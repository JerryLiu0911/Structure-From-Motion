"""Utility functions for GF4 Week 3 sparse reconstruction.

Plotting, CSV, and PLY helpers are mostly provided.
The core calibrated geometry, reprojection checks,
and third-view registration steps are marked with TODO and should be completed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import os
import tempfile
from typing import Iterable

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "matplotlib"),
)

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")


@dataclass
class TwoViewResult:
    """Container for Week 3 two-view reconstruction metrics."""

    image_i: str
    image_j: str
    filtered_matches: int
    essential_inliers: int
    pose_inliers: int
    triangulated_points: int
    positive_depth_points: int
    kept_points: int
    median_reprojection_error_px: float | None
    mean_reprojection_error_px: float | None
    focal_length_px: float

    def as_dict(self) -> dict:
        return {
            "image_i": self.image_i,
            "image_j": self.image_j,
            "filtered_matches": self.filtered_matches,
            "essential_inliers": self.essential_inliers,
            "pose_inliers": self.pose_inliers,
            "triangulated_points": self.triangulated_points,
            "positive_depth_points": self.positive_depth_points,
            "kept_points": self.kept_points,
            "median_reprojection_error_px": self.median_reprojection_error_px,
            "mean_reprojection_error_px": self.mean_reprojection_error_px,
            "focal_length_px": self.focal_length_px,
        }


@dataclass
class ThirdViewResult:
    """Container for Week 3 third-view registration metrics."""

    image_k: str
    anchor_matches: int
    pnp_correspondences: int
    pnp_inliers: int
    median_reprojection_error_px: float | None
    mean_reprojection_error_px: float | None
    focal_length_px: float

    def as_dict(self) -> dict:
        return {
            "image_k": self.image_k,
            "anchor_matches": self.anchor_matches,
            "pnp_correspondences": self.pnp_correspondences,
            "pnp_inliers": self.pnp_inliers,
            "median_reprojection_error_px": self.median_reprojection_error_px,
            "mean_reprojection_error_px": self.mean_reprojection_error_px,
            "focal_length_px": self.focal_length_px,
        }


def ensure_dir(path: Path) -> Path:
    """Create an output directory if needed and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_csv(path: Path, rows: Iterable[dict]) -> None:
    """Save a list of dictionaries as CSV."""
    rows = list(rows)
    ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def make_camera_matrix(
    image_shape: tuple[int, ...],
    focal_length_px: float | None = None,
    principal_point: tuple[float, float] | None = None,
) -> np.ndarray:
    """Build a simple camera intrinsic matrix for a resized image."""
    height, width = image_shape[:2]
    if focal_length_px is None:
        focal_length_px = 1.2 * max(width, height)
    if principal_point is None:
        principal_point = ((width - 1) / 2.0, (height - 1) / 2.0)

    cx, cy = principal_point
    return np.array(
        [
            [focal_length_px, 0.0, cx],
            [0.0, focal_length_px, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _mask_to_bool(mask: np.ndarray | None, length: int) -> np.ndarray:
    if mask is None:
        return np.ones(length, dtype=bool)

    mask = np.asarray(mask).reshape(-1)
    if mask.size != length:
        raise ValueError(f"Expected mask of length {length}, got {mask.size}")
    return mask.astype(bool)


def _essential_candidates(E: np.ndarray) -> list[np.ndarray]:
    E = np.asarray(E, dtype=np.float64)
    if E.shape == (3, 3):
        return [E]
    if E.ndim == 2 and E.shape[1] == 3 and E.shape[0] % 3 == 0:
        return [E[i : i + 3] for i in range(0, E.shape[0], 3)]
    raise ValueError(f"Essential matrix has unsupported shape {E.shape}")


def estimate_essential_matrix(
    pts1: np.ndarray,
    pts2: np.ndarray,
    K: np.ndarray,
    threshold: float = 1.0,
    confidence: float = 0.999,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate the essential matrix with OpenCV RANSAC.
    """
    pts1 = np.asarray(pts1, dtype=np.float64)
    pts2 = np.asarray(pts2, dtype=np.float64)
    if pts1.shape != pts2.shape or pts1.ndim != 2 or pts1.shape[1] != 2:
        raise ValueError("Point arrays must both have shape (N, 2)")
    if len(pts1) < 5:
        raise ValueError(f"Need at least 5 correspondences, got {len(pts1)}")

    E, mask = cv2.findEssentialMat(
        pts1,
        pts2,
        cameraMatrix=K,
        method=cv2.RANSAC,
        prob=confidence,
        threshold=threshold,
    )
    if E is None or mask is None:
        raise ValueError("Essential matrix estimation failed")
    return np.asarray(E, dtype=np.float64), _mask_to_bool(mask, len(pts1))


def recover_relative_pose(
    E: np.ndarray,
    pts1: np.ndarray,
    pts2: np.ndarray,
    K: np.ndarray,
    inlier_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Recover relative camera pose from an essential matrix.
    """
    pts1 = np.asarray(pts1, dtype=np.float64)
    pts2 = np.asarray(pts2, dtype=np.float64)
    if pts1.shape != pts2.shape or pts1.ndim != 2 or pts1.shape[1] != 2:
        raise ValueError("Point arrays must both have shape (N, 2)")
    if len(pts1) < 5:
        raise ValueError(f"Need at least 5 correspondences, got {len(pts1)}")

    input_mask = None
    if inlier_mask is not None:
        input_mask = (
            _mask_to_bool(inlier_mask, len(pts1)).astype(np.uint8) * 255
        ).reshape(-1, 1)

    best_result = None
    for candidate in _essential_candidates(E):
        mask_arg = None if input_mask is None else input_mask.copy()
        _, R, t, mask = cv2.recoverPose(
            candidate,
            pts1,
            pts2,
            cameraMatrix=K,
            mask=mask_arg,
        )
        pose_mask = _mask_to_bool(mask, len(pts1))
        score = int(np.sum(pose_mask))
        if best_result is None or score > best_result[0]:
            best_result = (score, R, t, pose_mask)

    if best_result is None:
        raise ValueError("Relative pose recovery failed")

    _, R, t, pose_mask = best_result
    return R.astype(np.float64), t.astype(np.float64), pose_mask


def make_projection_matrices(
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Create projection matrices P1 = K[I|0] and P2 = K[R|t]."""
    K = np.asarray(K, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64).reshape(3, 1)
    if K.shape != (3, 3) or R.shape != (3, 3):
        raise ValueError("K and R must both be 3x3 matrices")

    P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = K @ np.hstack((R, t))
    return P1, P2


def triangulate_points(
    pts1: np.ndarray,
    pts2: np.ndarray,
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
) -> np.ndarray:
    """Triangulate 3D points from corresponding image points."""
    pts1 = np.asarray(pts1, dtype=np.float64)
    pts2 = np.asarray(pts2, dtype=np.float64)
    if pts1.shape != pts2.shape or pts1.ndim != 2 or pts1.shape[1] != 2:
        raise ValueError("Point arrays must both have shape (N, 2)")
    if len(pts1) == 0:
        return np.empty((0, 3), dtype=np.float64)

    P1, P2 = make_projection_matrices(K, R, t)
    points4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)

    with np.errstate(divide="ignore", invalid="ignore"):
        points3d = (points4d[:3] / points4d[3:4]).T
    return points3d.astype(np.float64)


def project_points(
    points3d: np.ndarray,
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
) -> np.ndarray:
    """Project 3D points into an image using camera matrix K[R|t]."""
    points3d = np.asarray(points3d, dtype=np.float64)
    if points3d.ndim != 2 or points3d.shape[1] != 3:
        raise ValueError("3D points must have shape (N, 3)")
    if len(points3d) == 0:
        return np.empty((0, 2), dtype=np.float64)

    K = np.asarray(K, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64).reshape(3, 1)
    camera_points = (R @ points3d.T + t).T
    image_points = (K @ camera_points.T).T
    with np.errstate(divide="ignore", invalid="ignore"):
        return image_points[:, :2] / image_points[:, 2:3]


def compute_reprojection_errors(
    points3d: np.ndarray,
    observed_pts: np.ndarray,
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
) -> np.ndarray:
    """Compute Euclidean reprojection error in pixels."""
    observed_pts = np.asarray(observed_pts, dtype=np.float64)
    if observed_pts.ndim != 2 or observed_pts.shape[1] != 2:
        raise ValueError("Observed points must have shape (N, 2)")

    projected_pts = project_points(points3d, K, R, t)
    if len(projected_pts) != len(observed_pts):
        raise ValueError("Projected and observed point counts differ")
    return np.linalg.norm(projected_pts - observed_pts, axis=1)


def compute_depths(
    points3d: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute point depths in camera 1 and camera 2 coordinates.

    Camera 1 has extrinsics [I|0]. Camera 2 has extrinsics [R|t].
    """
    points3d = np.asarray(points3d, dtype=np.float64)
    if points3d.ndim != 2 or points3d.shape[1] != 3:
        raise ValueError("3D points must have shape (N, 3)")
    if len(points3d) == 0:
        empty = np.empty(0, dtype=np.float64)
        return empty, empty

    R = np.asarray(R, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64).reshape(3, 1)
    depths1 = points3d[:, 2]
    depths2 = (R @ points3d.T + t)[2]
    return depths1, depths2


def filter_reconstructed_points(
    points3d: np.ndarray,
    errors1: np.ndarray,
    errors2: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    max_reprojection_error: float = 4.0,
) -> np.ndarray:
    """Return a boolean mask for valid triangulated points."""
    points3d = np.asarray(points3d, dtype=np.float64)
    errors1 = np.asarray(errors1, dtype=np.float64).reshape(-1)
    errors2 = np.asarray(errors2, dtype=np.float64).reshape(-1)
    if points3d.ndim != 2 or points3d.shape[1] != 3:
        raise ValueError("3D points must have shape (N, 3)")
    if len(errors1) != len(points3d) or len(errors2) != len(points3d):
        raise ValueError("Error arrays must match the number of 3D points")

    finite_mask = np.isfinite(points3d).all(axis=1)
    depths1, depths2 = compute_depths(points3d, R, t)
    positive_depth_mask = (depths1 > 0) & (depths2 > 0)
    reprojection_error_mask = (
        np.isfinite(errors1)
        & np.isfinite(errors2)
        & (errors1 <= max_reprojection_error)
        & (errors2 <= max_reprojection_error)
    )
    return finite_mask & positive_depth_mask & reprojection_error_mask


def build_2d3d_correspondences(
    reconstructed_anchor_indices: np.ndarray,
    reconstructed_points: np.ndarray,
    anchor_to_new_matches: list[cv2.DMatch],
    new_keypoints: list[cv2.KeyPoint],
) -> tuple[np.ndarray, np.ndarray]:
    """Build 2D-3D correspondences for registering a third image."""
    anchor_indices = np.asarray(reconstructed_anchor_indices, dtype=int).reshape(-1)
    points = np.asarray(reconstructed_points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Reconstructed points must have shape (N, 3)")
    if len(anchor_indices) != len(points):
        raise ValueError("Anchor index and 3D point counts differ")

    point_by_anchor = {
        int(anchor_idx): point
        for anchor_idx, point in zip(anchor_indices, points)
    }
    points3d = []
    pts_new = []
    used_new_indices = set()
    for match in anchor_to_new_matches:
        if match.trainIdx in used_new_indices:
            continue
        point = point_by_anchor.get(match.queryIdx)
        if point is None:
            continue
        if match.trainIdx < 0 or match.trainIdx >= len(new_keypoints):
            continue

        points3d.append(point)
        pts_new.append(new_keypoints[match.trainIdx].pt)
        used_new_indices.add(match.trainIdx)

    if not points3d:
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 2), dtype=np.float64)
    return np.asarray(points3d, dtype=np.float64), np.asarray(pts_new, dtype=np.float64)


def estimate_camera_pose_pnp(
    points3d: np.ndarray,
    pts2d: np.ndarray,
    K: np.ndarray,
    threshold: float = 6.0,
    confidence: float = 0.999,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate a new camera pose from 2D-3D correspondences using PnP."""
    points3d = np.asarray(points3d, dtype=np.float64)
    pts2d = np.asarray(pts2d, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    if points3d.ndim != 2 or points3d.shape[1] != 3:
        raise ValueError("3D points must have shape (N, 3)")
    if pts2d.ndim != 2 or pts2d.shape[1] != 2:
        raise ValueError("2D points must have shape (N, 2)")
    if len(points3d) != len(pts2d):
        raise ValueError("2D and 3D point counts differ")
    if len(points3d) < 6:
        raise ValueError(f"Need at least 6 correspondences, got {len(points3d)}")

    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        points3d,
        pts2d,
        K,
        None,
        reprojectionError=threshold,
        confidence=confidence,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok or rvec is None or tvec is None:
        raise ValueError("PnP pose estimation failed")

    R, _ = cv2.Rodrigues(rvec)
    inlier_mask = np.zeros(len(points3d), dtype=bool)
    if inliers is not None:
        inlier_mask[np.asarray(inliers).reshape(-1)] = True
    return R.astype(np.float64), tvec.reshape(3, 1).astype(np.float64), inlier_mask


def sample_point_colours(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    if len(pts) == 0:
        return np.empty((0, 3), dtype=np.uint8)

    height, width = image.shape[:2]
    xy = np.rint(pts).astype(int)
    xy[:, 0] = np.clip(xy[:, 0], 0, width - 1)
    xy[:, 1] = np.clip(xy[:, 1], 0, height - 1)
    bgr = image[xy[:, 1], xy[:, 0]]
    return bgr[:, ::-1].astype(np.uint8)


def _draw_reprojection_axis(
    ax,
    image: np.ndarray,
    observed: np.ndarray,
    projected: np.ndarray,
    title: str,
) -> None:
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax.set_title(title)
    ax.axis("off")
    if len(observed) == 0:
        return

    ax.scatter(
        observed[:, 0],
        observed[:, 1],
        s=24,
        facecolors="none",
        edgecolors="#2b8cbe",
        linewidths=1.2,
        label="Observed",
    )
    ax.scatter(
        projected[:, 0],
        projected[:, 1],
        s=28,
        marker="x",
        c="#d95f02",
        linewidths=1.3,
        label="Reprojected",
    )
    for obs, proj in zip(observed, projected):
        ax.plot(
            [obs[0], proj[0]],
            [obs[1], proj[1]],
            c="#525252",
            linewidth=0.8,
            alpha=0.75,
        )
    ax.legend(loc="lower right", fontsize=8, framealpha=0.85)


def draw_reprojection_overlay(
    image1: np.ndarray,
    image2: np.ndarray,
    pts1: np.ndarray,
    pts2: np.ndarray,
    points3d: np.ndarray,
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    output_path: Path,
    max_draw: int = 120,
) -> None:
    """Save a two-view overlay of observed points and reprojected 3D points."""
    pts1 = np.asarray(pts1, dtype=np.float64)
    pts2 = np.asarray(pts2, dtype=np.float64)
    points3d = np.asarray(points3d, dtype=np.float64)
    if pts1.shape != pts2.shape or pts1.ndim != 2 or pts1.shape[1] != 2:
        raise ValueError("Observed point arrays must both have shape (N, 2)")
    if points3d.ndim != 2 or points3d.shape[1] != 3:
        raise ValueError("3D points must have shape (N, 3)")
    if len(points3d) != len(pts1):
        raise ValueError("3D point and observed point counts differ")

    projected1 = project_points(points3d, K, np.eye(3), np.zeros((3, 1)))
    projected2 = project_points(points3d, K, R, t)

    count = min(max_draw, len(points3d))
    if count:
        indices = np.linspace(0, len(points3d) - 1, count, dtype=int)
        pts1 = pts1[indices]
        pts2 = pts2[indices]
        projected1 = projected1[indices]
        projected2 = projected2[indices]
    else:
        pts1 = pts1[:0]
        pts2 = pts2[:0]
        projected1 = projected1[:0]
        projected2 = projected2[:0]

    import matplotlib.pyplot as plt

    ensure_dir(output_path.parent)
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    _draw_reprojection_axis(
        axes[0], image1, pts1, projected1, "Image 1 reprojection"
    )
    _draw_reprojection_axis(
        axes[1], image2, pts2, projected2, "Image 2 reprojection"
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def draw_single_image_reprojection_overlay(
    image: np.ndarray,
    observed_pts: np.ndarray,
    points3d: np.ndarray,
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    output_path: Path,
    max_draw: int = 120,
) -> None:
    """Save a reprojection check for one registered camera.

    TODO: Complete this function.

    Required behaviour:
    - project points3d into the image using camera pose [R|t],
    - draw observed_pts and projected points on top of the image,
    - draw a short line from each observed point to its reprojection,
    - save the figure to output_path.

    This is the corresponding correctness check for the image registered by
    PnP.
    """
    raise NotImplementedError("TODO: draw single-image reprojection overlay")


def _camera_center(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return (-R.T @ t.reshape(3, 1)).ravel()


def _camera_frustum(
    R_wc: np.ndarray,
    center: np.ndarray,
    scale: float,
) -> np.ndarray:
    corners = np.array(
        [
            [-0.5, -0.35, 1.0],
            [0.5, -0.35, 1.0],
            [0.5, 0.35, 1.0],
            [-0.5, 0.35, 1.0],
        ],
        dtype=np.float64,
    )
    return center[None, :] + scale * (R_wc @ corners.T).T


def _set_axes_equal(ax) -> None:
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])
    radius = 0.5 * max(x_range, y_range, z_range, 1e-9)

    x_mid = np.mean(x_limits)
    y_mid = np.mean(y_limits)
    z_mid = np.mean(z_limits)
    ax.set_xlim3d([x_mid - radius, x_mid + radius])
    ax.set_ylim3d([y_mid - radius, y_mid + radius])
    ax.set_zlim3d([z_mid - radius, z_mid + radius])


def _robust_bounds(points: np.ndarray, extra_points: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    point_cloud = np.empty((0, 3), dtype=np.float64)
    if len(points):
        point_cloud = points[np.isfinite(points).all(axis=1)]

    extra_cloud = np.empty((0, 3), dtype=np.float64)
    if extra_points is not None and len(extra_points):
        extra_cloud = extra_points[np.isfinite(extra_points).all(axis=1)]

    if len(point_cloud) == 0 and len(extra_cloud) == 0:
        return np.array([-1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0])

    if len(point_cloud) > 10:
        lower = np.percentile(point_cloud, 2, axis=0)
        upper = np.percentile(point_cloud, 98, axis=0)
    elif len(point_cloud):
        lower = np.min(point_cloud, axis=0)
        upper = np.max(point_cloud, axis=0)
    else:
        lower = np.min(extra_cloud, axis=0)
        upper = np.max(extra_cloud, axis=0)

    if len(extra_cloud):
        lower = np.minimum(lower, np.min(extra_cloud, axis=0))
        upper = np.maximum(upper, np.max(extra_cloud, axis=0))

    span = np.maximum(upper - lower, 1e-6)
    padding = 0.18 * span
    return lower - padding, upper + padding


def _apply_equal_limits(ax, lower: np.ndarray, upper: np.ndarray) -> None:
    center = 0.5 * (lower + upper)
    radius = 0.5 * max(np.max(upper - lower), 1e-6)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def plot_two_view_reconstruction(
    points3d: np.ndarray,
    colours: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    output_path: Path,
    max_points: int = 5000,
) -> None:
    import matplotlib.pyplot as plt

    ensure_dir(output_path.parent)

    points = points3d
    point_colours = colours
    if len(points) > max_points:
        indices = np.linspace(0, len(points) - 1, max_points, dtype=int)
        points = points[indices]
        point_colours = point_colours[indices]

    center1 = np.zeros(3)
    center2 = _camera_center(R, t)
    baseline = np.linalg.norm(center2 - center1)
    frustum_scale = max(0.65, 0.55 * baseline)

    camera_points = np.vstack([center1, center2])
    frustum1 = _camera_frustum(np.eye(3), center1, frustum_scale)
    frustum2 = _camera_frustum(R.T, center2, frustum_scale)
    plot_extra = np.vstack([camera_points, frustum1, frustum2])
    lower, upper = _robust_bounds(points, extra_points=plot_extra)
    point_size = 24 if len(points) < 200 else 9 if len(points) < 1000 else 3
    rgb = point_colours / 255.0 if len(point_colours) else "#456990"

    fig = plt.figure(figsize=(15, 8))
    grid = fig.add_gridspec(2, 2, width_ratios=[1.28, 1.0], height_ratios=[1.0, 0.82])
    ax = fig.add_subplot(grid[:, 0], projection="3d")
    ax_xz = fig.add_subplot(grid[0, 1])
    ax_cam = fig.add_subplot(grid[1, 1])

    def draw_camera_3d(axis, center, frustum, colour, label):
        axis.scatter(
            [center[0]],
            [center[1]],
            [center[2]],
            c=colour,
            s=170,
            edgecolors="white",
            linewidths=1.2,
            depthshade=False,
            label=label,
        )
        closed = np.vstack([frustum, frustum[0]])
        axis.plot(closed[:, 0], closed[:, 1], closed[:, 2], c=colour, linewidth=2.0)
        for corner in frustum:
            axis.plot(
                [center[0], corner[0]],
                [center[1], corner[1]],
                [center[2], corner[2]],
                c=colour,
                linewidth=1.2,
            )

    def draw_camera_xz(
        axis,
        center,
        frustum,
        colour,
        label,
        marker_size=150,
        label_offset=(0.0, 0.0),
        label_va="center",
    ):
        axis.scatter(
            [center[0]],
            [center[2]],
            c=colour,
            s=marker_size,
            edgecolors="white",
            linewidths=1.2,
            zorder=5,
            label=label,
        )
        closed = np.vstack([frustum, frustum[0]])
        axis.plot(closed[:, 0], closed[:, 2], c=colour, linewidth=2.0, zorder=4)
        for corner in frustum:
            axis.plot(
                [center[0], corner[0]],
                [center[2], corner[2]],
                c=colour,
                linewidth=1.2,
                zorder=4,
            )
        axis.text(
            center[0] + label_offset[0],
            center[2] + label_offset[1],
            f"  {label}",
            color=colour,
            weight="bold",
            va=label_va,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1.5},
            zorder=6,
        )

    if len(points):
        ax.scatter(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            s=point_size,
            c=rgb,
            alpha=0.9,
            depthshade=False,
        )
    else:
        ax.text(0, 0, 0, "No points after filtering", ha="center")

    draw_camera_3d(ax, center1, frustum1, "#2458a6", "Camera 1")
    draw_camera_3d(ax, center2, frustum2, "#a33b3b", "Camera 2")
    ax.plot([center1[0], center2[0]], [center1[1], center2[1]], [center1[2], center2[2]], c="#5b6675", linewidth=2.0)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z / depth")
    ax.set_title(f"Scene view ({len(points)} points)")
    ax.legend(loc="upper right")
    _apply_equal_limits(ax, lower, upper)
    ax.view_init(elev=18, azim=-68)
    try:
        ax.set_proj_type("ortho")
    except AttributeError:
        pass

    if len(points):
        ax_xz.scatter(points[:, 0], points[:, 2], s=point_size, c=rgb, alpha=0.9)
    ax_xz.plot([center1[0], center2[0]], [center1[2], center2[2]], c="#5b6675", linewidth=1.8)
    draw_camera_xz(ax_xz, center1, frustum1, "#2458a6", "C1")
    draw_camera_xz(ax_xz, center2, frustum2, "#a33b3b", "C2")

    x_padding = 0.05 * max(upper[0] - lower[0], 1e-6)
    z_padding = 0.05 * max(upper[2] - lower[2], 1e-6)
    ax_xz.set_xlim(lower[0] - x_padding, upper[0] + x_padding)
    ax_xz.set_ylim(lower[2] - z_padding, upper[2] + z_padding)
    ax_xz.grid(True, alpha=0.25)
    ax_xz.set_xlabel("X")
    ax_xz.set_ylabel("Z / depth")
    ax_xz.set_title("Full X-Z depth view")
    ax_xz.legend(loc="upper right")
    ax_xz.text(
        0.02,
        0.02,
        "Translation scale is arbitrary",
        transform=ax_xz.transAxes,
        fontsize=9,
        color="#5b6675",
    )

    detail_points = np.vstack([camera_points, frustum1, frustum2])
    detail_x = detail_points[:, 0]
    detail_z = detail_points[:, 2]
    detail_center = np.array([np.mean(detail_x), np.mean(detail_z)])
    detail_radius = 0.85 * max(np.ptp(detail_x), np.ptp(detail_z), baseline, 1.0)
    ax_cam.plot([center1[0], center2[0]], [center1[2], center2[2]], c="#5b6675", linewidth=2.2)
    label_offset = (0.02 * detail_radius, 0.05 * detail_radius)
    draw_camera_xz(
        ax_cam,
        center1,
        frustum1,
        "#2458a6",
        "Camera 1",
        marker_size=190,
        label_offset=label_offset,
        label_va="bottom",
    )
    draw_camera_xz(
        ax_cam,
        center2,
        frustum2,
        "#a33b3b",
        "Camera 2",
        marker_size=190,
        label_offset=label_offset,
        label_va="bottom",
    )
    ax_cam.set_xlim(detail_center[0] - detail_radius, detail_center[0] + detail_radius)
    ax_cam.set_ylim(detail_center[1] - detail_radius, detail_center[1] + detail_radius)
    ax_cam.set_aspect("equal", adjustable="box")
    ax_cam.grid(True, alpha=0.25)
    ax_cam.set_xlabel("X")
    ax_cam.set_ylabel("Z / depth")
    ax_cam.set_title("Camera baseline zoom")
    ax_cam.text(
        0.02,
        0.98,
        "Frustums show viewing direction; scale is arbitrary",
        transform=ax_cam.transAxes,
        fontsize=9,
        color="#5b6675",
        va="top",
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_multi_view_reconstruction(
    points3d: np.ndarray,
    colours: np.ndarray,
    camera_poses: list[tuple[str, np.ndarray, np.ndarray]],
    output_path: Path,
    max_points: int = 5000,
) -> None:
    """Save a sparse reconstruction plot consistent with the two-view figure."""
    import matplotlib.pyplot as plt

    ensure_dir(output_path.parent)

    points = points3d
    point_colours = colours
    if len(points) > max_points:
        indices = np.linspace(0, len(points) - 1, max_points, dtype=int)
        points = points[indices]
        point_colours = point_colours[indices]

    if camera_poses:
        camera_centers = np.array([_camera_center(R, t) for _, R, t in camera_poses])
    else:
        camera_centers = np.empty((0, 3), dtype=np.float64)

    if len(camera_centers) >= 2:
        baseline = np.median(
            [
                np.linalg.norm(camera_centers[i] - camera_centers[j])
                for i in range(len(camera_centers))
                for j in range(i + 1, len(camera_centers))
            ]
        )
    else:
        baseline = 1.0
    frustum_scale = max(0.65, 0.55 * baseline)

    frustums = [
        _camera_frustum(R.T, center, frustum_scale)
        for center, (_, R, _) in zip(camera_centers, camera_poses)
    ]
    plot_extra = camera_centers
    if frustums:
        plot_extra = np.vstack([camera_centers, *frustums])

    lower, upper = _robust_bounds(points, extra_points=plot_extra)
    point_size = 24 if len(points) < 200 else 9 if len(points) < 1000 else 3
    rgb = point_colours / 255.0 if len(point_colours) else "#456990"

    fig = plt.figure(figsize=(15, 8))
    grid = fig.add_gridspec(2, 2, width_ratios=[1.28, 1.0], height_ratios=[1.0, 0.82])
    ax = fig.add_subplot(grid[:, 0], projection="3d")
    ax_xz = fig.add_subplot(grid[0, 1])
    ax_cam = fig.add_subplot(grid[1, 1])

    colours_by_camera = ["#2458a6", "#a33b3b", "#2f7d32", "#7a4ea3", "#b46b00"]

    def draw_camera_3d(axis, center, frustum, colour, label):
        axis.scatter(
            [center[0]],
            [center[1]],
            [center[2]],
            c=colour,
            s=170,
            edgecolors="white",
            linewidths=1.2,
            depthshade=False,
            label=label,
        )
        closed = np.vstack([frustum, frustum[0]])
        axis.plot(closed[:, 0], closed[:, 1], closed[:, 2], c=colour, linewidth=2.0)
        for corner in frustum:
            axis.plot(
                [center[0], corner[0]],
                [center[1], corner[1]],
                [center[2], corner[2]],
                c=colour,
                linewidth=1.2,
            )

    def draw_camera_xz(
        axis,
        center,
        frustum,
        colour,
        label,
        marker_size=150,
        label_offset=(0.0, 0.0),
        label_va="center",
    ):
        axis.scatter(
            [center[0]],
            [center[2]],
            c=colour,
            s=marker_size,
            edgecolors="white",
            linewidths=1.2,
            zorder=5,
            label=label,
        )
        closed = np.vstack([frustum, frustum[0]])
        axis.plot(closed[:, 0], closed[:, 2], c=colour, linewidth=2.0, zorder=4)
        for corner in frustum:
            axis.plot(
                [center[0], corner[0]],
                [center[2], corner[2]],
                c=colour,
                linewidth=1.2,
                zorder=4,
            )
        axis.text(
            center[0] + label_offset[0],
            center[2] + label_offset[1],
            f"  {label}",
            color=colour,
            weight="bold",
            va=label_va,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1.5},
            zorder=6,
        )

    if len(points):
        ax.scatter(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            s=point_size,
            c=rgb,
            alpha=0.9,
            depthshade=False,
        )
    else:
        ax.text(0, 0, 0, "No points after filtering", ha="center")

    for idx, ((label, _, _), center, frustum) in enumerate(zip(camera_poses, camera_centers, frustums)):
        colour = colours_by_camera[idx % len(colours_by_camera)]
        draw_camera_3d(ax, center, frustum, colour, label)

    if len(camera_centers) >= 2:
        ax.plot(
            camera_centers[:, 0],
            camera_centers[:, 1],
            camera_centers[:, 2],
            c="#5b6675",
            linewidth=2.0,
        )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z / depth")
    ax.set_title(f"Scene view ({len(points)} points, {len(camera_poses)} cameras)")
    ax.legend(loc="upper right")
    _apply_equal_limits(ax, lower, upper)
    ax.view_init(elev=18, azim=-68)
    try:
        ax.set_proj_type("ortho")
    except AttributeError:
        pass

    if len(points):
        ax_xz.scatter(points[:, 0], points[:, 2], s=point_size, c=rgb, alpha=0.9)
    if len(camera_centers) >= 2:
        ax_xz.plot(camera_centers[:, 0], camera_centers[:, 2], c="#5b6675", linewidth=1.8)
    for idx, ((_, _, _), center, frustum) in enumerate(zip(camera_poses, camera_centers, frustums)):
        colour = colours_by_camera[idx % len(colours_by_camera)]
        draw_camera_xz(ax_xz, center, frustum, colour, f"C{idx + 1}")

    x_padding = 0.05 * max(upper[0] - lower[0], 1e-6)
    z_padding = 0.05 * max(upper[2] - lower[2], 1e-6)
    ax_xz.set_xlim(lower[0] - x_padding, upper[0] + x_padding)
    ax_xz.set_ylim(lower[2] - z_padding, upper[2] + z_padding)
    ax_xz.grid(True, alpha=0.25)
    ax_xz.set_xlabel("X")
    ax_xz.set_ylabel("Z / depth")
    ax_xz.set_title("Full X-Z depth view")
    ax_xz.legend(loc="upper right")
    ax_xz.text(
        0.02,
        0.02,
        "Translation scale is arbitrary",
        transform=ax_xz.transAxes,
        fontsize=9,
        color="#5b6675",
    )

    if len(plot_extra):
        detail_x = plot_extra[:, 0]
        detail_z = plot_extra[:, 2]
        detail_center = np.array([np.mean(detail_x), np.mean(detail_z)])
        detail_radius = 0.85 * max(np.ptp(detail_x), np.ptp(detail_z), baseline, 1.0)
    else:
        detail_center = np.zeros(2)
        detail_radius = 1.0

    if len(camera_centers) >= 2:
        ax_cam.plot(camera_centers[:, 0], camera_centers[:, 2], c="#5b6675", linewidth=2.2)
    label_offset = (0.02 * detail_radius, 0.05 * detail_radius)
    for idx, ((label, _, _), center, frustum) in enumerate(zip(camera_poses, camera_centers, frustums)):
        colour = colours_by_camera[idx % len(colours_by_camera)]
        draw_camera_xz(
            ax_cam,
            center,
            frustum,
            colour,
            label,
            marker_size=190,
            label_offset=label_offset,
            label_va="bottom",
        )

    ax_cam.set_xlim(detail_center[0] - detail_radius, detail_center[0] + detail_radius)
    ax_cam.set_ylim(detail_center[1] - detail_radius, detail_center[1] + detail_radius)
    ax_cam.set_aspect("equal", adjustable="box")
    ax_cam.grid(True, alpha=0.25)
    ax_cam.set_xlabel("X")
    ax_cam.set_ylabel("Z / depth")
    ax_cam.set_title("Camera trajectory zoom")
    ax_cam.text(
        0.02,
        0.98,
        "Frustums show viewing direction; scale is arbitrary",
        transform=ax_cam.transAxes,
        fontsize=9,
        color="#5b6675",
        va="top",
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close(fig)


def _extract_rgb_patch(image: np.ndarray, point: np.ndarray, patch_radius: int) -> np.ndarray:
    """Extract a small RGB image patch centred on a floating-point image coordinate."""
    height, width = image.shape[:2]
    x = int(round(float(point[0])))
    y = int(round(float(point[1])))
    x0 = max(0, x - patch_radius)
    x1 = min(width, x + patch_radius + 1)
    y0 = max(0, y - patch_radius)
    y1 = min(height, y + patch_radius + 1)

    patch = image[y0:y1, x0:x1]
    if patch.size == 0:
        size = 2 * patch_radius + 1
        return np.full((size, size, 3), 220, dtype=np.uint8)
    return cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)


def plot_patch_cloud_reconstruction(
    points3d: np.ndarray,
    source_image: np.ndarray,
    source_points: np.ndarray,
    camera_poses: list[tuple[str, np.ndarray, np.ndarray]],
    output_path: Path,
    max_patches: int = 70,
    patch_radius: int = 14,
    patch_zoom: float = 0.55,
) -> None:
    """Save an X-Z reconstruction view where points are shown as image patches."""
    import matplotlib.pyplot as plt
    from matplotlib.offsetbox import AnnotationBbox, OffsetImage

    ensure_dir(output_path.parent)

    n = min(len(points3d), len(source_points))
    points = points3d[:n]
    image_points = source_points[:n]

    if camera_poses:
        camera_centers = np.array([_camera_center(R, t) for _, R, t in camera_poses])
    else:
        camera_centers = np.empty((0, 3), dtype=np.float64)

    if len(camera_centers) >= 2:
        baseline = np.median(
            [
                np.linalg.norm(camera_centers[i] - camera_centers[j])
                for i in range(len(camera_centers))
                for j in range(i + 1, len(camera_centers))
            ]
        )
    else:
        baseline = 1.0
    frustum_scale = max(0.65, 0.55 * baseline)

    frustums = [
        _camera_frustum(R.T, center, frustum_scale)
        for center, (_, R, _) in zip(camera_centers, camera_poses)
    ]
    plot_extra = camera_centers
    if frustums:
        plot_extra = np.vstack([camera_centers, *frustums])
    lower, upper = _robust_bounds(points, extra_points=plot_extra)

    fig, ax = plt.subplots(1, 1, figsize=(11, 7))
    if len(points):
        ax.scatter(points[:, 0], points[:, 2], s=12, c="#9aa5b1", alpha=0.32, label="all points")
        if len(points) > max_patches:
            indices = np.linspace(0, len(points) - 1, max_patches, dtype=int)
        else:
            indices = np.arange(len(points))

        for idx in indices:
            patch_rgb = _extract_rgb_patch(source_image, image_points[idx], patch_radius)
            image_box = OffsetImage(patch_rgb, zoom=patch_zoom)
            ab = AnnotationBbox(
                image_box,
                (points[idx, 0], points[idx, 2]),
                frameon=True,
                pad=0.08,
                bboxprops={
                    "edgecolor": "#263238",
                    "linewidth": 0.6,
                    "alpha": 0.9,
                },
                zorder=4,
            )
            ax.add_artist(ab)
    else:
        ax.text(0.5, 0.5, "No points after filtering", transform=ax.transAxes, ha="center")

    colours_by_camera = ["#2458a6", "#a33b3b", "#2f7d32", "#7a4ea3", "#b46b00"]
    if len(camera_centers) >= 2:
        ax.plot(camera_centers[:, 0], camera_centers[:, 2], c="#5b6675", linewidth=2.0, zorder=2)

    for idx, ((label, _, _), center, frustum) in enumerate(zip(camera_poses, camera_centers, frustums)):
        colour = colours_by_camera[idx % len(colours_by_camera)]
        ax.scatter(
            [center[0]],
            [center[2]],
            c=colour,
            s=170,
            edgecolors="white",
            linewidths=1.2,
            zorder=6,
            label=label,
        )
        closed = np.vstack([frustum, frustum[0]])
        ax.plot(closed[:, 0], closed[:, 2], c=colour, linewidth=2.0, zorder=5)
        for corner in frustum:
            ax.plot([center[0], corner[0]], [center[2], corner[2]], c=colour, linewidth=1.2, zorder=5)
        ax.text(
            center[0],
            center[2],
            f"  {label}",
            color=colour,
            weight="bold",
            va="center",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1.5},
            zorder=7,
        )

    x_padding = 0.05 * max(upper[0] - lower[0], 1e-6)
    z_padding = 0.05 * max(upper[2] - lower[2], 1e-6)
    ax.set_xlim(lower[0] - x_padding, upper[0] + x_padding)
    ax.set_ylim(lower[2] - z_padding, upper[2] + z_padding)
    ax.grid(True, alpha=0.25)
    ax.set_xlabel("X")
    ax.set_ylabel("Z / depth")
    ax.set_title("Patch cloud X-Z view")
    ax.legend(loc="upper right")
    ax.text(
        0.02,
        0.02,
        "Each thumbnail is cropped around the source feature in image 1",
        transform=ax.transAxes,
        fontsize=9,
        color="#5b6675",
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close(fig)


def write_ply(path: Path, points3d: np.ndarray, colours: np.ndarray) -> None:
    """Write a simple ASCII PLY point cloud with RGB colours."""
    ensure_dir(path.parent)
    if len(colours) != len(points3d):
        colours = np.full((len(points3d), 3), 200, dtype=np.uint8)

    with path.open("w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points3d)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for point, colour in zip(points3d, colours):
            f.write(
                f"{point[0]:.8f} {point[1]:.8f} {point[2]:.8f} "
                f"{int(colour[0])} {int(colour[1])} {int(colour[2])}\n"
            )
