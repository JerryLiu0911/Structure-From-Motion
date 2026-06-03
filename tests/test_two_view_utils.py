from pathlib import Path

import cv2
import numpy as np

from gf4.week3.two_view_utils import (
    compute_depths,
    compute_reprojection_errors,
    draw_reprojection_overlay,
    estimate_essential_matrix,
    filter_reconstructed_points,
    make_projection_matrices,
    project_points,
    recover_relative_pose,
    triangulate_points,
)


def _skew(v):
    x, y, z = v.ravel()
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def _project(points3d, K, R, t):
    camera_points = (R @ points3d.T + t.reshape(3, 1)).T
    image_points = (K @ camera_points.T).T
    return image_points[:, :2] / image_points[:, 2:]


def _synthetic_pair():
    K = np.array(
        [
            [900.0, 0.0, 320.0],
            [0.0, 900.0, 240.0],
            [0.0, 0.0, 1.0],
        ]
    )
    angle = np.deg2rad(6.0)
    R = np.array(
        [
            [np.cos(angle), 0.0, np.sin(angle)],
            [0.0, 1.0, 0.0],
            [-np.sin(angle), 0.0, np.cos(angle)],
        ]
    )
    t = np.array([[0.7], [0.05], [0.1]])
    t /= np.linalg.norm(t)

    xs = np.linspace(-1.0, 1.0, 6)
    ys = np.linspace(-0.7, 0.7, 5)
    points = []
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            z = 5.0 + 0.2 * i + 0.15 * j
            points.append([x, y, z])
    points3d = np.asarray(points)

    pts1 = _project(points3d, K, np.eye(3), np.zeros((3, 1)))
    pts2 = _project(points3d, K, R, t)
    return K, R, t, pts1, pts2


def test_estimate_essential_matrix_returns_boolean_inlier_mask():
    K, _, _, pts1, pts2 = _synthetic_pair()

    E, mask = estimate_essential_matrix(pts1, pts2, K, threshold=0.5)

    assert E.shape[1] == 3
    assert mask.dtype == bool
    assert mask.shape == (len(pts1),)
    assert 0 < np.sum(mask) <= len(pts1)


def test_recover_relative_pose_uses_mask_and_returns_boolean_mask():
    K, expected_R, expected_t, pts1, pts2 = _synthetic_pair()
    E = _skew(expected_t) @ expected_R

    R, t, mask = recover_relative_pose(E, pts1, pts2, K, inlier_mask=np.ones(len(pts1)))

    t /= np.linalg.norm(t)
    assert mask.dtype == bool
    assert mask.shape == (len(pts1),)
    assert np.sum(mask) >= 0.8 * len(pts1)
    assert np.allclose(R, expected_R, atol=1e-5)
    assert float(t.ravel() @ expected_t.ravel()) > 0.99


def test_projection_matrices_have_expected_camera_blocks():
    K, R, t, _, _ = _synthetic_pair()

    P1, P2 = make_projection_matrices(K, R, t)

    assert P1.shape == (3, 4)
    assert P2.shape == (3, 4)
    assert np.allclose(P1, K @ np.hstack([np.eye(3), np.zeros((3, 1))]))
    assert np.allclose(P2, K @ np.hstack([R, t]))


def test_triangulated_points_reproject_and_pass_filter():
    K, R, t, pts1, pts2 = _synthetic_pair()

    points3d = triangulate_points(pts1, pts2, K, R, t)
    errors1 = compute_reprojection_errors(points3d, pts1, K, np.eye(3), np.zeros((3, 1)))
    errors2 = compute_reprojection_errors(points3d, pts2, K, R, t)
    depths1, depths2 = compute_depths(points3d, R, t)
    keep = filter_reconstructed_points(points3d, errors1, errors2, R, t, 1e-6)

    assert points3d.shape == (len(pts1), 3)
    assert project_points(points3d[:3], K, R, t).shape == (3, 2)
    assert np.max(errors1) < 1e-7
    assert np.max(errors2) < 1e-7
    assert np.all(depths1 > 0)
    assert np.all(depths2 > 0)
    assert np.all(keep)


def test_filter_rejects_bad_depth_nonfinite_and_large_errors():
    K, R, t, pts1, pts2 = _synthetic_pair()
    points3d = triangulate_points(pts1[:4], pts2[:4], K, R, t)
    points3d[1, 2] = -1.0
    points3d[2, 0] = np.nan
    errors1 = np.array([0.5, 0.5, 0.5, 5.0])
    errors2 = np.array([0.5, 0.5, 0.5, 0.5])

    keep = filter_reconstructed_points(points3d, errors1, errors2, R, t, 4.0)

    assert keep.tolist() == [True, False, False, False]


def test_draw_reprojection_overlay_writes_image(tmp_path: Path):
    K, R, t, pts1, pts2 = _synthetic_pair()
    points3d = triangulate_points(pts1, pts2, K, R, t)
    image1 = np.zeros((480, 640, 3), dtype=np.uint8)
    image2 = np.zeros((480, 640, 3), dtype=np.uint8)
    output_path = tmp_path / "overlay.png"

    draw_reprojection_overlay(image1, image2, pts1, pts2, points3d, K, R, t, output_path)

    saved = cv2.imread(str(output_path))
    assert saved is not None
    assert saved.size > 0
