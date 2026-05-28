"""Utility functions for GF4 Week 2 pairwise SfM front-end.

This file is intentionally a starter scaffold. Basic file handling and a few
plotting helpers are provided. The core SfM-front-end steps are marked with
TODO and should be completed by students.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import math
from typing import Iterable

import cv2
import numpy as np
import matplotlib.pyplot as plt


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass
class ImageFeatures:
    """Image, keypoints, and descriptors for one input image."""

    path: Path
    image: np.ndarray
    keypoints: list[cv2.KeyPoint]
    descriptors: np.ndarray


@dataclass
class PairAnalysis:
    """Container for pairwise matching and epipolar-geometry results."""

    image_i: str
    image_j: str
    keypoints_i: int
    keypoints_j: int
    raw_matches: int
    filtered_matches: int
    ransac_inliers: int
    inlier_ratio: float
    mean_epipolar_error_all: float | None
    median_epipolar_error_all: float | None
    mean_epipolar_error_inliers: float | None
    median_epipolar_error_inliers: float | None
    max_epipolar_error_inliers: float | None
    fundamental_matrix: list[list[float]] | None

    def as_dict(self) -> dict:
        return {
            "image_i": self.image_i,
            "image_j": self.image_j,
            "keypoints_i": self.keypoints_i,
            "keypoints_j": self.keypoints_j,
            "raw_matches": self.raw_matches,
            "filtered_matches": self.filtered_matches,
            "ransac_inliers": self.ransac_inliers,
            "inlier_ratio": self.inlier_ratio,
            "mean_epipolar_error_all": self.mean_epipolar_error_all,
            "median_epipolar_error_all": self.median_epipolar_error_all,
            "mean_epipolar_error_inliers": self.mean_epipolar_error_inliers,
            "median_epipolar_error_inliers": self.median_epipolar_error_inliers,
            "max_epipolar_error_inliers": self.max_epipolar_error_inliers,
            "fundamental_matrix": self.fundamental_matrix,
        }

    def csv_dict(self) -> dict:
        """Return scalar fields suitable for CSV output."""
        data = self.as_dict()
        data["fundamental_matrix"] = (
            "" if self.fundamental_matrix is None else str(self.fundamental_matrix)
        )
        return data


def ensure_dir(path: Path) -> Path:
    """Create an output directory if needed and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def list_image_paths(image_dir: Path, max_images: int | None = None) -> list[Path]:
    """Return sorted image paths from a directory."""
    image_dir = Path(image_dir)
    if not image_dir.exists() or not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {image_dir}")

    paths = sorted(
        p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if max_images is not None:
        paths = paths[:max_images]
    if not paths:
        raise ValueError(f"No images found in {image_dir}")
    return paths


def load_image(path: Path, max_size: int | None = None) -> np.ndarray:
    """Load an image with OpenCV in BGR order, optionally resizing the long edge."""
    path = Path(path)
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")

    if max_size is not None:
        height, width = image.shape[:2]
        scale = max_size / max(height, width)
        if scale < 1.0:
            image = cv2.resize(
                image,
                (int(round(width * scale)), int(round(height * scale))),
                interpolation=cv2.INTER_AREA,
            )
    return image


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


def detect_sift_features(
    image: np.ndarray,
    max_features: int = 4000,
) -> tuple[list[cv2.KeyPoint], np.ndarray]:
    """Detect SIFT keypoints and descriptors.
    
    TODO: Complete this function.

    Hints:
    - Convert the image to grayscale.
    - Create a SIFT detector with cv2.SIFT_create(nfeatures=max_features).
    - Return keypoints and descriptors from detector.detectAndCompute(...).
    - If no descriptors are found, return an empty array with shape (0, 128).
    - If OpenCV returns slightly more than max_features, keep only the first
      max_features keypoints and matching descriptor rows.
    """
    sift = cv2.SIFT_create(max_features) # creates a SIFT detector that returns at most max_features keypoints.
    keypoints, descriptors = sift.detectAndCompute(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), None) # returns keypoints and descriptors for the input image, after converting the image to greyscale.
    if descriptors is None:
        return [], np.empty((0, 128)) # returns empty keypoints and descriptors if no features are found.
    return keypoints[:max_features], descriptors[:max_features]

    raise NotImplementedError("TODO: implement SIFT feature detection")


def precompute_image_features(
    image_paths: list[Path],
    max_features: int = 4000,
    max_image_size: int | None = 1600,
) -> list[ImageFeatures]:
    """Load each image and compute SIFT features once.

    TODO: Complete this function after implementing detect_sift_features.

    Dataset mode should use this function so SIFT is not recomputed for the
    same image in every pair.
    """
    output_list = []
    for image_path in image_paths: # For each image path, load the image and compute SIFT features, then store them in an ImageFeatures dataclass and append to the output list.
        image = load_image(image_path, max_size=max_image_size)
        keypoints, descriptors = detect_sift_features(image, max_features=max_features)
        output_list.append(ImageFeatures(
            path=image_path,
            image=image,
            keypoints=keypoints,
            descriptors=descriptors,
        ))
        # Probably could be more efficient if implemented with a generator?
        # yield ImageFeatures(
        #     path=image_path,
        #     image=image,
        #     keypoints=keypoints,
        #     descriptors=descriptors,
        # )
    return output_list
    # raise NotImplementedError("TODO: implement feature precomputation") 


def raw_descriptor_matches(desc1: np.ndarray, desc2: np.ndarray) -> list[cv2.DMatch]:
    """Return one nearest-neighbour match per descriptor before Lowe filtering.

    TODO: Complete this function.

    Hints:
    - Handle empty descriptor arrays by returning an empty list.
    - For SIFT descriptors, use cv2.BFMatcher(cv2.NORM_L2).
    - Use matcher.match(desc1, desc2) to get the best match in image 2 for
      each descriptor in image 1.
    - Return the matches sorted by descriptor distance.
    """

    output_matches = []
    if desc1.size == 0 or desc2.size == 0: # edge case if descriptor arrays are empty, return an empty list of matches
        return output_matches
    matcher = cv2.BFMatcher(cv2.NORM_L2)  # brute-force matcher using L2 norm for SIFT descriptors, cross
    matches = matcher.match(desc1, desc2)   # returns a list of DMatch objects, which contains the best match in desc2 for each descriptor in desc1 (in index), and L2 distance between the matches
    output_matches = sorted(matches, key=lambda m: m.distance) # sort pairs by distance
    return output_matches

    raise NotImplementedError("TODO: compute raw nearest-neighbour matches")


def match_descriptors(
    desc1: np.ndarray,
    desc2: np.ndarray,
    ratio: float = 0.75,
) -> list[cv2.DMatch]:
    """Match SIFT descriptors using Lowe's ratio test.

    TODO: Complete this function.

    Hints:
    - Handle empty descriptor arrays by returning an empty list.
    - For SIFT descriptors, use cv2.BFMatcher(cv2.NORM_L2).
    - Use knnMatch(desc1, desc2, k=2) for the ratio test.
    - Keep a match when best_distance < ratio * second_best_distance.
    """

    output_matches = []
    if desc1.size == 0 or desc2.size == 0:
        return output_matches
    matcher = cv2.BFMatcher(cv2.NORM_L2) # Same as raw_descriptor_matches
    knn_matches = matcher.knnMatch(desc1, desc2, k=2) #Instead of best match, returns the top two matches also sorted by distance, for each descriptor in desc1. The output is a list of lists of DMatch objects, where each inner list contains the two best matches by clustering and sorted by distance.
    for m, n in knn_matches:
        if m.distance < ratio * n.distance: # Lowe's ratio test: filters by a ratio threshold between the best and second-best matches, only keep best match if it's significantly better than the second-best match, otherwise the match is discarded
            output_matches.append(m)
    return output_matches

    raise NotImplementedError("TODO: implement descriptor matching")


def count_raw_matches(desc1: np.ndarray, desc2: np.ndarray) -> int:
    """Return the number of descriptors that can be matched before filtering.

    TODO: Complete this function.

    A simple definition is len(raw_descriptor_matches(desc1, desc2)). This
    gives a useful denominator for comparing raw and filtered matching.
    """
    return len(raw_descriptor_matches(desc1, desc2))

    raise NotImplementedError("TODO: count raw descriptor matches")


def matched_keypoint_coords(
    keypoints1: list[cv2.KeyPoint],
    keypoints2: list[cv2.KeyPoint],
    matches: list[cv2.DMatch],
) -> tuple[np.ndarray, np.ndarray]:
    """Convert OpenCV matches into aligned Nx2 coordinate arrays.

    TODO: Complete this function.

    Remember: cv2.KeyPoint.pt is (x, y), not (row, column).
    """
    if not matches:
        return np.empty((0, 2)), np.empty((0, 2))
    coord_array_1 = np.array([keypoints1[m.queryIdx].pt for m in matches]) # m.queryIdx is the index of the descriptor in desc1, which corresponds to the keypoint in keypoints1, and .pt gives the (x, y) coordinates of that keypoint
    coord_array_2 = np.array([keypoints2[m.trainIdx].pt for m in matches]) # similarly m.trainIdx is the index of the descriptor in desc2, etc.
    return coord_array_1, coord_array_2

    raise NotImplementedError("TODO: convert matches to coordinate arrays")


def estimate_fundamental_ransac(
    pts1: np.ndarray,
    pts2: np.ndarray,
    threshold: float = 1.0,
    confidence: float = 0.99,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate the fundamental matrix with OpenCV RANSAC.

    TODO: Complete this function.

    Return:
    - F: 3x3 fundamental matrix
    - inlier_mask: boolean array of shape (N,)
    """
    F_matrix, inlier_mask = cv2.findFundamentalMat(pts1, pts2, cv2.RANSAC, threshold, confidence) # returns fundamental matrix and inlier mask of boolean values indicating which point pairs are inliers according to the RANSAC algorithm, based on the specified distance threshold and confidence level.
    return F_matrix, inlier_mask

    raise NotImplementedError("TODO: estimate fundamental matrix with RANSAC")


def compute_epipolar_errors(
    F: np.ndarray,
    pts1: np.ndarray,
    pts2: np.ndarray,
) -> np.ndarray:
    """Compute point-to-epipolar-line distances in image 2.

    TODO: Complete this function.

    For each point x1 in image 1, compute the epipolar line l2 = F x1.
    Then compute the distance from the corresponding x2 to l2.
    """

    distances = []
    for i in range(pts1.shape[0]): # For each coordinate pair in pts1, compute the epipolar line in 
        x1 = np.array([pts1[i][0], pts1[i][1], 1])
        line = F @ x1 # finds the epipolar line (aka the projection of the ray from image 1 to image 2) using the fundamental matrix
        a, b, c = line
        x2 = pts2[i]
        distance = abs(a * x2[0] + b * x2[1] + c) / np.sqrt(a**2 + b**2) # distance from point to line formula: |ax + by + c| / sqrt(a^2 + b^2), where (x, y) is the point and ax + by + c = 0 is the line equation.
        distances.append(distance)
    return np.array(distances)
    

    raise NotImplementedError("TODO: implement epipolar error calculation")


def draw_keypoints(
    image: np.ndarray,
    keypoints: list[cv2.KeyPoint],
    output_path: Path,
) -> None:
    """Save a keypoint visualisation."""
    ensure_dir(output_path.parent)
    vis = cv2.drawKeypoints(
        image,
        keypoints,
        None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )
    cv2.imwrite(str(output_path), vis)


def draw_matches(
    image1: np.ndarray,
    keypoints1: list[cv2.KeyPoint],
    image2: np.ndarray,
    keypoints2: list[cv2.KeyPoint],
    matches: list[cv2.DMatch],
    output_path: Path,
    max_draw: int = 80,
) -> None:
    """Save a feature-match visualisation."""
    ensure_dir(output_path.parent)
    matches_to_draw = sorted(matches, key=lambda m: m.distance)[:max_draw]
    vis = cv2.drawMatches(
        image1,
        keypoints1,
        image2,
        keypoints2,
        matches_to_draw,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    cv2.imwrite(str(output_path), vis)


def draw_epipolar_lines(
    image1: np.ndarray,
    image2: np.ndarray,
    pts1: np.ndarray,
    pts2: np.ndarray,
    F: np.ndarray,
    output_path: Path,
    max_lines: int = 20,
) -> None:
    """Save an epipolar-line visualisation.

    Draw sampled points from image1 on the left, and their corresponding
    epipolar lines plus matching points in image2 on the right.
    """
    n = min(max_lines, len(pts1), len(pts2)) # number of lines to draw by finding the minimum available matches and the max_lines limit.

    pts1_sample = pts1[:n]
    pts2_sample = pts2[:n]

    img1_rgb = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    img2_rgb = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

    height, width = image2.shape[:2]

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    ax1, ax2 = axes # plots the two images side by side, with keypoints on the left and epipolar lines on the right.
    ax1.imshow(img1_rgb)
    ax1.set_title("Image 1 points")
    ax1.axis("off")

    ax2.imshow(img2_rgb)
    ax2.set_title("Epipolar lines in image 2")
    ax2.axis("off")

    ax1.scatter(
        pts1_sample[:, 0],
        pts1_sample[:, 1],
        s=20,
        marker="o",
    )

    eps = 1e-12 # small epsilon to avoid division by 0 when computing line endpoints.

    for p1, p2 in zip(pts1_sample, pts2_sample):
        u, v = p1[:2]
        x1 = np.array([u, v, 1.0], dtype=float)

        a, b, c = F @ x1 # Finds each epipolar line in image 2 corresponding to the point in image 1, then computes two points on the line to draw it. 

        if abs(b) > eps: # line is not vertical, no division by 0 error.
            x_start = 0
            x_end = width - 1

            y_start = -c / b
            y_end = -(a * x_end + c) / b

            ax2.plot([x_start, x_end], [y_start, y_end])
        elif abs(a) > eps: # if line is vertical, just plot a vertical line at x = -c/a
            y_start = 0
            y_end = height - 1

            x = -c / a

            ax2.plot([x, x], [y_start, y_end])
        else:
            # Degenerate line; skip drawing it.
            continue

        ax2.scatter(
            p2[0],
            p2[1],
            s=20,
            marker="o",
        )

    ensure_dir(output_path.parent)
    plt.savefig(output_path)
    plt.close(fig)


def analyse_image_pair(
    image1_path: Path,
    image2_path: Path,
    output_dir: Path,
    max_features: int = 4000,
    ratio: float = 0.75,
    max_image_size: int | None = 1600,
    save_figures: bool = True,
    ransac_threshold: float = 1.0,
) -> PairAnalysis:
    """Run the full Week 2 analysis for one image pair."""
    # 1. Load both images
    image1 = load_image(image1_path, max_size=max_image_size)
    image2 = load_image(image2_path, max_size=max_image_size)

    # 2. Detect SIFT features
    kp1, desc1 = detect_sift_features(image1, max_features=max_features)
    kp2, desc2 = detect_sift_features(image2, max_features=max_features)

    features1 = ImageFeatures(image1_path, image1, kp1, desc1)
    features2 = ImageFeatures(image2_path, image2, kp2, desc2)

    return analyse_feature_pair(
        features1=features1,
        features2=features2,
        output_dir=output_dir,
        ratio=ratio,
        save_figures=save_figures,
        ransac_threshold=ransac_threshold,
    )


def analyse_feature_pair(
    features1: ImageFeatures,
    features2: ImageFeatures,
    output_dir: Path,
    ratio: float = 0.75,
    save_figures: bool = False,
    ransac_threshold: float = 1.0,
) -> PairAnalysis:
    """Run pair analysis using precomputed image features."""

    kp1, kp2 = features1.keypoints, features2.keypoints
    desc1, desc2 = features1.descriptors, features2.descriptors
    image1, image2 = features1.image, features2.image

    num_keypoints_1 = len(kp1)
    num_keypoints_2 = len(kp2)

    # 1. Count raw matches
    raw_count = count_raw_matches(desc1, desc2)

    # 2. Lowe-filtered matches
    matches = match_descriptors(desc1, desc2, ratio=ratio)
    filtered_count = len(matches)

    # 3. Convert filtered matches to point arrays
    if filtered_count > 0:
        pts1, pts2 = matched_keypoint_coords(kp1, kp2, matches)
    else:
        pts1 = np.empty((0, 2), dtype=np.float32)
        pts2 = np.empty((0, 2), dtype=np.float32)

    # 4. Estimate F with RANSAC
    F, mask = estimate_fundamental_ransac(pts1, pts2, threshold=ransac_threshold)

    if mask is not None:
        mask = mask.ravel().astype(bool)
    else:
        mask = np.zeros(filtered_count, dtype=bool)

    inlier_count = int(mask.sum())
    inlier_ratio = inlier_count / filtered_count if filtered_count > 0 else 0.0

    # Default epipolar error fields
    all_error_mean = None
    all_error_median = None
    inlier_error_mean = None
    inlier_error_median = None
    inlier_error_max = None

    # 5. Compute epipolar errors
    if F is not None and filtered_count > 0:
        all_errors = compute_epipolar_errors(F, pts1, pts2)

        if len(all_errors) > 0:
            all_error_mean = float(np.mean(all_errors))
            all_error_median = float(np.median(all_errors))

        if inlier_count > 0:
            inlier_errors = compute_epipolar_errors(F, pts1[mask], pts2[mask])

            if len(inlier_errors) > 0:
                inlier_error_mean = float(np.mean(inlier_errors))
                inlier_error_median = float(np.median(inlier_errors))
                inlier_error_max = float(np.max(inlier_errors))

    # 6. Save figures
    if save_figures:
        ensure_dir(output_dir)

        draw_keypoints(
            image1,
            kp1,
            output_dir / "keypoints_1.png",
        )

        draw_keypoints(
            image2,
            kp2,
            output_dir / "keypoints_2.png",
        )

        # Raw matches
        raw_matches = raw_descriptor_matches(desc1, desc2)

        draw_matches(
            image1,
            kp1,
            image2,
            kp2,
            raw_matches,
            output_dir / "matches_raw.png",
        )

        # Lowe-filtered matches
        draw_matches(
            image1,
            kp1,
            image2,
            kp2,
            matches,
            output_dir / "matches_filtered.png",
        )

        # RANSAC inlier matches
        inlier_matches = [
            m for m, keep in zip(matches, mask)
            if keep
        ]

        draw_matches(
            image1,
            kp1,
            image2,
            kp2,
            inlier_matches,
            output_dir / "matches_inliers.png",
        )

        # Epipolar lines
        if F is not None and inlier_count > 0:
            draw_epipolar_lines(
                image1,
                image2,
                pts1[mask],
                pts2[mask],
                F,
                output_dir / "epipolar_lines.png",
            )

    # 7. Return PairAnalysis
    return PairAnalysis(
        image_i=features1.path.name,
        image_j=features2.path.name,
        keypoints_i=num_keypoints_1,
        keypoints_j=num_keypoints_2,
        raw_matches=raw_count,
        filtered_matches=filtered_count,
        ransac_inliers=inlier_count,
        inlier_ratio=float(inlier_ratio),
        mean_epipolar_error_all=all_error_mean,
        median_epipolar_error_all=all_error_median,
        mean_epipolar_error_inliers=inlier_error_mean,
        median_epipolar_error_inliers=inlier_error_median,
        max_epipolar_error_inliers=inlier_error_max,
        fundamental_matrix=F.tolist() if F is not None else None,
    )

    """This avoids recomputing SIFT features during all-pairs dataset analysis.
    In dataset mode, save_figures is normally False, so this function should
    return metrics without creating an output folder for every image pair.
    """
    raise NotImplementedError("TODO: implement pair analysis from precomputed features")


def draw_match_graph(
    rows: list[dict],
    output_path: Path,
    min_inliers: int = 30,
) -> None:
    """Draw a match graph from pairwise metric rows.

    Edges with fewer than min_inliers are omitted to keep the graph readable.
    """
    import matplotlib.pyplot as plt

    ensure_dir(output_path.parent)

    nodes = sorted({row["image_i"] for row in rows} | {row["image_j"] for row in rows})
    edges = []
    for row in rows:
        inliers = int(row["ransac_inliers"])
        if inliers >= min_inliers:
            edges.append((row["image_i"], row["image_j"], inliers))

    plt.figure(figsize=(10, 8))
    if not nodes or not edges:
        plt.text(0.5, 0.5, "No edges above threshold", ha="center", va="center")
        plt.axis("off")
    else:
        radius = 1.0
        positions = {}
        for idx, node in enumerate(nodes):
            angle = 2 * math.pi * idx / len(nodes)
            positions[node] = (radius * math.cos(angle), radius * math.sin(angle))

        max_weight = max(weight for _, _, weight in edges)
        for image_i, image_j, weight in edges:
            x1, y1 = positions[image_i]
            x2, y2 = positions[image_j]
            width = 1.0 + 4.0 * (weight / max_weight)
            plt.plot([x1, x2], [y1, y2], color="#456990", linewidth=width, alpha=0.7)
            plt.text((x1 + x2) / 2, (y1 + y2) / 2, str(weight), fontsize=7)

        for node, (x, y) in positions.items():
            plt.scatter([x], [y], s=550, color="#d8e8ff", edgecolor="#456990", zorder=3)
            label = Path(node).stem
            plt.text(x, y, label, ha="center", va="center", fontsize=8, zorder=4)

        plt.axis("off")
        plt.axis("equal")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def select_top_initial_pairs(rows: list[dict], top_k: int = 3) -> list[dict]:
    """Select candidate Week 3 initial pairs from pairwise metrics.

    This starter version ranks by RANSAC inlier count first, then inlier ratio.
    Students should inspect the images too: the best numerical pair may have too
    little baseline for triangulation.
    """
    return sorted(
        rows,
        key=lambda row: (
            int(row["ransac_inliers"]),
            float(row["inlier_ratio"]),
            -float(row["median_epipolar_error_inliers"] or 1e9),
        ),
        reverse=True,
    )[:top_k]
