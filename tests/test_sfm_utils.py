"""Tests for gf4.week2.sfm_utils utility functions.

Tests cover the completed helper functions (file I/O, image loading, CSV, etc.)
as well as skeleton tests for the TODO functions so CI fails descriptively once
students start implementing them.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from gf4.week2.sfm_utils import (
    IMAGE_EXTENSIONS,
    ImageFeatures,
    PairAnalysis,
    ensure_dir,
    list_image_paths,
    load_image,
    save_csv,
)


# ---------------------------------------------------------------------------
# ensure_dir
# ---------------------------------------------------------------------------


def test_ensure_dir_creates_nested(tmp_path: Path) -> None:
    target = tmp_path / "a" / "b" / "c"
    result = ensure_dir(target)
    assert result == target
    assert target.is_dir()


def test_ensure_dir_idempotent(tmp_path: Path) -> None:
    ensure_dir(tmp_path)  # already exists — should not raise
    assert tmp_path.is_dir()


# ---------------------------------------------------------------------------
# list_image_paths
# ---------------------------------------------------------------------------


def test_list_image_paths_returns_sorted(tmp_image_dir: Path) -> None:
    paths = list_image_paths(tmp_image_dir)
    assert paths == sorted(paths)


def test_list_image_paths_count(tmp_image_dir: Path) -> None:
    paths = list_image_paths(tmp_image_dir)
    assert len(paths) == 3


def test_list_image_paths_max_images(tmp_image_dir: Path) -> None:
    paths = list_image_paths(tmp_image_dir, max_images=2)
    assert len(paths) == 2


def test_list_image_paths_missing_dir_raises() -> None:
    with pytest.raises(FileNotFoundError):
        list_image_paths(Path("/nonexistent/directory"))


def test_list_image_paths_empty_dir_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="No images found"):
        list_image_paths(tmp_path)


def test_list_image_paths_extensions(tmp_image_dir: Path) -> None:
    """All returned paths have a recognised image extension."""
    for path in list_image_paths(tmp_image_dir):
        assert path.suffix.lower() in IMAGE_EXTENSIONS


# ---------------------------------------------------------------------------
# load_image
# ---------------------------------------------------------------------------


def test_load_image_shape(tmp_image_dir: Path) -> None:
    path = list_image_paths(tmp_image_dir)[0]
    img = load_image(path)
    assert img.ndim == 3
    assert img.shape[2] == 3  # BGR


def test_load_image_resize(tmp_image_dir: Path) -> None:
    path = list_image_paths(tmp_image_dir)[0]
    img = load_image(path, max_size=50)
    assert max(img.shape[:2]) <= 50


def test_load_image_missing_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_image(Path("/nonexistent/image.jpg"))


# ---------------------------------------------------------------------------
# save_csv
# ---------------------------------------------------------------------------


def test_save_csv_writes_header_and_rows(tmp_path: Path) -> None:
    rows = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
    out = tmp_path / "out.csv"
    save_csv(out, rows)
    with out.open(newline="") as f:
        reader = list(csv.DictReader(f))
    assert reader == [{"a": "1", "b": "2"}, {"a": "3", "b": "4"}]


def test_save_csv_empty(tmp_path: Path) -> None:
    out = tmp_path / "empty.csv"
    save_csv(out, [])
    assert out.read_text() == ""


def test_save_csv_creates_parent_dirs(tmp_path: Path) -> None:
    out = tmp_path / "nested" / "dir" / "out.csv"
    save_csv(out, [{"x": 1}])
    assert out.exists()


# ---------------------------------------------------------------------------
# PairAnalysis dataclass helpers
# ---------------------------------------------------------------------------


def _make_pair_analysis(**kwargs) -> PairAnalysis:
    defaults = dict(
        image_i="a.jpg",
        image_j="b.jpg",
        keypoints_i=100,
        keypoints_j=120,
        raw_matches=80,
        filtered_matches=50,
        ransac_inliers=40,
        inlier_ratio=0.8,
        mean_epipolar_error_all=None,
        median_epipolar_error_all=None,
        mean_epipolar_error_inliers=None,
        median_epipolar_error_inliers=None,
        max_epipolar_error_inliers=None,
        fundamental_matrix=None,
    )
    defaults.update(kwargs)
    return PairAnalysis(**defaults)


def test_pair_analysis_as_dict_keys() -> None:
    pa = _make_pair_analysis()
    d = pa.as_dict()
    assert "ransac_inliers" in d
    assert "fundamental_matrix" in d


def test_pair_analysis_csv_dict_fundamental_matrix_none() -> None:
    pa = _make_pair_analysis()
    assert pa.csv_dict()["fundamental_matrix"] == ""


def test_pair_analysis_csv_dict_fundamental_matrix_value() -> None:
    F = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    pa = _make_pair_analysis(fundamental_matrix=F)
    assert pa.csv_dict()["fundamental_matrix"] != ""


# ---------------------------------------------------------------------------
# TODO function skeletons — these fail with NotImplementedError until
# students complete the implementations.
# ---------------------------------------------------------------------------


def test_detect_sift_features_not_implemented(bgr_image_100: np.ndarray) -> None:
    from gf4.week2.sfm_utils import detect_sift_features

    with pytest.raises(NotImplementedError):
        detect_sift_features(bgr_image_100)


def test_match_descriptors_not_implemented() -> None:
    from gf4.week2.sfm_utils import match_descriptors

    dummy = np.zeros((10, 128), dtype=np.float32)
    with pytest.raises(NotImplementedError):
        match_descriptors(dummy, dummy)


def test_estimate_fundamental_ransac_not_implemented() -> None:
    from gf4.week2.sfm_utils import estimate_fundamental_ransac

    pts = np.random.default_rng(0).random((20, 2)).astype(np.float32)
    with pytest.raises(NotImplementedError):
        estimate_fundamental_ransac(pts, pts)
