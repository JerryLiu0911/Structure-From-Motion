"""Shared pytest fixtures for the GF4 SfM test suite."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Synthetic image fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def grey_image_100() -> np.ndarray:
    """100×100 uint8 greyscale image with a simple gradient."""
    img = np.zeros((100, 100), dtype=np.uint8)
    for i in range(100):
        img[i, :] = i
    return img


@pytest.fixture()
def bgr_image_100() -> np.ndarray:
    """100×100 uint8 BGR colour image (gradient in green channel)."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    for i in range(100):
        img[i, :, 1] = i  # green channel
    return img


@pytest.fixture()
def tmp_image_dir(tmp_path: Path, bgr_image_100: np.ndarray) -> Path:
    """A temporary directory populated with 3 synthetic JPEG images."""
    import cv2

    image_dir = tmp_path / "images"
    image_dir.mkdir()
    for idx in range(3):
        # Add some random texture so SIFT can detect keypoints
        noise = np.random.default_rng(idx).integers(0, 255, (100, 100, 3), dtype=np.uint8)
        img = cv2.addWeighted(bgr_image_100, 0.5, noise, 0.5, 0)
        cv2.imwrite(str(image_dir / f"image_{idx:02d}.jpg"), img)
    return image_dir
