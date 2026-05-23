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