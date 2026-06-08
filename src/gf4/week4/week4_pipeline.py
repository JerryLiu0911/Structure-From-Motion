import cv2
import open3d as o3d
import matplotlib
import numpy as np
import argparse
import importlib.util
import sys
import increment_sfm
from pathlib import Path

DEFAULT_WEEK2_DIR = Path(__file__).resolve().parents[1] / "week2"


def load_week2_module(week2_dir: Path):
    """Load the completed Week 2 sfm_utils.py by path."""
    module_path = Path(week2_dir) / "sfm_utils.py"
    if not module_path.exists():
        raise FileNotFoundError(f"Could not find Week 2 sfm_utils.py: {module_path}")

    spec = importlib.util.spec_from_file_location("week2_sfm_utils", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import Week 2 module from {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module

DEFAULT_WEEK3_DIR = Path(__file__).resolve().parents[1] / "week3"


def load_week3_module(week3_dir: Path):
    """Load the completed Week 3 sfm_utils.py by path."""
    module_path = Path(week3_dir) / "sfm_utils.py"
    if not module_path.exists():
        raise FileNotFoundError(f"Could not find Week 3 sfm_utils.py: {module_path}")

    spec = importlib.util.spec_from_file_location("two_views_utils", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import Week 3 module from {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module

def run(dataset):
    """Run the full SfM pipeline on the provided dataset."""
    # Load Week 2 utilities
    week2_utils = load_week2_module(DEFAULT_WEEK2_DIR)
    # Load Week 3 utilities
    week3_utils = load_week3_module(DEFAULT_WEEK3_DIR)
    
    # Step 1: Feature detection and matching
    matches = week2_utils.detect_and_match_features(dataset.images)
    
    # Step 2: Estimate relative pose between image pairs
    relative_poses = week2_utils.estimate_relative_poses(matches, dataset.intrinsics)
    
    # Step 3: Incremental reconstruction
    reconstruction = week3_utils.incremental_reconstruction(relative_poses, matches, dataset.intrinsics)
    
    return reconstruction

def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full SfM pipeline on the provided dataset.")
    parser.add_argument("dataset_path", type=str, help="Path to the dataset directory containing images and camera intrinsics.")
    args = parser.parse_args()
    
    # Load dataset (images, intrinsics, etc.)
    dataset = increment_sfm.load_dataset(args.dataset_path)
    
    # Run the SfM pipeline
    reconstruction = increment_sfm.run_pipeline(dataset)

    # Save or visualize results
    increment_sfm.save_reconstruction(reconstruction, args.dataset_path)