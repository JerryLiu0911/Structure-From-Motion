"""Command-line driver for GF4 Week 2 pairwise matching experiments.

This script supports two modes:

1. Pair mode:
   python week2/week2_pipeline.py --image1 a.jpg --image2 b.jpg --output-dir out/

2. Dataset mode:
   python week2/week2_pipeline.py --image-dir images/ --output-dir out/ --max-images 20

The core computer-vision functions live in sfm_utils.py and are intentionally
left as TODOs for students to complete.
"""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path
import sys

from sfm_utils import (
    analyse_image_pair,
    analyse_feature_pair,
    draw_match_graph,
    ensure_dir,
    list_image_paths,
    precompute_image_features,
    save_csv,
    select_top_initial_pairs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GF4 Week 2 feature matching and epipolar geometry pipeline."
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--image-dir",
        type=Path,
        help="Directory of images for all-pairs dataset analysis.",
    )
    input_group.add_argument(
        "--image1",
        type=Path,
        help="First image for pair analysis. Must be used with --image2.",
    )

    parser.add_argument(
        "--image2",
        type=Path,
        help="Second image for pair analysis.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory where metrics and figures will be written.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=20,
        help="Maximum number of images to load in dataset mode.",
    )
    parser.add_argument(
        "--max-image-size",
        type=int,
        default=1600,
        help="Resize images so their long edge is at most this size. Use 0 to disable.",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=4000,
        help="Maximum number of SIFT features to retain per image.",
    )
    parser.add_argument(
        "--ratio",
        type=float,
        default=0.75,
        help="Lowe ratio-test threshold.",
    )
    parser.add_argument(
        "--min-graph-inliers",
        type=int,
        default=30,
        help="Minimum RANSAC inliers for drawing an edge in the match graph.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of candidate initial image pairs to save.",
    )

    args = parser.parse_args()

    if args.image1 is not None and args.image2 is None:
        parser.error("--image2 is required when using --image1")
    if args.image_dir is not None and args.image2 is not None:
        parser.error("--image2 should only be used with --image1")
    if args.max_images < 2:
        parser.error("--max-images must be at least 2")
    if args.max_features < 1:
        parser.error("--max-features must be positive")
    if not 0.0 < args.ratio < 1.0:
        parser.error("--ratio must be between 0 and 1")

    if args.max_image_size == 0:
        args.max_image_size = None

    return args


def run_pair_mode(args: argparse.Namespace) -> None:
    output_dir = ensure_dir(args.output_dir)
    analysis = analyse_image_pair(
        image1_path=args.image1,
        image2_path=args.image2,
        output_dir=output_dir,
        max_features=args.max_features,
        ratio=args.ratio,
        max_image_size=args.max_image_size,
        save_figures=True,
    )

    metrics = analysis.csv_dict()
    save_csv(output_dir / "pair_metrics.csv", [metrics])

    print("Pair analysis complete")
    print(f"  image 1: {metrics['image_i']}")
    print(f"  image 2: {metrics['image_j']}")
    print(f"  raw matches: {metrics['raw_matches']}")
    print(f"  filtered matches: {metrics['filtered_matches']}")
    print(f"  RANSAC inliers: {metrics['ransac_inliers']}")
    print(f"  inlier ratio: {metrics['inlier_ratio']:.3f}")
    print(f"  wrote: {output_dir}")


def run_dataset_mode(args: argparse.Namespace) -> None:
    output_dir = ensure_dir(args.output_dir)
    image_paths = list_image_paths(args.image_dir, max_images=args.max_images)

    print(f"Loaded {len(image_paths)} images")
    print(f"Running all-pairs analysis for {len(image_paths) * (len(image_paths) - 1) // 2} pairs")

    rows = []
    features = precompute_image_features(
        image_paths,
        max_features=args.max_features,
        max_image_size=args.max_image_size,
    )

    for features1, features2 in combinations(features, 2):
        pair_name = f"{features1.path.stem}__{features2.path.stem}"
        pair_output_dir = output_dir / "pairs" / pair_name
        print(f"Analysing {features1.path.name} <-> {features2.path.name}")

        analysis = analyse_feature_pair(
            features1=features1,
            features2=features2,
            output_dir=pair_output_dir,
            ratio=args.ratio,
            save_figures=False,
        )
        rows.append(analysis.csv_dict())

    save_csv(output_dir / "pairwise_metrics.csv", rows)

    top_pairs = select_top_initial_pairs(rows, top_k=args.top_k)
    save_csv(output_dir / "top_initial_pairs.csv", top_pairs)

    draw_match_graph(
        rows,
        output_dir / "match_graph.png",
        min_inliers=args.min_graph_inliers,
    )

    print("Dataset analysis complete")
    print(f"  pairwise metrics: {output_dir / 'pairwise_metrics.csv'}")
    print(f"  top initial pairs: {output_dir / 'top_initial_pairs.csv'}")
    print(f"  match graph: {output_dir / 'match_graph.png'}")


def main() -> int:
    args = parse_args()

    try:
        if args.image1 is not None:
            run_pair_mode(args)
        else:
            run_dataset_mode(args)
    except NotImplementedError as exc:
        print(f"\nStarter-code TODO reached: {exc}", file=sys.stderr)
        print("Complete the relevant function in week2/sfm_utils.py and run again.", file=sys.stderr)
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
