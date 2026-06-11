"""Command-line driver for GF4 Week 4 multi-view reconstruction.

This driver loads the completed Week 2 (feature detection / matching) and
Week 3 (two-view geometry) utilities, then uses the Week 4 tools in
`incremental_sfm.py` to build a multi-view reconstruction:

  1. load a pool of images and precompute SIFT features (Week 2),
  2. match every image pair once into a match cache (Week 2),
  3. select a wide-baseline initial pair using the Week 2 metrics CSV plus a
     triangulation-angle gate,
  4. bootstrap a two-view reconstruction and greedily register the remaining
     images by PnP, triangulating new points after each registration,
  5. write per-step metrics, a sparse point cloud (PLY), and figures.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

from incremental_sfm import IncrementalReconstruction
try:
    import view_cloud  # noqa: F401  kept importable as a CLI hint; not used in main()
except ModuleNotFoundError:
    view_cloud = None   # viewer deps (PyVista) are optional and not needed to reconstruct


DEFAULT_WEEK2_DIR = Path(__file__).resolve().parents[1] / "week2"
DEFAULT_WEEK3_DIR = Path(__file__).resolve().parents[1] / "week3"


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


def load_week3_module(week3_dir: Path):
    """Load the completed Week 3 two_view_utils.py by path."""
    module_path = Path(week3_dir) / "two_view_utils.py"
    if not module_path.exists():
        raise FileNotFoundError(f"Could not find Week 3 two_view_utils.py: {module_path}")

    spec = importlib.util.spec_from_file_location("week3_two_view_utils", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import Week 3 module from {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def read_metrics_csv(metrics_csv: Path) -> list[dict]:
    with open(metrics_csv, newline="") as handle:
        return list(csv.DictReader(handle))


def build_match_cache(features_by_id: dict, week2, ratio: float=0.8) -> dict: 
    """Match every image pair once: (i, j) with i < j -> [(kp_i, kp_j), ...]."""
    cache: dict[tuple[int, int], list[tuple[int, int]]] = {}  # Since week2 didn't export the matched keypoints, the cache structure maps pairs of image IDs to lists of matched keypoint index pairs.
    ids = sorted(features_by_id) # Get a sorted list of image IDs from the features_by_id dictionary
    for a in range(len(ids)): # Iterate over each image ID as the first element of the pair
        for b in range(a + 1, len(ids)): # Iterate over each image ID as the second element of the pair, ensuring that b > a to avoid duplicate pairs and self-matching
            i, j = ids[a], ids[b]
            matches = week2.match_descriptors(
                features_by_id[i].descriptors, features_by_id[j].descriptors, ratio=ratio
            )
            cache[(i, j)] = [(m.queryIdx, m.trainIdx) for m in matches]
    return cache


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GF4 Week 4 incremental multi-view reconstruction pipeline."
    )
    parser.add_argument("--metrics-csv", required=True, type=Path,
                        help="Week 2 pairwise_metrics.csv for the dataset.")
    parser.add_argument("--image-dir", required=True, type=Path,
                        help="Directory containing the dataset images.")
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="Directory where metrics and figures are written.")
    parser.add_argument("--week2-dir", type=Path, default=DEFAULT_WEEK2_DIR)
    parser.add_argument("--week3-dir", type=Path, default=DEFAULT_WEEK3_DIR)
    parser.add_argument("--max-images", type=int, default=200,
                        help="Maximum number of pool images to load.")
    parser.add_argument("--top-k", type=int, default=10,
                        help="How many top-inlier pairs to re-score for the seed.")
    parser.add_argument("--min-tri-angle", type=float, default=5.0,
                        help="Minimum median triangulation angle (deg) for the seed.")
    parser.add_argument("--min-inliers", type=int, default=100,
                        help="Minimum Week 2 RANSAC inliers to consider as a seed.")
    parser.add_argument("--min-corr", type=int, default=20,
                        help="Minimum 2D-3D correspondences to attempt registering an image.")
    parser.add_argument("--min-pnp-inliers", type=int, default=30,
                        help="Minimum PnP inliers (absolute floor) to accept a registration.")
    parser.add_argument("--min-pnp-ratio", type=float, default=0.3,
                        help="Minimum PnP inlier ratio (inliers / 2D-3D correspondences) to "
                             "accept a registration. Rejects drifted poses that clear the "
                             "absolute floor but explain only a small fraction of their matches.")
    parser.add_argument("--ratio", type=float, default=0.75)
    parser.add_argument("--ransac-threshold", type=float, default=1.0)
    parser.add_argument("--pnp-threshold", type=float, default=6.0)
    parser.add_argument("--confidence", type=float, default=0.999)
    parser.add_argument("--max-reprojection-error", type=float, default=2.5)
    parser.add_argument("--focal-length-px", type=float, default=None,
                        help="Override focal length in px. If unset, derived from "
                             "--focal-35mm-equiv and the resized image diagonal.")
    parser.add_argument("--focal-35mm-equiv", type=float, default=24.0,
                        help="35mm-equivalent focal length from EXIF (mm). Combined with the "
                             "full-frame diagonal (43.267mm) and the resized image diagonal "
                             "to recover focal length in pixels.")
    parser.add_argument("--max-features", type=int, default=4000)
    parser.add_argument("--max-image-size", type=int, default=1600)
    parser.add_argument("--retriangulate-interval", type=int, default=5,
                        help="Retriangulate every N registrations (0 disables).")

    args = parser.parse_args()
    if args.max_image_size == 0:
        args.max_image_size = None
    return args


def main() -> int:
    args = parse_args()

    week2 = load_week2_module(args.week2_dir)
    week3 = load_week3_module(args.week3_dir)
    output_dir = week3.ensure_dir(args.output_dir) # Re-use week3's ensure_dir to create the output directory if it doesn't exist

    pool_paths = week2.list_image_paths(args.image_dir, max_images=args.max_images)
   
    if len(pool_paths) < 3:
        raise ValueError(f"Need at least 3 pool images, found {len(pool_paths)}")
    
    features = week2.precompute_image_features(
        pool_paths, max_features=args.max_features, max_image_size=args.max_image_size
    )

    features_by_id = {idx: f for idx, f in enumerate(features)} # Creates a dictionary mapping image IDs to their corresponding feature data.
    name_to_id = {f.path.name: idx for idx, f in features_by_id.items()}
    print(f"Loaded {len(features)} pool images")

    # Assuming single camera, the intrinsics are shared and calculated based on image dimensions and the provided focal length parameters. The focal length in pixels is derived from the 35mm equivalent focal length and the resized image diagonal,
    # using the formula: focal_px = (focal_35mm_equiv / 43.267) * sqrt(w^2 + h^2), where 43.267mm is the diagonal of a full-frame sensor.
    h, w = features[0].image.shape[:2]
    focal_px = args.focal_length_px
    if focal_px is None:
        focal_px = args.focal_35mm_equiv * (w ** 2 + h ** 2) ** 0.5 / 43.267
    K = week3.make_camera_matrix(features[0].image.shape, focal_length_px=focal_px)
    print(f"Intrinsics: f={focal_px:.1f}px, principal=({(w - 1) / 2:.1f},{(h - 1) / 2:.1f}), "
          f"image {w}x{h}")
    print("Matching all image pairs...")
    pair_matches = build_match_cache(features_by_id, week2, args.ratio)

    engine = IncrementalReconstruction(K, features_by_id, pair_matches, week3)

    # Selects the initial seed pair. It reads the pairwise metrics from the provided CSV file, filters and ranks candidate pairs 
    # based on the number of RANSAC inliers, and then re-scores the top candidates using the actual two-view geometry to compute 
    # pose inliers and triangulation angles. The final selection is made using a combination of pose inliers and a minimum triangulation angle threshold to ensure a good baseline for reconstruction.
    rows = read_metrics_csv(args.metrics_csv)
    seed_i, seed_j, candidates = engine.select_initial_pair(
        rows,
        top_k=args.top_k, min_inliers=args.min_inliers,
        min_tri_angle=args.min_tri_angle,
        ransac_threshold=args.ransac_threshold, confidence=args.confidence,
    )
    chosen = next(
        c for c in candidates
        if name_to_id[c.image_i] == seed_i and name_to_id[c.image_j] == seed_j
    )
    print(
        f"Seed pair: {chosen.image_i} + {chosen.image_j} "
        f"(pose inliers={chosen.pose_inliers}, tri angle={chosen.median_tri_angle:.2f} deg)"
    )

    # 4. Bootstrap + greedy growth.
    n_seed = engine.initialise_two_view(
        seed_i, seed_j,
        ransac_threshold=args.ransac_threshold, confidence=args.confidence,
        max_reproj=args.max_reprojection_error,
    )
    print(f"Seed reconstruction: {n_seed} points from 2 cameras")

    metrics = engine.run(
        min_corr=args.min_corr,
        min_pnp_inliers=args.min_pnp_inliers,
        min_pnp_ratio=args.min_pnp_ratio,
        pnp_threshold=args.pnp_threshold,
        confidence=args.confidence,
        max_reproj=args.max_reprojection_error,
        retriangulate_interval=args.retriangulate_interval, # lets you run with in-loop retriangulation disabled (0) or every N registrations
    )

    # 5. Outputs.
    week3.save_csv(output_dir / "incremental_metrics.csv", [m.as_dict() for m in metrics])

    points, colours = engine.point_cloud()
    week3.write_ply(output_dir / "points3d.ply", points, colours)

    # Persist camera poses + intrinsics so the cloud + frusta can be re-viewed
    # (view_cloud.py) without re-running the reconstruction.
    cameras_json = {
        "K": np.asarray(K, dtype=np.float64).tolist(),
        "cameras": [
            {
                "name": name,
                "R": np.asarray(R, dtype=np.float64).tolist(),
                "t": np.asarray(t, dtype=np.float64).reshape(3).tolist(),
            }
            for name, R, t in engine.camera_list()
        ],
    }
    (output_dir / "cameras.json").write_text(json.dumps(cameras_json, indent=2))
    week3.plot_multi_view_reconstruction(
        points, colours, engine.camera_list(),
        output_dir / "multi_view_reconstruction.png",
    )

    errors = engine.reprojection_errors()
    registered_names = [self_name for self_name, _, _ in engine.camera_list()]
    rejected = [
        f.path.name for fid, f in features_by_id.items()
        if fid not in engine.registered
    ]

    print()
    print("Incremental reconstruction complete")
    print(f"  registered cameras : {len(engine.registered)} / {len(features)}")
    print(f"  sparse 3D points   : {len(engine.tracks)}")
    total_obs = sum(len(t.observations) for t in engine.tracks.values())
    if engine.tracks and engine.registered:
        print(f"  observations       : {total_obs}")
        print(f"  mean track length  : {total_obs / len(engine.tracks):.2f}  (observations per point)")
        print(f"  mean obs / image   : {total_obs / len(engine.registered):.1f}")
    if len(errors):
        print(f"  median reprojection error : {np.median(errors):.3f} px")
        print(f"  mean reprojection error   : {np.mean(errors):.3f} px")
    eb, ea = engine.errors_before_retri, engine.errors_after_retri
    if len(eb) and len(ea):
        # Same poses and the same point set on both sides of the final pass.
        # Shows that the change is attributable purely to retriangulation.
        # The win lands in the tail (p95 / sub-pixel fraction), not the median.
        print(f"  retriangulation (final pass): {engine.final_retri_updated} / "
              f"{len(engine.tracks)} points re-estimated")
        print(f"    reproj median : {np.median(eb):.3f} -> {np.median(ea):.3f} px")
        print(f"    reproj mean   : {np.mean(eb):.3f} -> {np.mean(ea):.3f} px")
        print(f"    reproj p95    : {np.percentile(eb, 95):.3f} -> {np.percentile(ea, 95):.3f} px")
        print(f"    obs under 1px : {np.mean(eb < 1):.1%} -> {np.mean(ea < 1):.1%}")
    print(f"  registered : {', '.join(registered_names)}")
    if rejected:
        print(f"  not registered : {', '.join(rejected)}")
        # Diagnostic: for each unregistered image, its 2D-3D foothold AND the PnP
        # inlier ratio it achieves against the FINAL cloud (read-only probe, not
        # committed). High footholds + low ratio => the ratio gate is the binding
        # constraint, i.e. the rejects' correspondences are mostly geometrically
        # inconsistent, not missing.
        footholds, ratios = [], []
        for fid in features_by_id:
            if fid in engine.registered:
                continue
            corr = engine.gather_2d3d(fid)
            footholds.append(corr["count"])
            r = 0.0
            if corr["count"] >= 6:
                try:
                    _, _, mask = week3.estimate_camera_pose_pnp(
                        corr["points3d"], corr["pts2d"], K,
                        threshold=args.pnp_threshold, confidence=args.confidence)
                    r = float(np.sum(mask)) / corr["count"]
                except Exception:
                    r = 0.0
            ratios.append(r)
        below = sum(1 for c in footholds if c < args.min_corr)
        clear = sum(1 for r in ratios if r >= args.min_pnp_ratio)
        print(f"  unregistered footholds (2D-3D vs final cloud): "
              f"median {int(np.median(footholds))}, max {max(footholds)}, "
              f"{below}/{len(footholds)} below min_corr={args.min_corr}")
        print(f"  unregistered PnP ratio (probe vs final cloud): "
              f"median {np.median(ratios):.2f}, max {max(ratios):.2f}, "
              f"{clear}/{len(ratios)} would clear min_pnp_ratio={args.min_pnp_ratio}")
        (output_dir / "reject_diag.json").write_text(json.dumps(
            {"reject_pnp_ratios": [round(r, 4) for r in ratios],
             "reject_footholds": footholds}))
    print(f"  wrote: {output_dir}")
    print(f"  view: python view_cloud.py --path {output_dir / 'points3d.ply'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
