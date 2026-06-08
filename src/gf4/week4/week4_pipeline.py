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
from pathlib import Path
import sys

import numpy as np

from incremental_sfm import (
    IncrementalReconstruction,
    PairCandidate,
    choose_initial_pair,
)


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


def build_match_cache(features_by_id: dict, week2, ratio: float) -> dict:
    """Match every image pair once: (i, j) with i < j -> [(kp_i, kp_j), ...]."""
    cache: dict[tuple[int, int], list[tuple[int, int]]] = {}
    ids = sorted(features_by_id)
    for a in range(len(ids)):
        for b in range(a + 1, len(ids)):
            i, j = ids[a], ids[b]
            matches = week2.match_descriptors(
                features_by_id[i].descriptors, features_by_id[j].descriptors, ratio=ratio
            )
            cache[(i, j)] = [(m.queryIdx, m.trainIdx) for m in matches]
    return cache


def select_seed(
    engine: IncrementalReconstruction,
    metrics_csv: Path,
    name_to_id: dict,
    *,
    top_k: int,
    min_inliers: int,
    min_tri_angle: float,
    ransac_threshold: float,
    confidence: float,
) -> tuple[int, int, list[PairCandidate]]:
    """Choose a seed pair from the Week 2 CSV using the triangulation-angle gate."""
    rows = read_metrics_csv(metrics_csv)
    shortlisted = [r for r in rows if int(r["ransac_inliers"]) >= min_inliers]
    shortlisted.sort(key=lambda r: int(r["ransac_inliers"]), reverse=True)
    shortlisted = shortlisted[:top_k]

    candidates: list[PairCandidate] = []
    for row in shortlisted:
        if row["image_i"] not in name_to_id or row["image_j"] not in name_to_id:
            continue
        i = name_to_id[row["image_i"]]
        j = name_to_id[row["image_j"]]
        geom = engine.two_view_geometry(
            i, j, ransac_threshold=ransac_threshold, confidence=confidence
        )
        if geom is None:
            continue
        pose_inliers, n_pts, angle = geom
        candidates.append(
            PairCandidate(
                image_i=row["image_i"],
                image_j=row["image_j"],
                ransac_inliers=int(row["ransac_inliers"]),
                pose_inliers=pose_inliers,
                triangulated_points=n_pts,
                median_tri_angle=angle,
            )
        )

    if not candidates:
        raise ValueError("No usable initial-pair candidates from the CSV / pool")

    chosen = choose_initial_pair(candidates, min_tri_angle=min_tri_angle)
    return name_to_id[chosen.image_i], name_to_id[chosen.image_j], candidates


def render_point_cloud(
    points: np.ndarray,
    colours: np.ndarray,
    *,
    screenshot_path: Path | None = None,
    show: bool = False,
    point_size: float = 3.0,
) -> None:
    """Visualise the sparse cloud with PyVista (Open3D-style interactive view).

    Open3D has no Python 3.13 wheel, so PyVista (VTK) is used instead. Colours
    are stored RGB in [0, 255] (Week 3 `sample_point_colours` already swaps
    BGR->RGB), so they only need scaling to [0, 1]. Imported lazily so the rest
    of the pipeline runs even if PyVista is absent.
    """
    import pyvista as pv

    points = np.asarray(points, dtype=np.float64)
    if len(points) == 0:
        raise ValueError("No 3D points to visualise")

    rgb = np.asarray(colours, dtype=np.float64)
    if rgb.size and rgb.max() > 1.0:
        rgb = rgb / 255.0

    plotter = pv.Plotter(off_screen=not show)
    plotter.add_points(
        pv.PolyData(points),
        scalars=rgb if rgb.size else None,
        rgb=rgb.size > 0,
        point_size=point_size,
        render_points_as_spheres=True,
    )
    plotter.add_axes()
    plotter.set_background("white")
    if screenshot_path is not None:
        plotter.screenshot(str(screenshot_path))
    if show:
        plotter.show()
    plotter.close()


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
    parser.add_argument("--max-images", type=int, default=20,
                        help="Maximum number of pool images to load.")
    parser.add_argument("--top-k", type=int, default=10,
                        help="How many top-inlier pairs to re-score for the seed.")
    parser.add_argument("--min-tri-angle", type=float, default=5.0,
                        help="Minimum median triangulation angle (deg) for the seed.")
    parser.add_argument("--min-inliers", type=int, default=100,
                        help="Minimum Week 2 RANSAC inliers to consider as a seed.")
    parser.add_argument("--min-corr", type=int, default=20,
                        help="Minimum 2D-3D correspondences to attempt registering an image.")
    parser.add_argument("--min-pnp-inliers", type=int, default=15,
                        help="Minimum PnP inliers to accept a registration.")
    parser.add_argument("--ratio", type=float, default=0.75)
    parser.add_argument("--ransac-threshold", type=float, default=1.0)
    parser.add_argument("--pnp-threshold", type=float, default=6.0)
    parser.add_argument("--confidence", type=float, default=0.999)
    parser.add_argument("--max-reprojection-error", type=float, default=4.0)
    parser.add_argument("--focal-length-px", type=float, default=None)
    parser.add_argument("--max-features", type=int, default=4000)
    parser.add_argument("--max-image-size", type=int, default=1600)
    parser.add_argument("--view", action="store_true",
                        help="Open an interactive PyVista window of the point cloud "
                             "(needs a display; DISPLAY is set under WSLg).")
    parser.add_argument("--no-screenshot", action="store_true",
                        help="Skip writing the PyVista point-cloud screenshot.")

    args = parser.parse_args()
    if args.max_image_size == 0:
        args.max_image_size = None
    return args


def main() -> int:
    args = parse_args()

    week2 = load_week2_module(args.week2_dir)
    week3 = load_week3_module(args.week3_dir)
    output_dir = week3.ensure_dir(args.output_dir)

    # 1. Pool of images + features.
    pool_paths = week2.list_image_paths(args.image_dir, max_images=args.max_images)
    if len(pool_paths) < 3:
        raise ValueError(f"Need at least 3 pool images, found {len(pool_paths)}")
    features = week2.precompute_image_features(
        pool_paths, max_features=args.max_features, max_image_size=args.max_image_size
    )
    features_by_id = {idx: f for idx, f in enumerate(features)}
    name_to_id = {f.path.name: idx for idx, f in features_by_id.items()}
    print(f"Loaded {len(features)} pool images")

    # 2. Shared intrinsics + all-pairs match cache.
    K = week3.make_camera_matrix(features[0].image.shape, focal_length_px=args.focal_length_px)
    print("Matching all image pairs...")
    pair_matches = build_match_cache(features_by_id, week2, args.ratio)

    engine = IncrementalReconstruction(K, features_by_id, pair_matches, week3)

    # 3. Seed selection (triangulation-angle gate).
    seed_i, seed_j, candidates = select_seed(
        engine, args.metrics_csv, name_to_id,
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
        pnp_threshold=args.pnp_threshold,
        confidence=args.confidence,
        max_reproj=args.max_reprojection_error,
    )

    # 5. Outputs.
    week3.save_csv(output_dir / "incremental_metrics.csv", [m.as_dict() for m in metrics])

    points, colours = engine.point_cloud()
    week3.write_ply(output_dir / "points3d.ply", points, colours)
    week3.plot_multi_view_reconstruction(
        points, colours, engine.camera_list(),
        output_dir / "multi_view_reconstruction.png",
    )

    if not args.no_screenshot or args.view:
        screenshot = None if args.no_screenshot else output_dir / "point_cloud_pyvista.png"
        try:
            render_point_cloud(points, colours, screenshot_path=screenshot, show=args.view)
        except Exception as exc:  # PyVista missing or no GL context -- non-fatal
            print(f"  (point-cloud view skipped: {exc})")

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
    if len(errors):
        print(f"  median reprojection error : {np.median(errors):.3f} px")
        print(f"  mean reprojection error   : {np.mean(errors):.3f} px")
    print(f"  registered : {', '.join(registered_names)}")
    if rejected:
        print(f"  not registered : {', '.join(rejected)}")
    print(f"  wrote: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
