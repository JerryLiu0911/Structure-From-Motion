"""Interactive (or off-screen) viewer for a sparse SfM point cloud.

Reads a PLY written by the Week 4 pipeline and renders it with PyVista. By
default it clips the small fraction of far low-parallax outliers that otherwise
force the camera to zoom out and crush the subject -- the clip is display-only
and never modifies the PLY.

Examples
--------
    # interactive window, default 5% outlier clip
    python view_cloud.py ../../../out/week4-doge-fixed/points3d.ply

    # show every point (no clip), bigger spheres
    python view_cloud.py path/to/points3d.ply --clip 100 --point-size 6

    # headless render to a file (no display needed)
    python view_cloud.py path/to/points3d.ply --screenshot cloud.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pyvista as pv


DEFAULT_PLY = Path(__file__).resolve().parents[3] / "out" / "week4-doge-fixed" / "points3d.ply"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--path", required=True, type=Path, nargs="?", default=DEFAULT_PLY,
                        help=f"Point-cloud PLY to view (default: {DEFAULT_PLY}).")
    parser.add_argument("--clip", type=float, default=95.0,
                        help="Keep points within this percentile of distance from the "
                             "cloud's median centre. 100 keeps everything. (default: 95)")
    parser.add_argument("--point-size", type=float, default=2.0,
                        help="Sphere size in pixels (default: 2.0).")
    parser.add_argument("--background", default="white", help="Background colour (default: white).")
    parser.add_argument("--screenshot", type=Path, default=None,
                        help="Render off-screen to this image instead of opening a window.")
    parser.add_argument("--camera", default="iso", choices=["iso", "xy", "xz", "yz"],
                        help="Initial camera position (default: iso).")
    return parser.parse_args()


def load_cloud(ply: Path, clip_percentile: float) -> pv.PolyData:
    """Read the PLY and (optionally) drop far outliers for display."""
    if not ply.exists():
        raise FileNotFoundError(f"Point cloud not found: {ply}")
    mesh = pv.read(ply)
    if clip_percentile >= 100.0 or mesh.n_points == 0:
        return mesh
    dist = np.linalg.norm(mesh.points - np.median(mesh.points, axis=0), axis=1)
    keep = dist < np.percentile(dist, clip_percentile)
    return mesh.extract_points(keep)


def main() -> int:
    args = parse_args()
    off_screen = args.screenshot is not None
    pv.OFF_SCREEN = off_screen

    mesh = pv.read(args.path)
    n_total = mesh.n_points
    cloud = load_cloud(args.path, args.clip)
    print(f"{args.path.name}: showing {cloud.n_points} / {n_total} points "
          f"(clip={args.clip}th pct)")

    plotter = pv.Plotter(off_screen=off_screen, window_size=[1400, 1000])
    plotter.add_points(
        cloud,
        rgb="RGB" in cloud.point_data,
        point_size=args.point_size,
        render_points_as_spheres=True,
    )
    plotter.set_background(args.background)
    plotter.add_axes()
    plotter.camera_position = args.camera

    if off_screen:
        plotter.screenshot(str(args.screenshot))
        print(f"wrote {args.screenshot}")
    else:
        plotter.show()
    plotter.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
