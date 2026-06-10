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
import json
from pathlib import Path

import numpy as np
import pyvista as pv


DEFAULT_PLY = Path(__file__).resolve().parents[3] / "out" / "week4-doge-fixed" / "points3d.ply"


def _camera_frustum(R: np.ndarray, t: np.ndarray, K: np.ndarray, depth: float):
    """Wireframe frustum (apex + 4 image-plane corners) for one camera, in WORLD
    coordinates.

    Pose is world->camera (`x_cam = R x + t`), so a camera-frame point maps back
    as `x_world = R^T (x_cam - t)` and the centre is `C = -R^T t`. The four image
    corners are back-projected through `K` to the given `depth`, giving a pyramid
    whose aspect/field-of-view matches the real camera. Returns
    `(points (5, 3), lines)` where `lines` is the VTK connectivity array (apex to
    each corner, plus the image-plane rectangle).
    """
    R = np.asarray(R, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64).reshape(3)
    K = np.asarray(K, dtype=np.float64)

    C = -R.T @ t
    W, H = 2.0 * K[0, 2], 2.0 * K[1, 2]               # principal point is auto-centred
    corners_px = np.array([[0, 0, 1], [W, 0, 1], [W, H, 1], [0, H, 1]], dtype=np.float64)
    dirs = (np.linalg.inv(K) @ corners_px.T).T        # ray directions in camera frame (z>0)
    corner_cam = dirs / dirs[:, 2:3] * depth          # each corner placed at depth `depth`
    corners_world = (R.T @ (corner_cam - t).T).T      # x_world = R^T (x_cam - t)

    pts = np.vstack([C, corners_world])               # 0 = apex, 1..4 = image-plane corners
    segs = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (2, 3), (3, 4), (4, 1)]
    lines = np.hstack([[2, a, b] for a, b in segs]).astype(np.int64)
    return pts, lines


def add_camera_frusta(plotter, cameras: list, K: np.ndarray, depth: float, *,
                      draw_trajectory: bool = True, colour: str = "red") -> None:
    """Draw each camera as a wireframe frustum + centre sphere into an existing
    PyVista plotter.

    `cameras` is `[(name, R, t), ...]` (the decoded `cameras.json`); `depth` is
    the frustum size in world units. When `draw_trajectory`, the centres are
    joined in list order, so the path tears visibly wherever an unregistered
    image is skipped.
    """
    centres = []
    for _name, R, t in cameras:
        pts, lines = _camera_frustum(R, t, K, depth)
        plotter.add_mesh(pv.PolyData(pts, lines=lines), color=colour, line_width=1)
        centres.append(pts[0])
    centres = np.asarray(centres, dtype=np.float64)
    if len(centres):
        plotter.add_points(pv.PolyData(centres), color=colour, point_size=8,
                           render_points_as_spheres=True)
    if draw_trajectory and len(centres) > 1:
        plotter.add_mesh(pv.lines_from_points(centres), color="black", line_width=1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--path", required=True, type=Path, nargs="?", default=DEFAULT_PLY,
                        help=f"Point-cloud PLY to view (default: {DEFAULT_PLY}).")
    parser.add_argument("--clip", type=float, default=95.0,
                        help="Keep points within this percentile of distance from the "
                             "cloud's median centre. 100 keeps everything. (default: 95)")
    parser.add_argument("--point-size", type=float, default=1.0,
                        help="Sphere size in pixels (default: 1.0).")
    parser.add_argument("--background", default="white", help="Background colour (default: white).")
    parser.add_argument("--screenshot", type=Path, default=None,
                        help="Render off-screen to this image instead of opening a window.")
    parser.add_argument("--camera", default="iso", choices=["iso", "xy", "xz", "yz"],
                        help="Initial camera position (default: iso).")
    parser.add_argument("--cameras", type=Path, default=None,
                        help="cameras.json (poses + K) to draw as frusta. "
                             "Default: cameras.json beside the PLY, if present.")
    parser.add_argument("--no-cameras", action="store_true",
                        help="Do not draw camera frusta even if cameras.json exists.")
    parser.add_argument("--camera-scale", type=float, default=0.04,
                        help="Frustum size as a fraction of the (clipped) cloud's "
                             "bounding-box diagonal (default: 0.04).")
    return parser.parse_args()


def load_cameras(path: Path):
    """Read a pipeline cameras.json into `([(name, R, t), ...], K)`.

    Returns `(None, None)` if the file is missing, so the viewer degrades to a
    points-only render.
    """
    if path is None or not path.exists():
        return None, None
    data = json.loads(path.read_text())
    K = np.asarray(data["K"], dtype=np.float64)
    cams = [
        (c["name"], np.asarray(c["R"], dtype=np.float64), np.asarray(c["t"], dtype=np.float64))
        for c in data["cameras"]
    ]
    return cams, K


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
    if not args.no_cameras:
        cameras_path = args.cameras or (Path(args.path).parent / "cameras.json")
        cams, K = load_cameras(cameras_path)
        if cams:
            depth = args.camera_scale * cloud.length   # scale frusta to the shown cloud
            add_camera_frusta(plotter, cams, K, depth)
            print(f"drew {len(cams)} camera frusta from {cameras_path.name}")
        elif args.cameras is not None:
            print(f"  (no cameras loaded from {cameras_path})")

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
