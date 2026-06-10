# Week 4 Pipeline — Implementation Notes

Detailed description of how the GF4 Week 4 multi-view Structure-from-Motion
pipeline is implemented. For the *assignment brief* see [README.md](README.md);
this document covers the *code*.

---

## 1. Goal and scope

Week 3 reconstructs **exactly three images**: an initial pair (images 1 & 2)
plus one extra image (image 3) registered against that pair by PnP. Week 4
generalises this to an **incremental reconstruction over a pool of images**
(here, 20 sequential `partial_doge_images` frames):

1. select a well-conditioned initial pair,
2. bootstrap a two-view reconstruction,
3. greedily register the remaining images one at a time,
4. triangulate new structure after each registration,
5. write metrics, a point cloud, and figures.

No bundle adjustment, and **no COLMAP** — COLMAP is only an external baseline for
the report, never used by this pipeline.

---

## 2. File layout

The implementation mirrors the Week 2 / Week 3 split of *pure tools* vs.
*driver*:

| File | Role |
|------|------|
| [`incremental_sfm.py`](incremental_sfm.py) | **Engine / tools.** Data structures + geometry + the greedy loop. No file I/O, no module loading. The Week 3 geometry module is *injected* by the driver. |
| [`week4_pipeline.py`](week4_pipeline.py) | **Driver.** Loads Week 2 / Week 3 utilities, loads images, builds the match cache, selects the seed, runs the engine, and writes outputs. |

### Reused code

- **Week 2 — [`sfm_utils.py`](../week2/sfm_utils.py):** `list_image_paths`,
  `precompute_image_features`, `match_descriptors`, and the `ImageFeatures`
  dataclass (`path`, `image`, `keypoints`, `descriptors`).
- **Week 3 — [`two_view_utils.py`](../week3/two_view_utils.py):**
  `make_camera_matrix`, `estimate_essential_matrix`, `recover_relative_pose`,
  `triangulate_points`, `compute_reprojection_errors`,
  `filter_reconstructed_points`, `estimate_camera_pose_pnp`,
  `sample_point_colours`, `project_points`, `plot_multi_view_reconstruction`,
  `write_ply`, `save_csv`, `ensure_dir`.

Both modules are loaded by **file path** with `importlib` (see
`load_week2_module` / `load_week3_module`), exactly as `week3_pipeline.py` does.
This avoids packaging issues from the modules' direct sibling imports.

---

## 3. Core data structures (`incremental_sfm.py`)

The central idea is a **persistent point map** that links every 3D point to the
2D observations that created it. Week 3 got away with loose index arrays
(`kept_image1_indices`) because it only ever had three images; that does not
scale, so Week 4 replaces it with:

### `Track`
One reconstructed 3D point.
```python
point_id: int
xyz: np.ndarray            # (3,)
colour: np.ndarray         # (3,) uint8, RGB
observations: dict[int, int]   # image_id -> keypoint_idx
```

### `RegisteredImage`
A camera that has been placed in the reconstruction frame.
```python
image_id: int
R: np.ndarray              # (3, 3) world->camera rotation
t: np.ndarray              # (3, 1) translation
kpidx_to_point: dict[int, int]   # keypoint_idx -> point_id
```
`kpidx_to_point` is the key to scaling: it answers *"does this keypoint already
correspond to a known 3D point?"* in **O(1)**, which is what makes greedy
next-image scoring cheap.

### `StepMetrics`
One incremental registration step (report evidence): `step`, `image`,
`n_2d3d`, `pnp_inliers`, `new_points`, `n_registered`, `n_points`.

### `IncrementalReconstruction`
The stateful engine. Constructed with:
```python
IncrementalReconstruction(K, features, pair_matches, geom)
```
- `K` — shared 3×3 intrinsics (single-camera assumption; all images same
  resolution after resizing).
- `features` — `{image_id: ImageFeatures}`.
- `pair_matches` — `{(i, j): [(kp_i, kp_j), ...]}` for `i < j` (Lowe-filtered
  matches, precomputed by the driver).
- `geom` — the loaded Week 3 `two_view_utils` module.

Mutable state: `registered`, `tracks`, `next_point_id`, `_last_pnp_inliers`.

---

## 4. Coordinate and matching conventions

- **Poses are world→camera:** a 3D point `X` projects in image *i* as
  `x ∝ K (R_i X + t_i)`, with points as **column vectors**. Camera *i*'s centre
  in world coordinates is `C_i = -R_iᵀ t_i`.
- **The first seed camera is the origin** (`R = I`, `t = 0`); everything else is
  expressed relative to it. The whole reconstruction is therefore **up to an
  unknown global scale**.
- **Matches are orientation-normalised.** `pair_matches` stores tuples as
  `(kp_i, kp_j)` for the canonical key `(i, j)` with `i < j`. The helper
  `matches_between(a, b)` returns matches oriented so the first index belongs to
  image `a`, flipping the stored tuples when `a > b`.

---

## 5. Geometry helpers (`incremental_sfm.py`)

### `triangulate_general(K, R1, t1, R2, t2, pts1, pts2)`
Triangulates points seen by **two cameras with arbitrary poses**, building both
projection matrices explicitly as `P = K [R | t]`. This is the one genuinely new
piece of geometry: Week 3's `triangulate_points` hardcodes the first camera at
the origin, which is fine for the seed but wrong for triangulating between two
already-registered cameras during growth. Division by the homogeneous coordinate
is wrapped in `np.errstate(...)` so points at infinity yield `inf`/`nan`
(silently filtered later) rather than warnings.

### `_depths_in_camera(points3d, R, t)`
Returns the camera-frame *z* of each point (`(R X + t)[2]`). Used for the
**cheirality** (positive-depth) check during general triangulation, since Week
3's `compute_depths` is hardwired to an origin first camera.

### `median_triangulation_angle(points3d, R, t)`
The **parallax / baseline measure** used for seed selection. For each point, the
angle between the rays from the point to each camera centre (`C1 = origin`,
`C2 = -Rᵀ t`); returns the median in degrees. **Scale-invariant**, so it is well
defined despite the unknown global scale. This is COLMAP's initialisation
criterion (`init_min_tri_angle`).

---

## 6. Initial-pair selection

Ranking pairs by RANSAC inlier count alone is a trap: near-coincident views
match beautifully but have near-parallel rays and triangulate terribly. So we
**gate on parallax**.

### Tools (`incremental_sfm.py`)
- `PairCandidate` — per-pair summary: `ransac_inliers` (from the Week 2 CSV),
  `pose_inliers`, `triangulated_points`, `median_tri_angle`, plus
  `passes_gate(min_tri_angle)`.
- `IncrementalReconstruction.two_view_geometry(i, j, ...)` — recovers two-view
  pose (`estimate_essential_matrix` → `recover_relative_pose`), triangulates the
  pose-inliers, and returns `(pose_inliers, finite_points, median_tri_angle)`.
  Does **not** mutate reconstruction state, so it is safe for scoring candidates.
- `choose_initial_pair(candidates, min_tri_angle=5.0)` — keep pairs whose median
  angle clears the gate; among those pick the one with the most pose inliers. If
  *none* clear the gate, fall back to the best by pose inliers so the caller is
  never left without a seed.

### Driver (`week4_pipeline.py`)
`select_seed(...)`:
1. read the Week 2 `pairwise_metrics.csv`,
2. shortlist the top-`top_k` rows by `ransac_inliers` (cheap filter; **no
   re-matching of all pairs**),
3. score each shortlisted pair with `engine.two_view_geometry`,
4. `choose_initial_pair` → return the two image ids.

This consumes the existing Week 2 CSV directly — no Week 2 changes and no CSV
regeneration.

> **Observed effect (partial-doge):** the #1 pair by inliers has only 2.41°
> parallax and is correctly rejected; the seed instead has 946 pose inliers
> *and* 10.54° parallax.

---

## 7. The incremental engine

### 7.1 `initialise_two_view(i, j, *, ransac_threshold, confidence, max_reproj)`
Bootstraps the reconstruction from the seed pair:
1. `estimate_essential_matrix` → `recover_relative_pose` → `R, t, pose_mask`,
2. register image `i` at the origin and image `j` at `(R, t)`,
3. `triangulate_points` on the pose-inliers (Week 3, origin first camera is
   correct here),
4. filter with `compute_reprojection_errors` (both views) +
   `filter_reconstructed_points`,
5. sample colours (`sample_point_colours`) and create a `Track` for each kept
   point, wiring `kpidx_to_point` on both images.

Returns the number of seed points.

### 7.2 `gather_2d3d(u)` — score an unregistered image
For an unregistered image `u`, accumulate its **2D–3D correspondences** against
the current cloud:
```text
for each registered image r:
    for each match (r_kp, u_kp) in matches_between(r, u):
        pid = registered[r].kpidx_to_point.get(r_kp)
        if pid is not None and u_kp not seen yet:
            seen[u_kp] = pid          # u sees existing 3D point `pid`
```
Returns `count`, the `points3d`/`pts2d` arrays for PnP, and the parallel
`u_kp` / `point_ids` lists. This is the generalisation of Week 3's
`build_2d3d_correspondences` — instead of two fixed anchors, it reads `point_id`
from `kpidx_to_point` across **all** registered images.

### 7.3 `_find_next(rejected)` — the greedy choice
Scans every unregistered, non-rejected image with `gather_2d3d` and returns the
one with the **most** 2D–3D correspondences. This is the "greedy" policy.

### 7.4 `_register(u, corr, *, min_pnp_inliers, pnp_threshold, confidence)`
Places image `u`:
1. `estimate_camera_pose_pnp(points3d, pts2d, K)` → `R, t, inlier_mask` (Week 3,
   `cv2.solvePnPRansac`),
2. reject if inliers `< min_pnp_inliers` (or fewer than 6 correspondences),
3. on success, add a `RegisteredImage` and, for each PnP inlier, link
   `u`'s keypoint into `kpidx_to_point` and append `u` to that track's
   observations.

This is exactly Week 3's image-3 step, made reusable.

### 7.5 `triangulate_new_points(u, max_reproj)` — grow structure
For each registered neighbour `r`, take matches where **neither** keypoint
already owns a 3D point, then:
1. `triangulate_general(K, R_r, t_r, R_u, t_u, pts_r, pts_u)`,
2. keep points that are finite, **positive depth in both cameras**
   (`_depths_in_camera`), and within `max_reproj` reprojection error in both,
3. create a `Track` for each survivor and link both keypoints.

A `u` keypoint matched to several neighbours is triangulated once (guarded by
`kpidx_to_point` being updated as we go).

### 7.6 `run(*, min_corr, min_pnp_inliers, pnp_threshold, confidence, max_reproj)`
The loop:
```text
while True:
    u, corr = _find_next(rejected)
    if u is None or corr.count < min_corr:      # nothing connects well enough
        break
    if not _register(u, corr, ...):
        rejected.add(u)                          # too weak for now; retry later
        continue
    triangulate_new_points(u, ...)
    rejected.clear()                             # cloud changed: re-allow failed
    record StepMetrics
```
Note the **reject-and-retry**: an image that fails PnP now may succeed after more
structure exists, so `rejected` is cleared whenever a registration succeeds. This
prevents both infinite spinning (an image can't be re-picked until the model
changes) and premature permanent rejection.

---

## 8. Outputs

Engine accessors:
- `point_cloud()` → `(points (N,3), colours (N,3))` over all tracks,
- `camera_list()` → `[(image_name, R, t), ...]` for plotting,
- `reprojection_errors()` → per-observation pixel errors (median/mean are the
  headline accuracy numbers).

The driver writes to `--output-dir`:
| File | Produced by |
|------|-------------|
| `incremental_metrics.csv` | `save_csv` of the per-step `StepMetrics` |
| `points3d.ply` | `write_ply` (ASCII PLY, RGB) |
| `multi_view_reconstruction.png` | `plot_multi_view_reconstruction` (matplotlib) |
| `point_cloud_pyvista.png` | `render_point_cloud` (PyVista screenshot) |

---

## 9. Point-cloud visualisation (PyVista)

Open3D has **no Python 3.13 wheel** (it ships compiled binaries built per
CPython ABI, and a `cp313` build does not yet exist), so visualisation uses
**PyVista** (VTK) instead. See `render_point_cloud(points, colours, *,
screenshot_path, show, point_size)`:
- colours are stored **RGB in [0, 255]** (`sample_point_colours` already swaps
  OpenCV BGR→RGB), so they only need `/255` for VTK;
- `pyvista` is **imported lazily** and the whole call is wrapped in `try/except`
  in `main`, so a missing library or GL context degrades gracefully instead of
  failing the reconstruction;
- `--view` opens an interactive window (needs a display; `DISPLAY=:0` under
  WSLg); otherwise an off-screen screenshot is written (disable with
  `--no-screenshot`).

PyVista is an **optional** dependency (`pip install -e '.[viz]'`, and listed in
`environment.yml`).

---

## 10. Driver flow (`week4_pipeline.py::main`)

1. `load_week2_module` / `load_week3_module` (+ `ensure_dir` the output dir).
2. `list_image_paths(image_dir, max_images)` → `precompute_image_features` →
   build `features_by_id` and `name_to_id`.
3. `make_camera_matrix(features[0].image.shape)` → shared `K`.
4. Features + all-pairs matches, **cached on disk**: if a valid
   `feature_match_cache.pkl` exists (signature matches the image set +
   `max_features`/`max_image_size`/`ratio`), keypoint coordinates and matches are
   loaded and images reloaded cheaply; otherwise `precompute_image_features` +
   `build_match_cache` run (for 20 images that is 190 pairs) and the result is
   saved. Controlled by `--cache-file` / `--no-cache` / `--rebuild-cache`. The
   cache stores keypoint coordinates and match index-pairs only — **not**
   descriptors or images (descriptors are unneeded once matches exist; images
   reload via `load_image`).
5. `IncrementalReconstruction(K, features_by_id, pair_matches, week3)`.
6. `select_seed(...)` → `initialise_two_view(...)`.
7. `engine.run(...)`.
8. write outputs (Section 8) and print a summary (registered/rejected cameras,
   median/mean reprojection error).

---

## 11. Command-line usage

Run **from `src/gf4/week4/`** (the directory-relative imports require it):

```bash
python week4_pipeline.py \
  --metrics-csv ../../../out/dataset-partial-doge/pairwise_metrics.csv \
  --image-dir ../../../images/partial_doge_images \
  --output-dir ../../../out/week4-partial-doge \
  --max-images 20
```

Add `--view` for an interactive PyVista window. Or view the saved cloud without
recomputing:

```bash
python -c "import pyvista as pv; pv.read('out/week4-partial-doge/points3d.ply').plot(rgb=True, point_size=4)"
```

### Key parameters

| Flag | Default | Meaning |
|------|---------|---------|
| `--max-images` | 20 | Pool size loaded from `--image-dir`. |
| `--top-k` | 10 | Top-inlier pairs re-scored for the seed. |
| `--min-tri-angle` | 5.0 | Seed gate: min median triangulation angle (deg). |
| `--min-inliers` | 100 | Min Week 2 RANSAC inliers to consider as a seed. |
| `--min-corr` | 20 | Min 2D–3D correspondences to attempt a registration. |
| `--min-pnp-inliers` | 15 | Min PnP inliers to accept a registration. |
| `--ratio` | 0.75 | Lowe ratio for matching. |
| `--ransac-threshold` | 1.0 | Essential-matrix RANSAC threshold (px). |
| `--pnp-threshold` | 6.0 | PnP RANSAC reprojection threshold (px). |
| `--max-reprojection-error` | 4.0 | Max error to keep a triangulated point (px). |
| `--focal-length-px` | none | Override focal length (default `1.2·max(w,h)`). |
| `--max-features` / `--max-image-size` | 4000 / 1600 | SIFT cap / resize. |
| `--cache-file` | `<out>/feature_match_cache.pkl` | Where to read/write the feature+match cache. |
| `--no-cache` / `--rebuild-cache` | off / off | Disable caching / force recompute and overwrite. |
| `--view` / `--no-screenshot` | off / off | Interactive window / skip screenshot. |

---

## 12. Reference result (partial-doge, defaults)

- **Seed:** `…332564` + `…333412` — 946 pose inliers, 10.54° parallax.
- **Registered:** 20 / 20 cameras.
- **Sparse points:** 7254.
- **Reprojection error:** median **0.553 px**, mean **1.039 px**.
- Greedy order recovered the capture (timestamp) order, since consecutive video
  frames share the most structure.

---

## 13. Design decisions and limitations

- **Greedy next-image (most 2D–3D correspondences).** Simple and defensible; on
  sequential video it reproduces capture order.
- **No bundle adjustment.** Poses and points are never jointly refined, so small
  errors accumulate (drift). The median (0.55 px) vs. mean (1.04 px) gap is a
  visible tail of higher-error points — the main "what's missing vs COLMAP"
  point for the report.
- **Single shared `K`.** Assumes one camera/resolution; mixed sources would need
  per-image intrinsics.
- **All-pairs matching up front.** O(N²) matches; fine for ~20 images, but for
  large pools a match graph / vocabulary-tree shortlist would be needed. Results
  are cached to disk (signature-keyed) so reruns skip detection + matching, but
  the *first* run on a large set is still O(N²).
- **No track merging across long loops.** A 3D point seen again after the camera
  loops back may be triangulated as a duplicate rather than merged.
