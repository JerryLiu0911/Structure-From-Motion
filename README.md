# GF4 — Structure from Motion

> **Language:** Python 3.13
> **Key dependencies:** pycolmap, OpenCV, NumPy, Matplotlib · PyVista (optional, Week 4 visualisation)

---

## Contents

```
src/gf4/
  week1/
  week2/
    sfm_utils.py         # SfM utility functions (features, matching, F-matrix)
    week2_pipeline.py    # Pairwise matching & epipolar geometry CLI
  week3/
    two_view_utils.py    # Two-view geometry: E, pose, triangulation, PnP, plots
    week3_pipeline.py    # Two-view + third-image reconstruction CLI
  week4/
    incremental_sfm.py   # Incremental SfM engine: point map, greedy PnP loop, retriangulation
    week4_pipeline.py    # Multi-view reconstruction CLI
    view_cloud.py        # PyVista point-cloud + camera-frustum viewer
```

---

## Setup

```bash
# 1. Clone
git clone <repo-url>
cd Structure-From-Motion

# 2. Create the conda environment (Python, OpenCV, pycolmap, dev tools)
conda env create -f environment.yml

# 3. Install the package in editable mode
conda run -n gf4-sfm pip install -e .

# 4. Install pre-commit hooks (runs ruff automatically on every git commit)
conda run -n gf4-sfm pre-commit install

# 5. Verify
conda run -n gf4-sfm python main.py
```

After installation the `sfm-week1` and `sfm-week2` commands are available inside the environment. Weeks 3 and 4 are run as scripts from their own directories (see below).

For the optional Week 4 point-cloud viewer (PyVista), install the `viz` extra:

```bash
conda run -n gf4-sfm pip install -e '.[viz]'
```

(PyVista is also listed in `environment.yml`, so a fresh `conda env create` already includes it.)

---

## Running the pipelines

Activate the environment once, then use the short commands:

```bash
conda activate gf4-sfm
```
### Week 2 — Pairwise feature matching & epipolar geometry

**Pair mode** (one image pair):
```bash
sfm-week2 --image1 img_a.jpg --image2 img_b.jpg --output-dir out/
```

**Dataset mode** (all pairs in a directory):
```bash
sfm-week2 --image-dir path/to/images --output-dir out/ --max-images 20
```

Full options:
```bash
sfm-week2 --help
```

### Week 3 — Two-view reconstruction (+ optional third image)

Estimates the essential matrix, recovers relative pose, triangulates a sparse
cloud from an image pair, and optionally registers a third image by PnP. Run as
a script from its directory (it imports its utilities by path):

```bash
cd src/gf4/week3
python week3_pipeline.py \
  --image1 img_a.jpg --image2 img_b.jpg \
  --image3 img_c.jpg \            # optional
  --output-dir out/week3
```

Outputs: reprojection overlays, 2-/3-view reconstruction plots, a patch cloud,
`points3d.ply`, and the camera matrices (`K/R/t/E.txt`). Full options:
`python week3_pipeline.py --help`.

### Week 4 — Incremental multi-view reconstruction



```bash
cd src/gf4/week4
python week4_pipeline.py \
  --metrics-csv ../../../out/dataset-doge/pairwise_metrics.csv \
  --image-dir   ../../../images/Doge \
  --output-dir  ../../../out/week4-doge \
  --max-images 200
```

Outputs (in `--output-dir`): per-step metrics (`incremental_metrics.csv`),
`points3d.ply`, camera poses + intrinsics (`cameras.json`), and a multi-view
matplotlib figure (`multi_view_reconstruction.png`). The run prints the
registered/rejected cameras, reprojection error, and the before/after
retriangulation comparison. Full options: `python week4_pipeline.py --help`.

**Viewing the cloud.** `view_cloud.py` renders the saved PLY (and optionally the
camera frusta) without re-running the reconstruction:

```bash
cd src/gf4/week4
python view_cloud.py --path ../../../out/week4-doge/points3d.ply
```

Useful flags: `--clip P` keeps points within the P-th distance percentile of the
cloud centre (drops far outliers that otherwise shrink the subject; `100` keeps
all); `--cameras ../../../out/week4-doge/cameras.json` overlays the camera frusta;
`--screenshot out.png` renders off-screen to a file instead of opening a window;
plus `--point-size` and `--camera {iso,xy,xz,yz}`. Full options:
`python view_cloud.py --help`.

---

## Coursework deadlines

| Deliverable | Due | Marks |
|---|---|---|
| Interim Report 1 | 21 May 2026 | 15 (individual) |
| Interim Report 2 | 29 May 2026 | 15 (individual) |
| Interim Code | 29 May 2026 | 5 (group) |
| Final code & Presentation | 11 June 2026 | 25 (group) |
| Final Report | 11 June 2026 | 40 (individual) |

Report template: [Overleaf](https://www.overleaf.com/read/jzfdccmknccp#17a7ee)

---

## References

- Torralba, Isola & Freeman — *Foundations of Computer Vision*, Chapter 44: [Multiview Geometry and Structure from Motion](https://visionbook.mit.edu/multiview.html)
- [COLMAP documentation](https://colmap.github.io/)
- [pycolmap documentation](https://pypi.org/project/pycolmap/)
