# GF4 — Structure from Motion

A guided project that teaches Structure from Motion (SfM) through hands-on experimentation: running COLMAP as a professional reference system and implementing key steps of the SfM pipeline from scratch in Python.

> **Course:** GF4 · University of Cambridge
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
    incremental_sfm.py   # Incremental SfM engine (point map + greedy loop)
    week4_pipeline.py    # Multi-view reconstruction CLI
    IMPLEMENTATION.md    # Detailed Week 4 implementation notes
colmap/                  # COLMAP helper scripts and notes
COLMAP projects/         # Local reconstruction data — gitignored
```

---

## Prerequisites

| Tool | Install |
|------|---------|
| [Miniconda](https://docs.anaconda.com/miniconda/) | via installer |
| [COLMAP](https://colmap.github.io/install.html) | OS package manager or binary |

Verify COLMAP is available:

```bash
colmap gui
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

### Week 1 — COLMAP sparse reconstruction

```bash
sfm-week1 --image-dir path/to/images --output-dir path/to/output

# or without activating:
conda run -n gf4-sfm sfm-week1 --image-dir path/to/images --output-dir path/to/output
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

Generalises Week 3 to a pool of images: it selects a wide-baseline initial pair
(parallax-gated), bootstraps a two-view reconstruction, then greedily registers
the remaining images by PnP and triangulates new structure after each. It reuses
the Week 2 and Week 3 utilities, reads the Week 2 `pairwise_metrics.csv` to pick
the seed, and uses **no COLMAP** (COLMAP is only an external baseline for the
report). Run as a script from its directory:

```bash
cd src/gf4/week4
python week4_pipeline.py \
  --metrics-csv ../../../out/dataset-partial-doge/pairwise_metrics.csv \
  --image-dir   ../../../images/partial_doge_images \
  --output-dir  ../../../out/week4-partial-doge \
  --max-images 20
```

Outputs: per-step metrics (`incremental_metrics.csv`), `points3d.ply`, a
multi-view matplotlib figure, and a PyVista screenshot. Add `--view` for an
interactive 3D window, or view a saved cloud directly:

```bash
python -c "import pyvista as pv; pv.read('out/week4-partial-doge/points3d.ply').plot(rgb=True, point_size=4)"
```

See [`src/gf4/week4/IMPLEMENTATION.md`](src/gf4/week4/IMPLEMENTATION.md) for a
detailed description of the engine, data structures, and design decisions. Full
options: `python week4_pipeline.py --help`.

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
