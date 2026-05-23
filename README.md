# GF4 — Structure from Motion

A guided project that teaches Structure from Motion (SfM) through hands-on experimentation: running COLMAP as a professional reference system and implementing key steps of the SfM pipeline from scratch in Python.

> **Course:** GF4 · University of Cambridge
> **Language:** Python 3.13
> **Key dependencies:** pycolmap, OpenCV, NumPy, Matplotlib

---

## Contents

```
src/gf4/
  week1/
  week2/
    sfm_utils.py         # SfM utility functions
    week2_pipeline.py    # Pairwise matching & epipolar geometry CLI
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

# 4. Verify
conda run -n gf4-sfm python main.py
```

After installation the `sfm-week1` and `sfm-week2` commands are available inside the environment.

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
