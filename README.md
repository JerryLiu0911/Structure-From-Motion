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
    test.py              # pycolmap sparse reconstruction pipeline (CLI)
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
| [Miniconda](https://docs.anaconda.com/miniconda/) | `brew install miniconda` or via installer |
| [COLMAP](https://colmap.github.io/install.html) | OS package manager or binary |
| [Task](https://taskfile.dev/installation/) | `brew install go-task` or `sh -c "$(curl …)"` |

Verify COLMAP is available before running Week 1:

```bash
colmap gui
```

---

## Quick start

```bash
# 1. Clone
git clone <repo-url>
cd Structure-From-Motion

# 2. Create the conda environment (installs Python, OpenCV, pycolmap, dev tools)
task setup          # or: conda env create -f environment.yml

# 3. Verify the package loads
conda run -n gf4-sfm python main.py
```

---

## Running the pipelines

### Week 1 — COLMAP sparse reconstruction (pycolmap)

```bash
task week1 IMAGE_DIR=path/to/images OUTPUT_DIR=path/to/output

# or directly:
conda run -n gf4-sfm python -m gf4.week1.test \
    --image-dir path/to/images \
    --output-dir path/to/output \
    --threads 4
```

### Week 2 — Pairwise feature matching & epipolar geometry

**Pair mode** (one image pair):
```bash
task week2-pair IMAGE1=img_a.jpg IMAGE2=img_b.jpg OUTPUT_DIR=out/

# or directly:
conda run -n gf4-sfm python -m gf4.week2.week2_pipeline \
    --image1 img_a.jpg --image2 img_b.jpg --output-dir out/
```

**Dataset mode** (all pairs in a directory):
```bash
task week2-dataset IMAGE_DIR=path/to/images OUTPUT_DIR=out/

# or directly:
conda run -n gf4-sfm python -m gf4.week2.week2_pipeline \
    --image-dir path/to/images --output-dir out/ --max-images 20
```

Full CLI options:
```bash
conda run -n gf4-sfm python -m gf4.week2.week2_pipeline --help
```

---

## Development workflow

```bash
task lint            # ruff check
task format          # ruff format (modifies files)
task format-check    # ruff format --check (CI-safe, read-only)
task typecheck       # mypy src/gf4
task test            # pytest
task test-cov        # pytest with coverage report
task check           # run all of the above (pre-push gate)
task clean           # remove build artefacts and caches
```

Pre-commit hooks (installed by `task setup`) run `ruff` automatically on every `git commit`.

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
