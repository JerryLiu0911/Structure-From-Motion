"""Structure from Motion — project entry point.

Usage:
    # Week 1 — pycolmap sparse reconstruction
    python -m gf4.week1.test --image-dir <images/> --output-dir <out/>

    # Week 2 — pairwise feature matching
    python -m gf4.week2.week2_pipeline --image-dir <images/> --output-dir <out/>
    python -m gf4.week2.week2_pipeline --image1 a.jpg --image2 b.jpg --output-dir <out/>

Run `python main.py` to print usage hints.
"""

from __future__ import annotations


def main() -> None:
    print("Structure from Motion — GF4")
    print()
    print("Week 1 (pycolmap reconstruction):")
    print("  python -m gf4.week1.test --image-dir <images/> --output-dir <out/>")
    print()
    print("Week 2 (feature matching & epipolar geometry):")
    print("  python -m gf4.week2.week2_pipeline --help")


if __name__ == "__main__":
    main()
