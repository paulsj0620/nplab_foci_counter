"""Process a single slide: detect nuclei (if not cached) + emit the output set.

The per-slide unit used by the SLURM job array (one array task = one slide).
Detection is skipped when the ``*_nuclei.npz`` cache already exists, so re-runs
are cheap and the array is resumable.

Usage::

    python process_one.py dataset/<slide>.czi
"""
from __future__ import annotations

import sys
from pathlib import Path

import detect_slide_nuclei
import run_slide


def main(czi: str):
    stem = Path(czi).stem
    npz = Path("results") / f"{stem}_nuclei.npz"
    if npz.exists():
        print(f"{stem}: nuclei cache found, skipping detection", flush=True)
    else:
        detect_slide_nuclei.run(czi)
    run_slide.run(str(npz), czi)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python process_one.py <slide.czi>")
    main(sys.argv[1])
