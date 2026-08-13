"""Batch runner: detect nuclei + emit the output set for many slides.

For each matching CZI it (1) runs nuclei detection unless the cache already
exists (so the batch is resumable / safe to re-run) and (2) runs the analysis +
output set. Finally it writes a combined workbook ``results/summary_all.xlsx``
with one row per slide.

Usage::

    python run_batch.py                       # all *_HE_* slides in dataset/
    python run_batch.py "dataset/*_HE_*.czi"   # explicit glob
    python run_batch.py dataset/a.czi dataset/b.czi   # explicit files
"""
from __future__ import annotations

import glob
import sys
import time
from pathlib import Path

import detect_slide_nuclei
import run_slide

DEFAULT_GLOB = "dataset/*_HE_*.czi"


def _resolve(args: list[str]) -> list[str]:
    if not args:
        return sorted(glob.glob(DEFAULT_GLOB))
    files: list[str] = []
    for a in args:
        files.extend(sorted(glob.glob(a)) if any(c in a for c in "*?[") else [a])
    return files


def batch(files: list[str]) -> list[dict]:
    rows = []
    t0 = time.time()
    for i, czi in enumerate(files, start=1):
        stem = Path(czi).stem
        npz = Path("results") / f"{stem}_nuclei.npz"
        print(f"\n[{i}/{len(files)}] {stem}", flush=True)
        if npz.exists():
            print("  nuclei cache found, skipping detection", flush=True)
        else:
            detect_slide_nuclei.run(czi)
        rows.append(run_slide.run(str(npz), czi))
    print(f"\nbatch done: {len(files)} slides in {(time.time()-t0)/60:.1f} min")
    return rows


def _write_summary(rows: list[dict], out_path: str = "results/summary_all.xlsx"):
    from openpyxl import Workbook

    cols = ["slide", "n_tissue_pieces", "n_nuclei", "n_inflammatory",
            "n_foci_wholeslide", "tissue_mm2", "n_ROIs", "ROI_coverage_pct",
            "ROI_tissue_mm2", "foci_in_ROIs", "FD_per_mm2", "FD_per_um2"]
    wb = Workbook()
    ws = wb.active
    ws.title = "all_slides"
    ws.append(cols)
    for r in rows:
        ws.append([r.get(c, "") for c in cols])
    wb.save(out_path)
    print(f"combined summary -> {out_path}")


if __name__ == "__main__":
    files = _resolve(sys.argv[1:])
    if not files:
        sys.exit("no slides matched")
    rows = batch(files)
    _write_summary(rows)
