"""Batch runner: detect nuclei + emit the output set for many slides.

For each matching CZI it (1) runs nuclei detection unless the cache already
exists (so the batch is resumable / safe to re-run) and (2) runs the analysis +
output set. A per-slide failure is logged and the batch continues. Everything
printed is also written to a timestamped log file, and a combined workbook
``results/summary_all.xlsx`` is written with one row per slide.

Usage::

    python run_batch.py                       # all *_HE_* slides in dataset/
    python run_batch.py "dataset/*_HE_*.czi"   # explicit glob
    python run_batch.py dataset/a.czi dataset/b.czi   # explicit files
"""
from __future__ import annotations

import glob
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import detect_slide_nuclei
import run_slide

DEFAULT_GLOB = "dataset/*_HE_*.czi"
LOG_DIR = Path("results/logs")


class _Tee:
    """Duplicate a stream to the console and a log file."""

    def __init__(self, stream, fh):
        self._stream = stream
        self._fh = fh

    def write(self, data):
        self._stream.write(data)
        self._fh.write(data)
        self._fh.flush()

    def flush(self):
        self._stream.flush()
        self._fh.flush()


def _resolve(args: list[str]) -> list[str]:
    if not args:
        return sorted(glob.glob(DEFAULT_GLOB))
    files: list[str] = []
    for a in args:
        files.extend(sorted(glob.glob(a)) if any(c in a for c in "*?[") else [a])
    return files


def _stamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def batch(files: list[str]) -> list[dict]:
    rows = []
    t0 = time.time()
    for i, czi in enumerate(files, start=1):
        stem = Path(czi).stem
        npz = Path("results") / f"{stem}_nuclei.npz"
        print(f"\n[{i}/{len(files)}] {stem}  ({_stamp()})", flush=True)
        t_slide = time.time()
        try:
            if npz.exists():
                print("  nuclei cache found, skipping detection", flush=True)
            else:
                detect_slide_nuclei.run(czi)
            res = run_slide.run(str(npz), czi)
            res["status"] = "ok"
            res["seconds"] = round(time.time() - t_slide, 1)
            print(f"  [{stem}] done in {res['seconds']}s", flush=True)
        except Exception as exc:                       # noqa: BLE001 keep batch alive
            print(f"  [{stem}] FAILED: {exc}", flush=True)
            traceback.print_exc()
            res = {"slide": stem, "status": f"FAILED: {exc}",
                   "seconds": round(time.time() - t_slide, 1)}
        rows.append(res)
    print(f"\nbatch done: {len(files)} slides in {(time.time()-t0)/60:.1f} min "
          f"({_stamp()})")
    ok = sum(1 for r in rows if r.get("status") == "ok")
    print(f"  {ok} ok, {len(rows) - ok} failed")
    return rows


def _write_summary(rows: list[dict], out_path: str = "results/summary_all.xlsx"):
    from openpyxl import Workbook

    cols = ["slide", "status", "seconds", "n_tissue_pieces", "n_nuclei",
            "n_inflammatory", "n_foci_wholeslide", "tissue_mm2", "n_ROIs",
            "ROI_coverage_pct", "ROI_tissue_mm2", "foci_in_ROIs",
            "FD_per_mm2", "FD_per_um2", "foci_per_field_3.1mm2",
            "Liang_grade", "Liang_grade_label"]
    wb = Workbook()
    ws = wb.active
    ws.title = "all_slides"
    ws.append(cols)
    for r in rows:
        ws.append([r.get(c, "") for c in cols])
    wb.save(out_path)
    print(f"combined summary -> {out_path}")


def main():
    files = _resolve(sys.argv[1:])
    if not files:
        sys.exit("no slides matched")

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"batch_{datetime.now():%Y%m%d_%H%M%S}.log"
    with open(log_path, "w") as fh:
        orig_out, orig_err = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = _Tee(orig_out, fh), _Tee(orig_err, fh)
        try:
            print(f"batch start {_stamp()}  ({len(files)} slides)")
            print("log: " + str(log_path))
            rows = batch(files)
            _write_summary(rows)
        finally:
            sys.stdout, sys.stderr = orig_out, orig_err
    print(f"log written -> {log_path}")


if __name__ == "__main__":
    main()
