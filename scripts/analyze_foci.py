"""Run steps 4-5 (inflammatory classification + foci clustering) on cached nuclei.

Loads the ``*_nuclei.npz`` produced by ``detect_slide_nuclei.py`` and, without
re-running the network, classifies inflammatory nuclei, clusters them into foci,
and drops edge/torn foci via the shared tissue-fraction QC (see foci_pipeline).
Fast to re-run, so it is the place to tune parameters. Produces a whole-slide
overlay (foci circled) and a one-row summary.

Usage::

    python analyze_foci.py results/20260505_Leber_HE_119_nuclei.npz \\
        dataset/20260505_Leber_HE_119.czi
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from czi_loader import CziSlide
from tissue_mask import tissue_mask, tissue_area_mm2
from foci_pipeline import analyze as analyze_nuclei

MASK_DS = 9


def analyze(npz_path: str, czi_path: str) -> dict:
    d = np.load(npz_path)
    points_xy = d["points_xy"]
    areas_um2 = d["areas_um2"]
    px = float(d["pixel_size_um"])

    with CziSlide(czi_path) as slide:
        info = slide.info()
        rgb = slide.read_level(MASK_DS)
    px_level = info.level_pixel_size_um(MASK_DS)
    mask = tissue_mask(rgb, pixel_size_um=px_level)

    fa = analyze_nuclei(points_xy, areas_um2, px, mask, MASK_DS, rgb=rgb)
    area_mm2 = tissue_area_mm2(mask, px_level)
    fd_mm2 = fa.n_foci / area_mm2 if area_mm2 else 0.0

    summary = {
        "slide": Path(npz_path).stem.replace("_nuclei", ""),
        "n_nuclei": int(points_xy.shape[0]),
        "n_inflammatory": int(fa.is_infl.sum()),
        "n_foci": fa.n_foci,
        "tissue_mm2": round(area_mm2, 1),
        # Focal Density: foci per tissue area, per mm² (practical) and per µm².
        "FD_per_mm2": round(fd_mm2, 3),
        "FD_per_um2": fd_mm2 / 1e6,
    }

    _save_overlay(rgb, fa, Path("results") / f"{summary['slide']}_foci_overlay.png")
    summary["overlay"] = str(Path("results") / f"{summary['slide']}_foci_overlay.png")
    return summary


def _save_overlay(rgb, fa, out_path):
    """Whole-slide overlay: a hollow red ring per kept focus."""
    from skimage.io import imsave
    from skimage.draw import circle_perimeter

    ov = rgb.copy()
    h, w = ov.shape[:2]
    for (cx, cy), sz in zip(fa.centers_xy, fa.sizes):
        r = 8 + int(np.sqrt(sz))            # ring grows with focus size
        yc, xc = int(cy / MASK_DS), int(cx / MASK_DS)
        for rr_off in (0, 1, 2):            # concentric perimeters = thickness
            yy, xx = circle_perimeter(yc, xc, r + rr_off, shape=(h, w))
            ov[yy, xx] = [255, 0, 0]
    imsave(out_path, ov)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit("usage: python analyze_foci.py <nuclei.npz> <slide.czi>")
    res = analyze(sys.argv[1], sys.argv[2])
    for k, v in res.items():
        print(f"{k:16s}: {v}")
