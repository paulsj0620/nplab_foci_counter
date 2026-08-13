"""Detect nuclei across all tissue tiles of one slide and cache the result.

Nuclei detection (StarDist) is the slow step (~0.3 s/tile). Running it once and
caching the global nucleus centroids + areas lets us iterate on the downstream
inflammatory-classification and foci-clustering logic quickly, without re-running
the network.

Usage::

    python detect_slide_nuclei.py dataset/20260505_Leber_HE_119.czi

Writes ``results/<stem>_nuclei.npz`` with arrays:
    points_xy   (N, 2) float  -- global full-res centroids (x, y)
    areas_um2   (N,)   float  -- nucleus area in µm²
    probs       (N,)   float  -- StarDist detection probability
and scalar metadata (pixel_size_um, full_width, full_height, n_tiles).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

from czi_loader import CziSlide
from tissue_mask import tissue_mask
from tiling import plan_tiles
from nuclei import detect_nuclei

MASK_DS = 9
TILE = 512
MIN_TISSUE_FRAC = 0.05


def run(path: str, out_dir: str = "results") -> str:
    t_start = time.time()
    with CziSlide(path) as slide:
        info = slide.info()
        px = info.pixel_size_um
        rgb = slide.read_level(MASK_DS)
        mask = tissue_mask(rgb, pixel_size_um=info.level_pixel_size_um(MASK_DS))
        tiles = plan_tiles(
            mask, MASK_DS, info.full_width, info.full_height,
            tile_size=TILE, min_tissue_frac=MIN_TISSUE_FRAC,
        )
        print(f"{Path(path).name}: {len(tiles)} tissue tiles to process")

        all_xy, all_area, all_prob = [], [], []
        for i, t in enumerate(tiles):
            tile = slide.read_region(t.x, t.y, t.w, t.h, downsample=1)
            res = detect_nuclei(tile)
            if res.count:
                all_xy.append(res.to_global((t.x, t.y)))
                all_area.append(res.areas_um2(px))
                all_prob.append(res.probs)
            if (i + 1) % 100 == 0 or i + 1 == len(tiles):
                elapsed = time.time() - t_start
                rate = (i + 1) / elapsed
                eta = (len(tiles) - i - 1) / rate
                total = sum(a.shape[0] for a in all_xy)
                print(f"  tile {i+1}/{len(tiles)}  nuclei={total}  "
                      f"{rate:.1f} tiles/s  ETA {eta/60:.1f} min", flush=True)

    points_xy = np.concatenate(all_xy) if all_xy else np.empty((0, 2))
    areas_um2 = np.concatenate(all_area) if all_area else np.empty((0,))
    probs = np.concatenate(all_prob) if all_prob else np.empty((0,))

    Path(out_dir).mkdir(exist_ok=True)
    out = Path(out_dir) / f"{Path(path).stem}_nuclei.npz"
    np.savez_compressed(
        out,
        points_xy=points_xy,
        areas_um2=areas_um2,
        probs=probs,
        pixel_size_um=px,
        full_width=info.full_width,
        full_height=info.full_height,
        n_tiles=len(tiles),
    )
    print(f"DONE: {points_xy.shape[0]} nuclei in {(time.time()-t_start)/60:.1f} min "
          f"-> {out}")
    return str(out)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: python detect_slide_nuclei.py <path.czi>")
    run(sys.argv[1])
