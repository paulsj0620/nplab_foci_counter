"""Build a numbered gallery of full-res crops around detected foci, for review.

Whole-slide overlays are too small to judge whether a called focus is really an
inflammatory cluster. This tool re-runs the (fast) shared step-4/5 analysis on
cached nuclei -- including the tissue-fraction QC that drops edge/torn foci --
samples some foci, reads a full-res crop around each, outlines the inflammatory
nuclei, and lays them out in a numbered grid. Focus numbers match the CSV rows.

Usage::

    python foci_gallery.py results/<stem>_nuclei.npz dataset/<stem>.czi [n_sample]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from skimage.io import imsave
from skimage.draw import circle_perimeter

from czi_loader import CziSlide
from tissue_mask import tissue_mask
from foci_pipeline import analyze as analyze_nuclei

MASK_DS = 9
CROP_UM = 250.0          # crop side length in microns (centered on the focus)
GRID_COLS = 5


def build_gallery(npz_path: str, czi_path: str, n_sample: int = 25):
    d = np.load(npz_path)
    points_xy, areas_um2 = d["points_xy"], d["areas_um2"]
    px = float(d["pixel_size_um"])

    with CziSlide(czi_path) as slide:
        info = slide.info()
        rgb = slide.read_level(MASK_DS)
        mask = tissue_mask(rgb, pixel_size_um=info.level_pixel_size_um(MASK_DS))
        fa = analyze_nuclei(points_xy, areas_um2, px, mask, MASK_DS, rgb=rgb)
        if fa.n_foci == 0:
            print("no foci to show")
            return

        # Sample foci across the size range (sorted, evenly spaced) so the grid
        # shows both large and small clusters, not only the biggest.
        order = np.argsort(-fa.sizes)
        pick = order[np.linspace(0, fa.n_foci - 1,
                                 min(n_sample, fa.n_foci)).astype(int)]

        crop_px = int(CROP_UM / px)
        half = crop_px // 2
        cell_r = max(4, int(5.0 / px))            # ~5µm ring radius per nucleus

        cells = []
        for rank, k in enumerate(pick, start=1):
            cid = fa.kept_ids[k]                    # original DBSCAN cluster id
            cx, cy = fa.centers_xy[k]
            x0, y0 = int(cx - half), int(cy - half)
            crop = slide.read_region(x0, y0, crop_px, crop_px, downsample=1)
            # Outline (not fill) each inflammatory nucleus: a thin dark backing
            # ring plus a 2-px bright-yellow ring so it stands out on any tissue.
            ch, cw = crop.shape[:2]
            for mx, my in fa.infl_xy[fa.labels == cid]:
                cyp, cxp = int(my - y0), int(mx - x0)
                yy, xx = circle_perimeter(cyp, cxp, cell_r + 2, shape=(ch, cw))
                crop[yy, xx] = [0, 0, 0]                    # dark contrast edge
                for rr in (cell_r, cell_r + 1):
                    yy, xx = circle_perimeter(cyp, cxp, rr, shape=(ch, cw))
                    crop[yy, xx] = [255, 255, 0]            # bright yellow ring
            cells.append((rank, int(fa.sizes[k]), crop))

    out = Path("results") / f"{Path(npz_path).stem.replace('_nuclei','')}_gallery.png"
    _save_grid(cells, out)


def _save_grid(cells, out_path):
    """Tile crops into a labeled grid image with a numbered header stripe."""
    n = len(cells)
    cols = GRID_COLS
    rows = -(-n // cols)
    ch, cw = cells[0][2].shape[:2]
    pad, header = 6, 22
    tile_h = ch + header
    canvas = np.full(
        (rows * tile_h + (rows + 1) * pad, cols * cw + (cols + 1) * pad, 3),
        255, dtype=np.uint8,
    )
    for i, (rank, size, crop) in enumerate(cells):
        r, c = divmod(i, cols)
        y = pad + r * (tile_h + pad) + header
        x = pad + c * (cw + pad)
        canvas[y:y + ch, x:x + cw] = crop
        canvas[y - header:y - 2, x:x + cw] = [0, 0, 0]      # black label bar
        _put_number(canvas, rank, size, x + 2, y - header + 3)
    imsave(out_path, canvas)
    print(f"saved {out_path}  ({n} foci: #1..#{n})")


def _put_number(canvas, rank, size, x, y):
    """Draw '#rank n=size' using PIL if available (crisp), else skip."""
    try:
        from PIL import Image, ImageDraw
        img = Image.fromarray(canvas)
        ImageDraw.Draw(img).text((x, y), f"#{rank}  n={size}", fill=(255, 255, 0))
        canvas[:] = np.asarray(img)
    except Exception:
        pass


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit("usage: python foci_gallery.py <nuclei.npz> <slide.czi> [n]")
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 25
    build_gallery(sys.argv[1], sys.argv[2], n)
