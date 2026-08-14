"""One-shot per-slide output set: ROI map + ROI gallery + Excel workbook.

Runs the shared step-4/5 analysis (foci_pipeline) once on cached nuclei, then
samples large non-overlapping ROIs covering ~25% of the tissue (roi_select) and
counts foci within them. Writes a review set into ``results/<stem>/``:

  1. ``<stem>_overview.png`` -- whole slide: every focus a small red ring, and
     the sampled ROIs as numbered yellow rectangles.
  2. ``<stem>_gallery.png``  -- one crop per ROI with its inflammatory nuclei
     outlined (yellow rings); numbered to match the map and workbook.
  3. ``<stem>_results.xlsx`` -- summary (foci, FD per mm²/µm², ROI coverage) and
     a per-ROI sheet.

Usage::

    python run_slide.py results/<stem>_nuclei.npz dataset/<stem>.czi
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from skimage.io import imsave
from skimage.draw import circle_perimeter, rectangle_perimeter

from czi_loader import CziSlide
from tissue_mask import tissue_mask, tissue_area_mm2
from foci_pipeline import analyze as analyze_nuclei
from roi_select import select_rois
from tissue_pieces import find_pieces, best_piece
import scoring

MASK_DS = 9
GALLERY_DS = 3          # read ROI crops at this level (detail vs size)
GRID_COLS = 5


def _label(canvas, text, x, y, color=(255, 255, 0)):
    try:
        from PIL import Image, ImageDraw
        img = Image.fromarray(canvas)
        ImageDraw.Draw(img).text((x, y), text, fill=color)
        canvas[:] = np.asarray(img)
    except Exception:
        pass


def _in_mask(points_xy, mask, mask_downsample):
    """Boolean: which full-res points fall on True pixels of a level mask."""
    h, w = mask.shape
    mx = np.clip((points_xy[:, 0] / mask_downsample).astype(int), 0, w - 1)
    my = np.clip((points_xy[:, 1] / mask_downsample).astype(int), 0, h - 1)
    return mask[my, mx]


def _foci_in_roi(fa, roi):
    """Indices of kept foci whose center lies inside the ROI box."""
    cx, cy = fa.centers_xy[:, 0], fa.centers_xy[:, 1]
    inside = (cx >= roi.x) & (cx < roi.x + roi.w) & \
             (cy >= roi.y) & (cy < roi.y + roi.h)
    return np.nonzero(inside)[0]


def _overview(rgb, fa, rois, out_path):
    ov = rgb.copy()
    h, w = ov.shape[:2]
    for (cx, cy), sz in zip(fa.centers_xy, fa.sizes):
        yc, xc = int(cy / MASK_DS), int(cx / MASK_DS)
        for off in (0, 1):
            yy, xx = circle_perimeter(yc, xc, 5 + int(np.sqrt(sz)) + off, shape=(h, w))
            ov[yy, xx] = [255, 0, 0]
    for rank, roi in enumerate(rois, start=1):
        y0, x0 = roi.y // MASK_DS, roi.x // MASK_DS
        y1, x1 = min(h - 1, (roi.y + roi.h) // MASK_DS), min(w - 1, (roi.x + roi.w) // MASK_DS)
        for off in (0, 1, 2):
            rr, cc = rectangle_perimeter((y0 - off, x0 - off), (y1 + off, x1 + off),
                                         shape=(h, w))
            ov[rr, cc] = [255, 255, 0]
        _label(ov, str(rank), x0 + 3, y0 + 2)
    imsave(out_path, ov)


def _gallery_cells(slide, fa, rois, px):
    """One crop per ROI (read at GALLERY_DS) with inflammatory nuclei outlined."""
    margin_um = 15.0                       # padding around the cluster
    cells = []
    for rank, roi in enumerate(rois, start=1):
        crop = slide.read_region(roi.x, roi.y, roi.w, roi.h, downsample=GALLERY_DS)
        ch, cw = crop.shape[:2]
        n_foci = 0
        for fi in _foci_in_roi(fa, roi):
            n_foci += 1
            cid = fa.kept_ids[fi]
            members = fa.infl_xy[fa.labels == cid]           # (M, 2) x, y full-res
            # One big circle enclosing the whole focus (cluster of cells).
            cxp = (members[:, 0].mean() - roi.x) / GALLERY_DS
            cyp = (members[:, 1].mean() - roi.y) / GALLERY_DS
            spread = np.hypot(members[:, 0] - members[:, 0].mean(),
                              members[:, 1] - members[:, 1].mean()).max()
            r = int((spread + margin_um / px) / GALLERY_DS) + 3
            yy, xx = circle_perimeter(int(cyp), int(cxp), r + 4, shape=(ch, cw))
            crop[yy, xx] = [0, 0, 0]                          # dark backing
            for off in range(4):                             # thick yellow ring
                yy, xx = circle_perimeter(int(cyp), int(cxp), r + off, shape=(ch, cw))
                crop[yy, xx] = [255, 255, 0]
        cells.append((rank, n_foci, crop))
    return cells


def _save_grid(cells, out_path):
    cols, rows = GRID_COLS, -(-len(cells) // GRID_COLS)
    ch, cw = cells[0][2].shape[:2]
    pad, header = 6, 22
    tile_h = ch + header
    canvas = np.full(
        (rows * tile_h + (rows + 1) * pad, cols * cw + (cols + 1) * pad, 3),
        255, dtype=np.uint8,
    )
    for i, (rank, n_foci, crop) in enumerate(cells):
        r, c = divmod(i, cols)
        y = pad + r * (tile_h + pad) + header
        x = pad + c * (cw + pad)
        canvas[y:y + crop.shape[0], x:x + crop.shape[1]] = crop
        canvas[y - header:y - 2, x:x + cw] = [0, 0, 0]
        _label(canvas, f"ROI #{rank}  foci={n_foci}", x + 2, y - header + 3)
    imsave(out_path, canvas)


def _excel(out_path, stem, fa, rois, foci_per_roi, roi_tissue_mm2, fd_mm2,
           tissue_mm2, coverage_pct, n_nuclei, px_level, n_pieces):
    from openpyxl import Workbook

    wb = Workbook()
    ws = wb.active
    ws.title = "summary"
    total_roi_foci = int(sum(foci_per_roi))
    for row in [
        ("slide", stem),
        ("n_tissue_pieces", n_pieces),
        ("analyzed", "cleanest piece" if n_pieces > 1 else "whole slide"),
        ("n_nuclei", n_nuclei),
        ("n_inflammatory", int(fa.is_infl.sum())),
        ("n_foci_wholeslide", fa.n_foci),
        ("tissue_mm2", round(tissue_mm2, 2)),
        ("n_ROIs", len(rois)),
        ("ROI_coverage_pct", round(coverage_pct, 1)),
        ("ROI_tissue_mm2", round(roi_tissue_mm2, 2)),
        ("foci_in_ROIs", total_roi_foci),
        ("FD_per_mm2", round(fd_mm2, 3)),
        ("FD_per_um2", fd_mm2 / 1e6),
    ]:
        ws.append(row)

    wr = wb.create_sheet("ROIs")
    wr.append(["ROI", "x_fullres", "y_fullres", "side_um",
               "tissue_frac", "n_foci"])
    for rank, (roi, nf) in enumerate(zip(rois, foci_per_roi), start=1):
        wr.append([rank, roi.x, roi.y, round(roi.w * px_level / MASK_DS, 0),
                   round(roi.tissue_frac, 3), int(nf)])
    wb.save(out_path)


def run(npz_path: str, czi_path: str) -> dict:
    d = np.load(npz_path)
    pts, areas = d["points_xy"], d["areas_um2"]
    px = float(d["pixel_size_um"])
    stem = Path(npz_path).stem.replace("_nuclei", "")
    out = Path("results") / stem
    out.mkdir(parents=True, exist_ok=True)

    with CziSlide(czi_path) as slide:
        info = slide.info()
        px_level = info.level_pixel_size_um(MASK_DS)
        rgb = slide.read_level(MASK_DS)
        full_mask = tissue_mask(rgb, pixel_size_um=px_level)

        # If the slide holds several tissue pieces, analyze only the cleanest.
        pieces = find_pieces(full_mask, rgb, px, MASK_DS)
        chosen = best_piece(pieces)
        mask = chosen.mask if chosen is not None else full_mask
        n_pieces = len(pieces)

        # Restrict nuclei to the chosen piece, then run analysis on it.
        keep = _in_mask(pts, mask, MASK_DS)
        fa = analyze_nuclei(pts[keep], areas[keep], px, mask, MASK_DS, rgb=rgb)
        tissue_mm2 = tissue_area_mm2(mask, px_level)

        rois = select_rois(mask, MASK_DS, px)
        foci_per_roi = [len(_foci_in_roi(fa, r)) for r in rois]
        # ROI tissue area (mm²) from the mask under each ROI.
        roi_tissue_px = 0
        for r in rois:
            roi_tissue_px += int(mask[r.y // MASK_DS:(r.y + r.h) // MASK_DS,
                                      r.x // MASK_DS:(r.x + r.w) // MASK_DS].sum())
        roi_tissue_mm2 = roi_tissue_px * (px_level / 1000) ** 2
        coverage_pct = 100 * roi_tissue_mm2 / tissue_mm2 if tissue_mm2 else 0.0
        fd_mm2 = sum(foci_per_roi) / roi_tissue_mm2 if roi_tissue_mm2 else 0.0

        _overview(rgb, fa, rois, out / f"{stem}_overview.png")
        cells = _gallery_cells(slide, fa, rois, px)

    if cells:
        _save_grid(cells, out / f"{stem}_gallery.png")
    _excel(out / f"{stem}_results.xlsx", stem, fa, rois, foci_per_roi,
           roi_tissue_mm2, fd_mm2, tissue_mm2, coverage_pct,
           int(keep.sum()), px_level, n_pieces)

    piece_note = f"1 of {n_pieces} pieces" if n_pieces > 1 else "single piece"
    print(f"{stem}: {piece_note}, {len(rois)} ROIs ({coverage_pct:.0f}% cover), "
          f"{sum(foci_per_roi)} foci in ROIs, FD={fd_mm2:.3f}/mm2 -> {out}/")
    return {
        "slide": stem,
        "n_tissue_pieces": n_pieces,
        "n_nuclei": int(keep.sum()),
        "n_inflammatory": int(fa.is_infl.sum()),
        "n_foci_wholeslide": fa.n_foci,
        "tissue_mm2": round(tissue_mm2, 2),
        "n_ROIs": len(rois),
        "ROI_coverage_pct": round(coverage_pct, 1),
        "ROI_tissue_mm2": round(roi_tissue_mm2, 2),
        "foci_in_ROIs": int(sum(foci_per_roi)),
        "FD_per_mm2": round(fd_mm2, 3),
        "FD_per_um2": fd_mm2 / 1e6,
        "out": str(out),
    }


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit("usage: python run_slide.py <nuclei.npz> <slide.czi>")
    run(sys.argv[1], sys.argv[2])
