"""Select representative ROIs covering a target fraction of the tissue.

Instead of reviewing tiny per-focus crops, we sample a handful of large,
non-overlapping square ROIs spread across the tissue -- like a pathologist
sampling representative fields. The ROIs together cover ~``target_coverage`` of
the tissue, number between ``min_rois`` and ``max_rois``, sit on mostly-tissue
areas, and (being grid cells) never overlap. Foci are then counted within them
and FD reported over the sampled ROI tissue area.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import remove_small_holes


@dataclass
class ROI:
    """A square review region in full-resolution pixel coordinates."""

    x: int
    y: int
    w: int
    h: int
    tissue_frac: float


def select_rois(
    mask: np.ndarray,
    mask_downsample: int,
    pixel_size_um: float,
    rgb: np.ndarray | None = None,
    target_coverage: float = 0.25,
    n_target: int = 11,
    min_rois: int = 5,
    max_rois: int = 20,
    min_tissue_frac: float = 0.98,
    lumen_max_pct: float = 1.0,
    dark_blob_max_um2: float = 1500.0,
) -> list[ROI]:
    """Pick non-overlapping square ROIs covering ~target_coverage of tissue.

    Parameters
    ----------
    mask : (H, W) bool tissue mask at ``mask_downsample``.
    pixel_size_um : full-res µm/px.
    rgb : (H, W, 3) image at the same level; enables vessel/fold rejection.
    target_coverage : desired fraction of tissue area covered by the ROIs.
    n_target : rough number of ROIs used to size each square.
    min_rois, max_rois : clamp on the ROI count.
    min_tissue_frac : a grid cell must be at least this much tissue to qualify.
    lumen_max_pct : reject a cell whose largest enclosed lumen exceeds this
        percent of the cell (big vessel / portal tract). Needs ``rgb`` is not
        required (uses the mask), but the check only runs when rgb is given.
    dark_blob_max_um2 : reject a cell with a connected very-dark blob larger
        than this (tissue fold / dark artifact). Needs ``rgb``.
    """
    px_level = pixel_size_um * mask_downsample
    tissue_px = int(mask.sum())
    tissue_area_um2 = tissue_px * (px_level ** 2)

    # Size each square so that n_target of them would cover target_coverage.
    roi_um = float(np.sqrt(target_coverage * tissue_area_um2 / max(1, n_target)))
    step = max(1, int(roi_um / px_level))          # ROI side in mask pixels
    H, W = mask.shape

    # Placement mask: fill small interior gaps (sinusoids/texture) so the
    # tissue-fraction test measures "inside the tissue region", not local
    # density. Genuine tears (large holes) and the slide edge stay non-tissue,
    # so a high threshold still excludes edge/torn ROIs. (`tissue_mask` uses
    # min_hole_um2=0 to keep tears out of the AREA denominator; here we only
    # fill sub-sinusoid gaps for *placement*.)
    fill_px = int(2500.0 / (px_level ** 2))        # fill holes up to ~2500 µm²
    place = remove_small_holes(mask, max_size=fill_px) if fill_px > 0 else mask
    gray = rgb.mean(axis=2) if rgb is not None else None

    def _is_clean(gy, gx):
        """Reject a cell holding a big vessel lumen or a fold (needs rgb)."""
        if gray is None:
            return True
        sub_m = mask[gy:gy + step, gx:gx + step]
        # Largest connected background blob (edge-touching included), so a big
        # vessel/portal lumen at the cell border is caught too, not only fully
        # enclosed lumens.
        lbl, k = ndi.label(~sub_m)
        if k and ndi.sum(np.ones_like(lbl), lbl, range(1, k + 1)).max() \
                / sub_m.size * 100 > lumen_max_pct:
            return False                            # big vessel / portal lumen
        dl, dk = ndi.label(gray[gy:gy + step, gx:gx + step] < 80)
        if dk and ndi.sum(np.ones_like(dl), dl, range(1, dk + 1)).max() \
                * (px_level ** 2) > dark_blob_max_um2:
            return False                            # tissue fold / dark blob
        return True

    def candidates(thresh):
        out = []  # (tissue_px, cy, cx, gy, gx, frac)
        for gy in range(0, H - step + 1, step):
            for gx in range(0, W - step + 1, step):
                sub = place[gy:gy + step, gx:gx + step]
                frac = float(sub.mean())
                if frac >= thresh and _is_clean(gy, gx):
                    out.append((int(mask[gy:gy + step, gx:gx + step].sum()),
                                gy + step // 2, gx + step // 2, gy, gx, frac))
        return out

    # Try the requested strictness, then relax step-wise until we have at least
    # min_rois candidates (some pieces are too textured/small for 0.98).
    cands = candidates(min_tissue_frac)
    thresh = min_tissue_frac
    while len(cands) < min_rois and thresh > 0.6:
        thresh = round(thresh - 0.05, 2)
        cands = candidates(thresh)
    if not cands:
        return []

    # Farthest-point sampling to spread ROIs out; stop at coverage or max_rois.
    centers = np.array([(c[1], c[2]) for c in cands], dtype=float)
    covered = 0.0
    target_area = target_coverage * tissue_area_um2
    chosen: list[int] = []
    # seed with the most-tissue cell
    nxt = int(np.argmax([c[0] for c in cands]))
    min_d = np.full(len(cands), np.inf)
    while len(chosen) < max_rois:
        chosen.append(nxt)
        covered += cands[nxt][0] * (px_level ** 2)
        d = np.hypot(centers[:, 0] - centers[nxt, 0],
                     centers[:, 1] - centers[nxt, 1])
        min_d = np.minimum(min_d, d)
        min_d[chosen] = -1
        if covered >= target_area and len(chosen) >= min_rois:
            break
        if np.all(min_d < 0):
            break
        nxt = int(np.argmax(min_d))

    rois = []
    for i in chosen:
        _, _, _, gy, gx, frac = cands[i]
        rois.append(ROI(
            x=int(gx * mask_downsample),
            y=int(gy * mask_downsample),
            w=int(step * mask_downsample),
            h=int(step * mask_downsample),
            tissue_frac=frac,
        ))
    return rois
