"""Split a slide into separate tissue pieces and pick the cleanest one.

Some slides carry two or more tissue sections. When asked, we analyze only the
best one: each connected tissue piece is scored by artifact load (tears +
dark debris/folds) and the lowest-scoring (cleanest) piece is selected. All
downstream steps (ROIs, foci, tissue area) are then restricted to it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi
from skimage.measure import label, regionprops


@dataclass
class Piece:
    """One connected tissue section and its quality metrics."""

    index: int              # 1-based rank by area (for display)
    mask: np.ndarray        # (H, W) bool, this piece only
    area_mm2: float
    tear_pct: float         # enclosed holes as % of piece area
    dark_pct: float         # very dark (fold/debris) pixels as % of piece area
    artifact_score: float   # lower = cleaner

    @property
    def centroid_xy(self):
        ys, xs = np.nonzero(self.mask)
        return float(xs.mean()), float(ys.mean())


def find_pieces(
    mask: np.ndarray,
    rgb: np.ndarray,
    pixel_size_um: float,
    mask_downsample: int,
    min_area_mm2: float = 1.0,
    dark_level: int = 60,
) -> list[Piece]:
    """Return tissue pieces (area >= min_area_mm2), largest first.

    ``pixel_size_um`` is the full-res value; the mask/rgb are at
    ``mask_downsample``.
    """
    px_level = pixel_size_um * mask_downsample
    gray = rgb.mean(axis=2)
    lbl = label(mask)
    pieces = []
    for p in sorted(regionprops(lbl), key=lambda r: -r.area):
        area_mm2 = p.area * (px_level / 1000) ** 2
        if area_mm2 < min_area_mm2:
            continue
        pm = lbl == p.label
        tear = float((ndi.binary_fill_holes(pm) & ~pm).sum()) / p.area * 100
        dark = float(((gray < dark_level) & pm).sum()) / p.area * 100
        pieces.append(Piece(
            index=len(pieces) + 1,
            mask=pm,
            area_mm2=area_mm2,
            tear_pct=tear,
            dark_pct=dark,
            artifact_score=tear + 2 * dark,     # tears + weighted dark debris
        ))
    return pieces


def best_piece(pieces: list[Piece]) -> Piece | None:
    """The cleanest piece (lowest artifact score), or None if there are none."""
    return min(pieces, key=lambda p: p.artifact_score) if pieces else None
