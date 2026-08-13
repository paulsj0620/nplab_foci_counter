"""Tiling plan: decide which full-res tiles to process.

Nuclei detection runs at full resolution, but the whole slide (~52000 x 29000)
is far too large to process at once. We split it into a grid of small tiles
(e.g. 512x512 full-res px) and, using the low-res tissue mask, keep only the
tiles that actually contain tissue -- background tiles are skipped, which is
the bulk of the slide.

This module only computes tile *coordinates*; reading the high-res pixels for a
tile is a separate step (a region reader on the loader).

Typical usage::

    from czi_loader import CziSlide
    from tissue_mask import tissue_mask
    from tiling import plan_tiles

    with CziSlide(path) as slide:
        info = slide.info()
        rgb = slide.read_level(9)
    mask = tissue_mask(rgb, pixel_size_um=info.level_pixel_size_um(9))
    tiles = plan_tiles(mask, mask_downsample=9,
                       full_width=info.full_width, full_height=info.full_height)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Tile:
    """One tile to be processed, in full-res pixel coordinates."""

    tx: int              # column index in the tile grid
    ty: int              # row index in the tile grid
    x: int               # full-res left
    y: int               # full-res top
    w: int               # full-res width (smaller than tile_size at right edge)
    h: int               # full-res height (smaller than tile_size at bottom edge)
    tissue_frac: float   # fraction of this tile covered by tissue (0-1)


def plan_tiles(
    mask: np.ndarray,
    mask_downsample: int,
    full_width: int,
    full_height: int,
    tile_size: int = 512,
    min_tissue_frac: float = 0.05,
) -> list[Tile]:
    """Return the list of tiles that contain tissue, in full-res coordinates.

    Parameters
    ----------
    mask : np.ndarray
        (H, W) boolean tissue mask, computed at ``mask_downsample``.
    mask_downsample : int
        Downsample factor of ``mask`` relative to full resolution (e.g. 9).
    full_width, full_height : int
        Full-resolution slide dimensions in pixels.
    tile_size : int
        Tile edge length in full-res pixels.
    min_tissue_frac : float
        Keep a tile only if at least this fraction of it is tissue.

    Returns
    -------
    list[Tile]
        Tiles overlapping tissue, in row-major order.
    """
    # Tile edge measured in mask pixels (mask is downsampled vs full-res).
    mask_tile = tile_size / mask_downsample

    n_cols = -(-full_width // tile_size)   # ceil division
    n_rows = -(-full_height // tile_size)
    mh, mw = mask.shape

    tiles: list[Tile] = []
    for ty in range(n_rows):
        # Mask row band for this tile row, clamped to the mask height.
        my0 = int(ty * mask_tile)
        my1 = min(int((ty + 1) * mask_tile), mh)
        if my1 <= my0:
            continue
        for tx in range(n_cols):
            mx0 = int(tx * mask_tile)
            mx1 = min(int((tx + 1) * mask_tile), mw)
            if mx1 <= mx0:
                continue

            sub = mask[my0:my1, mx0:mx1]
            frac = float(sub.mean()) if sub.size else 0.0
            if frac < min_tissue_frac:
                continue

            x = tx * tile_size
            y = ty * tile_size
            tiles.append(
                Tile(
                    tx=tx,
                    ty=ty,
                    x=x,
                    y=y,
                    w=min(tile_size, full_width - x),
                    h=min(tile_size, full_height - y),
                    tissue_frac=frac,
                )
            )
    return tiles


def overlay_tiles(
    rgb: np.ndarray,
    tiles: list[Tile],
    mask_downsample: int,
    color=(0, 128, 255),
) -> np.ndarray:
    """Draw tile boundaries on a low-res RGB image for visual QC.

    ``tiles`` are in full-res coordinates; ``mask_downsample`` maps them back to
    the resolution of ``rgb`` (which must be the same level as the mask used to
    plan the tiles).
    """
    out = rgb.copy()
    c = np.array(color, dtype=np.uint8)
    h, w = out.shape[:2]
    for t in tiles:
        x0 = int(t.x / mask_downsample)
        y0 = int(t.y / mask_downsample)
        x1 = min(int((t.x + t.w) / mask_downsample), w - 1)
        y1 = min(int((t.y + t.h) / mask_downsample), h - 1)
        if x1 <= x0 or y1 <= y0:
            continue
        out[y0:y1, x0] = c          # left edge
        out[y0:y1, x1] = c          # right edge
        out[y0, x0:x1] = c          # top edge
        out[y1, x0:x1] = c          # bottom edge
    return out
