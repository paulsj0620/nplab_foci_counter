"""Shared step-4/5 analysis: classify inflammatory -> cluster foci -> QC filter.

Used by both ``analyze_foci`` (counts + overlay) and ``foci_gallery`` (review
crops) so they always apply the *same* logic. On top of classification and
clustering it drops foci whose local neighborhood is mostly non-tissue -- these
sit on torn/edge regions with lots of empty space and are usually false
positives.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi
from skimage.color import rgb2hsv

from inflammation import classify_inflammatory
from foci import cluster_foci


@dataclass
class FociAnalysis:
    is_infl: np.ndarray       # (N,) bool over all nuclei
    infl_xy: np.ndarray       # (M, 2) inflammatory centroids (x, y)
    labels: np.ndarray        # (M,) DBSCAN cluster id per inflammatory nucleus
    kept_ids: np.ndarray      # cluster ids that passed the tissue-fraction QC
    centers_xy: np.ndarray    # (K, 2) centers of kept foci
    sizes: np.ndarray         # (K,) cell counts of kept foci

    @property
    def n_foci(self) -> int:
        return int(self.centers_xy.shape[0])


def _windows(centers_xy, mask_downsample, half_window_um, pixel_size_um, shape):
    """Yield (index, (y0, y1, x0, x1)) level-image windows around each center."""
    h, w = shape
    px_level = pixel_size_um * mask_downsample
    hw = max(1, int(half_window_um / px_level))
    for i, (cx, cy) in enumerate(centers_xy):
        mx, my = int(cx / mask_downsample), int(cy / mask_downsample)
        yield i, (max(0, my - hw), my + hw, max(0, mx - hw), mx + hw)


def local_tissue_fraction(
    centers_xy: np.ndarray,
    mask: np.ndarray,
    mask_downsample: int,
    half_window_um: float,
    pixel_size_um: float,
) -> np.ndarray:
    """Tissue fraction in a window around each focus center (0-1).

    Centers are in full-res coords; the mask is at ``mask_downsample``.
    """
    out = np.empty(centers_xy.shape[0])
    for i, (y0, y1, x0, x1) in _windows(
        centers_xy, mask_downsample, half_window_um, pixel_size_um, mask.shape
    ):
        sub = mask[y0:y1, x0:x1]
        out[i] = float(sub.mean()) if sub.size else 0.0
    return out


def local_brightness(
    centers_xy: np.ndarray,
    rgb: np.ndarray,
    mask_downsample: int,
    half_window_um: float,
    pixel_size_um: float,
) -> np.ndarray:
    """Mean brightness (0-255) of non-white pixels in a window around each focus.

    Over-stained / folded artifacts are abnormally dark; a low value here flags
    a focus that sits on such a region rather than on genuine inflammation.
    """
    out = np.empty(centers_xy.shape[0])
    for i, (y0, y1, x0, x1) in _windows(
        centers_xy, mask_downsample, half_window_um, pixel_size_um, rgb.shape[:2]
    ):
        sub = rgb[y0:y1, x0:x1]
        if sub.size == 0:
            out[i] = 255.0
            continue
        g = sub.mean(axis=2)
        tissue = g < 220
        out[i] = float(g[tissue].mean()) if tissue.any() else 255.0
    return out


def local_saturation(
    centers_xy: np.ndarray,
    rgb: np.ndarray,
    mask_downsample: int,
    half_window_um: float,
    pixel_size_um: float,
) -> np.ndarray:
    """Mean HSV saturation of non-white pixels in a window around each focus.

    Over-stained blotches are abnormally saturated; a high value flags a focus
    that sits on such a region rather than genuine inflammation.
    """
    out = np.empty(centers_xy.shape[0])
    for i, (y0, y1, x0, x1) in _windows(
        centers_xy, mask_downsample, half_window_um, pixel_size_um, rgb.shape[:2]
    ):
        sub = rgb[y0:y1, x0:x1]
        if sub.size == 0:
            out[i] = 0.0
            continue
        tissue = sub.mean(axis=2) < 220
        out[i] = float(rgb2hsv(sub)[..., 1][tissue].mean()) if tissue.any() else 0.0
    return out


def lumen_distance_um(
    centers_xy: np.ndarray,
    mask: np.ndarray,
    mask_downsample: int,
    pixel_size_um: float,
    min_lumen_um2: float = 1250.0,
) -> np.ndarray:
    """Distance (µm) from each focus to the nearest vessel lumen.

    A lumen is an enclosed background region inside the tissue (a hole in the
    mask) at least ``min_lumen_um2`` in area -- i.e. a vessel/duct cavity, not
    the outer glass background. Foci hugging a lumen are perivascular.
    """
    px_level = pixel_size_um * mask_downsample
    enclosed = ndi.binary_fill_holes(mask) & ~mask       # holes inside tissue
    lbl, n = ndi.label(enclosed)
    if n:
        sizes = ndi.sum(np.ones_like(lbl), lbl, range(1, n + 1))
        min_px = min_lumen_um2 / (px_level ** 2)
        big = np.isin(lbl, np.nonzero(sizes >= min_px)[0] + 1)
    else:
        big = np.zeros_like(mask)
    if not big.any():
        return np.full(centers_xy.shape[0], np.inf)
    dist = ndi.distance_transform_edt(~big) * px_level
    h, w = mask.shape
    out = np.empty(centers_xy.shape[0])
    for i, (cx, cy) in enumerate(centers_xy):
        my, mx = min(int(cy / mask_downsample), h - 1), min(int(cx / mask_downsample), w - 1)
        out[i] = dist[my, mx]
    return out


def inflammation_area_mm2(
    infl_points_xy: np.ndarray,
    mask_shape: tuple[int, int],
    mask_downsample: int,
    pixel_size_um: float,
    disc_um: float = 15.0,
) -> float:
    """Area (mm²) covered by inflammation, AIH-style.

    Each inflammatory nucleus is stamped onto a level-resolution canvas and
    dilated by a disc, and the union area is measured -- the "Inflammation
    Density (ID)" numerator. Pass the nuclei belonging to the kept foci.
    """
    from skimage.morphology import dilation, disk

    h, w = mask_shape
    if infl_points_xy.shape[0] == 0:
        return 0.0
    px_level = pixel_size_um * mask_downsample
    canvas = np.zeros((h, w), dtype=bool)
    my = np.clip((infl_points_xy[:, 1] / mask_downsample).astype(int), 0, h - 1)
    mx = np.clip((infl_points_xy[:, 0] / mask_downsample).astype(int), 0, w - 1)
    canvas[my, mx] = True
    # disk is symmetric, so dilation needs no footprint mirroring.
    canvas = dilation(canvas, disk(max(1, int(disc_um / px_level))))
    return float(canvas.sum()) * (px_level / 1000.0) ** 2


def analyze(
    points_xy: np.ndarray,
    areas_um2: np.ndarray,
    pixel_size_um: float,
    mask: np.ndarray,
    mask_downsample: int,
    rgb: np.ndarray | None = None,
    min_tissue_frac: float = 0.90,
    qc_window_um: float = 250.0,
    min_brightness: float = 95.0,
    max_saturation: float = 0.56,
    stain_window_um: float = 120.0,
    min_lumen_dist_um: float = 100.0,
) -> FociAnalysis:
    """Full step 4-5 analysis with QC filters applied.

    A focus is kept only if it is:
      * mostly tissue        (fraction >= ``min_tissue_frac`` over ``qc_window_um``),
      * not over-stained     (brightness >= ``min_brightness`` and
                              saturation <= ``max_saturation`` over the tighter
                              ``stain_window_um``, so a dark blotch is not diluted
                              by surrounding light tissue; needs ``rgb``),
      * not perivascular     (>= ``min_lumen_dist_um`` from a vessel lumen).
    """
    is_infl = classify_inflammatory(points_xy, areas_um2, pixel_size_um)
    infl_xy = points_xy[is_infl]
    foci = cluster_foci(infl_xy, pixel_size_um)

    if foci.n_foci == 0:
        return FociAnalysis(
            is_infl=is_infl, infl_xy=infl_xy, labels=foci.labels,
            kept_ids=np.empty((0,), dtype=int),
            centers_xy=np.empty((0, 2)), sizes=np.empty((0,), dtype=int),
        )

    half = qc_window_um / 2
    frac = local_tissue_fraction(
        foci.centers_xy, mask, mask_downsample, half, pixel_size_um
    )
    keep = frac >= min_tissue_frac
    keep &= lumen_distance_um(
        foci.centers_xy, mask, mask_downsample, pixel_size_um
    ) >= min_lumen_dist_um
    if rgb is not None:
        stain_half = stain_window_um / 2
        keep &= local_brightness(
            foci.centers_xy, rgb, mask_downsample, stain_half, pixel_size_um
        ) >= min_brightness
        keep &= local_saturation(
            foci.centers_xy, rgb, mask_downsample, stain_half, pixel_size_um
        ) <= max_saturation
    # cluster_foci emits one center per DBSCAN cluster id in id order, so the
    # center index equals the cluster id.
    kept_ids = np.nonzero(keep)[0]
    return FociAnalysis(
        is_infl=is_infl, infl_xy=infl_xy, labels=foci.labels,
        kept_ids=kept_ids,
        centers_xy=foci.centers_xy[keep], sizes=foci.sizes[keep],
    )
