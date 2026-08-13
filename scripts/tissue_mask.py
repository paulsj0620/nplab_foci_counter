"""Tissue mask: separate stained tissue from bright glass background.

An H&E whole-slide scan is mostly empty glass. Before detecting foci we build a
boolean **tissue mask** (True = tissue, False = background). This is used to
(a) skip background during later nuclei detection, (b) avoid counting dust/stain
specks as foci, and (c) measure tissue area, the denominator of Focal Density.

The tissue is stained (colored), so it has high **saturation**, while glass is
near-white / gray (low saturation). We threshold saturation, then clean up the
mask with morphology.

Typical usage::

    from czi_loader import CziSlide
    from tissue_mask import tissue_mask, tissue_area_mm2

    with CziSlide(path) as slide:
        rgb = slide.read_level(9)
        px_um = slide.info().level_pixel_size_um(9)
    mask = tissue_mask(rgb)
    area = tissue_area_mm2(mask, px_um)
"""
from __future__ import annotations

import numpy as np
from skimage.color import rgb2hsv
from skimage.morphology import remove_small_objects, remove_small_holes


def tissue_mask(
    rgb: np.ndarray,
    sat_threshold: float = 0.08,
    max_brightness: int = 235,
    min_object_um2: float = 5000.0,
    min_hole_um2: float = 0.0,
    pixel_size_um: float | None = None,
) -> np.ndarray:
    """Return a boolean tissue mask (True = tissue) for an RGB slide image.

    Parameters
    ----------
    rgb : np.ndarray
        (H, W, 3) uint8 RGB, e.g. from ``CziSlide.read_level``.
    sat_threshold : float
        Minimum HSV saturation to count as stained tissue (0-1).
    max_brightness : int
        Pixels brighter than this (0-255) are treated as background even if they
        pass the saturation test (guards against bright scanning artifacts).
    min_object_um2, min_hole_um2 : float
        Small blobs / holes below this physical area are removed / filled. Only
        applied when ``pixel_size_um`` is given; otherwise interpreted directly
        as pixel counts. ``min_hole_um2`` defaults to 0 so tissue tears/cracks
        stay excluded from the mask (and thus from the tissue-area denominator);
        raise it only to fill genuine interior gaps.
    pixel_size_um : float, optional
        Physical size of one pixel in the given image (µm/px), used to convert
        the µm² cleanup thresholds to pixels. If None, thresholds are pixels.

    Returns
    -------
    np.ndarray
        (H, W) boolean mask.
    """
    hsv = rgb2hsv(rgb)
    saturation = hsv[..., 1]
    brightness = rgb.mean(axis=2)

    mask = (saturation > sat_threshold) & (brightness < max_brightness)

    # Convert µm² cleanup thresholds to pixel counts when calibration is known.
    if pixel_size_um is not None:
        px_area = pixel_size_um ** 2
        min_obj_px = int(min_object_um2 / px_area)
        min_hole_px = int(min_hole_um2 / px_area)
    else:
        min_obj_px = int(min_object_um2)
        min_hole_px = int(min_hole_um2)

    # Fill small interior gaps, then drop small isolated specks.
    # NOTE: skimage >=0.26 renamed both size parameters to `max_size`
    # (removes features whose area is <= the value).
    if min_hole_px > 0:
        mask = remove_small_holes(mask, max_size=min_hole_px)
    if min_obj_px > 0:
        mask = remove_small_objects(mask, max_size=min_obj_px)
    return mask


def tissue_area_mm2(mask: np.ndarray, pixel_size_um: float) -> float:
    """Total tissue area in mm² from a boolean mask and its pixel size (µm/px)."""
    px_area_mm2 = (pixel_size_um / 1000.0) ** 2
    return float(mask.sum()) * px_area_mm2


def overlay_mask(
    rgb: np.ndarray, mask: np.ndarray, color=(255, 0, 0), alpha: float = 0.35
) -> np.ndarray:
    """Return an RGB image with the mask tinted on top, for visual QC."""
    out = rgb.astype(np.float32).copy()
    tint = np.array(color, dtype=np.float32)
    out[mask] = (1 - alpha) * out[mask] + alpha * tint
    return np.clip(out, 0, 255).astype(np.uint8)
