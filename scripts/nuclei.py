"""Nuclei detection on a full-res tile using a pretrained StarDist model.

Wraps the StarDist ``2D_versatile_he`` H&E model. The heavy model is loaded once
and cached at module level, so callers can run ``detect_nuclei`` over thousands
of tiles without reloading.

For each tile we return the per-nucleus centroids and areas -- the inputs needed
by the next step, which classifies nuclei as inflammatory using spatial
proximity and clusters them into foci.

Typical usage::

    from nuclei import detect_nuclei
    res = detect_nuclei(tile_rgb)          # tile_rgb: (H, W, 3) uint8
    print(res.count, res.centroids.shape)  # centroids are (N, 2) as (y, x)
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

# StarDist / TensorFlow are heavy and noisy on import; keep them lazy so that
# importing this module (e.g. for tests of pure functions) stays cheap.
_MODEL = None
_MODEL_NAME = "2D_versatile_he"


def get_model():
    """Return the pretrained StarDist H&E model, loading it once and caching."""
    global _MODEL
    if _MODEL is None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from stardist.models import StarDist2D
            _MODEL = StarDist2D.from_pretrained(_MODEL_NAME)
    return _MODEL


@dataclass
class NucleiResult:
    """Detected nuclei for a single tile."""

    centroids: np.ndarray   # (N, 2) float, (y, x) in tile pixel coordinates
    areas_px: np.ndarray    # (N,) int, nucleus area in pixels
    probs: np.ndarray       # (N,) float, StarDist detection probability
    labels: np.ndarray      # (H, W) int32 instance label map (0 = background)

    @property
    def count(self) -> int:
        return int(self.centroids.shape[0])

    def areas_um2(self, pixel_size_um: float) -> np.ndarray:
        """Nucleus areas in µm² given the tile's pixel size (µm/px)."""
        return self.areas_px.astype(float) * (pixel_size_um ** 2)

    def to_global(self, origin_xy: tuple[int, int], downsample: int = 1) -> np.ndarray:
        """Centroids mapped to whole-slide full-res coords, as (N, 2) (x, y).

        Parameters
        ----------
        origin_xy : (int, int)
            The tile's full-res top-left (x, y), e.g. ``(tile.x, tile.y)``.
        downsample : int
            Downsample the tile was read at (1 for full-res).
        """
        ox, oy = origin_xy
        xs = self.centroids[:, 1] * downsample + ox
        ys = self.centroids[:, 0] * downsample + oy
        return np.column_stack([xs, ys])


def detect_nuclei(
    tile_rgb: np.ndarray,
    prob_thresh: float | None = None,
    nms_thresh: float | None = None,
) -> NucleiResult:
    """Detect nuclei in a full-res RGB tile.

    Parameters
    ----------
    tile_rgb : np.ndarray
        (H, W, 3) uint8 RGB tile at full resolution.
    prob_thresh, nms_thresh : float, optional
        Override StarDist's default detection / non-max-suppression thresholds.
        Higher ``prob_thresh`` = fewer, more confident detections.

    Returns
    -------
    NucleiResult
    """
    from csbdeep.utils import normalize
    from skimage.measure import regionprops

    model = get_model()
    img = normalize(tile_rgb, 1, 99.8, axis=(0, 1))
    labels, details = model.predict_instances(
        img, prob_thresh=prob_thresh, nms_thresh=nms_thresh
    )

    centroids = np.asarray(details["points"], dtype=float)  # (N, 2) as (y, x)
    probs = np.asarray(details["prob"], dtype=float)
    if centroids.size == 0:
        return NucleiResult(
            centroids=np.empty((0, 2)),
            areas_px=np.empty((0,), dtype=int),
            probs=np.empty((0,)),
            labels=labels,
        )
    # Area per label, ordered to match StarDist's label ids (1..N).
    areas_px = np.array([p.area for p in regionprops(labels)], dtype=int)
    return NucleiResult(
        centroids=centroids, areas_px=areas_px, probs=probs, labels=labels
    )
