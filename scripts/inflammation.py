"""Step 4: classify nuclei as inflammatory using size + spatial proximity.

Inflammatory cells (lymphocytes, etc.) show up as **small nuclei packed densely
together**, unlike the larger, more spaced-out hepatocyte nuclei of normal liver
parenchyma. Following the reference pipeline's "spatial proximity criteria", a
nucleus is flagged inflammatory when it is both:

  * small       -- area below ``max_area_um2``, and
  * in a crowd  -- at least ``min_neighbors`` other nuclei within ``radius_um``.

Works on the global (whole-slide) point set so clusters spanning tile borders
are handled naturally.

Typical usage::

    from inflammation import classify_inflammatory
    is_infl = classify_inflammatory(points_xy, areas_um2, pixel_size_um)
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def neighbor_counts(points_xy: np.ndarray, radius_px: float) -> np.ndarray:
    """Number of *other* nuclei within ``radius_px`` of each nucleus.

    Parameters
    ----------
    points_xy : np.ndarray
        (N, 2) nucleus centroids in pixel coordinates.
    radius_px : float
        Neighborhood radius in pixels.

    Returns
    -------
    np.ndarray
        (N,) int, neighbor count excluding the nucleus itself.
    """
    if points_xy.shape[0] == 0:
        return np.empty((0,), dtype=int)
    tree = cKDTree(points_xy)
    # count_neighbors of the tree with itself gives, per point, the number of
    # points within radius (including self) -> subtract 1.
    counts = tree.query_ball_point(points_xy, r=radius_px, return_length=True)
    return np.asarray(counts, dtype=int) - 1


def classify_inflammatory(
    points_xy: np.ndarray,
    areas_um2: np.ndarray,
    pixel_size_um: float,
    radius_um: float = 25.0,
    min_neighbors: int = 8,
    max_area_um2: float = 45.0,
) -> np.ndarray:
    """Return a boolean (N,) mask flagging inflammatory nuclei.

    Parameters
    ----------
    points_xy : np.ndarray
        (N, 2) global centroids in full-res pixel coordinates.
    areas_um2 : np.ndarray
        (N,) nucleus areas in µm².
    pixel_size_um : float
        Full-res pixel size (µm/px), to convert ``radius_um`` to pixels.
    radius_um : float
        Neighborhood radius for the density test.
    min_neighbors : int
        Minimum neighbors within the radius to count as "in a crowd".
    max_area_um2 : float
        Upper size limit for a nucleus to be considered inflammatory.

    Returns
    -------
    np.ndarray
        (N,) boolean; True = inflammatory.
    """
    n = points_xy.shape[0]
    if n == 0:
        return np.empty((0,), dtype=bool)
    radius_px = radius_um / pixel_size_um
    dense = neighbor_counts(points_xy, radius_px) >= min_neighbors
    small = areas_um2 <= max_area_um2
    return dense & small
