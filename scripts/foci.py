"""Step 5: cluster inflammatory nuclei into foci and count them.

The reference pipeline dilates the inflammatory-nuclei mask with a disc kernel
and labels connected components. On a point set the equivalent -- and far
cheaper than rasterizing a 52000x29000 mask -- is **DBSCAN**: nuclei within a
linking distance of each other form one cluster (a focus).

A focus is kept only if it contains at least ``min_cells`` inflammatory nuclei.
This dataset is mouse liver, so the default follows the rodent NAFLD standard
(Liang et al. 2014): a focus is a cluster of **>= 5** inflammatory cells. (The
human NASH-CRN definition is >= 2; the value is configurable.)

Typical usage::

    from foci import cluster_foci
    result = cluster_foci(infl_points_xy, pixel_size_um)
    print(result.n_foci)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import DBSCAN


@dataclass
class FociResult:
    """Foci detected from a set of inflammatory nucleus centroids."""

    labels: np.ndarray        # (M,) cluster id per inflammatory nucleus (-1 = noise)
    centers_xy: np.ndarray    # (n_foci, 2) focus centroids (x, y), full-res px
    sizes: np.ndarray         # (n_foci,) number of nuclei in each focus

    @property
    def n_foci(self) -> int:
        return int(self.centers_xy.shape[0])


def cluster_foci(
    infl_points_xy: np.ndarray,
    pixel_size_um: float,
    link_distance_um: float = 20.0,
    min_cells: int = 5,
) -> FociResult:
    """Cluster inflammatory nuclei into foci.

    Parameters
    ----------
    infl_points_xy : np.ndarray
        (M, 2) centroids of nuclei already classified as inflammatory
        (full-res pixel coordinates).
    pixel_size_um : float
        Full-res pixel size (µm/px), to convert ``link_distance_um`` to pixels.
    link_distance_um : float
        Two inflammatory nuclei closer than this join the same focus.
    min_cells : int
        Minimum inflammatory nuclei for a cluster to count as a focus.

    Returns
    -------
    FociResult
    """
    m = infl_points_xy.shape[0]
    if m == 0:
        return FociResult(
            labels=np.empty((0,), dtype=int),
            centers_xy=np.empty((0, 2)),
            sizes=np.empty((0,), dtype=int),
        )

    eps_px = link_distance_um / pixel_size_um
    # min_samples=min_cells so that only sufficiently crowded points seed a
    # cluster; DBSCAN then also drops clusters that never reach min_cells.
    db = DBSCAN(eps=eps_px, min_samples=min_cells).fit(infl_points_xy)
    labels = db.labels_

    centers, sizes = [], []
    for cid in range(labels.max() + 1 if labels.size else 0):
        pts = infl_points_xy[labels == cid]
        if pts.shape[0] < min_cells:
            continue
        centers.append(pts.mean(axis=0))
        sizes.append(pts.shape[0])

    return FociResult(
        labels=labels,
        centers_xy=np.array(centers) if centers else np.empty((0, 2)),
        sizes=np.array(sizes, dtype=int),
    )
