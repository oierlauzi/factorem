"""Planes to look at the coordinates of the particles on."""

from __future__ import annotations

import numpy as np

# Share of the points, at each end of an axis, that may be left out of
# sight. A few particles far from all the others would otherwise squeeze
# the rest into a corner.
_OUT_OF_SIGHT = 1e-3


def umap_projection(coordinates: np.ndarray) -> np.ndarray:
    """Lay points out on a plane by UMAP (McInnes, Healy and Melville,
    2018), which keeps the neighbours of each point close to it.

    Args:
        coordinates: ``(points, dimensions)``.

    Returns:
        The points on the plane, ``(points, 2)``. Their layout changes
        from one time to the next: a fixed one would leave all the cores
        but one idle, and take several times longer.

    Raises:
        ImportError: Without umap-learn, which the ``umap`` extra brings.
    """
    try:
        import umap
    except ImportError as error:
        raise ImportError(
            "UMAP needs umap-learn: pip install 'factorem[umap]'."
        ) from error
    reducer = umap.UMAP(n_components=2)
    return np.asarray(reducer.fit_transform(coordinates), dtype=np.float64)


def sight(points: np.ndarray, aspect: float = 1.0) -> np.ndarray:
    """Limits of a view of points that distorts no distance and wastes
    no room on a few points far from the others.

    Args:
        points: ``(points, 2)``.
        aspect: Width of the view over its height.

    Returns:
        The limits of both axes, ``((left, right), (bottom, top))``.
    """
    low, high = np.quantile(points, (_OUT_OF_SIGHT, 1.0 - _OUT_OF_SIGHT), 0)
    centre = 0.5 * (low + high)
    # As tall as the view, with a margin. A degenerate cloud gets some room.
    extent = high - low
    half = 0.55 * max(extent[0] / aspect, extent[1]) or 1.0
    reach = half * np.array((aspect, 1.0))
    return np.stack((centre - reach, centre + reach), axis=1)
