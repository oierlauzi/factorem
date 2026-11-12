"""Common lines between the eigenimages of different projection directions.

The Fourier transforms of two projections of a volume coincide along the
line where their central slices intersect. The same holds for every
eigenvolume, so the eigenimages of any two directions, sampled on their
common line, are related by the transform between their latent frames. This
links all pairs of directions, not only those that share particles.
"""
from typing import Tuple

import numpy as np


def polar_resample(values: np.ndarray, frequencies: np.ndarray, angle_count: int,
                   max_radius: int) -> np.ndarray:
    """Bilinear samples of Hermitian images given on a half-plane.

    values: (..., features) complex coefficients at integer ``frequencies``
    (features, 2) as (kx, ky). Returns (..., angle_count, max_radius) samples
    at radii 1..max_radius along rays at angles 2 pi a / angle_count.
    """
    size = max_radius + 2
    dense = np.zeros(values.shape[:-1] + (2 * size + 1, 2 * size + 1), dtype=values.dtype)
    kx, ky = frequencies.T
    inside = (np.abs(kx) <= size) & (np.abs(ky) <= size)
    dense[..., ky[inside] + size, kx[inside] + size] = values[..., inside]
    dense[..., -ky[inside] + size, -kx[inside] + size] = np.conj(values[..., inside])

    angle = 2.0 * np.pi * np.arange(angle_count) / angle_count
    radius = np.arange(1, max_radius + 1)
    x = np.cos(angle)[:, None] * radius + size
    y = np.sin(angle)[:, None] * radius + size
    x0 = np.floor(x).astype(int)
    y0 = np.floor(y).astype(int)
    fx, fy = x - x0, y - y0
    return (
        dense[..., y0, x0] * ((1 - fx) * (1 - fy))
        + dense[..., y0, x0 + 1] * (fx * (1 - fy))
        + dense[..., y0 + 1, x0] * ((1 - fx) * fy)
        + dense[..., y0 + 1, x0 + 1] * (fx * fy)
    )


def common_line_angles(matrices: np.ndarray, left: np.ndarray, right: np.ndarray,
                       flip: Tuple[float, float] = (1.0, 1.0)) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """In-plane angles of the common line of each pair of directions, and
    the sine of the angle between the two directions."""
    line = np.cross(matrices[left, 2], matrices[right, 2])
    sine = np.linalg.norm(line, axis=1)
    line = line / np.maximum(sine, 1e-12)[:, None]

    def angle(index):
        x = np.einsum('nj,nj->n', matrices[index, 0], line) * flip[0]
        y = np.einsum('nj,nj->n', matrices[index, 1], line) * flip[1]
        return np.arctan2(y, x)

    return angle(left), angle(right), sine


def _lookup(polar: np.ndarray, index: np.ndarray, angle: np.ndarray) -> np.ndarray:
    """Samples of each image's ray at the given angle, linearly interpolated
    between the tabulated angles."""
    count = polar.shape[1]
    position = angle / (2.0 * np.pi) * count
    lower = np.floor(position).astype(int)
    fraction = (position - lower)[:, None, None] if polar.ndim == 4 else (position - lower)[:, None]
    return (
        polar[index, lower % count] * (1 - fraction)
        + polar[index, (lower + 1) % count] * fraction
    )


def line_similarities(polar: np.ndarray, matrices: np.ndarray, left: np.ndarray,
                      right: np.ndarray, flip=(1.0, 1.0)) -> Tuple[np.ndarray, np.ndarray]:
    """Cross products of the eigenimages of each pair on their common line.

    polar: (directions, angles, radii, components). Returns the (pairs,
    components, components) real blocks and the sine between directions.
    """
    angle_left, angle_right, sine = common_line_angles(matrices, left, right, flip)
    a = _lookup(polar, left, angle_left)
    b = _lookup(polar, right, angle_right)
    # Both rays of the line: the opposite one is the complex conjugate.
    return 2.0 * np.real(np.einsum('nrj,nrk->njk', np.conj(a), b)), sine


def random_pairs(count: int, partners: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """Pairs (left < right) linking every item to about ``partners`` random
    others; all pairs when ``partners`` is None or covers them."""
    if partners is None or partners >= count - 1:
        return np.triu_indices(count, 1)
    left = np.repeat(np.arange(count), partners // 2 + 1)
    right = (left + rng.integers(1, count, len(left))) % count
    pairs = np.unique(np.sort(np.stack((left, right), axis=1), axis=1), axis=0)
    return pairs[:, 0], pairs[:, 1]


def similarities(loadings: np.ndarray, frequencies: np.ndarray, matrices: np.ndarray,
                 partners=None, angle_count: int = 360, chunk: int = 20000,
                 rng: np.random.Generator = None):
    """Synchronization matrix from the common lines of the eigenimages.

    loadings: (directions, features, components) complex eigenimages at the
    integer ``frequencies`` (kx, ky); matrices: (directions, 3, 3) frames
    whose rows are the image axes and the projection direction.
    """
    import scipy.sparse
    m, _, k = loadings.shape
    max_radius = int(np.hypot(*frequencies.T).max()) - 1
    polar = polar_resample(np.moveaxis(loadings, -1, 1), frequencies, angle_count, max_radius)
    polar = np.moveaxis(polar, 1, -1)

    left, right = random_pairs(m, partners, rng or np.random.default_rng(0))
    blocks = np.empty((len(left), k, k))
    for start in range(0, len(left), chunk):
        span = slice(start, start + chunk)
        blocks[span], sine = line_similarities(polar, matrices, left[span], right[span])
        blocks[span][sine < 1e-3] = 0.0
    upper = scipy.sparse.coo_array(
        (np.arange(1, len(left) + 1), (left, right)), shape=(m, m)).tocsr()
    order = upper.data - 1
    upper = scipy.sparse.bsr_array(
        (blocks[order], upper.indices, upper.indptr), shape=(m * k, m * k))
    out = (upper + upper.T).tobsr(blocksize=(k, k))
    out.sort_indices()
    return out
