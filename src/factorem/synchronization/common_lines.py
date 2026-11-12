"""Synchronization by the common lines of the eigenimages.

The Fourier transforms of two projections of a volume coincide along the
line where their central slices intersect. The eigenimages of a projection
direction are projections of the eigenvolumes, mixed by the transform that
its frame differs by. So the eigenimages of any two directions, sampled
along their common line, tell the transform between their frames. This
relates all the pairs of directions, and not only the ones that share
particles, which keeps the frames from drifting across the sphere.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from ..embedding.directional import DirectionalEmbedding
from ..imaging import FourierBand
from .power_method import polar, synchronize

logger = logging.getLogger(__name__)

# Sine of the angle between two directions below which they have no common
# line to speak of.
_MIN_SINE = 1e-3


def random_pairs(
    count: int, partners: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Pairs ``(left < right)`` that link every item to about ``partners``
    others at random, or all the pairs if those are no more."""
    if partners >= count - 1:
        left, right = np.triu_indices(count, 1)
        return left, right
    left = np.repeat(np.arange(count), partners // 2 + 1)
    right = (left + rng.integers(1, count, len(left))) % count
    pairs = np.sort(np.stack((left, right), axis=1), axis=1)
    pairs = np.unique(pairs, axis=0)
    return pairs[:, 0], pairs[:, 1]


def _coefficient_table(band: FourierBand) -> tuple[np.ndarray, int]:
    """Where the coefficient of each frequency (ky + reach, kx) is, and the
    index past the last coefficient where there is none."""
    reach = int(band.max_radius) + 1
    table = np.full((2 * reach + 1, reach + 1), len(band), dtype=np.int64)
    kx, ky = band.frequencies.T
    table[ky + reach, kx] = np.arange(len(band))
    return table, reach


def _sample_lines(
    eigenimages: np.ndarray,
    table: np.ndarray,
    reach: int,
    items: np.ndarray,
    directions: np.ndarray,
    radii: np.ndarray,
) -> np.ndarray:
    """Eigenimages along a line through the origin, by bilinear
    interpolation.

    Args:
        eigenimages: ``(items, coefficients + 1, components)``, the last
            coefficient being zero.
        table: Where the coefficient of each frequency is.
        reach: Largest frequency of the table.
        items: Whose eigenimages to sample, ``(lines,)``.
        directions: Unit vector (x, y) of each line, ``(lines, 2)``.
        radii: Where along the line to sample, ``(samples,)``.

    Returns:
        ``(lines, samples, components)``.
    """
    # Only one of each Hermitian pair is stored: the other is its conjugate.
    mirrored = directions[:, 0] < 0.0
    directions = np.where(mirrored[:, None], -directions, directions)
    x = directions[:, 0, None] * radii
    y = directions[:, 1, None] * radii
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    fx = (x - x0).astype(np.float32)
    fy = (y - y0).astype(np.float32)

    rows = items[:, None]
    out = np.zeros((*x.shape, eigenimages.shape[-1]), dtype=eigenimages.dtype)
    for dy, wy in ((0, 1.0 - fy), (1, fy)):
        for dx, wx in ((0, 1.0 - fx), (1, fx)):
            coefficient = table[y0 + dy + reach, x0 + dx]
            out += (wx * wy)[..., None] * eigenimages[rows, coefficient]
    return np.where(mirrored[:, None, None], out.conj(), out)


def common_line_products(
    eigenimages: np.ndarray,
    band: FourierBand,
    frames: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    chunk: int = 16384,
) -> np.ndarray:
    """Products of the eigenimages of pairs of directions on their common
    lines.

    Args:
        eigenimages: Fourier coefficients of the eigenimages of each
            direction, ``(directions, coefficients, components)``.
        band: The coefficients they are made of.
        frames: Frame of each direction, ``(directions, 3, 3)``.
        left: First direction of each pair.
        right: Second direction of each pair.
        chunk: How many pairs to work on at once.

    Returns:
        A real ``(components, components)`` block for each pair: the inner
        products of the eigenimages of the first with those of the second,
        along the line. It is zero for the pairs that view along the same
        axis, which have no common line.
    """
    count, _, components = eigenimages.shape
    padded = np.zeros((count, len(band) + 1, components), eigenimages.dtype)
    padded[:, :-1] = eigenimages
    table, reach = _coefficient_table(band)
    radii = np.arange(1, int(band.max_radius), dtype=np.float64)

    blocks = np.zeros((len(left), components, components))
    for start in range(0, len(left), chunk):
        span = slice(start, start + chunk)
        line = np.cross(frames[left[span], 2], frames[right[span], 2])
        sine = np.linalg.norm(line, axis=1)
        line /= np.maximum(sine, 1e-12)[:, None]

        sides = []
        for items in (left[span], right[span]):
            directions = np.stack(
                [
                    np.einsum('nj,nj->n', frames[items, axis], line)
                    for axis in (0, 1)
                ],
                axis=1,
            )
            sides.append(
                _sample_lines(padded, table, reach, items, directions, radii)
            )
        a, b = sides
        # Both halves of the line: the opposite one is the conjugate.
        products = 2.0 * np.real(a.conj().swapaxes(-2, -1) @ b)
        blocks[span] = np.where((sine >= _MIN_SINE)[:, None, None], products, 0)
    return blocks


@dataclass(frozen=True)
class CommonLineSynchronizer:
    """Synchronizes the directions by the common lines of their
    eigenimages.

    Each direction is compared with a random few of the others, which is
    as good as comparing it with all of them, at a fraction of the cost.
    """

    partners: int = 64
    """About how many directions to compare each one with."""
    max_iterations: int = 200
    """Iterations of the power method to stop at."""
    tolerance: float = 1e-6
    """Relative gain of an iteration below which the power method stops."""
    seed: int = 0
    """Seed of the choice of pairs and of the initial transforms."""

    def __call__(self, embedding: DirectionalEmbedding) -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        count = len(embedding.groups)
        components = embedding.components
        left, right = random_pairs(count, self.partners, rng)
        blocks = common_line_products(
            embedding.eigenimages,
            embedding.band,
            embedding.groups.frames,
            left,
            right,
        )
        start = polar(rng.normal(size=(count, components, components)))
        transforms, iterations = synchronize(
            left, right, blocks, start, self.max_iterations, self.tolerance
        )
        logger.info(
            'Synchronized %d directions from %d common lines in %d iterations',
            count,
            len(left),
            iterations,
        )
        return transforms
