"""Orthogonal synchronization by the generalized power method."""

from __future__ import annotations

import numpy as np


def polar(blocks: np.ndarray) -> np.ndarray:
    """Closest matrices with orthonormal rows, or columns if they are
    fewer."""
    u, _, vh = np.linalg.svd(blocks, full_matrices=False)
    return np.asarray(u @ vh)


def synchronize(
    left: np.ndarray,
    right: np.ndarray,
    blocks: np.ndarray,
    start: np.ndarray,
    max_iterations: int = 200,
    tolerance: float = 1e-6,
) -> tuple[np.ndarray, int]:
    """Transforms that agree best with the ones measured between pairs.

    Maximizes the sum, over the pairs, of ``trace(X[l].T @ C @ X[r])`` for
    matrices ``X`` of orthonormal rows, by iterating
    ``X <- polar(S X)``, where ``S`` is the symmetric matrix that the blocks
    of the pairs make.

    Args:
        left: First item of each pair, ``(pairs,)``.
        right: Second item of each pair.
        blocks: Measurement ``C`` of each pair, ``(pairs, k, k)``: about
            ``X[l] @ X[r].T`` up to a positive scale.
        start: Initial transforms, ``(items, k, rank)``.
        max_iterations: Iterations to stop at.
        tolerance: Relative gain of the objective in one iteration below
            which to stop.

    Returns:
        The transforms, like ``start``, and the iterations it took.
    """
    x = np.asarray(start, dtype=np.float64)
    if len(left) == 0:
        return x, 0

    # Every pair, seen from each of its two items, sorted by the first.
    sources = np.concatenate((left, right))
    order = np.argsort(sources, kind='stable')
    sources = sources[order]
    targets = np.concatenate((right, left))[order]
    blocks = np.asarray(blocks, dtype=np.float64)
    blocks = np.concatenate((blocks, blocks.swapaxes(-2, -1)))[order]
    linked, first = np.unique(sources, return_index=True)

    previous = None
    iterations = 0
    while iterations < max_iterations:
        iterations += 1
        y = np.zeros_like(x)
        y[linked] = np.add.reduceat(blocks @ x[targets], first, axis=0)
        value = float(np.sum(x * y))
        if previous is not None and value - previous < tolerance * abs(value):
            break
        previous = value
        # What nothing links to stays where it is.
        x[linked] = polar(y[linked])
    return x, iterations
