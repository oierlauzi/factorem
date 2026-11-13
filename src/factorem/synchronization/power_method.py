"""Orthogonal synchronization by the generalized power method."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

# More threads than this gain little.
_MAX_THREADS = 16


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
        right: Second item of each pair. It may be the first one, with a
            symmetric block that tells how sure the item is of itself.
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

    # Every pair, seen from each of its two items, sorted by the first. An
    # item paired with itself is seen once.
    mirrored = left != right
    sources = np.concatenate((left, right[mirrored]))
    order = np.argsort(sources, kind='stable')
    sources = sources[order]
    targets = np.concatenate((right, left[mirrored]))[order]
    blocks = np.asarray(blocks, dtype=np.float64)
    blocks = np.concatenate((blocks, blocks[mirrored].swapaxes(-2, -1)))[order]
    linked, first = np.unique(sources, return_index=True)

    # NumPy goes through a stack of small matrices one by one, and lets
    # other threads run meanwhile, so the items are shared out among them.
    pieces = max(1, min(os.cpu_count() or 1, _MAX_THREADS, len(linked) // 64))
    bounds = np.linspace(0, len(linked), pieces + 1).astype(np.int64)
    edges = np.append(first, len(targets))

    def seen(piece: int) -> np.ndarray:
        """What the items of a piece are told by the ones they pair with."""
        items = slice(bounds[piece], bounds[piece + 1])
        pairs = slice(edges[items.start], edges[items.stop])
        told = blocks[pairs] @ x[targets[pairs]]
        total: np.ndarray = np.add.reduceat(
            told, first[items] - pairs.start, axis=0
        )
        return total

    previous = None
    iterations = 0
    with ThreadPoolExecutor(pieces) as pool:
        while iterations < max_iterations:
            iterations += 1
            y = list(pool.map(seen, range(pieces)))
            value = float(np.sum(x[linked] * np.concatenate(y)))
            if previous is not None and value - previous < tolerance * abs(
                value
            ):
                break
            previous = value
            # What nothing links to stays where it is.
            x[linked] = np.concatenate(list(pool.map(polar, y)))
    return x, iterations
