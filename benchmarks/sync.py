"""Orthogonal synchronization solvers tried in the benchmark."""
from typing import Optional, Tuple

import numpy as np
import scipy.sparse


def polar(blocks: np.ndarray) -> np.ndarray:
    """Closest matrices with orthonormal rows."""
    u, _, vh = np.linalg.svd(blocks, full_matrices=False)
    return u @ vh


def objective(similarities, x: np.ndarray) -> float:
    flat = x.reshape(-1, x.shape[-1])
    return float(np.sum(flat * (similarities @ flat)))


def power_iterations(
    similarities, x: np.ndarray, max_iter: int, tol: float = 1e-6,
) -> Tuple[np.ndarray, int]:
    """Generalized power method: x <- polar(S x). Stops when the relative
    gain of the objective in one iteration falls below ``tol``."""
    m, k, p = x.shape
    previous = None
    iteration = 0
    for iteration in range(1, max_iter + 1):
        y = similarities @ x.reshape(m * k, p)
        value = float(np.sum(x.reshape(m * k, p) * y))
        if previous is not None and value - previous < tol * abs(value):
            break
        previous = value
        x = polar(y.reshape(m, k, p))
    return x, iteration


def _match(similarities: scipy.sparse.bsr_array, rng) -> Tuple[np.ndarray, np.ndarray]:
    """Pair every node with its most similar free neighbour. Returns the
    aggregate of each node and, for the second node of a pair, the stored
    block that links it to the first (-1 otherwise)."""
    m = similarities.shape[0] // similarities.blocksize[0]
    indptr, indices = similarities.indptr, similarities.indices
    weights = np.linalg.norm(similarities.data, axis=(1, 2))
    aggregate = np.full(m, -1)
    link = np.full(m, -1)
    count = 0
    for g in rng.permutation(m):
        if aggregate[g] >= 0:
            continue
        aggregate[g] = count
        span = slice(indptr[g], indptr[g + 1])
        free = (aggregate[indices[span]] < 0)
        if free.any():
            best = indptr[g] + np.flatnonzero(free)[np.argmax(weights[span][free])]
            aggregate[indices[best]] = count
            link[indices[best]] = best
        count += 1
    return aggregate, link


def multigrid(
    similarities: scipy.sparse.bsr_array,
    smoothing: int = 16,
    rng: Optional[np.random.Generator] = None,
    tol: float = 1e-6,
) -> np.ndarray:
    """Coarse-to-fine orthogonal synchronization.

    Directions are merged in pairs by their relative alignment, the problem
    among the pairs is solved recursively and its solution is refined with a
    few power iterations. Long-range consistency is therefore settled on
    small problems instead of slowly diffusing across the fine one.
    """
    rng = rng or np.random.default_rng(0)
    k = similarities.blocksize[0]
    m = similarities.shape[0] // k
    identity = np.broadcast_to(np.eye(k), (m, k, k)).copy()
    if m == 1:
        return identity

    aggregate, link = _match(similarities, rng)
    coarse_count = aggregate.max() + 1
    if coarse_count == m:  # Nothing left to merge: disconnected pieces.
        return power_iterations(similarities, identity, smoothing, tol)[0]

    # Bring the second node of each pair to the frame of the first.
    start = identity
    paired = np.flatnonzero(link >= 0)
    start[paired] = polar(similarities.data[link[paired]]).swapaxes(-2, -1)

    prolongation = scipy.sparse.bsr_array(
        (start, aggregate, np.arange(m + 1)), shape=(m * k, coarse_count * k))
    coarse = (prolongation.T @ similarities @ prolongation).tobsr(blocksize=(k, k))
    coarse.sort_indices()
    solution = multigrid(coarse, smoothing, rng, tol)

    x = start @ solution[aggregate]
    return power_iterations(similarities, x, smoothing, tol)[0]
