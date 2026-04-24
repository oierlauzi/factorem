from typing import Optional, Tuple
import numpy as np
import scipy.sparse

def _orthogonalize_matrices(matrices: np.ndarray, special: bool) -> np.ndarray:
    u, _, vh = np.linalg.svd(matrices, full_matrices=False)
    result = u @ vh
    return result

def _optimization_step(
    samples: scipy.sparse.csr_matrix,
    transforms: np.ndarray,
    special: bool
) -> np.ndarray:
    result = samples @ transforms.reshape(-1, transforms.shape[-1])
    result = result.reshape(transforms.shape)
    result = _orthogonalize_matrices(result, special=special)
    return result

def _decompose_bases(bases: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n, k, p = bases.shape
    u, s, _ = np.linalg.svd(bases.reshape(n*k, p), full_matrices=False)

    u *= s
    transforms = np.reshape(u, (n, k, p))
    eigenvalues = np.square(s)

    return transforms, eigenvalues

def burer_monteiro_random_start(
    n: int, 
    k: int, 
    rng: np.random.Generator = np.random.default_rng()
):
    p = 2*k+1
    start = rng.standard_normal((n*k, p))
    start, _ = np.linalg.qr(start, mode='reduced')
    start = start.reshape(n, k, p)
    return start

def burer_monteiro_ortho_group_synchronization(
    samples: scipy.sparse.csr_matrix,
    start: np.ndarray,
    special: bool = False,
    tol: float = 1e-8,
    max_iter: int = 512,
) -> np.ndarray:
    x = start
    for _ in range(max_iter):
        prev_x = x
        x = _optimization_step(
            samples=samples,
            transforms=x,
            special=special
        )
        
        delta = np.linalg.norm(x-prev_x)
        if delta < tol:
            break

    return _decompose_bases(x)