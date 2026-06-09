from functools import partial
from typing import Callable, Optional, Union

import jax
import jax.numpy as jnp

from .processor import Processor

@jax.jit
def _crossed_pairwise_distance2(
    left_images: jax.Array,
    left_ctfs: jax.Array,
    right_images: jax.Array,
    right_ctfs: jax.Array,
) -> jax.Array:
    # Expand:
    # |L_i*b_j - R_j*a_i|^2 =
    # |L_i|^2*b_j^2 + |R_j|^2*a_i^2 - 2*a_i*b_j*Re(L_i*conj(R_j))
    n = left_images.shape[0]
    m = right_images.shape[0]
    L = left_images.reshape(n, -1)
    R = right_images.reshape(m, -1)
    a = left_ctfs.reshape(n, -1)
    b = right_ctfs.reshape(m, -1)

    L_abs2 = jnp.square(L.real) + jnp.square(L.imag)
    R_abs2 = jnp.square(R.real) + jnp.square(R.imag)

    term1 = L_abs2 @ (b**2).T
    term2 = (a**2) @ R_abs2.T
    term3 = (a * L.real) @ (b * R.real).T + (a * L.imag) @ (b * R.imag).T

    return term1 + term2 - 2*term3

@jax.jit
def _self_pairwise_distance2(
    images: jax.Array,
    ctfs: jax.Array
) -> jax.Array:
    # Expand: 
    # |A_i*c_j - A_j*c_i|^2
    # |A_i|^2*c_j^2 + |A_j|^2*c_i^2 - 2*c_i*c_j*Re(A_i*conj(A_j))
    # term2 == term1.T by symmetry, so only 3 matmuls needed instead of 4.
    n = images.shape[0]
    A = images.reshape(n, -1)
    c = ctfs.reshape(n, -1)

    A_abs2 = jnp.square(A.real) + jnp.square(A.imag)

    term1 = A_abs2 @ (c**2).T
    term2 = term1.T
    term3 = (c * A.real) @ (c * A.real).T + (c * A.imag) @ (c * A.imag).T

    return term1 + term2 - 2*term3

@jax.jit
def _radial_basis_function(
    distance2: jax.Array,
    sigma2: Union[float, jax.Array]
) -> jax.Array:
    return jnp.exp(-distance2 / sigma2)

@partial(jax.jit, static_argnames=('k',))
def _local_scaling_kernel(
    distance2: jax.Array,
    valid: jax.Array,
    k: int = 7,
) -> jax.Array:
    # Push padded columns to +inf so the k-th nearest is picked among valid neighbours.
    masked = jnp.where(valid[None, :], distance2, jnp.inf)
    sigma2 = jnp.partition(masked, k, axis=1)[:, k]
    sigma = jnp.sqrt(sigma2)
    scale = jnp.outer(sigma, sigma)
    return _radial_basis_function(distance2=distance2, sigma2=scale)

@jax.jit
def _median_scaling_kernel(
    distance2: jax.Array,
    valid: jax.Array,
) -> jax.Array:
    n = distance2.shape[0]
    triu = jnp.triu(jnp.ones((n, n), dtype=bool), k=1)
    pair_valid = valid[:, None] & valid[None, :]
    consider = triu & pair_valid
    sigma2 = jnp.nanmedian(jnp.where(consider, distance2, jnp.nan))
    return _radial_basis_function(distance2=distance2, sigma2=sigma2)

@jax.jit
def _fixed_sigma_kernel(
    distance2: jax.Array,
    valid: jax.Array,
    sigma2: Union[float, jax.Array],
) -> jax.Array:
    del valid
    return _radial_basis_function(distance2=distance2, sigma2=sigma2)

@partial(jax.jit, static_argnames=('n_components',))
def _spectral_embedding(
    affinity: jax.Array,
    valid: jax.Array,
    n_components: int
) -> jax.Array:
    pair_valid = valid[:, None] & valid[None, :]
    affinity = jnp.where(pair_valid, affinity, 0.0)

    degree = affinity.sum(axis=-1)
    safe_degree = jnp.where(valid, degree, 1.0)
    d_inv_sqrt = jnp.where(valid, jax.lax.rsqrt(safe_degree), 0.0)

    normalized = affinity * d_inv_sqrt[:, None] * d_inv_sqrt[None, :]
    huge_eigval = 2*len(valid)
    diag = jnp.where(valid, 1.0, huge_eigval)
    indices = jnp.arange(affinity.shape[0])
    laplacian = (-normalized).at[indices, indices].add(diag)
    laplacian = 0.5 * (laplacian + laplacian.T)
    
    _, eigvecs = jnp.linalg.eigh(laplacian)
    embedding =  d_inv_sqrt[:,None] * eigvecs[:, 1:n_components + 1]
    
    return embedding

@partial(
    jax.jit, 
    static_argnames=(
        'kernel', 
        'n_components', 
        'trim_iterations', 
        'outlier_threshold'
    )
)
def _fit_transform(
    images: jax.Array,
    ctfs: jax.Array,
    valid: jax.Array,
    kernel: Callable[[jax.Array, jax.Array], jax.Array],
    n_components: int,
    trim_iterations: int,
    outlier_threshold: float,
) -> tuple[jax.Array, jax.Array]:
    distances2 = _self_pairwise_distance2(images, ctfs)
    affinity = kernel(distances2, valid)
    embedding = _spectral_embedding(affinity, valid, n_components)
    for _ in range(trim_iterations):
        inliers = jnp.linalg.norm(embedding, axis=1) <= outlier_threshold
        valid = jnp.logical_and(valid, inliers)
        affinity = kernel(distances2, valid)
        embedding = _spectral_embedding(affinity, valid, n_components)
    return embedding, valid


class SpectralEmbedding(Processor):
    def __init__(
        self,
        n_components: int,
        kernel: str = 'median',
        k: Optional[int] = None,
        sigma2: Optional[float] = None,
        trim_iterations: int = 0,
        outlier_threshold: float = 5.0
    ):
        self.n_components = n_components
        self.trim_iterations = trim_iterations
        self.outlier_threshold = outlier_threshold

        if kernel == 'median':
            self.kernel = _median_scaling_kernel
        elif kernel == 'local':
            self.kernel = partial(_local_scaling_kernel, k=(k or 7))
        elif kernel == 'rbf':
            self.kernel = partial(_fixed_sigma_kernel, sigma2=sigma2)
        else:
            raise ValueError('Invalid kernel')

    def fit_transform(
        self,
        images: jax.Array,
        ctfs: jax.Array,
        count: int
    ) -> jax.Array:
        n_padded = images.shape[0]
        valid = jnp.arange(n_padded) < count
        embedding, valid = _fit_transform(
            images, ctfs, valid,
            kernel=self.kernel,
            n_components=self.n_components,
            trim_iterations=self.trim_iterations,
            outlier_threshold=self.outlier_threshold,
        )
        
        return embedding[:count]#, valid[:count]
