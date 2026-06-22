from functools import partial
from typing import Callable, Optional, Union

import jax
import jax.numpy as jnp

from .processor import Processor
from .map_average import estimate_map_reconstruction
from ..ctf import wiener_ctf_correct_2d

@partial(jax.jit, static_argnames=('box_size',))
def _rfft2_multiplicity(box_size: int) -> jax.Array:
    half = box_size // 2 + 1
    cols = jnp.full((half,), 2.0)
    cols = cols.at[0].set(1.0)
    if box_size % 2 == 0:
        cols = cols.at[-1].set(1.0)
    return jnp.broadcast_to(cols, (box_size, half))

@jax.jit
def _self_pairwise_distance2(
    images: jax.Array,
    ctfs: jax.Array,
    inv_ssnr: jax.Array,
    multiplicity: jax.Array
) -> jax.Array:
    # Expand: 
    # |A_i*c_j - A_j*c_i|^2
    # |A_i|^2*c_j^2 + |A_j|^2*c_i^2 - 2*c_i*c_j*Re(A_i*conj(A_j))
    # term2 == term1.T by symmetry, so only 3 matmuls needed instead of 4.
    n = images.shape[0]
    A = images.reshape(n, -1)
    c = ctfs.reshape(n, -1)
    w = multiplicity.reshape(-1)

    A_abs2 = jnp.square(A.real) + jnp.square(A.imag)
    cr = c * A.real
    ci = c * A.imag

    term1 = (A_abs2 * w) @ (c**2).T
    term2 = term1.T
    term3 = (cr * w) @ cr.T + (ci * w) @ ci.T

    return jnp.maximum(term1 + term2 - 2*term3, 0.0)

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

@partial(jax.jit, static_argnames=('n_components', 'alpha', 'diffusion_time'))
def _spectral_embedding(
    affinity: jax.Array,
    valid: jax.Array,
    n_components: int,
    alpha: float = 1.0,
    diffusion_time: float = 1.0,
) -> jax.Array:
    pair_valid = valid[:, None] & valid[None, :]
    affinity = jnp.where(pair_valid, affinity, 0.0)

    # Density normalization K_alpha = D^-alpha W D^-alpha decouples the manifold
    # geometry from the sampling density (alpha=1 -> Laplace-Beltrami operator).
    density = affinity.sum(axis=-1)
    safe_density = jnp.where(valid, density, 1.0)
    d_alpha = jnp.where(valid, safe_density ** (-alpha), 0.0)
    affinity = affinity * d_alpha[:, None] * d_alpha[None, :]

    # Symmetric conjugate of the row-stochastic diffusion operator P = D^-1 K_alpha.
    # M_sym = D^-1/2 K_alpha D^-1/2 shares P's spectrum but is symmetric.
    degree = affinity.sum(axis=-1)
    safe_degree = jnp.where(valid, degree, 1.0)
    d_inv_sqrt = jnp.where(valid, jax.lax.rsqrt(safe_degree), 0.0)

    normalized = affinity * d_inv_sqrt[:, None] * d_inv_sqrt[None, :]
    # Push padded nodes to a large negative eigenvalue so they fall outside the
    # leading diffusion modes (which are the largest eigenvalues here).
    huge_eigval = -2.0 * len(valid)
    diag = jnp.where(valid, 0.0, huge_eigval)
    indices = jnp.arange(affinity.shape[0])
    normalized = normalized.at[indices, indices].add(diag)
    normalized = 0.5 * (normalized + normalized.T)

    eigvals, eigvecs = jnp.linalg.eigh(normalized)
    # eigh returns ascending eigenvalues; the diffusion modes are the largest,
    # so reverse and drop the trivial stationary component (lambda = 1).
    eigvals = eigvals[::-1][1:n_components + 1]
    eigvecs = eigvecs[:, ::-1][:, 1:n_components + 1]

    # Right eigenvectors of P recovered as psi = D^-1/2 v, scaled by the
    # diffusion-time weighted eigenvalues lambda^t.
    psi = d_inv_sqrt[:, None] * eigvecs
    embedding = psi * (eigvals ** diffusion_time)[None, :]

    return embedding

@partial(jax.jit, static_argnames=('kernel', 'n_components'))
def _fit_transform(
    images: jax.Array,
    ctfs: jax.Array,
    valid: jax.Array,
    kernel: Callable[[jax.Array, jax.Array], jax.Array],
    n_components: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    average, inv_ssnr = estimate_map_reconstruction(
        images,
        ctfs,
        valid
    )

    multiplicity = _rfft2_multiplicity(images.shape[1])
    distances2 = _self_pairwise_distance2(images, ctfs, inv_ssnr, multiplicity)
    affinity = kernel(distances2, valid)
    embedding = _spectral_embedding(affinity, valid, n_components)
    return embedding

class SpectralEmbedding(Processor):
    def __init__(
        self,
        n_components: int,
        kernel: str = 'median',
        k: Optional[int] = None,
        sigma2: Optional[float] = None
    ):
        self.n_components = n_components

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
        embedding = _fit_transform(
            images=images,
            ctfs=ctfs,
            valid=valid,
            kernel=self.kernel,
            n_components=self.n_components,
        )
        return embedding[:count]
