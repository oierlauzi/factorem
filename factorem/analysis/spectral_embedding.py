from functools import partial
from typing import Union, Optional
import jax
import jax.numpy as jnp
import numpy as np
import sklearn.manifold

from .data_loader import DataLoader
from .processor import Processor

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
    L = left_images.reshape(n, -1)   # complex (n, p)
    R = right_images.reshape(m, -1)  # complex (m, p)
    a = left_ctfs.reshape(n, -1)     # real    (n, p)
    b = right_ctfs.reshape(m, -1)    # real    (m, p)

    L_abs2 = jnp.square(L.real) + jnp.square(L.imag)  # (n, p)
    R_abs2 = jnp.square(R.real) + jnp.square(R.imag)  # (m, p)

    term1 = L_abs2 @ (b**2).T
    term2 = (a**2) @ R_abs2.T
    term3 = (a * L.real) @ (b * R.real).T + (a * L.imag) @ (b * R.imag).T

    return term1 + term2 - 2*term3

def _self_pairwise_distance2(
    images: jax.Array,
    ctfs: jax.Array
) -> jax.Array:
    # Expand: 
    # |A_i*c_j - A_j*c_i|^2 = 
    # |A_i|^2*c_j^2 + |A_j|^2*c_i^2 - 2*c_i*c_j*Re(A_i*conj(A_j))
    # term2 == term1.T by symmetry, so only 3 matmuls needed instead of 4.
    n = images.shape[0]
    A = images.reshape(n, -1)   # complex (n, p)
    c = ctfs.reshape(n, -1)     # real    (n, p)

    A_abs2 = jnp.square(A.real) + jnp.square(A.imag) # (n, p)

    term1 = A_abs2 @ (c**2).T # (n, n)
    term2 = term1.T
    term3 = (c * A.real) @ (c * A.real).T + (c * A.imag) @ (c * A.imag).T # (n, n)

    return term1 + term2 - 2*term3

def _compute_pairwise_distance2_matrix(
    loader: DataLoader, 
    indices: np.ndarray, 
    direction_matrix: np.ndarray,
    batch_size: int
):
    n = len(indices)
    distances2 = jnp.empty((n, n))

    start0 = 0
    while start0 < n:
        end0 = min(start0 + batch_size, n)
        batch0_indices = indices[start0:end0]
        batch0_images, batch0_ctfs = loader.load(batch0_indices, direction_matrix)
    
        start1 = 0
        while start1 < start0:
            end1 = min(start1 + batch_size, n)
            batch1_indices = indices[start1:end1]
            batch1_images, batch1_ctfs = loader.load(batch1_indices, direction_matrix)
            
            tile_distances2 = _crossed_pairwise_distance2(
                batch0_images,
                batch0_ctfs,
                batch1_images,
                batch1_ctfs
            )
            distances2 = distances2.at[start0:end0,start1:end1].set(tile_distances2)
            distances2 = distances2.at[start1:end1,start0:end0].set(tile_distances2.T)
            
            start1 = end1
    
        tile_distances2 = _self_pairwise_distance2(
            batch0_images,
            batch0_ctfs
        )
        distances2 = distances2.at[start0:end0,start0:end0].set(tile_distances2)

        start0 = end0
    
    return distances2
    
def _radial_basis_function(
    distance2: jax.Array, 
    sigma2: Union[float, jax.Array]
) -> jax.Array:
    return jnp.exp(-distance2 / sigma2)

def _local_scaling_kernel(distance2: jax.Array, k: int = 7) -> jax.Array:
    sigma2 = jnp.partition(distance2, k, axis=1)[:, k]
    sigma = jnp.sqrt(sigma2)
    scale = jnp.outer(sigma, sigma)
    return _radial_basis_function(distance2=distance2, sigma2=scale)

def _median_scaling_kernel(distance2: jax.Array) -> jax.Array:
    n = distance2.shape[0]
    indices = jnp.triu_indices(n, k=1)
    flat_distances2 = distance2[indices]
    sigma2 = jnp.median(flat_distances2)
    return _radial_basis_function(distance2=distance2, sigma2=sigma2)

def _compute_laplacian(affinity: jnp.ndarray) -> jnp.ndarray:
    degree = affinity.sum(axis=-1)
    indices = jnp.arange(affinity.shape[0])
    return (-affinity).at[indices, indices].add(degree)

class SpectralEmbedding(Processor):
    def __init__(
        self, 
        n_components: int, 
        batch_size: int,
        kernel: str = 'median', 
        k: Optional[int] = None,
        sigma2: Optional[float] = None
    ):
        self.se = sklearn.manifold.SpectralEmbedding(
            n_components=n_components, 
            affinity='precomputed'
        )
        self.batch_size = batch_size
        
        if kernel == 'median':
            self.kernel = _median_scaling_kernel
        elif kernel == 'local':
            self.kernel = partial(_local_scaling_kernel, k=k)
        elif kernel == 'rbf':
            self.kernel = partial(_radial_basis_function, sigma2=sigma2)
        else:
            raise ValueError('Invalid kernel')
        
    def embed(
        self, 
        loader: DataLoader, 
        indices: np.ndarray, 
        direction_matrix: np.ndarray
    ) -> jax.Array:
        distances2 = _compute_pairwise_distance2_matrix(
            loader=loader, 
            indices=indices, 
            direction_matrix=direction_matrix,
            batch_size=self.batch_size
        )
        
        affinity = _median_scaling_kernel(distances2)
        return self.se.fit_transform(jax.device_get(affinity))
