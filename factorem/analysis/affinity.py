from typing import Union
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

def radial_basis_function(
    distance2: jax.Array, 
    sigma2: Union[float, jax.Array]
) -> jax.Array:
    return jnp.exp(-distance2 / sigma2)

def local_scaling_kernel(distance2: jax.Array, k: int = 7) -> jax.Array:
    sigma2 = jnp.partition(distance2, k, axis=1)[:, k]
    sigma = jnp.sqrt(sigma2)
    scale = jnp.outer(sigma, sigma)
    return radial_basis_function(distance2=distance2, sigma2=scale)

def median_scaling_kernel(distance2: jax.Array) -> jax.Array:
    n = distance2.shape[0]
    indices = jnp.triu_indices(n, k=1)
    flat_distances2 = distance2[indices]
    sigma2 = jnp.median(flat_distances2)
    return radial_basis_function(distance2=distance2, sigma2=sigma2)
