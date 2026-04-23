import jax.numpy as jnp

def compute_laplacian(affinity: jnp.ndarray) -> jnp.ndarray:
    degree = affinity.sum(axis=-1)
    indices = jnp.arange(affinity.shape[0])
    return (-affinity).at[indices, indices].add(degree)