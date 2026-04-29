import jax
import jax.numpy as jnp

from .processor import Processor
from ..ctf import wiener_ctf_correct_2d

@jax.jit
def _mean_center(
    x: jax.Array,
    valid: jax.Array
) -> jax.Array:
    valid = valid[:,None]
    mean = x.mean(axis=0, where=valid)
    return x - mean

class PCA(Processor):
    def __init__(
        self,
        n_components: int
    ):
        self.n_components = n_components

    def fit_transform(
        self,
        images: jax.Array,
        ctfs: jax.Array,
        count: int
    ) -> jax.Array:
        n_padded = images.shape[0]
        wiener_corrected_images = wiener_ctf_correct_2d(images, ctfs)
        
        valid = jnp.arange(n_padded) < count
        x = wiener_corrected_images.reshape(len(wiener_corrected_images), -1)
        x = _mean_center(x, valid)
        u, s, _ = jnp.linalg.svd(x[:count], full_matrices=False)
        y = u[:,:self.n_components] * s[:self.n_components]
        return y
    