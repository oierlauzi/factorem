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

def _remove_padding(
    images: jax.Array,
    particle_size: int
) -> jax.Array:
    return images[:,:particle_size,:particle_size]
    
class PCA(Processor):
    def __init__(
        self,
        n_components: int,
        particle_size: int
    ):
        self.n_components = n_components
        self.particle_size = particle_size

    def fit_transform(
        self,
        images: jax.Array,
        ctfs: jax.Array,
        count: int
    ) -> jax.Array:
        n_padded = images.shape[0]
        wiener_corrected_images = jnp.fft.irfft2(
            wiener_ctf_correct_2d(images, ctfs)
        )
        wiener_corrected_images = _remove_padding(
            wiener_corrected_images, 
            self.particle_size
        )
        
        valid = jnp.arange(n_padded) < count
        x = wiener_corrected_images.reshape(len(wiener_corrected_images), -1)
        x = _mean_center(x, valid)
        u, s, _ = jnp.linalg.svd(x[:count], full_matrices=False)
        y = u[:,:self.n_components] * s[:self.n_components]
        return y
    