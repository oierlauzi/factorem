import jax
import jax.numpy as jnp

from .processor import Processor

@jax.jit
def _wiener_ctf_correct(
    images_ft: jax.Array,
    ctfs: jax.Array
) -> jax.Array:
    ctfs2 = jnp.square(ctfs)
    wiener_factor = 0.1 * jnp.mean(ctfs2, axis=(-1, -2), keepdims=True)
    wiener_corrected_images_ft = (images_ft * ctfs) / (ctfs2 + wiener_factor)
    return jnp.fft.irfft2(wiener_corrected_images_ft)

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
        wiener_corrected_images = _wiener_ctf_correct(images, ctfs)
        
        valid = jnp.arange(n_padded) < count
        x = wiener_corrected_images.reshape(len(wiener_corrected_images), -1)
        x = _mean_center(x, valid)
        u, s, _ = jnp.linalg.svd(x[:count], full_matrices=False)
        y = u[:,:self.n_components] * s[:self.n_components]
        return y
    