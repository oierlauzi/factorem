import jax
import jax.numpy as jnp

@jax.jit
def wiener_ctf_correct_2d(
    images_ft: jax.Array,
    ctfs: jax.Array
) -> jax.Array:
    ctfs2 = jnp.square(ctfs)
    wiener_factor = 0.1 * jnp.mean(ctfs2, axis=(-1, -2), keepdims=True)
    wiener_corrected_images_ft = (images_ft * ctfs) / (ctfs2 + wiener_factor)
    return jnp.fft.irfft2(wiener_corrected_images_ft)
