from functools import partial
from typing import Tuple
import jax
import jax.numpy as jnp

@partial(jax.jit, static_argnames=('box_size',))
def _radial_index_grid(box_size: int) -> jax.Array:
    d = 1.0 / box_size
    ky = jnp.fft.fftfreq(box_size, d=d)
    kx = jnp.fft.rfftfreq(box_size, d=d)
    radius2 = jnp.square(ky[:, None]) + jnp.square(kx[None, :])
    return jnp.round(jnp.sqrt(radius2)).astype(jnp.int32)

@partial(jax.jit, static_argnames=('box_size',))
def _rfft2_multiplicity(box_size: int) -> jax.Array:
    half = box_size // 2 + 1
    cols = jnp.full((half,), 2.0)
    cols = cols.at[0].set(1.0)
    if box_size % 2 == 0:
        cols = cols.at[-1].set(1.0)
    return jnp.broadcast_to(cols, (box_size, half))

@jax.jit
def _estimate_map_average(
    images: jax.Array, 
    ctfs: jax.Array, 
    valid: jax.Array,
    inv_ssnr: jax.Array
) -> jax.Array:
    num = jnp.sum(images*ctfs, where=valid, axis=0)
    den = jnp.sum(jnp.square(ctfs), where=valid, axis=0) + inv_ssnr
    return num / den

@jax.jit
def _compute_inv_ssnr(
    half1: jax.Array,
    half2: jax.Array
):
    EPS = 1e-12
    box_size = half1.shape[0]
    n_bins = box_size // 2 + 1

    radius = _radial_index_grid(box_size).reshape(-1)
    multiplicity = _rfft2_multiplicity(box_size).reshape(-1)

    cross = jnp.real(half1*jnp.conj(half2)).reshape(-1)
    power1 = (jnp.square(half1.real) + jnp.square(half1.imag)).reshape(-1)
    power2 = (jnp.square(half2.real) + jnp.square(half2.imag)).reshape(-1)

    shell_cross = jnp.bincount(radius, weights=multiplicity*cross, length=n_bins)
    shell_power1 = jnp.bincount(radius, weights=multiplicity*power1, length=n_bins)
    shell_power2 = jnp.bincount(radius, weights=multiplicity*power2, length=n_bins)
    
    fsc = shell_cross / jnp.sqrt(jnp.maximum(shell_power1*shell_power2, EPS))
    inv_ssnr = (1 - fsc) / jnp.maximum(fsc, EPS)
    
    clipped = jnp.clip(radius, 0, n_bins - 1)
    return inv_ssnr[clipped].reshape(half1.shape)

@partial(jax.jit, static_argnames=('max_iter',))
def estimate_map_reconstruction(
    images: jax.Array,
    ctfs: jax.Array,
    valid: jax.Array,
    max_iter: int = 16
) -> Tuple[jax.Array, jax.Array]:
    even_mask = (jnp.arange(len(valid)) % 2) == 0
    valid = valid[:, None, None]
    even_mask = even_mask[:, None, None]
    valid_even = (valid & even_mask)
    valid_odd = (valid & ~even_mask)

    ctf2 = jnp.square(ctfs)
    ctf2_sum = jnp.sum(ctf2, where=valid, axis=0)
    ctf2_even_sum = jnp.sum(ctf2, where=valid_even, axis=0)
    ctf2_odd_sum = jnp.sum(ctf2, where=valid_odd, axis=0)
    
    inv_ssnr_half = jnp.zeros(images.shape[1:], dtype=ctfs.dtype)
    for _ in range(max_iter):
        inv_ssnr_even = inv_ssnr_half * ctf2_even_sum
        inv_ssnr_odd = inv_ssnr_half * ctf2_odd_sum
        half_even = _estimate_map_average(images, ctfs, valid_even, inv_ssnr_even)
        half_odd = _estimate_map_average(images, ctfs, valid_odd, inv_ssnr_odd)

        inv_ssnr_half = _compute_inv_ssnr(half_even, half_odd)

    inv_ssnr_full = 0.5*inv_ssnr_half
    average = _estimate_map_average(images, ctfs, valid, inv_ssnr_full*ctf2_sum)
    
    return average, inv_ssnr_full

    