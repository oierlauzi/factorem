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
def _radial_psd_average(spectra: jax.Array) -> jax.Array:
    box_size = spectra.shape[0]
    n_bins = box_size // 2 + 1

    radius = _radial_index_grid(box_size).reshape(-1)
    multiplicity = _rfft2_multiplicity(box_size).reshape(-1)
    values = spectra.reshape(-1)

    shell_sum = jnp.bincount(radius, weights=multiplicity*values, length=n_bins)
    shell_weight = jnp.bincount(radius, weights=multiplicity, length=n_bins)
    shell_average = shell_sum / shell_weight

    clipped = jnp.clip(radius, 0, n_bins - 1)
    return shell_average[clipped].reshape(spectra.shape)

@jax.jit
def _estimate_fsc_from_halves(
    half1: jax.Array,
    half2: jax.Array
) -> jax.Array:
    """Radially averaged Fourier shell correlation between two half maps.

    The FSC is a *normalised* cross-correlation, so the Wiener shrinkage gain
    that suppresses each half map cancels between numerator and denominator.
    This makes the derived SSNR independent of the regularisation used to build
    the halves, avoiding the feedback collapse that a raw cross-power estimate
    (``Re(half1 * conj(half2))``) suffers from at high frequency.
    """
    cross = _radial_psd_average(jnp.real(half1*jnp.conj(half2)))
    power1 = _radial_psd_average(jnp.square(half1.real) + jnp.square(half1.imag))
    power2 = _radial_psd_average(jnp.square(half2.real) + jnp.square(half2.imag))
    fsc = cross / jnp.sqrt(jnp.maximum(power1*power2, 1e-12))
    return jnp.clip(fsc, 0.0, 0.999)

@jax.jit
def _estimate_signal_spectra(
    half1: jax.Array,
    half2: jax.Array
) -> jax.Array:
    co_spectrum = jnp.real(half1*half2.conj())
    return _radial_psd_average(co_spectrum)

@jax.jit
def _estimate_noise_variance(
    images: jax.Array, 
    ctfs: jax.Array, 
    valid: jax.Array,
    average: jax.Array
) -> jax.Array:
    residuals = images - ctfs*average
    residuals2 = jnp.square(residuals.real) + jnp.square(residuals.imag)
    num = jnp.sum(residuals2, where=valid, axis=0)
    den = jnp.sum(valid, axis=0)
    return _radial_psd_average(num / den)

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

@partial(jax.jit, static_argnames=('max_iter',))
def estimate_map_reconstruction(
    images: jax.Array,
    ctfs: jax.Array,
    valid: jax.Array,
    max_iter: int = 16
) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    even_mask = (jnp.arange(len(valid)) % 2) == 0
    valid = valid[:, None, None]
    even_mask = even_mask[:, None, None]
    valid_even = (valid & even_mask)
    valid_odd = (valid & ~even_mask)

    tau2 = jnp.ones(images.shape[1:], dtype=ctfs.dtype)
    sigma2_even = jnp.zeros(images.shape[1:], dtype=ctfs.dtype)
    sigma2_odd = jnp.zeros(images.shape[1:], dtype=ctfs.dtype)
    for _ in range(max_iter):
        inv_ssnr_even = sigma2_even / jnp.maximum(tau2, 1e-12)
        inv_ssnr_odd = sigma2_odd / jnp.maximum(tau2, 1e-12)
        half_even = _estimate_map_average(images, ctfs, valid_even, inv_ssnr_even)
        half_odd = _estimate_map_average(images, ctfs, valid_odd, inv_ssnr_odd)

        tau2 = _estimate_signal_spectra(half_even, half_odd)
        sigma2_even = _estimate_noise_variance(images, ctfs, valid_even, half_even)
        sigma2_odd = _estimate_noise_variance(images, ctfs, valid_odd, half_odd)

     
    sigma2 = 0.5*(sigma2_even + sigma2_odd)
    inv_ssnr = sigma2 / jnp.maximum(tau2, 1e-12)
    average = _estimate_map_average(images, ctfs, valid, inv_ssnr)
    return average, tau2, sigma2

    