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
    """Iteratively regularised MAP reconstruction (RELION-style, 2D).

    Refines two independent even/odd half reconstructions. The regulariser is
    derived from the Fourier shell correlation (FSC) between the two halves,
    not from their raw cross-power: the FSC is a normalised correlation, so it
    is invariant to the Wiener shrinkage applied to each half and therefore
    does not collapse at high frequency.

    The per-shell FSC is turned into a *map* SSNR (``FSC / (1 - FSC)`` for a
    half, doubled for the full reconstruction) and then converted into the
    per-pixel object inverse-SSNR ``inv_ssnr = sigma^2 / tau^2`` via the CTF
    accumulator weight ``sum(CTF^2) / SSNR``. That single ``inv_ssnr`` is the
    regulariser for the half accumulators, the full accumulator, and any
    downstream per-particle Wiener filter (denominator ``CTF^2 + inv_ssnr``).

    Returns the full reconstruction (``average``), the object inverse-SSNR
    (``inv_ssnr``), the implied object signal power (``tau2``) and the pooled
    noise power (``sigma2``), each a ``(box, box // 2 + 1)`` spectrum.
    """
    eps = 1e-6
    even_mask = (jnp.arange(len(valid)) % 2) == 0
    valid = valid[:, None, None]
    even_mask = even_mask[:, None, None]
    valid_even = (valid & even_mask)
    valid_odd = (valid & ~even_mask)

    # CTF accumulator weights are constant across iterations.
    sum_c2_even = jnp.sum(jnp.square(ctfs), where=valid_even, axis=0)
    sum_c2_odd = jnp.sum(jnp.square(ctfs), where=valid_odd, axis=0)
    sum_c2_all = jnp.sum(jnp.square(ctfs), where=valid, axis=0)

    fsc = jnp.zeros(images.shape[1:])
    inv_ssnr_even = jnp.zeros(images.shape[1:])
    inv_ssnr_odd = jnp.zeros(images.shape[1:])
    for _ in range(max_iter):
        half_even = _estimate_map_average(images, ctfs, valid_even, inv_ssnr_even)
        half_odd = _estimate_map_average(images, ctfs, valid_odd, inv_ssnr_odd)

        fsc = _estimate_fsc_from_halves(half_even, half_odd)
        ssnr_half = fsc / jnp.maximum(1.0 - fsc, eps)
        inv_ssnr_even = sum_c2_even / jnp.maximum(ssnr_half, eps)
        inv_ssnr_odd = sum_c2_odd / jnp.maximum(ssnr_half, eps)

    ssnr_full = 2.0 * fsc / jnp.maximum(1.0 - fsc, eps)
    inv_ssnr = sum_c2_all / jnp.maximum(ssnr_full, eps)
    average = _estimate_map_average(images, ctfs, valid, inv_ssnr)
    return average, inv_ssnr

    