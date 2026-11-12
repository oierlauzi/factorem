from dataclasses import dataclass
from functools import partial
from typing import Mapping, Optional

import jax
import jax.numpy as jnp

from .. import ctf
from .low_pass_filter import butterworth_2d


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class DeviceBatch:
    """Particles of one direction group, aligned and in Fourier space."""
    images_ft: jax.Array
    ctfs: jax.Array


# Geometry is cheap, but GPUs default to reduced-precision float32 products,
# which would misplace pixels by a noticeable fraction.
_matmul = partial(jnp.matmul, precision=jax.lax.Precision.HIGHEST)


def _in_plane_rotation(
    direction_matrix: jax.Array,
    rotations: jax.Array
) -> jax.Array:
    """Closest 2D rotation taking each particle to the reference direction."""
    delta = _matmul(direction_matrix, rotations.swapaxes(-2, -1))
    u, _, vh = jnp.linalg.svd(delta[..., :2, :2])
    return _matmul(u, vh)


def _align_single(
    image: jax.Array,
    rotation: jax.Array,
    shift: jax.Array
) -> jax.Array:
    """Rotate an image about its centre and undo its (x, y) shift."""
    h, w = image.shape
    centre = jnp.array((w / 2, h / 2), dtype=image.dtype)
    y, x = jnp.mgrid[0:h, 0:w]
    destination = jnp.stack((x.ravel(), y.ravel())).astype(image.dtype)

    # Inverse of p -> R (p + shift - centre) + centre
    source = _matmul(rotation.T, destination - centre[:, None])
    source += (centre - shift)[:, None]
    coordinates = jnp.stack((source[1].reshape(h, w), source[0].reshape(h, w)))

    return jax.scipy.ndimage.map_coordinates(
        image, coordinates, order=1, mode='constant', cval=0.0
    )


def _zero_pad(images: jax.Array, box_size: int) -> jax.Array:
    h, w = images.shape[-2:]
    return jnp.pad(images, ((0, 0), (0, box_size - h), (0, box_size - w)))


@partial(jax.jit, static_argnames=('padded_box_size', 'ctf_context'))
def _preprocess(
    images: jax.Array,
    rotations: jax.Array,
    shifts: jax.Array,
    defocus: jax.Array,
    direction_matrix: jax.Array,
    frequency_mask: Optional[jax.Array],
    padded_box_size: int,
    ctf_context: ctf.CtfContext,
) -> DeviceBatch:
    in_plane = _in_plane_rotation(direction_matrix, rotations)
    aligned = jax.vmap(_align_single)(images, in_plane, shifts)

    h, w = images.shape[-2:]
    images_ft = jnp.fft.rfft2(_zero_pad(aligned, padded_box_size)) / (h * w)
    if frequency_mask is not None:
        images_ft = frequency_mask * images_ft

    ctfs = ctf.compute_ctf_image_2d(defocus, padded_box_size, ctf_context)
    return DeviceBatch(images_ft=images_ft, ctfs=ctfs)


class Preprocessor:
    """Aligns particles to a projection direction and moves them to Fourier
    space, padded, low-pass filtered and with their CTFs."""

    def __init__(
        self,
        padded_box_size: int,
        pixel_size_a: float,
        voltage_kv: float,
        spherical_aberration_mm: float,
        amplitude_contrast: float,
        max_freq: Optional[float] = None
    ):
        self.padded_box_size = padded_box_size
        self.ctf_context = ctf.CtfContext(
            pixel_size_a=pixel_size_a,
            spherical_aberration_mm=spherical_aberration_mm,
            voltage_kv=voltage_kv,
            q0=amplitude_contrast,
        )
        self.frequency_mask = None
        if max_freq is not None:
            self.frequency_mask = butterworth_2d(
                box_size=padded_box_size,
                cutoff=max_freq,
                order=2
            )

    def __call__(
        self,
        batch: Mapping[str, jax.Array],
        direction_matrix: jax.Array
    ) -> DeviceBatch:
        return _preprocess(
            batch['image'],
            batch['rotation'],
            batch['shift'],
            batch['defocus'],
            jnp.asarray(direction_matrix, dtype=jnp.float32),
            self.frequency_mask,
            padded_box_size=self.padded_box_size,
            ctf_context=self.ctf_context,
        )
