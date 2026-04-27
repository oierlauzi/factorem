from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp

from .. import ctf
from .data_loader import HostBatch

def _pad_0(x: jax.Array, padding: int) -> jax.Array:
    if padding <= 0:
        return x
    
    return jnp.concatenate((
        x,
        jnp.zeros((padding, ) + x.shape[1:], dtype=x.dtype, device=x.device)
    ))

def _pad_images_2d(images: jnp.ndarray, padded_box_size: int):
    batch_shape = images.shape[:-2]
    original_box_size_y, original_box_size_x = images.shape[-2:]
    result_shape = batch_shape + (padded_box_size, padded_box_size)

    result = jnp.zeros(result_shape, dtype=images.dtype)
    return result.at[..., :original_box_size_y, :original_box_size_x].set(images)

def _apply_affine_single(image, matrix_inv):
    H, W = image.shape
    y, x = jnp.mgrid[0:H, 0:W]
    coords = jnp.stack([x.ravel(), y.ravel(), jnp.ones_like(x.ravel())])

    src_coords = matrix_inv @ coords
    src_x = src_coords[0, :] / src_coords[2, :]
    src_y = src_coords[1, :] / src_coords[2, :]

    sample_coords = jnp.stack([src_y.reshape(H, W), src_x.reshape(H, W)])
    transformed = jax.scipy.ndimage.map_coordinates(
        image,
        sample_coords,
        order=1,
        mode='constant',
        cval=0.0
    )

    return transformed

_apply_affine_batch = jax.jit(jax.vmap(_apply_affine_single, in_axes=(0, 0)))


@partial(jax.jit, static_argnames=('padded_box_size',))
def _warp_pad_rfft2(
    images: jax.Array,
    affine_inv: jax.Array,
    norm: jax.Array,
    padded_box_size: int,
) -> jax.Array:
    transformed = _apply_affine_batch(images, affine_inv)
    padded = _pad_images_2d(transformed, padded_box_size)
    return jnp.fft.rfft2(padded) / norm

    
    
@dataclass(frozen=True)
class DeviceBatch:
    """Device-side outputs ready for downstream processors.

    The arrays are deferred (JAX async dispatch); consumers should not
    block until necessary. ``valid_count`` mirrors the host-side count so
    consumers can build a mask for padded rows.
    """
    images_ft: jax.Array
    ctfs: jax.Array
    valid_count: int


class Preprocessor:
    """Device-side stage: H2D copy + warp/pad/rfft + CTF.

    Separated from ``DataLoader`` so host I/O and JAX dispatch live in
    different classes. Must be called on the main thread.
    """

    def __init__(
        self,
        padded_box_size: int,
        pixel_size_a: float,
        voltage_kv: float,
        spherical_aberration_mm: float,
        amplitude_contrast: float,
        grain_size: int
    ):
        self.padded_box_size = padded_box_size
        self.grain_size = grain_size
        self.ctf_context = ctf.CtfContext(
            pixel_size_a=pixel_size_a,
            spherical_aberration_mm=spherical_aberration_mm,
            voltage_kv=voltage_kv,
            q0=amplitude_contrast,
        )

    def process(self, host_batch: HostBatch) -> DeviceBatch:
        """H2D + preprocessing. Returns deferred ``jax.Array``s (non-blocking).
        """
        n, box_size_y, box_size_x = host_batch.images.shape
        padding = (-n) % self.grain_size

        norm = jnp.float32(box_size_x * box_size_y)
        images_ft = _warp_pad_rfft2(
            _pad_0(jax.device_put(host_batch.images), padding),
            _pad_0(jax.device_put(host_batch.affine_inv), padding),
            norm,
            self.padded_box_size,
        )

        ctfs = ctf.compute_ctf_image_2d(
            _pad_0(jax.device_put(host_batch.defocus), padding),
            self.padded_box_size,
            self.ctf_context,
        )

        return DeviceBatch(
            images_ft=images_ft,
            ctfs=ctfs,
            valid_count=n,
        )
