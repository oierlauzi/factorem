from functools import partial

import numpy as np
import jax
import jax.numpy as jnp

from .. import image
from .. import geometry
from .. import ctf

def _pad_images_2d(images: jnp.ndarray, padded_box_size: int):
    batch_shape = images.shape[:-2]
    original_box_size_y, original_box_size_x = images.shape[-2:]
    result_shape = batch_shape + (padded_box_size, padded_box_size)

    result = jnp.zeros(result_shape, dtype=images.dtype)
    return result.at[...,:original_box_size_y,:original_box_size_x].set(images)

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

_apply_affine_batch = jax.vmap(_apply_affine_single, in_axes=(0, 0))

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

class DataLoader:
    def __init__(
        self,
        image_locations,
        image_prefix: str,
        rotations: np.ndarray,
        shifts: np.ndarray,
        defocus: np.ndarray,
        padded_box_size: int,
        pixel_size_a: float,
        voltage_kv: float,
        spherical_aberration_mm: float,
        amplitude_contrast: float
    ):
        self.image_locations = image_locations
        self.reader = image.BatchReader(prefix=image_prefix)
        self.rotations = rotations
        self.shifts = shifts
        self.defocus = defocus
        self.padded_box_size = padded_box_size
        self.ctf_context = ctf.CtfContext(
            pixel_size_a=pixel_size_a,
            spherical_aberration_mm=spherical_aberration_mm,
            voltage_kv=voltage_kv,
            q0=amplitude_contrast
        )
    
    def load(self, indices: np.ndarray, reference_transform: np.ndarray):
        batch_images = jax.device_put(
            self.reader.read_batch(self.image_locations[indices])
        )
        batch_rotations = self.rotations[indices]
        batch_shifts = self.shifts[indices]
        batch_defocus = self.defocus[indices]
        
        _, box_size_y, box_size_x = batch_images.shape
        centre = np.array((box_size_y/2, box_size_x/2))
        matrix_2d = geometry.compute_in_plane_alignment(
            reference_transform,
            batch_rotations
        )
        affine = geometry.make_affine(matrix_2d, batch_shifts, centre)
        affine_inv = np.linalg.inv(affine).astype(np.float32)

        norm = jnp.float32(box_size_x * box_size_y)
        transformed_images_ft = _warp_pad_rfft2(
            batch_images,
            jax.device_put(affine_inv),
            norm,
            self.padded_box_size,
        )

        ctf_images = ctf.compute_ctf_image_2d(
            jax.device_put(batch_defocus),
            self.padded_box_size,
            self.ctf_context
        )

        return (transformed_images_ft, ctf_images)
    