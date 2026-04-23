import numpy as np
import jax
import jax.numpy as jnp

from . import image
from . import geometry
from . import ctf
from . import analysis

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
        
        ctf_images = ctf.compute_ctf_image_2d(
            jnp.asarray(batch_defocus), 
            self.padded_box_size, 
            self.ctf_context
        )
        
        matrix_2d = geometry.compute_in_plane_alignment(
            reference_transform, 
            batch_rotations
        )
        affine = geometry.make_affine(matrix_2d, batch_shifts, centre)
        affine = np.linalg.inv(affine)

        transformed_images = analysis.apply_affine_batch(
            batch_images, 
            jax.device_put(affine)
        )
        transformed_images = analysis.pad_images_2d(
            transformed_images, 
            self.padded_box_size
        )
        
        transformed_images_ft = jnp.fft.rfft2(transformed_images)
        transformed_images_ft /= box_size_x*box_size_y
        
        #wiener_corrected_images_ft = (transformed_images_ft*ctf_images) / (np.square(ctf_images) + 0.1*np.mean(np.square(ctf_images), axis=(-1, -2), keepdims=True))
        #wiener_corrected_images = jnp.fft.irfft2(wiener_corrected_images_ft)

        return (transformed_images_ft, ctf_images)
    