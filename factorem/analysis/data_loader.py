from dataclasses import dataclass

import numpy as np

from .. import image
from .. import geometry


@dataclass(frozen=True)
class HostBatch:
    """Host-side, fully prepared inputs for a single batch.
    """
    images: np.ndarray
    affine_inv: np.ndarray
    defocus: np.ndarray



class DataLoader:
    """Host-side loader: reads images from disk and assembles affine matrices.

    Performs no JAX dispatch; safe to call from a worker thread.
    """

    def __init__(
        self,
        image_locations,
        image_prefix: str,
        rotations: np.ndarray,
        shifts: np.ndarray,
        defocus: np.ndarray,
    ):
        self.image_locations = image_locations
        self.reader = image.BatchReader(prefix=image_prefix)
        self.rotations = rotations
        self.shifts = shifts
        self.defocus = defocus

    def load(
        self,
        indices: np.ndarray,
        reference_transform: np.ndarray
    ) -> HostBatch:
        """CPU-only stage: read images from disk and compute affine matrices.
        """
        reference_transform = reference_transform.astype(np.float32)
        batch_images = self.reader.read_batch(self.image_locations[indices])
        batch_rotations = self.rotations[indices].astype(np.float32)
        batch_shifts = self.shifts[indices].astype(np.float32)
        batch_defocus = self.defocus[indices].astype(np.float32)

        _, box_size_y, box_size_x = batch_images.shape
        centre = np.array((box_size_y / 2, box_size_x / 2))
        matrix_2d = geometry.compute_in_plane_alignment(
            reference_transform,
            batch_rotations
        )
        affine = geometry.make_affine(matrix_2d, batch_shifts, centre)
        affine_inv = np.linalg.inv(affine)
        defocus = np.asarray(batch_defocus)

        return HostBatch(
            images=batch_images,
            affine_inv=affine_inv,
            defocus=defocus
        )
