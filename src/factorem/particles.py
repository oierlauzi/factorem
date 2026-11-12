"""What is known about a set of particles before analysing it."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, fields
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CtfParameters:
    """Contrast transfer function of each particle.

    Every field holds a value for each particle, so that particles of
    different micrographs, or of different microscopes, go together.
    """

    defocus_u: np.ndarray
    """Defocus along the major axis [A], positive for underfocus."""
    defocus_v: np.ndarray
    """Defocus along the minor axis [A]."""
    defocus_angle: np.ndarray
    """Angle of the major axis from the x axis of the image [rad]."""
    phase_shift: np.ndarray
    """Phase shift of a phase plate [rad]."""
    voltage: np.ndarray
    """Acceleration voltage [kV]."""
    spherical_aberration: np.ndarray
    """Spherical aberration [mm]."""
    amplitude_contrast: np.ndarray
    """Fraction of amplitude contrast."""

    def __post_init__(self) -> None:
        count = len(self.defocus_u)
        for field in fields(self):
            value = np.asarray(getattr(self, field.name), dtype=np.float64)
            if value.shape != (count,):
                raise ValueError(
                    f'{field.name} has shape {value.shape}, and not a '
                    f'value for each of the {count} particles.'
                )
            object.__setattr__(self, field.name, value)

    def __len__(self) -> int:
        return len(self.defocus_u)

    def as_array(self) -> np.ndarray:
        """All the parameters, ``(particles, 7)``, in the order of the
        fields."""
        return np.stack(
            [getattr(self, field.name) for field in fields(self)], axis=1
        )


@dataclass(frozen=True)
class Particles:
    """Particles with the alignment and the CTF of a consensus refinement."""

    images: Sequence[Any]
    """Where the image of each particle is: ``vitrio.ImageLocation``
    objects, as ``vitrio.data.parse_locations`` makes them."""
    rotations: np.ndarray
    """Orientation of each particle, ``(particles, 3, 3)``. See
    `factorem.geometry`."""
    shifts: np.ndarray
    """Translation (x, y) that centres each image [px]."""
    ctf: CtfParameters
    pixel_size: float
    """Size of a pixel of the images [A]."""

    def __post_init__(self) -> None:
        images = tuple(self.images)
        rotations = np.asarray(self.rotations, dtype=np.float64)
        shifts = np.asarray(self.shifts, dtype=np.float64)

        count = len(images)
        expected: dict[str, tuple[Any, tuple[int, ...]]] = {
            'rotations': (rotations.shape, (count, 3, 3)),
            'shifts': (shifts.shape, (count, 2)),
            'ctf': ((len(self.ctf),), (count,)),
        }
        for name, (shape, wanted) in expected.items():
            if shape != wanted:
                raise ValueError(
                    f'{name} has shape {shape} instead of {wanted}.'
                )
        if not self.pixel_size > 0.0:
            raise ValueError('The pixel size must be positive.')

        object.__setattr__(self, 'images', images)
        object.__setattr__(self, 'rotations', rotations)
        object.__setattr__(self, 'shifts', shifts)

    def __len__(self) -> int:
        return len(self.images)
