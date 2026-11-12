"""The Fourier coefficients of an image below a band limit."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FourierBand:
    """The Fourier coefficients of a square image below a band limit.

    Images are reduced to these coefficients, which are few when the
    analysis is limited to low resolution. They index the flattened output
    of ``rfft2``, which holds one of each pair of Hermitian mates.
    """

    box_size: int
    indices: np.ndarray
    """Position of each coefficient in the flattened ``rfft2``."""
    frequencies: np.ndarray
    """Integer frequency (kx, ky) of each coefficient, ``kx >= 0``."""
    shells: np.ndarray
    """Rounded radius of each coefficient."""
    multiplicity: np.ndarray
    """Coefficients of the full transform each one stands for: itself and
    its Hermitian mate, unless ``rfft2`` holds both."""

    def __len__(self) -> int:
        return len(self.indices)

    @property
    def shell_count(self) -> int:
        return int(self.shells.max()) + 1

    @property
    def max_radius(self) -> float:
        """Radius of the band, in frequency samples."""
        return float(np.hypot(*self.frequencies.T).max())


def make_fourier_band(box_size: int, max_frequency: float) -> FourierBand:
    """Select the coefficients up to a frequency.

    Args:
        box_size: Side of the images [px].
        max_frequency: Band limit as a digital frequency [1/px]. It is
            clipped to the Nyquist frequency, 0.5.
    """
    if not max_frequency > 0.0:
        raise ValueError('The band limit must be positive.')
    ky = np.fft.fftfreq(box_size, d=1.0 / box_size).round().astype(np.int32)
    kx = np.fft.rfftfreq(box_size, d=1.0 / box_size).round().astype(np.int32)
    ky, kx = np.meshgrid(ky, kx, indexing='ij')
    radius = np.hypot(ky, kx)
    keep = radius <= min(max_frequency, 0.5) * box_size

    # The mates of the first column, and of the last one for even sizes,
    # are in the same column, so they are both part of the output.
    multiplicity = np.full(radius.shape, 2.0, dtype=np.float32)
    multiplicity[:, 0] = 1.0
    if box_size % 2 == 0:
        multiplicity[:, -1] = 1.0

    return FourierBand(
        box_size=box_size,
        indices=np.flatnonzero(keep),
        frequencies=np.stack((kx[keep], ky[keep]), axis=1),
        shells=np.round(radius[keep]).astype(np.int32),
        multiplicity=multiplicity[keep],
    )
