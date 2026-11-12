"""Reading the images of the particles, once, at the sampling of the analysis.

Every particle is analysed with those of several projection directions, but
only up to a modest resolution. So the images are read from their files a
single time, centred, and resampled to pixels just fine enough for that
resolution. What is kept is small enough to stay in memory, where the
analysis of each direction finds its particles without touching the files
again.
"""

from __future__ import annotations

import logging
import math
import os
import tempfile
from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import vitrio.data as vio_data
from vitrio import ImageLoader

from ..particles import Particles

logger = logging.getLogger(__name__)

# Bytes of full-size images to process at once.
_BATCH_BYTES = 1 << 28


@dataclass(frozen=True)
class ImageStack:
    """The images of the particles, centred and resampled."""

    images: np.ndarray
    """``(particles, size, size)``, the centre of each particle at the
    pixel ``size / 2`` of both axes."""
    pixel_size: float
    """Size of a pixel of `images` [A]."""
    usable: np.ndarray
    """Which images hold data: finite, not constant, and of values that
    single precision can compute with."""

    def __len__(self) -> int:
        return len(self.images)

    @property
    def box_size(self) -> int:
        return int(self.images.shape[-1])


def resampled_box_size(box_size: int, pixel_size: float, target: float) -> int:
    """Side of the smallest even box whose pixels are no larger than
    ``target``, or the box itself when its pixels are already larger."""
    size = 2 * math.ceil(0.5 * box_size * pixel_size / target)
    return min(size, box_size)


@partial(jax.jit, static_argnames=('size',))
def _centre_and_resample(
    images: jax.Array, shifts: jax.Array, size: int
) -> tuple[jax.Array, jax.Array]:
    """Shift each image by its (x, y) translation and crop its Fourier
    transform to ``size`` samples. Both are exact for a periodic image."""
    h, w = images.shape[-2:]
    # The spread is not finite either for values that are, but too large
    # to compute with.
    spread = jnp.std(images, axis=(-2, -1))
    usable = jnp.isfinite(spread) & (spread > 0.0)
    images = jnp.where(usable[:, None, None], images, 0.0)

    # The centre of a box is at the pixel of index size // 2, and that of
    # the resampled one has to land on a pixel as well.
    shifts = shifts + jnp.array((w / 2 - w // 2, h / 2 - h // 2))
    kx = jnp.fft.rfftfreq(w)
    ky = jnp.fft.fftfreq(h)
    phase = (
        shifts[:, 0, None, None] * kx + shifts[:, 1, None, None] * ky[:, None]
    )
    images_ft = jnp.fft.rfft2(images) * jnp.exp(-2j * jnp.pi * phase)

    # The lowest frequencies of both signs, and one of each Hermitian pair.
    top, bottom, columns = (size + 1) // 2, size // 2, size // 2 + 1
    images_ft = jnp.concatenate(
        (images_ft[:, :top, :columns], images_ft[:, h - bottom :, :columns]),
        axis=1,
    )
    scale = size * size / (h * w)
    resampled = jnp.fft.irfft2(images_ft, s=(size, size)) * scale
    usable &= jnp.all(jnp.isfinite(resampled), axis=(-2, -1))
    resampled = jnp.where(usable[:, None, None], resampled, 0.0)
    return resampled.astype(jnp.float32), usable


def _allocate(
    shape: tuple[int, ...], scratch: str | os.PathLike[str] | None
) -> np.ndarray:
    if scratch is None:
        return np.empty(shape, dtype=np.float32)
    # The file has no name left once it is mapped, so it goes away with
    # the array even if the process is killed.
    with tempfile.TemporaryFile(dir=scratch, prefix='factorem-') as file:
        return np.memmap(file, dtype=np.float32, mode='w+', shape=shape)


def load_images(
    particles: Particles,
    max_pixel_size: float,
    *,
    device: jax.Device | None = None,
    scratch: str | os.PathLike[str] | None = None,
    loader: ImageLoader | None = None,
) -> ImageStack:
    """Read the images of some particles, centred and resampled.

    Args:
        particles: The particles to read.
        max_pixel_size: Largest size of the resampled pixels [A]. Images
            whose pixels are already larger are kept as they are.
        device: Where to resample. Defaults to the default device of JAX.
        scratch: A directory to keep the resampled images in, instead of
            in memory.
        loader: Where reads of files are dispatched. Defaults to the loader
            shared by the process.

    Returns:
        The resampled images, and which of them are usable.

    Raises:
        ValueError: If the images are not square, or not all of one shape.
    """
    source = vio_data.ImageSource(particles.images, data_type=np.float32)
    if len(source.shape) != 2 or source.shape[0] != source.shape[1]:
        raise ValueError(
            f'Images of shape {source.shape} were found, and the analysis '
            'needs square images.'
        )
    box_size = int(source.shape[0])
    size = resampled_box_size(box_size, particles.pixel_size, max_pixel_size)
    pixel_size = particles.pixel_size * box_size / size
    logger.info(
        'Reading %d images of %d px at %.3f A/px as %d px at %.3f A/px',
        len(particles),
        box_size,
        particles.pixel_size,
        size,
        pixel_size,
    )

    dataset = vio_data.Dataset(
        {'image': source, 'shift': particles.shifts.astype(np.float32)},
        index=None,
    )
    batch_size = int(
        np.clip(_BATCH_BYTES // (4 * box_size * box_size), 16, 1024)
    )
    batches = vio_data.DataLoader(
        dataset,
        batch_size,
        loader=loader,
        pad_last=True,
        output='jax',
        device=device,
    )

    images = _allocate((len(particles), size, size), scratch)
    usable = np.zeros(len(particles), dtype=bool)
    start = 0
    for batch in batches:
        resampled, valid = _centre_and_resample(
            batch['image'], batch['shift'], size
        )
        count = int(np.count_nonzero(np.asarray(batch['valid'])))
        images[start : start + count] = np.asarray(resampled)[:count]
        usable[start : start + count] = np.asarray(valid)[:count]
        start += count

    unusable = np.flatnonzero(~usable)
    if len(unusable):
        logger.warning(
            'Images left out for holding no usable data (%d): %s',
            len(unusable),
            unusable.tolist(),
        )
    return ImageStack(images=images, pixel_size=pixel_size, usable=usable)
