"""Reading the images of the particles as the analysis goes through them.

Every particle is analysed with those of several projection directions,
and the images of a direction are needed together. So they are read as
each direction comes, straight from their files, and centred and resampled
to pixels just fine enough for the resolution of the analysis. Only a few
batches of images are in memory at a time, and every image is read once
for each direction it belongs to. A copy of the images in memory, or on a
fast disk, makes those reads cheap where the files are slow to reach.
"""

from __future__ import annotations

import logging
import math
import os
import shutil
from collections import deque
from collections.abc import Iterator, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import vitrio as vio
import vitrio.data as vio_data

from ..particles import Particles

logger = logging.getLogger(__name__)

# Bytes of full-size images to process at once.
_BATCH_BYTES = 1 << 26

# Batches to bring to the device and resample at once: a copy to a device
# is as fast as one thread copies memory, and a few of them go faster.
_TRANSFERS = 4

# Images of a file that a copy takes in together, in one read of the file.
_RUN_LENGTH = 64

# Open files to keep between reads: the images of a batch are spread over
# many of them.
_OPEN_FILES = 64


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


def _make_copy(
    locations: list[vio.ImageLocation],
    preload: bool,
    scratch: str | os.PathLike[str] | None,
) -> vio.ImageScratch | None:
    if preload and scratch is not None:
        raise ValueError(
            'The images are copied to memory or to a scratch, not to both.'
        )
    if preload:
        return vio.create_host_image_scratch(locations, _RUN_LENGTH)
    if scratch is None:
        return None
    if not os.path.isdir(scratch):
        return vio.create_file_image_scratch(locations, scratch, _RUN_LENGTH)
    # A directory holds the copies of several sets of images, as many as
    # its disk has room for.
    held = sum(
        file.stat().st_size for file in Path(scratch).glob('*.vitrio-scratch')
    )
    return vio.create_directory_image_scratch(
        locations, scratch, _RUN_LENGTH, shutil.disk_usage(scratch).free + held
    )


@dataclass(frozen=True)
class ParticleImages:
    """The images of the particles, in their files, and how they are
    centred and resampled as they are read."""

    source: vio_data.ImageSource
    """The image of each particle."""
    shifts: np.ndarray
    """Translation (x, y) that centres each particle [px],
    ``(particles, 2)``."""
    box_size: int
    """Side of the resampled images [px]."""
    pixel_size: float
    """Size of a pixel of the resampled images [A]."""
    loader: vio.ImageLoader | None = None
    """Where reads of files are dispatched. Defaults to the loader shared
    by the process."""

    def __len__(self) -> int:
        return len(self.source)

    def read(
        self,
        groups: Sequence[np.ndarray],
        device: jax.Device | None = None,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Read the images of groups of particles, one group after another.

        The images are read a few batches ahead of the group handed out,
        and those of a particle are read again for every group it is in.

        Args:
            groups: Indices of the particles of each group.
            device: Where to resample. Defaults to the default device of
                JAX.

        Yields:
            The images of each group, ``(members, size, size)``, with the
            centre of each particle at the pixel ``size / 2`` of both
            axes, and which of them hold data: finite, not constant, and
            of values that single precision can compute with. The rest
            are zero.
        """
        size = self.box_size
        full = int(self.source.shape[-1])
        batch_size = int(np.clip(_BATCH_BYTES // (4 * full * full), 16, 1024))
        rows = np.concatenate([np.zeros(0, dtype=np.intp), *groups])

        # The batches go by whatever the groups, all of them of one size:
        # the last one is filled up with rows of its own.
        def draw() -> Iterator[Sequence[int]]:
            for start in range(0, len(rows), batch_size):
                batch = np.resize(rows[start : start + batch_size], batch_size)
                yield batch.tolist()

        def resample(
            batch: dict[str, np.ndarray],
        ) -> tuple[np.ndarray, np.ndarray]:
            images, shifts = jax.device_put(
                (batch['image'], batch['shift']), device
            )
            resampled, valid = _centre_and_resample(images, shifts, size)
            return np.asarray(resampled), np.asarray(valid)

        dataset = vio_data.Dataset(
            {'image': self.source, 'shift': self.shifts}, index=None
        )
        batches = iter(
            vio_data.DataLoader(
                dataset, batch_sampler=draw(), loader=self.loader
            )
        )
        with ThreadPoolExecutor(_TRANSFERS) as pool:
            # As many batches are under way as threads there are, and they
            # are taken in the order they were read.
            window: deque[Future[tuple[np.ndarray, np.ndarray]]] = deque()

            def advance() -> tuple[np.ndarray, np.ndarray]:
                for batch in batches:
                    window.append(pool.submit(resample, batch))
                    if len(window) > _TRANSFERS:
                        break
                return window.popleft().result()

            resampled = np.empty((0, size, size), dtype=np.float32)
            valid = np.empty(0, dtype=bool)
            position = 0
            for group in groups:
                images = np.empty((len(group), size, size), dtype=np.float32)
                usable = np.empty(len(group), dtype=bool)
                filled = 0
                while filled < len(group):
                    if position == len(resampled):
                        resampled, valid = advance()
                        position = 0
                    count = min(len(group) - filled, len(resampled) - position)
                    taken = slice(position, position + count)
                    images[filled : filled + count] = resampled[taken]
                    usable[filled : filled + count] = valid[taken]
                    filled += count
                    position += count
                yield images, usable


def open_images(
    particles: Particles,
    max_pixel_size: float,
    *,
    preload: bool = False,
    scratch: str | os.PathLike[str] | None = None,
) -> ParticleImages:
    """Find the images of some particles, to read them centred and
    resampled.

    Args:
        particles: The particles to read.
        max_pixel_size: Largest size of the resampled pixels [A]. Images
            whose pixels are already larger are kept as they are.
        preload: Whether to copy the images to memory, where reading them
            again takes no file. They are copied as their files hold them,
            which the memory must have room for.
        scratch: A file or a directory on a fast disk to copy the images
            to, and to read them again from. A file is kept, and the next
            analysis of the same images resumes from it. An existing
            directory keeps a file for each set of images, found again by
            the next analysis of it, and makes room for a new one by
            removing the ones used least recently.

    Returns:
        The images, of which none has been read yet.

    Raises:
        ValueError: If the images are not square, or not all of one shape,
            or if they are asked to be copied both to memory and to a
            scratch.
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
        'Found %d images of %d px at %.3f A/px, to read as %d px at %.3f A/px',
        len(particles),
        box_size,
        particles.pixel_size,
        size,
        pixel_size,
    )

    loader = None
    locations = list(source.locations)
    copy = _make_copy(locations, preload, scratch)
    if copy is not None:
        # The copy fills up meanwhile, and a read of what it does not hold
        # yet takes it in.
        loader = vio.loader(cache=_OPEN_FILES, scratch=copy)
        vio.prefetch_scratch_async(copy, locations)
    return ParticleImages(
        source=source,
        shifts=particles.shifts.astype(np.float32),
        box_size=size,
        pixel_size=pixel_size,
        loader=loader,
    )
