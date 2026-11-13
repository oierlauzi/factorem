"""Embedding the particles of every projection direction on its own.

Particles that view the molecule from about the same direction differ in
conformation, in composition and in little else, once they are rotated in
plane to match. So each direction is analysed by itself, as a set of 2D
images, with no need of a 3D model. `factorem.synchronization` then finds
how the coordinates of the different directions relate.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from typing import NamedTuple, TypeVar

import jax
import jax.numpy as jnp
import numpy as np

from .._device import resolve_device
from ..geometry import DirectionGroups
from ..imaging import (
    FourierBand,
    ctf,
    in_plane_rotations,
    make_fourier_band,
    soft_disc,
    transform_image,
    view_offsets,
)
from ..io.images import ImageStack
from ..particles import Particles
from .base import Embedder
from .ppca import ProbabilisticPCA

logger = logging.getLogger(__name__)

T = TypeVar('T')
U = TypeVar('U')

# Images go through alignment this many at a time, and groups are filled up
# to a multiple of it, so that few sizes have to be compiled.
_GRAIN = 256


@dataclass(frozen=True)
class DirectionalEmbedding:
    """Coordinates of the particles of each projection direction, each
    direction in a frame of its own."""

    groups: DirectionGroups
    coordinates: tuple[np.ndarray, ...]
    """Coordinates of the members of each direction,
    ``(members, components)``."""
    eigenimages: np.ndarray
    """Image of each coordinate of each direction, as complex Fourier
    coefficients in units of the noise,
    ``(directions, coefficients, components)``."""
    band: FourierBand
    """The Fourier coefficients the eigenimages are made of."""

    @property
    def components(self) -> int:
        return int(self.eigenimages.shape[-1])

    @property
    def strengths(self) -> np.ndarray:
        """Norm of the image of each coordinate of each direction, in units
        of the noise, ``(directions, components)``: how much of the images
        a unit of the coordinate stands for."""
        power = np.square(np.abs(self.eigenimages))
        strengths: np.ndarray = np.sqrt(
            np.einsum('c,dck->dk', self.band.multiplicity, power)
        )
        return strengths


class _Constants(NamedTuple):
    """What all the directions share, on the device."""

    mask: jax.Array
    background: jax.Array
    background_gain: jax.Array
    indices: jax.Array
    frequencies: jax.Array
    multiplicity: jax.Array
    shells: jax.Array


def _make_constants(
    stack: ImageStack,
    band: FourierBand,
    resolution: float,
    mask_diameter: float,
    device: jax.Device,
) -> _Constants:
    size = stack.box_size
    edge = resolution / stack.pixel_size
    radius = min(0.5 * mask_diameter / stack.pixel_size, 0.5 * size - 2 * edge)
    if radius <= 0.0:
        raise ValueError(
            f'A box of {size * stack.pixel_size:.0f} A is too small to '
            f'analyse at a resolution of {resolution:.0f} A.'
        )
    # The background is what is left of the largest disc that every in-plane
    # rotation keeps inside of the image.
    mask = soft_disc(size, radius, edge)
    background = soft_disc(size, 0.5 * size - edge, edge) - mask
    gain = np.square(mask).sum() / np.square(background).sum()

    box = size * stack.pixel_size

    def put(value: np.ndarray, dtype: type) -> jax.Array:
        on_device: jax.Array = jax.device_put(value.astype(dtype), device)
        return on_device

    return _Constants(
        mask=put(mask, np.float32),
        background=put(background, np.float32),
        background_gain=put(gain, np.float32),
        indices=put(band.indices, np.int32),
        frequencies=put(band.frequencies / box, np.float32),
        multiplicity=put(band.multiplicity, np.float32),
        shells=put(band.shells, np.int32),
    )


def _view_regressors(
    offsets: jax.Array, valid: jax.Array, order: int
) -> jax.Array:
    """Polynomials of where each particle views from within the cone."""
    mean_square = jnp.sum(jnp.square(offsets) * valid[:, None])
    scale = jnp.sqrt(mean_square / (2.0 * jnp.sum(valid))) + 1e-12
    x, y = (offsets / scale).T
    columns = [jnp.ones_like(x)]
    if order >= 1:
        columns += [x, y]
    if order >= 2:
        columns += [x * x - 1.0, y * y - 1.0, x * y]
    return jnp.stack(columns, axis=1)


def _remove_mean(
    images: jax.Array,
    transfer: jax.Array,
    regressors: jax.Array,
    valid: jax.Array,
    ridge: float = 1e-3,
) -> jax.Array:
    """Subtract the mean image, which changes smoothly with the view.

    Each coefficient of the mean is a combination of the regressors, fitted
    by least squares to what the CTF of each image lets through.
    """
    count = regressors.shape[1]
    weight = valid[:, None]
    products = regressors[:, :, None] * regressors[:, None, :]
    gram = (jnp.square(transfer) * weight).T @ products.reshape(-1, count**2)
    gram = gram.reshape(-1, count, count)
    rhs = (transfer * images * weight).T @ regressors.astype(images.dtype)

    scale = jnp.mean(jnp.trace(gram, axis1=-2, axis2=-1)) / count
    gram = gram + ridge * scale * jnp.eye(count)
    maps = jnp.linalg.solve(gram.astype(rhs.dtype), rhs[..., None])[..., 0]
    mean: jax.Array = transfer * (regressors.astype(maps.dtype) @ maps.T)
    return images - mean


def _shell_average(
    power: jax.Array, multiplicity: jax.Array, shells: jax.Array, count: int
) -> jax.Array:
    total = jnp.bincount(shells, weights=multiplicity * power, length=count)
    norm = jnp.bincount(shells, weights=multiplicity, length=count)
    return total / jnp.maximum(norm, 1.0)


def _to_band(image: jax.Array, indices: jax.Array) -> jax.Array:
    """Coefficients of the band, with the centre of the image as origin
    and white noise of unit variance kept as such."""
    h, w = image.shape
    image = jnp.roll(image, (-(h // 2), -(w // 2)), axis=(0, 1))
    image_ft = jnp.fft.rfft2(image) / jnp.sqrt(h * w)
    coefficients: jax.Array = image_ft.ravel()[indices]
    return coefficients


def _image_features(
    frame: jax.Array,
    constants: _Constants,
    particle: tuple[jax.Array, jax.Array, jax.Array],
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    image, rotation, ctf_parameters = particle
    matrix = in_plane_rotations(frame, rotation)
    aligned = transform_image(image, matrix)
    signal = _to_band(aligned * constants.mask, constants.indices)
    background = _to_band(aligned * constants.background, constants.indices)

    # The image was transformed and its CTF with it: the coefficient at k
    # of the aligned image is the one at M^T k of the original.
    qx, qy = (constants.frequencies @ matrix).T
    transfer = ctf(qx, qy, *ctf_parameters)

    power = jnp.square(background.real) + jnp.square(background.imag)
    return signal, transfer, power, view_offsets(frame, rotation)


@partial(
    jax.jit,
    static_argnames=('embedder', 'components', 'view_order', 'shell_count'),
)
def _embed_group(
    images: jax.Array,
    rotations: jax.Array,
    ctf_parameters: jax.Array,
    count: jax.Array,
    frame: jax.Array,
    constants: _Constants,
    *,
    embedder: Embedder,
    components: int,
    view_order: int,
    shell_count: int,
) -> tuple[jax.Array, jax.Array]:
    valid = (jnp.arange(len(images)) < count).astype(images.dtype)
    signal, transfer, power, offsets = jax.lax.map(
        partial(_image_features, frame, constants),
        (images, rotations, ctf_parameters),
        batch_size=_GRAIN,
    )

    regressors = _view_regressors(offsets, valid, view_order)
    residual = _remove_mean(signal, transfer, regressors, valid)

    # The noise is that of the background, as much of it as the mask lets
    # through. Dividing by it leaves unit variance in the real and in the
    # imaginary part of each coefficient.
    power = jnp.sum(power * valid[:, None], axis=0) / jnp.sum(valid)
    noise = constants.background_gain * _shell_average(
        power, constants.multiplicity, constants.shells, shell_count
    )
    weight = jnp.sqrt(constants.multiplicity)
    scale = weight / jnp.sqrt(jnp.maximum(noise[constants.shells], 1e-20))

    # A coefficient that stands for two counts twice.
    x = jnp.concatenate((residual.real * scale, residual.imag * scale), axis=1)
    c = jnp.concatenate((transfer, transfer), axis=1)
    shells = jnp.concatenate((constants.shells, constants.shells))
    coordinates, loadings = embedder(
        x * valid[:, None], c, shells, valid, components
    )

    half = len(constants.shells)
    eigenimages = (loadings[:half] + 1j * loadings[half:]) / weight[:, None]
    return coordinates, eigenimages


def _padded_size(count: int) -> int:
    """A size to fill a group up to: at most an eighth more, and one of few
    different sizes."""
    step = max(_GRAIN, 1 << max(count.bit_length() - 4, 0))
    return -(-count // step) * step


def _prefetch(
    function: Callable[[T], U], items: Iterable[T], depth: int = 4
) -> Iterator[U]:
    """Results of a function, computed on other threads ahead of use."""
    with ThreadPoolExecutor(max_workers=2) as pool:
        pending: deque[Future[U]] = deque()
        for item in items:
            pending.append(pool.submit(function, item))
            if len(pending) > depth:
                yield pending.popleft().result()
        while pending:
            yield pending.popleft().result()


def embed_directions(
    stack: ImageStack,
    particles: Particles,
    groups: DirectionGroups,
    *,
    resolution: float,
    mask_diameter: float,
    components: int,
    embedder: Embedder | None = None,
    view_order: int = 2,
    device: str | jax.Device | None = None,
) -> DirectionalEmbedding:
    """Embed the particles of each projection direction.

    The particles of a direction are rotated in plane to its frame, masked
    and reduced to their Fourier coefficients up to the resolution. Their
    mean is removed, their noise is whitened, and the embedder does the
    rest.

    Args:
        stack: The images of the particles, as `load_images` reads them.
        particles: The particles.
        groups: The projection directions and their particles.
        resolution: Resolution of the analysis [A].
        mask_diameter: Diameter of the circular mask on the images [A].
            Beyond it lies the background, which the noise is measured on.
        components: How many coordinates to give each particle.
        embedder: What embeds the particles of a direction. Defaults to
            `ProbabilisticPCA`.
        view_order: Order of the polynomial, of where each particle views
            from within the cone of a direction, that models the mean.
        device: Where to compute. Defaults to the default device of JAX.

    Returns:
        The coordinates of the particles of each direction and their
        eigenimages.

    Raises:
        ValueError: If the embedding of some direction is not finite.
    """
    if embedder is None:
        embedder = ProbabilisticPCA()
    device = resolve_device(device)
    band = make_fourier_band(stack.box_size, stack.pixel_size / resolution)
    constants = _make_constants(stack, band, resolution, mask_diameter, device)
    rotations = jax.device_put(particles.rotations.astype(np.float32), device)
    ctf_parameters = jax.device_put(
        particles.ctf.as_array().astype(np.float32), device
    )
    logger.info(
        'Embedding %d directions of %d to %d particles, with %d Fourier '
        'coefficients of each',
        len(groups),
        min(map(len, groups.members), default=0),
        max(map(len, groups.members), default=0),
        len(band),
    )

    def gather(members: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        indices = np.resize(members, _padded_size(len(members)))
        return indices, stack.images[indices]

    coordinates: list[np.ndarray] = []
    eigenimages: list[np.ndarray] = []
    # The device works on a few directions while the results of the ones
    # before are brought back.
    pending: deque[tuple[int, jax.Array, jax.Array]] = deque()

    def collect() -> None:
        count, y, w = pending.popleft()
        coordinates.append(np.asarray(y)[:count])
        eigenimages.append(np.asarray(w))

    start = last = time.monotonic()
    batches = _prefetch(gather, groups.members)
    for i, (members, frame, (indices, images)) in enumerate(
        zip(groups.members, groups.frames, batches, strict=True)
    ):
        on_device = jax.device_put(indices, device)
        y, w = _embed_group(
            jax.device_put(images, device),
            rotations[on_device],
            ctf_parameters[on_device],
            jax.device_put(np.int32(len(members)), device),
            jax.device_put(frame.astype(np.float32), device),
            constants,
            embedder=embedder,
            components=components,
            view_order=view_order,
            shell_count=band.shell_count,
        )
        pending.append((len(members), y, w))
        if len(pending) > 2:
            collect()
        if time.monotonic() - last > 30.0:
            last = time.monotonic()
            logger.info(
                'Embedded %d of %d directions in %.0f s',
                i + 1,
                len(groups),
                last - start,
            )
    while pending:
        collect()

    broken = sum(
        not (np.isfinite(y).all() and np.isfinite(w).all())
        for y, w in zip(coordinates, eigenimages, strict=True)
    )
    if broken:
        raise ValueError(
            f'The embedding of {broken} of the {len(groups)} directions is '
            'not finite. Some image may hold values that are not data.'
        )

    return DirectionalEmbedding(
        groups=groups,
        coordinates=tuple(coordinates),
        eigenimages=np.stack(eigenimages)
        if eigenimages
        else np.zeros((0, len(band), components), dtype=np.complex64),
        band=band,
    )
