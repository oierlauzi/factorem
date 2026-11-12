"""Bringing the particles of a projection direction to its frame."""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

# Geometry is cheap, but GPUs default to reduced-precision float32 products,
# which would misplace pixels by a noticeable fraction.
_matmul = partial(jnp.matmul, precision=jax.lax.Precision.HIGHEST)


def in_plane_rotations(frame: jax.Array, rotations: jax.Array) -> jax.Array:
    """Closest 2D transform taking each particle to the frame of a direction.

    Args:
        frame: Frame of the direction, ``(3, 3)``.
        rotations: Orientation of each particle, ``(..., 3, 3)``.

    Returns:
        Orthogonal matrices of shape ``(..., 2, 2)``. They are
        reflections for the particles that view the direction from the
        opposite side, whose images are mirrored.
    """
    delta = _matmul(frame, rotations.swapaxes(-2, -1))
    u, _, vh = jnp.linalg.svd(delta[..., :2, :2])
    return _matmul(u, vh)


def view_offsets(frame: jax.Array, rotations: jax.Array) -> jax.Array:
    """Where each particle views from, within the cone of a direction.

    Args:
        frame: Frame of the direction, ``(3, 3)``.
        rotations: Orientation of each particle, ``(..., 3, 3)``.

    Returns:
        The components (x, y) of the direction of each particle in the frame
        of the reference one, ``(..., 2)``. Opposite views are mirror
        images, equivalent to the negated direction.
    """
    delta = _matmul(frame, rotations.swapaxes(-2, -1))
    side = jnp.sign(delta[..., 2, 2])
    offsets: jax.Array = side[..., None] * delta[..., :2, 2]
    return offsets


def transform_image(image: jax.Array, matrix: jax.Array) -> jax.Array:
    """Apply an orthogonal 2D transform to an image, about its centre.

    The centre is the pixel ``size // 2`` of each axis for even sizes. What
    falls outside of the image is taken to be zero.

    Args:
        image: ``(height, width)``.
        matrix: ``(2, 2)``, acting on (x, y).
    """
    h, w = image.shape
    centre = jnp.array((w / 2, h / 2), dtype=image.dtype)
    y, x = jnp.mgrid[0:h, 0:w]
    destination = jnp.stack((x.ravel(), y.ravel())).astype(image.dtype)

    # Inverse of p -> M (p - centre) + centre
    source = _matmul(matrix.T, destination - centre[:, None]) + centre[:, None]
    coordinates = [source[1].reshape(h, w), source[0].reshape(h, w)]
    aligned: jax.Array = jax.scipy.ndimage.map_coordinates(
        image, coordinates, order=1, mode='constant', cval=0.0
    )
    return aligned


def transform_images(images: jax.Array, matrices: jax.Array) -> jax.Array:
    """Apply `transform_image` to each image of a batch with its matrix."""
    return jax.vmap(transform_image)(images, matrices)


def soft_disc(box_size: int, radius: float, edge: float) -> np.ndarray:
    """Disc that falls to zero along a raised cosine.

    Args:
        box_size: Side of the image [px].
        radius: Radius within which the disc is one [px].
        edge: Width of the fall beyond the radius [px].
    """
    y, x = np.mgrid[0:box_size, 0:box_size] - box_size / 2
    t = np.clip((np.hypot(x, y) - radius) / max(edge, 1e-6), 0.0, 1.0)
    disc: np.ndarray = 0.5 * (1.0 + np.cos(np.pi * t))
    return disc.astype(np.float32)
