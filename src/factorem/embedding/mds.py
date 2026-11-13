"""Multidimensional scaling of the Mahalanobis distances between images."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

# Terms of the distances to compute at once.
_BATCH_TERMS = 1 << 28


def signal_power(
    x: jax.Array, c: jax.Array, shells: jax.Array, valid: jax.Array
) -> jax.Array:
    """Variance of the signal behind each feature, one for each shell.

    The noise of a feature is of unit variance, so its mean square is
    ``c**2 * power + 1``. The power of a shell is the one that fits best,
    in least squares, the squares of its features. It is zero where they
    do not exceed the noise.
    """
    c2 = jnp.square(c) * valid[:, None]
    excess = jnp.sum(c2 * (jnp.square(x) - 1.0), axis=0)
    norm = jnp.sum(jnp.square(c2), axis=0)
    excess = jax.ops.segment_sum(excess, shells, x.shape[1])
    norm = jax.ops.segment_sum(norm, shells, x.shape[1])
    power = jnp.maximum(excess, 0.0) / jnp.maximum(norm, 1e-20)
    return power[shells]


def mahalanobis_distances(
    x: jax.Array, c: jax.Array, power: jax.Array
) -> jax.Array:
    """Squared Mahalanobis distance between every two images.

    With a Gaussian prior of the given power, what is known of the signal
    of an image is Gaussian too, with mean ``m`` and variance ``v``,

        ``v = power / (c**2 * power + 1)``,    ``m = v * c * x``.

    The distance between two images is that between their means, each
    feature over the variance of the difference, the sum of both
    variances (Bhamre, Zhao and Singer, 2017).

    Args:
        x: Features of each image, ``(images, features)``.
        c: CTF of each feature of each image, like ``x``.
        power: Variance of the signal behind each feature, ``(features,)``.

    Returns:
        The squares of the distances, ``(images, images)``.
    """
    variance = power / (jnp.square(c) * power + 1.0)
    mean = variance * c * x

    def row(index: jax.Array) -> jax.Array:
        # A feature of no power has no variance, and no difference either.
        total = jnp.maximum(variance + variance[index], 1e-30)
        distances: jax.Array = jnp.sum(
            jnp.square(mean - mean[index]) / total, axis=1
        )
        return distances

    batch = max(1, _BATCH_TERMS // (x.shape[0] * x.shape[1]))
    squared: jax.Array = jax.lax.map(row, jnp.arange(len(x)), batch_size=batch)
    return squared


def classical_mds(
    squared_distances: jax.Array, valid: jax.Array, components: int
) -> jax.Array:
    """Leading axes of the points that are at the given distances.

    The products of the points with each other, once centred, follow from
    the squares of their distances (Torgerson, 1952). Their leading
    eigenvectors are the coordinates of the points along their principal
    axes.

    Returns:
        The coordinates of each point along each axis, uncorrelated and
        of unit variance, ``(points, components)``.
    """
    count = jnp.sum(valid)
    weight = valid / count
    mean = squared_distances @ weight
    centred = squared_distances - mean[:, None] - mean[None, :] + mean @ weight
    products = -0.5 * centred * valid[:, None] * valid[None, :]
    _, vectors = jnp.linalg.eigh(products)
    coordinates: jax.Array = vectors[:, ::-1][:, :components] * jnp.sqrt(count)
    return coordinates


@dataclass(frozen=True)
class MahalanobisMDS:
    """Multidimensional scaling of the Mahalanobis distances between the
    images.

    Two images are as far as what is known of their signals differs, in
    units of how well it is known: a frequency that the CTF of either
    lets little through counts for little. The power of the signal that
    this takes is measured on the images, shell by shell. The coordinates
    are those of classical multidimensional scaling, the points whose
    distances are closest to the ones between the images, along their
    principal axes.

    It takes the distances between every two images of a direction, so
    its cost grows with the square of how many they are, and with their
    cube to find the axes. What is known of an image is the less the
    less its CTF lets through, and the closer that leaves it to the rest:
    how far from the centre the coordinates of an image are tells of its
    defocus too, which those of `ProbabilisticPCA` do not.

    The coordinates are given unit variance. The image of each one is
    the one that, behind the CTF of each image, fits the images best in
    least squares.
    """

    ridge: float = 1e-3
    """Regularization of the eigenimages, relative to the number of
    images. It only matters at the frequencies that no CTF lets through."""

    def __call__(
        self,
        x: jax.Array,
        c: jax.Array,
        shells: jax.Array,
        valid: jax.Array,
        components: int,
    ) -> tuple[jax.Array, jax.Array]:
        k = components
        count = jnp.sum(valid)
        power = signal_power(x, c, shells, valid)
        distances = mahalanobis_distances(x, c, power)
        coordinates = classical_mds(distances, valid, k)

        c2 = jnp.square(c) * valid[:, None]
        cx = c * x * valid[:, None]
        outer = coordinates[:, :, None] * coordinates[:, None, :]
        lhs = (c2.T @ outer.reshape(-1, k * k)).reshape(-1, k, k)
        lhs = lhs + self.ridge * count * jnp.eye(k, dtype=x.dtype)
        rhs = cx.T @ coordinates
        eigenimages: jax.Array = jnp.linalg.solve(lhs, rhs[..., None])[..., 0]
        return coordinates, eigenimages
