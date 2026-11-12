"""Probabilistic principal component analysis of images with a CTF."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp


def _top_singular(
    x: jax.Array, k: int
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Leading singular triplets, from the smaller of the Gram matrices."""
    n, d = x.shape
    if n <= d:
        values, vectors = jnp.linalg.eigh(x @ x.T)
        s = jnp.sqrt(jnp.maximum(values[::-1][:k], 1e-20))
        u = vectors[:, ::-1][:, :k]
        return u, s, (x.T @ u) / s
    values, vectors = jnp.linalg.eigh(x.T @ x)
    s = jnp.sqrt(jnp.maximum(values[::-1][:k], 1e-20))
    v = vectors[:, ::-1][:, :k]
    return (x @ v) / s, s, v


def _wiener(
    x: jax.Array, c: jax.Array, valid: jax.Array, inverse_snr: float
) -> jax.Array:
    c2 = jnp.square(c)
    floor = inverse_snr * jnp.mean(c2, axis=1, keepdims=True)
    corrected: jax.Array = (x * c) / (c2 + floor) * valid[:, None]
    return corrected


@dataclass(frozen=True)
class ProbabilisticPCA:
    """Principal components of the images behind their CTFs.

    The images are modelled as

        ``x = c * (W z) + noise``,    ``z ~ N(0, I)``,

    with the CTF, ``c``, of each one and noise of unit variance. The
    eigenimages, ``W``, are fitted by expectation maximization, and the
    coordinates of an image are the mean of ``z`` given the image. Each
    image is thus weighted by what its CTF lets through, with no need to
    correct or to fill in what it does not.

    The scale of the coordinates is that of the prior, which the noise
    sets, so nothing is left to tune.
    """

    iterations: int = 30
    """Iterations of expectation maximization."""
    start_inverse_snr: float = 0.1
    """Noise to signal ratio of the Wiener filter that corrects the images
    for the first guess of the eigenimages, a plain PCA."""
    ridge: float = 1e-3
    """Regularization of the eigenimages, relative to the number of
    images. It only matters at the frequencies that no CTF lets through."""

    def __call__(
        self, x: jax.Array, c: jax.Array, valid: jax.Array, components: int
    ) -> tuple[jax.Array, jax.Array]:
        k = components
        count = jnp.sum(valid)
        weight = valid[:, None]
        c2 = jnp.square(c) * weight
        cx = c * x * weight
        eye = jnp.eye(k, dtype=x.dtype)

        corrected = _wiener(x, c, valid, self.start_inverse_snr)
        _, s, v = _top_singular(corrected, k)
        loadings = v * (s / jnp.sqrt(count))

        def posterior(loadings: jax.Array) -> tuple[jax.Array, jax.Array]:
            outer = loadings[:, :, None] * loadings[:, None, :]
            precision = (c2 @ outer.reshape(-1, k * k)).reshape(-1, k, k)
            covariance = jnp.linalg.inv(precision + eye)
            mean = jnp.einsum('nkl,nl->nk', covariance, cx @ loadings)
            return mean, covariance

        def step(_: int, loadings: jax.Array) -> jax.Array:
            mean, covariance = posterior(loadings)
            second = mean[:, :, None] * mean[:, None, :] + covariance
            second = second * valid[:, None, None]
            lhs = (c2.T @ second.reshape(-1, k * k)).reshape(-1, k, k)
            rhs = cx.T @ mean
            lhs = lhs + self.ridge * count * eye
            updated: jax.Array = jnp.linalg.solve(lhs, rhs[..., None])[..., 0]
            return updated

        loadings = jax.lax.fori_loop(0, self.iterations, step, loadings)
        mean, _ = posterior(loadings)

        # Uncorrelated coordinates, in order of decreasing variance.
        _, rotation = jnp.linalg.eigh((mean * weight).T @ mean / count)
        rotation = rotation[:, ::-1]
        return mean @ rotation, loadings @ rotation
