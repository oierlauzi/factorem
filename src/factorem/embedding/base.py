"""What it takes to embed the particles of a projection direction."""

from __future__ import annotations

from typing import Protocol

import jax


class Embedder(Protocol):
    """Reduces the images of one projection direction to a few coordinates.

    The images come as real features with the mean removed and the noise
    whitened, so that each feature of each image is

        ``x = c * signal + noise``

    with ``c`` its CTF and the noise of unit variance. An embedder is
    called while tracing a compiled function, as one of its static
    arguments. So it has to be hashable, as a frozen dataclass is, and it
    has to be written with JAX.
    """

    def __call__(
        self,
        x: jax.Array,
        c: jax.Array,
        shells: jax.Array,
        valid: jax.Array,
        components: int,
    ) -> tuple[jax.Array, jax.Array]:
        """Embed the images of a direction.

        Args:
            x: Features of each image, ``(images, features)``.
            c: CTF of each feature of each image, like ``x``.
            shells: Resolution shell of each feature, ``(features,)``: the
                features of a shell are of about the same spatial
                frequency.
            valid: One for the images to consider and zero for the ones
                that only fill the batch, ``(images,)``.
            components: How many coordinates to give each image.

        Returns:
            The coordinates of each image, ``(images, components)``, and
            the image of each coordinate, ``(features, components)``: the
            signal is modelled as the product of the second by the first.
        """
        ...
