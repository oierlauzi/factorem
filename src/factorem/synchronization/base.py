"""What it takes to bring the projection directions to a common frame."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from ..embedding.directional import DirectionalEmbedding


class Synchronizer(Protocol):
    """Finds how the coordinates of each projection direction relate.

    The coordinates of a direction are known up to an orthogonal transform,
    as its embedder gives them in a frame of its own.

    `BurerMonteiroSynchronizer` relates the directions by the particles
    they share, and `CommonLineSynchronizer` by the common lines of their
    eigenimages.
    """

    def __call__(self, embedding: DirectionalEmbedding) -> np.ndarray:
        """Find the transform of each direction.

        Returns:
            Matrices ``X`` of shape ``(directions, components, rank)``.
            The coordinates ``z`` of a particle in a direction are
            ``z @ X`` in the frame that all the directions share. Their
            rows are orthogonal when the rank allows it, and of a length
            that tells how much each coordinate is to count.
        """
        ...
