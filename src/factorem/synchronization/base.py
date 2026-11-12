"""What it takes to bring the projection directions to a common frame."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from ..embedding.directional import DirectionalEmbedding


class Synchronizer(Protocol):
    """Finds how the coordinates of each projection direction relate.

    The coordinates of a direction are known up to an orthogonal transform,
    as its embedder gives them in a frame of its own.
    """

    def __call__(self, embedding: DirectionalEmbedding) -> np.ndarray:
        """Find the transform of each direction.

        Returns:
            Matrices ``X`` of shape ``(directions, components, rank)``,
            with orthonormal rows when the rank allows it. The coordinates
            ``z`` of a particle in a direction are ``z @ X`` in the frame
            that all the directions share.
        """
        ...
