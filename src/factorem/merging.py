"""Merging the embeddings of the projection directions into one."""

from __future__ import annotations

import numpy as np

from .embedding.directional import DirectionalEmbedding


def merge_directions(
    embedding: DirectionalEmbedding,
    transforms: np.ndarray,
    particle_count: int,
) -> np.ndarray:
    """Average the coordinates each particle has in its directions, once
    they are in the common frame.

    The average is the point closest, in least squares, to the coordinates
    that the directions of a particle give it. It is the consensus of a
    generalized Procrustes analysis (Gower, 1975) whose transforms are the
    ones the synchronization found, so it takes no parameters and no
    iterations, and a particle of a single direction keeps the coordinates
    of that one.

    No direction is given a scale or a weight of its own. The embedding
    gives the coordinates of all of them the same prior, which already
    makes them comparable.

    Args:
        embedding: The coordinates of the particles of each direction.
        transforms: The transform of each direction to the common frame,
            as a `Synchronizer` gives them.
        particle_count: How many particles there are.

    Returns:
        The coordinates of each particle, ``(particles, rank)``. They are
        NaN for the particles of no direction.
    """
    total = np.zeros((particle_count, transforms.shape[-1]))
    for members, coordinates, transform in zip(
        embedding.groups.members,
        embedding.coordinates,
        transforms,
        strict=True,
    ):
        total[members] += coordinates @ transform
    count = embedding.groups.multiplicity(particle_count)
    with np.errstate(invalid='ignore'):
        merged: np.ndarray = total / count[:, None]
    return merged


def principal_components(
    coordinates: np.ndarray, components: int
) -> tuple[np.ndarray, np.ndarray]:
    """Rotate coordinates to their principal axes and keep the leading
    ones.

    Args:
        coordinates: ``(particles, dimensions)``. The rows with something
            other than finite numbers are left out, and stay as NaN.
        components: How many axes to keep.

    Returns:
        The coordinates along the leading axes,
        ``(particles, components)``, and the variance along all of the
        axes, in decreasing order.
    """
    finite = np.isfinite(coordinates).all(axis=1)
    centred = coordinates[finite] - coordinates[finite].mean(axis=0)
    covariance = centred.T @ centred / max(len(centred), 1)
    variances, axes = np.linalg.eigh(covariance)
    variances, axes = variances[::-1], axes[:, ::-1]

    out = np.full((len(coordinates), components), np.nan)
    out[finite] = centred @ axes[:, :components]
    return out, variances
