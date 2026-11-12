"""Synchronization by the particles that projection directions share."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ..embedding.directional import DirectionalEmbedding
from .power_method import polar, synchronize

logger = logging.getLogger(__name__)


def shared_particle_products(
    members: Sequence[np.ndarray],
    coordinates: Sequence[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Products of the coordinates that two directions give to the
    particles they share.

    The coordinates of a particle in the directions ``l`` and ``r`` are
    those of the common frame transformed, ``z @ X[l].T`` and
    ``z @ X[r].T``. Their products, summed over the particles the two
    share, are thus about ``X[l] @ X[r].T`` times the covariance of the
    coordinates, which is the same for all the pairs.

    Args:
        members: Indices of the particles of each direction, with none
            repeated in a direction.
        coordinates: Coordinates of the members of each direction,
            ``(members, k)``.

    Returns:
        The pairs of directions with particles in common, as the first and
        the second of each (every direction is paired with itself too),
        and the products ``Z[l].T @ Z[r]`` of each, ``(pairs, k, k)``.
    """
    sizes = np.array([len(group) for group in members], dtype=np.int64)
    components = coordinates[0].shape[1] if len(coordinates) else 0
    if sizes.sum() == 0:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty, np.zeros((0, components, components))
    offsets = np.concatenate(([0], np.cumsum(sizes)))
    particle = np.concatenate(members)
    direction = np.repeat(np.arange(len(members)), sizes)
    z = np.concatenate(coordinates).astype(np.float64)

    # The memberships of each particle, one after another.
    by_particle = np.argsort(particle, kind='stable')
    first = np.concatenate(([0], np.cumsum(np.bincount(particle))))

    left: list[np.ndarray] = []
    right: list[np.ndarray] = []
    blocks: list[np.ndarray] = []
    for index, group in enumerate(members):
        # Every membership of the particles of this direction, with the
        # one of the direction itself that it meets.
        counts = first[group + 1] - first[group]
        skipped = np.cumsum(counts) - counts
        theirs = by_particle[
            np.repeat(first[group] - skipped, counts) + np.arange(counts.sum())
        ]
        own = offsets[index] + np.repeat(np.arange(len(group)), counts)

        # Each pair once, from the first of its directions.
        ahead = direction[theirs] >= index
        theirs, own = theirs[ahead], own[ahead]
        order = np.argsort(direction[theirs], kind='stable')
        theirs, own = theirs[order], own[order]
        partners, start = np.unique(direction[theirs], return_index=True)

        products = z[own][:, :, None] * z[theirs][:, None, :]
        blocks.append(np.add.reduceat(products, start, axis=0))
        left.append(np.full(len(partners), index, dtype=np.int64))
        right.append(partners)
    return np.concatenate(left), np.concatenate(right), np.concatenate(blocks)


@dataclass(frozen=True)
class BurerMonteiroSynchronizer:
    """Synchronizes the directions by the particles they share.

    A particle is a member of every direction close to its own, each of
    which gives it coordinates in a frame of its own. The transforms
    sought are the ones that make those coordinates agree best. They are
    given more columns than rows, the relaxation of Burer and Monteiro,
    which has room to leave the local maxima that square transforms get
    caught in.

    Each coordinate counts by the strength of its eigenimage, which is how
    much of the images it stands for. The ones that are mostly noise are
    weak, and would otherwise count as much as the rest: directions that
    share particles share their noise as well, and agree on it.

    Only neighbouring directions share particles, so how two distant ones
    relate follows from the chain of directions between them.
    """

    rank: int | None = None
    """Columns of the transforms. Defaults to one more than twice the
    components. The components asked for are the leading ones of what
    they give."""
    max_iterations: int = 1024
    """Iterations of the power method to stop at."""
    tolerance: float = 1e-8
    """Relative gain of an iteration below which the power method stops."""
    seed: int = 0
    """Seed of the initial transforms."""

    def __call__(self, embedding: DirectionalEmbedding) -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        count = len(embedding.groups)
        components = embedding.components
        rank = 2 * components + 1 if self.rank is None else self.rank
        if rank < components:
            raise ValueError(
                'The rank must be at least the number of components.'
            )
        strengths = embedding.strengths
        left, right, blocks = shared_particle_products(
            embedding.groups.members, embedding.coordinates
        )
        blocks *= strengths[left][:, :, None] * strengths[right][:, None, :]
        start = polar(rng.normal(size=(count, components, rank)))
        transforms, iterations = synchronize(
            left, right, blocks, start, self.max_iterations, self.tolerance
        )
        transforms *= strengths[:, :, None]
        logger.info(
            'Synchronized %d directions from %d pairs with particles in '
            'common in %d iterations',
            count,
            int(np.sum(left != right)),
            iterations,
        )
        return transforms
