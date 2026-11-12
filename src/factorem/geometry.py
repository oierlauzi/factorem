"""Orientations, projection directions and the particles that view each.

An orientation is a 3x3 rotation matrix in the convention of RELION: its
rows are the axes of the image, x and y, and the direction of projection,
expressed in the frame of the particle. The same holds for the frame of a
projection direction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

_GOLDEN_ANGLE = math.pi * (3.0 - math.sqrt(5.0))


def euler_to_matrix(
    rot: ArrayLike, tilt: ArrayLike, psi: ArrayLike
) -> np.ndarray:
    """Rotation matrices of ZYZ Euler angles [rad], as RELION defines them.

    Returns:
        Matrices of shape ``(..., 3, 3)``, the angles broadcast together.
    """
    rot, tilt, psi = np.broadcast_arrays(
        *(np.asarray(angle, dtype=np.float64) for angle in (rot, tilt, psi))
    )
    ca, sa = np.cos(rot), np.sin(rot)
    cb, sb = np.cos(tilt), np.sin(tilt)
    cg, sg = np.cos(psi), np.sin(psi)

    out = np.empty((*rot.shape, 3, 3))
    out[..., 0, 0] = cg * cb * ca - sg * sa
    out[..., 0, 1] = cg * cb * sa + sg * ca
    out[..., 0, 2] = -cg * sb
    out[..., 1, 0] = -sg * cb * ca - cg * sa
    out[..., 1, 1] = -sg * cb * sa + cg * ca
    out[..., 1, 2] = sg * sb
    out[..., 2, 0] = sb * ca
    out[..., 2, 1] = sb * sa
    out[..., 2, 2] = cb
    return out


def sample_directions(spacing: float) -> np.ndarray:
    """Frames of projection directions that cover half of the sphere.

    Opposite directions see mirrored images, so half of the sphere is all
    there is to cover. The directions follow a Fibonacci spiral, which
    spreads them evenly.

    Args:
        spacing: Angle between neighbouring directions [rad].

    Returns:
        Frames of shape ``(directions, 3, 3)``.
    """
    if not spacing > 0.0:
        raise ValueError('The spacing of the directions must be positive.')
    count = max(round(2.0 * math.pi / (spacing * spacing)), 1)
    rot = _GOLDEN_ANGLE * np.arange(count)
    tilt = np.arccos(np.linspace(0.0, 1.0, count))
    return euler_to_matrix(rot, tilt, 0.0)


@dataclass(frozen=True)
class DirectionGroups:
    """Projection directions and the particles that view the particle from
    close to each of them."""

    frames: np.ndarray
    """Frame of each direction, ``(directions, 3, 3)``."""
    members: tuple[np.ndarray, ...]
    """Sorted indices of the particles of each direction."""

    def __post_init__(self) -> None:
        if len(self.frames) != len(self.members):
            raise ValueError('Every direction needs its members.')

    def __len__(self) -> int:
        return len(self.members)

    def multiplicity(self, particle_count: int) -> np.ndarray:
        """How many directions each particle is a member of."""
        if not self.members:
            return np.zeros(particle_count, dtype=np.int64)
        indices: np.ndarray = np.concatenate(self.members)
        return np.bincount(indices, minlength=particle_count)


def _cosines(axes: np.ndarray, references: np.ndarray) -> np.ndarray:
    """Cosine of the angle between axes, whichever way each one points."""
    cosines: np.ndarray = np.abs(axes @ references.T)
    return cosines


def group_by_direction(
    rotations: np.ndarray,
    frames: np.ndarray,
    aperture: float,
    min_size: int = 1,
    usable: np.ndarray | None = None,
    chunk: int = 64,
) -> DirectionGroups:
    """Group particles by the projection directions they are close to.

    A particle belongs to every direction within ``aperture`` of its own,
    or of its opposite. Directions with fewer than ``min_size`` particles
    are dropped, and the particles left without a direction join the
    closest one that remains, so that every usable particle is in a group.

    Args:
        rotations: Orientation of each particle, ``(particles, 3, 3)``.
        frames: Frame of each direction, ``(directions, 3, 3)``.
        aperture: Half-angle of the cone around each direction [rad].
        min_size: Fewest particles a direction may have.
        usable: Which particles to group. Defaults to all of them.
        chunk: How many directions to compare against at once.

    Returns:
        The directions that were kept, with their particles.
    """
    axes = np.ascontiguousarray(rotations[:, 2, :], dtype=np.float32)
    references = np.ascontiguousarray(frames[:, 2, :], dtype=np.float32)
    if usable is None:
        usable = np.ones(len(axes), dtype=bool)
    threshold = math.cos(aperture)

    members: list[np.ndarray] = []
    for start in range(0, len(references), chunk):
        inside = _cosines(axes, references[start : start + chunk]) >= threshold
        inside &= usable[:, None]
        members.extend(np.flatnonzero(column) for column in inside.T)

    kept = [i for i, group in enumerate(members) if len(group) >= min_size]
    members = [members[i] for i in kept]
    if not kept:
        return DirectionGroups(frames[kept], ())

    covered = ~usable
    for group in members:
        covered[group] = True
    orphans = np.flatnonzero(~covered)
    if len(orphans):
        closest = np.concatenate(
            [
                np.argmax(_cosines(axes[rows], references[kept]), axis=1)
                for rows in np.array_split(orphans, len(orphans) // 4096 + 1)
            ]
        )
        for i in np.unique(closest):
            members[i] = np.union1d(members[i], orphans[closest == i])

    return DirectionGroups(frames[kept], tuple(members))
