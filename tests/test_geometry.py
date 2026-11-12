from __future__ import annotations

import numpy as np
import pytest

from factorem.geometry import (
    euler_to_matrix,
    group_by_direction,
    sample_directions,
)


def _random_rotations(count: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return euler_to_matrix(
        rng.uniform(-np.pi, np.pi, count),
        np.arccos(rng.uniform(-1.0, 1.0, count)),
        rng.uniform(-np.pi, np.pi, count),
    )


def test_euler_matrices_are_rotations():
    matrices = _random_rotations(100)
    identity = matrices @ matrices.swapaxes(-2, -1)
    np.testing.assert_allclose(
        identity, np.broadcast_to(np.eye(3), identity.shape), atol=1e-12
    )
    np.testing.assert_allclose(np.linalg.det(matrices), 1.0)


def test_euler_angles_follow_relion():
    # The projection direction has the polar angles (tilt, rot).
    rot, tilt = 0.3, 1.1
    matrix = euler_to_matrix(rot, tilt, 0.7)
    expected = (
        np.sin(tilt) * np.cos(rot),
        np.sin(tilt) * np.sin(rot),
        np.cos(tilt),
    )
    np.testing.assert_allclose(matrix[2], expected)

    # The in-plane angle turns the axes of the image about it.
    turned = euler_to_matrix(0.0, 0.0, np.pi / 2)
    np.testing.assert_allclose(turned[0], (0.0, 1.0, 0.0), atol=1e-12)
    np.testing.assert_allclose(turned[1], (-1.0, 0.0, 0.0), atol=1e-12)


def test_euler_angles_broadcast():
    assert euler_to_matrix(np.zeros(5), 0.1, 0.0).shape == (5, 3, 3)
    assert euler_to_matrix(0.0, 0.0, 0.0).shape == (3, 3)


def test_directions_cover_half_of_the_sphere():
    spacing = 0.1
    frames = sample_directions(spacing)
    axes = frames[:, 2]
    assert len(frames) == round(2 * np.pi / spacing**2)
    assert (axes[:, 2] >= -1e-12).all()

    # Every axis, whichever way it points, has a direction close by.
    probes = _random_rotations(2000)[:, 2]
    closest = np.abs(probes @ axes.T).max(axis=1)
    assert np.arccos(closest.min()) < spacing


def test_spacing_must_be_positive():
    with pytest.raises(ValueError):
        sample_directions(0.0)


def test_groups_hold_the_views_within_the_aperture():
    rotations = _random_rotations(5000)
    frames = sample_directions(0.3)
    aperture = 0.25
    groups = group_by_direction(rotations, frames, aperture)

    assert len(groups) == len(frames)
    cosine = np.abs(rotations[:, 2] @ groups.frames[:, 2].T)
    for i, members in enumerate(groups.members):
        inside = np.flatnonzero(cosine[:, i] >= np.cos(aperture))
        np.testing.assert_array_equal(members, inside)
    np.testing.assert_array_equal(
        groups.multiplicity(len(rotations)),
        (cosine >= np.cos(aperture)).sum(axis=1),
    )


def test_opposite_views_go_together():
    frames = euler_to_matrix(0.0, np.array([0.0, 1.0]), 0.0)
    rotations = euler_to_matrix(0.0, np.array([0.05, np.pi - 0.05, 1.0]), 0.0)
    groups = group_by_direction(rotations, frames, 0.1)
    np.testing.assert_array_equal(groups.members[0], [0, 1])
    np.testing.assert_array_equal(groups.members[1], [2])


def test_orphans_join_the_closest_direction():
    # Many particles around the pole, and a single one far from it.
    rng = np.random.default_rng(0)
    tilt = np.concatenate((rng.uniform(0.0, 0.05, 50), [1.2]))
    rotations = euler_to_matrix(rng.uniform(0, 6.28, 51), tilt, 0.0)
    frames = sample_directions(0.2)
    groups = group_by_direction(rotations, frames, 0.1, min_size=10)

    assert 0 < len(groups) < len(frames)
    assert all(len(members) >= 10 for members in groups.members)
    assert (groups.multiplicity(51) > 0).all()
    assert groups.multiplicity(51)[50] == 1


def test_unusable_particles_are_left_out():
    rotations = _random_rotations(500)
    usable = np.ones(500, dtype=bool)
    usable[[3, 77]] = False
    groups = group_by_direction(
        rotations, sample_directions(0.5), 0.5, min_size=5, usable=usable
    )
    count = groups.multiplicity(500)
    assert (count[usable] > 0).all()
    assert (count[~usable] == 0).all()


def test_no_direction_may_be_left():
    rotations = _random_rotations(10)
    groups = group_by_direction(
        rotations, sample_directions(0.5), 0.1, min_size=100
    )
    assert len(groups) == 0
    assert (groups.multiplicity(10) == 0).all()
