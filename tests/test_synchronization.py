from __future__ import annotations

import numpy as np
import pytest

from factorem.embedding import DirectionalEmbedding
from factorem.geometry import DirectionGroups, sample_directions
from factorem.imaging import make_fourier_band
from factorem.merging import merge_directions, principal_components
from factorem.synchronization import (
    BurerMonteiroSynchronizer,
    CommonLineSynchronizer,
    common_line_products,
    polar,
    random_pairs,
    shared_particle_products,
    synchronize,
)


def _random_orthogonal(rng, count: int, k: int) -> np.ndarray:
    return polar(rng.normal(size=(count, k, k)))


def _relative_error(transforms: np.ndarray, truth: np.ndarray) -> float:
    """How far the transforms between pairs are from the true ones. The
    transforms themselves are only known up to a common one."""
    found = np.einsum('gij,hkj->ghik', transforms, transforms)
    wanted = np.einsum('gij,hkj->ghik', truth, truth)
    return float(np.abs(found - wanted).max())


def test_polar_gives_the_closest_orthogonal_matrix():
    rng = np.random.default_rng(0)
    q = _random_orthogonal(rng, 5, 4)
    np.testing.assert_allclose(
        q @ q.swapaxes(-2, -1),
        np.broadcast_to(np.eye(4), (5, 4, 4)),
        atol=1e-12,
    )
    stretch = np.diag([1.0, 2.0, 3.0, 4.0])
    np.testing.assert_allclose(
        polar(q @ stretch @ q.swapaxes(-2, -1) @ q), q, atol=1e-12
    )


def test_random_pairs_link_everything():
    rng = np.random.default_rng(0)
    left, right = random_pairs(200, 16, rng)
    assert (left < right).all()
    assert len(np.unique(np.stack((left, right)), axis=1).T) == len(left)
    degree = np.bincount(np.concatenate((left, right)), minlength=200)
    assert degree.min() >= 9
    assert abs(degree.mean() - 18) < 2

    left, right = random_pairs(5, 16, rng)
    assert len(left) == 10


def test_synchronize_recovers_the_transforms():
    rng = np.random.default_rng(1)
    count, k = 60, 4
    truth = _random_orthogonal(rng, count, k)
    left, right = random_pairs(count, 8, rng)
    blocks = truth[left] @ truth[right].swapaxes(-2, -1)
    blocks = blocks * rng.uniform(0.5, 2.0, (len(left), 1, 1))
    blocks += 0.05 * rng.normal(size=blocks.shape)

    start = _random_orthogonal(rng, count, k)
    transforms, iterations = synchronize(left, right, blocks, start)
    assert 1 < iterations < 200
    assert _relative_error(transforms, truth) < 0.1


def test_synchronize_without_pairs_keeps_the_start():
    start = _random_orthogonal(np.random.default_rng(0), 3, 2)
    empty = np.zeros(0, dtype=np.int64)
    transforms, iterations = synchronize(
        empty, empty, np.zeros((0, 2, 2)), start
    )
    np.testing.assert_array_equal(transforms, start)
    assert iterations == 0


def _eigenvolume_slices(band, frames, rng, components: int) -> np.ndarray:
    """Central slices of volumes made of Gaussian blobs, as the Fourier
    coefficients of their projections along each direction."""
    blobs = 6
    centres = rng.uniform(-5.0, 5.0, (blobs, 3))
    weights = rng.normal(size=(blobs, components))
    sigma = 1.5
    kx, ky = band.frequencies.T / band.box_size
    # Where each coefficient of each direction is in 3D: the image axes
    # are the first two rows of the frame.
    q = kx[:, None] * frames[:, None, 0] + ky[:, None] * frames[:, None, 1]
    envelope = np.exp(-2.0 * np.square(np.pi * sigma) * np.sum(q * q, axis=-1))
    phases = np.exp(-2j * np.pi * (q @ centres.T))
    return envelope[..., None] * (phases @ weights)


def test_common_lines_agree():
    rng = np.random.default_rng(0)
    band = make_fourier_band(48, 0.25)
    frames = sample_directions(0.5)
    slices = _eigenvolume_slices(band, frames, rng, 1)
    left, right = np.triu_indices(len(frames), 1)
    products = common_line_products(slices, band, frames, left, right)[:, 0, 0]

    # The slices of two directions are the same along their common line,
    # so their product there is the power of either.
    norms = common_line_products(slices, band, frames, left, left)
    assert (norms == 0.0).all()  # No line in common with itself.
    power = np.sum(band.multiplicity * np.abs(slices[..., 0]) ** 2, axis=1)
    assert np.all(products > 0.0)
    assert np.all(products < power[left])


def test_common_lines_synchronize_the_frames():
    rng = np.random.default_rng(0)
    components = 4
    band = make_fourier_band(64, 0.25)
    frames = sample_directions(0.35)
    count = len(frames)
    truth = _random_orthogonal(rng, count, components)

    # The coordinates of a direction are those of the common frame
    # transformed, z @ X.T, so its eigenimages are transformed alike.
    slices = _eigenvolume_slices(band, frames, rng, components)
    eigenimages = slices @ truth.swapaxes(-2, -1)

    particles = 500
    latent = rng.normal(size=(particles, components)) * (4.0, 3.0, 2.0, 1.0)
    members = tuple(
        np.sort(rng.choice(particles, 100, replace=False)) for _ in range(count)
    )
    embedding = DirectionalEmbedding(
        groups=DirectionGroups(frames, members),
        coordinates=tuple(
            latent[group] @ transform.T
            for group, transform in zip(members, truth, strict=True)
        ),
        eigenimages=eigenimages.astype(np.complex64),
        band=band,
    )

    transforms = CommonLineSynchronizer(partners=16)(embedding)
    assert transforms.shape == (count, components, components)
    assert _relative_error(transforms, truth) < 0.2

    merged = merge_directions(embedding, transforms, particles)
    covered = embedding.groups.multiplicity(particles) > 0
    assert np.isnan(merged[~covered]).all()
    coordinates, variances = principal_components(merged, 2)
    assert np.isnan(coordinates[~covered]).all()
    assert np.all(np.diff(variances) <= 0.0)
    for axis in range(2):
        correlation = np.corrcoef(
            coordinates[covered, axis], latent[covered, axis]
        )
        assert abs(correlation[0, 1]) > 0.95


def test_synchronize_takes_pairs_of_an_item_with_itself():
    rng = np.random.default_rng(2)
    count, k = 200, 3
    truth = _random_orthogonal(rng, count, k)
    left, right = random_pairs(count, 8, rng)
    blocks = truth[left] @ truth[right].swapaxes(-2, -1)
    start = _random_orthogonal(rng, count, k)
    plain, _ = synchronize(left, right, blocks, start.copy(), 500, 1e-12)

    # An item agrees with itself whatever its transform, so the same
    # transforms are the best ones.
    items = np.arange(count)
    with_self, _ = synchronize(
        np.concatenate((left, items)),
        np.concatenate((right, items)),
        np.concatenate((blocks, np.broadcast_to(np.eye(k), (count, k, k)))),
        start.copy(),
        500,
        1e-12,
    )
    assert _relative_error(plain, truth) < 1e-4
    assert _relative_error(with_self, truth) < 1e-4


def _shared_embedding(
    rng: np.random.Generator,
    particles: int,
    components: int,
    noise: float = 0.0,
) -> tuple[DirectionalEmbedding, np.ndarray, np.ndarray]:
    """Directions that each embed the particles around them, in a frame of
    their own, and what they were made from."""
    band = make_fourier_band(16, 0.25)
    frames = sample_directions(0.35)
    count = len(frames)
    truth = _random_orthogonal(rng, count, components)
    scales = np.linspace(4.0, 1.0, components)
    latent = rng.normal(size=(particles, components)) * scales

    # Every particle views from somewhere, and belongs to the directions
    # close to it, opposite ones included.
    views = rng.normal(size=(particles, 3))
    views /= np.linalg.norm(views, axis=1, keepdims=True)
    close = np.abs(views @ frames[:, 2].T) > np.cos(0.6)
    members = tuple(np.flatnonzero(column) for column in close.T)

    # Eigenimages of unit norm, for every coordinate to count the same.
    eigenimages = rng.normal(size=(count, len(band), components)) + 0j
    eigenimages /= np.sqrt(
        np.einsum('c,dck->dk', band.multiplicity, np.abs(eigenimages) ** 2)
    )[:, None, :]
    embedding = DirectionalEmbedding(
        groups=DirectionGroups(frames, members),
        coordinates=tuple(
            latent[group] @ transform.T
            + noise * rng.normal(size=(len(group), components))
            for group, transform in zip(members, truth, strict=True)
        ),
        eigenimages=eigenimages.astype(np.complex64),
        band=band,
    )
    return embedding, truth, latent


def test_shared_particle_products():
    rng = np.random.default_rng(0)
    embedding, _, _ = _shared_embedding(rng, 300, 3)
    members = embedding.groups.members
    coordinates = embedding.coordinates
    left, right, blocks = shared_particle_products(members, coordinates)

    assert (left <= right).all()
    assert len(np.unique(np.stack((left, right)), axis=1).T) == len(left)
    shared = np.array(
        [[len(np.intersect1d(a, b)) for b in members] for a in members]
    )
    np.testing.assert_array_equal(
        np.stack((left, right)), np.nonzero(np.triu(shared))
    )
    for first, second, block in zip(left, right, blocks, strict=True):
        _, own, theirs = np.intersect1d(
            members[first], members[second], return_indices=True
        )
        np.testing.assert_allclose(
            block, coordinates[first][own].T @ coordinates[second][theirs]
        )

    left, right, blocks = shared_particle_products((), ())
    assert len(left) == len(right) == len(blocks) == 0


def test_shared_particles_synchronize_the_frames():
    rng = np.random.default_rng(0)
    particles, components = 3000, 4
    embedding, truth, latent = _shared_embedding(
        rng, particles, components, noise=0.2
    )
    count = len(embedding.groups)

    transforms = BurerMonteiroSynchronizer()(embedding)
    rank = 2 * components + 1
    assert transforms.shape == (count, components, rank)
    np.testing.assert_allclose(
        transforms @ transforms.swapaxes(-2, -1),
        np.broadcast_to(np.eye(components), (count, components, components)),
        atol=1e-5,
    )
    # No more than the components there are get used, of all the room.
    spectrum = np.linalg.svd(transforms.reshape(-1, rank), compute_uv=False)
    assert spectrum[components] < 0.05 * spectrum[0]
    assert _relative_error(transforms, truth) < 0.1

    merged = merge_directions(embedding, transforms, particles)
    coordinates, _ = principal_components(merged, 2)
    for axis in range(2):
        correlation = np.corrcoef(coordinates[:, axis], latent[:, axis])
        assert abs(correlation[0, 1]) > 0.95


def test_shared_particles_count_by_the_strength_of_the_eigenimages():
    rng = np.random.default_rng(0)
    embedding, _, _ = _shared_embedding(rng, 1000, 3)
    np.testing.assert_allclose(embedding.strengths, 1.0, rtol=1e-5)
    plain = BurerMonteiroSynchronizer(rank=3)(embedding)

    # The same embedding, in other units for each coordinate.
    units = np.array([1.0, 2.0, 4.0])
    scaled = DirectionalEmbedding(
        groups=embedding.groups,
        coordinates=tuple(c / units for c in embedding.coordinates),
        eigenimages=embedding.eigenimages * units.astype(np.float32),
        band=embedding.band,
    )
    np.testing.assert_allclose(
        scaled.strengths,
        np.broadcast_to(units, scaled.strengths.shape),
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        BurerMonteiroSynchronizer(rank=3)(scaled),
        plain * units[:, None],
        atol=1e-5,
    )


def test_the_rank_must_hold_the_components():
    rng = np.random.default_rng(0)
    embedding, _, _ = _shared_embedding(rng, 300, 3)
    with pytest.raises(ValueError, match='rank'):
        BurerMonteiroSynchronizer(rank=2)(embedding)
