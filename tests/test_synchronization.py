from __future__ import annotations

import numpy as np

from factorem.embedding import DirectionalEmbedding
from factorem.geometry import DirectionGroups, sample_directions
from factorem.imaging import make_fourier_band
from factorem.merging import merge_directions, principal_components
from factorem.synchronization import (
    CommonLineSynchronizer,
    common_line_products,
    polar,
    random_pairs,
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
