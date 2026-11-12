from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from factorem.geometry import euler_to_matrix
from factorem.imaging import (
    ctf,
    electron_wavelength,
    in_plane_rotations,
    make_fourier_band,
    soft_disc,
    transform_image,
    transform_images,
    view_offsets,
)


def _blobs(
    box_size: int, positions: np.ndarray, sigma: float = 2.0
) -> np.ndarray:
    """Gaussian blobs at positions (x, y) from the centre of the image."""
    grid = np.arange(box_size) - box_size // 2
    dx = grid[None, None, :] - positions[:, 0, None, None]
    dy = grid[None, :, None] - positions[:, 1, None, None]
    return np.exp(-0.5 * (dx**2 + dy**2) / sigma**2).sum(axis=0)


# ------------------------------------------------------------------ band


@pytest.mark.parametrize('box_size', [16, 17])
def test_band_holds_one_of_each_hermitian_pair(box_size):
    band = make_fourier_band(box_size, 0.3)
    k = np.fft.fftfreq(box_size, d=1.0 / box_size)
    radius = np.hypot(k[:, None], k[None, :])
    assert band.multiplicity.sum() == np.count_nonzero(radius <= 0.3 * box_size)
    assert (band.frequencies[:, 0] >= 0).all()
    assert band.shell_count == band.shells.max() + 1
    assert band.max_radius <= 0.3 * box_size


def test_band_indexes_the_real_transform():
    box_size = 16
    band = make_fourier_band(box_size, 0.4)
    kx, ky = 3, -2
    y, x = np.mgrid[0:box_size, 0:box_size]
    wave = np.cos(2 * np.pi * (kx * x + ky * y) / box_size)
    coefficients = np.fft.rfft2(wave).ravel()[band.indices]

    (found,) = np.flatnonzero(np.abs(coefficients) > 1e-6)
    np.testing.assert_array_equal(band.frequencies[found], (kx, ky))
    assert band.shells[found] == round(np.hypot(kx, ky))


def test_band_stops_at_nyquist():
    assert len(make_fourier_band(16, 2.0)) == len(make_fourier_band(16, 0.5))
    with pytest.raises(ValueError):
        make_fourier_band(16, 0.0)


# ------------------------------------------------------------------- ctf

_OPTICS = dict(voltage=300.0, spherical_aberration=2.7, amplitude_contrast=0.07)


def test_wavelength_of_usual_voltages():
    np.testing.assert_allclose(electron_wavelength(300.0), 0.019687, rtol=1e-4)
    np.testing.assert_allclose(electron_wavelength(200.0), 0.025079, rtol=1e-4)


def test_ctf_follows_the_textbook():
    q = np.linspace(0.0, 0.1, 50)
    defocus = 15000.0
    wavelength = 0.019687
    phase = (
        np.pi * wavelength * q**2 * defocus
        - 0.5 * np.pi * 2.7e7 * wavelength**3 * q**4
        + np.arcsin(0.07)
    )
    values = ctf(q, 0.0, defocus, defocus, 0.0, 0.0, **_OPTICS)
    np.testing.assert_allclose(values, np.sin(phase), atol=2e-3)
    np.testing.assert_allclose(values[0], 0.07, atol=1e-6)


def test_ctf_phase_plate_shifts_the_phase():
    values = ctf(0.0, 0.0, 1e4, 1e4, 0.0, np.pi / 2, **_OPTICS)
    np.testing.assert_allclose(values, np.cos(np.arcsin(0.07)), atol=1e-6)


def test_astigmatism_follows_the_major_axis():
    q = np.linspace(0.01, 0.08, 20)
    u, v, angle = 20000.0, 12000.0, 0.6
    along = ctf(
        q * np.cos(angle), q * np.sin(angle), u, v, angle, 0.0, **_OPTICS
    )
    across = ctf(
        -q * np.sin(angle), q * np.cos(angle), u, v, angle, 0.0, **_OPTICS
    )
    np.testing.assert_allclose(
        along, ctf(q, 0.0, u, u, 0.0, 0.0, **_OPTICS), atol=1e-3
    )
    np.testing.assert_allclose(
        across, ctf(q, 0.0, v, v, 0.0, 0.0, **_OPTICS), atol=1e-3
    )


def test_ctf_broadcasts_over_particles():
    q = np.linspace(0.0, 0.05, 7)
    defocus = np.array([1e4, 2e4, 3e4])[:, None]
    assert ctf(q, q, defocus, defocus, 0.0, 0.0, **_OPTICS).shape == (3, 7)


# ------------------------------------------------------------- alignment


def test_in_plane_rotation_undoes_psi():
    frame = jnp.asarray(euler_to_matrix(0.4, 0.9, 0.0), dtype=jnp.float32)
    psi = 0.5
    rotation = jnp.asarray(euler_to_matrix(0.4, 0.9, psi), dtype=jnp.float32)
    matrix = np.asarray(in_plane_rotations(frame, rotation))
    expected = [[np.cos(psi), -np.sin(psi)], [np.sin(psi), np.cos(psi)]]
    np.testing.assert_allclose(matrix, expected, atol=1e-5)


def test_opposite_views_are_mirrored():
    frame = jnp.asarray(euler_to_matrix(0.0, 0.0, 0.0), dtype=jnp.float32)
    rotation = jnp.asarray(euler_to_matrix(0.3, np.pi, 0.0), dtype=jnp.float32)
    matrix = np.asarray(in_plane_rotations(frame, rotation))
    np.testing.assert_allclose(np.linalg.det(matrix), -1.0, atol=1e-5)


def test_view_offsets_point_to_the_view():
    frame = jnp.asarray(euler_to_matrix(0.0, 0.0, 0.0), dtype=jnp.float32)
    tilt = 0.1
    rotations = jnp.asarray(
        euler_to_matrix(0.0, np.array([tilt, np.pi - tilt]), 0.0),
        dtype=jnp.float32,
    )
    offsets = np.asarray(view_offsets(frame, rotations))
    np.testing.assert_allclose(offsets[0], (np.sin(tilt), 0.0), atol=1e-6)
    # The opposite view is the same as the mirrored direction.
    np.testing.assert_allclose(offsets[1], (-np.sin(tilt), 0.0), atol=1e-6)


def test_aligned_projections_agree():
    # Blobs in 3D, seen from one direction with two in-plane angles and
    # from the opposite one: all match once aligned to the same frame.
    rng = np.random.default_rng(0)
    blobs = rng.uniform(-8.0, 8.0, (4, 3))
    frame = euler_to_matrix(0.7, 1.1, 0.0)
    rotations = np.stack(
        [
            euler_to_matrix(0.7, 1.1, 0.9),
            euler_to_matrix(0.7 + np.pi, np.pi - 1.1, -2.0),
        ]
    )
    np.testing.assert_allclose(rotations[1, 2], -frame[2], atol=1e-12)

    def project(rotation):
        return _blobs(48, blobs @ rotation[:2].T, 3.0).astype(np.float32)

    images = jnp.asarray(np.stack([project(r) for r in rotations]))
    matrices = in_plane_rotations(
        jnp.asarray(frame, dtype=jnp.float32),
        jnp.asarray(rotations, dtype=jnp.float32),
    )
    aligned = np.asarray(transform_images(images, matrices))
    reference = project(frame)
    assert reference.max() > 0.9
    for image in aligned:
        assert np.abs(image - reference).max() < 0.03


def test_transform_keeps_the_centre():
    image = _blobs(32, np.array([[0.0, 0.0]])).astype(np.float32)
    turn = jnp.asarray([[0.0, -1.0], [1.0, 0.0]])
    turned = np.asarray(transform_image(jnp.asarray(image), turn))
    np.testing.assert_allclose(turned, image, atol=1e-4)


def test_soft_disc():
    disc = soft_disc(32, 8.0, 4.0)
    assert disc[16, 16] == 1.0
    assert disc[16, 16 + 8] == 1.0
    assert 0.0 < disc[16, 16 + 10] < 1.0
    assert disc[16, 16 + 12] == 0.0
    np.testing.assert_allclose(disc, disc.T)
