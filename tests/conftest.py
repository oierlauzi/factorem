"""Simulated particles to test with."""

from __future__ import annotations

import os

# The tests are small enough for the CPU, and a GPU may be busy.
os.environ.setdefault('JAX_PLATFORMS', 'cpu')

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import starfile
import vitrio as vio

from factorem.geometry import euler_to_matrix


@dataclass(frozen=True)
class Simulation:
    """A STAR file of simulated particles and what they were made from."""

    path: Path
    root: Path
    latent: np.ndarray
    diameter: float
    pixel_size: float
    box_size: int


def _ctf(
    box_size: int,
    pixel_size: float,
    defocus: np.ndarray,
    voltage: float = 300.0,
    aberration: float = 2.7,
    contrast: float = 0.1,
) -> np.ndarray:
    """Plain numpy CTF, without astigmatism, on the grid of ``fft2``."""
    volts = 1e3 * voltage
    wavelength = 12.2643247 / np.sqrt(volts * (1.0 + 0.978466e-6 * volts))
    k = np.fft.fftfreq(box_size, d=pixel_size)
    q2 = np.square(k[:, None]) + np.square(k[None, :])
    phase = (
        np.pi * wavelength * q2 * defocus[:, None, None]
        - 0.5 * np.pi * (1e7 * aberration) * wavelength**3 * np.square(q2)
        + np.arcsin(contrast)
    )
    return np.sin(phase)


def simulate_particles(
    directory: Path,
    count: int = 4000,
    box_size: int = 32,
    pixel_size: float = 4.0,
    noise: float = 0.3,
    seed: int = 0,
) -> Simulation:
    """Write the images and the STAR file of a molecule that moves.

    The molecule is a few Gaussian blobs. Two of them move, each along a
    line and by a coordinate of its own, which makes for two degrees of
    freedom to recover.
    """
    rng = np.random.default_rng(seed)
    rot = rng.uniform(-180.0, 180.0, count)
    tilt = np.rad2deg(np.arccos(rng.uniform(-1.0, 1.0, count)))
    psi = rng.uniform(-180.0, 180.0, count)
    rotations = euler_to_matrix(*np.deg2rad((rot, tilt, psi)))
    shifts = rng.uniform(-2.0, 2.0, (count, 2))  # [px]
    defocus = rng.uniform(8000.0, 20000.0, count)
    latent = rng.normal(size=(count, 2))

    # Positions [A] of the blobs, and how the coordinates move them.
    centres = np.array(
        [
            [0.0, 0.0, 0.0],
            [18.0, 0.0, 6.0],
            [-12.0, 14.0, 0.0],
            [0.0, -16.0, -10.0],
            [8.0, 10.0, 14.0],
        ]
    )
    motions = np.zeros((2, *centres.shape))
    motions[0, 1] = (0.0, 9.0, 0.0)
    motions[1, 3] = (7.0, 0.0, 5.0)
    sigma = 6.0

    positions = centres + np.einsum('nl,lbj->nbj', latent, motions)
    # The image axes are the first two rows of the orientation. The image
    # is the projection moved back by the translation that centres it.
    projected = np.einsum('nij,nbj->nbi', rotations[:, :2], positions)
    projected = projected / pixel_size - shifts[:, None, :]
    grid = np.arange(box_size) - box_size // 2
    dx = grid[None, None, None, :] - projected[:, :, 0, None, None]
    dy = grid[None, None, :, None] - projected[:, :, 1, None, None]
    spread = np.square(sigma / pixel_size)
    images = np.exp(-0.5 * (np.square(dx) + np.square(dy)) / spread).sum(1)

    images_ft = np.fft.fft2(images) * _ctf(box_size, pixel_size, defocus)
    images = np.fft.ifft2(images_ft).real
    images += noise * rng.normal(size=images.shape)

    root = directory / 'project'
    (root / 'Particles').mkdir(parents=True)
    names = []
    per_stack = 1000
    for start in range(0, count, per_stack):
        name = f'Particles/stack_{start // per_stack:03d}.mrcs'
        chunk = images[start : start + per_stack].astype(np.float32)
        vio.write_stack(chunk, str(root / name))
        names += [f'{i + 1:06d}@{name}' for i in range(len(chunk))]

    optics = pd.DataFrame(
        {
            'rlnOpticsGroup': [1],
            'rlnOpticsGroupName': ['opticsGroup1'],
            'rlnVoltage': [300.0],
            'rlnSphericalAberration': [2.7],
            'rlnAmplitudeContrast': [0.1],
            'rlnImagePixelSize': [pixel_size],
            'rlnImageSize': [box_size],
            'rlnImageDimensionality': [2],
        }
    )
    particles = pd.DataFrame(
        {
            'rlnImageName': names,
            'rlnOpticsGroup': 1,
            'rlnAngleRot': rot,
            'rlnAngleTilt': tilt,
            'rlnAnglePsi': psi,
            'rlnOriginXAngst': shifts[:, 0] * pixel_size,
            'rlnOriginYAngst': shifts[:, 1] * pixel_size,
            'rlnDefocusU': defocus,
            'rlnDefocusV': defocus,
            'rlnDefocusAngle': 0.0,
        }
    )
    path = root / 'particles.star'
    starfile.write({'optics': optics, 'particles': particles}, path)
    return Simulation(
        path=path,
        root=root,
        latent=latent,
        diameter=64.0,
        pixel_size=pixel_size,
        box_size=box_size,
    )


@pytest.fixture(scope='session')
def simulation(tmp_path_factory: pytest.TempPathFactory) -> Simulation:
    return simulate_particles(tmp_path_factory.mktemp('simulation'))
