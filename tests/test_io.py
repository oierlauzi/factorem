from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import starfile
import vitrio as vio

from factorem.geometry import euler_to_matrix
from factorem.io import (
    StarDataset,
    StarFormatError,
    format_embedding,
    open_images,
    parse_embedding,
    particles_from_tables,
)
from factorem.io.images import resampled_box_size
from factorem.particles import CtfParameters, Particles


def _particle_table(count: int = 4) -> pd.DataFrame:
    return pd.DataFrame(
        {
            'rlnImageName': [f'{i + 1}@stack.mrcs' for i in range(count)],
            'rlnAngleRot': np.linspace(0.0, 90.0, count),
            'rlnAngleTilt': np.linspace(10.0, 170.0, count),
            'rlnAnglePsi': np.linspace(-45.0, 45.0, count),
            'rlnDefocusU': np.linspace(10000.0, 20000.0, count),
            'rlnDefocusV': np.linspace(9000.0, 19000.0, count),
            'rlnDefocusAngle': np.full(count, 90.0),
        }
    )


def _modern_tables(count: int = 4) -> dict[str, pd.DataFrame]:
    optics = pd.DataFrame(
        {
            'rlnOpticsGroup': [1, 2],
            'rlnVoltage': [300.0, 200.0],
            'rlnSphericalAberration': [2.7, 2.0],
            'rlnAmplitudeContrast': [0.1, 0.07],
            'rlnImagePixelSize': [2.0, 2.0],
        }
    )
    particles = _particle_table(count)
    particles['rlnOpticsGroup'] = [1 + i % 2 for i in range(count)]
    particles['rlnOriginXAngst'] = 3.0
    particles['rlnOriginYAngst'] = -1.0
    return {'optics': optics, 'particles': particles}


def _legacy_tables(count: int = 4) -> dict[str, pd.DataFrame]:
    particles = _particle_table(count)
    particles['rlnVoltage'] = 300.0
    particles['rlnSphericalAberration'] = 2.7
    particles['rlnAmplitudeContrast'] = 0.1
    particles['rlnDetectorPixelSize'] = 5.0
    particles['rlnMagnification'] = 25000.0
    particles['rlnOriginX'] = 1.5
    particles['rlnOriginY'] = -0.5
    return {'': particles}


# ------------------------------------------------------------------ star


def test_particles_of_the_layout_with_optics():
    tables = _modern_tables()
    particles = particles_from_tables(tables, root='/data')

    assert len(particles) == 4
    assert particles.pixel_size == 2.0
    np.testing.assert_allclose(particles.shifts, [[1.5, -0.5]] * 4)
    np.testing.assert_allclose(particles.ctf.voltage, [300.0, 200.0] * 2)
    np.testing.assert_allclose(
        particles.ctf.amplitude_contrast, [0.1, 0.07] * 2
    )
    np.testing.assert_allclose(particles.ctf.defocus_angle, np.pi / 2)
    np.testing.assert_allclose(particles.ctf.phase_shift, 0.0)
    table = tables['particles']
    np.testing.assert_allclose(
        particles.rotations,
        euler_to_matrix(
            *np.deg2rad(
                (
                    table['rlnAngleRot'],
                    table['rlnAngleTilt'],
                    table['rlnAnglePsi'],
                )
            )
        ),
    )
    assert particles.ctf.as_array().shape == (4, 7)
    assert str(particles.images[2]) == '3@/data/stack.mrcs'


def test_particles_of_the_layout_without_optics():
    particles = particles_from_tables(_legacy_tables())
    np.testing.assert_allclose(particles.pixel_size, 2.0)
    np.testing.assert_allclose(particles.shifts, [[1.5, -0.5]] * 4)
    np.testing.assert_allclose(particles.ctf.voltage, 300.0)


def test_translations_default_to_none():
    tables = _legacy_tables()
    del tables['']['rlnOriginX'], tables['']['rlnOriginY']
    np.testing.assert_array_equal(particles_from_tables(tables).shifts, 0.0)


@pytest.mark.parametrize(
    'label', ['rlnDefocusU', 'rlnAngleTilt', 'rlnVoltage', 'rlnMagnification']
)
def test_missing_labels_are_reported(label):
    tables = _legacy_tables()
    del tables[''][label]
    with pytest.raises(StarFormatError):
        particles_from_tables(tables)


def test_particles_are_needed():
    with pytest.raises(StarFormatError, match='table of particles'):
        particles_from_tables({'optics': _modern_tables()['optics']})


def test_pixel_sizes_must_agree():
    tables = _modern_tables()
    tables['optics']['rlnImagePixelSize'] = [2.0, 2.5]
    with pytest.raises(StarFormatError, match='pixel sizes'):
        particles_from_tables(tables)


def test_optics_groups_must_exist():
    tables = _modern_tables()
    tables['particles'].loc[0, 'rlnOpticsGroup'] = 7
    with pytest.raises(StarFormatError, match='optics group'):
        particles_from_tables(tables)


def test_embedding_text_round_trip():
    coordinates = np.random.default_rng(0).normal(size=(5, 3))
    text = format_embedding(coordinates)
    assert text[0].startswith('[') and text[0].endswith(']')
    np.testing.assert_allclose(parse_embedding(text), coordinates, rtol=1e-4)


@pytest.mark.parametrize('tables', [_modern_tables(), _legacy_tables()])
def test_star_round_trip(tables, tmp_path):
    source = tmp_path / 'particles.star'
    starfile.write(tables, source)
    dataset = StarDataset.read(source)
    coordinates = np.random.default_rng(0).normal(size=(4, 3))
    target = tmp_path / 'embedded.star'
    dataset.write(target, coordinates, group_count=np.arange(4))

    written = starfile.read(target, always_dict=True)
    assert list(written) == list(dataset.tables)
    block = 'particles' if 'particles' in written else next(iter(written))
    table = written[block]
    for label in tables[next(reversed(tables))].columns:
        assert label in table
    np.testing.assert_allclose(
        parse_embedding(table['factoremEmbedding']), coordinates, rtol=1e-4
    )
    np.testing.assert_array_equal(table['factoremGroupCount'], np.arange(4))
    # What was read is left as it was.
    assert 'factoremEmbedding' not in dataset.tables[block]

    with pytest.raises(ValueError):
        dataset.write(target, coordinates[:2])


# ------------------------------------------------------------- particles


def _ctf_parameters(count: int) -> CtfParameters:
    ones = np.ones(count)
    return CtfParameters(
        defocus_u=1e4 * ones,
        defocus_v=1e4 * ones,
        defocus_angle=0.0 * ones,
        phase_shift=0.0 * ones,
        voltage=300.0 * ones,
        spherical_aberration=2.7 * ones,
        amplitude_contrast=0.1 * ones,
    )


def test_particles_are_validated():
    images = vio.data.parse_locations(['1@a.mrcs', '2@a.mrcs'])
    rotations = np.broadcast_to(np.eye(3), (2, 3, 3))
    particles = Particles(
        images, rotations, np.zeros((2, 2)), _ctf_parameters(2), 1.5
    )
    assert len(particles) == 2

    with pytest.raises(ValueError):
        Particles(
            images, rotations[:1], np.zeros((2, 2)), _ctf_parameters(2), 1.5
        )
    with pytest.raises(ValueError):
        Particles(images, rotations, np.zeros((2, 2)), _ctf_parameters(3), 1.5)
    with pytest.raises(ValueError):
        Particles(images, rotations, np.zeros((2, 2)), _ctf_parameters(2), 0.0)


# ---------------------------------------------------------------- images


def _gaussian(
    size: int, centre: tuple[float, float], sigma: float
) -> np.ndarray:
    y, x = np.mgrid[0:size, 0:size]
    return np.exp(
        -0.5 * ((x - centre[0]) ** 2 + (y - centre[1]) ** 2) / sigma**2
    )


def _stack_particles(tmp_path, images, shifts, pixel_size=2.0) -> Particles:
    path = tmp_path / 'stack.mrcs'
    vio.write_stack(np.asarray(images, dtype=np.float32), str(path))
    count = len(images)
    return Particles(
        images=vio.data.parse_locations(
            [f'{i + 1}@{path}' for i in range(count)]
        ),
        rotations=np.broadcast_to(np.eye(3), (count, 3, 3)),
        shifts=np.asarray(shifts, dtype=np.float64),
        ctf=_ctf_parameters(count),
        pixel_size=pixel_size,
    )


def test_resampled_box_size():
    assert resampled_box_size(128, 3.0, 4.0) == 96
    assert resampled_box_size(320, 1.31, 4.0) == 106
    assert resampled_box_size(64, 5.0, 4.0) == 64
    assert resampled_box_size(65, 2.0, 4.0) == 34


@pytest.mark.parametrize('box_size', [32, 33])
def test_images_are_centred_and_resampled(tmp_path, box_size):
    shifts = np.array([[0.0, 0.0], [3.0, -2.0], [-1.25, 4.5]])
    centre = box_size // 2
    images = [
        _gaussian(box_size, (centre - sx, centre - sy), 3.0)
        for sx, sy in shifts
    ]
    particles = _stack_particles(tmp_path, images, shifts)

    opened = open_images(particles, max_pixel_size=4.0)
    size = opened.box_size
    assert len(opened) == 3
    assert size == resampled_box_size(box_size, 2.0, 4.0)
    np.testing.assert_allclose(opened.pixel_size * size, 2.0 * box_size)

    ((read, usable),) = opened.read([np.arange(3)])
    assert usable.all()
    scale = size / box_size
    expected = _gaussian(size, (size / 2, size / 2), 3.0 * scale)
    for image in read:
        np.testing.assert_allclose(image, expected, atol=2e-3)


def test_images_are_kept_when_coarse_enough(tmp_path):
    images = np.random.default_rng(0).normal(size=(3, 16, 16))
    particles = _stack_particles(tmp_path, images, np.zeros((3, 2)), 5.0)
    opened = open_images(particles, max_pixel_size=4.0)
    assert opened.pixel_size == 5.0
    ((read, _),) = opened.read([np.arange(3)])
    np.testing.assert_allclose(read, images, atol=1e-5)


def test_images_without_data_are_flagged(tmp_path):
    images = np.random.default_rng(0).normal(size=(6, 16, 16))
    images[1] = np.nan
    images[3] = 2.0
    # Finite, but its power is not.
    images[5, 4, 7] = 7.7e36
    particles = _stack_particles(tmp_path, images, np.zeros((6, 2)), 5.0)

    ((read, usable),) = open_images(particles, 4.0).read([np.arange(6)])
    np.testing.assert_array_equal(
        usable, [True, False, True, False, True, False]
    )
    np.testing.assert_array_equal(read[~usable], 0.0)
    np.testing.assert_allclose(read[usable], images[usable], atol=1e-5)


_GROUPS = [
    np.arange(30),
    np.array([5, 3]),
    np.array([], dtype=np.int64),
    np.arange(10, 40),
    np.array([39, 0, 39]),
]


def test_groups_are_read_one_after_another(tmp_path, monkeypatch):
    # Batches of 16 images, which the groups do not end at.
    monkeypatch.setattr('factorem.io.images._BATCH_BYTES', 1)
    images = np.random.default_rng(0).normal(size=(40, 8, 8))
    particles = _stack_particles(tmp_path, images, np.zeros((40, 2)), 5.0)

    groups = list(open_images(particles, 4.0).read(_GROUPS))
    assert len(groups) == len(_GROUPS)
    for members, (read, usable) in zip(_GROUPS, groups, strict=True):
        assert usable.shape == members.shape
        assert usable.all()
        np.testing.assert_allclose(read, images[members], atol=1e-5)


@pytest.mark.parametrize('copy', ['memory', 'file', 'directory'])
def test_copies_hold_the_same_images(tmp_path, copy):
    images = np.random.default_rng(0).normal(size=(40, 8, 8))
    particles = _stack_particles(tmp_path, images, np.zeros((40, 2)), 5.0)
    scratch = tmp_path / 'scratch'
    if copy == 'directory':
        scratch.mkdir()

    # A scratch is found again by the next analysis.
    for _ in range(2):
        opened = open_images(
            particles,
            4.0,
            preload=copy == 'memory',
            scratch=None if copy == 'memory' else scratch,
        )
        for members, (read, _) in zip(
            _GROUPS, opened.read(_GROUPS), strict=True
        ):
            np.testing.assert_allclose(read, images[members], atol=1e-5)

    assert scratch.is_file() == (copy == 'file')
    if copy == 'directory':
        assert len(list(scratch.glob('*.vitrio-scratch'))) == 1


def test_images_are_copied_to_one_place(tmp_path):
    images = np.random.default_rng(0).normal(size=(3, 8, 8))
    particles = _stack_particles(tmp_path, images, np.zeros((3, 2)))
    with pytest.raises(ValueError, match='not to both'):
        open_images(particles, 4.0, preload=True, scratch=tmp_path / 'scratch')
