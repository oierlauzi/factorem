"""The whole analysis on simulated particles."""

from __future__ import annotations

from dataclasses import dataclass, replace

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
import starfile
import vitrio as vio

import factorem
from factorem.cli import main
from factorem.io import parse_embedding


def _explained(coordinates: np.ndarray, latent: np.ndarray) -> np.ndarray:
    """Fraction of the variance of each latent coordinate that a linear
    function of the embedding accounts for."""
    design = np.column_stack((coordinates, np.ones(len(coordinates))))
    fit, *_ = np.linalg.lstsq(design, latent, rcond=None)
    residual = latent - design @ fit
    return 1.0 - residual.var(axis=0) / latent.var(axis=0)


@pytest.mark.slow
def test_recovers_the_motions(simulation):
    dataset = factorem.StarDataset.read(simulation.path, root=simulation.root)
    embedding = factorem.analyze(
        dataset.particles, diameter=simulation.diameter, components=4
    )

    assert embedding.coordinates.shape == (len(simulation.latent), 4)
    assert np.isfinite(embedding.coordinates).all()
    assert (embedding.group_count > 0).all()
    assert np.all(np.diff(embedding.variances) <= 0.0)
    assert np.all(_explained(embedding.coordinates, simulation.latent) > 0.45)


@pytest.mark.slow
def test_recovers_the_motions_from_distances(simulation):
    dataset = factorem.StarDataset.read(simulation.path, root=simulation.root)
    embedding = factorem.analyze(
        dataset.particles,
        diameter=simulation.diameter,
        components=4,
        embedder=factorem.MahalanobisMDS(),
    )

    assert np.isfinite(embedding.coordinates).all()
    assert np.all(_explained(embedding.coordinates, simulation.latent) > 0.45)


@pytest.mark.slow
def test_command_line(simulation, tmp_path, monkeypatch):
    monkeypatch.chdir(simulation.root)
    output = tmp_path / 'embedded.star'
    status = main(
        [
            'analyze',
            'particles.star',
            str(output),
            '--diameter',
            str(simulation.diameter),
            '--components',
            '3',
            '--synchronization',
            'burer-monteiro',
            '--scratch',
            str(tmp_path),
        ]
    )
    assert status == 0

    tables = starfile.read(output, always_dict=True)
    assert set(tables) == {'optics', 'particles'}
    coordinates = parse_embedding(tables['particles']['factoremEmbedding'])
    assert coordinates.shape == (len(simulation.latent), 3)
    assert (tables['particles']['factoremGroupCount'] > 0).all()
    assert np.all(_explained(coordinates, simulation.latent) > 0.3)


@pytest.mark.slow
def test_images_without_data_are_left_out(simulation, tmp_path, caplog):
    dataset = factorem.StarDataset.read(simulation.path, root=simulation.root)
    particles = dataset.particles
    blank = np.zeros((2, simulation.box_size, simulation.box_size))
    blank[1] = np.nan
    vio.write_stack(blank.astype(np.float32), str(tmp_path / 'blank.mrcs'))
    images = list(particles.images)
    images[7] = vio.ImageLocation(tmp_path / 'blank.mrcs', 0)
    images[1234] = vio.ImageLocation(tmp_path / 'blank.mrcs', 1)

    with caplog.at_level('WARNING'):
        embedding = factorem.analyze(
            replace(particles, images=images), diameter=simulation.diameter
        )
    assert '[7, 1234]' in caplog.text
    left_out = np.zeros(len(particles), dtype=bool)
    left_out[[7, 1234]] = True
    assert np.isnan(embedding.coordinates[left_out]).all()
    assert np.isfinite(embedding.coordinates[~left_out]).all()
    np.testing.assert_array_equal(embedding.group_count == 0, left_out)


@dataclass(frozen=True)
class _BrokenEmbedder:
    def __call__(
        self,
        x: jax.Array,
        c: jax.Array,
        shells: jax.Array,
        valid: jax.Array,
        components: int,
    ) -> tuple[jax.Array, jax.Array]:
        return (
            jnp.full((x.shape[0], components), jnp.nan),
            jnp.full((x.shape[1], components), jnp.nan),
        )


def test_embeddings_that_are_not_finite_are_reported(simulation):
    dataset = factorem.StarDataset.read(simulation.path, root=simulation.root)
    with pytest.raises(ValueError, match='not finite'):
        factorem.analyze(
            dataset.particles,
            diameter=simulation.diameter,
            embedder=_BrokenEmbedder(),
        )


def test_command_line_reports_bad_input(tmp_path, capsys):
    path = tmp_path / 'empty.star'
    starfile.write(
        {'particles': pd.DataFrame({'rlnAngleRot': [1.0, 2.0]})}, path
    )
    status = main(
        ['analyze', str(path), str(tmp_path / 'out.star'), '--diameter', '10']
    )
    assert status == 1
    assert 'factorem: error' in capsys.readouterr().err
