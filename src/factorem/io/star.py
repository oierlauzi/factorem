"""Particles of RELION STAR files.

Both layouts of RELION are read: the one since 3.1, with a table of optics
groups and translations in angstroms, and the older one, with everything in
a single table, translations in pixels and the pixel size given by the
magnification and the pixel size of the detector.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import starfile
import vitrio.data as vio_data

from ..geometry import euler_to_matrix
from ..particles import CtfParameters, Particles

EMBEDDING_LABEL = 'factoremEmbedding'
GROUP_COUNT_LABEL = 'factoremGroupCount'

_OPTICS_BLOCK = 'optics'
_OPTICS_GROUP = 'rlnOpticsGroup'


class StarFormatError(ValueError):
    """A STAR file lacks what the analysis needs, or contradicts itself."""


def _particle_block(tables: dict[str, pd.DataFrame]) -> str:
    if 'particles' in tables:
        return 'particles'
    candidates = [
        name for name, table in tables.items() if 'rlnImageName' in table
    ]
    if len(candidates) != 1:
        raise StarFormatError(
            'The STAR file needs one table of particles, with their '
            f'rlnImageName. Its tables are {list(tables)}.'
        )
    return candidates[0]


class _Columns:
    """The value of a label for each particle, whether it is written for
    each particle or for its optics group."""

    def __init__(self, table: pd.DataFrame, optics: pd.DataFrame | None):
        self._table = table
        self._optics = optics

    def __contains__(self, label: str) -> bool:
        return label in self._table or (
            self._optics is not None and label in self._optics
        )

    def _from_optics(self, label: str) -> np.ndarray:
        assert self._optics is not None
        if _OPTICS_GROUP in self._table and _OPTICS_GROUP in self._optics:
            by_group = self._optics.set_index(_OPTICS_GROUP)[label]
            values = by_group.reindex(self._table[_OPTICS_GROUP]).to_numpy()
            if pd.isna(values).any():
                raise StarFormatError(
                    'Some particles are of an optics group that the table '
                    'of optics lacks.'
                )
            return np.asarray(values)
        if len(self._optics) == 1:
            return np.repeat(self._optics[label].to_numpy(), len(self._table))
        raise StarFormatError(
            f'The particles need their {_OPTICS_GROUP} when there are '
            'several optics groups.'
        )

    def get(self, label: str, default: float | None = None) -> np.ndarray:
        if self._optics is not None and label in self._optics:
            values = self._from_optics(label)
        elif label in self._table:
            values = self._table[label].to_numpy()
        elif default is not None:
            return np.full(len(self._table), default, dtype=np.float64)
        else:
            raise StarFormatError(f'The STAR file lacks {label}.')
        return np.asarray(values, dtype=np.float64)


def _pixel_size(columns: _Columns) -> float:
    if 'rlnImagePixelSize' in columns:
        values = columns.get('rlnImagePixelSize')
    elif 'rlnDetectorPixelSize' in columns and 'rlnMagnification' in columns:
        values = (
            1e4
            * columns.get('rlnDetectorPixelSize')
            / columns.get('rlnMagnification')
        )
    else:
        raise StarFormatError(
            'The STAR file states no pixel size: neither rlnImagePixelSize '
            'nor rlnDetectorPixelSize with rlnMagnification.'
        )
    pixel_size = float(np.median(values))
    if np.ptp(values) > 1e-3 * pixel_size:
        raise StarFormatError(
            f'The particles have pixel sizes from {values.min():.4f} to '
            f'{values.max():.4f} A. Rescale them to a common one first.'
        )
    return pixel_size


def _shifts(columns: _Columns, pixel_size: float) -> np.ndarray:
    if 'rlnOriginXAngst' in columns:
        labels, scale = ('rlnOriginXAngst', 'rlnOriginYAngst'), 1.0 / pixel_size
    else:
        labels, scale = ('rlnOriginX', 'rlnOriginY'), 1.0
    return scale * np.stack(
        [columns.get(label, default=0.0) for label in labels], axis=1
    )


def particles_from_tables(
    tables: dict[str, pd.DataFrame],
    root: str | os.PathLike[str] | None = None,
) -> Particles:
    """Describe the particles of the tables of a STAR file.

    Args:
        tables: The tables by name, as ``starfile.read`` gives them with
            ``always_dict=True``.
        root: The directory the images are relative to, which RELION
            takes to be that of the project. Defaults to the working
            directory.

    Raises:
        StarFormatError: If the tables lack something the analysis needs.
    """
    table = tables[_particle_block(tables)]
    columns = _Columns(table, tables.get(_OPTICS_BLOCK))
    pixel_size = _pixel_size(columns)

    def angle(label: str, default: float | None = None) -> np.ndarray:
        radians: np.ndarray = np.deg2rad(columns.get(label, default))
        return radians

    return Particles(
        images=vio_data.parse_locations(table['rlnImageName'], root=root),
        rotations=euler_to_matrix(
            angle('rlnAngleRot'), angle('rlnAngleTilt'), angle('rlnAnglePsi')
        ),
        shifts=_shifts(columns, pixel_size),
        ctf=CtfParameters(
            defocus_u=columns.get('rlnDefocusU'),
            defocus_v=columns.get('rlnDefocusV'),
            defocus_angle=angle('rlnDefocusAngle', 0.0),
            phase_shift=angle('rlnPhaseShift', 0.0),
            voltage=columns.get('rlnVoltage'),
            spherical_aberration=columns.get('rlnSphericalAberration'),
            amplitude_contrast=columns.get('rlnAmplitudeContrast'),
        ),
        pixel_size=pixel_size,
    )


def format_embedding(coordinates: np.ndarray) -> list[str]:
    """Write the coordinates of each particle as a list, the way the
    ``factoremEmbedding`` column holds them."""
    return [
        '[' + ', '.join(f'{value:.4e}' for value in row) + ']'
        for row in np.asarray(coordinates)
    ]


def parse_embedding(column: Any) -> np.ndarray:
    """Read the coordinates back from a ``factoremEmbedding`` column."""
    return np.array(
        [
            [float(value) for value in str(text).strip('[]').split(',')]
            for text in column
        ]
    )


@dataclass(frozen=True)
class StarDataset:
    """A STAR file of particles: its tables, and what they say about the
    particles."""

    tables: dict[str, pd.DataFrame]
    particles: Particles

    @classmethod
    def read(
        cls,
        path: str | os.PathLike[str],
        root: str | os.PathLike[str] | None = None,
    ) -> StarDataset:
        """Read a STAR file. See `particles_from_tables`."""
        tables = starfile.read(path, always_dict=True)
        return cls(tables, particles_from_tables(tables, root))

    def write(
        self,
        path: str | os.PathLike[str],
        coordinates: np.ndarray,
        group_count: np.ndarray | None = None,
    ) -> None:
        """Write the tables with the coordinates of each particle added.

        Args:
            path: The STAR file to write.
            coordinates: Coordinates of each particle,
                ``(particles, components)``. They go in the
                ``factoremEmbedding`` column, as a list for each particle.
            group_count: How many projection directions each particle was
                analysed with, for the ``factoremGroupCount`` column.
        """
        block = _particle_block(self.tables)
        table = self.tables[block].copy()
        if len(coordinates) != len(table):
            raise ValueError(
                f'There are coordinates for {len(coordinates)} particles, '
                f'and {len(table)} particles.'
            )
        table[EMBEDDING_LABEL] = format_embedding(coordinates)
        if group_count is not None:
            table[GROUP_COUNT_LABEL] = np.asarray(group_count)
        starfile.write({**self.tables, block: table}, path)
