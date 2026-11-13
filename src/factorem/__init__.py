"""Continuous heterogeneity analysis of cryo-EM particles.

`analyze` does the whole analysis, and the subpackages hold its stages for
when they are to be combined in some other way::

    import factorem

    dataset = factorem.StarDataset.read('particles.star')
    embedding = factorem.analyze(dataset.particles, diameter=180.0)
    dataset.write('embedded.star', embedding.coordinates)
"""

from importlib import metadata

from .embedding import (
    DirectionalEmbedding,
    Embedder,
    MahalanobisMDS,
    ProbabilisticPCA,
    embed_directions,
)
from .geometry import (
    DirectionGroups,
    euler_to_matrix,
    group_by_direction,
    sample_directions,
)
from .io import ParticleImages, StarDataset, StarFormatError, open_images
from .merging import merge_directions, principal_components
from .particles import CtfParameters, Particles
from .pipeline import Design, Embedding, analyze
from .synchronization import (
    BurerMonteiroSynchronizer,
    CommonLineSynchronizer,
    Synchronizer,
)

try:
    __version__ = metadata.version('factorem')
except metadata.PackageNotFoundError:  # Not installed: run from the sources.
    __version__ = '0+unknown'

__all__ = [
    'BurerMonteiroSynchronizer',
    'CommonLineSynchronizer',
    'CtfParameters',
    'Design',
    'DirectionGroups',
    'DirectionalEmbedding',
    'Embedder',
    'Embedding',
    'MahalanobisMDS',
    'ParticleImages',
    'Particles',
    'ProbabilisticPCA',
    'StarDataset',
    'StarFormatError',
    'Synchronizer',
    'analyze',
    'embed_directions',
    'euler_to_matrix',
    'group_by_direction',
    'merge_directions',
    'open_images',
    'principal_components',
    'sample_directions',
]
