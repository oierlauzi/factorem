"""The whole analysis, from particles to their coordinates."""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass

import jax
import numpy as np

from ._device import resolve_device
from .embedding import Embedder, ProbabilisticPCA, embed_directions
from .geometry import group_by_direction, sample_directions
from .io.images import load_images
from .merging import merge_directions, principal_components
from .particles import Particles
from .synchronization import CommonLineSynchronizer, Synchronizer

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Design:
    """The constants of the analysis.

    They are stated in units that follow the particle and the resolution,
    so the same values serve every dataset: none of them is meant to be
    tuned for one. The unit of angles is the Shannon angle, the ratio of
    the resolution to the diameter of the particle, which is how far the
    view can turn before the projection changes at that resolution.
    """

    oversampling: float = 4.0
    """Pixels of the working images in a period at the resolution: twice
    as many as Nyquist asks for, which spares the in-plane rotation from
    interpolating close to the limit."""
    direction_spacing: float = 1.0
    """Angle between neighbouring projection directions [Shannon angles]."""
    aperture: float = 2.0
    """Half-angle of the cone of views analysed with each direction
    [Shannon angles]."""
    view_order: int = 2
    """Order of the polynomial that models how the mean image changes with
    the view within a cone."""
    mask_scale: float = 1.25
    """Diameter of the circular mask over that of the particle."""
    min_components: int = 10
    """Fewest components to analyse each direction with, however few are
    asked for: the leading ones are better told apart with some more
    behind them."""
    images_per_component: int = 10
    """Fewest particles a direction may have, for each of its
    components."""


@dataclass(frozen=True)
class Embedding:
    """Coordinates of the particles in the space of their heterogeneity."""

    coordinates: np.ndarray
    """``(particles, components)``, by decreasing variance. They are NaN
    for the particles left out of the analysis."""
    group_count: np.ndarray
    """How many projection directions each particle was analysed with. It
    is zero for the ones left out."""
    variances: np.ndarray
    """Variance along every axis found, the ones of `coordinates` first."""


def analyze(
    particles: Particles,
    *,
    diameter: float,
    resolution: float = 16.0,
    components: int = 6,
    design: Design | None = None,
    embedder: Embedder | None = None,
    synchronizer: Synchronizer | None = None,
    device: str | jax.Device | None = None,
    scratch: str | os.PathLike[str] | None = None,
) -> Embedding:
    """Embed particles according to their heterogeneity.

    The particles are grouped by projection direction and each group is
    embedded on its own, as a set of 2D images. The embeddings of all the
    directions are then brought to a common frame and averaged.

    Args:
        particles: The particles, with the alignment and the CTF of a
            consensus refinement.
        diameter: Diameter of the particle [A].
        resolution: Resolution of the analysis [A].
        components: How many coordinates to give each particle.
        design: The constants of the analysis.
        embedder: What embeds the particles of a direction. Defaults to
            `ProbabilisticPCA`.
        synchronizer: What brings the directions to a common frame.
            Defaults to `CommonLineSynchronizer`.
        device: Where to compute, such as ``'cpu'`` or ``'gpu:0'``.
            Defaults to the default device of JAX.
        scratch: A directory to keep the working images in, instead of in
            memory.

    Returns:
        The coordinates of each particle.

    Raises:
        ValueError: If the arguments make no sense, or if no projection
            direction has particles enough to analyse.
    """
    if not diameter > 0.0:
        raise ValueError('The diameter must be positive.')
    if not resolution > 0.0:
        raise ValueError('The resolution must be positive.')
    if components < 1:
        raise ValueError('At least one component is needed.')
    design = design or Design()
    embedder = embedder or ProbabilisticPCA()
    synchronizer = synchronizer or CommonLineSynchronizer()
    device = resolve_device(device)
    start = time.monotonic()

    stack = load_images(
        particles,
        resolution / design.oversampling,
        device=device,
        scratch=scratch,
    )
    logger.info('Read the images in %.0f s', time.monotonic() - start)

    shannon_angle = resolution / diameter
    internal = max(components, design.min_components)
    groups = group_by_direction(
        particles.rotations,
        sample_directions(design.direction_spacing * shannon_angle),
        design.aperture * shannon_angle,
        min_size=design.images_per_component * internal,
        usable=stack.usable,
    )
    if len(groups) == 0:
        raise ValueError(
            'No projection direction has particles enough to analyse. '
            'There may be too few particles for this resolution.'
        )

    checkpoint = time.monotonic()
    directional = embed_directions(
        stack,
        particles,
        groups,
        resolution=resolution,
        mask_diameter=design.mask_scale * diameter,
        components=internal,
        embedder=embedder,
        view_order=design.view_order,
        device=device,
    )
    logger.info(
        'Embedded the directions in %.0f s', time.monotonic() - checkpoint
    )

    checkpoint = time.monotonic()
    transforms = synchronizer(directional)
    merged = merge_directions(directional, transforms, len(particles))
    coordinates, variances = principal_components(merged, components)
    logger.info(
        'Synchronized and merged the directions in %.0f s',
        time.monotonic() - checkpoint,
    )
    return Embedding(
        coordinates=coordinates,
        group_count=groups.multiplicity(len(particles)),
        variances=variances,
    )
