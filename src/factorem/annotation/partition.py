"""Partitions of the plane that the particles are annotated on.

The classes are the Gaussians of a mixture, and each point goes to the one
that most likely gave it. So the classes of a partition never overlap and
leave no point out.

A mixture is fitted to twenty thousand of the points at most, taken at
random, and all of them are then sorted by it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal, get_args

import numpy as np
from sklearn.mixture import GaussianMixture

logger = logging.getLogger(__name__)

Covariance = Literal['full', 'tied', 'diag', 'spherical']
"""What the covariances of the Gaussians of a mixture may be: any, one
for all of them, aligned with the axes, or a multiple of the identity."""

COVARIANCES: tuple[Covariance, ...] = get_args(Covariance)

# Points that a mixture is fitted to, at most. They leave a thousand for
# each of the Gaussians when these are many, which settles them, and
# fitting them to all the particles would keep whoever annotates waiting.
_FITTED_POINTS = 20_000

# Standard deviations that the outline of a class spans to each side of its
# centre. It holds 86 % of a Gaussian of the plane.
OUTLINE_DEVIATIONS = 2.0


@dataclass(frozen=True)
class Outline:
    """The ellipse that outlines a class."""

    centre: np.ndarray
    """``(2,)``."""
    width: float
    """Length of the first axis, from end to end."""
    height: float
    """Length of the second axis, from end to end."""
    angle: float
    """Angle from the abscissas to the first axis [degrees]."""


@dataclass(frozen=True)
class Partition:
    """The points of a plane sorted into classes."""

    labels: np.ndarray
    """The class of each point, ``(points,)``, counted from zero."""
    outlines: tuple[Outline, ...]
    """The outline of each class."""


def _outline(centre: np.ndarray, covariance: np.ndarray) -> Outline:
    values, vectors = np.linalg.eigh(covariance)
    values, vectors = values[::-1], vectors[:, ::-1]
    width, height = (
        2.0 * OUTLINE_DEVIATIONS * np.sqrt(np.maximum(values, 0.0))
    ).tolist()
    angle = np.degrees(np.arctan2(vectors[1, 0], vectors[0, 0]))
    return Outline(np.array(centre, dtype=np.float64), width, height, angle)


def _covariances(mixture: GaussianMixture) -> np.ndarray:
    """The covariance of each Gaussian, however the mixture keeps them."""
    kept = np.asarray(mixture.covariances_, dtype=np.float64)
    classes, dimensions = mixture.means_.shape
    if mixture.covariance_type == 'tied':
        return np.broadcast_to(kept, (classes, dimensions, dimensions))
    if mixture.covariance_type == 'diag':
        return np.stack([np.diag(diagonal) for diagonal in kept])
    if mixture.covariance_type == 'spherical':
        return kept[:, None, None] * np.eye(dimensions)
    return kept


def _fit(
    points: np.ndarray,
    classes: int,
    covariance: Covariance,
    centres: np.ndarray | None = None,
) -> tuple[Partition, float]:
    """A partition by a mixture, and the BIC of the mixture."""
    sample = points
    if len(points) > _FITTED_POINTS:
        chosen = np.random.default_rng(0).choice(
            len(points), _FITTED_POINTS, replace=False
        )
        sample = points[chosen]
    mixture = GaussianMixture(
        n_components=classes,
        covariance_type=covariance,
        means_init=centres,
        max_iter=200,
        random_state=0,
    ).fit(sample)
    bic = float(mixture.bic(sample))
    logger.info(
        'Fitted a mixture of %d Gaussians, with a BIC of %.5g', classes, bic
    )

    labels = np.asarray(mixture.predict(points), dtype=np.int64)
    means = np.asarray(mixture.means_, dtype=np.float64)
    covariances = _covariances(mixture)
    if centres is None:
        # The order a fit leaves them in means nothing.
        order = np.argsort(-np.bincount(labels, minlength=classes))
        labels = np.argsort(order)[labels]
        means, covariances = means[order], covariances[order]
    outlines = tuple(
        _outline(mean, covariance)
        for mean, covariance in zip(means, covariances, strict=True)
    )
    return Partition(labels, outlines), bic


def fit_mixture(
    points: np.ndarray,
    classes: int,
    *,
    covariance: Covariance = 'full',
    centres: np.ndarray | None = None,
) -> Partition:
    """Partition points by the Gaussian mixture that fits them best.

    Args:
        points: ``(points, 2)``.
        classes: How many Gaussians the mixture has.
        covariance: What their covariances may be.
        centres: Where to start each Gaussian from, ``(classes, 2)``.
            They start from a k-means of the points without it.

    Returns:
        The partition, each class being one of the Gaussians. The ones
        started from ``centres`` keep their order, and the others go by
        decreasing number of points.

    Raises:
        ValueError: If there are fewer points than classes.
    """
    points = np.asarray(points, dtype=np.float64)
    if classes < 1:
        raise ValueError('At least one class is needed.')
    if centres is not None:
        centres = np.asarray(centres, dtype=np.float64)
        if centres.shape != (classes, points.shape[1]):
            raise ValueError('Each class needs a centre to start from.')
    return _fit(points, classes, covariance, centres)[0]


def fit_mixture_by_bic(
    points: np.ndarray,
    max_classes: int,
    *,
    covariance: Covariance = 'full',
) -> Partition:
    """Partition points by a Gaussian mixture of as many classes as its
    Bayesian information criterion asks for.

    Args:
        points: ``(points, 2)``.
        max_classes: The most classes to consider.
        covariance: What the covariances of the Gaussians may be.

    Returns:
        The partition of least BIC among those of one class to
        ``max_classes`` of them, by decreasing number of points.
    """
    points = np.asarray(points, dtype=np.float64)
    if max_classes < 1:
        raise ValueError('At least one class is needed.')
    candidates = [
        _fit(points, classes, covariance)
        for classes in range(1, min(max_classes, len(points)) + 1)
    ]
    if not candidates:
        raise ValueError('There are no points to partition.')
    return min(candidates, key=lambda candidate: candidate[1])[0]


def assign_to_spheres(
    points: np.ndarray, centres: np.ndarray, deviations: np.ndarray
) -> Partition:
    """Partition points by round Gaussians that are given, not fitted.

    Each point goes to the Gaussian of highest density at it, all of them
    being as likely beforehand. A wide Gaussian therefore takes the points
    that are far from all of them.

    Args:
        points: ``(points, 2)``.
        centres: The centre of each Gaussian, ``(classes, 2)``.
        deviations: Their standard deviations, ``(classes,)``.

    Returns:
        The partition, each class being one of the Gaussians, in order.
    """
    points = np.asarray(points, dtype=np.float64)
    centres = np.asarray(centres, dtype=np.float64)
    deviations = np.asarray(deviations, dtype=np.float64)
    if centres.ndim != 2 or len(centres) < 1:
        raise ValueError('At least one class is needed.')
    if deviations.shape != (len(centres),):
        raise ValueError('Each class needs a standard deviation.')
    if not (deviations > 0.0).all():
        raise ValueError('The standard deviations must be positive.')

    variances = np.square(deviations)
    distances = np.square(points[:, None, :] - centres).sum(axis=2)
    log_density = -0.5 * distances / variances - 0.5 * points.shape[1] * np.log(
        variances
    )
    labels = np.argmax(log_density, axis=1).astype(np.int64)
    outlines = tuple(
        _outline(centre, variance * np.eye(points.shape[1]))
        for centre, variance in zip(centres, variances, strict=True)
    )
    return Partition(labels, outlines)
