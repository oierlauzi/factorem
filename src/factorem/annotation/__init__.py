"""Looking at the coordinates of the particles and sorting them into
classes, by hand.

It needs matplotlib and scikit-learn, which the ``annotate`` extra brings,
so nothing else of the package imports it.
"""

from .partition import (
    COVARIANCES,
    Covariance,
    Outline,
    Partition,
    assign_to_spheres,
    fit_mixture,
    fit_mixture_by_bic,
)
from .projection import sight, umap_projection
from .window import MAX_CLASSES, AnnotationWindow

__all__ = [
    'COVARIANCES',
    'MAX_CLASSES',
    'AnnotationWindow',
    'Covariance',
    'Outline',
    'Partition',
    'assign_to_spheres',
    'fit_mixture',
    'fit_mixture_by_bic',
    'sight',
    'umap_projection',
]
