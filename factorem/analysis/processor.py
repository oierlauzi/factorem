from typing import Any
import numpy as np
import jax

from .data_loader import DataLoader
from .preprocessor import Preprocessor

Prepared = Any
"""Opaque, processor-specific host-side payload returned by ``prepare``.

Holds only host (numpy) state so it can be produced from a worker thread.
"""


class Processor:
    def fit_transform(
        self,
        images: jax.Array,
        ctfs: jax.Array,
        count: int
    ) -> jax.Array:
        raise NotImplementedError
