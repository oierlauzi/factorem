import numpy as np
import jax

from .data_loader import DataLoader

class Processor:
    def fit_transform(
        self, 
        loader: DataLoader, 
        indices: np.ndarray, 
        direction_matrix: np.ndarray
    ) -> jax.Array:
        pass