import jax
import jax.numpy as jnp
import numpy as np
import sklearn.decomposition

from .data_loader import DataLoader
from .processor import Processor

class PCA(Processor):
    def __init__(
        self, 
        n_components: int
    ):
        self.pca = sklearn.decomposition.PCA(n_components=n_components)

    def embed(
        self, 
        loader: DataLoader, 
        indices: np.ndarray, 
        direction_matrix: np.ndarray
    ) -> jax.Array:
        images, ctfs = loader.load(indices, direction_matrix)
        ctfs2 = jnp.square(ctfs)
        wiener_factor = 0.1*np.mean(ctfs2, axis=(-1, -2), keepdims=True)
        wiener_corrected_images_ft = (images*ctfs) / (ctfs2 + wiener_factor)
        wiener_corrected_images = jnp.fft.irfft2(wiener_corrected_images_ft)
        x = wiener_corrected_images.reshape(len(wiener_corrected_images), -1)
        return self.pca.fit_transform(jax.device_get(x))
