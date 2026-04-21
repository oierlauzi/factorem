import jax
import jax.numpy as jnp

def _apply_affine_single(image, matrix_inv):
    H, W = image.shape
    y, x = jnp.mgrid[0:H, 0:W]
    coords = jnp.stack([x.ravel(), y.ravel(), jnp.ones_like(x.ravel())])
    
    src_coords = matrix_inv @ coords
    src_x = src_coords[0, :] / src_coords[2, :]
    src_y = src_coords[1, :] / src_coords[2, :]
    
    sample_coords = jnp.stack([src_y.reshape(H, W), src_x.reshape(H, W)])
    transformed = jax.scipy.ndimage.map_coordinates(
        image, 
        sample_coords, 
        order=1, 
        mode='constant', 
        cval=0.0
    )
    
    return transformed

_apply_affine_batch = jax.vmap(_apply_affine_single, in_axes=(0, 0))

class ImageTransformer:
    def __init__(self, mask: jnp.ndarray):
        self._indices = jnp.nonzero(mask)
    
    def transform(
        self,
        images,
        transforms
    ):
        transformed_images = _apply_affine_batch(images, transforms)
        return transformed_images[self._indices]
