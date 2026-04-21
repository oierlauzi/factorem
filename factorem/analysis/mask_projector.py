import jax
import jax.numpy as jnp
from jax.scipy.ndimage import map_coordinates

def project_mask(mask, rotation):
    shape = mask.shape
    z, y, x = jnp.meshgrid(
        jnp.arange(shape[0]),
        jnp.arange(shape[1]),
        jnp.arange(shape[2]),
        indexing='ij'
    )
    
    coords = jnp.stack([z.flatten(), y.flatten(), x.flatten()])
    center = jnp.array([(s - 1) / 2.0 for s in shape])[:, None]
    coords_centered = coords - center
    
    rotated_coords_centered = rotation.T @ coords_centered
    rotated_coords = rotated_coords_centered + center
    rotated_coords = rotated_coords.reshape(3, *shape)
    rotated_mask = map_coordinates(
        mask, 
        rotated_coords, 
        order=1, 
        mode='constant', 
        cval=0
    )
    
    return rotated_mask.sum(dim=0)
    