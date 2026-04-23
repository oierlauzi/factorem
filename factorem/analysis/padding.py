import jax.numpy as jnp

def pad_images_2d(images: jnp.ndarray, padded_box_size: int):
    batch_shape = images.shape[:-2]
    original_box_size_y, original_box_size_x = images.shape[-2:]
    result_shape = batch_shape + (padded_box_size, padded_box_size)
    
    result = jnp.zeros(result_shape, dtype=images.dtype)
    return result.at[...,:original_box_size_y,:original_box_size_x].set(images)
