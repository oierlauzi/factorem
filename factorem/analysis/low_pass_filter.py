import jax.numpy as jnp

def _frequency2_grid_2d(box_size: int):
    kx = jnp.fft.rfftfreq(box_size)
    ky = jnp.fft.fftfreq(box_size)
    return jnp.square(kx[None,:]) + jnp.square(ky[:,None])

def butterworth_2d(box_size: int, cutoff: float, order: int):
    cutoff2 = cutoff*cutoff
    k2 = _frequency2_grid_2d(box_size)
    k2_cutoff2 = k2 / cutoff2
    
    if order == 1:
        term = k2_cutoff2
    elif order == 2:
        term = k2_cutoff2*k2_cutoff2
    else:
        term = jnp.pow(k2_cutoff2, order)
    return 1.0 / (1.0 + term)
