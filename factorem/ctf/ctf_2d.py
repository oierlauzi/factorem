from functools import partial
import jax
import jax.numpy as jnp

from .ctf_context import CtfContext

import matplotlib.pyplot as plt

def _frequency2_grid_2d(box_size: int, pixel_size: float):
    kx = jnp.fft.rfftfreq(box_size, d=pixel_size)
    ky = jnp.fft.fftfreq(box_size, d=pixel_size)
    return jnp.square(kx[None,:]) + jnp.square(ky[:,None])

@partial(jax.jit, static_argnames=('box_size', "context"))
def compute_ctf_image_2d(
    defocus_a: jnp.ndarray,
    box_size: int,
    context: CtfContext
):
    k2 = _frequency2_grid_2d(
        box_size=box_size, 
        pixel_size=context.pixel_size_a
    )
    
    wavelength = context.wavelength_a
    wavelength2 = wavelength*wavelength
    spherical_aberration = context.spherical_aberration_a
    q0 = context.q0
    
    angle = jnp.pi*wavelength*k2*(0.5*spherical_aberration*wavelength2*k2 + defocus_a[...,None,None])
    return jnp.sin(angle) - q0*jnp.cos(angle)
