"""The contrast transfer function of the microscope."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike


def electron_wavelength(voltage: ArrayLike) -> jax.Array:
    """Relativistic wavelength [A] of electrons accelerated by a voltage
    [kV]."""
    volts = 1e3 * jnp.asarray(voltage)
    return 12.2643247 / jnp.sqrt(volts * (1.0 + 0.978466e-6 * volts))


def ctf(
    qx: ArrayLike,
    qy: ArrayLike,
    defocus_u: ArrayLike,
    defocus_v: ArrayLike,
    defocus_angle: ArrayLike,
    phase_shift: ArrayLike,
    voltage: ArrayLike,
    spherical_aberration: ArrayLike,
    amplitude_contrast: ArrayLike,
) -> jax.Array:
    """Contrast transfer function at some frequencies, as RELION defines it.

    Args:
        qx: Frequency along the x axis of the image [1/A].
        qy: Frequency along the y axis of the image [1/A].
        defocus_u: Defocus along the major axis [A], positive for
            underfocus.
        defocus_v: Defocus along the minor axis [A].
        defocus_angle: Angle of the major axis from the x axis [rad].
        phase_shift: Phase shift of a phase plate [rad].
        voltage: Acceleration voltage [kV].
        spherical_aberration: Spherical aberration [mm].
        amplitude_contrast: Fraction of amplitude contrast.

    Returns:
        The values of the CTF, all the arguments broadcast together.
    """
    qx, qy = jnp.asarray(qx), jnp.asarray(qy)
    q2 = jnp.square(qx) + jnp.square(qy)
    # cos and sin of twice the angle of the frequency, without computing it.
    safe = jnp.where(q2 > 0.0, q2, 1.0)
    cos2 = (jnp.square(qx) - jnp.square(qy)) / safe
    sin2 = 2.0 * qx * qy / safe

    average = 0.5 * (jnp.asarray(defocus_u) + jnp.asarray(defocus_v))
    deviation = 0.5 * (jnp.asarray(defocus_u) - jnp.asarray(defocus_v))
    twice = 2.0 * jnp.asarray(defocus_angle)
    defocus = average + deviation * (
        cos2 * jnp.cos(twice) + sin2 * jnp.sin(twice)
    )

    wavelength = electron_wavelength(voltage)
    aberration = 1e7 * jnp.asarray(spherical_aberration)  # [A]
    contrast = jnp.asarray(amplitude_contrast)
    phase = (
        jnp.pi
        * wavelength
        * q2
        * (defocus - 0.5 * aberration * wavelength**2 * q2)
        + jnp.asarray(phase_shift)
        + jnp.arctan2(contrast, jnp.sqrt(1.0 - jnp.square(contrast)))
    )
    return jnp.sin(phase)
