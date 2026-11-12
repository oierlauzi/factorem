"""Choosing where JAX computes."""

from __future__ import annotations

import jax


def resolve_device(device: str | jax.Device | None = None) -> jax.Device:
    """Find a device of JAX by name.

    Args:
        device: ``'cpu'``, ``'gpu'`` or, for a given one, ``'gpu:1'``. The
            default device of JAX when none is named.

    Raises:
        ValueError: If there is no such device.
    """
    if device is None:
        return jax.devices()[0]
    if not isinstance(device, str):
        return device

    kind, _, number = device.lower().partition(':')
    try:
        index = int(number) if number else 0
        backend = {'cuda': 'gpu'}.get(kind, kind)
        return jax.devices(backend)[index]
    except (RuntimeError, IndexError, ValueError) as error:
        raise ValueError(
            f'There is no device {device!r} for JAX to compute on.'
        ) from error
