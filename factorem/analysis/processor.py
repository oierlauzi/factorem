import jax


class Processor:
    """Embeds the particles of one direction group.

    ``images`` and ``ctfs`` may hold padding rows past ``count``, which must
    not affect the result. Returns an array of shape ``(count, components)``.
    """

    def fit_transform(
        self,
        images: jax.Array,
        ctfs: jax.Array,
        count: int
    ) -> jax.Array:
        raise NotImplementedError
