from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from factorem.embedding import ProbabilisticPCA


def _simulate(count=600, features=80, components=3, seed=0):
    rng = np.random.default_rng(seed)
    loadings = rng.normal(size=(features, components)) * (4.0, 3.0, 2.0)
    z = rng.normal(size=(count, components))
    # A CTF of each image that crosses zero at different features.
    position = np.linspace(0.0, 1.0, features)
    c = np.sin(rng.uniform(3.0, 12.0, (count, 1)) * position + 0.3)
    x = c * (z @ loadings.T) + rng.normal(size=(count, features))
    return (
        jnp.asarray(x, dtype=jnp.float32),
        jnp.asarray(c, dtype=jnp.float32),
        z,
        loadings,
    )


def _subspace_cosines(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    qa, _ = np.linalg.qr(a)
    qb, _ = np.linalg.qr(b)
    return np.linalg.svd(qa.T @ qb, compute_uv=False)


def test_ppca_recovers_the_model():
    x, c, z, loadings = _simulate()
    valid = jnp.ones(len(x))
    y, w = ProbabilisticPCA()(x, c, valid, 3)

    assert y.shape == (len(x), 3)
    assert w.shape == loadings.shape
    assert _subspace_cosines(np.asarray(w), loadings).min() > 0.98
    assert _subspace_cosines(np.asarray(y), z).min() > 0.95

    # Uncorrelated, by decreasing variance, and on the scale of the prior.
    moments = np.asarray(y).T @ np.asarray(y) / len(y)
    variances = np.diag(moments)
    assert np.all(np.diff(variances) < 0.0)
    assert np.abs(moments - np.diag(variances)).max() < 1e-3
    assert variances.min() > 0.3 and variances.max() < 3.0


def test_ppca_ignores_the_filling():
    x, c, _, _ = _simulate()
    embedder = ProbabilisticPCA()
    y, w = embedder(x, c, jnp.ones(len(x)), 3)

    filling = 40
    valid = jnp.concatenate((jnp.ones(len(x)), jnp.zeros(filling)))
    x_filled = jnp.concatenate((x, 100.0 * x[:filling]))
    c_filled = jnp.concatenate((c, c[:filling]))
    y_filled, w_filled = embedder(x_filled, c_filled, valid, 3)

    signs = np.sign(np.sum(np.asarray(w) * np.asarray(w_filled), axis=0))
    np.testing.assert_allclose(w_filled * signs, w, atol=2e-2)
    np.testing.assert_allclose(y_filled[: len(x)] * signs, y, atol=2e-2)


def test_ppca_compiles_as_a_static_argument():
    @partial(jax.jit, static_argnames='embedder')
    def embed(x, c, valid, embedder: ProbabilisticPCA):
        return embedder(x, c, valid, 2)

    x, c, _, _ = _simulate(count=100, features=30)
    y, w = embed(
        x, c, jnp.ones(len(x)), embedder=ProbabilisticPCA(iterations=5)
    )
    assert y.shape == (100, 2)
    assert w.shape == (30, 2)
    assert hash(ProbabilisticPCA()) == hash(ProbabilisticPCA())
