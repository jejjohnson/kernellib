"""Kernel Taylor-diagram statistics.

The references are the centred Gram matrices themselves (Frobenius norms and
differences) and, for linear kernels on 1-D data, the classic variance and
Pearson correlation. Inputs use pinned keys: the randomness is incidental.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import kernellib as kl


def _pair(n=40, key=0):
    kx, ky = jax.random.split(jax.random.key(key))
    X = jax.random.normal(kx, (n, 2))
    Y = jnp.tanh(X) + 0.2 * jax.random.normal(ky, (n, 2))
    return X, Y


def _centred(K):
    K = np.asarray(K)
    return K - K.mean(0, keepdims=True) - K.mean(1, keepdims=True) + K.mean()


def _cosine_law(s):
    return s.norm_x**2 + s.norm_y**2 - 2 * s.norm_x * s.norm_y * s.correlation


@pytest.mark.parametrize("kernel", [kl.RBF(0.8), kl.Distance(), kl.Linear()])
def test_matches_centred_gram_geometry(kernel):
    X, Y = _pair()
    n = X.shape[0]
    Kx, Ky = _centred(kernel(X, X)), _centred(kernel(Y, Y))
    s = kl.taylor_statistics(kernel, kernel, X, Y)
    assert np.allclose(s.norm_x, np.linalg.norm(Kx) / n)
    assert np.allclose(s.norm_y, np.linalg.norm(Ky) / n)
    assert np.allclose(s.distance, np.linalg.norm(Kx - Ky) / n)
    assert np.allclose(s.correlation, kl.cka(kernel, kernel, X, Y))


@pytest.mark.parametrize("estimator", ["biased", "unbiased"])
def test_law_of_cosines(estimator):
    X, Y = _pair()
    k = kl.RBF(0.8)
    s = kl.taylor_statistics(k, k, X, Y, estimator=estimator)
    assert jnp.isclose(s.distance**2, _cosine_law(s), rtol=1e-9)


@pytest.mark.slow
def test_law_of_cosines_under_approx():
    X, Y = _pair(n=200)
    k = kl.RBF(0.8)
    approx = kl.RandomFourierFeatures(64, jax.random.key(1))
    s = kl.taylor_statistics(k, k, X, Y, approx=approx)
    assert jnp.isclose(s.distance**2, _cosine_law(s), rtol=1e-9)
    assert jnp.isclose(s.correlation, kl.cka(k, k, X, Y, approx=approx))


def test_linear_kernel_in_one_dimension_is_the_squared_classic_diagram():
    x = jax.random.normal(jax.random.key(0), (50, 1))
    y = 0.7 * x + 0.5 * jax.random.normal(jax.random.key(1), (50, 1))
    s = kl.taylor_statistics(kl.Linear(), kl.Linear(), x, y)
    assert np.allclose(s.norm_x, np.var(x))
    assert np.allclose(s.norm_y, np.var(y))
    assert np.allclose(s.correlation, np.corrcoef(x[:, 0], y[:, 0])[0, 1] ** 2)


def test_identical_inputs_sit_on_the_reference():
    X, _ = _pair()
    k = kl.RBF()
    s = kl.taylor_statistics(k, k, X, X)
    assert jnp.isclose(s.correlation, 1.0)
    assert jnp.isclose(s.norm_x, s.norm_y)
    assert float(s.distance) < 1e-6


def test_jit_and_errors():
    X, Y = _pair()
    k = kl.RBF()
    s = eqx.filter_jit(kl.taylor_statistics)(k, k, X, Y)
    assert jnp.isclose(s.distance, kl.taylor_statistics(k, k, X, Y).distance)
    with pytest.raises(ValueError, match="paired"):
        kl.taylor_statistics(k, k, X, Y[:-1])
    with pytest.raises(ValueError, match="estimator"):
        kl.taylor_statistics(k, k, X, Y, estimator="nope")  # type: ignore[arg-type]
