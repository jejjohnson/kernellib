"""Kernel Taylor-diagram statistics.

The references are the centred Gram matrices themselves (Frobenius norms and
differences) and, for linear kernels on 1-D data, the classic variance and
Pearson correlation. Inputs use pinned keys: the randomness is incidental.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
import pytest

import kernellib as kl
from kernellib import functional as F


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


def test_close_models_keep_their_distance_in_float32():
    # xx + yy - 2 xy rounds to zero here; the direct difference does not.
    X = jnp.array([[-1.0], [1.0]], dtype=jnp.float32)
    Y = 1.0002 * X
    s = kl.taylor_statistics(kl.Linear(), kl.Linear(), X, Y)
    Kx, Ky = _centred(X @ X.T), _centred(Y @ Y.T)
    expected = np.linalg.norm(Kx.astype(np.float64) - Ky.astype(np.float64)) / 2
    assert np.isclose(float(s.distance), expected, rtol=1e-3)


@pytest.mark.slow
@pytest.mark.parametrize("estimator", ["biased", "unbiased"])
def test_feature_distance_matches_its_gram(estimator):
    # Under approx the distance is exact for the features' own Gram matrices.
    X, Y = _pair(n=30)
    k = kl.RBF(0.8)
    approx = kl.RandomFourierFeatures(8, jax.random.key(3))
    kx_, ky_ = jax.random.split(jax.random.key(3))
    Px = kl.RandomFourierFeatures(8, kx_).fit(k, X)(X)
    Py = kl.RandomFourierFeatures(8, ky_).fit(k, Y)(Y)
    D = lx.MatrixLinearOperator(Px @ Px.T - Py @ Py.T, lx.symmetric_tag)
    expected = jnp.sqrt(jnp.maximum(F.hsic(D, D, estimator=estimator), 0.0))
    s = kl.taylor_statistics(k, k, X, Y, estimator=estimator, approx=approx)
    assert jnp.allclose(s.distance, expected, rtol=1e-8)


def test_distance_inputs_are_centred():
    kx, ky = jax.random.split(jax.random.key(0))
    X = jax.random.normal(kx, (50, 1), dtype=jnp.float32)
    Y = X**2 + 0.3 * jax.random.normal(ky, (50, 1), dtype=jnp.float32)
    k = kl.Distance()
    base = kl.taylor_statistics(k, k, X, Y)
    moved = kl.taylor_statistics(k, k, X + 1e3, Y - 1e3)
    for a, b in zip(base, moved, strict=True):
        assert jnp.allclose(a, b, rtol=2e-3)


@pytest.mark.slow
@pytest.mark.parametrize("estimator", ["biased", "unbiased"])
def test_gradient_is_finite_at_zero_distance(estimator):
    X, _ = _pair()
    k = kl.RBF()

    def distance(Y):
        return kl.taylor_statistics(k, k, X, Y, estimator=estimator).distance

    assert bool(jnp.all(jnp.isfinite(jax.grad(distance)(X))))
