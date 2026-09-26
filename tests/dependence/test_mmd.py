"""Maximum mean discrepancy on kernels and data."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

import kernellib as kl
from kernellib import functional as F


K = kl.RBF(lengthscale=1.0)


def _samples(m=40, n=30, shift=0.5, key=0):
    k1, k2 = jax.random.split(jax.random.key(key))
    return jax.random.normal(k1, (m, 2)), jax.random.normal(k2, (n, 2)) + shift


def test_biased_matches_functional():
    X, Y = _samples()
    expected = F.mmd_squared(K(X, X), K(Y, Y), K(X, Y))
    assert jnp.allclose(kl.mmd_squared(K, X, Y), expected)


def test_unbiased_formula():
    X, Y = _samples()
    Kxx, Kyy, Kxy = K(X, X), K(Y, Y), K(X, Y)
    off = lambda M: (jnp.sum(M) - jnp.trace(M)) / (M.shape[0] * (M.shape[0] - 1))
    expected = off(Kxx) + off(Kyy) - 2.0 * jnp.mean(Kxy)
    assert jnp.allclose(kl.mmd_squared(K, X, Y, estimator="unbiased"), expected)


def test_linear_formula_and_gram_only_kernels():
    X, Y = _samples(m=8, n=8)
    h = [
        K(X[i : i + 1], X[i + 1 : i + 2])
        + K(Y[i : i + 1], Y[i + 1 : i + 2])
        - K(X[i : i + 1], Y[i + 1 : i + 2])
        - K(X[i + 1 : i + 2], Y[i : i + 1])
        for i in range(0, 8, 2)
    ]
    expected = jnp.mean(jnp.stack(h))
    assert jnp.allclose(kl.mmd_squared(K, X, Y, estimator="linear"), expected)
    # A sum kernel is still a valid kernel for the linear estimator.
    assert jnp.isfinite(kl.mmd_squared(K + kl.Linear(), X, Y, estimator="linear"))


@pytest.mark.parametrize("estimator", ["biased", "unbiased"])
def test_features_are_exact_for_their_gram(estimator):
    X, Y = _samples()
    approx = kl.RandomFourierFeatures(32, jax.random.key(0))
    fitted = approx.fit(K, jnp.concatenate([X, Y]))
    Px, Py = fitted(X), fitted(Y)
    lin = kl.Linear(variance=1.0, bias=0.0)
    # MMD of the linear kernel on the features is MMD of Phi Phi^T.
    expected = kl.mmd_squared(lin, Px, Py, estimator=estimator)
    got = kl.mmd_squared(K, X, Y, estimator=estimator, approx=approx)
    assert jnp.allclose(got, expected)


def test_separates_shifted_samples():
    X, Y = _samples(m=100, n=100, shift=1.0)
    X2, _ = _samples(m=100, n=100, key=1)
    assert kl.mmd_squared(K, X, Y) > 10 * kl.mmd_squared(K, X, X2)
    assert jnp.abs(kl.mmd_squared(K, X, X2, estimator="unbiased")) < 0.02


def test_errors():
    X, Y = _samples()
    with pytest.raises(ValueError, match="estimator"):
        kl.mmd_squared(K, X, Y, estimator="quadratic")
    with pytest.raises(ValueError, match="equal sample sizes"):
        kl.mmd_squared(K, X, Y, estimator="linear")
    with pytest.raises(ValueError, match="does not take approx"):
        kl.mmd_squared(
            K,
            X,
            X,
            estimator="linear",
            approx=kl.RandomFourierFeatures(4, jax.random.key(0)),
        )
    with pytest.raises(ValueError, match="two points"):
        kl.mmd_squared(K, X[:1], Y, estimator="unbiased")
