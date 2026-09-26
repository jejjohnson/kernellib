"""Permutation tests."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

import kernellib as kl


K = kl.RBF(lengthscale=1.0)


def test_independence_test_rejects_dependence():
    X = jax.random.normal(jax.random.key(0), (80, 1))
    Y = jnp.cos(2.0 * X) + 0.1 * jax.random.normal(jax.random.key(1), (80, 1))
    res = kl.permutation_test(
        lambda X, Y: kl.hsic(K, K, X, Y),
        X,
        Y,
        key=jax.random.key(2),
        n_permutations=199,
    )
    assert res.p_value == pytest.approx(1 / 200)
    assert res.null_distribution.shape == (199,)
    assert jnp.all(res.null_distribution < res.statistic)


def test_independence_test_does_not_reject_independent_data():
    X = jax.random.normal(jax.random.key(0), (80, 1))
    Z = jax.random.normal(jax.random.key(1), (80, 1))
    res = kl.permutation_test(
        lambda X, Y: kl.hsic(K, K, X, Y), X, Z, key=jax.random.key(2)
    )
    # Pinned keys; under the null the p-value is uniform, and this draw is
    # far from the rejection region.
    assert res.p_value > 0.05


def test_two_sample_test():
    X = jax.random.normal(jax.random.key(0), (50, 2))
    Y = jax.random.normal(jax.random.key(1), (40, 2)) + 0.8
    stat = lambda X, Y: kl.mmd_squared(K, X, Y)
    res = kl.permutation_test(
        stat, X, Y, key=jax.random.key(2), n_permutations=99, kind="two_sample"
    )
    assert res.p_value == pytest.approx(0.01)
    same = kl.permutation_test(
        stat,
        X,
        jax.random.normal(jax.random.key(3), (40, 2)),
        key=jax.random.key(2),
        kind="two_sample",
    )
    assert same.p_value > 0.05


@pytest.mark.slow
def test_p_values_are_calibrated_under_the_null():
    # Exactness: under H0, P(p <= a) <= a. Over 100 independent null data
    # sets the rejection count at a = 0.1 is Binomial(100, 0.1): mean 10,
    # sd 3, so more than 10 + 7 * 3 = 31 rejections would be a real defect.
    def one(key):
        k1, k2, k3 = jax.random.split(key, 3)
        X = jax.random.normal(k1, (30, 1))
        Z = jax.random.normal(k2, (30, 1))
        return kl.permutation_test(
            lambda X, Y: kl.hsic(K, K, X, Y), X, Z, key=k3, n_permutations=99
        ).p_value

    p = jax.lax.map(one, jax.random.split(jax.random.key(0), 100))
    assert jnp.sum(p <= 0.1) <= 31


def test_errors():
    X = jnp.zeros((5, 1))
    stat = lambda X, Y: jnp.sum(X * Y)
    with pytest.raises(ValueError, match="kind"):
        kl.permutation_test(stat, X, X, key=jax.random.key(0), kind="paired")
    with pytest.raises(ValueError, match="n_permutations"):
        kl.permutation_test(stat, X, X, key=jax.random.key(0), n_permutations=0)
    with pytest.raises(ValueError, match="paired samples"):
        kl.permutation_test(stat, X, X[:3], key=jax.random.key(0))
