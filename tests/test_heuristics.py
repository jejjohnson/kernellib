"""Bandwidth heuristics, reimplemented from pysim."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import kernellib as kl


def _data(n=40, d=3, key=0):
    scales = jnp.array([0.5, 2.0, 5.0])[:d]
    return jax.random.normal(jax.random.key(key), (n, d)) * scales


def _pdist(X):
    X = np.asarray(X)
    D = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1))
    return D[np.triu_indices(len(X), k=1)], D


@pytest.mark.parametrize(("method", "agg"), [("median", np.median), ("mean", np.mean)])
def test_distance_methods_match_numpy(method, agg):
    X = _data()
    pairs, _ = _pdist(X)
    got = kl.estimate_lengthscale(X, method)
    assert np.isclose(float(got), agg(pairs))


@pytest.mark.parametrize(("method", "agg"), [("median", np.median), ("mean", np.mean)])
def test_percent_uses_the_kth_neighbour(method, agg):
    X = _data(n=20)
    _, D = _pdist(X)
    k = int(0.25 * 20)  # 5th neighbour; column 0 is the point itself
    expected = agg(np.sort(D, axis=1)[:, k])
    assert np.isclose(float(kl.estimate_lengthscale(X, method, percent=0.25)), expected)


def test_percent_counts_k_from_the_subsample():
    # pysim took k from the full n, which can run past the subsample.
    X = _data(n=100)
    got = kl.estimate_lengthscale(X, percent=0.5, subsample=10, key=jax.random.key(0))
    assert jnp.isfinite(got)


def test_percent_one_is_the_farthest_neighbour():
    X = jnp.array([[0.0], [1.0], [3.0]])
    # Farthest neighbours: 3, 2, 3.
    assert float(kl.estimate_lengthscale(X, "mean", percent=1.0)) == pytest.approx(
        8.0 / 3.0
    )


@pytest.mark.parametrize("method", ["silverman", "scott"])
def test_rules_scale_with_the_data(method):
    # pysim's rules ignored the data scale; these are equivariant.
    X = _data()
    base = kl.estimate_lengthscale(X, method)
    assert jnp.allclose(kl.estimate_lengthscale(10.0 * X, method), 10.0 * base)


def test_silverman_and_scott_formulas():
    X = _data(n=50, d=2)
    sigma = np.mean(np.std(np.asarray(X), axis=0, ddof=1))
    silverman = sigma * (50 * 4 / 4) ** (-1 / 6)
    scott = sigma * 50 ** (-1 / 6)
    assert np.isclose(float(kl.estimate_lengthscale(X, "silverman")), silverman)
    assert np.isclose(float(kl.estimate_lengthscale(X, "scott")), scott)


@pytest.mark.parametrize("method", ["median", "mean", "silverman", "scott"])
def test_ard_is_per_dimension(method):
    X = _data()
    ard = kl.estimate_lengthscale(X, method, ard=True)
    assert ard.shape == (3,)
    for d in range(3):
        assert jnp.allclose(ard[d], kl.estimate_lengthscale(X[:, d : d + 1], method))
    # The dimensions have scales 0.5, 2, 5.
    assert ard[0] < ard[1] < ard[2]


def test_scale_and_translation():
    X = _data()
    base = kl.estimate_lengthscale(X)
    assert jnp.allclose(kl.estimate_lengthscale(X, scale=3.0), 3.0 * base)
    assert jnp.allclose(kl.estimate_lengthscale(X + 7.0), base)


def test_subsample_is_deterministic_in_key_and_a_noop_when_large():
    X = _data(n=60)
    a = kl.estimate_lengthscale(X, subsample=20, key=jax.random.key(1))
    b = kl.estimate_lengthscale(X, subsample=20, key=jax.random.key(1))
    assert jnp.array_equal(a, b)
    assert jnp.allclose(
        kl.estimate_lengthscale(X, subsample=100), kl.estimate_lengthscale(X)
    )


def test_gives_a_sensible_rbf():
    # The median heuristic puts typical pairs in the kernel's sensitive range.
    X = _data()
    k = kl.RBF(lengthscale=kl.estimate_lengthscale(X))
    K = k(X, X)
    off = K[jnp.triu_indices(40, k=1)]
    assert jnp.isclose(jnp.median(off), jnp.exp(-0.5), atol=1e-6)


def test_jit():
    X = _data()
    f = jax.jit(kl.estimate_lengthscale, static_argnames=("method",))
    assert jnp.allclose(f(X, "median"), kl.estimate_lengthscale(X))


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"method": "iqr"}, "method"),
        ({"percent": 0.0}, "percent"),
        ({"percent": 1.5}, "percent"),
        ({"subsample": 5}, "PRNG key"),
    ],
)
def test_invalid_arguments(kwargs, match):
    with pytest.raises(ValueError, match=match):
        kl.estimate_lengthscale(_data(), **kwargs)


def test_needs_two_points():
    with pytest.raises(ValueError, match="two points"):
        kl.estimate_lengthscale(jnp.ones((1, 2)))


def test_gamma_round_trip():
    ell = jnp.array([0.1, 1.0, 3.0])
    assert jnp.allclose(kl.gamma_to_lengthscale(kl.lengthscale_to_gamma(ell)), ell)
    # exp(-gamma r^2) == exp(-r^2 / (2 l^2)).
    assert jnp.allclose(kl.lengthscale_to_gamma(1.0), 0.5)


def test_lengthscale_grid():
    grid = kl.lengthscale_grid(2.0, decades=2.0, n_points=5)
    assert jnp.allclose(grid, 2.0 * jnp.array([0.01, 0.1, 1.0, 10.0, 100.0]))
    with pytest.raises(ValueError, match="n_points"):
        kl.lengthscale_grid(1.0, n_points=0)
