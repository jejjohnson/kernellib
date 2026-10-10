"""Tests for the empirical variogram and its fit (GEO7)."""

from __future__ import annotations

import itertools

import einx
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import kernellib as kl
from kernellib._einx import rearrange
from kernellib.functional import great_circle_distance


def _brute_force(X, y, edges, max_distance, estimator, dist):
    """Double loop over i < j, bins [e_b, e_{b+1}) with the last one closed."""
    n_bins = len(edges) - 1
    counts = np.zeros(n_bins)
    stat = np.zeros(n_bins)
    for i, j in itertools.combinations(range(len(y)), 2):
        d = dist[i, j]
        if d > max_distance or d < edges[0] or d > edges[-1]:
            continue
        b = n_bins - 1 if d == edges[-1] else np.searchsorted(edges, d, "right") - 1
        counts[b] += 1
        dy = abs(y[i] - y[j])
        stat[b] += dy**2 if estimator == "matheron" else np.sqrt(dy)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = stat / counts
        if estimator == "matheron":
            gamma = 0.5 * mean
        else:
            gamma = mean**4 / (2.0 * (0.457 + 0.494 / counts))
    return np.where(counts > 0, gamma, np.nan), counts


@pytest.mark.parametrize("estimator", ["matheron", "cressie"])
@pytest.mark.parametrize("metric", ["euclidean", "great_circle"])
def test_matches_brute_force(estimator, metric):
    k1, k2 = jr.split(jr.key(0))
    n = 60
    if metric == "euclidean":
        X = jr.uniform(k1, (n, 2))
        diff = einx.subtract("n d, m d -> n m d", np.asarray(X), np.asarray(X))
        dist = np.sqrt(einx.sum("n m d -> n m", diff**2))
    else:
        X = jr.uniform(k1, (n, 2), minval=-40.0, maxval=40.0)
        dist = np.asarray(great_circle_distance(X, X))
    y = jr.normal(k2, (n,))
    v = kl.empirical_variogram(
        X, y, 8, estimator=estimator, metric=metric, batch_size=16
    )
    max_distance = 0.5 * dist.max()
    np.testing.assert_allclose(v.bin_edges, np.linspace(0, max_distance, 9))
    gamma, counts = _brute_force(
        np.asarray(X), np.asarray(y), np.asarray(v.bin_edges), max_distance,
        estimator, dist,
    )  # fmt: skip
    np.testing.assert_array_equal(v.counts, counts)
    np.testing.assert_allclose(v.gamma, gamma, rtol=1e-10)


def test_explicit_edges_and_max_distance():
    X = jr.uniform(jr.key(1), (40, 1))
    y = jr.normal(jr.key(2), (40,))
    edges = jnp.array([0.0, 0.1, 0.3, 0.6])
    v = kl.empirical_variogram(X, y, edges, max_distance=0.5, batch_size=7)
    x = np.asarray(X)[:, 0]
    dist = np.abs(einx.subtract("n, m -> n m", x, x))
    gamma, counts = _brute_force(
        np.asarray(X), np.asarray(y), np.asarray(edges), 0.5, "matheron", dist
    )
    np.testing.assert_array_equal(v.counts, counts)
    np.testing.assert_allclose(v.gamma, gamma, rtol=1e-10)


def test_constant_and_shifted_fields():
    X = jr.uniform(jr.key(3), (50, 2))
    y = jr.normal(jr.key(4), (50,))
    for est in ("matheron", "cressie"):
        flat = kl.empirical_variogram(X, jnp.full(50, 3.0), 10, estimator=est)
        assert np.all(np.asarray(flat.gamma)[np.asarray(flat.counts) > 0] == 0.0)
        a = kl.empirical_variogram(X, y, 10, estimator=est)
        b = kl.empirical_variogram(X, y + 7.5, 10, estimator=est)
        np.testing.assert_allclose(a.gamma, b.gamma, rtol=1e-10)
        np.testing.assert_array_equal(a.counts, b.counts)


def test_jit_and_empty_bins_are_nan():
    X = jnp.concatenate([jnp.zeros((5, 1)), jnp.ones((5, 1))])  # gaps at 0 < d < 1
    y = jr.normal(jr.key(5), (10,))
    f = jax.jit(kl.empirical_variogram, static_argnames=("bins",))
    v = f(X, y, bins=4, max_distance=1.0)
    assert v.gamma.shape == (4,)
    np.testing.assert_array_equal(v.counts, [20, 0, 0, 25])
    assert np.isnan(v.gamma[1]) and np.isnan(v.gamma[2])
    np.testing.assert_allclose(v.bin_centers[1:3], [0.375, 0.625])


def test_subsample():
    X = jr.uniform(jr.key(6), (100, 2))
    y = jr.normal(jr.key(7), (100,))
    v = kl.empirical_variogram(X, y, 5, subsample=30, key=jr.key(8))
    assert int(jnp.sum(v.counts)) <= 30 * 29 // 2
    with pytest.raises(ValueError, match="key"):
        kl.empirical_variogram(X, y, 5, subsample=30)


def test_bad_options_raise():
    X, y = jnp.zeros((4, 1)), jnp.zeros(4)
    with pytest.raises(ValueError, match="estimator"):
        kl.empirical_variogram(X, y, estimator="huber")  # ty: ignore[invalid-argument-type]
    with pytest.raises(ValueError, match="metric"):
        kl.empirical_variogram(X, y, metric="manhattan")  # ty: ignore[invalid-argument-type]
    with pytest.raises(ValueError, match="shapes"):
        kl.empirical_variogram(X, jnp.zeros(3))


def _theoretical(kernel, nugget, h, counts=None):
    gamma = (
        nugget
        + kernel.variance
        - kernel(rearrange(h, "n -> n 1"), jnp.zeros((1, 1)))[:, 0]
    )
    edges = jnp.concatenate([jnp.zeros(1), h + 0.5 * (h[1] - h[0])])
    counts = jnp.full(h.shape, 100) if counts is None else counts
    return kl.Variogram(edges, h, gamma, counts)


@pytest.mark.parametrize("weights", ["cressie", "counts", "uniform"])
def test_fit_recovers_matern_parameters(weights):
    truth = kl.Matern(nu=1.5, lengthscale=0.3, variance=2.0)
    v = _theoretical(truth, 0.1, jnp.linspace(0.02, 1.5, 30))
    fitted, tau2 = kl.fit_variogram(v, kl.Matern(nu=1.5), weights=weights)
    np.testing.assert_allclose(fitted.lengthscale, 0.3, rtol=1e-4)
    np.testing.assert_allclose(fitted.variance, 2.0, rtol=1e-4)
    np.testing.assert_allclose(tau2, 0.1, rtol=1e-4)
    assert fitted.nu == 1.5


def test_fit_rbf_without_nugget():
    truth = kl.RBF(lengthscale=0.5, variance=0.7)
    v = _theoretical(truth, 0.0, jnp.linspace(0.05, 2.0, 25))
    fitted, tau2 = kl.fit_variogram(v, kl.RBF(), nugget=False)
    np.testing.assert_allclose(fitted.lengthscale, 0.5, rtol=1e-4)
    np.testing.assert_allclose(fitted.variance, 0.7, rtol=1e-4)
    assert float(tau2) == 0.0


def test_fit_ignores_empty_bins():
    truth = kl.Matern(nu=1.5, lengthscale=0.3, variance=2.0)
    h = jnp.linspace(0.02, 1.5, 30)
    empty = jnp.zeros(30, bool).at[jnp.array([0, 5, 6, 20])].set(True)
    v = _theoretical(truth, 0.1, h, counts=jnp.where(empty, 0, 100))
    v = eqx.tree_at(lambda v: v.gamma, v, jnp.where(empty, jnp.nan, v.gamma))
    fitted, tau2 = kl.fit_variogram(v, kl.Matern(nu=1.5))
    np.testing.assert_allclose(fitted.lengthscale, 0.3, rtol=1e-4)
    np.testing.assert_allclose(tau2, 0.1, rtol=1e-4)


def test_fit_rejects_bad_inputs():
    v = _theoretical(kl.RBF(), 0.0, jnp.linspace(0.1, 1.0, 5))
    with pytest.raises(TypeError, match="Stationary"):
        kl.fit_variogram(v, kl.Linear())  # ty: ignore[invalid-argument-type]
    with pytest.raises(ValueError, match="scalar"):
        kl.fit_variogram(v, kl.RBF(lengthscale=jnp.ones(2)))
    with pytest.raises(ValueError, match="weights"):
        kl.fit_variogram(v, kl.RBF(), weights="ols")  # ty: ignore[invalid-argument-type]


@pytest.mark.slow
def test_fit_on_gp_draw_recovers_range():
    # A single fixed draw (jr.key(0)) with a loose 25% bound: one realisation
    # of a GP on the unit square carries sampling error in the empirical
    # variogram that no tight tolerance survives. The noise-free parameter
    # recovery test above carries the precise check.
    k1, k2, k3 = jr.split(jr.key(0), 3)
    n = 2000
    truth = kl.Matern(nu=1.5, lengthscale=0.05, variance=1.0)
    X = jr.uniform(k1, (n, 2))
    K = truth.gram(X) + 1e-8 * jnp.eye(n)
    y = jnp.linalg.cholesky(K) @ jr.normal(k2, (n,))
    y = y + jnp.sqrt(0.05) * jr.normal(k3, (n,))
    v = kl.empirical_variogram(X, y, 25, max_distance=0.5)
    fitted, _ = kl.fit_variogram(v, kl.Matern(nu=1.5))
    assert abs(float(fitted.lengthscale) / 0.05 - 1.0) < 0.25
