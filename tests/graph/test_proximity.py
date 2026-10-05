"""Proximity graphs: Delaunay, Gabriel and relative neighbourhood."""

from __future__ import annotations

import einx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.sparse.csgraph import minimum_spanning_tree

import kernellib as kl


BUILDERS = (kl.delaunay_graph, kl.gabriel_graph, kl.relative_neighborhood_graph)


def _X(n=200, d=2, seed=0):
    return jax.random.uniform(jax.random.key(seed), (n, d), dtype=jnp.float64)


def _pairs(graph):
    top = graph.topology
    return set(zip(top.senders.tolist(), top.receivers.tolist(), strict=True))


def _sq_distances(X):
    X = np.asarray(X, dtype=np.float64)
    return einx.sum("i j [d]", einx.subtract("i d, j d -> i j d", X, X) ** 2)


def _emst(X):
    D = np.sqrt(_sq_distances(X))
    T = minimum_spanning_tree(D).tocoo()
    return {(min(a, b), max(a, b)) for a, b in zip(T.row, T.col, strict=True)}


def _brute(X, rule):
    """O(n^3) definitions: (i, j) is an edge iff no k blocks it."""
    D2 = _sq_distances(X)
    ik = einx.id("i k -> i 1 k", D2)
    jk = einx.id("j k -> 1 j k", D2)
    ij = einx.id("i j -> i j 1", D2)
    blocks = (ik + jk < ij) if rule == "gabriel" else (np.maximum(ik, jk) < ij)
    n = D2.shape[0]
    idx = np.arange(n)
    # k must differ from i and j (on the diagonal the inequalities are false).
    edge = ~einx.any("i j [k]", blocks)
    edge &= einx.less("i, j -> i j", idx, idx)
    i, j = np.nonzero(edge)
    return set(zip(i.tolist(), j.tolist(), strict=True))


@pytest.mark.parametrize("d", [2, 3])
def test_nesting_emst_rng_gg_dt(d):
    X = _X(150, d, seed=1)
    dt, gg, rng = (_pairs(f(X)) for f in BUILDERS)
    emst = _emst(X)
    assert emst <= rng <= gg <= dt
    assert len(rng) < len(gg) < len(dt)


@pytest.mark.parametrize("d", [2, 3])
@pytest.mark.parametrize("builder", BUILDERS)
def test_connected_and_sparse(builder, d):
    X = _X(150, d, seed=2)
    g = builder(X)
    assert kl.n_components_graph(g) == 1
    assert g.topology.n_edges <= 8 * X.shape[0]  # O(N) edges


@pytest.mark.parametrize(("d", "n"), [(2, 200), (3, 100)])
def test_gabriel_matches_brute_force(d, n):
    X = _X(n, d, seed=3)
    assert _pairs(kl.gabriel_graph(X)) == _brute(X, "gabriel")


@pytest.mark.parametrize(("d", "n"), [(2, 200), (3, 100)])
def test_rng_matches_brute_force(d, n):
    X = _X(n, d, seed=4)
    assert _pairs(kl.relative_neighborhood_graph(X)) == _brute(X, "rng")


def test_delaunay_matches_scipy():
    from scipy.spatial import Delaunay

    X = _X(50, 2, seed=5)
    expected = set()
    for a, b, c in Delaunay(np.asarray(X)).simplices.tolist():
        expected |= {tuple(sorted(e)) for e in ((a, b), (b, c), (a, c))}
    assert _pairs(kl.delaunay_graph(X)) == expected


@pytest.mark.parametrize("builder", BUILDERS)
def test_collinear_points_raise(builder):
    X = jnp.stack([jnp.arange(6.0), 2.0 * jnp.arange(6.0)], axis=-1)
    with pytest.raises(ValueError, match="collinear"):
        builder(X)


def test_coplanar_points_raise():
    X = jnp.concatenate([_X(20, 2, seed=6), jnp.zeros((20, 1))], axis=-1)
    with pytest.raises(ValueError, match="coplanar"):
        kl.gabriel_graph(X)


@pytest.mark.parametrize("builder", BUILDERS)
def test_duplicate_points_raise(builder):
    X = _X(20, 2, seed=7)
    X = jnp.concatenate([X, X[:1]])
    with pytest.raises(ValueError, match="duplicate"):
        builder(X)


@pytest.mark.parametrize("builder", BUILDERS)
def test_higher_dimension_points_to_knn_graph(builder):
    with pytest.raises(ValueError, match="knn_graph"):
        builder(_X(20, 4, seed=8))


def test_too_few_points_raise():
    with pytest.raises(ValueError, match="at least 3"):
        kl.delaunay_graph(jnp.array([[0.0, 0.0], [1.0, 0.0]]))


def test_heat_weights_on_edge_lengths():
    X = _X(60, 2, seed=9)
    g = kl.gabriel_graph(X, bandwidth=0.1)
    top = g.topology
    Xh = np.asarray(X)
    d = np.sqrt(einx.sum("e [d]", (Xh[top.senders] - Xh[top.receivers]) ** 2))
    np.testing.assert_allclose(g.weights, np.exp(-(d**2) / (2 * 0.1**2)), rtol=1e-10)
    # Default bandwidth: the median edge length.
    g = kl.gabriel_graph(X)
    sigma = np.median(d)
    np.testing.assert_allclose(g.weights, np.exp(-(d**2) / (2 * sigma**2)), rtol=1e-10)


def test_connectivity_and_kernel_weighting():
    X = _X(60, 2, seed=10)
    g = kl.relative_neighborhood_graph(X, weighting="connectivity")
    np.testing.assert_array_equal(g.weights, 1.0)
    k = kl.RBF(lengthscale=0.3)
    g = kl.relative_neighborhood_graph(X, weighting=k)
    top = g.topology
    expected = k.elwise(X[top.senders], X[top.receivers])
    np.testing.assert_allclose(g.weights, expected, rtol=1e-10)
    with pytest.raises(ValueError, match="bandwidth"):
        kl.delaunay_graph(X, weighting="connectivity", bandwidth=1.0)


def test_weights_differentiable_in_bandwidth():
    X = _X(30, 2, seed=11)

    def total(sigma):
        return jnp.sum(kl.gabriel_graph(X, bandwidth=sigma).weights)

    g = jax.grad(total)(0.2)
    assert bool(jnp.isfinite(g))
    assert float(g) > 0.0  # wider heat kernel, larger weights
