"""Neighbour search and graph builders on the sphere (GEO4)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import kernellib as kl
from kernellib._einx import rearrange, reduce, repeat
from kernellib._testing import fibonacci_lonlat
from kernellib.functional import chordal_distance, great_circle_distance


def _lonlat():
    """A Fibonacci lattice plus random points, some either side of the
    dateline and some near both poles."""
    key_lon, key_lat = jax.random.split(jax.random.key(0))
    rand = jnp.column_stack(
        [
            jax.random.uniform(key_lon, (60,), minval=-180.0, maxval=180.0),
            jax.random.uniform(key_lat, (60,), minval=-90.0, maxval=90.0),
        ]
    )
    special = jnp.array(
        [
            [179.9, 10.0],
            [-179.9, 10.0],
            [179.5, -20.0],
            [-179.7, -20.1],
            [0.0, 89.95],
            [120.0, 89.9],
            [-60.0, 89.92],
            [30.0, -89.95],
            [-150.0, -89.9],
            [90.0, -89.93],
        ]
    )
    return jnp.concatenate([fibonacci_lonlat(300), rand, special])


def _brute(D, k):
    D = np.where(np.eye(D.shape[0], dtype=bool), np.inf, np.asarray(D))
    return np.argsort(D, kind="stable")[:, :k], np.sort(D)[:, :k]


@pytest.mark.parametrize(
    ("metric", "distance"),
    [("great_circle", great_circle_distance), ("chordal", chordal_distance)],
)
def test_nearest_neighbors_match_brute_force(metric, distance):
    X = _lonlat()
    R = kl.EARTH_RADIUS_KM
    g = kl.nearest_neighbors(X, 6, metric=metric, radius=R)
    idx, dist = _brute(distance(X, X, radius=R), 6)
    assert np.array_equal(np.asarray(g.indices), idx)
    # Relative to R: the distances are in km, and float64 resolves ~1e-16 R.
    np.testing.assert_allclose(np.asarray(g.distances), dist, rtol=0, atol=1e-10 * R)


def test_returned_distances_are_great_circle_to_1e10():
    X = _lonlat()
    g = kl.nearest_neighbors(X, 4, metric="great_circle")
    D = np.asarray(great_circle_distance(X, X))
    rows = np.asarray(repeat(jnp.arange(X.shape[0]), "n -> n k", k=4))
    expected = D[rows, np.asarray(g.indices)]
    np.testing.assert_allclose(np.asarray(g.distances), expected, rtol=0, atol=1e-10)


def test_dateline_and_pole_neighbours():
    X = jnp.array([[179.9, 0.0], [-179.9, 0.0], [0.0, 0.0], [0.0, 89.9], [180, 89.9]])
    g = kl.nearest_neighbors(X, 1, metric="great_circle")
    assert rearrange(g.indices, "n 1 -> n").tolist()[:2] == [1, 0]
    assert rearrange(g.indices, "n 1 -> n").tolist()[3:] == [4, 3]
    np.testing.assert_allclose(g.distances[0, 0], np.radians(0.2), atol=1e-12)


def test_radians_input():
    X = _lonlat()
    a = kl.nearest_neighbors(X, 3, metric="great_circle")
    b = kl.nearest_neighbors(jnp.radians(X), 3, metric="great_circle", degrees=False)
    assert np.array_equal(np.asarray(a.indices), np.asarray(b.indices))
    np.testing.assert_allclose(a.distances, b.distances, atol=1e-12)


def test_radius_neighbors_match_brute_force():
    X = _lonlat()
    R, r = kl.EARTH_RADIUS_KM, 1500.0
    dist = np.asarray(great_circle_distance(X, X, radius=R))
    within = (dist <= r) & ~np.eye(dist.shape[0], dtype=bool)
    k = (
        int(np.max(reduce(jnp.asarray(within, int), "n m -> n", "sum"))) + 1
    )  # room for every point within r
    g = kl.radius_neighbors(
        X, r, max_neighbors=k, metric="great_circle", sphere_radius=R
    )
    for i, row in enumerate(np.asarray(g.indices)):
        assert set(row[row >= 0].tolist()) == set(np.flatnonzero(within[i]).tolist())
    assert np.all(np.isinf(np.asarray(g.distances)[np.asarray(g.indices) < 0]))


def test_knn_graph_heat_weights_use_great_circle_distances():
    X = _lonlat()
    knn = kl.nearest_neighbors(X, 5, metric="great_circle")
    expected = kl.graph_from_neighbors(knn)
    g = kl.knn_graph(X, 5, metric="great_circle")
    assert np.array_equal(g.topology.senders, expected.topology.senders)
    assert np.array_equal(g.topology.receivers, expected.topology.receivers)
    np.testing.assert_allclose(g.weights, expected.weights, atol=1e-12)


def test_knn_graph_bridges_with_great_circle_lengths():
    # Two pairs across the dateline; the bridge is the 167-degree arc 0-3.
    X = jnp.array([[179.0, 0.0], [-179.0, 0.0], [10.0, 0.0], [12.0, 0.0]])
    sigma = 1.0
    g = kl.knn_graph(
        X, 1, metric="great_circle", bandwidth=sigma, ensure_connected=True
    )
    top = g.topology
    pairs = list(zip(top.senders.tolist(), top.receivers.tolist(), strict=True))
    assert pairs == [(0, 1), (0, 3), (2, 3)]
    d = np.radians(np.array([2.0, 167.0, 2.0]))
    np.testing.assert_allclose(g.weights, np.exp(-(d**2) / 2.0), atol=1e-12)


def test_cosine_weighting_uses_unit_vectors():
    X = jnp.array([[0.0, 0.0], [60.0, 0.0], [0.0, 30.0]])
    g = kl.knn_graph(X, 2, metric="great_circle", weighting="cosine")
    D = np.asarray(great_circle_distance(X, X))
    expected = np.cos(D[g.topology.senders, g.topology.receivers])
    np.testing.assert_allclose(g.weights, expected, atol=1e-12)


def test_radius_graph_on_the_sphere():
    X = jnp.array([[179.5, 0.0], [-179.5, 0.0], [0.0, 0.0]])
    g = kl.radius_graph(
        X,
        200.0,
        max_neighbors=2,
        metric="great_circle",
        sphere_radius=kl.EARTH_RADIUS_KM,
        weighting="connectivity",
    )
    assert g.topology.senders.tolist() == [0]
    assert g.topology.receivers.tolist() == [1]


@pytest.mark.parametrize("metric", ["great_circle", "chordal"])
def test_geo_metric_needs_lonlat(metric):
    with pytest.raises(ValueError, match=r"\(N, 2\)"):
        kl.nearest_neighbors(jnp.ones((5, 3)), 2, metric=metric)


def test_unknown_metric_raises():
    with pytest.raises(ValueError, match="metric must be"):
        kl.nearest_neighbors(_lonlat(), 2, metric="haversine")  # ty: ignore[invalid-argument-type]


def test_dtype_follows_input():
    X = _lonlat().astype(jnp.float32)
    g = kl.nearest_neighbors(X, 2, metric="great_circle")
    assert g.distances.dtype == jnp.float32


@pytest.mark.integration
def test_sklearn_backend_on_the_sphere():
    X = _lonlat()
    exact = kl.nearest_neighbors(X, 4, metric="great_circle")
    sk = kl.nearest_neighbors(X, 4, metric="great_circle", backend="sklearn")
    assert np.array_equal(np.asarray(sk.indices), np.asarray(exact.indices))
    np.testing.assert_allclose(sk.distances, exact.distances, atol=1e-12)
