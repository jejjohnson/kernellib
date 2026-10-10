"""Tests for the distances on the sphere (GEO1)."""

from __future__ import annotations

import math

import einx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from kernellib import EARTH_RADIUS_KM
from kernellib._testing import fibonacci_lonlat
from kernellib.functional import (
    chordal_distance,
    great_circle_distance,
    lonlat_to_unit,
)


def _haversine(X1, X2):
    lon1, lat1 = np.radians(X1[:, 0]), np.radians(X1[:, 1])
    lon2, lat2 = np.radians(X2[:, 0]), np.radians(X2[:, 1])
    dlat = np.subtract.outer(lat2, lat1).T
    dlon = np.subtract.outer(lon2, lon1).T
    h = (
        np.sin(dlat / 2) ** 2
        + np.outer(np.cos(lat1), np.cos(lat2)) * np.sin(dlon / 2) ** 2
    )
    return 2.0 * np.arcsin(np.sqrt(h))


def test_london_paris():
    london = jnp.array([[-0.1278, 51.5074]])
    paris = jnp.array([[2.3522, 48.8566]])
    d = float(great_circle_distance(london, paris, radius=EARTH_RADIUS_KM)[0, 0])
    expected = EARTH_RADIUS_KM * _haversine(np.asarray(london), np.asarray(paris))[0, 0]
    assert abs(d - expected) < 1e-9
    assert 343.0 < d < 344.0


def test_antipodes_and_poles():
    X = jnp.array([[0.0, 0.0], [180.0, 0.0], [0.0, 90.0], [0.0, -90.0]])
    d = np.asarray(great_circle_distance(X, X, radius=2.0))
    np.testing.assert_allclose([d[0, 1], d[2, 3]], 2.0 * math.pi, rtol=1e-12)


def test_matches_haversine_away_from_antipodes():
    rng = np.random.default_rng(0)
    base = np.column_stack([rng.uniform(-180, 180, 40), rng.uniform(-80, 80, 40)])
    offsets = np.degrees(np.logspace(-6, 0.4, 40))  # 1e-6 .. 2.5 rad
    other = base + np.column_stack([offsets, 0.5 * offsets])
    other[:, 1] = np.clip(other[:, 1], -89.0, 89.0)
    d = np.diag(
        np.asarray(great_circle_distance(jnp.asarray(base), jnp.asarray(other)))
    )
    ref = np.diag(_haversine(base, other))
    np.testing.assert_allclose(d, ref, rtol=1e-9, atol=1e-15)


def test_metric_properties_and_chordal_relation():
    X = fibonacci_lonlat(200)
    gc = np.asarray(great_circle_distance(X, X))
    ch = np.asarray(chordal_distance(X, X))
    np.testing.assert_allclose(gc, gc.T, atol=1e-12)
    np.testing.assert_allclose(np.diag(gc), 0.0, atol=1e-12)
    np.testing.assert_allclose(ch, 2.0 * np.sin(gc / 2.0), atol=1e-12)
    # d(i, k) <= d(i, j) + d(j, k) for every triple.
    detour = einx.add("i j, j k -> i j k", gc, gc)
    assert bool(einx.less_equal("i k, i j k -> i j k", gc, detour + 1e-12).all())


def test_units_and_longitude_wrap():
    X = fibonacci_lonlat(30)
    wrapped = X.at[:, 0].add(360.0)
    radians = jnp.radians(X)
    d = great_circle_distance(X, X)
    np.testing.assert_allclose(great_circle_distance(wrapped, X), d, atol=1e-10)
    np.testing.assert_allclose(
        great_circle_distance(radians, radians, degrees=False), d, atol=1e-12
    )
    np.testing.assert_allclose(
        lonlat_to_unit(radians, degrees=False), lonlat_to_unit(X), atol=1e-12
    )


@pytest.mark.parametrize("distance", [great_circle_distance, chordal_distance])
def test_gradient_finite_on_diagonal(distance):
    X = fibonacci_lonlat(10)

    def loss(points):
        return jnp.sum(jnp.exp(-distance(points, points)))

    assert jnp.isfinite(jax.grad(loss)(X)).all()


def test_float32_resolves_one_metre():
    with jax.enable_x64(False):
        a = jnp.array([[10.0, 45.0]], dtype=jnp.float32)
        metre_in_degrees = math.degrees(1e-3 / EARTH_RADIUS_KM)
        b = a + jnp.array([[0.0, metre_in_degrees]], dtype=jnp.float32)
        d = float(great_circle_distance(a, b, radius=EARTH_RADIUS_KM)[0, 0])
    assert abs(d - 1e-3) / 1e-3 < 1e-1  # float32 lon/lat input quantisation


def test_bad_shape_raises():
    with pytest.raises(ValueError, match="lon, lat"):
        great_circle_distance(jnp.zeros((3, 3)), jnp.zeros((3, 3)))
