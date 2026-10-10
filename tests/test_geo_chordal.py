"""Tests for the `Chordal` kernel wrapper (GEO2)."""

from __future__ import annotations

import einx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import kernellib as kl
from kernellib._einx import einsum
from kernellib._testing import fibonacci_lonlat
from kernellib.functional import chordal_distance, lonlat_to_unit


def _random_lonlat(n: int, seed: int = 0) -> jax.Array:
    rng = np.random.default_rng(seed)
    lon = rng.uniform(-180.0, 180.0, n)
    lat = np.degrees(np.arcsin(rng.uniform(-1.0, 1.0, n)))
    return jnp.asarray(np.column_stack([lon, lat]))


def test_matches_rbf_on_cartesian_points():
    X = _random_lonlat(30)
    R, ell = 3.0, 1.2
    k = kl.Chordal(kl.RBF(lengthscale=ell), radius=R)
    U = R * lonlat_to_unit(X)
    np.testing.assert_allclose(k(X, X), kl.RBF(lengthscale=ell)(U, U), atol=1e-12)
    np.testing.assert_allclose(k.to_cartesian(X), U, atol=1e-12)


def test_rbf_is_a_function_of_chordal_distance():
    X = _random_lonlat(20)
    k = kl.Chordal(kl.RBF(lengthscale=500.0), radius=kl.EARTH_RADIUS_KM)
    c = chordal_distance(X, X, radius=kl.EARTH_RADIUS_KM)
    np.testing.assert_allclose(k(X, X), jnp.exp(-0.5 * (c / 500.0) ** 2), atol=1e-12)


@pytest.mark.parametrize(
    "base",
    [
        kl.RBF(lengthscale=0.3),
        kl.Matern(nu=0.5, lengthscale=0.3),
        kl.Matern(nu=1.5, lengthscale=0.3),
        kl.Matern(nu=2.5, lengthscale=0.3),
        kl.RationalQuadratic(lengthscale=0.3, alpha=2.0),
    ],
    ids=["rbf", "matern12", "matern32", "matern52", "rq"],
)
def test_positive_definite_on_fibonacci_points(base):
    n = 400
    X = fibonacci_lonlat(n)
    K = kl.Chordal(base)(X, X)
    assert float(jnp.linalg.eigvalsh(K)[0]) >= -1e-10 * n


def test_longitude_wraps_and_radians_match():
    X = _random_lonlat(15)
    k = kl.Chordal(kl.Matern(nu=1.5, lengthscale=0.7))
    shifted = einx.add("n d, d -> n d", X, jnp.array([360.0, 0.0]))
    np.testing.assert_allclose(k(shifted, X), k(X, X), atol=1e-12)
    k_rad = kl.Chordal(kl.Matern(nu=1.5, lengthscale=0.7), degrees=False)
    np.testing.assert_allclose(
        k_rad(jnp.radians(X), jnp.radians(X)), k(X, X), atol=1e-12
    )


def test_diag_pairwise_and_cross_shapes():
    X1, X2 = _random_lonlat(5), _random_lonlat(7, seed=1)
    k = kl.Chordal(kl.RBF(lengthscale=0.5, variance=2.5), radius=2.0)
    np.testing.assert_allclose(k.diag(X1), jnp.full(5, 2.5), atol=1e-12)
    K = k(X1, X2)
    assert K.shape == (5, 7)
    assert k.is_pointwise and not k.is_stationary
    np.testing.assert_allclose(k.pairwise(X1[0], X2[3]), K[0, 3], atol=1e-12)


def test_grad_wrt_lengthscale_is_finite():
    X = _random_lonlat(10)

    def loss(ell):
        return jnp.sum(kl.Chordal(kl.Matern(nu=2.5, lengthscale=ell))(X, X))

    g = jax.grad(loss)(jnp.asarray(0.8))
    assert bool(jnp.isfinite(g))
    assert float(g) > 0.0  # a longer lengthscale raises every entry


def test_space_time_recipe_is_positive_definite():
    n = 60
    X = fibonacci_lonlat(n)
    t = jnp.linspace(0.0, 3.0, n)
    XT = jnp.concatenate([X, einx.id("n -> n 1", t)], axis=-1)
    k = kl.Product(
        kl.ActiveDims(kl.Chordal(kl.Matern(nu=1.5, lengthscale=0.5)), dims=(0, 1)),
        kl.ActiveDims(kl.Matern(nu=0.5, lengthscale=2.0), dims=(2,)),
    )
    assert float(jnp.linalg.eigvalsh(k(XT, XT))[0]) >= -1e-10 * n


def test_nystrom_accepts_chordal_directly():
    X = _random_lonlat(25)
    k = kl.Chordal(kl.Matern(nu=0.5, lengthscale=0.3))
    nys = kl.NystromFeatures(25, jax.random.key(0)).fit(k, X)
    Phi = nys(X)
    np.testing.assert_allclose(einsum(Phi, Phi, "n r, m r -> n m"), k(X, X), atol=1e-6)


@pytest.mark.slow
def test_rff_on_cartesian_points_converges_to_chordal():
    X = _random_lonlat(30)
    ell, var, F = 0.6, 1.7, 4000
    k = kl.Chordal(kl.RBF(lengthscale=ell, variance=var), radius=1.5)
    U = k.to_cartesian(X)
    rff = kl.RandomFourierFeatures(F, jax.random.key(0)).fit(k.kernel, U)
    Phi = rff(U)
    K_hat = einsum(Phi, Phi, "n r, m r -> n m")
    K = k(X, X)
    # Each entry is var/F * sum_j cos(w_j . delta) with w_j ~ N(0, I / ell^2),
    # so its sampling variance is var^2/F * ((1 + rho(2 delta)) / 2 - rho^2),
    # with rho(delta) = K / var and rho(2 delta) = rho^4 for the RBF.
    rho = K / var
    sd = jnp.sqrt(var**2 / F * jnp.clip((1.0 + rho**4) / 2.0 - rho**2, 0.0))
    # 6 sigma per entry: with 900 entries the chance of a false failure is
    # below 1e-6 (union bound).
    assert bool(jnp.all(jnp.abs(K_hat - K) <= 6.0 * sd + 1e-10))
