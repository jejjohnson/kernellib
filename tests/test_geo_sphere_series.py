"""Tests for the Legendre-series kernels on the sphere (GEO9)."""

from __future__ import annotations

import itertools
import math

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import eval_legendre

import kernellib as kl
from kernellib._einx import einsum, rearrange
from kernellib._testing import fibonacci_lonlat
from kernellib.functional import legendre_series, lonlat_to_unit


def _kernels(lengthscale: float = 0.5) -> list[kl.AbstractSphereSeriesKernel]:
    return [
        kl.SphereMatern(lengthscale=lengthscale, nu=0.5),
        kl.SphereMatern(lengthscale=lengthscale, nu=1.5, variance=2.0),
        kl.SphereMatern(lengthscale=lengthscale, nu=2.5),
        kl.SphereHeat(lengthscale=lengthscale, variance=0.7),
        kl.SphereSeries(lambda lam: (1.0 + lam) ** -2.0),
    ]


def _points(n: int = 12) -> jax.Array:
    return _uniform_lonlat(jax.random.key(0), n)


def _uniform_lonlat(key: jax.Array, n: int) -> jax.Array:
    k1, k2 = jax.random.split(key)
    lon = jax.random.uniform(k1, (n,), minval=-180.0, maxval=180.0)
    lat = jnp.rad2deg(jnp.arcsin(jax.random.uniform(k2, (n,), minval=-1.0)))
    return rearrange(jnp.stack([lon, lat]), "c n -> n c")


# --- legendre_series -------------------------------------------------------


def test_legendre_series_matches_scipy() -> None:
    L = 200
    rng = np.random.default_rng(0)
    coeffs = rng.standard_normal(L + 1)
    x = np.concatenate([np.linspace(-1.0, 1.0, 401), [-1.0, 1.0, 0.0]])
    expected = sum(c * eval_legendre(ell, x) for ell, c in enumerate(coeffs))
    got = np.asarray(legendre_series(jnp.asarray(x), jnp.asarray(coeffs)))
    np.testing.assert_allclose(got, expected, rtol=0.0, atol=1e-12)


def test_legendre_series_low_degrees_and_shapes() -> None:
    x = jnp.array([[-0.3, 0.2], [0.9, 1.0]])
    assert legendre_series(x, jnp.array([2.0])).tolist() == [[2.0, 2.0], [2.0, 2.0]]
    np.testing.assert_allclose(legendre_series(x, jnp.array([1.0, 3.0])), 1.0 + 3 * x)
    # Rounding outside [-1, 1] is clipped: P_l(1) = 1.
    assert float(legendre_series(jnp.array(1.0 + 1e-9), jnp.ones(5))) == 5.0
    with pytest.raises(ValueError, match="1-D"):
        legendre_series(x, jnp.ones((2, 2)))
    with pytest.raises(ValueError, match="1-D"):
        legendre_series(x, jnp.ones((0,)))


# --- the kernels ------------------------------------------------------------


@pytest.mark.parametrize("kernel", _kernels(), ids=repr)
def test_diag_is_variance(kernel: kl.AbstractSphereSeriesKernel) -> None:
    X = _points()
    K = kernel(X, X)
    np.testing.assert_allclose(kernel.diag(X), kernel.variance, rtol=1e-12)
    np.testing.assert_allclose(jnp.diag(K), kernel.variance, rtol=1e-12)
    np.testing.assert_allclose(K, rearrange(K, "i j -> j i"), atol=1e-14)
    a = kernel.degree_spectrum()
    assert a.shape == (kernel.max_degree + 1,)
    assert bool(jnp.all(a >= 0.0))
    np.testing.assert_allclose(jnp.sum(a), kernel.variance, rtol=1e-12)


@pytest.mark.parametrize("kernel", _kernels(), ids=repr)
def test_pairwise_matches_gram(kernel: kl.AbstractSphereSeriesKernel) -> None:
    X = _points(5)
    pw = jax.vmap(lambda x: jax.vmap(lambda y: kernel.pairwise(x, y))(X))(X)
    np.testing.assert_allclose(pw, kernel(X, X), atol=1e-12)


def _random_rotation(key: jax.Array) -> jax.Array:
    q, r = jnp.linalg.qr(jax.random.normal(key, (3, 3)))
    q = q * jnp.sign(jnp.diag(r))
    return q * jnp.sign(jnp.linalg.det(q))  # det = +1: a rotation


@pytest.mark.parametrize("kernel", _kernels(), ids=repr)
def test_rotation_invariant_and_unit_vector_inputs(
    kernel: kl.AbstractSphereSeriesKernel,
) -> None:
    X = _points()
    U = lonlat_to_unit(X)
    K = kernel(X, X)
    # (N, 3) inputs are accepted (and normalised to unit length).
    np.testing.assert_allclose(kernel(U, U), K, atol=1e-12)
    np.testing.assert_allclose(kernel(3.0 * U, U), K, atol=1e-12)
    Q = _random_rotation(jax.random.key(1))
    np.testing.assert_allclose(jnp.linalg.det(Q), 1.0, atol=1e-12)
    UR = einsum(Q, U, "i j, n j -> n i")
    np.testing.assert_allclose(kernel(UR, UR), K, atol=1e-12)


@pytest.mark.slow
@pytest.mark.parametrize(
    "kernel",
    [
        kl.SphereMatern(lengthscale=0.3, nu=0.5),
        kl.SphereMatern(lengthscale=0.3, nu=1.5),
        kl.SphereMatern(lengthscale=0.3, nu=2.5),
        kl.SphereHeat(lengthscale=0.3),
    ],
    ids=repr,
)
def test_positive_definite_on_fibonacci_points(
    kernel: kl.AbstractSphereSeriesKernel,
) -> None:
    n = 500
    X = fibonacci_lonlat(n)
    lam_min = float(np.linalg.eigvalsh(np.asarray(kernel(X, X))).min())
    assert lam_min >= -1e-10 * n


@pytest.mark.parametrize("lengthscale", [0.3, 0.5, 1.0])
def test_matern_tends_to_heat(lengthscale: float) -> None:
    X = _points(20)
    matern = kl.SphereMatern(lengthscale=lengthscale, nu=50.0)
    heat = kl.SphereHeat(lengthscale=lengthscale)
    np.testing.assert_allclose(matern(X, X), heat(X, X), rtol=0.0, atol=1e-2)


@pytest.mark.slow
@pytest.mark.parametrize("nu", [1.5, 2.5])
def test_locally_matches_euclidean_matern(nu: float) -> None:
    # For ell << R the sphere is locally flat: the intrinsic Matérn has the
    # Euclidean d = 2 spectrum (2 nu / ell^2 + |omega|^2)^-(nu + 1) with
    # |omega|^2 -> lambda, so at small separations it matches the Euclidean
    # Matérn of the same nu and ell. The differences are the curvature,
    # O((ell / R)^2) = 4e-4 here, and the truncation, whose tail at L = 2000 is
    # below 1e-4 (truncation_tail); the issue's 2 % tolerance is far above both.
    ell = 0.02
    k = kl.SphereMatern(lengthscale=ell, nu=nu, max_degree=2000)
    theta = jnp.linspace(0.0, 3.0 * ell, 13)  # angles, radians
    X = rearrange(jnp.stack([jnp.zeros_like(theta), jnp.rad2deg(theta)]), "c n -> n c")
    assert float(k.truncation_tail()) < 1e-4
    sphere = k(X[:1], X)[0]
    U = lonlat_to_unit(X)  # chordal distance ||u - v|| for radius 1
    euclid = kl.Matern(nu=nu, lengthscale=ell)(U[:1], U)[0]
    np.testing.assert_allclose(sphere, euclid, rtol=0.0, atol=2e-2)


def test_radius_sets_lengthscale_units() -> None:
    X = _points(8)
    unit = kl.SphereMatern(lengthscale=0.4, nu=1.5)
    earth = kl.SphereMatern(
        lengthscale=0.4 * kl.EARTH_RADIUS_KM, nu=1.5, radius=kl.EARTH_RADIUS_KM
    )
    np.testing.assert_allclose(earth(X, X), unit(X, X), atol=1e-12)


def test_series_kernel_matches_heat() -> None:
    X = _points(8)
    series = kl.SphereSeries(lambda lam: jnp.exp(-0.5 * 0.4**2 * lam), max_degree=40)
    heat = kl.SphereHeat(lengthscale=0.4, max_degree=40)
    np.testing.assert_allclose(series(X, X), heat(X, X), atol=1e-12)
    np.testing.assert_allclose(series.degree_spectrum(), heat.degree_spectrum())


# --- truncation --------------------------------------------------------------


def _exact_tail(kernel: kl.AbstractSphereSeriesKernel, l_far: int) -> float:
    ell = np.arange(l_far + 1, dtype=np.float64)
    terms = (2 * ell + 1) * np.asarray(
        kernel.spectrum(jnp.asarray(ell * (ell + 1) / kernel.radius**2))
    )
    return float(terms[kernel.max_degree + 1 :].sum() / terms.sum())


@pytest.mark.slow
@pytest.mark.parametrize(
    "make",
    [
        lambda L: kl.SphereMatern(lengthscale=0.1, nu=1.5, max_degree=L),
        lambda L: kl.SphereMatern(lengthscale=0.3, nu=0.5, max_degree=L),
        lambda L: kl.SphereHeat(lengthscale=0.1, max_degree=L),
        lambda L: kl.SphereSeries(lambda lam: (1.0 + lam) ** -2.0, max_degree=L),
    ],
)
def test_truncation_tail_decreases_and_bounds(make) -> None:
    tails = [float(make(L).truncation_tail()) for L in (8, 16, 32, 64, 128)]
    assert all(b < a for a, b in itertools.pairwise(tails)), tails
    # Against the series summed far out: the integral bound is an upper bound
    # within a factor of two; the summed estimate of SphereSeries is a lower
    # bound (it stops at 16 (L + 1)).
    kernel = make(32)
    exact = _exact_tail(kernel, 200_000)
    estimate = float(kernel.truncation_tail())
    if isinstance(kernel, kl.SphereSeries):
        assert 0.5 * exact <= estimate <= exact
    else:
        assert exact <= estimate <= 2.0 * exact


def test_truncation_tail_matern_at_256() -> None:
    k = kl.SphereMatern(lengthscale=0.1, nu=1.5, max_degree=256)
    assert float(k.truncation_tail()) < 1e-3


# --- gradients, transformations, Funk-Hecke, validation ----------------------


@pytest.mark.slow
@pytest.mark.parametrize(
    "kernel",
    [kl.SphereMatern(lengthscale=0.5, nu=1.5, variance=1.3), kl.SphereHeat()],
    ids=repr,
)
def test_gradients_are_finite(kernel: kl.AbstractSphereSeriesKernel) -> None:
    X = _points(6)

    @eqx.filter_grad
    def grad(k: kl.AbstractSphereSeriesKernel) -> jax.Array:
        return jnp.sum(k(X, X) ** 2) + k.truncation_tail()

    g = grad(kernel)
    for leaf in (g.lengthscale, g.variance):
        assert bool(jnp.isfinite(leaf))
    assert float(jnp.abs(g.lengthscale)) > 0.0
    # Input gradient at coincident and distinct points.
    gx = jax.grad(lambda x: kernel.pairwise(x, X[1]))(X[0])
    gx0 = jax.grad(lambda x: kernel.pairwise(x, X[0]))(X[0])
    assert bool(jnp.all(jnp.isfinite(gx))) and bool(jnp.all(jnp.isfinite(gx0)))


@pytest.mark.slow
def test_jit_and_vmap() -> None:
    k = kl.SphereMatern(lengthscale=0.5, nu=2.5)
    X = _points(6)
    np.testing.assert_allclose(eqx.filter_jit(lambda k: k(X, X))(k), k(X, X))
    ks = jax.vmap(lambda ell: kl.SphereMatern(lengthscale=ell, nu=2.5)(X, X))(
        jnp.array([0.3, 0.5])
    )
    np.testing.assert_allclose(ks[1], k(X, X), atol=1e-12)


@pytest.mark.slow
@pytest.mark.parametrize(
    "kernel",
    [kl.SphereMatern(lengthscale=0.5, nu=1.5, max_degree=40), kl.SphereHeat()],
    ids=repr,
)
def test_funk_hecke_recovers_degree_spectrum(
    kernel: kl.AbstractSphereSeriesKernel,
) -> None:
    # The kernel is a degree-L polynomial in u.v, so 256-node Gauss-Legendre
    # quadrature of kappa(t) P_l(t) is exact up to rounding.
    c = kl.funk_hecke_coefficients(kernel, 30, convention="legendre")
    np.testing.assert_allclose(c, kernel.degree_spectrum()[:31], atol=1e-12)
    a = kl.funk_hecke_coefficients(kernel, 30)
    ell = jnp.arange(31)
    np.testing.assert_allclose(
        a, 4 * math.pi / (2 * ell + 1) * kernel.degree_spectrum()[:31], atol=1e-11
    )


def test_validation() -> None:
    with pytest.raises(ValueError, match="nu"):
        kl.SphereMatern(nu=0.0)
    with pytest.raises(ValueError, match="max_degree"):
        kl.SphereHeat(max_degree=-1)
    with pytest.raises(ValueError, match="radius"):
        kl.SphereHeat(radius=0.0)
    with pytest.raises(ValueError, match=r"\(N, 2\)"):
        kl.SphereHeat()(jnp.ones((2, 4)), jnp.ones((2, 4)))
