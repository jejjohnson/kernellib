"""Funk-Hecke coefficients of zonal kernels on the sphere (GEO18)."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from geonnax.basis import legendre_polynomials
from scipy.special import spherical_in

import kernellib as kl
from kernellib._einx import einsum, rearrange


def _degree_factor(l_max: int) -> jnp.ndarray:
    ell = jnp.arange(l_max + 1, dtype=jnp.float64)
    return (2.0 * ell + 1.0) / (4.0 * math.pi)


@pytest.mark.parametrize("z", [0.5, 5.0, 50.0])
def test_rbf_matches_the_bessel_closed_form(z):
    # kappa(t) = s2 exp(-z) exp(z t), z = R^2 / ell^2, so
    # a_l = 4 pi s2 exp(-z) i_l(z) (modified spherical Bessel function).
    radius, variance = 2.0, 1.3
    lengthscale = radius / math.sqrt(z)
    k = kl.RBF(lengthscale=lengthscale, variance=variance)
    a = kl.funk_hecke_coefficients(k, 40, radius=radius)
    ell = np.arange(41)
    expected = 4.0 * math.pi * variance * math.exp(-z) * spherical_in(ell, z)
    np.testing.assert_allclose(np.asarray(a), expected, rtol=0.0, atol=1e-10)


def test_degree_sum_converges_to_the_variance():
    # sum_l (2l + 1)/(4 pi) a_l = kappa(1) = k(x, x) = variance; the RBF
    # spectrum at z = 4 has decayed below 1e-12 by degree 30.
    k = kl.RBF(lengthscale=0.5, variance=1.7)
    a = kl.funk_hecke_coefficients(k, 30)
    assert abs(float(jnp.sum(_degree_factor(30) * a)) - 1.7) < 1e-8


def test_legendre_series_reconstructs_a_matern_kernel():
    # The Matern-5/2 spectrum decays like l^-7 (nu + 3/2 = 4, doubled, minus 1),
    # so 100 degrees bring the series to ~1e-7 of kappa, inside the 1e-6 bound.
    k = kl.Matern(nu=2.5, lengthscale=0.6, variance=1.2)
    l_max = 100
    c = kl.funk_hecke_coefficients(k, l_max, convention="legendre")
    t = jnp.linspace(-1.0, 1.0, 50)
    series = einsum(legendre_polynomials(t, l_max), c, "q l, l -> q")
    s = jnp.sqrt(jnp.maximum(1.0 - t**2, 0.0))
    n_t = rearrange(jnp.stack([s, jnp.zeros_like(t), t]), "c q -> q c")
    kappa = k(jnp.array([[0.0, 0.0, 1.0]]), n_t)[0]
    np.testing.assert_allclose(np.asarray(series), np.asarray(kappa), atol=1e-6)


def test_legendre_convention_rescales_by_the_degree_factor():
    k = kl.Matern(nu=1.5, lengthscale=0.8)
    a = kl.funk_hecke_coefficients(k, 20)
    c = kl.funk_hecke_coefficients(k, 20, convention="legendre")
    np.testing.assert_allclose(c, _degree_factor(20) * a, rtol=1e-14)


@pytest.mark.parametrize(
    "kernel",
    [
        kl.RBF(lengthscale=0.3),
        kl.Matern(nu=0.5, lengthscale=2.0),
        kl.RationalQuadratic(lengthscale=0.5, alpha=1.0),
        kl.RBF(lengthscale=0.4) + 0.5 * kl.Matern(nu=1.5, lengthscale=1.5),
    ],
)
def test_coefficients_are_non_negative(kernel):
    a = kl.funk_hecke_coefficients(kernel, 60)
    assert a.shape == (61,)
    assert bool(jnp.all(a >= 0.0))


@pytest.mark.parametrize(
    "kernel",
    [
        kl.RBF(lengthscale=jnp.array([0.5, 1.0, 2.0])),
        kl.RBF() + kl.Matern(lengthscale=jnp.array([0.5, 1.0, 2.0])),
    ],
)
def test_ard_kernels_are_rejected(kernel):
    with pytest.raises(NotImplementedError, match="isotropic"):
        kl.funk_hecke_coefficients(kernel, 5)


def test_invalid_arguments_raise():
    k = kl.RBF()
    with pytest.raises(ValueError, match="l_max"):
        kl.funk_hecke_coefficients(k, -1)
    with pytest.raises(ValueError, match="num_quadrature"):
        kl.funk_hecke_coefficients(k, 3, num_quadrature=0)
    with pytest.raises(ValueError, match="convention"):
        kl.funk_hecke_coefficients(k, 3, convention="schmidt")


def test_l_max_zero_gives_the_mean():
    # a_0 = 2 pi int kappa = 4 pi s2 exp(-z) sinh(z) / z for the RBF (i_0).
    z = 1.0 / 0.7**2
    a = kl.funk_hecke_coefficients(kl.RBF(lengthscale=0.7), 0)
    assert a.shape == (1,)
    assert np.isclose(float(a[0]), 4.0 * math.pi * math.exp(-z) * math.sinh(z) / z)


# Pinned at R = 1, lengthscale 0.7, variance 1.5, l_max 12, 256 nodes. The
# values were produced by this implementation after it matched the RBF
# closed form above to 1e-14, and agree with pyrox-gp's
# `funk_hecke_coefficients` (pyrox 1643cdf) to within 1e-16 absolute.
_PINNED = {
    "rbf": [
        4.540187512325129,
        2.471403008189557,
        0.9072250902864907,
        0.2487015369876783,
        0.05417881841879717,
        0.009772947760854566,
        0.001502629987889468,
        0.0002011947381286691,
        2.384866279144452e-05,
        2.535377245928495e-06,
        2.443008029400508e-07,
        2.15221622101415e-08,
        1.746207147452338e-09,
    ],
    "matern32": [
        4.155914627460769,
        1.946164706950318,
        0.7739843810669443,
        0.3045620722939325,
        0.1267745487431521,
        0.05714611107135305,
        0.02794216983595819,
        0.01471051926077475,
        0.008256486140321945,
        0.004893545656619556,
        0.003037591427555349,
        0.001961254882898278,
        0.00130977695306449,
    ],
}


@pytest.mark.parametrize(
    ("name", "kernel"),
    [
        ("rbf", kl.RBF(lengthscale=0.7, variance=1.5)),
        ("matern32", kl.Matern(nu=1.5, lengthscale=0.7, variance=1.5)),
    ],
)
def test_matches_the_pinned_pyrox_table(name, kernel):
    a = kl.funk_hecke_coefficients(kernel, 12)
    np.testing.assert_allclose(np.asarray(a), _PINNED[name], rtol=1e-10, atol=1e-15)


@pytest.mark.slow
def test_gradient_wrt_lengthscale_is_finite():
    def total(log_ell):
        k = kl.Matern(nu=1.5, lengthscale=jnp.exp(log_ell))
        a = kl.funk_hecke_coefficients(k, 20, radius=1.5)
        return jnp.sum((_degree_factor(20) * a) ** 2)

    g = jax.grad(total)(jnp.log(0.6))
    assert bool(jnp.isfinite(g))
    assert float(g) != 0.0


def test_chordal_kernel_is_unwrapped():
    inner = kl.Matern(nu=1.5, lengthscale=500.0, variance=1.5)
    k = kl.Chordal(inner, radius=kl.EARTH_RADIUS_KM)
    np.testing.assert_allclose(
        kl.funk_hecke_coefficients(k, 20),
        kl.funk_hecke_coefficients(inner, 20, radius=kl.EARTH_RADIUS_KM),
    )
