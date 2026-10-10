"""Slepian features: spherical-harmonic features concentrated on a cap (GEO19)."""

from __future__ import annotations

import math

import einx
import equinox as eqx
import jax.numpy as jnp
import pytest
from geonnax.basis import harmonic_degrees, real_spherical_harmonics

import kernellib as kl
from kernellib._einx import einsum
from kernellib._testing import fibonacci_lonlat
from kernellib.functional import great_circle_distance, lonlat_to_unit


CENTRE = (20.0, -10.0)
CAP = math.radians(30.0)


def _split_by_cap(n: int):
    """Fibonacci points inside the cap and beyond twice its radius."""
    X = fibonacci_lonlat(n)
    d = great_circle_distance(X, jnp.asarray([CENTRE]), degrees=True)
    d = einx.id("n 1 -> n", d)
    return X[d < CAP], X[d > 2 * CAP]


def _norm_sq(Phi):
    return einx.sum("n k -> n", Phi**2)


# --- construction and errors (fast) ----------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"l_max": -1},
        {"cap_radius": 0.0},
        {"cap_radius": 4.0},
        {"cap_centre": (0.0,)},
        {"n_modes": 0},
        {"n_modes": 17},  # > (l_max + 1)^2 = 16
        {"num_quadrature": 0},
        {"jitter": -1.0},
    ],
)
def test_invalid_configuration_raises(kwargs):
    config = {"l_max": 3, "cap_radius": CAP, "cap_centre": CENTRE} | kwargs
    with pytest.raises(ValueError):
        kl.SlepianFeatures(**config)


def test_shannon_number():
    # (L + 1)^2 * area / 4 pi with area = 2 pi (1 - cos r): 289 * 0.067 = 19.4
    sf = kl.SlepianFeatures(l_max=16, cap_radius=CAP, cap_centre=CENTRE)
    assert sf.shannon_number == 19
    # The whole sphere keeps every harmonic.
    full = kl.SlepianFeatures(l_max=3, cap_radius=math.pi, cap_centre=CENTRE)
    assert full.shannon_number == 16


def test_unfitted_raises():
    sf = kl.SlepianFeatures(l_max=3, cap_radius=CAP, cap_centre=CENTRE)
    assert not sf.is_fitted
    with pytest.raises(RuntimeError, match="not fitted"):
        sf(jnp.zeros((2, 2)))


def test_fit_rejects_non_lonlat_inputs():
    sf = kl.SlepianFeatures(l_max=3, cap_radius=CAP, cap_centre=CENTRE)
    with pytest.raises(ValueError, match=r"\(N, 2\)"):
        sf.fit(kl.Chordal(kl.RBF()), jnp.zeros((4, 3)))


def test_fit_rejects_an_ard_kernel():
    sf = kl.SlepianFeatures(l_max=3, cap_radius=CAP, cap_centre=CENTRE)
    ard = kl.RBF(lengthscale=jnp.array([1.0, 2.0, 3.0]))
    with pytest.raises(NotImplementedError, match="per-dimension"):
        sf.fit(ard, jnp.zeros((4, 2)))


# --- numerics (slow: geonnax builds the basis and harmonics eagerly) --------


@pytest.mark.slow
def test_all_modes_recover_the_spherical_harmonic_gram():
    # With every mode kept C is orthogonal, so Phi Phi^T = Y diag(a^2 / (a +
    # eps)) Y^T: equal to Y diag(a) Y^T up to the jitter, ~1e-14 here.
    l_max = 5
    k = kl.Chordal(kl.Matern(nu=2.5, lengthscale=0.7))
    X = fibonacci_lonlat(40)
    sf = kl.SlepianFeatures(
        l_max=l_max,
        cap_radius=CAP,
        cap_centre=CENTRE,
        n_modes=(l_max + 1) ** 2,
        jitter=1e-14,
    ).fit(k, X)
    Phi = sf(X)
    assert Phi.shape == (40, 36)

    a = kl.funk_hecke_coefficients(k, l_max)[jnp.asarray(harmonic_degrees(l_max))]
    Y = real_spherical_harmonics(lonlat_to_unit(X), l_max)
    expected = einsum(einx.multiply("n m, m -> n m", Y, a), Y, "n m, p m -> n p")
    gram = einsum(Phi, Phi, "n k, p k -> n p")
    assert jnp.max(jnp.abs(gram - expected)) < 1e-8


@pytest.mark.slow
def test_shannon_truncation_approximates_the_chordal_matern_gram():
    inside, _ = _split_by_cap(4500)
    X = inside[:300]
    assert X.shape == (300, 2)
    k = kl.Chordal(kl.Matern(nu=2.5, lengthscale=0.5))
    sf = kl.SlepianFeatures(l_max=12, cap_radius=CAP, cap_centre=CENTRE).fit(k, X)
    assert sf.basis is not None and sf.basis.num_modes == 11  # Shannon number
    Phi = sf(X)
    K = k(X, X)
    err = jnp.linalg.norm(einsum(Phi, Phi, "n k, p k -> n p") - K) / jnp.linalg.norm(K)
    # Deterministic (fixed Fibonacci points, no randomness): measured 6.2e-3
    # once for l_max=12, 11 modes, lengthscale 0.5; pinned with headroom.
    assert err < 1e-2


@pytest.mark.slow
def test_features_are_concentrated_in_the_cap():
    inside, outside = _split_by_cap(4000)
    k = kl.Chordal(kl.Matern(nu=2.5, lengthscale=0.5))
    sf = kl.SlepianFeatures(l_max=12, cap_radius=CAP, cap_centre=CENTRE).fit(k, inside)
    ratio = jnp.mean(_norm_sq(sf(inside))) / jnp.mean(_norm_sq(sf(outside)))
    # Measured ~34 on these Fibonacci points; the spec asks for > 10.
    assert ratio > 10.0


@pytest.mark.slow
def test_moving_the_cap_moves_the_features():
    k = kl.Chordal(kl.Matern(nu=2.5, lengthscale=0.5))
    centre = jnp.asarray([CENTRE])
    antipode = jnp.asarray([[CENTRE[0] - 180.0, -CENTRE[1]]])
    sf = kl.SlepianFeatures(l_max=6, cap_radius=CAP, cap_centre=CENTRE).fit(k, centre)
    assert float(_norm_sq(sf(centre))[0]) > 100 * float(_norm_sq(sf(antipode))[0])


@pytest.mark.slow
def test_radians_match_degrees():
    k = kl.Chordal(kl.Matern(nu=2.5, lengthscale=0.5))
    X = fibonacci_lonlat(20)
    deg = kl.SlepianFeatures(l_max=4, cap_radius=CAP, cap_centre=CENTRE).fit(k, X)
    rad = kl.SlepianFeatures(
        l_max=4,
        cap_radius=CAP,
        cap_centre=(math.radians(CENTRE[0]), math.radians(CENTRE[1])),
        degrees=False,
    ).fit(k, X)
    assert jnp.allclose(deg(X), rad(jnp.deg2rad(X)), atol=1e-10)


@pytest.mark.slow
def test_gradient_reaches_the_lengthscale():
    X = fibonacci_lonlat(20)
    sf = kl.SlepianFeatures(l_max=4, cap_radius=CAP, cap_centre=CENTRE).fit(
        kl.Chordal(kl.Matern(nu=2.5, lengthscale=0.5)), X
    )

    def loss(ls):
        fitted = eqx.tree_at(lambda m: m.kernel.kernel.lengthscale, sf, ls)
        return jnp.sum(fitted(X) ** 2)

    g = eqx.filter_grad(loss)(jnp.asarray(0.5))
    assert jnp.isfinite(g) and g != 0.0
