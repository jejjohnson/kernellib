"""Tests for the great-circle kernels positive definite on the sphere (GEO3)."""

from __future__ import annotations

import math

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import kernellib as kl
from kernellib._testing import fibonacci_lonlat
from kernellib.functional import great_circle_distance


def _kernels(c: float) -> list[kl.AbstractGreatCircleKernel]:
    """One instance of every family (and parameter edge) at lengthscale ``c``."""
    kw = {"lengthscale": c, "degrees": True}
    return [
        kl.GreatCircleExponential(**kw),
        kl.GreatCirclePoweredExponential(alpha=0.5, **kw),
        kl.GreatCirclePoweredExponential(alpha=1.0, **kw),
        kl.GreatCircleCauchy(alpha=0.5, tau=0.5, **kw),
        kl.GreatCircleCauchy(alpha=1.0, tau=3.0, **kw),
        kl.GreatCircleSpherical(**kw),
        kl.GreatCircleAskey(tau=2.0, **kw),
        kl.GreatCircleAskey(tau=3.5, **kw),
        kl.GreatCircleWendland(order=2, **kw),
        kl.GreatCircleWendland(order=2, tau=5.0, **kw),
        kl.GreatCircleWendland(order=4, **kw),
    ]


_COMPACT = (kl.GreatCircleSpherical, kl.GreatCircleAskey, kl.GreatCircleWendland)


def _min_eigenvalues(n: int, c: float) -> dict[str, float]:
    X = fibonacci_lonlat(n)
    return {
        repr(k): float(np.linalg.eigvalsh(np.asarray(k.gram(X))).min())
        for k in _kernels(c)
    }


@pytest.mark.parametrize("c", [0.05, 0.5, math.pi])
def test_positive_definite_on_fibonacci_points(c: float) -> None:
    n = 200
    for name, lam_min in _min_eigenvalues(n, c).items():
        assert lam_min >= -1e-10 * n, (name, c, lam_min)


@pytest.mark.slow
@pytest.mark.parametrize("c", [0.5, 2.0, math.pi])
def test_positive_definite_at_dense_spacing(c: float) -> None:
    n = 1000
    # At this spacing invalid kernels fail: RBF of θ, or the spherical model
    # with 3t/2 in place of t/2, have eigenvalues far below zero.
    for name, lam_min in _min_eigenvalues(n, c).items():
        assert lam_min >= -1e-10 * n, (name, c, lam_min)


def test_rbf_on_great_circle_distance_is_not_positive_definite() -> None:
    # The counter-example the docs claim: exp(-θ²/2) with θ the great-circle
    # angle (lengthscale 1 rad) on 100 Fibonacci points has a clearly negative
    # eigenvalue (about -4.0e-3, pinned when this test was written).
    theta = great_circle_distance(fibonacci_lonlat(100), fibonacci_lonlat(100))
    K = np.asarray(jnp.exp(-0.5 * theta**2))
    lam_min = float(np.linalg.eigvalsh(K).min())
    assert lam_min < -1e-6
    assert lam_min == pytest.approx(-4.034e-3, rel=1e-3)


def test_spherical_profile_pinned_values() -> None:
    # ψ(t) = (1 + t/2)(1 - t)² = 1 - 3t/2 + t³/2 (Gneiting 2013, Table 1).
    k = kl.GreatCircleSpherical()
    t = jnp.array([0.0, 0.25, 0.5, 0.75, 1.0, 1.5])
    expected = 1.0 - 1.5 * t + 0.5 * t**3
    expected = jnp.where(t < 1.0, expected, 0.0)
    np.testing.assert_allclose(k.profile(t), expected, atol=1e-15)
    assert float(k.profile(jnp.array(0.5))) == pytest.approx(0.3125, abs=1e-15)


@pytest.mark.parametrize("c", [0.3, 1.0, math.pi])
def test_psi_zero_is_one_and_compact_support(c: float) -> None:
    t = jnp.array([0.0, 0.5, 0.999, 1.0, 1.5, 3.0])
    for k in _kernels(c):
        psi = k.profile(t)
        assert float(psi[0]) == 1.0
        if isinstance(k, _COMPACT):
            np.testing.assert_array_equal(psi[3:], 0.0)
            assert bool(jnp.all(psi[:3] > 0.0))
        else:
            assert bool(jnp.all(psi > 0.0))


def test_compact_kernel_is_exactly_zero_beyond_lengthscale_in_km() -> None:
    k = kl.GreatCircleWendland(lengthscale=1000.0, radius=kl.EARTH_RADIUS_KM)
    # 0, ~556 km, ~1112 km along the equator.
    X = jnp.array([[0.0, 0.0], [5.0, 0.0], [10.0, 0.0]])
    K = k(X, X)
    assert float(K[0, 1]) > 0.0
    assert float(K[0, 2]) == 0.0


def test_gram_matches_pairwise_and_diag() -> None:
    X = fibonacci_lonlat(7)
    for k in _kernels(0.8):
        k = eqx.tree_at(lambda m: m.variance, k, jnp.asarray(2.5))
        K = k(X, X)
        Kp = jax.vmap(lambda x, k=k: jax.vmap(lambda y: k.pairwise(x, y))(X))(X)
        np.testing.assert_allclose(K, Kp, atol=1e-14)
        np.testing.assert_allclose(k.diag(X), jnp.diag(K), atol=1e-14)


def test_radians_and_radius_conventions() -> None:
    Xd = jnp.array([[0.0, 0.0], [30.0, 20.0]])
    Xr = jnp.radians(Xd)
    k_deg = kl.GreatCircleSpherical(lengthscale=1.2)
    k_rad = kl.GreatCircleSpherical(lengthscale=1.2, degrees=False)
    k_km = kl.GreatCircleSpherical(
        lengthscale=1.2 * kl.EARTH_RADIUS_KM, radius=kl.EARTH_RADIUS_KM
    )
    np.testing.assert_allclose(k_deg(Xd, Xd), k_rad(Xr, Xr), atol=1e-14)
    np.testing.assert_allclose(k_deg(Xd, Xd), k_km(Xd, Xd), atol=1e-14)


@pytest.mark.parametrize(
    ("make", "match"),
    [
        (lambda: kl.GreatCirclePoweredExponential(alpha=1.5), "alpha"),
        (lambda: kl.GreatCirclePoweredExponential(alpha=0.0), "alpha"),
        (lambda: kl.GreatCircleCauchy(alpha=2.0), "alpha"),
        (lambda: kl.GreatCircleCauchy(tau=0.0), "tau"),
        (lambda: kl.GreatCircleAskey(tau=1.5), "tau"),
        (lambda: kl.GreatCircleWendland(order=3), "order"),  # ty: ignore[invalid-argument-type]
        (lambda: kl.GreatCircleWendland(order=2, tau=3.9), "tau"),
        (lambda: kl.GreatCircleWendland(order=4, tau=5.0), "tau"),
        (lambda: kl.GreatCircleSpherical(lengthscale=3.2), "π"),
        (lambda: kl.GreatCircleAskey(lengthscale=3.2), "π"),
        (lambda: kl.GreatCircleWendland(lengthscale=3.2), "π"),
        (
            lambda: kl.GreatCircleSpherical(
                lengthscale=20100.0, radius=kl.EARTH_RADIUS_KM
            ),
            "π",
        ),
    ],
)
def test_invalid_parameters_raise(make, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        make()


def test_lengthscale_pi_is_allowed_and_non_compact_have_no_bound() -> None:
    kl.GreatCircleSpherical(lengthscale=math.pi)
    kl.GreatCircleWendland(
        lengthscale=math.pi * kl.EARTH_RADIUS_KM, radius=kl.EARTH_RADIUS_KM
    )
    kl.GreatCircleExponential(lengthscale=10.0)


def test_traced_lengthscale_skips_the_concrete_check() -> None:
    @jax.jit
    def f(ell: jax.Array) -> jax.Array:
        X = jnp.array([[0.0, 0.0], [10.0, 0.0]])
        return kl.GreatCircleAskey(lengthscale=ell)(X, X)[0, 1]

    assert float(f(jnp.asarray(1.0))) > 0.0


@pytest.mark.parametrize("k", _kernels(0.5), ids=lambda k: type(k).__name__)
def test_gradients_are_finite_on_diagonal_and_beyond_support(
    k: kl.AbstractGreatCircleKernel,
) -> None:
    # Coincident points (the diagonal), a pair inside the support and a pair
    # beyond it (≈ 0.87 rad > c = 0.5).
    X = jnp.array([[0.0, 0.0], [0.0, 0.0], [10.0, 5.0], [50.0, 0.0]])

    def total(ell: jax.Array, X: jax.Array) -> jax.Array:
        return jnp.sum(eqx.tree_at(lambda m: m.lengthscale, k, ell)(X, X))

    g_ell, g_X = jax.grad(total, argnums=(0, 1))(jnp.asarray(0.5), X)
    assert bool(jnp.isfinite(g_ell))
    assert bool(jnp.all(jnp.isfinite(g_X)))


def test_gradient_beyond_support_is_zero() -> None:
    k = kl.GreatCircleAskey(lengthscale=0.5, degrees=False)
    x, y = jnp.array([0.0, 0.0]), jnp.array([1.0, 0.0])
    g = jax.grad(lambda x: k.pairwise(x, y))(x)
    np.testing.assert_array_equal(g, 0.0)
