"""Tests for the classical geostatistics covariance models (GEO5)."""

from __future__ import annotations

import math

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import kernellib as kl
from kernellib import functional as F
from kernellib.functional._distances import _pairwise_sq_dist


# Reference profiles psi(r), written out from the formulas in the docstrings
# (Chiles & Delfiner 2012 §2.5; Wendland 1995; Gneiting & Schlather 2004).
def _ref_spherical(r: float) -> float:
    return 1.0 - 1.5 * r + 0.5 * r**3 if r < 1 else 0.0


def _ref_cubic(r: float) -> float:
    if r >= 1:
        return 0.0
    return 1.0 - 7.0 * r**2 + 35.0 / 4.0 * r**3 - 7.0 / 2.0 * r**5 + 3.0 / 4.0 * r**7


def _ref_pentaspherical(r: float) -> float:
    if r >= 1:
        return 0.0
    return 1.0 - 15.0 / 8.0 * r + 5.0 / 4.0 * r**3 - 3.0 / 8.0 * r**5


def _ref_hole_effect(r: float) -> float:
    return 1.0 if r == 0 else math.sin(r) / r


def _ref_wendland2(r: float) -> float:
    return (1.0 - r) ** 4 * (4.0 * r + 1.0) if r < 1 else 0.0


def _ref_wendland4(r: float) -> float:
    return (1.0 - r) ** 6 * (35.0 * r**2 + 18.0 * r + 3.0) / 3.0 if r < 1 else 0.0


def _ref_stable(alpha: float):
    return lambda r: math.exp(-(r**alpha))


def _ref_gen_cauchy(alpha: float, beta: float):
    return lambda r: (1.0 + r**alpha) ** (-beta / alpha)


CASES = [
    ("spherical", kl.Spherical, {}, _ref_spherical),
    ("cubic", kl.Cubic, {}, _ref_cubic),
    ("pentaspherical", kl.Pentaspherical, {}, _ref_pentaspherical),
    ("hole_effect", kl.HoleEffect, {}, _ref_hole_effect),
    ("wendland2", kl.Wendland, {"order": 2}, _ref_wendland2),
    ("wendland4", kl.Wendland, {"order": 4}, _ref_wendland4),
    ("stable0.5", kl.Stable, {"alpha": 0.5}, _ref_stable(0.5)),
    ("stable1.5", kl.Stable, {"alpha": 1.5}, _ref_stable(1.5)),
    ("stable2", kl.Stable, {"alpha": 2.0}, _ref_stable(2.0)),
    ("cauchy0.7", kl.GeneralizedCauchy, {"alpha": 0.7, "beta": 0.3}, None),
    ("cauchy2", kl.GeneralizedCauchy, {"alpha": 2.0, "beta": 3.0}, None),
]
CASES = [
    (name, cls, kw, ref or _ref_gen_cauchy(kw["alpha"], kw["beta"]))
    for name, cls, kw, ref in CASES
]
IDS = [c[0] for c in CASES]
LOW_DIM = [kl.Spherical, kl.Cubic, kl.Pentaspherical, kl.HoleEffect, kl.Wendland]
COMPACT = {"spherical", "cubic", "pentaspherical", "wendland2", "wendland4"}

FUNCTIONAL = {
    "spherical": lambda X1, X2, v, ls: F.spherical_kernel(X1, X2, v, ls),
    "cubic": lambda X1, X2, v, ls: F.cubic_kernel(X1, X2, v, ls),
    "pentaspherical": lambda X1, X2, v, ls: F.pentaspherical_kernel(X1, X2, v, ls),
    "hole_effect": lambda X1, X2, v, ls: F.hole_effect_kernel(X1, X2, v, ls),
    "wendland2": lambda X1, X2, v, ls: F.wendland_kernel(X1, X2, v, ls, 2),
    "wendland4": lambda X1, X2, v, ls: F.wendland_kernel(X1, X2, v, ls, order=4),
    "stable0.5": lambda X1, X2, v, ls: F.stable_kernel(X1, X2, v, ls, 0.5),
    "stable1.5": lambda X1, X2, v, ls: F.stable_kernel(X1, X2, v, ls, 1.5),
    "stable2": lambda X1, X2, v, ls: F.stable_kernel(X1, X2, v, ls, 2.0),
    "cauchy0.7": lambda X1, X2, v, ls: F.generalized_cauchy_kernel(
        X1, X2, v, ls, 0.7, jnp.array(0.3)
    ),
    "cauchy2": lambda X1, X2, v, ls: F.generalized_cauchy_kernel(
        X1, X2, v, ls, 2.0, jnp.array(3.0)
    ),
}


@pytest.mark.parametrize(("name", "cls", "kw", "ref"), CASES, ids=IDS)
def test_exact_values(name, cls, kw, ref):
    k = cls(variance=2.0, **kw)
    x = jnp.array([0.0])
    for r in (0.0, 0.25, 0.5, 0.99, 1.0, 2.0):
        val = float(k.pairwise(x, jnp.array([r])))
        assert val == pytest.approx(2.0 * ref(r), rel=1e-12, abs=1e-14), r
    assert float(k.pairwise(x, x)) == 2.0


@pytest.mark.parametrize("name", sorted(COMPACT))
def test_compact_models_are_exactly_zero_beyond_range(name):
    _, cls, kw, _ = CASES[IDS.index(name)]
    k = cls(lengthscale=0.5, **kw)
    X = jnp.array([[0.0, 0.0], [0.3, 0.4], [1.0, 1.0], [0.5, 0.0]])
    K = k(X, X)
    # Pairs at scaled distance >= 1 (raw distance >= 0.5).
    assert float(K[0, 1]) == 0.0
    assert float(K[0, 2]) == 0.0
    assert float(K[0, 3]) == 0.0
    assert float(K[1, 3]) > 0.0


@pytest.mark.parametrize("d", [1, 2, 3])
@pytest.mark.parametrize(("name", "cls", "kw", "ref"), CASES, ids=IDS)
def test_positive_definite(name, cls, kw, ref, d):
    n = 300
    X = jax.random.uniform(jax.random.key(d), (n, d), maxval=3.0)
    K = cls(**kw)(X, X)
    assert float(np.linalg.eigvalsh(np.asarray(K)).min()) >= -1e-10 * n


def test_hole_effect_is_not_positive_definite_in_4d():
    # Why the d <= 3 guard exists: sin(r)/r is indefinite in R^4.
    X = jax.random.uniform(jax.random.key(0), (300, 4), maxval=3.0)
    r = jnp.sqrt(_pairwise_sq_dist(X, X))
    K = jnp.sinc(r / jnp.pi)
    assert float(np.linalg.eigvalsh(np.asarray(K)).min()) < -0.1


@pytest.mark.parametrize(("name", "cls", "kw", "ref"), CASES, ids=IDS)
def test_gram_matches_functional_and_pairwise(name, cls, kw, ref):
    ls = jnp.array([0.7, 1.3])
    k = cls(lengthscale=ls, variance=1.5, **kw)
    X1 = jax.random.uniform(jax.random.key(1), (6, 2), maxval=2.0)
    X2 = jax.random.uniform(jax.random.key(2), (5, 2), maxval=2.0)
    K = k(X1, X2)
    assert jnp.allclose(K, FUNCTIONAL[name](X1, X2, jnp.array(1.5), ls))
    Kp = jax.vmap(lambda x: jax.vmap(lambda y: k.pairwise(x, y))(X2))(X1)
    assert jnp.allclose(K, Kp, atol=1e-12)
    assert jnp.allclose(k.diag(X1), 1.5)


def test_stable_alpha_two_is_rbf():
    ls = jnp.array([0.6, 1.1, 2.0])
    X = jax.random.normal(jax.random.key(3), (8, 3))
    K = kl.Stable(lengthscale=ls, variance=1.7, alpha=2.0)(X, X)
    K_rbf = kl.RBF(lengthscale=ls / math.sqrt(2.0), variance=1.7)(X, X)
    assert jnp.allclose(K, K_rbf, rtol=1e-12)


def test_stable_alpha_one_is_matern_half():
    X = jax.random.normal(jax.random.key(4), (8, 3))
    K = kl.Stable(lengthscale=0.8, alpha=1.0)(X, X)
    K_mat = kl.Matern(lengthscale=0.8, nu=0.5)(X, X)
    assert jnp.allclose(K, K_mat, rtol=1e-12)


@pytest.mark.parametrize("beta", [0.5, 1.0, 3.0])
def test_generalized_cauchy_alpha_two_is_rational_quadratic(beta):
    # GC(2, beta): (1 + d²/l²)^(-beta/2). kernellib's RQ is
    # (1 + d²/(2 a l_rq²))^(-a); matching the exponent gives a = beta/2, and
    # then 2 a l_rq² = beta l_rq² = l², so l_rq = l / sqrt(beta).
    ls = jnp.array([0.9, 1.4])
    X = jax.random.normal(jax.random.key(5), (8, 2))
    K = kl.GeneralizedCauchy(lengthscale=ls, alpha=2.0, beta=beta)(X, X)
    K_rq = kl.RationalQuadratic(lengthscale=ls / math.sqrt(beta), alpha=beta / 2.0)(
        X, X
    )
    assert jnp.allclose(K, K_rq, rtol=1e-12)


@pytest.mark.parametrize("cls", LOW_DIM)
def test_four_dimensions_raise(cls):
    k = cls()
    X = jnp.zeros((3, 4))
    with pytest.raises(ValueError, match="D <= 3"):
        k(X, X)
    with pytest.raises(ValueError, match="D <= 3"):
        k.pairwise(X[0], X[1])
    assert k(X[:, :3], X[:, :3]).shape == (3, 3)


@pytest.mark.parametrize(
    "fn",
    [
        F.spherical_kernel,
        F.cubic_kernel,
        F.pentaspherical_kernel,
        F.hole_effect_kernel,
        F.wendland_kernel,
    ],
)
def test_functional_four_dimensions_raise(fn):
    X = jnp.zeros((3, 4))
    with pytest.raises(ValueError, match="D <= 3"):
        fn(X, X, jnp.array(1.0), jnp.array(1.0))


def test_global_models_accept_any_dimension():
    X = jax.random.normal(jax.random.key(6), (5, 6))
    assert kl.Stable(alpha=1.5)(X, X).shape == (5, 5)
    assert kl.GeneralizedCauchy(alpha=1.0, beta=2.0)(X, X).shape == (5, 5)


@pytest.mark.parametrize("alpha", [0.0, -1.0, 2.5])
def test_invalid_alpha_raises(alpha):
    with pytest.raises(ValueError, match="alpha"):
        kl.Stable(alpha=alpha)
    with pytest.raises(ValueError, match="alpha"):
        kl.GeneralizedCauchy(alpha=alpha)
    X = jnp.zeros((2, 1))
    with pytest.raises(ValueError, match="alpha"):
        F.stable_kernel(X, X, jnp.array(1.0), jnp.array(1.0), alpha)


@pytest.mark.parametrize("beta", [0.0, -0.5])
def test_invalid_beta_raises(beta):
    with pytest.raises(ValueError, match="beta"):
        kl.GeneralizedCauchy(alpha=1.0, beta=beta)


def test_generalized_cauchy_builds_under_jit():
    @eqx.filter_jit
    def value(beta):
        k = kl.GeneralizedCauchy(alpha=1.0, beta=beta)
        return k.pairwise(jnp.array([0.0]), jnp.array([1.0]))

    assert float(value(jnp.array(2.0))) == pytest.approx(0.25)


def test_invalid_wendland_order_raises():
    with pytest.raises(ValueError, match="order"):
        kl.Wendland(order=3)  # ty: ignore[invalid-argument-type]
    X = jnp.zeros((2, 1))
    with pytest.raises(ValueError, match="order"):
        F.wendland_kernel(X, X, jnp.array(1.0), jnp.array(1.0), order=3)


@pytest.mark.parametrize(("name", "cls", "kw", "ref"), CASES, ids=IDS)
def test_gradients_finite_at_origin_and_beyond_range(name, cls, kw, ref):
    k = cls(lengthscale=jnp.array(1.0), **kw)
    x = jnp.array([0.2, -0.1])
    for y in (x, x + jnp.array([1.0, 0.0]), x + jnp.array([2.0, 0.0])):
        gx = jax.grad(lambda x_, y=y: k.pairwise(x_, y))(x)
        assert bool(jnp.all(jnp.isfinite(gx))), y
        gk = eqx.filter_grad(lambda k_, y=y: k_.pairwise(x, y))(k)
        leaves = jax.tree_util.tree_leaves(gk)
        assert all(bool(jnp.all(jnp.isfinite(g))) for g in leaves), y
    assert jnp.allclose(jax.grad(lambda x_: k.pairwise(x_, x))(x), 0.0)
    if name in COMPACT:
        gx = jax.grad(lambda x_: k.pairwise(x_, x + jnp.array([2.0, 0.0])))(x)
        assert jnp.allclose(gx, 0.0)


@pytest.mark.parametrize(
    ("cls", "kw", "curvature"),
    [
        # psi(r) = 1 - c r² + o(r²) gives the Hessian -2c/l² I at x = y.
        (kl.Cubic, {}, 7.0),
        (kl.HoleEffect, {}, 1.0 / 6.0),
        (kl.Wendland, {"order": 2}, 10.0),
        (kl.Wendland, {"order": 4}, 28.0 / 3.0),
        (kl.Stable, {"alpha": 2.0}, 1.0),
        (kl.GeneralizedCauchy, {"alpha": 2.0, "beta": 3.0}, 1.5),
    ],
)
def test_hessian_at_coincident_points(cls, kw, curvature):
    ls = 0.5
    k = cls(lengthscale=ls, **kw)
    x = jnp.array([0.3, -0.2, 0.1])
    H = jax.hessian(lambda x_: k.pairwise(x_, x))(x)
    assert jnp.allclose(H, -2.0 * curvature / ls**2 * jnp.eye(3), rtol=1e-8)


@pytest.mark.parametrize(("name", "cls", "kw", "ref"), CASES, ids=IDS)
def test_no_spectral_density(name, cls, kw, ref):
    with pytest.raises(NotImplementedError):
        cls(**kw).spectral_density(jnp.zeros((2, 1)))
