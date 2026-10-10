"""Tests for the Gneiting non-separable space-time kernel (GEO12).

Inputs come from pinned keys: the randomness is incidental, so the
tolerances are deterministic.
"""

from __future__ import annotations

import einx
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

import kernellib as kl
from kernellib._einx import rearrange, reduce


X1 = jr.normal(jr.key(0), (7, 3))
X2 = jr.normal(jr.key(1), (5, 3))

PARAMS = {
    "variance": 1.3,
    "space_lengthscale": 0.8,
    "time_lengthscale": 1.4,
    "alpha": 0.8,
    "gamma": 0.7,
    "tau": 1.5,
}


def _gneiting(beta: float, **overrides) -> kl.Gneiting:
    return kl.Gneiting(space_dims=2, beta=beta, **(PARAMS | overrides))


def test_beta_zero_is_the_separable_product():
    # beta = 0: sigma^2 (a|u|^{2 alpha} + 1)^{-tau} exp(-c ||h||^{2 gamma}),
    # a stable model in space times a generalized Cauchy in time with
    # psi(r) = (1 + r^{2 alpha})^{-tau}, i.e. GeneralizedCauchy's
    # beta / alpha = tau at alpha_GC = 2 alpha.
    p = PARAMS
    space = kl.ActiveDims(
        kl.Stable(
            lengthscale=p["space_lengthscale"],
            variance=p["variance"],
            alpha=2.0 * p["gamma"],
        ),
        dims=(0, 1),
    )
    time = kl.ActiveDims(
        kl.GeneralizedCauchy(
            lengthscale=p["time_lengthscale"],
            alpha=2.0 * p["alpha"],
            beta=2.0 * p["alpha"] * p["tau"],
        ),
        dims=(2,),
    )
    expected = kl.Product(space, time)(X1, X2)
    assert jnp.allclose(_gneiting(0.0)(X1, X2), expected, rtol=0, atol=1e-12)


@pytest.mark.parametrize("beta", [0.0, 0.5, 1.0])
def test_zero_time_lag_is_purely_spatial(beta):
    k = _gneiting(beta)
    S = jr.normal(jr.key(2), (6, 2))
    T = jnp.full((6, 1), 0.3)
    X = jnp.concatenate([S, T], axis=1)
    c = PARAMS["space_lengthscale"] ** (-2.0 * PARAMS["gamma"])
    diff = einx.subtract("i d, j d -> i j d", S[:3], S)
    h = jnp.sqrt(reduce(diff**2, "i j d -> i j", "sum"))
    expected = PARAMS["variance"] * jnp.exp(-c * h ** (2.0 * PARAMS["gamma"]))
    assert jnp.allclose(k(X[:3], X), expected, rtol=0, atol=1e-12)


@pytest.mark.parametrize("beta", [0.0, 0.5, 1.0])
def test_zero_space_lag_is_purely_temporal(beta):
    k = _gneiting(beta)
    t = jnp.linspace(-2.0, 2.0, 6)
    X = jnp.concatenate([jnp.full((6, 2), 0.4), rearrange(t, "n -> n 1")], axis=1)
    a = PARAMS["time_lengthscale"] ** (-2.0 * PARAMS["alpha"])
    u = jnp.abs(t[0] - t)
    expected = PARAMS["variance"] * (a * u ** (2.0 * PARAMS["alpha"]) + 1.0) ** (
        -PARAMS["tau"]
    )
    assert jnp.allclose(k(X[:1], X)[0], expected, rtol=0, atol=1e-12)


def test_spatial_range_grows_with_the_time_lag():
    # The interaction a separable kernel cannot express: at beta > 0 the
    # spatial correlation C(h, u) / C(0, u) at a fixed h rises with |u|.
    k = _gneiting(1.0)
    origin = jnp.zeros(3)

    def spatial_corr(u):
        return k.pairwise(origin, jnp.array([0.5, 0.0, u])) / k.pairwise(
            origin, jnp.array([0.0, 0.0, u])
        )

    corrs = jnp.stack([spatial_corr(u) for u in (0.0, 1.0, 3.0)])
    assert bool(jnp.all(jnp.diff(corrs) > 0.0))
    separable = _gneiting(0.0)
    sep = [
        separable.pairwise(origin, jnp.array([0.5, 0.0, u]))
        / separable.pairwise(origin, jnp.array([0.0, 0.0, u]))
        for u in (0.0, 3.0)
    ]
    assert jnp.allclose(sep[0], sep[1], atol=1e-12)


@pytest.mark.parametrize("beta", [0.0, 0.5, 1.0])
def test_positive_definite(beta):
    # Bound from the issue: min eigenvalue >= -1e-10 n for n = 400 random
    # space-time points in d = 2 (float64 round-off on a unit-diagonal Gram).
    n = 400
    X = jr.uniform(jr.key(3), (n, 3), minval=-2.0, maxval=2.0)
    k = kl.Gneiting(space_dims=2, beta=beta, alpha=0.9, gamma=0.6)
    assert float(jnp.linalg.eigvalsh(k.gram(X)).min()) >= -1e-10 * n


@pytest.mark.parametrize("beta", [0.0, 0.5, 1.0])
def test_pointwise_diag_and_symmetry(beta):
    k = _gneiting(beta)
    assert k.is_pointwise
    assert k.is_stationary
    K_pw = jax.vmap(lambda x: jax.vmap(lambda y: k.pairwise(x, y))(X2))(X1)
    assert jnp.allclose(K_pw, k(X1, X2), atol=1e-12)
    assert jnp.allclose(k(X1, X2), rearrange(k(X2, X1), "i j -> j i"), atol=1e-12)
    assert jnp.allclose(k.diag(X1), jnp.diag(k.gram(X1)), atol=1e-12)
    assert jnp.allclose(k.diag(X1), PARAMS["variance"])


@pytest.mark.slow
def test_hyperparameter_grads_finite_at_zero_lags():
    # Rows 0/1 share a location (h = 0), rows 0/2 a time (u = 0), and the
    # diagonal has both; gamma, alpha < 1 put the safe-power branch to work.
    X = jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.5, -0.3, 0.0]])
    k = _gneiting(0.5)
    params, static = eqx.partition(k, eqx.is_inexact_array)
    grads = jax.grad(lambda p: jnp.sum(eqx.combine(p, static).gram(X)))(params)
    leaves = jax.tree_util.tree_leaves(grads)
    assert len(leaves) == 4  # variance, both lengthscales, beta
    assert all(bool(jnp.all(jnp.isfinite(g))) for g in leaves)
    x = X[0]
    g_pw = jax.grad(lambda p: eqx.combine(p, static).pairwise(x, x))(params)
    assert all(bool(jnp.isfinite(g)) for g in jax.tree_util.tree_leaves(g_pw))


def test_jit_and_vmap():
    k = _gneiting(0.5)
    K = eqx.filter_jit(lambda kern, A: kern(A, A))(k, X1)
    assert jnp.allclose(K, k(X1, X1), atol=1e-12)
    Ks = jax.vmap(lambda b: eqx.tree_at(lambda m: m.beta, k, b)(X1, X2))(
        jnp.array([0.0, 0.5, 1.0])
    )
    assert Ks.shape == (3, 7, 5)
    assert jnp.allclose(Ks[0], _gneiting(0.0)(X1, X2), atol=1e-12)


def test_tau_defaults_to_half_space_dims():
    assert kl.Gneiting(space_dims=2).tau == 1.0
    assert kl.Gneiting(space_dims=3).tau == 1.5
    assert kl.Gneiting(space_dims=3, tau=2).tau == 2.0


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"tau": 0.9}, "tau"),
        ({"alpha": 1.1}, "alpha"),
        ({"alpha": 0.0}, "alpha"),
        ({"gamma": 1.1}, "gamma"),
        ({"beta": 1.5}, "beta"),
        ({"beta": -0.1}, "beta"),
        ({"space_dims": 0}, "space_dims"),
    ],
)
def test_validation(kwargs, match):
    with pytest.raises(ValueError, match=match):
        kl.Gneiting(**({"space_dims": 2} | kwargs))


def test_rejects_wrong_input_width():
    k = kl.Gneiting(space_dims=2)
    with pytest.raises(ValueError, match="3 input columns"):
        k(jnp.ones((4, 2)), jnp.ones((4, 2)))
    with pytest.raises(ValueError, match="3 input columns"):
        k.pairwise(jnp.ones(4), jnp.ones(4))
