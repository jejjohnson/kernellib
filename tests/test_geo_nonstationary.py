"""`Gibbs` and `PaciorekSchervish`: stationary limits, the Gibbs / PS
equivalence, PSD under a sharp lengthscale transition, the unit diagonal and
gradients through an `eqx.nn.MLP` field."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

import kernellib as kl
from kernellib._einx import einsum, rearrange


BASES = [
    ("rbf", kl.RBF(lengthscale=0.7)),
    ("matern12", kl.Matern(nu=0.5, lengthscale=0.7)),
    ("matern32", kl.Matern(nu=1.5, lengthscale=0.7)),
    ("matern52", kl.Matern(nu=2.5, lengthscale=0.7)),
    ("rq", kl.RationalQuadratic(lengthscale=0.7, alpha=1.0)),
]


def _points(n, d=2, key=0):
    return jax.random.uniform(jax.random.key(key), (n, d), minval=-1.0, maxval=1.0)


def _sharp(x):
    return 0.05 + 0.5 * jax.nn.sigmoid(5.0 * x[0])


@pytest.mark.parametrize(("name", "stationary"), BASES)
def test_constant_lengthscale_recovers_stationary(name, stationary):
    X, Y = _points(7), _points(5, key=1)
    k = kl.Gibbs(lambda x: jnp.asarray(0.7), base=name)
    assert jnp.allclose(k(X, Y), stationary(X, Y), atol=1e-12)


@pytest.mark.parametrize(("name", "stationary"), BASES)
def test_constant_cov_is_ard(name, stationary):
    ell = jnp.array([0.4, 1.3])
    ard = eqx.tree_at(lambda s: s.lengthscale, stationary, ell)
    X, Y = _points(7), _points(5, key=1)
    k = kl.PaciorekSchervish(lambda x: jnp.diag(ell), base=name)
    assert jnp.allclose(k(X, Y), ard(X, Y), atol=1e-12)


def test_variance_scales():
    X = _points(4)
    k = kl.Gibbs(_sharp, variance=2.5)
    assert jnp.allclose(k(X, X), 2.5 * kl.Gibbs(_sharp)(X, X))
    assert jnp.allclose(k.diag(X), 2.5)


@pytest.mark.parametrize("name", [b[0] for b in BASES])
def test_gibbs_equals_ps_isotropic(name):
    X, Y = _points(8), _points(6, key=1)
    g = kl.Gibbs(_sharp, base=name)
    ps = kl.PaciorekSchervish(lambda x: _sharp(x) * jnp.eye(2), base=name)
    assert jnp.allclose(g(X, Y), ps(X, Y), atol=1e-12)


def test_gibbs_per_dimension_equals_ps_diagonal():
    def ells(x):
        return jnp.array([0.1 + 0.4 * jax.nn.sigmoid(x[0]), 0.3 + 0.2 * x[1] ** 2])

    X, Y = _points(8), _points(6, key=1)
    g = kl.Gibbs(ells, base="matern32")
    ps = kl.PaciorekSchervish(lambda x: jnp.diag(ells(x)), base="matern32")
    assert jnp.allclose(g(X, Y), ps(X, Y), atol=1e-12)


def test_gibbs_rbf_closed_form():
    x, y = jnp.array([0.2, -0.1]), jnp.array([0.5, 0.4])
    lx, ly = _sharp(x), _sharp(y)
    expected = (2 * lx * ly / (lx**2 + ly**2)) * jnp.exp(
        -jnp.sum((x - y) ** 2) / (lx**2 + ly**2)
    )
    assert jnp.allclose(kl.Gibbs(_sharp).pairwise(x, y), expected, atol=1e-12)


def test_ps_rotated_full_matrix():
    # A non-diagonal constant Sigma = L L^T is a stationary Mahalanobis kernel.
    L = jnp.array([[0.8, 0.0], [0.3, 0.5]])
    sigma = einsum(L, L, "i k, j k -> i j")
    x, y = jnp.array([0.2, -0.1]), jnp.array([0.5, 0.4])
    d = x - y
    q = jnp.dot(d, jnp.linalg.solve(sigma, d))
    k = kl.PaciorekSchervish(lambda x: L, base="matern52")
    expected = kl.Matern(nu=2.5).shape(q)
    assert jnp.allclose(k.pairwise(x, y), expected, atol=1e-12)


def _rotating_cov(x):
    th = 2.0 * x[0]
    R = jnp.array([[jnp.cos(th), -jnp.sin(th)], [jnp.sin(th), jnp.cos(th)]])
    return einsum(
        R,
        jnp.diag(jnp.array([0.1 + 0.3 * jax.nn.sigmoid(5 * x[1]), 0.8])),
        "i k, k j -> i j",
    )


@pytest.mark.parametrize(("name", "_"), BASES)
def test_psd_sharp_transition(name, _):
    X = jax.random.uniform(jax.random.key(3), (300, 2), minval=-1.0, maxval=1.0)
    n = X.shape[0]
    for k in (
        kl.Gibbs(_sharp, base=name),
        kl.PaciorekSchervish(_rotating_cov, base=name),
    ):
        K = k(X, X)
        assert jnp.allclose(K, rearrange(K, "i j -> j i"), atol=1e-12)
        assert float(jnp.linalg.eigvalsh(K).min()) >= -1e-10 * n


def test_diag_is_variance():
    X = _points(10)
    for k in (
        kl.Gibbs(_sharp, variance=1.7, base="matern52"),
        kl.PaciorekSchervish(_rotating_cov, variance=1.7, base="matern52"),
    ):
        assert jnp.allclose(k.diag(X), 1.7)
        assert jnp.allclose(jnp.diag(k(X, X)), 1.7)


def test_gradients_wrt_mlp_field_are_finite():
    X = _points(12)
    mlp = eqx.nn.MLP(2, "scalar", 8, 2, key=jax.random.key(0))
    mlp_cov = eqx.nn.MLP(2, 4, 8, 2, key=jax.random.key(1))

    def loss_g(field):
        k = kl.Gibbs(lambda x: 0.05 + jax.nn.softplus(field(x)))
        return jnp.sum(k(X, X))

    def loss_ps(field):
        def cov(x):
            m = rearrange(field(x), "(a b) -> a b", a=2)
            return jnp.tril(m, -1) + jnp.diag(0.1 + jax.nn.softplus(jnp.diagonal(m)))

        return jnp.sum(kl.PaciorekSchervish(cov, base="matern32")(X, X))

    for loss, field in ((loss_g, mlp), (loss_ps, mlp_cov)):
        g = eqx.filter_grad(loss)(field)
        leaves = jax.tree_util.tree_leaves(eqx.filter(g, eqx.is_array))
        assert leaves
        assert all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in leaves)
        assert any(bool(jnp.any(leaf != 0)) for leaf in leaves)


def test_float32_small_lengthscales_finite():
    X = _points(20).astype(jnp.float32)
    k = kl.Gibbs(lambda x: jnp.float32(1e-3) + 0.0 * x[0], base="rbf")
    K = k(X, X)
    assert bool(jnp.all(jnp.isfinite(K)))
    assert jnp.allclose(K, jnp.eye(20), atol=1e-6)


def test_invalid_inputs():
    with pytest.raises(ValueError, match="base"):
        kl.Gibbs(_sharp, base="spherical")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="base"):
        kl.PaciorekSchervish(_rotating_cov, base="wendland")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="cov_fn"):
        kl.PaciorekSchervish(lambda x: jnp.eye(3))(_points(2), _points(2))
    with pytest.raises(ValueError, match="lengthscale_fn"):
        kl.Gibbs(lambda x: jnp.ones(5))(_points(2), _points(2))
