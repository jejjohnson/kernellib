"""HSIC, CKA and kernel alignment on kernels and data."""

from __future__ import annotations

import einx
import jax
import jax.numpy as jnp
import lineax as lx
import pytest

import kernellib as kl
from kernellib import functional as F


def _pair(n=40, key=0, dependent=True):
    k1, k2 = jax.random.split(jax.random.key(key))
    X = jax.random.normal(k1, (n, 2))
    noise = jax.random.normal(k2, (n, 1))
    Y = X[:, :1] ** 2 + 0.1 * noise if dependent else noise
    return X, Y


KX = kl.RBF(lengthscale=1.0)
KY = kl.Matern(nu=1.5, lengthscale=0.7, variance=2.0)


def _op(M):
    return lx.MatrixLinearOperator(M, lx.symmetric_tag)


@pytest.mark.slow
@pytest.mark.parametrize("estimator", ["biased", "unbiased"])
def test_dense_matches_functional(estimator):
    X, Y = _pair()
    expected = F.hsic(_op(KX(X, X)), _op(KY(Y, Y)), estimator=estimator)
    assert jnp.allclose(kl.hsic(KX, KY, X, Y, estimator=estimator), expected)


@pytest.mark.slow
@pytest.mark.parametrize("estimator", ["biased", "unbiased"])
def test_feature_path_is_exact_for_its_gram(estimator):
    # The feature formulas equal the dense estimator on Phi Phi^T exactly.
    X, Y = _pair()
    approx = kl.RandomFourierFeatures(16, jax.random.key(3))
    k_x, k_y = jax.random.split(jax.random.key(3))
    Phi_x = kl.RandomFourierFeatures(16, k_x).fit(KX, X)(X)
    Phi_y = kl.RandomFourierFeatures(16, k_y).fit(KY, Y)(Y)
    expected = F.hsic(_op(Phi_x @ Phi_x.T), _op(Phi_y @ Phi_y.T), estimator=estimator)
    got = kl.hsic(KX, KY, X, Y, estimator=estimator, approx=approx)
    assert jnp.allclose(got, expected)


@pytest.mark.slow
@pytest.mark.parametrize("estimator", ["biased", "unbiased"])
def test_full_nystrom_recovers_the_dense_value(estimator):
    X, Y = _pair(n=25)
    approx = kl.NystromFeatures(25, jax.random.key(0), jitter=1e-12)
    dense = kl.hsic(KX, KY, X, Y, estimator=estimator)
    got = kl.hsic(KX, KY, X, Y, estimator=estimator, approx=approx)
    assert jnp.allclose(got, dense, rtol=1e-5)


@pytest.mark.slow
def test_random_features_approximate_the_dense_value():
    X, Y = _pair(n=200)
    dense = kl.hsic(KX, KY, X, Y)
    got = kl.hsic(
        KX, KY, X, Y, approx=kl.RandomFourierFeatures(2048, jax.random.key(0))
    )
    assert jnp.abs(got - dense) < 0.05 * dense


@pytest.mark.slow
def test_detects_dependence():
    X, Y = _pair(n=150)
    _, Z = _pair(n=150, key=1, dependent=False)
    assert kl.hsic(KX, KY, X, Y) > 5 * kl.hsic(KX, KY, X, Z)
    # Unbiased HSIC is centred on zero under independence.
    assert jnp.abs(kl.hsic(KX, KY, X, Z, estimator="unbiased")) < 0.2 * kl.hsic(
        KX, KY, X, Y, estimator="unbiased"
    )


@pytest.mark.slow
@pytest.mark.parametrize("approx", [None, kl.NystromFeatures(30, jax.random.key(0))])
def test_cka_bounds_and_scale_invariance(approx):
    X, Y = _pair(n=30)
    c = kl.cka(KX, KY, X, Y, approx=approx)
    assert 0.0 <= c <= 1.0
    assert jnp.allclose(kl.cka(KX, KY * 7.0, X, Y, approx=approx), c, rtol=1e-6)
    assert jnp.allclose(kl.cka(KX, KX, X, X, approx=approx), 1.0, rtol=1e-5)


def test_cka_matches_functional():
    X, Y = _pair()
    expected = F.cka(_op(KX(X, X)), _op(KY(Y, Y)))
    assert jnp.allclose(kl.cka(KX, KY, X, Y), expected)


def test_kernel_alignment():
    X, Y = _pair()
    K, L = KX(X, X), KY(Y, Y)
    expected = jnp.sum(K * L) / (jnp.linalg.norm(K) * jnp.linalg.norm(L))
    assert jnp.allclose(kl.kernel_alignment(KX, KY, X, Y), expected)
    # Uncentred: a constant offset changes it, unlike CKA.
    shifted = KY + kl.Constant(5.0)
    assert not jnp.allclose(kl.kernel_alignment(KX, shifted, X, Y), expected)
    approx = kl.NystromFeatures(40, jax.random.key(0), jitter=1e-12)
    assert jnp.allclose(
        kl.kernel_alignment(KX, KY, X, Y, approx=approx), expected, rtol=1e-5
    )


@pytest.mark.slow
def test_gradients_for_bandwidth_selection():
    X, Y = _pair()

    def objective(ell):
        return kl.hsic(kl.RBF(lengthscale=ell), KY, X, Y)

    g = jax.grad(objective)(0.8)
    eps = 1e-5
    fd = (objective(0.8 + eps) - objective(0.8 - eps)) / (2 * eps)
    assert jnp.allclose(g, fd, rtol=1e-4)
    # Input gradients go through pairwise too.
    gX = jax.grad(lambda X: kl.hsic(KX, KY, X, Y))(X)
    assert gX.shape == X.shape and jnp.all(jnp.isfinite(gX))


@pytest.mark.slow
def test_errors():
    X, Y = _pair()
    with pytest.raises(ValueError, match="paired"):
        kl.hsic(KX, KY, X, Y[:-1])
    with pytest.raises(ValueError, match="estimator"):
        kl.cka(KX, KY, X, Y, estimator="u-stat")
    with pytest.raises(ValueError, match="n >= 4"):
        kl.hsic(
            KX,
            KY,
            X[:3],
            Y[:3],
            estimator="unbiased",
            approx=kl.RandomFourierFeatures(4, jax.random.key(0)),
        )


@pytest.mark.parametrize("dy", [1, 2, 4])
def test_linear_cka_is_the_rv_coefficient(dy):
    # Escoufier's RV coefficient from the (Dx, Dy) cross-covariance matrices;
    # linear CKA computes it in sample space from N x N Gram matrices.
    X = jax.random.normal(jax.random.key(0), (40, 3))
    Y = jnp.tanh(X[:, :1]) + 0.5 * jax.random.normal(jax.random.key(1), (40, dy))
    Xc, Yc = X - X.mean(0), Y - Y.mean(0)
    S_xy, S_xx, S_yy = Xc.T @ Yc, Xc.T @ Xc, Yc.T @ Yc
    rv = jnp.sum(S_xy**2) / (jnp.linalg.norm(S_xx) * jnp.linalg.norm(S_yy))
    got = kl.cka(kl.Linear(), kl.Linear(), X, Y)
    assert jnp.allclose(got, rv, rtol=1e-10)


# --- #93: CKA on degenerate inputs --------------------------------------------


def _rbf32(f):
    """An RBF Gram matrix computed entirely in float32 (the suite runs in x64)."""
    d2 = einx.subtract("i, j -> i j", f[:, 0], f[:, 0]) ** 2
    return jnp.exp(-d2 / jnp.float32(2.0))


@pytest.mark.parametrize("estimator", ["biased", "unbiased"])
@pytest.mark.parametrize("scale", [0.0, 1e-2, 1e-4, 1e-5])
def test_cka_degenerate_input_is_finite(estimator, scale):
    # A network output at init: constant or nearly so. Value and gradient must
    # stay finite so the penalty never poisons an optimiser (#93).
    q = jax.random.normal(jax.random.key(1), (64, 1))
    noise = jax.random.normal(jax.random.key(2), (64, 1))
    k = kl.RBF(lengthscale=1.0)

    def top(f):
        return kl.cka(k, k, f, q, estimator=estimator)

    value, grad = jax.value_and_grad(top)(scale * noise)
    assert jnp.isfinite(value) and bool(jnp.all(jnp.isfinite(grad)))
    if estimator == "biased":
        assert 0.0 <= float(value) <= 1.0

    # The same through the functional API, in float32 end to end.
    q32, noise32 = q.astype(jnp.float32), noise.astype(jnp.float32)
    L32 = _op(_rbf32(q32))

    def functional(f):
        return F.cka(_op(_rbf32(f)), L32, estimator=estimator)

    value, grad = jax.value_and_grad(functional)(jnp.float32(scale) * noise32)
    assert jnp.isfinite(value) and bool(jnp.all(jnp.isfinite(grad)))
    if estimator == "biased":
        assert 0.0 <= float(value) <= 1.0


@pytest.mark.parametrize("estimator", ["biased", "unbiased"])
def test_cka_constant_is_zero(estimator):
    X, _ = _pair()
    const = jnp.zeros((X.shape[0], 1))
    value, grad = jax.value_and_grad(
        lambda c: kl.cka(KX, KY, X, c, estimator=estimator)
    )(const)
    # Biased: exactly 0 (the self-HSIC is exactly 0). Unbiased: U-centring a
    # constant matrix leaves rounding, so 0 up to rounding.
    if estimator == "biased":
        assert float(value) == 0.0
    else:
        assert abs(float(value)) < 1e-12
    assert bool(jnp.all(jnp.isfinite(grad)))


def test_cka_nan_still_propagates():
    X, Y = _pair()
    X = X.at[0, 0].set(jnp.nan)
    assert jnp.isnan(kl.cka(KX, KY, X, Y))
