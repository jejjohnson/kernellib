"""Sensitivity maps of predictors and dataset functionals (KD3)."""

from __future__ import annotations

import einx
import jax
import jax.numpy as jnp
import pytest

import kernellib as kl
from kernellib._derivatives._sensitivity import _derivative_matrix
from kernellib._einx import einsum, reduce


ELL = jnp.array([0.7, 1.3, 0.9])


def _krr(n=40, key=0):
    kx, ky = jax.random.split(jax.random.key(key))
    X = jax.random.normal(kx, (n, 3))
    y = jnp.sin(X[:, 0]) + 0.5 * X[:, 1] * X[:, 2] + 0.1 * jax.random.normal(ky, (n,))
    return kl.KRR(kl.RBF(lengthscale=ELL), regularization=1e-2).fit(X, y)


def _rbf_grad(model, Xq, ell):
    """Closed form sum_i alpha_i (-(x - x_i) / ell^2) k(x, x_i), shape (N, D)."""
    K = model.kernel(Xq, model.X_train)
    h = einx.subtract("n d, i d -> n i d", Xq, model.X_train)
    W = einx.multiply("n i, i -> n i", K, model.alpha)
    return -einsum(W, einx.divide("n i d, d -> n i d", h, ell**2), "n i, n i d -> n d")


# Linear predictor -----------------------------------------------------------


def test_linear_predictor_exact():
    a = jnp.array([2.0, -0.5, 1.5, 0.0])
    X = jax.random.normal(jax.random.key(0), (25, 4))
    f = lambda X: einsum(X, a, "n d, d -> n")
    assert jnp.allclose(kl.sensitivity(f, X), a**2, rtol=0, atol=1e-14)
    q = kl.point_sensitivity(f, X)
    assert q.shape == (25,)
    assert jnp.allclose(q, jnp.sum(a**2) / 4, rtol=0, atol=1e-14)
    assert jnp.allclose(
        kl.sensitivity(f, X, transform="abs"), jnp.abs(a), rtol=0, atol=1e-14
    )


# Kernel predictors ----------------------------------------------------------


def test_krr_rbf_matches_closed_form():
    model = _krr()
    Xq = jax.random.normal(jax.random.key(1), (30, 3))
    G = _rbf_grad(model, Xq, ELL)
    expected = reduce(G**2, "n d -> d", "mean")
    assert jnp.allclose(kl.sensitivity(model.predict, Xq), expected, atol=1e-10)
    assert jnp.allclose(
        kl.point_sensitivity(model.predict, Xq),
        reduce(G**2, "n d -> n", "mean"),
        atol=1e-10,
    )


def test_classifier_margin_chain_rule():
    model = _krr()
    Xq = jax.random.normal(jax.random.key(2), (30, 3))
    G = _rbf_grad(model, Xq, ELL)
    g = jnp.tanh(model.predict(Xq))
    expected = reduce(
        einx.multiply("n, n d -> n d", (1 - g**2) ** 2, G**2), "n d -> d", "mean"
    )
    got = kl.sensitivity(lambda X: jnp.tanh(model.predict(X)), Xq)
    assert jnp.allclose(got, expected, rtol=0, atol=1e-12)


def test_feature_ranking_drops_irrelevant_feature():
    X = jax.random.uniform(jax.random.key(3), (300, 3), minval=-1.0, maxval=1.0)
    y = jnp.sin(3 * X[:, 0]) + 0.3 * X[:, 1] + 0.0 * X[:, 2]
    kernel = kl.RBF(lengthscale=jnp.array([0.5, 2.0, 2.0]))
    model = kl.KRR(kernel, regularization=1e-3).fit(X, y)
    s = kl.sensitivity(model.predict, X)
    assert s[0] > s[1] > s[2]
    assert s[2] / s[0] < 1e-2


# The shared identities --------------------------------------------------------


def _hsic_problem(n=60, key=4):
    k1, k2 = jax.random.split(jax.random.key(key))
    X = jax.random.normal(k1, (n, 3))
    Y = X[:, :1] ** 2 + 0.5 * X[:, 2:3] + 0.1 * jax.random.normal(k2, (n, 1))
    kx, ky = kl.RBF(lengthscale=1.2), kl.RBF(lengthscale=0.8)
    return X, Y, kx, ky


@pytest.mark.parametrize("mode", ["point", "dataset"])
def test_mean_identity(mode):
    if mode == "point":
        f, X = _krr().predict, jax.random.normal(jax.random.key(5), (30, 3))
    else:
        X, Y, kx, ky = _hsic_problem()
        f = lambda X: kl.hsic(kx, ky, X, Y)
    s = kl.sensitivity(f, X, wrt=mode)
    q = kl.point_sensitivity(f, X, wrt=mode)
    G = _derivative_matrix(f, X, wrt=mode)
    frob = reduce(G**2, "n d ->", "sum") / G.size
    assert jnp.allclose(jnp.mean(s), jnp.mean(q), rtol=1e-12, atol=0)
    assert jnp.allclose(jnp.mean(s), frob, rtol=1e-12, atol=0)


# Normalisation ----------------------------------------------------------------


def test_normalize_sum_and_max():
    model = _krr()
    X = jax.random.normal(jax.random.key(6), (30, 3))
    s = kl.sensitivity(model.predict, X, normalize="sum")
    assert jnp.allclose(jnp.sum(s), 1.0, rtol=0, atol=1e-12)
    q = kl.point_sensitivity(model.predict, X, normalize="max")
    assert jnp.allclose(jnp.max(q), 1.0, rtol=0, atol=1e-15)
    assert jnp.all(q >= 0)


@pytest.mark.parametrize("normalize", [None, "sum", "max"])
def test_constant_function_gives_zeros_and_finite_gradient(normalize):
    X = jax.random.normal(jax.random.key(7), (20, 2))
    Xt = jax.random.normal(jax.random.key(8), (10, 2))

    def total(ell, alpha):
        f = lambda Z: einsum(kl.RBF(lengthscale=ell)(Z, Xt), alpha, "n i, i -> n")
        return jnp.sum(kl.sensitivity(f, X, normalize=normalize))

    zero = jnp.zeros(10)
    s = kl.sensitivity(lambda Z: 0.0 * Z[:, 0], X, normalize=normalize)
    assert jnp.array_equal(s, jnp.zeros(2))
    assert jnp.isfinite(jax.grad(total)(0.9, zero))
    alpha = jax.random.normal(jax.random.key(9), (10,))
    assert jnp.isfinite(jax.grad(total)(0.9, alpha))


# Dataset functionals: HSIC ------------------------------------------------------


def test_hsic_input_gradient_closed_form():
    X, Y, kx, ky = _hsic_problem()
    n = X.shape[0]
    hsic_of_X = lambda X: kl.hsic(kx, ky, X, Y)
    H = jnp.eye(n) - 1.0 / n
    A = H @ ky(Y, Y) @ H
    K = kx(X, X)
    h = einx.subtract("i d, j d -> i j d", X, X)
    dk = -einx.multiply("i j d, i j -> i j d", h, K) / 1.2**2  # d k(x_i, x_j) / d x_i
    expected = 2.0 / n**2 * einsum(A, dk, "i j, i j d -> i d")
    G = jax.grad(hsic_of_X)(X)
    assert jnp.allclose(G, expected, rtol=0, atol=1e-10)
    s = kl.sensitivity(hsic_of_X, X)
    assert jnp.allclose(s, reduce(G**2, "n d -> d", "mean"), rtol=0, atol=1e-14)


@pytest.mark.slow
def test_hsic_tutorial_ranks_irrelevant_feature_last():
    k1, k2 = jax.random.split(jax.random.key(7))
    X = jax.random.normal(k1, (500, 3))
    Y = X[:, :1] ** 2 + 0.5 * X[:, 2:3] + 0.1 * jax.random.normal(k2, (500, 1))
    kx = kl.RBF(lengthscale=kl.estimate_lengthscale(X, ard=True))
    ky = kl.RBF(lengthscale=kl.estimate_lengthscale(Y))
    s = kl.sensitivity(lambda X: kl.hsic(kx, ky, X, Y), X, normalize="sum")
    # The unused feature still moves the X Gram, so its finite-sample share is
    # not ~0 (0.158 here, vs 0.560 and 0.282): it ranks last, below 1/3.
    assert jnp.argmin(s) == 1
    assert jnp.argmax(s) == 0
    assert s[1] < 0.2


# Modes and errors ---------------------------------------------------------------


def test_auto_mode_detection():
    X = jax.random.normal(jax.random.key(10), (8, 2))
    f = lambda X: X[:, 0] * X[:, 1]
    F = lambda X: jnp.sum(X[:, 0] * X[:, 1])
    # Same G for the pointwise predictor and its sum (rows are independent).
    assert jnp.allclose(kl.sensitivity(f, X), kl.sensitivity(F, X), atol=1e-14)
    assert jnp.allclose(
        _derivative_matrix(f, X), _derivative_matrix(f, X, wrt="point"), atol=0
    )
    assert jnp.allclose(
        _derivative_matrix(F, X), _derivative_matrix(F, X, wrt="dataset"), atol=0
    )


def test_errors():
    X = jax.random.normal(jax.random.key(11), (8, 2))
    f = lambda X: X[:, 0]
    F = lambda X: jnp.sum(X)
    with pytest.raises(ValueError, match=r"f\(X\)\[:, c\]"):
        kl.sensitivity(lambda X: X, X)
    with pytest.raises(ValueError, match="wrt='point'"):
        kl.sensitivity(F, X, wrt="point")
    with pytest.raises(ValueError, match="wrt='dataset'"):
        kl.point_sensitivity(f, X, wrt="dataset")
    with pytest.raises(ValueError, match="order=2"):
        kl.sensitivity(F, X, order=2)
    with pytest.raises(ValueError, match="expected"):
        kl.sensitivity(lambda X: X[:3, 0], X)
    with pytest.raises(ValueError, match="transform"):
        kl.sensitivity(f, X, transform="cube")  # ty: ignore[invalid-argument-type]
    with pytest.raises(ValueError, match="normalize"):
        kl.sensitivity(f, X, normalize="l2")  # ty: ignore[invalid-argument-type]
    with pytest.raises(ValueError, match="wrt must"):
        kl.sensitivity(f, X, wrt="both")  # ty: ignore[invalid-argument-type]
    with pytest.raises(ValueError, match="order must"):
        kl.sensitivity(f, X, order=3)  # ty: ignore[invalid-argument-type]
    with pytest.raises(ValueError, match=r"\(N, D\)"):
        kl.sensitivity(f, X[:, 0])


def test_rough_kernel_guard():
    X = jax.random.normal(jax.random.key(12), (20, 2))
    model = kl.KRR(kl.Matern(nu=0.5), regularization=1e-2).fit(X, X[:, 0])
    with pytest.raises(ValueError):
        kl.sensitivity(model.predict, X)
    assert kl.sensitivity(model.predict, X, check=False).shape == (2,)


# Second order, batching, jit ----------------------------------------------------


def test_order_two_matches_hessian_diagonal():
    model = _krr()
    X = jax.random.normal(jax.random.key(13), (20, 3))
    Hd = einsum(kl.hessian(model.predict, X), jnp.eye(3), "n d e, d e -> n d")
    assert jnp.allclose(
        kl.sensitivity(model.predict, X, order=2),
        reduce(Hd**2, "n d -> d", "mean"),
        rtol=0,
        atol=1e-12,
    )
    assert jnp.allclose(
        kl.point_sensitivity(model.predict, X, order=2, transform="abs"),
        reduce(jnp.abs(Hd), "n d -> n", "mean"),
        rtol=0,
        atol=1e-12,
    )


def test_batch_size_and_jit_match_eager():
    model = _krr()
    X = jax.random.normal(jax.random.key(14), (40, 3))
    eager = kl.sensitivity(model.predict, X)
    batched = kl.sensitivity(model.predict, X, batch_size=16)
    assert jnp.allclose(eager, batched, rtol=1e-14, atol=0)  # vmap vs lax.map
    jitted = jax.jit(lambda X: kl.point_sensitivity(model.predict, X, normalize="sum"))
    assert jnp.allclose(
        jitted(X),
        kl.point_sensitivity(model.predict, X, normalize="sum"),
        rtol=1e-14,
        atol=0,
    )
