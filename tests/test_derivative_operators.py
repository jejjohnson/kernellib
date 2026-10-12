"""Derivative operators of callables: gradient, jacobian, hessian, laplacian,
divergence and curl (KD1)."""

from __future__ import annotations

import einx
import jax
import jax.numpy as jnp
import pytest

import kernellib as kl
from kernellib._derivatives._operators import _resolve_mode
from kernellib._einx import einsum, rearrange, reduce


# Fixtures -----------------------------------------------------------------

ELL = jnp.array([0.7, 1.3, 0.9])
VAR = 1.7


def _data(n=40, d=3, key=0):
    kx, ky = jax.random.split(jax.random.key(key))
    X = jax.random.normal(kx, (n, d))
    y = jnp.sin(X[:, 0]) + 0.5 * X[:, 1] * X[:, 2] + 0.1 * jax.random.normal(ky, (n,))
    return X, y


def _rbf_model():
    X, y = _data()
    kernel = kl.RBF(lengthscale=ELL, variance=VAR)
    return kl.KRR(kernel, regularization=1e-2).fit(X, y)


def _closed_forms(model, Xq):
    """RBF ARD closed forms of the gradient, Hessian and Laplacian of the
    kernel expansion sum_i alpha_i k(x, x_i)."""
    Xt, alpha = model.X_train, model.alpha
    K = model.kernel(Xq, Xt)  # (n, i)
    h = einx.subtract("n d, i d -> n i d", Xq, Xt)
    u = einx.divide("n i d, d -> n i d", h, ELL**2)  # Lambda^{-1} h
    inv = 1.0 / ELL**2
    W = einx.multiply("n i, i -> n i", K, alpha)  # alpha_i k(x_n, x_i)
    grad = -einsum(W, u, "n i, n i d -> n d")
    Wu = einx.multiply("n i, n i d -> n i d", W, u)
    outer = einsum(Wu, u, "n i d, n i e -> n d e")
    weight = reduce(W, "n i -> n", "sum")
    hess = outer - einx.multiply("n, d e -> n d e", weight, jnp.diag(inv))
    q = reduce(u**2, "n i d -> n i", "sum") - jnp.sum(inv)
    lap = einsum(W, q, "n i, n i -> n")
    return grad, hess, lap


# Closed forms on a KRR(RBF ARD) predictor ----------------------------------


def test_gradient_matches_rbf_closed_form():
    model = _rbf_model()
    Xq = jax.random.normal(jax.random.key(1), (50, 3))
    grad, _, _ = _closed_forms(model, Xq)
    assert jnp.allclose(kl.gradient(model.predict, Xq), grad, rtol=0, atol=1e-10)


def test_hessian_matches_rbf_closed_form_and_is_symmetric():
    model = _rbf_model()
    Xq = jax.random.normal(jax.random.key(2), (50, 3))
    _, hess, _ = _closed_forms(model, Xq)
    H = kl.hessian(model.predict, Xq)
    assert H.shape == (50, 3, 3)
    assert jnp.allclose(H, hess, rtol=0, atol=1e-10)
    assert jnp.allclose(H, rearrange(H, "n d e -> n e d"), rtol=0, atol=1e-12)


def test_laplacian_is_hessian_trace_and_closed_form():
    model = _rbf_model()
    Xq = jax.random.normal(jax.random.key(3), (50, 3))
    _, _, lap = _closed_forms(model, Xq)
    L = kl.laplacian(model.predict, Xq)
    H = kl.hessian(model.predict, Xq)
    trace = einsum(H, jnp.eye(3), "n d e, d e -> n")
    assert jnp.allclose(L, trace, rtol=0, atol=1e-10)
    assert jnp.allclose(L, lap, rtol=0, atol=1e-10)


def test_derivatives_at_training_points_are_finite_and_exact():
    model = _rbf_model()
    Xt = model.X_train
    grad, hess, lap = _closed_forms(model, Xt)
    G = kl.gradient(model.predict, Xt)
    H = kl.hessian(model.predict, Xt)
    L = kl.laplacian(model.predict, Xt)
    assert jnp.all(jnp.isfinite(G)) and jnp.all(jnp.isfinite(H))
    assert jnp.allclose(G, grad, rtol=0, atol=1e-10)
    assert jnp.allclose(H, hess, rtol=0, atol=1e-10)
    assert jnp.allclose(L, lap, rtol=0, atol=1e-10)


def test_gradient_matches_finite_differences_matern52():
    X, y = _data()
    model = kl.KRR(kl.Matern(nu=2.5, lengthscale=0.8), regularization=1e-2).fit(X, y)
    Xq = jax.random.normal(jax.random.key(4), (10, 3))
    eps = 1e-5
    E = eps * jnp.eye(3)
    fd = jnp.stack(
        [
            (model.predict(Xq + E[j]) - model.predict(Xq - E[j])) / (2 * eps)
            for j in range(3)
        ],
        axis=-1,
    )
    G = kl.gradient(model.predict, Xq)
    assert jnp.allclose(G, fd, rtol=1e-6, atol=1e-8)


def test_matern32_predictor_has_a_hessian():
    # Matern 3/2 is C^2 at x = x_i (k = 1 - 3/2 r^2 + sqrt(3) r^3 + ...).
    X, y = _data(n=10)
    model = kl.KRR(kl.Matern(nu=1.5), regularization=1e-2).fit(X, y)
    H = kl.hessian(model.predict, X)
    assert jnp.all(jnp.isfinite(H))


# Jacobian modes ------------------------------------------------------------


def _field(m):
    W = jax.random.normal(jax.random.key(m), (m, 5))

    def F(X):  # (N, D) -> (N, m)
        Z = einsum(X, W[:, : X.shape[1]], "n d, m d -> n m")
        scale = reduce(X**2, "n d -> n", "sum")
        return einx.multiply("n m, n -> n m", jnp.sin(Z), scale)

    return F


@pytest.mark.parametrize(("d", "m"), [(3, 5), (5, 2)])
def test_jacobian_forward_equals_reverse(d, m):
    F = _field(m)
    X = jax.random.normal(jax.random.key(5), (8, d))
    Jf = kl.jacobian(F, X, mode="forward")
    Jr = kl.jacobian(F, X, mode="reverse")
    assert Jf.shape == (8, m, d)
    assert jnp.allclose(Jf, Jr, rtol=0, atol=1e-12)
    assert jnp.allclose(kl.jacobian(F, X), Jf, rtol=0, atol=1e-12)


def test_auto_mode_picks_forward_iff_d_le_m():
    assert _resolve_mode("auto", 3, 5) == "forward"
    assert _resolve_mode("auto", 3, 3) == "forward"
    assert _resolve_mode("auto", 5, 2) == "reverse"
    assert _resolve_mode("reverse", 3, 5) == "reverse"
    with pytest.raises(ValueError, match="mode"):
        _resolve_mode("sideways", 3, 5)  # ty: ignore[invalid-argument-type]


def test_auto_mode_dispatches_through_jacfwd_or_jacrev(monkeypatch):
    calls = []
    jacfwd, jacrev = jax.jacfwd, jax.jacrev

    def spy(name, fn):
        def wrapped(*args, **kwargs):
            calls.append(name)
            return fn(*args, **kwargs)

        return wrapped

    monkeypatch.setattr(jax, "jacfwd", spy("forward", jacfwd))
    monkeypatch.setattr(jax, "jacrev", spy("reverse", jacrev))
    X3 = jax.random.normal(jax.random.key(6), (4, 3))
    X5 = jax.random.normal(jax.random.key(6), (4, 5))
    kl.jacobian(_field(5), X3)
    kl.jacobian(_field(2), X5)
    assert calls == ["forward", "reverse"]


def test_multi_output_krr_jacobian_rows_are_gradients():
    X, y = _data()
    Y = jnp.stack([y, jnp.cos(y)], axis=-1)
    model = kl.KRR(kl.RBF(lengthscale=ELL), regularization=1e-2).fit(X, Y)
    Xq = jax.random.normal(jax.random.key(7), (6, 3))
    J = kl.jacobian(model.predict, Xq)
    for c in range(2):
        g = kl.gradient(lambda Z, c=c: model.predict(Z)[:, c], Xq)
        assert jnp.allclose(J[:, c], g, rtol=0, atol=1e-12)


# Vector-calculus identities --------------------------------------------------


def phi2(X):
    return jnp.sin(X[:, 0]) * jnp.cos(2.0 * X[:, 1]) + X[:, 0] ** 3 * X[:, 1]


def phi3(X):
    return jnp.exp(-0.3 * X[:, 0] ** 2) * X[:, 1] + jnp.sin(X[:, 1] * X[:, 2])


def A3(X):  # a 3-D vector potential
    return jnp.stack(
        [X[:, 1] * X[:, 2] ** 2, jnp.sin(X[:, 0] * X[:, 2]), X[:, 0] ** 2 * X[:, 1]],
        axis=-1,
    )


def test_divergence_of_gradient_is_laplacian():
    X = jax.random.normal(jax.random.key(8), (20, 3))
    div = kl.divergence(lambda Z: kl.gradient(phi3, Z), X)
    assert jnp.allclose(div, kl.laplacian(phi3, X), rtol=0, atol=1e-10)


def test_jacobian_of_gradient_is_hessian():
    X = jax.random.normal(jax.random.key(9), (20, 3))
    J = kl.jacobian(lambda Z: kl.gradient(phi3, Z), X)
    assert jnp.allclose(J, kl.hessian(phi3, X), rtol=0, atol=1e-12)


@pytest.mark.parametrize("phi", [phi2, phi3])
def test_curl_of_gradient_vanishes(phi):
    d = 2 if phi is phi2 else 3
    X = jax.random.normal(jax.random.key(10), (20, d))
    c = kl.curl(lambda Z: kl.gradient(phi, Z), X)
    assert c.shape == ((20,) if d == 2 else (20, 3))
    assert jnp.max(jnp.abs(c)) < 1e-10


def test_divergence_of_rotated_gradient_vanishes_2d():
    X = jax.random.normal(jax.random.key(11), (20, 2))

    def rot_grad(Z):  # rot(a, b) = (b, -a)
        g = kl.gradient(phi2, Z)
        return jnp.stack([g[:, 1], -g[:, 0]], axis=-1)

    assert jnp.max(jnp.abs(kl.divergence(rot_grad, X))) < 1e-10
    # Control: its curl is minus the Laplacian of phi, not zero.
    assert jnp.allclose(
        kl.curl(rot_grad, X), -kl.laplacian(phi2, X), rtol=0, atol=1e-10
    )


def test_divergence_of_curl_vanishes_3d():
    X = jax.random.normal(jax.random.key(12), (20, 3))
    div = kl.divergence(lambda Z: kl.curl(A3, Z), X)
    assert jnp.max(jnp.abs(div)) < 1e-10
    assert jnp.max(jnp.abs(kl.curl(A3, X))) > 0.1  # control


def test_analytic_fields():
    X2 = jax.random.normal(jax.random.key(13), (10, 2))
    X3 = jax.random.normal(jax.random.key(14), (10, 3))

    def rotation(X):
        return jnp.stack([-X[:, 1], X[:, 0]], axis=-1)

    assert jnp.allclose(kl.curl(rotation, X2), 2.0, rtol=0, atol=1e-12)
    assert jnp.allclose(kl.divergence(rotation, X2), 0.0, rtol=0, atol=1e-12)
    assert jnp.allclose(kl.divergence(lambda X: X, X3), 3.0, rtol=0, atol=1e-12)
    assert jnp.allclose(kl.curl(lambda X: X, X3), 0.0, rtol=0, atol=1e-12)


# Vector-field GPs (GEO15) ----------------------------------------------------


def _vector_field_model(kernel):
    X = jax.random.uniform(jax.random.key(15), (25, 2), minval=-1.0, maxval=1.0)
    Y = jax.random.normal(jax.random.key(16), (25, 2))
    model = kl.KRR(kernel, regularization=1e-4)
    model = model.fit(kl.vector_inputs(X), rearrange(Y, "n i -> (n i)"))

    def F(Z):  # (N, 2) -> (N, 2)
        return rearrange(model.predict(kl.vector_inputs(Z)), "(n i) -> n i", i=2)

    return F


@pytest.mark.slow
def test_divergence_free_posterior_mean_has_zero_divergence():
    F = _vector_field_model(kl.DivergenceFree(kl.RBF(lengthscale=0.5)))
    Xq = jax.random.uniform(jax.random.key(17), (100, 2), minval=-1.0, maxval=1.0)
    assert jnp.max(jnp.abs(kl.divergence(F, Xq))) < 1e-8
    assert jnp.max(jnp.abs(kl.curl(F, Xq))) > 1e-2  # control


@pytest.mark.slow
def test_curl_free_posterior_mean_has_zero_curl():
    F = _vector_field_model(kl.CurlFree(kl.RBF(lengthscale=0.5)))
    Xq = jax.random.uniform(jax.random.key(18), (100, 2), minval=-1.0, maxval=1.0)
    assert jnp.max(jnp.abs(kl.curl(F, Xq))) < 1e-8
    assert jnp.max(jnp.abs(kl.divergence(F, Xq))) > 1e-2  # control


# Batching ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("op", "fn", "d"),
    [
        (kl.gradient, phi3, 3),
        (kl.hessian, phi3, 3),
        (kl.laplacian, phi3, 3),
        (kl.jacobian, A3, 3),
        (kl.divergence, A3, 3),
        (kl.curl, A3, 3),
    ],
)
def test_batch_size_matches_vmap(op, fn, d):
    X = jax.random.normal(jax.random.key(19), (50, d))
    assert jnp.allclose(op(fn, X, batch_size=7), op(fn, X), rtol=0, atol=1e-14)


def test_batch_size_must_be_positive():
    with pytest.raises(ValueError, match="batch_size"):
        kl.gradient(phi3, jnp.ones((4, 3)), batch_size=0)


# Guard ---------------------------------------------------------------------


@pytest.mark.parametrize("op", [kl.gradient, kl.hessian, kl.laplacian])
def test_rough_kernel_predictor_is_refused(op):
    X, y = _data(n=10)
    model = kl.KRR(kl.Matern(nu=0.5), regularization=1e-2).fit(X, y)
    with pytest.raises(ValueError, match=op.__name__):
        op(model.predict, X)
    assert jnp.all(jnp.isfinite(op(model.predict, X[:2] + 0.1, check=False)))


def test_rough_composite_and_distance_are_refused():
    X, y = _data(n=10)
    for kernel in (kl.RBF() + kl.White(1e-2), kl.Distance(exponent=1.5)):
        model = kl.KRR(kernel, regularization=1e-2).fit(X, y)
        with pytest.raises(ValueError, match="gradient"):
            kl.gradient(model.predict, X)


def test_opaque_callable_is_not_checked():
    X, y = _data(n=10)
    model = kl.KRR(kl.Matern(nu=0.5), regularization=1e-2).fit(X, y)
    kernel, Xt, alpha = model.kernel, model.X_train, model.alpha

    def expansion(Z):
        return kernel(Z, Xt) @ alpha

    G = kl.gradient(expansion, X[:3] + 0.1)
    assert G.shape == (3, 3)


def test_smooth_kernel_predictor_passes_the_guard():
    X, y = _data(n=10)
    model = kl.KRR(kl.Matern(nu=1.5) * kl.RBF(), regularization=1e-2).fit(X, y)
    assert kl.hessian(model.predict, X[:2]).shape == (2, 3, 3)


# Shape validation ------------------------------------------------------------


def test_scalar_operators_reject_vector_fields():
    X = jnp.ones((4, 3))
    for op in (kl.gradient, kl.hessian, kl.laplacian):
        with pytest.raises(ValueError, match=r"\(N, 3\)"):
            op(A3, X)


def test_jacobian_rejects_scalar_fields():
    with pytest.raises(ValueError, match="vector field"):
        kl.jacobian(phi3, jnp.ones((4, 3)))


def test_divergence_needs_square_field():
    with pytest.raises(ValueError, match="M = 2"):
        kl.divergence(_field(2), jnp.ones((4, 3)))


def test_curl_needs_two_or_three_dimensions():
    with pytest.raises(ValueError, match="D = 2"):
        kl.curl(lambda X: X, jnp.ones((4, 4)))


def test_x_must_be_two_dimensional():
    with pytest.raises(ValueError, match=r"\(N, D\)"):
        kl.gradient(phi3, jnp.ones(3))


def test_f_must_keep_the_batch_axis():
    with pytest.raises(ValueError, match="1, 3"):
        kl.gradient(lambda X: jnp.sum(X), jnp.ones((4, 3)))


# Transformations -------------------------------------------------------------


@pytest.mark.slow
def test_jit_and_hyperparameter_gradient_through_gradient():
    X, y = _data(n=20)
    Xq = jax.random.normal(jax.random.key(20), (5, 3))

    def objective(log_ell):
        model = kl.KRR(kl.RBF(lengthscale=jnp.exp(log_ell)), regularization=1e-2)
        model = model.fit(X, y)
        return jnp.sum(kl.gradient(model.predict, Xq) ** 2)

    value = jax.jit(objective)(0.1)
    assert jnp.isfinite(value)
    assert jnp.allclose(value, objective(0.1), rtol=1e-12)
    g = jax.grad(objective)(0.1)
    eps = 1e-5
    fd = (objective(0.1 + eps) - objective(0.1 - eps)) / (2 * eps)
    assert jnp.isfinite(g)
    assert jnp.allclose(g, fd, rtol=1e-5)
