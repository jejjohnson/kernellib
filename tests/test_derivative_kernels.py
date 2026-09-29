"""Derivative kernels (#58)."""

from __future__ import annotations

import gaussx as gx
import jax
import jax.numpy as jnp
import pytest

import kernellib as kl


ELL, VAR = 0.7, 1.3
X = jax.random.normal(jax.random.key(0), (6, 2))


def _rbf_blocks(x, y):
    # Closed-form RBF derivatives, tau = x - y.
    tau = x - y
    k = VAR * jnp.exp(-0.5 * jnp.sum(tau**2) / ELL**2)
    dx = -tau / ELL**2 * k
    dxdy = (jnp.eye(2) / ELL**2 - jnp.outer(tau, tau) / ELL**4) * k
    return k, dx, -dx, dxdy


@pytest.mark.slow
def test_derivative_matches_closed_form_rbf():
    k = kl.RBF(lengthscale=ELL, variance=VAR)
    x, y = X[0], X[3]
    _, dx, dy, dxdy = _rbf_blocks(x, y)
    for i in range(2):
        assert jnp.allclose(kl.Derivative(k, dx=i).pairwise(x, y), dx[i])
        assert jnp.allclose(kl.Derivative(k, dy=i).pairwise(x, y), dy[i])
        for j in range(2):
            got = kl.Derivative(k, dx=i, dy=j).pairwise(x, y)
            assert jnp.allclose(got, dxdy[i, j])


@pytest.mark.slow
@pytest.mark.parametrize(
    "kernel",
    [kl.Matern(nu=2.5, lengthscale=0.6), kl.RationalQuadratic(alpha=1.5)],
    ids=["matern25", "rq"],
)
def test_derivative_matches_finite_differences(kernel):
    x, y, eps = X[1], X[4], 1e-5
    e = jnp.eye(2)
    fd = (
        kernel.pairwise(x + eps * e[0], y + eps * e[1])
        - kernel.pairwise(x + eps * e[0], y - eps * e[1])
        - kernel.pairwise(x - eps * e[0], y + eps * e[1])
        + kernel.pairwise(x - eps * e[0], y - eps * e[1])
    ) / (4 * eps**2)
    got = kl.Derivative(kernel, dx=0, dy=1).pairwise(x, y)
    assert jnp.allclose(got, fd, rtol=1e-5)


@pytest.mark.slow
def test_indexed_gram_is_the_block_covariance_and_psd():
    k = kl.RBF(lengthscale=ELL, variance=VAR)
    Xa = kl.derivative_inputs(X)
    K = kl.DerivativeIndexed(k)(Xa, Xa)
    assert K.shape == (18, 18)
    n = X.shape[0]
    # Value block, gradient-value block and a gradient-gradient block.
    assert jnp.allclose(K[:n, :n], k(X, X))
    assert jnp.allclose(K[n : 2 * n, :n], kl.Derivative(k, dx=0)(X, X))
    assert jnp.allclose(K[n : 2 * n, 2 * n :], kl.Derivative(k, dx=0, dy=1)(X, X))
    assert jnp.allclose(K, K.T)
    assert jnp.linalg.eigvalsh(K)[0] > -1e-10


@pytest.mark.slow
def test_diagonal_is_finite_at_coincident_points():
    Xa = kl.derivative_inputs(X)
    for k in (kl.RBF(), kl.Matern(nu=1.5), kl.Matern(nu=2.5), kl.Periodic()):
        d = kl.DerivativeIndexed(k).diag(Xa)
        assert jnp.all(jnp.isfinite(d)) and jnp.all(d > 0.0)


@pytest.mark.slow
@pytest.mark.parametrize("d", [1, 2])
def test_gp_conditioned_on_gradients_recovers_the_function(d):
    # f(x) = sum_d sin(2 x_d): observe the value at 3 points and the gradient
    # at 24 d points, predict values elsewhere. The 3 values alone are far
    # off (max error ~1.7 in 2-D); the gradients pin f down.
    f = lambda x: jnp.sum(jnp.sin(2.0 * x))
    X_obs = jax.random.uniform(jax.random.key(d), (24 * d, d), minval=-1.5, maxval=1.5)
    X_test = jax.random.uniform(jax.random.key(12), (20, d), minval=-1.0, maxval=1.0)
    k = kl.DerivativeIndexed(kl.RBF(lengthscale=0.6, variance=1.0))

    def max_error(Xa, y):
        alpha = gx.solve(kl.to_operator(k, Xa, noise=1e-8), y)
        pred = k(kl.derivative_inputs(X_test, dims=()), Xa) @ alpha
        return jnp.max(jnp.abs(pred - jax.vmap(f)(X_test)))

    values = kl.derivative_inputs(X_obs[:3], dims=())
    Xa = jnp.concatenate([values, kl.derivative_inputs(X_obs, values=False)])
    grads = jax.vmap(jax.grad(f))(X_obs)
    y = jnp.concatenate([jax.vmap(f)(X_obs[:3]), grads.T.ravel()])
    with_gradients = max_error(Xa, y)
    assert with_gradients < 5e-3
    if d == 2:
        assert max_error(values, jax.vmap(f)(X_obs[:3])) > 100 * with_gradients


@pytest.mark.parametrize(
    "kernel",
    [
        kl.Matern(nu=0.5),
        kl.RBF() + kl.White(0.1),
        2.0 * kl.Matern(nu=0.5),
        kl.Distance(exponent=1.5),
        kl.RBF() + kl.Distance(),
    ],
    ids=["matern05", "white", "scaled-matern05", "distance15", "sum-distance"],
)
def test_rough_kernels_are_rejected(kernel):
    with pytest.raises(ValueError, match="mean-square differentiable"):
        kl.Derivative(kernel, dx=0)
    with pytest.raises(ValueError, match="mean-square differentiable"):
        kl.DerivativeIndexed(kernel)


def test_distance_with_exponent_two_is_differentiable():
    # Distance(exponent=2) is the linear kernel: its GP is differentiable.
    k = kl.Derivative(kl.Distance(exponent=2.0), dx=0, dy=0)
    assert float(k.pairwise(jnp.zeros(1), jnp.zeros(1))) == 1.0


def test_config_errors():
    with pytest.raises(ValueError, match="dx, dy"):
        kl.Derivative(kl.RBF())
    with pytest.raises(ValueError, match="out of range"):
        kl.Derivative(kl.RBF(), dx=5).pairwise(X[0], X[1])
    with pytest.raises(ValueError, match="dims"):
        kl.derivative_inputs(X, dims=[2])
    with pytest.raises(TypeError, match="Gram-only"):

        class _GramOnly(kl.AbstractKernel):
            def __call__(self, A, B):
                return A @ B.T

        kl.Derivative(_GramOnly(), dx=0)


@pytest.mark.slow
def test_hyperparameter_gradients_are_finite():
    Xa = kl.derivative_inputs(X)
    y = jnp.linspace(-1.0, 1.0, Xa.shape[0])

    def nll(ell):
        op = kl.to_operator(
            kl.DerivativeIndexed(kl.RBF(lengthscale=ell)), Xa, noise=1e-3
        )
        return gx.logdet(op) + y @ gx.solve(op, y)

    g = jax.jit(jax.grad(nll))(jnp.array(0.8))
    assert jnp.isfinite(g) and g != 0.0
