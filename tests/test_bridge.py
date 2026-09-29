"""Tests for the kernel-to-operator bridge (``to_operator``, ``to_cross_operator``).

Pinned keys throughout: inputs are incidental to what is checked.
"""

from __future__ import annotations

import gaussx as gx
import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
import pytest

import kernellib as kl


X = jr.normal(jr.key(0), (30, 2))
Z = jr.normal(jr.key(1), (8, 2))
Y = jr.normal(jr.key(2), (30,))
NOISE = 0.1


def _kernel(lengthscale=0.7):
    return kl.RBF(lengthscale=lengthscale, variance=1.2) + kl.Linear(variance=0.3)


class _GramOnly(kl.AbstractKernel):
    def __call__(self, A, B):
        return A @ B.T


def test_dense_operator_matches_gram_plus_noise():
    op = kl.to_operator(_kernel(), X, noise=NOISE)
    expected = _kernel()(X, X) + NOISE * jnp.eye(30)
    assert isinstance(op, lx.MatrixLinearOperator)
    assert jnp.allclose(op.as_matrix(), expected, atol=1e-12)
    assert lx.is_symmetric(op)
    assert lx.is_positive_semidefinite(op)


def test_implicit_operator_matches_dense():
    dense = kl.to_operator(_kernel(), X, noise=NOISE)
    implicit = kl.to_operator(_kernel(), X, noise=NOISE, implicit=True)
    assert isinstance(implicit, kl.ImplicitKernelOperator)
    assert jnp.allclose(implicit.mv(Y), dense.mv(Y), atol=1e-10)
    assert jnp.allclose(implicit.as_matrix(), dense.as_matrix(), atol=1e-10)
    assert lx.is_positive_semidefinite(implicit)


def test_cg_solve_through_implicit_operator_matches_dense_solve():
    implicit = kl.to_operator(_kernel(), X, noise=NOISE, implicit=True)
    dense = kl.to_operator(_kernel(), X, noise=NOISE)
    x_cg = gx.CGSolver(rtol=1e-10, atol=1e-10, max_steps=500).solve(implicit, Y)
    x_dense = jnp.linalg.solve(dense.as_matrix(), Y)
    assert jnp.allclose(x_cg, x_dense, atol=1e-6)


def test_solve_gradient_wrt_lengthscale_matches_dense():
    """Exercises the implicit operator's custom JVP with kernellib params."""

    def loss_implicit(ls):
        op = kl.to_operator(_kernel(ls), X, noise=NOISE, implicit=True)
        return jnp.sum(
            gx.solve(op, Y, solver=lx.CG(rtol=1e-12, atol=1e-12, max_steps=1000))
        )

    def loss_dense(ls):
        return jnp.sum(
            jnp.linalg.solve(kl.to_operator(_kernel(ls), X, noise=NOISE).as_matrix(), Y)
        )

    g_implicit = jax.grad(loss_implicit)(jnp.array(0.7))
    g_dense = jax.grad(loss_dense)(jnp.array(0.7))
    assert jnp.allclose(g_implicit, g_dense, rtol=1e-5)


def test_cross_operator_dense_and_implicit_match():
    k = _kernel()
    dense = kl.to_cross_operator(k, X, Z)
    implicit = kl.to_cross_operator(k, X, Z, implicit=True, batch_size=7)
    assert isinstance(implicit, kl.ImplicitCrossKernelOperator)
    v = jnp.arange(8.0)
    assert jnp.allclose(dense.as_matrix(), k(X, Z), atol=1e-12)
    assert jnp.allclose(implicit.mv(v), dense.mv(v), atol=1e-10)


def test_implicit_rejects_gram_only_kernel():
    with pytest.raises(TypeError, match="pointwise"):
        kl.to_operator(_GramOnly(), X, implicit=True)
    with pytest.raises(TypeError, match="pointwise"):
        kl.to_cross_operator(_GramOnly(), X, Z, implicit=True)


def test_dense_accepts_gram_only_kernel():
    op = kl.to_operator(_GramOnly(), X, noise=NOISE)
    assert jnp.allclose(op.as_matrix(), X @ X.T + NOISE * jnp.eye(30))


def test_implicit_rejects_traced_noise():
    def build(noise):
        return kl.to_operator(_kernel(), X, noise=noise, implicit=True).mv(Y)

    with pytest.raises(TypeError, match="concrete noise"):
        jax.jit(build)(jnp.array(NOISE))


def test_dense_noise_can_be_differentiated():
    def loss(noise):
        return jnp.trace(kl.to_operator(_kernel(), X, noise=noise).as_matrix())

    assert jnp.allclose(jax.grad(loss)(jnp.array(NOISE)), 30.0)


# ---------------------------------------------------------------------------
# Structured Gram operators (#55)
# ---------------------------------------------------------------------------

STRUCTURED = [
    (kl.White(0.3), lx.TaggedLinearOperator),
    (kl.Constant(2.0), gx.LowRankUpdate),
    (kl.Linear(), gx.LowRankUpdate),
    (kl.Linear(variance=0.4, bias=0.5) + kl.White(0.1), gx.LowRankUpdate),
    (2.0 * (kl.Linear() + kl.Constant()) + kl.White(0.2), gx.LowRankUpdate),
    (kl.ActiveDims(kl.Linear(), (1,)) + kl.White(0.1), gx.LowRankUpdate),
    (kl.Warped(kl.Linear(), lambda x: jnp.tanh(x)) + kl.White(0.1), gx.LowRankUpdate),
    (kl.RBF() + kl.White(0.1), lx.MatrixLinearOperator),
    (kl.Linear() * kl.Linear(), lx.MatrixLinearOperator),
]
STRUCTURED_IDS = [
    "white",
    "constant",
    "linear",
    "linear+white",
    "scaled-sum",
    "active-dims",
    "warped",
    "rbf+white-dense",
    "product-dense",
]


@pytest.mark.parametrize(("kernel", "cls"), STRUCTURED, ids=STRUCTURED_IDS)
def test_structured_operator_matches_the_gram(kernel, cls):
    op = kl.to_operator(kernel, X, noise=NOISE)
    assert isinstance(op, cls)
    assert jnp.allclose(op.as_matrix(), kernel(X, X) + NOISE * jnp.eye(30), atol=1e-12)
    assert lx.is_symmetric(op) and lx.is_positive_semidefinite(op)


@pytest.mark.parametrize(("kernel", "cls"), STRUCTURED, ids=STRUCTURED_IDS)
def test_structured_solve_and_logdet_match_dense(kernel, cls):
    op = kl.to_operator(kernel, X, noise=NOISE)
    dense = kl.to_operator(kernel, X, noise=NOISE, structure="dense")
    assert isinstance(dense, lx.MatrixLinearOperator)
    assert jnp.allclose(gx.solve(op, Y), jnp.linalg.solve(dense.as_matrix(), Y))
    assert jnp.allclose(gx.logdet(op), jnp.linalg.slogdet(dense.as_matrix())[1])


def test_structured_logdet_gradients_match_dense():
    def loss(params, structure):
        variance, bias, noise = params
        k = kl.Linear(variance=variance, bias=bias) + kl.White(noise)
        op = kl.to_operator(k, X, structure=structure)
        return gx.logdet(op) + Y @ gx.solve(op, Y)

    # bias = 0 is the default: the zero-weight column must not break gradients.
    params = (jnp.array(0.7), jnp.array(0.0), jnp.array(0.2))
    structured = jax.grad(loss)(params, "auto")
    dense = jax.grad(loss)(params, "dense")
    for a, b in zip(structured, dense, strict=True):
        assert jnp.isfinite(a) and jnp.allclose(a, b, rtol=1e-8)


def test_structured_solve_never_forms_an_n_by_n_matrix():
    n = 4096
    Xn = jr.normal(jr.key(3), (n, 3))
    yn = jr.normal(jr.key(4), (n,))
    k = kl.Linear(bias=0.3) + kl.White(0.1)
    jaxpr = str(jax.make_jaxpr(lambda X, y: gx.solve(kl.to_operator(k, X), y))(Xn, yn))
    assert f"{n},{n}" not in jaxpr


def test_structured_white_is_observation_noise_on_repeated_rows():
    # The dense delta on locations also correlates repeated rows; the
    # structured Gram is sigma^2 I, independent noise per observation.
    Xd = jnp.concatenate([X[:3], X[:3]])
    op = kl.to_operator(kl.White(0.5), Xd)
    assert jnp.allclose(op.as_matrix(), 0.5 * jnp.eye(6))
    assert kl.White(0.5)(Xd, Xd)[0, 3] == 0.5


def test_unknown_structure_raises():
    with pytest.raises(ValueError, match="structure"):
        kl.to_operator(kl.RBF(), X, structure="sparse")
