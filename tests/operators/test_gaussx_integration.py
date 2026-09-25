"""The moved kernel operators work with gaussx's solvers from kernellib.

gaussx's strategies and lineax's iterative solvers dispatch on lineax's
structural predicates, and ``lineax.linearise`` must be registered for
matrix-free solves. These tests fail if the registrations in
``kernellib._operators`` are missing or wrong, which the moved operator
tests alone would not catch.

Inputs use a pinned key: the randomness is incidental (any well-conditioned
system would do), so the tolerances are deterministic.
"""

from __future__ import annotations

import gaussx as gx
import jax.numpy as jnp
import jax.random as jr
import lineax as lx

from kernellib import ImplicitKernelOperator, KernelOperator


def _rbf(params, x, y):
    return params["variance"] * jnp.exp(
        -0.5 * jnp.sum((x - y) ** 2) / params["lengthscale"] ** 2
    )


PARAMS = {"variance": jnp.array(1.0), "lengthscale": jnp.array(0.7)}
NOISE = 0.1


def _problem():
    kx, ky = jr.split(jr.key(0))
    X = jr.normal(kx, (40, 2))
    y = jr.normal(ky, (40,))
    return X, y


def _dense_solution(X, y):
    K = KernelOperator(_rbf, X, X, PARAMS).as_matrix()
    return jnp.linalg.solve(K + NOISE * jnp.eye(X.shape[0]), y)


def test_implicit_operator_predicates_follow_tags():
    X, _ = _problem()
    tags = frozenset({lx.symmetric_tag, lx.positive_semidefinite_tag})
    op = ImplicitKernelOperator(_rbf, X, NOISE, params=PARAMS, tags=tags)
    assert lx.is_symmetric(op)
    assert lx.is_positive_semidefinite(op)
    assert not lx.is_negative_semidefinite(op)
    assert not lx.is_diagonal(op)
    assert not lx.is_tridiagonal(op)
    assert not lx.is_lower_triangular(op)
    assert not lx.is_upper_triangular(op)
    assert not lx.has_unit_diagonal(op)
    assert lx.linearise(op) is op


def test_gaussx_cg_solver_on_implicit_operator_matches_dense():
    X, y = _problem()
    tags = frozenset({lx.symmetric_tag, lx.positive_semidefinite_tag})
    op = ImplicitKernelOperator(_rbf, X, NOISE, params=PARAMS, tags=tags)
    x = gx.CGSolver(rtol=1e-10, atol=1e-10, max_steps=500).solve(op, y)
    assert jnp.allclose(x, _dense_solution(X, y), atol=1e-6)


def test_gaussx_solve_on_implicit_operator_matches_dense():
    X, y = _problem()
    tags = frozenset({lx.symmetric_tag, lx.positive_semidefinite_tag})
    op = ImplicitKernelOperator(_rbf, X, NOISE, params=PARAMS, tags=tags)
    x = gx.solve(op, y, solver=lx.CG(rtol=1e-10, atol=1e-10, max_steps=500))
    assert jnp.allclose(x, _dense_solution(X, y), atol=1e-6)


# ---------------------------------------------------------------------------
# lineax.diagonal, needed by gaussx's partial-Cholesky preconditioner
# ---------------------------------------------------------------------------


def _rbf_plain(x, y):
    return jnp.exp(-0.5 * jnp.sum((x - y) ** 2) / 0.7**2)


def _diag_cases():
    from kernellib import ImplicitCrossKernelOperator

    kx, kz, kb = jr.split(jr.key(3), 3)
    X = jr.normal(kx, (12, 2))
    Z = jr.normal(kz, (5, 2))
    Xb = jr.normal(kb, (3, 7, 2))
    cross = ImplicitCrossKernelOperator(_rbf, X, Z, 4, params=PARAMS)
    return [
        ("implicit-params", ImplicitKernelOperator(_rbf, X, NOISE, params=PARAMS)),
        ("implicit-plain", ImplicitKernelOperator(_rbf_plain, X, NOISE)),
        ("implicit-batched", ImplicitKernelOperator(_rbf_plain, Xb, NOISE)),
        ("kernel-square", KernelOperator(_rbf, X, X, PARAMS)),
        ("kernel-rect", KernelOperator(_rbf, X, Z, PARAMS)),
        ("cross", cross),
        ("cross-plain", ImplicitCrossKernelOperator(_rbf_plain, X, Z, 4)),
        ("cross-transposed", cross.transpose()),
    ]


def test_diagonal_matches_dense_for_every_kernel_operator():
    for name, op in _diag_cases():
        expected = jnp.diagonal(op.as_matrix(), axis1=-2, axis2=-1)
        assert jnp.allclose(lx.diagonal(op), expected, atol=1e-12), name


def test_preconditioned_cg_on_implicit_operator_matches_dense():
    """Regression: gaussx's partial-Cholesky preconditioner calls lx.diagonal."""
    X, y = _problem()
    tags = frozenset({lx.symmetric_tag, lx.positive_semidefinite_tag})
    op = ImplicitKernelOperator(_rbf, X, NOISE, params=PARAMS, tags=tags)
    solver = gx.PreconditionedCGSolver(
        preconditioner_rank=10, shift=NOISE, rtol=1e-10, atol=1e-10
    )
    assert jnp.allclose(solver.solve(op, y), _dense_solution(X, y), atol=1e-6)
