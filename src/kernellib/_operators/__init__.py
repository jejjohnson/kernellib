"""Kernel linear operators, moved from gaussx.

Every operator here is a `lineax.AbstractLinearOperator`, so it plugs into
`gaussx.solve`, `gaussx.logdet` and every gaussx solver strategy unchanged:
those dispatch on lineax's structural predicates, which are registered below.
"""

from __future__ import annotations

import jax
import lineax as lx

from kernellib._operators._batched import (
    batched_kernel_matvec,
    batched_kernel_rmatvec,
)
from kernellib._operators._bridge import to_cross_operator, to_operator
from kernellib._operators._implicit import ImplicitKernelOperator
from kernellib._operators._implicit_cross import (
    ImplicitCrossKernelOperator,
    _TransposedCrossKernelOperator,
    implicit_cross_kernel,
)
from kernellib._operators._kernel import KernelOperator
from kernellib._operators._low_rank import nystrom_operator, rff_operator
from kernellib._operators._utils import vmap_over_batch_dims


# ---------------------------------------------------------------------------
# lineax predicate registrations
#
# Carried over unchanged from gaussx's ``_operators/__init__.py``. lineax
# 0.1.1 made every structural predicate a required dispatch, and
# ``lineax.linearise`` is registered only for lineax's own classes, so
# without these the operators fail under ``lineax.CG`` and are misread by
# gaussx's strategies. Symmetry and definiteness come from the operator's
# tags; no kernel operator claims diagonal, tridiagonal, triangular or
# unit-diagonal structure.
# ---------------------------------------------------------------------------

_KERNEL_OPERATORS = (
    KernelOperator,
    ImplicitKernelOperator,
    ImplicitCrossKernelOperator,
    _TransposedCrossKernelOperator,
)

for _cls in _KERNEL_OPERATORS:
    lx.is_symmetric.register(_cls)(lambda operator: lx.symmetric_tag in operator.tags)
    lx.is_positive_semidefinite.register(_cls)(
        lambda operator: lx.positive_semidefinite_tag in operator.tags
    )
    lx.is_negative_semidefinite.register(_cls)(
        lambda operator: lx.negative_semidefinite_tag in operator.tags
    )
    lx.is_diagonal.register(_cls)(lambda _operator: False)
    lx.is_tridiagonal.register(_cls)(lambda _operator: False)
    lx.has_unit_diagonal.register(_cls)(lambda _operator: False)
    lx.is_lower_triangular.register(_cls)(lambda _operator: False)
    lx.is_upper_triangular.register(_cls)(lambda _operator: False)
    lx.linearise.register(_cls)(lambda operator: operator)

del _cls


# ---------------------------------------------------------------------------
# lineax.diagonal
#
# gaussx's partial-Cholesky preconditioner (``PreconditionedCGSolver``) reads
# ``lx.diagonal(operator)``, which lineax only implements for its own classes.
# gaussx never registered it for the kernel operators, so preconditioned CG
# on them raised ``NotImplementedError`` from gaussx 0.1.0 on. The diagonal
# of ``K(A, B)`` is ``k(A[i], B[i])`` for ``i < min(N, M)``: exact, O(N), and
# equal to ``jnp.diagonal(op.as_matrix())`` without materializing anything.
# ---------------------------------------------------------------------------


def _kernel_diag(kernel_fn, params, has_params, A, B, batch_shape):
    n = min(A.shape[-2], B.shape[-2])

    def single(a, b):
        if has_params:
            return jax.vmap(lambda x, y: kernel_fn(params, x, y))(a[:n], b[:n])
        return jax.vmap(kernel_fn)(a[:n], b[:n])

    return vmap_over_batch_dims(single, len(batch_shape))(A, B)


@lx.diagonal.register(ImplicitKernelOperator)
def _(operator: ImplicitKernelOperator):
    diag = _kernel_diag(
        operator.kernel_fn,
        operator.params,
        operator._has_params,
        operator.X,
        operator.X,
        operator._batch_shape,
    )
    return diag + operator.noise_var


@lx.diagonal.register(KernelOperator)
def _(operator: KernelOperator):
    return _kernel_diag(
        operator.kernel_fn,
        operator.params,
        True,
        operator.X1,
        operator.X2,
        operator._batch_shape,
    )


@lx.diagonal.register(ImplicitCrossKernelOperator)
def _(operator: ImplicitCrossKernelOperator):
    return _kernel_diag(
        operator.kernel_fn,
        operator.params,
        operator._has_params,
        operator.X_data,
        operator.X_inducing,
        operator._batch_shape,
    )


@lx.diagonal.register(_TransposedCrossKernelOperator)
def _(operator: _TransposedCrossKernelOperator):
    parent = operator._parent
    return _kernel_diag(
        parent.kernel_fn,
        parent.params,
        parent._has_params,
        parent.X_inducing,
        parent.X_data,
        parent._batch_shape,
    )


__all__ = [
    "ImplicitCrossKernelOperator",
    "ImplicitKernelOperator",
    "KernelOperator",
    "batched_kernel_matvec",
    "batched_kernel_rmatvec",
    "implicit_cross_kernel",
    "nystrom_operator",
    "rff_operator",
    "to_cross_operator",
    "to_operator",
]
