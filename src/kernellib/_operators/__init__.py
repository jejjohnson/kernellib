"""Kernel linear operators, moved from gaussx.

Every operator here is a `lineax.AbstractLinearOperator`, so it plugs into
`gaussx.solve`, `gaussx.logdet` and every gaussx solver strategy unchanged:
those dispatch on lineax's structural predicates, which are registered below.
"""

from __future__ import annotations

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
