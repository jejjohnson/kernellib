"""Derivative operators of callables (layer 0).

Batched, exact (autodiff) derivatives of scalar fields ``(N, D) -> (N,)`` and
vector fields ``(N, D) -> (N, M)`` under one pointwise contract: `gradient`,
`jacobian`, `hessian`, `laplacian`, `divergence`, `curl`; and the
sensitivity maps `sensitivity` / `point_sensitivity` of predictors and
dataset functionals. Kernel derivative *Grams* stay kernels
(`Derivative`, `DerivativeIndexed`). See ``docs/api/derivatives.md``.
"""

from __future__ import annotations

from kernellib._derivatives._operators import (
    curl,
    divergence,
    gradient,
    hessian,
    jacobian,
    laplacian,
)
from kernellib._derivatives._sensitivity import point_sensitivity, sensitivity


__all__ = [
    "curl",
    "divergence",
    "gradient",
    "hessian",
    "jacobian",
    "laplacian",
    "point_sensitivity",
    "sensitivity",
]
