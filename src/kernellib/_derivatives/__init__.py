"""Derivative operators of callables (layer 0).

Batched, exact (autodiff) derivatives of scalar fields ``(N, D) -> (N,)`` and
vector fields ``(N, D) -> (N, M)`` under one pointwise contract: `gradient`,
`jacobian`, `hessian`, `laplacian`, `divergence`, `curl`. Kernel derivative
*Grams* stay kernels (`Derivative`, `DerivativeIndexed`). See
``docs/api/derivatives.md``.
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


__all__ = ["curl", "divergence", "gradient", "hessian", "jacobian", "laplacian"]
