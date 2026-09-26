"""Kernel regression: estimators and the Falkon / EigenPro primitives.

The primitives moved from gaussx; the estimators (`KRR`, and `Falkon` /
`EigenPro` on top of the primitives) follow the `AbstractEstimator` contract.
"""

from __future__ import annotations

from kernellib._regression._base import AbstractEstimator
from kernellib._regression._eigenpro import (
    EigenProPreconditioner,
    eigenpro_correction,
    eigenpro_preconditioner,
    eigenpro_step_size,
)
from kernellib._regression._falkon import (
    FalkonPreconditioner,
    falkon_preconditioner,
    falkon_predict,
    falkon_solve,
)
from kernellib._regression._krr import KRR


__all__ = [
    "KRR",
    "AbstractEstimator",
    "EigenProPreconditioner",
    "FalkonPreconditioner",
    "eigenpro_correction",
    "eigenpro_preconditioner",
    "eigenpro_step_size",
    "falkon_preconditioner",
    "falkon_predict",
    "falkon_solve",
]
