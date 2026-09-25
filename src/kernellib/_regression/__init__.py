"""Kernel regression: the Falkon and EigenPro primitives, moved from gaussx.

The estimator-level workflows (``KRR``, ``Falkon``, ``EigenPro``) will live
here too; they arrive with the algorithms phase.
"""

from __future__ import annotations

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


__all__ = [
    "EigenProPreconditioner",
    "FalkonPreconditioner",
    "eigenpro_correction",
    "eigenpro_preconditioner",
    "eigenpro_step_size",
    "falkon_preconditioner",
    "falkon_predict",
    "falkon_solve",
]
