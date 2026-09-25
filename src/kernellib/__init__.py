"""Kernels and scalable kernel methods for JAX.

``kernellib`` sits between `gaussx <https://github.com/jejjohnson/gaussx>`_
(structured linear operators, solvers, preconditioners) and
`pyrox-gp <https://github.com/jejjohnson/pyrox>`_ (Gaussian-process models
with NumPyro priors). It owns kernel functions and their composition, spectral
densities and feature maps, kernel ridge regression, dependence measures, and
kernel derivatives. Scale is inherited from gaussx: every kernel can be turned
into a gaussx linear operator, and every algorithm here solves through gaussx's
solver strategies.

The package never imports ``numpyro``; the modelling layer above it does.

Everything listed in ``__all__`` is importable straight from the top-level
package.
"""

from __future__ import annotations

from kernellib import functional
from kernellib._operators import (
    ImplicitCrossKernelOperator,
    ImplicitKernelOperator,
    KernelOperator,
    batched_kernel_matvec,
    batched_kernel_rmatvec,
    implicit_cross_kernel,
    nystrom_operator,
    rff_operator,
)


__version__ = "0.0.1"

__all__ = [
    "ImplicitCrossKernelOperator",
    "ImplicitKernelOperator",
    "KernelOperator",
    "__version__",
    "batched_kernel_matvec",
    "batched_kernel_rmatvec",
    "functional",
    "implicit_cross_kernel",
    "nystrom_operator",
    "rff_operator",
]
