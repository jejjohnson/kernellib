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
from kernellib._kernels import (
    RBF,
    AbstractKernel,
    AbstractPointwiseKernel,
    AbstractStationaryKernel,
    ActiveDims,
    Constant,
    Cosine,
    Linear,
    Matern,
    Periodic,
    Polynomial,
    Product,
    RationalQuadratic,
    Scaled,
    Sum,
    Warped,
    White,
)
from kernellib._operators import (
    FastFoodParams,
    ImplicitCrossKernelOperator,
    ImplicitKernelOperator,
    KernelOperator,
    batched_kernel_matvec,
    batched_kernel_rmatvec,
    fastfood_features,
    fastfood_frequencies,
    fastfood_operator,
    fastfood_params,
    hadamard_transform,
    implicit_cross_kernel,
    nystrom_operator,
    rff_operator,
    to_cross_operator,
    to_operator,
)
from kernellib._regression import (
    EigenProPreconditioner,
    FalkonPreconditioner,
    eigenpro_correction,
    eigenpro_preconditioner,
    eigenpro_step_size,
    falkon_preconditioner,
    falkon_predict,
    falkon_solve,
)


__version__ = "0.0.4"

__all__ = [
    "RBF",
    "AbstractKernel",
    "AbstractPointwiseKernel",
    "AbstractStationaryKernel",
    "ActiveDims",
    "Constant",
    "Cosine",
    "EigenProPreconditioner",
    "FalkonPreconditioner",
    "FastFoodParams",
    "ImplicitCrossKernelOperator",
    "ImplicitKernelOperator",
    "KernelOperator",
    "Linear",
    "Matern",
    "Periodic",
    "Polynomial",
    "Product",
    "RationalQuadratic",
    "Scaled",
    "Sum",
    "Warped",
    "White",
    "__version__",
    "batched_kernel_matvec",
    "batched_kernel_rmatvec",
    "eigenpro_correction",
    "eigenpro_preconditioner",
    "eigenpro_step_size",
    "falkon_preconditioner",
    "falkon_predict",
    "falkon_solve",
    "fastfood_features",
    "fastfood_frequencies",
    "fastfood_operator",
    "fastfood_params",
    "functional",
    "hadamard_transform",
    "implicit_cross_kernel",
    "nystrom_operator",
    "rff_operator",
    "to_cross_operator",
    "to_operator",
]
