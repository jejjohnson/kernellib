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
from kernellib._spectral import (
    AbstractFeatureMap,
    FastFoodFeatures,
    LaplaceEigenfunctionFeatures,
    NystromFeatures,
    OrthogonalRandomFeatures,
    RandomFourierFeatures,
    draw_rff_cosine_basis,
    evaluate_rff_cosine_paths,
)


__version__ = "0.0.5"

__all__ = [
    "RBF",
    "AbstractFeatureMap",
    "AbstractKernel",
    "AbstractPointwiseKernel",
    "AbstractStationaryKernel",
    "ActiveDims",
    "Constant",
    "Cosine",
    "EigenProPreconditioner",
    "FalkonPreconditioner",
    "FastFoodFeatures",
    "FastFoodParams",
    "ImplicitCrossKernelOperator",
    "ImplicitKernelOperator",
    "KernelOperator",
    "LaplaceEigenfunctionFeatures",
    "Linear",
    "Matern",
    "NystromFeatures",
    "OrthogonalRandomFeatures",
    "Periodic",
    "Polynomial",
    "Product",
    "RandomFourierFeatures",
    "RationalQuadratic",
    "Scaled",
    "Sum",
    "Warped",
    "White",
    "__version__",
    "batched_kernel_matvec",
    "batched_kernel_rmatvec",
    "draw_rff_cosine_basis",
    "eigenpro_correction",
    "eigenpro_preconditioner",
    "eigenpro_step_size",
    "evaluate_rff_cosine_paths",
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
