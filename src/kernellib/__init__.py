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
from kernellib._decomposition import (
    KernelPCA,
    KNNGraph,
    LaplacianEigenmaps,
    LocalityPreservingProjections,
    SchrodingerEigenmaps,
    adjacency_matrix,
    barrier_potential,
    commute_time_kernel,
    cosine_graph_kernel,
    diffusion_kernel,
    graph_laplacian,
    label_potential,
    laplacian_eigenmap,
    nearest_neighbors,
    random_walk_kernel,
    regularized_laplacian_kernel,
    schrodinger_eigenmap,
    spatial_spectral_potential,
)
from kernellib._dependence import (
    PermutationTestResult,
    cka,
    hsic,
    kernel_alignment,
    mmd_squared,
    permutation_test,
)
from kernellib._heuristics import (
    estimate_lengthscale,
    gamma_to_lengthscale,
    lengthscale_grid,
    lengthscale_to_gamma,
)
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
    KRR,
    AbstractEstimator,
    EigenPro,
    EigenProPreconditioner,
    Falkon,
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


__version__ = "0.0.7"

__all__ = [
    "KRR",
    "RBF",
    "AbstractEstimator",
    "AbstractFeatureMap",
    "AbstractKernel",
    "AbstractPointwiseKernel",
    "AbstractStationaryKernel",
    "ActiveDims",
    "Constant",
    "Cosine",
    "EigenPro",
    "EigenProPreconditioner",
    "Falkon",
    "FalkonPreconditioner",
    "FastFoodFeatures",
    "FastFoodParams",
    "ImplicitCrossKernelOperator",
    "ImplicitKernelOperator",
    "KNNGraph",
    "KernelOperator",
    "KernelPCA",
    "LaplaceEigenfunctionFeatures",
    "LaplacianEigenmaps",
    "Linear",
    "LocalityPreservingProjections",
    "Matern",
    "NystromFeatures",
    "OrthogonalRandomFeatures",
    "Periodic",
    "PermutationTestResult",
    "Polynomial",
    "Product",
    "RandomFourierFeatures",
    "RationalQuadratic",
    "Scaled",
    "SchrodingerEigenmaps",
    "Sum",
    "Warped",
    "White",
    "__version__",
    "adjacency_matrix",
    "barrier_potential",
    "batched_kernel_matvec",
    "batched_kernel_rmatvec",
    "cka",
    "commute_time_kernel",
    "cosine_graph_kernel",
    "diffusion_kernel",
    "draw_rff_cosine_basis",
    "eigenpro_correction",
    "eigenpro_preconditioner",
    "eigenpro_step_size",
    "estimate_lengthscale",
    "evaluate_rff_cosine_paths",
    "falkon_preconditioner",
    "falkon_predict",
    "falkon_solve",
    "fastfood_features",
    "fastfood_frequencies",
    "fastfood_operator",
    "fastfood_params",
    "functional",
    "gamma_to_lengthscale",
    "graph_laplacian",
    "hadamard_transform",
    "hsic",
    "implicit_cross_kernel",
    "kernel_alignment",
    "label_potential",
    "laplacian_eigenmap",
    "lengthscale_grid",
    "lengthscale_to_gamma",
    "mmd_squared",
    "nearest_neighbors",
    "nystrom_operator",
    "permutation_test",
    "random_walk_kernel",
    "regularized_laplacian_kernel",
    "rff_operator",
    "schrodinger_eigenmap",
    "spatial_spectral_potential",
    "to_cross_operator",
    "to_operator",
]
