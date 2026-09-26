# API Reference

Everything listed in `kernellib.__all__` is importable from the top level:

```python
import kernellib as kl

kl.__version__
```

The submodules are an implementation detail you are welcome to reach into,
but never have to. Modules appear here as each phase of the
[architecture](../architecture/) lands.

## Modules

| Module | Contents |
|---|---|
| [Kernels](kernels.md) | `AbstractKernel`, `AbstractPointwiseKernel`, `AbstractStationaryKernel`, `RBF`, `Matern`, `RationalQuadratic`, `Periodic`, `Cosine`, `White`, `Constant`, `Linear`, `Polynomial`, `Sum`, `Product`, `Scaled`, `ActiveDims`, `Warped`, `to_operator`, `to_cross_operator` |
| [Functional](functional.md) | `rbf_kernel`, `matern_kernel`, `rational_quadratic_kernel`, `periodic_kernel`, `cosine_kernel`, `linear_kernel`, `polynomial_kernel`, `white_kernel`, `constant_kernel`, `kernel_add`, `kernel_mul`, `stable_rbf_kernel`; `centering_operator`, `center_kernel`, `hsic`, `cka`, `mmd_squared` |
| [Operators](operators.md) | `KernelOperator`, `ImplicitKernelOperator`, `ImplicitCrossKernelOperator`, `implicit_cross_kernel`, `batched_kernel_matvec`, `batched_kernel_rmatvec`, `nystrom_operator`, `rff_operator`, `fastfood_params`, `fastfood_features`, `fastfood_operator`, `FastFoodParams`, `fastfood_frequencies`, `hadamard_transform` |
| [Spectral](spectral.md) | `AbstractStationaryKernel.spectral_density` / `sample_frequencies`, `AbstractFeatureMap`, `RandomFourierFeatures`, `OrthogonalRandomFeatures`, `FastFoodFeatures`, `NystromFeatures`, `LaplaceEigenfunctionFeatures`, `draw_rff_cosine_basis`, `evaluate_rff_cosine_paths` |
| [Heuristics](heuristics.md) | `estimate_lengthscale`, `lengthscale_to_gamma`, `gamma_to_lengthscale`, `lengthscale_grid` |
| [Regression](regression.md) | `AbstractEstimator`, `KRR`, `falkon_preconditioner`, `falkon_solve`, `falkon_predict`, `FalkonPreconditioner`, `eigenpro_preconditioner`, `eigenpro_step_size`, `eigenpro_correction`, `EigenProPreconditioner` |

## Package overview

::: kernellib
    options:
      members: false
      show_root_heading: false
      show_source: false
