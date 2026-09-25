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
| [Functional](functional.md) | `rbf_kernel`, `matern_kernel`, `rational_quadratic_kernel`, `periodic_kernel`, `cosine_kernel`, `linear_kernel`, `polynomial_kernel`, `white_kernel`, `constant_kernel`, `kernel_add`, `kernel_mul`; `centering_operator`, `center_kernel`, `hsic`, `cka`, `mmd_squared` |
| [Operators](operators.md) | `KernelOperator`, `ImplicitKernelOperator`, `ImplicitCrossKernelOperator`, `implicit_cross_kernel`, `batched_kernel_matvec`, `batched_kernel_rmatvec`, `nystrom_operator`, `rff_operator` |

## Package overview

::: kernellib
    options:
      members: false
      show_root_heading: false
      show_source: false
