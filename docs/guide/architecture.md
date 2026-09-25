# Architecture

kernellib sits between gaussx and pyrox-gp. This page is the short version of
the design document committed at `design_docs/kernellib/architecture.md`,
which fixes the kernel contract, the package layout, the migration phases, and
the boundaries with gaussx, pyrox-gp, geonnax and pysim.

## The placement test

Every family of functionality is split with one rule:

- **Needs a kernel object or a random key** → kernellib.
- **Takes matrices, operators or a callable; returns matrices, operators or a
  scalar** → gaussx.
- **Kernel-agnostic feature-map arithmetic and basis functions** → geonnax.

| Family | gaussx (operations) | kernellib (algorithms) |
|---|---|---|
| Kernels | `KernelOperator`, `ImplicitKernelOperator`, `ImplicitCrossKernelOperator` | `RBF`, `Matern`, `Periodic`, ..., `Sum`, `Product`, `to_operator` |
| Approximation | `nystrom_operator`, `rff_operator`, `fastfood_operator` | `NystromFeatures`, `RandomFourierFeatures`, `OrthogonalRandomFeatures`, `FastFoodFeatures` |
| Dependence | `hsic`, `cka`, `mmd_squared`, `center_kernel` on matrices | `hsic`, `cka`, `mmd`, `permutation_test` on kernels and data, randomized variants |
| Regression | `solve` with any strategy, Falkon and EigenPro primitives | `KRR`, `Falkon`, `EigenPro` estimators |
| Derivatives | autodiff | `kernel_jacobian`, `derivative_gram`, `predictor_gradient` |

## The kernel contract

Three abstract levels, each adding one capability:

```python
class AbstractKernel(eqx.Module):            # Gram-only: __call__(X1, X2) -> (N1, N2)
class AbstractPointwiseKernel(AbstractKernel):   # adds pairwise(x, y) -> scalar
class AbstractStationaryKernel(AbstractPointwiseKernel):  # adds shape(r²), spectral_density(ω)
```

Hyperparameters are plain array fields with no transforms and no priors; the
modelling layer above adds those. Stationary kernels override the Gram path
with the closed form; the pointwise form exists for autodiff and implicit
operators.

## The bridge to gaussx

`to_operator(kernel, X, noise=..., implicit=...)` returns a gaussx linear
operator. With `implicit=True` and a pointwise kernel it is a matrix-free
`ImplicitKernelOperator` whose custom JVP differentiates through the kernel's
hyperparameters. After the bridge, everything else is gaussx:

```python
K = kl.to_operator(kernel, X, noise=1e-2, implicit=True)
alpha = gx.solve(K, y, solver=gx.PreconditionedCGSolver(preconditioner_rank=100))
```

## Layout and phases

| Path | Contents | Phase |
|---|---|---|
| `functional/`, `_kernels/`, `_operators.py` | Kernel math, abstractions, composition, the bridge | 1 |
| `_spectral/` | Spectral densities, feature maps | 3 (FastFood in 5) |
| `_heuristics.py`, `_regression/`, `_dependence/`, `_decomposition/`, `_derivatives.py` | Estimators, dependence measures, embeddings, derivatives | 5 |

Phase 0 is this scaffold. Phase 2 makes pyrox-gp consume kernellib, and
phase 4 adds four small operations to gaussx (unbiased HSIC and matrix-level
CKA, a low-rank `trace_product` fast path, a low-rank-preserving
`center_kernel`, and the FastFood operator).
