# Architecture

kernellib sits between gaussx and pyrox-gp. This page is the short version of
the design document committed at `design_docs/kernellib/architecture.md`,
which fixes the kernel contract, the package layout, the migration phases, and
the boundaries with gaussx, pyrox-gp, geonnax and pysim.

## The rule

**Anything with a kernel in it lives in kernellib, from the operators up.
gaussx is kernel-agnostic linear algebra. geonnax is kernel-agnostic
feature-map arithmetic and basis functions.**

That means gaussx's current kernel layer moves here: the kernel operators,
the Nyström and RFF low-rank operators, the matrix-level HSIC and MMD
statistics, and the Falkon and EigenPro primitives. gaussx keeps what those
build on: structured operators such as `LowRankUpdate`, the solver
strategies, preconditioners, `trace_product`, and `stable_squared_distances`
(a distance primitive that gaussx's own ensemble localization uses).

Two namespaces inside kernellib:

| Namespace | Takes | Examples |
|---|---|---|
| `kernellib.functional` and the operator constructors | arrays, matrices, operators, callables | `rbf_kernel`, `functional.hsic(K, L)`, `nystrom_operator(K_XZ, K_ZZ)`, `ImplicitKernelOperator(kernel_fn, X)` |
| `kernellib` top level | kernel objects and data, often a random key | `RBF`, `hsic(kx, ky, X, Y)`, `NystromFeatures(...).fit(k, X)`, `KRR(k, ...).fit(X, y)` |

| Family | Arrays in | Kernels in |
|---|---|---|
| Kernels | `KernelOperator`, `ImplicitKernelOperator`, `ImplicitCrossKernelOperator` | `RBF`, `Matern`, `Periodic`, ..., `Sum`, `Product`, `to_operator` |
| Approximation | `nystrom_operator`, `rff_operator`, `fastfood_operator` | `NystromFeatures`, `RandomFourierFeatures`, `OrthogonalRandomFeatures`, `FastFoodFeatures` |
| Dependence | `functional.hsic`, `functional.cka`, `functional.mmd_squared`, `functional.center_kernel` | `hsic`, `cka`, `mmd`, `permutation_test`, randomized variants |
| Regression | Falkon and EigenPro primitives | `KRR`, `Falkon`, `EigenPro` estimators |
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

`to_operator(kernel, X, noise=..., implicit=...)` returns a lineax operator.
With `implicit=True` and a pointwise kernel it is a matrix-free
`ImplicitKernelOperator` whose custom JVP differentiates through the kernel's
hyperparameters. The operators carry lineax's symmetric and
positive-semidefinite tags, and gaussx's strategies dispatch on tags rather
than on operator type, so after the bridge everything else is gaussx:

```python
K = kl.to_operator(kernel, X, noise=1e-2, implicit=True)
alpha = gx.solve(K, y, solver=gx.PreconditionedCGSolver(preconditioner_rank=100))
```

## Layout and phases

| Path | Contents | Phase |
|---|---|---|
| `functional/`, `_kernels/`, `_operators/` | Kernel math and matrix statistics, abstractions, composition, the operators moved from gaussx, the bridge | 1 |
| `_spectral/` | Spectral densities, feature maps including FastFood | 4 |
| `_heuristics.py`, `_regression/`, `_dependence/`, `_decomposition/`, `_derivatives.py` | Estimators (with the moved Falkon / EigenPro primitives), dependence measures, embeddings, derivatives | 5 |

Phase 0 is this scaffold. Phase 2 makes pyrox-gp consume kernellib. Phase 3
removes the kernel layer from gaussx in gaussx 0.2.0, with no shims, and adds
a low-rank fast path to `trace_product`, the one generic piece of linear
algebra the randomized dependence measures need.
