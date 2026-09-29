# Kernels

Kernel objects: the abstract contract, concrete kernels, composition, and the
bridge to gaussx-compatible operators.

The contract has three levels. `AbstractKernel` produces a Gram matrix;
`AbstractPointwiseKernel` adds ``pairwise(x, y)``, which autodiff and the
matrix-free operators use; `AbstractStationaryKernel` is
``variance * shape(||(x - x') / lengthscale||²)`` with a closed-form Gram path.
Hyperparameters are plain array fields.

```python
import gaussx as gx
import kernellib as kl

k = kl.RBF(lengthscale=0.5) + kl.White(1e-2)
K = kl.to_operator(k, X, noise=1e-2, implicit=True)
alpha = gx.solve(K, y, solver=gx.PreconditionedCGSolver(preconditioner_rank=100))
```

Kernels are equinox modules and `pairwise` is a scalar JAX function, so
derivatives, derivative Gram blocks, predictor gradients and hyperparameter
gradients are compositions of `jax.grad`, `jax.jacfwd` and `jax.vmap`; the
[Kernels and JAX](../../kernels-and-jax/) tutorial shows each.

## Contract

::: kernellib.AbstractKernel

::: kernellib.AbstractPointwiseKernel

::: kernellib.AbstractStationaryKernel

## Stationary

::: kernellib.RBF

::: kernellib.Matern

::: kernellib.RationalQuadratic

::: kernellib.Periodic

::: kernellib.Cosine

::: kernellib.White

::: kernellib.Constant

## Inner product

::: kernellib.Linear

::: kernellib.Polynomial

## Composition

Besides the constructors below, every kernel has shorthand methods returning
them: `k.stretch(c)` (`Warped` with `Stretch`), `k.shift(c)` (`Shift`; a
no-op for stationary kernels), `k.transform(w)` (`Warped`), `k.select(dims)`
(`ActiveDims`) and `k.periodic(p)` (`Periodised`). `k.elwise(X1, X2)`
evaluates the kernel on paired rows, and `k.is_stationary` reports
stationarity for composites too.

| mlkernels | kernellib |
|---|---|
| `EQ()`, `Matern12/32/52()`, `RQ(a)` | `RBF()`, `Matern(nu=0.5/1.5/2.5)`, `RationalQuadratic(alpha=a)` |
| `Delta()`, `Linear()` | `White()`, `Linear()` |
| `k.stretch(c)`, `k.shift(c)`, `k.select(d)`, `k.transform(f)`, `k.periodic(p)` | the same methods |
| `k.elwise(x, y)`, `k.stationary` | `k.elwise(X1, X2)`, `k.is_stationary` |
| `TensorProductKernel(f)`, `f * k` | `FeatureKernel(f)`, `Modulated(k, f)` |
| `SubspaceKernel`, `PosteriorKernel` (noise-free) | `nystrom_kernel(k, Z)`, `Residual(k, approx)` |

::: kernellib.Sum

::: kernellib.Product

::: kernellib.Scaled

::: kernellib.ActiveDims

::: kernellib.Warped

::: kernellib.Stretch

::: kernellib.Shift

::: kernellib.Periodised

## Feature kernels and modulation

`FeatureKernel(phi)` is the inner product of features, rank `R` and kept
low-rank by `to_operator`; with a fitted feature map it is that map's kernel
approximation as a kernel. `Modulated(k, a)` scales a kernel by an
input-dependent amplitude.

::: kernellib.FeatureKernel

::: kernellib.Modulated

## Approximations as kernels

`nystrom_kernel(k, Z)` is the Nyström kernel $k(x, Z) K_{ZZ}^{-1} k(Z, x')$
as a (low-rank) kernel, and `Residual(k, approx)` what an approximation
misses; with a Nyström approximation that is the GP covariance given the
values at the landmarks.

::: kernellib.nystrom_kernel

::: kernellib.Residual

## Bridge to gaussx

::: kernellib.to_operator

::: kernellib.to_cross_operator
