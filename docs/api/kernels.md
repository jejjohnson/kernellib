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

::: kernellib.Sum

::: kernellib.Product

::: kernellib.Scaled

::: kernellib.ActiveDims

::: kernellib.Warped

## Bridge to gaussx

::: kernellib.to_operator

::: kernellib.to_cross_operator
