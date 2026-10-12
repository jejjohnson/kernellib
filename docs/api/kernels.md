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
kernel derivatives and hyperparameter gradients are compositions of
`jax.grad`, `jax.jacfwd` and `jax.vmap`; the
[Kernels and JAX](../../kernels-and-jax/) tutorial shows each. Derivatives of
a fitted predictor or a vector field (gradient, Hessian, Laplacian,
divergence, curl) are the operators on the [Derivatives](derivatives.md)
page.

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

::: kernellib.Tanh

!!! warning "Tanh is not positive semidefinite in general"
    `is_psd` is `False`: `to_operator` warns and tags the operator symmetric
    only. Add a ridge larger than $-\lambda_{\min}(K)$ before a Cholesky-type
    solve (`KRR` with a large enough `regularization` does this).

References: Smola, Ovari & Williamson (2001), *Regularization with dot-product
kernels* (positive-definite dot-product kernels need non-negative Taylor
coefficients, which `tanh` lacks); Lin & Lin (2003), *A study on sigmoid
kernels for SVM and the training of non-PSD kernels by SMO-type methods*.

### Paper and scikit-learn parameter mapping

Johnson, Laparra, Piles & Camps-Valls (2020), *Kernel methods and their
derivatives*, PLOS ONE 15(10): e0235885 ([arXiv:2007.14706](https://arxiv.org/abs/2007.14706)),
write the kernels in the parameterisation on the left.

| Paper (Table 1) | kernellib | Notes |
|---|---|---|
| Linear $x^\top y$ | `Linear(variance=1, bias=0)` | kernellib: $\sigma^2 x^\top y + b$ |
| Poly $(\gamma x^\top y + c_0)^p$ | `Polynomial(variance=γ**p, bias=c0/γ, degree=p)` | exact for $\gamma > 0$ |
| RBF $\exp(-\gamma\lVert x - y\rVert^2)$ | `RBF(lengthscale=1/sqrt(2γ), variance=1)` | $\gamma = 1/(2\ell^2)$ |
| Tanh $\tanh(\gamma x^\top y + c_0)$ | `Tanh(variance=1, scale=γ, bias=c0)` | scikit-learn `"sigmoid"` |
| ARD $\nu^2 e^{-\frac12\sum_d (h^d/\lambda_d)^2}$ | `RBF(lengthscale=λ, variance=ν²)` | $h = x - y$ |

### Kernel derivative tables

Derivatives of $k(x, y)$ with respect to the first argument, with
$u = \gamma x^\top y + c_0$, $h = x - y$ and $j \ne k$. Every entry is
pinned against `jax.grad` / `jax.hessian` of `pairwise` in
`tests/test_kernel_derivative_tables.py`.

| Kernel | $\partial k/\partial x^j$ | $\partial^2k/\partial (x^j)^2$ | $\partial^2k/\partial x^j\partial x^k$ |
|---|---|---|---|
| Linear | $\sigma^2 y^j$ | $0$ | $0$ |
| Poly $(\gamma x^\top y + c_0)^p$ | $\gamma p\,y^j u^{p-1}$ | $p(p-1)(\gamma y^j)^2u^{p-2}$ | $p(p-1)\gamma^2y^jy^ku^{p-2}$ |
| RBF $e^{-\gamma\lVert h\rVert^2}$ | $-2\gamma h^j k$ | $2\gamma(2\gamma (h^j)^2 - 1)k$ | $4\gamma^2h^jh^kk$ |
| Tanh $\tanh u$ | $\gamma y^j\operatorname{sech}^2u$ | $-2(\gamma y^j)^2\operatorname{sech}^2(u)\,k$ | $-2\gamma^2y^jy^k\operatorname{sech}^2(u)\,k$ |
| ARD | $-\dfrac{h^j}{\lambda_j^2}k$ | $\Big(\dfrac{(h^j)^2}{\lambda_j^4} - \dfrac1{\lambda_j^2}\Big)k$ | $\dfrac{h^j}{\lambda_j^2}\dfrac{h^k}{\lambda_k^2}k$ |

!!! note "Errata in the paper's ARD row"
    The ARD entries above are the corrected ones. Table 1 of the paper prints
    $+(h^j/\lambda_j^2)\,k$ (sign error) and Table 2 prints
    $(1/\lambda_j^2 + (h^j/\lambda_j^2)^2)\,k$ (sign of the $1/\lambda_j^2$
    term). Both disagree with autodiff and with the paper's own RBF row
    ($\gamma = 1/(2\lambda^2)$ gives $(h^2/\lambda^4 - 1/\lambda^2)k$). Checked
    against the arXiv e-print LaTeX (2020-10-06); confirm against the version
    of record before citing the errata publicly. The tests assert the corrected
    forms and that the printed ones fail.

## Distance-induced

`Distance` turns HSIC and MMD into the distance-based statistics of Székely
et al.; see [Dependence](dependence.md#distance-based-statistics).

::: kernellib.Distance

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

## Derivative kernels

Covariances of a GP's partial derivatives, as kernels.
`Derivative(k, dx=i, dy=j)` is $\partial_{x_i}\partial_{x'_j} k$;
`DerivativeIndexed(k)` takes rows `[x, i]` (`i = -1` for a value,
`i = d` for $\partial_d f$) and gives the joint covariance of any mix of
value and gradient observations, which `derivative_inputs` builds.

```python
Xa = kl.derivative_inputs(X)  # values, then every partial
K = kl.to_operator(kl.DerivativeIndexed(kl.RBF()), Xa, noise=1e-4)
alpha = gx.solve(K, jnp.concatenate([y, dy.T.ravel()]))
```

::: kernellib.Derivative

::: kernellib.DerivativeIndexed

::: kernellib.derivative_inputs

## Bridge to gaussx

::: kernellib.to_operator

::: kernellib.to_cross_operator
