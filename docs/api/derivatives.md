# Derivatives

Exact, batched derivatives of **functions**: the gradient, Jacobian, Hessian
and Laplacian of a fitted predictor, and the divergence and curl of a vector
field. A **scalar field** is any callable `f: (N, D) -> (N,)` and a **vector
field** any callable `F: (N, D) -> (N, M)`: a fitted `KRR` / `Falkon` /
`EigenPro` `predict`, a closure over a kernel expansion, or any JAX function.
Every operator is autodiff, so the results are exact to rounding.

```python
import kernellib as kl

model = kl.KRR(kl.RBF(lengthscale=0.5), regularization=1e-3).fit(X, y)
G = kl.gradient(model.predict, X_test)  # (N, D)
H = kl.hessian(model.predict, X_test)  # (N, D, D)
L = kl.laplacian(model.predict, X_test)  # (N,)
```

These are derivatives of functions. Derivative **kernels**, the covariances of
a GP's partial derivatives, are kernels: [`Derivative`][kernellib.Derivative],
[`DerivativeIndexed`][kernellib.DerivativeIndexed] (on the
[Kernels](kernels.md#derivative-kernels) page). A one-off derivative of a kernel's
`pairwise` is still `jax.grad(kernel.pairwise)(x, y)`.

## Contract

- **Pointwise.** `f` is evaluated one row at a time,
  $f_n(x) := f(x[\text{None}])[0]$, and the operators return the derivative
  of that single-point function at $x_n$. When `f(X)[n]` depends only on
  `X[n]`, which holds for every predictor, this is the derivative of
  `f(X)[n]` with respect to `X[n]`. For a functional of the whole dataset
  (HSIC as a function of $X$), use `jax.grad` directly.
- **Index convention.** $J_n[i, j] = \partial F_i(x_n) / \partial x^j$, so
  `jacobian` returns `(N, M, D)`, and the 2-D curl is $J_{10} - J_{01}$
  (0-indexed), as in the invariant tests of `CurlFree` / `DivergenceFree`.
- **Batching.** `batch_size=None` maps the per-point derivative over all `N`
  rows with `jax.vmap`. An integer uses
  `jax.lax.map(..., batch_size=batch_size)`, so peak memory is
  $O(\texttt{batch\_size}\cdot\operatorname{cost}(f))$: for KRR, a
  `(batch_size, N_train)` kernel block instead of `(N, N_train)`. Both give
  the same numbers.
- **Shapes.** `X` must be `(N, D)`. The output shape of `f` is checked with
  `jax.eval_shape` on one row before anything is computed: `gradient`,
  `hessian` and `laplacian` need `(N,)`, `jacobian` needs `(N, M)`,
  `divergence` and `curl` need `(N, D)`, and `curl` needs $D \in \{2, 3\}$.
  Each violation raises a `ValueError` naming the shapes.
- **Transformations.** The operators are plain JAX functions: they `jit`, and
  `jax.grad` through them with respect to a kernel hyperparameter (in a
  closure) is the mixed derivative $\partial_\theta \nabla_x \hat f$.

## Operators

| Operator | Input | Output | Autodiff | Cost per point |
|---|---|---|---|---|
| `gradient(f, X)` | scalar field | `(N, D)` | reverse (`jax.grad`) | 1 VJP |
| `jacobian(F, X, mode=)` | vector field | `(N, M, D)` | `"auto"`: `jax.jacfwd` if $D \le M$, else `jax.jacrev` | $\min(D, M)$ passes |
| `hessian(f, X)` | scalar field | `(N, D, D)` | forward-over-reverse (`jacfwd(grad)`) | $D$ HVPs |
| `laplacian(f, X)` | scalar field | `(N,)` | trace of the per-point Hessian | $D$ HVPs |
| `divergence(F, X)` | vector field, $M = D$ | `(N,)` | trace of the forward Jacobian | $D$ JVPs |
| `curl(F, X)` | vector field, $D = M \in \{2, 3\}$ | `(N,)` in 2-D, `(N, 3)` in 3-D | forward Jacobian | $D$ JVPs |

A VJP, JVP or Hessian-vector product costs a small constant times one
evaluation of $f$ at one point. For large $D$ the $D$ passes of the exact
Laplacian and divergence dominate; matrix-free stochastic estimates with
$m \ll D$ probes are planned (#269).

## Mathematics

A kernel predictor is linear in its weights,
$\hat f(x) = \sum_{i=1}^n \alpha_i k(x, x_i)$, so every derivative is the same
expansion over the derivative of the kernel's first argument (Johnson et al.,
2020, Eqs 4–10; the derivative reproducing property of Zhou, 2008):

$$
\nabla \hat f(x) = \sum_i \alpha_i \nabla_x k(x, x_i),\qquad
H_{jk}(x) = \sum_i \alpha_i \partial_j \partial_k k(x, x_i),\qquad
\Delta \hat f(x) = \operatorname{tr} H(x).
$$

For the ARD RBF kernel, with $h_i = x - x_i$ and
$\Lambda = \operatorname{diag}(\ell_d^2)$ (the closed forms the tests check
against):

$$
\nabla_x k = -\Lambda^{-1} h\, k,\qquad
\nabla_x^2 k = \big(\Lambda^{-1} h h^\top \Lambda^{-1} - \Lambda^{-1}\big) k,\qquad
\Delta_x k = \Big(\sum_d \frac{h_d^2}{\ell_d^4} - \sum_d \frac1{\ell_d^2}\Big) k.
$$

For a vector field, with $J_{ij} = \partial F_i / \partial x^j$:

$$
\nabla \cdot F = \operatorname{tr} J,\qquad
\operatorname{curl}_{2D} F = \partial_0 F_1 - \partial_1 F_0 = J_{10} - J_{01},\qquad
\operatorname{curl}_{3D} F = (J_{21} - J_{12},\ J_{02} - J_{20},\ J_{10} - J_{01}).
$$

The operators compute exactly these, one point at a time:

```text
f_n(x)       = f(x[None])[0]                       # pointwise contract
gradient     : x -> grad(f_n)(x)                   # (D,)
hessian      : x -> jacfwd(grad(f_n))(x)           # (D, D)
laplacian    : x -> trace(jacfwd(grad(f_n))(x))    # ()
jacobian     : x -> jacfwd(F_n)(x) if D <= M else jacrev(F_n)(x)   # (M, D)
divergence   : x -> trace(jacfwd(F_n)(x))          # ()
curl         : J = jacfwd(F_n)(x)
               D = 2: J[1, 0] - J[0, 1]
               D = 3: [J[2, 1] - J[1, 2], J[0, 2] - J[2, 0], J[1, 0] - J[0, 1]]
over points  : vmap(op)(X)               if batch_size is None
               lax.map(op, X, batch_size) otherwise
```

The tests also pin the vector-calculus identities
$\operatorname{curl}\nabla\phi = 0$,
$\nabla\cdot(\operatorname{rot}\nabla\psi) = 0$ in 2-D with
$\operatorname{rot}(a, b) = (b, -a)$, $\nabla\cdot(\nabla\times A) = 0$ in
3-D, $\Delta f = \nabla\cdot\nabla f$ and
$\operatorname{jacobian}(\nabla f) = \operatorname{hessian}(f)$.

## Differentiability guard

$\hat f \in C^1$ needs $k(\cdot, x_i) \in C^1$ at $x = x_i$, and the Hessian
needs $C^2$. Matérn-½ ($e^{-r}$) has a kink at every training point, so the
gradient of its predictor does not exist there. Matérn-3/2 qualifies for both
orders: $k = 1 - \tfrac32 r^2 + \sqrt3 r^3 + O(r^4)$, and $r^3$ is $C^2$.

With `check=True` (the default), when `f` is a bound method whose owner has a
`kernel` field that is an `AbstractKernel` (`KRR.predict`, `Falkon.predict`,
`EigenPro.predict`, a scikit-learn adapter's `model_.predict`), the operator
refuses kernels that contain `Matern(nu=0.5)`, `White` or
`Distance(exponent < 2)`, the same list that `Derivative` rejects. The test is
conservative for `Distance` with $1 < a < 2$, whose predictor is $C^1$ but not
$C^2$. The lookup is duck-typed. Other callables (lambdas, closures over a
kernel expansion) are not checked, and `check=False` skips the check.

Kernels that are smooth in $r^2$ (RBF, Matérn-3/2 and 5/2, rational
quadratic) are differentiated exactly **at** the training points: the Gram
path clips the squared distance at zero in value only, so autodiff sees the
true curvature at $x = x_i$.

## Vector fields (GEO15)

`CurlFree`, `DivergenceFree` and `Helmholtz` GPs predict on point-major rows
`vector_inputs(X)`, so wrap the predictor as an `(N, D) -> (N, D)` field:

```python
from kernellib._einx import rearrange

model = kl.KRR(kl.DivergenceFree(kl.RBF(lengthscale=0.5)), regularization=1e-4)
model = model.fit(kl.vector_inputs(X), rearrange(Y, "n i -> (n i)"))


def F(Z):
    return rearrange(model.predict(kl.vector_inputs(Z)), "(n i) -> n i", i=Z.shape[-1])


div, vort = kl.divergence(F, Z), kl.curl(F, Z)  # div ≈ 0 up to rounding
```

## Reference

::: kernellib.gradient

::: kernellib.jacobian

::: kernellib.hessian

::: kernellib.laplacian

::: kernellib.divergence

::: kernellib.curl

## References

- Johnson, Laparra, Pérez-Suay, Mahecha & Camps-Valls (2020). Kernel methods
  and their derivatives: concept and perspectives for the earth system
  sciences. PLOS ONE 15(10): e0235885.
  [doi:10.1371/journal.pone.0235885](https://doi.org/10.1371/journal.pone.0235885),
  [arXiv:2007.14706](https://arxiv.org/abs/2007.14706)
- Zhou (2008). Derivative reproducing properties for kernel methods in
  learning theory. Journal of Computational and Applied Mathematics 220(1–2):
  456–463. [doi:10.1016/j.cam.2007.08.023](https://doi.org/10.1016/j.cam.2007.08.023)
