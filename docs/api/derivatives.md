# Derivatives

Exact, batched derivatives of **functions**: the gradient, Jacobian, Hessian
and Laplacian of a fitted predictor, and the divergence and curl of a vector
field. A **scalar field** is any callable `f: (N, D) -> (N,)` and a **vector
field** any callable `F: (N, D) -> (N, M)`: a fitted `KRR` / `Falkon` /
`EigenPro` `predict`, a closure over a kernel expansion, or any JAX function.
Every operator is autodiff, so the results are exact to rounding. On top of
them, `sensitivity` and `point_sensitivity` summarise a gradient matrix into a
per-feature and a per-sample relevance, for predictors and for dataset
functionals such as HSIC ([Sensitivity maps](#sensitivity-maps)).

```python
import kernellib as kl

model = kl.KRR(kl.RBF(lengthscale=0.5), regularization=1e-3).fit(X, y)
G = kl.gradient(model.predict, X_test)  # (N, D)
H = kl.hessian(model.predict, X_test)  # (N, D, D)
L = kl.laplacian(model.predict, X_test)  # (N,)
s = kl.sensitivity(model.predict, X_test)  # (D,)
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

## Sensitivity maps

A sensitivity map ranks the input features by how strongly a function varies
along them, and the samples by how strongly it varies around them (Johnson et
al., 2020, §2.2). Both reduce one $(N, D)$ derivative matrix $G$:

```python
model = kl.KRR(kl.RBF(lengthscale=ell_ard), regularization=1e-3).fit(X, y)
s = kl.sensitivity(model.predict, X, normalize="sum")  # (D,) feature ranking
q = kl.point_sensitivity(model.predict, X)  # (N,) sample relevance
s_clf = kl.sensitivity(lambda X: jnp.tanh(model.predict(X)), X)  # margin


def hsic_of_X(X):  # a dataset functional (N, D) -> ()
    return kl.hsic(kx, ky, X, Y)


s_hsic = kl.sensitivity(hsic_of_X, X, normalize="sum")  # features driving HSIC
q_hsic = kl.point_sensitivity(hsic_of_X, X)  # samples driving HSIC
```

### Two kinds of callable

| `f(X)` shape | Mode (`wrt=`) | $G_{nj}$ | Computed with |
|---|---|---|---|
| `(N,)`: a pointwise predictor (`model.predict`, a classifier margin, any JAX function) | `"point"` | $\partial_j f(x_n)$, how the output **at** $x_n$ changes with $x_n$ | `gradient` (the pointwise contract) |
| `()`: a dataset functional (`hsic`, `mmd_squared`, a loss of the sample) | `"dataset"` | $\partial F / \partial X_{nj}$, how the statistic of the **whole sample** changes when sample $n$ moves | `jax.grad(F)(X)` |

`wrt="auto"` (the default) reads `jax.eval_shape(f, X)` and picks the mode
from the output shape; forcing `wrt` checks that the shape agrees. A
multi-output predictor `(N, C)` raises: select an output first with
`lambda X: f(X)[:, c]`. The two modes are different objects that share only
the reduction: for a predictor, $G$ is local derivative information; for a
functional, row $n$ is the influence of sample $n$ on $F$.

### Formulation

With $\tau(u) = u^2$ (`transform="square"`, the default) or $|u|$
(`transform="abs"`):

$$
\hat s_j = \frac1N \sum_{n=1}^N \tau(G_{nj}),\qquad
q_n = \frac1D \sum_{j=1}^D \tau(G_{nj}).
$$

$\hat s_j$ is the Monte-Carlo estimate (Eq. 12) of the population
sensitivity (Eq. 11). The paper writes Eq. 11 with the marginal $p(x^j)$;
since $\partial_j f$ depends on all of $x$, the well-defined quantity is the
expectation under the joint density,

$$
s_j = \mathbb E_{x \sim p}\big[(\partial_j f(x))^2\big]
    = \int (\partial_j f(x))^2\, p(x)\, dx .
$$

$q_n$ is the paper's point sensitivity (Eq. 13), a **mean** over features:
$q_n = \|\nabla f(x_n)\|_2^2 / D$. Both are means, so

$$
\frac1D \sum_j \hat s_j = \frac1N \sum_n q_n = \frac{\|G\|_F^2}{N D},
$$

and the paper's §3.3 sum $\sum_n \|\nabla f(x_n)\|^2$ is $N \sum_j \hat s_j$.

For a kernel expansion $\hat f(x) = \sum_i \alpha_i k(x, x_i)$,
$G_{nj} = \sum_i \alpha_i \partial_j k(x_n, x_i)$; for the ARD RBF kernel,
$\partial_j k(x, y) = -(x^j - y^j) / \ell_j^2\, k(x, y)$. For a classifier
margin $g = \tanh \circ f$, autodiff applies the chain rule
$\partial_j g = (1 - g^2)\, \partial_j f$ (§4).

`order=2` replaces $G$ by the per-feature curvature
$G_{nj} = \partial_j^2 f(x_n)$ (Eq. 6), the diagonal of the exact per-point
`hessian`. It is pointwise only (a dataset functional would need the
$(ND) \times (ND)$ Hessian) and holds an $(N, D, D)$ array, so pass
`batch_size` for large $D$.

### HSIC as a dataset functional

With the biased estimator $\mathrm{HSIC}_b = \tfrac1{n^2}\operatorname{tr}(KHLH)$,
$A = HLH$ and a symmetric $k$, the input gradient (the paper's §6) is

$$
\frac{\partial\, \mathrm{HSIC}_b}{\partial x_i^q}
  = \frac{2}{n^2} \sum_{j=1}^n A_{ij}\, \frac{\partial k(x_i, x_j)}{\partial x_i^q}.
$$

The diagonal term vanishes for a stationary $k$. The rows are $O(1/n)$, so
dataset-mode sensitivities are $O(1/n^2)$: compare features with
`normalize="sum"`, not across sample sizes. A feature that $Y$ does not depend
on still moves the Gram $K$, so its finite-sample share is small but not
zero.

### Normalisation

`normalize="sum"` returns $\hat s / \sum_j \hat s_j$ (shares that sum to 1),
`"max"` returns $\hat s / \max_j \hat s_j$. A zero divisor (a constant $f$)
gives zeros, not NaN, and the division is guarded with a double `where`, so
`jax.grad` through a normalised sensitivity (with respect to a lengthscale,
say) stays finite.

### Pseudocode

```text
mode = "auto" ? {(N,): "point", (): "dataset"}[eval_shape(f, X).shape] : wrt
G = mode == "dataset"            ? grad(f)(X)                    # (N, D)
  : order == 1                   ? gradient(f, X)                # (N, D)
  :                                diag(hessian(f, X))           # (N, D)
T = transform == "square" ? G * G : |G|
s = mean_n T                                                     # (D,) sensitivity
q = mean_d T                                                     # (N,) point_sensitivity
normalize: v -> where(c > 0, v / where(c > 0, c, 1), 0),  c = sum(v) or max(v)
```

### Context: derivative-based global sensitivity

$s_j$ is the derivative-based global sensitivity measure $\nu_j$ of Sobol'
and Kucherenko (2009). For independent uniform inputs on $[0, 1]^D$ it bounds
the total Sobol' index, $S_j^{\text{tot}} \le \nu_j / (\pi^2 \operatorname{Var} f)$.
This is context, not a guarantee for kernel models fitted on other input
distributions.

## Reference

::: kernellib.gradient

::: kernellib.jacobian

::: kernellib.hessian

::: kernellib.laplacian

::: kernellib.divergence

::: kernellib.curl

::: kernellib.sensitivity

::: kernellib.point_sensitivity

## References

- Johnson, Laparra, Pérez-Suay, Mahecha & Camps-Valls (2020). Kernel methods
  and their derivatives: concept and perspectives for the earth system
  sciences. PLOS ONE 15(10): e0235885.
  [doi:10.1371/journal.pone.0235885](https://doi.org/10.1371/journal.pone.0235885),
  [arXiv:2007.14706](https://arxiv.org/abs/2007.14706)
- Pérez-Suay & Camps-Valls (2018). Sensitivity maps of the Hilbert–Schmidt
  independence criterion. Applied Soft Computing 70: 1054–1063.
  [doi:10.1016/j.asoc.2017.04.024](https://doi.org/10.1016/j.asoc.2017.04.024)
- Rasmussen, Madsen, Lund & Hansen (2011). Visualization of nonlinear kernel
  models in neuroimaging by sensitivity maps. NeuroImage 55(3): 1120–1131.
  [doi:10.1016/j.neuroimage.2010.12.035](https://doi.org/10.1016/j.neuroimage.2010.12.035)
- Sobol' & Kucherenko (2009). Derivative based global sensitivity measures
  and their link with global sensitivity indices. Mathematics and Computers
  in Simulation 79(10): 3009–3017.
  [doi:10.1016/j.matcom.2009.01.023](https://doi.org/10.1016/j.matcom.2009.01.023)
- Zhou (2008). Derivative reproducing properties for kernel methods in
  learning theory. Journal of Computational and Applied Mathematics 220(1–2):
  456–463. [doi:10.1016/j.cam.2007.08.023](https://doi.org/10.1016/j.cam.2007.08.023)
