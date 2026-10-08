# Regression

Kernel regression estimators, and the Falkon and EigenPro primitives they
build on (moved from gaussx).

## Estimators

Every estimator follows `AbstractEstimator`: configuration in the constructor,
`fit(X, y, *, key=None)` returns a fitted copy whose state (training inputs,
weights ``alpha``, landmarks) is plain fields, and `predict(X)` evaluates it.
Targets may be ``(N,)`` or ``(N, C)``. Fitted estimators are PyTrees, so a
validation loss differentiates straight through `fit`:

```python
import gaussx as gx
import jax
import kernellib as kl

model = kl.KRR(kl.RBF(lengthscale=0.5), regularization=1e-3).fit(X, y)
y_hat = model.predict(X_test)

# matrix-free: O(N) memory, conjugate gradients
model = kl.KRR(k, 1e-3, solver=gx.CGSolver(), implicit=True).fit(X, y)

# matrix-free and preconditioned: CG iterations stay O(1) as N grows
model = kl.KRR(
    k, 1e-6, implicit=True, preconditioner="rpcholesky", preconditioner_rank=1000
).fit(X, y, key=key)

# tune the lengthscale on a validation set
grad = jax.grad(lambda ell: kl.KRR(kl.RBF(ell), 1e-3).fit(X, y).loss(X_val, y_val))
```

The ridge is scaled by the number of points: `KRR` solves
$(K + \lambda n I)\alpha = y$, the same convention as `Falkon`.

### Preconditioned KRR

Plain CG on that system needs more iterations as $\lambda$ shrinks. With
`preconditioner="nystrom"` or `"rpcholesky"`, `KRR` builds a rank-`r`
preconditioner from $K$ (gaussx's `NystromPreconditioner` or randomly pivoted
`PartialCholeskyPreconditioner`) and solves by preconditioned CG, whose
iteration count no longer depends on $\lambda$ once `r` is near the effective
dimension. Choose `"rpcholesky"` when kernel evaluations are expensive: it
needs `O(N r)` of them instead of `r` full matvecs. The maths, the costs and
the failure modes are under [Formulations](#formulations).

| Field | Values | Default |
|---|---|---|
| `preconditioner` | `"none"`, `"nystrom"`, `"rpcholesky"` | `"none"` |
| `preconditioner_rank` | rank `r` (capped at `N`) | `200` |

A preconditioner needs a `key` in `fit` (both constructions are randomized)
and replaces `solver`: passing both is an error. Combine it with
`implicit=True` so that $K$ is never formed:

```python
model = kl.KRR(
    kl.Matern(nu=1.5, lengthscale=0.3),
    regularization=1e-7,
    implicit=True,
    preconditioner="nystrom",
    preconditioner_rank=500,
).fit(X, y, key=jax.random.key(0))  # n = 1e5: O(N r) memory
```

### Falkon and EigenPro

`Falkon` is Nyström KRR for large ``N``: ``M`` centres, a preconditioned CG
solve and a streamed ``N x M`` cross kernel, so memory is ``O(M^2)``.
`EigenPro` fits the interpolating solution by preconditioned mini-batch SGD
and never forms more than a ``B x N`` block.

```python
model = kl.Falkon(k, n_inducing=2000, regularization=1e-4).fit(X, y, key=key)
model.landmarks  # the chosen centres

model = kl.EigenPro(
    k, epochs=10, batch_size=512, subsample_size=4000, n_components=100
).fit(X, y, key=key)
```

See [Formulations](#formulations) for what each one solves and what it
costs.

## Formulations

Throughout, $K = K_{XX}$ is the ``N x N`` training Gram matrix with
eigenvalues $\sigma_1 \ge \dots \ge \sigma_N \ge 0$, $\lambda$ is
``regularization`` and $n = N$.

### KRR and the ridge convention

`KRR` minimises the mean squared error plus an RKHS penalty,

$$
\min_\alpha\ \frac1n\|y - K\alpha\|^2 + \lambda\,\alpha^\top K\alpha
\quad\Longrightarrow\quad
(K + \lambda n I)\,\alpha = y, \qquad f(x) = k(x, X)\,\alpha .
$$

The ridge is $\lambda n$, not $\lambda$: the loss is an average, so the same
$\lambda$ means the same amount of smoothing as $N$ grows, and `KRR` and
`Falkon` (with every point a centre) solve the same system.
`sklearn.kernel_ridge.KernelRidge(alpha=a)` solves $(K + aI)\alpha = y$,
which is $\lambda = a / n$ here; kernellib's own adapter
`kernellib.sklearn.KernelRidge` keeps the $\lambda n$ convention. The default solver is a dense Cholesky,
$O(N^3)$ time and $O(N^2)$ memory.

### Why conjugate gradients stalls

CG on $K + \lambda n I$ converges at a rate set by the condition number

$$
\kappa = \frac{\sigma_1 + \lambda n}{\sigma_N + \lambda n}
\approx \frac{\sigma_1}{\lambda n},
\qquad
\frac{\|e_t\|_{K + \lambda n I}}{\|e_0\|_{K + \lambda n I}} \le
2\Big(\frac{\sqrt\kappa - 1}{\sqrt\kappa + 1}\Big)^{t},
$$

for the error $e_t$ in the energy norm, so reaching a relative error $\varepsilon$ takes
$t \approx \tfrac12\sqrt\kappa\,\log(2/\varepsilon)$ iterations. For a kernel
with $k(x, x) = 1$, $\sigma_1$ grows like $n$ times the top eigenvalue of
the kernel's integral operator, so $\kappa$ is about that eigenvalue divided
by $\lambda$, and it is the small ridges that a good fit needs which make CG
slow: $\lambda = 10^{-6}$ means $\sqrt\kappa \sim 10^3$ iterations, each an
$O(N^2)$ matvec. Only the top of the spectrum is the problem. The number of
eigenvalues above the ridge is the **effective dimension**

$$
d_{\mathrm{eff}}(\lambda n) = \operatorname{tr}\big(K (K + \lambda n I)^{-1}\big)
= \sum_i \frac{\sigma_i}{\sigma_i + \lambda n},
$$

which for smooth kernels is far smaller than $N$, and a preconditioner that
captures those directions removes the dependence on $\lambda$.

### Preconditioned conjugate gradients

With `preconditioner=`, `KRR` builds a rank-$r$ approximation $\hat K
\approx K$ once, takes $P = \hat K + \mu I$ with $\mu = \lambda n$ (the shift
is passed separately, so the ridge is never counted twice), applies
$P^{-1}$ through the Woodbury identity in $O(Nr)$, and runs preconditioned
CG on $K + \mu I$.

- **`"nystrom"`**: randomized Nyström (`gaussx.randomized_nystrom`),
  $\hat K = U\hat\Lambda U^\top$ from $r$ matvecs of $K$ with a Gaussian
  test matrix (Frangella, Tropp & Udell, 2023). The inverse is applied as
  $$
  P^{-1}x = (\hat\lambda_r + \mu)\,U(\hat\Lambda + \mu I)^{-1}U^\top x
  + (x - UU^\top x),
  $$
  and $\kappa(P^{-1/2}(K + \mu I)P^{-1/2}) \le (\hat\lambda_r + \mu +
  \|K - \hat K\|)/\mu$, which is $O(1)$ once $r \gtrsim
  d_{\mathrm{eff}}(\mu)$ (gaussx documents $r = 2\lceil 1.5\,
  d_{\mathrm{eff}}\rceil + 1$ for an expected $\kappa < 28$).
- **`"rpcholesky"`**: a partial Cholesky $\hat K = FF^\top$ with randomly
  pivoted columns (`gaussx.rp_cholesky`; Chen, Epperly, Tropp & Webber,
  2023; used as a KRR preconditioner by Díaz et al., 2023), which is the
  column Nyström approximation on the pivots (see
  [landmark selection](spectral.md#landmark-selection)). It needs the
  diagonal and $r$ columns of $K$, $O(Nr)$ kernel evaluations, instead of
  $r$ full matvecs, and is applied as
  $$
  P^{-1}x = \mu^{-1}\big(x - F(\mu I + F^\top F)^{-1}F^\top x\big).
  $$

**Cost.** Building: $r$ matvecs of $K$, $O(N^2 r)$ kernel evaluations when
$K$ is implicit, for `"nystrom"`; $O(Nr)$ kernel evaluations and
$O(Nr^2)$ flops for `"rpcholesky"`. Each CG step: one matvec of $K$
($O(N^2)$) plus $O(Nr)$. Memory: $O(Nr)$ for the factor, plus $O(N)$ with
`implicit=True` or $O(N^2)$ without.

**Numerics.**

- CG stops at `rtol = atol = tol` (default `1e-6`) or after `max_steps`
  (default 1000) steps. If it does not converge, `fit` **raises**
  (lineax's "maximum number of solver steps was reached"); with
  `throw=False` it returns the iterate at its budget, as `Falkon` always
  does, sets `converged=False` and warns.
- The rank is capped at $N$; past the numerical rank of $K$ the
  RPCholesky factor stops adding columns (pivots are guarded as in LAPACK
  `?pstrf`) instead of producing NaNs.
- In float32, a $\lambda$ whose $\mu = \lambda n$ is near ``eps`` times
  $\sigma_1$ is below the working precision of $K + \mu I$. CG can then
  fail to converge (and raise), or stop early on a poor solution without
  raising. Use float64 for small ridges, and check a validation loss.

### The Falkon system

Nyström KRR on $M$ centres $Z$, $f = \sum_j \alpha_j k(\cdot, z_j)$, solves
$(K_{nm}^\top K_{nm} + \lambda n K_{mm})\,\alpha = K_{nm}^\top y$ by CG
preconditioned with two $M \times M$ Choleskys,
$T = \operatorname{chol}(K_{mm} + \epsilon I)$ and
$A = \operatorname{chol}(\tfrac1m TT^\top + \lambda I)$, which invert the
Nyström approximation $\frac nm K_{mm}^2 + \lambda n K_{mm}$ of the system
matrix. The full derivation, pseudocode, jitter rule and failure modes are
in the [`Falkon`][kernellib.Falkon] docstring below, and the primitives are
under [Falkon](#falkon).

### EigenPro's preconditioned SGD

Mini-batch SGD on $\frac1{2n}\|K\alpha - y\|^2$ (no ridge; `epochs` is the
regulariser) with the spectral preconditioner
$P = I - \sum_{i \le q}(1 - (\lambda_{q+1}/\lambda_i)^a)\, e_i\otimes e_i$
built from the top-$q$ eigenpairs of $K_{SS}/m$ on an $m$-point subsample
($a$ = `decay`, $a = 1$ in Ma & Belkin). The largest stable step grows from
about $2/\lambda_1$ to about $2/\lambda_{q+1}$ (in eigenvalues of
$K_{SS}/m$; $2m/\sigma_{q+1}$ in those of $K_{SS}$). See the
[`EigenPro`][kernellib.EigenPro] docstring for the update, the step-size
rule and pseudocode.

### Cost and memory

| Estimator | Setup | Per iteration | Iterations | Memory |
|---|---|---|---|---|
| `KRR`, dense | $O(N^3)$ Cholesky | – | – | $O(N^2)$ |
| `KRR`, CG | – | $O(N^2)$ | $\sim\sqrt{\sigma_1/\lambda n}$ | $O(N)$ implicit |
| `KRR`, preconditioned CG | $O(N^2 r)$ (`"nystrom"`) or $O(Nr^2)$ (`"rpcholesky"`) | $O(N^2 + Nr)$ | $O(1)$ in $\lambda$ once $r \gtrsim d_{\mathrm{eff}}$ | $O(Nr)$ implicit |
| `Falkon` | $O(M^3)$, $O(M^2)$ kernel evaluations | $O(NM)$ kernel evaluations $+\,O(M^2)$ | a few tens (`max_iter`) | $O(M^2)$ implicit, $O(NM)$ otherwise |
| `EigenPro` | $O(m^3)$ `eigh` $+\,O(Nm)$ kernel evaluations | $O(bN + bm)$ per mini-batch | `epochs` $\times\ N/b$ | $O(bN + m^2)$ |

### References

- Frangella, Tropp & Udell (2023). Randomized Nyström preconditioning.
  SIAM J. Matrix Anal. Appl. [arXiv:2110.02820](https://arxiv.org/abs/2110.02820)
- Chen, Epperly, Tropp & Webber (2023). Randomly pivoted Cholesky:
  practical approximation of a kernel matrix with few entry evaluations.
  [arXiv:2207.06503](https://arxiv.org/abs/2207.06503)
- Díaz, Epperly, Frangella, Tropp & Webber (2023). Robust, randomized
  preconditioning for kernel ridge regression.
  [arXiv:2304.12465](https://arxiv.org/abs/2304.12465)
- Rudi, Carratino & Rosasco (2017). FALKON: An optimal large scale kernel
  method. NeurIPS. [arXiv:1705.10958](https://arxiv.org/abs/1705.10958)
- Meanti, Carratino, Rosasco & Rudi (2020). Kernel methods through the
  roof: handling billions of points efficiently. NeurIPS.
  [arXiv:2006.10350](https://arxiv.org/abs/2006.10350)
- Ma & Belkin (2017). Diving into the shallows: a computational perspective
  on large-scale shallow learning. NeurIPS.
  [arXiv:1703.10622](https://arxiv.org/abs/1703.10622)
- Ma & Belkin (2019). Kernel machines that adapt to GPUs for effective large
  batch training (EigenPro 2.0). SysML.
  [arXiv:1806.06144](https://arxiv.org/abs/1806.06144)

## Penalised KRR: fairness and graph smoothness

`KRR.fit` takes an optional penalty operator $M$ and a mask of labelled
points, and minimises
$\frac1l\|J(y - K\alpha)\|^2 + \lambda\,\alpha^\top K\alpha
+ \mu\,\alpha^\top K M K\alpha$ in closed form
(``penalty_weight`` is $\mu$):

```python
# Fair KRR: predictions (nearly) independent of protected attributes S
fair = kl.KRR(kl.RBF(1.0), 1e-3, penalty_weight=30.0).fit(
    X,
    y,
    penalty=kl.hsic_penalty(kl.Linear(), S),  # rank P: P + 1 KRR solves
)

# Laplacian-regularised least squares: a few labels, many unlabelled points
graph = kl.knn_graph(X_all, 10)  # sparse: the penalty stays O(nnz)
laprls = kl.KRR(kl.RBF(0.3), 1e-4, penalty_weight=100.0).fit(
    X_all, y_all, mask=is_labelled, penalty=kl.laplacian_penalty(graph)
)
```

A pure low-rank penalty (`hsic_penalty` with a `Linear` kernel or
``approx``) and no mask is solved by Woodbury through ``solver``, so any
strategy applies. Anything else is solved by dense LU, or matrix-free by
GMRES with ``implicit=True``. GMRES takes ``tol`` and ``max_steps`` like
the preconditioned CG, its steps counted in restart cycles of up to
``min(N, 50)`` Krylov iterations; an explicit ``solver`` with tolerances
overrides them.

::: kernellib.AbstractEstimator

::: kernellib.KRR

::: kernellib.hsic_penalty

::: kernellib.laplacian_penalty

::: kernellib.Falkon

::: kernellib.EigenPro

## Falkon

Nyström kernel ridge regression solves
$(K_{nm}^\top K_{nm} + \lambda n K_{mm})\alpha = K_{nm}^\top y$. Falkon
(Rudi et al., 2017; Meanti et al., 2020) preconditions it with two
upper-triangular $M \times M$ Choleskys, so conjugate gradients needs only
triangular solves and matvecs with $K_{nm}$, which an
`ImplicitCrossKernelOperator` provides without forming the $N \times M$ matrix.

```python
import kernellib as kl

precond = kl.falkon_preconditioner(K_mm, regularization=lam)
K_nm = kl.ImplicitCrossKernelOperator(kernel_fn, X_train, Z)
alpha = kl.falkon_solve(K_nm, y_train, precond, regularization=lam)
y_pred = kl.falkon_predict(kernel_fn, Z, alpha, X_test)

# CG steps taken and whether the tolerance was reached
alpha, info = kl.falkon_solve(K_nm, y_train, precond, lam, return_info=True)
info.n_iter, info.converged
```

::: kernellib.falkon_preconditioner

::: kernellib.falkon_solve

::: kernellib.falkon_predict

::: kernellib.FalkonPreconditioner

::: kernellib.FalkonInfo

## EigenPro

Spectral preconditioning for kernel stochastic gradient descent: damp the top
eigendirections of the kernel operator so the step size is governed by the
residual spectrum (Ma & Belkin, 2017).

::: kernellib.eigenpro_preconditioner

::: kernellib.eigenpro_step_size

::: kernellib.eigenpro_correction

::: kernellib.EigenProPreconditioner

The [regression at scale](../../regression-at-scale/) notebook benchmarks
dense, CG and preconditioned `KRR`, `Falkon`, `EigenPro` and ridge on random
features against each other as $N$ grows, and ends with a table of which to
use for which $N$ and with what settings.
