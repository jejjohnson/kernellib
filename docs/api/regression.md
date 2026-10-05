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

Plain CG on that system needs more iterations as $\lambda$ shrinks, since the
condition number is $(\lambda_1 + \lambda n) / \lambda n$. With
`preconditioner="nystrom"` or `"rpcholesky"`, `KRR` builds a rank-`r`
preconditioner from $K$ (gaussx's `NystromPreconditioner` or randomly pivoted
`PartialCholeskyPreconditioner`) and solves by preconditioned CG. A rank near
the effective dimension $\sum_i \lambda_i / (\lambda_i + \lambda n)$ makes the
iteration count independent of $\lambda$. Choose `"rpcholesky"` when kernel
evaluations are expensive: it needs `O(N r)` of them instead of `r` full matvecs.

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

| Estimator | Solves | Time | Memory |
|---|---|---|---|
| `KRR` | $(K + \lambda n I)\alpha = y$ exactly (or by CG, optionally preconditioned) | $O(N^3)$ dense, $O(N^2 t)$ CG | $O(N^2)$ dense, $O(N)$ implicit, $+O(N r)$ preconditioner |
| `Falkon` | the Nyström system on ``M`` centres | $O(N M t + M^3)$ | $O(M^2)$ |
| `EigenPro` | $K\alpha = y$, early-stopped by ``epochs`` | $O(N^2 \cdot \text{epochs})$ | $O(B N + m^2)$ |

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
W = kl.adjacency_matrix(kl.nearest_neighbors(X_all, 10))
laprls = kl.KRR(kl.RBF(0.3), 1e-4, penalty_weight=100.0).fit(
    X_all, y_all, mask=is_labelled, penalty=kl.laplacian_penalty(W)
)
```

A pure low-rank penalty (`hsic_penalty` with a `Linear` kernel or
``approx``) and no mask is solved by Woodbury through ``solver``, so any
strategy applies. Anything else is solved by dense LU, or matrix-free by
GMRES with ``implicit=True``.

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
