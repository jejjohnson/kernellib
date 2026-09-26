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

# tune the lengthscale on a validation set
grad = jax.grad(lambda ell: kl.KRR(kl.RBF(ell), 1e-3).fit(X, y).loss(X_val, y_val))
```

The ridge is scaled by the number of points: `KRR` solves
$(K + \lambda n I)\alpha = y$, the same convention as `Falkon`.

::: kernellib.AbstractEstimator

::: kernellib.KRR

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
```

::: kernellib.falkon_preconditioner

::: kernellib.falkon_solve

::: kernellib.falkon_predict

::: kernellib.FalkonPreconditioner

## EigenPro

Spectral preconditioning for kernel stochastic gradient descent: damp the top
eigendirections of the kernel operator so the step size is governed by the
residual spectrum (Ma & Belkin, 2017).

::: kernellib.eigenpro_preconditioner

::: kernellib.eigenpro_step_size

::: kernellib.eigenpro_correction

::: kernellib.EigenProPreconditioner
