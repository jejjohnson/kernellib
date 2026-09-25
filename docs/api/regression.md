# Regression

The Falkon and EigenPro primitives, moved from gaussx. The estimator-level
workflows (`KRR`, `Falkon`, `EigenPro`) will build on these.

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
