# Operators

Kernel linear operators, moved from gaussx. Each is a
`lineax.AbstractLinearOperator` with lineax's structural predicates registered
(including `lineax.diagonal`, which gaussx's partial-Cholesky preconditioner
reads), so it works with `gaussx.solve`, `gaussx.logdet` and every gaussx
solver strategy.

Most code should not build these directly: `to_operator(kernel, X, ...)` and
`to_cross_operator(kernel, X1, X2, ...)` turn a kernel object into the right
operator and set its tags. See the
[matrix-free GP example](../../matrix-free-gp/) for an end-to-end walkthrough.

## Choosing between the three kernel operators

All three are **matrix-free and scan-based**: none of them ever materializes
its kernel block. The word *implicit* in two of the names is therefore not the
distinction it appears to be. What separates them is the **shape contract**,
whether a **noise term is fused in**, and the **scan granularity**:

| | `KernelOperator` | `ImplicitKernelOperator` | `ImplicitCrossKernelOperator` |
|---|---|---|---|
| **Shape** | Rectangular $(N, M)$ | Square $(N, N)$ | Rectangular $(N, M)$ |
| **Points** | `X1`, `X2` (independent) | `X` (one set) | `X_data`, `X_inducing` |
| **Noise term** | — | **fused** `+ noise_var * I` | — |
| **Scan step** | one row of `X1` | one row of `X` | `batch_size` rows (default 1024) |
| **Peak memory/step** | $O(M)$ | $O(N)$ | $O(\texttt{batch\_size} \times M)$ |
| **Kernel signature** | `k(params, x, x')` (required) | `k(x, x')` or `k(params, x, x')` | `k(x, z)` or `k(params, x, z)` |
| **`jax.custom_jvp`** | always | only with `params=` | only with `params=` |
| **Built by** | — | `to_operator(..., implicit=True)` | `to_cross_operator(..., implicit=True)` |

!!! warning "The custom JVP follows `params`, not the class"
    `KernelOperator` takes `params` as a required argument, so its matvec
    always runs under a `jax.custom_jvp` that differentiates the kernel
    without materializing Jacobians. The two `Implicit*` operators default to
    `params=None`, and in that mode `mv` runs the ordinary scan with no custom
    rule: autodiff falls back to differentiating straight through the scan.
    Pass a `params` pytree (and use the `k(params, x, x')` signature) if you
    differentiate with respect to hyperparameters and want the efficient
    path. `to_operator` and `to_cross_operator` always do this for you: they
    split a kernel's arrays out as `params`.

!!! note "Tags are claims, not inferences"
    An operator does not work out for itself that a kernel is symmetric or
    positive semidefinite. `to_operator` tags its result symmetric and PSD. If
    you build an `ImplicitKernelOperator` from a callable, pass
    `tags=frozenset({lx.symmetric_tag, lx.positive_semidefinite_tag})`
    yourself, or gaussx cannot pick its symmetric / PSD fast paths.

Practical guidance:

- Building a **training covariance** you will pass to CG, preconditioned CG
  or BBMM? Use `to_operator(kernel, X, noise=..., implicit=True)`, which
  returns an `ImplicitKernelOperator`. The fused noise term means $K$ and
  $\sigma^2 I$ never exist as separate operators. The noise must be a
  concrete value on this path; use `implicit=False` to trace or
  differentiate it.
- Need a **general kernel block** between two point sets from a raw callable?
  `KernelOperator`.
- Need a **data-inducing block** and want to trade peak memory for throughput?
  `to_cross_operator(kernel, X, Z, implicit=True, batch_size=...)`, which
  returns an `ImplicitCrossKernelOperator`. Tune `batch_size`.

!!! note "Naming rule for future kernel operators"
    Every kernel operator here is matrix-free, so *implicit*, *lazy*, and
    *matrix-free* carry no information in a class name, and neither would
    *scan*, since all three scan. Name a new kernel operator after the part of
    its **contract** that differs: what shape it produces, what it fuses in,
    or what point sets it relates. The existing `Implicit*` names predate this
    rule and are kept for compatibility; see
    [gaussx#135](https://github.com/jejjohnson/gaussx/issues/135) for the
    rename discussion that began before the move.

::: kernellib.KernelOperator

::: kernellib.ImplicitKernelOperator

::: kernellib.ImplicitCrossKernelOperator

::: kernellib.implicit_cross_kernel

::: kernellib.batched_kernel_matvec

::: kernellib.batched_kernel_rmatvec

## Low-rank approximations

Nyström ($K \approx K_{nm} K_{mm}^{-1} K_{mn}$) and random-Fourier-feature
approximations, returned as `gaussx.LowRankUpdate` so solves and
log-determinants go through Woodbury automatically. See the
[kernel approximations example](../../kernel-approximations/).

::: kernellib.nystrom_operator

::: kernellib.rff_operator
