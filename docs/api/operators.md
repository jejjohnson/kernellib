# Operators

Kernel linear operators, moved from gaussx. Each is a
`lineax.AbstractLinearOperator` with lineax's structural predicates
registered, so it works with `gaussx.solve`, `gaussx.logdet` and every gaussx
solver strategy.

| Operator | Evaluation | Memory |
|---|---|---|
| `KernelOperator` | `K(X1, X2; params)` with a first-order custom JVP | `O(N)` per matvec via `lax.scan` |
| `ImplicitKernelOperator` | Square `K(X, X) + σ²I`, noise fused | `O(N)` per step |
| `ImplicitCrossKernelOperator` | Rectangular `K(X_data, X_inducing)`, batched rows | `O(batch · M)` |
| `nystrom_operator`, `rff_operator` | Low-rank `LowRankUpdate` approximations | `O(N · M)` |

::: kernellib.KernelOperator

::: kernellib.ImplicitKernelOperator

::: kernellib.ImplicitCrossKernelOperator

::: kernellib.implicit_cross_kernel

::: kernellib.batched_kernel_matvec

::: kernellib.batched_kernel_rmatvec

::: kernellib.nystrom_operator

::: kernellib.rff_operator
