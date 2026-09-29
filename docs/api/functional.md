# Functional

Arrays in, arrays out: pure kernel functions and matrix-level statistics.

The kernel functions each take
``X1`` of shape ``(N1, D)``, ``X2`` of shape ``(N2, D)`` and hyperparameters as
JAX arrays, and returns the ``(N1, N2)`` Gram matrix. The distance-based
kernels accept a scalar (isotropic) or ``(D,)`` (ARD) lengthscale, except
``periodic_kernel`` and ``cosine_kernel``, which are isotropic only.

The statistics (`centering_operator`, `center_kernel`, `hsic`, `cka`,
`mmd_squared`) take kernel matrices or lineax operators and return a scalar or
an operator. `hsic` and `cka` offer the biased estimator (default) and the
unbiased estimator of Song et al. (2012).

Low-rank operators stay low-rank. For a `gaussx.LowRankUpdate` on a diagonal
base (what `nystrom_operator`, `rff_operator` and `feature_map.operator(X)`
return, with or without a noise diagonal), `center_kernel` returns a
`LowRankUpdate` of rank ``R + 3``. `hsic` and `cka` on two such operators
cost ``O(N R_x R_y)`` and never form an ``N x N`` matrix. Other operators are
materialised.

```python
K = feature_map_x.operator(X)  # gx.LowRankUpdate
L = feature_map_y.operator(Y)
kl.functional.hsic(K, L)  # O(N R_x R_y)
```

::: kernellib.functional
    options:
      show_root_heading: false
