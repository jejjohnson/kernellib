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

::: kernellib.functional
    options:
      show_root_heading: false
