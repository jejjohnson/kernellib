# Functional

Arrays in, arrays out: pure kernel functions on input matrices. Each takes
``X1`` of shape ``(N1, D)``, ``X2`` of shape ``(N2, D)`` and hyperparameters as
JAX arrays, and returns the ``(N1, N2)`` Gram matrix. The distance-based
kernels accept a scalar (isotropic) or ``(D,)`` (ARD) lengthscale, except
``periodic_kernel`` and ``cosine_kernel``, which are isotropic only.

::: kernellib.functional
    options:
      show_root_heading: false
