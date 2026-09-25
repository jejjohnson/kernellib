"""Arrays in, arrays out: pure kernel functions and matrix-level statistics.

Every function here takes plain JAX arrays or lineax operators and returns an
array, a scalar, or an operator. Nothing here takes a kernel object or a random
key; those APIs live at the top level of `kernellib`.

Examples:
    >>> import jax.numpy as jnp
    >>> import kernellib as kl
    >>> X = jnp.array([[0.0], [1.0]])
    >>> K = kl.functional.rbf_kernel(X, X, jnp.array(1.0), jnp.array(1.0))
    >>> K.shape
    (2, 2)
"""

from __future__ import annotations

from kernellib.functional._compose import kernel_add, kernel_mul
from kernellib.functional._nonstationary import linear_kernel, polynomial_kernel
from kernellib.functional._stationary import (
    constant_kernel,
    cosine_kernel,
    matern_kernel,
    periodic_kernel,
    rational_quadratic_kernel,
    rbf_kernel,
    white_kernel,
)
from kernellib.functional._statistics import (
    center_kernel,
    centering_operator,
    cka,
    hsic,
    mmd_squared,
)


__all__ = [
    "center_kernel",
    "centering_operator",
    "cka",
    "constant_kernel",
    "cosine_kernel",
    "hsic",
    "kernel_add",
    "kernel_mul",
    "linear_kernel",
    "matern_kernel",
    "mmd_squared",
    "periodic_kernel",
    "polynomial_kernel",
    "rational_quadratic_kernel",
    "rbf_kernel",
    "white_kernel",
]
