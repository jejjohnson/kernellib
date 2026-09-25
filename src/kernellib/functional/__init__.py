"""Arrays in, arrays out: pure kernel functions on input matrices.

Every function here takes plain JAX arrays (inputs and hyperparameters) and
returns a plain JAX array. Nothing here takes a kernel object or a random
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


__all__ = [
    "constant_kernel",
    "cosine_kernel",
    "kernel_add",
    "kernel_mul",
    "linear_kernel",
    "matern_kernel",
    "periodic_kernel",
    "polynomial_kernel",
    "rational_quadratic_kernel",
    "rbf_kernel",
    "white_kernel",
]
