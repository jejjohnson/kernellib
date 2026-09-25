"""Composition of already-evaluated Gram matrices.

These act on matrices, not on kernels. Kernel-level composition (``Sum``,
``Product``) arrives with the kernel classes.
"""

from __future__ import annotations

from jaxtyping import Array, Float


__all__ = [
    "kernel_add",
    "kernel_mul",
]


def kernel_add(
    K1: Float[Array, "N1 N2"],
    K2: Float[Array, "N1 N2"],
) -> Float[Array, "N1 N2"]:
    """Pointwise sum of two already-evaluated Gram matrices.

    Examples:
        >>> import jax.numpy as jnp
        >>> from kernellib.functional import kernel_add
        >>> float(kernel_add(jnp.ones((2, 2)), jnp.ones((2, 2)))[0, 0])
        2.0
    """
    return K1 + K2


def kernel_mul(
    K1: Float[Array, "N1 N2"],
    K2: Float[Array, "N1 N2"],
) -> Float[Array, "N1 N2"]:
    """Pointwise (Hadamard) product of two already-evaluated Gram matrices.

    Examples:
        >>> import jax.numpy as jnp
        >>> from kernellib.functional import kernel_mul
        >>> float(kernel_mul(2.0 * jnp.ones((2, 2)), 3.0 * jnp.ones((2, 2)))[0, 0])
        6.0
    """
    return K1 * K2
