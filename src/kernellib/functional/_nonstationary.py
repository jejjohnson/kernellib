"""Inner-product kernels on arrays: linear and polynomial.

Ported from ``pyrox_gp._src.kernels``; the math is unchanged.
"""

from __future__ import annotations

import einx
from jaxtyping import Array, Float


__all__ = [
    "linear_kernel",
    "polynomial_kernel",
]


def linear_kernel(
    X1: Float[Array, "N1 D"],
    X2: Float[Array, "N2 D"],
    variance: Float[Array, ""],
    bias: Float[Array, ""],
) -> Float[Array, "N1 N2"]:
    r"""Linear kernel.

    $$
    k(x, x') = \sigma^2\, x^\top x' + b
    $$

    Args:
        X1: ``(N1, D)`` inputs.
        X2: ``(N2, D)`` inputs.
        variance: Scalar variance multiplier on the dot product.
        bias: Scalar additive bias.

    Returns:
        ``(N1, N2)`` Gram matrix.

    Examples:
        >>> import jax.numpy as jnp
        >>> from kernellib.functional import linear_kernel
        >>> X = jnp.array([[1.0, 2.0]])
        >>> float(linear_kernel(X, X, jnp.array(1.0), jnp.array(0.5))[0, 0])
        5.5
    """
    return variance * einx.dot("n1 d, n2 d -> n1 n2", X1, X2) + bias


def polynomial_kernel(
    X1: Float[Array, "N1 D"],
    X2: Float[Array, "N2 D"],
    variance: Float[Array, ""],
    bias: Float[Array, ""],
    degree: int,
) -> Float[Array, "N1 N2"]:
    r"""Polynomial kernel.

    $$
    k(x, x') = \sigma^2 \bigl(x^\top x' + b\bigr)^d
    $$

    `linear_kernel` is the special case ``degree == 1`` without the
    outer power. ``degree`` is a static Python int so the kernel specializes
    at trace time.

    Args:
        X1: ``(N1, D)`` inputs.
        X2: ``(N2, D)`` inputs.
        variance: Scalar multiplier.
        bias: Scalar additive bias inside the power.
        degree: Positive integer polynomial degree.

    Returns:
        ``(N1, N2)`` Gram matrix.

    Examples:
        >>> import jax.numpy as jnp
        >>> from kernellib.functional import polynomial_kernel
        >>> X = jnp.array([[1.0], [2.0]])
        >>> K = polynomial_kernel(X, X, jnp.array(1.0), jnp.array(0.0), 2)
        >>> float(K[0, 1])
        4.0
    """
    if degree < 1:
        raise ValueError(f"polynomial_kernel requires degree >= 1, got {degree!r}")
    dot = einx.dot("n1 d, n2 d -> n1 n2", X1, X2)
    return variance * (dot + bias) ** degree
