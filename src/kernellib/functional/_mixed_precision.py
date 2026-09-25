"""Mixed-precision RBF Gram matrix, moved from ``gaussx._linalg._mixed_precision``.

The distance step stays in gaussx: `gaussx.stable_squared_distances` is a
kernel-agnostic distance primitive that gaussx's ensemble localization also
uses, so only the kernel moved.
"""

from __future__ import annotations

import gaussx as gx
import jax.numpy as jnp
from jaxtyping import Array, Float


__all__ = [
    "stable_rbf_kernel",
]


def stable_rbf_kernel(
    X: Float[Array, "N D"],
    Z: Float[Array, "M D"],
    lengthscale: float | Float[Array, ""],
    variance: float | Float[Array, ""] = 1.0,
    *,
    compute_dtype: jnp.dtype = jnp.float32,
    accumulate_dtype: jnp.dtype = jnp.float64,
) -> Float[Array, "N M"]:
    r"""RBF (squared exponential) kernel with mixed-precision stability.

    Computes ``variance * exp(-0.5 * ||x - z||^2 / lengthscale^2)``
    using `gaussx.stable_squared_distances` for the distance computation.

    Args:
        X: First set of points, shape ``(N, D)``.
        Z: Second set of points, shape ``(M, D)``.
        lengthscale: Kernel lengthscale.
        variance: Kernel signal variance (default 1.0).
        compute_dtype: Dtype for dot products (default float32).
        accumulate_dtype: Dtype for subtraction (default float64).

    Returns:
        Kernel matrix, shape ``(N, M)``.

    Examples:
        >>> import jax.numpy as jnp
        >>> from kernellib.functional import stable_rbf_kernel
        >>> X = jnp.zeros((3, 2), dtype=jnp.float32)
        >>> K = stable_rbf_kernel(X, X, 1.0, accumulate_dtype=jnp.float32)
        >>> K.shape
        (3, 3)
    """
    dist_sq = gx.stable_squared_distances(
        X, Z, compute_dtype=compute_dtype, accumulate_dtype=accumulate_dtype
    )
    return variance * jnp.exp(-0.5 * dist_sq / lengthscale**2)
