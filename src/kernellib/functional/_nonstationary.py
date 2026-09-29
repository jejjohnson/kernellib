"""Non-stationary kernels on arrays: linear, polynomial and distance-induced.

Ported from ``pyrox_gp._src.kernels``; the math is unchanged.
"""

from __future__ import annotations

import einx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


__all__ = [
    "distance_kernel",
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


def distance_kernel(
    X1: Float[Array, "N1 D"],
    X2: Float[Array, "N2 D"],
    variance: Float[Array, ""],
    exponent: float = 1.0,
) -> Float[Array, "N1 N2"]:
    r"""Distance-induced kernel (Sejdinovic et al., 2013).

    $$
    k(x, x') = \frac{\sigma^2}{2}\bigl(\|x\|^a + \|x'\|^a - \|x - x'\|^a\bigr)
    $$

    PSD for $0 < a \le 2$, where $\|x - x'\|^a$ is conditionally negative
    definite; at $a = 2$ it is the linear kernel $\sigma^2 x^\top x'$. HSIC
    and MMD under it are distance covariance and energy distance. Powers of
    zero are guarded, so gradients stay finite at coincident points.

    Distances are taken from coordinate differences, not the norm expansion
    the other kernels use: under the ``a``-th root the expansion's
    cancellation error would erase separations that are small next to the
    inputs' magnitude. Rows of ``X1`` are processed in chunks, so the
    ``(N1, N2, D)`` difference tensor is never formed whole.

    Args:
        X1: ``(N1, D)`` inputs.
        X2: ``(N2, D)`` inputs.
        variance: Scalar multiplier.
        exponent: The exponent $a$ in ``(0, 2]``; a static Python float.

    Returns:
        ``(N1, N2)`` Gram matrix.

    Raises:
        ValueError: If ``exponent`` is outside ``(0, 2]``.

    Examples:
        >>> import jax.numpy as jnp
        >>> from kernellib.functional import distance_kernel
        >>> X = jnp.array([[3.0], [4.0]])
        >>> float(distance_kernel(X, X, jnp.array(1.0))[0, 1])  # (3 + 4 - 1) / 2
        3.0
    """
    _check_exponent(exponent)
    n1 = _norm_pow(einx.dot("n1 d, n1 d -> n1", X1, X1), exponent)
    n2 = _norm_pow(einx.dot("n2 d, n2 d -> n2", X2, X2), exponent)
    between = _pairwise_distance_pow(X1, X2, exponent)
    return 0.5 * variance * (einx.add("n1, n2 -> n1 n2", n1, n2) - between)


def _check_exponent(exponent: float) -> None:
    if not 0.0 < exponent <= 2.0:
        raise ValueError(
            f"The distance kernel needs an exponent in (0, 2], got {exponent!r}."
        )


# Elements of the per-chunk (rows, N2, D) difference tensor: 32 MiB in float64.
_CHUNK_ELEMENTS = 2**22


def _pairwise_distance_pow(
    X1: Float[Array, "N1 D"], X2: Float[Array, "N2 D"], exponent: float
) -> Float[Array, "N1 N2"]:
    """``||x - x'||^exponent`` from coordinate differences, in row chunks."""
    n2, d = X2.shape
    batch = max(1, min(X1.shape[0], _CHUNK_ELEMENTS // max(1, n2 * d)))

    def row(x: Float[Array, " D"]) -> Float[Array, " N2"]:
        diff = einx.subtract("d, n2 d -> n2 d", x, X2)
        return _norm_pow(einx.dot("n2 d, n2 d -> n2", diff, diff), exponent)

    return jax.lax.map(row, X1, batch_size=batch)


def _norm_pow(r2: Float[Array, ...], exponent: float) -> Float[Array, ...]:
    """``r2 ** (exponent / 2)``, zero at zero with a finite gradient there.

    Only an exact zero is guarded, so a NaN input stays NaN.
    """
    zero = r2 == 0
    safe = jnp.where(zero, 1.0, r2)
    return jnp.where(zero, 0.0, safe ** (0.5 * exponent))
