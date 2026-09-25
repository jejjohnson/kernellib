"""Distance-based kernels on arrays: RBF, Matern, rational quadratic, periodic,
cosine, white noise, constant.

Each function takes ``X1`` of shape ``(N1, D)``, ``X2`` of shape ``(N2, D)``,
hyperparameters as JAX arrays (structural ones such as ``nu`` as Python
values), and returns the ``(N1, N2)`` Gram matrix. All inputs are 2-D; add a
trailing singleton axis to 1-D inputs first.

Ported from ``pyrox_gp._src.kernels``; the math is unchanged.
"""

from __future__ import annotations

import einx
import jax.numpy as jnp
from jaxtyping import Array, Float

from kernellib.functional._distances import _pairwise_sq_dist


__all__ = [
    "constant_kernel",
    "cosine_kernel",
    "matern_kernel",
    "periodic_kernel",
    "rational_quadratic_kernel",
    "rbf_kernel",
    "white_kernel",
]


def rbf_kernel(
    X1: Float[Array, "N1 D"],
    X2: Float[Array, "N2 D"],
    variance: Float[Array, ""],
    lengthscale: Float[Array, ""] | Float[Array, " D"],
) -> Float[Array, "N1 N2"]:
    r"""Radial basis function (squared exponential) kernel.

    $$
    k(x, x') = \sigma^2 \exp\!\left(-\frac{\|x - x'\|^2}{2\ell^2}\right)
    $$

    With a ``(D,)`` lengthscale each input dimension is scaled by its own
    $\ell_d$ (automatic relevance determination, ARD).

    Args:
        X1: ``(N1, D)`` inputs.
        X2: ``(N2, D)`` inputs.
        variance: Scalar signal variance ``sigma^2``.
        lengthscale: Scalar (isotropic) or ``(D,)`` (ARD) lengthscale ``ell``.

    Returns:
        ``(N1, N2)`` kernel Gram matrix.

    Examples:
        >>> import jax.numpy as jnp
        >>> from kernellib.functional import rbf_kernel
        >>> X = jnp.array([[0.0], [1.0], [2.0]])
        >>> K = rbf_kernel(X, X, jnp.array(2.5), jnp.array(1.0))
        >>> K.shape
        (3, 3)
        >>> bool(jnp.allclose(jnp.diag(K), 2.5))
        True
    """
    sq = _pairwise_sq_dist(X1, X2, lengthscale)
    return variance * jnp.exp(-0.5 * sq)


def matern_kernel(
    X1: Float[Array, "N1 D"],
    X2: Float[Array, "N2 D"],
    variance: Float[Array, ""],
    lengthscale: Float[Array, ""] | Float[Array, " D"],
    nu: float,
) -> Float[Array, "N1 N2"]:
    r"""Matern kernel with closed-form ``nu in {1/2, 3/2, 5/2}``.

    $$
    k(x, x') = \sigma^2\, f_\nu(r / \ell),
    \qquad r = \|x - x'\|
    $$

    Only the three common half-integer orders are supported because those
    admit closed-form expressions without Bessel evaluations. ``nu`` is a
    static Python float (not a JAX array) so the branch specializes at
    trace time. With a ``(D,)`` lengthscale each input dimension is scaled
    by its own $\ell_d$ (automatic relevance determination, ARD).

    Args:
        X1: ``(N1, D)`` inputs.
        X2: ``(N2, D)`` inputs.
        variance: Scalar signal variance.
        lengthscale: Scalar (isotropic) or ``(D,)`` (ARD) lengthscale.
        nu: Smoothness parameter; must be ``0.5``, ``1.5``, or ``2.5``.

    Returns:
        ``(N1, N2)`` Gram matrix.

    Examples:
        >>> import jax.numpy as jnp
        >>> from kernellib.functional import matern_kernel
        >>> X = jnp.array([[0.0], [1.0]])
        >>> K = matern_kernel(X, X, jnp.array(1.0), jnp.array(1.0), 0.5)
        >>> bool(jnp.allclose(K[0, 1], jnp.exp(-1.0)))
        True
    """
    sq = _pairwise_sq_dist(X1, X2, lengthscale)
    # Jitter inside sqrt avoids NaN gradients at r = 0 (sqrt' is undefined).
    r = jnp.sqrt(jnp.clip(sq, min=1e-30))
    if nu == 0.5:
        shape = jnp.exp(-r)
    elif nu == 1.5:
        a = jnp.sqrt(3.0) * r
        shape = (1.0 + a) * jnp.exp(-a)
    elif nu == 2.5:
        a = jnp.sqrt(5.0) * r
        shape = (1.0 + a + (a * a) / 3.0) * jnp.exp(-a)
    else:
        raise ValueError(f"matern_kernel supports nu in {{0.5, 1.5, 2.5}}, got {nu!r}")
    return variance * shape


def periodic_kernel(
    X1: Float[Array, "N1 D"],
    X2: Float[Array, "N2 D"],
    variance: Float[Array, ""],
    lengthscale: Float[Array, ""],
    period: Float[Array, ""],
) -> Float[Array, "N1 N2"]:
    r"""Periodic (MacKay) kernel.

    $$
    k(x, x') = \sigma^2 \exp\!\left(
        -\frac{2 \sin^2(\pi \|x - x'\| / p)}{\ell^2}
    \right)
    $$

    For multi-dimensional inputs the argument uses the Euclidean distance,
    matching the common GPML convention.

    Args:
        X1: ``(N1, D)`` inputs.
        X2: ``(N2, D)`` inputs.
        variance: Scalar signal variance.
        lengthscale: Scalar lengthscale.
        period: Scalar period ``p``.

    Returns:
        ``(N1, N2)`` Gram matrix.

    Examples:
        >>> import jax.numpy as jnp
        >>> from kernellib.functional import periodic_kernel
        >>> x0, x1 = jnp.array([[0.0]]), jnp.array([[2.0]])
        >>> args = (jnp.array(1.0), jnp.array(0.5), jnp.array(2.0))
        >>> bool(jnp.allclose(periodic_kernel(x0, x1, *args), 1.0, atol=1e-5))
        True
    """
    sq = _pairwise_sq_dist(X1, X2)
    # Jitter inside sqrt avoids NaN gradients at r = 0 (sqrt' is undefined).
    r = jnp.sqrt(jnp.clip(sq, min=1e-30))
    sinsq = jnp.sin(jnp.pi * r / period) ** 2
    return variance * jnp.exp(-2.0 * sinsq / (lengthscale * lengthscale))


def rational_quadratic_kernel(
    X1: Float[Array, "N1 D"],
    X2: Float[Array, "N2 D"],
    variance: Float[Array, ""],
    lengthscale: Float[Array, ""] | Float[Array, " D"],
    alpha: Float[Array, ""],
) -> Float[Array, "N1 N2"]:
    r"""Rational quadratic kernel.

    $$
    k(x, x') = \sigma^2 \left(
        1 + \frac{\|x - x'\|^2}{2\alpha \ell^2}
    \right)^{-\alpha}
    $$

    Scale mixture of RBF kernels: the limit ``alpha -> infty`` recovers the
    RBF, small ``alpha`` yields heavier-tailed correlations. With a ``(D,)``
    lengthscale each input dimension is scaled by its own $\ell_d$
    (automatic relevance determination, ARD).

    Args:
        X1: ``(N1, D)`` inputs.
        X2: ``(N2, D)`` inputs.
        variance: Scalar signal variance.
        lengthscale: Scalar (isotropic) or ``(D,)`` (ARD) lengthscale.
        alpha: Scalar shape parameter; must be positive.

    Returns:
        ``(N1, N2)`` Gram matrix.

    Examples:
        >>> import jax.numpy as jnp
        >>> from kernellib.functional import rational_quadratic_kernel
        >>> X = jnp.array([[0.0], [1.0]])
        >>> K = rational_quadratic_kernel(
        ...     X, X, jnp.array(1.5), jnp.array(1.0), jnp.array(2.0)
        ... )
        >>> bool(jnp.allclose(jnp.diag(K), 1.5))
        True
    """
    sq = _pairwise_sq_dist(X1, X2, lengthscale)
    return variance * (1.0 + sq / (2.0 * alpha)) ** (-alpha)


def cosine_kernel(
    X1: Float[Array, "N1 D"],
    X2: Float[Array, "N2 D"],
    variance: Float[Array, ""],
    period: Float[Array, ""],
) -> Float[Array, "N1 N2"]:
    r"""Cosine kernel.

    $$
    k(x, x') = \sigma^2 \cos\!\left(
        \frac{2 \pi \|x - x'\|}{p}
    \right)
    $$

    Useful as a simple periodic building block alongside
    `periodic_kernel`; unlike the Mackay form this one uses plain
    cosine of distance and can go negative.

    Args:
        X1: ``(N1, D)`` inputs.
        X2: ``(N2, D)`` inputs.
        variance: Scalar signal variance.
        period: Scalar period.

    Returns:
        ``(N1, N2)`` Gram matrix.

    Examples:
        >>> import jax.numpy as jnp
        >>> from kernellib.functional import cosine_kernel
        >>> x0, x1 = jnp.array([[0.0]]), jnp.array([[1.0]])
        >>> K = cosine_kernel(x0, x1, jnp.array(1.0), jnp.array(2.0))
        >>> bool(jnp.allclose(K, -1.0, atol=1e-5))
        True
    """
    sq = _pairwise_sq_dist(X1, X2)
    # Jitter inside sqrt avoids NaN gradients at r = 0 (sqrt' is undefined).
    r = jnp.sqrt(jnp.clip(sq, min=1e-30))
    return variance * jnp.cos(2.0 * jnp.pi * r / period)


def white_kernel(
    X1: Float[Array, "N1 D"],
    X2: Float[Array, "N2 D"],
    variance: Float[Array, ""],
) -> Float[Array, "N1 N2"]:
    r"""White-noise kernel.

    $$
    k(x, x') = \sigma^2 \,\delta(x, x')
    $$

    Nonzero only where ``X1[i]`` exactly matches ``X2[j]`` across all feature
    dimensions. When evaluated at ``X1 == X2`` this yields ``sigma^2 * I``.

    Args:
        X1: ``(N1, D)`` inputs.
        X2: ``(N2, D)`` inputs.
        variance: Scalar noise variance.

    Returns:
        ``(N1, N2)`` Gram matrix.

    Examples:
        >>> import jax.numpy as jnp
        >>> from kernellib.functional import white_kernel
        >>> X = jnp.array([[0.0], [1.0]])
        >>> bool(jnp.allclose(white_kernel(X, X, jnp.array(0.5)), 0.5 * jnp.eye(2)))
        True
    """
    # Pairwise feature difference (N1, N2, D) via named broadcasting.
    diff = einx.subtract("n1 d, n2 d -> n1 n2 d", X1, X2)
    match = jnp.all(diff == 0.0, axis=-1)
    return variance * match.astype(X1.dtype)


def constant_kernel(
    X1: Float[Array, "N1 D"],
    X2: Float[Array, "N2 D"],
    variance: Float[Array, ""],
) -> Float[Array, "N1 N2"]:
    r"""Constant kernel.

    $$
    k(x, x') = \sigma^2
    $$

    A rank-one kernel useful as a scalar offset additive component.

    Args:
        X1: ``(N1, D)`` inputs.
        X2: ``(N2, D)`` inputs.
        variance: Scalar value.

    Returns:
        ``(N1, N2)`` Gram matrix filled with ``variance``.

    Examples:
        >>> import jax.numpy as jnp
        >>> from kernellib.functional import constant_kernel
        >>> K = constant_kernel(
        ...     jnp.zeros((3, 1)), jnp.zeros((4, 1)), jnp.array(1.8)
        ... )
        >>> K.shape
        (3, 4)
    """
    return variance * jnp.ones((X1.shape[0], X2.shape[0]), dtype=X1.dtype)
