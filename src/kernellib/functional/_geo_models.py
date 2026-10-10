"""Classical geostatistics covariance models on arrays (GEO5).

Spherical, cubic, pentaspherical, hole-effect, stable (powered exponential),
generalized Cauchy and Wendland models. Each kernel is
``variance * psi(r)`` with ``r = ||(x - x') / lengthscale||``; for the
compactly supported models the lengthscale is the **range**, and ``psi`` is 0
for ``r >= 1``.

The profiles ``psi`` are written as functions of the scaled squared distance
``r2``, shared with the kernel classes in `kernellib._geo`. Where ``psi`` is
twice differentiable at the origin (cubic, hole effect, Wendland, stable and
generalized Cauchy at ``alpha = 2``) the profile switches to its expansion in
``r2`` near ``r = 0``, as `kernellib.Matern` does, so Hessians at coincident
points are exact.

The models marked "d <= 3" are positive definite in at most three dimensions;
their functions raise for wider inputs.
"""

from __future__ import annotations

from collections.abc import Callable

import jax.numpy as jnp
from geonnax.basis import wendland_c2, wendland_c4
from jaxtyping import Array, Float

from kernellib.functional._distances import _pairwise_sq_dist


__all__ = [
    "cubic_kernel",
    "generalized_cauchy_kernel",
    "hole_effect_kernel",
    "pentaspherical_kernel",
    "spherical_kernel",
    "stable_kernel",
    "wendland_kernel",
]


# Below this scaled squared distance the profiles that are smooth in r² use
# their expansion in r² (see ``kernellib._kernels._stationary._R2_SMALL``).
_R2_SMALL = 1e-10

# Largest input dimension in which the "d <= 3" models are positive definite.
_MAX_DIM = 3


def _r_safe(r2: Float[Array, ...]) -> Float[Array, ...]:
    """``r = sqrt(r2)``, exactly 0 with a zero (sub)gradient at ``r2 = 0``.

    A jittered ``sqrt`` would leave ``r ~ 1e-15`` at the origin, so a profile
    linear in ``r`` would miss ``psi(0) = 1`` by a few ulp.
    """
    positive = r2 > 0.0
    return jnp.where(positive, jnp.sqrt(jnp.where(positive, r2, 1.0)), 0.0)


def _smooth_in_r2(
    r2: Float[Array, ...],
    exact: Callable[[Float[Array, ...]], Float[Array, ...]],
    taylor: Callable[[Float[Array, ...]], Float[Array, ...]],
) -> Float[Array, ...]:
    """``exact(r)`` away from ``r = 0``, ``taylor(r²)`` near it (double where)."""
    small = r2 < _R2_SMALL
    r = jnp.sqrt(jnp.where(small, 1.0, r2))
    return jnp.where(small, taylor(r2), exact(r))


def _compact(
    r: Float[Array, ...], poly: Callable[[Float[Array, ...]], Float[Array, ...]]
) -> Float[Array, ...]:
    """``poly(r)`` for ``r < 1``, else exactly 0.

    The polynomials are finite everywhere, so the discarded branch leaks no
    NaN into the gradient.
    """
    return jnp.where(r < 1.0, poly(r), 0.0)


def _pow_r(r2: Float[Array, ...], alpha: float) -> Float[Array, ...]:
    """``r**alpha`` from ``r2``, exactly 0 (with zero gradient) at ``r = 0``.

    ``alpha = 2`` returns ``r2`` itself, so the profile stays smooth in ``r2``.
    """
    if alpha == 2.0:
        return r2
    positive = r2 > 0.0
    safe = jnp.where(positive, r2, 1.0)
    return jnp.where(positive, safe ** (0.5 * alpha), 0.0)


def _check_dim(d: int, name: str) -> None:
    if d > _MAX_DIM:
        raise ValueError(
            f"{name} is positive definite only for inputs with D <= {_MAX_DIM}; "
            f"got D = {d}."
        )


def _check_alpha(alpha: float, name: str) -> None:
    if not 0.0 < alpha <= 2.0:
        raise ValueError(f"{name} requires 0 < alpha <= 2, got {alpha!r}")


# -- profiles psi(r2), shared with the kernel classes ---------------------------


def _spherical_shape(r2: Float[Array, ...]) -> Float[Array, ...]:
    return _compact(_r_safe(r2), lambda r: 1.0 - 1.5 * r + 0.5 * r**3)


def _cubic_shape(r2: Float[Array, ...]) -> Float[Array, ...]:
    def exact(r: Float[Array, ...]) -> Float[Array, ...]:
        return _compact(
            r,
            lambda r: 1.0 - 7.0 * r**2 + 8.75 * r**3 - 3.5 * r**5 + 0.75 * r**7,
        )

    # The r³ term is kept through a guarded sqrt; r⁵ and beyond are < 1e-24.
    return _smooth_in_r2(r2, exact, lambda s: 1.0 - 7.0 * s + 8.75 * s * _r_safe(s))


def _pentaspherical_shape(r2: Float[Array, ...]) -> Float[Array, ...]:
    return _compact(_r_safe(r2), lambda r: 1.0 - 1.875 * r + 1.25 * r**3 - 0.375 * r**5)


def _hole_effect_shape(r2: Float[Array, ...]) -> Float[Array, ...]:
    return _smooth_in_r2(
        r2,
        lambda r: jnp.sinc(r / jnp.pi),
        lambda s: 1.0 - s / 6.0 + s * s / 120.0,
    )


def _stable_shape(r2: Float[Array, ...], alpha: float) -> Float[Array, ...]:
    return jnp.exp(-_pow_r(r2, alpha))


def _generalized_cauchy_shape(
    r2: Float[Array, ...], alpha: float, beta: Float[Array, ""]
) -> Float[Array, ...]:
    return (1.0 + _pow_r(r2, alpha)) ** (-beta / alpha)


def _wendland_shape(r2: Float[Array, ...], order: int) -> Float[Array, ...]:
    if order == 2:
        # (1 - r)⁴(4r + 1) = 1 - 10r² + 20r³ - 15r⁴ + 4r⁵.
        return _smooth_in_r2(
            r2, wendland_c2, lambda s: 1.0 - 10.0 * s + 20.0 * s * _r_safe(s)
        )
    if order == 4:
        # (1 - r)⁶(35r² + 18r + 3)/3 = 1 - 28r²/3 + 70r⁴ - 448r⁵/3 + ...
        return _smooth_in_r2(
            r2, wendland_c4, lambda s: 1.0 - 28.0 * s / 3.0 + 70.0 * s * s
        )
    raise ValueError(f"Wendland supports order in {{2, 4}}, got {order!r}")


# -- functional kernels --------------------------------------------------------


def spherical_kernel(
    X1: Float[Array, "N1 D"],
    X2: Float[Array, "N2 D"],
    variance: Float[Array, ""],
    lengthscale: Float[Array, ""] | Float[Array, " D"],
) -> Float[Array, "N1 N2"]:
    r"""Spherical covariance model, compactly supported on ``r < 1``.

    $$
    \psi(r) = 1 - \tfrac32 r + \tfrac12 r^3 \quad (r < 1), \qquad 0 \text{ else}
    $$

    with $r = \|(x - x')/\ell\|$: the lengthscale is the range. Positive
    definite for $D \le 3$ (Chilès & Delfiner 2012, §2.5).

    Args:
        X1: ``(N1, D)`` inputs, ``D <= 3``.
        X2: ``(N2, D)`` inputs.
        variance: Scalar sill ``sigma^2``.
        lengthscale: Scalar (isotropic) or ``(D,)`` (ARD) range.

    Returns:
        ``(N1, N2)`` Gram matrix.

    Raises:
        ValueError: If ``D > 3``.

    Examples:
        >>> import jax.numpy as jnp
        >>> from kernellib.functional import spherical_kernel
        >>> X = jnp.array([[0.0], [0.5], [2.0]])
        >>> K = spherical_kernel(X, X, jnp.array(1.0), jnp.array(1.0))
        >>> [round(float(v), 4) for v in K[0]]
        [1.0, 0.3125, 0.0]
    """
    _check_dim(X1.shape[-1], "spherical_kernel")
    return variance * _spherical_shape(_pairwise_sq_dist(X1, X2, lengthscale))


def cubic_kernel(
    X1: Float[Array, "N1 D"],
    X2: Float[Array, "N2 D"],
    variance: Float[Array, ""],
    lengthscale: Float[Array, ""] | Float[Array, " D"],
) -> Float[Array, "N1 N2"]:
    r"""Cubic covariance model, compactly supported on ``r < 1``.

    $$
    \psi(r) = 1 - 7r^2 + \tfrac{35}{4} r^3 - \tfrac72 r^5 + \tfrac34 r^7
    \quad (r < 1), \qquad 0 \text{ else}
    $$

    Twice differentiable at the origin. Positive definite for $D \le 3$
    (Chilès & Delfiner 2012, §2.5).

    Args:
        X1: ``(N1, D)`` inputs, ``D <= 3``.
        X2: ``(N2, D)`` inputs.
        variance: Scalar sill.
        lengthscale: Scalar (isotropic) or ``(D,)`` (ARD) range.

    Returns:
        ``(N1, N2)`` Gram matrix.

    Raises:
        ValueError: If ``D > 3``.

    Examples:
        >>> import jax.numpy as jnp
        >>> from kernellib.functional import cubic_kernel
        >>> X = jnp.array([[0.0], [0.5], [1.0]])
        >>> K = cubic_kernel(X, X, jnp.array(1.0), jnp.array(1.0))
        >>> [round(float(v), 4) for v in K[0]]
        [1.0, 0.2402, 0.0]
    """
    _check_dim(X1.shape[-1], "cubic_kernel")
    return variance * _cubic_shape(_pairwise_sq_dist(X1, X2, lengthscale))


def pentaspherical_kernel(
    X1: Float[Array, "N1 D"],
    X2: Float[Array, "N2 D"],
    variance: Float[Array, ""],
    lengthscale: Float[Array, ""] | Float[Array, " D"],
) -> Float[Array, "N1 N2"]:
    r"""Pentaspherical covariance model, compactly supported on ``r < 1``.

    $$
    \psi(r) = 1 - \tfrac{15}{8} r + \tfrac54 r^3 - \tfrac38 r^5
    \quad (r < 1), \qquad 0 \text{ else}
    $$

    Positive definite for $D \le 3$ (Chilès & Delfiner 2012, §2.5).

    Args:
        X1: ``(N1, D)`` inputs, ``D <= 3``.
        X2: ``(N2, D)`` inputs.
        variance: Scalar sill.
        lengthscale: Scalar (isotropic) or ``(D,)`` (ARD) range.

    Returns:
        ``(N1, N2)`` Gram matrix.

    Raises:
        ValueError: If ``D > 3``.

    Examples:
        >>> import jax.numpy as jnp
        >>> from kernellib.functional import pentaspherical_kernel
        >>> X = jnp.array([[0.0], [0.5]])
        >>> K = pentaspherical_kernel(X, X, jnp.array(1.0), jnp.array(1.0))
        >>> round(float(K[0, 1]), 4)
        0.207
    """
    _check_dim(X1.shape[-1], "pentaspherical_kernel")
    return variance * _pentaspherical_shape(_pairwise_sq_dist(X1, X2, lengthscale))


def hole_effect_kernel(
    X1: Float[Array, "N1 D"],
    X2: Float[Array, "N2 D"],
    variance: Float[Array, ""],
    lengthscale: Float[Array, ""] | Float[Array, " D"],
) -> Float[Array, "N1 N2"]:
    r"""Hole-effect (cardinal sine) covariance model.

    $$
    \psi(r) = \frac{\sin r}{r}, \qquad \psi(0) = 1
    $$

    Takes negative values (the "hole"), modelling periodic-like dependence.
    Positive definite for $D \le 3$ (Chilès & Delfiner 2012, §2.5).

    Args:
        X1: ``(N1, D)`` inputs, ``D <= 3``.
        X2: ``(N2, D)`` inputs.
        variance: Scalar sill.
        lengthscale: Scalar (isotropic) or ``(D,)`` (ARD) lengthscale.

    Returns:
        ``(N1, N2)`` Gram matrix.

    Raises:
        ValueError: If ``D > 3``.

    Examples:
        >>> import jax.numpy as jnp
        >>> from kernellib.functional import hole_effect_kernel
        >>> X = jnp.array([[0.0], [jnp.pi]])
        >>> K = hole_effect_kernel(X, X, jnp.array(1.0), jnp.array(1.0))
        >>> [round(float(v), 4) + 0.0 for v in K[0]]
        [1.0, 0.0]
    """
    _check_dim(X1.shape[-1], "hole_effect_kernel")
    return variance * _hole_effect_shape(_pairwise_sq_dist(X1, X2, lengthscale))


def stable_kernel(
    X1: Float[Array, "N1 D"],
    X2: Float[Array, "N2 D"],
    variance: Float[Array, ""],
    lengthscale: Float[Array, ""] | Float[Array, " D"],
    alpha: float,
) -> Float[Array, "N1 N2"]:
    r"""Stable (powered exponential) covariance model.

    $$
    \psi(r) = \exp(-r^\alpha), \qquad 0 < \alpha \le 2
    $$

    Positive definite in every dimension. ``alpha = 1`` is the exponential
    (Matérn ½) model and ``alpha = 2`` the RBF with lengthscale
    $\ell / \sqrt2$; ``alpha`` is a static Python float.

    Args:
        X1: ``(N1, D)`` inputs.
        X2: ``(N2, D)`` inputs.
        variance: Scalar sill.
        lengthscale: Scalar (isotropic) or ``(D,)`` (ARD) lengthscale.
        alpha: Shape exponent in ``(0, 2]``.

    Returns:
        ``(N1, N2)`` Gram matrix.

    Raises:
        ValueError: If ``alpha`` is outside ``(0, 2]``.

    Examples:
        >>> import jax.numpy as jnp
        >>> from kernellib.functional import stable_kernel
        >>> X = jnp.array([[0.0], [1.0]])
        >>> K = stable_kernel(X, X, jnp.array(1.0), jnp.array(1.0), 1.0)
        >>> bool(jnp.allclose(K[0, 1], jnp.exp(-1.0)))
        True
    """
    _check_alpha(alpha, "stable_kernel")
    return variance * _stable_shape(_pairwise_sq_dist(X1, X2, lengthscale), alpha)


def generalized_cauchy_kernel(
    X1: Float[Array, "N1 D"],
    X2: Float[Array, "N2 D"],
    variance: Float[Array, ""],
    lengthscale: Float[Array, ""] | Float[Array, " D"],
    alpha: float,
    beta: Float[Array, ""],
) -> Float[Array, "N1 N2"]:
    r"""Generalized Cauchy covariance model (Gneiting & Schlather 2004).

    $$
    \psi(r) = (1 + r^\alpha)^{-\beta/\alpha},
    \qquad 0 < \alpha \le 2,\ \beta > 0
    $$

    Positive definite in every dimension. ``alpha`` sets the fractal
    dimension (roughness at the origin) and ``beta`` the power-law tail
    (long-range dependence), independently. ``alpha = 2`` is the rational
    quadratic with $\alpha_{RQ} = \beta/2$ and $\ell_{RQ} = \ell/\sqrt\beta$.
    ``alpha`` is a static Python float.

    Args:
        X1: ``(N1, D)`` inputs.
        X2: ``(N2, D)`` inputs.
        variance: Scalar sill.
        lengthscale: Scalar (isotropic) or ``(D,)`` (ARD) lengthscale.
        alpha: Shape exponent in ``(0, 2]``.
        beta: Positive tail exponent.

    Returns:
        ``(N1, N2)`` Gram matrix.

    Raises:
        ValueError: If ``alpha`` is outside ``(0, 2]``.

    Examples:
        >>> import jax.numpy as jnp
        >>> from kernellib.functional import generalized_cauchy_kernel
        >>> X = jnp.array([[0.0], [1.0]])
        >>> K = generalized_cauchy_kernel(
        ...     X, X, jnp.array(1.0), jnp.array(1.0), 1.0, jnp.array(2.0)
        ... )
        >>> round(float(K[0, 1]), 4)
        0.25
    """
    _check_alpha(alpha, "generalized_cauchy_kernel")
    return variance * _generalized_cauchy_shape(
        _pairwise_sq_dist(X1, X2, lengthscale), alpha, beta
    )


def wendland_kernel(
    X1: Float[Array, "N1 D"],
    X2: Float[Array, "N2 D"],
    variance: Float[Array, ""],
    lengthscale: Float[Array, ""] | Float[Array, " D"],
    order: int = 2,
) -> Float[Array, "N1 N2"]:
    r"""Wendland covariance model, compactly supported on ``r < 1``.

    $$
    \psi_{C^2}(r) = (1 - r)_+^4 (4r + 1), \qquad
    \psi_{C^4}(r) = (1 - r)_+^6 (35r^2 + 18r + 3) / 3
    $$

    Wendland's $\phi_{3,1}$ and $\phi_{3,2}$ (Wendland 1995), normalised to
    $\psi(0) = 1$; positive definite for $D \le 3$. The profiles are
    `geonnax.basis.wendland_c2` / ``wendland_c4``.

    Args:
        X1: ``(N1, D)`` inputs, ``D <= 3``.
        X2: ``(N2, D)`` inputs.
        variance: Scalar sill.
        lengthscale: Scalar (isotropic) or ``(D,)`` (ARD) support radius.
        order: Smoothness, ``2`` (C², default) or ``4`` (C⁴); static.

    Returns:
        ``(N1, N2)`` Gram matrix.

    Raises:
        ValueError: If ``D > 3`` or ``order`` is not 2 or 4.

    Examples:
        >>> import jax.numpy as jnp
        >>> from kernellib.functional import wendland_kernel
        >>> X = jnp.array([[0.0], [0.5], [1.5]])
        >>> K = wendland_kernel(X, X, jnp.array(1.0), jnp.array(1.0))
        >>> [round(float(v), 4) for v in K[0]]
        [1.0, 0.1875, 0.0]
    """
    _check_dim(X1.shape[-1], "wendland_kernel")
    if order not in (2, 4):
        raise ValueError(f"wendland_kernel supports order in {{2, 4}}, got {order!r}")
    return variance * _wendland_shape(_pairwise_sq_dist(X1, X2, lengthscale), order)
