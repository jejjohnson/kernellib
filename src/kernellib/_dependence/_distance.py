r"""Distance covariance, distance correlation and energy distance.

Sejdinovic et al. (2013): under the distance-induced kernel `Distance`, the
distance-based statistics of Székely et al. are the kernel ones up to a
constant,

$$
\mathrm{dCov}^2 = 4\,\mathrm{HSIC}, \qquad
\mathrm{dCor}^2 = \mathrm{CKA}, \qquad
\mathcal E = 2\,\mathrm{MMD}^2,
$$

so each function here is one call to `hsic`, `cka` or `mmd_squared`, with
their estimators, ``approx`` paths and gradients.
"""

from __future__ import annotations

from typing import Literal

from jaxtyping import Array, Float

from kernellib._dependence._hsic import Estimator, cka, hsic
from kernellib._dependence._mmd import mmd_squared
from kernellib._kernels import Distance
from kernellib._spectral import AbstractFeatureMap


__all__ = [
    "distance_correlation_squared",
    "distance_covariance_squared",
    "energy_distance",
]


def distance_covariance_squared(
    X: Float[Array, "N Dx"],
    Y: Float[Array, "N Dy"],
    *,
    exponent: float = 1.0,
    estimator: Estimator = "biased",
    approx: AbstractFeatureMap | None = None,
) -> Float[Array, ""]:
    r"""Squared distance covariance (Székely et al., 2007), ``4 * hsic``.

    ``"biased"`` is the V-statistic $\frac{1}{n^2}\sum_{ij} A_{ij} B_{ij}$
    over double-centred distance matrices; ``"unbiased"`` is the
    U-centred estimator of Székely & Rizzo (2014), which can be slightly
    negative and needs ``N >= 4``. Zero in the population exactly when
    ``X`` and ``Y`` are independent (for ``0 < exponent < 2``).

    Args:
        X: Samples, shape ``(N, Dx)``.
        Y: Paired samples, shape ``(N, Dy)``.
        exponent: Distance exponent in ``(0, 2]``; see `Distance`.
        estimator: ``"biased"`` or ``"unbiased"``.
        approx: Optional unfitted feature map, as in `hsic` (Nyström; the
            kernel has no spectral density).

    Returns:
        The dCov² estimate.

    Examples:
        >>> import jax.numpy as jnp
        >>> import kernellib as kl
        >>> X = jnp.array([[0.0], [1.0], [2.0]])
        >>> round(float(kl.distance_covariance_squared(X, X)), 6)  # dVar² of X
        0.493827
    """
    kernel = Distance(exponent=exponent)
    return 4.0 * hsic(kernel, kernel, X, Y, estimator=estimator, approx=approx)


def distance_correlation_squared(
    X: Float[Array, "N Dx"],
    Y: Float[Array, "N Dy"],
    *,
    exponent: float = 1.0,
    estimator: Estimator = "biased",
    approx: AbstractFeatureMap | None = None,
) -> Float[Array, ""]:
    r"""Squared distance correlation (Székely et al., 2007), ``cka``.

    $\mathrm{dCov}^2(X, Y) / \sqrt{\mathrm{dCov}^2(X, X)\,\mathrm{dCov}^2(Y, Y)}$.
    With the biased estimator it lies in ``[0, 1]``: zero for independent
    samples in the limit, one when ``Y`` is a similarity transform of ``X``.
    Unlike Pearson's $\rho^2$ it detects nonlinear dependence. The square
    root is dCor; it is left to the caller because the unbiased estimate can
    be negative.

    Same arguments as `distance_covariance_squared`.

    Examples:
        >>> import jax
        >>> import kernellib as kl
        >>> X = jax.random.normal(jax.random.key(0), (200, 1))
        >>> Z = jax.random.normal(jax.random.key(1), (200, 1))
        >>> dcor2 = kl.distance_correlation_squared
        >>> bool(dcor2(X, X**2) > 5 * dcor2(X, Z))  # rho(X, X^2) is ~0
        True
        >>> round(float(dcor2(X, 3.0 * X + 1.0)), 6)
        1.0
    """
    kernel = Distance(exponent=exponent)
    return cka(kernel, kernel, X, Y, estimator=estimator, approx=approx)


def energy_distance(
    X: Float[Array, "Nx D"],
    Y: Float[Array, "Ny D"],
    *,
    exponent: float = 1.0,
    estimator: Literal["biased", "unbiased", "linear"] = "biased",
    approx: AbstractFeatureMap | None = None,
) -> Float[Array, ""]:
    r"""Energy distance (Székely & Rizzo, 2004), ``2 * mmd_squared``.

    $$
    \mathcal E(P, Q) = 2\,\mathbb E\|X - Y\|^a - \mathbb E\|X - X'\|^a
        - \mathbb E\|Y - Y'\|^a,
    $$

    zero exactly when ``P = Q`` (for ``0 < exponent < 2``). The estimators
    are those of `mmd_squared`: ``"biased"`` averages over all pairs,
    ``"unbiased"`` leaves out the zero self-distances, ``"linear"`` is the
    ``O(N)`` estimator.

    Args:
        X: First sample, shape ``(Nx, D)``.
        Y: Second sample, shape ``(Ny, D)``.
        exponent: Distance exponent in ``(0, 2]``; see `Distance`.
        estimator: ``"biased"``, ``"unbiased"`` or ``"linear"``.
        approx: Optional unfitted feature map, as in `mmd_squared`.

    Returns:
        The energy-distance estimate.

    Examples:
        >>> import jax.numpy as jnp
        >>> import kernellib as kl
        >>> X, Y = jnp.array([[0.0], [1.0]]), jnp.array([[3.0]])
        >>> float(kl.energy_distance(X, Y))  # 2 * 2.5 - 0.5 - 0
        4.5
    """
    kernel = Distance(exponent=exponent)
    return 2.0 * mmd_squared(kernel, X, Y, estimator=estimator, approx=approx)
