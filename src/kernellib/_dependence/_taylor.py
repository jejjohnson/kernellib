r"""Coordinates for a kernel Taylor diagram.

A Taylor diagram (Taylor, 2001) places each model at radius $\sigma_m$ and
angle $\arccos\rho$ from a reference, so the distance to the reference point
is the centred RMSE, by the law of cosines. Centred Gram matrices are
vectors in $\mathbb R^{N \times N}$ with inner product HSIC, so the same
triangle holds for them: the radii are HSIC norms, the cosine is CKA and the
third side is the distance between the centred Gram matrices. That gives a
Taylor diagram for multivariate, nonlinear similarity.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
from jaxtyping import Array, Float

from kernellib import functional as F
from kernellib._dependence._features import _fit_pair
from kernellib._dependence._hsic import Estimator, _check_estimator, _check_paired
from kernellib._kernels import AbstractKernel
from kernellib._operators._bridge import to_operator
from kernellib._spectral import AbstractFeatureMap
from kernellib.functional._statistics import _hsic_features


__all__ = ["TaylorStatistics", "taylor_statistics"]


class TaylorStatistics(NamedTuple):
    r"""The sides and angle of a kernel Taylor-diagram triangle.

    They satisfy ``distance**2 == norm_x**2 + norm_y**2 - 2 * norm_x *
    norm_y * correlation`` (clipped at zero), because all four come from the
    same three HSIC values.

    Attributes:
        norm_x: $\sqrt{\mathrm{HSIC}(x, x)}$, the reference's radius;
            $\|\tilde K_x\|_F / n$ with the biased estimator.
        norm_y: $\sqrt{\mathrm{HSIC}(y, y)}$, the model's radius.
        correlation: $\mathrm{CKA}(x, y)$; the model sits at angle
            ``arccos(correlation)``.
        distance: $\sqrt{\mathrm{HSIC}(x, x) + \mathrm{HSIC}(y, y) -
            2\,\mathrm{HSIC}(x, y)}$; $\|\tilde K_x - \tilde K_y\|_F / n$
            with the biased estimator.
    """

    norm_x: Float[Array, ""]
    norm_y: Float[Array, ""]
    correlation: Float[Array, ""]
    distance: Float[Array, ""]


def taylor_statistics(
    kernel_x: AbstractKernel,
    kernel_y: AbstractKernel,
    X: Float[Array, "N Dx"],
    Y: Float[Array, "N Dy"],
    *,
    estimator: Estimator = "biased",
    approx: AbstractFeatureMap | None = None,
) -> TaylorStatistics:
    r"""Radii, correlation and distance for a kernel Taylor diagram.

    ``X`` is the reference and ``Y`` a model output on the same ``N``
    samples. Plot the reference at ``(norm_x, 0)`` and the model at radius
    ``norm_y`` and angle ``arccos(correlation)``; their separation is
    ``distance``. Call once per model with the same reference.

    With `Linear` kernels this is the RV-coefficient diagram: for 1-D data
    the radius is the variance $\sigma^2$ and the correlation is $\rho^2$,
    the classic diagram on a squared scale. With nonlinear kernels (`RBF`,
    `Distance`) it also registers nonlinear agreement.

    Args:
        kernel_x: Kernel on the reference ``X``.
        kernel_y: Kernel on the model ``Y``; usually the same kernel, so the
            radii are comparable.
        X: Reference samples, shape ``(N, Dx)``.
        Y: Paired model samples, shape ``(N, Dy)``.
        estimator: ``"biased"`` or ``"unbiased"``; with ``"unbiased"`` the
            squared distance can dip below zero and is clipped.
        approx: Optional unfitted feature map, as in `hsic`.

    Returns:
        A `TaylorStatistics`.

    Raises:
        ValueError: On mismatched sample sizes or an unknown estimator.

    Examples:
        >>> import jax
        >>> import jax.numpy as jnp
        >>> import kernellib as kl
        >>> X = jax.random.normal(jax.random.key(0), (100, 2))
        >>> k = kl.RBF()
        >>> s = kl.taylor_statistics(k, k, X, X + 0.1)
        >>> cosine_law = (
        ...     s.norm_x**2 + s.norm_y**2 - 2 * s.norm_x * s.norm_y * s.correlation
        ... )
        >>> bool(jnp.isclose(s.distance**2, cosine_law))
        True
        >>> bool(s.correlation > 0.9)
        True
    """
    _check_paired(X, Y)
    _check_estimator(estimator)
    if approx is None:
        K_x, K_y = to_operator(kernel_x, X), to_operator(kernel_y, Y)
        xy = F.hsic(K_x, K_y, estimator=estimator)
        xx = F.hsic(K_x, K_x, estimator=estimator)
        yy = F.hsic(K_y, K_y, estimator=estimator)
    else:
        Phi_x, Phi_y = _fit_pair(approx, kernel_x, X, kernel_y, Y)
        xy = _hsic_features(Phi_x, Phi_y, estimator)
        xx = _hsic_features(Phi_x, Phi_x, estimator)
        yy = _hsic_features(Phi_y, Phi_y, estimator)
    norm_x, norm_y = jnp.sqrt(xx), jnp.sqrt(yy)
    return TaylorStatistics(
        norm_x=norm_x,
        norm_y=norm_y,
        correlation=xy / (norm_x * norm_y),
        distance=jnp.sqrt(jnp.maximum(xx + yy - 2.0 * xy, 0.0)),
    )
