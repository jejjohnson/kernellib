r"""Maximum mean discrepancy between two samples under one kernel."""

from __future__ import annotations

from typing import Literal

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from kernellib import functional as F
from kernellib._kernels import AbstractKernel
from kernellib._spectral import AbstractFeatureMap


__all__ = ["mmd_squared"]


def mmd_squared(
    kernel: AbstractKernel,
    X: Float[Array, "Nx D"],
    Y: Float[Array, "Ny D"],
    *,
    estimator: Literal["biased", "unbiased", "linear"] = "biased",
    approx: AbstractFeatureMap | None = None,
) -> Float[Array, ""]:
    r"""Squared maximum mean discrepancy between samples ``X`` and ``Y``.

    Estimators (Gretton et al., 2012):

    - ``"biased"``: $\overline{K_{xx}} + \overline{K_{yy}} - 2\,\overline{K_{xy}}$,
      the V-statistic; ``kernellib.functional.mmd_squared``.
    - ``"unbiased"``: the U-statistic, with the diagonals of $K_{xx}$ and
      $K_{yy}$ left out of their means. Can be slightly negative.
    - ``"linear"``: the linear-time estimator, averaging
      $h = k(x_1, x_2) + k(y_1, y_2) - k(x_1, y_2) - k(x_2, y_1)$ over
      disjoint pairs; ``O(N)`` time and memory, higher variance. Needs
      ``Nx == Ny``.

    With ``approx``, one feature map is fitted on the pooled sample and the
    Gram matrices are replaced by $\Phi\Phi^\top$: the biased estimate is
    $\|\bar\phi_x - \bar\phi_y\|^2$, the distance between mean embeddings,
    in ``O((Nx + Ny) R)``.

    Args:
        kernel: The kernel.
        X: First sample, shape ``(Nx, D)``.
        Y: Second sample, shape ``(Ny, D)``.
        estimator: ``"biased"``, ``"unbiased"`` or ``"linear"``.
        approx: Optional unfitted feature map (not with ``"linear"``).

    Returns:
        The MMD² estimate.

    Raises:
        ValueError: On an unknown estimator, ``"linear"`` with unequal sizes
            or with ``approx``, or ``"unbiased"`` with fewer than two points
            per sample.

    Examples:
        >>> import jax
        >>> import kernellib as kl
        >>> X = jax.random.normal(jax.random.key(0), (100, 2))
        >>> Y = jax.random.normal(jax.random.key(1), (100, 2)) + 1.0
        >>> k = kl.RBF()
        >>> bool(kl.mmd_squared(k, X, Y) > kl.mmd_squared(k, X, X[::-1]))
        True
    """
    if estimator not in ("biased", "unbiased", "linear"):
        raise ValueError(
            f"estimator must be 'biased', 'unbiased' or 'linear', got {estimator!r}."
        )
    m, n = X.shape[0], Y.shape[0]
    if estimator == "linear":
        if approx is not None:
            raise ValueError("The linear-time estimator does not take approx.")
        if m != n:
            raise ValueError(
                f"The linear-time estimator needs equal sample sizes, got {m} and {n}."
            )
        return _mmd_linear(kernel, X, Y)
    if estimator == "unbiased" and min(m, n) < 2:
        raise ValueError("The unbiased estimator needs two points per sample.")

    if approx is not None:
        fitted = approx.fit(kernel, jnp.concatenate([X, Y]))
        Phi_x, Phi_y = fitted(X), fitted(Y)
        sx, sy = jnp.sum(Phi_x, axis=0), jnp.sum(Phi_y, axis=0)
        if estimator == "biased":
            return jnp.sum((sx / m - sy / n) ** 2)
        within_x = (jnp.sum(sx**2) - jnp.sum(Phi_x**2)) / (m * (m - 1))
        within_y = (jnp.sum(sy**2) - jnp.sum(Phi_y**2)) / (n * (n - 1))
        return within_x + within_y - 2.0 * jnp.dot(sx, sy) / (m * n)

    K_xx, K_yy, K_xy = kernel(X, X), kernel(Y, Y), kernel(X, Y)
    if estimator == "biased":
        return F.mmd_squared(K_xx, K_yy, K_xy)
    within_x = (jnp.sum(K_xx) - jnp.trace(K_xx)) / (m * (m - 1))
    within_y = (jnp.sum(K_yy) - jnp.trace(K_yy)) / (n * (n - 1))
    return within_x + within_y - 2.0 * jnp.mean(K_xy)


def _mmd_linear(
    kernel: AbstractKernel, X: Float[Array, "N D"], Y: Float[Array, "N D"]
) -> Float[Array, ""]:
    pairs = X.shape[0] // 2
    x1, x2 = X[0 : 2 * pairs : 2], X[1 : 2 * pairs : 2]
    y1, y2 = Y[0 : 2 * pairs : 2], Y[1 : 2 * pairs : 2]

    def k(a: Float[Array, " D"], b: Float[Array, " D"]) -> Float[Array, ""]:
        # One entry of the Gram matrix: works for Gram-only kernels too.
        return kernel(a[None], b[None])[0, 0]

    h = jax.vmap(lambda a1, a2, b1, b2: k(a1, a2) + k(b1, b2) - k(a1, b2) - k(a2, b1))(
        x1, x2, y1, y2
    )
    return jnp.mean(h)
