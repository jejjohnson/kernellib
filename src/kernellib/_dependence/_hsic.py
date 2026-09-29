r"""HSIC, CKA and kernel alignment on kernels and data.

Each takes two kernels and paired samples and ends in a matrix-level
computation. The dense path forms both Gram matrices and calls
`kernellib.functional`. With ``approx``, a feature map (Nyström, random
Fourier, FastFood, ...) replaces each Gram matrix by $\Phi\Phi^\top$, and the
statistic is computed from the features in ``O(N R_x R_y)`` time and
``O(N (R_x + R_y))`` memory, without an ``N x N`` matrix.
"""

from __future__ import annotations

from typing import Literal

import jax.numpy as jnp
from jaxtyping import Array, Float

from kernellib import functional as F
from kernellib._dependence._features import _fit_pair
from kernellib._kernels import AbstractKernel, Distance
from kernellib._operators._bridge import to_operator
from kernellib._spectral import AbstractFeatureMap
from kernellib.functional._statistics import _frob_sq, _hsic_features


__all__ = ["cka", "hsic", "kernel_alignment"]

Estimator = Literal["biased", "unbiased"]


def hsic(
    kernel_x: AbstractKernel,
    kernel_y: AbstractKernel,
    X: Float[Array, "N Dx"],
    Y: Float[Array, "N Dy"],
    *,
    estimator: Estimator = "biased",
    approx: AbstractFeatureMap | None = None,
) -> Float[Array, ""]:
    r"""Hilbert-Schmidt Independence Criterion between paired samples.

    ``kernellib.functional.hsic`` on ``kernel_x(X, X)`` and ``kernel_y(Y, Y)``;
    see there for the biased and unbiased (Song et al., 2012) estimators.
    With ``approx``, the biased estimator is
    $\|\bar\Phi_x^\top \bar\Phi_y\|_F^2 / n^2$ with column-centred features
    $\bar\Phi$, and the unbiased one is evaluated from the features in the
    same way. Differentiable in the kernels' hyperparameters and the inputs.

    Args:
        kernel_x: Kernel on ``X``.
        kernel_y: Kernel on ``Y``.
        X: Samples, shape ``(N, Dx)``.
        Y: Paired samples, shape ``(N, Dy)``.
        estimator: ``"biased"`` or ``"unbiased"`` (needs ``N >= 4``).
        approx: Optional unfitted feature map used as a template; it is
            fitted once per kernel, with independent keys.

    Returns:
        The HSIC estimate.

    Raises:
        ValueError: On mismatched sample sizes or an unknown estimator.

    Examples:
        >>> import jax
        >>> import kernellib as kl
        >>> X = jax.random.normal(jax.random.key(0), (200, 1))
        >>> Z = jax.random.normal(jax.random.key(1), (200, 1))  # independent
        >>> k = kl.RBF(lengthscale=1.0)
        >>> bool(kl.hsic(k, k, X, X**2) > 10 * kl.hsic(k, k, X, Z))
        True
    """
    _check_paired(X, Y)
    _check_estimator(estimator)
    X, Y = _anchor(kernel_x, X), _anchor(kernel_y, Y)
    if approx is None:
        return F.hsic(
            to_operator(kernel_x, X), to_operator(kernel_y, Y), estimator=estimator
        )
    Phi_x, Phi_y = _fit_pair(approx, kernel_x, X, kernel_y, Y)
    return _hsic_features(Phi_x, Phi_y, estimator)


def cka(
    kernel_x: AbstractKernel,
    kernel_y: AbstractKernel,
    X: Float[Array, "N Dx"],
    Y: Float[Array, "N Dy"],
    *,
    estimator: Estimator = "biased",
    approx: AbstractFeatureMap | None = None,
) -> Float[Array, ""]:
    r"""Centred kernel alignment, $\mathrm{HSIC}(x, y) /
    \sqrt{\mathrm{HSIC}(x, x)\,\mathrm{HSIC}(y, y)}$.

    Same arguments as `hsic`. With the biased estimator the value lies in
    ``[0, 1]`` and is invariant to rescaling either kernel.

    Examples:
        >>> import jax
        >>> import kernellib as kl
        >>> X = jax.random.normal(jax.random.key(0), (30, 2))
        >>> k = kl.RBF()
        >>> round(float(kl.cka(k, k, X, X)), 6)
        1.0
    """
    xy, xx, yy = _cka_parts(kernel_x, kernel_y, X, Y, estimator, approx)
    return xy / jnp.sqrt(xx * yy)


def _cka_parts(
    kernel_x: AbstractKernel,
    kernel_y: AbstractKernel,
    X: Float[Array, "N Dx"],
    Y: Float[Array, "N Dy"],
    estimator: Estimator,
    approx: AbstractFeatureMap | None,
) -> tuple[Float[Array, ""], Float[Array, ""], Float[Array, ""]]:
    """``HSIC(x, y)``, ``HSIC(x, x)``, ``HSIC(y, y)`` from one pair of Grams
    (or one fit of the features)."""
    _check_paired(X, Y)
    _check_estimator(estimator)
    X, Y = _anchor(kernel_x, X), _anchor(kernel_y, Y)
    if approx is None:
        K_x, K_y = to_operator(kernel_x, X), to_operator(kernel_y, Y)
        return (
            F.hsic(K_x, K_y, estimator=estimator),
            F.hsic(K_x, K_x, estimator=estimator),
            F.hsic(K_y, K_y, estimator=estimator),
        )
    Phi_x, Phi_y = _fit_pair(approx, kernel_x, X, kernel_y, Y)
    return (
        _hsic_features(Phi_x, Phi_y, estimator),
        _hsic_features(Phi_x, Phi_x, estimator),
        _hsic_features(Phi_y, Phi_y, estimator),
    )


def _anchor(kernel: AbstractKernel, X: Float[Array, "N D"]) -> Float[Array, "N D"]:
    """Centre the inputs of an origin-anchored `Distance` kernel.

    Centred statistics under `Distance` are translation invariant, but its
    Gram carries ``|x|^a`` terms that only cancel on paper; with a large
    offset they would bury the pairwise distances.
    """
    if isinstance(kernel, Distance):
        return X - jnp.mean(X, axis=0)
    return X


def kernel_alignment(
    kernel_x: AbstractKernel,
    kernel_y: AbstractKernel,
    X: Float[Array, "N Dx"],
    Y: Float[Array, "N Dy"],
    *,
    approx: AbstractFeatureMap | None = None,
) -> Float[Array, ""]:
    r"""Uncentred kernel alignment, $\langle K, L\rangle_F / (\|K\|_F \|L\|_F)$.

    Cristianini et al. (2002). Unlike `cka` the Gram matrices are not
    centred, so a constant offset in either kernel changes the value.

    Examples:
        >>> import jax.numpy as jnp
        >>> import kernellib as kl
        >>> X = jnp.linspace(0.0, 1.0, 10)[:, None]
        >>> round(float(kl.kernel_alignment(kl.RBF(), kl.RBF(), X, X)), 6)
        1.0
    """
    _check_paired(X, Y)
    if approx is None:
        K, L = kernel_x(X, X), kernel_y(Y, Y)
        return jnp.sum(K * L) / jnp.sqrt(jnp.sum(K * K) * jnp.sum(L * L))
    Phi_x, Phi_y = _fit_pair(approx, kernel_x, X, kernel_y, Y)
    kl_ = _frob_sq(Phi_x.T @ Phi_y)
    return kl_ / jnp.sqrt(_frob_sq(Phi_x.T @ Phi_x) * _frob_sq(Phi_y.T @ Phi_y))


def _check_paired(X: Array, Y: Array) -> None:
    if X.shape[0] != Y.shape[0]:
        raise ValueError(
            f"X and Y must be paired samples of the same size, got {X.shape[0]} "
            f"and {Y.shape[0]}."
        )


def _check_estimator(estimator: str) -> None:
    if estimator not in ("biased", "unbiased"):
        raise ValueError(
            f"estimator must be 'biased' or 'unbiased', got {estimator!r}."
        )
