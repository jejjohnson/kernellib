"""The estimator contract shared by `KRR`, `Falkon` and `EigenPro`.

Configuration goes in the constructor; `fit` returns a new, fitted module
whose fitted state (training inputs, weights, landmarks) is plain fields;
`predict` evaluates it. Fitted estimators are PyTrees, so ``jax.grad``
through ``fit`` and ``predict`` reaches the kernel's hyperparameters.
"""

from __future__ import annotations

import warnings
from abc import abstractmethod
from typing import Self

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray


__all__ = ["AbstractEstimator"]


class AbstractEstimator(eqx.Module):
    """Configure, `fit`, `predict`.

    Subclasses keep their fitted state in fields that are ``None`` until
    `fit`, including an ``alpha`` field for the weights, and implement `fit`
    and `_predict`. Targets may be ``(N,)`` or ``(N, C)``.
    """

    @abstractmethod
    def fit(
        self,
        X: Float[Array, "N D"],
        y: Float[Array, " N"] | Float[Array, "N C"],
        *,
        key: PRNGKeyArray | None = None,
    ) -> Self:
        """Return a fitted copy."""
        raise NotImplementedError

    @abstractmethod
    def _predict(self, X: Float[Array, "Nt D"]) -> Float[Array, " Nt"]:
        raise NotImplementedError

    @property
    def is_fitted(self) -> bool:
        """Whether `fit` has been called."""
        # Subclasses declare ``alpha`` after their required fields; declaring
        # it here would put a defaulted field ahead of them in the dataclass.
        return getattr(self, "alpha", None) is not None

    def predict(
        self, X: Float[Array, "Nt D"]
    ) -> Float[Array, " Nt"] | Float[Array, "Nt C"]:
        """Predictions at ``X``, shaped like the training targets.

        Raises:
            RuntimeError: If the estimator has not been fitted.
        """
        if not self.is_fitted:
            raise RuntimeError(
                f"{type(self).__name__} is not fitted; call .fit(X, y) first."
            )
        return self._predict(X)

    def loss(
        self,
        X: Float[Array, "Nt D"],
        y: Float[Array, " Nt"] | Float[Array, "Nt C"],
    ) -> Float[Array, ""]:
        """Mean squared error of the predictions at ``X``.

        Differentiable, so a validation loss can drive hyperparameters:
        ``jax.grad(lambda k: KRR(k).fit(X, y).loss(X_val, y_val))``.
        """
        return jnp.mean((self.predict(X) - y) ** 2)


def _check_targets(X: Float[Array, "N D"], y: Array) -> Array:
    y = jnp.asarray(y)
    if y.ndim not in (1, 2) or y.shape[0] != X.shape[0]:
        raise ValueError(
            f"y must have shape (N,) or (N, C) with N={X.shape[0]}, got {y.shape}."
        )
    return y


def _concrete(value: float | Array) -> float | None:
    """``value`` as a Python float, or ``None`` when it is traced (under ``jit``)."""
    try:
        return float(value)
    except jax.errors.ConcretizationTypeError:
        return None


def _warn_small_regularization(
    name: str, regularization: float | Float[Array, ""], diag: Float[Array, " N"]
) -> None:
    r"""Warn when the ridge is below the working precision of the kernel matrix.

    The system $K + \lambda n I$ has $\|K\|_2 \le n \max_i K_{ii}$, so once
    $\lambda n < \epsilon\, n \max_i K_{ii}$, i.e. $\lambda < \epsilon
    \max_i K_{ii}$ with $\epsilon$ the machine epsilon of the kernel matrix's
    dtype, the ridge is lost in the rounding of the largest eigenvalue: the
    condition number can exceed $1/\epsilon$ and any solve, direct or
    iterative, may return garbage. Skipped when either value is traced.
    """
    lam = _concrete(regularization)
    scale = _concrete(jnp.max(jnp.abs(diag)))
    if lam is None or scale is None:
        return
    dtype = jnp.result_type(diag, jnp.float32)
    floor = float(jnp.finfo(dtype).eps) * scale
    if lam < floor:
        warnings.warn(
            f"{name}: regularization={lam:g} is below the {jnp.dtype(dtype).name} "
            f"precision floor eps * max diag(K) = {floor:.3g}, so the system "
            "can be numerically singular and the fit unreliable. Use float64 "
            "or a larger regularization.",
            RuntimeWarning,
            stacklevel=4,  # past the helper, fit and equinox's method wrapper
        )
