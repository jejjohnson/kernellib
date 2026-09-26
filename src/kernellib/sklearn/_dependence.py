"""scikit-learn-style estimators for HSIC / CKA and MMD.

They are not predictors, so they do not pass scikit-learn's estimator checks,
but they share its conventions: parameters in the constructor (with nested
``kernel_x__lengthscale`` etc.), ``fit`` returning ``self`` with trailing
underscore results, ``clone``, ``get_params`` / ``set_params``. ``fit``
estimates any missing kernel bandwidths by the median heuristic and computes
the statistic, and a permutation p-value when ``n_permutations > 0``;
``score`` re-evaluates the statistic on new data with the fitted kernels.
"""

from __future__ import annotations

from typing import Any, Literal

import jax
import jax.numpy as jnp
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_array, check_is_fitted

from kernellib._dependence import cka, hsic, mmd_squared, permutation_test
from kernellib._kernels import AbstractKernel
from kernellib._spectral import AbstractFeatureMap
from kernellib.sklearn._base import _default_kernel, _KernelParamsMixin, _key


__all__ = ["HSIC", "MMD"]

_FLOAT = (np.float64, np.float32)


def _as_2d(A: Any, name: str) -> np.ndarray:
    A = np.asarray(A)
    if A.ndim == 1:
        A = A[:, None]
    return check_array(A, dtype=_FLOAT, input_name=name)


class HSIC(_KernelParamsMixin, BaseEstimator):
    r"""Hilbert-Schmidt independence criterion between paired samples.

    Reimplements pysim's ``HSIC`` estimator on `kernellib.hsic` /
    `kernellib.cka`. ``normalize=True`` gives centred kernel alignment.

    Args:
        kernel_x: Kernel on ``X``, or ``None`` for a median-heuristic `RBF`.
        kernel_y: Kernel on ``y``, or ``None`` for a median-heuristic `RBF`.
        estimator: ``"biased"`` or ``"unbiased"``.
        normalize: Return CKA instead of HSIC.
        approx: Optional unfitted kernellib feature map for an ``O(n)``
            randomised estimate, e.g. ``kl.NystromFeatures(300, key)``.
        n_permutations: If positive, also compute a permutation p-value.
        random_state: Seed for the permutations and the heuristic subsample.

    Attributes:
        statistic_: HSIC (or CKA) of the fitted data.
        p_value_: Permutation p-value, or ``None`` if ``n_permutations == 0``.
        kernel_x_: Kernel used on ``X``.
        kernel_y_: Kernel used on ``y``.
        n_features_in_: Number of features of ``X``.

    Examples:
        >>> import numpy as np
        >>> from kernellib.sklearn import HSIC
        >>> rng = np.random.default_rng(0)
        >>> X = rng.normal(size=(100, 1))
        >>> dep = HSIC(n_permutations=99, random_state=0).fit(X, X**2)
        >>> round(dep.p_value_, 4)
        0.01
        >>> ind = HSIC(n_permutations=99, random_state=0).fit(
        ...     X, rng.normal(size=100)
        ... )
        >>> bool(ind.p_value_ > 0.05)
        True
    """

    _kernel_params = ("kernel_x", "kernel_y")

    def __init__(
        self,
        kernel_x: AbstractKernel | None = None,
        kernel_y: AbstractKernel | None = None,
        *,
        estimator: Literal["biased", "unbiased"] = "biased",
        normalize: bool = False,
        approx: AbstractFeatureMap | None = None,
        n_permutations: int = 0,
        random_state: Any = None,
    ) -> None:
        self.kernel_x = kernel_x
        self.kernel_y = kernel_y
        self.estimator = estimator
        self.normalize = normalize
        self.approx = approx
        self.n_permutations = n_permutations
        self.random_state = random_state

    def fit(self, X: Any, y: Any) -> HSIC:
        """Compute the statistic (and p-value) for paired ``X`` and ``y``."""
        X, Y = _as_2d(X, "X"), _as_2d(y, "y")
        if X.shape[0] != Y.shape[0]:
            raise ValueError(
                f"X and y must be paired samples, got {X.shape[0]} and {Y.shape[0]} "
                "rows."
            )
        self.n_features_in_ = X.shape[1]
        key_x, key_y, key_test = jax.random.split(_key(self.random_state), 3)
        X_j, Y_j = jnp.asarray(X), jnp.asarray(Y)
        self.kernel_x_ = _default_kernel(self.kernel_x, X_j, key_x)
        self.kernel_y_ = _default_kernel(self.kernel_y, Y_j, key_y)
        self.statistic_ = self._statistic(X_j, Y_j)
        self.p_value_ = None
        if self.n_permutations > 0:
            result = permutation_test(
                self._statistic,
                X_j,
                Y_j,
                key=key_test,
                n_permutations=self.n_permutations,
            )
            self.p_value_ = float(result.p_value)
        self.statistic_ = float(self.statistic_)
        return self

    def _statistic(self, X: Any, Y: Any) -> Any:
        measure = cka if self.normalize else hsic
        return measure(
            self.kernel_x_,
            self.kernel_y_,
            X,
            Y,
            estimator=self.estimator,
            approx=self.approx,
        )

    def score(self, X: Any, y: Any) -> float:
        """The statistic on new paired data, with the fitted kernels."""
        check_is_fitted(self)
        X, Y = _as_2d(X, "X"), _as_2d(y, "y")
        return float(self._statistic(jnp.asarray(X), jnp.asarray(Y)))


class MMD(_KernelParamsMixin, BaseEstimator):
    r"""Squared maximum mean discrepancy between two samples.

    ``fit(X, Y)`` takes the two samples; they may have different sizes (the
    linear-time estimator needs equal ones).

    Args:
        kernel: The kernel, or ``None`` for an `RBF` at the median-heuristic
            lengthscale of the pooled sample.
        estimator: ``"biased"``, ``"unbiased"`` or ``"linear"``.
        approx: Optional unfitted kernellib feature map.
        n_permutations: If positive, also compute a permutation p-value.
        random_state: Seed for the permutations and the heuristic subsample.

    Attributes:
        statistic_: MMD² of the fitted samples.
        p_value_: Permutation p-value, or ``None``.
        kernel_: The kernel used.
        n_features_in_: Number of features.

    Examples:
        >>> import numpy as np
        >>> from kernellib.sklearn import MMD
        >>> rng = np.random.default_rng(0)
        >>> X, Y = rng.normal(size=(80, 2)), rng.normal(size=(60, 2)) + 1.0
        >>> round(MMD(n_permutations=99, random_state=0).fit(X, Y).p_value_, 4)
        0.01
    """

    def __init__(
        self,
        kernel: AbstractKernel | None = None,
        *,
        estimator: Literal["biased", "unbiased", "linear"] = "biased",
        approx: AbstractFeatureMap | None = None,
        n_permutations: int = 0,
        random_state: Any = None,
    ) -> None:
        self.kernel = kernel
        self.estimator = estimator
        self.approx = approx
        self.n_permutations = n_permutations
        self.random_state = random_state

    def fit(self, X: Any, y: Any) -> MMD:
        """Compute MMD² (and p-value) between samples ``X`` and ``y``."""
        X, Y = _as_2d(X, "X"), _as_2d(y, "y")
        if X.shape[1] != Y.shape[1]:
            raise ValueError(
                f"X and y must have the same features, got {X.shape[1]} and "
                f"{Y.shape[1]}."
            )
        self.n_features_in_ = X.shape[1]
        key_kernel, key_test = jax.random.split(_key(self.random_state))
        X_j, Y_j = jnp.asarray(X), jnp.asarray(Y)
        self.kernel_ = _default_kernel(
            self.kernel, jnp.concatenate([X_j, Y_j]), key_kernel
        )
        statistic = self._statistic(X_j, Y_j)
        self.p_value_ = None
        if self.n_permutations > 0:
            result = permutation_test(
                self._statistic,
                X_j,
                Y_j,
                key=key_test,
                n_permutations=self.n_permutations,
                kind="two_sample",
            )
            self.p_value_ = float(result.p_value)
        self.statistic_ = float(statistic)
        return self

    def _statistic(self, X: Any, Y: Any) -> Any:
        return mmd_squared(
            self.kernel_, X, Y, estimator=self.estimator, approx=self.approx
        )

    def score(self, X: Any, y: Any) -> float:
        """MMD² between new samples, with the fitted kernel."""
        check_is_fitted(self)
        X, Y = _as_2d(X, "X"), _as_2d(y, "y")
        return float(self._statistic(jnp.asarray(X), jnp.asarray(Y)))
