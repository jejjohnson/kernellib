"""scikit-learn regressors over `KRR`, `Falkon` and `EigenPro`."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted, validate_data

from kernellib._kernels import AbstractKernel
from kernellib._regression import KRR, AbstractEstimator, EigenPro, Falkon
from kernellib.sklearn._base import _default_kernel, _KernelParamsMixin, _key


__all__ = ["EigenProRegressor", "FalkonRegressor", "KernelRidge"]

_FLOAT = (np.float64, np.float32)


class _Regressor(_KernelParamsMixin, RegressorMixin, BaseEstimator):
    """fit / predict around a kernellib estimator built by ``_build``."""

    kernel: AbstractKernel | None
    # The concrete kernellib estimator (KRR, Falkon, EigenPro), set by fit.
    model_: Any

    def _build(self, kernel: AbstractKernel, n: int) -> AbstractEstimator:
        raise NotImplementedError

    def _random_state(self) -> Any:
        return 0  # deterministic median-heuristic subsample

    def __sklearn_tags__(self) -> Any:
        tags = super().__sklearn_tags__()
        # (n, c) targets are fitted jointly; (n, 1) stays (n, 1).
        tags.target_tags.multi_output = True
        return tags

    def fit(self, X: Any, y: Any) -> Any:
        """Fit on ``X`` of shape ``(n, d)`` and ``y`` of shape ``(n,)`` or ``(n, c)``.

        Returns:
            ``self``, with ``model_`` (the fitted kernellib estimator),
            ``kernel_`` (the kernel used) and ``alpha_`` (the weights).
        """
        X, y = validate_data(
            self, X, y, multi_output=True, y_numeric=True, dtype=_FLOAT
        )
        # One working precision: the kernel is evaluated in X's dtype.
        y = y.astype(X.dtype)
        key_kernel, key_fit = jax.random.split(_key(self._random_state()))
        X_j = jnp.asarray(X)
        self.kernel_ = _default_kernel(self.kernel, X_j, key_kernel)
        model: Any = self._build(self.kernel_, X.shape[0]).fit(
            X_j, jnp.asarray(y), key=key_fit
        )
        self.model_ = model
        self.alpha_ = np.asarray(model.alpha)
        return self

    def predict(self, X: Any) -> np.ndarray:
        """Predictions at ``X``, shaped like the training targets."""
        check_is_fitted(self)
        X = validate_data(self, X, reset=False, dtype=_FLOAT)
        return np.asarray(self.model_.predict(jnp.asarray(X)))


class KernelRidge(_Regressor):
    r"""Kernel ridge regression (`kernellib.KRR`) as a scikit-learn regressor.

    Unlike ``sklearn.kernel_ridge.KernelRidge``, whose ``alpha`` is added to
    the kernel matrix as is, ``regularization`` is scaled by the number of
    training points: the system is $(K + \lambda n I)\alpha = y$.

    Args:
        kernel: A kernellib kernel. ``None`` uses an `RBF` at the
            median-heuristic lengthscale of the training inputs. Its fields
            are nested parameters, e.g. ``kernel__lengthscale``.
        regularization: Ridge $\lambda$.
        solver: A gaussx solver strategy; ``None`` for dense Cholesky.
        implicit: Matrix-free kernel operator (pair with ``gx.CGSolver()``).

    Attributes:
        model_: The fitted `kernellib.KRR`.
        kernel_: The kernel used.
        alpha_: Dual weights, ``(n,)`` or ``(n, c)``.
        n_features_in_: Number of input features.

    Examples:
        >>> import numpy as np
        >>> import kernellib as kl
        >>> from kernellib.sklearn import KernelRidge
        >>> from sklearn.model_selection import GridSearchCV
        >>> X = np.linspace(0, 1, 40)[:, None]
        >>> y = np.sin(6 * X[:, 0])
        >>> search = GridSearchCV(
        ...     KernelRidge(kernel=kl.RBF()),
        ...     {"kernel__lengthscale": [0.01, 0.2, 5.0]},
        ...     cv=4,
        ... ).fit(X, y)
        >>> search.best_params_
        {'kernel__lengthscale': 0.2}
    """

    def __init__(
        self,
        kernel: AbstractKernel | None = None,
        *,
        regularization: float = 1e-3,
        solver: Any = None,
        implicit: bool = False,
    ) -> None:
        self.kernel = kernel
        self.regularization = regularization
        self.solver = solver
        self.implicit = implicit

    def _build(self, kernel: AbstractKernel, n: int) -> AbstractEstimator:
        kwargs = {} if self.solver is None else {"solver": self.solver}
        return KRR(
            kernel,
            regularization=self.regularization,
            implicit=self.implicit,
            **kwargs,
        )


class FalkonRegressor(_Regressor):
    r"""Falkon Nyström kernel ridge regression (`kernellib.Falkon`).

    Args:
        kernel: A kernellib kernel, or ``None`` for a median-heuristic `RBF`.
        n_inducing: Number of centres (capped at ``n``).
        regularization: Ridge $\lambda$, scaled by ``n`` as in `KernelRidge`.
        max_iter: Conjugate-gradient budget.
        tol: Conjugate-gradient tolerance.
        implicit: Stream the ``n x m`` cross kernel.
        batch_size: Rows per streamed step.
        random_state: Seed for the centres and the heuristic subsample.

    Attributes:
        model_: The fitted `kernellib.Falkon`.
        kernel_: The kernel used.
        alpha_: Weights on the centres.
        landmarks_: The centres, ``(m, d)``.
        n_iter_: The conjugate-gradient budget, ``max_iter``.
            `kernellib.falkon_solve` does not report the iterations used.
        n_features_in_: Number of input features.

    Examples:
        >>> import numpy as np
        >>> from kernellib.sklearn import FalkonRegressor
        >>> X = np.random.default_rng(0).uniform(size=(500, 2))
        >>> y = np.sin(4 * X[:, 0]) + X[:, 1]
        >>> model = FalkonRegressor(n_inducing=100, random_state=0).fit(X, y)
        >>> model.landmarks_.shape, bool(model.score(X, y) > 0.99)
        ((100, 2), True)
    """

    def __init__(
        self,
        kernel: AbstractKernel | None = None,
        *,
        n_inducing: int = 1000,
        regularization: float = 1e-3,
        max_iter: int = 20,
        tol: float = 1e-6,
        implicit: bool = True,
        batch_size: int = 1024,
        random_state: Any = None,
    ) -> None:
        self.kernel = kernel
        self.n_inducing = n_inducing
        self.regularization = regularization
        self.max_iter = max_iter
        self.tol = tol
        self.implicit = implicit
        self.batch_size = batch_size
        self.random_state = random_state

    def _random_state(self) -> Any:
        return self.random_state

    def _build(self, kernel: AbstractKernel, n: int) -> AbstractEstimator:
        return Falkon(
            kernel,
            n_inducing=min(self.n_inducing, n),
            regularization=self.regularization,
            max_iter=self.max_iter,
            tol=self.tol,
            implicit=self.implicit,
            batch_size=self.batch_size,
        )

    def fit(self, X: Any, y: Any) -> FalkonRegressor:
        super().fit(X, y)
        self.landmarks_ = np.asarray(self.model_.landmarks)
        self.n_iter_ = self.max_iter
        return self


class EigenProRegressor(_Regressor):
    r"""EigenPro-preconditioned kernel SGD (`kernellib.EigenPro`).

    Fits the interpolating solution, early-stopped by ``epochs``. On small
    data the sizes are capped: ``subsample_size`` and ``batch_size`` at
    ``n``, and ``n_components`` below the subsample size.

    Args:
        kernel: A kernellib kernel, or ``None`` for a median-heuristic `RBF`.
        epochs: Passes over the data.
        batch_size: Mini-batch size.
        subsample_size: Subsample for the eigendecomposition.
        n_components: Damped eigendirections.
        decay: Spectral exponent in ``(0, 1]``.
        random_state: Seed for the subsample, the batches and the heuristic.

    Attributes:
        model_: The fitted `kernellib.EigenPro`.
        kernel_: The kernel used.
        alpha_: Weights on the training points.
        n_features_in_: Number of input features.

    Examples:
        >>> import numpy as np
        >>> from kernellib.sklearn import EigenProRegressor
        >>> X = np.random.default_rng(0).uniform(size=(400, 2))
        >>> y = np.sin(4 * X[:, 0]) + X[:, 1]
        >>> model = EigenProRegressor(
        ...     epochs=5, subsample_size=200, n_components=20, random_state=0
        ... ).fit(X, y)
        >>> bool(model.score(X, y) > 0.99)
        True
    """

    def __init__(
        self,
        kernel: AbstractKernel | None = None,
        *,
        epochs: int = 10,
        batch_size: int = 256,
        subsample_size: int = 2000,
        n_components: int = 100,
        decay: float = 0.95,
        random_state: Any = None,
    ) -> None:
        self.kernel = kernel
        self.epochs = epochs
        self.batch_size = batch_size
        self.subsample_size = subsample_size
        self.n_components = n_components
        self.decay = decay
        self.random_state = random_state

    def _random_state(self) -> Any:
        return self.random_state

    def _build(self, kernel: AbstractKernel, n: int) -> AbstractEstimator:
        if n < 3:
            raise ValueError(
                "EigenProRegressor needs at least 3 samples for its subsample "
                f"eigendecomposition; got n_samples = {n}."
            )
        m = min(self.subsample_size, n)
        return EigenPro(
            kernel,
            epochs=self.epochs,
            batch_size=min(self.batch_size, n),
            subsample_size=m,
            n_components=max(1, min(self.n_components, m - 1)),
            decay=self.decay,
        )
