"""scikit-learn adapter for the weighted Parzen density `kernellib.KernelDensity`."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.validation import (
    _check_sample_weight,
    check_is_fitted,
    validate_data,
)

from kernellib._density import KernelDensity as _KernelDensity
from kernellib._kernels import RBF, AbstractKernel
from kernellib.sklearn._base import _KernelParamsMixin, _key


__all__ = ["KernelDensity"]

_FLOAT = (np.float64, np.float32)


class KernelDensity(_KernelParamsMixin, BaseEstimator):
    """`kernellib.KernelDensity` with ``sklearn.neighbors.KernelDensity``'s API.

    ``score_samples`` is the log-density and ``score`` the **total**
    log-likelihood, as in scikit-learn, so ``GridSearchCV`` over
    ``kernel__lengthscale`` selects a bandwidth by held-out likelihood. (The
    core object's ``score`` is the gradient of the log-density instead; reach
    it through ``model_``.)

    Args:
        kernel: A non-negative kernellib kernel, or ``None`` for ``RBF()``
            (lengthscale 1, scikit-learn's default bandwidth).
        bandwidth: ``None`` (keep the kernel's lengthscale), a lengthscale, or
            ``"scott"``, ``"silverman"``, ``"loo"``.
        normalize: Divide by the kernel's integral (else a log-density up to a
            constant).
        batch_size: Chunk size of the streaming path, or ``None`` (dense).
        loo_max_iter: BFGS iterations for ``bandwidth="loo"``.

    Attributes:
        model_: The fitted `kernellib.KernelDensity`.
        kernel_: The kernel with the selected lengthscale.
        bandwidth_: The selected lengthscale (``None`` for a kernel without
            one).
        n_features_in_: Number of input features.

    Examples:
        >>> import numpy as np
        >>> import kernellib as kl
        >>> from kernellib.sklearn import KernelDensity
        >>> X = np.random.default_rng(0).normal(size=(100, 2))
        >>> kde = KernelDensity(kernel=kl.RBF(), bandwidth="scott").fit(X)
        >>> kde.score_samples(X[:3]).shape, kde.sample(5, random_state=0).shape
        ((3,), (5, 2))
    """

    def __init__(
        self,
        kernel: AbstractKernel | None = None,
        *,
        bandwidth: Any = None,
        normalize: bool = True,
        batch_size: int | None = None,
        loo_max_iter: int = 100,
    ) -> None:
        self.kernel = kernel
        self.bandwidth = bandwidth
        self.normalize = normalize
        self.batch_size = batch_size
        self.loo_max_iter = loo_max_iter

    def fit(self, X: Any, y: Any = None, sample_weight: Any = None) -> KernelDensity:
        X = validate_data(self, X, dtype=_FLOAT)
        weights = None
        if sample_weight is not None:
            weights = jnp.asarray(
                _check_sample_weight(
                    sample_weight, X, dtype=X.dtype, ensure_non_negative=True
                )
            )
        self.model_ = _KernelDensity(
            RBF() if self.kernel is None else self.kernel,
            bandwidth=self.bandwidth,
            normalize=self.normalize,
            batch_size=self.batch_size,
            loo_max_iter=self.loo_max_iter,
        ).fit(jnp.asarray(X), weights=weights)
        self.kernel_ = self.model_.kernel
        ell = getattr(self.kernel_, "lengthscale", None)
        self.bandwidth_ = None if ell is None else np.asarray(ell)
        return self

    def score_samples(self, X: Any) -> np.ndarray:
        """Log-density of each sample, ``(n_samples,)``."""
        check_is_fitted(self)
        X = validate_data(self, X, reset=False, dtype=_FLOAT)
        return np.asarray(self.model_.log_density(jnp.asarray(X)))

    def score(self, X: Any, y: Any = None) -> float:
        """Total log-likelihood of ``X`` under the fitted density."""
        return float(np.sum(self.score_samples(X)))

    def sample(self, n_samples: int = 1, random_state: Any = None) -> np.ndarray:
        """Draw ``n_samples`` points (`RBF` kernel only)."""
        check_is_fitted(self)
        return np.asarray(self.model_.sample(_key(random_state), n_samples))
