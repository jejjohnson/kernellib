"""scikit-learn transformers over the kernellib feature maps."""

from __future__ import annotations

import warnings
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from sklearn.base import (
    BaseEstimator,
    ClassNamePrefixFeaturesOutMixin,
    TransformerMixin,
)
from sklearn.utils.validation import check_is_fitted, validate_data

from kernellib import _spectral as spectral
from kernellib._kernels import AbstractKernel
from kernellib.sklearn._base import _default_kernel, _KernelParamsMixin, _key


__all__ = [
    "FastFoodFeatures",
    "LaplaceEigenfunctionFeatures",
    "NystromFeatures",
    "OrthogonalRandomFeatures",
    "RandomFourierFeatures",
]

_FLOAT = (np.float64, np.float32)


class _FeatureMap(
    ClassNamePrefixFeaturesOutMixin, _KernelParamsMixin, TransformerMixin, BaseEstimator
):
    """fit / transform around a kernellib feature map built by ``_build``."""

    kernel: AbstractKernel | None
    random_state: Any = None

    def _build(self, key: Any, X: np.ndarray) -> spectral.AbstractFeatureMap:
        raise NotImplementedError

    def fit(self, X: Any, y: Any = None) -> Any:
        """Draw the feature map for the kernel on inputs like ``X``.

        Returns:
            ``self``, with ``feature_map_`` (the fitted kernellib map) and
            ``kernel_``.
        """
        X = validate_data(self, X, dtype=_FLOAT)
        key_kernel, key_map = jax.random.split(_key(self.random_state))
        X_j = jnp.asarray(X)
        self.kernel_ = _default_kernel(self.kernel, X_j, key_kernel)
        self.feature_map_ = self._build(key_map, X).fit(self.kernel_, X_j)
        self._n_features_out = int(self.feature_map_(X_j[:1]).shape[1])
        return self

    def transform(self, X: Any) -> np.ndarray:
        """Feature matrix ``(n, n_features_out)``."""
        check_is_fitted(self)
        X = validate_data(self, X, reset=False, dtype=_FLOAT)
        return np.asarray(self.feature_map_(jnp.asarray(X)))


class RandomFourierFeatures(_FeatureMap):
    r"""Random Fourier features for any kernellib kernel with a spectral sampler.

    Like ``sklearn.kernel_approximation.RBFSampler``, but for `RBF`, `Matern`
    and `RationalQuadratic` kernels with ARD lengthscales, and with a
    ``[cos, sin]`` pair per frequency: the output has ``2 * n_components``
    columns.

    Args:
        n_components: Number of frequencies.
        kernel: A stationary kernellib kernel, or ``None`` for a
            median-heuristic `RBF`.
        random_state: Seed.

    Attributes:
        feature_map_: The fitted `kernellib.RandomFourierFeatures`.
        kernel_: The kernel used.
        n_features_in_: Number of input features.

    Examples:
        >>> import numpy as np
        >>> import kernellib as kl
        >>> from kernellib.sklearn import RandomFourierFeatures
        >>> from sklearn.linear_model import Ridge
        >>> from sklearn.pipeline import make_pipeline
        >>> X = np.random.default_rng(0).uniform(size=(200, 3))
        >>> y = np.sin(4 * X[:, 0])
        >>> pipe = make_pipeline(
        ...     RandomFourierFeatures(
        ...         256, kernel=kl.Matern(nu=2.5), random_state=0
        ...     ),
        ...     Ridge(alpha=1e-3),
        ... ).fit(X, y)
        >>> bool(pipe.score(X, y) > 0.99)
        True
    """

    def __init__(
        self,
        n_components: int = 100,
        *,
        kernel: AbstractKernel | None = None,
        random_state: Any = None,
    ) -> None:
        self.n_components = n_components
        self.kernel = kernel
        self.random_state = random_state

    def _build(self, key: Any, X: np.ndarray) -> spectral.AbstractFeatureMap:
        return spectral.RandomFourierFeatures(self.n_components, key)


class OrthogonalRandomFeatures(RandomFourierFeatures):
    """Orthogonal random features: `RandomFourierFeatures` with orthogonal
    frequency blocks and lower variance. Same parameters and output width.
    """

    def _build(self, key: Any, X: np.ndarray) -> spectral.AbstractFeatureMap:
        return spectral.OrthogonalRandomFeatures(self.n_components, key)


class FastFoodFeatures(RandomFourierFeatures):
    """FastFood random features: `RandomFourierFeatures` in ``O(n_components
    log d)`` time and ``O(n_components)`` memory. Same parameters and output
    width.
    """

    def _build(self, key: Any, X: np.ndarray) -> spectral.AbstractFeatureMap:
        return spectral.FastFoodFeatures(self.n_components, key)


class NystromFeatures(RandomFourierFeatures):
    """Nyström features for any kernellib kernel, like
    ``sklearn.kernel_approximation.Nystroem``. ``n_components`` landmarks are
    drawn uniformly from the training inputs (capped at ``n`` with a
    warning); the output has ``n_components`` columns.
    """

    def _build(self, key: Any, X: np.ndarray) -> spectral.AbstractFeatureMap:
        n_components = self.n_components
        if n_components > X.shape[0]:
            warnings.warn(
                f"n_components={n_components} > n_samples={X.shape[0]}; using "
                f"n_components={X.shape[0]}.",
                stacklevel=3,
            )
            n_components = X.shape[0]
        return spectral.NystromFeatures(n_components, key)


class LaplaceEigenfunctionFeatures(_FeatureMap):
    """Hilbert-space (HSGP) features on a box around the training inputs.

    Deterministic. The output has ``n_per_dim ** d`` columns, so keep ``d``
    small. With ``L=None`` the half-widths are ``boundary_factor * max|X_d|``
    (a dimension that is identically zero gets half-width 1).

    Args:
        n_per_dim: Basis functions per input dimension.
        L: Half-widths of the box, or ``None`` to set them from the data.
        boundary_factor: Multiplier when ``L`` is ``None``.
        kernel: A kernellib `RBF` or `Matern`, or ``None`` for a
            median-heuristic `RBF`.

    Attributes:
        feature_map_: The fitted `kernellib.LaplaceEigenfunctionFeatures`.
        kernel_: The kernel used.
        n_features_in_: Number of input features.
    """

    # No draw: seed only the median-heuristic subsample, deterministically.
    random_state = 0

    def __init__(
        self,
        n_per_dim: int = 16,
        *,
        L: float | None = None,
        boundary_factor: float = 1.5,
        kernel: AbstractKernel | None = None,
    ) -> None:
        self.n_per_dim = n_per_dim
        self.L = L
        self.boundary_factor = boundary_factor
        self.kernel = kernel

    def _build(self, key: Any, X: np.ndarray) -> spectral.AbstractFeatureMap:
        L = self.L
        if L is None:
            half = self.boundary_factor * np.max(np.abs(X), axis=0)
            L = tuple(float(h) if h > 0 else 1.0 for h in half)
        return spectral.LaplaceEigenfunctionFeatures(self.n_per_dim, L=L)
