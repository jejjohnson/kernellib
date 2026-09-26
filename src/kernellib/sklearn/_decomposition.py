"""scikit-learn adapters for kernel PCA, LPP and the graph eigenmaps."""

from __future__ import annotations

from typing import Any, Literal

import jax.numpy as jnp
import numpy as np
from sklearn.base import (
    BaseEstimator,
    ClassNamePrefixFeaturesOutMixin,
    TransformerMixin,
)
from sklearn.utils.validation import check_is_fitted, validate_data

from kernellib import _decomposition as dec
from kernellib._kernels import AbstractKernel
from kernellib._spectral import AbstractFeatureMap
from kernellib.sklearn._base import _default_kernel, _KernelParamsMixin, _key


__all__ = [
    "KernelPCA",
    "LaplacianEigenmaps",
    "LocalityPreservingProjections",
    "SchrodingerEigenmaps",
]

_FLOAT = (np.float64, np.float32)


def _check_n_samples(n: int, minimum: int, name: str) -> None:
    if n < minimum:
        raise ValueError(
            f"{name} needs at least {minimum} samples; got n_samples = {n}."
        )


class KernelPCA(
    ClassNamePrefixFeaturesOutMixin, _KernelParamsMixin, TransformerMixin, BaseEstimator
):
    """Kernel PCA (`kernellib.KernelPCA`) as a scikit-learn transformer.

    Args:
        n_components: Number of components (capped at ``n_samples``).
        kernel: A kernellib kernel, or ``None`` for a median-heuristic `RBF`.
        approx: Optional unfitted kernellib feature map for ``O(n R^2)``
            kernel PCA.

    Attributes:
        model_: The fitted `kernellib.KernelPCA`.
        kernel_: The kernel used.
        eigenvalues_: Eigenvalues of the centred Gram matrix.
        n_features_in_: Number of input features.

    Examples:
        >>> import numpy as np
        >>> from kernellib.sklearn import KernelPCA
        >>> X = np.random.default_rng(0).normal(size=(50, 3))
        >>> KernelPCA(n_components=2).fit_transform(X).shape
        (50, 2)
    """

    def __init__(
        self,
        n_components: int = 2,
        *,
        kernel: AbstractKernel | None = None,
        approx: AbstractFeatureMap | None = None,
    ) -> None:
        self.n_components = n_components
        self.kernel = kernel
        self.approx = approx

    def fit(self, X: Any, y: Any = None) -> KernelPCA:
        X = validate_data(self, X, dtype=_FLOAT)
        X_j = jnp.asarray(X)
        self.kernel_ = _default_kernel(self.kernel, X_j, _key(0))
        n_components = min(self.n_components, X.shape[0])
        self.model_ = dec.KernelPCA(
            self.kernel_, n_components=n_components, approx=self.approx
        ).fit(X_j)
        self.eigenvalues_ = np.asarray(self.model_.eigenvalues)
        self._n_features_out = n_components
        return self

    def transform(self, X: Any) -> np.ndarray:
        check_is_fitted(self)
        X = validate_data(self, X, reset=False, dtype=_FLOAT)
        return np.asarray(self.model_.transform(jnp.asarray(X)))


class LocalityPreservingProjections(
    ClassNamePrefixFeaturesOutMixin, TransformerMixin, BaseEstimator
):
    """Locality preserving projections (`kernellib.LocalityPreservingProjections`).

    A linear graph embedding with an out-of-sample ``transform``.

    Args:
        n_components: Output dimension (at most ``n_features``).
        n_neighbors: Neighbours per point (capped at ``n_samples - 1``).
        weighting: ``"heat"`` or ``"connectivity"``.
        bandwidth: Heat-kernel width, or ``None`` for the median distance.
        neighbors_backend: ``"exact"``, ``"pynndescent"`` or ``"sklearn"``.
        random_state: Seed for the approximate neighbours.

    Attributes:
        model_: The fitted kernellib model.
        projection_: ``(n_features, n_components)``.
        n_features_in_: Number of input features.
    """

    def __init__(
        self,
        n_components: int = 2,
        *,
        n_neighbors: int = 10,
        weighting: Literal["heat", "connectivity"] = "heat",
        bandwidth: float | None = None,
        neighbors_backend: Literal["exact", "pynndescent", "sklearn"] = "exact",
        random_state: int | None = None,
    ) -> None:
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.weighting = weighting
        self.bandwidth = bandwidth
        self.neighbors_backend = neighbors_backend
        self.random_state = random_state

    def fit(self, X: Any, y: Any = None) -> LocalityPreservingProjections:
        X = validate_data(self, X, dtype=_FLOAT)
        _check_n_samples(X.shape[0], 2, type(self).__name__)
        n_components = min(self.n_components, X.shape[1])
        self.model_ = dec.LocalityPreservingProjections(
            n_components=n_components,
            n_neighbors=min(self.n_neighbors, X.shape[0] - 1),
            weighting=self.weighting,
            bandwidth=self.bandwidth,
            neighbors_backend=self.neighbors_backend,
            random_state=self.random_state,
        ).fit(jnp.asarray(X))
        self.projection_ = np.asarray(self.model_.projection)
        self._n_features_out = n_components
        return self

    def transform(self, X: Any) -> np.ndarray:
        check_is_fitted(self)
        X = validate_data(self, X, reset=False, dtype=_FLOAT)
        return np.asarray(self.model_.transform(jnp.asarray(X)))


class _Eigenmap(BaseEstimator):
    """Transductive embedding: ``fit`` / ``fit_transform``, no ``transform``."""

    def _config(self, n: int) -> dict[str, Any]:
        _check_n_samples(n, 3, type(self).__name__)
        return {
            "n_components": min(self.n_components, n - 2),  # ty: ignore[unresolved-attribute]
            "n_neighbors": min(self.n_neighbors, n - 1),  # ty: ignore[unresolved-attribute]
            "weighting": self.weighting,  # ty: ignore[unresolved-attribute]
            "bandwidth": self.bandwidth,  # ty: ignore[unresolved-attribute]
            "constraint": self.constraint,  # ty: ignore[unresolved-attribute]
            "neighbors_backend": self.neighbors_backend,  # ty: ignore[unresolved-attribute]
            "eigen_solver": self.eigen_solver,  # ty: ignore[unresolved-attribute]
            "random_state": self.random_state,  # ty: ignore[unresolved-attribute]
        }

    def fit_transform(self, X: Any, y: Any = None) -> np.ndarray:
        """Fit and return the embedding of ``X``."""
        return self.fit(X, y).embedding_  # ty: ignore[unresolved-attribute]


class LaplacianEigenmaps(_Eigenmap):
    """Laplacian eigenmaps (`kernellib.LaplacianEigenmaps`), like
    ``sklearn.manifold.SpectralEmbedding``: ``fit`` / ``fit_transform`` only.

    Args:
        n_components: Embedding dimension.
        n_neighbors: Neighbours per point (capped at ``n_samples - 1``).
        weighting: ``"heat"`` or ``"connectivity"``.
        bandwidth: Heat-kernel width, or ``None`` for the median distance.
        constraint: ``"degree"`` or ``"identity"``.
        neighbors_backend: ``"exact"``, ``"pynndescent"`` or ``"sklearn"``.
        eigen_solver: ``"dense"`` or ``"arpack"``.
        random_state: Seed for the approximate neighbours and ARPACK.

    Attributes:
        embedding_: ``(n_samples, n_components)``.
        eigenvalues_: The generalised eigenvalues.
        model_: The fitted kernellib model.
        n_features_in_: Number of input features.

    Examples:
        >>> import numpy as np
        >>> from kernellib.sklearn import LaplacianEigenmaps
        >>> X = np.random.default_rng(0).normal(size=(60, 3))
        >>> LaplacianEigenmaps(n_components=2).fit_transform(X).shape
        (60, 2)
    """

    def __init__(
        self,
        n_components: int = 2,
        *,
        n_neighbors: int = 10,
        weighting: Literal["heat", "connectivity"] = "heat",
        bandwidth: float | None = None,
        constraint: Literal["degree", "identity"] = "degree",
        neighbors_backend: Literal["exact", "pynndescent", "sklearn"] = "exact",
        eigen_solver: Literal["dense", "arpack"] = "dense",
        random_state: int | None = None,
    ) -> None:
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.weighting = weighting
        self.bandwidth = bandwidth
        self.constraint = constraint
        self.neighbors_backend = neighbors_backend
        self.eigen_solver = eigen_solver
        self.random_state = random_state

    def fit(self, X: Any, y: Any = None) -> LaplacianEigenmaps:
        X = validate_data(self, X, dtype=_FLOAT)
        self.model_ = dec.LaplacianEigenmaps(**self._config(X.shape[0])).fit(
            jnp.asarray(X)
        )
        self.embedding_ = np.asarray(self.model_.embedding)
        self.eigenvalues_ = np.asarray(self.model_.eigenvalues)
        return self


class SchrodingerEigenmaps(_Eigenmap):
    """Semi-supervised Schrödinger eigenmaps (`kernellib.SchrodingerEigenmaps`).

    ``fit(X, y)`` takes partial labels in scikit-learn's semi-supervised
    convention, ``-1`` for unlabelled points. ``potential="labels"`` pulls
    points sharing a label together (`label_potential`);
    ``potential="barrier"`` pins every labelled point near the origin
    (`barrier_potential`). With ``y=None`` (or no labelled points) it is
    `LaplacianEigenmaps`. The other parameters (``n_components``,
    ``n_neighbors``, ``weighting``, ``bandwidth``, ``constraint``,
    ``neighbors_backend``, ``eigen_solver``, ``random_state``) and the fitted
    attributes (``embedding_``, ``eigenvalues_``, ``model_``,
    ``n_features_in_``) are as in `LaplacianEigenmaps`.

    Args:
        alpha: Weight of the potential, relative to the graph's scale.
        potential: ``"labels"`` or ``"barrier"``.

    Examples:
        >>> import numpy as np
        >>> from kernellib.sklearn import SchrodingerEigenmaps
        >>> rng = np.random.default_rng(0)
        >>> X = rng.normal(size=(60, 3))
        >>> y = np.full(60, -1)
        >>> y[:5], y[5:10] = 0, 1
        >>> SchrodingerEigenmaps(alpha=10.0).fit_transform(X, y).shape
        (60, 2)
    """

    def __init__(
        self,
        n_components: int = 2,
        *,
        alpha: float = 1.0,
        potential: Literal["labels", "barrier"] = "labels",
        n_neighbors: int = 10,
        weighting: Literal["heat", "connectivity"] = "heat",
        bandwidth: float | None = None,
        constraint: Literal["degree", "identity"] = "degree",
        neighbors_backend: Literal["exact", "pynndescent", "sklearn"] = "exact",
        eigen_solver: Literal["dense", "arpack"] = "dense",
        random_state: int | None = None,
    ) -> None:
        self.n_components = n_components
        self.alpha = alpha
        self.potential = potential
        self.n_neighbors = n_neighbors
        self.weighting = weighting
        self.bandwidth = bandwidth
        self.constraint = constraint
        self.neighbors_backend = neighbors_backend
        self.eigen_solver = eigen_solver
        self.random_state = random_state

    def fit(self, X: Any, y: Any = None) -> SchrodingerEigenmaps:
        X = validate_data(self, X, dtype=_FLOAT)
        n = X.shape[0]
        config = self._config(n)
        labels = np.full(n, -1) if y is None else np.asarray(y).astype(int)
        if labels.shape != (n,):
            raise ValueError(f"y must have shape ({n},), got {labels.shape}.")
        labelled = labels != -1
        if self.potential == "labels":
            V = dec.label_potential(jnp.asarray(labels))
            drop_first = True
        elif self.potential == "barrier":
            V = dec.barrier_potential(n, jnp.flatnonzero(jnp.asarray(labelled)))
            drop_first = False
        else:
            raise ValueError(
                f"potential must be 'labels' or 'barrier', got {self.potential!r}."
            )
        # No usable labels (none, or no class with two labelled points for the
        # label potential): the potential vanishes and SE reduces to LE.
        strength = float(jnp.trace(V) if V.ndim == 2 else jnp.sum(V))
        if strength == 0.0:
            model = dec.LaplacianEigenmaps(**config).fit(jnp.asarray(X))
        else:
            model = dec.SchrodingerEigenmaps(
                alpha=self.alpha, drop_first=drop_first, **config
            ).fit(jnp.asarray(X), V)
        self.model_ = model
        self.embedding_ = np.asarray(model.embedding)
        self.eigenvalues_ = np.asarray(model.eigenvalues)
        return self
