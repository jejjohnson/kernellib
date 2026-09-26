"""The feature-map contract: ``fit`` a kernel, then map inputs to features.

A feature map approximates a kernel by an explicit finite-dimensional map,
``k(x, x') ≈ φ(x)ᵀ φ(x')``. Configuration (feature count, PRNG key) goes in
the constructor; `fit` returns a new module holding the kernel and whatever
it drew; calling the fitted map gives the ``(N, R)`` feature matrix, and
`operator` wraps it as a gaussx `LowRankUpdate`, so solves and
log-determinants go through Woodbury in ``O(N R²)``.
"""

from __future__ import annotations

from abc import abstractmethod

import equinox as eqx
import gaussx as gx
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Float

from kernellib._kernels import AbstractKernel, AbstractStationaryKernel


__all__ = ["AbstractFeatureMap"]


class AbstractFeatureMap(eqx.Module):
    """A finite-dimensional approximation ``k(x, x') ≈ φ(x)ᵀ φ(x')``.

    Subclasses hold their configuration as static fields, keep the fitted
    state (including a ``kernel`` field) as fields that are ``None`` until
    `fit`,
    and implement `fit` and `features`. Fitted maps are PyTrees, so they
    ``jit`` and differentiate like any other equinox module; the kernel's
    hyperparameters are read at call time, so gradients flow to them.
    """

    @abstractmethod
    def fit(self, kernel: AbstractKernel, X: Float[Array, "N D"]) -> AbstractFeatureMap:
        """Return a fitted copy for ``kernel`` on inputs like ``X``.

        ``X`` fixes the input dimension; landmark methods also select from it.
        """
        raise NotImplementedError

    @abstractmethod
    def features(self, X: Float[Array, "N D"]) -> Float[Array, "N R"]:
        """Feature matrix ``Φ(X)`` of a fitted map."""
        raise NotImplementedError

    @property
    def is_fitted(self) -> bool:
        """Whether `fit` has been called."""
        # Subclasses declare ``kernel`` after their required fields; declaring
        # it here would put a defaulted field ahead of them in the dataclass.
        return getattr(self, "kernel", None) is not None

    def __call__(self, X: Float[Array, "N D"]) -> Float[Array, "N R"]:
        """Feature matrix ``Φ(X)``, shape ``(N, R)``.

        Raises:
            RuntimeError: If the map has not been fitted.
        """
        if not self.is_fitted:
            raise RuntimeError(
                f"{type(self).__name__} is not fitted; call .fit(kernel, X) first."
            )
        return self.features(X)

    def operator(self, X: Float[Array, "N D"]) -> gx.LowRankUpdate:
        """``Φ(X) Φ(X)ᵀ`` as a zero-base, symmetric PSD `gaussx.LowRankUpdate`.

        Add noise by passing it to a solver or by swapping in a diagonal base;
        the ``N x N`` matrix is never formed.
        """
        Phi = self(X)
        return gx.LowRankUpdate(
            base=lx.DiagonalLinearOperator(jnp.zeros(Phi.shape[0], dtype=Phi.dtype)),
            U=Phi,
            d=jnp.ones(Phi.shape[1], dtype=Phi.dtype),
            V=Phi,
            tags=frozenset({lx.symmetric_tag, lx.positive_semidefinite_tag}),
        )


def _require_spectral(kernel: AbstractKernel, name: str) -> AbstractStationaryKernel:
    if not isinstance(kernel, AbstractStationaryKernel):
        raise NotImplementedError(
            f"{name} needs a stationary kernel with a spectral sampler; got "
            f"{type(kernel).__name__}. NystromFeatures works for any kernel."
        )
    return kernel
