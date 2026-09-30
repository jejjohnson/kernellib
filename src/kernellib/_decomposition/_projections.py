"""Linear graph projections: locality preserving projections (He & Niyogi, 2003)."""

from __future__ import annotations

import dataclasses
from typing import Literal

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float

from kernellib._decomposition._eigenmaps import (
    _check_common,
    _GraphEmbedding,
)
from kernellib._graph._construct import adjacency_matrix
from kernellib._graph._laplacian import graph_laplacian
from kernellib._graph._neighbors import Backend


__all__ = ["LocalityPreservingProjections"]


class LocalityPreservingProjections(_GraphEmbedding):
    r"""Locality preserving projections (He & Niyogi, 2003).

    A linear Laplacian eigenmap: find $A$ (``D x n``) minimising
    $\sum_{ij} W_{ij}\|A^\top x_i - A^\top x_j\|^2$ subject to
    $A^\top \bar X^\top D \bar X A = I$, i.e. the smallest solutions of
    $\bar X^\top L \bar X a = \lambda \bar X^\top D \bar X a$, with $\bar X$ the
    inputs centred at their degree-weighted mean. Unlike the eigenmaps it
    embeds new points: `transform` is $(x - \mu) A$. The graph settings
    (``n_components``, ``n_neighbors``, ``weighting``, ``bandwidth``,
    ``neighbors_backend``, ``random_state``) are as in `LaplacianEigenmaps`.

    Attributes:
        regularization: Ridge on $\bar X^\top D \bar X$, relative to its mean
            eigenvalue.
        projection: ``(D, n_components)``, ``None`` before `fit`.
        mean: ``(D,)``, ``None`` before `fit`.
        eigenvalues: ``(n_components,)``, ``None`` before `fit`.

    Examples:
        >>> import jax
        >>> import jax.numpy as jnp
        >>> import kernellib as kl
        >>> X = jax.random.normal(jax.random.key(0), (100, 5))
        >>> lpp = kl.LocalityPreservingProjections(n_components=2).fit(X)
        >>> lpp.transform(X[:3]).shape
        (3, 2)
    """

    n_components: int = eqx.field(default=2, static=True)
    n_neighbors: int = eqx.field(default=10, static=True)
    weighting: Literal["heat", "connectivity"] = eqx.field(default="heat", static=True)
    bandwidth: float | None = None
    regularization: float = 1e-8
    neighbors_backend: Backend = eqx.field(default="exact", static=True)
    random_state: int | None = eqx.field(default=None, static=True)
    projection: Float[Array, "D n"] | None = None
    mean: Float[Array, " D"] | None = None
    eigenvalues: Float[Array, " n"] | None = None

    def __check_init__(self) -> None:
        _check_common(self)

    def fit(self, X: Float[Array, "N D"]) -> LocalityPreservingProjections:
        """Learn the projection from ``X``.

        Raises:
            ValueError: If ``n_components`` exceeds the input dimension.
        """
        d = X.shape[1]
        if self.n_components > d:
            raise ValueError(
                f"n_components={self.n_components} exceeds the input dimension {d}."
            )
        W = adjacency_matrix(
            self._graph(X), weighting=self.weighting, bandwidth=self.bandwidth
        )
        degree = jnp.sum(W, axis=1)
        mu = degree @ X / jnp.sum(degree)
        Xc = X - mu
        A = Xc.T @ graph_laplacian(W) @ Xc
        B = (Xc * degree[:, None]).T @ Xc
        B = B + self.regularization * jnp.trace(B) / d * jnp.eye(d, dtype=B.dtype)
        # B = C Cᵀ turns the generalised problem into C⁻¹ A C⁻ᵀ u = λ u.
        C = jnp.linalg.cholesky(B)
        Ci = jnp.linalg.inv(C)
        lam, U = jnp.linalg.eigh(Ci @ A @ Ci.T)
        P = Ci.T @ U[:, : self.n_components]
        return dataclasses.replace(
            self, projection=P, mean=mu, eigenvalues=lam[: self.n_components]
        )

    def transform(self, X: Float[Array, "M D"]) -> Float[Array, "M n"]:
        """Project new points.

        Raises:
            RuntimeError: If not fitted.
        """
        if self.projection is None or self.mean is None:
            raise RuntimeError("LocalityPreservingProjections is not fitted.")
        return (X - self.mean) @ self.projection
