r"""Linear graph projections: locality preserving projections (He & Niyogi,
2003) and Schrödinger eigenmap projections.

Both restrict a graph embedding to a linear map of the centred inputs,
$y = \bar X a$, which turns the ``N x N`` eigenproblem of the eigenmaps into a
``D x D`` one and embeds new points as $(x - \mu) A$:

- **LPP**: $\bar X^\top L \bar X a = \lambda \bar X^\top D \bar X a$;
- **SEP**: $\bar X^\top (L + \alpha V) \bar X a = \lambda \bar X^\top D \bar X a$.

Both solve through `gaussx.eigh_generalized`, with the constraint matrix
$\bar X^\top D \bar X$ (plus a relative ridge) tagged positive definite, so
the solve is a Cholesky whitening.
"""

from __future__ import annotations

import dataclasses
from typing import Literal

import einx
import equinox as eqx
import gaussx as gx
import jax
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Float

from kernellib._decomposition._eigenmaps import (
    GraphLike,
    Potential,
    _check_common,
    _check_graph,
    _check_potential,
    _check_signed,
    _GraphEmbedding,
    _laplacian_and_degree,
    _potential_operator,
    _trace,
)
from kernellib._einx import einsum
from kernellib._graph._construct import adjacency_matrix
from kernellib._graph._neighbors import Backend


__all__ = ["LocalityPreservingProjections", "SchrodingerEigenmapProjections"]


# -- shared solvers (also used by the kernel projections) ---------------------


def _quadratic(
    M: lx.AbstractLinearOperator | Float[Array, "N N"], F: Float[Array, "N p"]
) -> Float[Array, "p p"]:
    """$F^\\top M F$, for a dense matrix or a (sparse) operator $M$."""
    if isinstance(M, lx.MatrixLinearOperator):
        M = M.as_matrix()
    if isinstance(M, lx.AbstractLinearOperator):
        MF = jax.vmap(M.mv, in_axes=1, out_axes=1)(F)
    else:
        MF = einsum(M, F, "n m, m p -> n p")
    return einsum(F, MF, "n a, n b -> a b")


def _weighted_gram(
    F: Float[Array, "N p"], degree: Float[Array, " N"]
) -> Float[Array, "p p"]:
    """$F^\\top D F$."""
    return einsum(einx.multiply("n a, n -> n a", F, degree), F, "n a, n b -> a b")


def _degree_centre(
    F: Float[Array, "N p"], degree: Float[Array, " N"]
) -> tuple[Float[Array, "N p"], Float[Array, " p"]]:
    """Centre the rows at their degree-weighted mean: $d^\\top \\bar F = 0$."""
    mu = degree @ F / jnp.sum(degree)
    return einx.subtract("n p, p -> n p", F, mu), mu


def _potential_term(
    F: Float[Array, "N p"],
    potential: Potential,
    alpha: float | Float[Array, ""],
    normalize: bool,
    degree: Float[Array, " N"],
) -> tuple[Float[Array, ""], Float[Array, "p p"]]:
    """$\\alpha$, trace-normalised in ``N`` space, and $F^\\top V F$."""
    if normalize:
        alpha = alpha * jnp.sum(degree) / jnp.maximum(_trace(potential), 1e-30)
    return jnp.asarray(alpha), _quadratic(_potential_operator(potential), F)


def _smallest_constrained(
    A: Float[Array, "p p"], B: Float[Array, "p p"], k: int, regularization: float
) -> tuple[Float[Array, " k"], Float[Array, "p k"]]:
    """Smallest solutions of $A a = \\lambda (B + \\epsilon I) a$, with
    $\\epsilon$ = ``regularization`` times the mean eigenvalue of $B$."""
    p = B.shape[0]
    B = B + regularization * jnp.trace(B) / p * jnp.eye(p, dtype=B.dtype)
    return gx.eigh_generalized(
        lx.MatrixLinearOperator(A, lx.symmetric_tag),
        lx.MatrixLinearOperator(B, lx.positive_semidefinite_tag),
        rank=k,
    )


def _smallest_penalised(
    A: Float[Array, "p p"], B: Float[Array, "p p"], k: int, regularization: float
) -> tuple[Float[Array, " k"], Float[Array, "p k"]]:
    """Smallest solutions of $(A + \\epsilon I) a = \\lambda B a$, $B$ PSD.

    The ridge penalises $\\|a\\|^2$ (an RKHS norm for kernel features), so
    directions where $B$ is (nearly) singular get a large eigenvalue rather
    than a spurious zero one. Solved as the largest $\\mu = 1/\\lambda$ of
    $B a = \\mu (A + \\epsilon I) a$, whose right-hand side is positive
    definite. The solutions are rescaled to $a^\top B a = 1$, as LPP's.
    """
    p = B.shape[0]
    A = A + regularization * jnp.trace(B) / p * jnp.eye(p, dtype=A.dtype)
    mu, P = gx.eigh_generalized(
        lx.MatrixLinearOperator(B, lx.symmetric_tag),
        lx.MatrixLinearOperator(A, lx.positive_semidefinite_tag),
        rank=k,
        which="largest",
    )
    mu = jnp.clip(mu[::-1], min=jnp.finfo(mu.dtype).tiny)
    return 1.0 / mu, einx.divide("p k, k -> p k", P[:, ::-1], jnp.sqrt(mu))


# -- estimators --------------------------------------------------------------


class _LinearProjection(_GraphEmbedding):
    """Shared by LPP and SEP: graph, centring and the out-of-sample map."""

    def _prepare(
        self, X: Float[Array, "N D"], graph: GraphLike | None
    ) -> tuple[
        Float[Array, "N D"],
        Float[Array, " D"],
        lx.AbstractLinearOperator | Float[Array, "N N"],
        Float[Array, " N"],
    ]:
        """Centred inputs, their mean, the Laplacian and the degrees."""
        d = X.shape[1]
        n_components = self.n_components  # ty: ignore[unresolved-attribute]
        if n_components > d:
            raise ValueError(
                f"n_components={n_components} exceeds the input dimension {d}."
            )
        if graph is None:
            graph = adjacency_matrix(
                self._graph(X),
                weighting=self.weighting,  # ty: ignore[unresolved-attribute]
                bandwidth=self.bandwidth,  # ty: ignore[unresolved-attribute]
            )
        else:
            _check_graph(graph, X.shape[0])
            # The constraint X^T D X is the degree constraint: no signed graph.
            graph = _check_signed(graph, type(self).__name__)
        L, degree = _laplacian_and_degree(graph)
        Xc, mu = _degree_centre(X, degree)
        return Xc, mu, L, degree

    def transform(self, X: Float[Array, "M D"]) -> Float[Array, "M n"]:
        """Project new points.

        Raises:
            RuntimeError: If not fitted.
        """
        projection = self.projection  # ty: ignore[unresolved-attribute]
        mean = self.mean  # ty: ignore[unresolved-attribute]
        if projection is None or mean is None:
            raise RuntimeError(f"{type(self).__name__} is not fitted.")
        return einx.subtract("m d, d -> m d", X, mean) @ projection


class LocalityPreservingProjections(_LinearProjection):
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

    def fit(
        self, X: Float[Array, "N D"], *, graph: GraphLike | None = None
    ) -> LocalityPreservingProjections:
        """Learn the projection from ``X``.

        Args:
            X: Inputs, ``(N, D)``.
            graph: Optional precomputed graph (an `AbstractGraph` or adjacency
                matrix) instead of the k-NN graph of ``X``.

        Raises:
            ValueError: If ``n_components`` exceeds the input dimension, or
                ``graph`` does not match ``X`` or has negative edge weights
                (the constraint ``X^T D X`` needs ``w >= 0``).
        """
        Xc, mu, L, degree = self._prepare(X, graph)
        lam, P = _smallest_constrained(
            _quadratic(L, Xc),
            _weighted_gram(Xc, degree),
            self.n_components,
            self.regularization,
        )
        return dataclasses.replace(self, projection=P, mean=mu, eigenvalues=lam)


class SchrodingerEigenmapProjections(_LinearProjection):
    r"""Schrödinger eigenmap projections (SEP): linear Schrödinger eigenmaps.

    The smallest solutions of
    $\bar X^\top (L + \alpha V) \bar X a = \lambda \bar X^\top D \bar X a$:
    `SchrodingerEigenmaps` restricted to $y = \bar X a$, so new points embed
    as $(x - \mu) A$. The centring, ridge and graph settings are as in
    `LocalityPreservingProjections`, and ``alpha = 0`` is LPP exactly. The
    potential $V$ lives on the training points (`label_potential`,
    `barrier_potential`, ``spatial_spectral_graph(...).laplacian_operator()``
    or `combine_potentials`), and its trace normalisation is done in ``N``
    space, before projection (Cahill et al.).

    Attributes:
        alpha: Weight of the potential.
        normalize_potential: Scale ``alpha`` by ``tr(L) / tr(V)``.
        projection: ``(D, n_components)``, ``None`` before `fit`.
        mean: ``(D,)``, ``None`` before `fit`.
        eigenvalues: ``(n_components,)``, ``None`` before `fit`.

    Examples:
        >>> import jax
        >>> import jax.numpy as jnp
        >>> import kernellib as kl
        >>> X = jax.random.normal(jax.random.key(0), (100, 5))
        >>> labels = jnp.where(jnp.arange(100) < 10, 0, -1)  # 10 labelled
        >>> sep = kl.SchrodingerEigenmapProjections(n_components=2, alpha=5.0)
        >>> sep = sep.fit(X, kl.label_potential(labels))
        >>> sep.transform(X[:3]).shape  # out-of-sample
        (3, 2)
    """

    n_components: int = eqx.field(default=2, static=True)
    n_neighbors: int = eqx.field(default=10, static=True)
    alpha: float = 1.0
    normalize_potential: bool = eqx.field(default=True, static=True)
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

    def fit(
        self,
        X: Float[Array, "N D"],
        potential: Potential,
        *,
        graph: GraphLike | None = None,
    ) -> SchrodingerEigenmapProjections:
        """Learn the projection from ``X`` under ``potential``.

        Args:
            X: Inputs, ``(N, D)``.
            potential: Diagonal ``(N,)``, matrix ``(N, N)`` or lineax operator
                on the points of ``X``.
            graph: Optional precomputed graph instead of the k-NN graph of
                ``X``.

        Raises:
            ValueError: If ``n_components`` exceeds the input dimension,
                the potential or ``graph`` does not match ``X``, or ``graph``
                has negative edge weights (the constraint
                ``X^T D X`` needs ``w >= 0``).
        """
        if not isinstance(potential, lx.AbstractLinearOperator):
            potential = jnp.asarray(potential)
        _check_potential(potential, X.shape[0])
        Xc, mu, L, degree = self._prepare(X, graph)
        alpha, XVX = _potential_term(
            Xc, potential, self.alpha, self.normalize_potential, degree
        )
        # Separate terms, so alpha = 0 reproduces LPP to the last bit.
        lam, P = _smallest_constrained(
            _quadratic(L, Xc) + alpha * XVX,
            _weighted_gram(Xc, degree),
            self.n_components,
            self.regularization,
        )
        return dataclasses.replace(self, projection=P, mean=mu, eigenvalues=lam)
