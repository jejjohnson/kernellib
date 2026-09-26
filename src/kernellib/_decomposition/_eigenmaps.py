r"""Graph embeddings: Laplacian eigenmaps, Schrödinger eigenmaps, LPP.

All three solve a generalised symmetric eigenproblem on a neighbourhood graph
with adjacency $W$, degree matrix $D$ and Laplacian $L = D - W$:

- **Laplacian eigenmaps** (Belkin & Niyogi, 2003) minimise
  $\sum_{ij} W_{ij}\|y_i - y_j\|^2 = 2\,\mathrm{tr}(Y^\top L Y)$ subject to
  $Y^\top D Y = I$: the smallest non-trivial solutions of $L y = \lambda D y$.
- **Schrödinger eigenmaps** (Czaja & Ehler, 2013) add a potential $V \succeq 0$
  that steers the embedding, $(L + \alpha V) y = \lambda D y$. A diagonal
  *barrier* potential pulls chosen points to the origin; a non-diagonal
  potential (itself a graph Laplacian) pulls chosen pairs together, e.g.
  points sharing a label (semi-supervised) or spatial neighbours in an image
  (Cahill, Czaja & Messinger, 2014).
- **Locality preserving projections** (He & Niyogi, 2003) restrict $y = X a$ to
  be linear in the inputs, so new points can be embedded:
  $X^\top L X a = \lambda X^\top D X a$.

``constraint="identity"`` replaces $D$ by $I$ in the constraint. The dense
path (default) is JAX end to end and differentiable in the edge weights;
``eigen_solver="arpack"`` keeps the graph sparse and calls SciPy's ARPACK
(CPU, not traced), for graphs too large for an ``N x N`` matrix.
"""

from __future__ import annotations

import dataclasses
from typing import Literal

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from jaxtyping import Array, Float, Int

from kernellib._decomposition._graph import adjacency_matrix, graph_laplacian
from kernellib._decomposition._neighbors import (
    Backend,
    KNNGraph,
    nearest_neighbors,
)
from kernellib._einx import rearrange


__all__ = [
    "LaplacianEigenmaps",
    "LocalityPreservingProjections",
    "SchrodingerEigenmaps",
    "barrier_potential",
    "label_potential",
    "laplacian_eigenmap",
    "schrodinger_eigenmap",
    "spatial_spectral_potential",
]

Constraint = Literal["degree", "identity"]
Solver = Literal["dense", "arpack"]


# -- matrix level ------------------------------------------------------------


def _smallest_generalized(
    A: Float[Array, "N N"],
    degree: Float[Array, " N"] | None,
    n_components: int,
    drop_first: bool,
) -> tuple[Float[Array, " n"], Float[Array, "N n"]]:
    """Smallest solutions of ``A y = λ B y``, ``B = diag(degree)`` or ``I``."""
    if degree is None:
        lam, U = jnp.linalg.eigh(A)
        scale = jnp.ones(A.shape[0], A.dtype)
    else:
        scale = 1.0 / jnp.sqrt(jnp.where(degree > 0, degree, 1.0))
        lam, U = jnp.linalg.eigh(scale[:, None] * A * scale[None, :])
    start = 1 if drop_first else 0
    stop = start + n_components
    return lam[start:stop], scale[:, None] * U[:, start:stop]


def laplacian_eigenmap(
    W: Float[Array, "N N"],
    n_components: int = 2,
    *,
    constraint: Constraint = "degree",
    drop_first: bool = True,
) -> tuple[Float[Array, " n"], Float[Array, "N n"]]:
    r"""Laplacian eigenmap of a weighted graph: smallest solutions of
    $L y = \lambda D y$.

    Args:
        W: Symmetric adjacency matrix, ``(N, N)``.
        n_components: Embedding dimension.
        constraint: ``"degree"`` ($Y^\top D Y = I$) or ``"identity"``.
        drop_first: Drop the trivial constant solution ($\lambda = 0$).

    Returns:
        ``(eigenvalues, embedding)``, shapes ``(n,)`` and ``(N, n)``.

    Examples:
        >>> import jax.numpy as jnp
        >>> import kernellib as kl
        >>> # A path graph 0 - 1 - 2 - 3: the first coordinate is monotone.
        >>> W = jnp.diag(jnp.ones(3), 1) + jnp.diag(jnp.ones(3), -1)
        >>> _, Y = kl.laplacian_eigenmap(W, 1)
        >>> y = Y[:, 0] * jnp.sign(Y[-1, 0])
        >>> bool(jnp.all(jnp.diff(y) > 0))
        True
    """
    L = graph_laplacian(W)
    degree = jnp.sum(W, axis=1) if constraint == "degree" else None
    _check_constraint(constraint)
    return _smallest_generalized(L, degree, n_components, drop_first)


def schrodinger_eigenmap(
    W: Float[Array, "N N"],
    potential: Float[Array, " N"] | Float[Array, "N N"],
    n_components: int = 2,
    *,
    alpha: float | Float[Array, ""] = 1.0,
    normalize_potential: bool = True,
    constraint: Constraint = "degree",
    drop_first: bool = True,
) -> tuple[Float[Array, " n"], Float[Array, "N n"]]:
    r"""Schrödinger eigenmap: smallest solutions of $(L + \alpha V) y = \lambda D y$.

    Args:
        W: Symmetric adjacency matrix, ``(N, N)``.
        potential: $V$, either its diagonal ``(N,)`` (a barrier potential, see
            `barrier_potential`) or a symmetric PSD matrix ``(N, N)`` (e.g.
            `label_potential`, `spatial_spectral_potential`).
        n_components: Embedding dimension.
        alpha: Weight of the potential.
        normalize_potential: Scale $\alpha$ by $\mathrm{tr}(L) / \mathrm{tr}(V)$
            (Cahill et al.), so ``alpha`` is relative to the graph's own scale
            and comparable across data sets.
        constraint: ``"degree"`` or ``"identity"``.
        drop_first: Drop the first solution. Keep ``True`` for non-diagonal
            potentials that are Laplacians (the constant vector stays a
            $\lambda = 0$ solution); a barrier potential removes it, so pass
            ``False`` to keep all solutions.

    Returns:
        ``(eigenvalues, embedding)``.

    Examples:
        >>> import jax.numpy as jnp
        >>> import kernellib as kl
        >>> W = jnp.diag(jnp.ones(4), 1) + jnp.diag(jnp.ones(4), -1)
        >>> V = kl.barrier_potential(5, jnp.array([0]))  # pin node 0 to 0
        >>> _, Y = kl.schrodinger_eigenmap(W, V, 1, alpha=100.0, drop_first=False)
        >>> bool(jnp.abs(Y[0, 0]) < 0.05 * jnp.max(jnp.abs(Y)))
        True
    """
    _check_constraint(constraint)
    L = graph_laplacian(W)
    V = jnp.diag(potential) if potential.ndim == 1 else potential
    if normalize_potential:
        alpha = alpha * jnp.trace(L) / jnp.maximum(jnp.trace(V), 1e-30)
    degree = jnp.sum(W, axis=1) if constraint == "degree" else None
    return _smallest_generalized(L + alpha * V, degree, n_components, drop_first)


def barrier_potential(
    n: int, indices: Int[Array, " m"], strength: float = 1.0
) -> Float[Array, " N"]:
    """Diagonal potential that is ``strength`` at ``indices`` and 0 elsewhere.

    In `schrodinger_eigenmap` it pulls the embedding of those points towards
    the origin (Czaja & Ehler, 2013).

    Examples:
        >>> import jax.numpy as jnp
        >>> import kernellib as kl
        >>> kl.barrier_potential(4, jnp.array([1, 3])).tolist()
        [0.0, 1.0, 0.0, 1.0]
    """
    return jnp.zeros(n).at[indices].set(strength)


def label_potential(
    labels: Int[Array, " N"], *, unlabeled: int = -1
) -> Float[Array, "N N"]:
    r"""Non-diagonal potential pulling together points that share a label.

    The Laplacian of the graph joining every pair of labelled points with the
    same label, $V = \sum_{i \sim j} (e_i - e_j)(e_i - e_j)^\top$: in
    `schrodinger_eigenmap` it adds $\alpha \sum_{i \sim j} \|y_i - y_j\|^2$ to
    the objective (semi-supervised Schrödinger eigenmaps). Points labelled
    ``unlabeled`` are free.

    Examples:
        >>> import jax.numpy as jnp
        >>> import kernellib as kl
        >>> kl.label_potential(jnp.array([0, 0, -1])).tolist()
        [[1.0, -1.0, 0.0], [-1.0, 1.0, 0.0], [0.0, 0.0, 0.0]]
    """
    labels = jnp.asarray(labels)
    same = (labels[:, None] == labels[None, :]) & (labels[:, None] != unlabeled)
    A = same.astype(jnp.result_type(float)) * (1.0 - jnp.eye(labels.shape[0]))
    return graph_laplacian(A)


def spatial_spectral_potential(
    X: Float[Array, "N D"],
    coordinates: Float[Array, "N S"],
    n_neighbors: int = 4,
    *,
    bandwidth: float | None = None,
) -> Float[Array, "N N"]:
    r"""Spatial-spectral potential for images (Cahill, Czaja & Messinger, 2014).

    Joins each point to its ``n_neighbors`` nearest points in *space*
    (``coordinates``, e.g. pixel positions) and weights the edge by the heat
    kernel of their *spectral* distance $\|x_i - x_j\|$, returning that graph's
    Laplacian. Spatial neighbours that also look alike are pulled together.

    Args:
        X: Spectral features, ``(N, D)``.
        coordinates: Spatial positions, ``(N, S)``.
        n_neighbors: Spatial neighbours per point.
        bandwidth: Heat-kernel width on spectral distances; ``None`` for the
            median spectral distance over the spatial edges.
    """
    graph = nearest_neighbors(coordinates, n_neighbors)
    rows = jnp.repeat(jnp.arange(X.shape[0]), n_neighbors)
    cols = rearrange(graph.indices, "n k -> (n k)")
    spectral = rearrange(
        jnp.linalg.norm(X[rows] - X[cols], axis=1), "(n k) -> n k", k=n_neighbors
    )
    W = adjacency_matrix(
        KNNGraph(indices=graph.indices, distances=spectral),
        weighting="heat",
        bandwidth=bandwidth,
    )
    return graph_laplacian(W)


def _check_constraint(constraint: str) -> None:
    if constraint not in ("degree", "identity"):
        raise ValueError(
            f"constraint must be 'degree' or 'identity', got {constraint!r}."
        )


# -- sparse (ARPACK) path ----------------------------------------------------


def _sparse_adjacency(
    graph: KNNGraph, weighting: str, bandwidth: float | None
) -> sp.csr_matrix:
    idx = np.asarray(graph.indices)
    dist = np.asarray(graph.distances)
    n, k = idx.shape
    if weighting == "heat":
        sigma = float(np.median(dist)) if bandwidth is None else float(bandwidth)
        sigma = sigma if sigma > 0 else 1.0
        w = np.exp(-(dist**2) / (2.0 * sigma**2))
    else:
        w = np.ones_like(dist)
    W = sp.csr_matrix((w.ravel(), (np.repeat(np.arange(n), k), idx.ravel())), (n, n))
    W = W.maximum(W.T)
    W.setdiag(0.0)
    W.eliminate_zeros()
    return W


def _smallest_sparse(
    A: sp.spmatrix,
    degree: np.ndarray | None,
    n_components: int,
    drop_first: bool,
    seed: int,
) -> tuple[Float[Array, " n"], Float[Array, "N n"]]:
    n = A.shape[0]
    scale = (
        np.ones(n)
        if degree is None
        else 1.0 / np.sqrt(np.where(degree > 0, degree, 1.0))
    )
    S = sp.diags(scale) @ A @ sp.diags(scale)
    # Smallest eigenvalues of S are the largest of c I - S, with c a
    # Gershgorin bound on S's spectrum; ARPACK converges fast on those.
    c = float(np.max(np.abs(S).sum(axis=1)))
    k = n_components + int(drop_first)
    v0 = np.random.default_rng(seed).uniform(size=n)
    mu, U = spla.eigsh(c * sp.identity(n) - S, k=k, which="LA", v0=v0)
    order = np.argsort(c - mu)
    lam, U = (c - mu)[order], U[:, order]
    start = int(drop_first)
    Y = scale[:, None] * U[:, start:]
    return jnp.asarray(lam[start:]), jnp.asarray(Y)


# -- estimators --------------------------------------------------------------


class _GraphEmbedding(eqx.Module):
    """Shared configuration: how the neighbourhood graph is built and solved."""

    def _graph(self, X: Float[Array, "N D"]) -> KNNGraph:
        return nearest_neighbors(
            X,
            self.n_neighbors,  # ty: ignore[unresolved-attribute]
            backend=self.neighbors_backend,  # ty: ignore[unresolved-attribute]
            random_state=self.random_state,  # ty: ignore[unresolved-attribute]
        )


class LaplacianEigenmaps(_GraphEmbedding):
    r"""Laplacian eigenmaps (Belkin & Niyogi, 2003) of the k-NN graph of ``X``.

    ``fit(X)`` builds the graph (`nearest_neighbors`, `adjacency_matrix`) and
    solves $L y = \lambda D y$ (`laplacian_eigenmap`). The embedding is
    transductive: there is no ``transform`` for new points (use
    `LocalityPreservingProjections` for that).

    Attributes:
        n_components: Embedding dimension.
        n_neighbors: Neighbours per point.
        weighting: ``"heat"`` or ``"connectivity"`` edge weights.
        bandwidth: Heat-kernel width; ``None`` for the median neighbour
            distance.
        constraint: ``"degree"`` or ``"identity"``.
        neighbors_backend: ``"exact"``, ``"pynndescent"`` or ``"sklearn"``.
        eigen_solver: ``"dense"`` (JAX, differentiable) or ``"arpack"``
            (sparse, SciPy, for large ``N``).
        random_state: Seed for the approximate neighbours and ARPACK.
        embedding: ``(N, n_components)``, ``None`` before `fit`.
        eigenvalues: ``(n_components,)``, ``None`` before `fit`.
        graph: The `KNNGraph`, ``None`` before `fit`.

    Examples:
        >>> import jax.numpy as jnp
        >>> import kernellib as kl
        >>> t = jnp.linspace(0.0, 3.0, 60)
        >>> X = jnp.stack([jnp.cos(t), jnp.sin(t)], axis=1)  # an arc
        >>> le = kl.LaplacianEigenmaps(n_components=1, n_neighbors=4).fit(X)
        >>> y = le.embedding[:, 0] * jnp.sign(le.embedding[-1, 0])
        >>> bool(jnp.all(jnp.diff(y) > 0))  # unrolls the arc in order
        True
    """

    n_components: int = eqx.field(default=2, static=True)
    n_neighbors: int = eqx.field(default=10, static=True)
    weighting: Literal["heat", "connectivity"] = eqx.field(default="heat", static=True)
    bandwidth: float | None = None
    constraint: Constraint = eqx.field(default="degree", static=True)
    neighbors_backend: Backend = eqx.field(default="exact", static=True)
    eigen_solver: Solver = eqx.field(default="dense", static=True)
    random_state: int | None = eqx.field(default=None, static=True)
    embedding: Float[Array, "N n"] | None = None
    eigenvalues: Float[Array, " n"] | None = None
    graph: KNNGraph | None = None

    def __check_init__(self) -> None:
        _check_common(self)

    def fit(self, X: Float[Array, "N D"]) -> LaplacianEigenmaps:
        """Embed ``X``."""
        graph = self._graph(X)
        if self.eigen_solver == "arpack":
            W = _sparse_adjacency(graph, self.weighting, self.bandwidth)
            L = sp.diags(np.asarray(W.sum(axis=1)).ravel()) - W
            degree = np.asarray(W.sum(axis=1)).ravel()
            lam, Y = _smallest_sparse(
                L,
                degree if self.constraint == "degree" else None,
                self.n_components,
                True,
                self.random_state or 0,
            )
        else:
            W = adjacency_matrix(
                graph, weighting=self.weighting, bandwidth=self.bandwidth
            )
            lam, Y = laplacian_eigenmap(
                W, self.n_components, constraint=self.constraint
            )
        return dataclasses.replace(self, embedding=Y, eigenvalues=lam, graph=graph)


class SchrodingerEigenmaps(_GraphEmbedding):
    r"""Schrödinger eigenmaps (Czaja & Ehler, 2013) of the k-NN graph of ``X``.

    Laplacian eigenmaps steered by a potential $V$:
    $(L + \alpha V) y = \lambda D y$. Pass the potential to `fit`:

    - `barrier_potential` (diagonal): pins chosen points near the origin;
    - `label_potential`: pulls points sharing a label together
      (semi-supervised);
    - `spatial_spectral_potential`: pulls spatially adjacent, spectrally
      similar pixels together (hyperspectral imagery).

    With ``alpha = 0`` it is `LaplacianEigenmaps`. The graph and solver
    settings (``n_components``, ``n_neighbors``, ``weighting``, ``bandwidth``,
    ``constraint``, ``neighbors_backend``, ``eigen_solver``, ``random_state``)
    are as in `LaplacianEigenmaps`.

    Attributes:
        alpha: Weight of the potential.
        normalize_potential: Scale ``alpha`` by ``tr(L) / tr(V)``.
        drop_first: Drop the first solution (see `schrodinger_eigenmap`).
        embedding: ``(N, n_components)``, ``None`` before `fit`.
        eigenvalues: ``(n_components,)``, ``None`` before `fit`.
        graph: The `KNNGraph`, ``None`` before `fit`.

    Examples:
        >>> import jax
        >>> import jax.numpy as jnp
        >>> import kernellib as kl
        >>> X = jax.random.normal(jax.random.key(0), (80, 2))
        >>> labels = jnp.where(jnp.arange(80) < 10, 0, -1)  # 10 labelled
        >>> se = kl.SchrodingerEigenmaps(n_components=2, alpha=10.0)
        >>> se = se.fit(X, kl.label_potential(labels))
        >>> Y = se.embedding
        >>> spread = jnp.linalg.norm(Y[:10] - Y[:10].mean(0), axis=1).mean()
        >>> bool(spread < 0.2 * jnp.linalg.norm(Y - Y.mean(0), axis=1).mean())
        True
    """

    n_components: int = eqx.field(default=2, static=True)
    n_neighbors: int = eqx.field(default=10, static=True)
    alpha: float = 1.0
    normalize_potential: bool = eqx.field(default=True, static=True)
    drop_first: bool = eqx.field(default=True, static=True)
    weighting: Literal["heat", "connectivity"] = eqx.field(default="heat", static=True)
    bandwidth: float | None = None
    constraint: Constraint = eqx.field(default="degree", static=True)
    neighbors_backend: Backend = eqx.field(default="exact", static=True)
    eigen_solver: Solver = eqx.field(default="dense", static=True)
    random_state: int | None = eqx.field(default=None, static=True)
    embedding: Float[Array, "N n"] | None = None
    eigenvalues: Float[Array, " n"] | None = None
    graph: KNNGraph | None = None

    def __check_init__(self) -> None:
        _check_common(self)

    def fit(
        self,
        X: Float[Array, "N D"],
        potential: Float[Array, " N"] | Float[Array, "N N"],
    ) -> SchrodingerEigenmaps:
        """Embed ``X`` under ``potential`` (diagonal ``(N,)`` or ``(N, N)``).

        Raises:
            ValueError: If the potential does not match ``X``.
        """
        n = X.shape[0]
        potential = jnp.asarray(potential)
        if potential.shape not in ((n,), (n, n)):
            raise ValueError(
                f"potential must have shape ({n},) or ({n}, {n}), got "
                f"{potential.shape}."
            )
        graph = self._graph(X)
        if self.eigen_solver == "arpack":
            W = _sparse_adjacency(graph, self.weighting, self.bandwidth)
            degree = np.asarray(W.sum(axis=1)).ravel()
            L = sp.diags(degree) - W
            V_np = np.asarray(potential)
            V = sp.diags(V_np) if V_np.ndim == 1 else sp.csr_matrix(V_np)
            alpha = self.alpha
            if self.normalize_potential:
                alpha = alpha * L.diagonal().sum() / max(V.diagonal().sum(), 1e-30)
            lam, Y = _smallest_sparse(
                (L + alpha * V).tocsr(),
                degree if self.constraint == "degree" else None,
                self.n_components,
                self.drop_first,
                self.random_state or 0,
            )
        else:
            W = adjacency_matrix(
                graph, weighting=self.weighting, bandwidth=self.bandwidth
            )
            lam, Y = schrodinger_eigenmap(
                W,
                potential,
                self.n_components,
                alpha=self.alpha,
                normalize_potential=self.normalize_potential,
                constraint=self.constraint,
                drop_first=self.drop_first,
            )
        return dataclasses.replace(self, embedding=Y, eigenvalues=lam, graph=graph)


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


def _check_common(model: eqx.Module) -> None:
    if model.n_components < 1:  # ty: ignore[unresolved-attribute]
        raise ValueError("n_components must be >= 1.")
    if model.n_neighbors < 1:  # ty: ignore[unresolved-attribute]
        raise ValueError("n_neighbors must be >= 1.")
    constraint = getattr(model, "constraint", "degree")
    _check_constraint(constraint)
    solver = getattr(model, "eigen_solver", "dense")
    if solver not in ("dense", "arpack"):
        raise ValueError(f"eigen_solver must be 'dense' or 'arpack', got {solver!r}.")
