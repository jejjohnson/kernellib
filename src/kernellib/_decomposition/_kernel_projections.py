r"""Kernel graph projections: kernel LPP and kernel Schrödinger projections.

By the representer theorem the embedding is $y = \bar K \alpha$, with
$\bar K = H K H^\top$ the Gram matrix centred at the degree-weighted mean in
feature space ($H = I - \mathbf 1 d^\top / \mathbf 1^\top d$, the kernel
analogue of LPP's centring, which keeps $y$ $D$-orthogonal to the constant).
With the factor $\bar K = F F^\top$, $F = U \Lambda^{1/2}$ from
$\bar K = U \Lambda U^\top$, and $y = F \beta$,

$$
F^\top (L + \alpha V) F \beta + \epsilon \beta = \lambda\, F^\top D F \beta ,
$$

the kernel analogue of LPP (He & Niyogi, 2003) and SEP. The ridge
$\epsilon\|\beta\|^2 = \epsilon\|f\|_{\mathcal H}^2$ penalises the RKHS norm of
the embedding function. New points embed as $\bar k(x, X)\,\alpha$ with
$\alpha = U \Lambda^{-1/2} \beta$.

With ``approx=`` a feature map $\phi$ (Nyström, random Fourier, ...) replaces
the kernel, $F = \bar\Phi$ (``N x M``, degree-centred), and the problem is
linear LPP / SEP in feature space, in $O(N M^2)$ instead of $O(N^3)$.
"""

from __future__ import annotations

import dataclasses
from typing import Literal

import einx
import equinox as eqx
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
)
from kernellib._decomposition._kpca import _symmetrise
from kernellib._decomposition._projections import (
    _degree_centre,
    _potential_term,
    _quadratic,
    _smallest_penalised,
    _weighted_gram,
)
from kernellib._einx import einsum
from kernellib._graph._construct import adjacency_matrix
from kernellib._graph._neighbors import Backend
from kernellib._kernels import AbstractKernel
from kernellib._spectral import AbstractFeatureMap


__all__ = ["KernelLocalityPreservingProjections", "KernelSchrodingerProjections"]


class _KernelProjection(_GraphEmbedding):
    """Shared by kernel LPP and kernel SEP: features, solve, transform."""

    def _fit(
        self,
        X: Float[Array, "N D"],
        graph: GraphLike | None,
        potential: Potential | None,
    ) -> _KernelProjection:
        n = X.shape[0]
        k = self.n_components  # ty: ignore[unresolved-attribute]
        if graph is None:
            graph = adjacency_matrix(
                self._graph(X),
                weighting=self.weighting,  # ty: ignore[unresolved-attribute]
                bandwidth=self.bandwidth,  # ty: ignore[unresolved-attribute]
            )
        else:
            _check_graph(graph, n)
            # The constraint F^T D F is the degree constraint: no signed graph.
            graph = _check_signed(graph, type(self).__name__)
        L, degree = _laplacian_and_degree(graph)
        approx: AbstractFeatureMap | None = self.approx  # ty: ignore[unresolved-attribute]
        kernel: AbstractKernel = self.kernel  # ty: ignore[unresolved-attribute]
        fitted: dict = {}
        if approx is not None:
            fmap = approx.fit(kernel, X)
            F, mu = _degree_centre(fmap(X), degree)
            if k > F.shape[1]:
                raise ValueError(f"n_components={k} exceeds the {F.shape[1]} features.")
            fitted.update(feature_map=fmap, feature_mean=mu)
        else:
            K = kernel(X, X)
            w = degree / jnp.sum(degree)
            r = K @ w
            t = w @ r
            Kc = K - einx.add("i, j -> i j", r, r) + t
            lam, U = jnp.linalg.eigh(_symmetrise(Kc))
            keep = lam > jnp.max(lam) * n * jnp.finfo(lam.dtype).eps
            _check_rank(keep, k)
            root = jnp.where(keep, jnp.sqrt(jnp.where(keep, lam, 1.0)), 0.0)
            inv_root = jnp.where(keep, 1.0 / jnp.where(keep, root, 1.0), 0.0)
            F = einx.multiply("n a, a -> n a", U, root)
            fitted.update(
                X_train=X,
                degree_weights=w,
                gram_weighted_means=r,
                gram_weighted_mean=t,
            )
        A = _quadratic(L, F)
        if potential is not None:
            alpha, FVF = _potential_term(
                F,
                potential,
                self.alpha,  # ty: ignore[unresolved-attribute]
                self.normalize_potential,  # ty: ignore[unresolved-attribute]
                degree,
            )
            A = A + alpha * FVF
        eigenvalues, beta = _smallest_penalised(
            A,
            _weighted_gram(F, degree),
            k,
            self.regularization,  # ty: ignore[unresolved-attribute]
        )
        if approx is not None:
            fitted.update(projection=beta)
        else:
            fitted.update(
                alphas=einsum(
                    einx.multiply("n a, a -> n a", U, inv_root),
                    beta,
                    "n a, a k -> n k",
                )
            )
        return dataclasses.replace(
            self,
            eigenvalues=eigenvalues,
            embedding=einsum(F, beta, "n a, a k -> n k"),
            **fitted,
        )

    def transform(self, X: Float[Array, "M D"]) -> Float[Array, "M n"]:
        """Embed new points: $\\bar k(x, X)\\,\\alpha$, or $(\\phi(x) - \\mu) B$.

        Raises:
            RuntimeError: If not fitted.
        """
        if self.projection is not None:  # ty: ignore[unresolved-attribute]
            Phi = einx.subtract(
                "m p, p -> m p",
                self.feature_map(X),  # ty: ignore[unresolved-attribute]
                self.feature_mean,  # ty: ignore[unresolved-attribute]
            )
            return Phi @ self.projection  # ty: ignore[unresolved-attribute]
        if self.alphas is None:  # ty: ignore[unresolved-attribute]
            raise RuntimeError(f"{type(self).__name__} is not fitted.")
        Kx = self.kernel(X, self.X_train)  # ty: ignore[unresolved-attribute]
        Kxc = (
            Kx
            - einx.add(
                "m, n -> m n",
                Kx @ self.degree_weights,  # ty: ignore[unresolved-attribute]
                self.gram_weighted_means,  # ty: ignore[unresolved-attribute]
            )
            + self.gram_weighted_mean  # ty: ignore[unresolved-attribute]
        )
        return Kxc @ self.alphas  # ty: ignore[unresolved-attribute]


def _check_rank(keep: Float[Array, " N"], k: int) -> None:
    """Raise when ``k`` exceeds the rank of the centred Gram (if concrete)."""
    try:
        rank = int(jnp.sum(keep))
    except jax.errors.ConcretizationTypeError:  # traced: can't check
        return
    if k > rank:
        raise ValueError(
            f"n_components={k} exceeds the rank {rank} of the centred Gram matrix."
        )


class KernelLocalityPreservingProjections(_KernelProjection):
    r"""Kernel locality preserving projections.

    LPP in the feature space of a kernel: the embedding $y = \bar K \alpha$
    minimises $y^\top L y + \epsilon \|f\|_{\mathcal H}^2$ subject to
    $y^\top D y = 1$ (see the module docstring), on the k-NN graph of ``X``
    or a ``graph`` passed to `fit` (e.g. a spatial one). With a `Linear`
    kernel and a vanishing ridge it spans the same embedding as
    `LocalityPreservingProjections`. The graph settings (``n_components``,
    ``n_neighbors``, ``weighting``, ``bandwidth``, ``neighbors_backend``,
    ``random_state``) are as in `LaplacianEigenmaps`.

    Attributes:
        kernel: The kernel.
        regularization: $\epsilon$, relative to the mean eigenvalue of
            $F^\top D F$ ($\mathrm{tr}(D\bar K) / N$ on the exact path).
        approx: Optional unfitted feature map for the ``O(N M^2)`` path.
        eigenvalues: ``(n_components,)``, ``None`` before `fit`.
        embedding: The training points' embedding ``(N, n_components)``.
        X_train: Training inputs (exact path).
        alphas: Dual coefficients ``(N, n_components)`` (exact path).
        feature_map: The fitted feature map (approximate path).
        projection: ``(M, n_components)`` (approximate path).

    Examples:
        >>> import jax
        >>> import jax.numpy as jnp
        >>> import kernellib as kl
        >>> X = jax.random.normal(jax.random.key(0), (60, 3))
        >>> klpp = kl.KernelLocalityPreservingProjections(
        ...     kl.RBF(1.0), n_components=2
        ... )
        >>> klpp = klpp.fit(X)
        >>> bool(jnp.allclose(klpp.transform(X), klpp.embedding, atol=1e-4))
        True

        With Nyström features, in $O(N M^2)$:

        >>> approx = kl.NystromFeatures(30, jax.random.key(1))
        >>> fast = kl.KernelLocalityPreservingProjections(
        ...     kl.RBF(1.0), n_components=2, approx=approx
        ... ).fit(X)
        >>> fast.transform(X[:5]).shape
        (5, 2)
    """

    kernel: AbstractKernel
    n_components: int = eqx.field(default=2, static=True)
    n_neighbors: int = eqx.field(default=10, static=True)
    weighting: Literal["heat", "connectivity"] = eqx.field(default="heat", static=True)
    bandwidth: float | None = None
    regularization: float = 1e-3
    approx: AbstractFeatureMap | None = None
    neighbors_backend: Backend = eqx.field(default="exact", static=True)
    random_state: int | None = eqx.field(default=None, static=True)
    eigenvalues: Float[Array, " n"] | None = None
    embedding: Float[Array, "N n"] | None = None
    X_train: Float[Array, "N D"] | None = None
    alphas: Float[Array, "N n"] | None = None
    degree_weights: Float[Array, " N"] | None = None
    gram_weighted_means: Float[Array, " N"] | None = None
    gram_weighted_mean: Float[Array, ""] | None = None
    feature_map: AbstractFeatureMap | None = None
    feature_mean: Float[Array, " M"] | None = None
    projection: Float[Array, "M n"] | None = None

    def __check_init__(self) -> None:
        _check_common(self)

    def fit(
        self, X: Float[Array, "N D"], *, graph: GraphLike | None = None
    ) -> KernelLocalityPreservingProjections:
        """Fit the projection on ``X``.

        Args:
            X: Training inputs, ``(N, D)``.
            graph: Optional precomputed graph (an `AbstractGraph` or adjacency
                matrix) instead of the k-NN graph of ``X``.

        Raises:
            ValueError: If ``n_components`` exceeds the rank of the centred
                Gram matrix (or the number of features), or ``graph`` does
                not match ``X`` or has negative edge weights (the degree
                constraint needs ``w >= 0``).
        """
        return self._fit(X, graph, None)  # ty: ignore[invalid-return-type]


class KernelSchrodingerProjections(_KernelProjection):
    r"""Kernel Schrödinger eigenmap projections.

    `KernelLocalityPreservingProjections` steered by a potential $V$ on the
    training points: $y = \bar K \alpha$ minimises
    $y^\top (L + \alpha V) y + \epsilon \|f\|_{\mathcal H}^2$ subject to
    $y^\top D y = 1$. ``alpha = 0`` is kernel LPP; the trace normalisation
    of $\alpha$ is done in ``N`` space, as in `SchrodingerEigenmapProjections`.
    The other settings and attributes are as in
    `KernelLocalityPreservingProjections`.

    Attributes:
        alpha: Weight of the potential.
        normalize_potential: Scale ``alpha`` by ``tr(L) / tr(V)``.

    Examples:
        >>> import jax
        >>> import jax.numpy as jnp
        >>> import kernellib as kl
        >>> X = jax.random.normal(jax.random.key(0), (60, 3))
        >>> labels = jnp.where(jnp.arange(60) < 10, 0, -1)
        >>> ksep = kl.KernelSchrodingerProjections(kl.RBF(1.0), alpha=5.0)
        >>> ksep = ksep.fit(X, kl.label_potential(labels))
        >>> ksep.transform(X[:4]).shape
        (4, 2)
    """

    kernel: AbstractKernel
    n_components: int = eqx.field(default=2, static=True)
    n_neighbors: int = eqx.field(default=10, static=True)
    alpha: float = 1.0
    normalize_potential: bool = eqx.field(default=True, static=True)
    weighting: Literal["heat", "connectivity"] = eqx.field(default="heat", static=True)
    bandwidth: float | None = None
    regularization: float = 1e-3
    approx: AbstractFeatureMap | None = None
    neighbors_backend: Backend = eqx.field(default="exact", static=True)
    random_state: int | None = eqx.field(default=None, static=True)
    eigenvalues: Float[Array, " n"] | None = None
    embedding: Float[Array, "N n"] | None = None
    X_train: Float[Array, "N D"] | None = None
    alphas: Float[Array, "N n"] | None = None
    degree_weights: Float[Array, " N"] | None = None
    gram_weighted_means: Float[Array, " N"] | None = None
    gram_weighted_mean: Float[Array, ""] | None = None
    feature_map: AbstractFeatureMap | None = None
    feature_mean: Float[Array, " M"] | None = None
    projection: Float[Array, "M n"] | None = None

    def __check_init__(self) -> None:
        _check_common(self)

    def fit(
        self,
        X: Float[Array, "N D"],
        potential: Potential,
        *,
        graph: GraphLike | None = None,
    ) -> KernelSchrodingerProjections:
        """Fit the projection on ``X`` under ``potential``.

        Args:
            X: Training inputs, ``(N, D)``.
            potential: Diagonal ``(N,)``, matrix ``(N, N)`` or lineax operator
                on the points of ``X``.
            graph: Optional precomputed graph instead of the k-NN graph of
                ``X``.

        Raises:
            ValueError: If ``n_components`` exceeds the rank of the centred
                Gram matrix (or the number of features), the potential or
                ``graph`` does not match ``X``, or ``graph`` has negative
                edge weights (the degree constraint needs ``w >= 0``).
        """
        if not isinstance(potential, lx.AbstractLinearOperator):
            potential = jnp.asarray(potential)
        _check_potential(potential, X.shape[0])
        return self._fit(X, graph, potential)  # ty: ignore[invalid-return-type]
