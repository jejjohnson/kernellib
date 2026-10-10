r"""Rank-``M`` features of the graph Matérn and heat kernels.

A spectral graph kernel is $K = U\,\Phi(\Lambda)\,U^\top$ for the Laplacian
$L = U \Lambda U^\top$. Keeping the ``M`` smallest eigenpairs (the smooth
eigenvectors, which carry the most prior variance) gives the features
$\phi(i) = U_{i,:M}\,\operatorname{diag}(\sqrt{\Phi(\lambda_{1:M})})$, so that
$\phi(i)^\top \phi(j) \approx k(i, j)$ without a ``V x V`` matrix
(Borovitskiy et al., 2021).
"""

from __future__ import annotations

import dataclasses
from typing import Literal

import einx
import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, PRNGKeyArray

from kernellib._graph._eigpairs import laplacian_eigpairs
from kernellib._graph._kernel_classes import AbstractGraphKernel, _indices
from kernellib._kernels import AbstractKernel
from kernellib._spectral._base import AbstractFeatureMap


__all__ = ["GraphEigenFeatures"]


class GraphEigenFeatures(AbstractFeatureMap):
    r"""Low-rank features of `GraphMatern` / `GraphHeat` from Laplacian eigenpairs.

    $$
    \phi(i) = U_{i,:M}\,\operatorname{diag}\big(\sqrt{\Phi(\lambda_{1:M})}\big),
    $$

    with $(\lambda_{1:M}, U_{:, :M})$ the ``M`` smallest eigenpairs of the
    kernel's Laplacian (`laplacian_eigpairs`, computed once at `fit`) and
    $\Phi$ the kernel's spectrum (`kernellib.functional.graph_matern_spectrum`
    or `graph_heat_spectrum`). With ``M = V`` the features reproduce the dense
    kernel exactly.

    **Truncated normalisation.** $\Phi$ is normalised over the *retained*
    ``M`` eigenvalues, $\sum_{k \le M} \Phi_k = V\sigma^2$, so the rank-``M``
    prior has average marginal variance $\sigma^2$ at every ``M`` (the choice
    of pyrox-gp's ``LaplacianInducingFeatures``). It therefore differs from
    the dense kernel by more than the dropped tail: every retained mode is
    scaled up by $\sum_{k \le V} \Phi_k / \sum_{k \le M} \Phi_k$.

    The eigenpairs are fixed at `fit`; the kernel's ``lengthscale`` and
    ``variance`` are read at call time (only the ``M``-vector $\Phi$ is
    recomputed), so gradients reach them. Gradients do not reach the graph's
    edge weights: refit after reweighting. Cost: the eigensolve at `fit`,
    then $O(NM)$ per call. A truncation at ``M`` inside a repeated eigenvalue
    (common on grids) depends on the eigensolver's basis of that eigenspace;
    choose ``M`` at a gap for solver-independent features.

    Attributes:
        n_components: Number of eigenpairs ``M``, ``1 <= M <= V``.
        method: Eigensolver of `laplacian_eigpairs`: ``"dense"``,
            ``"kronecker"`` (a face-connected `GridGraph` with the
            ``"unnormalized"`` Laplacian), ``"lanczos"`` (needs ``key`` at
            `fit`), ``"arpack"``, or ``None`` for its type-based default.
        eigvals: Fitted Laplacian eigenvalues, shape ``(M,)``, ascending.
        eigvecs: Fitted eigenvectors, shape ``(V, M)``.
        kernel: The fitted `GraphMatern` or `GraphHeat`, ``None`` before `fit`.

    Examples:
        >>> import einx
        >>> import jax.numpy as jnp
        >>> import kernellib as kl
        >>> g = kl.grid_graph((10, 10))
        >>> k = kl.GraphMatern(
        ...     g,
        ...     nu=1.5,
        ...     lengthscale=3.0,
        ...     variance=2.0,
        ...     normalization="unnormalized",
        ... )
        >>> nodes = einx.id("n -> n 1", jnp.arange(g.n_nodes))
        >>> fm = kl.GraphEigenFeatures(20).fit(k, nodes)  # Kronecker eigenpairs
        >>> Phi = fm(nodes)
        >>> Phi.shape
        (100, 20)
        >>> var = einx.dot("n m, n m -> n", Phi, Phi)
        >>> round(float(jnp.mean(var)), 6)  # average variance at any M
        2.0
    """

    n_components: int = eqx.field(static=True)
    method: Literal["dense", "kronecker", "lanczos", "arpack"] | None = eqx.field(
        default=None, static=True
    )
    eigvals: Float[Array, " M"] | None = None
    eigvecs: Float[Array, "V M"] | None = None
    kernel: AbstractGraphKernel | None = None

    def __check_init__(self) -> None:
        if self.n_components < 1:
            raise ValueError(f"n_components must be >= 1, got {self.n_components}.")

    def fit(
        self,
        kernel: AbstractKernel,
        X: Int[Array, "N 1"],
        *,
        key: PRNGKeyArray | None = None,
    ) -> GraphEigenFeatures:
        """Return a copy holding the kernel's ``M`` smallest Laplacian eigenpairs.

        ``X`` (node indices) is only checked: the eigenpairs cover every node.

        Args:
            kernel: A `GraphMatern` or `GraphHeat`.
            X: ``(N, 1)`` node indices.
            key: PRNG key for ``method="lanczos"`` (required) or ``"arpack"``.

        Raises:
            TypeError: If ``kernel`` is not a `GraphMatern` or `GraphHeat`.
            ValueError: If ``n_components`` exceeds the number of nodes, or the
                eigensolver rejects the graph (see `laplacian_eigpairs`).
        """
        if not isinstance(kernel, AbstractGraphKernel):
            raise TypeError(
                "GraphEigenFeatures needs a GraphMatern or GraphHeat kernel, got "
                f"{type(kernel).__name__}."
            )
        _indices(X)
        eigvals, eigvecs = laplacian_eigpairs(
            kernel.graph,
            self.n_components,
            normalization=kernel.normalization,
            method=self.method,
            key=key,
        )
        return dataclasses.replace(
            self, eigvals=eigvals, eigvecs=eigvecs, kernel=kernel
        )

    @property
    def weights(self) -> Float[Array, " M"]:
        r"""$\Phi(\lambda_{1:M})$, normalised to $\sum_k \Phi_k = V\sigma^2$."""
        assert self.kernel is not None and self.eigvals is not None
        return self.kernel.spectrum(jnp.clip(self.eigvals, min=0.0))

    def features(self, X: Int[Array, "N 1"]) -> Float[Array, "N M"]:
        assert self.eigvecs is not None
        rows = self.eigvecs[_indices(X)]
        phi = self.weights
        # High modes can underflow to 0 (heat, long lengthscale); the double
        # where keeps sqrt's infinite derivative there out of the gradient.
        positive = phi > 0.0
        sqrt_phi = jnp.where(positive, jnp.sqrt(jnp.where(positive, phi, 1.0)), 0.0)
        return einx.multiply("n m, m -> n m", rows, sqrt_phi)
