r"""Graph Matérn and heat kernels as kernel objects on node indices.

`matern_graph_kernel` and `diffusion_kernel` return the dense ``V x V``
matrix of a graph. `GraphMatern` and `GraphHeat` hold the graph and the
hyperparameters, take ``(N, 1)`` node indices as inputs, and gather rows and
columns of that matrix, so they follow the kernel contract (`fit(kernel, X)`
of `GraphEigenFeatures`, `KRR`, `to_operator`). Both are normalised to
average marginal variance $\sigma^2$ over the ``V`` nodes.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Literal

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from kernellib._einx import rearrange
from kernellib._graph._kernels import _spectral
from kernellib._graph._types import AbstractGraph
from kernellib._kernels import AbstractKernel
from kernellib.functional._graph import graph_heat_spectrum, graph_matern_spectrum


__all__ = ["GraphHeat", "GraphMatern"]

GraphNormalization = Literal["unnormalized", "symmetric"]


def _as_graph(
    graph: AbstractGraph | Float[Array, "V V"],
) -> AbstractGraph | Float[Array, "V V"]:
    return graph if isinstance(graph, AbstractGraph) else jnp.asarray(graph)


class AbstractGraphKernel(AbstractKernel):
    r"""A spectral graph kernel $U\,\Phi(\Lambda)\,U^\top$ on node indices.

    Subclasses provide the fields ``graph``, ``lengthscale``, ``variance`` and
    ``normalization`` and the spectrum $\Phi$. Gram-only: the kernel is a
    lookup in a ``V x V`` matrix, so it has no pointwise form to
    differentiate in the inputs.
    """

    if TYPE_CHECKING:  # declared by subclasses, in their own field order
        graph: AbstractGraph | Float[Array, "V V"]
        normalization: GraphNormalization

    @abstractmethod
    def spectrum(self, eigvals: Float[Array, " M"]) -> Float[Array, " M"]:
        r"""$\Phi(\lambda)$, scaled so that $\sum_k \Phi_k = V \sigma^2$.

        Over all ``V`` eigenvalues this is the kernel's average marginal
        variance $\sigma^2$; over the ``M`` smallest it normalises the
        truncated (rank-``M``) kernel the same way.
        """
        raise NotImplementedError

    @property
    def n_nodes(self) -> int:
        """Number of graph nodes ``V``."""
        if isinstance(self.graph, AbstractGraph):
            return self.graph.n_nodes
        return self.graph.shape[0]

    def matrix(self) -> Float[Array, "V V"]:
        """The dense ``V x V`` kernel matrix over all nodes."""
        return _spectral(self.graph, self.spectrum, self.normalization)

    def __call__(
        self, X1: Int[Array, "N1 1"], X2: Int[Array, "N2 1"]
    ) -> Float[Array, "N1 N2"]:
        K = self.matrix()
        return K[_indices(X1)][:, _indices(X2)]

    def diag(self, X: Int[Array, "N 1"]) -> Float[Array, " N"]:
        return jnp.diag(self.matrix())[_indices(X)]


def _indices(X: Int[Array, "N 1"]) -> Int[Array, " N"]:
    """Node indices from an ``(N, 1)`` array (integers or integral floats)."""
    X = jnp.asarray(X)
    if X.ndim != 2 or X.shape[-1] != 1:
        raise ValueError(
            f"Graph kernels take (N, 1) node indices, got shape {X.shape}."
        )
    return rearrange(X, "n 1 -> n").astype(jnp.int32)


def _check_normalization(normalization: str) -> None:
    if normalization not in ("unnormalized", "symmetric"):
        raise ValueError(
            "normalization must be 'unnormalized' or 'symmetric', got "
            f"{normalization!r}."
        )


class GraphMatern(AbstractGraphKernel):
    r"""Graph Matérn kernel on node indices (Borovitskiy et al., 2021).

    $k(i, j) = (U \Phi(\Lambda) U^\top)_{ij}$ with
    $\Phi(\lambda) \propto (2\nu/\ell^2 + \lambda)^{-\nu}$
    (`kernellib.functional.graph_matern_spectrum`) for the Laplacian
    $L = U \Lambda U^\top$, normalised so that the average marginal variance
    over the ``V`` nodes is ``variance``: the kernel object of
    `matern_graph_kernel`. Inputs are ``(N, 1)`` integer node indices.

    The Gram path is dense (a ``V x V`` eigendecomposition per call). For a
    large graph use `GraphEigenFeatures`, a rank-``M`` approximation from the
    smallest Laplacian eigenpairs.

    Attributes:
        graph: A graph, or a dense symmetric adjacency matrix.
        nu: Graph smoothness $\nu > 0$ (the SPDE exponent; no dimension
            enters). Keyword-only, as in `matern_graph_kernel`.
        lengthscale: $\ell > 0$.
        variance: Average marginal variance $\sigma^2$.
        normalization: Laplacian: ``"symmetric"`` (default) or
            ``"unnormalized"``.

    Examples:
        >>> import jax.numpy as jnp
        >>> import kernellib as kl
        >>> g = kl.grid_graph((5,))  # a path of 5 nodes
        >>> k = kl.GraphMatern(g, nu=1.5, lengthscale=2.0)
        >>> idx = jnp.array([[0], [1], [4]])
        >>> K = k(idx, idx)
        >>> K.shape
        (3, 3)
        >>> bool(K[0, 1] > K[0, 2] > 0)  # nearer nodes covary more
        True
        >>> K_full = kl.matern_graph_kernel(g, nu=1.5, lengthscale=2.0)
        >>> bool(jnp.allclose(K[0, 2], K_full[0, 4]))
        True
    """

    graph: AbstractGraph | Float[Array, "V V"] = eqx.field(converter=_as_graph)
    nu: float = eqx.field(static=True, kw_only=True)
    lengthscale: Float[Array, ""] = eqx.field(default=1.0, converter=jnp.asarray)
    variance: Float[Array, ""] = eqx.field(default=1.0, converter=jnp.asarray)
    normalization: GraphNormalization = eqx.field(default="symmetric", static=True)

    def __check_init__(self) -> None:
        if not self.nu > 0:
            raise ValueError(f"nu must be > 0, got {self.nu}.")
        _check_normalization(self.normalization)

    def spectrum(self, eigvals: Float[Array, " M"]) -> Float[Array, " M"]:
        return graph_matern_spectrum(
            eigvals,
            nu=self.nu,
            lengthscale=self.lengthscale,
            variance=self.variance,
            n_nodes=self.n_nodes,
        )


class GraphHeat(AbstractGraphKernel):
    r"""Graph heat (diffusion) kernel on node indices.

    $k(i, j) = (U \Phi(\Lambda) U^\top)_{ij}$ with
    $\Phi(\lambda) \propto e^{-\ell^2 \lambda / 2}$
    (`kernellib.functional.graph_heat_spectrum`), normalised so that the
    average marginal variance over the ``V`` nodes is ``variance``: the
    `diffusion_kernel` with $\beta = \ell^2 / 2$, rescaled. It is the
    $\nu \to \infty$ limit of `GraphMatern`. Inputs are ``(N, 1)`` integer
    node indices; the Gram path is dense, see `GraphEigenFeatures` for a
    rank-``M`` approximation.

    Attributes:
        graph: A graph, or a dense symmetric adjacency matrix.
        lengthscale: $\ell > 0$; the diffusion time is $\ell^2 / 2$.
        variance: Average marginal variance $\sigma^2$.
        normalization: Laplacian: ``"symmetric"`` (default) or
            ``"unnormalized"``.

    Examples:
        >>> import einx
        >>> import jax.numpy as jnp
        >>> import kernellib as kl
        >>> g = kl.grid_graph((5,))
        >>> k = kl.GraphHeat(g, lengthscale=2.0, variance=3.0)
        >>> idx = einx.id("n -> n 1", jnp.arange(5))  # every node
        >>> round(float(jnp.mean(k.diag(idx))), 6)  # average variance
        3.0
    """

    graph: AbstractGraph | Float[Array, "V V"] = eqx.field(converter=_as_graph)
    lengthscale: Float[Array, ""] = eqx.field(default=1.0, converter=jnp.asarray)
    variance: Float[Array, ""] = eqx.field(default=1.0, converter=jnp.asarray)
    normalization: GraphNormalization = eqx.field(default="symmetric", static=True)

    def __check_init__(self) -> None:
        _check_normalization(self.normalization)

    def spectrum(self, eigvals: Float[Array, " M"]) -> Float[Array, " M"]:
        return graph_heat_spectrum(
            eigvals,
            lengthscale=self.lengthscale,
            variance=self.variance,
            n_nodes=self.n_nodes,
        )
