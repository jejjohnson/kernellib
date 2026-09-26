r"""Graphs from neighbours, graph Laplacians, and graph kernels.

A weighted, undirected graph on ``N`` points is its symmetric adjacency matrix
$W$ (``N x N``, non-negative, zero diagonal). Its degree matrix is
$D = \mathrm{diag}(W\mathbf{1})$ and its Laplacian $L = D - W$ (or a
normalised form). Graph kernels are spectral functions of the Laplacian,
$K = U f(\Lambda) U^\top$ for $L = U\Lambda U^\top$ (Smola & Kondor, 2003):
they are positive semidefinite similarity matrices between the nodes, so they
plug into anything in kernellib that takes a Gram matrix (``functional.hsic``,
`KRR` via a precomputed operator, `KernelPCA` on nodes).

Everything here is dense and differentiable; the ``N x N`` matrices limit it
to graphs of a few thousand nodes. The eigenmaps' ``eigen_solver="arpack"``
path works on the sparse graph instead.
"""

from __future__ import annotations

from typing import Literal

import jax.numpy as jnp
from jaxtyping import Array, Float

from kernellib._decomposition._neighbors import KNNGraph
from kernellib._einx import rearrange


__all__ = [
    "adjacency_matrix",
    "commute_time_kernel",
    "cosine_graph_kernel",
    "diffusion_kernel",
    "graph_laplacian",
    "random_walk_kernel",
    "regularized_laplacian_kernel",
]

Normalization = Literal["unnormalized", "symmetric", "random_walk"]


def adjacency_matrix(
    graph: KNNGraph,
    *,
    weighting: Literal["heat", "connectivity"] = "heat",
    bandwidth: float | Float[Array, ""] | None = None,
    symmetrize: Literal["max", "mean", "min"] = "max",
) -> Float[Array, "N N"]:
    r"""Symmetric weighted adjacency matrix of a k-NN graph.

    Edge weights are $w_{ij} = \exp(-d_{ij}^2 / 2\sigma^2)$ (``"heat"``, the
    RBF kernel on the edge) or $1$ (``"connectivity"``). The k-NN relation is
    not symmetric; ``symmetrize`` makes it so: ``"max"`` keeps an edge if
    either end lists the other (the union), ``"min"`` only if both do (mutual
    neighbours), ``"mean"`` averages the two directed weights.

    Args:
        graph: From `nearest_neighbors`.
        weighting: ``"heat"`` or ``"connectivity"``.
        bandwidth: Heat-kernel $\sigma$; ``None`` uses the median neighbour
            distance.
        symmetrize: ``"max"``, ``"min"`` or ``"mean"``.

    Returns:
        ``(N, N)`` symmetric, non-negative, zero-diagonal matrix.

    Examples:
        >>> import jax.numpy as jnp
        >>> import kernellib as kl
        >>> X = jnp.array([[0.0], [1.0], [3.0]])
        >>> W = kl.adjacency_matrix(
        ...     kl.nearest_neighbors(X, 1), weighting="connectivity"
        ... )
        >>> W.tolist()
        [[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]
    """
    n, k = graph.indices.shape
    if weighting == "heat":
        sigma = jnp.median(graph.distances) if bandwidth is None else bandwidth
        sigma = jnp.where(sigma > 0, sigma, 1.0)
        weights = jnp.exp(-(graph.distances**2) / (2.0 * sigma**2))
    elif weighting == "connectivity":
        weights = jnp.ones_like(graph.distances)
    else:
        raise ValueError(
            f"weighting must be 'heat' or 'connectivity', got {weighting!r}."
        )
    rows = jnp.repeat(jnp.arange(n), k)
    directed = (
        jnp.zeros((n, n), weights.dtype)
        .at[rows, rearrange(graph.indices, "n k -> (n k)")]
        .max(rearrange(weights, "n k -> (n k)"))
    )
    if symmetrize == "max":
        W = jnp.maximum(directed, directed.T)
    elif symmetrize == "min":
        W = jnp.minimum(directed, directed.T)
    elif symmetrize == "mean":
        W = 0.5 * (directed + directed.T)
    else:
        raise ValueError(
            f"symmetrize must be 'max', 'min' or 'mean', got {symmetrize!r}."
        )
    return W * (1.0 - jnp.eye(n, dtype=W.dtype))


def graph_laplacian(
    W: Float[Array, "N N"], normalization: Normalization = "unnormalized"
) -> Float[Array, "N N"]:
    r"""Graph Laplacian of a symmetric adjacency matrix.

    - ``"unnormalized"``: $L = D - W$.
    - ``"symmetric"``: $L = I - D^{-1/2} W D^{-1/2}$, eigenvalues in $[0, 2]$.
    - ``"random_walk"``: $L = I - D^{-1} W$ (not symmetric).

    Isolated nodes (zero degree) get a zero row in the normalised forms.

    Examples:
        >>> import jax.numpy as jnp
        >>> import kernellib as kl
        >>> W = jnp.array([[0.0, 1.0], [1.0, 0.0]])
        >>> kl.graph_laplacian(W).tolist()
        [[1.0, -1.0], [-1.0, 1.0]]
    """
    degree = jnp.sum(W, axis=1)
    if normalization == "unnormalized":
        return jnp.diag(degree) - W
    safe = jnp.where(degree > 0, degree, 1.0)
    connected = (degree > 0).astype(W.dtype)
    if normalization == "symmetric":
        d_isqrt = connected / jnp.sqrt(safe)
        return jnp.diag(connected) - d_isqrt[:, None] * W * d_isqrt[None, :]
    if normalization == "random_walk":
        return jnp.diag(connected) - (connected / safe)[:, None] * W
    raise ValueError(
        "normalization must be 'unnormalized', 'symmetric' or 'random_walk', got "
        f"{normalization!r}."
    )


def _spectral(
    W: Float[Array, "N N"], fn, normalization: Normalization
) -> Float[Array, "N N"]:
    """``U f(Λ) Uᵀ`` for the (symmetric) Laplacian ``U Λ Uᵀ``."""
    if normalization == "random_walk":
        raise ValueError("Graph kernels need a symmetric Laplacian.")
    lam, U = jnp.linalg.eigh(graph_laplacian(W, normalization))
    return (U * fn(jnp.clip(lam, min=0.0))) @ U.T


def diffusion_kernel(
    W: Float[Array, "N N"],
    beta: float | Float[Array, ""] = 1.0,
    *,
    normalization: Normalization = "symmetric",
) -> Float[Array, "N N"]:
    r"""Diffusion (heat) kernel $K = \exp(-\beta L)$ (Kondor & Lafferty, 2002).

    Heat diffusing on the graph for time $\beta$: small $\beta$ is local,
    large $\beta$ spreads similarity along the graph.

    Examples:
        >>> import jax.numpy as jnp
        >>> import kernellib as kl
        >>> W = jnp.array([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
        >>> K = kl.diffusion_kernel(W, beta=0.5)
        >>> bool(K[0, 1] > K[0, 2] > 0)  # neighbours are more similar
        True
    """
    return _spectral(W, lambda lam: jnp.exp(-beta * lam), normalization)


def regularized_laplacian_kernel(
    W: Float[Array, "N N"],
    sigma: float | Float[Array, ""] = 1.0,
    *,
    normalization: Normalization = "symmetric",
) -> Float[Array, "N N"]:
    r"""Regularised Laplacian kernel $K = (I + \sigma^2 L)^{-1}$.

    Smola & Kondor (2003); $\sigma$ sets how far similarity spreads.
    """
    return _spectral(W, lambda lam: 1.0 / (1.0 + sigma**2 * lam), normalization)


def random_walk_kernel(
    W: Float[Array, "N N"],
    p: int = 1,
    a: float = 2.0,
    *,
    normalization: Normalization = "symmetric",
) -> Float[Array, "N N"]:
    r"""$p$-step random-walk kernel $K = (aI - L)^p$ (Smola & Kondor, 2003).

    Positive semidefinite for $a \ge \lambda_{\max}(L)$, i.e. $a \ge 2$ with
    the symmetric normalisation (the default).

    Raises:
        ValueError: If ``p < 1``.
    """
    if p < 1:
        raise ValueError(f"p must be >= 1, got {p}.")
    return _spectral(W, lambda lam: (a - lam) ** p, normalization)


def cosine_graph_kernel(
    W: Float[Array, "N N"], *, normalization: Normalization = "symmetric"
) -> Float[Array, "N N"]:
    r"""Inverse-cosine kernel $K = \cos(\pi L / 4)$ (Smola & Kondor, 2003).

    Positive semidefinite for the symmetric normalisation, whose spectrum lies
    in $[0, 2]$.
    """
    return _spectral(W, lambda lam: jnp.cos(jnp.pi * lam / 4.0), normalization)


def commute_time_kernel(
    W: Float[Array, "N N"],
    *,
    normalization: Normalization = "unnormalized",
    rtol: float = 1e-10,
) -> Float[Array, "N N"]:
    r"""Commute-time kernel $K = L^{+}$, the Laplacian's pseudo-inverse.

    $K_{ii} + K_{jj} - 2K_{ij}$ is proportional to the expected commute time of
    a random walk between nodes $i$ and $j$ (Fouss et al., 2007). Eigenvalues
    below ``rtol`` times the largest (the constant vector on each connected
    component) are treated as zero.
    """

    def pinv(lam: Float[Array, " N"]) -> Float[Array, " N"]:
        keep = lam > rtol * jnp.max(lam)
        return jnp.where(keep, 1.0 / jnp.where(keep, lam, 1.0), 0.0)

    return _spectral(W, pinv, normalization)
