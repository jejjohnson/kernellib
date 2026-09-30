"""Graph construction: adjacency matrices from neighbour graphs."""

from __future__ import annotations

from typing import Literal

import jax.numpy as jnp
from jaxtyping import Array, Float

from kernellib._einx import rearrange
from kernellib._graph._neighbors import KNNGraph


__all__ = ["adjacency_matrix"]


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
