r"""Edge weights: from distances or points to affinities.

Heat weights are the RBF kernel evaluated on the edges,
$w_{ij} = \exp(-d_{ij}^2 / 2\sigma^2)$, so any kernellib kernel can weight a
graph: $w_{ij} = k(x_i, x_j)$, restricted to a sparse topology. Self-tuning
(Zelnik-Manor & Perona, 2004) replaces $\sigma^2$ with $\sigma_i\sigma_j$,
the product of the endpoints' distances to their $k$-th neighbour, so the
bandwidth adapts to the local density.

Every builder turns its edges into weights through `edge_weigher`, so edges a
builder adds later (the bridges of ``ensure_connected``) are weighted exactly
like the rest.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from kernellib._einx import einsum, rearrange
from kernellib._kernels import AbstractKernel


__all__ = ["Bandwidth", "Weighting", "edge_weigher"]

Weighting = Literal["heat", "connectivity", "cosine"] | AbstractKernel
Bandwidth = float | Float[Array, ""] | Literal["median", "local"] | None

# ``(senders, receivers, distances) -> weights`` for a batch of edges; the
# endpoints are host index arrays, the distances a JAX array.
EdgeWeigher = Callable[[np.ndarray, np.ndarray, Float[Array, " E"]], Float[Array, " E"]]


def edge_weigher(
    weighting: Weighting,
    bandwidth: Bandwidth,
    *,
    distances: Float[Array, " M"],
    X: Float[Array, "N D"] | None = None,
    local_scale: Float[Array, " N"] | None = None,
) -> EdgeWeigher:
    r"""The edge-weighting function of a builder.

    Args:
        weighting: ``"heat"``, ``"connectivity"``, ``"cosine"`` or a kernel.
        bandwidth: For ``"heat"`` only: ``None`` / ``"median"`` (the median
            of ``distances``), ``"local"`` (self-tuning, needs
            ``local_scale``) or a fixed $\sigma$.
        distances: The builder's edge distances, for the median heuristic.
        X: The points, if the builder has them. ``"cosine"`` and
            non-stationary kernels need them; a stationary kernel without
            them is evaluated on the distance.
        local_scale: Each node's distance to its ``k``-th neighbour, for
            ``bandwidth="local"``.

    Returns:
        A function from edges to weights.

    Raises:
        ValueError: For an unknown ``weighting``, a ``bandwidth`` with a
            weighting other than ``"heat"``, or a weighting that needs the
            points when they are not available.
    """
    if bandwidth is not None and not (
        isinstance(weighting, str) and weighting == "heat"
    ):
        raise ValueError(
            "bandwidth applies to weighting='heat' only; a kernel carries its "
            "own lengthscale."
        )
    if isinstance(weighting, AbstractKernel):
        return _kernel_weigher(weighting, X)
    if weighting == "connectivity":
        return lambda s, r, d: jnp.ones_like(d)
    if weighting == "cosine":
        if X is None:
            raise ValueError(
                "weighting='cosine' needs the points: use knn_graph, "
                "radius_graph or edge_weights."
            )
        return lambda s, r, d: cosine_weights(X[s], X[r])
    if weighting != "heat":
        raise ValueError(
            "weighting must be 'heat', 'connectivity', 'cosine' or a kernel, "
            f"got {weighting!r}."
        )
    if isinstance(bandwidth, str) and bandwidth == "local":
        if local_scale is None:
            raise ValueError("bandwidth='local' needs k-nearest-neighbour distances.")
        scale = local_scale
        return lambda s, r, d: local_heat_weights(d, scale[s], scale[r])
    if bandwidth is None or (isinstance(bandwidth, str) and bandwidth == "median"):
        # No edges, no median: the bandwidth is then never used.
        sigma = jnp.median(distances) if distances.size else 1.0
    elif isinstance(bandwidth, str):
        raise ValueError(
            f"bandwidth must be a number, 'median', 'local' or None, got {bandwidth!r}."
        )
    else:
        sigma = bandwidth
    return lambda s, r, d: heat_weights(d, sigma)


def heat_weights(
    d: Float[Array, "*shape"], sigma: float | Float[Array, ""]
) -> Float[Array, "*shape"]:
    r"""$\exp(-d^2 / 2\sigma^2)$; a non-positive $\sigma$ is replaced by 1."""
    sigma = jnp.where(sigma > 0, sigma, 1.0)
    return jnp.exp(-(d**2) / (2.0 * sigma**2))


def local_heat_weights(
    d: Float[Array, " E"], sigma_i: Float[Array, " E"], sigma_j: Float[Array, " E"]
) -> Float[Array, " E"]:
    r"""Self-tuning $\exp(-d^2 / \sigma_i\sigma_j)$ (Zelnik-Manor & Perona)."""
    scale = sigma_i * sigma_j
    return jnp.exp(-(d**2) / jnp.where(scale > 0, scale, 1.0))


def cosine_weights(
    xi: Float[Array, "E D"], xj: Float[Array, "E D"]
) -> Float[Array, " E"]:
    """Cosine similarity of the endpoints, clipped at 0 (zero vectors give 0)."""
    dot = einsum(xi, xj, "e d, e d -> e")
    norms = jnp.sqrt(einsum(xi, xi, "e d, e d -> e") * einsum(xj, xj, "e d, e d -> e"))
    cos = dot / jnp.where(norms > 0, norms, 1.0)
    return jnp.clip(cos, min=0.0)


def _kernel_weigher(kernel: AbstractKernel, X: Float[Array, "N D"] | None):
    if X is not None:
        return lambda s, r, d: kernel.elwise(X[s], X[r])
    if not kernel.is_stationary:
        raise ValueError(
            "A non-stationary kernel needs the points: use knn_graph, "
            "radius_graph or edge_weights."
        )

    # An isotropic stationary kernel depends on the distance only:
    # k(x, y) = k(d e_1, 0). An ARD lengthscale fails the feature check.
    def weigh(s: np.ndarray, r: np.ndarray, d: Float[Array, " E"]) -> Array:
        d = rearrange(jnp.asarray(d), "e -> e 1")
        return kernel.elwise(d, jnp.zeros_like(d))

    return weigh
