r"""Proximity graphs of points in two or three dimensions: Delaunay, Gabriel
and relative neighbourhood.

The Delaunay triangulation $DT$ of points in general position contains a
nested family of parameter-free proximity graphs,

$$
\mathrm{EMST} \subseteq \mathrm{RNG} \subseteq \mathrm{GG} \subseteq DT,
$$

so every one contains the Euclidean minimum spanning tree and is therefore
**connected**, and each has $O(N)$ edges. Unlike a k-NN graph there is no
``k`` to tune. They suit ICAR / Besag priors on irregular sites
(`structure_matrix`).

- **Gabriel graph (GG).** $(i, j)$ is an edge when the ball with diameter
  $x_ix_j$ contains no other point, $d_{ik}^2 + d_{jk}^2 \ge d_{ij}^2$ for
  all $k$.
- **Relative neighbourhood graph (RNG).** $(i, j)$ is an edge when no point
  is closer to both, $\max(d_{ik}, d_{jk}) \ge d_{ij}$ for all $k$.

Both are filters of the Delaunay edges. A point that blocks the edge
$(i, j)$ can always be replaced by a blocking Delaunay neighbour of $i$ or
$j$, so only those are checked; in 2-D the Gabriel test needs just the one
or two vertices opposite the edge (the angle there is at least 90°).

The triangulation is `scipy.spatial.Delaunay` (Qhull) on the host, imported
lazily: like every builder these run eagerly, outside ``jax.jit``. The edge
weights are JAX arrays (differentiable in a heat ``bandwidth`` or a kernel's
hyperparameters); the points themselves must be concrete.
"""

from __future__ import annotations

from typing import Literal

import einx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike, Float

from kernellib._graph._construct import (
    _concrete,
    _triangle_edges,
    graph_from_edges,
)
from kernellib._graph._types import Graph
from kernellib._graph._weights import Weighting


__all__ = ["delaunay_graph", "gabriel_graph", "relative_neighborhood_graph"]

Bandwidth = float | Float[Array, ""] | Literal["median"] | None

# The four triangular faces of a tetrahedron, as corner indices.
_TET_FACES = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])


def delaunay_graph(
    X: Float[ArrayLike, "N d"],
    *,
    weighting: Weighting = "heat",
    bandwidth: Bandwidth = None,
) -> Graph:
    r"""The edge graph of the Delaunay triangulation of ``X``.

    Two points are joined when they share a triangle (2-D) or tetrahedron
    (3-D) of the Delaunay triangulation. It contains the Gabriel and
    relative neighbourhood graphs and the Euclidean minimum spanning tree,
    so it is connected.

    Args:
        X: Points, shape ``(N, 2)`` or ``(N, 3)`` (concrete).
        weighting: ``"heat"``, ``"connectivity"`` or a stationary kernel,
            applied to the edge lengths (see `graph_from_edges`).
        bandwidth: Heat $\sigma$; ``None`` / ``"median"`` uses the median
            edge length.

    Returns:
        A `Graph` on ``N`` nodes.

    Raises:
        ValueError: For ``d`` other than 2 or 3 (use `knn_graph`), fewer
            than ``d + 1`` points, duplicate points, or points that are all
            collinear (2-D) or coplanar (3-D).

    Examples:
        >>> import jax.numpy as jnp
        >>> import kernellib as kl
        >>> X = jnp.array([[0.0, 0.0], [2.0, 0.0], [1.0, 0.5], [1.0, 2.0]])
        >>> g = kl.delaunay_graph(X, weighting="connectivity")
        >>> g.topology.senders.tolist(), g.topology.receivers.tolist()
        ([0, 0, 0, 1, 1, 2], [1, 2, 3, 2, 3, 3])
    """
    X, Xh, tri = _delaunay(X, "delaunay_graph")
    topology, _ = _simplex_edges(tri.simplices, Xh.shape[0])
    return _weighted(X, Xh, topology.senders, topology.receivers, weighting, bandwidth)


def gabriel_graph(
    X: Float[ArrayLike, "N d"],
    *,
    weighting: Weighting = "heat",
    bandwidth: Bandwidth = None,
) -> Graph:
    r"""The Gabriel graph of ``X``.

    $(i, j)$ is an edge when no other point lies inside the ball with
    diameter $x_ix_j$: $d_{ik}^2 + d_{jk}^2 \ge d_{ij}^2$ for all $k$. It
    sits between the relative neighbourhood graph and the Delaunay graph,
    so it is connected, with $O(N)$ edges and no parameter to tune. In 2-D
    each Delaunay edge is tested against the one or two vertices opposite
    it; in 3-D against the Delaunay neighbours of its endpoints.

    Args:
        X: Points, shape ``(N, 2)`` or ``(N, 3)`` (concrete).
        weighting: ``"heat"``, ``"connectivity"`` or a stationary kernel,
            applied to the edge lengths (see `graph_from_edges`).
        bandwidth: Heat $\sigma$; ``None`` / ``"median"`` uses the median
            edge length.

    Returns:
        A `Graph` on ``N`` nodes.

    Raises:
        ValueError: As `delaunay_graph`.

    Examples:
        An ICAR prior on irregular monitoring sites, with no ``k`` to tune:

        >>> import jax.numpy as jnp
        >>> import kernellib as kl
        >>> station_xy = jnp.array([[0.0, 0.0], [2.0, 0.0], [1.0, 0.5], [1.0, 2.0]])
        >>> stations = kl.gabriel_graph(station_xy, weighting="connectivity")
        >>> stations.topology.senders.tolist(), stations.topology.receivers.tolist()
        ([0, 1, 2], [2, 2, 3])
        >>> prior = kl.structure_matrix(stations)  # Besag structure R = L
        >>> prior.as_matrix().tolist()[2]
        [-1.0, -1.0, 3.0, -1.0]
    """
    X, Xh, tri = _delaunay(X, "gabriel_graph")
    n = Xh.shape[0]
    topology, pair = _simplex_edges(tri.simplices, n)
    s, r = topology.senders, topology.receivers
    if Xh.shape[1] == 2:
        # Corner k of each triangle is opposite the edge (k+1, k+2); the edge
        # fails when that angle is obtuse, d_ik^2 + d_jk^2 < d_ij^2.
        p = Xh[tri.simplices]  # (T, 3, 2)
        u = p[:, [1, 2, 0]] - p
        v = p[:, [2, 0, 1]] - p
        obtuse = einx.dot("t k d, t k d -> (t k)", u, v) < 0.0
        blocked = np.bincount(pair[obtuse], minlength=topology.n_edges) > 0
    else:
        blocked = _blocked(Xh, s, r, tri, "gabriel")
    keep = np.flatnonzero(~blocked)
    return _weighted(X, Xh, s[keep], r[keep], weighting, bandwidth)


def relative_neighborhood_graph(
    X: Float[ArrayLike, "N d"],
    *,
    weighting: Weighting = "heat",
    bandwidth: Bandwidth = None,
) -> Graph:
    r"""The relative neighbourhood graph of ``X``.

    $(i, j)$ is an edge when no point is closer to both endpoints than they
    are to each other: $\max(d_{ik}, d_{jk}) \ge d_{ij}$ for all $k$ (the
    *lune* of $x_ix_j$ is empty). It contains the Euclidean minimum
    spanning tree and is contained in the Gabriel graph, so it is the
    sparsest connected graph of the family. Each Delaunay edge is tested
    against the Delaunay neighbours of its endpoints.

    Args:
        X: Points, shape ``(N, 2)`` or ``(N, 3)`` (concrete).
        weighting: ``"heat"``, ``"connectivity"`` or a stationary kernel,
            applied to the edge lengths (see `graph_from_edges`).
        bandwidth: Heat $\sigma$; ``None`` / ``"median"`` uses the median
            edge length.

    Returns:
        A `Graph` on ``N`` nodes.

    Raises:
        ValueError: As `delaunay_graph`.

    Examples:
        >>> import jax.numpy as jnp
        >>> import kernellib as kl
        >>> X = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.8]])  # 0-1 is longest
        >>> g = kl.relative_neighborhood_graph(X, weighting="connectivity")
        >>> g.topology.senders.tolist(), g.topology.receivers.tolist()
        ([0, 1], [2, 2])
    """
    X, Xh, tri = _delaunay(X, "relative_neighborhood_graph")
    topology, _ = _simplex_edges(tri.simplices, Xh.shape[0])
    s, r = topology.senders, topology.receivers
    keep = np.flatnonzero(~_blocked(Xh, s, r, tri, "rng"))
    return _weighted(X, Xh, s[keep], r[keep], weighting, bandwidth)


# -- helpers -----------------------------------------------------------------


def _delaunay(X: Float[ArrayLike, "N d"], name: str):
    """Validate ``X`` and triangulate it: ``(X, host float64 copy, Delaunay)``."""
    from scipy.spatial import Delaunay, QhullError

    Xh = _concrete(X, "X").astype(np.float64)
    if Xh.ndim != 2:
        raise ValueError(f"X must have shape (N, d), got {Xh.shape}.")
    n, d = Xh.shape
    if d not in (2, 3):
        raise ValueError(
            f"{name} needs points in 2 or 3 dimensions, got d={d}: the Delaunay "
            "triangulation grows exponentially with d. Use knn_graph instead."
        )
    if not np.all(np.isfinite(Xh)):
        raise ValueError("X must be finite.")
    if n < d + 1:
        raise ValueError(f"{name} needs at least {d + 1} points in {d}-D, got {n}.")
    if np.unique(Xh, axis=0).shape[0] < n:
        raise ValueError(
            f"{name} needs distinct points: X has duplicate rows (merge them first)."
        )
    try:
        tri = Delaunay(Xh)
    except QhullError as err:
        flat = "collinear" if d == 2 else "coplanar"
        raise ValueError(
            f"{name} cannot triangulate X: the points are (nearly) {flat}, so "
            f"they span fewer than {d} dimensions."
        ) from err
    if tri.coplanar.size:
        raise ValueError(
            f"{name}: {tri.coplanar.shape[0]} point(s) are numerically "
            "indistinguishable from others and left out of the triangulation "
            "(near-duplicates); merge them first."
        )
    return jnp.asarray(X), Xh, tri


def _simplex_edges(simplices: np.ndarray, n: int):
    """The edges of the Delaunay simplices, via the mesh triangle helper.

    A tetrahedron contributes its four faces; repeated edges are merged.
    """
    if simplices.shape[1] == 4:
        simplices = np.concatenate([simplices[:, face] for face in _TET_FACES])
    return _triangle_edges(np.asarray(simplices), n)


def _blocked(
    Xh: np.ndarray,
    s: np.ndarray,
    r: np.ndarray,
    tri,
    rule: Literal["gabriel", "rng"],
) -> np.ndarray:
    """Whether a Delaunay neighbour of either endpoint blocks each edge.

    Gabriel: $d_{ik}^2 + d_{jk}^2 < d_{ij}^2$; RNG:
    $\\max(d_{ik}^2, d_{jk}^2) < d_{ij}^2$. Host-side, in float64.
    """
    indptr, indices = tri.vertex_neighbor_vertices
    n_edges = s.shape[0]
    # Candidate pairs (edge, k): every neighbour k of the sender, then of the
    # receiver (CSR rows gathered without a Python loop).
    ends = np.concatenate([s, r])
    edge = np.concatenate([np.arange(n_edges), np.arange(n_edges)])
    counts = indptr[ends + 1] - indptr[ends]
    e = np.repeat(edge, counts)
    offset = np.arange(counts.sum()) - np.repeat(np.cumsum(counts) - counts, counts)
    k = indices[np.repeat(indptr[ends], counts) + offset]
    i, j = s[e], r[e]
    other = (k != i) & (k != j)
    e, i, j, k = e[other], i[other], j[other], k[other]
    d_ij = einx.sum("m [d]", (Xh[i] - Xh[j]) ** 2)
    d_ik = einx.sum("m [d]", (Xh[i] - Xh[k]) ** 2)
    d_jk = einx.sum("m [d]", (Xh[j] - Xh[k]) ** 2)
    inside = d_ik + d_jk < d_ij if rule == "gabriel" else np.maximum(d_ik, d_jk) < d_ij
    return np.bincount(e[inside], minlength=n_edges) > 0


def _weighted(
    X: Array,
    Xh: np.ndarray,
    s: np.ndarray,
    r: np.ndarray,
    weighting: Weighting,
    bandwidth: Bandwidth,
) -> Graph:
    """Route edges through `graph_from_edges`, weighting their lengths.

    The lengths are computed on the host (``X`` is concrete anyway), which
    spares a gather compiled for every new edge count.
    """
    d = np.sqrt(einx.sum("e [d]", (Xh[s] - Xh[r]) ** 2))
    dtype = X.dtype if jnp.issubdtype(X.dtype, jnp.inexact) else None
    return graph_from_edges(
        s,
        r,
        Xh.shape[0],
        distances=jnp.asarray(d, dtype=dtype),
        weighting=weighting,
        bandwidth=bandwidth,
    )
