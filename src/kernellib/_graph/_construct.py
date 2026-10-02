"""Graph construction: sparse graphs from neighbours, radii, lattices, dense
adjacency matrices and edge lists.

Builders run eagerly, outside ``jax.jit``: the topology is a host-side
`GraphTopology` derived from concrete indices. The weights they produce are
JAX arrays, so everything downstream of a builder is differentiable in them.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import einx
import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp
from jaxtyping import Array, ArrayLike, Float, Int
from scipy.sparse.csgraph import connected_components

from kernellib._einx import einsum, rearrange
from kernellib._graph._neighbors import (
    Backend,
    KNNGraph,
    nearest_neighbors,
    radius_neighbors,
)
from kernellib._graph._types import AbstractGraph, Graph, GraphTopology, GridGraph
from kernellib._graph._weights import (
    Bandwidth,
    EdgeWeigher,
    Weighting,
    edge_weigher,
    heat_weights,
)
from kernellib._kernels import AbstractKernel


__all__ = [
    "adjacency_matrix",
    "edge_weights",
    "graph_from_adjacency",
    "graph_from_edges",
    "graph_from_neighbors",
    "grid_graph",
    "knn_graph",
    "radius_graph",
]

Symmetrize = Literal["max", "min", "mean"]


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

    This is ``graph_from_neighbors(graph, ...).to_dense()``. Under
    ``jax.jit``, where the neighbour indices are traced, it builds the same
    matrix densely instead.

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
    if weighting not in ("heat", "connectivity"):
        raise ValueError(
            f"weighting must be 'heat' or 'connectivity', got {weighting!r}."
        )
    _check_symmetrize(symmetrize)
    if weighting == "connectivity":
        bandwidth = None  # only heat weights have a bandwidth
    try:
        np.asarray(graph.indices)
    except jax.errors.TracerArrayConversionError:
        return _dense_adjacency(graph, weighting, bandwidth, symmetrize)
    return graph_from_neighbors(
        graph, weighting=weighting, bandwidth=bandwidth, symmetrize=symmetrize
    ).to_dense()


def _dense_adjacency(
    graph: KNNGraph,
    weighting: str,
    bandwidth: float | Float[Array, ""] | None,
    symmetrize: str,
) -> Float[Array, "N N"]:
    """The traceable dense construction (traced neighbour indices)."""
    n, k = graph.indices.shape
    if weighting == "heat":
        sigma = jnp.median(graph.distances) if bandwidth is None else bandwidth
        weights = heat_weights(graph.distances, sigma)
    else:
        weights = jnp.ones_like(graph.distances)
    rows = jnp.repeat(jnp.arange(n), k)
    directed = (
        jnp.zeros((n, n), weights.dtype)
        .at[rows, rearrange(graph.indices, "n k -> (n k)")]
        .max(rearrange(weights, "n k -> (n k)"))
    )
    transposed = rearrange(directed, "i j -> j i")
    if symmetrize == "max":
        W = jnp.maximum(directed, transposed)
    elif symmetrize == "min":
        W = jnp.minimum(directed, transposed)
    else:
        W = 0.5 * (directed + transposed)
    return W * (1.0 - jnp.eye(n, dtype=W.dtype))


def graph_from_neighbors(
    knn: KNNGraph,
    *,
    weighting: Weighting = "heat",
    bandwidth: Bandwidth = None,
    symmetrize: Symmetrize = "max",
) -> Graph:
    r"""The sparse, symmetric graph of a k-nearest-neighbour relation.

    Each directed entry ``i -> knn.indices[i, m]`` is weighted from its
    distance, and the two directions of a pair are merged by ``symmetrize``,
    a missing direction counting as weight 0:

    - ``"max"`` keeps an edge if either end lists the other (the union);
    - ``"min"`` keeps it only if both do (mutual neighbours);
    - ``"mean"`` averages the two directed weights.

    Padding entries (index ``-1``, from `radius_neighbors`) and self-pairs are
    skipped.

    Args:
        knn: From `nearest_neighbors` or `radius_neighbors`.
        weighting: ``"heat"`` ($\exp(-d^2/2\sigma^2)$), ``"connectivity"``
            (1), or an isotropic stationary kernel evaluated on the distance.
            ``"cosine"`` and other kernels need the points: use `knn_graph`.
        bandwidth: For ``"heat"``: ``None`` / ``"median"`` (the median
            neighbour distance), ``"local"`` (self-tuning
            $\exp(-d_{ij}^2/\sigma_i\sigma_j)$, $\sigma_i$ the distance to the
            farthest listed neighbour of ``i``) or a fixed $\sigma$.
        symmetrize: ``"max"``, ``"min"`` or ``"mean"``.

    Returns:
        A `Graph` on ``knn.n_points`` nodes.

    Raises:
        TypeError: If the indices are traced (builders run eagerly).
        ValueError: For an invalid option.

    Examples:
        >>> import jax.numpy as jnp
        >>> import kernellib as kl
        >>> X = jnp.array([[0.0], [1.0], [3.0], [7.0]])
        >>> knn = kl.nearest_neighbors(X, 1)  # 0->1, 1->0, 2->1, 3->2
        >>> g = kl.graph_from_neighbors(knn, weighting="connectivity")
        >>> g.topology.senders.tolist(), g.topology.receivers.tolist()
        ([0, 1, 2], [1, 2, 3])
        >>> kl.graph_from_neighbors(knn, symmetrize="min").topology.n_edges
        1
    """
    return _from_neighbors(knn, None, weighting, bandwidth, symmetrize)


def knn_graph(
    X: Float[Array, "N D"],
    n_neighbors: int,
    *,
    weighting: Weighting = "heat",
    bandwidth: Bandwidth = None,
    symmetrize: Symmetrize = "max",
    backend: Backend = "exact",
    random_state: int | None = None,
    ensure_connected: bool = False,
) -> Graph:
    r"""The k-nearest-neighbour graph of ``X``.

    `nearest_neighbors` followed by `graph_from_neighbors`, with the points
    available, so ``weighting`` may also be ``"cosine"`` (the endpoints'
    cosine similarity, clipped at 0) or any kernellib kernel, evaluated as
    $k(x_i, x_j)$ on each edge.

    With ``ensure_connected=True``, a graph that falls apart into several
    components is joined up: each component gets the shortest edge from it
    to any other point (Borůvka's step), until one component remains. That
    takes at most $\lceil\log_2 c\rceil$ rounds for $c$ components. Each
    added edge is the shortest across a cut, so it is an edge of the
    Euclidean minimum spanning tree, and it is weighted like every other
    edge (the median bandwidth is that of the k-NN distances alone).

    Args:
        X: Points, shape ``(N, D)``.
        n_neighbors: Neighbours per point.
        weighting: ``"heat"``, ``"connectivity"``, ``"cosine"`` or a kernel.
        bandwidth: See `graph_from_neighbors`.
        symmetrize: ``"max"``, ``"min"`` or ``"mean"``.
        backend: Neighbour search, see `nearest_neighbors`.
        random_state: Seed for ``backend="pynndescent"``.
        ensure_connected: Add bridge edges until the graph is connected.

    Returns:
        A `Graph` on ``N`` nodes.

    Examples:
        >>> import jax.numpy as jnp
        >>> import kernellib as kl
        >>> X = jnp.array([[0.0], [1.0], [10.0], [11.0]])
        >>> kl.knn_graph(X, 1).topology.n_edges  # two separate pairs
        2
        >>> g = kl.knn_graph(X, 1, ensure_connected=True)
        >>> g.topology.senders.tolist(), g.topology.receivers.tolist()
        ([0, 1, 2], [1, 2, 3])
    """
    X = jnp.asarray(X)
    knn = nearest_neighbors(X, n_neighbors, backend=backend, random_state=random_state)
    graph, weigh = _from_neighbors(
        knn, X, weighting, bandwidth, symmetrize, return_weigher=True
    )
    if ensure_connected:
        graph = _connect(graph, X, weigh)
    return graph


def radius_graph(
    X: Float[Array, "N D"],
    radius: float,
    *,
    max_neighbors: int,
    weighting: Weighting = "heat",
    bandwidth: Bandwidth = None,
    backend: Backend = "exact",
    random_state: int | None = None,
) -> Graph:
    """The graph joining points within ``radius`` of each other.

    Shapes in JAX are static, so this runs a k-NN search with
    ``max_neighbors`` (`radius_neighbors`) and then drops the longer edges. A
    point with more than ``max_neighbors`` others within ``radius`` keeps
    only its nearest ones (though a farther pair is still joined if the other
    end lists it). The pruning makes the edge count data-dependent, so this
    function is eager and cannot be ``jit``-ted.

    Args:
        X: Points, shape ``(N, D)``.
        radius: Largest edge length (inclusive).
        max_neighbors: Neighbours searched per point, ``< N``.
        weighting: ``"heat"``, ``"connectivity"``, ``"cosine"`` or a kernel.
        bandwidth: See `graph_from_neighbors`; the median is over the kept
            edges.
        backend: Neighbour search, see `nearest_neighbors`.
        random_state: Seed for ``backend="pynndescent"``.

    Returns:
        A `Graph` on ``N`` nodes.

    Examples:
        >>> import jax.numpy as jnp
        >>> import kernellib as kl
        >>> X = jnp.array([[0.0], [1.0], [1.5], [9.0]])
        >>> g = kl.radius_graph(X, 1.0, max_neighbors=2, weighting="connectivity")
        >>> g.topology.senders.tolist(), g.topology.receivers.tolist()
        ([0, 1], [1, 2])
    """
    X = jnp.asarray(X)
    knn = radius_neighbors(
        X,
        radius,
        max_neighbors=max_neighbors,
        backend=backend,
        random_state=random_state,
    )
    return _from_neighbors(knn, X, weighting, bandwidth, "max")


def grid_graph(
    shape: Sequence[int],
    *,
    connectivity: Literal["face", "full"] = "face",
    periodic: bool | tuple[bool, ...] = False,
    spacing: float | Sequence[float] | Float[Array, " d"] | None = None,
) -> GridGraph:
    r"""The lattice graph of an array of shape ``shape``.

    Nodes are numbered in row-major (C) order, so the node values of an image
    are ``rearrange("h w -> (h w)", image)``. The edge list is never stored;
    see `GridGraph`.

    ``spacing`` $h_k$ sets the edge weight along axis $k$ to $1/h_k^2$, so
    the unnormalised Laplacian is the finite-difference approximation of the
    negative continuous Laplacian $-\Delta$ on that grid.

    Args:
        shape: Lattice shape, e.g. ``image.shape[:2]``.
        connectivity: ``"face"`` (4 / 6 neighbours) or ``"full"`` (8 / 26).
        periodic: Wrap-around, per axis (``(False, True)`` for a global
            latitude-longitude raster) or for all axes.
        spacing: Grid spacing, a scalar or one per axis; ``None`` for unit
            spacing (unit weights).

    Returns:
        A `GridGraph`.

    Raises:
        ValueError: For a non-positive ``spacing`` or the wrong number of
            entries, or an invalid lattice (see `GridGraph`).

    Examples:
        >>> import kernellib as kl
        >>> g = kl.grid_graph((3, 4), spacing=(1.0, 0.5))
        >>> g.axis_weights.tolist()
        [1.0, 4.0]
        >>> g.n_nodes
        12
    """
    shape = tuple(int(n) for n in shape)
    if spacing is None:
        return GridGraph(shape, connectivity=connectivity, periodic=periodic)
    h = jnp.asarray(spacing)
    if h.ndim == 0:
        h = jnp.full(len(shape), h)
    if h.shape != (len(shape),):
        raise ValueError(
            f"spacing must be a scalar or have one entry per axis ({len(shape)}), "
            f"got shape {h.shape}."
        )
    if not np.all(np.asarray(h) > 0):
        raise ValueError(f"spacing must be positive, got {np.asarray(h).tolist()}.")
    if not jnp.issubdtype(h.dtype, jnp.inexact):
        h = h.astype(jnp.result_type(h.dtype, jnp.float32))
    return GridGraph(
        shape, connectivity=connectivity, periodic=periodic, axis_weights=1.0 / h**2
    )


def graph_from_adjacency(W: Float[ArrayLike, "N N"], *, atol: float = 0.0) -> Graph:
    """The sparse graph of a dense, symmetric adjacency matrix.

    An edge is kept where ``|W_ij| > atol``; the diagonal (self-loops) is
    ignored. For large graphs, prefer the sparse builders: this reads the
    whole ``N x N`` matrix on the host.

    Args:
        W: Symmetric adjacency matrix, shape ``(N, N)``.
        atol: Entries with magnitude at most this are not edges.

    Returns:
        A `Graph`, with weights taken from ``W``.

    Raises:
        TypeError: If ``W`` is traced (builders run eagerly).
        ValueError: If ``W`` is not square and symmetric.

    Examples:
        >>> import jax.numpy as jnp
        >>> import kernellib as kl
        >>> W = jnp.array([[0.0, 2.0, 0.0], [2.0, 0.0, 1e-9], [0.0, 1e-9, 0.0]])
        >>> kl.graph_from_adjacency(W, atol=1e-6).topology.n_edges
        1
    """
    W_host = _concrete(W, "W")
    if W_host.ndim != 2 or W_host.shape[0] != W_host.shape[1]:
        raise ValueError(f"W must be square, got shape {W_host.shape}.")
    if not np.allclose(W_host, W_host.T):
        raise ValueError("W must be symmetric.")
    s, r = np.nonzero(np.triu(np.abs(W_host) > atol, k=1))
    W = jnp.asarray(W)
    if not jnp.issubdtype(W.dtype, jnp.inexact):
        W = W.astype(jnp.result_type(W.dtype, jnp.float32))
    return Graph(GraphTopology(s, r, W_host.shape[0]), W[s, r])


def graph_from_edges(
    senders: Int[ArrayLike, " E"],
    receivers: Int[ArrayLike, " E"],
    n_nodes: int,
    *,
    weights: Float[ArrayLike, " E"] | None = None,
    distances: Float[ArrayLike, " E"] | None = None,
    weighting: Weighting = "connectivity",
    bandwidth: float | Float[Array, ""] | Literal["median"] | None = None,
    symmetrize: Symmetrize = "max",
) -> Graph:
    r"""The sparse graph of an edge list: the seam for other graph libraries.

    Any tool that produces an edge list (city2graph, libpysal contiguity,
    NetworkX, OpenStreetMap road networks) hands over plain integer arrays;
    kernellib never imports those libraries. The pairs may come in either
    order and more than once: every occurrence of the pair $\{i, j\}$ is
    merged by ``symmetrize`` (the max, min or mean of their weights).
    Self-loops are dropped.

    **Distances are not weights.** A road network's ``weight`` column is
    usually a length, and using it as an affinity inverts the Laplacian: far
    neighbours would couple most strongly. Pass lengths as ``distances`` and
    let ``weighting`` convert them: ``"heat"`` gives $\exp(-d^2/2\sigma^2)$,
    an isotropic stationary kernel is evaluated on $d$, and
    ``"connectivity"`` ignores them.

    Args:
        senders: One endpoint of each edge, shape ``(E,)``.
        receivers: The other endpoint, shape ``(E,)``.
        n_nodes: Number of nodes.
        weights: Affinities, used as given. Excludes ``distances``.
        distances: Edge lengths, converted by ``weighting``.
        weighting: ``"connectivity"``, ``"heat"`` or a stationary kernel;
            applies to ``distances`` (without them, only
            ``"connectivity"``).
        bandwidth: Heat $\sigma$; ``None`` / ``"median"`` uses the median
            distance.
        symmetrize: ``"max"``, ``"min"`` or ``"mean"`` over the occurrences of
            a pair.

    Returns:
        A `Graph` on ``n_nodes`` nodes.

    Raises:
        TypeError: If the indices are traced or not integers.
        ValueError: For out-of-range indices, mismatched lengths, both
            ``weights`` and ``distances``, or a ``weighting`` that cannot
            apply.

    Examples:
        >>> import jax.numpy as jnp
        >>> import kernellib as kl
        >>> g = kl.graph_from_edges(
        ...     [0, 1, 2, 1],
        ...     [1, 0, 1, 1],
        ...     3,
        ...     weights=jnp.array([1.0, 3.0, 2.0, 5.0]),
        ... )  # (0, 1) twice, a self-loop on 1
        >>> g.topology.senders.tolist(), g.topology.receivers.tolist()
        ([0, 1], [1, 2])
        >>> g.weights.tolist()
        [3.0, 2.0]
    """
    s = _concrete(senders, "senders")
    r = _concrete(receivers, "receivers")
    n_nodes = int(n_nodes)
    if s.ndim != 1 or r.ndim != 1 or s.shape != r.shape:
        raise ValueError(
            "senders and receivers must be rank-1 arrays of equal length, "
            f"got {s.shape} and {r.shape}."
        )
    if s.size and not (
        np.issubdtype(s.dtype, np.integer) and np.issubdtype(r.dtype, np.integer)
    ):
        raise TypeError(f"Indices must be integers, got {s.dtype} and {r.dtype}.")
    if s.size and (min(s.min(), r.min()) < 0 or max(s.max(), r.max()) >= n_nodes):
        raise ValueError(f"Node indices out of range for n_nodes={n_nodes}.")
    _check_symmetrize(symmetrize)
    if weights is not None and distances is not None:
        raise ValueError("Pass weights or distances, not both.")
    if weights is not None:
        if not (isinstance(weighting, str) and weighting == "connectivity"):
            raise ValueError("weighting converts distances; weights are used as given.")
        w = _edge_array(weights, s.shape, "weights")
    elif distances is not None:
        d = _edge_array(distances, s.shape, "distances")
        if isinstance(bandwidth, str) and bandwidth == "local":
            raise ValueError("bandwidth='local' needs k-nearest-neighbour distances.")
        keep = s != r
        weigh = edge_weigher(weighting, bandwidth, distances=d[np.flatnonzero(keep)])
        w = weigh(s, r, d)
    else:
        if not (isinstance(weighting, str) and weighting == "connectivity"):
            raise ValueError(f"weighting={weighting!r} needs distances.")
        w = jnp.ones(s.shape)
    keep = np.flatnonzero(s != r)
    s, r, w = s[keep], r[keep], w[keep]
    a, b = np.minimum(s, r), np.maximum(s, r)
    keys, pair = np.unique(a.astype(np.int64) * n_nodes + b, return_inverse=True)
    pair = pair.ravel()
    n_edges = keys.shape[0]
    if symmetrize == "max":
        merged = jax.ops.segment_max(w, pair, n_edges)
    elif symmetrize == "min":
        merged = jax.ops.segment_min(w, pair, n_edges)
    else:
        counts = np.bincount(pair, minlength=n_edges)
        merged = jax.ops.segment_sum(w, pair, n_edges) / counts
    topology = GraphTopology(keys // n_nodes, keys % n_nodes, n_nodes)
    return Graph(topology, merged)


def edge_weights(
    graph: AbstractGraph, X: Float[Array, "N D"], kernel: AbstractKernel
) -> Graph:
    r"""Reweight a graph's edges by a kernel on the nodes' features.

    $w_{ij} = k(x_i, x_j)$ on every edge of ``graph``, the topology
    unchanged. A lattice weighted by spectral similarity is Cahill's
    spatial-spectral graph: pixels that are adjacent *and* alike. Only the
    topology is static, so this is ``jit``-able and differentiable in ``X``
    and in the kernel's hyperparameters.

    Args:
        graph: Any graph, e.g. from `grid_graph` or `knn_graph`.
        X: Node features, shape ``(N, D)``.
        kernel: Any kernellib kernel.

    Returns:
        A `Graph` with the same topology.

    Examples:
        >>> import jax.numpy as jnp
        >>> import kernellib as kl
        >>> pixels = jnp.array([[0.0], [0.0], [3.0], [3.0]])  # 2 x 2 image, 1 band
        >>> g = kl.edge_weights(kl.grid_graph((2, 2)), pixels, kl.RBF())
        >>> [round(w, 4) for w in g.weights.tolist()]  # edges 0-1, 0-2, 1-3, 2-3
        [1.0, 0.0111, 0.0111, 1.0]
    """
    top = graph.topology
    X = jnp.asarray(X)
    if X.shape[0] != top.n_nodes:
        raise ValueError(
            f"X must have one row per node ({top.n_nodes}), got {X.shape[0]}."
        )
    return Graph(top, kernel.elwise(X[top.senders], X[top.receivers]))


# -- helpers -----------------------------------------------------------------


def _from_neighbors(
    knn: KNNGraph,
    X: Float[Array, "N D"] | None,
    weighting: Weighting,
    bandwidth: Bandwidth,
    symmetrize: Symmetrize,
    *,
    return_weigher: bool = False,
):
    _check_symmetrize(symmetrize)
    idx = _concrete(knn.indices, "knn.indices")
    n, k = idx.shape
    rows = np.repeat(np.arange(n), k)
    cols = idx.ravel()  # host index bookkeeping, row-major like (n k)
    valid = (cols >= 0) & (cols != rows)
    entries = np.flatnonzero(valid)
    rows, cols = rows[entries], cols[entries]
    d = rearrange(jnp.asarray(knn.distances), "n k -> (n k)")[entries]
    local_scale = None
    if isinstance(bandwidth, str) and bandwidth == "local":
        # Distance to the farthest listed neighbour; rows are nearest first.
        n_valid = einx.sum("n [k]", (idx >= 0).astype(np.int64))
        last = jnp.asarray(knn.distances)[np.arange(n), np.maximum(n_valid - 1, 0)]
        local_scale = jnp.where(jnp.asarray(n_valid) > 0, last, 1.0)
    # With no padding, the median is over all distances, exactly as the dense
    # `adjacency_matrix` always computed it.
    median_over = jnp.asarray(knn.distances) if bool(valid.all()) else d
    weigh = edge_weigher(
        weighting, bandwidth, distances=median_over, X=X, local_scale=local_scale
    )
    graph = _symmetrize_directed(rows, cols, weigh(rows, cols, d), n, symmetrize)
    return (graph, weigh) if return_weigher else graph


def _symmetrize_directed(
    rows: np.ndarray,
    cols: np.ndarray,
    w: Float[Array, " M"],
    n: int,
    symmetrize: Symmetrize,
) -> Graph:
    """Merge directed entries ``rows -> cols`` into undirected edges.

    Each pair has a forward slot (``i < j`` listed by ``i``) and a backward
    slot; repeated entries in a slot take the max, and a missing slot is
    weight 0. This mirrors the dense ``max`` / ``min`` / ``mean`` of
    ``W`` and ``Wᵀ`` bit for bit.
    """
    if rows.size == 0:
        return Graph(GraphTopology(rows, cols, n), w)
    a = np.minimum(rows, cols).astype(np.int64)
    b = np.maximum(rows, cols).astype(np.int64)
    keys, pair = np.unique(a * n + b, return_inverse=True)
    pair = pair.ravel()
    n_edges = keys.shape[0]
    slot = 2 * pair + (rows > cols)
    present = np.zeros(2 * n_edges, dtype=bool)
    present[slot] = True
    slots = jax.ops.segment_max(w, slot, 2 * n_edges)
    slots = jnp.where(present, slots, jnp.zeros((), w.dtype))
    slots = rearrange(slots, "(e two) -> two e", e=n_edges, two=2)
    forward, backward = slots[0], slots[1]
    if symmetrize == "max":
        merged = jnp.maximum(forward, backward)
    elif symmetrize == "min":
        mutual = np.flatnonzero(present[0::2] & present[1::2])
        keys, merged = keys[mutual], jnp.minimum(forward, backward)[mutual]
    else:
        merged = 0.5 * (forward + backward)
    return Graph(GraphTopology(keys // n, keys % n, n), merged)


def _connect(graph: Graph, X: Float[Array, "N D"], weigh: EdgeWeigher) -> Graph:
    """Join the components of ``graph`` with Borůvka bridge edges."""
    top = graph.topology
    n = top.n_nodes
    s, r, w = top.senders, top.receivers, graph.weights
    while True:
        adjacency = sp.coo_matrix((np.ones(s.shape[0]), (s, r)), shape=(n, n))
        n_comp, labels = connected_components(adjacency, directed=False)
        if n_comp == 1:
            break
        nearest, dist = _nearest_outside(X, jnp.asarray(labels))
        nearest = np.asarray(nearest)
        dist_host = np.asarray(dist)
        # Per component, the member whose nearest outside point is closest.
        order = np.lexsort((dist_host, labels))
        first = np.flatnonzero(np.r_[True, labels[order][1:] != labels[order][:-1]])
        here = order[first]
        there = nearest[here]
        a, b = np.minimum(here, there), np.maximum(here, there)
        _, unique = np.unique(a.astype(np.int64) * n + b, return_index=True)
        a, b, here = a[unique], b[unique], here[unique]
        s = np.concatenate([s, a])
        r = np.concatenate([r, b])
        w = jnp.concatenate([w, weigh(a, b, dist[here])])
    order = np.lexsort((r, s))
    return Graph(GraphTopology(s[order], r[order], n), w[order])


def _nearest_outside(
    X: Float[Array, "N D"], labels: Int[Array, " N"], batch: int = 1024
) -> tuple[Int[Array, " N"], Float[Array, " N"]]:
    """Each point's nearest point with a different label, and its distance."""
    n = X.shape[0]
    batch = min(batch, n)
    n_blocks = -(-n // batch)
    pad = n_blocks * batch - n
    X_pad = jnp.concatenate([X, jnp.zeros((pad, X.shape[1]), X.dtype)])
    labels_pad = jnp.concatenate([labels, jnp.full(pad, -1, labels.dtype)])
    sq_norms = einsum(X, X, "n d, n d -> n")

    def block(rows: Int[Array, " B"]) -> tuple[Array, Array]:
        Xb = X_pad[rows]
        d2 = einx.add("b, n -> b n", einsum(Xb, Xb, "b d, b d -> b"), sq_norms)
        d2 = jnp.clip(d2 - 2.0 * einsum(Xb, X, "b d, n d -> b n"), min=0.0)
        same = einx.equal("b, n -> b n", labels_pad[rows], labels)
        d2 = jnp.where(same, jnp.inf, d2)
        j = rearrange(einx.argmin("b [n]", d2), "b 1 -> b")
        return j, jnp.sqrt(einx.min("b [n]", d2))

    rows = rearrange(jnp.arange(n_blocks * batch), "(b r) -> b r", r=batch)
    j, d = jax.lax.map(block, rows)
    return rearrange(j, "b r -> (b r)")[:n], rearrange(d, "b r -> (b r)")[:n]


def _concrete(a: ArrayLike, name: str) -> np.ndarray:
    try:
        return np.asarray(a)
    except jax.errors.TracerArrayConversionError as err:
        raise TypeError(
            f"{name} must be concrete: graph builders run eagerly, outside "
            "jax.jit. Only the edge weights of the result are traced."
        ) from err


def _edge_array(values: ArrayLike, shape: tuple[int, ...], name: str) -> Array:
    values = jnp.asarray(values)
    if values.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {values.shape}.")
    if not jnp.issubdtype(values.dtype, jnp.inexact):
        values = values.astype(jnp.result_type(values.dtype, jnp.float32))
    return values


def _check_symmetrize(symmetrize: str) -> None:
    if symmetrize not in ("max", "min", "mean"):
        raise ValueError(
            f"symmetrize must be 'max', 'min' or 'mean', got {symmetrize!r}."
        )
