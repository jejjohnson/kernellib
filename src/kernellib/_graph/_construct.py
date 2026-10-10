"""Graph construction: sparse graphs from neighbours, radii, lattices, dense
adjacency matrices and edge lists.

Builders run eagerly, outside ``jax.jit``: the topology is a host-side
`GraphTopology` derived from concrete indices. The weights they produce are
JAX arrays, so everything downstream of a builder is differentiable in them.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Literal

import einx
import equinox as eqx
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
    Metric,
    _geo_embed,
    _sphere_distance,
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
    "mesh_graph",
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
    metric: Metric = "euclidean",
    radius: float = 1.0,
    degrees: bool = True,
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

    With a geo ``metric`` (``X`` is ``(N, 2)`` ``(lon, lat)``), neighbours,
    edge distances, the heat bandwidth and the bridge edges are all in that
    metric, in units of ``radius``. A kernel ``weighting`` is evaluated on
    the ``(lon, lat)`` points, and ``"cosine"`` on their unit vectors (the
    cosine of the great-circle angle, clipped at 0).

    Args:
        X: Points, shape ``(N, D)``; ``(N, 2)`` ``(lon, lat)`` for a geo
            metric.
        n_neighbors: Neighbours per point.
        weighting: ``"heat"``, ``"connectivity"``, ``"cosine"`` or a kernel.
        bandwidth: See `graph_from_neighbors`.
        symmetrize: ``"max"``, ``"min"`` or ``"mean"``.
        backend: Neighbour search, see `nearest_neighbors`.
        random_state: Seed for ``backend="pynndescent"``.
        ensure_connected: Add bridge edges until the graph is connected.
        metric: ``"euclidean"``, ``"great_circle"`` or ``"chordal"``; see
            `nearest_neighbors`.
        radius: Sphere radius for a geo metric. Ignored for ``"euclidean"``.
        degrees: Whether ``(lon, lat)`` are in degrees. Ignored for
            ``"euclidean"``.

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

        On the sphere, 179° and -179° are neighbours, and the bridge between
        the two pairs is the shortest great-circle arc:

        >>> lonlat = jnp.array(
        ...     [[179.0, 0.0], [-179.0, 0.0], [10.0, 0.0], [12.0, 0.0]]
        ... )
        >>> g = kl.knn_graph(
        ...     lonlat,
        ...     1,
        ...     metric="great_circle",
        ...     ensure_connected=True,
        ...     weighting="connectivity",
        ... )
        >>> g.topology.senders.tolist(), g.topology.receivers.tolist()
        ([0, 0, 2], [1, 3, 3])
    """
    X = jnp.asarray(X)
    knn = nearest_neighbors(
        X,
        n_neighbors,
        backend=backend,
        random_state=random_state,
        metric=metric,
        radius=radius,
        degrees=degrees,
    )
    U = None if metric == "euclidean" else _geo_embed(X, metric, degrees)
    points = U if U is not None and _is_cosine(weighting) else X
    graph, weigh = _from_neighbors(
        knn, points, weighting, bandwidth, symmetrize, return_weigher=True
    )
    if ensure_connected:
        if U is None:
            graph = _connect(graph, X, weigh)
        else:
            graph = _connect(
                graph,
                U,
                weigh,
                lambda a, b: _sphere_distance(U[a], U[b], metric, radius),
            )
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
    metric: Metric = "euclidean",
    sphere_radius: float = 1.0,
    degrees: bool = True,
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
        metric: ``"euclidean"``, ``"great_circle"`` or ``"chordal"``; see
            `radius_neighbors`. ``"cosine"`` weighting then uses the unit
            vectors, as in `knn_graph`.
        sphere_radius: Sphere radius for a geo metric; ``radius`` is in its
            units. Ignored for ``"euclidean"``.
        degrees: Whether ``(lon, lat)`` are in degrees. Ignored for
            ``"euclidean"``.

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
        metric=metric,
        sphere_radius=sphere_radius,
        degrees=degrees,
    )
    if metric != "euclidean" and _is_cosine(weighting):
        X = _geo_embed(X, metric, degrees)
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
    if not np.allclose(W_host, einx.id("i j -> j i", W_host)):
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


def mesh_graph(
    vertices: Float[ArrayLike, "V D"],
    triangles: Int[ArrayLike, "T 3"],
    *,
    weighting: Literal["connectivity", "cotangent"] = "connectivity",
    on_negative: Literal["raise", "clip", "allow"] = "raise",
) -> Graph:
    r"""The edge graph of a triangle mesh.

    Two vertices are joined when they share a triangle edge. With
    ``weighting="cotangent"`` the edge weight is
    $w_{ij} = \tfrac12(\cot\alpha_{ij} + \cot\beta_{ij})$, with
    $\alpha_{ij}, \beta_{ij}$ the angles opposite the edge (one on a boundary
    edge). The graph Laplacian is then the P1 finite-element stiffness
    matrix $G$ (`gaussx.fem_matrices`), the discrete $-\Delta$ on the
    surface. The weights are differentiable in ``vertices``.

    A cotangent weight is negative on an interior edge when
    $\alpha + \beta > \pi$, i.e. the mesh is not Delaunay there, and on a
    boundary edge when its single opposite angle is obtuse. The second case
    is common on a Delaunay mesh of scattered points: a thin triangle on the
    convex hull. $G$ is then still positive semidefinite, but not a graph
    Laplacian with non-negative weights (its incidence matrix would need
    $\sqrt{w_e}$). ``on_negative`` chooses what happens:

    - ``"raise"`` (default): raise, naming the interior (not Delaunay) and
      the boundary (obtuse hull angle) cases separately. Clamping silently
      would change the operator.
    - ``"clip"``: set the negative weights to 0, the usual graph-Laplacian
      fix. The Laplacian is no longer exactly $G$.
    - ``"allow"``: keep the signed weights, so the Laplacian is exactly $G$
      and still positive semidefinite, but some weights are negative. Such a
      `Graph` supports the unnormalised `Graph.laplacian_operator` (and
      `structure_matrix`, `Graph.dirichlet_energy`, `Graph.degree`); the
      operations that need non-negative weights (`Graph.incidence_operator`,
      with its $\sqrt{w_e}$, and the normalised Laplacians) raise, or fail
      an `equinox.error_if` under ``jit``.

    `gaussx.fem_matrices` also holds the signed stiffness, as a
    `gaussx.SparseOperator`. Weights within rounding error of zero (right
    angles on both sides; ``1e3`` machine epsilons of the largest weight) are
    set to 0. Under ``jit``, ``grad`` or ``vmap`` the checks run at run time
    (`equinox.error_if`).

    Args:
        vertices: Vertex coordinates, ``(V, 2)`` or ``(V, 3)`` for a surface.
        triangles: Vertex indices of each triangle, ``(T, 3)`` (concrete).
        weighting: ``"connectivity"`` (1 per edge) or ``"cotangent"``.
        on_negative: What to do with negative cotangent weights:
            ``"raise"``, ``"clip"`` (set to 0) or ``"allow"`` (keep; only
            the unnormalised Laplacian then supports the signed weights).
            Ignored for ``weighting="connectivity"``.

    Returns:
        A `Graph` on the ``V`` vertices.

    Raises:
        ValueError: For malformed ``triangles``, a degenerate triangle, an
            unknown ``on_negative``, or a negative cotangent weight with
            ``on_negative="raise"``.

    Examples:
        >>> import jax.numpy as jnp
        >>> import kernellib as kl
        >>> V = jnp.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        >>> T = jnp.array([[0, 1, 2], [0, 2, 3]])  # a unit square, two halves
        >>> g = kl.mesh_graph(V, T, weighting="cotangent")
        >>> g.laplacian_operator().as_matrix()[0].tolist()
        [1.0, -0.5, 0.0, -0.5]

        A thin triangle on the hull: its boundary edge faces an obtuse angle.

        >>> V = jnp.array([[0.0, 0.0], [2.0, 0.0], [1.0, 0.2]])
        >>> T = jnp.array([[0, 1, 2]])
        >>> g = kl.mesh_graph(V, T, weighting="cotangent", on_negative="clip")
        >>> bool(jnp.all(g.weights >= 0.0))
        True
    """
    if on_negative not in ("raise", "clip", "allow"):
        raise ValueError(
            f"on_negative must be 'raise', 'clip' or 'allow', got {on_negative!r}."
        )
    tri = _concrete(triangles, "triangles")
    vertices = jnp.asarray(vertices)
    n_vertices = vertices.shape[0]
    if tri.ndim != 2 or tri.shape[1] != 3:
        raise ValueError(f"triangles must have shape (T, 3), got {tri.shape}.")
    if tri.size and not np.issubdtype(tri.dtype, np.integer):
        raise TypeError(f"triangles must be integers, got {tri.dtype}.")
    if tri.size and (tri.min() < 0 or tri.max() >= n_vertices):
        raise ValueError(f"Triangle indices out of range for {n_vertices} vertices.")
    topology, pair = _triangle_edges(tri, n_vertices)
    if weighting == "connectivity":
        return Graph(topology, jnp.ones(topology.n_edges, dtype=vertices.dtype))
    if weighting != "cotangent":
        raise ValueError(
            f"weighting must be 'connectivity' or 'cotangent', got {weighting!r}."
        )
    # Corner k of each triangle is opposite the edge (k+1, k+2).
    p = vertices[tri]  # (T, 3, D)
    u = p[:, [1, 2, 0]] - p
    v = p[:, [2, 0, 1]] - p
    dot = einsum(u, v, "t k d, t k d -> t k")
    uu = einsum(u, u, "t k d, t k d -> t k")
    vv = einsum(v, v, "t k d, t k d -> t k")
    area2 = jnp.sqrt(jnp.clip(uu * vv - dot**2, min=0.0))  # |u x v|
    area2 = _check_nondegenerate(area2)
    cot = rearrange(dot / area2, "t k -> (t k)")
    w = 0.5 * jax.ops.segment_sum(cot, pair, topology.n_edges)
    # An edge seen by one triangle corner is on the boundary.
    boundary = np.bincount(pair, minlength=topology.n_edges) == 1
    w = _check_cotangent_weights(w, boundary, on_negative)
    return Graph(topology, w)


def _triangle_edges(
    triangles: np.ndarray, n_vertices: int
) -> tuple[GraphTopology, np.ndarray]:
    """The unique undirected edges of a triangle list, and for each corner
    ``(t, k)`` (row-major) the index of the opposite edge ``(k+1, k+2)``.

    Shared with the proximity graphs built from a Delaunay triangulation.
    """
    a = triangles[:, [1, 2, 0]].ravel().astype(np.int64)
    b = triangles[:, [2, 0, 1]].ravel().astype(np.int64)
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    keys, pair = np.unique(lo * n_vertices + hi, return_inverse=True)
    return (
        GraphTopology(keys // n_vertices, keys % n_vertices, n_vertices),
        pair.ravel(),
    )


_DEGENERATE = "mesh_graph needs non-degenerate triangles (zero area found)."
_NOT_DELAUNAY = (
    "cotangent weight(s) are negative on interior edges: the mesh is not "
    "Delaunay there (the two angles facing an edge sum to more than pi). A "
    "graph needs non-negative weights; use gaussx.fem_matrices for the signed "
    "stiffness matrix, re-mesh, or pass on_negative='clip' or 'allow'."
)
_OBTUSE_BOUNDARY = (
    "cotangent weight(s) are negative on boundary edges: a boundary edge "
    "faces a single obtuse angle (common on the convex hull of a Delaunay "
    "mesh, which is still Delaunay). Pass on_negative='clip' to set them to 0 "
    "or on_negative='allow' to keep the signed FEM stiffness, refine the "
    "boundary, or use weighting='connectivity' (or kl.delaunay_graph with "
    "weighting='heat')."
)


def _check_nondegenerate(area2: Array) -> Array:
    """Raise on a zero-area triangle; under a JAX transform, a run-time check."""
    try:
        host = np.asarray(area2)
    except jax.errors.TracerArrayConversionError:
        return eqx.error_if(area2, jnp.any(area2 <= 0.0), _DEGENERATE)
    if host.size and np.any(host <= 0.0):
        raise ValueError(_DEGENERATE)
    return area2


def _check_cotangent_weights(
    w: Array,
    boundary: np.ndarray,
    on_negative: Literal["raise", "clip", "allow"],
) -> Array:
    """Handle negative weights per ``on_negative`` and zero the ones within
    rounding of 0.

    ``boundary`` marks the edges with a single opposite angle, so a raise can
    tell a non-Delaunay interior edge from an obtuse hull triangle. The
    tolerance scales with the precision of ``w`` and the largest weight, so
    float32 cancellation on a theoretically zero weight (right angles on both
    sides) is not mistaken for a negative one. Under a JAX transform the
    check is a run-time `equinox.error_if`.
    """
    if not w.size:
        return w
    tol = 1e3 * jnp.finfo(w.dtype).eps * jnp.max(jnp.abs(w))
    w = jnp.where(jnp.abs(w) <= tol, 0.0, w)
    if on_negative == "clip":
        return jnp.clip(w, min=0.0)
    if on_negative == "allow":
        return w
    negative = w < 0.0
    interior_neg = negative & ~boundary
    boundary_neg = negative & boundary
    try:
        host_interior = np.asarray(interior_neg)
        host_boundary = np.asarray(boundary_neg)
    except jax.errors.TracerArrayConversionError:
        w = eqx.error_if(w, jnp.any(interior_neg), _NOT_DELAUNAY)
        return eqx.error_if(w, jnp.any(boundary_neg), _OBTUSE_BOUNDARY)
    problems = [
        f"{int(np.sum(host))} {message}"
        for host, message in (
            (host_interior, _NOT_DELAUNAY),
            (host_boundary, _OBTUSE_BOUNDARY),
        )
        if np.any(host)
    ]
    if problems:
        raise ValueError(" ".join(problems))
    return w


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
    cols = einx.id("n k -> (n k)", idx)  # host index bookkeeping
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


def _is_cosine(weighting: Weighting) -> bool:
    return isinstance(weighting, str) and weighting == "cosine"


def _connect(
    graph: Graph,
    X: Float[Array, "N D"],
    weigh: EdgeWeigher,
    distance: Callable[[np.ndarray, np.ndarray], Array] | None = None,
) -> Graph:
    """Join the components of ``graph`` with Borůvka bridge edges.

    The search is Euclidean in ``X``; ``distance(a, b)``, if given, replaces
    the Euclidean length of a bridge (a geo search embeds the points so that
    Euclidean order is the metric's order).
    """
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
        d = dist[here] if distance is None else distance(a, b)
        w = jnp.concatenate([w, weigh(a, b, d)])
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
