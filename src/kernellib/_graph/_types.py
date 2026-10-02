r"""Sparse graph types: a static topology with traced edge weights.

A weighted, undirected graph stores each edge **once**, as a pair
``senders[e] < receivers[e]`` with weight ``w_e``. Its Laplacian is the
matrix of the Dirichlet energy,

$$
f^\top L f = \sum_{e=(i,j)} w_e (f_i - f_j)^2 = \|B f\|^2,
\qquad L = B^\top B,\quad B_{e,:} = \sqrt{w_e}\,(e_i - e_j)^\top,
$$

so $L$ is positive semidefinite and the incidence matrix $B$ falls out of the
edge list directly.

The topology is host-side NumPy and static under ``jax.jit``, like gaussx's
`gaussx.SparsityPattern`: the sparsity pattern of every operator is derived
from it once, and only the weights are traced. A symbolic sparse Cholesky of
the Laplacian is therefore reused across every reweighting, ``vmap`` and
hyperparameter value.
"""

from __future__ import annotations

import functools as ft
import hashlib
import itertools
import math
from typing import Any, Literal

import einx
import equinox as eqx
import gaussx as gx
import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
from jax.experimental import sparse as jsparse
from jaxtyping import Array, ArrayLike, Float, Int

from kernellib._einx import einsum, reduce
from kernellib._graph._laplacian import Normalization


__all__ = ["AbstractGraph", "Graph", "GraphTopology", "GridGraph"]

_NORMALIZATIONS = ("unnormalized", "symmetric", "random_walk")


class GraphTopology:
    r"""Static, hashable edge list of an undirected graph without self-loops.

    Each edge is stored once, smaller node index first
    (``senders[e] < receivers[e]``). The index arrays live on the host as
    read-only ``int32`` NumPy arrays, so a topology is a compile-time constant
    under ``jax.jit``: it is a static field of `Graph`, and the sparsity
    patterns of the graph's operators are derived from it once and cached on
    it.

    The hash is a content hash of the index arrays and ``n_nodes``, so two
    topologies with the same edges in the same order compare equal.

    Args:
        senders: Smaller endpoint of each edge, shape ``(E,)``.
        receivers: Larger endpoint of each edge, shape ``(E,)``.
        n_nodes: Number of nodes ``N``; nodes without edges are allowed.

    Raises:
        TypeError: If the indices are traced or not integers.
        ValueError: If the arrays are not rank 1 of equal length, an index is
            out of range, an edge has ``senders[e] >= receivers[e]`` (a
            reversed pair or a self-loop), or an edge is repeated.

    Examples:
        >>> import kernellib as kl
        >>> top = kl.GraphTopology([0, 1], [1, 2], n_nodes=3)  # path 0 - 1 - 2
        >>> top
        GraphTopology(n_nodes=3, n_edges=2)
        >>> top == kl.GraphTopology([0, 1], [1, 2], n_nodes=3)
        True
    """

    senders: np.ndarray
    receivers: np.ndarray
    n_nodes: int
    _digest: str

    def __init__(self, senders: ArrayLike, receivers: ArrayLike, n_nodes: int) -> None:
        try:
            s = np.asarray(senders)
            r = np.asarray(receivers)
        except jax.errors.TracerArrayConversionError as err:
            raise TypeError(
                "Graph topology must be concrete host arrays; only the edge "
                "weights may be traced."
            ) from err
        if s.ndim != 1 or r.ndim != 1 or s.shape != r.shape:
            raise ValueError(
                "senders and receivers must be rank-1 arrays of equal length, "
                f"got {s.shape} and {r.shape}."
            )
        if s.size and not (
            np.issubdtype(s.dtype, np.integer) and np.issubdtype(r.dtype, np.integer)
        ):
            raise TypeError(f"Indices must be integers, got {s.dtype} and {r.dtype}.")
        n_nodes = int(n_nodes)
        if n_nodes < 0:
            raise ValueError(f"n_nodes must be non-negative, got {n_nodes}.")
        s = s.astype(np.int64)
        r = r.astype(np.int64)
        if s.size:
            if s.min() < 0 or r.max() >= n_nodes:
                raise ValueError(f"Node indices out of range for n_nodes={n_nodes}.")
            if np.any(s >= r):
                raise ValueError(
                    "Each edge must be stored once with senders < receivers "
                    "(no reversed pairs, no self-loops)."
                )
            if np.unique(s * n_nodes + r).size != s.size:
                raise ValueError("Each edge must be stored once; found duplicates.")
        s = np.ascontiguousarray(s, dtype=np.int32)
        r = np.ascontiguousarray(r, dtype=np.int32)
        s.flags.writeable = False
        r.flags.writeable = False
        digest = hashlib.sha256()
        digest.update(repr(n_nodes).encode())
        digest.update(s.astype("<i4").tobytes())
        digest.update(r.astype("<i4").tobytes())
        object.__setattr__(self, "senders", s)
        object.__setattr__(self, "receivers", r)
        object.__setattr__(self, "n_nodes", n_nodes)
        object.__setattr__(self, "_digest", digest.hexdigest())

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("GraphTopology is immutable.")

    @property
    def n_edges(self) -> int:
        """Number of undirected edges ``E``."""
        return int(self.senders.shape[0])

    def __hash__(self) -> int:
        return int(self._digest[:15], 16)

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if not isinstance(other, GraphTopology):
            return NotImplemented
        return (
            self._digest == other._digest
            and self.n_nodes == other.n_nodes
            and np.array_equal(self.senders, other.senders)
            and np.array_equal(self.receivers, other.receivers)
        )

    def __repr__(self) -> str:
        return f"GraphTopology(n_nodes={self.n_nodes}, n_edges={self.n_edges})"

    # Host-side plans, built once per topology. ``cached_property`` writes to
    # the instance ``__dict__`` directly, so it bypasses ``__setattr__``.

    @ft.cached_property
    def _symmetric_plan(self) -> tuple[gx.SparsityPattern, np.ndarray, np.ndarray]:
        """Lower-triangle ``(N, N)`` pattern and the positions of the edges
        ``(receiver, sender)`` and of the diagonal in it."""
        n = self.n_nodes
        pattern = gx.SparsityPattern(
            self.receivers, self.senders, (n, n), symmetric=True
        )
        nodes = np.arange(n)
        return (
            pattern,
            _positions(pattern, self.receivers, self.senders),
            _positions(pattern, nodes, nodes),
        )

    @ft.cached_property
    def _general_plan(
        self,
    ) -> tuple[gx.SparsityPattern, np.ndarray, np.ndarray, np.ndarray]:
        """Full ``(N, N)`` pattern and the positions of ``(receiver, sender)``,
        ``(sender, receiver)`` and the diagonal in it."""
        n = self.n_nodes
        pattern = gx.SparsityPattern(
            np.concatenate([self.receivers, self.senders]),
            np.concatenate([self.senders, self.receivers]),
            (n, n),
        )
        nodes = np.arange(n)
        return (
            pattern,
            _positions(pattern, self.receivers, self.senders),
            _positions(pattern, self.senders, self.receivers),
            _positions(pattern, nodes, nodes),
        )

    @ft.cached_property
    def _incidence_plan(self) -> tuple[gx.SparsityPattern, np.ndarray, np.ndarray]:
        """``(E, N)`` pattern and the positions of ``(e, sender)`` and
        ``(e, receiver)`` in it."""
        e = np.arange(self.n_edges)
        pattern = gx.SparsityPattern(
            np.concatenate([e, e]),
            np.concatenate([self.senders, self.receivers]),
            (self.n_edges, self.n_nodes),
        )
        return (
            pattern,
            _positions(pattern, e, self.senders),
            _positions(pattern, e, self.receivers),
        )


def _positions(
    pattern: gx.SparsityPattern, rows: np.ndarray, cols: np.ndarray
) -> np.ndarray:
    """Positions of the entries ``(rows, cols)`` in a canonical (row-major
    sorted) pattern; every entry must be present."""
    n_cols = pattern.shape[1]
    keys = pattern.rows.astype(np.int64) * n_cols + pattern.cols
    query = np.asarray(rows, np.int64) * n_cols + np.asarray(cols, np.int64)
    return np.searchsorted(keys, query).astype(np.int32)


class AbstractGraph(eqx.Module):
    r"""Weighted, undirected graph without self-loops.

    A subclass supplies a static `GraphTopology` and one weight per edge; the
    degrees, dense and sparse matrices, operators and the Dirichlet energy are
    shared. Everything is differentiable in the edge weights.

    For symmetric weights $W$ and degrees $D = \mathrm{diag}(W\mathbf 1)$:

    - ``"unnormalized"``: $L = D - W$;
    - ``"symmetric"``: $L = I - D^{-1/2} W D^{-1/2}$, spectrum in $[0, 2]$;
    - ``"random_walk"``: $L = I - D^{-1} W$ (not symmetric).

    Isolated nodes get a zero row in the normalised forms, as in
    `graph_laplacian`.
    """

    n_nodes: eqx.AbstractVar[int]
    topology: eqx.AbstractVar[GraphTopology]
    weights: eqx.AbstractVar[Float[Array, " E"]]

    def edges(
        self,
    ) -> tuple[Int[Array, " E"], Int[Array, " E"], Float[Array, " E"]]:
        """The edge list, each edge once.

        Returns:
            ``(senders, receivers, weights)``, each of shape ``(E,)``, with
            ``senders < receivers``.
        """
        top = self.topology
        return jnp.asarray(top.senders), jnp.asarray(top.receivers), self.weights

    def degree(self) -> Float[Array, " N"]:
        r"""Weighted degrees $d_i = \sum_j W_{ij}$.

        Returns:
            Shape ``(N,)``.
        """
        top = self.topology
        w = self.weights
        n = top.n_nodes
        return jax.ops.segment_sum(w, top.senders, n) + jax.ops.segment_sum(
            w, top.receivers, n
        )

    def to_bcoo(self) -> jsparse.BCOO:
        """The symmetric adjacency matrix as a ``jax.experimental.sparse.BCOO``.

        Returns:
            ``(N, N)`` with both triangles stored and an empty diagonal.
        """
        top = self.topology
        w = self.weights
        indices = np.column_stack(
            [
                np.concatenate([top.senders, top.receivers]),
                np.concatenate([top.receivers, top.senders]),
            ]
        )
        return jsparse.BCOO(
            (jnp.concatenate([w, w]), jnp.asarray(indices, dtype=jnp.int32)),
            shape=(top.n_nodes, top.n_nodes),
            unique_indices=True,
        )

    def to_dense(self) -> Float[Array, "N N"]:
        """The symmetric adjacency matrix as a dense array.

        Returns:
            ``(N, N)``, non-negative for non-negative weights, zero diagonal.
        """
        top = self.topology
        w = self.weights
        n = top.n_nodes
        return (
            jnp.zeros((n, n), w.dtype)
            .at[top.senders, top.receivers]
            .add(w)
            .at[top.receivers, top.senders]
            .add(w)
        )

    def adjacency_operator(self) -> gx.SparseOperator:
        """The adjacency matrix $W$ as a symmetric `gaussx.SparseOperator`.

        Returns:
            ``(N, N)`` operator on the topology's (cached) pattern.
        """
        pattern, edge_pos, _ = self.topology._symmetric_plan
        w = self.weights
        values = jnp.zeros(pattern.nnz, w.dtype).at[edge_pos].set(w)
        return gx.SparseOperator(values, pattern)

    def laplacian_operator(
        self, normalization: Normalization = "unnormalized"
    ) -> lx.AbstractLinearOperator:
        """The graph Laplacian as a sparse operator.

        The pattern comes from the topology and is built once, so the
        operator's symbolic structure is shared by every reweighting.

        Args:
            normalization: ``"unnormalized"``, ``"symmetric"`` or
                ``"random_walk"``.

        Returns:
            A `gaussx.SparseOperator`, tagged symmetric and positive
            semidefinite except for ``"random_walk"``, which carries no tag.

        Raises:
            ValueError: For an unknown ``normalization``.
        """
        _check_normalization(normalization)
        top = self.topology
        w = self.weights
        degree = self.degree()
        if normalization == "unnormalized":
            pattern, edge_pos, diag_pos = top._symmetric_plan
            values = (
                jnp.zeros(pattern.nnz, w.dtype)
                .at[diag_pos]
                .set(degree)
                .at[edge_pos]
                .set(-w)
            )
            return gx.SparseOperator(values, pattern, tags=lx.positive_semidefinite_tag)
        safe = jnp.where(degree > 0, degree, 1.0)
        connected = (degree > 0).astype(w.dtype)
        if normalization == "symmetric":
            pattern, edge_pos, diag_pos = top._symmetric_plan
            d_isqrt = connected / jnp.sqrt(safe)
            off = -w * d_isqrt[top.senders] * d_isqrt[top.receivers]
            values = (
                jnp.zeros(pattern.nnz, w.dtype)
                .at[diag_pos]
                .set(connected)
                .at[edge_pos]
                .set(off)
            )
            return gx.SparseOperator(values, pattern, tags=lx.positive_semidefinite_tag)
        pattern, pos_rs, pos_sr, diag_pos = top._general_plan
        d_inv = connected / safe
        values = (
            jnp.zeros(pattern.nnz, w.dtype)
            .at[diag_pos]
            .set(connected)
            .at[pos_rs]
            .set(-w * d_inv[top.receivers])
            .at[pos_sr]
            .set(-w * d_inv[top.senders])
        )
        return gx.SparseOperator(values, pattern)

    def incidence_operator(self) -> gx.SparseOperator:
        r"""The weighted incidence matrix $B$, with $B^\top B = L$.

        Row $e$ of an edge $(i, j)$, $i < j$, is $\sqrt{w_e}\,(e_i - e_j)^\top$.
        It is the precision factor of an intrinsic CAR prior: $\|Bf\|^2$ is
        the Dirichlet energy. The gradient in a weight is infinite where that
        weight is zero, through the square root.

        Returns:
            ``(E, N)`` `gaussx.SparseOperator`.
        """
        pattern, pos_s, pos_r = self.topology._incidence_plan
        root = jnp.sqrt(self.weights)
        values = (
            jnp.zeros(pattern.nnz, root.dtype).at[pos_s].set(root).at[pos_r].set(-root)
        )
        return gx.SparseOperator(values, pattern)

    def dirichlet_energy(self, f: Float[Array, "N ..."]) -> Float[Array, ...]:
        r"""The Dirichlet energy $f^\top L f = \sum_e w_e (f_i - f_j)^2$.

        Args:
            f: Node values, shape ``(N,)`` or ``(N, ...)`` for several signals.

        Returns:
            The energy of each signal, shape ``f.shape[1:]``.
        """
        top = self.topology
        f = jnp.asarray(f)
        diff = f[top.senders] - f[top.receivers]
        return einsum(self.weights, diff**2, "e, e ... -> ...")

    def reweight(self, weights: Float[ArrayLike, " E"]) -> Graph:
        """The same topology with new edge weights.

        Args:
            weights: One weight per edge, in the order of `edges`.

        Returns:
            A `Graph`.
        """
        return Graph(self.topology, weights)


def _check_normalization(normalization: str) -> None:
    if normalization not in _NORMALIZATIONS:
        raise ValueError(
            "normalization must be 'unnormalized', 'symmetric' or 'random_walk', "
            f"got {normalization!r}."
        )


class Graph(AbstractGraph):
    r"""Sparse weighted graph: a static topology and traced edge weights.

    The weights are the only pytree leaf, so ``jit``, ``grad`` and ``vmap``
    act on them while the topology stays a compile-time constant.

    Args:
        topology: The edge list, each edge once.
        weights: Non-negative edge weights, shape ``(E,)``.

    Raises:
        ValueError: If ``weights`` does not have one entry per edge.

    Examples:
        >>> import jax.numpy as jnp
        >>> import kernellib as kl
        >>> g = kl.Graph(kl.GraphTopology([0, 1], [1, 2], 3), jnp.array([1.0, 2.0]))
        >>> g.degree().tolist()
        [1.0, 3.0, 2.0]
        >>> g.laplacian_operator().as_matrix().tolist()
        [[1.0, -1.0, 0.0], [-1.0, 3.0, -2.0], [0.0, -2.0, 2.0]]
        >>> g.dirichlet_energy(jnp.array([0.0, 1.0, 3.0])).item()  # 1*1 + 2*4
        9.0
    """

    topology: GraphTopology = eqx.field(static=True)
    weights: Float[Array, " E"]

    def __init__(self, topology: GraphTopology, weights: Float[ArrayLike, " E"]):
        weights = jnp.asarray(weights)
        if weights.shape != (topology.n_edges,):
            raise ValueError(
                f"weights must have shape ({topology.n_edges},) to match the "
                f"topology, got {weights.shape}."
            )
        if not jnp.issubdtype(weights.dtype, jnp.inexact):
            weights = weights.astype(jnp.result_type(weights.dtype, jnp.float32))
        self.topology = topology
        self.weights = weights

    @property
    def n_nodes(self) -> int:
        """Number of nodes ``N``."""
        return self.topology.n_nodes


class GridGraph(AbstractGraph):
    r"""Regular lattice graph that never stores its edges.

    Nodes are the cells of an array of shape ``shape``, numbered in
    **row-major (C) order**, matching ``rearrange("h w c -> (h w) c", image)``.
    (The original MATLAB and Python eigenmap code used column-major order.)

    - ``connectivity="face"`` joins cells that differ by one step along one
      axis (4 neighbours in 2-D, 6 in 3-D); ``"full"`` also joins diagonal
      neighbours (8 in 2-D, 26 in 3-D).
    - ``periodic`` wraps an axis around, per axis: a global latitude-longitude
      raster is ``periodic=(False, True)``. A ``bool`` applies to every axis.
      A periodic axis needs at least 3 cells.
    - An edge along axis ``k`` has weight ``axis_weights[k]`` (anisotropic
      spacing). A diagonal edge, which steps along several axes, takes the
      mean of their weights.

    The edge list is built on the host only when asked for (and cached per
    lattice). With ``connectivity="face"``, the unnormalised Laplacian is the
    Kronecker sum of one 1-D Laplacian per axis,

    $$L = a_1 L_1 \oplus a_2 L_2 \oplus \cdots,$$

    with $L_k$ the path Laplacian (free boundary) or the cycle Laplacian
    (periodic) and $a_k$ the axis weight. That is exact, border degrees
    included, and `laplacian_operator` returns it as a nested
    `gaussx.KroneckerSum`; every other combination is a sparse operator.

    Args:
        shape: Lattice shape, e.g. ``(H, W)``.
        connectivity: ``"face"`` or ``"full"``.
        periodic: Wrap-around, per axis or for all axes.
        axis_weights: Edge weight per axis, shape ``(len(shape),)``; ones by
            default.

    Raises:
        ValueError: For an empty or non-positive ``shape``, an unknown
            ``connectivity``, a ``periodic`` tuple of the wrong length, a
            periodic axis with fewer than 3 cells, or ``axis_weights`` of the
            wrong shape.

    Examples:
        >>> import jax.numpy as jnp
        >>> import kernellib as kl
        >>> g = kl.GridGraph((2, 3))  # nodes 0 1 2 / 3 4 5
        >>> g.n_nodes, g.topology.n_edges
        (6, 7)
        >>> type(g.laplacian_operator()).__name__
        'KroneckerSum'
        >>> g.degree().tolist()
        [2.0, 3.0, 2.0, 2.0, 3.0, 2.0]
    """

    shape: tuple[int, ...] = eqx.field(static=True)
    connectivity: Literal["face", "full"] = eqx.field(static=True)
    periodic: tuple[bool, ...] = eqx.field(static=True)
    axis_weights: Float[Array, " d"]
    n_nodes: int = eqx.field(static=True)

    def __init__(
        self,
        shape: tuple[int, ...],
        *,
        connectivity: Literal["face", "full"] = "face",
        periodic: bool | tuple[bool, ...] = False,
        axis_weights: Float[ArrayLike, " d"] | None = None,
    ):
        shape = tuple(int(n) for n in shape)
        if not shape or min(shape) < 1:
            raise ValueError(f"shape must be non-empty and positive, got {shape}.")
        if connectivity not in ("face", "full"):
            raise ValueError(
                f"connectivity must be 'face' or 'full', got {connectivity!r}."
            )
        ndim = len(shape)
        if isinstance(periodic, bool | np.bool_):
            periodic = (bool(periodic),) * ndim
        periodic = tuple(bool(p) for p in periodic)
        if len(periodic) != ndim:
            raise ValueError(
                f"periodic must be a bool or have one entry per axis ({ndim}), "
                f"got {periodic}."
            )
        for n, p in zip(shape, periodic, strict=True):
            if p and n < 3:
                raise ValueError(
                    f"A periodic axis needs at least 3 cells, got shape {shape} "
                    f"with periodic={periodic}."
                )
        if axis_weights is None:
            axis_weights = jnp.ones(ndim)
        axis_weights = jnp.asarray(axis_weights)
        if axis_weights.shape != (ndim,):
            raise ValueError(
                f"axis_weights must have shape ({ndim},), got {axis_weights.shape}."
            )
        if not jnp.issubdtype(axis_weights.dtype, jnp.inexact):
            axis_weights = axis_weights.astype(
                jnp.result_type(axis_weights.dtype, jnp.float32)
            )
        self.shape = shape
        self.connectivity = connectivity
        self.periodic = periodic
        self.axis_weights = axis_weights
        self.n_nodes = math.prod(shape)

    @property
    def topology(self) -> GraphTopology:
        """The lattice's edge list, built once per lattice and cached."""
        return _lattice(self.shape, self.connectivity, self.periodic)[0]

    @property
    def weights(self) -> Float[Array, " E"]:
        """One weight per edge, from ``axis_weights``."""
        _, edge_class, class_axes = _lattice(
            self.shape, self.connectivity, self.periodic
        )
        mask = jnp.asarray(class_axes, self.axis_weights.dtype)
        class_weights = einsum(mask, self.axis_weights, "c d, d -> c") / reduce(
            mask, "c d -> c", "sum"
        )
        return class_weights[edge_class]

    def laplacian_operator(
        self, normalization: Normalization = "unnormalized"
    ) -> lx.AbstractLinearOperator:
        """The graph Laplacian.

        With ``connectivity="face"`` and ``"unnormalized"`` this is the nested
        `gaussx.KroneckerSum` of the per-axis 1-D Laplacians (the factor
        itself for a 1-D lattice), with exact eigendecompositions, solves and
        log-determinants from the factors. Otherwise it is the sparse operator
        of `AbstractGraph.laplacian_operator`.

        Args:
            normalization: ``"unnormalized"``, ``"symmetric"`` or
                ``"random_walk"``.

        Returns:
            The Laplacian, as a symmetric positive semidefinite operator
            except for ``"random_walk"``.
        """
        _check_normalization(normalization)
        if self.connectivity != "face" or normalization != "unnormalized":
            return super().laplacian_operator(normalization)
        factors = [
            Graph(
                top,
                jnp.full(top.n_edges, self.axis_weights[k]),
            ).laplacian_operator()
            for k, top in enumerate(
                _lattice((n,), "face", (p,))[0]
                for n, p in zip(self.shape, self.periodic, strict=True)
            )
        ]
        op = factors[-1]
        for factor in reversed(factors[:-1]):
            op = gx.KroneckerSum(factor, op)
        return op


@ft.lru_cache(maxsize=32)
def _lattice(
    shape: tuple[int, ...],
    connectivity: str,
    periodic: tuple[bool, ...],
) -> tuple[GraphTopology, np.ndarray, np.ndarray]:
    """Edge list of a lattice, in row-major node order.

    Returns the topology, each edge's offset class, and the ``(C, d)`` 0/1
    mask of the axes each class steps along. Edges are sorted by
    ``(sender, receiver)``.
    """
    ndim = len(shape)
    steps = (np.array(delta) for delta in itertools.product((-1, 0, 1), repeat=ndim))
    # One of each pair ``±delta``: the first non-zero step is positive.
    offsets = [d for d in steps if np.any(d) and d[np.flatnonzero(d)[0]] > 0]
    if connectivity == "face":
        offsets = [d for d in offsets if np.count_nonzero(d) == 1]
    dims = np.array(shape)
    nodes = np.arange(math.prod(shape))
    coords = np.stack(np.unravel_index(nodes, shape))
    senders, receivers, classes = [], [], []
    for c, delta in enumerate(offsets):
        target = einx.add("d n, d -> d n", coords, delta)
        valid = np.ones(nodes.shape[0], dtype=bool)
        for k in range(ndim):
            if periodic[k]:
                target[k] %= dims[k]
            else:
                valid &= (target[k] >= 0) & (target[k] < dims[k])
        other = np.ravel_multi_index(tuple(target[:, valid]), shape)
        here = nodes[valid]
        senders.append(np.minimum(here, other))
        receivers.append(np.maximum(here, other))
        classes.append(np.full(here.shape[0], c))
    s = np.concatenate(senders) if senders else np.zeros(0, np.int64)
    r = np.concatenate(receivers) if receivers else np.zeros(0, np.int64)
    cls = np.concatenate(classes) if classes else np.zeros(0, np.int64)
    order = np.lexsort((r, s))
    class_axes = (
        np.array([d != 0 for d in offsets], dtype=np.int8)
        if offsets
        else np.zeros((0, ndim), np.int8)
    )
    return (
        GraphTopology(s[order], r[order], int(nodes.shape[0])),
        cls[order].astype(np.int32),
        class_axes,
    )
