r"""Rooted trees and the path geometry of sites on them.

A `RootedTree` is a tree with one root, edge lengths and an additive weight
function. `RootedTree.sites` places points anywhere on its edges and computes,
on the host in NumPy, the pairwise geometry the tail-up / tail-down kernels
(`kernellib.TailUp`, `kernellib.TailDown`) need: path distances, the distances
to the closest common ancestor and the flow-connection weights. The
construction is from Ver Hoef & Peterson (2010) for stream networks; nothing
in it is specific to water.

Terminology (general / stream networks): root / outlet, toward the root /
downstream, toward the leaves / upstream, ancestor--descendant pair /
flow-connected pair, junction (closest common ancestor) / confluence,
additive weights / watershed area, Shreve order.
"""

from __future__ import annotations

from typing import Literal

import einx
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike, Bool, Float, Int

from kernellib._graph._types import AbstractGraph


__all__ = ["RootedTree", "TreeSites"]

TreeWeights = Literal["leaf_count", "uniform"]

_RTOL = 1e-6
_ATOL = 1e-9


def _host(a: ArrayLike, name: str) -> np.ndarray:
    try:
        return np.asarray(a)
    except jax.errors.TracerArrayConversionError as err:
        raise TypeError(
            f"{name} must be concrete: rooted-tree geometry is computed "
            "eagerly on the host, outside jax.jit."
        ) from err


class _Skeleton:
    """Host-side traversal data of a validated parent array."""

    def __init__(self, parent: np.ndarray) -> None:
        n = parent.size
        if n == 0:
            raise ValueError("A rooted tree needs at least one node.")
        if np.any(parent >= n) or np.any(parent < -1):
            raise ValueError("parent indices must lie in [-1, V).")
        roots = np.flatnonzero(parent < 0)
        if roots.size != 1:
            raise ValueError(
                f"A rooted tree has exactly one root (parent = -1), got {roots.size}."
            )
        children: list[list[int]] = [[] for _ in range(n)]
        for v in range(n):
            if parent[v] >= 0:
                children[int(parent[v])].append(v)
        root = int(roots[0])
        order = [root]  # breadth-first: parents before children
        for v in order:
            order.extend(children[v])
        if len(order) != n:
            raise ValueError(
                "parent contains a cycle: some nodes do not reach the root."
            )
        # Pre-order numbering, so a subtree is a contiguous range of `tin`.
        size = np.ones(n, dtype=np.int64)
        for v in reversed(order):
            if parent[v] >= 0:
                size[parent[v]] += size[v]
        tin = np.zeros(n, dtype=np.int64)
        stack, count = [root], 0
        while stack:
            v = stack.pop()
            tin[v] = count
            count += 1
            stack.extend(reversed(children[v]))
        self.n = n
        self.root = root
        self.parent = parent
        self.children = children
        self.order = order
        self.size = size
        self.tin = tin

    def depth(self, length: np.ndarray) -> np.ndarray:
        depth = np.zeros(self.n)
        for v in self.order:
            if self.parent[v] >= 0:
                depth[v] = depth[self.parent[v]] + length[v]
        return depth

    def default_weights(self, kind: str) -> np.ndarray:
        w = np.ones(self.n)
        if kind == "leaf_count":
            counts = np.array([1.0 if not c else 0.0 for c in self.children])
            for v in reversed(self.order):
                if self.parent[v] >= 0:
                    counts[self.parent[v]] += counts[v]
            return counts / counts[self.root]
        if kind == "uniform":
            for v in self.order:
                kids = self.children[v]
                for c in kids:
                    w[c] = w[v] / len(kids)
            return w
        raise ValueError(
            f"weights must be an array, 'leaf_count' or 'uniform', got {kind!r}."
        )

    def check_weights(self, w: np.ndarray) -> None:
        if w.shape != (self.n,):
            raise ValueError(f"weights must have shape ({self.n},), got {w.shape}.")
        if not np.all(np.isfinite(w)) or np.any(w <= 0) or np.any(w > 1 + _ATOL):
            raise ValueError("weights must lie in (0, 1].")
        if not np.isclose(w[self.root], 1.0, rtol=_RTOL, atol=_ATOL):
            raise ValueError(f"The root weight must be 1, got {w[self.root]}.")
        for v in range(self.n):
            kids = self.children[v]
            if kids and not np.isclose(
                w[v], sum(w[c] for c in kids), rtol=_RTOL, atol=_ATOL
            ):
                raise ValueError(
                    "weights are not additive: the weight of node "
                    f"{v} ({w[v]:.6g}) must equal the sum over its children "
                    f"({sum(w[c] for c in kids):.6g})."
                )


def _prepare(
    parent: ArrayLike,
    length: ArrayLike,
    weights: ArrayLike | str,
) -> tuple[_Skeleton, np.ndarray, np.ndarray]:
    p = _host(parent, "parent")
    if p.ndim != 1 or not np.issubdtype(p.dtype, np.integer):
        raise ValueError("parent must be a rank-1 integer array.")
    p = p.astype(np.int64)
    ln = _host(length, "length").astype(np.float64)
    if ln.shape != p.shape:
        raise ValueError(f"length must have shape {p.shape}, got {ln.shape}.")
    skel = _Skeleton(p)
    if not np.all(np.isfinite(ln)) or np.any(ln < 0):
        raise ValueError("length must be finite and non-negative.")
    ln = ln.copy()
    ln[skel.root] = 0.0
    if isinstance(weights, str):
        w = skel.default_weights(weights)
    else:
        w = _host(weights, "weights").astype(np.float64)
        skel.check_weights(w)
    return skel, ln, w


class TreeSites(eqx.Module):
    r"""Pairwise geometry of ``N`` sites on a `RootedTree` (host-computed).

    The sparsity pattern (``connected``) is static; the kernels use the
    values. For sites $s_i, s_j$ with path-to-root depths $t_i, t_j$:

    - **connected** (ancestor--descendant, i.e. flow-connected) pairs: one
      site lies on the other's path to the root, $h_{ij} = |t_i - t_j|$,
      $a_{ij} = 0$, $b_{ij} = h_{ij}$ and
      $\pi_{ij} = \sqrt{\Omega(\text{leafward}) / \Omega(\text{rootward})}$;
    - **other** pairs: $a_{ij} \le b_{ij}$ are the distances from the two
      sites to their closest common ancestor (the junction / confluence),
      $h_{ij} = a_{ij} + b_{ij}$ and $\pi_{ij} = 0$.

    Setting $a = 0$, $b = h$ on connected pairs lets `kernellib.TailDown` use
    one formula in $(a, b)$ for every pair.

    Attributes:
        h: Path distance, shape ``(N, N)``.
        a: Shorter distance to the junction (0 if connected), ``(N, N)``.
        b: Longer distance to the junction (``h`` if connected), ``(N, N)``.
        connected: Ancestor--descendant pairs, ``(N, N)`` bool.
        weight: $\pi_{ij}$ for connected pairs (1 on the diagonal), else 0,
            ``(N, N)``.
    """

    h: Float[Array, "N N"]
    a: Float[Array, "N N"]
    b: Float[Array, "N N"]
    connected: Bool[Array, "N N"]
    weight: Float[Array, "N N"]

    @property
    def n_sites(self) -> int:
        """Number of sites ``N``."""
        return self.h.shape[0]


class RootedTree(eqx.Module):
    r"""A tree with one root: every non-root node has exactly one parent.

    A tree with edge lengths and an additive weight function $\Omega$ is all
    the tail-up / tail-down covariances (Ver Hoef & Peterson, 2010) need. They
    were built for stream networks but apply to any rooted tree (drainage or
    sewer systems, radial feeders, airway or vascular trees, dendrites).
    Hydrology's terms are in the table on the Graphs page: root / outlet,
    toward the root / downstream, ancestor--descendant pair / flow-connected
    pair, junction / confluence, weights / watershed area or Shreve order.

    Node $v$ owns the edge from $v$ to its parent $p(v)$, of length
    ``length[v]``, and the additive weight ``weights[v]`` of that edge. The
    root has no edge: ``parent = -1``, ``length = 0``. Weights satisfy
    $\Omega \in (0, 1]$, $\Omega(\text{root}) = 1$ and
    $\Omega(v) = \sum_{c:\,p(c) = v} \Omega(c)$ at every junction (so they
    are conserved there).

    The structure is computed on the host in NumPy; build it eagerly, outside
    ``jax.jit``. For a graph with loops (braided channels, meshed grids) the
    construction is not valid: use `matern_graph_kernel` or another graph
    kernel.

    Attributes:
        parent: Parent node index, ``-1`` at the root, shape ``(V,)``.
        length: Length of the edge from each node to its parent, ``(V,)``.
        weights: Additive weights $\Omega$, ``(V,)``.

    Examples:
        >>> import kernellib as kl
        >>> #       0 (root)
        >>> #       |  length 2
        >>> #       1 (junction)
        >>> #      / \
        >>> #     2   3   (lengths 3 and 4)
        >>> tree = kl.RootedTree.from_parents([-1, 0, 1, 1], [0.0, 2.0, 3.0, 4.0])
        >>> tree.weights.tolist()  # Shreve order / number of leaves
        [1.0, 1.0, 0.5, 0.5]
        >>> sites = tree.sites([2, 3], [1.0, 1.0])  # one site on each branch
        >>> sites.h.tolist()[0]  # path distance via the junction
        [0.0, 5.0]
        >>> sites.connected.tolist()[0]
        [True, False]
    """

    parent: Int[Array, " V"]
    length: Float[Array, " V"]
    weights: Float[Array, " V"]

    @property
    def n_nodes(self) -> int:
        """Number of nodes ``V``."""
        return self.parent.shape[0]

    @classmethod
    def from_parents(
        cls,
        parent: Int[ArrayLike, " V"],
        length: Float[ArrayLike, " V"],
        weights: Float[ArrayLike, " V"] | TreeWeights = "leaf_count",
    ) -> RootedTree:
        r"""A tree from its parent array.

        Args:
            parent: Parent of each node, ``-1`` for the single root.
            length: Edge length from each node to its parent (the root's entry
                is ignored and set to 0).
            weights: A user array, ``"leaf_count"`` (Shreve order: the number
                of leaves at or below a node over the total number of leaves)
                or ``"uniform"`` (a junction splits its weight equally among
                its children). A user array is checked for additivity,
                $(0, 1]$ and a value of 1 at the root.

        Raises:
            ValueError: For no root or several roots, a cycle, an out-of-range
                parent, negative lengths or invalid weights.
            TypeError: If an input is traced.

        Examples:
            >>> import kernellib as kl
            >>> t = kl.RootedTree.from_parents([-1, 0, 0], [0.0, 1.0, 1.0], "uniform")
            >>> t.weights.tolist()
            [1.0, 0.5, 0.5]
            >>> kl.RootedTree.from_parents(
            ...     [-1, 0, 0], [0.0, 1.0, 1.0], [1.0, 0.7, 0.7]
            ... )  # doctest: +ELLIPSIS
            Traceback (most recent call last):
                ...
            ValueError: weights are not additive: ...
        """
        _, ln, w = _prepare(parent, length, weights)
        p = _host(parent, "parent").astype(np.int64)
        return cls(jnp.asarray(p), jnp.asarray(ln), jnp.asarray(w))

    @classmethod
    def from_graph(
        cls,
        graph: AbstractGraph,
        root: int,
        weights: Float[ArrayLike, " V"] | TreeWeights = "leaf_count",
        *,
        lengths: Float[ArrayLike, " E"] | None = None,
    ) -> RootedTree:
        r"""Orient a tree-shaped graph away from ``root``.

        Args:
            graph: A `Graph` (for example from `graph_from_edges`) whose edges
                form a tree.
            root: The root node.
            weights: As in `from_parents`.
            lengths: Edge lengths in the order of ``graph.edges()``. Default:
                the graph's edge weights (so build the graph with
                ``weights=lengths``). A connectivity graph has weight 1 and
                therefore unit lengths.

        Raises:
            ValueError: If the graph has a cycle (the construction is not
                valid on graphs with loops; use `matern_graph_kernel` or
                another graph kernel) or more than one connected component.

        Examples:
            >>> import kernellib as kl
            >>> g = kl.graph_from_edges(
            ...     [0, 1, 1], [1, 2, 3], 4, weights=[2.0, 3.0, 4.0]
            ... )
            >>> t = kl.RootedTree.from_graph(g, root=0)
            >>> t.parent.tolist(), t.length.tolist()
            ([-1, 0, 1, 1], [0.0, 2.0, 3.0, 4.0])
            >>> loop = kl.graph_from_edges([0, 1, 2], [1, 2, 0], 3)
            >>> kl.RootedTree.from_graph(loop, root=0)  # doctest: +ELLIPSIS
            Traceback (most recent call last):
                ...
            ValueError: The graph contains a cycle; ...
        """
        n = graph.n_nodes
        root = int(root)
        if not 0 <= root < n:
            raise ValueError(f"root must lie in [0, {n}), got {root}.")
        s, r, w = graph.edges()
        s, r = np.asarray(s), np.asarray(r)
        ln = _host(w if lengths is None else lengths, "lengths").astype(np.float64)
        if ln.shape != s.shape:
            raise ValueError(f"lengths must have shape {s.shape}, got {ln.shape}.")
        adj: list[list[tuple[int, float]]] = [[] for _ in range(n)]
        for u, v, d in zip(s.tolist(), r.tolist(), ln.tolist(), strict=True):
            adj[u].append((v, d))
            adj[v].append((u, d))
        seen = np.zeros(n, dtype=bool)
        n_comp = 0
        parent = np.full(n, -1, dtype=np.int64)
        length = np.zeros(n)
        for start in [root, *range(n)]:
            if seen[start]:
                continue
            n_comp += 1
            seen[start] = True
            queue = [start]
            for u in queue:
                for v, d in adj[u]:
                    if not seen[v]:
                        seen[v] = True
                        parent[v], length[v] = u, d
                        queue.append(v)
        if s.size - n + n_comp > 0:
            raise ValueError(
                "The graph contains a cycle; rooted-tree kernels are not valid "
                "on graphs with loops. Use matern_graph_kernel or another "
                "graph kernel."
            )
        if n_comp > 1:
            raise ValueError(
                f"The graph has {n_comp} connected components; a rooted tree "
                "has exactly one."
            )
        return cls.from_parents(parent, length, weights)

    def sites(
        self, edge: Int[ArrayLike, " N"], offset: Float[ArrayLike, " N"]
    ) -> TreeSites:
        r"""Place ``N`` points on the edges and compute their pairwise geometry.

        Site $i$ lies on the edge from node ``edge[i]`` to its parent, at
        distance ``offset[i]`` from the node end (``0`` = at the node,
        ``length[edge[i]]`` = at the parent end). A site at a node $u$ is
        ``(edge=u, offset=0)``; the root has no edge, so it cannot carry a
        site. Weights are constant along an edge: $\Omega(s_i)$ is the weight
        of the edge, so give a site at a junction to the edge it is the node
        of, not to a child edge's parent end.

        Costs $O(N \cdot \text{depth})$ host work per site (vectorised over
        the other sites) and ``O(N^2)`` memory.

        Raises:
            ValueError: For an invalid edge index (or the root), an offset
                outside ``[0, length]``, or an invalid tree.
            TypeError: If an input is traced.

        Examples:
            >>> import kernellib as kl
            >>> t = kl.RootedTree.from_parents([-1, 0, 1, 1], [0.0, 2.0, 3.0, 4.0])
            >>> s = t.sites([1, 2, 2], [1.0, 1.0, 2.0])
            >>> s.h.tolist()[0]
            [0.0, 3.0, 2.0]
            >>> [round(x, 4) for x in s.weight.tolist()[0]]  # sqrt(Omega ratio)
            [1.0, 0.7071, 0.7071]
        """
        skel, ln, w = _prepare(self.parent, self.length, self.weights)
        v = _host(edge, "edge")
        o = _host(offset, "offset").astype(np.float64)
        if v.ndim != 1 or o.shape != v.shape or not np.issubdtype(v.dtype, np.integer):
            raise ValueError(
                "edge and offset must be rank-1, equal length, edge integer."
            )
        v = v.astype(np.int64)
        if np.any(v < 0) or np.any(v >= skel.n) or np.any(v == skel.root):
            raise ValueError(
                "edge must index a non-root node (the root has no edge) "
                f"in [0, {skel.n})."
            )
        if np.any(o < -_ATOL) or np.any(o > ln[v] + _ATOL * (1.0 + ln[v])):
            raise ValueError("offset must lie in [0, length[edge]].")
        o = np.clip(o, 0.0, ln[v])
        depth = skel.depth(ln)
        t = depth[v] - o  # distance of each site to the root
        n = v.size
        tin_v = skel.tin[v]
        # Depth of the closest common ancestor node of the edges' lower nodes.
        lca = np.zeros((n, n))
        for i in range(n):
            path, u = [], int(v[i])
            while u >= 0:
                path.append(u)
                u = int(skel.parent[u])
            for u in reversed(path):  # root first, so the deepest wins
                inside = (tin_v >= skel.tin[u]) & (tin_v < skel.tin[u] + skel.size[u])
                lca[i, inside] = depth[u]
        d_i = einx.subtract("i, i j -> i j", t, lca)
        d_j = einx.id("i j -> j i", d_i)
        # i's edge node is an ancestor-or-self of j's edge node.
        anc = einx.less_equal("i, j -> i j", tin_v, tin_v) & einx.less(
            "j, i -> i j", tin_v, tin_v + skel.size[v]
        )
        connected = anc | einx.id("i j -> j i", anc)
        h_conn = np.abs(einx.subtract("i, j -> i j", t, t))
        w_site = w[v]
        ratio = np.sqrt(
            np.minimum(
                einx.divide("i, j -> i j", w_site, w_site),
                einx.divide("j, i -> i j", w_site, w_site),
            )
        )
        a_oth = np.minimum(d_i, d_j)
        b_oth = np.maximum(d_i, d_j)
        a = np.where(connected, 0.0, a_oth)
        b = np.where(connected, h_conn, b_oth)
        return TreeSites(
            h=jnp.asarray(a + b),
            a=jnp.asarray(a),
            b=jnp.asarray(b),
            connected=jnp.asarray(connected),
            weight=jnp.asarray(np.where(connected, ratio, 0.0)),
        )
