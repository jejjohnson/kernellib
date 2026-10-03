r"""GMRF structure on graphs: null spaces and (scaled) Besag structure matrices.

The Besag / intrinsic CAR (ICAR) structure matrix is the graph Laplacian,
$R = L$. It is positive semidefinite, and its null space is spanned by the
indicator vectors $\mathbf 1_C / \sqrt{|C|}$ of the connected components:
an intrinsic GMRF with precision $\tau R$ needs one sum-to-zero constraint
per component.

**BYM2 scaling** (Riebler et al., 2016; Sørbye & Rue, 2014) rescales
$R^\ast = sR$ so that the geometric mean of the marginal variances under the
constraints is 1, $s = \exp\big(\frac1N \sum_i \log [R^{+}]_{ii}\big)$, so
that a precision $\tau$ means the same on every graph. Following R-INLA, each
connected component is scaled separately (`gaussx.generalized_variance_scale`
computes $s$); an isolated node has no edges and nothing to scale.

**Covariance and precision form on one graph.** The graph Matérn kernel
(`matern_graph_kernel`, `kernellib.functional.graph_matern_spectrum`) has
spectrum $(2\nu/\ell^2 + \lambda)^{-\nu}$; gaussx's grid SPDE precision
(`gaussx.spde_precision_grid`) has covariance spectrum
$\tau^{-2}(\kappa^2 + \lambda)^{-\alpha}$ on the same unnormalised Laplacian.
They are the same family with $\nu = \alpha$ and $2\nu / \ell^2 = \kappa^2$,
i.e. ``lengthscale = sqrt(2 * alpha) / kappa``, and agree once each is
normalised to average marginal variance 1. The graph side needs
``normalization="unnormalized"``: the symmetric normalisation differs at the
border of a non-periodic grid.
"""

from __future__ import annotations

import gaussx as gx
import jax.numpy as jnp
import lineax as lx
import numpy as np
from jaxtyping import Array, Float

from kernellib._graph._eigpairs import _component_labels
from kernellib._graph._types import AbstractGraph, Graph, GraphTopology, GridGraph


__all__ = ["graph_null_space", "structure_matrix"]


def graph_null_space(graph: AbstractGraph) -> Float[Array, "N c"]:
    r"""Orthonormal basis of the Laplacian's null space.

    One column per connected component $C$ (an isolated node is one): the
    normalised indicator $\mathbf 1_C / \sqrt{|C|}$. These are the
    constraints $V^\top x = 0$ of an intrinsic GMRF with structure
    `structure_matrix`. Columns are ordered by each component's smallest
    node index.

    Args:
        graph: Any graph.

    Returns:
        ``(N, c)`` with ``c = n_components_graph(graph)``.

    Examples:
        >>> import kernellib as kl
        >>> g = kl.graph_from_edges([0], [1], 3)  # 0 - 1, and node 2 alone
        >>> [[round(v, 4) for v in row] for row in kl.graph_null_space(g).tolist()]
        [[0.7071, 0.0], [0.7071, 0.0], [0.0, 1.0]]
    """
    labels = _component_labels(graph)
    n_comp = int(labels.max()) + 1
    sizes = np.bincount(labels, minlength=n_comp)
    V = np.zeros((labels.shape[0], n_comp))
    V[np.arange(labels.shape[0]), labels] = 1.0 / np.sqrt(sizes[labels])
    dtype = graph.weights.dtype if isinstance(graph, Graph) else None
    return jnp.asarray(V, dtype=dtype)


def structure_matrix(
    graph: AbstractGraph, *, scaled: bool = False
) -> lx.AbstractLinearOperator:
    r"""The Besag (ICAR) structure matrix $R = L$, optionally BYM2-scaled.

    The unnormalised Laplacian operator, tagged symmetric and positive
    semidefinite: a `gaussx.SparseOperator` for a `Graph` (sparse Cholesky
    and selected inversion in gaussx), a `gaussx.KroneckerSum` for a
    face-connected `GridGraph` (exact eigen-structured solves).

    With ``scaled=True`` each connected component $C$ is multiplied by
    $s_C$ (`gaussx.generalized_variance_scale` on that component), so the
    geometric mean of its marginal variances under the sum-to-zero
    constraint is 1: the scaled structure of BYM2. A lattice is connected,
    so a `GridGraph` keeps its Kronecker structure (its axis weights are
    multiplied by $s$).

    Args:
        graph: Any graph.
        scaled: Apply the BYM2 scaling per connected component.

    Returns:
        The structure matrix as a lineax operator.

    Examples:
        >>> import jax.numpy as jnp
        >>> import kernellib as kl
        >>> g = kl.grid_graph((4,))  # an RW1 on 4 nodes
        >>> R = kl.structure_matrix(g, scaled=True)
        >>> P = jnp.eye(4) - jnp.full((4, 4), 0.25)  # sum-to-zero constraint
        >>> var = jnp.diag(P @ jnp.linalg.pinv(R.as_matrix()) @ P)
        >>> round(float(jnp.exp(jnp.mean(jnp.log(var)))), 4)  # geometric mean
        1.0
    """
    if not scaled:
        return graph.laplacian_operator()
    if isinstance(graph, GridGraph) and graph.connectivity == "face":
        L = graph.laplacian_operator()
        s = gx.generalized_variance_scale(L, graph_null_space(graph))
        return GridGraph(
            graph.shape,
            connectivity=graph.connectivity,
            periodic=graph.periodic,
            axis_weights=s * graph.axis_weights,
        ).laplacian_operator()
    top = graph.topology
    labels = _component_labels(graph)
    n_comp = int(labels.max()) + 1
    weights = graph.weights
    edge_scale = jnp.ones_like(weights)
    for c in range(n_comp):
        nodes = np.flatnonzero(labels == c)
        if nodes.shape[0] < 2:
            continue  # an isolated node: no edges, nothing to scale
        s_c = gx.generalized_variance_scale(
            _component_laplacian(top, weights, nodes),
            jnp.ones(nodes.shape[0], dtype=weights.dtype),
        )
        edge_scale = jnp.where(jnp.asarray(labels[top.senders] == c), s_c, edge_scale)
    return graph.reweight(weights * edge_scale).laplacian_operator()


def _component_laplacian(
    top: GraphTopology, weights: Float[Array, " E"], nodes: np.ndarray
) -> gx.SparseOperator:
    """The Laplacian of the subgraph induced by ``nodes`` (one component)."""
    local = np.full(top.n_nodes, -1)
    local[nodes] = np.arange(nodes.shape[0])
    inside = np.flatnonzero(local[top.senders] >= 0)
    sub = GraphTopology(
        local[top.senders[inside]], local[top.receivers[inside]], nodes.shape[0]
    )
    L = Graph(sub, weights[inside]).laplacian_operator()
    assert isinstance(L, gx.SparseOperator)
    return L
