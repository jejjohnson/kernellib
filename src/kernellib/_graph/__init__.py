r"""Graphs: neighbour search, adjacency, Laplacians and graph kernels.

A weighted, undirected graph on ``N`` points is its symmetric adjacency matrix
$W$ (``N x N``, non-negative, zero diagonal). Its degree matrix is
$D = \mathrm{diag}(W\mathbf{1})$ and its Laplacian $L = D - W$ (or a
normalised form). Graph kernels are spectral functions of the Laplacian,
$K = U f(\Lambda) U^\top$ for $L = U\Lambda U^\top$ (Smola & Kondor, 2003):
they are positive semidefinite similarity matrices between the nodes, so they
plug into anything in kernellib that takes a Gram matrix (``functional.hsic``,
`KRR` via a precomputed operator, `KernelPCA` on nodes).

`Graph` and `GridGraph` store a graph sparsely: a static edge list with traced
weights, whose Laplacian, adjacency and incidence matrices are sparse gaussx
operators (a `gaussx.KroneckerSum` for a lattice). The adjacency-matrix
functions and the graph kernels are dense and differentiable; their ``N x N``
matrices limit them to graphs of a few thousand nodes. The eigenmaps'
``eigen_solver="arpack"`` path works on the sparse graph instead.
"""

from kernellib._graph._construct import (
    adjacency_matrix,
    edge_weights,
    graph_from_adjacency,
    graph_from_edges,
    graph_from_neighbors,
    grid_graph,
    knn_graph,
    mesh_graph,
    radius_graph,
)
from kernellib._graph._eigpairs import laplacian_eigpairs, n_components_graph
from kernellib._graph._kernels import (
    commute_time_kernel,
    cosine_graph_kernel,
    diffusion_kernel,
    matern_graph_kernel,
    random_walk_kernel,
    regularized_laplacian_kernel,
)
from kernellib._graph._laplacian import graph_laplacian
from kernellib._graph._neighbors import KNNGraph, nearest_neighbors, radius_neighbors
from kernellib._graph._proximity import (
    delaunay_graph,
    gabriel_graph,
    relative_neighborhood_graph,
)
from kernellib._graph._structure import graph_null_space, structure_matrix
from kernellib._graph._types import AbstractGraph, Graph, GraphTopology, GridGraph


__all__ = [
    "AbstractGraph",
    "Graph",
    "GraphTopology",
    "GridGraph",
    "KNNGraph",
    "adjacency_matrix",
    "commute_time_kernel",
    "cosine_graph_kernel",
    "delaunay_graph",
    "diffusion_kernel",
    "edge_weights",
    "gabriel_graph",
    "graph_from_adjacency",
    "graph_from_edges",
    "graph_from_neighbors",
    "graph_laplacian",
    "graph_null_space",
    "grid_graph",
    "knn_graph",
    "laplacian_eigpairs",
    "matern_graph_kernel",
    "mesh_graph",
    "n_components_graph",
    "nearest_neighbors",
    "radius_graph",
    "radius_neighbors",
    "random_walk_kernel",
    "regularized_laplacian_kernel",
    "relative_neighborhood_graph",
    "structure_matrix",
]
