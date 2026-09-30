r"""Graphs: neighbour search, adjacency, Laplacians and graph kernels.

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

from kernellib._graph._construct import adjacency_matrix
from kernellib._graph._kernels import (
    commute_time_kernel,
    cosine_graph_kernel,
    diffusion_kernel,
    random_walk_kernel,
    regularized_laplacian_kernel,
)
from kernellib._graph._laplacian import graph_laplacian
from kernellib._graph._neighbors import KNNGraph, nearest_neighbors


__all__ = [
    "KNNGraph",
    "adjacency_matrix",
    "commute_time_kernel",
    "cosine_graph_kernel",
    "diffusion_kernel",
    "graph_laplacian",
    "nearest_neighbors",
    "random_walk_kernel",
    "regularized_laplacian_kernel",
]
