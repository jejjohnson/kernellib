"""Kernel and graph decompositions: kernel PCA, neighbourhood graphs, graph
kernels, and Laplacian / Schrödinger eigenmaps and LPP."""

from kernellib._decomposition._eigenmaps import (
    LaplacianEigenmaps,
    LocalityPreservingProjections,
    SchrodingerEigenmaps,
    barrier_potential,
    label_potential,
    laplacian_eigenmap,
    schrodinger_eigenmap,
    spatial_spectral_potential,
)
from kernellib._decomposition._graph import (
    adjacency_matrix,
    commute_time_kernel,
    cosine_graph_kernel,
    diffusion_kernel,
    graph_laplacian,
    random_walk_kernel,
    regularized_laplacian_kernel,
)
from kernellib._decomposition._kpca import KernelPCA
from kernellib._decomposition._neighbors import KNNGraph, nearest_neighbors


__all__ = [
    "KNNGraph",
    "KernelPCA",
    "LaplacianEigenmaps",
    "LocalityPreservingProjections",
    "SchrodingerEigenmaps",
    "adjacency_matrix",
    "barrier_potential",
    "commute_time_kernel",
    "cosine_graph_kernel",
    "diffusion_kernel",
    "graph_laplacian",
    "label_potential",
    "laplacian_eigenmap",
    "nearest_neighbors",
    "random_walk_kernel",
    "regularized_laplacian_kernel",
    "schrodinger_eigenmap",
    "spatial_spectral_potential",
]
