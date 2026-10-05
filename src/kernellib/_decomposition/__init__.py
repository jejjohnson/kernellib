"""Kernel and graph decompositions: kernel PCA, Laplacian / Schrödinger
eigenmaps, LPP / SEP and their kernel versions. The graphs they are built on
live in `kernellib._graph`."""

from kernellib._decomposition._eigenmaps import (
    LaplacianEigenmaps,
    SchrodingerEigenmaps,
    barrier_potential,
    combine_potentials,
    label_potential,
    laplacian_eigenmap,
    schrodinger_eigenmap,
    spatial_spectral_graph,
    spatial_spectral_potential,
)
from kernellib._decomposition._kernel_projections import (
    KernelLocalityPreservingProjections,
    KernelSchrodingerProjections,
)
from kernellib._decomposition._kpca import KernelPCA
from kernellib._decomposition._projections import (
    LocalityPreservingProjections,
    SchrodingerEigenmapProjections,
)


__all__ = [
    "KernelLocalityPreservingProjections",
    "KernelPCA",
    "KernelSchrodingerProjections",
    "LaplacianEigenmaps",
    "LocalityPreservingProjections",
    "SchrodingerEigenmapProjections",
    "SchrodingerEigenmaps",
    "barrier_potential",
    "combine_potentials",
    "label_potential",
    "laplacian_eigenmap",
    "schrodinger_eigenmap",
    "spatial_spectral_graph",
    "spatial_spectral_potential",
]
