"""Kernel and graph decompositions: kernel PCA, Laplacian / Schrödinger
eigenmaps and LPP. The graphs they are built on live in `kernellib._graph`."""

from kernellib._decomposition._eigenmaps import (
    LaplacianEigenmaps,
    SchrodingerEigenmaps,
    barrier_potential,
    label_potential,
    laplacian_eigenmap,
    schrodinger_eigenmap,
    spatial_spectral_potential,
)
from kernellib._decomposition._kpca import KernelPCA
from kernellib._decomposition._projections import LocalityPreservingProjections


__all__ = [
    "KernelPCA",
    "LaplacianEigenmaps",
    "LocalityPreservingProjections",
    "SchrodingerEigenmaps",
    "barrier_potential",
    "label_potential",
    "laplacian_eigenmap",
    "schrodinger_eigenmap",
    "spatial_spectral_potential",
]
