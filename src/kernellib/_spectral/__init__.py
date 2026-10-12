"""The spectral side of kernels: feature maps and random Fourier feature draws.

Spectral densities and frequency samplers are methods on
`AbstractStationaryKernel`; this package builds on them.
"""

from kernellib._spectral._base import AbstractFeatureMap
from kernellib._spectral._feature_maps import (
    FastFoodFeatures,
    NystromFeatures,
    OrthogonalRandomFeatures,
    RandomFourierFeatures,
)
from kernellib._spectral._funk_hecke import funk_hecke_coefficients
from kernellib._spectral._graph import GraphEigenFeatures
from kernellib._spectral._kronecker import KroneckerFeatures
from kernellib._spectral._landmarks import select_landmarks
from kernellib._spectral._laplace import LaplaceEigenfunctionFeatures
from kernellib._spectral._mesh import MeshLaplaceFeatures
from kernellib._spectral._periodic import PeriodicFeatures
from kernellib._spectral._rff import draw_rff_cosine_basis, evaluate_rff_cosine_paths
from kernellib._spectral._slepian import SlepianFeatures
from kernellib._spectral._spherical import SphericalHarmonicFeatures
from kernellib._spectral._vector import (
    CurlFreeFeatures,
    DivergenceFreeFeatures,
    HelmholtzFeatures,
)


__all__ = [
    "AbstractFeatureMap",
    "CurlFreeFeatures",
    "DivergenceFreeFeatures",
    "FastFoodFeatures",
    "GraphEigenFeatures",
    "HelmholtzFeatures",
    "KroneckerFeatures",
    "LaplaceEigenfunctionFeatures",
    "MeshLaplaceFeatures",
    "NystromFeatures",
    "OrthogonalRandomFeatures",
    "PeriodicFeatures",
    "RandomFourierFeatures",
    "SlepianFeatures",
    "SphericalHarmonicFeatures",
    "draw_rff_cosine_basis",
    "evaluate_rff_cosine_paths",
    "funk_hecke_coefficients",
    "select_landmarks",
]
