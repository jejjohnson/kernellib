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
from kernellib._spectral._rff import draw_rff_cosine_basis, evaluate_rff_cosine_paths


__all__ = [
    "AbstractFeatureMap",
    "FastFoodFeatures",
    "NystromFeatures",
    "OrthogonalRandomFeatures",
    "RandomFourierFeatures",
    "draw_rff_cosine_basis",
    "evaluate_rff_cosine_paths",
]
