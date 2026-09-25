"""The spectral side of stationary kernels: random Fourier feature draws.

Spectral densities and frequency samplers are methods on
`AbstractStationaryKernel`; this package builds on them.
"""

from kernellib._spectral._rff import draw_rff_cosine_basis, evaluate_rff_cosine_paths


__all__ = ["draw_rff_cosine_basis", "evaluate_rff_cosine_paths"]
