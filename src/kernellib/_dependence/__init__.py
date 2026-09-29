"""Dependence measures on kernels and data: HSIC, CKA, alignment, MMD.

The matrix-level versions live in `kernellib.functional`; these take kernels
and samples, and optionally a feature map for a randomised ``O(N)`` path.
"""

from kernellib._dependence._distance import (
    distance_correlation_squared,
    distance_covariance_squared,
    energy_distance,
)
from kernellib._dependence._hsic import cka, hsic, kernel_alignment
from kernellib._dependence._mmd import mmd_squared
from kernellib._dependence._permutation import PermutationTestResult, permutation_test
from kernellib._dependence._taylor import TaylorStatistics, taylor_statistics


__all__ = [
    "PermutationTestResult",
    "TaylorStatistics",
    "cka",
    "distance_correlation_squared",
    "distance_covariance_squared",
    "energy_distance",
    "hsic",
    "kernel_alignment",
    "mmd_squared",
    "permutation_test",
    "taylor_statistics",
]
