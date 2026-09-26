"""Dependence measures on kernels and data: HSIC, CKA, alignment, MMD.

The matrix-level versions live in `kernellib.functional`; these take kernels
and samples, and optionally a feature map for a randomised ``O(N)`` path.
"""

from kernellib._dependence._hsic import cka, hsic, kernel_alignment
from kernellib._dependence._mmd import mmd_squared
from kernellib._dependence._permutation import PermutationTestResult, permutation_test


__all__ = [
    "PermutationTestResult",
    "cka",
    "hsic",
    "kernel_alignment",
    "mmd_squared",
    "permutation_test",
]
