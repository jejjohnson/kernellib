"""Kernel density estimation (layer 2): the weighted Parzen estimator
`KernelDensity`, its score and Hessian, and bandwidth selection. See
``docs/api/density.md``."""

from kernellib._density._kde import KernelDensity


__all__ = ["KernelDensity"]
