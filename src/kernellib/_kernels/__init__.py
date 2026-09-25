"""Kernel objects: the abstract contract, concrete kernels, composition."""

from __future__ import annotations

from kernellib._kernels._base import (
    AbstractKernel,
    AbstractPointwiseKernel,
    AbstractStationaryKernel,
)
from kernellib._kernels._compose import ActiveDims, Product, Scaled, Sum, Warped
from kernellib._kernels._nonstationary import Linear, Polynomial
from kernellib._kernels._stationary import (
    RBF,
    Constant,
    Cosine,
    Matern,
    Periodic,
    RationalQuadratic,
    White,
)


__all__ = [
    "RBF",
    "AbstractKernel",
    "AbstractPointwiseKernel",
    "AbstractStationaryKernel",
    "ActiveDims",
    "Constant",
    "Cosine",
    "Linear",
    "Matern",
    "Periodic",
    "Polynomial",
    "Product",
    "RationalQuadratic",
    "Scaled",
    "Sum",
    "Warped",
    "White",
]
