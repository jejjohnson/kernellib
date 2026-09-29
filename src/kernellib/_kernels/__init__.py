"""Kernel objects: the abstract contract, concrete kernels, composition."""

from __future__ import annotations

from kernellib._kernels._base import (
    AbstractKernel,
    AbstractPointwiseKernel,
    AbstractStationaryKernel,
)
from kernellib._kernels._compose import (
    ActiveDims,
    Periodised,
    Product,
    Scaled,
    Shift,
    Stretch,
    Sum,
    Warped,
)
from kernellib._kernels._derivative import (
    Derivative,
    DerivativeIndexed,
    derivative_inputs,
)
from kernellib._kernels._feature import FeatureKernel, Modulated
from kernellib._kernels._nonstationary import Distance, Linear, Polynomial
from kernellib._kernels._residual import Residual, nystrom_kernel
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
    "Derivative",
    "DerivativeIndexed",
    "Distance",
    "FeatureKernel",
    "Linear",
    "Matern",
    "Modulated",
    "Periodic",
    "Periodised",
    "Polynomial",
    "Product",
    "RationalQuadratic",
    "Residual",
    "Scaled",
    "Shift",
    "Stretch",
    "Sum",
    "Warped",
    "White",
    "derivative_inputs",
    "nystrom_kernel",
]
