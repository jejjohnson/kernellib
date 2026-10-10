"""Kernels and tools for data on the Earth.

Geographic inputs are ``(N, 2)`` arrays of ``(lon, lat)``, longitude first, in
degrees by default. See ``docs/api/geo.md`` and the distances in
`kernellib.functional` (`great_circle_distance`, `chordal_distance`).
"""

from __future__ import annotations

from kernellib._geo._anisotropy import GeometricAnisotropy, LinearTransform
from kernellib._geo._chordal import Chordal
from kernellib._geo._great_circle import (
    AbstractGreatCircleKernel,
    GreatCircleAskey,
    GreatCircleCauchy,
    GreatCircleExponential,
    GreatCirclePoweredExponential,
    GreatCircleSpherical,
    GreatCircleWendland,
)
from kernellib._geo._models import (
    Cubic,
    GeneralizedCauchy,
    HoleEffect,
    Pentaspherical,
    Spherical,
    Stable,
    Wendland,
)


__all__ = [
    "AbstractGreatCircleKernel",
    "Chordal",
    "Cubic",
    "GeneralizedCauchy",
    "GeometricAnisotropy",
    "GreatCircleAskey",
    "GreatCircleCauchy",
    "GreatCircleExponential",
    "GreatCirclePoweredExponential",
    "GreatCircleSpherical",
    "GreatCircleWendland",
    "HoleEffect",
    "LinearTransform",
    "Pentaspherical",
    "Spherical",
    "Stable",
    "Wendland",
]
