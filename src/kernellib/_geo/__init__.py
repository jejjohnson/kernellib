"""Kernels and tools for data on the Earth.

Geographic inputs are ``(N, 2)`` arrays of ``(lon, lat)``, longitude first, in
degrees by default. See ``docs/api/geo.md`` and the distances in
`kernellib.functional` (`great_circle_distance`, `chordal_distance`).
"""

from __future__ import annotations

from kernellib._geo._chordal import Chordal


__all__ = ["Chordal"]
