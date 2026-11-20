"""kernellib's Earth radius equals geophysix's.

kernellib sits below geophysix in the stack and must not import it at run time
(geophysix design decision 5), so it repeats the constant. This test, the only
place kernellib touches geophysix, keeps the copy equal. It is skipped where
geophysix is not installed.
"""

from __future__ import annotations

import math

import pytest

import kernellib as kl


def test_earth_radius_km_is_the_iugg_mean_radius() -> None:
    earth = pytest.importorskip("geophysix.earth")
    assert math.isclose(
        kl.EARTH_RADIUS_KM * 1000.0, earth.SPHERE.semi_major, rel_tol=1e-12
    )
    assert math.isclose(
        kl.EARTH_RADIUS_KM * 1000.0, earth.WGS84.mean_radius, rel_tol=1e-12
    )
