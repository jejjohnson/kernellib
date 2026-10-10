"""Arrays in, arrays out: pure kernel functions and matrix-level statistics.

Every function here takes plain JAX arrays or lineax operators and returns an
array, a scalar, or an operator. Nothing here takes a kernel object or a random
key; those APIs live at the top level of `kernellib`.

Examples:
    >>> import jax.numpy as jnp
    >>> import kernellib as kl
    >>> X = jnp.array([[0.0], [1.0]])
    >>> K = kl.functional.rbf_kernel(X, X, jnp.array(1.0), jnp.array(1.0))
    >>> K.shape
    (2, 2)
"""

from __future__ import annotations

from kernellib.functional._compose import kernel_add, kernel_mul
from kernellib.functional._geo import (
    EARTH_RADIUS_KM,
    chordal_distance,
    great_circle_distance,
    legendre_series,
    lonlat_to_unit,
)
from kernellib.functional._geo_models import (
    cubic_kernel,
    generalized_cauchy_kernel,
    hole_effect_kernel,
    pentaspherical_kernel,
    spherical_kernel,
    stable_kernel,
    wendland_kernel,
)
from kernellib.functional._graph import graph_heat_spectrum, graph_matern_spectrum
from kernellib.functional._mixed_precision import stable_rbf_kernel
from kernellib.functional._nonstationary import (
    distance_kernel,
    linear_kernel,
    polynomial_kernel,
)
from kernellib.functional._stationary import (
    constant_kernel,
    cosine_kernel,
    matern_kernel,
    periodic_kernel,
    rational_quadratic_kernel,
    rbf_kernel,
    white_kernel,
)
from kernellib.functional._statistics import (
    center_cross_kernel,
    center_kernel,
    centering_operator,
    cka,
    hsic,
    mmd_squared,
)


__all__ = [
    "EARTH_RADIUS_KM",
    "center_cross_kernel",
    "center_kernel",
    "centering_operator",
    "chordal_distance",
    "cka",
    "constant_kernel",
    "cosine_kernel",
    "cubic_kernel",
    "distance_kernel",
    "generalized_cauchy_kernel",
    "graph_heat_spectrum",
    "graph_matern_spectrum",
    "great_circle_distance",
    "hole_effect_kernel",
    "hsic",
    "kernel_add",
    "kernel_mul",
    "legendre_series",
    "linear_kernel",
    "lonlat_to_unit",
    "matern_kernel",
    "mmd_squared",
    "pentaspherical_kernel",
    "periodic_kernel",
    "polynomial_kernel",
    "rational_quadratic_kernel",
    "rbf_kernel",
    "spherical_kernel",
    "stable_kernel",
    "stable_rbf_kernel",
    "wendland_kernel",
    "white_kernel",
]
