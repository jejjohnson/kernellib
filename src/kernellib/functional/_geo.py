r"""Distances on the sphere for ``(lon, lat)`` inputs.

Geographic inputs are ``(N, 2)`` arrays of ``(lon, lat)``, longitude first
(the order of `geonnax.geo.lonlat_to_cartesian3d`), in degrees by default.
Distances come back in units of ``radius``: ``radius=1.0`` gives the
great-circle angle in radians, ``radius=EARTH_RADIUS_KM`` kilometres.

Both distances go through the unit vectors $u, v$. The great-circle angle uses
$\tan(\theta/2) = \lVert u - v\rVert / \lVert u + v\rVert$, which is accurate
at every separation (the haversine loses precision near antipodes, the arccos
form near zero), and the chordal distance is $R\,\lVert u - v\rVert =
2R\sin(\theta/2)$. Square roots are guarded so gradients stay finite at
coincident points, the diagonal of a Gram matrix.
"""

from __future__ import annotations

import einx
import jax.numpy as jnp
from geonnax.geo import lonlat_to_cartesian3d
from jaxtyping import Array, Float


EARTH_RADIUS_KM: float = 6371.0088
"""IUGG mean Earth radius, in kilometres."""


def _safe_sqrt(s: Float[Array, ...]) -> Float[Array, ...]:
    """``sqrt`` with a zero (not infinite) gradient at ``s = 0``."""
    positive = s > 0
    return jnp.where(positive, jnp.sqrt(jnp.where(positive, s, 1.0)), 0.0)


def lonlat_to_unit(
    X: Float[Array, "N 2"], *, degrees: bool = True
) -> Float[Array, "N 3"]:
    """Unit vectors in R³ for ``(lon, lat)`` points.

    Args:
        X: ``(N, 2)`` longitude/latitude.
        degrees: Whether ``X`` is in degrees (else radians).

    Returns:
        ``(N, 3)`` unit vectors $(\\cos\\phi\\cos\\lambda, \\cos\\phi\\sin\\lambda,
        \\sin\\phi)$.

    Examples:
        >>> import jax.numpy as jnp
        >>> from kernellib.functional import lonlat_to_unit
        >>> (lonlat_to_unit(jnp.array([[90.0, 0.0]])).round(6) + 0.0).tolist()
        [[0.0, 1.0, 0.0]]
    """
    X = jnp.asarray(X)
    if X.ndim != 2 or X.shape[-1] != 2:
        raise ValueError(f"X must be (N, 2) (lon, lat); got shape {X.shape}.")
    return lonlat_to_cartesian3d(X, input_unit="degrees" if degrees else "radians")


def _chord_and_sum(
    U1: Float[Array, "N1 3"], U2: Float[Array, "N2 3"]
) -> tuple[Float[Array, "N1 N2"], Float[Array, "N1 N2"]]:
    """Squared norms of ``u - v`` and ``u + v`` for every pair."""
    diff = einx.subtract("n d, m d -> n m d", U1, U2)
    total = einx.add("n d, m d -> n m d", U1, U2)
    return (
        einx.sum("n m d -> n m", diff**2),
        einx.sum("n m d -> n m", total**2),
    )


def great_circle_distance(
    X1: Float[Array, "N1 2"],
    X2: Float[Array, "N2 2"],
    *,
    radius: float = 1.0,
    degrees: bool = True,
) -> Float[Array, "N1 N2"]:
    r"""Pairwise great-circle distance $R\,\theta$ between ``(lon, lat)`` points.

    $\theta = 2\operatorname{atan2}(\lVert u - v\rVert, \lVert u + v\rVert)$
    for the unit vectors $u, v$, accurate from coincident points to antipodes.

    Args:
        X1: ``(N1, 2)`` longitude/latitude.
        X2: ``(N2, 2)`` longitude/latitude.
        radius: Sphere radius; the result is in its units. ``1.0`` gives the
            angle in radians, `EARTH_RADIUS_KM` kilometres.
        degrees: Whether the inputs are in degrees (else radians).

    Returns:
        ``(N1, N2)`` distances in ``[0, π·radius]``.

    Examples:
        >>> import jax.numpy as jnp
        >>> from kernellib.functional import EARTH_RADIUS_KM, great_circle_distance
        >>> london = jnp.array([[-0.1278, 51.5074]])
        >>> paris = jnp.array([[2.3522, 48.8566]])
        >>> d = great_circle_distance(london, paris, radius=EARTH_RADIUS_KM)
        >>> round(float(d[0, 0]), 1)  # km
        343.6
    """
    chord2, sum2 = _chord_and_sum(
        lonlat_to_unit(X1, degrees=degrees), lonlat_to_unit(X2, degrees=degrees)
    )
    return radius * 2.0 * jnp.arctan2(_safe_sqrt(chord2), _safe_sqrt(sum2))


def chordal_distance(
    X1: Float[Array, "N1 2"],
    X2: Float[Array, "N2 2"],
    *,
    radius: float = 1.0,
    degrees: bool = True,
) -> Float[Array, "N1 N2"]:
    r"""Pairwise chordal (straight-line, through the sphere) distance.

    $c = R\,\lVert u - v\rVert = 2R\sin(\theta/2)$, with $\theta$ the
    great-circle angle. It is a monotone function of $\theta$, so it orders
    neighbours exactly as great-circle distance does, and any Euclidean kernel
    of it is positive definite on the sphere (see `kernellib.Chordal`).

    Args:
        X1: ``(N1, 2)`` longitude/latitude.
        X2: ``(N2, 2)`` longitude/latitude.
        radius: Sphere radius; the result is in its units.
        degrees: Whether the inputs are in degrees (else radians).

    Returns:
        ``(N1, N2)`` distances in ``[0, 2·radius]``.

    Examples:
        >>> import jax.numpy as jnp
        >>> from kernellib.functional import chordal_distance
        >>> X = jnp.array([[0.0, 0.0], [180.0, 0.0]])  # antipodes
        >>> chordal_distance(X, X).round(6).tolist()
        [[0.0, 2.0], [2.0, 0.0]]
    """
    chord2, _ = _chord_and_sum(
        lonlat_to_unit(X1, degrees=degrees), lonlat_to_unit(X2, degrees=degrees)
    )
    return radius * _safe_sqrt(chord2)


__all__ = [
    "EARTH_RADIUS_KM",
    "chordal_distance",
    "great_circle_distance",
    "lonlat_to_unit",
]
