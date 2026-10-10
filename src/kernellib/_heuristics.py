r"""Bandwidth heuristics for RBF-type kernels.

Reimplemented from pysim's ``estimate_sigma`` with three corrections:

- Silverman's and Scott's rules are scaled by the data's standard deviation.
  pysim returned only the ``n``-dependent factor, which is a bandwidth for
  standardised data only.
- The ``percent`` (k-th neighbour) rule counts ``k`` from the subsample it
  actually uses, not from the full data.
- The self-distance is excluded from every distance statistic.

For the RBF kernel $\exp(-\|x - x'\|^2 / 2\ell^2)$ the lengthscale $\ell$ is
pysim's $\sigma$, and ``gamma`` is the scikit-learn parameterisation
$\exp(-\gamma \|x - x'\|^2)$, so $\gamma = 1 / (2 \ell^2)$.
"""

from __future__ import annotations

from typing import Literal

import einx
import jax
import jax.numpy as jnp
from jax.scipy.special import gammaln
from jaxtyping import Array, Float, PRNGKeyArray

from kernellib._einx import rearrange, reduce
from kernellib.functional._distances import _pairwise_sq_dist
from kernellib.functional._geo import chordal_distance, great_circle_distance


__all__ = [
    "estimate_lengthscale",
    "gamma_to_lengthscale",
    "lengthscale_grid",
    "lengthscale_to_gamma",
]

Method = Literal["median", "mean", "silverman", "scott", "gaussian"]
Metric = Literal["euclidean", "great_circle", "chordal"]


def estimate_lengthscale(
    X: Float[Array, "N D"],
    method: Method = "median",
    *,
    percent: float | None = None,
    subsample: int | None = None,
    key: PRNGKeyArray | None = None,
    scale: float = 1.0,
    ard: bool = False,
    metric: Metric = "euclidean",
    radius: float = 1.0,
    degrees: bool = True,
) -> Float[Array, ""] | Float[Array, " D"]:
    r"""A data-driven lengthscale for an RBF-type kernel.

    Methods:

    - ``"median"`` / ``"mean"``: the median / mean pairwise Euclidean
      distance between distinct points (the median heuristic). With
      ``percent``, each point's distance to its ``k``-th nearest neighbour,
      ``k = max(1, floor(percent * n))``, is taken first and the median /
      mean is over points; small ``percent`` gives local bandwidths.
    - ``"silverman"``: $\hat\sigma\,(n (d + 2) / 4)^{-1/(d + 4)}$.
    - ``"scott"``: $\hat\sigma\, n^{-1/(d + 4)}$.
    - ``"gaussian"``: $2\hat\sigma\,\Gamma(\tfrac{d+1}{2})/\Gamma(\tfrac d2)$,
      the expected distance between two independent draws from
      $\mathcal N(\mu, \hat\sigma^2 I_d)$. It is a closed form: it needs
      no pairwise distances (``O(n d)``), and it is a smooth function of the
      data, unlike the median.

    $\hat\sigma$ is the mean per-dimension standard deviation (or each
    dimension's own with ``ard``).

    With ``metric="great_circle"`` or ``"chordal"``, ``X`` is ``(N, 2)``
    ``(lon, lat)`` and ``"median"`` / ``"mean"`` run on that distance, so the
    lengthscale is in units of ``radius`` (km with ``radius=EARTH_RADIUS_KM``).
    The other methods, and ``ard``, assume Euclidean coordinates and are
    rejected with a geo metric.

    Args:
        X: Data, shape ``(N, D)``; ``(N, 2)`` ``(lon, lat)`` for a geo metric.
        method: One of ``"median"``, ``"mean"``, ``"silverman"``, ``"scott"``,
            ``"gaussian"``.
        percent: For ``"median"`` / ``"mean"``, the neighbour fraction in
            ``(0, 1]``; ``None`` uses all pairwise distances.
        subsample: Use a random subset of this many points (the distance
            methods cost ``O(n^2)``). Needs ``key`` when smaller than ``N``.
        key: PRNG key for ``subsample``.
        scale: Multiplier on the result.
        ard: Return one lengthscale per dimension, each from that
            coordinate alone.
        metric: ``"euclidean"``, ``"great_circle"`` or ``"chordal"``.
        radius: Sphere radius for a geo metric; the result is in its units.
            Ignored for ``"euclidean"``.
        degrees: Whether ``(lon, lat)`` are in degrees (else radians).
            Ignored for ``"euclidean"``.

    Returns:
        A scalar lengthscale, or ``(D,)`` with ``ard``.

    Raises:
        ValueError: On an unknown method or metric, a ``percent`` outside
            ``(0, 1]``, fewer than two points, ``subsample`` without ``key``,
            or a geo metric with a method other than ``"median"`` /
            ``"mean"``, with ``ard``, or on ``X`` that is not ``(N, 2)``.

    Examples:
        >>> import einx
        >>> import jax.numpy as jnp
        >>> import kernellib as kl
        >>> X = jnp.array([[0.0], [1.0], [3.0]])
        >>> float(kl.estimate_lengthscale(X))  # distances 1, 2, 3
        2.0
        >>> kl.estimate_lengthscale(
        ...     einx.multiply("n d, n -> n d", jnp.ones((10, 3)), jnp.arange(10.0)),
        ...     "scott",
        ...     ard=True,
        ... ).shape
        (3,)

        On the sphere, in km (two points a quarter of the equator apart):

        >>> lonlat = jnp.array([[0.0, 0.0], [90.0, 0.0]])
        >>> d = kl.estimate_lengthscale(
        ...     lonlat, metric="great_circle", radius=kl.EARTH_RADIUS_KM
        ... )
        >>> round(float(d))
        10008
    """
    if method not in ("median", "mean", "silverman", "scott", "gaussian"):
        raise ValueError(
            "method must be 'median', 'mean', 'silverman', 'scott' or 'gaussian', "
            f"got {method!r}."
        )
    if percent is not None and not 0.0 < percent <= 1.0:
        raise ValueError(f"percent must be in (0, 1], got {percent}.")
    X = jnp.asarray(X)
    if metric not in ("euclidean", "great_circle", "chordal"):
        raise ValueError(
            f"metric must be 'euclidean', 'great_circle' or 'chordal', got {metric!r}."
        )
    if metric != "euclidean":
        if method not in ("median", "mean") or ard:
            raise ValueError(
                f"metric={metric!r} supports method='median' or 'mean' without "
                f"ard; got method={method!r}, ard={ard}."
            )
        if X.ndim != 2 or X.shape[1] != 2:
            raise ValueError(
                f"metric={metric!r} needs (N, 2) (lon, lat) points, got shape "
                f"{X.shape}."
            )
    if subsample is not None and subsample < X.shape[0]:
        if key is None:
            raise ValueError("subsample needs a PRNG key.")
        X = X[jax.random.choice(key, X.shape[0], (subsample,), replace=False)]
    n = X.shape[0]
    if n < 2:
        raise ValueError(f"Need at least two points, got {n}.")

    if ard:
        per_dim = jax.vmap(
            lambda col: _estimate(rearrange(col, "n -> n 1"), method, percent),
            in_axes=1,
        )(X)
        return scale * per_dim
    if metric != "euclidean":
        geo = great_circle_distance if metric == "great_circle" else chordal_distance
        dist = geo(X, X, radius=radius, degrees=degrees)
        return scale * _aggregate_distances(dist, method, percent)
    return scale * _estimate(X, method, percent)


def _estimate(
    X: Float[Array, "N D"], method: Method, percent: float | None
) -> Float[Array, ""]:
    n, d = X.shape
    if method == "gaussian":
        centred = einx.subtract("n d, d -> n d", X, reduce(X, "n d -> d", "mean"))
        variance = reduce(centred**2, "n d -> d", "sum") / (n - 1)
        # Mean per-dimension std (ddof=1). A zero-safe sqrt keeps the gradient
        # finite when a column is constant (sqrt has an infinite slope at 0).
        constant = variance <= 0
        std = jnp.where(constant, 0.0, jnp.sqrt(jnp.where(constant, 1.0, variance)))
        sigma = jnp.mean(std)
        return 2.0 * sigma * jnp.exp(gammaln((d + 1) / 2.0) - gammaln(d / 2.0))
    if method in ("silverman", "scott"):
        # Per-column std (ddof=1); einx's std has no ddof.
        sigma = jnp.mean(jax.vmap(lambda col: jnp.std(col, ddof=1), in_axes=1)(X))
        if method == "silverman":
            return sigma * (n * (d + 2.0) / 4.0) ** (-1.0 / (d + 4.0))
        return sigma * n ** (-1.0 / (d + 4.0))

    dist = jnp.sqrt(jnp.clip(_pairwise_sq_dist(X, X, 1.0), min=0.0))
    return _aggregate_distances(dist, method, percent)


def _aggregate_distances(
    dist: Float[Array, "N N"], method: Method, percent: float | None
) -> Float[Array, ""]:
    """The ``"median"`` / ``"mean"`` statistic of a pairwise distance matrix."""
    n = dist.shape[0]
    aggregate = jnp.median if method == "median" else jnp.mean
    if percent is None:
        rows, cols = jnp.triu_indices(n, k=1)
        return aggregate(dist[rows, cols])
    # Column 0 of each sorted row is the point itself (distance 0).
    k = min(max(1, int(percent * n)), n - 1)
    return aggregate(einx.sort("i [j]", dist)[:, k])


def lengthscale_to_gamma(
    lengthscale: float | Float[Array, ...],
) -> Float[Array, ...]:
    r"""``gamma = 1 / (2 lengthscale^2)``, the scikit-learn RBF parameter.

    Examples:
        >>> import kernellib as kl
        >>> float(kl.lengthscale_to_gamma(0.5))
        2.0
    """
    return 1.0 / (2.0 * jnp.asarray(lengthscale) ** 2)


def gamma_to_lengthscale(gamma: float | Float[Array, ...]) -> Float[Array, ...]:
    r"""``lengthscale = 1 / sqrt(2 gamma)``, the inverse of `lengthscale_to_gamma`.

    Examples:
        >>> import kernellib as kl
        >>> float(kl.gamma_to_lengthscale(2.0))
        0.5
    """
    return 1.0 / jnp.sqrt(2.0 * jnp.asarray(gamma))


def lengthscale_grid(
    center: float | Float[Array, ""],
    *,
    decades: float = 2.0,
    n_points: int = 20,
) -> Float[Array, " n_points"]:
    r"""Log-spaced lengthscales within ``decades`` orders of magnitude of ``center``.

    A search grid around a heuristic estimate. For a grid in ``gamma``, map
    it through `lengthscale_to_gamma` (pysim's ``get_gamma_grid`` returned
    lengthscales by mistake).

    Examples:
        >>> import kernellib as kl
        >>> grid = kl.lengthscale_grid(1.0, decades=1.0, n_points=3)
        >>> [round(float(g), 6) for g in grid]
        [0.1, 1.0, 10.0]
    """
    if n_points < 1:
        raise ValueError(f"n_points must be >= 1, got {n_points}.")
    log_center = jnp.log10(jnp.asarray(center))
    return jnp.logspace(log_center - decades, log_center + decades, n_points)
