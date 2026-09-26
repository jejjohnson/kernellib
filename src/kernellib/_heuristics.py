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

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from kernellib.functional._distances import _pairwise_sq_dist


__all__ = [
    "estimate_lengthscale",
    "gamma_to_lengthscale",
    "lengthscale_grid",
    "lengthscale_to_gamma",
]

Method = Literal["median", "mean", "silverman", "scott"]


def estimate_lengthscale(
    X: Float[Array, "N D"],
    method: Method = "median",
    *,
    percent: float | None = None,
    subsample: int | None = None,
    key: PRNGKeyArray | None = None,
    scale: float = 1.0,
    ard: bool = False,
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

    $\hat\sigma$ is the mean per-dimension standard deviation (or each
    dimension's own with ``ard``).

    Args:
        X: Data, shape ``(N, D)``.
        method: One of ``"median"``, ``"mean"``, ``"silverman"``, ``"scott"``.
        percent: For ``"median"`` / ``"mean"``, the neighbour fraction in
            ``(0, 1]``; ``None`` uses all pairwise distances.
        subsample: Use a random subset of this many points (the distance
            methods cost ``O(n^2)``). Needs ``key`` when smaller than ``N``.
        key: PRNG key for ``subsample``.
        scale: Multiplier on the result.
        ard: Return one lengthscale per dimension, each from that
            coordinate alone.

    Returns:
        A scalar lengthscale, or ``(D,)`` with ``ard``.

    Raises:
        ValueError: On an unknown method, a ``percent`` outside ``(0, 1]``,
            fewer than two points, or ``subsample`` without ``key``.

    Examples:
        >>> import jax.numpy as jnp
        >>> import kernellib as kl
        >>> X = jnp.array([[0.0], [1.0], [3.0]])
        >>> float(kl.estimate_lengthscale(X))  # distances 1, 2, 3
        2.0
        >>> kl.estimate_lengthscale(
        ...     jnp.ones((10, 3)) * jnp.arange(10.0)[:, None], "scott", ard=True
        ... ).shape
        (3,)
    """
    if method not in ("median", "mean", "silverman", "scott"):
        raise ValueError(
            f"method must be 'median', 'mean', 'silverman' or 'scott', got {method!r}."
        )
    if percent is not None and not 0.0 < percent <= 1.0:
        raise ValueError(f"percent must be in (0, 1], got {percent}.")
    X = jnp.asarray(X)
    if subsample is not None and subsample < X.shape[0]:
        if key is None:
            raise ValueError("subsample needs a PRNG key.")
        X = X[jax.random.choice(key, X.shape[0], (subsample,), replace=False)]
    n = X.shape[0]
    if n < 2:
        raise ValueError(f"Need at least two points, got {n}.")

    if ard:
        per_dim = jax.vmap(
            lambda col: _estimate(col[:, None], method, percent), in_axes=1
        )(X)
        return scale * per_dim
    return scale * _estimate(X, method, percent)


def _estimate(
    X: Float[Array, "N D"], method: Method, percent: float | None
) -> Float[Array, ""]:
    n, d = X.shape
    if method in ("silverman", "scott"):
        sigma = jnp.mean(jnp.std(X, axis=0, ddof=1))
        if method == "silverman":
            return sigma * (n * (d + 2.0) / 4.0) ** (-1.0 / (d + 4.0))
        return sigma * n ** (-1.0 / (d + 4.0))

    aggregate = jnp.median if method == "median" else jnp.mean
    dist = jnp.sqrt(jnp.clip(_pairwise_sq_dist(X, X, 1.0), min=0.0))
    if percent is None:
        rows, cols = jnp.triu_indices(n, k=1)
        return aggregate(dist[rows, cols])
    # Column 0 of each sorted row is the point itself (distance 0).
    k = min(max(1, int(percent * n)), n - 1)
    return aggregate(jnp.sort(dist, axis=1)[:, k])


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
