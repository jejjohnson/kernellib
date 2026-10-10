r"""Empirical semivariograms and weighted least-squares variogram fits.

The semivariogram of a stationary field is
$\gamma(h) = \tfrac12\,\mathbb E[(y(x) - y(x'))^2]$ at separation
$h = d(x, x')$. For a covariance $k$ with sill $\sigma^2 = k(0)$ and a nugget
$\tau^2$ it is $\gamma(h) = \tau^2 + \sigma^2 - k(h)$ for $h > 0$.

- `empirical_variogram` bins the pairs $i < j$ by distance (Euclidean or
  great-circle) and estimates $\gamma$ in each bin, with Matheron's
  moment estimator or Cressie and Hawkins' robust one. The pairs are
  streamed in row blocks with `jax.lax.scan`, so memory is
  $O(\text{batch}\cdot N)$, never $N^2$.
- `fit_variogram` fits a stationary kernel plus a nugget to the binned
  estimate by weighted least squares (Cressie 1985), with BFGS on the
  log-parameters.

Directional variograms (an angle and a tolerance) and cross-variograms are
out of scope.
"""

from __future__ import annotations

from typing import Literal

import einx
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jax.scipy.optimize import minimize
from jaxtyping import Array, Float, Int, PRNGKeyArray

from kernellib._einx import rearrange
from kernellib._kernels._base import AbstractStationaryKernel
from kernellib.functional._geo import great_circle_distance


__all__ = ["Variogram", "empirical_variogram", "fit_variogram"]

# Cressie & Hawkins (1980): E|Z|^{1/2} bias correction, 0.457 + 0.494 / N.
_CH_A = 0.457
_CH_B = 0.494


class Variogram(eqx.Module):
    """A binned empirical semivariogram.

    Attributes:
        bin_edges: ``(B + 1,)`` distance bin edges.
        bin_centers: ``(B,)`` mean pair distance in each bin (the edge
            midpoint for an empty bin).
        gamma: ``(B,)`` semivariance $\\hat\\gamma(h_b)$; NaN for an empty bin.
        counts: ``(B,)`` number of pairs $N_b$ in each bin.

    Examples:
        >>> import jax.numpy as jnp
        >>> import kernellib as kl
        >>> v = kl.Variogram(
        ...     bin_edges=jnp.array([0.0, 1.0, 2.0]),
        ...     bin_centers=jnp.array([0.5, 1.5]),
        ...     gamma=jnp.array([0.2, 0.4]),
        ...     counts=jnp.array([10, 12]),
        ... )
        >>> v.gamma.shape
        (2,)
    """

    bin_edges: Float[Array, " B1"] = eqx.field(converter=jnp.asarray)
    bin_centers: Float[Array, " B"] = eqx.field(converter=jnp.asarray)
    gamma: Float[Array, " B"] = eqx.field(converter=jnp.asarray)
    counts: Int[Array, " B"] = eqx.field(converter=jnp.asarray)


def _block_distances(
    Xb: Float[Array, "b D"],
    X: Float[Array, "N D"],
    metric: str,
    radius: float,
    degrees: bool,
) -> Float[Array, "b N"]:
    """Distances from a block of rows to every point."""
    if metric == "great_circle":
        return great_circle_distance(Xb, X, radius=radius, degrees=degrees)
    diff = einx.subtract("b d, n d -> b n d", Xb, X)
    return jnp.sqrt(einx.sum("b n d -> b n", diff**2))


def empirical_variogram(
    X: Float[Array, "N D"],
    y: Float[Array, " N"],
    bins: int | Float[Array, " B1"] = 20,
    *,
    max_distance: float | None = None,
    estimator: Literal["matheron", "cressie"] = "matheron",
    metric: Literal["euclidean", "great_circle"] = "euclidean",
    radius: float = 1.0,
    degrees: bool = True,
    subsample: int | None = None,
    key: PRNGKeyArray | None = None,
    batch_size: int = 1024,
) -> Variogram:
    r"""Binned empirical semivariogram of ``y`` observed at ``X``.

    Over the pairs $i < j$ whose distance $d_{ij}$ falls in bin $b$:

    - Matheron: $\hat\gamma_b = \frac{1}{2N_b}\sum (y_i - y_j)^2$;
    - Cressie-Hawkins (robust): $\bar\gamma_b = \big(\frac{1}{N_b}\sum
      |y_i - y_j|^{1/2}\big)^4 / \big(2\,(0.457 + 0.494/N_b)\big)$.

    A bin is $[e_b, e_{b+1})$, the last one closed. Pairs beyond
    ``max_distance`` are dropped; empty bins get ``gamma = nan`` and
    ``counts = 0``. With an integer ``bins`` the number of bins is static,
    so the output has a fixed shape and the function can be jit-compiled
    (pass ``bins`` and the string options as static arguments). Pairs are
    streamed over row blocks of ``batch_size`` with `jax.lax.scan`, in
    $O(\text{batch}\cdot N)$ memory.

    Directional variograms and cross-variograms are not supported.

    Args:
        X: ``(N, D)`` locations; ``(N, 2)`` ``(lon, lat)`` for
            ``metric="great_circle"``.
        y: ``(N,)`` observations.
        bins: Number of equal-width bins on ``[0, max_distance]``, or the
            ``(B + 1,)`` bin edges.
        max_distance: Largest pair distance kept. Defaults to the last edge
            when ``bins`` gives edges, else half the largest pairwise
            distance.
        estimator: ``"matheron"`` or the robust ``"cressie"``.
        metric: ``"euclidean"`` or ``"great_circle"``
            (`kernellib.functional.great_circle_distance`).
        radius: Sphere radius for ``"great_circle"``; distances are in its
            units (``1.0`` gives radians).
        degrees: Whether ``(lon, lat)`` are in degrees for ``"great_circle"``.
        subsample: Use a random subset of this many rows (for very large N).
        key: PRNG key, required with ``subsample``.
        batch_size: Rows per block of the pair scan.

    Returns:
        The `Variogram`.

    Raises:
        ValueError: On an unknown ``estimator`` or ``metric``, mismatched
            shapes, or ``subsample`` without ``key``.

    Examples:
        >>> import einx
        >>> import jax.numpy as jnp
        >>> import kernellib as kl
        >>> X = einx.id("n -> n 1", jnp.arange(10.0))
        >>> v = kl.empirical_variogram(X, X[:, 0], jnp.array([0.5, 1.5, 2.5, 3.5]))
        >>> v.gamma.tolist(), v.counts.tolist()
        ([0.5, 2.0, 4.5], [9, 8, 7])
    """
    if estimator not in ("matheron", "cressie"):
        raise ValueError(
            f"estimator must be 'matheron' or 'cressie'; got {estimator!r}."
        )
    if metric not in ("euclidean", "great_circle"):
        raise ValueError(
            f"metric must be 'euclidean' or 'great_circle'; got {metric!r}."
        )
    X = jnp.asarray(X)
    y = jnp.asarray(y, dtype=X.dtype)
    if X.ndim != 2 or y.ndim != 1 or X.shape[0] != y.shape[0]:
        raise ValueError(
            f"X must be (N, D) and y (N,); got shapes {X.shape} and {y.shape}."
        )
    if subsample is not None and subsample < X.shape[0]:
        if key is None:
            raise ValueError("subsample requires a PRNG key.")
        rows = jr.choice(key, X.shape[0], (subsample,), replace=False)
        X, y = X[rows], y[rows]
    return _empirical_variogram(
        X, y, bins, max_distance, estimator, metric, radius, degrees, batch_size
    )


@eqx.filter_jit
def _empirical_variogram(
    X: Float[Array, "N D"],
    y: Float[Array, " N"],
    bins: int | Float[Array, " B1"],
    max_distance: float | Array | None,
    estimator: str,
    metric: str,
    radius: float,
    degrees: bool,
    batch_size: int,
) -> Variogram:
    """The jit-compiled pair scan behind `empirical_variogram`."""
    n = X.shape[0]
    batch = max(1, min(batch_size, n))
    n_blocks = -(-n // batch)
    pad = n_blocks * batch - n
    Xp = rearrange(
        jnp.concatenate([X, jnp.zeros((pad, X.shape[1]), X.dtype)]),
        "(k b) d -> k b d",
        b=batch,
    )
    yp = rearrange(
        jnp.concatenate([y, jnp.zeros(pad, y.dtype)]), "(k b) -> k b", b=batch
    )
    starts = jnp.arange(n_blocks) * batch
    cols = jnp.arange(n)

    def pair_mask(start: Array) -> Array:
        rows = start + jnp.arange(batch)
        return einx.less("b, n -> b n", rows, cols)  # i < j, so padded rows drop

    def block_dist(Xb: Array) -> Array:
        return _block_distances(Xb, X, metric, radius, degrees)

    if isinstance(bins, int):
        if max_distance is None:

            def max_body(carry: Array, inp: tuple[Array, Array]) -> tuple[Array, None]:
                Xb, start = inp
                d = jnp.where(pair_mask(start), block_dist(Xb), 0.0)
                return jnp.maximum(carry, jnp.max(d)), None

            dmax, _ = jax.lax.scan(max_body, jnp.zeros((), X.dtype), (Xp, starts))
            max_distance = 0.5 * dmax
        edges = jnp.linspace(0.0, max_distance, bins + 1, dtype=X.dtype)
    else:
        edges = jnp.asarray(bins, dtype=X.dtype)
        if max_distance is None:
            max_distance = edges[-1]
    n_bins = edges.shape[0] - 1

    def body(
        carry: tuple[Array, Array, Array], inp: tuple[Array, Array, Array]
    ) -> tuple[tuple[Array, Array, Array], None]:
        Xb, yb, start = inp
        d = block_dist(Xb)
        idx = jnp.searchsorted(edges, d, side="right") - 1
        idx = jnp.where(d == edges[-1], n_bins - 1, idx)
        keep = pair_mask(start) & (idx >= 0) & (idx < n_bins) & (d <= max_distance)
        seg = rearrange(jnp.where(keep, idx, n_bins), "b n -> (b n)")
        dy = jnp.abs(einx.subtract("b, n -> b n", yb, y))
        stat = dy**2 if estimator == "matheron" else jnp.sqrt(dy)

        def acc(v: Array) -> Array:
            flat = rearrange(jnp.where(keep, v, 0.0), "b n -> (b n)")
            return jax.ops.segment_sum(flat, seg, num_segments=n_bins + 1)[:n_bins]

        cnt, dsum, ssum = carry
        return (cnt + acc(jnp.ones_like(d)), dsum + acc(d), ssum + acc(stat)), None

    zeros = jnp.zeros(n_bins, X.dtype)
    (counts, dsum, ssum), _ = jax.lax.scan(
        body, (zeros, zeros, zeros), (Xp, yp, starts)
    )

    empty = counts == 0
    safe = jnp.where(empty, 1.0, counts)
    mean = ssum / safe
    if estimator == "matheron":
        gamma = 0.5 * mean
    else:
        gamma = mean**4 / (2.0 * (_CH_A + _CH_B / safe))
    midpoints = 0.5 * (edges[:-1] + edges[1:])
    return Variogram(
        bin_edges=edges,
        bin_centers=jnp.where(empty, midpoints, dsum / safe),
        gamma=jnp.where(empty, jnp.nan, gamma),
        counts=jnp.round(counts).astype(jnp.int32),
    )


def _with_params(
    kernel: AbstractStationaryKernel, lengthscale: Array, variance: Array
) -> AbstractStationaryKernel:
    return eqx.tree_at(
        lambda k: (k.lengthscale, k.variance), kernel, (lengthscale, variance)
    )


def _unit_range(kernel: AbstractStationaryKernel, dtype: jnp.dtype) -> Array:
    """Scaled distance at which the unit kernel's variogram reaches 95%."""
    r = jnp.geomspace(1e-3, 1e3, 4001, dtype=dtype)
    reached = 1.0 - kernel.shape(r**2) >= 0.95
    return jnp.where(jnp.any(reached), r[jnp.argmax(reached)], 1.0)


def fit_variogram(
    variogram: Variogram,
    kernel: AbstractStationaryKernel,
    *,
    nugget: bool = True,
    weights: Literal["cressie", "counts", "uniform"] = "cressie",
    max_steps: int = 500,
) -> tuple[AbstractStationaryKernel, Float[Array, ""]]:
    r"""Fit a stationary kernel plus a nugget to an empirical variogram.

    The model is $\gamma_\theta(h) = \tau^2 + \sigma^2 - k(h)$ with
    $k(0) = \sigma^2$ (``kernel.variance``) and nugget $\tau^2$. It minimises
    $\sum_b w_b(\hat\gamma_b - \gamma_\theta(h_b))^2$ over the logs of the
    lengthscale, variance and nugget with BFGS
    (`jax.scipy.optimize.minimize`). The weights are Cressie's
    $w_b = N_b / \gamma_\theta(h_b)^2$, the counts $w_b = N_b$, or uniform.
    Empty (NaN) bins are ignored.

    The sill starts at the mean of $\hat\gamma$ over the last third of the
    bins, the nugget at $\hat\gamma$ in the first bin (clipped to
    $[10^{-3}, 0.5]$ of the sill), and the range at the first bin centre
    where $\hat\gamma$ reaches 95% of the sill.

    Any `AbstractStationaryKernel` with a scalar ``lengthscale`` and a
    ``variance`` works (for example `Matern`, `RBF`, `RationalQuadratic`);
    its other fields stay fixed. Variograms on great-circle distance are fit
    the same way, in the variogram's distance units.

    Args:
        variogram: The `Variogram` to fit.
        kernel: The kernel to fit; its ``lengthscale`` and ``variance`` are
            replaced.
        nugget: Fit a nugget $\tau^2$ (else it is fixed at zero).
        weights: ``"cressie"``, ``"counts"`` or ``"uniform"``.
        max_steps: Maximum BFGS iterations.

    Returns:
        The fitted kernel and the nugget variance $\tau^2$.

    Raises:
        TypeError: If ``kernel`` is not an `AbstractStationaryKernel`.
        ValueError: On an ARD lengthscale or unknown ``weights``.

    Examples:
        >>> import einx
        >>> import jax.numpy as jnp
        >>> import kernellib as kl
        >>> h = jnp.linspace(0.05, 1.5, 30)
        >>> truth = kl.Matern(nu=1.5, lengthscale=0.3, variance=2.0)
        >>> gamma = (
        ...     0.1 + 2.0 - truth(einx.id("n -> n 1", h), jnp.zeros((1, 1)))[:, 0]
        ... )
        >>> v = kl.Variogram(
        ...     jnp.linspace(0.0, 1.55, 31), h, gamma, jnp.full(30, 100)
        ... )
        >>> fitted, tau2 = kl.fit_variogram(v, kl.Matern(nu=1.5))
        >>> round(float(fitted.lengthscale), 3), round(float(fitted.variance), 3)
        (0.3, 2.0)
        >>> round(float(tau2), 3)
        0.1
    """
    if not isinstance(kernel, AbstractStationaryKernel):
        raise TypeError(
            "fit_variogram needs an AbstractStationaryKernel; got "
            f"{type(kernel).__name__}."
        )
    if jnp.ndim(kernel.lengthscale) != 0:
        raise ValueError("fit_variogram needs a scalar (isotropic) lengthscale.")
    if weights not in ("cressie", "counts", "uniform"):
        raise ValueError(
            f"weights must be 'cressie', 'counts' or 'uniform'; got {weights!r}."
        )

    return _fit_variogram(variogram, kernel, nugget, weights, max_steps)


@eqx.filter_jit
def _fit_variogram(
    variogram: Variogram,
    kernel: AbstractStationaryKernel,
    nugget: bool,
    weights: str,
    max_steps: int,
) -> tuple[AbstractStationaryKernel, Float[Array, ""]]:
    """The jit-compiled weighted least-squares fit behind `fit_variogram`."""
    h = jnp.asarray(variogram.bin_centers)
    dtype = h.dtype
    valid = (variogram.counts > 0) & jnp.isfinite(variogram.gamma)
    counts = jnp.where(valid, variogram.counts, 0).astype(dtype)
    g_hat = jnp.where(valid, variogram.gamma, 0.0).astype(dtype)

    # Initial values from the shape of the empirical variogram.
    rank = jnp.cumsum(valid)
    n_valid = rank[-1]
    tail = valid & (3 * rank > 2 * n_valid)
    sill = jnp.sum(jnp.where(tail, g_hat, 0.0)) / jnp.maximum(jnp.sum(tail), 1)
    sill = jnp.where(sill > 0, sill, 1.0)
    first = jnp.argmax(valid)
    nug0 = jnp.clip(g_hat[first], 1e-3 * sill, 0.5 * sill)
    above = valid & (g_hat >= 0.95 * sill)
    last = jnp.argmax(valid & (rank == n_valid))
    range0 = jnp.where(jnp.any(above), h[jnp.argmax(above)], h[last])
    ell0 = jnp.maximum(range0, 1e-12) / _unit_range(kernel, dtype)
    var0 = sill - nug0 if nugget else sill

    # Fit on gamma / sill so the objective and its tolerance are scale-free.
    g_norm = g_hat / sill
    base = (
        counts if weights in ("cressie", "counts") else valid.astype(dtype)
    ) / jnp.maximum(jnp.sum(counts if weights != "uniform" else valid), 1)

    def model(theta: Array) -> Array:
        ell, var = jnp.exp(theta[0]), jnp.exp(theta[1])
        tau2 = jnp.exp(theta[2]) if nugget else jnp.zeros((), dtype)
        return tau2 + var - var * kernel.shape((h / ell) ** 2)

    def loss(theta: Array) -> Array:
        g = model(theta)
        w = base / jnp.where(valid, g, 1.0) ** 2 if weights == "cressie" else base
        return jnp.sum(jnp.where(valid, w * (g_norm - g) ** 2, 0.0))

    theta0 = jnp.log(jnp.stack([ell0, var0 / sill, nug0 / sill])).astype(dtype)
    if not nugget:
        theta0 = theta0[:2]
    tol = 1e-12 if dtype == jnp.float64 else 1e-6
    # ``minimize`` ignores ``tol``; the gradient tolerance goes in ``options``.
    res = minimize(
        loss, theta0, method="BFGS", options={"maxiter": max_steps, "gtol": tol}
    )
    theta = res.x
    fitted = _with_params(kernel, jnp.exp(theta[0]), sill * jnp.exp(theta[1]))
    tau2 = sill * jnp.exp(theta[2]) if nugget else jnp.zeros((), dtype)
    return fitted, tau2
