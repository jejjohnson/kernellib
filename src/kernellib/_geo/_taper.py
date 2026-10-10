r"""Covariance tapering into a sparse gaussx operator (GEO8).

A taper $T$ is a compactly supported correlation function. The tapered
covariance is the Hadamard product $K_{tap} = K \circ T$: by the Schur product
theorem it is positive semidefinite whenever $K$ and $T$ are, it keeps $K$'s
behaviour at short range (which dominates kriging weights), and it is exactly
zero beyond the taper's range. `tapered_operator` stores only those non-zeros,
as a `gaussx.SparseOperator` that `gaussx.SparseCholeskySolver` factorises.

References:
    Furrer, R., Genton, M. G. & Nychka, D. (2006). Covariance tapering for
    interpolation of large spatial datasets. *JCGS* 15(3), 502-523.
"""

from __future__ import annotations

from typing import Literal

import gaussx as gx
import jax.numpy as jnp
import lineax as lx
import numpy as np
from jaxtyping import Array, Float

from kernellib._einx import reduce, repeat
from kernellib._geo._great_circle import GreatCircleWendland
from kernellib._geo._models import Wendland
from kernellib._graph._neighbors import KNNGraph, radius_neighbors
from kernellib._kernels._base import AbstractKernel


__all__ = ["Tapered", "tapered_operator"]


class Tapered(AbstractKernel):
    r"""A covariance multiplied by a compactly supported taper.

    $$
    k_{tap}(x, x') = k(x, x')\, T(x, x')
    $$

    Positive semidefinite when both factors are (Schur product theorem), and
    zero wherever the taper is. Evaluation here is dense; `tapered_operator`
    builds the sparse Gram. The taper should have unit variance, so that the
    tapered covariance keeps $k$'s variance: `Wendland` in $\mathbb{R}^d$,
    $d \le 3$, or `GreatCircleWendland` on the sphere.

    Attributes:
        kernel: The covariance $k$.
        taper: The taper $T$, compactly supported with unit variance.

    Examples:
        >>> import jax.numpy as jnp
        >>> import kernellib as kl
        >>> k = kl.Tapered(kl.Matern(nu=1.5), kl.Wendland(lengthscale=1.0))
        >>> X = jnp.array([[0.0], [0.5], [1.5]])
        >>> k(X, X).round(4).tolist()
        [[1.0, 0.1472, 0.0], [0.1472, 1.0, 0.0], [0.0, 0.0, 1.0]]
    """

    kernel: AbstractKernel
    taper: AbstractKernel

    def __call__(
        self, X1: Float[Array, "N1 D"], X2: Float[Array, "N2 D"]
    ) -> Float[Array, "N1 N2"]:
        return self.kernel(X1, X2) * self.taper(X1, X2)

    def diag(self, X: Float[Array, "N D"]) -> Float[Array, " N"]:
        return self.kernel.diag(X) * self.taper.diag(X)

    def elwise(
        self, X1: Float[Array, "N D"], X2: Float[Array, "N D"]
    ) -> Float[Array, " N"]:
        return self.kernel.elwise(X1, X2) * self.taper.elwise(X1, X2)

    def pairwise(
        self, x: Float[Array, " D"], y: Float[Array, " D"]
    ) -> Float[Array, ""]:
        return self.kernel.pairwise(x, y) * self.taper.pairwise(x, y)

    @property
    def is_pointwise(self) -> bool:
        return self.kernel.is_pointwise and self.taper.is_pointwise

    @property
    def is_stationary(self) -> bool:
        return self.kernel.is_stationary and self.taper.is_stationary


def tapered_operator(
    kernel: AbstractKernel,
    X: Float[Array, "N D"],
    *,
    taper_range: float,
    taper: Literal["wendland2", "wendland4"] = "wendland2",
    metric: Literal["euclidean", "great_circle"] = "euclidean",
    radius: float = 1.0,
    degrees: bool = True,
    max_neighbors: int = 256,
    pattern: gx.SparsityPattern | None = None,
) -> gx.SparseOperator:
    r"""The tapered Gram $K(X, X) \circ T(X, X)$ as a sparse gaussx operator.

    The sparsity pattern (every pair closer than ``taper_range``, plus the
    diagonal) comes from `radius_neighbors` on the host, so ``X`` must be
    concrete; the values ``Tapered(kernel, T).elwise(X[rows], X[cols])`` are
    traced and differentiable in the kernel's hyperparameters. To reuse the
    pattern across optimisation steps, or under ``jax.jit``, pass the
    ``pattern`` of an earlier result: the neighbour search is then skipped
    and only the values are computed. The result holds about
    $n \bar m$ non-zeros, $\bar m$ the mean neighbour count within range.

    **Taper choice** (Furrer, Genton & Nychka 2006, Thm 2.2 and §3): the
    taper should be at least as smooth at the origin as the covariance. For a
    Matérn with smoothness $\nu$ in $d \le 3$, use ``"wendland2"`` (C²) for
    $\nu \le 1.5$ and ``"wendland4"`` (C⁴) for $\nu \le 2.5$. The rule is
    documented, not enforced. On the sphere (``metric="great_circle"``) the
    taper is `GreatCircleWendland`, which needs
    ``taper_range <= π · radius``.

    The one- and two-taper likelihood bias corrections (Kaufman, Schervish &
    Nychka 2008) are not included.

    Args:
        kernel: The covariance; with ``metric="great_circle"``, a kernel of
            ``(lon, lat)`` inputs (such as `GreatCircleExponential` or
            `Chordal`).
        X: Points, ``(N, D)`` with ``D <= 3``; ``(N, 2)`` ``(lon, lat)`` for
            ``metric="great_circle"``.
        taper_range: Support radius of the taper, in the units of ``X``
            (Euclidean) or of ``radius`` (great circle).
        taper: ``"wendland2"`` (C², default) or ``"wendland4"`` (C⁴).
        metric: ``"euclidean"`` (`Wendland` taper) or ``"great_circle"``
            (`GreatCircleWendland` taper).
        radius: Sphere radius for ``"great_circle"`` (`EARTH_RADIUS_KM` for
            kilometres). Ignored for ``"euclidean"``.
        degrees: Whether ``(lon, lat)`` are in degrees. Ignored for
            ``"euclidean"``.
        max_neighbors: Most neighbours a point may have within
            ``taper_range``. Dropping some would break positive
            definiteness, so more raises instead.
        pattern: A pattern from an earlier call on the same ``X`` and
            ``taper_range``, to skip the neighbour search.

    Returns:
        The symmetric sparse operator, tagged positive semidefinite. Shift it
        with ``add_diagonal`` and factorise it with
        `gaussx.SparseCholeskySolver`.

    Raises:
        ValueError: On an unknown ``taper`` or ``metric``, or when a point
            has more than ``max_neighbors`` neighbours within ``taper_range``
            (the message names the ``max_neighbors`` needed).

    Examples:
        >>> import gaussx as gx
        >>> import jax.numpy as jnp
        >>> import kernellib as kl
        >>> X = jnp.array([[0.0, 0.0], [0.1, 0.0], [0.25, 0.0], [1.0, 1.0]])
        >>> K = kl.tapered_operator(
        ...     kl.Matern(lengthscale=0.2, nu=1.5), X, taper_range=0.3
        ... )
        >>> K.pattern.nnz  # 4 diagonal entries and the 3 pairs within range
        7
        >>> round(float(K.as_matrix()[0, 1]), 4)
        0.3618
        >>> K = K.add_diagonal(jnp.full(4, 0.1))
        >>> round(float(gx.SparseCholeskySolver().logdet(K)), 4)
        0.254
    """
    if taper not in ("wendland2", "wendland4"):
        raise ValueError(f"taper must be 'wendland2' or 'wendland4', got {taper!r}.")
    order = 2 if taper == "wendland2" else 4
    if metric == "euclidean":
        T: AbstractKernel = Wendland(lengthscale=taper_range, order=order)
    elif metric == "great_circle":
        T = GreatCircleWendland(
            lengthscale=taper_range, radius=radius, degrees=degrees, order=order
        )
    else:
        raise ValueError(
            f"metric must be 'euclidean' or 'great_circle', got {metric!r}."
        )
    if pattern is None:
        pattern = _taper_pattern(X, taper_range, metric, radius, degrees, max_neighbors)
    values = Tapered(kernel, T).elwise(X[pattern.rows], X[pattern.cols])
    return gx.SparseOperator(values, pattern, tags=lx.positive_semidefinite_tag)


def _taper_pattern(
    X: Float[Array, "N D"],
    taper_range: float,
    metric: Literal["euclidean", "great_circle"],
    radius: float,
    degrees: bool,
    max_neighbors: int,
) -> gx.SparsityPattern:
    """The symmetric pattern of the pairs closer than ``taper_range``.

    Runs eagerly on the host. Raises if a row may have been truncated.
    """
    n = X.shape[0]
    if n == 1:
        return gx.SparsityPattern(
            np.zeros(1, dtype=np.int32),
            np.zeros(1, dtype=np.int32),
            (1, 1),
            symmetric=True,
        )
    # One more than allowed, so a row with too many neighbours shows up as a
    # full row.
    k = min(max_neighbors + 1, n - 1)

    def search(k: int) -> KNNGraph:
        return radius_neighbors(
            X,
            taper_range,
            max_neighbors=k,
            metric=metric,
            sphere_radius=radius,
            degrees=degrees,
        )

    g = search(k)
    if k > max_neighbors and bool(jnp.any(g.indices[:, -1] >= 0)):
        # Some row has more than max_neighbors: widen the search until no row
        # is full, to report the count needed.
        while True:
            k = min(2 * k, n - 1)
            wide = search(k).indices
            if k == n - 1 or not bool(jnp.any(wide[:, -1] >= 0)):
                break
        needed = int(jnp.max(reduce(wide >= 0, "n k -> n", "sum")))
        raise ValueError(
            f"max_neighbors={max_neighbors} truncates the taper's neighbourhoods: "
            f"some point has {needed} neighbours within taper_range={taper_range}. "
            f"Pass max_neighbors >= {needed}."
        )
    # Strictly inside the support: the taper is exactly 0 at the range.
    inside = np.asarray((g.indices >= 0) & (g.distances < taper_range))
    rows = np.asarray(repeat(jnp.arange(n), "n -> n k", k=k))[inside]
    cols = np.asarray(g.indices)[inside]
    return gx.SparsityPattern(rows, cols, (n, n), symmetric=True)
