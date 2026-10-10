r"""k-nearest-neighbour graphs: exact in JAX, or approximate through optional
backends.

The eigenmaps need a sparse neighbourhood graph. The default backend is an
exact brute-force search in JAX (``O(N^2 D)`` time, ``O(B N)`` memory for a
row block of ``B``). For large ``N``, ``backend="pynndescent"`` (extra
``kernellib[neighbors]``) builds an approximate graph with NN-descent, and
``backend="sklearn"`` (extra ``kernellib[sklearn]``) uses scikit-learn's
tree-based search. The optional backends are imported only when chosen, so
``import kernellib`` never loads them.

``metric="great_circle"`` or ``"chordal"`` takes ``(lon, lat)`` points. The
search then runs on their unit vectors in R³, whatever the backend: the
chordal distance $2R\sin(\theta/2)$ is monotone in the great-circle angle
$\theta$, so Euclidean neighbours of the unit vectors are exactly the
great-circle neighbours. The distances of the neighbours found are then
recomputed in the chosen metric from the unit vectors.
"""

from __future__ import annotations

from typing import Literal

import einx
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, Int

from kernellib._einx import einsum, rearrange, reduce, repeat
from kernellib.functional._geo import _safe_sqrt, lonlat_to_unit


__all__ = ["KNNGraph", "nearest_neighbors", "radius_neighbors"]

Backend = Literal["exact", "pynndescent", "sklearn"]
Metric = Literal["euclidean", "great_circle", "chordal"]


class KNNGraph(eqx.Module):
    """A directed k-nearest-neighbour graph, self excluded.

    Attributes:
        indices: Neighbour indices, ``(N, k)``, nearest first.
        distances: Distances to them, ``(N, k)``, in the search's metric
            (Euclidean by default).
    """

    indices: Int[Array, "N k"]
    distances: Float[Array, "N k"]

    @property
    def n_points(self) -> int:
        return self.indices.shape[0]

    @property
    def n_neighbors(self) -> int:
        return self.indices.shape[1]


def nearest_neighbors(
    X: Float[Array, "N D"],
    n_neighbors: int,
    *,
    backend: Backend = "exact",
    batch_size: int = 1024,
    random_state: int | None = None,
    metric: Metric = "euclidean",
    radius: float = 1.0,
    degrees: bool = True,
) -> KNNGraph:
    """The ``n_neighbors`` nearest other points of every point in ``X``.

    With a geo ``metric``, ``X`` is ``(N, 2)`` ``(lon, lat)`` and the search
    runs on the unit vectors in R³. Chordal distance is monotone in
    great-circle distance, so every backend returns exactly the great-circle
    neighbours, at no extra search cost.

    Args:
        X: Points, shape ``(N, D)``; ``(N, 2)`` ``(lon, lat)`` for a geo
            metric.
        n_neighbors: Neighbours per point, ``1 <= k < N``.
        backend: ``"exact"`` (JAX brute force), ``"pynndescent"`` (approximate,
            needs ``kernellib[neighbors]``) or ``"sklearn"`` (needs
            ``kernellib[sklearn]``).
        batch_size: Rows per block for the exact search.
        random_state: Seed for ``"pynndescent"``.
        metric: ``"euclidean"``, ``"great_circle"`` or ``"chordal"``.
        radius: Sphere radius for a geo metric; the distances are in its
            units (``1.0`` gives radians for ``"great_circle"``,
            `EARTH_RADIUS_KM` kilometres). Ignored for ``"euclidean"``.
        degrees: Whether ``(lon, lat)`` are in degrees (else radians).
            Ignored for ``"euclidean"``.

    Returns:
        The graph. A point is never its own neighbour; duplicates of it are
        (at distance zero). Whatever the backend, the distances have ``X``'s
        floating dtype, or the default float dtype when ``X`` is an integer
        array.

    Raises:
        ValueError: On an invalid ``n_neighbors``, ``backend`` or ``metric``,
            or a geo metric on ``X`` that is not ``(N, 2)``.
        ImportError: If an optional backend is not installed.

    Examples:
        >>> import jax.numpy as jnp
        >>> import kernellib as kl
        >>> X = jnp.array([[0.0], [1.0], [3.0], [7.0]])
        >>> g = kl.nearest_neighbors(X, 2)
        >>> g.indices.tolist()
        [[1, 2], [0, 2], [1, 0], [2, 1]]

        On the sphere, across the dateline (distances in km):

        >>> lonlat = jnp.array([[179.0, 0.0], [-179.0, 0.0], [170.0, 0.0]])
        >>> g = kl.nearest_neighbors(
        ...     lonlat, 1, metric="great_circle", radius=kl.EARTH_RADIUS_KM
        ... )
        >>> g.indices.tolist(), [round(float(d), 1) for d in g.distances[:, 0]]
        ([[1], [0], [0]], [222.4, 222.4, 1000.8])
    """
    n = X.shape[0]
    if not 1 <= n_neighbors < n:
        raise ValueError(
            f"n_neighbors must be in [1, {n - 1}] for {n} points, got {n_neighbors}."
        )
    if metric != "euclidean":
        dtype = _distance_dtype(X)
        U = _geo_embed(X, metric, degrees)
        knn = nearest_neighbors(
            U,
            n_neighbors,
            backend=backend,
            batch_size=batch_size,
            random_state=random_state,
        )
        Ua = repeat(U, "n d -> n k d", k=n_neighbors)
        distances = _sphere_distance(Ua, U[knn.indices], metric, radius)
        return KNNGraph(indices=knn.indices, distances=distances.astype(dtype))
    if backend == "exact":
        return _exact(jnp.asarray(X), n_neighbors, min(batch_size, n))
    # The optional backends compute in their own precision (pynndescent always
    # in float32); their distances are cast back to the exact backend's dtype.
    dtype = _distance_dtype(X)
    if backend == "pynndescent":
        # A writable copy: numba cannot type the read-only view of a JAX array.
        return _pynndescent(np.array(X), n_neighbors, random_state, dtype)
    if backend == "sklearn":
        return _sklearn(np.asarray(X), n_neighbors, dtype)
    raise ValueError(
        f"backend must be 'exact', 'pynndescent' or 'sklearn', got {backend!r}."
    )


def radius_neighbors(
    X: Float[Array, "N D"],
    radius: float,
    *,
    max_neighbors: int,
    backend: Backend = "exact",
    batch_size: int = 1024,
    random_state: int | None = None,
    metric: Metric = "euclidean",
    sphere_radius: float = 1.0,
    degrees: bool = True,
) -> KNNGraph:
    """The other points within ``radius`` of every point, at most
    ``max_neighbors`` of them.

    Shapes are static, so this is a ``max_neighbors``-nearest-neighbour search
    whose entries beyond ``radius`` are replaced by padding: index ``-1`` and
    distance ``inf``. Rows stay nearest first, so the padding comes last.
    `graph_from_neighbors` and `radius_graph` skip it.

    With a geo ``metric``, ``radius`` is a distance in that metric, in units
    of ``sphere_radius`` (km with ``sphere_radius=EARTH_RADIUS_KM``). The
    sphere's radius is named ``sphere_radius`` here, not ``radius`` as in
    `nearest_neighbors`, because ``radius`` is the search radius.

    Args:
        X: Points, shape ``(N, D)``; ``(N, 2)`` ``(lon, lat)`` for a geo
            metric.
        radius: Largest neighbour distance (inclusive).
        max_neighbors: Neighbours searched per point, ``1 <= k < N``.
        backend: See `nearest_neighbors`.
        batch_size: Rows per block for the exact search.
        random_state: Seed for ``"pynndescent"``.
        metric: ``"euclidean"``, ``"great_circle"`` or ``"chordal"``.
        sphere_radius: Sphere radius for a geo metric (`nearest_neighbors`'s
            ``radius``). Ignored for ``"euclidean"``.
        degrees: Whether ``(lon, lat)`` are in degrees (else radians).
            Ignored for ``"euclidean"``.

    Returns:
        The padded graph.

    Examples:
        >>> import jax.numpy as jnp
        >>> import kernellib as kl
        >>> X = jnp.array([[0.0], [1.0], [1.5], [9.0]])
        >>> g = kl.radius_neighbors(X, 1.0, max_neighbors=2)
        >>> g.indices.tolist()
        [[1, -1], [2, 0], [1, -1], [-1, -1]]
        >>> lonlat = jnp.array([[0.0, 0.0], [0.0, 1.0], [0.0, 4.0]])
        >>> g = kl.radius_neighbors(
        ...     lonlat,
        ...     250.0,
        ...     max_neighbors=2,
        ...     metric="great_circle",
        ...     sphere_radius=kl.EARTH_RADIUS_KM,
        ... )  # 1 degree of latitude is about 111 km
        >>> g.indices.tolist()
        [[1, -1], [0, -1], [-1, -1]]
    """
    # The threshold is applied to the distances in the chosen metric, so a
    # geo radius needs no conversion to a chordal one.
    knn = nearest_neighbors(
        X,
        max_neighbors,
        backend=backend,
        batch_size=batch_size,
        random_state=random_state,
        metric=metric,
        radius=sphere_radius,
        degrees=degrees,
    )
    inside = knn.distances <= radius
    return KNNGraph(
        indices=jnp.where(inside, knn.indices, -1),
        distances=jnp.where(inside, knn.distances, jnp.inf),
    )


def _geo_embed(
    X: Float[Array, "N 2"], metric: str, degrees: bool
) -> Float[Array, "N 3"]:
    """The unit vectors of ``(lon, lat)`` points, the space a geo search runs in.

    Euclidean distance between them is the chordal distance on the unit
    sphere, monotone in the great-circle angle, so their Euclidean neighbours
    are the great-circle (and chordal) neighbours.
    """
    if metric not in ("great_circle", "chordal"):
        raise ValueError(
            f"metric must be 'euclidean', 'great_circle' or 'chordal', got {metric!r}."
        )
    X = jnp.asarray(X)
    if X.ndim != 2 or X.shape[1] != 2:
        raise ValueError(
            f"metric={metric!r} needs (N, 2) (lon, lat) points, got shape {X.shape}."
        )
    return lonlat_to_unit(X.astype(_distance_dtype(X)), degrees=degrees)


def _sphere_distance(
    U: Float[Array, "*B 3"], V: Float[Array, "*B 3"], metric: str, radius: float
) -> Float[Array, "*B"]:
    r"""Great-circle or chordal distance between paired unit vectors.

    Recomputed from the vectors rather than converted from a backend's
    chordal distance: the latter loses precision for close points (and
    pynndescent's is float32). The angle is $2\operatorname{atan2}(\lVert u
    - v\rVert, \lVert u + v\rVert)$, as in `great_circle_distance`.
    """
    chord2 = einx.sum("... [d]", (U - V) ** 2)
    if metric == "chordal":
        return radius * _safe_sqrt(chord2)
    sum2 = einx.sum("... [d]", (U + V) ** 2)
    return radius * 2.0 * jnp.arctan2(_safe_sqrt(chord2), _safe_sqrt(sum2))


def _exact(X: Float[Array, "N D"], k: int, batch: int) -> KNNGraph:
    n = X.shape[0]
    n_blocks = -(-n // batch)
    pad = n_blocks * batch - n
    rows = jnp.arange(n_blocks * batch)
    X_pad = jnp.concatenate([X, jnp.zeros((pad, X.shape[1]), X.dtype)])
    sq_norms = reduce(X * X, "n d -> n", "sum")

    def block(start_rows: Int[Array, " B"]) -> tuple[Array, Array]:
        Xb = X_pad[start_rows]
        d2 = einx.add("b, n -> b n", reduce(Xb * Xb, "b d -> b", "sum"), sq_norms)
        d2 = d2 - 2.0 * einsum(Xb, X, "b d, n d -> b n")
        d2 = jnp.clip(d2, min=0.0)
        # A point is not its own neighbour.
        self_pair = einx.equal("b, n -> b n", start_rows, jnp.arange(n))
        d2 = jnp.where(self_pair, jnp.inf, d2)
        neg, idx = jax.lax.top_k(-d2, k)
        return idx, jnp.sqrt(-neg)

    idx, dist = jax.lax.map(block, rearrange(rows, "(b r) -> b r", r=batch))
    idx = rearrange(idx, "b r k -> (b r) k")[:n]
    dist = rearrange(dist, "b r k -> (b r) k")[:n]
    return KNNGraph(indices=idx, distances=dist)


def _distance_dtype(X: Array | np.ndarray) -> jnp.dtype:
    # X's floating dtype, or the default float dtype for an integer X (the
    # exact backend's sqrt promotes integers the same way); canonicalised, so
    # float64 becomes float32 when x64 is off.
    dtype = jnp.result_type(X)
    if not jnp.issubdtype(dtype, jnp.inexact):
        dtype = jnp.result_type(float)
    return dtype


def _drop_self(
    indices: np.ndarray, distances: np.ndarray, k: int, dtype: jnp.dtype
) -> KNNGraph:
    # The query set is the data, so each row normally starts with the point
    # itself; drop it by index (not by position, which ties can reorder).
    n = indices.shape[0]
    keep = einx.not_equal("n k, n -> n k", indices, np.arange(n))
    idx = np.stack([row[m][:k] for row, m in zip(indices, keep, strict=True)])
    dist = np.stack([row[m][:k] for row, m in zip(distances, keep, strict=True)])
    return KNNGraph(indices=jnp.asarray(idx), distances=jnp.asarray(dist, dtype))


def _pynndescent(
    X: np.ndarray, k: int, random_state: int | None, dtype: jnp.dtype
) -> KNNGraph:
    try:
        from pynndescent import NNDescent
    except ImportError as err:  # pragma: no cover - depends on the environment
        raise ImportError(
            "backend='pynndescent' needs pynndescent: pip install "
            "'kernellib[neighbors]'."
        ) from err
    index = NNDescent(X, n_neighbors=k + 1, random_state=random_state)
    indices, distances = index.neighbor_graph
    return _drop_self(np.asarray(indices), np.asarray(distances), k, dtype)


def _sklearn(X: np.ndarray, k: int, dtype: jnp.dtype) -> KNNGraph:
    try:
        from sklearn.neighbors import NearestNeighbors
    except ImportError as err:  # pragma: no cover - depends on the environment
        raise ImportError(
            "backend='sklearn' needs scikit-learn: pip install 'kernellib[sklearn]'."
        ) from err
    distances, indices = NearestNeighbors(n_neighbors=k + 1).fit(X).kneighbors(X)
    return _drop_self(indices, distances, k, dtype)
