"""k-nearest-neighbour graphs: exact in JAX, or approximate through optional
backends.

The eigenmaps need a sparse neighbourhood graph. The default backend is an
exact brute-force search in JAX (``O(N^2 D)`` time, ``O(B N)`` memory for a
row block of ``B``). For large ``N``, ``backend="pynndescent"`` (extra
``kernellib[neighbors]``) builds an approximate graph with NN-descent, and
``backend="sklearn"`` (extra ``kernellib[sklearn]``) uses scikit-learn's
tree-based search. The optional backends are imported only when chosen, so
``import kernellib`` never loads them.
"""

from __future__ import annotations

from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, Int

from kernellib._einx import rearrange


__all__ = ["KNNGraph", "nearest_neighbors"]

Backend = Literal["exact", "pynndescent", "sklearn"]


class KNNGraph(eqx.Module):
    """A directed k-nearest-neighbour graph, self excluded.

    Attributes:
        indices: Neighbour indices, ``(N, k)``, nearest first.
        distances: Euclidean distances to them, ``(N, k)``.
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
) -> KNNGraph:
    """The ``n_neighbors`` nearest other points of every point in ``X``.

    Args:
        X: Points, shape ``(N, D)``.
        n_neighbors: Neighbours per point, ``1 <= k < N``.
        backend: ``"exact"`` (JAX brute force), ``"pynndescent"`` (approximate,
            needs ``kernellib[neighbors]``) or ``"sklearn"`` (needs
            ``kernellib[sklearn]``).
        batch_size: Rows per block for the exact search.
        random_state: Seed for ``"pynndescent"``.

    Returns:
        The graph. A point is never its own neighbour; duplicates of it are
        (at distance zero).

    Raises:
        ValueError: On an invalid ``n_neighbors`` or ``backend``.
        ImportError: If an optional backend is not installed.

    Examples:
        >>> import jax.numpy as jnp
        >>> import kernellib as kl
        >>> X = jnp.array([[0.0], [1.0], [3.0], [7.0]])
        >>> g = kl.nearest_neighbors(X, 2)
        >>> g.indices.tolist()
        [[1, 2], [0, 2], [1, 0], [2, 1]]
    """
    n = X.shape[0]
    if not 1 <= n_neighbors < n:
        raise ValueError(
            f"n_neighbors must be in [1, {n - 1}] for {n} points, got {n_neighbors}."
        )
    if backend == "exact":
        return _exact(jnp.asarray(X), n_neighbors, min(batch_size, n))
    if backend == "pynndescent":
        return _pynndescent(np.asarray(X), n_neighbors, random_state)
    if backend == "sklearn":
        return _sklearn(np.asarray(X), n_neighbors)
    raise ValueError(
        f"backend must be 'exact', 'pynndescent' or 'sklearn', got {backend!r}."
    )


def _exact(X: Float[Array, "N D"], k: int, batch: int) -> KNNGraph:
    n = X.shape[0]
    n_blocks = -(-n // batch)
    pad = n_blocks * batch - n
    rows = jnp.arange(n_blocks * batch)
    X_pad = jnp.concatenate([X, jnp.zeros((pad, X.shape[1]), X.dtype)])
    sq_norms = jnp.sum(X * X, axis=1)

    def block(start_rows: Int[Array, " B"]) -> tuple[Array, Array]:
        Xb = X_pad[start_rows]
        d2 = jnp.sum(Xb * Xb, axis=1)[:, None] + sq_norms[None, :] - 2.0 * Xb @ X.T
        d2 = jnp.clip(d2, min=0.0)
        # A point is not its own neighbour.
        d2 = jnp.where(start_rows[:, None] == jnp.arange(n)[None, :], jnp.inf, d2)
        neg, idx = jax.lax.top_k(-d2, k)
        return idx, jnp.sqrt(-neg)

    idx, dist = jax.lax.map(block, rearrange(rows, "(b r) -> b r", r=batch))
    idx = rearrange(idx, "b r k -> (b r) k")[:n]
    dist = rearrange(dist, "b r k -> (b r) k")[:n]
    return KNNGraph(indices=idx, distances=dist)


def _drop_self(indices: np.ndarray, distances: np.ndarray, k: int) -> KNNGraph:
    # The query set is the data, so each row normally starts with the point
    # itself; drop it by index (not by position, which ties can reorder).
    n = indices.shape[0]
    keep = indices != np.arange(n)[:, None]
    idx = np.stack([row[m][:k] for row, m in zip(indices, keep, strict=True)])
    dist = np.stack([row[m][:k] for row, m in zip(distances, keep, strict=True)])
    return KNNGraph(indices=jnp.asarray(idx), distances=jnp.asarray(dist))


def _pynndescent(X: np.ndarray, k: int, random_state: int | None) -> KNNGraph:
    try:
        from pynndescent import NNDescent
    except ImportError as err:  # pragma: no cover - depends on the environment
        raise ImportError(
            "backend='pynndescent' needs pynndescent: pip install "
            "'kernellib[neighbors]'."
        ) from err
    index = NNDescent(X, n_neighbors=k + 1, random_state=random_state)
    indices, distances = index.neighbor_graph
    return _drop_self(np.asarray(indices), np.asarray(distances), k)


def _sklearn(X: np.ndarray, k: int) -> KNNGraph:
    try:
        from sklearn.neighbors import NearestNeighbors
    except ImportError as err:  # pragma: no cover - depends on the environment
        raise ImportError(
            "backend='sklearn' needs scikit-learn: pip install 'kernellib[sklearn]'."
        ) from err
    distances, indices = NearestNeighbors(n_neighbors=k + 1).fit(X).kneighbors(X)
    return _drop_self(indices, distances, k)
