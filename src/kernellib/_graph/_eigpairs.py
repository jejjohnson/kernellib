"""Sparse Laplacian eigenpairs through SciPy ARPACK (CPU, not traced)."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from jaxtyping import Array, Float

from kernellib._graph._neighbors import KNNGraph


def _sparse_adjacency(
    graph: KNNGraph, weighting: str, bandwidth: float | None
) -> sp.csr_matrix:
    idx = np.asarray(graph.indices)
    dist = np.asarray(graph.distances)
    n, k = idx.shape
    if weighting == "heat":
        sigma = float(np.median(dist)) if bandwidth is None else float(bandwidth)
        sigma = sigma if sigma > 0 else 1.0
        w = np.exp(-(dist**2) / (2.0 * sigma**2))
    else:
        w = np.ones_like(dist)
    W = sp.csr_matrix((w.ravel(), (np.repeat(np.arange(n), k), idx.ravel())), (n, n))
    W = W.maximum(W.T)
    W.setdiag(0.0)
    W.eliminate_zeros()
    return W


def _smallest_sparse(
    A: sp.spmatrix,
    degree: np.ndarray | None,
    n_components: int,
    drop_first: bool,
    seed: int,
) -> tuple[Float[Array, " n"], Float[Array, "N n"]]:
    n = A.shape[0]
    scale = (
        np.ones(n)
        if degree is None
        else 1.0 / np.sqrt(np.where(degree > 0, degree, 1.0))
    )
    S = sp.diags(scale) @ A @ sp.diags(scale)
    # Smallest eigenvalues of S are the largest of c I - S, with c a
    # Gershgorin bound on S's spectrum; ARPACK converges fast on those.
    c = float(np.max(np.abs(S).sum(axis=1)))
    k = n_components + int(drop_first)
    v0 = np.random.default_rng(seed).uniform(size=n)
    mu, U = spla.eigsh(c * sp.identity(n) - S, k=k, which="LA", v0=v0)
    order = np.argsort(c - mu)
    lam, U = (c - mu)[order], U[:, order]
    start = int(drop_first)
    Y = scale[:, None] * U[:, start:]
    return jnp.asarray(lam[start:]), jnp.asarray(Y)
