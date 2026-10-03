r"""Smallest Laplacian eigenpairs at every scale, and connected components.

`laplacian_eigpairs` picks the method from the graph's type:

- **dense** `eigh` for a `Graph` or an adjacency array;
- **Kronecker** for a face-connected `GridGraph`: a path Laplacian has the
  closed-form (DCT-II) eigenpairs $\lambda_k = 2 - 2\cos(\pi k / n)$, and
  Kronecker sums add eigenvalues, $(A \oplus B)(u \otimes v) =
  (\lambda + \mu)(u \otimes v)$, so the ``n`` smallest eigenpairs of an
  ``H x W`` grid need only a sort of ``H W`` scalars;
- **Lanczos** (``method="lanczos"``) in JAX for large sparse graphs: the
  smallest eigenvalues of $L$ are the largest of $cI - L$ for a Gershgorin
  bound $c \ge \lambda_{\max}$, where Lanczos converges fastest;
- **ARPACK** (``method="arpack"``) through SciPy, on the CPU, untraced, for
  graphs where Lanczos converges badly.
"""

from __future__ import annotations

from typing import Literal

import einx
import gaussx as gx
import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from jaxtyping import Array, Float, PRNGKeyArray
from scipy.sparse.csgraph import connected_components

from kernellib._einx import rearrange, reduce
from kernellib._graph._laplacian import graph_laplacian
from kernellib._graph._neighbors import KNNGraph
from kernellib._graph._types import AbstractGraph, GridGraph


__all__ = ["laplacian_eigpairs", "n_components_graph"]

EigpairMethod = Literal["dense", "kronecker", "lanczos", "arpack"]


def laplacian_eigpairs(
    graph: AbstractGraph | Float[Array, "N N"],
    n: int,
    *,
    normalization: Literal["unnormalized", "symmetric"] = "unnormalized",
    method: EigpairMethod | None = None,
    key: PRNGKeyArray | None = None,
    oversample: int = 200,
) -> tuple[Float[Array, " n"], Float[Array, "N n"]]:
    r"""The ``n`` smallest eigenpairs of a graph Laplacian.

    - ``"dense"``: `jnp.linalg.eigh` of the dense Laplacian.
    - ``"kronecker"``: the eigenpairs of each 1-D factor of a `GridGraph`,
      the ``n`` smallest of their sums, outer-product eigenvectors.
    - ``"lanczos"``: `gaussx.eig` of $cI - L$, with Krylov dimension
      ``n + oversample``.
    - ``"arpack"``: SciPy ``eigsh``, on the CPU. The only path that is not
      traced or differentiable.

    The default is chosen by type, not size: ``"kronecker"`` for a
    `GridGraph` with ``connectivity="face"`` and ``"unnormalized"``,
    ``"dense"`` otherwise. Pass ``"lanczos"`` or ``"arpack"`` for a large
    sparse graph.

    Lanczos needs a Krylov space well beyond ``n`` to converge at the bottom
    of a graph spectrum, where the eigenvalues are small and clustered:
    ``oversample`` (default 200) is that margin. If the eigenvalues are
    still inaccurate, raise it or use ``"arpack"``.

    Every eigenvector is ``ℓ²``-normalised and sign-fixed: its
    largest-magnitude entry is positive. A disconnected graph has one zero
    eigenvalue per connected component (`n_components_graph`); a caller
    dropping "the trivial" eigenvector must drop all of them. Within a
    repeated eigenvalue (common on grids) only the eigenspace is unique.

    Args:
        graph: A graph, or a dense symmetric adjacency matrix.
        n: Number of eigenpairs, ``1 <= n <= N``.
        normalization: ``"unnormalized"`` ($L = D - W$) or ``"symmetric"``
            ($I - D^{-1/2} W D^{-1/2}$).
        method: ``"dense"``, ``"kronecker"``, ``"lanczos"``, ``"arpack"``,
            or ``None`` for the type-based default.
        key: PRNG key; required by ``"lanczos"``, seeds ``"arpack"``.
        oversample: Extra Krylov dimension for ``"lanczos"``.

    Returns:
        ``(eigenvalues, eigenvectors)``: shapes ``(n,)`` ascending and
        ``(N, n)``.

    Raises:
        ValueError: For an invalid ``n``, ``method`` or ``normalization``,
            ``"kronecker"`` on anything but a face-connected unnormalised
            `GridGraph`, or ``"lanczos"`` without a ``key``.

    Examples:
        >>> import jax
        >>> import kernellib as kl
        >>> lam, U = kl.laplacian_eigpairs(kl.grid_graph((300, 300)), 4)
        >>> bool(abs(lam[0]) < 1e-5)  # the constant vector
        True
        >>> [round(v, 6) for v in lam[1:].tolist()]  # 2 - 2 cos(pi k / 300)
        [0.00011, 0.00011, 0.000219]
        >>> U.shape
        (90000, 4)
    """
    if normalization not in ("unnormalized", "symmetric"):
        raise ValueError(
            "normalization must be 'unnormalized' or 'symmetric', got "
            f"{normalization!r}."
        )
    n_nodes = graph.n_nodes if isinstance(graph, AbstractGraph) else graph.shape[0]
    if not 1 <= n <= n_nodes:
        raise ValueError(f"n must be in [1, {n_nodes}], got {n}.")
    if method is None:
        method = "kronecker" if _kronecker_ok(graph, normalization) else "dense"
    if method == "dense":
        lam, U = jnp.linalg.eigh(_dense_laplacian(graph, normalization))
        lam, U = lam[:n], U[:, :n]
    elif method == "kronecker":
        if not _kronecker_ok(graph, normalization):
            raise ValueError(
                "method='kronecker' needs a GridGraph with connectivity='face' "
                "and normalization='unnormalized'."
            )
        assert isinstance(graph, GridGraph)
        lam, U = _kronecker(graph, n)
    elif method == "lanczos":
        if key is None:
            raise ValueError("method='lanczos' needs a PRNG key.")
        lam, U = _lanczos(graph, n, normalization, oversample, key)
    elif method == "arpack":
        lam, U = _arpack(graph, n, normalization, key)
    else:
        raise ValueError(
            "method must be 'dense', 'kronecker', 'lanczos', 'arpack' or None, "
            f"got {method!r}."
        )
    return lam, _fix_signs(U)


def n_components_graph(graph: AbstractGraph | Float[Array, "N N"]) -> int:
    """Number of connected components (an isolated node is one).

    It is the multiplicity of the Laplacian's zero eigenvalue. Computed on
    the host from the topology in ``O(N + E)`` (a lattice is always
    connected).

    Args:
        graph: A graph, or a dense symmetric adjacency matrix.

    Returns:
        The number of components.

    Examples:
        >>> import kernellib as kl
        >>> g = kl.graph_from_edges([0, 2], [1, 3], 5)  # 0-1, 2-3, and 4
        >>> kl.n_components_graph(g)
        3
    """
    return int(_component_labels(graph).max()) + 1


def _component_labels(graph: AbstractGraph | Float[Array, "N N"]) -> np.ndarray:
    """Connected-component label of every node, ``0 .. c - 1``."""
    if isinstance(graph, GridGraph):
        return np.zeros(graph.n_nodes, dtype=np.int32)
    if isinstance(graph, AbstractGraph):
        top = graph.topology
        n = top.n_nodes
        adjacency = sp.coo_matrix(
            (np.ones(top.n_edges), (top.senders, top.receivers)), shape=(n, n)
        )
    else:
        adjacency = sp.coo_matrix(np.asarray(graph) != 0)
    _, labels = connected_components(adjacency, directed=False)
    return labels.astype(np.int32)


def _kronecker_ok(graph: object, normalization: str) -> bool:
    return (
        isinstance(graph, GridGraph)
        and graph.connectivity == "face"
        and normalization == "unnormalized"
    )


def _dense_laplacian(
    graph: AbstractGraph | Float[Array, "N N"], normalization: str
) -> Float[Array, "N N"]:
    if isinstance(graph, AbstractGraph):
        return graph.laplacian_operator(normalization).as_matrix()  # ty: ignore[invalid-argument-type]
    return graph_laplacian(jnp.asarray(graph), normalization)  # ty: ignore[invalid-argument-type]


def _kronecker_factors(
    op: lx.AbstractLinearOperator,
) -> list[lx.AbstractLinearOperator]:
    """The 1-D factors of a nested `gaussx.KroneckerSum`, in axis order."""
    if isinstance(op, gx.KroneckerSum):
        return [op.A, *_kronecker_factors(op.B)]
    return [op]


def _kronecker(
    graph: GridGraph, n: int
) -> tuple[Float[Array, " n"], Float[Array, "N n"]]:
    factors = [gx.eig(f) for f in _kronecker_factors(graph.laplacian_operator())]
    # All prod(n_k) eigenvalue sums, in row-major order like the nodes.
    total = factors[0][0]
    for lam_k, _ in factors[1:]:
        total = rearrange(einx.add("a, b -> a b", total, lam_k), "a b -> (a b)")
    order = jnp.argsort(total, stable=True)[:n]
    multi = jnp.unravel_index(order, graph.shape)
    U = factors[0][1][:, multi[0]]
    for (_, U_k), idx in zip(factors[1:], multi[1:], strict=True):
        U = rearrange(
            einx.multiply("a m, b m -> a b m", U, U_k[:, idx]), "a b m -> (a b) m"
        )
    return total[order], U


def _lanczos(
    graph: AbstractGraph | Float[Array, "N N"],
    n: int,
    normalization: str,
    oversample: int,
    key: PRNGKeyArray,
) -> tuple[Float[Array, " n"], Float[Array, "N n"]]:
    if isinstance(graph, AbstractGraph):
        L = graph.laplacian_operator(normalization)  # ty: ignore[invalid-argument-type]
        degree = graph.degree()
    else:
        L = lx.MatrixLinearOperator(_dense_laplacian(graph, normalization))
        degree = reduce(jnp.asarray(graph), "i j -> i", "sum")
    # Gershgorin: lambda_max(L) <= 2 max(degree); the symmetric form is <= 2.
    c = 2.0 * jnp.max(degree) if normalization == "unnormalized" else 2.0
    shifted = lx.FunctionLinearOperator(
        lambda v: c * v - L.mv(v),
        L.in_structure(),
        tags=(lx.symmetric_tag, lx.positive_semidefinite_tag),
    )
    rank = min(n + oversample, L.in_size())
    mu, V = gx.eig(shifted, rank=rank, key=key)
    top = jnp.argsort(-mu)[:n]
    return c - mu[top], V[:, top]


def _arpack(
    graph: AbstractGraph | Float[Array, "N N"],
    n: int,
    normalization: str,
    key: PRNGKeyArray | None,
) -> tuple[Float[Array, " n"], Float[Array, "N n"]]:
    if isinstance(graph, AbstractGraph):
        top = graph.topology
        w = np.asarray(graph.weights)
        N = top.n_nodes
        W = sp.coo_matrix(
            (
                np.concatenate([w, w]),
                (
                    np.concatenate([top.senders, top.receivers]),
                    np.concatenate([top.receivers, top.senders]),
                ),
            ),
            shape=(N, N),
        ).tocsr()
    else:
        W = sp.csr_matrix(np.asarray(graph))
        W.setdiag(0.0)
        W.eliminate_zeros()
    N = W.shape[0]
    if n >= N:
        raise ValueError(f"method='arpack' needs n < N = {N}; use method='dense'.")
    degree = np.asarray(W.sum(axis=1)).ravel()
    if normalization == "symmetric":
        # I - D^{-1/2} W D^{-1/2}, with a zero row for isolated nodes. Built
        # here rather than through _smallest_sparse's degree scaling, which
        # returns the generalised (D^{-1/2}-scaled) eigenvectors.
        connected = (degree > 0).astype(float)
        d_isqrt = connected / np.sqrt(np.where(degree > 0, degree, 1.0))
        L = sp.diags(connected) - sp.diags(d_isqrt) @ W @ sp.diags(d_isqrt)
    else:
        L = sp.diags(degree) - W
    seed = 0 if key is None else int(jax.random.randint(key, (), 0, 2**31 - 1))
    return _smallest_sparse(L, None, n, False, seed)


def _fix_signs(U: Float[Array, "N n"]) -> Float[Array, "N n"]:
    """Flip each column so its largest-magnitude entry is positive."""
    peak = rearrange(einx.argmax("[N] n", jnp.abs(U)), "1 n -> n")
    signs = jnp.sign(U[peak, jnp.arange(U.shape[1])])
    return einx.multiply("N n, n -> N n", U, jnp.where(signs == 0, 1.0, signs))


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
