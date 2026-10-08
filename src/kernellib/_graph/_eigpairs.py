r"""Smallest Laplacian eigenpairs at every scale, and connected components.

`laplacian_eigpairs` picks the method from the graph's type:

- **dense** `eigh` for a `Graph` or an adjacency array;
- **Kronecker** for a face-connected `GridGraph`: a path Laplacian has the
  closed-form (DCT-II) eigenpairs $\lambda_k = 2 - 2\cos(\pi k / n)$,
  $u_k[i] \propto \cos(\pi k (i + 1/2) / n)$, a cycle (periodic axis)
  $\lambda_k = 2 - 2\cos(2\pi k / n)$ with real Fourier modes, each scaled
  by its axis weight; Kronecker sums add eigenvalues,
  $(A \oplus B)(u \otimes v) = (\lambda + \mu)(u \otimes v)$, so
  the ``n`` smallest eigenpairs of an ``H x W`` grid need only a sort of
  ``H W`` scalars;
- **Lanczos** (``method="lanczos"``) in JAX for large sparse graphs: the
  smallest eigenvalues of $L$ are the largest of $cI - L$ for a Gershgorin
  bound $c \ge \lambda_{\max}$, where Lanczos converges fastest. The
  returned pairs are residual-checked (`_check_eigpairs`) and an unconverged
  run is repeated with a larger Krylov space (`_lanczos_until_converged`);
- **ARPACK** (``method="arpack"``) through SciPy, on the CPU, untraced, for
  graphs where Lanczos converges badly.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import einx
import equinox as eqx
import gaussx as gx
import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from jax.core import Tracer
from jaxtyping import Array, Float, Int, PRNGKeyArray
from scipy.sparse.csgraph import connected_components

from kernellib._einx import rearrange, reduce
from kernellib._graph._laplacian import graph_laplacian
from kernellib._graph._neighbors import KNNGraph
from kernellib._graph._types import AbstractGraph, GridGraph, _check_nonnegative


__all__ = ["laplacian_eigpairs", "n_components_graph"]

EigpairMethod = Literal["dense", "kronecker", "lanczos", "arpack"]

# Who failed and what to do instead, in a `laplacian_eigpairs` Lanczos error.
_LANCZOS_SOLVER = "laplacian_eigpairs(method='lanczos')"
_LANCZOS_HINT = "Use method='arpack' (SciPy, CPU), or method='dense' for a small graph."


def laplacian_eigpairs(
    graph: AbstractGraph | Float[Array, "N N"],
    n: int,
    *,
    normalization: Literal["unnormalized", "symmetric"] = "unnormalized",
    method: EigpairMethod | None = None,
    key: PRNGKeyArray | None = None,
    oversample: int = 200,
    max_oversample: int = 1600,
) -> tuple[Float[Array, " n"], Float[Array, "N n"]]:
    r"""The ``n`` smallest eigenpairs of a graph Laplacian.

    - ``"dense"``: `jnp.linalg.eigh` of the dense Laplacian.
    - ``"kronecker"``: the closed-form eigenpairs of each 1-D factor of a
      `GridGraph` (path or cycle, times its axis weight), the ``n`` smallest
      of their sums, outer-product eigenvectors. No eigensolver runs: the
      cost is a sort of the ``N`` sums, and float32 keeps full relative
      precision at the bottom of the spectrum.
    - ``"lanczos"``: `gaussx.eig` of $cI - L$, for the Gershgorin bound
      $c = \max_i (d_i + \sum_j |W_{ij}|) \ge \lambda_{\max}(L)$ ($2$ for
      ``"symmetric"``), with Krylov dimension ``n + oversample``,
      residual-checked (below).
    - ``"arpack"``: SciPy ``eigsh``, on the CPU. The only path that is not
      traced or differentiable.

    The default is chosen by type, not size: ``"kronecker"`` for a
    `GridGraph` with ``connectivity="face"`` and ``"unnormalized"``,
    ``"dense"`` otherwise. Pass ``"lanczos"`` or ``"arpack"`` for a large
    sparse graph.

    Lanczos needs a Krylov space well beyond ``n`` to converge at the bottom
    of a graph spectrum, where the eigenvalues are small and clustered:
    ``oversample`` (default 200) is that margin. ``"lanczos"`` checks what
    it returns: the residual $\|L u_i - \lambda_i u_i\|$ of every pair must
    be at most $\sqrt{\varepsilon}\, \max(c, 1)$, with $\varepsilon$ the
    machine epsilon of the dtype. A failing run is repeated with twice the
    oversample, up to ``max_oversample`` or until the Krylov space spans all
    ``N`` nodes; if the last run still fails, a ``RuntimeError`` is raised
    rather than wrong eigenpairs returned. Under ``jit`` there is a single
    run, at ``oversample``, and the check is attached to the outputs with
    `equinox.error_if`.

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
        oversample: Extra Krylov dimension of the first ``"lanczos"`` run.
        max_oversample: Largest extra Krylov dimension a ``"lanczos"``
            restart tries; ``max_oversample <= oversample`` means one run.

    Returns:
        ``(eigenvalues, eigenvectors)``: shapes ``(n,)`` ascending and
        ``(N, n)``.

    Raises:
        ValueError: For an invalid ``n``, ``method``, ``normalization`` or
            oversample, ``"kronecker"`` on anything but a face-connected
            unnormalised `GridGraph`, or ``"lanczos"`` without a ``key``.
        RuntimeError: If ``"lanczos"`` has not converged at the largest
            Krylov space it tries (under ``jit``, when the result is
            computed).

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
    return _laplacian_eigpairs(
        graph,
        n,
        normalization=normalization,
        method=method,
        key=key,
        oversample=oversample,
        max_oversample=max_oversample,
        solver=_LANCZOS_SOLVER,
        hint=_LANCZOS_HINT,
    )


def _laplacian_eigpairs(
    graph: AbstractGraph | Float[Array, "N N"],
    n: int,
    *,
    normalization: str,
    method: EigpairMethod | None,
    key: PRNGKeyArray | None,
    oversample: int,
    max_oversample: int,
    solver: str,
    hint: str,
) -> tuple[Float[Array, " n"], Float[Array, "N n"]]:
    """`laplacian_eigpairs`, with a Lanczos failure named by ``solver``.

    ``solver`` and ``hint`` name the failing call and what to do instead in
    the ``"lanczos"`` error (`_check_eigpairs`), so a caller that takes the
    solver by another parameter (the eigenmaps' ``eigen_solver``) can say so.
    """
    if oversample < 0:
        raise ValueError(f"oversample must be >= 0, got {oversample}.")
    if max_oversample < 0:
        raise ValueError(f"max_oversample must be >= 0, got {max_oversample}.")
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
        lam, U = _lanczos(
            graph,
            n,
            normalization,
            key,
            oversample=oversample,
            max_oversample=max_oversample,
            solver=solver,
            hint=hint,
        )
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

    It is the multiplicity of the Laplacian's zero eigenvalue, so an edge
    whose weight is zero does not connect its endpoints. Computed on the
    host in ``O(N + E)`` from the topology and the concrete weights (under
    a JAX transform, where the weights are traced, every topology edge
    counts).

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
    if isinstance(graph, GridGraph) and _all_nonzero(graph.axis_weights):
        # Every lattice edge has weight > 0 along some axis: connected.
        return np.zeros(graph.n_nodes, dtype=np.int32)
    if isinstance(graph, AbstractGraph):
        top = graph.topology
        n = top.n_nodes
        keep = _nonzero_mask(graph.weights)
        adjacency = sp.coo_matrix(
            (
                np.ones(int(keep.sum())),
                (top.senders[keep], top.receivers[keep]),
            ),
            shape=(n, n),
        )
    else:
        adjacency = sp.coo_matrix(np.asarray(graph) != 0)
    _, labels = connected_components(adjacency, directed=False)
    return labels.astype(np.int32)


def _nonzero_mask(weights: Float[Array, " E"]) -> np.ndarray:
    """Edges with a non-zero weight; all of them if the weights are traced."""
    try:
        return np.asarray(weights) != 0
    except jax.errors.TracerArrayConversionError:
        return np.ones(weights.shape, dtype=bool)


def _all_nonzero(weights: Float[Array, " d"]) -> bool:
    return bool(np.all(_nonzero_mask(weights)))


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


def _kronecker(
    graph: GridGraph, n: int
) -> tuple[Float[Array, " n"], Float[Array, "N n"]]:
    dtype = graph.axis_weights.dtype
    lams = [
        graph.axis_weights[k] * _lattice_eigvals(n_k, p_k, dtype)
        for k, (n_k, p_k) in enumerate(zip(graph.shape, graph.periodic, strict=True))
    ]
    # All prod(n_k) eigenvalue sums, in row-major order like the nodes.
    total = lams[0]
    for lam_k in lams[1:]:
        total = rearrange(einx.add("a, b -> a b", total, lam_k), "a b -> (a b)")
    order = jnp.argsort(total, stable=True)[:n]
    multi = jnp.unravel_index(order, graph.shape)
    # Only the n selected columns of each factor's eigenbasis are built.
    vecs = [
        _lattice_eigvecs(n_k, p_k, idx, dtype)
        for n_k, p_k, idx in zip(graph.shape, graph.periodic, multi, strict=True)
    ]
    U = vecs[0]
    for U_k in vecs[1:]:
        U = rearrange(einx.multiply("a m, b m -> a b m", U, U_k), "a b m -> (a b) m")
    return total[order], U


def _lattice_eigvals(n: int, periodic: bool, dtype: jnp.dtype) -> Float[Array, " n"]:
    r"""Eigenvalues of the unit-weight path (or cycle) Laplacian on ``n`` nodes.

    Path: $2 - 2\cos(\pi k / n) = 4 \sin^2(\pi k / 2n)$; cycle:
    $2 - 2\cos(2\pi k / n) = 4 \sin^2(\pi m / n)$ with the folded frequency
    $m = \min(k, n - k)$, for $k = 0, \dots, n - 1$. The sine form has no
    cancellation at small $k$, and the fold keeps the half-angle in
    $[0, \pi/2]$, so the bottom of the spectrum (which includes the cycle's
    aliased $k = n - 1$) keeps full relative precision in float32, and the
    two members of a degenerate cycle pair are bitwise equal.
    """
    k = jnp.arange(n)
    if periodic:
        k = jnp.minimum(k, n - k)
    half_angle = (jnp.pi / n if periodic else jnp.pi / (2 * n)) * k.astype(dtype)
    return 4.0 * jnp.sin(half_angle) ** 2


def _lattice_eigvecs(
    n: int, periodic: bool, k: Int[Array, " m"], dtype: jnp.dtype
) -> Float[Array, "n m"]:
    r"""Columns ``k`` of the orthonormal eigenbasis matching `_lattice_eigvals`.

    Path (DCT-II): $u_k[i] \propto \cos(\pi k (i + 1/2) / n)$. Cycle (real
    Fourier modes) with the folded frequency $m = \min(k, n - k)$:
    $\cos(2\pi m i / n)$ for $k \le n / 2$ and $\sin(2\pi m i / n)$ for
    $k > n / 2$, the partner of $k' = n - k$ in the same eigenspace. The
    phase is $2\pi \cdot \mathrm{ticks} / \mathrm{period}$ with ticks
    reduced modulo the period exactly, in integer arithmetic (`_mulmod`), so
    no float phase exceeds $2\pi$ and float32 keeps its precision at any
    ``n``.
    """
    i = jnp.arange(n)
    k = k.astype(i.dtype)
    if periodic:
        # phase = 2 pi (i m mod n) / n.
        period, steps, freq = n, i, jnp.minimum(k, n - k)
    else:
        # phase = pi k (2i + 1) / 2n = 2 pi (k (2i + 1) mod 4n) / 4n.
        period, steps, freq = 4 * n, 2 * i + 1, k
    ticks = _mulmod(steps, freq, period).astype(dtype)
    phase = (2 * jnp.pi / period) * ticks
    if periodic:
        U = einx.where("m, i m, i m -> i m", 2 * k > n, jnp.sin(phase), jnp.cos(phase))
        flat = (k == 0) | (2 * k == n)  # the constant and alternating modes
    else:
        U = jnp.cos(phase)
        flat = k == 0
    scale = jnp.where(flat, 1.0 / jnp.sqrt(n), jnp.sqrt(2.0 / n)).astype(dtype)
    return einx.multiply("i m, m -> i m", U, scale)


def _mulmod(a: Int[Array, " i"], b: Int[Array, " m"], p: int) -> Int[Array, "i m"]:
    r"""The outer product $(a_i b_m) \bmod p$, exact, for $0 \le a, b < p$.

    Horner's rule over base-$2^s$ digits of $a$,
    $r \leftarrow (r 2^s + d\, b) \bmod p$, with $s$ the largest shift for
    which $2 p\, 2^s$ fits the integer dtype: every intermediate is
    representable, so int32 (JAX's default) cannot overflow. With int64 or a
    small ``p`` this is a single digit, the plain product.
    """
    limit = jnp.iinfo(a.dtype).max
    if 4 * p > limit:
        raise ValueError(f"Lattice axis too long for {a.dtype} phases: period {p}.")
    shift = (limit // (2 * p)).bit_length() - 1  # 2 p 2^shift <= limit
    n_digits = max(1, -(-(p - 1).bit_length() // shift))
    r = jnp.zeros((a.shape[0], b.shape[0]), a.dtype)
    for j in reversed(range(n_digits)):
        digit = (a >> (j * shift)) & ((1 << shift) - 1)
        r = jnp.mod((r << shift) + einx.multiply("i, m -> i m", digit, b), p)
    return r


def _gershgorin_bound(
    graph: AbstractGraph | Float[Array, "N N"], normalization: str
) -> float | Float[Array, ""]:
    r"""A Gershgorin bound $c \ge \lambda_{\max}(L)$, safe for signed weights.

    $\max_i (d_i + \sum_j |W_{ij}|)$ for the unnormalised Laplacian, which
    is $2 \max_i d_i$ for non-negative weights (signed ones: a `mesh_graph`
    with ``on_negative="allow"``); $2$ for the symmetric one.
    """
    if normalization != "unnormalized":
        return 2.0
    if isinstance(graph, AbstractGraph):
        degree = graph.degree()
        spread = graph._degree(jnp.abs(graph.weights))
    else:
        degree = reduce(jnp.asarray(graph), "i j -> i", "sum")
        spread = reduce(jnp.abs(jnp.asarray(graph)), "i j -> i", "sum")
    return jnp.max(degree + spread)


def _lanczos(
    graph: AbstractGraph | Float[Array, "N N"],
    n: int,
    normalization: str,
    key: PRNGKeyArray,
    *,
    oversample: int,
    max_oversample: int,
    solver: str,
    hint: str,
) -> tuple[Float[Array, " n"], Float[Array, "N n"]]:
    """Lanczos on $cI - L$, residual-checked, with a growing Krylov space."""
    if isinstance(graph, AbstractGraph):
        L = graph.laplacian_operator(normalization)  # ty: ignore[invalid-argument-type]
    else:
        L = lx.MatrixLinearOperator(_dense_laplacian(graph, normalization))
    c = _gershgorin_bound(graph, normalization)
    shifted = lx.FunctionLinearOperator(
        lambda v: c * v - L.mv(v),
        L.in_structure(),
        tags=(lx.symmetric_tag, lx.positive_semidefinite_tag),
    )
    n_nodes = L.in_size()

    def solve(extra: int):
        mu, V = gx.eig(shifted, rank=min(n + extra, n_nodes), key=key)
        top = jnp.argsort(-mu)[:n]
        return c - mu[top], V[:, top], c

    return _lanczos_until_converged(
        solve,
        L.mv,
        n,
        n_nodes,
        oversample=oversample,
        max_oversample=max_oversample,
        solver=solver,
        hint=hint,
    )


# -- eigenpair verification --------------------------------------------------


def _eigpair_residuals(
    matvec: Callable[[Float[Array, " N"]], Float[Array, " N"]],
    lam: Float[Array, " n"],
    U: Float[Array, "N n"],
) -> Float[Array, " n"]:
    r"""Relative residuals $\|A u_i - \lambda_i u_i\| / \|u_i\|$ of eigenpairs.

    ``n`` matvecs. ``matvec`` is $v \mapsto A v$ for the symmetric matrix
    the pairs claim to diagonalise (not a shifted or flipped one).
    """
    AU = jax.vmap(matvec, in_axes=1, out_axes=1)(U)
    R = AU - einx.multiply("N n, n -> N n", U, lam)
    norm_r = jnp.sqrt(reduce(R**2, "N n -> n", "sum"))
    norm_u = jnp.sqrt(reduce(U**2, "N n -> n", "sum"))
    return norm_r / jnp.maximum(norm_u, jnp.finfo(U.dtype).tiny)


def _residual_tolerance(
    bound: float | Float[Array, ""], dtype: jnp.dtype
) -> Float[Array, ""]:
    r"""$\sqrt{\varepsilon}\, \max(c, 1)$: half the digits of the dtype.

    $c \approx \lambda_{\max}(A)$, a bound or a close estimate. A residual
    $r$ bounds the eigenvalue error by $r$, and by $r^2 / \mathrm{gap}$ once
    the pair is isolated, so a passing pair is accurate to about
    $\varepsilon c^2 / \mathrm{gap}$.
    """
    return jnp.sqrt(jnp.finfo(dtype).eps) * jnp.maximum(jnp.abs(bound), 1.0)


def _check_eigpairs(
    matvec: Callable[[Float[Array, " N"]], Float[Array, " N"]],
    lam: Float[Array, " n"],
    U: Float[Array, "N n"],
    bound: float | Float[Array, ""],
    *,
    solver: str,
    hint: str,
) -> tuple[Float[Array, " n"], Float[Array, "N n"]]:
    r"""Raise if any returned eigenpair of $A$ has not converged.

    The residual test of `_eigpair_residuals` against `_residual_tolerance`
    (``bound`` is $c \approx \lambda_{\max}(A)$). An iterative solver that
    returns the wrong pairs gives a wrong embedding with no other symptom,
    so this raises rather than warns. With concrete values it raises a
    ``RuntimeError``; under ``jit`` it attaches the same check to the
    outputs with `equinox.error_if`.

    Args:
        matvec: $v \mapsto A v$.
        lam: Claimed eigenvalues, ``(n,)``.
        U: Claimed eigenvectors, ``(N, n)``.
        bound: Spectral scale $c \approx \lambda_{\max}(A)$.
        solver: Name of the solver, for the message.
        hint: What to do instead, for the message.

    Returns:
        ``(lam, U)``, unchanged (under ``jit``, wrapped in the check).

    Raises:
        RuntimeError: If a residual exceeds the tolerance.
    """
    res = _eigpair_residuals(matvec, lam, U)
    tol = _residual_tolerance(bound, U.dtype)
    if isinstance(res, Tracer) or isinstance(tol, Tracer):
        message = (
            f"{solver} did not converge: eigenpair residuals above tolerance. {hint}"
        )
        # ~all(res <= tol), not any(res > tol): NaN compares False, so a
        # non-finite residual must fail the check, as it does eagerly.
        return eqx.error_if((lam, U), ~jnp.all(res <= tol), message)
    if not bool(jnp.all(res <= tol)):
        raise RuntimeError(
            f"{solver} did not converge: eigenpair residuals "
            f"||A u - lambda u|| = {np.asarray(res).tolist()} exceed the "
            f"tolerance {float(tol):.3g} (sqrt(eps) * {float(bound):.3g}), so "
            f"the eigenvalues {np.asarray(lam).tolist()} are not trustworthy. "
            f"{hint}"
        )
    return lam, U


def _lanczos_until_converged(
    solve: Callable[
        [int],
        tuple[Float[Array, " n"], Float[Array, "N n"], float | Float[Array, ""]],
    ],
    matvec: Callable[[Float[Array, " N"]], Float[Array, " N"]],
    n_wanted: int,
    n_nodes: int,
    *,
    oversample: int,
    max_oversample: int,
    solver: str,
    hint: str,
) -> tuple[Float[Array, " n"], Float[Array, "N n"]]:
    """Run ``solve(oversample)`` until `_check_eigpairs` passes.

    ``solve`` returns ``(lam, U, bound)`` from a Krylov space of dimension
    ``n_wanted + oversample`` (capped at ``n_nodes``). The oversample starts
    at ``oversample`` and doubles on failure, up to ``max_oversample`` or
    the full space; the last run is checked and raises, its message naming
    ``solver`` and suggesting ``hint``. (A Krylov space near the full space
    is no safe harbour: without breakdown detection Lanczos can lose the
    pairs there, which the check then reports.) Under ``jit`` there is one
    run, checked by `equinox.error_if`.
    """
    while True:
        lam, U, bound = solve(oversample)
        last = oversample >= max_oversample or n_wanted + oversample >= n_nodes
        res = _eigpair_residuals(matvec, lam, U)
        if last or isinstance(res, Tracer):
            return _check_eigpairs(matvec, lam, U, bound, solver=solver, hint=hint)
        if bool(jnp.all(res <= _residual_tolerance(bound, U.dtype))):
            return lam, U
        oversample = min(max(2 * oversample, 1), max_oversample)


def _arpack(
    graph: AbstractGraph | Float[Array, "N N"],
    n: int,
    normalization: str,
    key: PRNGKeyArray | None,
) -> tuple[Float[Array, " n"], Float[Array, "N n"]]:
    if isinstance(graph, AbstractGraph):
        top = graph.topology
        w = graph.weights
        if normalization == "symmetric":
            w = _check_nonnegative(w, "a 'symmetric' graph eigenbasis")
        w = np.asarray(w)
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
    rows = np.repeat(np.arange(n), k)
    cols = einx.id("n k -> (n k)", idx)
    W = sp.csr_matrix((einx.id("n k -> (n k)", w), (rows, cols)), (n, n))
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
    Y = einx.multiply("N, N n -> N n", scale, U[:, start:])
    return jnp.asarray(lam[start:]), jnp.asarray(Y)
