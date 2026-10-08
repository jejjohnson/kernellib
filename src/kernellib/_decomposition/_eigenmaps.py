r"""Graph embeddings: Laplacian eigenmaps, Schrödinger eigenmaps, LPP.

All three solve a generalised symmetric eigenproblem on a neighbourhood graph
with adjacency $W$, degree matrix $D$ and Laplacian $L = D - W$:

- **Laplacian eigenmaps** (Belkin & Niyogi, 2003) minimise
  $\sum_{ij} W_{ij}\|y_i - y_j\|^2 = 2\,\mathrm{tr}(Y^\top L Y)$ subject to
  $Y^\top D Y = I$: the smallest non-trivial solutions of $L y = \lambda D y$.
- **Schrödinger eigenmaps** (Czaja & Ehler, 2013) add a potential $V \succeq 0$
  that steers the embedding, $(L + \alpha V) y = \lambda D y$. A diagonal
  *barrier* potential pulls chosen points to the origin; a non-diagonal
  potential (itself a graph Laplacian) pulls chosen pairs together, e.g.
  points sharing a label (semi-supervised) or spatial neighbours in an image
  (Cahill, Czaja & Messinger, 2014).
- **Locality preserving projections** (He & Niyogi, 2003; in
  `_projections.py`) restrict $y = X a$ to
  be linear in the inputs, so new points can be embedded:
  $X^\top L X a = \lambda X^\top D X a$.

``constraint="identity"`` replaces $D$ by $I$ in the constraint. The dense
path (default) is JAX end to end and differentiable in the edge weights;
``eigen_solver="arpack"`` keeps the graph sparse and calls SciPy's ARPACK
(CPU, not traced), for graphs too large for an ``N x N`` matrix.

**Graph inputs.** The functions and estimators also take a sparse
`AbstractGraph`. Under the degree constraint $L y = \lambda D y$ is
$L_{\mathrm{sym}} u = \lambda u$ with $y = D^{-1/2} u$, so Laplacian eigenmaps
go through `laplacian_eigpairs` (``normalization="symmetric"``) and every
method there applies. Schrödinger eigenmaps solve
$D^{-1/2}(L + \alpha V) D^{-1/2} u = \lambda u$, a lineax composition, by
``"dense"``, ``"lanczos"`` or ``"arpack"``. `combine_potentials` weights
several potentials into one.

**Lanczos is checked.** Both ``"lanczos"`` paths verify the returned pairs:
the residual $r_i = \|A u_i - \lambda_i u_i\|$ of each must be at most
$\sqrt{\varepsilon}\, c$, with $\varepsilon$ the machine epsilon of the
dtype and $c \approx \lambda_{\max}(A)$ the spectral scale: the Gershgorin
bound for a Laplacian, the largest Ritz value for a Schrödinger operator
(``_check_eigpairs``).
A failing run is repeated with twice the Krylov oversample, up to
``_MAX_OVERSAMPLE``; if it still fails, a ``RuntimeError`` is raised rather
than a wrong embedding returned (use ``"arpack"``).

**LOBPCG.** ``eigen_solver="lobpcg"`` keeps the graph a
``jax.experimental.sparse.BCOO`` and runs ``lobpcg_standard`` on the
shift-and-flip $cI - S$ (LOBPCG finds the largest eigenpairs), on the
device and under ``jit``. It needs float64 and ``5 k < N``, and is checked
like Lanczos (``_lobpcg_eigenmap``).
"""

from __future__ import annotations

import contextlib
import dataclasses
from collections.abc import Iterator, Sequence
from contextvars import ContextVar
from typing import Literal

import einx
import equinox as eqx
import gaussx as gx
import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
import scipy.sparse as sp
from jax.experimental import sparse as jsparse
from jax.experimental.sparse.linalg import lobpcg_standard
from jaxtyping import Array, Float, Int, PRNGKeyArray

from kernellib._einx import rearrange, reduce
from kernellib._graph._construct import (
    adjacency_matrix,
    edge_weights,
    graph_from_neighbors,
)
from kernellib._graph._eigpairs import (
    _check_eigpairs,
    # Re-exported: the residual check lives with laplacian_eigpairs (#181).
    _eigpair_residuals as _eigpair_residuals,
    _lanczos_until_converged,
    _laplacian_eigpairs,
    _smallest_sparse,
    _sparse_adjacency,
    laplacian_eigpairs,
)
from kernellib._graph._laplacian import graph_laplacian
from kernellib._graph._neighbors import Backend, KNNGraph, nearest_neighbors
from kernellib._graph._types import AbstractGraph, Graph
from kernellib._kernels import RBF


__all__ = [
    "LaplacianEigenmaps",
    "SchrodingerEigenmaps",
    "barrier_potential",
    "combine_potentials",
    "label_potential",
    "laplacian_eigenmap",
    "schrodinger_eigenmap",
    "spatial_spectral_graph",
    "spatial_spectral_potential",
]

Constraint = Literal["degree", "identity"]
Solver = Literal["dense", "kronecker", "lanczos", "arpack", "lobpcg"]
# A potential: its diagonal, a dense matrix, or a (sparse) lineax operator.
Potential = Float[Array, " N"] | Float[Array, "N N"] | lx.AbstractLinearOperator
GraphLike = Float[Array, "N N"] | AbstractGraph

_SOLVERS = ("dense", "kronecker", "lanczos", "arpack", "lobpcg")
# Krylov margins of the "lanczos" paths, the defaults of laplacian_eigpairs: a
# run whose residuals fail `_check_eigpairs` is repeated with twice the
# margin, up to _MAX_OVERSAMPLE (three doublings), before raising.
_OVERSAMPLE = 200
_MAX_OVERSAMPLE = 1600
# Iteration cap of "lobpcg" (its `m`). LOBPCG stops early once every pair
# meets its eps-level tolerance, so the cap only costs on graphs that need it:
# a 10-NN Swiss roll converges in ~260 iterations at N = 1000, ~420 at
# N = 5000 and ~840 at N = 20 000 (#91).
_LOBPCG_MAX_ITER = 1000
# The estimator whose fit is running, if any: a solver failure then names the
# estimator and its ``eigen_solver`` rather than the function and ``method``.
_ESTIMATOR: ContextVar[str | None] = ContextVar("_ESTIMATOR", default=None)


# -- matrix level ------------------------------------------------------------


def _smallest_generalized(
    A: Float[Array, "N N"],
    degree: Float[Array, " N"] | None,
    n_components: int,
    drop_first: bool,
) -> tuple[Float[Array, " n"], Float[Array, "N n"]]:
    """Smallest solutions of ``A y = λ B y``, ``B = diag(degree)`` or ``I``."""
    if degree is None:
        lam, U = jnp.linalg.eigh(A)
        scale = jnp.ones(A.shape[0], A.dtype)
    else:
        scale = 1.0 / jnp.sqrt(jnp.where(degree > 0, degree, 1.0))
        scaled = einx.multiply("i, i j -> i j", scale, A)
        lam, U = jnp.linalg.eigh(einx.multiply("i j, j -> i j", scaled, scale))
    start = 1 if drop_first else 0
    stop = start + n_components
    return lam[start:stop], einx.multiply("N, N n -> N n", scale, U[:, start:stop])


def laplacian_eigenmap(
    W: GraphLike,
    n_components: int = 2,
    *,
    constraint: Constraint = "degree",
    drop_first: bool = True,
    method: Solver | None = None,
    key: PRNGKeyArray | None = None,
    max_iter: int = _LOBPCG_MAX_ITER,
) -> tuple[Float[Array, " n"], Float[Array, "N n"]]:
    r"""Laplacian eigenmap of a weighted graph: smallest solutions of
    $L y = \lambda D y$.

    A dense ``W`` with ``method=None`` (or ``"dense"``) is solved as before,
    by ``eigh`` of $D^{-1/2} L D^{-1/2}$. A graph, or any other ``method``,
    goes through `laplacian_eigpairs`: of $L_{\mathrm{sym}}$ under the degree
    constraint (then $y = D^{-1/2} u$), of $L$ under the identity constraint
    (where a face-connected `GridGraph` gets the Kronecker path).
    ``"lanczos"`` is checked as in `schrodinger_eigenmap` (by
    `laplacian_eigpairs`' own check), with the Gershgorin bound $c$ ($2$, or
    $\max_i (d_i + \sum_j |W_{ij}|)$ for the identity constraint, which is
    $2 \max_i d_i$ for non-negative weights) as the scale. ``"lobpcg"`` is
    as in `schrodinger_eigenmap` with $\alpha = 0$: LOBPCG on
    $2I - L_{\mathrm{sym}} = I + D^{-1/2} W D^{-1/2}$.

    Args:
        W: Symmetric adjacency matrix ``(N, N)``, or an `AbstractGraph`.
        n_components: Embedding dimension.
        constraint: ``"degree"`` ($Y^\top D Y = I$) or ``"identity"``.
        drop_first: Drop the trivial constant solution ($\lambda = 0$).
        method: ``"dense"``, ``"kronecker"`` (identity constraint only),
            ``"lanczos"``, ``"arpack"``, ``"lobpcg"``, or ``None`` for the
            default of `laplacian_eigpairs`.
        key: PRNG key; required by ``"lanczos"`` and ``"lobpcg"``, seeds
            ``"arpack"``.
        max_iter: Iteration cap of ``"lobpcg"``.

    Returns:
        ``(eigenvalues, embedding)``, shapes ``(n,)`` and ``(N, n)``.

    Raises:
        ValueError: For an invalid ``constraint`` or ``method``, or
            ``"kronecker"`` with the degree constraint; for ``"lobpcg"``
            without x64, without a ``key`` or on a graph with
            ``N <= 5 (n_components + drop_first)``.
        RuntimeError: If ``"lanczos"`` has not converged at the largest
            Krylov space it tries, or ``"lobpcg"`` within ``max_iter``.

    Examples:
        >>> import jax.numpy as jnp
        >>> import kernellib as kl
        >>> # A path graph 0 - 1 - 2 - 3: the first coordinate is monotone.
        >>> W = jnp.diag(jnp.ones(3), 1) + jnp.diag(jnp.ones(3), -1)
        >>> _, Y = kl.laplacian_eigenmap(W, 1)
        >>> y = Y[:, 0] * jnp.sign(Y[-1, 0])
        >>> bool(jnp.all(jnp.diff(y) > 0))
        True

        The same embedding of the sparse graph, by Lanczos:

        >>> import jax
        >>> g = kl.graph_from_adjacency(W)
        >>> _, Yg = kl.laplacian_eigenmap(
        ...     g, 1, method="lanczos", key=jax.random.key(0)
        ... )
        >>> bool(jnp.allclose(jnp.abs(Yg), jnp.abs(Y), atol=1e-6))
        True
    """
    _check_constraint(constraint)
    if not isinstance(W, AbstractGraph) and method in (None, "dense"):
        L = graph_laplacian(W)
        degree = reduce(W, "i j -> i", "sum") if constraint == "degree" else None
        return _smallest_generalized(L, degree, n_components, drop_first)
    _check_method(method)
    if constraint == "degree" and method == "kronecker":
        raise ValueError(
            "method='kronecker' solves L u = lambda u: it needs constraint='identity'."
        )
    if method == "lobpcg":
        lam, Y, _ = _lobpcg_eigenmap(
            W,
            None,
            n_components,
            constraint=constraint,
            drop_first=drop_first,
            key=key,
            max_iter=max_iter,
        )
        return lam, Y
    start = int(drop_first)
    normalization = "symmetric" if constraint == "degree" else "unnormalized"
    if method == "lanczos":
        lam, U = _laplacian_lanczos(W, n_components + start, normalization, key)
    else:
        lam, U = laplacian_eigpairs(
            W, n_components + start, normalization=normalization, method=method, key=key
        )
    lam, U = lam[start:], U[:, start:]
    if constraint == "identity":
        return lam, U
    return lam, einx.multiply("n, n k -> n k", _degree_scale(_degree(W)), U)


def schrodinger_eigenmap(
    W: GraphLike,
    potential: Potential,
    n_components: int = 2,
    *,
    alpha: float | Float[Array, ""] = 1.0,
    normalize_potential: bool = True,
    constraint: Constraint = "degree",
    drop_first: bool = True,
    method: Solver | None = None,
    key: PRNGKeyArray | None = None,
    max_iter: int = _LOBPCG_MAX_ITER,
) -> tuple[Float[Array, " n"], Float[Array, "N n"]]:
    r"""Schrödinger eigenmap: smallest solutions of $(L + \alpha V) y = \lambda D y$.

    A dense ``W`` with ``method=None`` (or ``"dense"``) is solved as before.
    A graph, or another ``method``, solves
    $D^{-1/2}(L + \alpha V) D^{-1/2} u = \lambda u$ (``"identity"``: without
    the $D^{-1/2}$), built as a lineax composition of the sparse operators:
    ``"dense"`` materialises it, ``"lanczos"`` runs `gaussx.eig` on it
    matrix-free (all Ritz pairs of a Krylov space ``200`` wider than needed,
    the smallest kept), ``"arpack"`` converts it to SciPy.

    ``"lanczos"`` checks every returned pair: its residual
    $\|A u - \lambda u\|$ must be at most $\sqrt{\varepsilon}\,c$, with $c$
    the largest Ritz value ($\approx \lambda_{\max}(A)$). A strong potential
    widens the spectrum and crowds the wanted eigenvalues near 0, where a
    fixed Krylov space may not converge; a failing run is repeated with twice
    the oversample (up to ``1600``), and then raises a ``RuntimeError``
    instead of returning wrong eigenpairs. ``"arpack"`` is the robust
    fallback.

    ``"lobpcg"`` keeps the graph a ``jax.experimental.sparse.BCOO`` and runs
    `jax.experimental.sparse.linalg.lobpcg_standard`, on the device and
    under ``jit``, with no SciPy and no ``N x N`` matrix (unless the
    potential is one). LOBPCG finds the *largest* eigenpairs, so it runs on
    the shift-and-flip $cI - S$ of $S = D^{-1/2}(L + \alpha V)D^{-1/2}$, with
    $c \ge \lambda_{\max}(S)$: $2$ for the Laplacian part plus $\alpha$
    times the Gershgorin bound of $D^{-1/2} V D^{-1/2}$. Then
    $\lambda_i = c - \theta_i$ and $y_i = D^{-1/2} u_i$. It needs float64
    (``jax_enable_x64``): the wanted eigenvalues sit within about $10^{-4}$
    of 0 in a spectrum of width $c$, and float32 LOBPCG cannot separate them
    from the bulk, so it returns a *wrong* embedding rather than an imprecise
    one. It needs ``5 (n_components + drop_first) < N``. It stops when every
    pair has converged to the dtype's eps, or after ``max_iter`` iterations.
    The iterations needed grow with $N$ and with $c$, since the relative gap
    of the wanted eigenvalues shrinks: a 10-NN Swiss roll takes about 420 at
    $N = 5000$ and 840 at $N = 20\,000$, but a barrier potential with
    ``alpha=1`` at $N = 5000$ ($c \approx 195$) takes several thousand, where
    ``"arpack"`` is the better choice. The result passes the same residual
    check as ``"lanczos"`` or raises a ``RuntimeError`` (raise
    ``max_iter``). No gradients.

    Args:
        W: Symmetric adjacency matrix ``(N, N)``, or an `AbstractGraph`.
        potential: $V$, either its diagonal ``(N,)`` (a barrier potential, see
            `barrier_potential`), a symmetric PSD matrix ``(N, N)`` (e.g.
            `label_potential`, `spatial_spectral_potential`), or a symmetric
            PSD lineax operator (e.g. the ``laplacian_operator()`` of
            `spatial_spectral_graph`, or `combine_potentials`).
        n_components: Embedding dimension.
        alpha: Weight of the potential.
        normalize_potential: Scale $\alpha$ by $\mathrm{tr}(L) / \mathrm{tr}(V)$
            (Cahill et al.), so ``alpha`` is relative to the graph's own scale
            and comparable across data sets.
        constraint: ``"degree"`` or ``"identity"``.
        drop_first: Drop the first solution. Keep ``True`` for non-diagonal
            potentials that are Laplacians (the constant vector stays a
            $\lambda = 0$ solution); a barrier potential removes it, so pass
            ``False`` to keep all solutions.
        method: ``"dense"``, ``"lanczos"``, ``"arpack"``, ``"lobpcg"``, or
            ``None`` (``"dense"``).
        key: PRNG key; required by ``"lanczos"`` and ``"lobpcg"``, seeds
            ``"arpack"``.
        max_iter: Iteration cap of ``"lobpcg"``.

    Returns:
        ``(eigenvalues, embedding)``.

    Raises:
        ValueError: For an invalid ``constraint`` or ``method`` (including
            ``"kronecker"``), ``"lanczos"`` or ``"lobpcg"`` without a
            ``key``, or ``"lobpcg"`` without x64 or with
            ``N <= 5 (n_components + drop_first)``.
        RuntimeError: If ``"lanczos"`` has not converged at the largest
            Krylov space it tries, or ``"lobpcg"`` within ``max_iter``.

    Examples:
        >>> import jax.numpy as jnp
        >>> import kernellib as kl
        >>> W = jnp.diag(jnp.ones(4), 1) + jnp.diag(jnp.ones(4), -1)
        >>> V = kl.barrier_potential(5, jnp.array([0]))  # pin node 0 to 0
        >>> _, Y = kl.schrodinger_eigenmap(W, V, 1, alpha=100.0, drop_first=False)
        >>> bool(jnp.abs(Y[0, 0]) < 0.05 * jnp.max(jnp.abs(Y)))
        True
        >>> g = kl.graph_from_adjacency(W)  # sparse: the same embedding
        >>> _, Yg = kl.schrodinger_eigenmap(g, V, 1, alpha=100.0, drop_first=False)
        >>> bool(jnp.allclose(jnp.abs(Yg), jnp.abs(Y), atol=1e-6))
        True
    """
    _check_constraint(constraint)
    if not isinstance(W, AbstractGraph) and method in (None, "dense"):
        L = graph_laplacian(W)
        V = _potential_matrix(potential)
        if normalize_potential:
            alpha = alpha * jnp.trace(L) / jnp.maximum(jnp.trace(V), 1e-30)
        degree = reduce(W, "i j -> i", "sum") if constraint == "degree" else None
        return _smallest_generalized(L + alpha * V, degree, n_components, drop_first)
    _check_method(method)
    if method == "kronecker":
        raise ValueError(
            "method='kronecker' does not apply to Schrödinger eigenmaps: the "
            "potential breaks the Kronecker structure. Use 'dense', 'lanczos', "
            "'arpack' or 'lobpcg'."
        )
    if method == "lobpcg":
        lam, Y, _ = _lobpcg_eigenmap(
            W,
            potential,
            n_components,
            alpha=alpha,
            normalize_potential=normalize_potential,
            constraint=constraint,
            drop_first=drop_first,
            key=key,
            max_iter=max_iter,
        )
        return lam, Y
    L, degree = _laplacian_and_degree(W)
    if normalize_potential:
        alpha = alpha * jnp.sum(degree) / jnp.maximum(_trace(potential), 1e-30)
    if method == "arpack":
        A = (_to_scipy(L) + float(alpha) * _to_scipy(potential)).tocsr()
        return _smallest_sparse(
            A,
            np.asarray(degree) if constraint == "degree" else None,
            n_components,
            drop_first,
            _seed(key),
        )
    scale = _degree_scale(degree) if constraint == "degree" else jnp.ones_like(degree)
    S = lx.DiagonalLinearOperator(scale)
    op = lx.TaggedLinearOperator(
        S @ (L + _potential_operator(potential) * alpha) @ S, lx.symmetric_tag
    )
    start = int(drop_first)
    stop = start + n_components
    if method == "lanczos":
        if key is None:
            raise ValueError("method='lanczos' needs a PRNG key.")
        # Krylov spaces are shift-invariant, and gaussx.eig fully
        # reorthogonalises: the smallest Ritz pairs of op are those of the
        # shifted cI - op that laplacian_eigpairs uses, so no shift is applied
        # (it changes no digit). What decides convergence is the Krylov
        # dimension, which _lanczos_until_converged grows until the residuals
        # pass.
        mv = lx.FunctionLinearOperator(op.mv, op.in_structure(), lx.symmetric_tag)

        def solve(oversample: int):
            mu, V = gx.eig(mv, rank=min(stop + oversample, op.in_size()), key=key)
            order = jnp.argsort(mu)[:stop]
            # The largest Ritz value converges first and bounds lambda_max(op)
            # from below: the spectral scale of the residual tolerance.
            return mu[order], V[:, order], jnp.max(mu)

        solver, hint = _solver_names("schrodinger_eigenmap", "lanczos")
        lam, U = _lanczos_until_converged(
            solve,
            op.mv,
            stop,
            op.in_size(),
            oversample=_OVERSAMPLE,
            max_oversample=_MAX_OVERSAMPLE,
            solver=solver,
            hint=hint,
        )
    else:
        lam, U = jnp.linalg.eigh(op.as_matrix())
    return lam[start:stop], einx.multiply("n, n k -> n k", scale, U[:, start:stop])


def combine_potentials(
    W: GraphLike,
    terms: Sequence[tuple[Potential, float | Float[Array, ""]]],
    *,
    normalize: bool = True,
) -> Potential:
    r"""Several Schrödinger potentials as one: $\sum_k \alpha_k c_k V_k$.

    With ``normalize`` each term gets Cahill's scaling,
    $c_k = \mathrm{tr}(L) / \mathrm{tr}(V_k)$, so every $\alpha_k$ is
    relative to the graph's own scale; otherwise $c_k = 1$. Pass the result
    to `schrodinger_eigenmap` (or `SchrodingerEigenmaps.fit`) with
    ``alpha=1.0, normalize_potential=False``. One term reproduces that
    function's own ``alpha`` normalisation. Each weight stays with its own
    potential, in the order given (the old ``'sspl'`` mode swapped them).

    Args:
        W: Symmetric adjacency matrix ``(N, N)``, or an `AbstractGraph`: the
            graph whose $\mathrm{tr}(L)$ (total degree) sets the scale.
        terms: ``(potential, alpha)`` pairs; a potential is a diagonal
            ``(N,)``, a matrix ``(N, N)`` or a lineax operator.
        normalize: Apply the trace normalisation.

    Returns:
        A diagonal ``(N,)`` if every potential is one, a matrix ``(N, N)``
        if the potentials are arrays, otherwise a lineax operator (the sum
        stays sparse).

    Raises:
        ValueError: If ``terms`` is empty.

    Examples:
        >>> import jax.numpy as jnp
        >>> import kernellib as kl
        >>> W = jnp.diag(jnp.ones(3), 1) + jnp.diag(jnp.ones(3), -1)  # tr L = 6
        >>> pins = kl.barrier_potential(4, jnp.array([0, 3]))  # tr V = 2
        >>> kl.combine_potentials(W, [(pins, 0.5)]).tolist()
        [1.5, 0.0, 0.0, 1.5]
        >>> labels = kl.label_potential(jnp.array([0, 0, -1, -1]))  # tr V = 2
        >>> V = kl.combine_potentials(W, [(pins, 0.5), (labels, 1.0)])
        >>> V.diagonal().tolist()
        [4.5, 3.0, 0.0, 1.5]
    """
    if not terms:
        raise ValueError("combine_potentials needs at least one (potential, alpha).")
    trace_l = jnp.sum(_degree(W))
    weights = [
        alpha * trace_l / jnp.maximum(_trace(V), 1e-30) if normalize else alpha
        for V, alpha in terms
    ]
    potentials = [V for V, _ in terms]
    if any(isinstance(V, lx.AbstractLinearOperator) for V in potentials):
        ops = [
            _potential_operator(V) * w for V, w in zip(potentials, weights, strict=True)
        ]
        total = ops[0]
        for op in ops[1:]:
            total = total + op
        return total
    arrays = [jnp.asarray(V) for V in potentials]
    if any(V.ndim == 2 for V in arrays):
        arrays = [_potential_matrix(V) for V in arrays]
    return sum(
        (w * V for V, w in zip(arrays[1:], weights[1:], strict=True)),
        start=weights[0] * arrays[0],
    )


def barrier_potential(
    n: int, indices: Int[Array, " m"], strength: float = 1.0
) -> Float[Array, " N"]:
    """Diagonal potential that is ``strength`` at ``indices`` and 0 elsewhere.

    In `schrodinger_eigenmap` it pulls the embedding of those points towards
    the origin (Czaja & Ehler, 2013).

    Examples:
        >>> import jax.numpy as jnp
        >>> import kernellib as kl
        >>> kl.barrier_potential(4, jnp.array([1, 3])).tolist()
        [0.0, 1.0, 0.0, 1.0]
    """
    return jnp.zeros(n).at[indices].set(strength)


def label_potential(
    labels: Int[Array, " N"], *, unlabeled: int = -1
) -> Float[Array, "N N"]:
    r"""Non-diagonal potential pulling together points that share a label.

    The Laplacian of the graph joining every pair of labelled points with the
    same label, $V = \sum_{i \sim j} (e_i - e_j)(e_i - e_j)^\top$: in
    `schrodinger_eigenmap` it adds $\alpha \sum_{i \sim j} \|y_i - y_j\|^2$ to
    the objective (semi-supervised Schrödinger eigenmaps). Points labelled
    ``unlabeled`` are free.

    Examples:
        >>> import jax.numpy as jnp
        >>> import kernellib as kl
        >>> kl.label_potential(jnp.array([0, 0, -1])).tolist()
        [[1.0, -1.0, 0.0], [-1.0, 1.0, 0.0], [0.0, 0.0, 0.0]]
    """
    labels = jnp.asarray(labels)
    same = einx.logical_and(
        "i j, i -> i j",
        einx.equal("i, j -> i j", labels, labels),
        labels != unlabeled,
    )
    A = same.astype(jnp.result_type(float)) * (1.0 - jnp.eye(labels.shape[0]))
    return graph_laplacian(A)


def spatial_spectral_potential(
    X: Float[Array, "N D"],
    coordinates: Float[Array, "N S"],
    n_neighbors: int = 4,
    *,
    bandwidth: float | None = None,
) -> Float[Array, "N N"]:
    r"""Spatial-spectral potential for images (Cahill, Czaja & Messinger, 2014).

    Joins each point to its ``n_neighbors`` nearest points in *space*
    (``coordinates``, e.g. pixel positions) and weights the edge by the heat
    kernel of their *spectral* distance $\|x_i - x_j\|$, returning that graph's
    Laplacian. Spatial neighbours that also look alike are pulled together.

    The spatial search is a dense ``O(N^2)`` k-NN on ``coordinates``, and the
    potential is an ``N x N`` matrix. For an image, whose pixel neighbours are
    known, use ``spatial_spectral_graph(X, grid_graph(image.shape[:2]))``
    instead: a sparse graph whose ``laplacian_operator()`` is the potential.

    Args:
        X: Spectral features, ``(N, D)``.
        coordinates: Spatial positions, ``(N, S)``.
        n_neighbors: Spatial neighbours per point.
        bandwidth: Heat-kernel width on spectral distances; ``None`` for the
            median spectral distance over the spatial edges.
    """
    graph = nearest_neighbors(coordinates, n_neighbors)
    rows = jnp.repeat(jnp.arange(X.shape[0]), n_neighbors)
    cols = rearrange(graph.indices, "n k -> (n k)")
    spectral = rearrange(
        jnp.sqrt(reduce((X[rows] - X[cols]) ** 2, "e d -> e", "sum")),
        "(n k) -> n k",
        k=n_neighbors,
    )
    W = adjacency_matrix(
        KNNGraph(indices=graph.indices, distances=spectral),
        weighting="heat",
        bandwidth=bandwidth,
    )
    return graph_laplacian(W)


def spatial_spectral_graph(
    X: Float[Array, "N D"],
    spatial_graph: AbstractGraph,
    *,
    bandwidth: float | Float[Array, ""] | None = None,
) -> Graph:
    r"""Spatial-spectral graph for images (Cahill, Czaja & Messinger, 2014).

    The topology of ``spatial_graph`` (e.g. `grid_graph` of the image, so
    adjacent pixels) with every edge reweighted by the heat kernel of its
    spectral distance, `edge_weights` with ``RBF(bandwidth)``. Its
    ``laplacian_operator()`` is the sparse Schrödinger potential that
    `spatial_spectral_potential` builds densely: pixels that are adjacent
    *and* alike are pulled together.

    Args:
        X: Spectral features, one row per node of ``spatial_graph``,
            ``(N, D)`` (an ``(H, W, D)`` cube flattened row-major).
        spatial_graph: The spatial neighbourhood, e.g. ``grid_graph((H, W))``.
        bandwidth: Heat-kernel width on spectral distances; ``None`` for the
            median spectral distance over the edges.

    Returns:
        A `Graph` with the topology of ``spatial_graph``.

    Examples:
        >>> import einx
        >>> import jax
        >>> import jax.numpy as jnp
        >>> import kernellib as kl
        >>> cube = jax.random.normal(jax.random.key(0), (8, 8, 3))  # H x W x bands
        >>> X = einx.id("h w d -> (h w) d", cube)
        >>> g = kl.spatial_spectral_graph(X, kl.grid_graph(cube.shape[:2]))
        >>> g.n_nodes, g.topology.n_edges
        (64, 112)
        >>> V = g.laplacian_operator()  # the potential, sparse
        >>> bool(jnp.allclose(V.mv(jnp.ones(64)), 0.0, atol=1e-6))  # a Laplacian
        True
    """
    top = spatial_graph.topology
    if bandwidth is None:
        diff = X[top.senders] - X[top.receivers]
        distances = jnp.sqrt(reduce(diff**2, "e d -> e", "sum"))
        bandwidth = jnp.median(distances) if distances.size else 1.0
    bandwidth = jnp.where(bandwidth > 0, bandwidth, 1.0)
    return edge_weights(spatial_graph, X, RBF(lengthscale=bandwidth))


def _laplacian_lanczos(
    W: GraphLike,
    n: int,
    normalization: Literal["unnormalized", "symmetric"],
    key: PRNGKeyArray | None,
) -> tuple[Float[Array, " n"], Float[Array, "N n"]]:
    """`laplacian_eigpairs` by Lanczos, its failure named for the eigenmap.

    The residual check, the Krylov growth and the (signed-safe Gershgorin)
    tolerance scale are `laplacian_eigpairs`'; this sets the margins to
    ``_OVERSAMPLE`` / ``_MAX_OVERSAMPLE`` and the message to
    `_solver_names`'.
    """
    solver, hint = _solver_names("laplacian_eigenmap", "lanczos")
    return _laplacian_eigpairs(
        W,
        n,
        normalization=normalization,
        method="lanczos",
        key=key,
        oversample=_OVERSAMPLE,
        max_oversample=_MAX_OVERSAMPLE,
        solver=solver,
        hint=hint,
    )


def _check_constraint(constraint: str) -> None:
    if constraint not in ("degree", "identity"):
        raise ValueError(
            f"constraint must be 'degree' or 'identity', got {constraint!r}."
        )


def _check_method(method: str | None) -> None:
    if method is not None and method not in _SOLVERS:
        raise ValueError(
            "method must be 'dense', 'kronecker', 'lanczos', 'arpack', 'lobpcg' "
            f"or None, got {method!r}."
        )


def _degree(W: GraphLike) -> Float[Array, " N"]:
    if isinstance(W, AbstractGraph):
        return W.degree()
    return reduce(jnp.asarray(W), "i j -> i", "sum")


def _degree_scale(degree: Float[Array, " N"]) -> Float[Array, " N"]:
    """$D^{-1/2}$, with 1 for isolated nodes (as `_smallest_generalized`)."""
    return 1.0 / jnp.sqrt(jnp.where(degree > 0, degree, 1.0))


def _laplacian_and_degree(
    W: GraphLike,
) -> tuple[lx.AbstractLinearOperator, Float[Array, " N"]]:
    """The unnormalised Laplacian as an operator, and the degrees."""
    if isinstance(W, AbstractGraph):
        return W.laplacian_operator(), W.degree()
    W = jnp.asarray(W)
    L = lx.MatrixLinearOperator(graph_laplacian(W), lx.symmetric_tag)
    return L, reduce(W, "i j -> i", "sum")


def _potential_matrix(V: Potential) -> Float[Array, "N N"]:
    if isinstance(V, lx.AbstractLinearOperator):
        return V.as_matrix()
    V = jnp.asarray(V)
    return jnp.diag(V) if V.ndim == 1 else V


def _potential_operator(V: Potential) -> lx.AbstractLinearOperator:
    if isinstance(V, lx.AbstractLinearOperator):
        return V
    V = jnp.asarray(V)
    if V.ndim == 1:
        return lx.DiagonalLinearOperator(V)
    return lx.MatrixLinearOperator(V, lx.symmetric_tag)


def _trace(V: Potential) -> Float[Array, ""]:
    """Trace of a potential, from its diagonal where the operator has one."""
    if isinstance(V, lx.AbstractLinearOperator):
        try:
            return jnp.sum(lx.diagonal(V))
        except NotImplementedError:  # e.g. a gaussx.KroneckerSum
            return gx.trace(V)
    V = jnp.asarray(V)
    return jnp.sum(V) if V.ndim == 1 else jnp.trace(V)


def _check_potential(potential: Potential, n: int) -> None:
    if isinstance(potential, lx.AbstractLinearOperator):
        shape: tuple[int, ...] = (potential.out_size(), potential.in_size())
        ok = shape == (n, n)
    else:
        shape = jnp.shape(potential)
        ok = shape in ((n,), (n, n))
    if not ok:
        raise ValueError(
            f"potential must have shape ({n},) or ({n}, {n}), got {shape}."
        )


def _check_graph(graph: GraphLike, n: int) -> None:
    n_nodes = graph.n_nodes if isinstance(graph, AbstractGraph) else graph.shape[0]
    if n_nodes != n:
        raise ValueError(f"graph must have one node per point ({n}), got {n_nodes}.")


def _to_scipy(A: Potential | lx.AbstractLinearOperator) -> sp.csr_matrix:
    """A diagonal, matrix or (sparse, composed) operator as a SciPy matrix."""
    if isinstance(A, lx.TaggedLinearOperator):
        return _to_scipy(A.operator)
    if isinstance(A, lx.AddLinearOperator):
        return (_to_scipy(A.operator1) + _to_scipy(A.operator2)).tocsr()
    if isinstance(A, lx.MulLinearOperator):
        return (float(A.scalar) * _to_scipy(A.operator)).tocsr()
    if isinstance(A, lx.DiagonalLinearOperator):
        return sp.diags(np.asarray(lx.diagonal(A))).tocsr()
    if isinstance(A, gx.SparseOperator):
        bcoo = A.to_bcoo()
        idx = np.asarray(bcoo.indices)
        return sp.csr_matrix(
            (np.asarray(bcoo.data), (idx[:, 0], idx[:, 1])), shape=bcoo.shape
        )
    if isinstance(A, gx.KroneckerSum):
        a, b = _to_scipy(A.A), _to_scipy(A.B)
        return (
            sp.kron(a, sp.identity(b.shape[0])) + sp.kron(sp.identity(a.shape[0]), b)
        ).tocsr()
    M = np.asarray(A.as_matrix() if isinstance(A, lx.AbstractLinearOperator) else A)
    return sp.diags(M).tocsr() if M.ndim == 1 else sp.csr_matrix(M)


def _seed(key: PRNGKeyArray | None) -> int:
    return 0 if key is None else int(jax.random.randint(key, (), 0, 2**31 - 1))


# -- solver failures ---------------------------------------------------------


@contextlib.contextmanager
def _fitting(estimator: str) -> Iterator[None]:
    """Solver failures inside this block name ``estimator`` (`_solver_names`).

    Decorates the estimators' ``fit`` (a ``contextmanager`` is also a
    decorator); the check message is built at trace time, so it holds under
    ``jit`` too.
    """
    token = _ESTIMATOR.set(estimator)
    try:
        yield
    finally:
        _ESTIMATOR.reset(token)


def _solver_param(function: str) -> tuple[str, str]:
    """Who chose the solver, and by which parameter, for error messages.

    The public function takes the solver as ``method``, the estimators as
    ``eigen_solver``; inside an estimator's `fit` (`_fitting`) a message
    names the estimator and its parameter, so the advice can be followed.
    """
    estimator = _ESTIMATOR.get()
    if estimator is None:
        return function, "method"
    return estimator, "eigen_solver"


def _solver_names(function: str, method: str) -> tuple[str, str]:
    """The ``solver`` and ``hint`` of a `_check_eigpairs` message."""
    name, param = _solver_param(function)
    return (
        f"{name}({param}={method!r})",
        f"Use {param}='arpack' (SciPy, CPU), or {param}='dense' for a small graph.",
    )


# -- LOBPCG ------------------------------------------------------------------

# A term of a potential: ("diag", (N,)), ("dense", (N, N)) or ("sparse", BCOO).
# A pytree with a static kind, so the jitted solve is cached across calls.
_Term = tuple[str, Array | jsparse.BCOO]


def _potential_terms(V: Potential) -> list[_Term]:
    """A potential as a sum of diagonal, dense and BCOO terms.

    Lineax compositions (sums, scalings, tags) are unpacked, so a sparse
    potential stays sparse. Summing the entrywise absolute values term by
    term over-estimates $|V|$, which keeps the Gershgorin bound a bound.
    """
    if isinstance(V, lx.TaggedLinearOperator):
        return _potential_terms(V.operator)
    if isinstance(V, lx.AddLinearOperator):
        return _potential_terms(V.operator1) + _potential_terms(V.operator2)
    if isinstance(V, lx.MulLinearOperator):
        return [
            (kind, value * V.scalar) for kind, value in _potential_terms(V.operator)
        ]
    if isinstance(V, gx.SparseOperator):
        return [("sparse", V.to_bcoo())]
    if isinstance(V, lx.DiagonalLinearOperator):
        V = lx.diagonal(V)
    elif isinstance(V, lx.MatrixLinearOperator):
        V = V.matrix
    elif isinstance(V, lx.AbstractLinearOperator):
        _, param = _solver_param("schrodinger_eigenmap")
        raise ValueError(
            f"{param}='lobpcg' takes a potential that is an array, or a "
            "lineax sum / scaling of diagonal, matrix and gaussx.SparseOperator "
            f"terms; got {type(V).__name__}. Pass V.as_matrix() or use "
            f"{param}='arpack'."
        )
    V = jnp.asarray(V)
    return [("diag" if V.ndim == 1 else "dense", V)]


def _term_mv(term: _Term, x: Array, *, absolute: bool = False) -> Array:
    """``V x`` (or ``|V| x``, entrywise) for a vector or an ``(N, k)`` block."""
    kind, V = term
    if isinstance(V, jsparse.BCOO):
        if absolute:
            V = jsparse.BCOO((jnp.abs(V.data), V.indices), shape=V.shape)
        return V @ x
    V = jnp.abs(V) if absolute else V
    if kind == "diag":
        return einx.multiply("N, N ... -> N ...", V, x)
    return V @ x


def _term_trace(term: _Term) -> Array:
    kind, V = term
    if isinstance(V, jsparse.BCOO):
        rows, cols = V.indices[:, 0], V.indices[:, 1]
        return jnp.sum(jnp.where(rows == cols, V.data, 0.0))
    return jnp.sum(V) if kind == "diag" else jnp.trace(V)


def _schrodinger_mv(
    A: Float[Array, "N N"] | jsparse.BCOO,
    degree: Float[Array, " N"],
    s: Float[Array, " N"],
    terms: list[_Term],
    alpha: Float[Array, ""],
    x: Array,
) -> Array:
    r"""$S x = s (D - W + \alpha V) s x$, for a vector or an ``(N, k)`` block."""
    z = einx.multiply("N, N ... -> N ...", s, x)
    out = einx.multiply("N, N ... -> N ...", degree, z) - A @ z
    for term in terms:
        out = out + alpha * _term_mv(term, z)
    return einx.multiply("N, N ... -> N ...", s, out)


@eqx.filter_jit
def _shifted_lobpcg(
    A: Float[Array, "N N"] | jsparse.BCOO,
    degree: Float[Array, " N"],
    s: Float[Array, " N"],
    terms: list[_Term],
    alpha: Float[Array, ""],
    c: Float[Array, ""],
    X0: Float[Array, "N k"],
    max_iter: int,
) -> tuple[Float[Array, " k"], Float[Array, "N k"], Int[Array, ""]]:
    """The ``k`` smallest eigenpairs of ``S``, ascending, as the top of
    ``cI - S``; and the iteration count."""

    def shifted(x: Array) -> Array:
        return c * x - _schrodinger_mv(A, degree, s, terms, alpha, x)

    theta, U, n_iter = lobpcg_standard(shifted, X0, m=max_iter)
    order = jnp.argsort(-theta)
    return c - theta[order], U[:, order], n_iter


def _lobpcg_eigenmap(
    W: GraphLike,
    potential: Potential | None,
    n_components: int,
    *,
    alpha: float | Float[Array, ""] = 0.0,
    normalize_potential: bool = True,
    constraint: Constraint = "degree",
    drop_first: bool = True,
    key: PRNGKeyArray | None,
    max_iter: int = _LOBPCG_MAX_ITER,
) -> tuple[Float[Array, " n"], Float[Array, "N n"], Int[Array, ""]]:
    r"""Smallest solutions of $(L + \alpha V) y = \lambda B y$ by LOBPCG.

    $B = D$ (``"degree"``) or $I$. With $s = \mathrm{diag}(B)^{-1/2}$ and
    $S = s (L + \alpha V) s$, the wanted pairs are the bottom of $S$, and
    `jax.experimental.sparse.linalg.lobpcg_standard` finds the top of a
    standard problem only, so it runs on the shift-and-flip $cI - S$:
    $\lambda_i = c - \theta_i$, $y_i = s\,u_i$. $c$ bounds $\lambda_{\max}(S)$
    (Weyl): $2$ for $s L s = L_{\mathrm{sym}}$ (``2 max(d)``, Gershgorin, for
    the identity constraint), plus $|\alpha|$ times the Gershgorin bound of
    $s V s$. The graph is a BCOO (`AbstractGraph.to_bcoo`), so memory is
    $O(N k + |E|)$, and the whole solve traces under ``jit``.

    The wanted eigenvalues sit near 0 of a spectrum of width $c$, so the
    relative gap is small: float32 cannot separate them from the bulk and
    returns a wrong embedding (#91), hence the float64 requirement; and the
    iteration count grows with $N$ and with $c$ (a strong potential).
    Shift-invert or a preconditioner would fix the latter, but
    ``lobpcg_standard`` has neither. ``tol`` stays at its default (dtype
    eps): the criterion is relative and scaled by $10 N$, so a
    "tighter-looking" ``1e-9`` is looser and stops early. The returned pairs
    are checked by `_check_eigpairs`. No gradients.

    Returns:
        ``(eigenvalues, embedding, n_iter)``.

    Raises:
        ValueError: Without ``jax_enable_x64``; without a ``key``; unless
            ``0 < 5 (n_components + drop_first) < N``; or for a potential
            operator it cannot unpack.
        RuntimeError: If the returned pairs have not converged (raise
            ``max_iter``).
    """
    name, param = _solver_param(
        "laplacian_eigenmap" if potential is None else "schrodinger_eigenmap"
    )
    if jax.dtypes.canonicalize_dtype(jnp.float64) != jnp.float64:
        raise ValueError(
            f"{param}='lobpcg' needs float64: the smallest graph eigenvalues "
            "crowd near 0 and float32 LOBPCG returns a wrong embedding (#91). "
            'Enable x64 with jax.config.update("jax_enable_x64", True), or use '
            f"{param}='dense' or 'arpack'."
        )
    if key is None:
        raise ValueError("method='lobpcg' needs a PRNG key.")
    n_nodes = W.n_nodes if isinstance(W, AbstractGraph) else jnp.shape(W)[0]
    k = n_components + int(drop_first)
    if not 0 < 5 * k < n_nodes:
        raise ValueError(
            f"{param}='lobpcg' needs 5 * {k} eigenpairs < N = {n_nodes} "
            "(a limit of jax.experimental.sparse.linalg.lobpcg_standard); "
            f"use {param}='dense' for a graph this small."
        )
    f64 = jnp.float64
    A: Float[Array, "N N"] | jsparse.BCOO
    if isinstance(W, AbstractGraph):
        bcoo = W.to_bcoo()
        A = jsparse.BCOO((bcoo.data.astype(f64), bcoo.indices), shape=bcoo.shape)
        degree = W.degree().astype(f64)
    else:
        A = jnp.asarray(W, f64)
        degree = reduce(A, "i j -> i", "sum")
    terms = [] if potential is None else _potential_terms(potential)
    terms = [
        (kind, V.astype(f64) if isinstance(V, jsparse.BCOO) else jnp.asarray(V, f64))
        for kind, V in terms
    ]
    alpha = jnp.asarray(alpha, f64)
    if terms and normalize_potential:
        trace_v = sum(_term_trace(term) for term in terms)
        alpha = alpha * jnp.sum(degree) / jnp.maximum(trace_v, 1e-30)
    if constraint == "degree":
        s, c = _degree_scale(degree), jnp.asarray(2.0, f64)
    else:
        s, c = jnp.ones_like(degree), 2.0 * jnp.max(degree)
    for term in terms:
        c = c + jnp.abs(alpha) * jnp.max(s * _term_mv(term, s, absolute=True))
    X0 = jax.random.normal(key, (n_nodes, k), f64)
    lam, U, n_iter = _shifted_lobpcg(A, degree, s, terms, alpha, c, X0, max_iter)
    lam, U = _check_eigpairs(
        lambda x: _schrodinger_mv(A, degree, s, terms, alpha, x),
        lam,
        U,
        c,
        solver=f"{name}({param}='lobpcg', max_iter={max_iter})",
        hint=f"Raise max_iter, or use {param}='arpack' (SciPy, CPU).",
    )
    start = int(drop_first)
    return lam[start:], einx.multiply("N, N n -> N n", s, U[:, start:]), n_iter


# -- estimators --------------------------------------------------------------


class _GraphEmbedding(eqx.Module):
    """Shared configuration: how the neighbourhood graph is built and solved."""

    def _graph(self, X: Float[Array, "N D"]) -> KNNGraph:
        return nearest_neighbors(
            X,
            self.n_neighbors,  # ty: ignore[unresolved-attribute]
            backend=self.neighbors_backend,  # ty: ignore[unresolved-attribute]
            random_state=self.random_state,  # ty: ignore[unresolved-attribute]
        )

    def _sparse_graph(self, knn: KNNGraph) -> Graph:
        """The k-NN graph as a `Graph`, weighted like `adjacency_matrix`."""
        return graph_from_neighbors(
            knn,
            weighting=self.weighting,  # ty: ignore[unresolved-attribute]
            bandwidth=self.bandwidth,  # ty: ignore[unresolved-attribute]
        )

    def _key(self) -> PRNGKeyArray:
        return jax.random.key(self.random_state or 0)  # ty: ignore[unresolved-attribute]


class LaplacianEigenmaps(_GraphEmbedding):
    r"""Laplacian eigenmaps (Belkin & Niyogi, 2003) of the k-NN graph of ``X``.

    ``fit(X)`` builds the graph (`nearest_neighbors`, `adjacency_matrix`) and
    solves $L y = \lambda D y$ (`laplacian_eigenmap`); ``fit(X, graph=g)``
    embeds a precomputed graph (an `AbstractGraph` or adjacency matrix)
    instead. The embedding is transductive: there is no ``transform`` for new
    points (use `LocalityPreservingProjections` for that).

    Attributes:
        n_components: Embedding dimension.
        n_neighbors: Neighbours per point.
        weighting: ``"heat"`` or ``"connectivity"`` edge weights.
        bandwidth: Heat-kernel width; ``None`` for the median neighbour
            distance.
        constraint: ``"degree"`` or ``"identity"``.
        neighbors_backend: ``"exact"``, ``"pynndescent"`` or ``"sklearn"``.
        eigen_solver: ``"dense"`` (JAX, differentiable), ``"arpack"``
            (sparse, SciPy, for large ``N``), ``"lanczos"`` (sparse, JAX),
            ``"lobpcg"`` (sparse BCOO, JAX, jittable; float64 only, see
            `schrodinger_eigenmap`) or ``"kronecker"`` (a face-connected
            `GridGraph` passed to `fit`, identity constraint).
        max_iter: Iteration cap of ``"lobpcg"``; ignored otherwise.
        random_state: Seed for the approximate neighbours, ARPACK, Lanczos
            and LOBPCG's initial block.
        embedding: ``(N, n_components)``, ``None`` before `fit`.
        eigenvalues: ``(n_components,)``, ``None`` before `fit`.
        graph: The `KNNGraph`, or the graph passed to `fit`; ``None`` before
            `fit`.
        n_iter: LOBPCG iterations taken (``max_iter`` when it hit the cap,
            which is normal on large graphs); ``None`` for the other solvers.

    Examples:
        >>> import jax.numpy as jnp
        >>> import kernellib as kl
        >>> t = jnp.linspace(0.0, 3.0, 60)
        >>> X = jnp.stack([jnp.cos(t), jnp.sin(t)], axis=1)  # an arc
        >>> le = kl.LaplacianEigenmaps(n_components=1, n_neighbors=4).fit(X)
        >>> y = le.embedding[:, 0] * jnp.sign(le.embedding[-1, 0])
        >>> bool(jnp.all(jnp.diff(y) > 0))  # unrolls the arc in order
        True
    """

    n_components: int = eqx.field(default=2, static=True)
    n_neighbors: int = eqx.field(default=10, static=True)
    weighting: Literal["heat", "connectivity"] = eqx.field(default="heat", static=True)
    bandwidth: float | None = None
    constraint: Constraint = eqx.field(default="degree", static=True)
    neighbors_backend: Backend = eqx.field(default="exact", static=True)
    eigen_solver: Solver = eqx.field(default="dense", static=True)
    max_iter: int = eqx.field(default=_LOBPCG_MAX_ITER, static=True)
    random_state: int | None = eqx.field(default=None, static=True)
    embedding: Float[Array, "N n"] | None = None
    eigenvalues: Float[Array, " n"] | None = None
    graph: KNNGraph | GraphLike | None = None
    n_iter: Int[Array, ""] | None = None

    def __check_init__(self) -> None:
        _check_common(self)

    @_fitting("LaplacianEigenmaps")
    def fit(
        self, X: Float[Array, "N D"], *, graph: GraphLike | None = None
    ) -> LaplacianEigenmaps:
        """Embed ``X``, on its k-NN graph or on ``graph``.

        Raises:
            ValueError: If ``graph`` does not have one node per row of ``X``.
        """
        if self.eigen_solver == "lobpcg":
            fitted: KNNGraph | GraphLike
            if graph is not None:
                _check_graph(graph, X.shape[0])
                fitted = target = graph
            else:
                fitted = self._graph(X)
                target = self._sparse_graph(fitted)
            lam, Y, n_iter = _lobpcg_eigenmap(
                target,
                None,
                self.n_components,
                constraint=self.constraint,
                key=self._key(),
                max_iter=self.max_iter,
            )
            return dataclasses.replace(
                self,
                embedding=Y,
                eigenvalues=lam,
                graph=fitted,
                n_iter=n_iter,
            )
        if graph is not None:
            _check_graph(graph, X.shape[0])
            lam, Y = laplacian_eigenmap(
                graph,
                self.n_components,
                constraint=self.constraint,
                method=self.eigen_solver,
                key=self._key(),
            )
            return dataclasses.replace(self, embedding=Y, eigenvalues=lam, graph=graph)
        knn = self._graph(X)
        if self.eigen_solver in ("lanczos", "kronecker"):
            lam, Y = laplacian_eigenmap(
                self._sparse_graph(knn),
                self.n_components,
                constraint=self.constraint,
                method=self.eigen_solver,
                key=self._key(),
            )
        elif self.eigen_solver == "arpack":
            W = _sparse_adjacency(knn, self.weighting, self.bandwidth)
            L = sp.diags(np.asarray(W.sum(axis=1)).ravel()) - W
            degree = np.asarray(W.sum(axis=1)).ravel()
            lam, Y = _smallest_sparse(
                L,
                degree if self.constraint == "degree" else None,
                self.n_components,
                True,
                self.random_state or 0,
            )
        else:
            W = adjacency_matrix(
                knn, weighting=self.weighting, bandwidth=self.bandwidth
            )
            lam, Y = laplacian_eigenmap(
                W, self.n_components, constraint=self.constraint
            )
        return dataclasses.replace(self, embedding=Y, eigenvalues=lam, graph=knn)


class SchrodingerEigenmaps(_GraphEmbedding):
    r"""Schrödinger eigenmaps (Czaja & Ehler, 2013) of the k-NN graph of ``X``.

    Laplacian eigenmaps steered by a potential $V$:
    $(L + \alpha V) y = \lambda D y$. Pass the potential to `fit`:

    - `barrier_potential` (diagonal): pins chosen points near the origin;
    - `label_potential`: pulls points sharing a label together
      (semi-supervised);
    - `spatial_spectral_potential`: pulls spatially adjacent, spectrally
      similar pixels together (hyperspectral imagery).

    With ``alpha = 0`` it is `LaplacianEigenmaps`. The graph and solver
    settings (``n_components``, ``n_neighbors``, ``weighting``, ``bandwidth``,
    ``constraint``, ``neighbors_backend``, ``eigen_solver``, ``max_iter``,
    ``random_state``) and ``n_iter`` are as in `LaplacianEigenmaps`, except
    that ``"kronecker"`` does not apply. Several potentials combine into one
    with `combine_potentials` (then ``alpha=1.0, normalize_potential=False``).

    Attributes:
        alpha: Weight of the potential.
        normalize_potential: Scale ``alpha`` by ``tr(L) / tr(V)``.
        drop_first: Drop the first solution (see `schrodinger_eigenmap`).
        embedding: ``(N, n_components)``, ``None`` before `fit`.
        eigenvalues: ``(n_components,)``, ``None`` before `fit`.
        graph: The `KNNGraph`, or the graph passed to `fit`; ``None`` before
            `fit`.

    Examples:
        >>> import jax
        >>> import jax.numpy as jnp
        >>> import kernellib as kl
        >>> X = jax.random.normal(jax.random.key(0), (80, 2))
        >>> labels = jnp.where(jnp.arange(80) < 10, 0, -1)  # 10 labelled
        >>> se = kl.SchrodingerEigenmaps(n_components=2, alpha=10.0)
        >>> se = se.fit(X, kl.label_potential(labels))
        >>> Y = se.embedding
        >>> import einx
        >>> def spread(Z):  # mean distance to the centroid
        ...     sq = einx.sum("n [k]", (Z - einx.mean("[n] k", Z)) ** 2)
        ...     return jnp.mean(jnp.sqrt(sq))
        >>> bool(spread(Y[:10]) < 0.2 * spread(Y))
        True
    """

    n_components: int = eqx.field(default=2, static=True)
    n_neighbors: int = eqx.field(default=10, static=True)
    alpha: float = 1.0
    normalize_potential: bool = eqx.field(default=True, static=True)
    drop_first: bool = eqx.field(default=True, static=True)
    weighting: Literal["heat", "connectivity"] = eqx.field(default="heat", static=True)
    bandwidth: float | None = None
    constraint: Constraint = eqx.field(default="degree", static=True)
    neighbors_backend: Backend = eqx.field(default="exact", static=True)
    eigen_solver: Solver = eqx.field(default="dense", static=True)
    max_iter: int = eqx.field(default=_LOBPCG_MAX_ITER, static=True)
    random_state: int | None = eqx.field(default=None, static=True)
    embedding: Float[Array, "N n"] | None = None
    eigenvalues: Float[Array, " n"] | None = None
    graph: KNNGraph | GraphLike | None = None
    n_iter: Int[Array, ""] | None = None

    def __check_init__(self) -> None:
        _check_common(self)

    @_fitting("SchrodingerEigenmaps")
    def fit(
        self,
        X: Float[Array, "N D"],
        potential: Potential,
        *,
        graph: GraphLike | None = None,
    ) -> SchrodingerEigenmaps:
        """Embed ``X`` under ``potential``, on its k-NN graph or on ``graph``.

        Args:
            X: Inputs, ``(N, D)``.
            potential: Diagonal ``(N,)``, matrix ``(N, N)`` or lineax
                operator.
            graph: Optional precomputed graph (an `AbstractGraph` or
                adjacency matrix) to embed instead of the k-NN graph of ``X``.

        Raises:
            ValueError: If the potential or ``graph`` does not match ``X``.
        """
        n = X.shape[0]
        if not isinstance(potential, lx.AbstractLinearOperator):
            potential = jnp.asarray(potential)
        _check_potential(potential, n)
        if self.eigen_solver == "lobpcg":
            fitted: KNNGraph | GraphLike
            if graph is not None:
                _check_graph(graph, n)
                fitted = target = graph
            else:
                fitted = self._graph(X)
                target = self._sparse_graph(fitted)
            lam, Y, n_iter = _lobpcg_eigenmap(
                target,
                potential,
                self.n_components,
                alpha=self.alpha,
                normalize_potential=self.normalize_potential,
                constraint=self.constraint,
                drop_first=self.drop_first,
                key=self._key(),
                max_iter=self.max_iter,
            )
            return dataclasses.replace(
                self,
                embedding=Y,
                eigenvalues=lam,
                graph=fitted,
                n_iter=n_iter,
            )
        if graph is not None:
            _check_graph(graph, n)
            lam, Y = self._embed(graph, potential, self.eigen_solver)
            return dataclasses.replace(self, embedding=Y, eigenvalues=lam, graph=graph)
        knn = self._graph(X)
        if self.eigen_solver in ("lanczos", "kronecker"):
            lam, Y = self._embed(self._sparse_graph(knn), potential, self.eigen_solver)
        elif self.eigen_solver == "arpack":
            W = _sparse_adjacency(knn, self.weighting, self.bandwidth)
            degree = np.asarray(W.sum(axis=1)).ravel()
            L = sp.diags(degree) - W
            V = _to_scipy(potential)
            alpha = self.alpha
            if self.normalize_potential:
                alpha = alpha * L.diagonal().sum() / max(V.diagonal().sum(), 1e-30)
            lam, Y = _smallest_sparse(
                (L + alpha * V).tocsr(),
                degree if self.constraint == "degree" else None,
                self.n_components,
                self.drop_first,
                self.random_state or 0,
            )
        else:
            W = adjacency_matrix(
                knn, weighting=self.weighting, bandwidth=self.bandwidth
            )
            lam, Y = self._embed(W, potential, None)
        return dataclasses.replace(self, embedding=Y, eigenvalues=lam, graph=knn)

    def _embed(
        self, W: GraphLike, potential: Potential, method: Solver | None
    ) -> tuple[Float[Array, " n"], Float[Array, "N n"]]:
        return schrodinger_eigenmap(
            W,
            potential,
            self.n_components,
            alpha=self.alpha,
            normalize_potential=self.normalize_potential,
            constraint=self.constraint,
            drop_first=self.drop_first,
            method=method,
            key=self._key(),
        )


def _check_common(model: eqx.Module) -> None:
    if model.n_components < 1:  # ty: ignore[unresolved-attribute]
        raise ValueError("n_components must be >= 1.")
    if model.n_neighbors < 1:  # ty: ignore[unresolved-attribute]
        raise ValueError("n_neighbors must be >= 1.")
    constraint = getattr(model, "constraint", "degree")
    _check_constraint(constraint)
    solver = getattr(model, "eigen_solver", "dense")
    if solver not in _SOLVERS:
        raise ValueError(
            "eigen_solver must be 'dense', 'kronecker', 'lanczos', 'arpack' or "
            f"'lobpcg', got {solver!r}."
        )
    if getattr(model, "max_iter", 1) < 1:
        raise ValueError("max_iter must be >= 1.")
