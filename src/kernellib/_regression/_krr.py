r"""Kernel ridge regression through any gaussx solver strategy."""

from __future__ import annotations

import dataclasses
from typing import Literal, TypeGuard

import einx
import equinox as eqx
import gaussx as gx
import jax
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray

from kernellib._einx import einsum
from kernellib._kernels import AbstractKernel
from kernellib._operators._bridge import to_cross_operator, to_operator
from kernellib._regression._base import AbstractEstimator, _check_targets


__all__ = ["KRR"]

Preconditioner = Literal["none", "nystrom", "rpcholesky"]
# One column's solve: the solution, the iteration count and whether it
# converged; the last two are None for a direct (non-iterative) solve.
_Solved = tuple[Float[Array, " N"], Int[Array, ""] | None, Bool[Array, ""] | None]


class KRR(AbstractEstimator):
    r"""Kernel ridge regression.

    Minimises $\frac{1}{n}\|y - K\alpha\|^2 + \lambda\, \alpha^\top K \alpha$,
    whose solution is

    $$
    (K + \lambda n I)\, \alpha = y, \qquad f(x) = k(x, X)\, \alpha.
    $$

    The ridge is scaled by ``n`` so that ``regularization`` means the same
    thing here as in `Falkon` (which with every point as a centre reduces to
    this) and does not need retuning as the data grows. The system is solved
    by ``solver``: `gaussx.DenseSolver` (Cholesky) by default, or any other
    strategy, e.g. ``gaussx.CGSolver()`` with ``implicit=True`` for a
    matrix-free ``O(N)``-memory solve.

    **Preconditioned CG.** Plain CG on $(K + \lambda n I)\alpha = y$ needs
    $O(\sqrt\kappa)$ iterations with $\kappa = (\lambda_1 + \lambda n) /
    \lambda n$, which explodes for small $\lambda$. With
    ``preconditioner="nystrom"`` or ``"rpcholesky"``, `fit` builds a rank
    ``preconditioner_rank`` preconditioner from $K$ (with the shift
    $\lambda n$ passed separately, so the ridge is never counted twice) and
    solves by `gaussx.PreconditionedCGSolver`. A rank near the effective
    dimension $d_{\mathrm{eff}}(\lambda n) = \sum_i \lambda_i / (\lambda_i +
    \lambda n)$ makes $\kappa = O(1)$. With ``implicit=True`` the kernel
    matrix is never formed, so this scales to ``n = 10^5`` and beyond.

    - ``"nystrom"``: `gaussx.NystromPreconditioner` (randomized Nyström,
      ``preconditioner_rank`` matvecs with $K$).
    - ``"rpcholesky"``: `gaussx.PartialCholeskyPreconditioner` with random
      pivoting (Díaz, Epperly, Frangella, Tropp & Webber, 2023),
      ``O(N r)`` kernel evaluations instead of ``r`` full matvecs: the
      better choice when kernel evaluations are expensive.

    ``solver=None`` (the default) means `gaussx.DenseSolver` without a
    preconditioner, as before, and preconditioned CG with one. An explicit
    ``solver`` is used as given, and cannot be combined with a
    preconditioner.

    **Quadratic penalties.** `fit` also takes a penalty operator $M$ and a
    mask $J$ of labelled points ($l = \operatorname{tr} J$), and then
    minimises

    $$
    \tfrac1l\|J(y - K\alpha)\|^2 + \lambda\,\alpha^\top K\alpha
    + \mu\,\alpha^\top K M K\alpha
    \quad\Longrightarrow\quad
    (JK + l\lambda I + l\mu\, M K)\,\alpha = J y,
    $$

    with $\mu$ = ``penalty_weight``. `hsic_penalty` makes this fair kernel
    learning (predictions independent of protected attributes) and
    `laplacian_penalty` makes it Laplacian-regularised least squares
    (smooth along a graph, which with a ``mask`` uses unlabelled points).
    How it is solved:

    - **No mask, a pure low-rank** $M = Q\,\mathrm{diag}(w)\,Q^\top$
      (`hsic_penalty` with a `Linear` kernel or ``approx``): the system is a
      rank-``r`` update of $K + \lambda n I$, solved by Woodbury with
      ``r + 1`` solves through ``solver``, so any strategy applies.
    - **Otherwise:** the system above is not symmetric, but it is
      $PK + l\lambda I$ with $P = J + l\mu M$ PSD, so its eigenvalues are
      real and at least $l\lambda$: it is as well conditioned as KRR. With
      ``implicit=False`` it is solved by dense LU; with ``implicit=True``,
      matrix-free by GMRES, with the tolerances (``rtol``, ``atol``,
      ``max_steps``) of ``solver`` when it has them. A ``preconditioner``
      applies to the Woodbury path only.

    **Iterative solves.** When `fit` solves iteratively (preconditioned CG,
    an explicit `gaussx.CGSolver` or `gaussx.PreconditionedCGSolver`, or the
    matrix-free GMRES of the penalised path), the fitted model records
    ``n_iter``, the iterations taken, and ``converged``, whether the
    tolerance was reached within the budget, one per target column for
    ``(N, C)`` targets; on the Woodbury path ``converged`` also requires the
    ``r`` solves against the penalty factor to have converged. Direct
    solves leave both ``None``. The preconditioned CG built from
    ``preconditioner`` stops at ``rtol = atol = tol`` or after
    ``max_steps`` iterations; an explicit ``solver`` keeps its own
    tolerances. With ``throw=True`` (the default) an iterative solve that
    hits its budget raises; with ``throw=False`` it returns the last
    iterate and ``converged=False``, so check it (and a validation loss).

    Attributes:
        kernel: The kernel.
        regularization: Ridge $\lambda > 0$.
        solver: A gaussx solver strategy; ``None`` to choose from
            ``preconditioner``.
        preconditioner: ``"none"`` (default), ``"nystrom"`` or
            ``"rpcholesky"``; the latter two need a ``key`` in `fit`.
        preconditioner_rank: Rank of the preconditioner (capped at ``N``).
        tol: Relative and absolute tolerance of the preconditioned CG solve.
        max_steps: Iteration budget of the preconditioned CG solve.
        throw: Raise when an iterative solve does not converge within its
            budget; ``False`` returns the last iterate instead.
        penalty_weight: Weight $\mu \ge 0$ of the penalty passed to `fit`.
        implicit: Build the kernel matrix matrix-free
            (`ImplicitKernelOperator`); needs a pointwise kernel and a concrete
            ``regularization``.
        X_train: Training inputs, ``None`` before `fit`.
        alpha: Weights, ``(N,)`` or ``(N, C)``, ``None`` before `fit`.
        n_iter: Iterations of an iterative solve, one per target column for
            ``(N, C)`` targets; ``None`` before `fit` or for a direct solve.
        converged: Whether the iterative solve reached its tolerance, per
            target column; ``None`` before `fit` or for a direct solve.

    Examples:
        >>> import jax.numpy as jnp
        >>> import kernellib as kl
        >>> X = jnp.linspace(0.0, 1.0, 20)[:, None]
        >>> y = jnp.sin(6.0 * X[:, 0])
        >>> model = kl.KRR(kl.RBF(lengthscale=0.2), regularization=1e-6).fit(X, y)
        >>> bool(jnp.max(jnp.abs(model.predict(X) - y)) < 1e-2)
        True

        Matrix-free, with a rank-10 Nyström preconditioner for CG:

        >>> import jax
        >>> pcg = kl.KRR(
        ...     kl.RBF(lengthscale=0.2),
        ...     regularization=1e-6,
        ...     implicit=True,
        ...     preconditioner="nystrom",
        ...     preconditioner_rank=10,
        ... ).fit(X, y, key=jax.random.key(0))
        >>> bool(jnp.max(jnp.abs(pcg.predict(X) - model.predict(X))) < 1e-3)
        True
        >>> bool(pcg.converged), bool(0 < pcg.n_iter < pcg.max_steps)
        (True, True)
    """

    kernel: AbstractKernel
    regularization: float | Float[Array, ""] = 1e-3
    solver: gx.AbstractSolverStrategy | None = None
    implicit: bool = eqx.field(default=False, static=True)
    penalty_weight: float | Float[Array, ""] = 0.0
    preconditioner: Preconditioner = eqx.field(default="none", static=True)
    preconditioner_rank: int = eqx.field(default=200, static=True)
    tol: float = eqx.field(default=1e-6, static=True)
    max_steps: int = eqx.field(default=1000, static=True)
    throw: bool = eqx.field(default=True, static=True)
    X_train: Float[Array, "N D"] | None = None
    alpha: Float[Array, " N"] | Float[Array, "N C"] | None = None
    n_iter: Int[Array, ""] | Int[Array, " C"] | None = None
    converged: Bool[Array, ""] | Bool[Array, " C"] | None = None

    def __check_init__(self) -> None:
        if self.preconditioner not in ("none", "nystrom", "rpcholesky"):
            raise ValueError(
                "preconditioner must be 'none', 'nystrom' or 'rpcholesky', got "
                f"{self.preconditioner!r}."
            )
        if self.preconditioner != "none" and self.solver is not None:
            raise ValueError(
                "Pass a solver or a preconditioner, not both: with a "
                "preconditioner, KRR solves by preconditioned CG."
            )
        if self.preconditioner_rank < 1:
            raise ValueError(
                f"preconditioner_rank must be >= 1, got {self.preconditioner_rank}."
            )
        if not self.tol > 0:
            raise ValueError(f"tol must be positive, got {self.tol}.")
        if self.max_steps < 1:
            raise ValueError(f"max_steps must be >= 1, got {self.max_steps}.")

    def fit(
        self,
        X: Float[Array, "N D"],
        y: Float[Array, " N"] | Float[Array, "N C"],
        *,
        penalty: lx.AbstractLinearOperator | None = None,
        mask: Bool[Array, " N"] | None = None,
        key: PRNGKeyArray | None = None,
    ) -> KRR:
        """Solve for the weights.

        Args:
            X: Training inputs, shape ``(N, D)``.
            y: Targets, ``(N,)`` or ``(N, C)``. Values at unlabelled points
                (``mask`` false) are ignored and may be NaN.
            penalty: Optional penalty operator $M$ on the training points,
                e.g. from `hsic_penalty` or `laplacian_penalty`. It must be
                positive semidefinite (otherwise the objective is unbounded);
                one not tagged symmetric is symmetrised, since only
                $(M + M^\top)/2$ enters the objective. A symmetric
                `gaussx.LowRankUpdate` takes the Woodbury path and must have
                a zero diagonal base.
            mask: Optional boolean mask of labelled points, shape ``(N,)``.
            key: PRNG key for the preconditioner; required when
                ``preconditioner`` is not ``"none"``, unused otherwise.

        Raises:
            ValueError: If ``y`` or ``mask`` does not match ``X``, a
                low-rank ``penalty`` has a non-zero diagonal base, or
                ``key`` is missing for a preconditioner.
        """
        y = _check_targets(X, y)
        if self.preconditioner != "none" and key is None:
            raise ValueError(
                f"preconditioner={self.preconditioner!r} needs a PRNG key in fit."
            )
        if mask is not None or not _no_penalty(penalty, self.penalty_weight):
            alpha, n_iter, converged = self._fit_penalised(X, y, penalty, mask, key)
        else:
            solver = self._solver(X, key)
            n = X.shape[0]
            K = to_operator(
                self.kernel, X, noise=self.regularization * n, implicit=self.implicit
            )
            alpha, n_iter, converged = _per_column(
                lambda col: self._solve(solver, K, col), y
            )
        return dataclasses.replace(
            self, X_train=X, alpha=alpha, n_iter=n_iter, converged=converged
        )

    def _solve(
        self,
        solver: gx.AbstractSolverStrategy,
        A: lx.AbstractLinearOperator,
        b: Float[Array, " N"],
    ) -> _Solved:
        """``A^{-1} b`` by ``solver``; for gaussx's CG strategies, the same
        lineax CG solve, with ``throw`` and the iteration statistics."""
        if isinstance(solver, gx.PreconditionedCGSolver):
            precond = solver.preconditioner
            if precond is None:  # built per solve, as gaussx does
                precond = gx.PartialCholeskyPreconditioner(
                    rank=solver.preconditioner_rank, shift=solver.shift
                )
            solver = gx.CGSolver(
                rtol=solver.rtol,
                atol=solver.atol,
                max_steps=solver.max_steps,
                preconditioner=precond,
            )
        if not isinstance(solver, gx.CGSolver):
            return solver.solve(A, b), None, None
        options: dict[str, lx.AbstractLinearOperator] = {}
        if solver.preconditioner is not None:
            P = solver.preconditioner.as_operator(A)
            if P is not None:
                # As in gaussx: lineax rejects tangents through ``options``,
                # and the CG solution does not depend on the preconditioner.
                dynamic, static = eqx.partition(P, eqx.is_array)
                options["preconditioner"] = eqx.combine(
                    jax.lax.stop_gradient(dynamic), static
                )
        solution = lx.linear_solve(
            A,
            b,
            lx.CG(rtol=solver.rtol, atol=solver.atol, max_steps=solver.max_steps),
            options=options,
            throw=self.throw,
        )
        return (
            solution.value,
            solution.stats["num_steps"],
            solution.result == lx.RESULTS.successful,
        )

    def _solver(
        self, X: Float[Array, "N D"], key: PRNGKeyArray | None
    ) -> gx.AbstractSolverStrategy:
        """The solver strategy for ``K + lambda n I``: as given, dense, or
        preconditioned CG with a preconditioner built from ``K`` once."""
        if self.preconditioner == "none":
            return gx.DenseSolver() if self.solver is None else self.solver
        assert key is not None  # checked in fit
        n = X.shape[0]
        K = to_operator(self.kernel, X, implicit=self.implicit)
        rank = min(self.preconditioner_rank, n)
        shift = self.regularization * n
        if self.preconditioner == "nystrom":
            precond = gx.NystromPreconditioner.from_operator(
                K, rank, shift=shift, key=key
            )
        else:
            precond = gx.PartialCholeskyPreconditioner.from_operator(
                K, rank, shift=shift, pivoting="random", key=key
            )
        return gx.PreconditionedCGSolver(
            preconditioner=precond,
            rtol=self.tol,
            atol=self.tol,
            max_steps=self.max_steps,
        )

    def _fit_penalised(
        self,
        X: Float[Array, "N D"],
        y: Float[Array, " N"] | Float[Array, "N C"],
        penalty: lx.AbstractLinearOperator | None,
        mask: Bool[Array, " N"] | None,
        key: PRNGKeyArray | None,
    ) -> tuple[
        Float[Array, " N"] | Float[Array, "N C"],
        Int[Array, ""] | Int[Array, " C"] | None,
        Bool[Array, ""] | Bool[Array, " C"] | None,
    ]:
        """The weights, iteration counts and convergence flags of a
        penalised (or masked) fit; the latter two ``None`` for a direct solve."""
        n = X.shape[0]
        mu = jnp.asarray(0.0 if penalty is None else self.penalty_weight)
        if mask is None:
            J = jnp.ones(n, dtype=bool)
        else:
            J = jnp.asarray(mask, dtype=bool)
            if J.shape != (n,):
                raise ValueError(f"mask must have shape ({n},), got {J.shape}.")
            J = _require(jnp.any(J), "mask selects no labelled points.", J)
        Jf = J.astype(X.dtype)
        n_lab = jnp.sum(Jf) if mask is not None else n
        # Unlabelled targets are ignored; zero them so a NaN placeholder
        # cannot leak through J y.
        Jy = (
            jnp.where(J, y, 0.0)
            if y.ndim == 1
            else einx.where("n, n c, -> n c", J, y, 0.0)
        )

        if mask is None and _is_low_rank(penalty):
            # A rank-r update of K + lambda n I: its solves are the plain
            # KRR solves, so they take the same (preconditioned) solver.
            return self._fit_woodbury(X, y, penalty, n * mu, self._solver(X, key))

        K_op = to_operator(self.kernel, X, implicit=self.implicit)
        M_mv = _symmetric_mv(penalty)
        lam, reg = self.regularization, mu

        if not self.implicit:
            K = K_op.as_matrix()
            MK = jax.vmap(M_mv, in_axes=1, out_axes=1)(K)
            B = (
                einx.multiply("i, i j -> i j", Jf, K)
                + n_lab * lam * jnp.eye(n, dtype=K.dtype)
                + n_lab * reg * MK
            )
            return jnp.linalg.solve(B, Jy), None, None

        # Matrix-free: GMRES on the non-symmetric system itself. B = P K + cI
        # with P = J + l mu M PSD is similar to a PSD matrix plus cI, so its
        # eigenvalues are real and >= c = l lambda: KRR's conditioning. The
        # symmetric normal form K B would square the conditioning of K.
        def system_mv(v: Float[Array, " N"]) -> Float[Array, " N"]:
            Kv = K_op.mv(v)
            return Jf * Kv + n_lab * lam * v + n_lab * reg * M_mv(Kv)

        B = lx.FunctionLinearOperator(system_mv, jax.ShapeDtypeStruct((n,), X.dtype))
        gmres = lx.GMRES(
            rtol=getattr(self.solver, "rtol", 1e-6),
            atol=getattr(self.solver, "atol", 1e-6),
            max_steps=getattr(self.solver, "max_steps", None) or 10 * n,
            restart=min(n, 50),
        )

        def one(b: Float[Array, " N"]) -> _Solved:
            solution = lx.linear_solve(B, b, gmres, throw=self.throw)
            return (
                solution.value,
                solution.stats["num_steps"],
                solution.result == lx.RESULTS.successful,
            )

        return _per_column(one, Jy)

    def _fit_woodbury(
        self,
        X: Float[Array, "N D"],
        y: Float[Array, " N"] | Float[Array, "N C"],
        penalty: gx.LowRankUpdate,
        scale: Float[Array, ""],
        solver: gx.AbstractSolverStrategy,
    ) -> tuple[
        Float[Array, " N"] | Float[Array, "N C"],
        Int[Array, ""] | Int[Array, " C"] | None,
        Bool[Array, ""] | Bool[Array, " C"] | None,
    ]:
        """``(A + Q diag(scale w) (K Q)^T) alpha = y`` with ``A = K + lambda n I``."""
        n = X.shape[0]
        A = to_operator(
            self.kernel, X, noise=self.regularization * n, implicit=self.implicit
        )
        K_op = to_operator(self.kernel, X, implicit=self.implicit)
        U = _require(
            jnp.all(lx.diagonal(penalty.base) == 0),
            "A low-rank penalty must have a zero diagonal base; build it with "
            "hsic_penalty, or pass the full operator.",
            penalty.U,
        )
        w = scale * penalty.d
        KV = jax.vmap(K_op.mv, in_axes=1, out_axes=1)(penalty.V)
        Ainv_U, _, U_converged = jax.vmap(
            lambda u: self._solve(solver, A, u), in_axes=1, out_axes=(1, 0, 0)
        )(U)
        # Capacitance I + diag(w) (K V)^T A^{-1} U, free of 1 / w.
        weighted = einx.multiply(
            "a, a b -> a b", w, einsum(KV, Ainv_U, "n a, n b -> a b")
        )
        capacitance = jnp.eye(U.shape[1], dtype=U.dtype) + weighted

        def one(b: Float[Array, " N"]) -> _Solved:
            Ainv_b, n_iter, converged = self._solve(solver, A, b)
            coef = jnp.linalg.solve(capacitance, w * einsum(KV, Ainv_b, "n a, n -> a"))
            if converged is not None and U_converged is not None:
                converged = converged & jnp.all(U_converged)
            return Ainv_b - Ainv_U @ coef, n_iter, converged

        return _per_column(one, y)

    def _predict(
        self, X: Float[Array, "Nt D"]
    ) -> Float[Array, " Nt"] | Float[Array, "Nt C"]:
        assert self.X_train is not None and self.alpha is not None
        K_xs = to_cross_operator(self.kernel, X, self.X_train, implicit=self.implicit)
        if self.alpha.ndim == 1:
            return K_xs.mv(self.alpha)
        return jax.vmap(K_xs.mv, in_axes=1, out_axes=1)(self.alpha)


def _per_column(fn, y: Float[Array, " N"] | Float[Array, "N C"]):
    """``fn`` on ``(N,)`` targets, or on each column of ``(N, C)`` targets:
    the solutions stacked as columns, the statistics along the first axis."""
    if y.ndim == 1:
        return fn(y)
    return jax.vmap(fn, in_axes=1, out_axes=(1, 0, 0))(y)


def _no_penalty(
    penalty: lx.AbstractLinearOperator | None, weight: float | Float[Array, ""]
) -> bool:
    """No penalty, or a weight that is a concrete zero: the plain KRR path."""
    if penalty is None:
        return True
    return _is_concrete_zero(weight)


def _is_concrete_zero(value: float | Float[Array, ""]) -> bool:
    """Whether ``value`` is a known zero: a Python number, or a concrete array."""
    if isinstance(value, (int, float)):
        return value == 0
    try:
        return bool(jnp.asarray(value) == 0)
    except jax.errors.ConcretizationTypeError:  # traced: not known to be zero
        return False


def _require(ok: Bool[Array, ""], message: str, carry: Array) -> Array:
    """Raise ``ValueError(message)`` unless ``ok``; under ``jit``, where ``ok`` is

    traced, attach the same check to ``carry`` as an ``equinox`` runtime error.
    """
    try:
        concrete = bool(ok)
    except jax.errors.ConcretizationTypeError:
        return eqx.error_if(carry, ~ok, message)
    if not concrete:
        raise ValueError(message)
    return carry


def _symmetric_mv(penalty: lx.AbstractLinearOperator | None):
    """``M v``, symmetrised as ``(M + M^T) v / 2`` unless ``M`` is tagged symmetric.

    The quadratic term only sees the symmetric part of ``M``, and its gradient
    is ``K (M + M^T) K alpha / 2``.
    """
    if penalty is None:
        return lambda v: jnp.zeros_like(v)
    if lx.is_symmetric(penalty):
        return penalty.mv
    transpose = penalty.T
    return lambda v: 0.5 * (penalty.mv(v) + transpose.mv(v))


def _is_low_rank(
    penalty: lx.AbstractLinearOperator | None,
) -> TypeGuard[gx.LowRankUpdate]:
    """A symmetric-tagged `gaussx.LowRankUpdate` on a diagonal base, for Woodbury.

    The base must be zero; `_fit_woodbury` checks it (eagerly, or at run time
    under ``jit``). Untagged low-rank operators take the general path, which
    symmetrises them.
    """
    return (
        isinstance(penalty, gx.LowRankUpdate)
        and isinstance(penalty.base, lx.DiagonalLinearOperator)
        and lx.is_symmetric(penalty)
    )
