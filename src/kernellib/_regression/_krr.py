r"""Kernel ridge regression through any gaussx solver strategy."""

from __future__ import annotations

import dataclasses
from typing import TypeGuard

import einx
import equinox as eqx
import gaussx as gx
import jax
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Bool, Float, PRNGKeyArray

from kernellib._einx import einsum
from kernellib._kernels import AbstractKernel
from kernellib._operators._bridge import to_cross_operator, to_operator
from kernellib._regression._base import AbstractEstimator, _check_targets


__all__ = ["KRR"]


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
      ``max_steps``) of ``solver`` when it has them.

    Attributes:
        kernel: The kernel.
        regularization: Ridge $\lambda > 0$.
        solver: A gaussx solver strategy.
        penalty_weight: Weight $\mu \ge 0$ of the penalty passed to `fit`.
        implicit: Build the kernel matrix matrix-free
            (`ImplicitKernelOperator`); needs a pointwise kernel and a concrete
            ``regularization``.
        X_train: Training inputs, ``None`` before `fit`.
        alpha: Weights, ``(N,)`` or ``(N, C)``, ``None`` before `fit`.

    Examples:
        >>> import jax.numpy as jnp
        >>> import kernellib as kl
        >>> X = jnp.linspace(0.0, 1.0, 20)[:, None]
        >>> y = jnp.sin(6.0 * X[:, 0])
        >>> model = kl.KRR(kl.RBF(lengthscale=0.2), regularization=1e-6).fit(X, y)
        >>> bool(jnp.max(jnp.abs(model.predict(X) - y)) < 1e-2)
        True
    """

    kernel: AbstractKernel
    regularization: float | Float[Array, ""] = 1e-3
    solver: gx.AbstractSolverStrategy = gx.DenseSolver()
    implicit: bool = eqx.field(default=False, static=True)
    penalty_weight: float | Float[Array, ""] = 0.0
    X_train: Float[Array, "N D"] | None = None
    alpha: Float[Array, " N"] | Float[Array, "N C"] | None = None

    def fit(
        self,
        X: Float[Array, "N D"],
        y: Float[Array, " N"] | Float[Array, "N C"],
        *,
        penalty: lx.AbstractLinearOperator | None = None,
        mask: Bool[Array, " N"] | None = None,
        key: PRNGKeyArray | None = None,
    ) -> KRR:
        """Solve for the weights. ``key`` is unused.

        Args:
            X: Training inputs, shape ``(N, D)``.
            y: Targets, ``(N,)`` or ``(N, C)``. Values at unlabelled points
                (``mask`` false) are ignored and may be NaN.
            penalty: Optional penalty operator $M$ on the training points,
                e.g. from `hsic_penalty` or `laplacian_penalty`. A
                `gaussx.LowRankUpdate` must have a zero diagonal base.
            mask: Optional boolean mask of labelled points, shape ``(N,)``.
            key: Unused.

        Raises:
            ValueError: If ``y`` or ``mask`` does not match ``X``, or a
                low-rank ``penalty`` has a non-zero diagonal base.
        """
        del key
        y = _check_targets(X, y)
        if mask is not None or not _no_penalty(penalty, self.penalty_weight):
            alpha = self._fit_penalised(X, y, penalty, mask)
            return dataclasses.replace(self, X_train=X, alpha=alpha)
        n = X.shape[0]
        K = to_operator(
            self.kernel, X, noise=self.regularization * n, implicit=self.implicit
        )
        if y.ndim == 1:
            alpha = self.solver.solve(K, y)
        else:
            alpha = jax.vmap(
                lambda col: self.solver.solve(K, col), in_axes=1, out_axes=1
            )(y)
        return dataclasses.replace(self, X_train=X, alpha=alpha)

    def _fit_penalised(
        self,
        X: Float[Array, "N D"],
        y: Float[Array, " N"] | Float[Array, "N C"],
        penalty: lx.AbstractLinearOperator | None,
        mask: Bool[Array, " N"] | None,
    ) -> Float[Array, " N"] | Float[Array, "N C"]:
        n = X.shape[0]
        mu = jnp.asarray(0.0 if penalty is None else self.penalty_weight)
        if mask is None:
            J = jnp.ones(n, dtype=bool)
        else:
            J = jnp.asarray(mask, dtype=bool)
            if J.shape != (n,):
                raise ValueError(f"mask must have shape ({n},), got {J.shape}.")
            try:
                empty = not bool(jnp.any(J))
            except jax.errors.ConcretizationTypeError:  # traced: can't check
                empty = False
            if empty:
                raise ValueError("mask selects no labelled points.")
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
            return self._fit_woodbury(X, y, penalty, n * mu)

        K_op = to_operator(self.kernel, X, implicit=self.implicit)
        M_mv = (lambda v: jnp.zeros_like(v)) if penalty is None else penalty.mv
        lam, reg = self.regularization, mu

        if not self.implicit:
            K = K_op.as_matrix()
            MK = jax.vmap(M_mv, in_axes=1, out_axes=1)(K)
            B = (
                einx.multiply("i, i j -> i j", Jf, K)
                + n_lab * lam * jnp.eye(n, dtype=K.dtype)
                + n_lab * reg * MK
            )
            return jnp.linalg.solve(B, Jy)

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

        def one(b: Float[Array, " N"]) -> Float[Array, " N"]:
            return lx.linear_solve(B, b, gmres).value

        if y.ndim == 1:
            return one(Jy)
        return jax.vmap(one, in_axes=1, out_axes=1)(Jy)

    def _fit_woodbury(
        self,
        X: Float[Array, "N D"],
        y: Float[Array, " N"] | Float[Array, "N C"],
        penalty: gx.LowRankUpdate,
        scale: Float[Array, ""],
    ) -> Float[Array, " N"] | Float[Array, "N C"]:
        """``(A + Q diag(scale w) (K Q)^T) alpha = y`` with ``A = K + lambda n I``."""
        n = X.shape[0]
        A = to_operator(
            self.kernel, X, noise=self.regularization * n, implicit=self.implicit
        )
        K_op = to_operator(self.kernel, X, implicit=self.implicit)
        U, w = penalty.U, scale * penalty.d
        KV = jax.vmap(K_op.mv, in_axes=1, out_axes=1)(penalty.V)
        solve = self.solver.solve
        Ainv_U = jax.vmap(lambda u: solve(A, u), in_axes=1, out_axes=1)(U)
        # Capacitance I + diag(w) (K V)^T A^{-1} U, free of 1 / w.
        weighted = einx.multiply(
            "a, a b -> a b", w, einsum(KV, Ainv_U, "n a, n b -> a b")
        )
        capacitance = jnp.eye(U.shape[1], dtype=U.dtype) + weighted

        def one(b: Float[Array, " N"]) -> Float[Array, " N"]:
            Ainv_b = solve(A, b)
            coef = jnp.linalg.solve(capacitance, w * einsum(KV, Ainv_b, "n a, n -> a"))
            return Ainv_b - Ainv_U @ coef

        if y.ndim == 1:
            return one(y)
        return jax.vmap(one, in_axes=1, out_axes=1)(y)

    def _predict(
        self, X: Float[Array, "Nt D"]
    ) -> Float[Array, " Nt"] | Float[Array, "Nt C"]:
        assert self.X_train is not None and self.alpha is not None
        K_xs = to_cross_operator(self.kernel, X, self.X_train, implicit=self.implicit)
        if self.alpha.ndim == 1:
            return K_xs.mv(self.alpha)
        return jax.vmap(K_xs.mv, in_axes=1, out_axes=1)(self.alpha)


def _no_penalty(
    penalty: lx.AbstractLinearOperator | None, weight: float | Float[Array, ""]
) -> bool:
    """No penalty, or a weight that is a concrete zero: the plain KRR path."""
    if penalty is None:
        return True
    return _is_concrete_zero(weight)


def _is_concrete_zero(value: float | Float[Array, ""]) -> bool:
    """Whether ``value`` is a known zero: a Python number, or a concrete array."""
    try:
        return bool(jnp.asarray(value) == 0)
    except jax.errors.ConcretizationTypeError:  # traced: not known to be zero
        return False


def _is_low_rank(
    penalty: lx.AbstractLinearOperator | None,
) -> TypeGuard[gx.LowRankUpdate]:
    """A `gaussx.LowRankUpdate` with a zero diagonal base (checked when concrete)."""
    if not isinstance(penalty, gx.LowRankUpdate):
        return False
    if not isinstance(penalty.base, lx.DiagonalLinearOperator):
        return False
    try:
        nonzero = bool(jnp.any(lx.diagonal(penalty.base) != 0))
    except jax.errors.ConcretizationTypeError:  # traced: trust the builder
        nonzero = False
    if nonzero:
        raise ValueError(
            "A low-rank penalty must have a zero diagonal base; build it with "
            "hsic_penalty, or pass the full operator."
        )
    return True
