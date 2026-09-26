r"""Kernel ridge regression through any gaussx solver strategy."""

from __future__ import annotations

import dataclasses

import equinox as eqx
import gaussx as gx
import jax
from jaxtyping import Array, Float, PRNGKeyArray

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

    Attributes:
        kernel: The kernel.
        regularization: Ridge $\lambda > 0$.
        solver: A gaussx solver strategy.
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
    X_train: Float[Array, "N D"] | None = None
    alpha: Float[Array, " N"] | Float[Array, "N C"] | None = None

    def fit(
        self,
        X: Float[Array, "N D"],
        y: Float[Array, " N"] | Float[Array, "N C"],
        *,
        key: PRNGKeyArray | None = None,
    ) -> KRR:
        """Solve for the weights. ``key`` is unused.

        Raises:
            ValueError: If ``y`` does not match ``X``.
        """
        del key
        y = _check_targets(X, y)
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

    def _predict(
        self, X: Float[Array, "Nt D"]
    ) -> Float[Array, " Nt"] | Float[Array, "Nt C"]:
        assert self.X_train is not None and self.alpha is not None
        K_xs = to_cross_operator(self.kernel, X, self.X_train, implicit=self.implicit)
        if self.alpha.ndim == 1:
            return K_xs.mv(self.alpha)
        return jax.vmap(K_xs.mv, in_axes=1, out_axes=1)(self.alpha)
