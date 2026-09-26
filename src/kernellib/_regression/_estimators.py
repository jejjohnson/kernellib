r"""`Falkon` and `EigenPro` estimators over the moved primitives."""

from __future__ import annotations

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from kernellib._einx import rearrange
from kernellib._kernels import AbstractKernel
from kernellib._operators._bridge import to_cross_operator, to_operator
from kernellib._regression._base import AbstractEstimator, _check_targets
from kernellib._regression._eigenpro import (
    EigenProPreconditioner,
    eigenpro_correction,
    eigenpro_preconditioner,
    eigenpro_step_size,
)
from kernellib._regression._falkon import falkon_preconditioner, falkon_solve


__all__ = ["EigenPro", "Falkon"]


def _per_column(fn, y: Array) -> Array:
    """Apply a vector solver to ``(N,)`` targets or each column of ``(N, C)``."""
    if y.ndim == 1:
        return fn(y)
    return jax.vmap(fn, in_axes=1, out_axes=1)(y)


class Falkon(AbstractEstimator):
    r"""Nyström kernel ridge regression solved by Falkon (Rudi et al., 2017).

    Picks ``n_inducing`` centres $Z$ uniformly from the training inputs and
    solves $(K_{nm}^\top K_{nm} + \lambda n K_{mm})\alpha = K_{nm}^\top y$ by
    preconditioned conjugate gradients (`falkon_preconditioner`,
    `falkon_solve`). With ``implicit=True`` the ``N x M`` cross kernel is
    streamed in ``batch_size`` rows and never stored: ``O(NM)`` time per
    iteration, ``O(M^2)`` memory. With every point as a centre it is `KRR`
    with the same ``regularization``.

    Attributes:
        kernel: The kernel.
        n_inducing: Number of centres ``M`` (capped at ``N``).
        regularization: Ridge $\lambda > 0$, scaled by ``n`` as in `KRR`.
        max_iter: CG iteration budget; Falkon needs a few tens.
        tol: CG relative tolerance.
        implicit: Stream ``K_nm`` (needs a pointwise kernel; ignored
            otherwise).
        batch_size: Rows per step of the streamed ``K_nm``.
        jitter: Diagonal jitter on ``K_mm``; ``None`` for the default.
        landmarks: The centres ``Z``, ``None`` before `fit`.
        alpha: Weights on the centres, ``None`` before `fit`.

    Examples:
        >>> import jax
        >>> import jax.numpy as jnp
        >>> import kernellib as kl
        >>> X = jnp.linspace(0.0, 1.0, 200)[:, None]
        >>> y = jnp.sin(6.0 * X[:, 0])
        >>> model = kl.Falkon(
        ...     kl.RBF(lengthscale=0.2), n_inducing=30, regularization=1e-6
        ... ).fit(X, y, key=jax.random.key(0))
        >>> model.landmarks.shape
        (30, 1)
        >>> bool(model.loss(X, y) < 1e-4)
        True
    """

    kernel: AbstractKernel
    n_inducing: int = eqx.field(default=1000, static=True)
    regularization: float | Float[Array, ""] = 1e-3
    max_iter: int = eqx.field(default=20, static=True)
    tol: float = eqx.field(default=1e-6, static=True)
    implicit: bool = eqx.field(default=True, static=True)
    batch_size: int = eqx.field(default=1024, static=True)
    jitter: float | None = eqx.field(default=None, static=True)
    landmarks: Float[Array, "M D"] | None = None
    alpha: Float[Array, " M"] | Float[Array, "M C"] | None = None

    def fit(
        self,
        X: Float[Array, "N D"],
        y: Float[Array, " N"] | Float[Array, "N C"],
        *,
        key: PRNGKeyArray | None = None,
    ) -> Falkon:
        """Choose the centres and solve for the weights.

        Raises:
            ValueError: If ``y`` does not match ``X``, or ``key`` is missing
                when ``n_inducing < N``.
        """
        y = _check_targets(X, y)
        n = X.shape[0]
        if self.n_inducing < n:
            if key is None:
                raise ValueError("Falkon needs a PRNG key to choose its centres.")
            Z = X[jax.random.choice(key, n, (self.n_inducing,), replace=False)]
        else:
            Z = X
        precond = falkon_preconditioner(
            self.kernel(Z, Z), self.regularization, jitter=self.jitter
        )
        K_nm = self._cross(X, Z)
        alpha = _per_column(
            lambda col: falkon_solve(
                K_nm,
                col,
                precond,
                self.regularization,
                max_iter=self.max_iter,
                tol=self.tol,
            ),
            y,
        )
        return dataclasses.replace(self, landmarks=Z, alpha=alpha)

    def _cross(self, X: Float[Array, "N D"], Z: Float[Array, "M D"]):
        implicit = self.implicit and self.kernel.is_pointwise
        return to_cross_operator(
            self.kernel,
            X,
            Z,
            implicit=implicit,
            batch_size=min(self.batch_size, X.shape[0]),
        )

    def _predict(
        self, X: Float[Array, "Nt D"]
    ) -> Float[Array, " Nt"] | Float[Array, "Nt C"]:
        assert self.landmarks is not None and self.alpha is not None
        return _per_column(self._cross(X, self.landmarks).mv, self.alpha)


class EigenPro(AbstractEstimator):
    r"""Kernel regression by EigenPro-preconditioned SGD (Ma & Belkin, 2017).

    Fits the interpolating solution of $K\alpha = y$ (no ridge; early
    stopping by ``epochs`` is the regulariser) by mini-batch SGD on
    $\frac{1}{2n}\|K\alpha - y\|^2$ in function space. Plain kernel SGD is
    throttled by the top eigenvalues of the kernel operator; EigenPro damps
    the top ``n_components`` eigendirections, estimated on a subsample
    (`eigenpro_preconditioner`), so the step size (`eigenpro_step_size`) is
    set by the residual spectrum and is typically orders of magnitude
    larger. Each step costs ``O(B N)`` kernel evaluations; nothing ``N x N``
    is formed.

    Per mini-batch $B$ with residuals $g = K_{BX}\alpha - y_B$:

    $$
    \alpha_B \leftarrow \alpha_B - \tfrac{\eta}{b}\, g, \qquad
    \alpha_S \leftarrow \alpha_S + \tfrac{\eta}{b\,m}\, V D V^\top K_{SB}\, g,
    $$

    with $S$ the ``m``-point subsample and $V$, $D$ the preconditioner's
    eigenvectors and weights (`eigenpro_correction`).

    Attributes:
        kernel: The kernel.
        epochs: Passes over the data.
        batch_size: Mini-batch size ``b`` (capped at ``N``).
        subsample_size: Subsample ``m`` for the eigendecomposition (capped at
            ``N``).
        n_components: Number of damped eigendirections ``k < m``.
        decay: Spectral exponent in ``(0, 1]`` (the primitive's ``alpha``).
        X_train: Training inputs, ``None`` before `fit`.
        alpha: Weights, ``None`` before `fit`.
        preconditioner: The fitted `EigenProPreconditioner`.

    Examples:
        >>> import jax
        >>> import jax.numpy as jnp
        >>> import kernellib as kl
        >>> X = jax.random.uniform(jax.random.key(0), (300, 1))
        >>> y = jnp.sin(6.0 * X[:, 0])
        >>> model = kl.EigenPro(
        ...     kl.RBF(lengthscale=0.2),
        ...     epochs=5,
        ...     batch_size=64,
        ...     subsample_size=150,
        ...     n_components=10,
        ... )
        >>> model = model.fit(X, y, key=jax.random.key(1))
        >>> bool(model.loss(X, y) < 1e-3)
        True
    """

    kernel: AbstractKernel
    epochs: int = eqx.field(default=10, static=True)
    batch_size: int = eqx.field(default=256, static=True)
    subsample_size: int = eqx.field(default=2000, static=True)
    n_components: int = eqx.field(default=100, static=True)
    decay: float = eqx.field(default=0.95, static=True)
    X_train: Float[Array, "N D"] | None = None
    alpha: Float[Array, " N"] | Float[Array, "N C"] | None = None
    preconditioner: EigenProPreconditioner | None = None

    def __check_init__(self) -> None:
        if self.epochs < 1 or self.batch_size < 1:
            raise ValueError("epochs and batch_size must be >= 1.")

    def fit(
        self,
        X: Float[Array, "N D"],
        y: Float[Array, " N"] | Float[Array, "N C"],
        *,
        key: PRNGKeyArray | None = None,
    ) -> EigenPro:
        """Run ``epochs`` passes of preconditioned SGD.

        Raises:
            ValueError: If ``y`` does not match ``X``, ``key`` is missing, or
                ``n_components`` is not below the (capped) subsample size.
        """
        y = _check_targets(X, y)
        if key is None:
            raise ValueError("EigenPro needs a PRNG key for its subsample and batches.")
        n = X.shape[0]
        m = min(self.subsample_size, n)
        b = min(self.batch_size, n)
        if not 0 < self.n_components < m:
            raise ValueError(
                f"n_components must be in (0, {m}), below the subsample size; got "
                f"{self.n_components}."
            )
        key_pre, key_sgd = jax.random.split(key)
        op = to_operator(self.kernel, X, implicit=self.kernel.is_pointwise)
        precond = eigenpro_preconditioner(
            op,
            subsample_size=m,
            n_components=self.n_components,
            alpha=self.decay,
            key=key_pre,
        )
        eta = eigenpro_step_size(precond, b)
        S = precond.subsample_indices
        X_S = X[S]
        Y = y if y.ndim == 2 else y[:, None]
        n_batches = n // b

        def step(alpha: Array, idx: Array) -> tuple[Array, None]:
            X_B = X[idx]
            g = self.kernel(X_B, X) @ alpha - Y[idx]
            alpha = alpha.at[idx].add(-(eta / b) * g)
            correction = eigenpro_correction(
                precond, self.kernel(X_B, X_S), g, eta / (b * m)
            )
            return alpha.at[S].add(correction), None

        def epoch(alpha: Array, k: PRNGKeyArray) -> tuple[Array, None]:
            order = jax.random.permutation(k, n)[: n_batches * b]
            alpha, _ = jax.lax.scan(step, alpha, rearrange(order, "(t b) -> t b", b=b))
            return alpha, None

        alpha0 = jnp.zeros_like(Y, dtype=jnp.result_type(Y, X, float))
        alpha, _ = jax.lax.scan(epoch, alpha0, jax.random.split(key_sgd, self.epochs))
        if y.ndim == 1:
            alpha = alpha[:, 0]
        return dataclasses.replace(self, X_train=X, alpha=alpha, preconditioner=precond)

    def _predict(
        self, X: Float[Array, "Nt D"]
    ) -> Float[Array, " Nt"] | Float[Array, "Nt C"]:
        assert self.X_train is not None and self.alpha is not None
        return self.kernel(X, self.X_train) @ self.alpha
