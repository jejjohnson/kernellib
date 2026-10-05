r"""`Falkon` and `EigenPro` estimators over the moved primitives."""

from __future__ import annotations

import dataclasses
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray

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
from kernellib._spectral._landmarks import LandmarkMethod, select_landmarks


__all__ = ["EigenPro", "Falkon"]


def _per_column(fn, y: Array, out_axes: Any = 1) -> Any:
    """Apply a vector solver to ``(N,)`` targets or each column of ``(N, C)``."""
    if y.ndim == 1:
        return fn(y)
    if y.shape[1] == 1:
        # Solve the lone column unbatched, so (N, 1) targets give bit-for-bit
        # the (N,) result: vmapped matvecs round differently, and an
        # iterative solver that stops at a tolerance does not wash that out.
        out = fn(y[:, 0])
        if not isinstance(out_axes, tuple):
            return jax.tree.map(lambda leaf: jnp.expand_dims(leaf, out_axes), out)
        return tuple(
            jax.tree.map(lambda leaf, axis=axis: jnp.expand_dims(leaf, axis), part)
            for part, axis in zip(out, out_axes, strict=True)
        )
    return jax.vmap(fn, in_axes=1, out_axes=out_axes)(y)


class Falkon(AbstractEstimator):
    r"""Nyström kernel ridge regression solved by Falkon (Rudi et al., 2017).

    **Problem.** KRR restricted to functions $f = \sum_{j=1}^m \alpha_j
    k(\cdot, z_j)$ on ``M`` centres $Z \subset X$ minimises
    $\frac1n\|y - K_{nm}\alpha\|^2 + \lambda\,\alpha^\top K_{mm}\alpha$,
    whose normal equations are

    $$
    H\alpha = K_{nm}^\top y, \qquad H = K_{nm}^\top K_{nm} + \lambda n K_{mm},
    $$

    with the same $\lambda n$ ridge convention as `KRR`. Forming $H$ costs
    $O(NM^2)$ and squares the conditioning of $K_{nm}$, so Falkon never forms
    it.

    **Preconditioner.** With centres sampled uniformly,
    $K_{nm}^\top K_{nm} \approx \frac nm K_{mm}^2$, so $H \approx \frac nm
    K_{mm}^2 + \lambda n K_{mm}$, which factors through two ``M x M``
    upper Choleskys (`falkon_preconditioner`):

    $$
    T = \operatorname{chol}(K_{mm} + \epsilon I), \qquad
    A = \operatorname{chol}\big(\tfrac1m T T^\top + \lambda I\big), \qquad
    P = T^{-1} A^{-1},
    $$

    with $T^\top T = K_{mm} + \epsilon I$ and $PP^\top = n\,(\frac nm K_{mm}^2
    + \lambda n K_{mm})^{-1}$. CG runs on $\beta = P^{-1}\alpha$ against
    $P^\top H P\,\beta = P^\top K_{nm}^\top y$, where $K_{mm}$ cancels:

    $$
    P^\top H P = A^{-\top}\big[T^{-\top} K_{nm}^\top K_{nm} T^{-1}
    + \lambda n I\big] A^{-1},
    $$

    and its condition number is $O(1)$ once $M \gtrsim
    d_{\mathrm{eff}}(\lambda)$, so a few tens of iterations suffice
    (`falkon_solve`).

    ```text
    Z  = X[select_landmarks(kernel, X, M, method=centers)]   # or X if M >= N
    T  = chol_upper(K(Z, Z) + eps I)
    A  = chol_upper(T T^T / M + lambda I)
    b  = A^-T T^-T K_nm^T y
    beta = CG(v -> A^-T [T^-T K_nm^T K_nm T^-1 (A^-1 v) + lambda n A^-1 v], b,
              max_iter, tol)            # K_nm streamed, never stored
    alpha = T^-1 A^-1 beta
    predict(x) = K(x, Z) alpha
    ```

    **Cost.** $O(M^3)$ once for the two Choleskys, then per CG step one
    matvec each with $K_{nm}$ and $K_{nm}^\top$ ($O(NM)$ kernel evaluations)
    and four triangular solves ($O(M^2)$): $O(NMt + M^3)$ time for $t$
    iterations, $O(M^2)$ memory with ``implicit=True`` (the ``N x M`` cross
    kernel is streamed in ``batch_size`` rows), $O(NM)$ without. With every
    point as a centre it is `KRR` with the same ``regularization``.

    **Numerics.**

    - $K_{mm}$ gets a diagonal jitter $\epsilon$, by default
      ``M * eps * max(diag(K_mm))`` floored at the dtype's smallest normal
      number (``jitter`` overrides it), so a numerically singular $K_{mm}$
      (duplicated centres, long lengthscales) still factors.
    - Everything runs in the promoted dtype of the kernel matrix and
      ``regularization``, at least float32. CG stops when every component
      of the preconditioned residual is below ``tol * (max|b| + |b_i|)``;
      ``max_iter`` is a budget, not a failure: the iterate at the budget
      is returned without raising, and ``n_iter`` / ``converged`` report
      what happened. In float32 the default ``tol=1e-6`` is near machine
      precision, so ``converged`` is often ``False`` with an accurate fit.
    - $\lambda$ is not validated, so it must be positive. As
      $\lambda \to 0$, $\frac1m TT^\top + \lambda I$ is dominated by the
      jitter, $P$ stops approximating $H^{-1}$, and CG at a fixed budget
      degrades quietly: no error, just a worse fit. The threshold is
      relative to the dtype's precision, so float32 breaks down at a much
      larger $\lambda$ than float64. Use float64 for small $\lambda$, and
      check ``converged`` and a validation loss.
    - The $\frac nm K_{mm}^2$ approximation assumes uniformly sampled
      centres. Other ``centers`` methods give better centres, but the
      preconditioner is not reweighted for them, so CG may need a few more
      iterations.

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
        centers: How the centres are chosen from ``X``: ``"uniform"``
            (default), ``"leverage"``, ``"rpcholesky"`` (recommended: no
            tuning, near-optimal Nyström) or ``"greedy"``; see
            `select_landmarks`.
        landmarks: The centres ``Z``, ``None`` before `fit`.
        alpha: Weights on the centres, ``None`` before `fit`.
        n_iter: CG steps taken, one per target column for ``(N, C)``
            targets; ``None`` before `fit`.
        converged: Whether CG reached ``tol`` within ``max_iter``, per
            target column; ``None`` before `fit`.

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

    References:
        - Rudi, Carratino & Rosasco (2017). FALKON: An optimal large scale
          kernel method. NeurIPS. [arXiv:1705.10958](https://arxiv.org/abs/1705.10958)
        - Meanti, Carratino, Rosasco & Rudi (2020). Kernel methods through
          the roof: handling billions of points efficiently. NeurIPS.
          [arXiv:2006.10350](https://arxiv.org/abs/2006.10350)
    """

    kernel: AbstractKernel
    n_inducing: int = eqx.field(default=1000, static=True)
    regularization: float | Float[Array, ""] = 1e-3
    max_iter: int = eqx.field(default=20, static=True)
    tol: float = eqx.field(default=1e-6, static=True)
    implicit: bool = eqx.field(default=True, static=True)
    batch_size: int = eqx.field(default=1024, static=True)
    jitter: float | None = eqx.field(default=None, static=True)
    centers: LandmarkMethod = eqx.field(default="uniform", static=True)
    landmarks: Float[Array, "M D"] | None = None
    alpha: Float[Array, " M"] | Float[Array, "M C"] | None = None
    n_iter: Int[Array, ""] | Int[Array, " C"] | None = None
    converged: Bool[Array, ""] | Bool[Array, " C"] | None = None

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
            Z = X[
                select_landmarks(
                    self.kernel, X, self.n_inducing, method=self.centers, key=key
                )
            ]
        else:
            Z = X
        precond = falkon_preconditioner(
            self.kernel(Z, Z), self.regularization, jitter=self.jitter
        )
        K_nm = self._cross(X, Z)
        alpha, info = _per_column(
            lambda col: falkon_solve(
                K_nm,
                col,
                precond,
                self.regularization,
                max_iter=self.max_iter,
                tol=self.tol,
                return_info=True,
            ),
            y,
            # Weights stack as columns, the per-column info along axis 0.
            out_axes=(1, 0),
        )
        return dataclasses.replace(
            self,
            landmarks=Z,
            alpha=alpha,
            n_iter=info.n_iter,
            converged=info.converged,
        )

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

    **Problem.** Fits the interpolating solution of $K\alpha = y$ (no ridge;
    early stopping by ``epochs`` is the regulariser) by mini-batch SGD on
    $\frac{1}{2n}\|K\alpha - y\|^2$ in function space, $f = \sum_i \alpha_i
    k(\cdot, x_i)$. Plain kernel SGD converges at a rate set by
    $\lambda_1 / \lambda_{\min}$ of the kernel's integral operator, and its
    largest stable step is about $2/\lambda_1$, which is tiny because kernel
    spectra decay fast. EigenPro flattens the top of the spectrum so the
    step is set by $\lambda_{q+1}$ instead.

    **Preconditioner.** On an ``m``-point subsample $S$ (chosen by
    ``subsample``), $K_{SS}/m = V \Lambda V^\top$ estimates the operator's
    eigenpairs $\lambda_1 \ge \lambda_2 \ge \dots$, with RKHS-normalised
    Nyström eigenfunctions $e_i(x) = k(x, S) V_{:,i} / \sqrt{m\lambda_i}$.
    With $q$ = ``n_components`` and $a$ = ``decay``,

    $$
    P = I - \sum_{i \le q}\Big(1 - \big(\tfrac{\lambda_{q+1}}{\lambda_i}
    \big)^{a}\Big)\, e_i \otimes e_i ,
    $$

    which maps $\lambda_i \mapsto \lambda_i^{1-a}\lambda_{q+1}^{a}$ for
    $i \le q$ and leaves the rest alone; $a = 1$ is Ma & Belkin's
    $P = I - \sum_{i\le q}(1 - \lambda_{q+1}/\lambda_i)\, e_i \otimes e_i$,
    which flattens the top $q$ eigenvalues to $\lambda_{q+1}$. The default
    $a = 0.95$ stops slightly short of that, so the preconditioned
    spectrum keeps its order. The stored weights are
    $D_i = (1 - (\lambda_{q+1}/\lambda_i)^a)/\lambda_i$
    (`eigenpro_preconditioner`).

    **Step size.** With $\lambda_P = \lambda_1 (\lambda_{q+1}/\lambda_1)^a$
    the top preconditioned eigenvalue and $\beta = \max_i k_P(x_i, x_i)$ the
    largest preconditioned kernel diagonal over the training points,
    `eigenpro_step_size` takes Ma & Belkin's

    $$
    \eta = \begin{cases} b/\beta, & b < \beta/\lambda_P,\\[2pt]
    2b / (\beta + (b-1)\lambda_P), & \text{otherwise,} \end{cases}
    $$

    so for large batches $\eta \to 2/\lambda_P \approx 2/\lambda_{q+1}$,
    against $2/\lambda_1$ for plain SGD: a speed-up of about
    $\lambda_1/\lambda_{q+1}$. In terms of the eigenvalues
    $\sigma_i = m\lambda_i$ of the unnormalised $K_{SS}$ this is
    $\eta \approx 2m/\sigma_{q+1}$.

    **Update.** Per mini-batch $B$ with residuals $g = K_{BX}\alpha - y_B$:

    $$
    \alpha_B \leftarrow \alpha_B - \tfrac{\eta}{b}\, g, \qquad
    \alpha_S \leftarrow \alpha_S + \tfrac{\eta}{b\,m}\, V D V^\top K_{SB}\, g,
    $$

    the second term being the preconditioner's correction on the subsample
    (`eigenpro_correction`).

    ```text
    S = select_landmarks(kernel, X, m, method=subsample)
    lam, V = eigh(K(S, S) / m)                 # descending
    D = (1 - (lam[q] / lam[:q])**a) / lam[:q]; V = V[:, :q]
    beta = max_i [k(x_i, x_i) - sum_j D_j (K(x_i, S) V_j)^2 / m]
    eta = step_size(beta, lam_P, b)
    alpha = 0
    for epoch in range(epochs):
        for B in batches(permutation(N), b):   # N // b batches
            g = K(X_B, X) alpha - y_B
            alpha[B] -= eta / b * g
            alpha[S] += eta / (b m) * V D V^T K(S, X_B) g
    predict(x) = K(x, X) alpha
    ```

    **Cost.** Setup: $O(m^2)$ kernel evaluations and an $O(m^3)$ dense
    ``eigh``, plus $O(Nm)$ kernel evaluations for $\beta$ (streamed in
    row chunks). Each step: $O(bN)$ kernel evaluations for $K_{BX}\alpha$
    and $O(bm + mq)$ for the correction, so $O(N^2)$ kernel evaluations
    per epoch. Memory: $O(bN + m^2)$; nothing ``N x N`` is formed.

    **Numerics.**

    - The eigenvalues are floored at the dtype's ``eps`` and the ratio
      $\lambda_{q+1}/\lambda_i$ is clipped to 1, so a numerically singular
      $K_{SS}$ does not produce negative or infinite weights; $\beta$ is
      floored at ``eps`` too.
    - $\eta \propto 1/\lambda_{q+1}$: taking ``n_components`` so large
      that $\lambda_{q+1}$ reaches the noise floor of the ``eigh``
      (relative ``eps`` of the dtype, about $10^{-7}$ in float32) gives an
      unreliable, very large step and SGD can diverge. A subsample that is
      too small misestimates the spectrum in the same way: keep
      ``n_components`` well below ``subsample_size``.
    - There is no ridge: with noisy targets, more epochs fit the noise.
      Treat ``epochs`` as the regularisation parameter.
    - Each epoch uses ``N // b`` full batches of a fresh permutation, so
      up to ``b - 1`` points (different ones each epoch) are skipped per
      epoch.

        Attributes:
        kernel: The kernel.
        epochs: Passes over the data.
        batch_size: Mini-batch size ``b`` (capped at ``N``).
        subsample_size: Subsample ``m`` for the eigendecomposition (capped at
            ``N``).
        n_components: Number of damped eigendirections ``k < m``.
        decay: Spectral exponent in ``(0, 1]`` (the primitive's ``alpha``).
        subsample: How the eigendecomposition subsample is chosen:
            ``"uniform"`` (default), ``"leverage"``, ``"rpcholesky"`` or
            ``"greedy"``; see `select_landmarks`.
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

    References:
        - Ma & Belkin (2017). Diving into the shallows: a computational
          perspective on large-scale shallow learning. NeurIPS.
          [arXiv:1703.10622](https://arxiv.org/abs/1703.10622)
        - Ma & Belkin (2019). Kernel machines that adapt to GPUs for
          effective large batch training (EigenPro 2.0). SysML.
          [arXiv:1806.06144](https://arxiv.org/abs/1806.06144)
    """

    kernel: AbstractKernel
    epochs: int = eqx.field(default=10, static=True)
    batch_size: int = eqx.field(default=256, static=True)
    subsample_size: int = eqx.field(default=2000, static=True)
    n_components: int = eqx.field(default=100, static=True)
    decay: float = eqx.field(default=0.95, static=True)
    subsample: LandmarkMethod = eqx.field(default="uniform", static=True)
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
            subsample_indices=select_landmarks(
                self.kernel, X, m, method=self.subsample, key=key_pre
            ),
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
