r"""Kernel principal component analysis.

Kernel PCA (Schölkopf, Smola & Müller, 1998) is PCA in the feature space of a
kernel. With the centred Gram matrix $\tilde K = H K H$, $H = I - \tfrac1n
\mathbf 1 \mathbf 1^\top$, and its top eigenpairs
$\tilde K u_j = \lambda_j u_j$, the $j$-th component of a point $x$ is

$$
z_j(x) = \sum_i \alpha_{ij}\, \tilde k(x, x_i), \qquad
\alpha_{\cdot j} = u_j / \sqrt{\lambda_j},
$$

where $\tilde k(x, \cdot)$ is the kernel row centred with the training
statistics. On the training points $z_j = \sqrt{\lambda_j}\, u_j$. The
eigenvalues divided by $n$ are the feature-space variances.

The exact path costs $O(N^3)$. With ``approx``, a feature map
$\phi$ (Nyström, random Fourier, ...) replaces the kernel and kernel PCA
becomes ordinary PCA of $\phi(X)$, in $O(N R^2)$.

**Randomized.** With ``eigen_solver="randomized"`` the kernel is exact but
the eigendecomposition is approximate: `gaussx.randomized_eigh`, a range
finder with $q$ = ``n_power_iter`` power iterations, runs on the doubly
centred, matrix-free operator $HKH$. It needs $(2q + 1)(k + p)$ matvecs
($p$ = ``oversample``), $O(N^2 (k + p) q)$ kernel work and $O(N (k + p))$
memory: the ``N x N`` Gram is never formed. The error is governed by
$\lambda_{k+1}$ and damped like $(\lambda_{k+1} / \lambda_k)^{2q}$, so raise
``n_power_iter`` for slowly decaying spectra (rough kernels, short
lengthscales). Use ``approx`` to approximate the kernel itself when even
$O(N^2)$ kernel evaluations are too many.

**Supervised and fair kernel PCA.** Given targets $T$ with a kernel $K_T$,
the components $Z = \tilde K A$ maximise the variance plus $\gamma$ times
the linear-kernel HSIC between $Z$ and $T$,

$$
\max_A\ \operatorname{tr}\Big(A^\top \tilde K\big(\tfrac1n I
    + \tfrac{\gamma}{n^2} H K_T H\big)\tilde K A\Big)
\quad\text{s.t.}\quad A^\top \tilde K A = I .
$$

$\gamma > 0$ is supervised kernel PCA (Barshan et al., 2011) and
$\gamma < 0$ fair kernel PCA (Pérez-Suay et al., 2017). With
$\tilde K = U\Lambda U^\top$ over its positive eigenpairs and
$A = U\Lambda^{-1/2}B$, the constraint is $B^\top B = I$ and $B$ is the top
eigenvectors of $C = \Lambda^{1/2}U^\top(\tfrac1n I
+ \tfrac{\gamma}{n^2}HK_TH)U\Lambda^{1/2}$; the embedding is
$Z = U\Lambda^{1/2}B$. At $\gamma = 0$, $C = \Lambda / n$ and this is plain
kernel PCA.

**Pre-images** (Bakir, Weston & Schölkopf, 2004). With
``fit_inverse_transform``, a `KRR` from the training embedding back to the
centred inputs is fitted, and `inverse_transform` maps components to input
space,

$$
\hat x(z) = \bar x + k(z, Z)\,(K_Z + n\mu I)^{-1}(X - \mathbf 1 \bar x^\top),
$$

with $\bar x$ the training mean and $\mu$ = ``inverse_regularization``.
Centring the targets acts as an intercept: a `Linear` inverse kernel can then
restore the mean, and far from the training embedding, where $k(z, Z) \to 0$,
a stationary kernel's pre-image falls back to $\bar x$ rather than to $0$.
"""

from __future__ import annotations

import dataclasses
from typing import Literal

import einx
import equinox as eqx
import gaussx as gx
import jax
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Bool, Float, PRNGKeyArray

from kernellib._einx import einsum, rearrange, reduce
from kernellib._heuristics import estimate_lengthscale
from kernellib._kernels import RBF, AbstractKernel, Linear
from kernellib._operators._bridge import to_operator
from kernellib._regression._krr import KRR, _is_concrete_zero, _require
from kernellib._spectral import AbstractFeatureMap
from kernellib.functional._statistics import (
    _double_centre,
    center_cross_kernel,
    centering_operator,
)


__all__ = ["KernelPCA"]


class KernelPCA(eqx.Module):
    r"""Kernel principal component analysis.

    Attributes:
        kernel: The kernel.
        n_components: Number of components.
        approx: Optional unfitted feature map for the ``O(N R^2)`` path.
        eigen_solver: ``"dense"`` (default, ``eigh`` of the centred Gram) or
            ``"randomized"`` (`gaussx.randomized_eigh` on the matrix-free
            $HKH$; needs a ``key`` in `fit`). Plain kernel PCA only.
        n_power_iter: Power iterations $q$ of the randomized range finder.
        oversample: Extra range-finder columns $p$ (capped so that
            $k + p \le N$).
        X_train: Training inputs (exact path), ``None`` before `fit`.
        alphas: Dual coefficients ``(N, n)`` (exact path).
        eigenvalues: Eigenvalues of the centred Gram matrix (or of
            $\bar\Phi^\top \bar\Phi$), ``(n,)``.
        explained_variance: ``eigenvalues / N``, the feature-space variances.
        embedding: The training points' components, ``(N, n)``.
        feature_map: The fitted feature map (approximate path).
        components: Principal directions in feature space ``(R, n)``
            (approximate path).
        target_kernel: Kernel on the targets passed to `fit`; ``None`` means
            `Linear`.
        target_weight: $\gamma$: ``> 0`` supervised, ``< 0`` fair, ``0``
            (default) plain kernel PCA. With a target, ``eigenvalues`` are
            ``n`` times the eigenvalues of $C$ (so they equal plain kernel
            PCA's at $\gamma = 0$) and can be negative for $\gamma < 0$;
            ``explained_variance`` is each component's variance.
        fit_inverse_transform: Also fit the pre-image map, for
            `inverse_transform`.
        inverse_kernel: Kernel on the embedding for the pre-image `KRR`;
            ``None`` means an `RBF` with the median-heuristic lengthscale
            (the mean distance, or ``1``, if the median is ``0``).
        inverse_regularization: Ridge of the pre-image `KRR`.
        inverse_model: The fitted pre-image `KRR`, on the centred inputs.
        inverse_mean: The training mean $\bar x$ ``(D,)``, added back by
            `inverse_transform`.

    Examples:
        >>> import jax
        >>> import jax.numpy as jnp
        >>> import kernellib as kl
        >>> X = jax.random.normal(jax.random.key(0), (50, 3))
        >>> kpca = kl.KernelPCA(kl.RBF(lengthscale=2.0), n_components=2).fit(X)
        >>> bool(jnp.allclose(kpca.transform(X), kpca.embedding, atol=1e-4))
        True

        Matrix-free, by a randomized eigendecomposition of $HKH$:

        >>> rand = kl.KernelPCA(
        ...     kl.RBF(lengthscale=2.0), n_components=2, eigen_solver="randomized"
        ... ).fit(X, key=jax.random.key(1))
        >>> bool(jnp.allclose(rand.eigenvalues, kpca.eigenvalues, rtol=1e-3))
        True
    """

    kernel: AbstractKernel
    n_components: int = eqx.field(default=2, static=True)
    approx: AbstractFeatureMap | None = None
    eigen_solver: Literal["dense", "randomized"] = eqx.field(
        default="dense", static=True
    )
    n_power_iter: int = eqx.field(default=2, static=True)
    oversample: int = eqx.field(default=10, static=True)
    X_train: Float[Array, "N D"] | None = None
    alphas: Float[Array, "N n"] | None = None
    eigenvalues: Float[Array, " n"] | None = None
    explained_variance: Float[Array, " n"] | None = None
    embedding: Float[Array, "N n"] | None = None
    gram_column_means: Float[Array, " N"] | None = None
    gram_mean: Float[Array, ""] | None = None
    feature_map: AbstractFeatureMap | None = None
    feature_mean: Float[Array, " R"] | None = None
    components: Float[Array, "R n"] | None = None
    target_kernel: AbstractKernel | None = None
    target_weight: float | Float[Array, ""] = 0.0
    fit_inverse_transform: bool = eqx.field(default=False, static=True)
    inverse_kernel: AbstractKernel | None = None
    inverse_regularization: float | Float[Array, ""] = 1e-3
    inverse_model: KRR | None = None
    inverse_mean: Float[Array, " D"] | None = None

    def __check_init__(self) -> None:
        if self.n_components < 1:
            raise ValueError(f"n_components must be >= 1, got {self.n_components}.")
        if self.eigen_solver not in ("dense", "randomized"):
            raise ValueError(
                "eigen_solver must be 'dense' or 'randomized', got "
                f"{self.eigen_solver!r}."
            )
        if self.eigen_solver == "randomized" and self.approx is not None:
            raise ValueError(
                "eigen_solver='randomized' decomposes the exact kernel; it "
                "cannot be combined with approx."
            )
        if self.n_power_iter < 0 or self.oversample < 0:
            raise ValueError("n_power_iter and oversample must be >= 0.")

    def fit(
        self,
        X: Float[Array, "N D"],
        *,
        target: Float[Array, "N P"] | Float[Array, " N"] | None = None,
        key: PRNGKeyArray | None = None,
    ) -> KernelPCA:
        """Fit the components on ``X``.

        Args:
            X: Training inputs, shape ``(N, D)``.
            target: Targets (supervised) or protected attributes (fair) for
                ``target_weight``; ignored when the weight is ``0``.
            key: PRNG key for ``eigen_solver="randomized"`` (required
                there); ignored by the dense path.

        Raises:
            ValueError: If ``n_components`` exceeds what the data (or the
                feature map) supports, ``target_weight`` is non-zero
                without a ``target``, or ``eigen_solver="randomized"`` is
                used without a ``key`` or with a ``target``.
        """
        randomized = self.eigen_solver == "randomized"
        if randomized and key is None:
            raise ValueError("eigen_solver='randomized' needs a PRNG key in fit.")
        # Without a target there is nothing to supervise, so the weight must
        # be 0: a ValueError when it is concrete, and a run-time check under
        # jit, where even the default 0.0 is traced. Either way the plain path
        # never silently ignores a non-zero weight.
        if target is None and not _is_concrete_zero(self.target_weight):
            X = _require(
                jnp.asarray(self.target_weight) == 0,
                "target_weight is non-zero but no target was given.",
                X,
            )
        if target is None or _is_concrete_zero(self.target_weight):
            fitted = self._fit_plain(X, key)
        elif randomized:
            raise ValueError(
                "eigen_solver='randomized' supports plain kernel PCA only; use "
                "'dense' for supervised or fair kernel PCA."
            )
        else:
            fitted = self._fit_supervised(X, _target_penalty(self, target))
        if self.fit_inverse_transform:
            assert fitted.embedding is not None
            inverse_model, inverse_mean = self._fit_pre_image(fitted.embedding, X)
            fitted = dataclasses.replace(
                fitted, inverse_model=inverse_model, inverse_mean=inverse_mean
            )
        return fitted

    def inverse_transform(self, Z: Float[Array, "M n"]) -> Float[Array, "M D"]:
        """Approximate pre-images of components ``Z`` in input space.

        Raises:
            RuntimeError: If fitted without ``fit_inverse_transform``.
        """
        if self.inverse_model is None or self.inverse_mean is None:
            raise RuntimeError(
                "inverse_transform needs KernelPCA(fit_inverse_transform=True)."
            )
        return einx.add(
            "m d, d -> m d", self.inverse_model.predict(Z), self.inverse_mean
        )

    def _fit_pre_image(
        self, Z: Float[Array, "N n"], X: Float[Array, "N D"]
    ) -> tuple[KRR, Float[Array, " D"]]:
        """The pre-image `KRR` on the centred inputs, and their mean."""
        kernel = self.inverse_kernel
        if kernel is None:
            # Median distance, falling back to the mean and then 1: with
            # duplicated samples the median can be 0, and RBF(0) gives NaN.
            median = estimate_lengthscale(Z)
            mean = estimate_lengthscale(Z, "mean")
            lengthscale = jnp.where(median > 0, median, jnp.where(mean > 0, mean, 1.0))
            kernel = RBF(lengthscale=lengthscale)
        mean = reduce(X, "n d -> d", "mean")
        centred = einx.subtract("n d, d -> n d", X, mean)
        krr = KRR(kernel, regularization=self.inverse_regularization).fit(Z, centred)
        return krr, mean

    def _fit_supervised(
        self, X: Float[Array, "N D"], M: Float[Array, "N N"]
    ) -> KernelPCA:
        """The generalised eigenproblem, as a standard one of size N (or R)."""
        n, k, gamma = X.shape[0], self.n_components, self.target_weight
        if self.approx is not None:
            fmap = self.approx.fit(self.kernel, X)
            Phi = fmap(X)
            if k > min(Phi.shape):
                raise ValueError(
                    f"n_components={k} exceeds the rank bound "
                    f"{min(Phi.shape)} of the features."
                )
            mu = reduce(Phi, "n r -> r", "mean")
            Pc = einx.subtract("n r, r -> n r", Phi, mu)
            # Work in the eigenbasis V of Pc^T Pc, so that directions in Pc's
            # null space (Z = Pc v = 0) are coordinate axes that can be ruled out.
            gram = einsum(Pc, Pc, "n a, n b -> a b")
            lam, V = jnp.linalg.eigh(_symmetrise(gram))
            positive = _positive(lam, k, "the features")
            PcV = einsum(Pc, V, "n r, r a -> n a")
            C = jnp.diag(lam) / n + gamma * _quad(M, PcV)
            rho, B = _top(_exclude(C, positive), k)
            W = einsum(V, B, "r a, a k -> r k")
            Z = einsum(Pc, W, "n r, r k -> n k")
            return dataclasses.replace(
                self,
                eigenvalues=n * rho,
                explained_variance=reduce(Z**2, "n k -> k", "sum") / n,
                embedding=Z,
                feature_map=fmap,
                feature_mean=mu,
                components=W,
            )
        if k > n:
            raise ValueError(f"n_components={k} exceeds the number of points {n}.")
        K = self.kernel(X, X)
        col = reduce(K, "i j -> j", "mean")
        total = jnp.mean(K)
        Kc = _double_centre(K)
        lam, U = jnp.linalg.eigh(_symmetrise(Kc))
        # Positive eigenpairs only: the constraint A^T K A = I lives there.
        positive = _positive(lam, k, "the centred Gram matrix")
        w = jnp.where(positive, jnp.sqrt(jnp.where(positive, lam, 1.0)), 0.0)
        Uw = einx.multiply("n a, a -> n a", U, w)
        C = jnp.diag(w**2) / n + gamma * _quad(M, Uw)
        rho, B = _top(_exclude(C, positive), k)
        inv_w = jnp.where(positive, 1.0 / jnp.where(positive, w, 1.0), 0.0)
        Z = einsum(Uw, B, "n a, a k -> n k")
        return dataclasses.replace(
            self,
            X_train=X,
            alphas=einsum(
                einx.multiply("n a, a -> n a", U, inv_w), B, "n a, a k -> n k"
            ),
            eigenvalues=n * rho,
            explained_variance=reduce(Z**2, "n k -> k", "sum") / n,
            embedding=Z,
            gram_column_means=col,
            gram_mean=total,
        )

    def _fit_plain(
        self, X: Float[Array, "N D"], key: PRNGKeyArray | None = None
    ) -> KernelPCA:
        n = X.shape[0]
        if self.approx is not None:
            fmap = self.approx.fit(self.kernel, X)
            Phi = fmap(X)
            if self.n_components > min(Phi.shape):
                raise ValueError(
                    f"n_components={self.n_components} exceeds the rank bound "
                    f"{min(Phi.shape)} of the features."
                )
            mu = jnp.mean(Phi, axis=0)
            Pc = Phi - mu
            lam, U = jnp.linalg.eigh(Pc.T @ Pc)
            lam, U = lam[::-1][: self.n_components], U[:, ::-1][:, : self.n_components]
            return dataclasses.replace(
                self,
                eigenvalues=lam,
                explained_variance=lam / n,
                embedding=Pc @ U,
                feature_map=fmap,
                feature_mean=mu,
                components=U,
            )
        if self.n_components > n:
            raise ValueError(
                f"n_components={self.n_components} exceeds the number of points {n}."
            )
        if self.eigen_solver == "randomized":
            assert key is not None  # checked in fit
            K_op = to_operator(self.kernel, X, implicit=self.kernel.is_pointwise)
            # K is symmetric, so its column means are its row means: one matvec.
            col = K_op.mv(jnp.ones(n, dtype=X.dtype)) / n
            total = jnp.mean(col)
            # randomized_eigh returns the eigenpairs in ascending order.
            lam, U = gx.randomized_eigh(
                _doubly_centred(K_op, n),
                self.n_components,
                oversample=min(self.oversample, n - self.n_components),
                n_power_iter=self.n_power_iter,
                which="largest",
                key=key,
            )
            lam, U = lam[::-1], U[:, ::-1]
        else:
            K = self.kernel(X, X)
            col = jnp.mean(K, axis=0)
            total = jnp.mean(K)
            Kc = K - col[None, :] - col[:, None] + total
            lam, U = jnp.linalg.eigh(0.5 * (Kc + Kc.T))
            lam = lam[::-1][: self.n_components]
            U = U[:, ::-1][:, : self.n_components]
        safe = jnp.sqrt(jnp.clip(lam, min=jnp.finfo(lam.dtype).tiny))
        return dataclasses.replace(
            self,
            X_train=X,
            alphas=U / safe,
            eigenvalues=lam,
            explained_variance=lam / n,
            embedding=U * safe,
            gram_column_means=col,
            gram_mean=total,
        )

    def transform(self, X: Float[Array, "M D"]) -> Float[Array, "M n"]:
        """Components of new points.

        Raises:
            RuntimeError: If not fitted.
        """
        if self.components is not None:
            assert self.feature_map is not None and self.feature_mean is not None
            return (self.feature_map(X) - self.feature_mean) @ self.components
        if self.alphas is None:
            raise RuntimeError("KernelPCA is not fitted; call .fit(X) first.")
        assert self.X_train is not None and self.gram_column_means is not None
        assert self.gram_mean is not None
        Kt = self.kernel(X, self.X_train)
        Ktc = center_cross_kernel(Kt, self.gram_column_means, self.gram_mean)
        return Ktc @ self.alphas


def _doubly_centred(K: lx.AbstractLinearOperator, n: int) -> lx.FunctionLinearOperator:
    """``H K H`` with ``H = centering_operator(n)``, matrix-free.

    ``H`` on both sides keeps it symmetric positive semidefinite (a one-sided
    ``H K`` is neither); each ``H`` costs ``O(n)``.
    """
    H = centering_operator(n)
    return lx.FunctionLinearOperator(
        lambda v: H.mv(K.mv(H.mv(v))),
        K.in_structure(),
        tags=(lx.symmetric_tag, lx.positive_semidefinite_tag),
    )


def _target_penalty(
    model: KernelPCA, target: Float[Array, "N P"] | Float[Array, " N"]
) -> Float[Array, "N N"]:
    """``H K_T H / n^2`` for the targets."""
    T = jnp.asarray(target)
    if T.ndim == 1:
        T = rearrange(T, "n -> n 1")
    kernel = Linear() if model.target_kernel is None else model.target_kernel
    n = T.shape[0]
    return _double_centre(kernel(T, T)) / n**2


def _quad(M: Float[Array, "N N"], V: Float[Array, "N r"]) -> Float[Array, "r r"]:
    """``V^T M V``."""
    return einsum(V, einsum(M, V, "i j, j b -> i b"), "i a, i b -> a b")


def _top(
    C: Float[Array, "r r"], k: int
) -> tuple[Float[Array, " k"], Float[Array, "r k"]]:
    """The ``k`` largest eigenpairs of a symmetric matrix, descending."""
    rho, B = jnp.linalg.eigh(_symmetrise(C))
    return rho[::-1][:k], B[:, ::-1][:, :k]


def _positive(lam: Float[Array, " r"], k: int, what: str) -> Bool[Array, " r"]:
    """Eigenvalues above rounding; raises when fewer than ``k`` (if concrete)."""
    positive = lam > jnp.max(lam) * lam.shape[0] * jnp.finfo(lam.dtype).eps
    try:
        rank = int(jnp.sum(positive))
    except jax.errors.ConcretizationTypeError:  # traced: can't check
        return positive
    if k > rank:
        raise ValueError(
            f"n_components={k} exceeds the rank {rank} of {what}; supervised and "
            "fair kernel PCA only have that many valid components."
        )
    return positive


def _exclude(C: Float[Array, "r r"], keep: Bool[Array, " r"]) -> Float[Array, "r r"]:
    """Push the directions outside ``keep`` below every valid eigenvalue.

    Their rows and columns of ``C`` are zero, so without this their zero
    eigenvalues would outrank valid directions with negative eigenvalues
    (``gamma < 0``). The valid eigenpairs are unchanged.
    """
    shift = 1.0 + jnp.sum(jnp.abs(C))
    return C - jnp.diag(jnp.where(keep, 0.0, shift))


def _symmetrise(A: Float[Array, "n n"]) -> Float[Array, "n n"]:
    """``(A + A^T) / 2``, against rounding before ``eigh``."""
    return 0.5 * (A + rearrange(A, "i j -> j i"))
