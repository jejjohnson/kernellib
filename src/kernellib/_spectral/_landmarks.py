r"""Landmark selection for Nyström-type methods.

Nyström on landmarks $Z$ approximates the Gram matrix by
$\hat K = K_{XZ} K_{ZZ}^{+} K_{ZX}$. Its trace error is a sum of conditional
variances,

$$
\operatorname{tr}(K - \hat K)
    = \sum_i \big(k(x_i, x_i) - k_{iZ} K_{ZZ}^{-1} k_{Zi}\big),
$$

exactly the residual diagonal that randomly pivoted Cholesky samples from.
Ridge leverage scores $\ell_i(\lambda) = [K (K + \lambda n I)^{-1}]_{ii}$ sum
to the effective dimension $d_{\mathrm{eff}}(\lambda)$, the number of
landmarks that matter.
"""

from __future__ import annotations

from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsl
from gaussx import rp_cholesky
from jaxtyping import Array, Float, Int, PRNGKeyArray

from kernellib._einx import einsum, rearrange, reduce
from kernellib._kernels import AbstractKernel


__all__ = ["LandmarkMethod", "select_landmarks"]

LandmarkMethod = Literal["uniform", "leverage", "rpcholesky", "greedy"]
_METHODS = ("uniform", "leverage", "rpcholesky", "greedy")


def select_landmarks(
    kernel: AbstractKernel,
    X: Float[Array, "N D"],
    n_landmarks: int,
    *,
    method: LandmarkMethod = "uniform",
    key: PRNGKeyArray,
    regularization: float | None = None,
    uniform_mixing: float = 0.5,
) -> Int[Array, " M"]:
    r"""Indices into ``X`` of ``n_landmarks`` distinct landmarks.

    - ``"uniform"``: uniformly, without replacement.
    - ``"leverage"``: proportionally to approximate ridge leverage scores
      (Alaoui & Mahoney, 2015; Rudi et al., 2018), from a uniform pilot
      Nyström map on ``min(2 M, N)`` points, mixed with the uniform
      distribution with weight ``uniform_mixing``; see `NystromFeatures`.
    - ``"rpcholesky"``: randomly pivoted Cholesky (Chen, Epperly, Tropp &
      Webber, 2023) through `gaussx.rp_cholesky`. Each landmark is drawn in
      proportion to the variance the landmarks so far leave unexplained.
      ``O(N M)`` kernel evaluations, ``O(N M^2)`` arithmetic, the ``N x N``
      Gram never formed, and no tuning; its Nyström error is within a small
      factor of the best rank-``M`` approximation in expectation. **The
      recommended choice.**
    - ``"greedy"``: the largest residual variance each time (pivoted
      Cholesky, the conditional-variance rule of Burt, Rasmussen & van der
      Wilk, 2020). Deterministic: ``key`` is not used. Without the
      randomisation it has no error guarantee: it chases isolated points,
      and on clustered data it can do worse than ``"uniform"``.

    If the kernel matrix has numerical rank below ``n_landmarks`` (repeated
    points, say), the Cholesky methods run out of informative pivots; the
    remaining landmarks are then filled with unused points, uniformly at
    random for ``"rpcholesky"`` and in index order for ``"greedy"``.

    Args:
        kernel: The kernel.
        X: Candidate points, shape ``(N, D)``.
        n_landmarks: Number of landmarks ``M <= N``.
        method: ``"uniform"`` (default), ``"leverage"``, ``"rpcholesky"`` or
            ``"greedy"``.
        key: PRNG key.
        regularization: The $\lambda$ of the ridge leverage scores
            (``"leverage"`` only); ``None`` for ``1e-3``.
        uniform_mixing: Weight in ``[0, 1]`` of the uniform distribution in
            the leverage sampling distribution (``"leverage"`` only).

    Returns:
        Distinct indices, shape ``(M,)``, in the order chosen.

    Raises:
        ValueError: For an unknown ``method``, ``n_landmarks`` outside
            ``[1, N]``, or invalid leverage options.

    Examples:
        >>> import jax
        >>> import kernellib as kl
        >>> X = jax.random.normal(jax.random.key(0), (500, 2))
        >>> idx = kl.select_landmarks(
        ...     kl.Matern(nu=1.5, lengthscale=0.3),
        ...     X,
        ...     20,
        ...     method="rpcholesky",
        ...     key=jax.random.key(1),
        ... )
        >>> idx.shape, len(set(idx.tolist()))
        ((20,), 20)
    """
    _check_options(n_landmarks, X.shape[0], method, regularization, uniform_mixing)
    return _select(
        kernel,
        X,
        n_landmarks,
        method,
        key,
        1e-3 if regularization is None else regularization,
        uniform_mixing,
        1e-6,
    )


def _check_options(
    n_landmarks: int,
    n: int,
    method: str,
    regularization: float | None,
    uniform_mixing: float,
) -> None:
    if method not in _METHODS:
        raise ValueError(
            "method must be 'uniform', 'leverage', 'rpcholesky' or 'greedy', "
            f"got {method!r}."
        )
    if not 1 <= n_landmarks <= n:
        raise ValueError(f"n_landmarks must be in [1, {n}], got {n_landmarks}.")
    if not 0.0 <= uniform_mixing <= 1.0:
        raise ValueError(f"uniform_mixing must be in [0, 1], got {uniform_mixing}.")
    if regularization is not None and not regularization > 0.0:
        raise ValueError(f"regularization must be positive, got {regularization}.")


# Jitted, so repeated calls with the same shapes reuse one compilation (the
# Cholesky methods' loop is slow to retrace). Non-array arguments (``method``,
# ``n_landmarks``, the float options) are static under ``filter_jit``.
@eqx.filter_jit
def _select(
    kernel: AbstractKernel,
    X: Float[Array, "N D"],
    n_landmarks: int,
    method: str,
    key: PRNGKeyArray,
    regularization: float,
    uniform_mixing: float,
    jitter: float,
) -> Int[Array, " M"]:
    """`select_landmarks` without validation, with `NystromFeatures`' jitter."""
    n = X.shape[0]
    if method == "uniform":
        return jax.random.choice(key, n, (n_landmarks,), replace=False)
    if method == "leverage":
        key_pilot, key_draw = jax.random.split(key)
        m0 = min(2 * n_landmarks, n)
        pilot = jax.random.choice(key_pilot, n, (m0,), replace=False)
        scores = _ridge_leverage_scores(kernel, X, pilot, regularization, jitter)
        return jax.random.choice(
            key_draw,
            n,
            (n_landmarks,),
            replace=False,
            p=(1.0 - uniform_mixing) * scores / jnp.sum(scores) + uniform_mixing / n,
        )
    key_pivots, key_fill = jax.random.split(key)

    def column(s: Int[Array, ""]) -> Float[Array, " N"]:
        return rearrange(kernel(X, rearrange(X[s], "d -> 1 d")), "n 1 -> n")

    _, pivots = rp_cholesky(
        kernel.diag(X),
        column,
        n_landmarks,
        pivoting="greedy" if method == "greedy" else "random",
        key=key_pivots,
    )
    fill_order = (
        jnp.arange(n) if method == "greedy" else jax.random.permutation(key_fill, n)
    )
    return _fill_exhausted(pivots, fill_order)


def _fill_exhausted(
    pivots: Int[Array, " M"], fill_order: Int[Array, " N"]
) -> Int[Array, " M"]:
    """Replace the ``-1`` pivots past the numerical rank by unused indices,
    taken in ``fill_order``. Jittable: every shape is static."""
    n = fill_order.shape[0]
    chosen = (
        jnp.zeros(n, dtype=bool)
        .at[jnp.where(pivots >= 0, pivots, n)]
        .set(True, mode="drop")
    )
    # Unused indices first, in fill order (a stable sort keeps that order).
    candidates = fill_order[jnp.argsort(chosen[fill_order], stable=True)]
    missing = pivots < 0
    slot = jnp.cumsum(missing) - 1
    return jnp.where(missing, candidates[jnp.maximum(slot, 0)], pivots)


def _nystrom_features(
    kernel: AbstractKernel,
    Z: Float[Array, "M D"],
    X: Float[Array, "N D"],
    jitter: float,
) -> Float[Array, "N M"]:
    """``L^{-1} k(Z, X)``, transposed, with ``K_ZZ + jitter I = L Lᵀ``."""
    K_zz = kernel(Z, Z)
    # Relative to the mean diagonal, with an absolute floor when that is zero
    # (e.g. a `Distance` kernel whose only landmark is the origin).
    scale = jnp.mean(jnp.diag(K_zz))
    eps = jitter * jnp.where(scale > 0, scale, 1.0)
    L = jnp.linalg.cholesky(K_zz + eps * jnp.eye(Z.shape[0], dtype=K_zz.dtype))
    return rearrange(jsl.solve_triangular(L, kernel(Z, X), lower=True), "m n -> n m")


def _ridge_leverage_scores(
    kernel: AbstractKernel,
    X: Float[Array, "N D"],
    pilot: Int[Array, " m0"],
    regularization: float,
    jitter: float,
) -> Float[Array, " N"]:
    r"""Approximate ridge leverage scores from a pilot Nyström map.

    With pilot features $\Phi$ (``N x m0``, $K \approx \Phi\Phi^\top$) the
    push-through identity gives the leverage of $\Phi\Phi^\top$ as
    $\phi_i^\top (\Phi^\top\Phi + \lambda n I)^{-1} \phi_i$; the Nyström
    residual on the diagonal, over $\lambda n$, is added back. Non-negative;
    they sum to about the effective dimension $d_{\mathrm{eff}}(\lambda)$.
    """
    n = X.shape[0]
    Phi = _nystrom_features(kernel, X[pilot], X, jitter)  # (N, m0)
    ridge = regularization * n
    gram = einsum(Phi, Phi, "n a, n b -> a b")
    chol = jnp.linalg.cholesky(gram + ridge * jnp.eye(gram.shape[0], dtype=gram.dtype))
    solved = jsl.solve_triangular(
        chol, rearrange(Phi, "n a -> a n"), lower=True
    )  # (m0, N)
    explained = reduce(solved**2, "a n -> n", "sum")
    residual = jnp.maximum(kernel.diag(X) - reduce(Phi**2, "n a -> n", "sum"), 0.0)
    return explained + residual / ridge
