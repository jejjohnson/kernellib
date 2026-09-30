"""Matrix-level kernel statistics: centering, HSIC, CKA, MMD.

Arrays or operators in, scalar or operator out. `centering_operator`,
`center_kernel`, the biased `hsic` and `mmd_squared` are moved from
``gaussx._kernels._kernel_approx`` unchanged; the unbiased HSIC estimator and
`cka` are new. Kernel-and-data versions (``kernellib.hsic(kx, ky, X, Y)``)
build on these.

Low-rank operands -- a `gaussx.LowRankUpdate` on a diagonal base, which is
what `nystrom_operator`, `rff_operator` and ``feature_map.operator(X)``
return -- stay low-rank: `center_kernel` returns one, and `hsic` / `cka` on
two of them cost ``O(N R_x R_y)`` with no ``N x N`` intermediate.
"""

from __future__ import annotations

from typing import Literal, TypeGuard

import einx
import gaussx as gx
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Float

from kernellib._einx import einsum, reduce


__all__ = [
    "center_cross_kernel",
    "center_kernel",
    "centering_operator",
    "cka",
    "hsic",
    "mmd_squared",
]


def centering_operator(n: int) -> gx.LowRankUpdate:
    r"""Centering matrix ``H = I - (1/n) \mathbf{1}\mathbf{1}^T``.

    Returns as a `LowRankUpdate` so the structure is
    preserved for downstream operations like ``H K H``.

    Args:
        n: Dimension of the centering matrix.

    Returns:
        `LowRankUpdate` operator of shape ``(n, n)``.

    Examples:
        >>> import jax.numpy as jnp
        >>> from kernellib.functional import centering_operator
        >>> H = centering_operator(4).as_matrix()
        >>> bool(jnp.allclose(H @ jnp.ones(4), 0.0))
        True
    """
    base = lx.DiagonalLinearOperator(jnp.ones(n))
    ones = jnp.ones((n, 1))
    D = jnp.array([-1.0 / n])
    return gx.LowRankUpdate(base=base, U=ones, d=D, V=ones, tags=frozenset())


def center_kernel(
    K: lx.AbstractLinearOperator,
) -> lx.MatrixLinearOperator | gx.LowRankUpdate:
    r"""Center a kernel matrix: ``H K H``.

    Computes the centered Gram matrix where ``H = I - (1/n) 11^T``.
    Used in kernel PCA, HSIC, and other centered kernel methods.

    A `gaussx.LowRankUpdate` ``K = D_0 + U \operatorname{diag}(d) V^\top``
    on a diagonal base ``D_0`` (the Nyström, RFF and feature-map operators,
    with or without a noise diagonal) is centered without forming ``K``:

    $$
    H K H = D_0 + (HU) \operatorname{diag}(d) (HV)^\top
        - \tfrac{1}{n}\left(\delta \mathbf{1}^\top + \mathbf{1} \delta^\top\right)
        + \tfrac{\mathbf{1}^\top \delta}{n^2} \mathbf{1}\mathbf{1}^\top,
    $$

    with $\delta = \operatorname{diag}(D_0)$ and $HU$ the column-centred
    factor. The result is a `gaussx.LowRankUpdate` of rank ``k + 3`` on the
    same base, ``O(N k)`` to build. Any other operator is materialised.

    Args:
        K: Kernel (Gram) matrix operator, shape ``(n, n)``.

    Returns:
        Centered kernel operator, shape ``(n, n)``: a `gaussx.LowRankUpdate`
        for a low-rank input on a diagonal base, a dense operator otherwise.

    Examples:
        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> from kernellib.functional import center_kernel
        >>> K = lx.MatrixLinearOperator(jnp.ones((3, 3)) + jnp.eye(3))
        >>> Kc = center_kernel(K).as_matrix()
        >>> bool(jnp.allclose(Kc.sum(axis=0), 0.0))
        True

        A low-rank operator stays low-rank:

        >>> import gaussx as gx
        >>> Phi = jnp.arange(8.0).reshape(4, 2)
        >>> K = gx.LowRankUpdate(lx.DiagonalLinearOperator(jnp.zeros(4)), Phi)
        >>> type(center_kernel(K)).__name__
        'LowRankUpdate'
    """
    tags = frozenset()
    if lx.is_symmetric(K):
        tags = frozenset({lx.symmetric_tag})
    if _is_diagonal_low_rank(K):
        return _center_low_rank(K, tags)
    K_mat = K.as_matrix()
    row_mean = jnp.mean(K_mat, axis=1, keepdims=True)
    col_mean = jnp.mean(K_mat, axis=0, keepdims=True)
    total_mean = jnp.mean(K_mat)
    K_centered = K_mat - row_mean - col_mean + total_mean
    return lx.MatrixLinearOperator(K_centered, tags)


def _is_diagonal_low_rank(
    K: lx.AbstractLinearOperator,
) -> TypeGuard[gx.LowRankUpdate]:
    return isinstance(K, gx.LowRankUpdate) and isinstance(
        K.base, lx.DiagonalLinearOperator
    )


def _center_low_rank(K: gx.LowRankUpdate, tags: frozenset) -> gx.LowRankUpdate:
    """``H K H`` for ``K = D_0 + U diag(d) V^T`` as a rank ``k + 3`` update."""
    delta = lx.diagonal(K.base)
    n = delta.shape[0]
    dtype = jnp.result_type(delta, K.U, K.d, K.V)
    ones = jnp.ones((n, 1), dtype=dtype)
    delta_col = delta[:, None].astype(dtype)
    # H D_0 H = D_0 - (δ1ᵀ + 1δᵀ)/n + (1ᵀδ/n²) 11ᵀ: three rank-one terms.
    U = jnp.concatenate([K.U - jnp.mean(K.U, axis=0), delta_col, ones, ones], axis=1)
    V = jnp.concatenate([K.V - jnp.mean(K.V, axis=0), ones, delta_col, ones], axis=1)
    correction = jnp.stack([-1.0 / n, -1.0 / n, jnp.sum(delta) / n**2]).astype(dtype)
    d = jnp.concatenate([K.d.astype(dtype), correction])
    return gx.LowRankUpdate(base=K.base, U=U, d=d, V=V, tags=tags)


def center_cross_kernel(
    K_t: Float[Array, "M N"],
    col_means: Float[Array, " N"],
    mean: Float[Array, ""] | float,
) -> Float[Array, "M N"]:
    r"""Centre a cross-kernel matrix with the training Gram's statistics.

    For new points $x_t$ against training points, the feature-space
    centring that `center_kernel` applies to the training Gram $K$ is

    $$
    \tilde K_t = K_t - \mathbf 1\,\bar k^\top
        - \bar k_t\,\mathbf 1^\top + \bar{\bar k},
    $$

    with $\bar k$ the column means of $K$, $\bar k_t$ the row means of
    $K_t$ and $\bar{\bar k}$ the grand mean of $K$. With $x_t$ the training
    points themselves it equals $HKH$. `KernelPCA.transform` uses it.

    Args:
        K_t: Cross-kernel matrix ``k(X_t, X)``, shape ``(M, N)``.
        col_means: Column means of the training Gram ``k(X, X)``, ``(N,)``.
        mean: Grand mean of the training Gram.

    Returns:
        The centred cross-kernel matrix, shape ``(M, N)``.

    Examples:
        >>> import einx
        >>> import jax.numpy as jnp
        >>> from kernellib.functional import center_cross_kernel
        >>> K = jnp.array([[2.0, 1.0], [1.0, 2.0]])
        >>> center_cross_kernel(K, einx.mean("i j -> j", K), jnp.mean(K)).tolist()
        [[0.5, -0.5], [-0.5, 0.5]]
    """
    minus_cols = einx.subtract("m n, n -> m n", K_t, col_means)
    minus_rows = einx.subtract(
        "m n, m -> m n", minus_cols, reduce(K_t, "m n -> m", "mean")
    )
    return minus_rows + mean


def hsic(
    K_f: lx.AbstractLinearOperator,
    K_q: lx.AbstractLinearOperator,
    *,
    estimator: Literal["biased", "unbiased"] = "biased",
) -> Float[Array, ""]:
    r"""Hilbert-Schmidt Independence Criterion from two kernel matrices.

    The biased estimator (Gretton et al., 2005) is

    $$
    \mathrm{HSIC}_b = \frac{1}{n^2} \operatorname{tr}(K_f H K_q H),
    \qquad H = I - \tfrac{1}{n}\mathbf{1}\mathbf{1}^\top.
    $$

    The unbiased estimator (Song et al., 2012) removes the ``O(1/n)`` bias.
    With $\tilde K$ the kernel matrix with its diagonal set to zero,

    $$
    \mathrm{HSIC}_u = \frac{1}{n(n-3)}\left[
        \operatorname{tr}(\tilde K_f \tilde K_q)
        + \frac{\mathbf{1}^\top \tilde K_f \mathbf{1}\;
                \mathbf{1}^\top \tilde K_q \mathbf{1}}{(n-1)(n-2)}
        - \frac{2}{n-2}\, \mathbf{1}^\top \tilde K_f \tilde K_q \mathbf{1}
    \right].
    $$

    It is a U-statistic, so it can be slightly negative for independent
    samples, and it needs ``n >= 4``. Like the closed form it implements, it
    assumes symmetric kernel matrices, which any kernel produces.

    Numerically it is evaluated as $\langle \tilde K_f^{U}, \tilde K_q^{U}
    \rangle_F / (n(n-3))$ with U-centred matrices (Székely & Rizzo, 2014),
    which is the same estimator algebraically. The expanded form above sums
    three ``O(n^2)`` terms that cancel, and in float32 it loses every digit
    once a Gram matrix is within about ``1e-3`` of constant. Centring first
    removes the common mode before anything is summed. The low-rank path
    centres its factors instead, which leaves the estimate unchanged because
    it is invariant to ``K -> H K H``.

    When both operands are `gaussx.LowRankUpdate` on a diagonal base (e.g.
    ``feature_map.operator(X)``), both estimators run on the factors in
    ``O(N R_f R_q)`` and never form an ``N x N`` matrix. The diagonal base
    drops out of the unbiased estimator, which zeroes the diagonal.

    Args:
        K_f: First kernel matrix, shape ``(n, n)``.
        K_q: Second kernel matrix, shape ``(n, n)``.
        estimator: ``"biased"`` (default, as in gaussx) or ``"unbiased"``.

    Returns:
        Scalar HSIC estimate.

    Raises:
        ValueError: If ``estimator`` is unknown, or if ``estimator`` is
            ``"unbiased"`` and ``n < 4``.

    Examples:
        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> from kernellib.functional import hsic
        >>> K = lx.MatrixLinearOperator(jnp.eye(5), lx.symmetric_tag)
        >>> hsic(K, K).shape
        ()
        >>> hsic(K, K, estimator="unbiased").shape
        ()
    """
    if estimator == "biased":
        K_f_centered = center_kernel(K_f)
        K_q_centered = center_kernel(K_q)
        n = K_f_centered.in_size()
        return gx.trace_product(K_f_centered, K_q_centered) / (n * n)
    if estimator == "unbiased":
        if _is_diagonal_low_rank(K_f) and _is_diagonal_low_rank(K_q):
            return _hsic_unbiased_low_rank(
                (K_f.U, K_f.d, K_f.V),
                (K_q.U, K_q.d, K_q.V),
            )
        return _hsic_unbiased(K_f.as_matrix(), K_q.as_matrix())
    raise ValueError(f"estimator must be 'biased' or 'unbiased', got {estimator!r}.")


def _hsic_unbiased(K: Float[Array, "n n"], L: Float[Array, "n n"]) -> Float[Array, ""]:
    """Song et al. (2012) unbiased HSIC on dense matrices, via U-centring."""
    n = K.shape[0]
    if n < 4:
        raise ValueError(f"The unbiased HSIC estimator needs n >= 4, got n={n}.")
    return jnp.sum(_u_centre(K) * _u_centre(L)) / (n * (n - 3))


def _u_centre(K: Float[Array, "n n"]) -> Float[Array, "n n"]:
    """Székely & Rizzo (2014) U-centring: zero diagonal, off-diagonal entries

    ``K_ij - r_i / (n-2) - c_j / (n-2) + S / ((n-1)(n-2))``, with row sums
    ``r``, column sums ``c`` and total ``S`` of ``K`` with its diagonal zeroed.
    """
    n = K.shape[0]
    off = 1.0 - jnp.eye(n, dtype=K.dtype)
    # U-centring ignores a constant shift, so remove the off-diagonal mean
    # first: every sum below then runs on small numbers, not on ~1 + tiny.
    common = jnp.sum(K * off) / (n * (n - 1))
    K = (K - common) * off
    rows = reduce(K, "i j -> i", "sum") / (n - 2)
    cols = reduce(K, "i j -> j", "sum") / (n - 2)
    total = jnp.sum(K) / ((n - 1) * (n - 2))
    return (K + total - einx.add("i, j -> i j", rows, cols)) * off


def _hsic_features(
    Phi_x: Float[Array, "N Rx"],
    Phi_y: Float[Array, "N Ry"],
    estimator: Literal["biased", "unbiased"],
) -> Float[Array, ""]:
    """HSIC of ``K = Φx Φxᵀ`` and ``L = Φy Φyᵀ`` without forming them."""
    n = Phi_x.shape[0]
    if estimator == "biased":
        Cx = Phi_x - jnp.mean(Phi_x, axis=0)
        Cy = Phi_y - jnp.mean(Phi_y, axis=0)
        return _frob_sq(einsum(Cx, Cy, "n a, n b -> a b")) / (n * n)
    ones_x = jnp.ones(Phi_x.shape[1], dtype=Phi_x.dtype)
    ones_y = jnp.ones(Phi_y.shape[1], dtype=Phi_y.dtype)
    return _hsic_unbiased_low_rank((Phi_x, ones_x, Phi_x), (Phi_y, ones_y, Phi_y))


LowRankFactors = tuple[Float[Array, "N R"], Float[Array, " R"], Float[Array, "N R"]]


def _hsic_unbiased_low_rank(K: LowRankFactors, L: LowRankFactors) -> Float[Array, ""]:
    """Song et al. (2012) unbiased HSIC of ``U diag(d) Vᵀ`` factors.

    ``K~ = K - diag(K)``, so a diagonal base never contributes, and every
    term is a product of the factors minus a diagonal correction:
    ``O(N R_K R_L)``.
    """
    (Uk, dk, Vk), (Ul, dl, Vl) = K, L
    n = Uk.shape[0]
    if n < 4:
        raise ValueError(f"The unbiased HSIC estimator needs n >= 4, got n={n}.")
    # Centre the factors, i.e. K -> H K H. The estimator is invariant to it
    # (it only adds row and column constants off the diagonal), and it removes
    # the common mode that the sums below would otherwise cancel in float32.
    Uk, Vk, Ul, Vl = (_centre_columns(F) for F in (Uk, Vk, Ul, Vl))
    # einx contracts an axis over exactly two operands, so fold d in first.
    diag_k = einsum(Uk * dk, Vk, "n a, n a -> n")
    diag_l = einsum(Ul * dl, Vl, "n b, n b -> n")
    # sum_ij K_ij L_ij = tr(Kᵀ L) = sum_ab dk_a dl_b (Ukᵀ Ul)_ab (Vlᵀ Vk)_ba.
    frob = einsum(
        einsum(Uk * dk, Ul * dl, "n a, n b -> a b"),
        einsum(Vk, Vl, "n a, n b -> a b"),
        "a b, a b -> ",
    )
    trace_term = frob - jnp.sum(diag_k * diag_l)
    K1 = einsum(Uk, dk * reduce(Vk, "n a -> a", "sum"), "n a, a -> n") - diag_k
    L1 = einsum(Ul, dl * reduce(Vl, "n b -> b", "sum"), "n b, b -> n") - diag_l
    ones_term = jnp.sum(K1) * jnp.sum(L1) / ((n - 1) * (n - 2))
    cross_term = 2.0 / (n - 2) * jnp.sum(K1 * L1)
    return (trace_term + ones_term - cross_term) / (n * (n - 3))


def _double_centre(K: Float[Array, "N N"]) -> Float[Array, "N N"]:
    """``H K H`` of a dense matrix, from its row, column and grand means."""
    rows = reduce(K, "i j -> i", "mean")
    cols = reduce(K, "i j -> j", "mean")
    return K + jnp.mean(K) - einx.add("i, j -> i j", rows, cols)


def _centre_columns(F: Float[Array, "N R"]) -> Float[Array, "N R"]:
    """Subtract each column's mean: ``H F``."""
    return einx.subtract("n r, r -> n r", F, reduce(F, "n r -> r", "mean"))


def _frob_sq(A: Float[Array, "a b"]) -> Float[Array, ""]:
    return jnp.sum(A * A)


def cka(
    K_f: lx.AbstractLinearOperator,
    K_q: lx.AbstractLinearOperator,
    *,
    estimator: Literal["biased", "unbiased"] = "biased",
) -> Float[Array, ""]:
    r"""Centered kernel alignment.

    $$
    \mathrm{CKA}(K_f, K_q) = \frac{\mathrm{HSIC}(K_f, K_q)}
        {\sqrt{\mathrm{HSIC}(K_f, K_f)\, \mathrm{HSIC}(K_q, K_q)}}
    $$

    With the biased estimator and PSD kernels the value lies in ``[0, 1]``
    (the result is clipped there, against rounding) and is invariant to
    rescaling either kernel. The unbiased estimator gives debiased CKA.
    Low-rank operands on a diagonal base take the ``O(N R_f R_q)`` path of
    `hsic`.

    **Degenerate inputs.** When either self-HSIC is not positive (a constant
    variable, or a slightly negative unbiased estimate), CKA is defined as
    ``0``, with a zero gradient: a constant is independent of everything.
    That keeps it usable as a training penalty, for example for a network
    whose output starts constant. NaN inputs still give NaN.

    Args:
        K_f: First kernel matrix, shape ``(n, n)``.
        K_q: Second kernel matrix, shape ``(n, n)``.
        estimator: HSIC estimator, ``"biased"`` (default) or ``"unbiased"``.

    Returns:
        Scalar CKA.

    Examples:
        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> from kernellib.functional import cka
        >>> X = jnp.arange(10.0).reshape(5, 2)
        >>> K = lx.MatrixLinearOperator(X @ X.T, lx.symmetric_tag)
        >>> bool(jnp.allclose(cka(K, K), 1.0))
        True
    """
    cross = hsic(K_f, K_q, estimator=estimator)
    self_f = hsic(K_f, K_f, estimator=estimator)
    self_q = hsic(K_q, K_q, estimator=estimator)
    return _cka_ratio(cross, self_f, self_q, estimator)


def _cka_ratio(
    cross: Float[Array, ""],
    self_x: Float[Array, ""],
    self_y: Float[Array, ""],
    estimator: Literal["biased", "unbiased"],
) -> Float[Array, ""]:
    """``cross / sqrt(self_x * self_y)``, and 0 with a 0 gradient where either

    self term is not positive. A double ``where`` keeps ``sqrt`` away from 0
    (its slope is infinite there). A NaN in any term propagates.
    The biased ratio is clipped to ``[0, 1]``, where it lies in exact
    arithmetic.
    """
    denom = self_x * self_y
    # Test each term: two slightly negative (unbiased) estimates would give a
    # positive product. A NaN anywhere is never "degenerate", so it reaches
    # the ratio and propagates, even when the other side is constant.
    has_nan = jnp.isnan(cross) | jnp.isnan(self_x) | jnp.isnan(self_y)
    degenerate = ((self_x <= 0) | (self_y <= 0)) & ~has_nan
    ratio = jnp.where(
        degenerate, 0.0, cross / jnp.sqrt(jnp.where(degenerate, 1.0, denom))
    )
    return jnp.clip(ratio, 0.0, 1.0) if estimator == "biased" else ratio


def mmd_squared(
    K_xx: Float[Array, "Nx Nx"],
    K_yy: Float[Array, "Ny Ny"],
    K_xy: Float[Array, "Nx Ny"],
) -> Float[Array, ""]:
    r"""Biased squared Maximum Mean Discrepancy.

    Computes:

        MMD^2 = mean(K_{xx}) + mean(K_{yy}) - 2 \cdot mean(K_{xy})

    Args:
        K_xx: Kernel matrix within first sample, shape ``(Nx, Nx)``.
        K_yy: Kernel matrix within second sample, shape ``(Ny, Ny)``.
        K_xy: Cross-kernel matrix between samples, shape ``(Nx, Ny)``.

    Returns:
        Scalar biased MMD^2 estimate.

    Examples:
        >>> import jax.numpy as jnp
        >>> from kernellib.functional import mmd_squared
        >>> K = jnp.ones((3, 3))
        >>> float(mmd_squared(K, K, K))
        0.0
    """
    return jnp.mean(K_xx) + jnp.mean(K_yy) - 2.0 * jnp.mean(K_xy)
