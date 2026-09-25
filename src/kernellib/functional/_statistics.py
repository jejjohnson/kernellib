"""Matrix-level kernel statistics: centering, HSIC, CKA, MMD.

Arrays or operators in, scalar or operator out. `centering_operator`,
`center_kernel`, the biased `hsic` and `mmd_squared` are moved from
``gaussx._kernels._kernel_approx`` unchanged; the unbiased HSIC estimator and
`cka` are new. Kernel-and-data versions (``kernellib.hsic(kx, ky, X, Y)``)
build on these.
"""

from __future__ import annotations

from typing import Literal

import gaussx as gx
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Float


__all__ = [
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
) -> lx.MatrixLinearOperator:
    r"""Center a kernel matrix: ``H K H``.

    Computes the centered Gram matrix where ``H = I - (1/n) 11^T``.
    Used in kernel PCA, HSIC, and other centered kernel methods.

    Args:
        K: Kernel (Gram) matrix operator, shape ``(n, n)``.

    Returns:
        Centered kernel operator, shape ``(n, n)``.

    Examples:
        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> from kernellib.functional import center_kernel
        >>> K = lx.MatrixLinearOperator(jnp.ones((3, 3)) + jnp.eye(3))
        >>> Kc = center_kernel(K).as_matrix()
        >>> bool(jnp.allclose(Kc.sum(axis=0), 0.0))
        True
    """
    K_mat = K.as_matrix()
    row_mean = jnp.mean(K_mat, axis=1, keepdims=True)
    col_mean = jnp.mean(K_mat, axis=0, keepdims=True)
    total_mean = jnp.mean(K_mat)
    K_centered = K_mat - row_mean - col_mean + total_mean
    tags = frozenset()
    if lx.is_symmetric(K):
        tags = frozenset({lx.symmetric_tag})
    return lx.MatrixLinearOperator(K_centered, tags)


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
        n = K_f_centered.as_matrix().shape[0]
        return gx.trace_product(K_f_centered, K_q_centered) / (n * n)
    if estimator == "unbiased":
        return _hsic_unbiased(K_f.as_matrix(), K_q.as_matrix())
    raise ValueError(f"estimator must be 'biased' or 'unbiased', got {estimator!r}.")


def _hsic_unbiased(K: Float[Array, "n n"], L: Float[Array, "n n"]) -> Float[Array, ""]:
    """Song et al. (2012) unbiased HSIC on dense matrices."""
    n = K.shape[0]
    if n < 4:
        raise ValueError(f"The unbiased HSIC estimator needs n >= 4, got n={n}.")
    K_t = K - jnp.diag(jnp.diag(K))
    L_t = L - jnp.diag(jnp.diag(L))
    # tr(K_t L_t) = sum_ij K_t[i, j] L_t[i, j] for symmetric kernel matrices.
    trace_term = jnp.sum(K_t * L_t)
    ones_term = jnp.sum(K_t) * jnp.sum(L_t) / ((n - 1) * (n - 2))
    cross_term = 2.0 / (n - 2) * jnp.sum(K_t @ L_t)
    return (trace_term + ones_term - cross_term) / (n * (n - 3))


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
    and is invariant to rescaling either kernel. The unbiased estimator gives
    debiased CKA; its self-HSIC terms can be non-positive for very small
    samples, in which case the result is not finite.

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
    return cross / jnp.sqrt(self_f * self_q)


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
