"""Low-rank kernel operators: Nystrom and random Fourier features.

Moved from ``gaussx._kernels._kernel_approx``. Both return a
`gaussx.LowRankUpdate`, so solves and log-determinants go through Woodbury
without ever forming the ``N x N`` matrix.
"""

from __future__ import annotations

import gaussx as gx
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Float


__all__ = [
    "nystrom_operator",
    "rff_operator",
]


def nystrom_operator(
    K_XZ: Float[Array, "N M"],
    K_ZZ_op: lx.AbstractLinearOperator,
) -> gx.LowRankUpdate:
    r"""Nystrom low-rank kernel approximation.

    Approximates ``K_{XX} \approx K_{XZ} K_{ZZ}^{-1} K_{ZX}`` as a
    `LowRankUpdate` with zero base:

        K_{XX} \approx U D U^T

    where ``U = K_{XZ} L_{ZZ}^{-T}`` and ``D = I`` (i.e. ``UU^T``),
    with ``L_{ZZ} = cholesky(K_{ZZ})``.

    Args:
        K_XZ: Cross-covariance between data and inducing points,
            shape ``(N, M)``.
        K_ZZ_op: Inducing-point covariance operator, shape ``(M, M)``.

    Returns:
        `LowRankUpdate` operator of shape ``(N, N)``.
    """

    L = gx.cholesky(K_ZZ_op)
    # U = K_XZ @ L^{-T} = solve(L^T, K_XZ^T)^T
    # Solve L @ A_col = K_XZ^T_col for each column
    K_ZX = K_XZ.T  # (M, N)
    A = gx.solve_columns(L, K_ZX)
    U = A.T  # (N, M)

    N = K_XZ.shape[0]
    M = K_XZ.shape[1]
    base = lx.DiagonalLinearOperator(jnp.zeros(N))
    D = jnp.ones(M)
    return gx.LowRankUpdate(
        base=base,
        U=U,
        d=D,
        V=U,
        tags=frozenset({lx.symmetric_tag, lx.positive_semidefinite_tag}),
    )


def rff_operator(
    X: Float[Array, "N D"],
    omega: Float[Array, "D_rff D"],
    b: Float[Array, " D_rff"],
) -> gx.LowRankUpdate:
    r"""Random Fourier Features kernel approximation.

    Approximates ``K_{XX} \approx \Phi \Phi^T`` where:

        \Phi_{i,j} = \sqrt{2/D_{rff}} \cos(X_i \cdot \omega_j + b_j)

    Returns a `LowRankUpdate` that never materializes
    the ``N x N`` matrix.

    Args:
        X: Data points, shape ``(N, D)``.
        omega: Random frequencies, shape ``(D_rff, D)``.
            Sample from the spectral density of the kernel.
        b: Random phase offsets, shape ``(D_rff,)``.
            Sample uniformly from ``[0, 2*pi]``.

    Returns:
        `LowRankUpdate` operator of shape ``(N, N)``.
    """
    D_rff = omega.shape[0]
    N = X.shape[0]
    Phi = jnp.sqrt(2.0 / D_rff) * jnp.cos(X @ omega.T + b[None, :])  # (N, D_rff)

    base = lx.DiagonalLinearOperator(jnp.zeros(N))
    D = jnp.ones(D_rff)
    return gx.LowRankUpdate(
        base=base,
        U=Phi,
        d=D,
        V=Phi,
        tags=frozenset({lx.symmetric_tag, lx.positive_semidefinite_tag}),
    )
