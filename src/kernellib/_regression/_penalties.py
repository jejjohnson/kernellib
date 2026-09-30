r"""Quadratic penalties for `KRR`: dependence (HSIC) and graph smoothness.

`KRR` with a penalty operator $M$ minimises
$\frac1l\|J(y - K\alpha)\|^2 + \lambda\,\alpha^\top K\alpha
+ \mu\,\alpha^\top K M K\alpha$. These build $M$ for the two common cases.

- `hsic_penalty`: $M = H K_s H / n^2$, so the penalty is the biased HSIC
  between the predictions $f = K\alpha$ (under a linear kernel) and protected
  attributes $S$. That is fair kernel learning (Pérez-Suay et al., 2017).
- `laplacian_penalty`: $M = L / n^2$, so the penalty is the smoothness
  $f^\top L f$ along a graph. That is Laplacian-regularised least squares
  (Belkin, Niyogi & Sindhwani, 2006).
"""

from __future__ import annotations

from typing import Literal

import gaussx as gx
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Float

from kernellib._graph._laplacian import graph_laplacian
from kernellib._kernels import AbstractKernel, Linear
from kernellib._spectral import AbstractFeatureMap


__all__ = ["hsic_penalty", "laplacian_penalty"]


def hsic_penalty(
    kernel: AbstractKernel,
    S: Float[Array, "N P"],
    *,
    approx: AbstractFeatureMap | None = None,
) -> lx.AbstractLinearOperator:
    r"""The HSIC penalty operator $M = H K_s H / n^2$ on protected attributes.

    With it, $\alpha^\top K M K \alpha$ is the biased HSIC between the
    predictions $f = K\alpha$ (under a linear kernel) and ``S`` under
    ``kernel``, so `KRR` pushes its predictions towards independence from
    ``S`` as ``penalty_weight`` grows.

    When $K_s$ has an exact low-rank form, the result is a pure low-rank
    `gaussx.LowRankUpdate` (zero diagonal base), which `KRR` solves by
    Woodbury at the cost of ``r + 1`` ordinary KRR solves:

    - a `Linear` kernel: $H K_s H = v\,H S S^\top H$ (the bias cancels under
      centring), so ``r = P``, the number of attributes;
    - ``approx``: a feature map fitted on ``S``, $K_s \approx \Phi\Phi^\top$.

    Any other kernel gives a dense ``N x N`` operator.

    Args:
        kernel: Kernel on the protected attributes.
        S: Protected attributes, shape ``(N, P)``, aligned with the training
            inputs.
        approx: Optional unfitted feature map for a low-rank $K_s$.

    Returns:
        The penalty operator $M$, shape ``(N, N)``.

    Examples:
        >>> import jax
        >>> import kernellib as kl
        >>> S = jax.random.normal(jax.random.key(0), (50, 2))
        >>> M = kl.hsic_penalty(kl.Linear(), S)
        >>> M.U.shape  # rank 2: Woodbury in KRR
        (50, 2)
    """
    S = jnp.asarray(S)
    if S.ndim == 1:
        S = S[:, None]
    n = S.shape[0]
    if approx is not None or isinstance(kernel, Linear):
        if approx is not None:
            Phi = approx.fit(kernel, S)(S)
            weights = jnp.ones(Phi.shape[1], dtype=Phi.dtype)
        else:
            assert isinstance(kernel, Linear)
            Phi = S
            weights = jnp.full(S.shape[1], kernel.variance, dtype=S.dtype)
        Q = (Phi - jnp.mean(Phi, axis=0)) / n
        zero = lx.DiagonalLinearOperator(jnp.zeros(n, dtype=Q.dtype))
        return gx.LowRankUpdate(
            base=zero, U=Q, d=weights, V=Q, tags=frozenset({lx.symmetric_tag})
        )
    K = kernel(S, S)
    Kc = K - jnp.mean(K, axis=0)[None, :] - jnp.mean(K, axis=1)[:, None] + jnp.mean(K)
    return lx.MatrixLinearOperator(
        Kc / n**2, frozenset({lx.symmetric_tag, lx.positive_semidefinite_tag})
    )


def laplacian_penalty(
    W: Float[Array, "N N"],
    *,
    normalization: Literal["unnormalized", "symmetric"] = "unnormalized",
) -> lx.AbstractLinearOperator:
    r"""The graph-smoothness penalty operator $M = L / n^2$.

    With it, $\alpha^\top K M K \alpha = f^\top L f / n^2$ for
    $f = K\alpha$: the Dirichlet energy
    $\tfrac12\sum_{ij} W_{ij}(f_i - f_j)^2$ of the predictions on the graph,
    which is Laplacian-regularised least squares when combined with `KRR`'s
    ``mask`` for unlabelled points.

    Args:
        W: Symmetric, non-negative adjacency matrix over the training points,
            e.g. from `adjacency_matrix`.
        normalization: ``"unnormalized"`` ($L = D - W$, the default) or
            ``"symmetric"``.

    Returns:
        The penalty operator $M$, shape ``(N, N)``.

    Examples:
        >>> import jax.numpy as jnp
        >>> import kernellib as kl
        >>> W = jnp.array([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
        >>> f = jnp.array([1.0, 2.0, 4.0])
        >>> float(f @ kl.laplacian_penalty(W).mv(f) * 3**2)  # (1-2)^2 + (2-4)^2
        5.0
    """
    n = W.shape[0]
    L = graph_laplacian(W, normalization)
    return lx.MatrixLinearOperator(
        L / n**2, frozenset({lx.symmetric_tag, lx.positive_semidefinite_tag})
    )
