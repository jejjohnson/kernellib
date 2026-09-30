"""Graph kernels: spectral functions of the Laplacian (Smola & Kondor, 2003)."""

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, Float

from kernellib._graph._laplacian import Normalization, graph_laplacian


__all__ = [
    "commute_time_kernel",
    "cosine_graph_kernel",
    "diffusion_kernel",
    "random_walk_kernel",
    "regularized_laplacian_kernel",
]


def _spectral(
    W: Float[Array, "N N"], fn, normalization: Normalization
) -> Float[Array, "N N"]:
    """``U f(Λ) Uᵀ`` for the (symmetric) Laplacian ``U Λ Uᵀ``."""
    if normalization == "random_walk":
        raise ValueError("Graph kernels need a symmetric Laplacian.")
    lam, U = jnp.linalg.eigh(graph_laplacian(W, normalization))
    return (U * fn(jnp.clip(lam, min=0.0))) @ U.T


def diffusion_kernel(
    W: Float[Array, "N N"],
    beta: float | Float[Array, ""] = 1.0,
    *,
    normalization: Normalization = "symmetric",
) -> Float[Array, "N N"]:
    r"""Diffusion (heat) kernel $K = \exp(-\beta L)$ (Kondor & Lafferty, 2002).

    Heat diffusing on the graph for time $\beta$: small $\beta$ is local,
    large $\beta$ spreads similarity along the graph.

    Examples:
        >>> import jax.numpy as jnp
        >>> import kernellib as kl
        >>> W = jnp.array([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
        >>> K = kl.diffusion_kernel(W, beta=0.5)
        >>> bool(K[0, 1] > K[0, 2] > 0)  # neighbours are more similar
        True
    """
    return _spectral(W, lambda lam: jnp.exp(-beta * lam), normalization)


def regularized_laplacian_kernel(
    W: Float[Array, "N N"],
    sigma: float | Float[Array, ""] = 1.0,
    *,
    normalization: Normalization = "symmetric",
) -> Float[Array, "N N"]:
    r"""Regularised Laplacian kernel $K = (I + \sigma^2 L)^{-1}$.

    Smola & Kondor (2003); $\sigma$ sets how far similarity spreads.
    """
    return _spectral(W, lambda lam: 1.0 / (1.0 + sigma**2 * lam), normalization)


def random_walk_kernel(
    W: Float[Array, "N N"],
    p: int = 1,
    a: float = 2.0,
    *,
    normalization: Normalization = "symmetric",
) -> Float[Array, "N N"]:
    r"""$p$-step random-walk kernel $K = (aI - L)^p$ (Smola & Kondor, 2003).

    Positive semidefinite for $a \ge \lambda_{\max}(L)$, i.e. $a \ge 2$ with
    the symmetric normalisation (the default).

    Raises:
        ValueError: If ``p < 1``.
    """
    if p < 1:
        raise ValueError(f"p must be >= 1, got {p}.")
    return _spectral(W, lambda lam: (a - lam) ** p, normalization)


def cosine_graph_kernel(
    W: Float[Array, "N N"], *, normalization: Normalization = "symmetric"
) -> Float[Array, "N N"]:
    r"""Inverse-cosine kernel $K = \cos(\pi L / 4)$ (Smola & Kondor, 2003).

    Positive semidefinite for the symmetric normalisation, whose spectrum lies
    in $[0, 2]$.
    """
    return _spectral(W, lambda lam: jnp.cos(jnp.pi * lam / 4.0), normalization)


def commute_time_kernel(
    W: Float[Array, "N N"],
    *,
    normalization: Normalization = "unnormalized",
    rtol: float = 1e-10,
) -> Float[Array, "N N"]:
    r"""Commute-time kernel $K = L^{+}$, the Laplacian's pseudo-inverse.

    $K_{ii} + K_{jj} - 2K_{ij}$ is proportional to the expected commute time of
    a random walk between nodes $i$ and $j$ (Fouss et al., 2007). Eigenvalues
    below ``rtol`` times the largest (the constant vector on each connected
    component) are treated as zero.
    """

    def pinv(lam: Float[Array, " N"]) -> Float[Array, " N"]:
        keep = lam > rtol * jnp.max(lam)
        return jnp.where(keep, 1.0 / jnp.where(keep, lam, 1.0), 0.0)

    return _spectral(W, pinv, normalization)
