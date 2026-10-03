r"""Spectral densities of graph kernels, on Laplacian eigenvalues.

A spectral graph kernel is $K = U\,\Phi(\Lambda)\,U^\top$ for the Laplacian
$L = U \Lambda U^\top$ and a decreasing $\Phi \ge 0$: smooth eigenvectors
(small $\lambda$) get large prior variance. These functions return
$\Phi(\lambda)$ for given eigenvalues, the ingredient of a truncated
(rank-$M$) graph GP prior $f = U \operatorname{diag}(\sqrt{\Phi})\,\varepsilon$.

- **Heat (diffusion):** $\Phi(\lambda) = \exp(-\ell^2 \lambda / 2)$, the heat
  equation $\partial_t u = -L u$ run for time $t = \ell^2 / 2$.
- **Matérn** (Borovitskiy et al., 2021): $\Phi(\lambda) =
  (2\nu / \ell^2 + \lambda)^{-\nu}$, the graph analogue of the SPDE
  $(\kappa^2 - \Delta)^{\nu/2} f = \mathcal W$ with $\kappa^2 = 2\nu/\ell^2$.
  There is no dimension in the exponent: on a graph $\nu$ plays the role of
  the SPDE's $\alpha$, not of the Euclidean $\nu = \alpha - d/2$. As
  $\nu \to \infty$, $(2\nu/\ell^2 + \lambda)^{-\nu} \propto
  (1 + \ell^2 \lambda / 2\nu)^{-\nu} \to e^{-\ell^2 \lambda / 2}$: the heat
  spectrum.

With ``n_nodes``, $\Phi$ is scaled so that the average marginal variance of
$U \operatorname{diag}(\Phi) U^\top$ over the ``n_nodes`` nodes is
``variance``. For orthonormal $U$ that average is $\sum_k \Phi_k / N$, so the
eigenvectors are not needed, and with $M < N$ eigenpairs it normalises the
truncated prior. Both spectra are evaluated in log space, so the
normalisation is stable for any $\nu$ (the raw Matérn spectrum underflows for
large $\nu$).
"""

from __future__ import annotations

import jax.numpy as jnp
from jax.scipy.special import logsumexp
from jaxtyping import Array, Float


__all__ = ["graph_heat_spectrum", "graph_matern_spectrum"]


def graph_heat_spectrum(
    eigvals: Float[Array, " M"],
    *,
    lengthscale: float | Float[Array, ""],
    variance: float | Float[Array, ""] = 1.0,
    n_nodes: int | None = None,
) -> Float[Array, " M"]:
    r"""Heat-kernel spectrum $\Phi(\lambda) = \sigma^2 \exp(-\ell^2 \lambda / 2)$.

    Args:
        eigvals: Laplacian eigenvalues, shape ``(M,)``.
        lengthscale: $\ell$; the diffusion time is $\ell^2 / 2$.
        variance: $\sigma^2$, or the average marginal variance with
            ``n_nodes``.
        n_nodes: Number of graph nodes $N$. If given, $\Phi$ is scaled so the
            average marginal variance of $U \operatorname{diag}(\Phi) U^\top$
            is ``variance``: $\sum_k \Phi_k = N \sigma^2$.

    Returns:
        $\Phi$, shape ``(M,)``.

    Examples:
        >>> import jax.numpy as jnp
        >>> from kernellib.functional import graph_heat_spectrum
        >>> lam = jnp.array([0.0, 1.0, 2.0])
        >>> phi = graph_heat_spectrum(lam, lengthscale=1.0, n_nodes=3)
        >>> round(float(phi.sum()), 6)  # average variance 1 over 3 nodes
        3.0
    """
    log_phi = -0.5 * jnp.asarray(lengthscale) ** 2 * jnp.asarray(eigvals)
    return _scaled(log_phi, variance, n_nodes)


def graph_matern_spectrum(
    eigvals: Float[Array, " M"],
    *,
    nu: float | Float[Array, ""],
    lengthscale: float | Float[Array, ""],
    variance: float | Float[Array, ""] = 1.0,
    n_nodes: int | None = None,
) -> Float[Array, " M"]:
    r"""Graph Matérn spectrum $\Phi(\lambda) = \sigma^2 (2\nu/\ell^2 + \lambda)^{-\nu}$.

    Borovitskiy et al. (2021). $\nu$ is the graph smoothness (the SPDE's
    $\alpha$; no dimension enters the exponent), and $2\nu / \ell^2$ is the
    SPDE's $\kappa^2$. As $\nu \to \infty$, the normalised spectrum tends to
    `graph_heat_spectrum` with the same $\ell$.

    Args:
        eigvals: Laplacian eigenvalues, shape ``(M,)``.
        nu: Smoothness $\nu > 0$.
        lengthscale: $\ell > 0$.
        variance: $\sigma^2$, or the average marginal variance with
            ``n_nodes``.
        n_nodes: Number of graph nodes $N$. If given, $\Phi$ is scaled so the
            average marginal variance of $U \operatorname{diag}(\Phi) U^\top$
            is ``variance``: $\sum_k \Phi_k = N \sigma^2$.

    Returns:
        $\Phi$, shape ``(M,)``.

    Examples:
        >>> import jax.numpy as jnp
        >>> from kernellib.functional import graph_matern_spectrum
        >>> lam = jnp.array([0.0, 0.5, 1.0])
        >>> phi = graph_matern_spectrum(lam, nu=1.5, lengthscale=1.0)
        >>> bool(phi[0] > phi[1] > phi[2] > 0)  # decreasing in lambda
        True
    """
    nu = jnp.asarray(nu)
    kappa2 = 2.0 * nu / jnp.asarray(lengthscale) ** 2
    log_phi = -nu * jnp.log(kappa2 + jnp.asarray(eigvals))
    return _scaled(log_phi, variance, n_nodes)


def _scaled(
    log_phi: Float[Array, " M"],
    variance: float | Float[Array, ""],
    n_nodes: int | None,
) -> Float[Array, " M"]:
    """``variance * exp(log_phi)``, or with ``n_nodes`` rescaled so that
    ``sum(phi) = n_nodes * variance``, computed without underflow."""
    if n_nodes is None:
        return variance * jnp.exp(log_phi)
    return variance * n_nodes * jnp.exp(log_phi - logsumexp(log_phi))
