r"""Random Fourier feature prior draws for stationary kernels.

Moved from ``pyrox_gp._basis._rff``. A single prior function path is
factored into ``(variance, lengthscale, omega, phase, weights)`` and
evaluated at arbitrary inputs as

$$
\tilde{f}(x) = \sum_{j=1}^F w_j
    \sqrt{2 \sigma^2 / F}\,
    \cos\!\bigl(\omega_j^\top (x / \ell) + b_j\bigr),
\qquad w_j \sim \mathcal{N}(0, 1),
\quad b_j \sim \mathrm{Unif}(0, 2\pi),
$$

with $\omega_j$ drawn from the kernel's unit-lengthscale spectral density
(`AbstractStationaryKernel.sample_unit_frequencies`). The frequencies are
kept free of the lengthscale so a caller can pair one draw with
hyperparameters resolved elsewhere, which is how pyrox-gp's pathwise
samplers reuse the values cached on a conditioned GP.

The helpers are stateless, deterministic in a PRNG key, batched along a
leading path axis, and friendly to ``jax.jit`` / ``jax.grad``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.typing import DTypeLike
from jaxtyping import Array, Float, PRNGKeyArray

from kernellib._einx import einsum, rearrange
from kernellib._kernels import AbstractKernel, AbstractStationaryKernel


__all__ = ["draw_rff_cosine_basis", "evaluate_rff_cosine_paths"]


def draw_rff_cosine_basis(
    kernel: AbstractKernel,
    key: PRNGKeyArray,
    *,
    n_paths: int,
    n_features: int,
    in_features: int,
    dtype: DTypeLike | None = None,
) -> tuple[
    Float[Array, ""],
    Float[Array, ""] | Float[Array, " D"],
    Float[Array, "S D F"],
    Float[Array, "S F"],
    Float[Array, "S F"],
]:
    """Draw ``(variance, lengthscale, omega, phase, weights)`` for a kernel.

    Args:
        kernel: A stationary kernel with a spectral sampler (`RBF`,
            `Matern`, `RationalQuadratic`).
        key: PRNG key, split internally into frequency / phase / weight keys.
        n_paths: Number of independent prior function draws ``S``.
        n_features: Number of random features per draw ``F``.
        in_features: Input dimension ``D``.
        dtype: Floating dtype for all outputs; JAX's default float if ``None``.

    Returns:
        ``(variance, lengthscale, omega, phase, weights)``: the kernel's
        hyperparameters, unit-lengthscale frequencies of shape ``(S, D, F)``,
        and phases and weights of shape ``(S, F)``.

    Raises:
        ValueError: If ``n_paths < 1`` or ``n_features < 1``.
        NotImplementedError: If the kernel is not stationary or has no
            spectral sampler.

    Examples:
        >>> import jax
        >>> import kernellib as kl
        >>> basis = kl.draw_rff_cosine_basis(
        ...     kl.RBF(), jax.random.key(0), n_paths=2, n_features=16, in_features=3
        ... )
        >>> [a.shape for a in basis]
        [(), (), (2, 3, 16), (2, 16), (2, 16)]
    """
    if n_features < 1:
        raise ValueError(f"n_features must be >= 1, got {n_features}.")
    if n_paths < 1:
        raise ValueError(f"n_paths must be >= 1, got {n_paths}.")
    if not isinstance(kernel, AbstractStationaryKernel):
        raise NotImplementedError(
            "Random Fourier features need a stationary kernel with a spectral "
            f"sampler; got {type(kernel).__name__}."
        )
    dtype = jnp.result_type(float) if dtype is None else dtype

    freq_key, phase_key, weight_key = jax.random.split(key, 3)
    omega = kernel.sample_unit_frequencies(
        freq_key, (n_paths, n_features, in_features), dtype
    )
    omega = rearrange(omega, "s f d -> s d f")
    phase = jax.random.uniform(
        phase_key,
        shape=(n_paths, n_features),
        minval=0.0,
        maxval=2.0 * jnp.pi,
        dtype=dtype,
    )
    weights = jax.random.normal(weight_key, shape=(n_paths, n_features), dtype=dtype)
    variance = jnp.asarray(kernel.variance, dtype=dtype)
    lengthscale = jnp.asarray(kernel.lengthscale, dtype=dtype)
    return variance, lengthscale, omega, phase, weights


def evaluate_rff_cosine_paths(
    X: Float[Array, "N D"],
    *,
    variance: Float[Array, ""],
    lengthscale: Float[Array, ""] | Float[Array, " D"],
    omega: Float[Array, "S D F"],
    phase: Float[Array, "S F"],
    weights: Float[Array, "S F"],
) -> Float[Array, "S N"]:
    r"""Evaluate the zero-mean RFF prior path(s) at inputs ``X``.

    $$
    \tilde f_s(x_n) = \sum_{j=1}^{F} w_{s,j}\,
        \sqrt{2\sigma^2 / F}\,
        \cos\!\bigl(\omega_{s,\cdot,j}^\top (x_n / \ell) + b_{s,j}\bigr),
    $$

    vectorised over path index ``s`` and input index ``n``. The empirical
    covariance of the paths converges to the kernel as ``F`` grows.

    Args:
        X: Inputs, shape ``(N, D)``.
        variance: Kernel variance.
        lengthscale: Scalar or ``(D,)`` (ARD) lengthscale.
        omega: Unit-lengthscale frequencies, shape ``(S, D, F)``.
        phase: Phases, shape ``(S, F)``.
        weights: Feature weights, shape ``(S, F)``.

    Returns:
        Path values, shape ``(S, N)``.

    Raises:
        ValueError: If an ARD lengthscale does not match the feature axes of
            ``X`` and ``omega``.

    Examples:
        >>> import jax
        >>> import jax.numpy as jnp
        >>> import kernellib as kl
        >>> v, ell, omega, phase, w = kl.draw_rff_cosine_basis(
        ...     kl.RBF(), jax.random.key(0), n_paths=4, n_features=32, in_features=1
        ... )
        >>> X = jnp.linspace(-1.0, 1.0, 5)[:, None]
        >>> kl.evaluate_rff_cosine_paths(
        ...     X, variance=v, lengthscale=ell, omega=omega, phase=phase, weights=w
        ... ).shape
        (4, 5)
    """
    # Scale along the *input* axis before contracting it. For a scalar
    # lengthscale this equals dividing the projection, but it is also correct
    # for a ``(D,)`` ARD lengthscale, where dividing the ``(S, N, F)``
    # projection would broadcast F against D.
    if jnp.ndim(lengthscale) != 0:
        d = jnp.size(lengthscale)
        if X.shape[-1] != d or omega.shape[-2] != d:
            raise ValueError(
                f"ARD lengthscale of size {d} requires inputs and frequencies "
                f"with that many features; got X with {X.shape[-1]} and omega "
                f"with {omega.shape[-2]}. A singleton feature axis on X would "
                "broadcast silently and repeat one coordinate across every "
                "dimension."
            )
    projected = einsum(X / lengthscale, omega, "n d, s d f -> s n f")
    angles = projected + rearrange(phase, "s f -> s 1 f")
    features = jnp.sqrt(2.0 * variance / omega.shape[-1]) * jnp.cos(angles)
    return einsum(features, weights, "s n f, s f -> s n")
