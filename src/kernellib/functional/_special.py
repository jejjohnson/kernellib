r"""Special functions JAX does not provide.

Private: `log_bessel_kv` backs `RationalQuadratic.unit_spectral_density`.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Float


# Trapezoid nodes on [0, T]. The integrand below is entire and even in t, so
# the rule converges geometrically: 128 nodes match scipy.special.kve to
# ~1e-14 relative over nu in [0, 25], x in [1e-4, 3e2].
_KV_NODES: int = 128
# The truncated tail sits at least exp(-_KV_TAIL) below the integrand's peak.
_KV_TAIL: float = 45.0


def log_bessel_kv(nu: ArrayLike, x: ArrayLike) -> Float[Array, ...]:
    r"""Log of the modified Bessel function of the second kind, $\log K_\nu(x)$.

    For real order $\nu$ and $x > 0$, from the integral representation

    $$
    K_\nu(x) = e^{-x} \int_0^\infty
        \exp\!\bigl(-x (\cosh t - 1)\bigr) \cosh(\nu t)\, dt,
    $$

    evaluated by the trapezoidal rule in log space on $[0, T]$, with $T$
    chosen per $(\nu, x)$ so the discarded tail is negligible. The integrand
    is entire and even in $t$, so the rule converges geometrically, and
    because it is plain arithmetic, `jax.grad` differentiates it in both $x$
    and the order $\nu$ (for which there is no closed-form derivative).
    Working in log space keeps it finite where $K_\nu$ itself under- or
    overflows ($x \to \infty$, or $x \to 0$ with large $\nu$).

    ``nu`` and ``x`` broadcast against each other. $K_{-\nu} = K_\nu$.

    Args:
        nu: Order, any real.
        x: Argument, positive.

    Returns:
        $\log K_\nu(x)$, with the broadcast shape of ``nu`` and ``x``.

    Examples:
        >>> import jax.numpy as jnp
        >>> from kernellib.functional._special import log_bessel_kv
        >>> # K_{1/2}(x) = sqrt(pi / (2x)) e^{-x}
        >>> x = jnp.array([0.1, 1.0, 10.0])
        >>> exact = 0.5 * jnp.log(jnp.pi / (2 * x)) - x
        >>> bool(jnp.allclose(log_bessel_kv(0.5, x), exact, rtol=1e-6))
        True
    """
    nu, x = jnp.broadcast_arrays(*_promote(nu, x))
    nu = jnp.abs(nu)
    # The log-integrand g(t) = log cosh(nu t) - x (cosh t - 1) peaks at
    # sinh t = nu / x (approximately); integrate until it has dropped
    # _KV_TAIL below that peak. The endpoint only sets the grid, so it carries
    # no gradient: the integrand vanishes there.
    t_peak = jnp.arcsinh(nu / x)
    peak = nu * t_peak - x * (jnp.cosh(t_peak) - 1.0)
    T = t_peak + 1.0
    for _ in range(4):  # fixed point of x (cosh T - 1) = tail + peak + nu T
        T = jnp.arccosh(1.0 + (_KV_TAIL + peak + nu * T) / x)
    T = jax.lax.stop_gradient(jnp.maximum(T, t_peak + 1e-3))
    h = T / _KV_NODES

    nodes = jnp.arange(_KV_NODES + 1, dtype=x.dtype)
    t = h[..., None] * nodes
    nu_t = nu[..., None] * t
    log_cosh = nu_t + jnp.log1p(jnp.exp(-2.0 * nu_t)) - jnp.log(2.0)
    g = log_cosh - x[..., None] * (jnp.cosh(t) - 1.0)
    weights = jnp.ones(_KV_NODES + 1, dtype=x.dtype).at[0].set(0.5).at[-1].set(0.5)
    g_max = jax.lax.stop_gradient(jnp.max(g, axis=-1))
    log_sum = jnp.log(jnp.sum(weights * jnp.exp(g - g_max[..., None]), axis=-1))
    return g_max + log_sum + jnp.log(h) - x


def _promote(nu: ArrayLike, x: ArrayLike) -> tuple[Array, Array]:
    dtype = jnp.result_type(nu, x, jnp.float32)
    return jnp.asarray(nu, dtype=dtype), jnp.asarray(x, dtype=dtype)
