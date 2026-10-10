r"""Exact Fourier (Bessel-series) features of the `Periodic` kernel.

With $z = 1 / \ell^2$ and $\omega = 2\pi / p$, the periodic kernel is
$k(\tau) = \sigma^2 e^{-z} \exp(z \cos \omega\tau)$, and the Jacobi-Anger
expansion $e^{z\cos\theta} = I_0(z) + 2\sum_{k \ge 1} I_k(z) \cos k\theta$
gives its Mercer series (Solin & Särkkä, 2014),

$$
k(\tau) = q_0 + \sum_{k \ge 1} q_k \cos(k\omega\tau), \qquad
q_0 = \sigma^2 \tilde I_0(z), \quad q_k = 2\sigma^2 \tilde I_k(z),
$$

with $\tilde I_k(z) = e^{-z} I_k(z)$ the exponentially scaled modified Bessel
functions. Truncating at $K$ harmonics and splitting each cosine of a
difference into a cosine and a sine product gives $2K + 1$ deterministic
features.
"""

from __future__ import annotations

import dataclasses
import functools
import math

import einx
import equinox as eqx
import jax
import jax.numpy as jnp
from jax.scipy.special import i0e
from jaxtyping import Array, Float

from kernellib._einx import rearrange
from kernellib._kernels import AbstractKernel, Periodic
from kernellib._spectral._base import AbstractFeatureMap


__all__ = ["PeriodicFeatures"]


def _scaled_bessel_i(
    z: Float[Array, ""], n_max: int, n_start: int | None = None
) -> Float[Array, " n_start"]:
    r"""$\tilde I_k(z) = e^{-z} I_k(z)$ for $k = 0, \dots, n_{start}$.

    Miller's backward recurrence $I_{k-1} = I_{k+1} + \frac{2k}{z} I_k$, run
    on the ratios $r_k = I_k / I_{k-1} = z / (2k + z\, r_{k+1})$ so that
    nothing overflows, started at $r_{n_{start} + 1} = 0$ and normalised by
    ``i0e(z)``. The ratios are the continued fraction of the minimal solution,
    so the error at index $k$ falls like $\exp(-(n_{start}^2 - k^2) / z)$ for
    large $z$: start $\gtrsim 8\sqrt{z}$ above the largest index needed.
    Differentiable in ``z`` through the recurrence.

    Args:
        z: Non-negative argument.
        n_max: Largest order needed (the result has at least ``n_max + 1``
            entries).
        n_start: Start of the recurrence; ``2 * n_max + 32`` if ``None``.

    Returns:
        ``(n_start + 1,)`` values, indexed by order; the leading
        ``n_max + 1`` are accurate.
    """
    if n_start is None:
        n_start = 2 * n_max + 32
    z = jnp.asarray(z, dtype=jnp.result_type(z, 1.0))
    return _bessel_series(z, max(n_start, n_max))


@functools.partial(jax.jit, static_argnums=1)
def _bessel_series(z: Float[Array, ""], n_start: int) -> Float[Array, " n_start"]:
    """The recurrence of `_scaled_bessel_i`, compiled once per ``n_start``."""
    orders = jnp.arange(n_start, 0, -1, dtype=z.dtype)

    def step(r_next, k):
        r = z / (2.0 * k + z * r_next)
        return r, r

    _, r_desc = jax.lax.scan(step, jnp.zeros((), dtype=z.dtype), orders)
    ratios = r_desc[::-1]  # r_1, ..., r_{n_start}
    i0 = i0e(z)
    return einx.id(", k -> (1 + k)", i0, i0 * jnp.cumprod(ratios))


class PeriodicFeatures(AbstractFeatureMap):
    r"""Exact truncated Fourier features of a 1-D `Periodic` kernel.

    $$
    \phi(x) = \big[\sqrt{q_0},\
    \sqrt{q_k}\cos(k\omega x)_{k=1..K},\ \sqrt{q_k}\sin(k\omega x)_{k=1..K}\big],
    $$

    with $q_0 = \sigma^2 e^{-z} I_0(z)$, $q_k = 2\sigma^2 e^{-z} I_k(z)$,
    $z = 1/\ell^2$ and $\omega = 2\pi / p$ (Solin & Särkkä, 2014), so that
    $\phi(x)^\top\phi(x') = k(x, x')$ up to the dropped harmonics $k > K$.
    Columns are the constant, then the cosines, then the sines (the order of
    ``geonnax.basis.seasonal_features``). There is no sampling: the error is
    the truncation tail alone, at most
    $\sigma^2$ · `truncation_tail` in every entry of the Gram matrix. Cost
    $O(K)$ per input.

    The coefficients decay like $\exp(-k^2 / 2z)$, so a short lengthscale
    needs $K$ of a few times $\sqrt{z} = 1/\ell$. Smallest $K$ with
    `truncation_tail` below the target:

    | $\ell$ | $10^{-6}$ | $10^{-12}$ |
    |---|---|---|
    | 2 | 4 | 8 |
    | 1 | 7 | 11 |
    | 0.5 | 11 | 18 |
    | 0.2 | 25 | 38 |
    | 0.1 | 49 | 73 |
    | 0.05 | 98 | 143 |

    Roughly, $K \approx 5/\ell$ and $7/\ell$ for short lengthscales.

    The kernel's hyperparameters are read at call time, so gradients reach
    them. Only ``D = 1``: for ``D > 1`` `Periodic` acts on the Euclidean
    distance, which is not separable (and not PSD); use `Periodised` for
    multi-dimensional periodicity, or a Kronecker product of per-axis maps.

    Attributes:
        n_harmonics: Number of harmonics $K$; there are ``2K + 1`` features.
        n_recurrence: Start index of the Bessel recurrence. ``None`` lets
            `fit` choose ``K + 32 + ceil(8 / lengthscale)`` from the fitted
            lengthscale (``2K + 32`` when the lengthscale is traced). Refit,
            or set it, if the lengthscale shrinks a lot after `fit`.
        kernel: The fitted `Periodic` kernel, ``None`` before `fit`.

    Examples:
        >>> import einx
        >>> import jax.numpy as jnp
        >>> import kernellib as kl
        >>> X = einx.id("n -> n 1", jnp.linspace(0.0, 3.0, 40))
        >>> k = kl.Periodic(lengthscale=0.7, variance=2.0, period=1.0)
        >>> pf = kl.PeriodicFeatures(10).fit(k, X)
        >>> Phi = pf(X)
        >>> Phi.shape
        (40, 21)
        >>> bool(pf.truncation_tail() < 1e-6)
        True
        >>> err = jnp.abs(einx.dot("n f, m f -> n m", Phi, Phi) - k(X, X))
        >>> bool(jnp.max(err) < 1e-5)
        True
    """

    n_harmonics: int = eqx.field(static=True)
    n_recurrence: int | None = eqx.field(default=None, static=True)
    kernel: Periodic | None = None

    def __check_init__(self) -> None:
        if self.n_harmonics < 1:
            raise ValueError(f"n_harmonics must be >= 1, got {self.n_harmonics}.")
        if self.n_recurrence is not None and self.n_recurrence < self.n_harmonics:
            raise ValueError(
                f"n_recurrence ({self.n_recurrence}) must be >= n_harmonics "
                f"({self.n_harmonics})."
            )

    def fit(self, kernel: AbstractKernel, X: Float[Array, "N 1"]) -> PeriodicFeatures:
        """Return a copy fitted to a `Periodic` kernel on 1-D inputs.

        Raises:
            TypeError: If ``kernel`` is not a `Periodic`.
            ValueError: If ``X`` is not ``(N, 1)``.
        """
        if not isinstance(kernel, Periodic):
            raise TypeError(
                "PeriodicFeatures needs a Periodic kernel, got "
                f"{type(kernel).__name__}."
            )
        _check_1d(X)
        n_recurrence = self.n_recurrence
        if n_recurrence is None:
            try:
                ell = float(kernel.lengthscale)
            except (
                jax.errors.ConcretizationTypeError,
                jax.errors.TracerArrayConversionError,
            ):
                n_recurrence = 2 * self.n_harmonics + 32
            else:
                n_recurrence = self.n_harmonics + 32 + math.ceil(8.0 / abs(ell))
        return dataclasses.replace(self, n_recurrence=n_recurrence, kernel=kernel)

    def _scaled_coefficients(self) -> Float[Array, " n_start"]:
        r"""$\tilde I_k(z)$ for $k = 0, \dots,$ ``n_recurrence``."""
        assert self.kernel is not None
        z = 1.0 / jnp.asarray(self.kernel.lengthscale) ** 2
        return _scaled_bessel_i(z, self.n_harmonics, self.n_recurrence)

    @property
    def coefficients(self) -> Float[Array, " K1"]:
        r"""Mercer weights $q_0, \dots, q_K$; non-negative, sum $\le \sigma^2$."""
        assert self.kernel is not None
        i_k = self._scaled_coefficients()[: self.n_harmonics + 1]
        weight = jnp.where(jnp.arange(self.n_harmonics + 1) == 0, 1.0, 2.0)
        return self.kernel.variance * weight * i_k

    def truncation_tail(self) -> Float[Array, ""]:
        r"""Relative variance beyond $K$, $1 - \sum_{k \le K} q_k / \sigma^2$.

        Computed as $2\sum_{k > K} \tilde I_k(z)$ from the rest of the
        recurrence rather than by subtraction, so it stays accurate far below
        machine epsilon and decreases monotonically in $K$.
        """
        i_k = self._scaled_coefficients()
        return 2.0 * jnp.sum(i_k[self.n_harmonics + 1 :])

    def features(self, X: Float[Array, "N 1"]) -> Float[Array, "N F"]:
        assert self.kernel is not None
        _check_1d(X)
        q = self.coefficients.astype(X.dtype)
        # High harmonics underflow to q = 0 at long lengthscales; the double
        # where keeps sqrt's infinite derivative there out of the gradient.
        positive = q > 0.0
        sqrt_q = jnp.where(positive, jnp.sqrt(jnp.where(positive, q, 1.0)), 0.0)
        omega = 2.0 * jnp.pi / self.kernel.period
        harmonics = jnp.arange(1, self.n_harmonics + 1, dtype=X.dtype)
        theta = einx.multiply("n, k -> n k", rearrange(X, "n 1 -> n"), harmonics)
        theta = theta * omega
        cos = einx.multiply("n k, k -> n k", jnp.cos(theta), sqrt_q[1:])
        sin = einx.multiply("n k, k -> n k", jnp.sin(theta), sqrt_q[1:])
        const = jnp.full(X.shape[0], sqrt_q[0], dtype=X.dtype)
        return einx.id("n, n k, n k -> n (1 + k + k)", const, cos, sin)


def _check_1d(X: Float[Array, "N D"]) -> None:
    if X.ndim != 2 or X.shape[-1] != 1:
        raise ValueError(
            "PeriodicFeatures supports 1-D inputs of shape (N, 1) only, got "
            f"shape {X.shape}. For D > 1, Periodic acts on the Euclidean "
            "distance, which is not separable; use Periodised for "
            "multi-dimensional periodicity, or a Kronecker product of per-axis "
            "feature maps (GEO24)."
        )
