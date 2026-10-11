r"""Non-separable space-time covariances (GEO12).

`Gneiting` is the class of Gneiting (2002), eq. (14): a stationary covariance
on $\mathbb{R}^d \times \mathbb{R}$ whose spatial range grows with
the time lag, which a separable ``Product(space, time)`` cannot express.

Reference:
    Gneiting, T. (2002). Nonseparable, stationary covariance functions for
    space-time data. *Journal of the American Statistical Association*
    97(458), 590-600.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jax.core import Tracer
from jaxtyping import Array, Float

from kernellib._kernels._base import AbstractPointwiseKernel
from kernellib.functional._distances import _pairwise_sq_dist
from kernellib.functional._geo_models import _pow_r


__all__ = ["Gneiting"]


class Gneiting(AbstractPointwiseKernel):
    r"""Gneiting's non-separable space-time covariance.

    Inputs are ``(N, space_dims + 1)``: the spatial coordinates first, the time
    last. For a spatial lag $h \in \mathbb{R}^d$ and a time lag
    $u \in \mathbb{R}$ (Gneiting 2002, eq. 14),

    $$
    C(h, u) = \frac{\sigma^2}{\psi(u)^{\tau}}
    \exp\!\left(-\frac{c\,\lVert h\rVert^{2\gamma}}{\psi(u)^{\beta\gamma}}\right),
    \qquad \psi(u) = a|u|^{2\alpha} + 1,
    $$

    with $c = \ell_s^{-2\gamma}$ and $a = \ell_t^{-2\alpha}$. It is positive
    definite on $\mathbb{R}^d \times \mathbb{R}$ for $\alpha, \gamma \in (0, 1]$,
    $\beta \in [0, 1]$ and $\tau \ge d/2$.

    The separability parameter $\beta$ sets the space-time interaction: the
    effective spatial range at lag $u$ is $\ell_s\,\psi(u)^{\beta/2}$, so it
    grows with the lag for $\beta > 0$. $\beta = 0$ is separable, a temporal
    generalized Cauchy times a spatial stable model:
    ``Product(ActiveDims(Stable(alpha=2 * gamma), space),
    ActiveDims(GeneralizedCauchy(alpha=2 * alpha, beta=2 * alpha * tau), time))``.
    ``beta`` is a trainable array; when you optimise it, keep it in
    $[0, 1]$ with a reparameterisation such as ``beta = sigmoid(z)``.

    ``alpha`` = ``gamma`` = 1 is smooth; smaller values give rough sample paths
    in time or space. The powers use a safe branch at zero lag, so gradients
    with respect to the hyperparameters are finite at $h = 0$ and $u = 0$.

    Attributes:
        space_dims: Number of spatial columns $d$ (static); the input has
            ``space_dims + 1`` columns.
        variance: Scalar sill $\sigma^2$, ``> 0``.
        space_lengthscale: Scalar spatial lengthscale $\ell_s$, ``> 0``.
        time_lengthscale: Scalar temporal lengthscale $\ell_t$, ``> 0``.
        beta: Separability $\beta \in [0, 1]$ (default ``0.5``).
        alpha: Temporal smoothness $\alpha \in (0, 1]$ (default ``1.0``);
            static.
        gamma: Spatial smoothness $\gamma \in (0, 1]$ (default ``0.5``);
            static.
        tau: Temporal decay exponent $\tau \ge d/2$; ``None`` (the default)
            means $d/2$. Static.

    Raises:
        ValueError: If ``space_dims < 1``, ``alpha`` or ``gamma`` is outside
            ``(0, 1]``, ``tau < space_dims / 2``, a concrete ``beta`` is
            outside ``[0, 1]``, or the inputs do not have ``space_dims + 1``
            columns.

    Examples:
        >>> import jax.numpy as jnp
        >>> import kernellib as kl
        >>> k = kl.Gneiting(space_dims=2, beta=1.0)
        >>> X = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 1.0]])
        >>> [round(float(v), 4) for v in k(X, X)[0]]
        [1.0, 0.3679, 0.2465]
        >>> k.tau
        1.0
    """

    space_dims: int = eqx.field(static=True)
    variance: Float[Array, ""] = eqx.field(default=1.0, converter=jnp.asarray)
    space_lengthscale: Float[Array, ""] = eqx.field(default=1.0, converter=jnp.asarray)
    time_lengthscale: Float[Array, ""] = eqx.field(default=1.0, converter=jnp.asarray)
    beta: Float[Array, ""] = eqx.field(default=0.5, converter=jnp.asarray)
    alpha: float = eqx.field(default=1.0, static=True)
    gamma: float = eqx.field(default=0.5, static=True)
    tau: float | None = eqx.field(
        default=None,
        static=True,
        converter=lambda t: None if t is None else float(t),
    )

    def __post_init__(self) -> None:
        if self.tau is None:
            object.__setattr__(self, "tau", self.space_dims / 2.0)

    def __check_init__(self) -> None:
        if self.space_dims < 1:
            raise ValueError(
                f"Gneiting requires space_dims >= 1, got {self.space_dims}"
            )
        for name, value in (("alpha", self.alpha), ("gamma", self.gamma)):
            if not 0.0 < value <= 1.0:
                raise ValueError(f"Gneiting requires {name} in (0, 1], got {value}")
        tau = self.space_dims / 2.0 if self.tau is None else self.tau
        if tau < self.space_dims / 2.0:
            raise ValueError(
                f"Gneiting requires tau >= space_dims / 2 = {self.space_dims / 2.0}, "
                f"got {tau}"
            )
        # Under a transform (jit, vmap) beta may be a tracer; check only values.
        if not isinstance(self.beta, Tracer) and bool(
            jnp.any((self.beta < 0.0) | (self.beta > 1.0))
        ):
            raise ValueError(f"Gneiting requires beta in [0, 1], got {self.beta}")

    @property
    def is_stationary(self) -> bool:
        return True

    def _check_width(self, d: int) -> None:
        if d != self.space_dims + 1:
            raise ValueError(
                f"Gneiting with space_dims={self.space_dims} expects "
                f"{self.space_dims + 1} input columns (space then time), got {d}."
            )

    def _profile(
        self, h2: Float[Array, ...], u2: Float[Array, ...]
    ) -> Float[Array, ...]:
        """``C`` from the scaled squared lags ``(h / l_s)**2`` and ``(u / l_t)**2``."""
        tau = self.space_dims / 2.0 if self.tau is None else self.tau
        psi = 1.0 + _pow_r(u2, 2.0 * self.alpha)
        spatial = _pow_r(h2, 2.0 * self.gamma) * psi ** (-self.beta * self.gamma)
        return self.variance * psi ** (-tau) * jnp.exp(-spatial)

    def pairwise(
        self, x: Float[Array, " D"], y: Float[Array, " D"]
    ) -> Float[Array, ""]:
        self._check_width(x.shape[-1])
        d = self.space_dims
        h2 = jnp.sum(((x[:d] - y[:d]) / self.space_lengthscale) ** 2)
        u2 = ((x[d] - y[d]) / self.time_lengthscale) ** 2
        return self._profile(h2, u2)

    def __call__(
        self, X1: Float[Array, "N1 D"], X2: Float[Array, "N2 D"]
    ) -> Float[Array, "N1 N2"]:
        self._check_width(X1.shape[-1])
        d = self.space_dims
        h2 = _pairwise_sq_dist(X1[:, :d], X2[:, :d], self.space_lengthscale)
        u2 = _pairwise_sq_dist(X1[:, d:], X2[:, d:], self.time_lengthscale)
        return self._profile(h2, u2)

    def diag(self, X: Float[Array, "N D"]) -> Float[Array, " N"]:
        self._check_width(X.shape[-1])
        return self.variance * jnp.ones(X.shape[0], dtype=X.dtype)
