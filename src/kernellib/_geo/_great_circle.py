r"""Isotropic kernels of the great-circle angle, positive definite on the sphere.

Every kernel here is $k(x, x') = \sigma^2\,\psi(\theta(x, x') / c)$, with
$\theta \in [0, \pi]$ the great-circle angle between the ``(lon, lat)`` points
and $c = \ell / R$ the lengthscale in radians. ``lengthscale`` is in the units
of ``radius``: kilometres with ``radius=EARTH_RADIUS_KM``.

An isotropic function of $\theta$ is positive definite on the sphere only for
special profiles $\psi$: the RBF, and the Matérn with $\nu > \tfrac12$, are not
(Gneiting 2013; Huang, Zhang & Robeson 2011). The profiles here are the
families of Gneiting (2013), Table 1, with their parameter constraints
enforced. The last three are compactly supported: they vanish for
$\theta \ge c$, which requires $c \le \pi$.

References:
    Gneiting, T. (2013). Strictly and non-strictly positive definite functions
    on spheres. *Bernoulli* 19(4), 1327-1349.
"""

from __future__ import annotations

import math
from abc import abstractmethod
from typing import ClassVar, Literal

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float

from kernellib._einx import rearrange
from kernellib._kernels._base import AbstractPointwiseKernel
from kernellib.functional._geo import great_circle_distance


__all__ = [
    "AbstractGreatCircleKernel",
    "GreatCircleAskey",
    "GreatCircleCauchy",
    "GreatCircleExponential",
    "GreatCirclePoweredExponential",
    "GreatCircleSpherical",
    "GreatCircleWendland",
]


def _safe_pow(t: Float[Array, ...], p: float) -> Float[Array, ...]:
    """``t ** p`` for ``t >= 0`` with a zero (not NaN) gradient at ``t = 0``."""
    positive = t > 0
    return jnp.where(positive, jnp.where(positive, t, 1.0) ** p, 0.0)


def _truncated(t: Float[Array, ...], p: float) -> Float[Array, ...]:
    """``(1 - t)_+ ** p``, clamped before the power so gradients beyond the
    support are 0, not NaN."""
    return jnp.where(t < 1.0, 1.0 - t, 0.0) ** p


def _check_alpha(name: str, alpha: float) -> None:
    if not 0.0 < alpha <= 1.0:
        raise ValueError(
            f"{name} is positive definite on the sphere only for alpha in "
            f"(0, 1]; got {alpha!r}."
        )


class AbstractGreatCircleKernel(AbstractPointwiseKernel):
    r"""$k(x, x') = \sigma^2\,\psi(\theta(x, x') / c)$ on ``(lon, lat)`` inputs.

    $\theta$ is the great-circle angle (`great_circle_distance` with
    ``radius=1``) and $c = \ell / R$ the lengthscale in radians. Subclasses
    implement the unit profile `profile` $\psi(t)$, with $\psi(0) = 1$.

    For compactly supported profiles $c \le \pi$ (``lengthscale <= π·radius``)
    is required. It is checked at construction when ``lengthscale`` is
    concrete; under a transformation (``jit``, ``grad``, ``vmap``) it cannot
    be, and stays the caller's responsibility.

    Attributes:
        lengthscale: Scalar lengthscale, in units of ``radius`` (km with
            ``radius=EARTH_RADIUS_KM``).
        variance: Scalar signal variance.
        radius: Sphere radius (static); ``1.0`` makes ``lengthscale`` an angle
            in radians.
        degrees: Whether inputs are in degrees (static; else radians).
    """

    lengthscale: Float[Array, ""] = eqx.field(default=1.0, converter=jnp.asarray)
    variance: Float[Array, ""] = eqx.field(default=1.0, converter=jnp.asarray)
    radius: float = eqx.field(default=1.0, static=True)
    degrees: bool = eqx.field(default=True, static=True)

    compact: ClassVar[bool] = False
    """Whether the profile vanishes for ``t >= 1`` (requires ``c <= π``)."""

    def __check_init__(self) -> None:
        if not type(self).compact:
            return
        try:
            ell = float(self.lengthscale)
        except TypeError:  # a tracer: not checkable here
            return
        if ell > math.pi * self.radius * (1.0 + 1e-12):
            raise ValueError(
                f"{type(self).__name__} is positive definite on the sphere only "
                f"for lengthscale <= π·radius = {math.pi * self.radius:g}; got "
                f"{ell:g}."
            )

    @abstractmethod
    def profile(self, t: Float[Array, ...]) -> Float[Array, ...]:
        r"""Unit profile $\psi(t)$ of the scaled angle $t = \theta / c \ge 0$."""
        raise NotImplementedError

    def __call__(
        self, X1: Float[Array, "N1 2"], X2: Float[Array, "N2 2"]
    ) -> Float[Array, "N1 N2"]:
        d = great_circle_distance(X1, X2, radius=self.radius, degrees=self.degrees)
        return self.variance * self.profile(d / self.lengthscale)

    def pairwise(
        self, x: Float[Array, " 2"], y: Float[Array, " 2"]
    ) -> Float[Array, ""]:
        return self(rearrange(x, "d -> 1 d"), rearrange(y, "d -> 1 d"))[0, 0]

    def diag(self, X: Float[Array, "N 2"]) -> Float[Array, " N"]:
        X = jnp.asarray(X)
        return self.variance * jnp.ones(X.shape[0], dtype=jnp.result_type(X, 1.0))


class GreatCircleExponential(AbstractGreatCircleKernel):
    r"""Exponential kernel of the great-circle angle, $\psi(t) = e^{-t}$.

    The Matérn with $\nu = \tfrac12$; positive definite on every sphere for
    any $c > 0$ (Gneiting 2013, Table 1). Smoother Matérns of $\theta$ are not.

    Examples:
        >>> import jax.numpy as jnp
        >>> import kernellib as kl
        >>> k = kl.GreatCircleExponential(
        ...     lengthscale=1000.0, radius=kl.EARTH_RADIUS_KM
        ... )
        >>> X = jnp.array([[0.0, 0.0], [0.0, 10.0]])  # 10 degrees ~ 1112 km apart
        >>> k(X, X).round(4).tolist()
        [[1.0, 0.3289], [0.3289, 1.0]]
    """

    def profile(self, t: Float[Array, ...]) -> Float[Array, ...]:
        return jnp.exp(-t)


class GreatCirclePoweredExponential(AbstractGreatCircleKernel):
    r"""Powered exponential kernel, $\psi(t) = e^{-t^\alpha}$.

    Positive definite on every sphere for $c > 0$ and $\alpha \in (0, 1]$
    (Gneiting 2013, Table 1). $\alpha = 2$, the RBF, is not.

    Attributes:
        alpha: Static exponent in ``(0, 1]`` (default 1, the exponential).

    Raises:
        ValueError: If ``alpha`` is outside ``(0, 1]``.

    Examples:
        >>> import jax.numpy as jnp
        >>> import kernellib as kl
        >>> k = kl.GreatCirclePoweredExponential(alpha=0.5, degrees=False)
        >>> X = jnp.array([[0.0, 0.0], [0.25, 0.0]])  # 0.25 rad apart
        >>> round(float(k(X, X)[0, 1]), 6)  # exp(-0.5)
        0.606531
    """

    alpha: float = eqx.field(default=1.0, static=True)

    def __check_init__(self) -> None:
        _check_alpha(type(self).__name__, self.alpha)

    def profile(self, t: Float[Array, ...]) -> Float[Array, ...]:
        return jnp.exp(-_safe_pow(t, self.alpha))


class GreatCircleCauchy(AbstractGreatCircleKernel):
    r"""Generalized Cauchy kernel, $\psi(t) = (1 + t^\alpha)^{-\tau/\alpha}$.

    Positive definite on every sphere for $c > 0$, $\alpha \in (0, 1]$ and
    $\tau > 0$ (Gneiting 2013, Table 1). Its tail decays like $t^{-\tau}$.

    Attributes:
        alpha: Static shape exponent in ``(0, 1]`` (default 1).
        tau: Static tail exponent, ``> 0`` (default 1).

    Raises:
        ValueError: If ``alpha`` is outside ``(0, 1]`` or ``tau <= 0``.

    Examples:
        >>> import jax.numpy as jnp
        >>> import kernellib as kl
        >>> k = kl.GreatCircleCauchy(tau=2.0, degrees=False)
        >>> X = jnp.array([[0.0, 0.0], [1.0, 0.0]])  # t = 1
        >>> round(float(k(X, X)[0, 1]), 6)  # (1 + 1)^-2
        0.25
    """

    alpha: float = eqx.field(default=1.0, static=True)
    tau: float = eqx.field(default=1.0, static=True)

    def __check_init__(self) -> None:
        _check_alpha(type(self).__name__, self.alpha)
        if not self.tau > 0.0:
            raise ValueError(f"GreatCircleCauchy requires tau > 0; got {self.tau!r}.")

    def profile(self, t: Float[Array, ...]) -> Float[Array, ...]:
        return (1.0 + _safe_pow(t, self.alpha)) ** (-self.tau / self.alpha)


class GreatCircleSpherical(AbstractGreatCircleKernel):
    r"""The geostatistics spherical model of the great-circle angle.

    $$
    \psi(t) = \left(1 + \frac{t}{2}\right)(1 - t)_+^2
            = 1 - \frac{3}{2}t + \frac{1}{2}t^3 \quad (t \le 1),
    $$

    compactly supported on $\theta < c$. Positive definite on the sphere
    $S^2$ for $c \in (0, \pi]$ (Gneiting 2013, Table 1).

    Raises:
        ValueError: If a concrete ``lengthscale`` exceeds ``π·radius``.

    Examples:
        >>> import jax.numpy as jnp
        >>> import kernellib as kl
        >>> k = kl.GreatCircleSpherical(lengthscale=1.0, degrees=False)
        >>> X = jnp.array([[0.0, 0.0], [0.5, 0.0], [1.5, 0.0]])
        >>> k(X, X)[0].tolist()  # t = 0, 0.5, 1.5
        [1.0, 0.3125, 0.0]
    """

    compact: ClassVar[bool] = True

    def profile(self, t: Float[Array, ...]) -> Float[Array, ...]:
        return (1.0 + 0.5 * t) * _truncated(t, 2.0)


class GreatCircleAskey(AbstractGreatCircleKernel):
    r"""Askey's truncated power kernel, $\psi(t) = (1 - t)_+^\tau$.

    Compactly supported on $\theta < c$; positive definite on the sphere for
    $c \in (0, \pi]$ and $\tau \ge 2$ (Gneiting 2013, Table 1).

    Attributes:
        tau: Static exponent, ``>= 2`` (default 2).

    Raises:
        ValueError: If ``tau < 2`` or a concrete ``lengthscale`` exceeds
            ``π·radius``.

    Examples:
        >>> import jax.numpy as jnp
        >>> import kernellib as kl
        >>> k = kl.GreatCircleAskey(lengthscale=1.0, degrees=False)
        >>> X = jnp.array([[0.0, 0.0], [0.5, 0.0]])
        >>> float(k(X, X)[0, 1])  # (1 - 0.5)^2
        0.25
    """

    tau: float = eqx.field(default=2.0, static=True)

    compact: ClassVar[bool] = True

    def __check_init__(self) -> None:
        if not self.tau >= 2.0:
            raise ValueError(f"GreatCircleAskey requires tau >= 2; got {self.tau!r}.")

    def profile(self, t: Float[Array, ...]) -> Float[Array, ...]:
        return _truncated(t, self.tau)


class GreatCircleWendland(AbstractGreatCircleKernel):
    r"""C²- or C⁴-Wendland kernel of the great-circle angle.

    $$
    \psi_{2}(t) = (1 + \tau t)(1 - t)_+^\tau, \qquad
    \psi_{4}(t) = \left(1 + \tau t + \frac{(\tau^2 - 1)}{3} t^2\right)
        (1 - t)_+^\tau,
    $$

    compactly supported on $\theta < c$. Positive definite on the sphere for
    $c \in (0, \pi]$ with $\tau \ge 4$ (``order=2``) or $\tau \ge 6$
    (``order=4``) (Gneiting 2013, Table 1).

    Attributes:
        order: Static smoothness, ``2`` (C², default) or ``4`` (C⁴).
        tau: Static exponent; ``None`` (default) takes the minimum for
            ``order``, 4 or 6.

    Raises:
        ValueError: If ``order`` is not 2 or 4, ``tau`` is below the minimum
            for ``order``, or a concrete ``lengthscale`` exceeds ``π·radius``.

    Examples:
        >>> import jax.numpy as jnp
        >>> import kernellib as kl
        >>> k = kl.GreatCircleWendland(lengthscale=1.0, degrees=False)
        >>> X = jnp.array([[0.0, 0.0], [0.5, 0.0]])
        >>> float(k(X, X)[0, 1])  # (1 + 4 * 0.5)(1 - 0.5)^4
        0.1875
    """

    order: Literal[2, 4] = eqx.field(default=2, static=True)
    tau: float | None = eqx.field(default=None, static=True)

    compact: ClassVar[bool] = True

    def __check_init__(self) -> None:
        if self.order not in (2, 4):
            raise ValueError(
                f"GreatCircleWendland supports order 2 or 4; got {self.order!r}."
            )
        minimum = 4.0 if self.order == 2 else 6.0
        if self.tau is not None and not self.tau >= minimum:
            raise ValueError(
                f"GreatCircleWendland(order={self.order}) requires tau >= "
                f"{minimum:g}; got {self.tau!r}."
            )

    @property
    def exponent(self) -> float:
        """The exponent $\\tau$ in use: ``tau``, or the minimum for ``order``."""
        if self.tau is not None:
            return float(self.tau)
        return 4.0 if self.order == 2 else 6.0

    def profile(self, t: Float[Array, ...]) -> Float[Array, ...]:
        tau = self.exponent
        poly = 1.0 + tau * t
        if self.order == 4:
            poly = poly + (tau * tau - 1.0) / 3.0 * t * t
        return poly * _truncated(t, tau)
