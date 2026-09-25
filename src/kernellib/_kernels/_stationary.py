"""Stationary kernels: RBF, Matern, rational quadratic, periodic, cosine,
white noise, constant.

Gram matrices agree with the matching `kernellib.functional` functions;
``tests/test_kernels.py`` checks it for every class. Defaults match
``pyrox_gp``'s kernel classes.
"""

from __future__ import annotations

import math

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.typing import DTypeLike
from jaxtyping import Array, Float, PRNGKeyArray

from kernellib._kernels._base import AbstractPointwiseKernel, AbstractStationaryKernel
from kernellib.functional import _stationary as _f


__all__ = [
    "RBF",
    "Constant",
    "Cosine",
    "Matern",
    "Periodic",
    "RationalQuadratic",
    "White",
]


def _float_dtype(dtype: DTypeLike | None) -> DTypeLike:
    # jax.random needs a concrete float dtype; None means JAX's default.
    return jnp.result_type(float) if dtype is None else dtype


def _sqrt_safe(r2: Float[Array, ...]) -> Float[Array, ...]:
    # Jitter inside sqrt keeps gradients finite at r = 0, as in functional.
    return jnp.sqrt(jnp.clip(r2, min=1e-30))


class RBF(AbstractStationaryKernel):
    r"""Radial basis function (squared exponential) kernel.

    $$
    k(x, x') = \sigma^2 \exp\!\left(-\tfrac12 \|(x - x')/\ell\|^2\right)
    $$

    Attributes:
        lengthscale: Scalar (isotropic) or ``(D,)`` (ARD) lengthscale.
        variance: Scalar signal variance.

    Examples:
        >>> import jax.numpy as jnp
        >>> import kernellib as kl
        >>> k = kl.RBF(lengthscale=1.0, variance=2.0)
        >>> X = jnp.array([[0.0], [1.0]])
        >>> bool(jnp.allclose(k.diag(X), 2.0))
        True
    """

    lengthscale: Float[Array, ""] | Float[Array, " D"] = eqx.field(
        default=1.0, converter=jnp.asarray
    )
    variance: Float[Array, ""] = eqx.field(default=1.0, converter=jnp.asarray)

    def shape(self, r2: Float[Array, ...]) -> Float[Array, ...]:
        return jnp.exp(-0.5 * r2)

    def unit_spectral_density(
        self, omega_sq: Float[Array, ...], d: int
    ) -> Float[Array, ...]:
        r"""$s(\omega) = (2\pi)^{d/2} \exp(-\|\omega\|^2 / 2)$."""
        return (2.0 * math.pi) ** (d / 2.0) * jnp.exp(-0.5 * omega_sq)

    def sample_unit_frequencies(
        self,
        key: PRNGKeyArray,
        shape: tuple[int, ...],
        dtype: DTypeLike | None = None,
    ) -> Float[Array, ...]:
        """Standard normal frequencies."""
        return jax.random.normal(key, shape, dtype=_float_dtype(dtype))


class Matern(AbstractStationaryKernel):
    r"""Matern kernel with closed-form ``nu in {0.5, 1.5, 2.5}``.

    ``nu`` is static: it selects a code path, not an optimisation target.

    Attributes:
        lengthscale: Scalar (isotropic) or ``(D,)`` (ARD) lengthscale.
        variance: Scalar signal variance.
        nu: Smoothness, ``0.5``, ``1.5`` or ``2.5`` (default, as in pyrox-gp).

    Raises:
        ValueError: If ``nu`` is not one of the supported values.

    Examples:
        >>> import jax.numpy as jnp
        >>> import kernellib as kl
        >>> k = kl.Matern(nu=0.5)
        >>> x, y = jnp.array([0.0]), jnp.array([1.0])
        >>> bool(jnp.allclose(k.pairwise(x, y), jnp.exp(-1.0)))
        True
    """

    lengthscale: Float[Array, ""] | Float[Array, " D"] = eqx.field(
        default=1.0, converter=jnp.asarray
    )
    variance: Float[Array, ""] = eqx.field(default=1.0, converter=jnp.asarray)
    nu: float = eqx.field(default=2.5, static=True)

    def __check_init__(self) -> None:
        if self.nu not in (0.5, 1.5, 2.5):
            raise ValueError(
                f"Matern supports nu in {{0.5, 1.5, 2.5}}, got {self.nu!r}"
            )

    def shape(self, r2: Float[Array, ...]) -> Float[Array, ...]:
        r = _sqrt_safe(r2)
        if self.nu == 0.5:
            return jnp.exp(-r)
        if self.nu == 1.5:
            a = jnp.sqrt(3.0) * r
            return (1.0 + a) * jnp.exp(-a)
        a = jnp.sqrt(5.0) * r
        return (1.0 + a + (a * a) / 3.0) * jnp.exp(-a)

    def unit_spectral_density(
        self, omega_sq: Float[Array, ...], d: int
    ) -> Float[Array, ...]:
        r"""Matern density in ``d`` dimensions.

        $$
        s(\omega) = \frac{2^d \pi^{d/2}\, \Gamma(\nu + d/2)\, (2\nu)^\nu}
        {\Gamma(\nu)}\, (2\nu + \|\omega\|^2)^{-(\nu + d/2)}
        $$
        """
        nu = self.nu
        log_c = (
            d * math.log(2.0)
            + (d / 2.0) * math.log(math.pi)
            + math.lgamma(nu + d / 2.0)
            - math.lgamma(nu)
            + nu * math.log(2.0 * nu)
        )
        return math.exp(log_c) * (2.0 * nu + omega_sq) ** (-(nu + d / 2.0))

    def sample_unit_frequencies(
        self,
        key: PRNGKeyArray,
        shape: tuple[int, ...],
        dtype: DTypeLike | None = None,
    ) -> Float[Array, ...]:
        """Multivariate Student-t frequencies with ``2 nu`` degrees of freedom.

        Drawn jointly as ``g * sqrt(nu / u)``, ``g ~ N(0, I)``,
        ``u ~ Gamma(nu)`` shared across the last axis. A coordinate-wise
        Student-t draw would give a product of 1-D densities, which is the
        Matern spectrum only for ``d = 1``.
        """
        dtype = _float_dtype(dtype)
        key_g, key_u = jax.random.split(key)
        g = jax.random.normal(key_g, shape, dtype=dtype)
        u = jax.random.gamma(key_u, self.nu, (*shape[:-1], 1), dtype=dtype)
        return g * jnp.sqrt(self.nu / u)


class RationalQuadratic(AbstractStationaryKernel):
    r"""Rational quadratic kernel, a scale mixture of RBFs.

    $$
    k(x, x') = \sigma^2 \left(1 + \frac{r^2}{2\alpha}\right)^{-\alpha}
    $$

    Attributes:
        lengthscale: Scalar (isotropic) or ``(D,)`` (ARD) lengthscale.
        variance: Scalar signal variance.
        alpha: Positive shape parameter.
    """

    lengthscale: Float[Array, ""] | Float[Array, " D"] = eqx.field(
        default=1.0, converter=jnp.asarray
    )
    variance: Float[Array, ""] = eqx.field(default=1.0, converter=jnp.asarray)
    alpha: Float[Array, ""] = eqx.field(default=1.0, converter=jnp.asarray)

    def shape(self, r2: Float[Array, ...]) -> Float[Array, ...]:
        return (1.0 + r2 / (2.0 * self.alpha)) ** (-self.alpha)

    def sample_unit_frequencies(
        self,
        key: PRNGKeyArray,
        shape: tuple[int, ...],
        dtype: DTypeLike | None = None,
    ) -> Float[Array, ...]:
        """Frequencies of the Gamma scale mixture of RBFs.

        ``k`` is ``E[exp(-tau r^2 / 2)]`` with ``tau ~ Gamma(alpha, rate=alpha)``,
        so a frequency is ``g * sqrt(tau)`` with ``g ~ N(0, I)``. The density
        itself needs a modified Bessel function of the second kind, which JAX
        does not provide, so `spectral_density` is not available.
        """
        dtype = _float_dtype(dtype)
        key_g, key_tau = jax.random.split(key)
        g = jax.random.normal(key_g, shape, dtype=dtype)
        alpha = jnp.asarray(self.alpha, dtype=dtype)
        tau = jax.random.gamma(key_tau, alpha, (*shape[:-1], 1), dtype=dtype) / alpha
        return g * jnp.sqrt(tau)


class Periodic(AbstractPointwiseKernel):
    r"""Periodic (MacKay) kernel on the Euclidean distance.

    $$
    k(x, x') = \sigma^2 \exp\!\left(-\frac{2\sin^2(\pi \|x - x'\| / p)}{\ell^2}\right)
    $$

    Stationary, but not of the ``shape(||(x - x') / lengthscale||²)`` form (the
    lengthscale acts after the sine), so it subclasses
    `AbstractPointwiseKernel`. Isotropic only, as in pyrox-gp.

    Warning:
        Applied to the Euclidean distance, this kernel is positive
        semidefinite for 1-D inputs but not in general for ``D > 1``. For
        multi-dimensional periodicity use a `Product` of per-dimension
        periodic kernels via `ActiveDims`.

    Attributes:
        lengthscale: Scalar lengthscale.
        variance: Scalar signal variance.
        period: Scalar period ``p``.
    """

    lengthscale: Float[Array, ""] = eqx.field(default=1.0, converter=jnp.asarray)
    variance: Float[Array, ""] = eqx.field(default=1.0, converter=jnp.asarray)
    period: Float[Array, ""] = eqx.field(default=1.0, converter=jnp.asarray)

    def pairwise(
        self, x: Float[Array, " D"], y: Float[Array, " D"]
    ) -> Float[Array, ""]:
        r = _sqrt_safe(jnp.sum((x - y) ** 2))
        sinsq = jnp.sin(jnp.pi * r / self.period) ** 2
        return self.variance * jnp.exp(-2.0 * sinsq / (self.lengthscale**2))

    def __call__(
        self, X1: Float[Array, "N1 D"], X2: Float[Array, "N2 D"]
    ) -> Float[Array, "N1 N2"]:
        return _f.periodic_kernel(X1, X2, self.variance, self.lengthscale, self.period)

    def diag(self, X: Float[Array, "N D"]) -> Float[Array, " N"]:
        return self.variance * jnp.ones(X.shape[0], dtype=X.dtype)


class Cosine(AbstractPointwiseKernel):
    r"""Cosine kernel, ``sigma^2 cos(2 pi ||x - x'|| / p)``. Can go negative.

    Like `Periodic`, it is positive semidefinite for 1-D inputs only.

    Attributes:
        variance: Scalar signal variance.
        period: Scalar period.
    """

    variance: Float[Array, ""] = eqx.field(default=1.0, converter=jnp.asarray)
    period: Float[Array, ""] = eqx.field(default=1.0, converter=jnp.asarray)

    def pairwise(
        self, x: Float[Array, " D"], y: Float[Array, " D"]
    ) -> Float[Array, ""]:
        r = _sqrt_safe(jnp.sum((x - y) ** 2))
        return self.variance * jnp.cos(2.0 * jnp.pi * r / self.period)

    def __call__(
        self, X1: Float[Array, "N1 D"], X2: Float[Array, "N2 D"]
    ) -> Float[Array, "N1 N2"]:
        return _f.cosine_kernel(X1, X2, self.variance, self.period)

    def diag(self, X: Float[Array, "N D"]) -> Float[Array, " N"]:
        return self.variance * jnp.ones(X.shape[0], dtype=X.dtype)


class White(AbstractPointwiseKernel):
    r"""White-noise kernel, ``sigma^2 delta(x, x')``.

    Attributes:
        variance: Scalar noise variance.
    """

    variance: Float[Array, ""] = eqx.field(default=1.0, converter=jnp.asarray)

    def pairwise(
        self, x: Float[Array, " D"], y: Float[Array, " D"]
    ) -> Float[Array, ""]:
        return self.variance * jnp.all(x == y).astype(x.dtype)

    def __call__(
        self, X1: Float[Array, "N1 D"], X2: Float[Array, "N2 D"]
    ) -> Float[Array, "N1 N2"]:
        return _f.white_kernel(X1, X2, self.variance)

    def diag(self, X: Float[Array, "N D"]) -> Float[Array, " N"]:
        return self.variance * jnp.ones(X.shape[0], dtype=X.dtype)


class Constant(AbstractPointwiseKernel):
    r"""Constant kernel, ``k(x, x') = sigma^2``.

    Attributes:
        variance: Scalar value.
    """

    variance: Float[Array, ""] = eqx.field(default=1.0, converter=jnp.asarray)

    def pairwise(
        self, x: Float[Array, " D"], y: Float[Array, " D"]
    ) -> Float[Array, ""]:
        return self.variance * jnp.ones((), dtype=x.dtype)

    def __call__(
        self, X1: Float[Array, "N1 D"], X2: Float[Array, "N2 D"]
    ) -> Float[Array, "N1 N2"]:
        return _f.constant_kernel(X1, X2, self.variance)

    def diag(self, X: Float[Array, "N D"]) -> Float[Array, " N"]:
        return self.variance * jnp.ones(X.shape[0], dtype=X.dtype)
