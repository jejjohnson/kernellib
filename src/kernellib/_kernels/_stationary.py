"""Stationary kernels: RBF, Matern, rational quadratic, periodic, cosine,
white noise, constant.

Gram matrices agree with the matching `kernellib.functional` functions;
``tests/test_kernels.py`` checks it for every class. Defaults match
``pyrox_gp``'s kernel classes.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float

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
