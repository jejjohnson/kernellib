"""Non-stationary kernels: linear, polynomial and distance-induced.

Linear and polynomial defaults match pyrox-gp.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float

from kernellib._einx import reduce
from kernellib._kernels._base import AbstractPointwiseKernel, GramParts
from kernellib.functional import _nonstationary as _f


__all__ = [
    "Distance",
    "Linear",
    "Polynomial",
]


class Linear(AbstractPointwiseKernel):
    r"""Linear kernel, ``k(x, x') = sigma^2 x^T x' + b``.

    Attributes:
        variance: Scalar multiplier on the dot product.
        bias: Scalar additive bias.

    Examples:
        >>> import jax.numpy as jnp
        >>> import kernellib as kl
        >>> x = jnp.array([1.0, 2.0])
        >>> float(kl.Linear(bias=0.5).pairwise(x, x))
        5.5
    """

    variance: Float[Array, ""] = eqx.field(default=1.0, converter=jnp.asarray)
    bias: Float[Array, ""] = eqx.field(default=0.0, converter=jnp.asarray)

    def pairwise(
        self, x: Float[Array, " D"], y: Float[Array, " D"]
    ) -> Float[Array, ""]:
        return self.variance * jnp.dot(x, y) + self.bias

    def __call__(
        self, X1: Float[Array, "N1 D"], X2: Float[Array, "N2 D"]
    ) -> Float[Array, "N1 N2"]:
        return _f.linear_kernel(X1, X2, self.variance, self.bias)

    def diag(self, X: Float[Array, "N D"]) -> Float[Array, " N"]:
        return self.variance * reduce(X * X, "n d -> n", "sum") + self.bias

    def _gram_structure(self, X: Float[Array, "N D"]) -> GramParts:
        # sigma^2 X Xᵀ + b 11ᵀ: rank D + 1 (a zero bias is a zero-weight column).
        n, d = X.shape
        factors = jnp.concatenate([X, jnp.ones((n, 1), dtype=X.dtype)], axis=1)
        dtype = jnp.result_type(self.variance, self.bias, X)
        weights = jnp.concatenate(
            [
                jnp.full((d,), self.variance, dtype=dtype),
                jnp.atleast_1d(self.bias).astype(dtype),
            ]
        )
        return GramParts(factors=factors, weights=weights)


class Polynomial(AbstractPointwiseKernel):
    r"""Polynomial kernel, ``k(x, x') = sigma^2 (x^T x' + b)^d``.

    ``degree`` is static.

    Attributes:
        variance: Scalar multiplier.
        bias: Scalar additive bias inside the power.
        degree: Positive integer degree (default 2, as in pyrox-gp).

    Raises:
        ValueError: If ``degree < 1``.
    """

    variance: Float[Array, ""] = eqx.field(default=1.0, converter=jnp.asarray)
    bias: Float[Array, ""] = eqx.field(default=0.0, converter=jnp.asarray)
    degree: int = eqx.field(default=2, static=True)

    def __check_init__(self) -> None:
        if self.degree < 1:
            raise ValueError(f"Polynomial requires degree >= 1, got {self.degree!r}")

    def pairwise(
        self, x: Float[Array, " D"], y: Float[Array, " D"]
    ) -> Float[Array, ""]:
        return self.variance * (jnp.dot(x, y) + self.bias) ** self.degree

    def __call__(
        self, X1: Float[Array, "N1 D"], X2: Float[Array, "N2 D"]
    ) -> Float[Array, "N1 N2"]:
        return _f.polynomial_kernel(X1, X2, self.variance, self.bias, self.degree)

    def diag(self, X: Float[Array, "N D"]) -> Float[Array, " N"]:
        return (
            self.variance
            * (reduce(X * X, "n d -> n", "sum") + self.bias) ** self.degree
        )


class Distance(AbstractPointwiseKernel):
    r"""Distance-induced kernel, ``k(x, x') = σ² (|x|^a + |x'|^a - |x - x'|^a) / 2``.

    Sejdinovic et al. (2013): with this kernel HSIC is a quarter of the
    distance covariance and MMD² half the energy distance (Székely et al.,
    2007), so `distance_covariance_squared`, `distance_correlation_squared`
    and `energy_distance` are `hsic`, `cka` and `mmd_squared` under it. The
    anchor is the origin; HSIC and MMD do not depend on it.

    PSD for ``0 < exponent <= 2``. ``exponent=2`` is the linear kernel;
    smaller exponents weight large distances less and suit heavy tails.
    ``exponent`` is static. Gradients are finite at coincident points. The
    GP it defines is mean-square differentiable only at ``exponent=2``, so
    `Derivative` and `DerivativeIndexed` reject smaller exponents. `hsic`,
    `cka` and `mmd_squared` centre its inputs, since the kernel is anchored
    at the origin and the centred statistics are translation invariant.

    Attributes:
        variance: Scalar multiplier.
        exponent: The exponent ``a`` in ``(0, 2]`` (default 1, the standard
            distance covariance).

    Raises:
        ValueError: If ``exponent`` is outside ``(0, 2]``.

    Examples:
        >>> import jax.numpy as jnp
        >>> import kernellib as kl
        >>> x, y = jnp.array([3.0, 0.0]), jnp.array([0.0, 4.0])
        >>> float(kl.Distance().pairwise(x, y))  # (3 + 4 - 5) / 2
        1.0
        >>> float(kl.Distance(exponent=2.0).pairwise(x, x))  # x . x
        9.0
    """

    variance: Float[Array, ""] = eqx.field(default=1.0, converter=jnp.asarray)
    exponent: float = eqx.field(default=1.0, static=True)

    def __check_init__(self) -> None:
        _f._check_exponent(self.exponent)

    def pairwise(
        self, x: Float[Array, " D"], y: Float[Array, " D"]
    ) -> Float[Array, ""]:
        a = self.exponent
        if a == 2.0:
            return self.variance * jnp.dot(x, y)
        diff = x - y
        return (
            0.5
            * self.variance
            * (
                _f._norm_pow(jnp.dot(x, x), a)
                + _f._norm_pow(jnp.dot(y, y), a)
                - _f._norm_pow(jnp.dot(diff, diff), a)
            )
        )

    def __call__(
        self, X1: Float[Array, "N1 D"], X2: Float[Array, "N2 D"]
    ) -> Float[Array, "N1 N2"]:
        return _f.distance_kernel(X1, X2, self.variance, self.exponent)

    def diag(self, X: Float[Array, "N D"]) -> Float[Array, " N"]:
        return self.variance * _f._norm_pow(
            reduce(X * X, "n d -> n", "sum"), self.exponent
        )
