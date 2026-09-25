"""Inner-product kernels: linear and polynomial. Defaults match pyrox-gp."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float

from kernellib._kernels._base import AbstractPointwiseKernel
from kernellib.functional import _nonstationary as _f


__all__ = [
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
        return self.variance * jnp.sum(X * X, axis=-1) + self.bias


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
        return self.variance * (jnp.sum(X * X, axis=-1) + self.bias) ** self.degree
