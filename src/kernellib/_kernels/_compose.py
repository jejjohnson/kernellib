"""Kernel-level composition: sums, products, scaling, input selection and
input warping.

Composites are pointwise exactly when all their children are, so a sum of
pointwise kernels still gets the matrix-free operator path. Their Gram matrix
is built from each child's own Gram path, so closed forms are kept.
"""

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from kernellib._kernels._base import AbstractKernel


__all__ = [
    "ActiveDims",
    "Product",
    "Scaled",
    "Sum",
    "Warped",
]


def _check_kernels(kernels: tuple[object, ...], name: str) -> None:
    if not kernels:
        raise ValueError(f"{name} needs at least one kernel.")
    for k in kernels:
        if not isinstance(k, AbstractKernel):
            raise TypeError(f"{name} takes kernels, got {type(k).__name__}.")


class Sum(AbstractKernel):
    """Sum of kernels, ``k(x, x') = sum_i k_i(x, x')``.

    Usually built with ``+``, which flattens nested sums.

    Examples:
        >>> import jax.numpy as jnp
        >>> import kernellib as kl
        >>> k = kl.RBF() + kl.White(0.1)
        >>> type(k).__name__, len(k.kernels)
        ('Sum', 2)
        >>> X = jnp.array([[0.0], [1.0]])
        >>> bool(jnp.allclose(k.diag(X), 1.1))
        True
    """

    kernels: tuple[AbstractKernel, ...]

    def __init__(self, *kernels: AbstractKernel) -> None:
        _check_kernels(kernels, "Sum")
        self.kernels = tuple(kernels)

    def __call__(
        self, X1: Float[Array, "N1 D"], X2: Float[Array, "N2 D"]
    ) -> Float[Array, "N1 N2"]:
        out = self.kernels[0](X1, X2)
        for k in self.kernels[1:]:
            out = out + k(X1, X2)
        return out

    def diag(self, X: Float[Array, "N D"]) -> Float[Array, " N"]:
        out = self.kernels[0].diag(X)
        for k in self.kernels[1:]:
            out = out + k.diag(X)
        return out

    def pairwise(
        self, x: Float[Array, " D"], y: Float[Array, " D"]
    ) -> Float[Array, ""]:
        out = self.kernels[0].pairwise(x, y)
        for k in self.kernels[1:]:
            out = out + k.pairwise(x, y)
        return out

    @property
    def is_pointwise(self) -> bool:
        return all(k.is_pointwise for k in self.kernels)


class Product(AbstractKernel):
    """Elementwise product of kernels, ``k(x, x') = prod_i k_i(x, x')``.

    Usually built with ``*`` between kernels, which flattens nested products.
    """

    kernels: tuple[AbstractKernel, ...]

    def __init__(self, *kernels: AbstractKernel) -> None:
        _check_kernels(kernels, "Product")
        self.kernels = tuple(kernels)

    def __call__(
        self, X1: Float[Array, "N1 D"], X2: Float[Array, "N2 D"]
    ) -> Float[Array, "N1 N2"]:
        out = self.kernels[0](X1, X2)
        for k in self.kernels[1:]:
            out = out * k(X1, X2)
        return out

    def diag(self, X: Float[Array, "N D"]) -> Float[Array, " N"]:
        out = self.kernels[0].diag(X)
        for k in self.kernels[1:]:
            out = out * k.diag(X)
        return out

    def pairwise(
        self, x: Float[Array, " D"], y: Float[Array, " D"]
    ) -> Float[Array, ""]:
        out = self.kernels[0].pairwise(x, y)
        for k in self.kernels[1:]:
            out = out * k.pairwise(x, y)
        return out

    @property
    def is_pointwise(self) -> bool:
        return all(k.is_pointwise for k in self.kernels)


class Scaled(AbstractKernel):
    """A kernel times a scalar, ``k(x, x') = scale * k_0(x, x')``.

    Usually built with ``scale * kernel``. ``scale`` is a differentiable leaf.
    """

    kernel: AbstractKernel
    scale: Float[Array, ""] = eqx.field(converter=jnp.asarray)

    def __call__(
        self, X1: Float[Array, "N1 D"], X2: Float[Array, "N2 D"]
    ) -> Float[Array, "N1 N2"]:
        return self.scale * self.kernel(X1, X2)

    def diag(self, X: Float[Array, "N D"]) -> Float[Array, " N"]:
        return self.scale * self.kernel.diag(X)

    def pairwise(
        self, x: Float[Array, " D"], y: Float[Array, " D"]
    ) -> Float[Array, ""]:
        return self.scale * self.kernel.pairwise(x, y)

    @property
    def is_pointwise(self) -> bool:
        return self.kernel.is_pointwise


class ActiveDims(AbstractKernel):
    """Apply a kernel to a subset of input dimensions.

    Attributes:
        kernel: The kernel to apply.
        dims: Input columns to keep (static).

    Examples:
        >>> import jax.numpy as jnp
        >>> import kernellib as kl
        >>> k = kl.ActiveDims(kl.Linear(), dims=(1,))
        >>> x, y = jnp.array([5.0, 2.0]), jnp.array([7.0, 3.0])
        >>> float(k.pairwise(x, y))
        6.0
    """

    kernel: AbstractKernel
    dims: tuple[int, ...] = eqx.field(static=True, converter=tuple)

    def __check_init__(self) -> None:
        if not self.dims:
            raise ValueError("ActiveDims needs at least one dimension.")

    def _select(self, X: Float[Array, "... D"]) -> Float[Array, "... d"]:
        return jnp.take(X, jnp.asarray(self.dims), axis=-1)

    def __call__(
        self, X1: Float[Array, "N1 D"], X2: Float[Array, "N2 D"]
    ) -> Float[Array, "N1 N2"]:
        return self.kernel(self._select(X1), self._select(X2))

    def diag(self, X: Float[Array, "N D"]) -> Float[Array, " N"]:
        return self.kernel.diag(self._select(X))

    def pairwise(
        self, x: Float[Array, " D"], y: Float[Array, " D"]
    ) -> Float[Array, ""]:
        return self.kernel.pairwise(self._select(x), self._select(y))

    @property
    def is_pointwise(self) -> bool:
        return self.kernel.is_pointwise


class Warped(AbstractKernel):
    """Apply a kernel to warped inputs, ``k(x, x') = k_0(w(x), w(x'))``.

    ``warp`` maps one input point ``(D,)`` to ``(D',)``. It may be a plain
    function or an equinox module with its own parameters.

    Attributes:
        kernel: The kernel on warped inputs.
        warp: Pointwise input warping.
    """

    kernel: AbstractKernel
    warp: Callable[[Float[Array, " D"]], Float[Array, " Dw"]]

    def __call__(
        self, X1: Float[Array, "N1 D"], X2: Float[Array, "N2 D"]
    ) -> Float[Array, "N1 N2"]:
        return self.kernel(jax.vmap(self.warp)(X1), jax.vmap(self.warp)(X2))

    def diag(self, X: Float[Array, "N D"]) -> Float[Array, " N"]:
        return self.kernel.diag(jax.vmap(self.warp)(X))

    def pairwise(
        self, x: Float[Array, " D"], y: Float[Array, " D"]
    ) -> Float[Array, ""]:
        return self.kernel.pairwise(self.warp(x), self.warp(y))

    @property
    def is_pointwise(self) -> bool:
        return self.kernel.is_pointwise
