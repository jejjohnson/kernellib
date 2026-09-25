"""The kernel contract: three abstract levels, each adding one capability.

- `AbstractKernel` produces a Gram matrix. Only ``__call__`` is abstract, so
  Gram-only kernels (multi-output, structured) subclass it directly. This is
  the class ``pyrox_gp.Kernel`` becomes an alias of.
- `AbstractPointwiseKernel` is defined by ``k(x, x')``. The pointwise form is
  what autodiff and the matrix-free operators use.
- `AbstractStationaryKernel` is ``variance * shape(r²)`` with
  ``r² = ||(x - x') / lengthscale||²``. It overrides the Gram path with the
  closed-form, rank-2 distance expansion.

Hyperparameters are plain array fields: no transforms, no priors. The
modelling layer above adds those.
"""

from __future__ import annotations

from abc import abstractmethod

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from kernellib.functional._distances import _pairwise_sq_dist


__all__ = [
    "AbstractKernel",
    "AbstractPointwiseKernel",
    "AbstractStationaryKernel",
]


class AbstractKernel(eqx.Module):
    """Anything that produces a Gram matrix.

    Subclasses implement ``__call__``. ``gram`` and ``diag`` derive from it;
    override ``diag`` when a cheaper form exists. ``+`` and ``*`` build
    `Sum`, `Product` and `Scaled` kernels.
    """

    @abstractmethod
    def __call__(
        self, X1: Float[Array, "N1 D"], X2: Float[Array, "N2 D"]
    ) -> Float[Array, "N1 N2"]:
        """Gram matrix ``K(X1, X2)``."""
        raise NotImplementedError

    def gram(self, X: Float[Array, "N D"]) -> Float[Array, "N N"]:
        """Symmetric Gram matrix ``K(X, X)``."""
        return self(X, X)

    def diag(self, X: Float[Array, "N D"]) -> Float[Array, " N"]:
        """Diagonal of ``K(X, X)``. Default: extract from the full Gram."""
        return jnp.diag(self(X, X))

    @property
    def is_pointwise(self) -> bool:
        """Whether ``pairwise`` is available (and so the implicit operators)."""
        return False

    def pairwise(
        self, x: Float[Array, " D"], y: Float[Array, " D"]
    ) -> Float[Array, ""]:
        """Scalar ``k(x, y)``. Gram-only kernels do not provide it."""
        raise TypeError(
            f"{type(self).__name__} is a Gram-only kernel and has no pointwise "
            "form; subclass AbstractPointwiseKernel to provide pairwise(x, y)."
        )

    def __add__(self, other: object) -> AbstractKernel:
        if not isinstance(other, AbstractKernel):
            return NotImplemented
        from kernellib._kernels._compose import Sum

        return Sum(*_flatten(self, Sum), *_flatten(other, Sum))

    def __mul__(self, other: object) -> AbstractKernel:
        from kernellib._kernels._compose import Product, Scaled

        if isinstance(other, AbstractKernel):
            return Product(*_flatten(self, Product), *_flatten(other, Product))
        if isinstance(other, (int, float, jax.Array)):
            return Scaled(self, other)
        return NotImplemented

    def __rmul__(self, other: object) -> AbstractKernel:
        from kernellib._kernels._compose import Scaled

        if isinstance(other, (int, float, jax.Array)):
            return Scaled(self, other)
        return NotImplemented


def _flatten(kernel: AbstractKernel, cls: type) -> tuple[AbstractKernel, ...]:
    """Children of ``kernel`` if it is a ``cls`` composite, else ``(kernel,)``."""
    if isinstance(kernel, cls):
        return kernel.kernels  # ty: ignore[unresolved-attribute]
    return (kernel,)


class AbstractPointwiseKernel(AbstractKernel):
    """A kernel defined by ``k(x, x')``.

    Subclasses implement ``pairwise``. The Gram and diagonal default to
    ``vmap`` over it; subclasses override them with closed forms where one
    exists.
    """

    @abstractmethod
    def pairwise(
        self, x: Float[Array, " D"], y: Float[Array, " D"]
    ) -> Float[Array, ""]:
        """Scalar ``k(x, y)``."""
        raise NotImplementedError

    def __call__(
        self, X1: Float[Array, "N1 D"], X2: Float[Array, "N2 D"]
    ) -> Float[Array, "N1 N2"]:
        return jax.vmap(lambda x: jax.vmap(lambda y: self.pairwise(x, y))(X2))(X1)

    def diag(self, X: Float[Array, "N D"]) -> Float[Array, " N"]:
        return jax.vmap(lambda x: self.pairwise(x, x))(X)

    @property
    def is_pointwise(self) -> bool:
        return True


class AbstractStationaryKernel(AbstractPointwiseKernel):
    """``k(x, x') = variance * shape(||(x - x') / lengthscale||²)``.

    ``lengthscale`` is a scalar (isotropic) or ``(D,)`` array (ARD).
    Subclasses implement ``shape`` of the scaled squared distance.
    """

    lengthscale: eqx.AbstractVar[Float[Array, ""] | Float[Array, " D"]]
    variance: eqx.AbstractVar[Float[Array, ""]]

    @abstractmethod
    def shape(self, r2: Float[Array, ...]) -> Float[Array, ...]:
        """Unit-variance profile as a function of scaled squared distance."""
        raise NotImplementedError

    def pairwise(
        self, x: Float[Array, " D"], y: Float[Array, " D"]
    ) -> Float[Array, ""]:
        if jnp.ndim(self.lengthscale) == 1 and x.shape[-1] != jnp.size(
            self.lengthscale
        ):
            raise ValueError(
                f"ARD lengthscale of size {jnp.size(self.lengthscale)} requires "
                f"inputs with that many features; got {x.shape[-1]}."
            )
        r2 = jnp.sum(((x - y) / self.lengthscale) ** 2)
        return self.variance * self.shape(r2)

    def __call__(
        self, X1: Float[Array, "N1 D"], X2: Float[Array, "N2 D"]
    ) -> Float[Array, "N1 N2"]:
        return self.variance * self.shape(_pairwise_sq_dist(X1, X2, self.lengthscale))

    def diag(self, X: Float[Array, "N D"]) -> Float[Array, " N"]:
        return self.variance * jnp.ones(X.shape[0], dtype=X.dtype)
