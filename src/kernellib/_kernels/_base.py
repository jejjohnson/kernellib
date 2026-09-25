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
from jax.typing import DTypeLike
from jaxtyping import Array, Float, PRNGKeyArray

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
    Subclasses implement ``shape`` of the scaled squared distance. The
    spectral side (`spectral_density`, `sample_frequencies`) is optional:
    a subclass enables it by overriding `unit_spectral_density` and
    `sample_unit_frequencies` for the unit-variance, unit-lengthscale
    kernel, and the base class adds the hyperparameters.
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

    # -- spectral side ------------------------------------------------------

    def unit_spectral_density(
        self, omega_sq: Float[Array, ...], d: int
    ) -> Float[Array, ...]:
        """Spectral density of the unit-variance, unit-lengthscale kernel.

        A function of the squared frequency magnitude ``omega_sq`` in ``d``
        dimensions. Subclasses with a closed form override it;
        `spectral_density` adds the variance and lengthscale.

        Raises:
            NotImplementedError: If the kernel has no registered density.
        """
        raise NotImplementedError(
            f"{type(self).__name__} has no closed-form spectral density; "
            "override unit_spectral_density(omega_sq, d) to provide one."
        )

    def sample_unit_frequencies(
        self,
        key: PRNGKeyArray,
        shape: tuple[int, ...],
        dtype: DTypeLike | None = None,
    ) -> Float[Array, ...]:
        """Draw frequencies of the unit-variance, unit-lengthscale kernel.

        The last axis of ``shape`` is the input dimension; every leading
        index is an independent frequency vector. `sample_frequencies`
        divides by the lengthscale.

        Raises:
            NotImplementedError: If the kernel has no registered sampler.
        """
        raise NotImplementedError(
            f"{type(self).__name__} has no spectral sampler; override "
            "sample_unit_frequencies(key, shape, dtype) to provide one."
        )

    def spectral_density(
        self, omega: Float[Array, "*batch D"]
    ) -> Float[Array, "*batch"]:
        r"""Spectral density $S(\omega)$ at frequency vectors ``omega``.

        The convention is the one Bochner's theorem gives with the
        $(2\pi)^{-D}$ on the inverse transform,

        $$
        k(\tau) = \frac{1}{(2\pi)^D} \int S(\omega)\, e^{i \omega^\top \tau}
        \, d\omega,
        $$

        so $S$ integrates to $(2\pi)^D \sigma^2$. With a diagonal lengthscale
        $\Lambda$, $S(\omega) = \sigma^2 \det\Lambda\; s(\|\Lambda\omega\|^2)$,
        where $s$ is `unit_spectral_density`.

        Args:
            omega: Frequencies, shape ``(..., D)``.

        Returns:
            Density values, shape ``(...)``.

        Raises:
            ValueError: If an ARD lengthscale does not match ``D``.
            NotImplementedError: If the kernel has no registered density.

        Examples:
            >>> import jax.numpy as jnp
            >>> import kernellib as kl
            >>> k = kl.RBF(lengthscale=1.0, variance=1.0)
            >>> s0 = k.spectral_density(jnp.zeros((1, 1)))  # sqrt(2 pi)
            >>> bool(jnp.allclose(s0, jnp.sqrt(2 * jnp.pi)))
            True
        """
        omega = jnp.asarray(omega)
        d = omega.shape[-1]
        ell = self._lengthscale_vector(d)
        omega_sq = jnp.sum((omega * ell) ** 2, axis=-1)
        return self.variance * jnp.prod(ell) * self.unit_spectral_density(omega_sq, d)

    def sample_frequencies(
        self,
        key: PRNGKeyArray,
        n: int,
        d: int,
        dtype: DTypeLike | None = None,
    ) -> Float[Array, "n d"]:
        r"""Draw ``n`` frequencies from the normalised spectral density.

        The draws have density $S(\omega) / ((2\pi)^D \sigma^2)$, so
        $\mathbb{E}[\cos(\omega^\top \tau)] = k(\tau) / \sigma^2$: the random
        Fourier feature identity.

        Args:
            key: PRNG key.
            n: Number of frequency vectors.
            d: Input dimension.
            dtype: Floating dtype; JAX's default float if ``None``.

        Returns:
            Frequencies, shape ``(n, d)``.

        Raises:
            ValueError: If an ARD lengthscale does not match ``d``.
            NotImplementedError: If the kernel has no registered sampler.

        Examples:
            >>> import jax
            >>> import kernellib as kl
            >>> k = kl.Matern(nu=1.5, lengthscale=0.5)
            >>> k.sample_frequencies(jax.random.key(0), 8, 3).shape
            (8, 3)
        """
        ell = self._lengthscale_vector(d)
        return self.sample_unit_frequencies(key, (n, d), dtype) / ell

    def _lengthscale_vector(self, d: int) -> Float[Array, " D"]:
        """The lengthscale broadcast to ``(d,)``, checking an ARD size."""
        if jnp.ndim(self.lengthscale) == 1 and jnp.size(self.lengthscale) != d:
            raise ValueError(
                f"ARD lengthscale of size {jnp.size(self.lengthscale)} requires "
                f"{d}-dimensional frequencies."
            )
        return jnp.broadcast_to(self.lengthscale, (d,))
