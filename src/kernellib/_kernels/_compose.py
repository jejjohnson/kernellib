"""Kernel-level composition: sums, products, scaling, input selection and
input warping.

Composites are pointwise exactly when all their children are, so a sum of
pointwise kernels still gets the matrix-free operator path. Their Gram matrix
is built from each child's own Gram path, so closed forms are kept.

`Scaled` and `Sum` of stationary kernels are stationary, and their spectra
are the scaled and summed parts: they carry `spectral_density`,
`sample_frequencies` and `spectral_variance`, so the random-feature and
Laplace maps accept them. `Product` has a spectrum too (the convolution of
the parts), but not a closed-form one, so it does not.
"""

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.typing import DTypeLike
from jaxtyping import Array, Float, PRNGKeyArray

from kernellib._kernels._base import AbstractKernel, AbstractStationaryKernel, GramParts


__all__ = [
    "ActiveDims",
    "Periodised",
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


SpectralComponents = tuple[tuple[Float[Array, ""], AbstractStationaryKernel], ...]


def _spectral_components(kernel: AbstractKernel) -> SpectralComponents | None:
    """``((c_j, k_j), ...)`` with ``kernel = sum_j c_j k_j``, or ``None``.

    Each ``k_j`` is an `AbstractStationaryKernel` and ``c_j`` the product of
    the `Scaled` factors above it. ``None`` when any part is not a stationary
    kernel, a `Scaled` one or a `Sum` of them.
    """
    if isinstance(kernel, AbstractStationaryKernel):
        return ((jnp.asarray(1.0), kernel),)
    if isinstance(kernel, Scaled):
        inner = _spectral_components(kernel.kernel)
        if inner is None:
            return None
        return tuple((kernel.scale * c, k) for c, k in inner)
    if isinstance(kernel, Sum):
        parts = [_spectral_components(k) for k in kernel.kernels]
        if any(p is None for p in parts):
            return None
        return tuple(c for p in parts for c in p)  # ty: ignore[not-iterable]
    return None


def _require_components(kernel: AbstractKernel) -> SpectralComponents:
    components = _spectral_components(kernel)
    if components is None:
        raise NotImplementedError(
            f"This {type(kernel).__name__} has no spectral density: every part "
            "must be a stationary kernel, or a Scaled or Sum of them (a Product "
            "or a non-stationary part has none in closed form)."
        )
    return components


def _components_density(
    components: SpectralComponents, omega: Float[Array, "*batch D"]
) -> Float[Array, "*batch"]:
    """``sum_j c_j S_j(omega)``."""
    total = components[0][0] * components[0][1].spectral_density(omega)
    for c, k in components[1:]:
        total = total + c * k.spectral_density(omega)
    return total


class _SpectralComposite(AbstractKernel):
    """Spectral methods of a `Scaled` or `Sum` of stationary kernels."""

    def spectral_density(
        self, omega: Float[Array, "*batch D"]
    ) -> Float[Array, "*batch"]:
        r"""$S(\omega) = \sum_j c_j S_j(\omega)$ over the stationary parts.

        Same convention as `AbstractStationaryKernel.spectral_density`.

        Raises:
            NotImplementedError: If a part has no spectral density.
        """
        return _components_density(_require_components(self), omega)

    @property
    def spectral_variance(self) -> Float[Array, ""]:
        r"""$k(0) = \sum_j c_j \sigma_j^2$.

        Raises:
            NotImplementedError: If a part has no spectral density.
        """
        components = _require_components(self)
        total = components[0][0] * components[0][1].variance
        for c, k in components[1:]:
            total = total + c * k.variance
        return total

    def sample_frequencies(
        self,
        key: PRNGKeyArray,
        n: int,
        d: int,
        dtype: DTypeLike | None = None,
    ) -> Float[Array, "n d"]:
        r"""Draw ``n`` frequencies from the normalised density.

        The normalised density is the mixture $\sum_j w_j p_j$ with
        $w_j = c_j \sigma_j^2 / \sum_l c_l \sigma_l^2$: each draw picks a
        part from $\mathrm{Categorical}(w)$, then a frequency from it. The
        scales $c_j$ must be positive (as they must for a valid kernel).

        Raises:
            NotImplementedError: If a part has no spectral sampler.
            ValueError: If an ARD lengthscale does not match ``d``.
        """
        components = _require_components(self)
        if len(components) == 1:
            return components[0][1].sample_frequencies(key, n, d, dtype)
        key_part, *keys = jax.random.split(key, len(components) + 1)
        weights = jnp.stack([c * k.variance for c, k in components])
        part = jax.random.categorical(key_part, jnp.log(weights), shape=(n,))
        draws = jnp.stack(
            [
                k.sample_frequencies(key_j, n, d, dtype)
                for (_, k), key_j in zip(components, keys, strict=True)
            ]
        )
        return draws[part, jnp.arange(n)]


class Sum(_SpectralComposite):
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

    def _gram_structure(self, X: Float[Array, "N D"]) -> GramParts | None:
        parts = [k._gram_structure(X) for k in self.kernels]
        if any(p is None for p in parts):
            return None
        return GramParts.merge(parts)  # ty: ignore[invalid-argument-type]

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


class Scaled(_SpectralComposite):
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

    def _gram_structure(self, X: Float[Array, "N D"]) -> GramParts | None:
        parts = self.kernel._gram_structure(X)
        return None if parts is None else parts.scaled(self.scale)

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

    def _gram_structure(self, X: Float[Array, "N D"]) -> GramParts | None:
        return self.kernel._gram_structure(self._select(X))

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

    def _gram_structure(self, X: Float[Array, "N D"]) -> GramParts | None:
        return self.kernel._gram_structure(jax.vmap(self.warp)(X))

    def pairwise(
        self, x: Float[Array, " D"], y: Float[Array, " D"]
    ) -> Float[Array, ""]:
        return self.kernel.pairwise(self.warp(x), self.warp(y))

    @property
    def is_pointwise(self) -> bool:
        return self.kernel.is_pointwise


class Periodised(AbstractKernel):
    r"""Make any kernel periodic: apply it to inputs wrapped onto circles.

    Each input coordinate is embedded as
    $z_d \mapsto (\sin(2\pi z_d / p_d), \cos(2\pi z_d / p_d))$, and
    ``kernel`` is evaluated on the ``2D``-dimensional embedding (MacKay,
    1998). A kernel of warped inputs is positive semidefinite whenever the
    base kernel is, in any dimension, unlike `Periodic`, which applies the
    sine to the Euclidean distance and is not PSD for ``D > 1``.

    With a stationary base the result is stationary and periodic in each
    coordinate: $\lVert\phi(x) - \phi(y)\rVert^2
    = \sum_d 4 \sin^2(\pi (x_d - y_d) / p_d)$, so
    ``Periodised(RBF(lengthscale=l), p)`` is
    $\exp(-2 \sum_d \sin^2(\pi (x_d - y_d) / p_d) / l^2)$: exactly
    `Periodic` in 1-D, and the product of per-dimension periodic kernels in
    higher dimensions. Any base works (`Matern`, `RationalQuadratic`,
    ``Linear``, ...).

    The base kernel acts on the embedding, so an ARD lengthscale on it needs
    ``2 * D`` entries (sine and cosine of each coordinate).

    Attributes:
        kernel: The base kernel, applied to the ``(2D,)`` embedding.
        period: Scalar, or ``(D,)`` for a period per input dimension.

    Examples:
        >>> import jax.numpy as jnp
        >>> import kernellib as kl
        >>> k = kl.Periodised(kl.Matern(nu=1.5, lengthscale=0.8), period=2.0)
        >>> x = jnp.array([0.3, -0.4])
        >>> bool(jnp.allclose(k.pairwise(x, x + 2.0), k.pairwise(x, x)))
        True
    """

    kernel: AbstractKernel
    period: Float[Array, ""] | Float[Array, " D"] = eqx.field(converter=jnp.asarray)

    def _embed(self, x: Float[Array, " D"]) -> Float[Array, " 2D"]:
        z = 2.0 * jnp.pi * x / self.period
        return jnp.concatenate([jnp.sin(z), jnp.cos(z)], axis=-1)

    def __call__(
        self, X1: Float[Array, "N1 D"], X2: Float[Array, "N2 D"]
    ) -> Float[Array, "N1 N2"]:
        return self.kernel(self._embed(X1), self._embed(X2))

    def diag(self, X: Float[Array, "N D"]) -> Float[Array, " N"]:
        return self.kernel.diag(self._embed(X))

    def pairwise(
        self, x: Float[Array, " D"], y: Float[Array, " D"]
    ) -> Float[Array, ""]:
        return self.kernel.pairwise(self._embed(x), self._embed(y))

    def _gram_structure(self, X: Float[Array, "N D"]) -> GramParts | None:
        return self.kernel._gram_structure(self._embed(X))

    @property
    def is_pointwise(self) -> bool:
        return self.kernel.is_pointwise
