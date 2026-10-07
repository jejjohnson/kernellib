"""Kernels built from functions of the inputs: feature kernels and modulation."""

from __future__ import annotations

from collections.abc import Callable

import einx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from kernellib._einx import einsum, reduce
from kernellib._kernels._base import AbstractKernel, AbstractPointwiseKernel, GramParts


__all__ = ["FeatureKernel", "Modulated"]


class FeatureKernel(AbstractPointwiseKernel):
    r"""Inner product of features, $k(x, x') = \langle \phi(x), \phi(x') \rangle$.

    ``features`` maps one input point ``(D,)`` to ``(R,)`` (or a scalar, the
    rank-1 $\phi(x)\phi(x')$), and may be a plain function or an equinox
    module with trainable parameters, e.g. the last layer of a deep kernel.
    A fitted `AbstractFeatureMap` also works: it maps a batch ``(N, D)`` to
    ``(N, R)``, so ``FeatureKernel(rff)`` is the random-feature approximation
    of the kernel ``rff`` was fitted to, and ``FeatureKernel(nystrom)`` its
    Nyström approximation, as kernels.

    The Gram $\Phi_1 \Phi_2^\top$ has rank at most ``R``; `to_operator`
    keeps it as a `gaussx.LowRankUpdate`.

    Attributes:
        features: Pointwise feature function ``(D,) -> (R,)`` or ``()``, or
            a fitted `AbstractFeatureMap`.

    Examples:
        >>> import jax.numpy as jnp
        >>> import kernellib as kl
        >>> k = kl.FeatureKernel(lambda x: jnp.array([1.0, x[0], x[0] ** 2]))
        >>> X = jnp.array([[0.0], [1.0], [2.0]])
        >>> float(k(X, X)[1, 2])  # 1 + 1*2 + 1*4
        7.0
    """

    features: Callable

    def _batch(self, X: Float[Array, "N D"]) -> Float[Array, "N R"]:
        from kernellib._spectral._base import AbstractFeatureMap

        if isinstance(self.features, AbstractFeatureMap):
            return self.features(X)
        return jax.vmap(lambda x: jnp.atleast_1d(self.features(x)))(X)

    def pairwise(
        self, x: Float[Array, " D"], y: Float[Array, " D"]
    ) -> Float[Array, ""]:
        phi = self._batch(jnp.stack([x, y]))
        return jnp.dot(phi[0], phi[1])

    def __call__(
        self, X1: Float[Array, "N1 D"], X2: Float[Array, "N2 D"]
    ) -> Float[Array, "N1 N2"]:
        return einsum(self._batch(X1), self._batch(X2), "n r, m r -> n m")

    def diag(self, X: Float[Array, "N D"]) -> Float[Array, " N"]:
        return reduce(self._batch(X) ** 2, "n r -> n", "sum")

    def _gram_structure(self, X: Float[Array, "N D"]) -> GramParts:
        Phi = self._batch(X)
        return GramParts(factors=Phi, weights=jnp.ones(Phi.shape[1], dtype=Phi.dtype))


class Modulated(AbstractKernel):
    r"""A kernel with an input-dependent amplitude,
    $k(x, x') = a(x)\, k_0(x, x')\, a(x')$.

    Positive semidefinite whenever ``kernel`` is (a congruence
    $A K_0 A$ with $A = \mathrm{diag}(a(X))$); the variance at $x$ becomes
    $a(x)^2 k_0(x, x)$, so a smooth ``amplitude`` gives a smoothly varying
    signal strength. Not stationary unless ``amplitude`` is constant.

    Attributes:
        kernel: The modulated kernel.
        amplitude: ``(D,) -> ()``, a function or an equinox module.

    Examples:
        >>> import jax.numpy as jnp
        >>> import kernellib as kl
        >>> k = kl.Modulated(kl.RBF(), amplitude=lambda x: 1.0 + x[0] ** 2)
        >>> k.diag(jnp.array([[0.0], [1.0]])).tolist()
        [1.0, 4.0]
    """

    kernel: AbstractKernel
    amplitude: Callable

    def _amplitudes(self, X: Float[Array, "N D"]) -> Float[Array, " N"]:
        return jax.vmap(lambda x: jnp.squeeze(self.amplitude(x)))(X)

    def __call__(
        self, X1: Float[Array, "N1 D"], X2: Float[Array, "N2 D"]
    ) -> Float[Array, "N1 N2"]:
        a1, a2 = self._amplitudes(X1), self._amplitudes(X2)
        scaled = einx.multiply("n1, n1 n2 -> n1 n2", a1, self.kernel(X1, X2))
        return einx.multiply("n1 n2, n2 -> n1 n2", scaled, a2)

    def diag(self, X: Float[Array, "N D"]) -> Float[Array, " N"]:
        return self._amplitudes(X) ** 2 * self.kernel.diag(X)

    def pairwise(
        self, x: Float[Array, " D"], y: Float[Array, " D"]
    ) -> Float[Array, ""]:
        a = jnp.squeeze(self.amplitude(x))
        b = jnp.squeeze(self.amplitude(y))
        return a * self.kernel.pairwise(x, y) * b

    def _gram_structure(self, X: Float[Array, "N D"]) -> GramParts | None:
        parts = self.kernel._gram_structure(X)
        if parts is None:
            return None
        a = self._amplitudes(X)
        return GramParts(
            None if parts.diagonal is None else a**2 * parts.diagonal,
            None
            if parts.factors is None
            else einx.multiply("n, n r -> n r", a, parts.factors),
            parts.weights,
        )

    @property
    def is_pointwise(self) -> bool:
        return self.kernel.is_pointwise
