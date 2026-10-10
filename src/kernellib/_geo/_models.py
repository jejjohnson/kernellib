"""Classical geostatistics covariance models (GEO5).

Every model is ``k(x, x') = variance * psi(r)`` with
``r = ||(x - x') / lengthscale||``, an `AbstractStationaryKernel` whose
``shape(r2)`` evaluates ``psi`` on the scaled squared distance. For the
compactly supported models (spherical family, Wendland) the lengthscale is the
**range**: ``psi`` is exactly 0 for ``r >= 1``, so their Grams are sparse.

None of these models registers a spectral density, so `spectral_density` and
random Fourier features raise ``NotImplementedError``. (Closed forms exist for
some, such as Wendland and Askey; they are a possible follow-up.)

The profiles live in `kernellib.functional` (``spherical_kernel`` and friends),
which the Gram matrices agree with.
"""

from __future__ import annotations

from typing import Literal

import equinox as eqx
import jax.numpy as jnp
from jax.core import Tracer
from jaxtyping import Array, Float

from kernellib._kernels._base import AbstractStationaryKernel
from kernellib.functional import _geo_models as _f


__all__ = [
    "Cubic",
    "GeneralizedCauchy",
    "HoleEffect",
    "Pentaspherical",
    "Spherical",
    "Stable",
    "Wendland",
]


class _AbstractLowDimKernel(AbstractStationaryKernel):
    """A stationary model positive definite only for ``D <= 3``.

    The input dimension is static, so ``pairwise`` and ``__call__`` raise for
    wider inputs, as the ARD-size check in `AbstractStationaryKernel` does.
    """

    def pairwise(
        self, x: Float[Array, " D"], y: Float[Array, " D"]
    ) -> Float[Array, ""]:
        _f._check_dim(x.shape[-1], type(self).__name__)
        return super().pairwise(x, y)

    def __call__(
        self, X1: Float[Array, "N1 D"], X2: Float[Array, "N2 D"]
    ) -> Float[Array, "N1 N2"]:
        _f._check_dim(X1.shape[-1], type(self).__name__)
        return super().__call__(X1, X2)


class Spherical(_AbstractLowDimKernel):
    r"""Spherical covariance model, compactly supported.

    $$
    \psi(r) = 1 - \tfrac32 r + \tfrac12 r^3 \quad (r < 1), \qquad 0 \text{ else}
    $$

    Positive definite for $D \le 3$; continuous but not differentiable at
    the origin (like the exponential). Reference: Chilès & Delfiner (2012),
    *Geostatistics*, §2.5.

    Attributes:
        lengthscale: Scalar (isotropic) or ``(D,)`` (ARD) range, ``> 0``.
        variance: Scalar sill, ``> 0``.

    Raises:
        ValueError: When evaluated on inputs with ``D > 3``.

    Examples:
        >>> import jax.numpy as jnp
        >>> import kernellib as kl
        >>> k = kl.Spherical(lengthscale=2.0)
        >>> X = jnp.array([[0.0], [1.0], [2.0]])
        >>> [round(float(v), 4) for v in k(X, X)[0]]
        [1.0, 0.3125, 0.0]
    """

    lengthscale: Float[Array, ""] | Float[Array, " D"] = eqx.field(
        default=1.0, converter=jnp.asarray
    )
    variance: Float[Array, ""] = eqx.field(default=1.0, converter=jnp.asarray)

    def shape(self, r2: Float[Array, ...]) -> Float[Array, ...]:
        return _f._spherical_shape(r2)


class Cubic(_AbstractLowDimKernel):
    r"""Cubic covariance model, compactly supported.

    $$
    \psi(r) = 1 - 7r^2 + \tfrac{35}{4} r^3 - \tfrac72 r^5 + \tfrac34 r^7
    \quad (r < 1), \qquad 0 \text{ else}
    $$

    Positive definite for $D \le 3$; twice differentiable at the origin.
    Reference: Chilès & Delfiner (2012), *Geostatistics*, §2.5.

    Attributes:
        lengthscale: Scalar (isotropic) or ``(D,)`` (ARD) range, ``> 0``.
        variance: Scalar sill, ``> 0``.

    Raises:
        ValueError: When evaluated on inputs with ``D > 3``.

    Examples:
        >>> import jax.numpy as jnp
        >>> import kernellib as kl
        >>> k = kl.Cubic()
        >>> x, y = jnp.array([0.0]), jnp.array([0.5])
        >>> round(float(k.pairwise(x, y)), 4)
        0.2402
    """

    lengthscale: Float[Array, ""] | Float[Array, " D"] = eqx.field(
        default=1.0, converter=jnp.asarray
    )
    variance: Float[Array, ""] = eqx.field(default=1.0, converter=jnp.asarray)

    def shape(self, r2: Float[Array, ...]) -> Float[Array, ...]:
        return _f._cubic_shape(r2)


class Pentaspherical(_AbstractLowDimKernel):
    r"""Pentaspherical covariance model, compactly supported.

    $$
    \psi(r) = 1 - \tfrac{15}{8} r + \tfrac54 r^3 - \tfrac38 r^5
    \quad (r < 1), \qquad 0 \text{ else}
    $$

    Positive definite for $D \le 3$; not differentiable at the origin.
    Reference: Chilès & Delfiner (2012), *Geostatistics*, §2.5.

    Attributes:
        lengthscale: Scalar (isotropic) or ``(D,)`` (ARD) range, ``> 0``.
        variance: Scalar sill, ``> 0``.

    Raises:
        ValueError: When evaluated on inputs with ``D > 3``.

    Examples:
        >>> import jax.numpy as jnp
        >>> import kernellib as kl
        >>> k = kl.Pentaspherical()
        >>> x, y = jnp.array([0.0]), jnp.array([1.0])
        >>> float(k.pairwise(x, y))
        0.0
    """

    lengthscale: Float[Array, ""] | Float[Array, " D"] = eqx.field(
        default=1.0, converter=jnp.asarray
    )
    variance: Float[Array, ""] = eqx.field(default=1.0, converter=jnp.asarray)

    def shape(self, r2: Float[Array, ...]) -> Float[Array, ...]:
        return _f._pentaspherical_shape(r2)


class HoleEffect(_AbstractLowDimKernel):
    r"""Hole-effect (cardinal sine, "wave") covariance model.

    $$
    \psi(r) = \frac{\sin r}{r}, \qquad \psi(0) = 1
    $$

    Dips below zero (the "hole"), modelling periodic-like dependence such as
    dunes or ocean swell. Positive definite for $D \le 3$ and infinitely
    differentiable. Reference: Chilès & Delfiner (2012), *Geostatistics*, §2.5.

    Attributes:
        lengthscale: Scalar (isotropic) or ``(D,)`` (ARD) lengthscale, ``> 0``.
        variance: Scalar sill, ``> 0``.

    Raises:
        ValueError: When evaluated on inputs with ``D > 3``.

    Examples:
        >>> import jax.numpy as jnp
        >>> import kernellib as kl
        >>> k = kl.HoleEffect()
        >>> x, y = jnp.array([0.0]), jnp.array([1.5 * jnp.pi])
        >>> round(float(k.pairwise(x, y)), 4)
        -0.2122
    """

    lengthscale: Float[Array, ""] | Float[Array, " D"] = eqx.field(
        default=1.0, converter=jnp.asarray
    )
    variance: Float[Array, ""] = eqx.field(default=1.0, converter=jnp.asarray)

    def shape(self, r2: Float[Array, ...]) -> Float[Array, ...]:
        return _f._hole_effect_shape(r2)


class Stable(AbstractStationaryKernel):
    r"""Stable (powered exponential) covariance model.

    $$
    \psi(r) = \exp(-r^\alpha), \qquad 0 < \alpha \le 2
    $$

    Positive definite in every dimension. ``alpha = 1`` is the exponential
    (`Matern` with ``nu = 0.5``); ``alpha = 2`` is `RBF` with lengthscale
    $\ell/\sqrt2$, the only infinitely differentiable member. ``alpha`` is
    static: it selects a code path, not an optimisation target. Reference:
    Gneiting & Schlather (2004), *SIAM Review* 46.

    Attributes:
        lengthscale: Scalar (isotropic) or ``(D,)`` (ARD) lengthscale, ``> 0``.
        variance: Scalar sill, ``> 0``.
        alpha: Shape exponent in ``(0, 2]`` (default ``1.0``).

    Raises:
        ValueError: If ``alpha`` is outside ``(0, 2]``.

    Examples:
        >>> import jax.numpy as jnp
        >>> import kernellib as kl
        >>> k = kl.Stable(alpha=2.0)
        >>> x, y = jnp.array([0.0]), jnp.array([1.0])
        >>> bool(jnp.allclose(k.pairwise(x, y), jnp.exp(-1.0)))
        True
    """

    lengthscale: Float[Array, ""] | Float[Array, " D"] = eqx.field(
        default=1.0, converter=jnp.asarray
    )
    variance: Float[Array, ""] = eqx.field(default=1.0, converter=jnp.asarray)
    alpha: float = eqx.field(default=1.0, static=True)

    def __check_init__(self) -> None:
        _f._check_alpha(self.alpha, "Stable")

    def shape(self, r2: Float[Array, ...]) -> Float[Array, ...]:
        return _f._stable_shape(r2, self.alpha)


class GeneralizedCauchy(AbstractStationaryKernel):
    r"""Generalized Cauchy covariance model.

    $$
    \psi(r) = (1 + r^\alpha)^{-\beta/\alpha},
    \qquad 0 < \alpha \le 2,\ \beta > 0
    $$

    Positive definite in every dimension. ``alpha`` fixes the fractal
    dimension ($D + 1 - \alpha/2$ for a surface in $\mathbb{R}^D$) and
    ``beta`` the power-law tail, independently. ``alpha = 2`` is
    `RationalQuadratic` with $\alpha_{RQ} = \beta/2$ and
    $\ell_{RQ} = \ell/\sqrt\beta$. ``alpha`` is static; ``beta`` is a
    trainable array. Reference: Gneiting & Schlather (2004), *SIAM Review* 46.

    Attributes:
        lengthscale: Scalar (isotropic) or ``(D,)`` (ARD) lengthscale, ``> 0``.
        variance: Scalar sill, ``> 0``.
        alpha: Shape exponent in ``(0, 2]`` (default ``1.0``).
        beta: Positive tail exponent (default ``1.0``).

    Raises:
        ValueError: If ``alpha`` is outside ``(0, 2]``, or a concrete
            ``beta`` is not positive.

    Examples:
        >>> import jax.numpy as jnp
        >>> import kernellib as kl
        >>> k = kl.GeneralizedCauchy(alpha=1.0, beta=2.0)
        >>> x, y = jnp.array([0.0]), jnp.array([1.0])
        >>> round(float(k.pairwise(x, y)), 4)
        0.25
    """

    lengthscale: Float[Array, ""] | Float[Array, " D"] = eqx.field(
        default=1.0, converter=jnp.asarray
    )
    variance: Float[Array, ""] = eqx.field(default=1.0, converter=jnp.asarray)
    alpha: float = eqx.field(default=1.0, static=True)
    beta: Float[Array, ""] = eqx.field(default=1.0, converter=jnp.asarray)

    def __check_init__(self) -> None:
        _f._check_alpha(self.alpha, "GeneralizedCauchy")
        # Under a transform (jit, vmap) beta may be a tracer; check only values.
        if not isinstance(self.beta, Tracer) and bool(jnp.any(self.beta <= 0.0)):
            raise ValueError(f"GeneralizedCauchy requires beta > 0, got {self.beta}")

    def shape(self, r2: Float[Array, ...]) -> Float[Array, ...]:
        return _f._generalized_cauchy_shape(r2, self.alpha, self.beta)


class Wendland(_AbstractLowDimKernel):
    r"""Wendland covariance model, compactly supported.

    $$
    \psi_{C^2}(r) = (1 - r)_+^4 (4r + 1), \qquad
    \psi_{C^4}(r) = (1 - r)_+^6 (35r^2 + 18r + 3) / 3
    $$

    Wendland's $\phi_{3,1}$ (``order=2``) and $\phi_{3,2}$ (``order=4``),
    normalised to $\psi(0) = 1$: positive definite for $D \le 3$, and $C^2$
    or $C^4$ at the origin. The usual taper for covariance tapering.
    Reference: Wendland (1995), *Adv. Comput. Math.* 4.

    Attributes:
        lengthscale: Scalar (isotropic) or ``(D,)`` (ARD) support radius,
            ``> 0``.
        variance: Scalar sill, ``> 0``.
        order: Smoothness, ``2`` (default) or ``4``; static.

    Raises:
        ValueError: If ``order`` is not 2 or 4, or when evaluated on inputs
            with ``D > 3``.

    Examples:
        >>> import jax.numpy as jnp
        >>> import kernellib as kl
        >>> k = kl.Wendland(order=4)
        >>> x, y = jnp.array([0.0, 0.0]), jnp.array([0.3, 0.4])
        >>> round(float(k.pairwise(x, y)), 4)
        0.1081
    """

    lengthscale: Float[Array, ""] | Float[Array, " D"] = eqx.field(
        default=1.0, converter=jnp.asarray
    )
    variance: Float[Array, ""] = eqx.field(default=1.0, converter=jnp.asarray)
    order: Literal[2, 4] = eqx.field(default=2, static=True)

    def __check_init__(self) -> None:
        if self.order not in (2, 4):
            raise ValueError(f"Wendland supports order in {{2, 4}}, got {self.order!r}")

    def shape(self, r2: Float[Array, ...]) -> Float[Array, ...]:
        return _f._wendland_shape(r2, self.order)
