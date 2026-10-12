r"""Latitude-dependent and axially symmetric kernels on the sphere.

Global fields are rarely isotropic: variance and correlation change with
latitude. Two constructions give valid covariances that do, and `JunStein`
is also invariant to rotations about the polar axis (*axially symmetric*):

- `LatitudeScaled`, an amplitude modulation $s(\phi_x)\,s(\phi_y)\,k_0(x, y)$;
- `JunStein`, the covariance of a latitude-dependent first-order differential
  operator applied to an isotropic process (Jun & Stein 2008).

`LegendreLatitudeFn` is a learnable function of latitude, to pass as the
scale $s$ or as the operator coefficients $a, b, c$.
"""

from __future__ import annotations

from collections.abc import Callable

import einx
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from kernellib._einx import einsum, rearrange
from kernellib._geo._great_circle import AbstractGreatCircleKernel
from kernellib._geo._sphere_series import SphereMatern
from kernellib._kernels._base import AbstractKernel, AbstractPointwiseKernel
from kernellib._kernels._derivative import (
    _check_differentiable,
    _value_gradient_block,
)
from kernellib._kernels._stationary import Matern
from kernellib.functional._geo import legendre_series


__all__ = ["JunStein", "LatitudeScaled", "LegendreLatitudeFn"]


def _latitude(X: Float[Array, "N D"], degrees: bool) -> Float[Array, " N"]:
    """Latitude column of ``(lon, lat, ...)`` inputs, in radians."""
    lat = X[:, 1]
    return jnp.deg2rad(lat) if degrees else lat


class LegendreLatitudeFn(eqx.Module):
    r"""A function of latitude, $f(\phi) = \sum_{j=0}^{J-1} w_j\,P_j(\sin\phi)$.

    Meant for the scale $s$ of `LatitudeScaled` and the coefficients
    $a, b, c$ of `JunStein`. The Legendre polynomials $P_j$ of $\sin\phi$ are
    smooth on the whole sphere (even or odd about the equator by the parity
    of $j$), so $f$ is smooth at the poles and the weights are plain
    differentiable parameters.

    With ``cos_factor=True`` the series is multiplied by $\cos\phi$,
    $f(\phi) = \cos\phi \sum_j w_j P_j(\sin\phi)$, which vanishes at the poles
    (the form $b(\phi) = \tilde b(\phi)\cos\phi$ of Jun & Stein 2008).

    Pseudocode::

        t = sin(phi)
        f = sum_j weights[j] * P_j(t)      # Legendre recurrence
        if cos_factor: f = cos(phi) * f

    **Numerics.** Evaluated by the Legendre three-term recurrence
    (`kernellib.functional.legendre_series`), stable for any $J$; $\phi$ is in
    **radians** and may be a scalar or an array. The output follows the
    dtype of ``weights`` and ``phi``.

    Attributes:
        weights: ``(J,)`` coefficients $w_j$, by degree.
        cos_factor: Whether to multiply by $\cos\phi$ (static).

    Examples:
        >>> import jax.numpy as jnp
        >>> import kernellib as kl
        >>> f = kl.LegendreLatitudeFn(jnp.array([1.0, 0.0, 0.5]))
        >>> # 1 + 0.5 P_2(sin phi); P_2(1) = 1 at the north pole
        >>> round(float(f(jnp.pi / 2)), 6), round(float(f(0.0)), 6)
        (1.5, 0.75)
        >>> g = kl.LegendreLatitudeFn(jnp.array([1.0]), cos_factor=True)
        >>> round(float(g(jnp.pi / 2)), 6), round(float(g(0.0)), 6)
        (0.0, 1.0)
    """

    weights: Float[Array, " J"] = eqx.field(converter=jnp.asarray)
    cos_factor: bool = eqx.field(default=False, static=True)

    def __check_init__(self) -> None:
        if jnp.ndim(self.weights) != 1 or jnp.size(self.weights) == 0:
            raise ValueError(
                f"weights must be a non-empty 1-D array; got shape "
                f"{jnp.shape(self.weights)}."
            )

    def __call__(self, phi: Float[Array, ...]) -> Float[Array, ...]:
        phi = jnp.asarray(phi)
        value = legendre_series(jnp.sin(phi), self.weights)
        return jnp.cos(phi) * value if self.cos_factor else value


class LatitudeScaled(AbstractKernel):
    r"""Amplitude modulation by latitude, $k(x, y) = s(\phi_x)\,s(\phi_y)\,k_0(x, y)$.

    Scales an isotropic (or any) kernel ``kernel`` on ``(lon, lat)`` inputs
    so the standard deviation varies with latitude $\phi$ as
    $\sqrt{k(x, x)} = |s(\phi_x)|\sqrt{k_0(x, x)}$, for example to give
    polar SST a smaller variance than equatorial SST. The correlation
    $k/\sqrt{k(x,x)\,k(y,y)}$ is unchanged. With $s \equiv 1$ it is
    ``kernel``.

    **Positive definite** for every real $s$ and every positive-definite
    ``kernel``: for any $v$, $v^\top D K_0 D v = (Dv)^\top K_0 (Dv) \ge 0$
    with $D = \operatorname{diag}(s(\phi_i))$ (a congruence of a PSD matrix;
    Stein 2005, Porcu, Alegria & Furrer 2018). Use a ``kernel`` valid on the
    sphere (`Chordal`, `SphereMatern`, ...). The product is not stationary
    in latitude but keeps the symmetries of ``kernel`` in longitude: with an
    isotropic ``kernel`` it is axially symmetric.

    Pseudocode::

        phi = lat in radians  # (N,)
        s = vmap(scale_fn)(phi)  # (N,)
        K = kernel(X1, X2)
        return einx.multiply("n, n m, m", s1, K, s2)  # D1 K D2

    **Numerics.** ``scale_fn`` is called on a scalar latitude in
    **radians** (also when ``degrees=True``, which only says that the inputs
    are in degrees), and is vmapped, so it must be JAX-traceable. Latitude is
    column 1 of the inputs; extra columns are passed to ``kernel`` untouched.
    The Gram is formed as ``D1 K D2`` in $O(N_1 N_2)$ extra work, in the
    result dtype of ``K`` and ``s``. Nothing is singular at the poles.

    Attributes:
        kernel: The base kernel on ``(lon, lat)`` inputs.
        scale_fn: $s(\phi)$ for a scalar latitude in radians, a callable or
            an `equinox.Module` such as `LegendreLatitudeFn`.
        degrees: Whether inputs are in degrees (else radians), static.

    References:
        Stein (2005), *J. Amer. Statist. Assoc.* 100, 310-321.
        Porcu, Alegria & Furrer (2018), *Int. Stat. Rev.* 86, 344-377.

    Examples:
        >>> import jax.numpy as jnp
        >>> import kernellib as kl
        >>> base = kl.SphereMatern(lengthscale=0.5, nu=1.5, max_degree=64)
        >>> s = lambda phi: 1.0 + 0.5 * jnp.cos(phi)  # larger near the equator
        >>> k = kl.LatitudeScaled(base, s)
        >>> X = jnp.array([[0.0, 0.0], [0.0, 90.0]])
        >>> [round(float(v), 4) for v in k.diag(X)]  # s(0)^2, s(pi/2)^2
        [2.25, 1.0]
        >>> bool(jnp.allclose(k(X, X), k.gram(X)))
        True
    """

    kernel: AbstractKernel
    scale_fn: Callable[[Float[Array, ""]], Float[Array, ""]]
    degrees: bool = eqx.field(default=True, static=True)

    def _scale(self, X: Float[Array, "N D"]) -> Float[Array, " N"]:
        return jax.vmap(self.scale_fn)(_latitude(X, self.degrees))

    def __call__(
        self, X1: Float[Array, "N1 D"], X2: Float[Array, "N2 D"]
    ) -> Float[Array, "N1 N2"]:
        return einx.multiply(
            "n, n m, m -> n m",
            self._scale(X1),
            self.kernel(X1, X2),
            self._scale(X2),
        )

    def diag(self, X: Float[Array, "N D"]) -> Float[Array, " N"]:
        return self._scale(X) ** 2 * self.kernel.diag(X)

    def pairwise(
        self, x: Float[Array, " D"], y: Float[Array, " D"]
    ) -> Float[Array, ""]:
        sx = self._scale(rearrange(x, "d -> 1 d"))[0]
        sy = self._scale(rearrange(y, "d -> 1 d"))[0]
        return sx * sy * self.kernel.pairwise(x, y)

    @property
    def is_pointwise(self) -> bool:
        return self.kernel.is_pointwise


def _check_smooth(kernel: AbstractKernel) -> None:
    """Reject base kernels without two mean-square derivatives."""
    _check_differentiable(kernel, "JunStein")
    leaves = jax.tree.leaves(
        kernel,
        is_leaf=lambda n: isinstance(
            n, Matern | SphereMatern | AbstractGreatCircleKernel
        ),
    )
    for leaf in leaves:
        if isinstance(leaf, Matern | SphereMatern) and not leaf.nu > 1.0:
            raise ValueError(
                f"JunStein needs a base kernel whose sample paths are mean-square "
                f"differentiable, but {type(leaf).__name__}(nu={leaf.nu}) is not "
                "(it needs nu > 1; use nu >= 1.5). Use Chordal(Matern(nu >= 1.5)), "
                "Chordal(RBF) or SphereMatern(nu > 1)."
            )
        if isinstance(leaf, AbstractGreatCircleKernel):
            raise ValueError(
                "JunStein needs a twice-differentiable base kernel; "
                f"{type(leaf).__name__} is not. Use Chordal(Matern(nu >= 1.5)), "
                "Chordal(RBF) or SphereMatern(nu > 1)."
            )


class JunStein(AbstractPointwiseKernel):
    r"""Axially symmetric kernel of Jun & Stein (2008), $L_x L_y k_0(x, y)$.

    Let $Z_0 \sim \mathcal{GP}(0, k_0)$ be isotropic on $S^2$ and apply a
    latitude-dependent first-order operator in $(\lambda, \phi)$ =
    (longitude, latitude), in radians,

    $$
    Z(x) = a(\phi)\,\partial_\phi Z_0 + b(\phi)\,\partial_\lambda Z_0
           + c(\phi)\,Z_0 .
    $$

    Then

    $$
    \operatorname{Cov}(Z(x), Z(y)) = \sum_{p, q \in \{\phi, \lambda, 1\}}
    \alpha_p(\phi_x)\,\alpha_q(\phi_y)\,\partial_p^{(x)}\partial_q^{(y)}
    k_0(x, y),
    \qquad (\alpha_\phi, \alpha_\lambda, \alpha_1) = (a, b, c),
    $$

    with $\partial_1 = \mathrm{id}$. Since $k_0$ depends on $(x, y)$ only
    through $\phi_x$, $\phi_y$ and $\lambda_x - \lambda_y$, so does $k$: it is
    **axially symmetric** (invariant to a common shift of all longitudes) but
    its variance and correlation length vary with latitude through
    $a, b, c$. With $a = b = 0$, $c \equiv 1$ it is ``kernel``.

    **Positive definite** (positive semi-definite, and positive definite for
    distinct points whenever $L$ is non-degenerate): $Z$ is a linear
    operator applied to a mean-square differentiable Gaussian process, hence
    a Gaussian process, and the covariance of any GP is PSD. Explicitly,
    for weights $v_i$ and $\xi = \sum_i v_i Z(x_i)$,
    $\sum_{ij} v_i v_j k(x_i, x_j) = \operatorname{Var}(\xi) \ge 0$
    (Jun & Stein 2008, Sec. 2; Stein 2005). This needs $k_0$ to be twice
    differentiable at the origin: `Chordal(Matern(nu >= 1.5))`,
    `Chordal(RBF)` or `SphereMatern(nu > 1)` (the base is checked, and
    exponential / $\nu \le 1$ Matérn and great-circle kernels are rejected
    with a `ValueError`).

    Pseudocode::

        for one pair (x, y):                  # lon/lat of the user's units
            v   = k0(x, y)
            gx  = d k0 / d x      (2,)        # jax.grad, argnum 0
            gy  = d k0 / d y      (2,)        # jax.grad, argnum 1
            H   = d2 k0 / dx dy   (2, 2)      # jax.jacfwd of gx w.r.t. y
            B   = [[v, gy], [gx, H]]          # (3, 3) in order (1, lon, lat)
            ax  = [c(phi_x), b(phi_x), a(phi_x)] ; ay likewise at y
            return ax @ B @ ay

    **Numerics.** The derivatives are exact autodiff of ``kernel.pairwise``
    (no finite differences), so the Gram is the vmap of ``pairwise`` at
    $O(N_1 N_2)$ small Hessians: use moderate $N$ or a low-rank / chunked
    path for large ones. Derivatives are taken in the units of the inputs
    and converted to radians (a factor $180/\pi$ when ``degrees=True``), so
    ``kernel`` must use the same unit convention (a `Chordal` or
    `SphereMatern` with the same ``degrees``); a mismatch raises. ``a``,
    ``b``, ``c`` are called on a scalar latitude in **radians**. Use
    float64 for tight tolerances; float32 loses about the digits of the
    second derivative.

    **Poles.** In the chart $(\lambda, \phi)$, $\partial_\phi$ is the
    derivative to the north, whose direction at a pole depends on $\lambda$,
    so $Z$ is continuous at $\phi = \pm\pi/2$ only if $a(\pm\pi/2) = 0$.
    $\partial_\lambda Z_0 = \cos\phi\,\partial_{\text{east}} Z_0$ already
    vanishes there, so any bounded $b$ is harmless (Jun & Stein use
    $b = \tilde b\cos\phi$). Positive definiteness holds regardless;
    `LegendreLatitudeFn(..., cos_factor=True)` gives the vanishing factor.
    Autodiff of the trigonometric chart is finite at the poles (no
    $1/\cos\phi$ appears).

    Attributes:
        kernel: Isotropic twice-differentiable kernel $k_0$ on ``(lon, lat)``.
        a: $a(\phi)$, coefficient of $\partial_\phi$, latitude in radians.
        b: $b(\phi)$, coefficient of $\partial_\lambda$.
        c: $c(\phi)$, coefficient of the identity.
        degrees: Whether inputs are in degrees (else radians), static.

    References:
        Jun & Stein (2008), *Ann. Appl. Stat.* 2(4), 1271-1296.
        Jun & Stein (2007), *J. Amer. Statist. Assoc.* 102, 1085-1095.
        Stein (2005), *J. Amer. Statist. Assoc.* 100, 310-321.
        Porcu, Alegria & Furrer (2018), *Int. Stat. Rev.* 86, 344-377.

    Examples:
        >>> import einx
        >>> import jax.numpy as jnp
        >>> import kernellib as kl
        >>> base = kl.Chordal(kl.Matern(nu=2.5, lengthscale=0.4))
        >>> k = kl.JunStein(
        ...     base,
        ...     a=lambda phi: 0.3 * jnp.cos(phi),
        ...     b=lambda phi: 0.2 * jnp.cos(phi),
        ...     c=lambda phi: 1.0 + 0.5 * jnp.sin(phi),
        ... )
        >>> X = jnp.array([[0.0, 10.0], [40.0, -20.0], [200.0, 60.0]])
        >>> K = k(X, X)
        >>> K.shape, bool(jnp.allclose(K, einx.id("i j -> j i", K)))
        ((3, 3), True)
        >>> # rotating every longitude by the same angle leaves K unchanged
        >>> Xr = X + jnp.array([75.0, 0.0])
        >>> bool(jnp.allclose(K, k(Xr, Xr)))
        True
        >>> # a = b = 0, c = 1 recovers the base kernel
        >>> zero = lambda phi: 0.0 * phi
        >>> one = lambda phi: 1.0 + 0.0 * phi
        >>> k0 = kl.JunStein(base, zero, zero, one)
        >>> bool(jnp.allclose(k0(X, X), base(X, X)))
        True
    """

    kernel: AbstractPointwiseKernel
    a: Callable[[Float[Array, ""]], Float[Array, ""]]
    b: Callable[[Float[Array, ""]], Float[Array, ""]]
    c: Callable[[Float[Array, ""]], Float[Array, ""]]
    degrees: bool = eqx.field(default=True, static=True)

    def __check_init__(self) -> None:
        _check_smooth(self.kernel)
        for leaf in jax.tree.leaves(
            self.kernel, is_leaf=lambda n: hasattr(n, "degrees")
        ):
            declared = getattr(leaf, "degrees", None)
            if isinstance(declared, bool) and declared != self.degrees:
                raise ValueError(
                    f"JunStein(degrees={self.degrees}) differentiates in the "
                    f"input units, but {type(leaf).__name__} has "
                    f"degrees={declared}; make them agree."
                )

    def _alpha(self, x: Float[Array, " D"]) -> Float[Array, " 3"]:
        """Operator coefficients ``(c, b, a)`` per unit of the input coordinates."""
        phi = jnp.deg2rad(x[1]) if self.degrees else x[1]
        # d/d(radian) = (180/pi) d/d(degree)
        unit = jnp.rad2deg(1.0) if self.degrees else 1.0
        return jnp.stack([self.c(phi), unit * self.b(phi), unit * self.a(phi)])

    def pairwise(
        self, x: Float[Array, " D"], y: Float[Array, " D"]
    ) -> Float[Array, ""]:
        if x.shape[-1] != 2:
            raise ValueError(
                f"JunStein needs (lon, lat) inputs; got {x.shape[-1]} columns."
            )
        block = _value_gradient_block(self.kernel.pairwise, x, y)
        return einsum(self._alpha(x), block, self._alpha(y), "i, i j, j ->")

    @property
    def is_stationary(self) -> bool:
        return False
