r"""Spherical-harmonic features for zonal kernels on the sphere.

A zonal kernel on $S^2$, one that depends on unit vectors $u, v$ only through
$t = u \cdot v$, is diagonal in the spherical harmonics. With the addition
theorem for real harmonics orthonormal on the unit sphere,

$$
\sum_{m=-l}^{l} Y_{lm}(u)\, Y_{lm}(v) = \frac{2l+1}{4\pi}\, P_l(u \cdot v),
$$

a Legendre series $k(u, v) = \sum_l a_l P_l(u \cdot v)$ is exactly
$\sum_{l, m} \phi_{lm}(u)\, \phi_{lm}(v)$ with
$\phi_{lm} = \sqrt{4\pi a_l / (2l+1)}\, Y_{lm}$. This is the sphere's
analogue of the Laplace-eigenfunction (HSGP) features on a box: the basis is
kernel-independent (the real harmonics of
``geonnax.basis.real_spherical_harmonics``, in its column order) and only
the per-degree weights carry the hyperparameters.

References:
    Dutordoir, V., Durrande, N. & Hensman, J. (2020). Sparse Gaussian
    processes with spherical harmonic features. *ICML*.
"""

from __future__ import annotations

import dataclasses
import functools
import math

import einx
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from geonnax.basis import harmonic_degrees
from jaxtyping import Array, Float

from kernellib._einx import rearrange
from kernellib._geo._anisotropy import _AbstractLinearTransform
from kernellib._geo._chordal import Chordal
from kernellib._geo._great_circle import AbstractGreatCircleKernel
from kernellib._geo._sphere_series import AbstractSphereSeriesKernel
from kernellib._kernels import AbstractKernel
from kernellib._kernels._compose import ActiveDims, Periodised, Warped
from kernellib._spectral._base import AbstractFeatureMap
from kernellib._spectral._funk_hecke import _submodules, funk_hecke_coefficients
from kernellib.functional._geo import lonlat_to_unit


__all__ = ["SphericalHarmonicFeatures"]

# Kernels whose inputs are not points of R^3, or that are not rotation
# invariant there, so the Funk-Hecke path would silently be wrong.
_NOT_ZONAL = (
    ActiveDims,
    Warped,
    Periodised,
    _AbstractLinearTransform,
    AbstractGreatCircleKernel,
    Chordal,
)


class SphericalHarmonicFeatures(AbstractFeatureMap):
    r"""Weighted real spherical harmonics up to degree $L$ for a zonal kernel.

    $\phi_{lm}(u) = w_l\, Y_{lm}(u)$ for $0 \le l \le L$, $-l \le m \le l$:
    $(L+1)^2$ deterministic features with $\Phi\Phi^\top = k_L$, the kernel's
    Legendre series truncated at degree $L$. The weights depend on the kernel:

    - a sphere-series kernel (`SphereMatern`, `SphereHeat`, `SphereSeries`)
      gives $w_l = \sqrt{4\pi a_l / (2l+1)}$ from its `degree_spectrum`, so
      with ``max_degree`` equal to the kernel's the features reproduce its
      Gram exactly;
    - a `Chordal` kernel, or any other isotropic kernel on $\mathbb{R}^3$
      (restricted to the sphere of radius ``radius``), gives
      $w_l = \sqrt{a_l}$ with $a_l$ from `funk_hecke_coefficients`
      (``num_quadrature`` Gauss-Legendre nodes). The error is the series tail
      $\sum_{l > L} \frac{2l+1}{4\pi} a_l$ at $u = v$.

    Anything else is rejected: ARD lengthscales, `GeometricAnisotropy`,
    `LinearTransform`, `ActiveDims`, `Warped`, `Periodised`, great-circle
    kernels (their inputs are ``(lon, lat)``, not points of
    $\mathbb{R}^3$), and a `Chordal` nested inside a composite. Isotropy of
    other kernels is assumed, not checked.

    **Inputs.** ``(N, 2)`` ``(lon, lat)`` points (in degrees unless
    ``degrees`` resolves to ``False``), or ``(N, 3)`` points in
    $\mathbb{R}^3$, normalised to unit length.

    **When to use it.** Features cost $O(N (L+1)^2)$ time and memory and turn
    a GP into Bayesian linear regression with $(L+1)^2$ weights, so for
    $N \gg (L+1)^2$ (dense global data with a smooth field) they beat the
    dense $O(N^3)$ kernel. A short lengthscale or a rough kernel needs a large
    $L$ (see `truncation_tail` and the table on the Geo API page), and the
    feature count grows as $L^2$: $L = 64$ gives 4225 features. Then the dense
    kernel, or a sparse method on it, is cheaper.

    Attributes:
        max_degree: Truncation degree $L$; ``None`` takes the sphere-series
            kernel's ``max_degree`` at `fit` (required for other kernels).
        degrees: Whether ``(lon, lat)`` inputs are in degrees; ``None``
            follows the kernel's ``degrees`` for a sphere-series or `Chordal`
            kernel, and ``True`` otherwise.
        radius: Sphere radius for a bare kernel on $\mathbb{R}^3$; ``None``
            means 1 (a `Chordal` kernel uses its own radius, a sphere-series
            kernel needs none).
        num_quadrature: Gauss-Legendre nodes for the Funk-Hecke path.
        kernel: The fitted kernel, ``None`` before `fit`.

    Examples:
        >>> import jax.numpy as jnp
        >>> import kernellib as kl
        >>> from kernellib._einx import einsum
        >>> from kernellib._testing import fibonacci_lonlat
        >>> X = fibonacci_lonlat(50)
        >>> k = kl.SphereMatern(lengthscale=0.5, nu=1.5, max_degree=16)
        >>> sh = kl.SphericalHarmonicFeatures().fit(k, X)
        >>> sh.n_features
        289
        >>> Phi = sh(X)
        >>> Phi.shape
        (50, 289)
        >>> err = einsum(Phi, Phi, "n f, m f -> n m") - k(X, X)
        >>> bool(jnp.max(jnp.abs(err)) < 1e-5)
        True

        Any isotropic kernel through `Chordal`, by Funk-Hecke quadrature:

        >>> c = kl.Chordal(kl.RBF(lengthscale=0.5))
        >>> sh = kl.SphericalHarmonicFeatures(max_degree=20).fit(c, X)
        >>> err = einsum(sh(X), sh(X), "n f, m f -> n m") - c(X, X)
        >>> bool(jnp.max(jnp.abs(err)) < 1e-5)
        True
    """

    max_degree: int | None = eqx.field(default=None, static=True)
    degrees: bool | None = eqx.field(default=None, static=True)
    radius: float | None = eqx.field(default=None, static=True)
    num_quadrature: int = eqx.field(default=256, static=True)
    kernel: AbstractKernel | None = None

    def __check_init__(self) -> None:
        if self.max_degree is not None and (
            not isinstance(self.max_degree, int) or self.max_degree < 0
        ):
            raise ValueError(
                f"max_degree must be a non-negative int; got {self.max_degree!r}."
            )
        if self.radius is not None and not self.radius > 0:
            raise ValueError(f"radius must be positive; got {self.radius!r}.")
        if self.num_quadrature < 1:
            raise ValueError(f"num_quadrature must be >= 1; got {self.num_quadrature}.")

    def fit(
        self, kernel: AbstractKernel, X: Float[Array, "N D"]
    ) -> SphericalHarmonicFeatures:
        """Resolve the degree, input units and radius for ``kernel``.

        ``X`` only fixes the input layout, ``(N, 2)`` or ``(N, 3)``.

        Raises:
            NotImplementedError: If the kernel is not zonal on the sphere.
            ValueError: If ``X`` is not ``(N, 2)`` or ``(N, 3)``, if
                ``max_degree`` is missing (non-series kernel) or exceeds a
                series kernel's, or if ``degrees`` / ``radius`` contradict
                the kernel's own.
        """
        _check_inputs(X)
        max_degree, degrees, radius = self.max_degree, self.degrees, self.radius
        if isinstance(kernel, AbstractSphereSeriesKernel):
            if max_degree is None:
                max_degree = kernel.max_degree
            elif max_degree > kernel.max_degree:
                raise ValueError(
                    f"max_degree={max_degree} exceeds the kernel's "
                    f"max_degree={kernel.max_degree}; the extra degrees would "
                    "have zero weight."
                )
            degrees = _resolve(degrees, kernel.degrees, "degrees")
            if radius is not None:
                raise ValueError(
                    "radius is only for a bare kernel on R^3; a sphere-series "
                    "kernel carries its own."
                )
        else:
            inner = kernel
            if isinstance(kernel, Chordal):
                degrees = _resolve(degrees, kernel.degrees, "degrees")
                radius = _resolve(radius, kernel.radius, "radius")
                inner = kernel.kernel
            for module in _submodules(inner):
                if isinstance(module, _NOT_ZONAL):
                    raise NotImplementedError(
                        "SphericalHarmonicFeatures needs a zonal kernel: a "
                        "sphere-series kernel, a Chordal kernel, or an "
                        f"isotropic kernel on R^3; got {type(module).__name__}"
                        f" inside {type(kernel).__name__}."
                    )
            if max_degree is None:
                raise ValueError(
                    "max_degree is required for a kernel that is not a "
                    f"sphere-series kernel; got {type(kernel).__name__}."
                )
            degrees = True if degrees is None else degrees
            radius = 1.0 if radius is None else radius
            # Fail here (ARD lengthscales) rather than at the first call.
            funk_hecke_coefficients(
                kernel, 0, radius=radius, num_quadrature=self.num_quadrature
            )
        return dataclasses.replace(
            self, kernel=kernel, max_degree=max_degree, degrees=degrees, radius=radius
        )

    @property
    def n_features(self) -> int:
        """Number of features, ``(max_degree + 1) ** 2``."""
        if self.max_degree is None:
            raise RuntimeError(
                "n_features is unknown until max_degree is set or the map is "
                "fitted to a sphere-series kernel."
            )
        return (self.max_degree + 1) ** 2

    def degree_weights(self) -> Float[Array, " L"]:
        r"""Per-degree feature weights $w_0, \ldots, w_L$ (zero where $a_l = 0$)."""
        assert self.kernel is not None and self.max_degree is not None
        L = self.max_degree
        if isinstance(self.kernel, AbstractSphereSeriesKernel):
            a = self.kernel.degree_spectrum()[: L + 1]
            ell = jnp.arange(L + 1, dtype=a.dtype)
            variance = 4.0 * math.pi * a / (2.0 * ell + 1.0)
        else:
            variance = funk_hecke_coefficients(
                self.kernel,
                L,
                radius=1.0 if self.radius is None else self.radius,
                num_quadrature=self.num_quadrature,
            )
        # sqrt has an infinite derivative at 0 (an underflowed or clipped
        # a_l); route those through a safe value so gradients stay finite.
        positive = variance > 0
        safe = jnp.where(positive, variance, 1.0)
        return jnp.where(positive, jnp.sqrt(safe), 0.0)

    def features(self, X: Float[Array, "N D"]) -> Float[Array, "N M"]:
        assert self.max_degree is not None
        X = jnp.asarray(X)
        _check_inputs(X)
        if X.shape[-1] == 3:
            norm = jnp.sqrt(einx.sum("n d -> n", X**2))
            unit = einx.divide("n d, n -> n d", X, norm)
        else:
            unit = lonlat_to_unit(X, degrees=self.degrees is not False)
        Y = _real_spherical_harmonics(unit, self.max_degree)
        w = self.degree_weights().astype(Y.dtype)
        columns = w[jnp.asarray(harmonic_degrees(self.max_degree))]
        return einx.multiply("n f, f -> n f", Y, columns)


@functools.partial(jax.jit, static_argnums=1)
def _real_spherical_harmonics(
    unit: Float[Array, "N 3"], l_max: int
) -> Float[Array, "N M"]:
    r"""Orthonormal real spherical harmonics, as ``geonnax.basis``'s.

    The same functions, normalisation and column order as
    ``geonnax.basis.real_spherical_harmonics`` (tested to agree), evaluated
    with fixed-shape scans instead of one array per harmonic: geonnax stacks
    $(L+1)^2$ separate arrays, which XLA takes minutes to compile beyond
    $L \approx 20$, and its unnormalised $Q_l^m$ overflow beyond
    $L \approx 85$. Here the fully normalised recursion

    $$
    \bar q_m^m = -\sqrt{\tfrac{2m+1}{2m}}\,\bar q_{m-1}^{m-1},\quad
    \bar q_{m+1}^m = \sqrt{2m+3}\,z\,\bar q_m^m,\quad
    \bar q_l^m = a_{lm}\big(z\,\bar q_{l-1}^m - b_{lm}\,\bar q_{l-2}^m\big),
    $$

    with $a_{lm} = \sqrt{(4l^2-1)/(l^2-m^2)}$ and
    $b_{lm} = \sqrt{((l-1)^2-m^2)/(4(l-1)^2-1)}$, runs over all orders at
    once; $\bar q_l^m = N_l^m P_l^m(z) / (1-z^2)^{m/2}$, multiplied by
    $\sqrt2\,\mathrm{Re}/\mathrm{Im}\,(x+iy)^m$ for $m \ne 0$.
    """
    dtype = unit.dtype
    x, y, z = unit[:, 0], unit[:, 1], unit[:, 2]
    m = jnp.arange(l_max + 1, dtype=dtype)

    # Diagonal seeds qbar_m^m (constants), and the l-recursion coefficients.
    ratio = jnp.where(
        m > 0, -jnp.sqrt((2.0 * m + 1.0) / jnp.maximum(2.0 * m, 1.0)), 1.0
    )
    diag = jnp.cumprod(ratio) / math.sqrt(4.0 * math.pi)

    def degree(carry, ell):
        prev, prev2 = carry  # qbar_{l-1}^m, qbar_{l-2}^m: (L+1, N)
        l2, m2 = ell**2, m**2
        a = jnp.sqrt((4.0 * l2 - 1.0) / jnp.maximum(l2 - m2, 1.0))
        b = jnp.sqrt(
            jnp.maximum((ell - 1.0) ** 2 - m2, 0.0)
            / jnp.maximum(4.0 * (ell - 1.0) ** 2 - 1.0, 1.0)
        )
        general = einx.multiply("m, m n -> m n", a, z * prev) - einx.multiply(
            "m, m n -> m n", a * b, prev2
        )
        first = einx.multiply("m, n -> m n", jnp.sqrt(2.0 * m + 3.0) * diag, z)
        seed = einx.multiply("m, n -> m n", diag, jnp.ones_like(z))
        q = jnp.where(
            einx.id("m -> m 1", m == ell),
            seed,
            jnp.where(
                einx.id("m -> m 1", m == ell - 1.0),
                first,
                jnp.where(einx.id("m -> m 1", m < ell - 1.0), general, 0.0),
            ),
        )
        return (q, prev), q

    zeros = jnp.zeros((l_max + 1, unit.shape[0]), dtype=dtype)
    _, Q = jax.lax.scan(degree, (zeros, zeros), m)  # (l, m, N)

    def power(carry, _):
        re, im = carry
        return (x * re - y * im, x * im + y * re), carry

    _, (re, im) = jax.lax.scan(
        power, (jnp.ones_like(x), jnp.zeros_like(x)), None, length=l_max + 1
    )  # Re / Im (x + iy)^m: (m, N)

    ell = np.asarray(harmonic_degrees(l_max))
    order = np.concatenate([np.arange(-l, l + 1) for l in range(l_max + 1)])
    am = np.abs(order)
    trig = jnp.where(
        einx.id("f -> f 1", jnp.asarray(order > 0)),
        math.sqrt(2.0) * re[am],
        jnp.where(
            einx.id("f -> f 1", jnp.asarray(order < 0)), math.sqrt(2.0) * im[am], 1.0
        ),
    )
    return rearrange(Q[ell, am] * trig, "f n -> n f")


def _check_inputs(X: Float[Array, "N D"]) -> None:
    if X.ndim != 2 or X.shape[-1] not in (2, 3):
        raise ValueError(
            "SphericalHarmonicFeatures takes (N, 2) (lon, lat) or (N, 3) "
            f"inputs; got shape {X.shape}."
        )


def _resolve(value, own, name: str):
    """The kernel's own setting, or ``value`` if it agrees with it."""
    if value is not None and value != own:
        raise ValueError(
            f"{name}={value!r} contradicts the kernel's {name}={own!r}; leave "
            f"it as None to follow the kernel."
        )
    return own
