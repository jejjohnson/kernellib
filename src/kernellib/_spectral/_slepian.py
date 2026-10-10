r"""Slepian features: spherical-harmonic features concentrated on a cap.

A zonal kernel on the sphere is diagonal in real spherical harmonics,
$k(u, v) = \sum_{l,m} a_l\, Y_{lm}(u)\, Y_{lm}(v)$, with $a_l$ its
Funk-Hecke spectrum. For data on one region of the globe most of the
$(L+1)^2$ harmonics of degree $\le L$ carry energy outside the region.
The Slepian functions of a spherical cap (Simons, Dahlen & Wieczorek 2006)
are the band-limited combinations $g = Y(Ru)\, C$ that are optimally
concentrated in the cap; about the Shannon number
$N = (L+1)^2 A / 4\pi$ of them are well concentrated. Projecting the kernel
onto their span (as inducing features, Hensman et al. 2017) gives

$$
K_{uu} = C^\top \mathrm{diag}(a)\, C, \qquad
k_u(x) = C^\top \mathrm{diag}(a)\, Y(Rx)^\top, \qquad
\phi(x) = L^{-1} k_u(x),
$$

with $L L^\top = K_{uu} + \epsilon I$. The maths is ported from pyrox-gp's
``SlepianInducingFeatures``; the basis is geonnax's `SlepianCapBasis`.
"""

from __future__ import annotations

import dataclasses
import math

import einx
import equinox as eqx
import jax.numpy as jnp
import jax.scipy.linalg as jsl
from geonnax.basis import (
    SlepianCapBasis,
    harmonic_degrees,
    real_spherical_harmonics,
    shannon_number,
    slepian_cap_basis,
)
from jaxtyping import Array, Float

from kernellib._einx import einsum, rearrange
from kernellib._kernels import AbstractKernel
from kernellib._spectral._base import AbstractFeatureMap
from kernellib._spectral._funk_hecke import funk_hecke_coefficients
from kernellib.functional._geo import lonlat_to_unit


__all__ = ["SlepianFeatures"]


def _degree_coefficients(
    kernel: AbstractKernel,
    l_max: int,
    num_quadrature: int,
    *,
    radius: float = 1.0,
) -> Float[Array, " L"]:
    """Per-degree spectrum ``a_l`` (Funk-Hecke convention), ``l = 0..l_max``.

    The single place a zonal kernel becomes its degree spectrum, so kernels
    with a closed-form series can be special-cased here later. For now every
    kernel goes through `funk_hecke_coefficients` (which unwraps `Chordal`).
    """
    return funk_hecke_coefficients(
        kernel, l_max, radius=radius, num_quadrature=num_quadrature
    )


class SlepianFeatures(AbstractFeatureMap):
    r"""Features of a zonal kernel concentrated on a spherical cap.

    The map projects the kernel onto the span of the ``K`` best-concentrated
    Slepian functions of degree $\le$ ``l_max`` on the cap,
    $\phi(x) = L^{-1} C^\top \mathrm{diag}(a)\, Y(Rx)^\top$, where $Y$ are the
    real spherical harmonics evaluated in the frame that rotates the cap
    centre to the north pole, $C$ the Slepian coefficients
    (`SlepianCapBasis.coeffs`), $a$ the kernel's Funk-Hecke spectrum repeated
    over the orders $m$, and $L L^\top = C^\top \mathrm{diag}(a) C + \epsilon I$.
    $\Phi \Phi^\top$ is then the Nyström-type projection of the kernel onto
    the Slepian span: exact (up to $\epsilon$) when all $(L+1)^2$ modes are
    kept, and accurate inside the cap with about the Shannon number
    $(L+1)^2 (1 - \cos r) / 2$ of them. Outside the cap the features decay,
    so do not use the map there.

    The kernel must be zonal: a [`Chordal`][kernellib.Chordal] kernel, or an
    isotropic kernel on $\mathbb{R}^3$ evaluated on the sphere of radius
    ``radius``. Its spectrum is recomputed at call time, so gradients reach
    its hyperparameters. The basis is built once by `fit`; inputs are
    ``(lon, lat)``.

    Cost: $O(N (L+1)^2 K)$ per call plus $O(K^3)$ for the Cholesky factor,
    and a Funk-Hecke quadrature of ``num_quadrature`` kernel evaluations.

    Attributes:
        l_max: Maximum spherical-harmonic degree $L$.
        cap_radius: Cap half-angle, in radians, in ``(0, pi]``.
        cap_centre: Cap centre ``(lon, lat)``, in degrees when ``degrees``
            (else radians).
        n_modes: Number of Slepian modes kept. ``None`` keeps the rounded
            Shannon number, unless ``eig_threshold`` is set, which then
            alone decides.
        eig_threshold: Keep only modes whose concentration ratio exceeds it
            (applied before ``n_modes``).
        degrees: Whether ``cap_centre`` and the inputs are in degrees.
        num_quadrature: Gauss-Legendre nodes for the Funk-Hecke spectrum.
        radius: Sphere radius the kernel is evaluated on; ignored for a
            ``Chordal`` kernel, which carries its own.
        jitter: Relative diagonal jitter $\epsilon$ on $K_{uu}$, scaled by
            its mean diagonal.
        basis: The fitted `geonnax.basis.SlepianCapBasis`, ``None`` before
            `fit`.
        kernel: The fitted kernel, ``None`` before `fit`.

    Examples:
        >>> import math
        >>> import einx
        >>> import jax.numpy as jnp
        >>> import kernellib as kl
        >>> k = kl.Chordal(kl.Matern(nu=2.5, lengthscale=0.5))
        >>> sf = kl.SlepianFeatures(
        ...     l_max=8, cap_radius=math.radians(30.0), cap_centre=(10.0, 45.0)
        ... )
        >>> # (lon, lat) in degrees: two points in the cap, one at its antipode
        >>> X = jnp.array([[10.0, 45.0], [15.0, 50.0], [-170.0, -45.0]])
        >>> sf = sf.fit(k, X)
        >>> sf.basis.num_modes  # Shannon number: 81 * (1 - cos 30°) / 2
        5
        >>> Phi = sf(X)
        >>> Phi.shape
        (3, 5)
        >>> # ||phi(x)||^2 is close to k(x, x) = 1 in the cap, ~0 far from it
        >>> [round(float(v), 2) for v in einx.sum("n k -> n", Phi**2)]
        [0.93, 0.94, 0.0]
    """

    l_max: int = eqx.field(static=True)
    cap_radius: float = eqx.field(static=True)
    cap_centre: tuple[float, float] = eqx.field(static=True)
    n_modes: int | None = eqx.field(default=None, static=True)
    eig_threshold: float | None = eqx.field(default=None, static=True)
    degrees: bool = eqx.field(default=True, static=True)
    num_quadrature: int = eqx.field(default=256, static=True)
    radius: float = eqx.field(default=1.0, static=True)
    jitter: float = eqx.field(default=1e-6, static=True)
    basis: SlepianCapBasis | None = None
    kernel: AbstractKernel | None = None

    def __check_init__(self) -> None:
        if self.l_max < 0:
            raise ValueError(f"l_max must be >= 0, got {self.l_max}.")
        if not 0.0 < self.cap_radius <= math.pi:
            raise ValueError(
                f"cap_radius must lie in (0, pi] radians, got {self.cap_radius}."
            )
        if len(self.cap_centre) != 2:
            raise ValueError(f"cap_centre must be (lon, lat), got {self.cap_centre}.")
        n_harmonics = (self.l_max + 1) ** 2
        if self.n_modes is not None and not 1 <= self.n_modes <= n_harmonics:
            raise ValueError(
                f"n_modes must lie in [1, (l_max + 1)^2 = {n_harmonics}], "
                f"got {self.n_modes}."
            )
        if self.num_quadrature < 1:
            raise ValueError(f"num_quadrature must be >= 1, got {self.num_quadrature}.")
        if self.jitter < 0.0:
            raise ValueError(f"jitter must be >= 0, got {self.jitter}.")

    @property
    def shannon_number(self) -> int:
        """The cap's rounded Shannon number $(L+1)^2 (1 - \\cos r) / 2$, ``>= 1``."""
        area = 2.0 * math.pi * (1.0 - math.cos(self.cap_radius))
        n = round(float(shannon_number(self.l_max, area)))
        return min(max(n, 1), (self.l_max + 1) ** 2)

    def fit(self, kernel: AbstractKernel, X: Float[Array, "N 2"]) -> SlepianFeatures:
        """Build the Slepian basis of the cap and store ``kernel``.

        The basis depends on the cap and ``l_max`` only; ``X`` is checked
        for shape. The kernel's spectrum is computed here once, so an
        unsupported (e.g. ARD) kernel fails at fit time.

        Raises:
            ValueError: If ``X`` is not ``(N, 2)``, or ``eig_threshold``
                leaves fewer than ``n_modes`` modes.
            NotImplementedError: If the kernel has an ARD lengthscale.
        """
        if X.ndim != 2 or X.shape[-1] != 2:
            raise ValueError(f"X must be (N, 2) (lon, lat); got shape {X.shape}.")
        _degree_coefficients(
            kernel, self.l_max, self.num_quadrature, radius=self.radius
        )
        n_modes = self.n_modes
        if n_modes is None and self.eig_threshold is None:
            n_modes = self.shannon_number
        lon, lat = self.cap_centre
        if self.degrees:
            lon, lat = math.radians(lon), math.radians(lat)
        basis = slepian_cap_basis(
            self.l_max,
            float(self.cap_radius),
            n_modes=n_modes,
            eig_threshold=self.eig_threshold,
            lonlat_centre=jnp.asarray([lon, lat]),
        )
        return dataclasses.replace(self, kernel=kernel, basis=basis)

    def features(self, X: Float[Array, "N 2"]) -> Float[Array, "N K"]:
        assert self.kernel is not None and self.basis is not None
        if X.ndim != 2 or X.shape[-1] != 2:
            raise ValueError(f"X must be (N, 2) (lon, lat); got shape {X.shape}.")
        C = self.basis.coeffs.astype(X.dtype)  # (M, K)
        a = _degree_coefficients(
            self.kernel, self.l_max, self.num_quadrature, radius=self.radius
        ).astype(X.dtype)
        a = a[jnp.asarray(harmonic_degrees(self.l_max))]  # per harmonic, (M,)

        aC = einx.multiply("m k, m -> m k", C, a)
        K_uu = einsum(C, aC, "m i, m j -> i j")
        scale = jnp.mean(jnp.diag(K_uu))
        eps = self.jitter * jnp.where(scale > 0, scale, 1.0)
        L = jnp.linalg.cholesky(K_uu + eps * jnp.eye(K_uu.shape[0], dtype=X.dtype))

        unit = lonlat_to_unit(X, degrees=self.degrees)
        Y = real_spherical_harmonics(self.basis.centred_coordinates(unit), self.l_max)
        k_u = einsum(Y.astype(X.dtype), aC, "n m, m k -> k n")  # (K, N)
        return rearrange(jsl.solve_triangular(L, k_u, lower=True), "k n -> n k")
