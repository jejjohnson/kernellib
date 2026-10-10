r"""Intrinsic kernels on the sphere as a truncated Legendre (Schoenberg) series.

On the sphere $S^2$ of radius $R$, the Laplace-Beltrami eigenvalues are
$\lambda_l = l(l+1)/R^2$, each with the $2l+1$ spherical harmonics of degree
$l$. A kernel that is a function $\Phi \ge 0$ of the Laplacian is, by the
addition theorem, a Legendre series in $t = u \cdot v$ for unit vectors
$u, v$:

$$
k(u, v) = \frac{\sigma^2}{C} \sum_{l=0}^{L} (2l+1)\,\Phi(\lambda_l)\,P_l(u \cdot v),
\qquad C = \sum_{l=0}^{L} (2l+1)\,\Phi(\lambda_l),
$$

so $k(u, u) = \sigma^2$. Non-negative coefficients make it positive definite
on $S^2$ for every truncation $L$ (Schoenberg 1942): the truncated kernel is
exactly valid, not an approximation of a valid one.

References:
    Borovitskiy, V., Terenin, A., Mostowsky, P. & Deisenroth, M. P. (2020).
    Matérn Gaussian processes on Riemannian manifolds. *NeurIPS*.

    Schoenberg, I. J. (1942). Positive definite functions on spheres. *Duke
    Math. J.* 9(1), 96-108.
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable

import einx
import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float

from kernellib._einx import einsum, rearrange
from kernellib._kernels._base import AbstractPointwiseKernel
from kernellib.functional._geo import legendre_series, lonlat_to_unit


__all__ = [
    "AbstractSphereSeriesKernel",
    "SphereHeat",
    "SphereMatern",
    "SphereSeries",
]


class AbstractSphereSeriesKernel(AbstractPointwiseKernel):
    r"""A kernel on $S^2$ given by its Laplace-Beltrami spectrum $\Phi(\lambda)$.

    $k(u, v) = \sum_{l=0}^{L} a_l\,P_l(u \cdot v)$ with the normalised
    per-degree coefficients

    $$
    a_l = \frac{\sigma^2\,(2l+1)\,\Phi(\lambda_l)}{C},
    \qquad C = \sum_{m=0}^{L} (2m+1)\,\Phi(\lambda_m),
    \qquad \lambda_l = \frac{l(l+1)}{R^2},
    $$

    returned by `degree_spectrum`, so that $k(u, u) = \sum_l a_l = \sigma^2$.
    Subclasses implement `spectrum`, which only matters up to a constant
    factor (it cancels in $a_l$).

    The series is evaluated by the Legendre recurrence
    (`kernellib.functional.legendre_series`): a Gram matrix costs
    $O(N_1 N_2 L)$ time and $O(N_1 N_2)$ memory.

    **Inputs.** ``(N, 2)`` ``(lon, lat)`` points, in degrees unless
    ``degrees=False``; or, when ``X.shape[1] == 3``, points in $\mathbb{R}^3$,
    which are normalised to unit length (so $R\,u$ on a sphere of any radius
    works, as `funk_hecke_coefficients` passes them).

    **Truncation.** The kernel is positive definite for every
    ``max_degree`` $L$, but the coefficients beyond $L$ are dropped. A rough
    kernel (small $\nu$) or a short lengthscale relative to $R$ puts mass at
    high degrees; `truncation_tail` estimates the fraction lost, and the table
    on the Geo API page gives the $L$ needed for a 1 % tail.

    Attributes:
        variance: Scalar signal variance $\sigma^2$.
        radius: Sphere radius $R$ (static). Lengthscales are in its units:
            ``radius=1`` makes them angles in radians, `EARTH_RADIUS_KM`
            kilometres.
        degrees: Whether ``(lon, lat)`` inputs are in degrees (static).
        max_degree: Truncation degree $L \ge 0$ (static).
    """

    variance: Float[Array, ""] = eqx.field(
        default=1.0, converter=jnp.asarray, kw_only=True
    )
    radius: float = eqx.field(default=1.0, static=True, kw_only=True)
    degrees: bool = eqx.field(default=True, static=True, kw_only=True)
    max_degree: int = eqx.field(default=64, static=True, kw_only=True)

    def __check_init__(self) -> None:
        if not isinstance(self.max_degree, int) or self.max_degree < 0:
            raise ValueError(
                f"max_degree must be a non-negative int; got {self.max_degree!r}."
            )
        if not self.radius > 0:
            raise ValueError(f"radius must be positive; got {self.radius!r}.")

    @abstractmethod
    def spectrum(self, lam: Float[Array, " L"]) -> Float[Array, " L"]:
        r"""Unnormalised spectrum $\Phi(\lambda) \ge 0$ at eigenvalues ``lam``.

        Only the shape matters: any constant factor cancels in
        `degree_spectrum`.
        """
        raise NotImplementedError

    def _eigenvalues(self) -> tuple[Float[Array, " L"], Float[Array, " L"]]:
        """Degrees ``l = 0..L`` and their eigenvalues ``l(l+1)/R²``."""
        dtype = jnp.result_type(self.variance, 1.0)
        ell = jnp.arange(self.max_degree + 1, dtype=dtype)
        return ell, ell * (ell + 1.0) / self.radius**2

    def degree_spectrum(self) -> Float[Array, " L"]:
        r"""Normalised per-degree coefficients $a_0, \ldots, a_L$.

        $a_l = \sigma^2 (2l+1)\Phi(\lambda_l) / C$, the coefficients of the
        Legendre series $k(u, v) = \sum_l a_l P_l(u \cdot v)$; they sum to
        $\sigma^2$. Each degree-$l$ spherical harmonic $Y_{lm}$ (orthonormal
        on the unit sphere) carries prior variance $4\pi a_l / (2l+1)$.

        Returns:
            The ``(max_degree + 1,)`` coefficients, indexed by degree.
        """
        ell, lam = self._eigenvalues()
        weights = (2.0 * ell + 1.0) * self.spectrum(lam)
        return self.variance * weights / jnp.sum(weights)

    def _tail_integral(self, lam0: Float[Array, ""]) -> Float[Array, ""] | None:
        r"""Closed-form $\int_{\lambda_0}^\infty \Phi(\lambda)\,d\lambda$, or None."""
        return None

    def truncation_tail(self) -> Float[Array, ""]:
        r"""Estimated fraction of the series' mass beyond ``max_degree``.

        Estimates $\sum_{l > L} (2l+1)\Phi_l \big/ \sum_{l \ge 0}
        (2l+1)\Phi_l$. With $s = l(l+1)$, $ds = (2l+1)\,dl$, the tail is
        bounded by the integral $\int_L^\infty (2l+1)\Phi(\lambda_l)\, dl =
        R^2 \int_{\lambda_L}^\infty \Phi(\lambda)\, d\lambda$ whenever
        $(2l+1)\Phi(\lambda_l)$ decreases beyond $L$; `SphereMatern` and
        `SphereHeat` use that closed form. For a user spectrum
        (`SphereSeries`) the next $15(L+1)$ degrees are summed instead.

        Returns:
            A scalar in $[0, 1)$.
        """
        ell, lam = self._eigenvalues()
        head = jnp.sum((2.0 * ell + 1.0) * self.spectrum(lam))
        tail = self._tail_integral(lam[-1])
        if tail is None:
            extra = jnp.arange(
                self.max_degree + 1, 16 * (self.max_degree + 1), dtype=ell.dtype
            )
            lam_extra = extra * (extra + 1.0) / self.radius**2
            tail = jnp.sum((2.0 * extra + 1.0) * self.spectrum(lam_extra))
        else:
            tail = self.radius**2 * tail
        return tail / (head + tail)

    def to_unit(self, X: Float[Array, "N D"]) -> Float[Array, "N 3"]:
        """Unit vectors for ``(N, 2)`` ``(lon, lat)`` or ``(N, 3)`` inputs."""
        X = jnp.asarray(X)
        if X.ndim == 2 and X.shape[-1] == 3:
            norm = jnp.sqrt(einx.sum("n d -> n", X**2))
            return einx.divide("n d, n -> n d", X, norm)
        return lonlat_to_unit(X, degrees=self.degrees)

    def __call__(
        self, X1: Float[Array, "N1 D"], X2: Float[Array, "N2 D"]
    ) -> Float[Array, "N1 N2"]:
        cos = einsum(self.to_unit(X1), self.to_unit(X2), "n d, m d -> n m")
        return legendre_series(cos, self.degree_spectrum())

    def pairwise(
        self, x: Float[Array, " D"], y: Float[Array, " D"]
    ) -> Float[Array, ""]:
        u = rearrange(self.to_unit(rearrange(x, "d -> 1 d")), "1 d -> d")
        v = rearrange(self.to_unit(rearrange(y, "d -> 1 d")), "1 d -> d")
        return legendre_series(jnp.dot(u, v), self.degree_spectrum())

    def diag(self, X: Float[Array, "N D"]) -> Float[Array, " N"]:
        X = jnp.asarray(X)
        dtype = jnp.result_type(X, self.variance, 1.0)
        return self.variance * jnp.ones(X.shape[0], dtype=dtype)


class SphereMatern(AbstractSphereSeriesKernel):
    r"""Intrinsic Matérn kernel on the sphere (Borovitskiy et al. 2020).

    The Matérn GP on a $d$-dimensional manifold solves the SPDE
    $(2\nu/\ell^2 - \Delta)^{\nu/2 + d/4} f = \mathcal{W}$, so its spectrum is

    $$
    \Phi(\lambda) = \Big(\frac{2\nu}{\ell^2} + \lambda\Big)^{-\nu - d/2}
    = \Big(\frac{2\nu}{\ell^2} + \lambda\Big)^{-(\nu + 1)}
    \quad (d = 2),
    $$

    the Euclidean Matérn spectral density with $|\omega|^2 \to \lambda$. The
    manifold dimension **enters** the exponent, which is what makes $\nu$ the
    usual smoothness: for $\ell \ll R$ the kernel matches the Euclidean
    `Matern` with the same $\nu$ and $\ell$ locally. The graph Matérn
    (`kernellib.functional.graph_matern_spectrum`) instead uses
    $(2\nu/\ell^2 + \lambda)^{-\nu}$, with no dimension, following the graph
    convention in which $\nu$ plays the SPDE's $\alpha$.

    Unlike a Matérn of the great-circle distance (valid only for
    $\nu \le \tfrac12$), it is positive definite for every $\nu > 0$. As
    $\nu \to \infty$ it tends to `SphereHeat`. The spectrum is computed as
    $(1 + \ell^2\lambda / 2\nu)^{-(\nu+1)}$, i.e. divided by the constant
    $(2\nu/\ell^2)^{-(\nu+1)}$, which cancels in the normalisation and keeps
    large $\nu$ from underflowing.

    **Truncation.** $(2l+1)\Phi(\lambda_l) \sim l^{-2\nu-1}$, so the
    relative tail beyond $L$ is about $(1 + L(L+1)\ell^2/(2\nu R^2))^{-\nu}$:
    small $\nu$ and $\ell/R$ need a large ``max_degree`` (see
    `truncation_tail`).

    Attributes:
        lengthscale: Scalar lengthscale $\ell$, in units of ``radius``.
        nu: Smoothness $\nu > 0$ (static).
        variance: Scalar signal variance.
        radius: Sphere radius (static).
        degrees: Whether ``(lon, lat)`` inputs are in degrees (static).
        max_degree: Truncation degree $L$ (static).

    Examples:
        >>> import jax.numpy as jnp
        >>> import kernellib as kl
        >>> k = kl.SphereMatern(lengthscale=0.5, nu=1.5, max_degree=128)
        >>> X = jnp.array([[0.0, 0.0], [0.0, 30.0], [180.0, 0.0]])
        >>> [round(float(v), 3) for v in k(X, X)[0]]  # 0, 30 and 180 degrees away
        [1.0, 0.477, 0.002]
        >>> a = k.degree_spectrum()  # a_l, used by spherical-harmonic features
        >>> a.shape, round(float(a.sum()), 6)
        ((129,), 1.0)
        >>> bool(k.truncation_tail() < 1e-3)
        True
    """

    lengthscale: Float[Array, ""] = eqx.field(default=1.0, converter=jnp.asarray)
    nu: float = eqx.field(default=1.5, static=True)

    def __check_init__(self) -> None:
        if not self.nu > 0:
            raise ValueError(f"nu must be positive; got {self.nu!r}.")

    def spectrum(self, lam: Float[Array, " L"]) -> Float[Array, " L"]:
        x = self.lengthscale**2 * lam / (2.0 * self.nu)
        return (1.0 + x) ** (-(self.nu + 1.0))

    def _tail_integral(self, lam0: Float[Array, ""]) -> Float[Array, ""]:
        kappa2 = 2.0 * self.nu / self.lengthscale**2
        return kappa2 / self.nu * (1.0 + lam0 / kappa2) ** (-self.nu)


class SphereHeat(AbstractSphereSeriesKernel):
    r"""Heat (diffusion, squared-exponential) kernel on the sphere.

    $\Phi(\lambda) = \exp(-\ell^2\lambda / 2)$: the heat equation on $S^2$
    run for time $\ell^2/2$, the $\nu \to \infty$ limit of `SphereMatern`,
    and the intrinsic analogue of the RBF (which, as a function of the
    great-circle distance, is not positive definite). The same spectrum as
    `kernellib.functional.graph_heat_spectrum`.

    **Truncation.** The tail beyond $L$ is about
    $\exp(-\ell^2 L(L+1) / 2R^2)$, so $L \approx 3R/\ell$ suffices for 1 %.

    Attributes:
        lengthscale: Scalar lengthscale $\ell$, in units of ``radius``.
        variance: Scalar signal variance.
        radius: Sphere radius (static).
        degrees: Whether ``(lon, lat)`` inputs are in degrees (static).
        max_degree: Truncation degree $L$ (static).

    Examples:
        >>> import jax.numpy as jnp
        >>> import kernellib as kl
        >>> k = kl.SphereHeat(lengthscale=0.5)
        >>> X = jnp.array([[0.0, 0.0], [0.0, 30.0], [180.0, 0.0]])
        >>> [round(float(v), 2) for v in k(X, X)[0]]
        [1.0, 0.59, 0.0]
    """

    lengthscale: Float[Array, ""] = eqx.field(default=1.0, converter=jnp.asarray)

    def spectrum(self, lam: Float[Array, " L"]) -> Float[Array, " L"]:
        return jnp.exp(-0.5 * self.lengthscale**2 * lam)

    def _tail_integral(self, lam0: Float[Array, ""]) -> Float[Array, ""]:
        c = 0.5 * self.lengthscale**2
        return jnp.exp(-c * lam0) / c


class SphereSeries(AbstractSphereSeriesKernel):
    r"""A sphere kernel from any user spectrum $\Phi(\lambda) \ge 0$.

    ``spectrum_fn`` maps an array of Laplace-Beltrami eigenvalues
    $\lambda_l = l(l+1)/R^2$ to non-negative values (negative ones would break
    positive definiteness; they are not checked). It may close over arrays,
    or be an `equinox.Module`, to make its parameters differentiable.

    Attributes:
        spectrum_fn: ``lam -> Phi(lam)``, elementwise on a 1-D array.
        variance: Scalar signal variance.
        radius: Sphere radius (static).
        degrees: Whether ``(lon, lat)`` inputs are in degrees (static).
        max_degree: Truncation degree $L$ (static).

    Examples:
        >>> import jax.numpy as jnp
        >>> import kernellib as kl
        >>> heat = kl.SphereSeries(lambda lam: jnp.exp(-0.125 * lam))
        >>> X = jnp.array([[0.0, 0.0], [0.0, 30.0]])
        >>> bool(jnp.allclose(heat(X, X), kl.SphereHeat(lengthscale=0.5)(X, X)))
        True
    """

    spectrum_fn: Callable[[Float[Array, " L"]], Float[Array, " L"]]

    def spectrum(self, lam: Float[Array, " L"]) -> Float[Array, " L"]:
        return jnp.asarray(self.spectrum_fn(lam))
