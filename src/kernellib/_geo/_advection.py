r"""Transport kernels: space-time covariance carried by a flow.

Inputs are space-time points $z = (x, t)$, stored as ``(N, D + 1)`` arrays
``[space (D) | time (1)]``, with lags $h = x - x'$ and $u = t - t'$.

- `Advected` is the frozen-field ("Taylor hypothesis") kernel: a spatial
  field $Z_0 \sim \mathcal{GP}(0, k_s)$ moving at a constant velocity
  $v$, $Z(x, t) = Z_0(x - v t)$, so $k = k_s(h - v u)$.
- `RandomAdvectionRBF` averages the frozen-field RBF over a Gaussian
  velocity $V \sim \mathcal N(\mu, \Sigma_V)$, in closed form (Cox & Isham,
  1988; Schlather, 2010).

Both depend on $z - z'$ only, so they are stationary in space-time. A
deterministic wind field that varies in space or time (true Lagrangian
trajectories) is out of scope here.
"""

from __future__ import annotations

from typing import ClassVar

import equinox as eqx
import jax.numpy as jnp
import jax.scipy.linalg as jsl
from jaxtyping import Array, Float

from kernellib._einx import einsum, rearrange, reduce
from kernellib._geo._anisotropy import _AbstractLinearTransform
from kernellib._kernels._base import AbstractPointwiseKernel


__all__ = ["Advected", "RandomAdvectionRBF"]


def _as_1d(x: float | Float[Array, ...]) -> Float[Array, " n"]:
    return jnp.atleast_1d(jnp.asarray(x))


class Advected(_AbstractLinearTransform):
    r"""A frozen spatial field moving at a constant velocity.

    $$
    k\bigl((x, t), (x', t')\bigr) = k_s\bigl((x - v t) - (x' - v t')\bigr)
    = k_s(h - v u),
    $$

    with $h = x - x'$ and $u = t - t'$: the covariance of
    $Z(x, t) = Z_0(x - v t)$ for $Z_0 \sim \mathcal{GP}(0, k_s)$. Inputs
    are ``[space (D) | time (1)]``. An observation at $(x, t)$ is perfectly
    correlated with one at $(x + v\Delta, t + \Delta)$, downwind and later.

    It is `Warped` with the linear warp $w(x, t) = x - v t$, written as the
    ``(D, D + 1)`` map $A = [\,I \mid -v\,]$, so it is PSD whenever $k_s$
    is, stationary in space-time, and differentiable in ``velocity``. Its
    frequencies are $A^\top\omega_0 = (\omega_0, -v^\top\omega_0)$ with
    $\omega_0$ drawn from $k_s$, so `RandomFourierFeatures` and
    `draw_rff_cosine_basis` accept it. Its space-time spectral measure is
    singular (it lives on a ``D``-dimensional subspace), so
    `spectral_density` raises, and so do the paths built on it
    (`LaplaceEigenfunctionFeatures`); `OrthogonalRandomFeatures` and
    `FastFoodFeatures` refuse it as anisotropic.

    Pure transport never decorrelates in time. For decay on top of
    transport, multiply by a kernel on the time column:
    ``Advected(Matern(...), v) * Matern(...).select((D,))``, the
    `Product` of the two (Gneiting, Genton & Guttorp, 2007, §5).

    Attributes:
        kernel: The stationary spatial kernel $k_s$ on $\mathbb{R}^D$.
        velocity: $v$, shape ``(D,)`` (a scalar is ``D = 1``); learnable.

    Raises:
        TypeError: If ``kernel`` is not stationary.
        ValueError: If ``velocity`` is not a vector, or the inputs do not
            have ``D + 1`` columns.

    Examples:
        >>> import jax.numpy as jnp
        >>> import kernellib as kl
        >>> v = jnp.array([2.0, -1.0])  # wind, in space units per time unit
        >>> k = kl.Advected(kl.RBF(lengthscale=0.5), v)
        >>> z = jnp.array([0.3, 0.1, 0.0])  # (x, y, t)
        >>> # three time units later the parcel is 3 v downwind: correlation 1
        >>> downwind = z + jnp.concatenate([3.0 * v, jnp.array([3.0])])
        >>> round(float(k.pairwise(z, downwind)), 6)
        1.0
        >>> round(float(k.pairwise(z, z + jnp.array([0.0, 0.0, 3.0]))), 6)
        0.0
        >>> # with temporal decay on top of transport
        >>> kd = k * kl.Matern(nu=1.5, lengthscale=10.0).select((2,))
        >>> round(float(kd.pairwise(z, downwind)), 4)  # Matérn-3/2 at r = 0.3
        0.9038
    """

    velocity: Float[Array, " D"] = eqx.field(converter=_as_1d)

    # The density on R^{D+1} is singular; nothing reads it as radial.
    _radial_unit_spectrum: ClassVar[bool] = False

    def __check_init__(self) -> None:
        if self.velocity.ndim != 1:
            raise ValueError(
                f"velocity must be a vector of shape (D,), got {self.velocity.shape}."
            )

    @property
    def space_dims(self) -> int:
        """``D``, the number of spatial columns (from ``velocity``)."""
        return self.velocity.shape[0]

    def _matrix(self) -> Float[Array, "D D1"]:
        """``A = [I | -v]``, shape ``(D, D + 1)``."""
        v = self.velocity
        eye = jnp.eye(self.space_dims, dtype=v.dtype)
        return jnp.concatenate([eye, rearrange(-v, "d -> d 1")], axis=1)

    def spectral_density(
        self, omega: Float[Array, "*batch D1"]
    ) -> Float[Array, "*batch"]:
        """Not available: the space-time spectral measure is singular.

        Raises:
            NotImplementedError: Always; sample frequencies instead.
        """
        raise NotImplementedError(
            "Advected has a singular spectral measure on space-time (it lives on "
            "the D-dimensional subspace omega_t = -v . omega_x); use "
            "sample_frequencies or RandomFourierFeatures."
        )


class RandomAdvectionRBF(AbstractPointwiseKernel):
    r"""The frozen-field RBF averaged over a Gaussian velocity.

    With $V \sim \mathcal N(\mu, \Sigma_V)$ independent of the field and
    $k_s(h) = \sigma^2 \exp(-\tfrac12 h^\top h / \ell^2)$,
    $h - V u \sim \mathcal N(h - \mu u, u^2 \Sigma_V)$, and the Gaussian
    expectation $\mathbb E_V[k_s(h - V u)]$ is

    $$
    C(h, u) = \sigma^2 \bigl|I + u^2 \Sigma_V / \ell^2\bigr|^{-1/2}
    \exp\Bigl(-\tfrac12 (h - \mu u)^\top
    (\ell^2 I + u^2 \Sigma_V)^{-1} (h - \mu u)\Bigr)
    $$

    (Cox & Isham, 1988; Schlather, 2010). Inputs are
    ``[space (D) | time (1)]``, $h = x - x'$, $u = t - t'$. It is PSD (an
    expectation of PSD kernels) and stationary in space-time. At $u = 0$
    it is the spatial RBF; as $\Sigma_V \to 0$ it is
    ``Advected(RBF(lengthscale, variance), velocity_mean)``. An uncertain
    wind spreads and damps the correlation downwind, the more the longer
    the time lag.

    Each pair takes a Cholesky factorisation of the ``D x D`` matrix
    $\ell^2 I + u^2 \Sigma_V$, cheap for ``D <= 3``.

    Attributes:
        velocity_mean: $\mu$, shape ``(D,)`` (a scalar is ``D = 1``).
        velocity_cov_tril: $L$ with $\Sigma_V = L L^\top$, shape
            ``(D, D)``; the Cholesky factor of $\Sigma_V$ (any square $L$
            gives a valid covariance).
        variance: $\sigma^2$.
        lengthscale: $\ell$, a scalar.

    Raises:
        ValueError: If the shapes of ``velocity_mean``,
            ``velocity_cov_tril`` and ``lengthscale`` disagree with the
            above, or the inputs do not have ``D + 1`` columns.

    Examples:
        >>> import jax.numpy as jnp
        >>> import kernellib as kl
        >>> mu = jnp.array([2.0, -1.0])
        >>> k = kl.RandomAdvectionRBF(mu, 0.5 * jnp.eye(2), lengthscale=0.5)
        >>> z = jnp.array([0.3, 0.1, 0.0])
        >>> downwind = z + jnp.concatenate([3.0 * mu, jnp.array([3.0])])
        >>> # |I + 9 (0.25 I) / 0.25|^(-1/2) = 1 / 10 at the mean displacement
        >>> round(float(k.pairwise(z, downwind)), 6)
        0.1
        >>> frozen = kl.Advected(kl.RBF(lengthscale=0.5), mu)
        >>> round(float(frozen.pairwise(z, downwind)), 6)
        1.0
    """

    velocity_mean: Float[Array, " D"] = eqx.field(converter=_as_1d)
    velocity_cov_tril: Float[Array, "D D"] = eqx.field(converter=jnp.asarray)
    variance: Float[Array, ""] = eqx.field(default=1.0, converter=jnp.asarray)
    lengthscale: Float[Array, ""] = eqx.field(default=1.0, converter=jnp.asarray)

    def __check_init__(self) -> None:
        if self.velocity_mean.ndim != 1:
            raise ValueError(
                "velocity_mean must be a vector of shape (D,), got "
                f"{self.velocity_mean.shape}."
            )
        d = self.velocity_mean.shape[0]
        if self.velocity_cov_tril.shape != (d, d):
            raise ValueError(
                f"velocity_cov_tril must have shape ({d}, {d}) to match "
                f"velocity_mean, got {self.velocity_cov_tril.shape}."
            )
        if jnp.ndim(self.lengthscale) != 0:
            raise ValueError(
                "lengthscale must be a scalar, got shape "
                f"{jnp.shape(self.lengthscale)}."
            )

    @property
    def space_dims(self) -> int:
        """``D``, the number of spatial columns (from ``velocity_mean``)."""
        return self.velocity_mean.shape[0]

    @property
    def is_stationary(self) -> bool:
        return True

    @property
    def velocity_cov(self) -> Float[Array, "D D"]:
        r"""$\Sigma_V = L L^\top$."""
        L = self.velocity_cov_tril
        return einsum(L, L, "i k, j k -> i j")

    def pairwise(
        self, x: Float[Array, " D1"], y: Float[Array, " D1"]
    ) -> Float[Array, ""]:
        d = self.space_dims
        if x.shape[-1] != d + 1:
            raise ValueError(
                f"RandomAdvectionRBF has D = {d} spatial dimensions and expects "
                f"[space | time] inputs with {d + 1} columns; got {x.shape[-1]}."
            )
        lag = x - y
        h, u = lag[:d], lag[d]
        r = h - self.velocity_mean * u
        ell2 = self.lengthscale**2
        M = ell2 * jnp.eye(d, dtype=r.dtype) + u**2 * self.velocity_cov
        L = jnp.linalg.cholesky(M)
        w = jsl.solve_triangular(L, r, lower=True)
        quad = reduce(w**2, "d ->", "sum")
        # |I + u² Σ / ell²|^{-1/2} = ell^D |M|^{-1/2}, |M|^{1/2} = prod diag L.
        log_norm = 0.5 * d * jnp.log(ell2) - reduce(
            jnp.log(jnp.diagonal(L)), "d ->", "sum"
        )
        return self.variance * jnp.exp(log_norm - 0.5 * quad)

    def diag(self, X: Float[Array, "N D1"]) -> Float[Array, " N"]:
        return self.variance * jnp.ones(X.shape[0], dtype=X.dtype)
