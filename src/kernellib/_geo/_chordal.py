r"""`Chordal`: any Euclidean kernel on ``(lon, lat)`` inputs, via R³.

Restricting a positive-definite kernel on R³ to a sphere keeps it positive
definite, since a Gram matrix on the sphere is a Gram matrix in R³. So every
Euclidean kernel of the **chordal** distance $2R\sin(\theta/2)$ is valid on
S², unlike an RBF, or a Matérn with $\nu > \tfrac12$, of the great-circle
distance (Gneiting 2013).
"""

from __future__ import annotations

import equinox as eqx
from jaxtyping import Array, Float

from kernellib._einx import rearrange
from kernellib._kernels._base import AbstractKernel, GramParts
from kernellib.functional._geo import lonlat_to_unit


__all__ = ["Chordal"]


class Chordal(AbstractKernel):
    r"""A Euclidean kernel on points of the sphere, $k_0(R\,u(x), R\,u(x'))$.

    Each ``(lon, lat)`` input is mapped to $R\,u(x) \in \mathbb{R}^3$, with
    $u(x)$ its unit vector, and ``kernel`` is evaluated there. For a
    stationary ``kernel`` the result is a function of the chordal distance
    $c = 2R\sin(\theta/2) \approx R\theta$ for small great-circle angles
    $\theta$, so the lengthscale is in the units of ``radius`` (kilometres
    with `EARTH_RADIUS_KM`) at local scales.

    It is positive definite for every positive-definite ``kernel`` (RBF,
    Matérn of any $\nu$, rational quadratic, sums and products): the Gram on
    points $x_i$ is the Gram of ``kernel`` on $R\,u(x_i)$. The smoothness of
    ``kernel`` carries over to the sphere; its spectrum differs from that of
    an intrinsic sphere Matérn, but both are valid priors.

    Feature maps take the 3-D points: fit `RandomFourierFeatures` or
    `OrthogonalRandomFeatures` to ``chordal.kernel`` on
    ``chordal.to_cartesian(X)`` and evaluate them on the same points.

    For space-time inputs ``(lon, lat, t)``, combine it with a kernel in
    time: ``Product(ActiveDims(Chordal(...), dims=(0, 1)),
    ActiveDims(Matern(...), dims=(2,)))``.

    Attributes:
        kernel: The kernel on R³, with lengthscales in units of ``radius``.
        radius: Sphere radius ``R``.
        degrees: Whether inputs are in degrees (else radians).

    Examples:
        >>> import jax.numpy as jnp
        >>> import kernellib as kl
        >>> X = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 90.0]])
        >>> k = kl.Chordal(
        ...     kl.Matern(nu=1.5, lengthscale=500.0), radius=kl.EARTH_RADIUS_KM
        ... )
        >>> k(X, X).shape
        (3, 3)
        >>> bool(
        ...     jnp.allclose(
        ...         k(X, X), k.kernel(k.to_cartesian(X), k.to_cartesian(X))
        ...     )
        ... )
        True

        Space-time inputs ``(lon, lat, t)``:

        >>> st = kl.Product(
        ...     kl.ActiveDims(k, dims=(0, 1)),
        ...     kl.ActiveDims(kl.Matern(nu=0.5, lengthscale=2.0), dims=(2,)),
        ... )
        >>> XT = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 1.0]])
        >>> st(XT, XT).shape
        (2, 2)
    """

    kernel: AbstractKernel
    radius: float = eqx.field(default=1.0, static=True)
    degrees: bool = eqx.field(default=True, static=True)

    def to_cartesian(self, X: Float[Array, "N 2"]) -> Float[Array, "N 3"]:
        """Points ``R·u(x)`` in R³ for ``(N, 2)`` ``(lon, lat)`` inputs."""
        return self.radius * lonlat_to_unit(X, degrees=self.degrees)

    def __call__(
        self, X1: Float[Array, "N1 2"], X2: Float[Array, "N2 2"]
    ) -> Float[Array, "N1 N2"]:
        return self.kernel(self.to_cartesian(X1), self.to_cartesian(X2))

    def diag(self, X: Float[Array, "N 2"]) -> Float[Array, " N"]:
        return self.kernel.diag(self.to_cartesian(X))

    def _gram_structure(self, X: Float[Array, "N 2"]) -> GramParts | None:
        return self.kernel._gram_structure(self.to_cartesian(X))

    def pairwise(
        self, x: Float[Array, " 2"], y: Float[Array, " 2"]
    ) -> Float[Array, ""]:
        u = rearrange(self.to_cartesian(rearrange(x, "d -> 1 d")), "1 d -> d")
        v = rearrange(self.to_cartesian(rearrange(y, "d -> 1 d")), "1 d -> d")
        return self.kernel.pairwise(u, v)

    @property
    def is_pointwise(self) -> bool:
        return self.kernel.is_pointwise
