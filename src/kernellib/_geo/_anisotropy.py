r"""Anisotropic stationary kernels: a linear map of the lag.

`LinearTransform` evaluates a stationary kernel on linearly transformed
inputs, $k(x, x') = k_0(A x, A x') = k_0(A(x - x'))$, and
`GeometricAnisotropy` builds $A$ from rotation angles and axis ratios: the
geostatistics model of a correlation ellipse (2-D) or ellipsoid (3-D).

Both stay stationary and keep the spectral side in closed form. If $k_0$ has
density $S_0$, then

$$
S_A(\omega) = |\det A|^{-1}\, S_0(A^{-\top}\omega),
$$

and frequencies sample as $\omega = A^\top \omega_0$ with
$\omega_0 \sim S_0$, so `RandomFourierFeatures`, `draw_rff_cosine_basis`
and `LaplaceEigenfunctionFeatures` accept them. `OrthogonalRandomFeatures`
and `FastFoodFeatures` draw only the *lengths* of isotropic frequencies, so
they refuse these kernels.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import ClassVar

import einx
import equinox as eqx
import jax.numpy as jnp
from jax.typing import DTypeLike
from jaxtyping import Array, Float, PRNGKeyArray

from kernellib._einx import einsum, rearrange
from kernellib._kernels._base import AbstractStationaryKernel, GramParts


__all__ = ["GeometricAnisotropy", "LinearTransform"]


class _AbstractLinearTransform(AbstractStationaryKernel):
    r"""``k(x, x') = k_0(A x, A x')`` for a stationary ``k_0``.

    The transform absorbs the base kernel's lengthscale: as an
    `AbstractStationaryKernel` the wrapper reports ``lengthscale = 1`` and
    the base variance, its profile is ``shape(||ℓ₀⁻¹ A (x - x')||²)``, and
    its *unit* frequencies are the full frequencies $A^\top \omega_0$. That
    is what lets the random Fourier feature paths, which divide unit
    frequencies by the lengthscale, use it unchanged.
    """

    kernel: AbstractStationaryKernel

    # ``lengthscale`` and ``variance`` are fields of AbstractStationaryKernel;
    # declaring them ClassVar here drops them from ``__init__`` so the
    # properties below provide them.
    lengthscale: ClassVar[Float[Array, ""]]
    variance: ClassVar[Float[Array, ""]]

    # The unit density is not a function of |ω| alone: maps that draw only
    # frequency lengths (orthogonal, FastFood) must refuse it.
    _radial_unit_spectrum: ClassVar[bool] = False

    def __check_init__(self) -> None:
        if not isinstance(self.kernel, AbstractStationaryKernel):
            raise TypeError(
                f"{type(self).__name__} needs a stationary kernel, got "
                f"{type(self.kernel).__name__}."
            )

    @abstractmethod
    def _matrix(self) -> Float[Array, "D D"]:
        """The transform ``A``, shape ``(D, D)``."""
        raise NotImplementedError

    @property
    def lengthscale(self) -> Float[Array, ""]:
        """``1``: the base lengthscale is part of the transform."""
        return jnp.ones((), dtype=jnp.result_type(self._matrix()))

    @property
    def variance(self) -> Float[Array, ""]:
        """The base kernel's variance."""
        return self.kernel.variance

    def _apply(self, X: Float[Array, "... D"]) -> Float[Array, "... D"]:
        A = self._matrix()
        if X.shape[-1] != A.shape[-1]:
            raise ValueError(
                f"{type(self).__name__} has a {A.shape[0]}x{A.shape[1]} transform; "
                f"got {X.shape[-1]}-dimensional inputs."
            )
        return einsum(A, X, "i j, ... j -> ... i")

    def shape(self, r2: Float[Array, ...]) -> Float[Array, ...]:
        return self.kernel.shape(r2)

    def pairwise(
        self, x: Float[Array, " D"], y: Float[Array, " D"]
    ) -> Float[Array, ""]:
        return self.kernel.pairwise(self._apply(x), self._apply(y))

    def __call__(
        self, X1: Float[Array, "N1 D"], X2: Float[Array, "N2 D"]
    ) -> Float[Array, "N1 N2"]:
        return self.kernel(self._apply(X1), self._apply(X2))

    def _gram_structure(self, X: Float[Array, "N D"]) -> GramParts | None:
        return self.kernel._gram_structure(self._apply(X))

    # -- spectral side ------------------------------------------------------

    def unit_spectral_density(
        self, omega_sq: Float[Array, ...], d: int
    ) -> Float[Array, ...]:
        """Not available: the density is not a function of ``|ω|²``.

        Raises:
            NotImplementedError: Always; use `spectral_density`.
        """
        raise NotImplementedError(
            f"{type(self).__name__} is anisotropic: its density is not a function "
            "of |omega|^2. Use spectral_density(omega) on frequency vectors."
        )

    def spectral_density(
        self, omega: Float[Array, "*batch D"]
    ) -> Float[Array, "*batch"]:
        r"""$S_A(\omega) = |\det A|^{-1} S_0(A^{-\top}\omega)$.

        Same convention as `AbstractStationaryKernel.spectral_density`.

        Raises:
            ValueError: If ``D`` does not match the transform.
            NotImplementedError: If the base kernel has no density.
        """
        omega = jnp.asarray(omega)
        A = self._matrix()
        if omega.shape[-1] != A.shape[0]:
            raise ValueError(
                f"{type(self).__name__} has a {A.shape[0]}x{A.shape[1]} transform; "
                f"got {omega.shape[-1]}-dimensional frequencies."
            )
        # (A^{-T} ω)_i = sum_j (A^{-1})_{ji} ω_j, for every frequency.
        v = einsum(jnp.linalg.inv(A), omega, "j i, ... j -> ... i")
        _, logabsdet = jnp.linalg.slogdet(A)
        return jnp.exp(-logabsdet) * self.kernel.spectral_density(v)

    def sample_frequencies(
        self,
        key: PRNGKeyArray,
        n: int,
        d: int,
        dtype: DTypeLike | None = None,
    ) -> Float[Array, "n d"]:
        r"""Draw ``n`` frequencies $\omega = A^\top \omega_0$, $\omega_0 \sim S_0$.

        The draws have density $S_A(\omega) / ((2\pi)^D \sigma^2)$.

        Raises:
            ValueError: If ``d`` does not match the transform.
            NotImplementedError: If the base kernel has no sampler.
        """
        return self.sample_unit_frequencies(key, (n, d), dtype)

    def sample_unit_frequencies(
        self,
        key: PRNGKeyArray,
        shape: tuple[int, ...],
        dtype: DTypeLike | None = None,
    ) -> Float[Array, ...]:
        r"""Draw $\omega = A^\top \omega_0$ with $\omega_0 \sim S_0$.

        These are the wrapper's frequencies in full: its reported
        lengthscale is ``1``, so `sample_frequencies` returns them as is.

        Raises:
            ValueError: If the last axis of ``shape`` does not match the
                transform, or an ARD base lengthscale does not.
            NotImplementedError: If the base kernel has no sampler.
        """
        A = self._matrix()
        d = shape[-1]
        if d != A.shape[0]:
            raise ValueError(
                f"{type(self).__name__} has a {A.shape[0]}x{A.shape[1]} transform; "
                f"got {d}-dimensional frequencies."
            )
        omega0 = self.kernel.sample_unit_frequencies(
            key, shape, dtype
        ) / self.kernel._lengthscale_vector(d)
        omega = einsum(A, omega0, "i j, ... i -> ... j")
        return omega if dtype is None else omega.astype(dtype)


class LinearTransform(_AbstractLinearTransform):
    r"""A stationary kernel of a linearly transformed lag.

    $$
    k(x, x') = k_0(A x, A x') = k_0\bigl(A (x - x')\bigr),
    $$

    for a stationary $k_0$ and a learnable square matrix $A$. A diagonal
    $A$ is an ARD rescaling; a full one also rotates and shears. The kernel
    stays stationary, its spectral density is
    $|\det A|^{-1} S_0(A^{-\top}\omega)$, and its frequencies are
    $A^\top\omega_0$, so `RandomFourierFeatures` accepts it.

    Attributes:
        kernel: The stationary base kernel ``k_0``.
        A: Transform, shape ``(D, D)``; must be invertible for the spectral
            density.

    Examples:
        >>> import jax.numpy as jnp
        >>> import kernellib as kl
        >>> A = jnp.array([[2.0, 0.0], [0.0, 0.5]])
        >>> k = kl.LinearTransform(kl.RBF(), A)
        >>> ard = kl.RBF(lengthscale=jnp.array([0.5, 2.0]))
        >>> X = jnp.array([[0.0, 0.0], [1.0, 2.0], [-1.0, 0.5]])
        >>> bool(jnp.allclose(k(X, X), ard(X, X)))
        True
    """

    A: Float[Array, "D D"] = eqx.field(converter=jnp.asarray)

    def __check_init__(self) -> None:
        if self.A.ndim != 2 or self.A.shape[0] != self.A.shape[1]:
            raise ValueError(f"A must be a square matrix, got shape {self.A.shape}.")

    def _matrix(self) -> Float[Array, "D D"]:
        return self.A


def _as_1d(x: float | Float[Array, ...]) -> Float[Array, " n"]:
    return jnp.atleast_1d(jnp.asarray(x))


def _rotation_x(t: Float[Array, ""]) -> Float[Array, "3 3"]:
    c, s = jnp.cos(t), jnp.sin(t)
    o, z = jnp.ones_like(t), jnp.zeros_like(t)
    return jnp.stack(
        [jnp.stack([o, z, z]), jnp.stack([z, c, -s]), jnp.stack([z, s, c])]
    )


def _rotation_y(t: Float[Array, ""]) -> Float[Array, "3 3"]:
    c, s = jnp.cos(t), jnp.sin(t)
    o, z = jnp.ones_like(t), jnp.zeros_like(t)
    return jnp.stack(
        [jnp.stack([c, z, s]), jnp.stack([z, o, z]), jnp.stack([-s, z, c])]
    )


def _rotation_z(t: Float[Array, ""]) -> Float[Array, "3 3"]:
    c, s = jnp.cos(t), jnp.sin(t)
    o, z = jnp.ones_like(t), jnp.zeros_like(t)
    return jnp.stack(
        [jnp.stack([c, -s, z]), jnp.stack([s, c, z]), jnp.stack([z, z, o])]
    )


def _rotation(angles: Float[Array, " n_angles"]) -> Float[Array, "D D"]:
    """The rotation ``R`` whose columns are the main axes (major first)."""
    if angles.shape[0] == 1:
        c, s = jnp.cos(angles[0]), jnp.sin(angles[0])
        return jnp.stack([jnp.stack([c, -s]), jnp.stack([s, c])])
    alpha, beta, gamma = angles[0], angles[1], angles[2]
    return _rotation_z(alpha) @ _rotation_y(beta) @ _rotation_x(gamma)


class GeometricAnisotropy(_AbstractLinearTransform):
    r"""A rotated, stretched stationary kernel: a correlation ellipse or
    ellipsoid.

    `LinearTransform` with $A = \operatorname{diag}(1, 1/a_1, \ldots)\,
    R^\top$, where the columns of the rotation $R$ are the main axes,
    major first. The base kernel's ``lengthscale`` is the range along the
    major axis, and the ratios $a_i \in (0, 1]$ (minor / major) shrink it
    along the others: the range along axis $i + 1$ is $a_i \ell$.

    - **2-D:** ``angles`` $= (\alpha,)$, ``ratios`` $= (a,)$. $\alpha$ is the
      counter-clockwise angle from the x-axis to the major axis, and
      $R(\alpha) = \begin{pmatrix}\cos\alpha & -\sin\alpha\\
      \sin\alpha & \cos\alpha\end{pmatrix}$.
    - **3-D:** ``angles`` $= (\alpha, \beta, \gamma)$, ``ratios``
      $= (a_1, a_2)$, and
      $R = R_z(\alpha) R_y(\beta) R_x(\gamma)$: yaw about $z$, then pitch
      about the new $y$, then roll about the new $x$ (intrinsic
      Tait-Bryan angles), each a right-handed rotation. This is the
      ordering of gstools' ``rotated_main_axes`` (Müller et al., 2022);
      Chilès & Delfiner (2012), §2.5.2.

    Angles are in radians (`from_angles` takes degrees). Ratio 1 makes the
    kernel isotropic and independent of the angles, and $\alpha$ and
    $\alpha + \pi$ give the
    same kernel. ``angles`` and ``ratios`` are differentiable leaves; the
    ratios are not range-checked, since they may be traced.

    Attributes:
        kernel: The stationary base kernel; its ``lengthscale`` is the
            major-axis range.
        angles: ``(1,)`` in 2-D or ``(3,)`` in 3-D, radians.
        ratios: ``(1,)`` in 2-D or ``(2,)`` in 3-D, minor / major.

    Raises:
        ValueError: Unless ``angles`` and ``ratios`` have the 2-D or 3-D
            sizes above.

    Examples:
        >>> import jax.numpy as jnp
        >>> import kernellib as kl
        >>> base = kl.Matern(nu=1.5, lengthscale=2.0)
        >>> k = kl.GeometricAnisotropy(base, angles=jnp.pi / 4, ratios=0.5)
        >>> major = jnp.array([[1.0, 1.0]]) / jnp.sqrt(2.0)  # unit vector at 45°
        >>> minor = jnp.array([[-1.0, 1.0]]) / jnp.sqrt(2.0)
        >>> o = jnp.zeros((1, 2))
        >>> # range 2 along the major axis, 0.5 * 2 = 1 along the minor one
        >>> bool(jnp.allclose(k(o, 2.0 * major), k(o, 1.0 * minor)))
        True
        >>> round(float(k(o, 2.0 * major)[0, 0]), 4)  # Matérn-3/2 at r = 1
        0.4834
    """

    angles: Float[Array, " n_angles"] = eqx.field(converter=_as_1d)
    ratios: Float[Array, " n_ratios"] = eqx.field(converter=_as_1d)

    def __check_init__(self) -> None:
        sizes = (self.angles.shape, self.ratios.shape)
        if sizes not in (((1,), (1,)), ((3,), (2,))):
            raise ValueError(
                "GeometricAnisotropy supports 2-D (angles (1,), ratios (1,)) and "
                "3-D (angles (3,), ratios (2,)); got angles of shape "
                f"{self.angles.shape} and ratios of shape {self.ratios.shape}."
            )

    @classmethod
    def from_angles(
        cls,
        kernel: AbstractStationaryKernel,
        angles: float | Float[Array, " n_angles"],
        ratios: float | Float[Array, " n_ratios"],
        *,
        degrees: bool = False,
    ) -> GeometricAnisotropy:
        """Build from angles in radians, or in degrees with ``degrees=True``.

        Examples:
            >>> import jax.numpy as jnp
            >>> import kernellib as kl
            >>> k = kl.GeometricAnisotropy.from_angles(
            ...     kl.RBF(), [30.0, 0.0, 0.0], [0.5, 0.2], degrees=True
            ... )
            >>> k.A.shape, bool(jnp.allclose(k.angles[0], jnp.pi / 6))
            ((3, 3), True)
        """
        angles = jnp.asarray(angles)
        if degrees:
            angles = jnp.deg2rad(angles)
        # ty reads the inherited lengthscale / variance as fields; the
        # ClassVar redeclaration above removes them from __init__.
        return cls(kernel=kernel, angles=angles, ratios=jnp.asarray(ratios))  # ty: ignore[missing-argument]

    @property
    def rotation(self) -> Float[Array, "D D"]:
        """``R``: its columns are the main axes, major first."""
        return _rotation(self.angles)

    @property
    def A(self) -> Float[Array, "D D"]:
        r"""$A = \operatorname{diag}(1, 1/a_1, \ldots)\, R^\top$."""
        Rt = rearrange(self.rotation, "i j -> j i")
        scale = jnp.concatenate([jnp.ones((1,), dtype=self.ratios.dtype), self.ratios])
        return einx.divide("i j, i -> i j", Rt, scale)

    def _matrix(self) -> Float[Array, "D D"]:
        return self.A
