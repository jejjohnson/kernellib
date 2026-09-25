r"""Laplace-eigenfunction (Hilbert-space GP) features for stationary kernels.

On a box $\Omega = [-L_1, L_1] \times \dots \times [-L_D, L_D]$ the
Dirichlet eigenfunctions of the Laplacian are products of sines with
per-axis frequencies $\omega_{j,d} = \pi j_d / (2 L_d)$. Solin & Särkkä
(2020) show that a stationary kernel restricted to the box is approximated by

$$
k(x, x') \approx \sum_j S(\omega_j)\, \phi_j(x)\, \phi_j(x'),
$$

with $S$ the kernel's spectral density. The basis is kernel-independent (it
is ``geonnax.basis.fourier_basis``); only the weights $S(\omega_j)$ carry the
hyperparameters, so they are cheap to differentiate.
"""

from __future__ import annotations

import dataclasses

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from geonnax.basis import fourier_basis, fourier_eigenvalues_1d
from jaxtyping import Array, Float

from kernellib._einx import rearrange
from kernellib._kernels import AbstractKernel, AbstractStationaryKernel
from kernellib._spectral._base import AbstractFeatureMap


__all__ = ["LaplaceEigenfunctionFeatures"]


class LaplaceEigenfunctionFeatures(AbstractFeatureMap):
    r"""Hilbert-space (HSGP) approximation of a stationary kernel on a box.

    $\phi_j(x) = \sqrt{S(\omega_j)}\, \prod_d \psi_{j_d}(x_d)$, with the
    $\psi$ the 1-D Dirichlet eigenfunctions on $[-L_d, L_d]$ and
    $\omega_j = (\pi j_d / 2 L_d)_d$ the matching frequency vector, so ARD
    lengthscales are handled exactly. There are ``prod(n_per_dim)``
    features. Unlike the random maps it is deterministic and needs a kernel
    with a closed-form `spectral_density` (`RBF`, `Matern`).

    The approximation is accurate inside the box and away from its edge;
    it is zero on the boundary. With ``L=None`` the half-widths are
    ``boundary_factor * max|X_d|`` over the fitting inputs, so centre the
    inputs first. The margin has to be wide relative to the lengthscale:
    Riutort-Mayol et al. (2023) recommend a factor of at least
    ``max(1.2, 3.2 * lengthscale / max|X_d|)`` for RBF, with enough basis
    functions to resolve the kernel on the wider box. Inputs outside the box
    are not rejected: the sines extrapolate periodically, which is wrong.

    Attributes:
        n_per_dim: Basis functions per input dimension, an ``int`` or one
            per dimension.
        L: Half-widths of the box, a ``float`` or one per dimension; ``None``
            to set them from the fitting inputs.
        boundary_factor: Multiplier on ``max|X_d|`` when ``L`` is ``None``.
        kernel: The fitted kernel, ``None`` before `fit`.
        half_widths: The resolved half-widths, ``None`` before `fit`.

    Examples:
        >>> import jax.numpy as jnp
        >>> import kernellib as kl
        >>> X = jnp.linspace(-1.0, 1.0, 50)[:, None]
        >>> k = kl.RBF(lengthscale=0.3)
        >>> lap = kl.LaplaceEigenfunctionFeatures(32, boundary_factor=2.5)
        >>> Phi = lap.fit(k, X)(X)
        >>> Phi.shape
        (50, 32)
        >>> bool(jnp.max(jnp.abs(Phi @ Phi.T - k(X, X))) < 1e-4)
        True
    """

    n_per_dim: int | tuple[int, ...] = eqx.field(static=True)
    L: float | tuple[float, ...] | None = eqx.field(default=None, static=True)
    boundary_factor: float = eqx.field(default=1.5, static=True)
    kernel: AbstractStationaryKernel | None = None
    half_widths: tuple[float, ...] | None = eqx.field(default=None, static=True)

    def __check_init__(self) -> None:
        sizes = (
            self.n_per_dim if isinstance(self.n_per_dim, tuple) else (self.n_per_dim,)
        )
        if any(m < 1 for m in sizes):
            raise ValueError(f"n_per_dim must be >= 1, got {self.n_per_dim}.")
        if self.boundary_factor <= 1.0:
            raise ValueError(
                f"boundary_factor must be > 1, got {self.boundary_factor}."
            )

    def fit(
        self, kernel: AbstractKernel, X: Float[Array, "N D"]
    ) -> LaplaceEigenfunctionFeatures:
        """Resolve the box for ``kernel`` on inputs like ``X``.

        Needs concrete ``X`` when ``L`` is ``None``: the half-widths are
        static, as the basis construction requires.

        Raises:
            NotImplementedError: If the kernel has no spectral density.
            ValueError: On a size mismatch between ``X`` and ``n_per_dim``,
                ``L`` or an ARD lengthscale, or a non-positive half-width.
        """
        if not isinstance(kernel, AbstractStationaryKernel):
            raise NotImplementedError(
                "LaplaceEigenfunctionFeatures needs a stationary kernel with a "
                f"spectral density; got {type(kernel).__name__}."
            )
        d = X.shape[-1]
        # Fail here rather than at the first call for kernels without a
        # density (RationalQuadratic, user subclasses without the hook).
        kernel.spectral_density(jnp.zeros((1, d), dtype=X.dtype))
        _per_dim(self.n_per_dim, d, "n_per_dim")
        if self.L is None:
            half_widths = tuple(
                float(v) for v in self.boundary_factor * np.max(np.abs(X), axis=0)
            )
        else:
            half_widths = tuple(float(v) for v in _per_dim(self.L, d, "L"))
        if any(h <= 0 for h in half_widths):
            raise ValueError(f"Half-widths must be positive, got {half_widths}.")
        return dataclasses.replace(self, kernel=kernel, half_widths=half_widths)

    @property
    def frequencies(self) -> Float[Array, "M D"]:
        """Frequency vectors ``omega_j``, in the basis' row-major order."""
        assert self.half_widths is not None
        d = len(self.half_widths)
        sizes = _per_dim(self.n_per_dim, d, "n_per_dim")
        axes = [
            jnp.sqrt(fourier_eigenvalues_1d(m, h, dtype=jnp.result_type(float)))
            for m, h in zip(sizes, self.half_widths, strict=True)
        ]
        grid = jnp.stack(jnp.meshgrid(*axes, indexing="ij"), axis=-1)
        return rearrange(grid, "... d -> (...) d")

    def features(self, X: Float[Array, "N D"]) -> Float[Array, "N M"]:
        assert self.kernel is not None and self.half_widths is not None
        if X.shape[-1] != len(self.half_widths):
            raise ValueError(
                f"The map was fitted on {len(self.half_widths)}-dimensional "
                f"inputs, got {X.shape[-1]}."
            )
        basis, _ = fourier_basis(X, self.n_per_dim, self.half_widths)
        S = self.kernel.spectral_density(self.frequencies.astype(X.dtype))
        return basis * jnp.sqrt(S)


def _per_dim(value, d: int, name: str) -> tuple:
    if isinstance(value, tuple | list):
        if len(value) != d:
            raise ValueError(
                f"{name} has {len(value)} entries for {d}-dimensional inputs."
            )
        return tuple(value)
    return (value,) * d
