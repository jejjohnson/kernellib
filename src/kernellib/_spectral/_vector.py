r"""Hilbert-space features for 2-D vector fields: divergence-free and curl-free.

The low-rank (HSGP-style) counterpart of the matrix-valued kernels
`CurlFree`, `DivergenceFree` and `Helmholtz`. Where those cost
$O((ND)^3)$, the feature maps here give an $(ND) \times M$ matrix whose
Gram $\Phi\Phi^\top$ approximates the kernel, so a GP on ``N`` drifter
velocities costs $O(NDM^2)$ through Woodbury.

**Construction.** On a box $[-L_1, L_1] \times [-L_2, L_2]$ let
$(\lambda_j, \varphi_j)$ be the eigenpairs of the Laplacian with Dirichlet
or Neumann boundary conditions (``geonnax.basis.fourier_basis``). For a
stationary scalar kernel $k$ with spectral density $S$, Solin & Särkkä
(2020) approximate

$$
k(x, x') \approx \sum_j S(\omega_j)\, \varphi_j(x)\, \varphi_j(x'),
\qquad \omega_j = \sqrt{\lambda_j}\ \text{(per axis, so ARD is exact)},
$$

i.e. the GP $\psi \approx \sum_j \sqrt{S_j}\, w_j \varphi_j$ with
$w_j \sim \mathcal{N}(0, 1)$. A linear operator $\mathcal{L}$ applies
term by term, so $\mathcal{L}\psi \approx \sum_j \sqrt{S_j}\, w_j\,
\mathcal{L}\varphi_j$ has the feature columns $\sqrt{S_j}\, \mathcal{L}\varphi_j$:

- **Divergence-free** ($\mathcal{L} = R\nabla$, the stream function),
  $R = \begin{pmatrix} 0 & 1 \\ -1 & 0 \end{pmatrix}$, so
  $F = (\partial_1 \Psi, -\partial_0 \Psi)$ and the columns are
  $\sqrt{S_j}\, R\nabla\varphi_j(x)$.
- **Curl-free** ($\mathcal{L} = \nabla$, the potential): columns
  $\sqrt{S_j}\, \nabla\varphi_j(x)$.

**Why the fields are exact.** Each $\varphi_j$ is a smooth function, and
$\nabla\cdot R\nabla\varphi = \partial_0\partial_1\varphi -
\partial_1\partial_0\varphi = 0$ and $\nabla\times\nabla\varphi = 0$ by the
symmetry of second derivatives. These hold for every column, hence for
every linear combination at any truncation $M$ and any box. The
approximation error is only in the covariance, never in the invariant.

**Convention.** Same as `kernellib.vector_inputs`: row $n D + i$ of the
feature matrix is component $i$ at location $n$ (``"n i m -> (n i) m"``)
and $R$ is the rotation used by `DivergenceFree`, so
$\Phi\Phi^\top \to$ ``DivergenceFree(k)(Xv, Xv)`` and
$\Phi\Phi^\top \to$ ``CurlFree(k)(Xv, Xv)`` with ``Xv = vector_inputs(X)``.
(``geonnax.basis.divfree_basis`` is the same basis with the opposite sign,
$-R\nabla\varphi_j$, which leaves $\Phi\Phi^\top$ unchanged.)

References:
    Solin & Särkkä (2020). Hilbert space methods for reduced-rank Gaussian
    process regression. Statistics and Computing 30. arXiv:1401.5508.
    Riutort-Mayol, Bürkner, Andersen, Solin & Vehtari (2023). Practical
    Hilbert space approximate Bayesian Gaussian processes for probabilistic
    programming. Statistics and Computing 33.
    Berlinghieri, Trippe, Burt, Chen, Terenin & Broderick (2023). Gaussian
    processes at the Helm(holtz): a more fluid model for ocean currents.
    ICML.
"""

from __future__ import annotations

import dataclasses
from typing import Literal

import einx
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from geonnax.basis import fourier_basis, fourier_eigenvalues_1d
from jaxtyping import Array, Float

from kernellib._einx import einsum, rearrange
from kernellib._kernels import AbstractKernel
from kernellib._kernels._compose import _spectral_components
from kernellib._spectral._base import AbstractFeatureMap
from kernellib._spectral._laplace import _density, _per_dim


__all__ = ["CurlFreeFeatures", "DivergenceFreeFeatures", "HelmholtzFeatures"]

_BC = Literal["dirichlet", "neumann"]

# Rotation of the gradient: R grad = (d_1, -d_0). Shared with `DivergenceFree`.
_ROTATION = ((0.0, 1.0), (-1.0, 0.0))


class _AbstractVectorFeatures(AbstractFeatureMap):
    """Shared box, eigenpair and spectral-weight plumbing (private)."""

    num_basis_per_dim: int | tuple[int, int] = eqx.field(static=True)
    L: float | tuple[float, float] | None = eqx.field(default=None, static=True)
    boundary_factor: float = eqx.field(default=1.5, static=True)
    boundary: _BC | tuple[_BC, _BC] = eqx.field(default="dirichlet", static=True)
    kernel: AbstractKernel | None = None
    half_widths: tuple[float, float] | None = eqx.field(default=None, static=True)

    def __check_init__(self) -> None:
        sizes = (
            self.num_basis_per_dim
            if isinstance(self.num_basis_per_dim, tuple)
            else (self.num_basis_per_dim,)
        )
        if any(m < 1 for m in sizes):
            raise ValueError(f"num_basis_per_dim must be >= 1, got {sizes}.")
        if self.boundary_factor <= 1.0:
            raise ValueError(
                f"boundary_factor must be > 1, got {self.boundary_factor}."
            )
        for b in _per_dim(self.boundary, 2, "boundary"):
            if b not in ("dirichlet", "neumann"):
                raise ValueError(
                    f"boundary must be 'dirichlet' or 'neumann', got {b!r}."
                )

    def fit(self, kernel: AbstractKernel, X: Float[Array, "N 2"]):
        """Resolve the box for the scalar ``kernel`` on inputs like ``X``.

        ``kernel`` is the kernel of the stream function (divergence-free) or
        the potential (curl-free). Needs concrete ``X`` when ``L`` is
        ``None``: the half-widths are static.

        Raises:
            NotImplementedError: If the kernel has no spectral density.
            ValueError: If ``X`` is not ``(N, 2)``, on a size mismatch of
                ``num_basis_per_dim``, ``L`` or an ARD lengthscale, or a
                non-positive half-width.
        """
        name = type(self).__name__
        if X.ndim != 2 or X.shape[-1] != 2:
            raise ValueError(f"{name} needs (N, 2) inputs; got shape {X.shape}.")
        if _spectral_components(kernel) is None:
            raise NotImplementedError(
                f"{name} needs a stationary kernel with a spectral density, or "
                f"a Scaled or Sum of them; got {type(kernel).__name__}."
            )
        # Fail here, not at the first call, for kernels without a density.
        _density(kernel, jnp.zeros((1, 2), dtype=X.dtype))
        _per_dim(self.num_basis_per_dim, 2, "num_basis_per_dim")
        if self.L is None:
            half_widths = tuple(
                float(v) for v in self.boundary_factor * einx.max("n d -> d", np.abs(X))
            )
        else:
            half_widths = tuple(float(v) for v in _per_dim(self.L, 2, "L"))
        if any(h <= 0 for h in half_widths):
            raise ValueError(f"Half-widths must be positive, got {half_widths}.")
        return dataclasses.replace(self, kernel=kernel, half_widths=half_widths)

    @property
    def frequencies(self) -> Float[Array, "M 2"]:
        """Frequency vectors ``omega_j = sqrt(lambda_j)``, per axis, row-major."""
        assert self.half_widths is not None
        sizes = _per_dim(self.num_basis_per_dim, 2, "num_basis_per_dim")
        bcs = _per_dim(self.boundary, 2, "boundary")
        axes = [
            jnp.sqrt(
                fourier_eigenvalues_1d(m, h, dtype=jnp.result_type(float), boundary=b)
            )
            for m, h, b in zip(sizes, self.half_widths, bcs, strict=True)
        ]
        grid = jnp.stack(jnp.meshgrid(*axes, indexing="ij"), axis=-1)
        return rearrange(grid, "... d -> (...) d")

    def _gradients(self, X: Float[Array, "N 2"]) -> Float[Array, "N M 2"]:
        """``grad phi_j(x)`` of the eigenfunctions by forward-mode autodiff."""
        assert self.half_widths is not None
        n, hw, bc = self.num_basis_per_dim, self.half_widths, self.boundary

        def phi(x: Float[Array, " 2"]) -> Float[Array, " M"]:
            basis, _ = fourier_basis(x[None], n, hw, boundary=bc)
            return basis[0]

        return jax.vmap(jax.jacfwd(phi))(X)

    def _check_inputs(self, X: Float[Array, "N 2"]) -> None:
        if X.ndim != 2 or X.shape[-1] != 2:
            raise ValueError(
                f"{type(self).__name__} needs (N, 2) inputs; got shape {X.shape}."
            )

    def _weighted_gradients(self, X: Float[Array, "N 2"]) -> Float[Array, "N M 2"]:
        self._check_inputs(X)
        assert self.kernel is not None
        S = _density(self.kernel, self.frequencies.astype(X.dtype))
        return einx.multiply("n m a, m -> n m a", self._gradients(X), jnp.sqrt(S))


class DivergenceFreeFeatures(_AbstractVectorFeatures):
    r"""Hilbert-space features of a divergence-free 2-D vector field.

    Columns $\sqrt{S_j}\, R\nabla\varphi_j(x)$ with
    $R = \begin{pmatrix} 0 & 1 \\ -1 & 0 \end{pmatrix}$, so every combination
    $F = \Phi w$ has $\partial_0 F_0 + \partial_1 F_1 = 0$ exactly, at any
    truncation. $\Phi\Phi^\top \approx$ ``DivergenceFree(k)`` on
    `vector_inputs` rows, inside the box and away from its edge.

    Math. With the stream-function GP
    $\Psi \approx \sum_j \sqrt{S(\omega_j)}\, w_j \varphi_j$ (Solin & Särkkä
    2020), $F = R\nabla\Psi$ and
    $\operatorname{cov}(F) \approx \sum_j S_j\, R\nabla\varphi_j
    (R\nabla\varphi_j)^\top = R\,[\nabla\nabla'^\top k]\,R^\top$, the
    kernel of `DivergenceFree`. The divergence of each column,
    $\partial_0\partial_1\varphi_j - \partial_1\partial_0\varphi_j$, vanishes
    by symmetry of second derivatives.

    Pseudocode::

        omega[j]  = (sqrt(lam_a), sqrt(lam_b))     # eigen-frequencies per axis
        S[j]      = k.spectral_density(omega[j])
        G[n,j,:]  = grad phi_j(x_n)                # jax.jacfwd of fourier_basis
        F[n,j,i]  = sqrt(S[j]) * (R @ G[n,j,:])[i] # R = [[0, 1], [-1, 0]]
        Phi       = rearrange(F, "n m i -> (n i) m")

    Numerics. The basis vanishes (Dirichlet) or has zero normal derivative
    (Neumann) on the box edge, so the covariance is accurate only in the
    interior. Choose the box wider than the data: ``boundary_factor`` of at
    least ``max(1.2, 3.2 * lengthscale / max|X_d|)`` for an RBF
    (Riutort-Mayol et al. 2023), and enough basis functions to resolve
    the lengthscale on that wider box, roughly
    ``num_basis_per_dim >= 1.75 * L / lengthscale`` per axis. The kernel is
    the one of the *stream function*, not of the velocity. Inputs are plain
    positions ``(N, 2)``, not `vector_inputs` rows; inputs outside the box
    are not rejected.

    Attributes:
        num_basis_per_dim: Basis functions per axis, an ``int`` or a pair.
            ``M = prod`` of them.
        L: Half-widths of the box ``[-L, L]``, a ``float`` or a pair; ``None``
            to set them to ``boundary_factor * max|X_d|`` from the fitting
            inputs (centre the inputs first).
        boundary_factor: Multiplier on ``max|X_d|`` when ``L`` is ``None``.
        boundary: ``"dirichlet"`` or ``"neumann"``, one value or one per axis.
        kernel: The fitted scalar kernel, ``None`` before `fit`.
        half_widths: The resolved half-widths, ``None`` before `fit`.

    Examples:
        >>> import jax
        >>> import jax.numpy as jnp
        >>> import kernellib as kl
        >>> X = jax.random.uniform(jax.random.key(0), (20, 2), minval=-1, maxval=1)
        >>> k = kl.RBF(lengthscale=0.5)
        >>> feat = kl.DivergenceFreeFeatures(10, L=3.0).fit(k, X)
        >>> feat(X).shape  # (N * 2, M), rows (n i)
        (40, 100)
        >>> J = jax.jacfwd(lambda x: feat(x[None]))(X[0])  # (2, M, 2)
        >>> bool(jnp.max(jnp.abs(J[0, :, 0] + J[1, :, 1])) < 1e-10)
        True
    """

    def features(self, X: Float[Array, "N 2"]) -> Float[Array, "N2 M"]:
        grads = self._weighted_gradients(X)
        R = jnp.asarray(_ROTATION, dtype=grads.dtype)
        return rearrange(einsum(R, grads, "i a, n m a -> n m i"), "n m i -> (n i) m")


class CurlFreeFeatures(_AbstractVectorFeatures):
    r"""Hilbert-space features of a curl-free 2-D vector field.

    Columns $\sqrt{S_j}\, \nabla\varphi_j(x)$, so every combination
    $F = \Phi w$ has $\partial_0 F_1 - \partial_1 F_0 = 0$ exactly.
    $\Phi\Phi^\top \approx$ ``CurlFree(k)`` on `vector_inputs` rows, inside
    the box and away from its edge.

    Math. With the potential GP
    $\Phi_\mathrm{pot} \approx \sum_j \sqrt{S(\omega_j)}\, w_j \varphi_j$,
    $F = \nabla\Phi_\mathrm{pot}$ has
    $\operatorname{cov}(F) \approx \sum_j S_j \nabla\varphi_j
    \nabla\varphi_j^\top = \nabla\nabla'^\top k$, the kernel of `CurlFree`.
    The scalar curl of each column is $\partial_0\partial_1\varphi_j -
    \partial_1\partial_0\varphi_j = 0$.

    Pseudocode::

        omega[j] = (sqrt(lam_a), sqrt(lam_b))
        S[j]     = k.spectral_density(omega[j])
        G[n,j,:] = grad phi_j(x_n)                 # jax.jacfwd of fourier_basis
        Phi      = rearrange(sqrt(S) * G, "n m i -> (n i) m")

    Numerics. As for `DivergenceFreeFeatures`: widen the box past the data
    (``boundary_factor``) and use enough basis functions for the lengthscale.
    The kernel is the one of the *potential*. With Neumann boundaries the
    constant mode is included and has zero gradient, so its column is zero.

    Attributes:
        num_basis_per_dim: Basis functions per axis, an ``int`` or a pair.
        L: Half-widths of the box, a ``float`` or a pair, or ``None``.
        boundary_factor: Multiplier on ``max|X_d|`` when ``L`` is ``None``.
        boundary: ``"dirichlet"`` or ``"neumann"``, one value or one per axis.
        kernel: The fitted scalar kernel, ``None`` before `fit`.
        half_widths: The resolved half-widths, ``None`` before `fit`.

    Examples:
        >>> import jax
        >>> import jax.numpy as jnp
        >>> import kernellib as kl
        >>> X = jax.random.uniform(jax.random.key(0), (20, 2), minval=-1, maxval=1)
        >>> k = kl.RBF(lengthscale=0.5)
        >>> feat = kl.CurlFreeFeatures(10, L=3.0).fit(k, X)
        >>> feat(X).shape
        (40, 100)
        >>> J = jax.jacfwd(lambda x: feat(x[None]))(X[0])  # (2, M, 2)
        >>> bool(jnp.max(jnp.abs(J[1, :, 0] - J[0, :, 1])) < 1e-10)
        True
    """

    def features(self, X: Float[Array, "N 2"]) -> Float[Array, "N2 M"]:
        return rearrange(self._weighted_gradients(X), "n m i -> (n i) m")


class HelmholtzFeatures(AbstractFeatureMap):
    r"""Helmholtz features: curl-free plus divergence-free columns.

    The columns of `CurlFreeFeatures` (potential) followed by those of
    `DivergenceFreeFeatures` (stream function), so
    $\Phi\Phi^\top \approx$ ``Helmholtz(potential, stream)``
    (Berlinghieri et al. 2023) and a combination
    $F = \nabla\Phi_\mathrm{pot} + R\nabla\Psi$ has the Helmholtz
    decomposition by construction. ``M = M_Φ + M_Ψ`` columns.

    Pseudocode::

        Phi = hstack([CurlFreeFeatures(X), DivergenceFreeFeatures(X)])

    Numerics. See `DivergenceFreeFeatures`; the two parts have separate
    kernels, box and basis size.

    Attributes:
        potential: The curl-free part, unfitted or fitted.
        stream: The divergence-free part, unfitted or fitted.

    Examples:
        >>> import jax
        >>> import kernellib as kl
        >>> X = jax.random.uniform(jax.random.key(0), (20, 2), minval=-1, maxval=1)
        >>> feat = kl.HelmholtzFeatures(
        ...     kl.CurlFreeFeatures(6, L=3.0), kl.DivergenceFreeFeatures(8, L=3.0)
        ... ).fit(kl.RBF(lengthscale=0.5), X, stream_kernel=kl.RBF(lengthscale=0.8))
        >>> feat(X).shape  # M = 36 + 64
        (40, 100)
    """

    potential: CurlFreeFeatures
    stream: DivergenceFreeFeatures

    @property
    def is_fitted(self) -> bool:
        return self.potential.is_fitted and self.stream.is_fitted

    def fit(
        self,
        kernel: AbstractKernel,
        X: Float[Array, "N 2"],
        stream_kernel: AbstractKernel | None = None,
    ) -> HelmholtzFeatures:
        """Fit both parts.

        Args:
            kernel: The potential kernel (and the stream kernel if
                ``stream_kernel`` is ``None``).
            X: Inputs ``(N, 2)``.
            stream_kernel: The stream-function kernel, if different.
        """
        return dataclasses.replace(
            self,
            potential=self.potential.fit(kernel, X),
            stream=self.stream.fit(
                kernel if stream_kernel is None else stream_kernel, X
            ),
        )

    def features(self, X: Float[Array, "N 2"]) -> Float[Array, "N2 M"]:
        return einx.id("r a, r b -> r (a + b)", self.potential(X), self.stream(X))
