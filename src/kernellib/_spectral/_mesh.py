r"""Mesh Laplace features: FEM eigenvectors for kernels on a triangulated surface.

The mesh kernels `MeshMatern` and `MeshHeat` are finite eigen-expansions,
$k(v, w) = \sum_{j \le M} s_j\, \varphi_j(v)\, \varphi_j(w)$, so the features
$\phi_j(x) = \sqrt{s_j}\, \varphi_j(x)$ reproduce them exactly. This is the
mesh analogue of `LaplaceEigenfunctionFeatures` (a box) and
`SphericalHarmonicFeatures` (the sphere): the basis depends only on the mesh,
and only the weights $s_j$ carry the hyperparameters.
"""

from __future__ import annotations

import dataclasses
from typing import Literal

import einx
import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float

from kernellib._geo._mesh import AbstractMeshKernel
from kernellib._kernels import AbstractKernel
from kernellib._spectral._base import AbstractFeatureMap


__all__ = ["MeshLaplaceFeatures"]


class MeshLaplaceFeatures(AbstractFeatureMap):
    r"""FEM Laplace-Beltrami eigenvector features of a mesh kernel.

    $\phi_j(x) = \sqrt{s_j}\, \varphi_j(x)$ for the first $M$ eigenpairs of
    a `MeshMatern` or `MeshHeat` kernel, with $s_j$ its `spectral_weights`.
    At the kernel's own ``n_components`` the features reproduce its Gram
    exactly; fewer drop the highest modes (their weight is the truncation
    error). Features cost $O(NM)$ and turn a GP on the mesh into Bayesian
    linear regression with $M$ weights.

    **Inputs.** With ``inputs="vertices"``, ``(N, 1)`` vertex indices: the
    features are rows of the eigenvector matrix, and may be traced. With
    ``inputs="points"``, ``(N, D)`` points on the mesh: the eigenvectors are
    interpolated barycentrically (`gaussx.fem_projector`), so the features
    are a linear map of the vertex features and the kernel stays positive
    definite; point location runs on the host, so the points must be
    concrete.

    Gradients reach the kernel's ``lengthscale`` and ``variance`` through the
    weights; the eigenbasis is fixed by the mesh.

    Attributes:
        n_components: Number $M$ of features; ``None`` takes the kernel's
            ``n_components`` at `fit`.
        inputs: ``"vertices"`` or ``"points"``.
        kernel: The fitted kernel, ``None`` before `fit`.

    Examples:
        >>> import einx
        >>> import jax.numpy as jnp
        >>> import kernellib as kl
        >>> from geonnax.basis import icosphere
        >>> V, T = icosphere(2)
        >>> k = kl.MeshMatern(V, T, lengthscale=0.5, nu=1.5, n_components=100)
        >>> X = einx.id("n -> n 1", jnp.arange(162.0))
        >>> Phi = kl.MeshLaplaceFeatures().fit(k, X)(X)
        >>> Phi.shape
        (162, 100)
        >>> err = einx.dot("n f, m f -> n m", Phi, Phi) - k(X, X)
        >>> bool(jnp.max(jnp.abs(err)) < 1e-5)
        True

        Off-vertex points on the mesh, here the face centroids:

        >>> P = einx.mean("t [3] d -> t d", V[T[:4]])
        >>> mf = kl.MeshLaplaceFeatures(n_components=50, inputs="points").fit(k, P)
        >>> mf(P).shape
        (4, 50)
    """

    n_components: int | None = eqx.field(default=None, static=True)
    inputs: Literal["vertices", "points"] = eqx.field(default="vertices", static=True)
    kernel: AbstractMeshKernel | None = None

    def __check_init__(self) -> None:
        if self.n_components is not None and (
            not isinstance(self.n_components, int) or self.n_components < 1
        ):
            raise ValueError(
                f"n_components must be a positive int; got {self.n_components!r}."
            )
        if self.inputs not in ("vertices", "points"):
            raise ValueError(
                f"inputs must be 'vertices' or 'points'; got {self.inputs!r}."
            )

    def fit(
        self, kernel: AbstractKernel, X: Float[Array, "N D"]
    ) -> MeshLaplaceFeatures:
        """Resolve the feature count for ``kernel``; ``X`` fixes the input layout.

        Raises:
            NotImplementedError: If ``kernel`` is not a mesh kernel.
            ValueError: If ``X`` does not match ``inputs``, or
                ``n_components`` exceeds the kernel's.
        """
        if not isinstance(kernel, AbstractMeshKernel):
            raise NotImplementedError(
                "MeshLaplaceFeatures needs a mesh kernel (MeshMatern, MeshHeat); "
                f"got {type(kernel).__name__}."
            )
        _check_inputs(jnp.asarray(X), self.inputs, kernel.vertices.shape[1])
        n = kernel.n_components if self.n_components is None else self.n_components
        if n > kernel.n_components:
            raise ValueError(
                f"n_components={n} exceeds the kernel's "
                f"n_components={kernel.n_components}."
            )
        return dataclasses.replace(self, kernel=kernel, n_components=n)

    @property
    def n_features(self) -> int:
        """Number of features ``M``."""
        if self.n_components is None:
            raise RuntimeError("n_features is unknown until the map is fitted.")
        return self.n_components

    def features(self, X: Float[Array, "N D"]) -> Float[Array, "N M"]:
        assert self.kernel is not None and self.n_components is not None
        X = jnp.asarray(X)
        _check_inputs(X, self.inputs, self.kernel.vertices.shape[1])
        m = self.n_components
        basis = self.kernel.eigenfunctions(X)[:, :m]
        weights = self.kernel.spectral_weights()[:m]
        # sqrt has an infinite derivative at 0 (an underflowed heat weight);
        # route those through a safe value so gradients stay finite.
        positive = weights > 0
        scale = jnp.where(positive, jnp.sqrt(jnp.where(positive, weights, 1.0)), 0.0)
        return einx.multiply("n m, m -> n m", basis, scale.astype(basis.dtype))


def _check_inputs(X: Float[Array, "N D"], inputs: str, dim: int) -> None:
    width = 1 if inputs == "vertices" else dim
    if X.ndim != 2 or X.shape[1] != width:
        expected = "(N, 1) vertex indices" if width == 1 else f"(N, {dim}) points"
        raise ValueError(
            f"MeshLaplaceFeatures(inputs={inputs!r}) takes {expected}; got shape "
            f"{X.shape}."
        )
