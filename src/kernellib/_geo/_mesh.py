r"""Matérn and heat kernels on a triangulated surface, from FEM eigenpairs.

On a triangle mesh of a domain $\mathcal{M}$ (a planar region bounded by a
coastline, a lake, an icosphere, terrain), the P1 finite-element
discretisation of the Laplace-Beltrami operator is the generalised
eigenproblem

$$
G \varphi = \lambda\, C \varphi,
$$

with $G$ the stiffness and $C$ the lumped (diagonal) mass matrix of
`gaussx.fem_matrices`. Symmetrising, $S = C^{-1/2} G C^{-1/2}$,
$S\psi = \lambda\psi$, $\varphi = C^{-1/2}\psi$, gives eigenvectors with
$\varphi^\top C \varphi = I$: an $L^2(\mathcal{M})$-orthonormal basis.
Natural (Neumann) boundary conditions on open boundaries come for free.

A kernel that is a function $\Phi \ge 0$ of the Laplacian is then

$$
k(v, w) = \frac{\sigma^2}{Z} \sum_{j \le M} \Phi(\lambda_j)\,
\varphi_j(v)\, \varphi_j(w), \qquad
Z = \frac{1}{A} \sum_{j \le M} \Phi(\lambda_j), \quad A = \operatorname{tr} C,
$$

so the area-weighted mean of the marginal variance is $\sigma^2$ (each
$\varphi_j$ has $\sum_v C_{vv} \varphi_j(v)^2 = 1$). Off the vertices,
$\varphi(x) = P(x)\, \varphi$ with $P$ the barycentric weights of
`gaussx.fem_projector`, a linear map of the features, so still positive
definite.

References:
    Borovitskiy, V., Terenin, A., Mostowsky, P. & Deisenroth, M. P. (2020).
    Matérn Gaussian processes on Riemannian manifolds. *NeurIPS*, §4.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Literal

import einx
import equinox as eqx
import gaussx as gx
import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
import numpy as np
from jax.experimental import sparse as jsparse
from jaxtyping import Array, ArrayLike, Float, Int, PRNGKeyArray

from kernellib._einx import einsum
from kernellib._graph._eigpairs import _fix_signs, _lanczos_until_converged
from kernellib._kernels._base import AbstractKernel, GramParts


__all__ = ["AbstractMeshKernel", "MeshHeat", "MeshMatern"]

MeshEigenMethod = Literal["dense", "lanczos"]

_LANCZOS_HINT = (
    "Use method='dense' for a mesh of up to a few thousand vertices, or a "
    "larger oversample / max_oversample."
)


class AbstractMeshKernel(AbstractKernel):
    r"""A kernel on a triangle mesh given by its Laplace-Beltrami spectrum.

    $k(v, w) = \sum_{j \le M} s_j\, \varphi_j(v)\, \varphi_j(w)$ with the FEM
    eigenpairs $(\lambda_j, \varphi_j)$ of the mesh (module docstring) and the
    weights $s_j = \sigma^2 \Phi(\lambda_j) / Z$ of `spectral_weights`.
    Subclasses implement `spectrum`, which only matters up to a constant
    factor (it cancels in $Z$).

    The eigenpairs are computed once, when the kernel is built, from the
    concrete mesh: ``method="dense"`` runs `jnp.linalg.eigh` on the
    $V \times V$ matrix $S$ ($O(V^3)$, fine up to a few thousand vertices);
    ``method="lanczos"`` runs Lanczos on $cI - S$ (matrix-free in the sparse
    stiffness, residual-checked, restarted with a larger Krylov space if it
    has not converged, as `laplacian_eigpairs` does). Changing the
    lengthscale or variance never re-solves; `eqx.tree_at` on them is cheap.
    Gradients flow to ``lengthscale`` and ``variance`` only: the eigenbasis
    is a constant of the kernel, so to differentiate with respect to the
    vertex positions build the kernel inside the differentiated function.

    **Inputs.** ``(N, 1)`` vertex indices (integers, or floats holding
    integers), or ``(N, D)`` points on the mesh (``D`` as ``vertices``),
    which are interpolated barycentrically with `gaussx.fem_projector`.
    Point location runs on the host, so points must be concrete (no ``jit``
    over them); vertex indices may be traced.

    **Accuracy.** The FEM eigenvalues converge as $O(h^2)$ in the mesh width
    $h$, fastest for the smooth low modes, so a moderate mesh resolves a
    lengthscale of a few triangle widths. The truncation at $M$ eigenpairs
    drops $\sum_{j > M} \Phi(\lambda_j)$; with $M$ inside a degenerate
    eigenspace (a sphere's $2l + 1$ harmonics of degree $l$) the kernel is
    still positive semidefinite but not exactly symmetric under the
    domain's symmetries.

    Attributes:
        vertices: Vertex coordinates, ``(V, 2)`` for a planar mesh or
            ``(V, 3)`` for a surface.
        triangles: Vertex indices of each triangle, ``(T, 3)``. Every vertex
            must belong to a triangle.
        lengthscale: Scalar lengthscale $\ell$, in units of the coordinates.
        variance: Scalar signal variance $\sigma^2$.
        n_components: Number $M$ of eigenpairs kept (static), at most $V$.
        method: ``"dense"`` or ``"lanczos"`` (static).
        eigenvalues: The $M$ smallest FEM eigenvalues, ascending.
        eigenvectors: ``(V, M)`` $C$-orthonormal eigenvectors $\varphi$.
        mass: ``(V,)`` lumped vertex areas, the diagonal of $C$.
    """

    vertices: Float[Array, "V D"]
    triangles: Int[Array, "T 3"]
    lengthscale: Float[Array, ""]
    variance: Float[Array, ""]
    n_components: int = eqx.field(static=True)
    method: MeshEigenMethod = eqx.field(static=True)
    eigenvalues: Float[Array, " M"]
    eigenvectors: Float[Array, "V M"]
    mass: Float[Array, " V"]

    def _build(
        self,
        vertices: Float[ArrayLike, "V D"],
        triangles: Int[ArrayLike, "T 3"],
        lengthscale: float | Float[Array, ""],
        variance: float | Float[Array, ""],
        n_components: int,
        method: MeshEigenMethod,
        key: PRNGKeyArray | None,
        oversample: int,
        max_oversample: int,
    ) -> None:
        """Set the fields and solve the eigenproblem (called from ``__init__``)."""
        vertices = jnp.asarray(vertices)
        if not jnp.issubdtype(vertices.dtype, jnp.floating):
            vertices = vertices.astype(jnp.result_type(float))
        tri = np.asarray(triangles)
        if tri.ndim != 2 or tri.shape[1] != 3:
            raise ValueError(f"triangles must have shape (T, 3); got {tri.shape}.")
        if vertices.ndim != 2 or vertices.shape[1] not in (2, 3):
            raise ValueError(
                f"vertices must have shape (V, 2) or (V, 3); got {vertices.shape}."
            )
        if not isinstance(n_components, int) or n_components < 1:
            raise ValueError(
                f"n_components must be a positive int; got {n_components!r}."
            )
        if method not in ("dense", "lanczos"):
            raise ValueError(f"method must be 'dense' or 'lanczos'; got {method!r}.")
        n_vertices = vertices.shape[0]
        if not np.isin(np.arange(n_vertices), tri).all():
            raise ValueError(
                "Every vertex must belong to a triangle: an unused vertex has no "
                "area, so the mass matrix is singular. Drop unused vertices first."
            )
        self.vertices = vertices
        self.triangles = jnp.asarray(tri)
        self.lengthscale = jnp.asarray(lengthscale)
        self.variance = jnp.asarray(variance)
        self.n_components = min(n_components, n_vertices)
        self.method = method
        lam, phi, mass = _mesh_eigenpairs(
            vertices,
            tri,
            self.n_components,
            method=method,
            key=key,
            oversample=oversample,
            max_oversample=max_oversample,
        )
        self.eigenvalues = lam
        self.eigenvectors = phi
        self.mass = mass

    @abstractmethod
    def spectrum(self, lam: Float[Array, " M"]) -> Float[Array, " M"]:
        r"""Unnormalised spectrum $\Phi(\lambda) \ge 0$ at eigenvalues ``lam``."""
        raise NotImplementedError

    def spectral_weights(self) -> Float[Array, " M"]:
        r"""Per-eigenpair weights $s_j = \sigma^2 \Phi(\lambda_j) / Z$.

        $Z = \sum_i \Phi(\lambda_i) / A$ with $A = \operatorname{tr} C$. The
        kernel is $\sum_j s_j \varphi_j(v) \varphi_j(w)$, and
        $\sum_j s_j / A = \sigma^2$ is its area-weighted mean variance.
        """
        phi = self.spectrum(self.eigenvalues)
        return self.variance * phi * jnp.sum(self.mass) / jnp.sum(phi)

    def eigenfunctions(self, X: Float[Array, "N D"]) -> Float[Array, "N M"]:
        r"""The eigenvectors $\varphi_j$ at vertex indices or points.

        Args:
            X: ``(N, 1)`` vertex indices, or ``(N, D)`` points on the mesh.

        Returns:
            ``(N, M)``: rows of $\varphi$, or their barycentric interpolants.

        Raises:
            ValueError: If ``X`` is neither ``(N, 1)`` nor ``(N, D)``, or a
                point is not on the mesh.
        """
        X = jnp.asarray(X)
        D = self.vertices.shape[1]
        if X.ndim != 2 or X.shape[1] not in (1, D):
            raise ValueError(
                f"{type(self).__name__} takes (N, 1) vertex indices or (N, {D}) "
                f"points; got shape {X.shape}."
            )
        if X.shape[1] == 1:
            index = X[:, 0]
            if jnp.issubdtype(index.dtype, jnp.floating):
                index = jnp.round(index)
            return self.eigenvectors[index.astype(jnp.int32)]
        P = gx.fem_projector(self.vertices, np.asarray(self.triangles), X)
        return P.to_bcoo() @ self.eigenvectors

    def __call__(
        self, X1: Float[Array, "N1 D"], X2: Float[Array, "N2 D"]
    ) -> Float[Array, "N1 N2"]:
        B1 = einx.multiply(
            "n m, m -> n m", self.eigenfunctions(X1), self.spectral_weights()
        )
        return einsum(B1, self.eigenfunctions(X2), "n m, k m -> n k")

    def diag(self, X: Float[Array, "N D"]) -> Float[Array, " N"]:
        return einx.dot(
            "n m, m -> n", self.eigenfunctions(X) ** 2, self.spectral_weights()
        )

    def _gram_structure(self, X: Float[Array, "N D"]) -> GramParts:
        return GramParts(None, self.eigenfunctions(X), self.spectral_weights())


class MeshMatern(AbstractMeshKernel):
    r"""Matérn kernel on a triangulated surface (Borovitskiy et al. 2020).

    The FEM version of `SphereMatern` for any mesh: a coastline-bounded
    basin, a lake, an icosphere, terrain. The spectrum of the Matérn SPDE on
    a 2-D manifold is

    $$
    \Phi(\lambda) = \Big(\frac{2\nu}{\ell^2} + \lambda\Big)^{-\nu - 1},
    $$

    computed as $(1 + \ell^2\lambda / 2\nu)^{-(\nu + 1)}$ (the constant
    cancels in the normalisation). Correlation follows the intrinsic
    geometry of the mesh: two points on either side of a peninsula, close in
    the plane but far along the water, are weakly correlated, which no
    Euclidean kernel can express. See `AbstractMeshKernel` for the inputs,
    the eigen-solve and the normalisation.

    Args:
        vertices: ``(V, 2)`` or ``(V, 3)`` vertex coordinates.
        triangles: ``(T, 3)`` vertex indices of the triangles.
        lengthscale: Scalar lengthscale $\ell$.
        nu: Smoothness $\nu > 0$.
        variance: Scalar signal variance $\sigma^2$.
        n_components: Eigenpairs kept, capped at $V$.
        method: ``"dense"`` (`jnp.linalg.eigh`) or ``"lanczos"``.
        key: Lanczos start vector; ``None`` uses ``jax.random.key(0)``.
        oversample: Extra Krylov dimension of the first Lanczos run.
        max_oversample: Largest extra Krylov dimension a Lanczos restart
            tries.

    Raises:
        ValueError: On malformed ``vertices`` / ``triangles``, an unused
            vertex, a bad ``n_components``, ``method`` or ``nu``.
        RuntimeError: If Lanczos has not converged.

    Examples:
        >>> import einx
        >>> import jax.numpy as jnp
        >>> import kernellib as kl
        >>> from geonnax.basis import icosphere
        >>> V, T = icosphere(2)  # 162 vertices on the unit sphere
        >>> k = kl.MeshMatern(V, T, lengthscale=0.5, nu=1.5, n_components=100)
        >>> [round(float(v), 1) for v in k.eigenvalues[:5]]  # l(l + 1), to O(h²)
        [0.0, 2.0, 2.0, 2.0, 5.9]
        >>> X = einx.id("n -> n 1", jnp.arange(162.0))  # vertex indices
        >>> K = k(X, X)
        >>> K.shape
        (162, 162)
        >>> round(float(jnp.sum(k.mass * jnp.diag(K)) / jnp.sum(k.mass)), 6)
        1.0
    """

    nu: float = eqx.field(static=True)

    def __init__(
        self,
        vertices: Float[ArrayLike, "V D"],
        triangles: Int[ArrayLike, "T 3"],
        lengthscale: float | Float[Array, ""] = 1.0,
        nu: float = 1.5,
        variance: float | Float[Array, ""] = 1.0,
        *,
        n_components: int = 200,
        method: MeshEigenMethod = "dense",
        key: PRNGKeyArray | None = None,
        oversample: int = 200,
        max_oversample: int = 1600,
    ) -> None:
        if not nu > 0:
            raise ValueError(f"nu must be positive; got {nu!r}.")
        self.nu = nu
        self._build(
            vertices,
            triangles,
            lengthscale,
            variance,
            n_components,
            method,
            key,
            oversample,
            max_oversample,
        )

    def spectrum(self, lam: Float[Array, " M"]) -> Float[Array, " M"]:
        x = self.lengthscale**2 * lam / (2.0 * self.nu)
        return (1.0 + x) ** (-(self.nu + 1.0))


class MeshHeat(AbstractMeshKernel):
    r"""Heat (diffusion) kernel on a triangulated surface.

    $\Phi(\lambda) = \exp(-\ell^2 \lambda / 2)$, the $\nu \to \infty$ limit
    of `MeshMatern` and the FEM version of `SphereHeat`: the heat equation
    on the mesh run for time $\ell^2 / 2$. Its spectrum decays fastest, so
    it needs the fewest eigenpairs. See `AbstractMeshKernel` for the inputs,
    the eigen-solve and the normalisation.

    Args:
        vertices: ``(V, 2)`` or ``(V, 3)`` vertex coordinates.
        triangles: ``(T, 3)`` vertex indices of the triangles.
        lengthscale: Scalar lengthscale $\ell$.
        variance: Scalar signal variance $\sigma^2$.
        n_components: Eigenpairs kept, capped at $V$.
        method: ``"dense"`` (`jnp.linalg.eigh`) or ``"lanczos"``.
        key: Lanczos start vector; ``None`` uses ``jax.random.key(0)``.
        oversample: Extra Krylov dimension of the first Lanczos run.
        max_oversample: Largest extra Krylov dimension a Lanczos restart
            tries.

    Examples:
        >>> import jax.numpy as jnp
        >>> import kernellib as kl
        >>> from geonnax.basis import icosphere
        >>> V, T = icosphere(2)
        >>> k = kl.MeshHeat(V, T, lengthscale=0.5, n_components=64)
        >>> K = k(V[:3], V[:3])  # points on the mesh: barycentric interpolation
        >>> bool(jnp.allclose(K, k(jnp.array([[0.0], [1.0], [2.0]]), V[:3])))
        True
    """

    def __init__(
        self,
        vertices: Float[ArrayLike, "V D"],
        triangles: Int[ArrayLike, "T 3"],
        lengthscale: float | Float[Array, ""] = 1.0,
        variance: float | Float[Array, ""] = 1.0,
        *,
        n_components: int = 200,
        method: MeshEigenMethod = "dense",
        key: PRNGKeyArray | None = None,
        oversample: int = 200,
        max_oversample: int = 1600,
    ) -> None:
        self._build(
            vertices,
            triangles,
            lengthscale,
            variance,
            n_components,
            method,
            key,
            oversample,
            max_oversample,
        )

    def spectrum(self, lam: Float[Array, " M"]) -> Float[Array, " M"]:
        return jnp.exp(-0.5 * self.lengthscale**2 * lam)


def _mesh_eigenpairs(
    vertices: Float[Array, "V D"],
    triangles: np.ndarray,
    n: int,
    *,
    method: MeshEigenMethod,
    key: PRNGKeyArray | None,
    oversample: int,
    max_oversample: int,
) -> tuple[Float[Array, " n"], Float[Array, "V n"], Float[Array, " V"]]:
    r"""The ``n`` smallest FEM eigenpairs $G\varphi = \lambda C\varphi$.

    Returns ``(lam, phi, mass)``: eigenvalues ascending (clipped at 0, so
    the constant mode's round-off is not negative), $C$-orthonormal
    eigenvectors (sign-fixed: each column's largest-magnitude entry of
    $C^{1/2}\varphi$ is positive) and the lumped mass diagonal.
    """
    C, G = gx.fem_matrices(vertices, triangles)
    mass = C.diagonal
    s = 1.0 / jnp.sqrt(mass)
    if method == "dense":
        S = einx.multiply("i j, i, j -> i j", G.as_matrix(), s, s)
        lam, U = jnp.linalg.eigh(S)
        lam, U = lam[:n], U[:, :n]
    else:
        lam, U = _lanczos_mesh(G, s, n, key, oversample, max_oversample)
    U = _fix_signs(U)
    return jnp.maximum(lam, 0.0), einx.multiply("v n, v -> v n", U, s), mass


def _lanczos_mesh(
    G: gx.SparseOperator,
    s: Float[Array, " V"],
    n: int,
    key: PRNGKeyArray | None,
    oversample: int,
    max_oversample: int,
) -> tuple[Float[Array, " n"], Float[Array, "V n"]]:
    r"""Lanczos on $cI - S$, $S = \mathrm{diag}(s)\, G\, \mathrm{diag}(s)$.

    $c = \max_i s_i \sum_j |G_{ij}| s_j \ge \lambda_{\max}(S)$ (Gershgorin),
    so the smallest eigenvalues of $S$ are the largest of $cI - S$.
    """
    if oversample < 0 or max_oversample < 0:
        raise ValueError(
            f"oversample and max_oversample must be >= 0; got {oversample}, "
            f"{max_oversample}."
        )
    key = jr.key(0) if key is None else key
    B = G.to_bcoo()
    absG = jsparse.BCOO((jnp.abs(B.data), B.indices), shape=B.shape)
    c = jnp.max(s * (absG @ s))
    n_vertices = s.shape[0]

    def matvec(v: Float[Array, " V"]) -> Float[Array, " V"]:
        return s * G.mv(s * v)

    shifted = lx.FunctionLinearOperator(
        lambda v: c * v - matvec(v),
        jax.ShapeDtypeStruct((n_vertices,), s.dtype),
        tags=(lx.symmetric_tag, lx.positive_semidefinite_tag),
    )

    def solve(extra: int):
        mu, V = gx.eig(shifted, rank=min(n + extra, n_vertices), key=key)
        top = jnp.argsort(-mu)[:n]
        return c - mu[top], V[:, top], c

    return _lanczos_until_converged(
        solve,
        matvec,
        n,
        n_vertices,
        oversample=oversample,
        max_oversample=max_oversample,
        solver="the mesh kernel's method='lanczos'",
        hint=_LANCZOS_HINT,
    )
