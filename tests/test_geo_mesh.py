"""Mesh kernels: Matérn and heat from FEM Laplace-Beltrami eigenpairs (GEO21)."""

from __future__ import annotations

import math

import einx
import equinox as eqx
import gaussx as gx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from geonnax.basis import icosphere

import kernellib as kl
from kernellib._einx import einsum, rearrange


def _grid_mesh(
    n: int, h: float, keep=None
) -> tuple[np.ndarray, np.ndarray, dict[tuple[int, int], int]]:
    """Right-triangle mesh of the ``n h x n h`` square, minus dropped triangles.

    ``keep(cx, cy)`` decides from a triangle's centroid whether to keep it;
    unused vertices are dropped. Returns the vertices, triangles and a map
    from grid index ``(i, j)`` to vertex index.
    """
    triangles = []
    for i in range(n):
        for j in range(n):
            a, b, c, d = (i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)
            for tri in ((a, b, c), (a, c, d)):
                cx = h * sum(p[0] for p in tri) / 3.0
                cy = h * sum(p[1] for p in tri) / 3.0
                if keep is None or keep(cx, cy):
                    triangles.append(tri)
    used = sorted({p for tri in triangles for p in tri})
    index = {p: k for k, p in enumerate(used)}
    vertices = h * np.array(used, dtype=float)
    tris = np.array([[index[p] for p in tri] for tri in triangles])
    return vertices, tris, index


def _indices(*idx: int) -> jax.Array:
    return einx.id("n -> n 1", jnp.asarray(idx, dtype=float))


def _all_vertices(k: kl.AbstractMeshKernel) -> jax.Array:
    return einx.id("n -> n 1", jnp.arange(k.vertices.shape[0], dtype=float))


@pytest.fixture(scope="module")
def ico2() -> tuple[jax.Array, jax.Array]:
    return icosphere(2)


@pytest.fixture(scope="module")
def ico4_matern() -> kl.MeshMatern:
    V, T = icosphere(4)  # 2562 vertices, h ~ 0.075
    # 196 = (13 + 1)^2: the degree-0..13 clusters exactly.
    return kl.MeshMatern(V, T, lengthscale=0.5, nu=1.5, n_components=196)


# --- eigenvalues -------------------------------------------------------------


@pytest.mark.slow
def test_square_eigenvalues_approach_neumann() -> None:
    # Neumann Laplacian on [0, 1]^2: pi^2 (m^2 + n^2). The lumped-mass P1
    # error is O(h^2) (~ (pi m h)^2 / 12, 0.7 % at m = 3, h = 1/32).
    V, T, _ = _grid_mesh(32, 1.0 / 32)
    k = kl.MeshMatern(V, T, n_components=10)
    exact = sorted(math.pi**2 * (m * m + n * n) for m in range(5) for n in range(5))
    exact = np.array(exact[:10])
    lam = np.asarray(k.eigenvalues)
    assert abs(lam[0]) < 1e-8
    np.testing.assert_allclose(lam[1:], exact[1:], rtol=0.01)


@pytest.mark.slow
def test_icosphere_eigenvalues_cluster_at_l_l_plus_1(ico4_matern) -> None:
    # S^2: l (l + 1) with multiplicity 2l + 1. FEM error at level 4 is O(h^2),
    # measured <= 0.7 % for l <= 4; the issue's bound is 2 %.
    lam = np.asarray(ico4_matern.eigenvalues)
    assert abs(lam[0]) < 1e-8
    for ell in range(1, 5):
        cluster = lam[ell**2 : (ell + 1) ** 2]
        np.testing.assert_allclose(cluster, ell * (ell + 1), rtol=0.02)
    # The next cluster starts above l = 4's.
    assert lam[25] > 1.2 * lam[24]


@pytest.mark.slow
def test_icosphere_matern_matches_sphere_matern(ico4_matern) -> None:
    # Same truncation (degrees 0..13 = 196 eigenpairs) on both sides, so the
    # gap is the FEM error of eigenpairs and of the mesh's area (an inscribed
    # polyhedron): measured 0.86 % relative Frobenius; the issue asks 2 %.
    k = ico4_matern
    sphere = kl.SphereMatern(lengthscale=0.5, nu=1.5, max_degree=13)
    K_mesh = k(_all_vertices(k), _all_vertices(k))
    K_sphere = sphere(k.vertices, k.vertices)
    rel = jnp.linalg.norm(K_mesh - K_sphere) / jnp.linalg.norm(K_sphere)
    assert float(rel) < 0.02


@pytest.mark.slow
def test_lanczos_matches_dense(ico2) -> None:
    V, T = ico2
    # 64 = (7 + 1)^2 closes the degree-7 cluster, so the Gram is unique even
    # though the eigenvectors inside a cluster are not.
    dense = kl.MeshMatern(V, T, lengthscale=0.4, n_components=64)
    lanczos = kl.MeshMatern(
        V, T, lengthscale=0.4, n_components=64, method="lanczos", key=jax.random.key(1)
    )
    np.testing.assert_allclose(lanczos.eigenvalues, dense.eigenvalues, atol=1e-8)
    X = _all_vertices(dense)
    np.testing.assert_allclose(lanczos(X, X), dense(X, X), atol=1e-8)


# --- kernel properties -------------------------------------------------------


@pytest.mark.parametrize("cls", [kl.MeshMatern, kl.MeshHeat])
def test_psd_and_area_weighted_variance(ico2, cls) -> None:
    V, T = ico2
    k = cls(V, T, lengthscale=0.4, variance=1.7, n_components=100)
    X = _all_vertices(k)
    K = k(X, X)
    np.testing.assert_allclose(K, rearrange(K, "i j -> j i"), atol=1e-12)
    assert float(jnp.min(jnp.linalg.eigvalsh(K))) > -1e-10
    # sum_v C_vv k(v, v) / A = sigma^2 by the C-orthonormality of phi.
    mean = jnp.sum(k.mass * jnp.diag(K)) / jnp.sum(k.mass)
    np.testing.assert_allclose(mean, 1.7, rtol=1e-10)
    np.testing.assert_allclose(k.diag(X), jnp.diag(K), rtol=1e-10)
    # C-orthonormal eigenvectors.
    weighted = einx.multiply("v i, v -> v i", k.eigenvectors, k.mass)
    gram = einsum(weighted, k.eigenvectors, "v i, v j -> i j")
    np.testing.assert_allclose(gram, jnp.eye(100), atol=1e-10)


@pytest.mark.slow
def test_gradient_wrt_lengthscale_is_finite(ico2) -> None:
    V, T = ico2
    k = kl.MeshMatern(V, T, lengthscale=0.4, n_components=50)
    X = _indices(0, 5, 40, 100)

    def f(ell):
        return jnp.sum(eqx.tree_at(lambda m: m.lengthscale, k, ell)(X, X))

    g = jax.grad(f)(jnp.asarray(0.4))
    assert bool(jnp.isfinite(g)) and float(g) != 0.0
    gk = eqx.filter_grad(lambda m: jnp.sum(m(X, X)))(k)
    assert bool(jnp.isfinite(gk.lengthscale) & jnp.isfinite(gk.variance))


@pytest.mark.slow
def test_points_interpolate_the_vertex_basis(ico2) -> None:
    V, T = ico2
    k = kl.MeshHeat(V, T, lengthscale=0.5, n_components=64)
    idx = _indices(3, 17, 90)
    np.testing.assert_allclose(k(V[3:4], V[17:18]), k(idx[:1], idx[1:2]), atol=1e-10)
    # Face centroids: barycentric (1/3, 1/3, 1/3) averages of their corners.
    P = einx.mean("t [3] d -> t d", V[T[:5]])
    corners = einx.mean("t [3] m -> t m", k.eigenvectors[T[:5]])
    np.testing.assert_allclose(k.eigenfunctions(P), corners, atol=1e-10)
    K = k(P, P)
    assert float(jnp.min(jnp.linalg.eigvalsh(K))) > -1e-10


def test_vertex_indices_jit(ico2) -> None:
    V, T = ico2
    k = kl.MeshMatern(V, T, lengthscale=0.4, n_components=30)
    X = _indices(0, 1, 2, 3)
    jitted = eqx.filter_jit(lambda m, x: m(x, x))(k, X)
    np.testing.assert_allclose(jitted, k(X, X), atol=1e-12)
    # Integer indices work too.
    np.testing.assert_allclose(k(X.astype(jnp.int32), X), k(X, X))


def test_to_operator_keeps_low_rank(ico2) -> None:
    V, T = ico2
    k = kl.MeshMatern(V, T, lengthscale=0.4, n_components=30)
    X = _all_vertices(k)
    op = kl.to_operator(k, X)
    assert isinstance(op, gx.LowRankUpdate)
    np.testing.assert_allclose(op.as_matrix(), k(X, X), atol=1e-10)


@pytest.mark.slow
def test_barrier_reduces_correlation_across_a_gap() -> None:
    # A U-shaped domain on [0, 3]^2: the slot x in (1, 2), y > 0.8 is land.
    # a = (0.9, 2.5) and b = (2.1, 2.5) face each other across the slot;
    # c = (0.9, 1.3) is as far from a, along the same arm.
    h = 0.1
    V, T, index = _grid_mesh(30, h, keep=lambda x, y: not (1.0 < x < 2.0 and y > 0.8))
    k = kl.MeshMatern(V, T, lengthscale=1.0, nu=1.5, n_components=150)
    a, b, c = index[(9, 25)], index[(21, 25)], index[(9, 13)]
    np.testing.assert_allclose(
        np.linalg.norm(V[a] - V[b]), np.linalg.norm(V[a] - V[c]), rtol=1e-12
    )
    K = k(_indices(a), _indices(b, c))
    across, along = float(K[0, 0]), float(K[0, 1])
    assert across < 0.5 * along
    # A Euclidean Matérn cannot tell them apart.
    euclid = kl.Matern(lengthscale=1.0, nu=1.5)(V[a : a + 1], V[[b, c]])
    np.testing.assert_allclose(euclid[0, 0], euclid[0, 1], rtol=1e-12)


# --- validation --------------------------------------------------------------


def test_validation(ico2) -> None:
    V, T = ico2
    with pytest.raises(ValueError, match="triangles"):
        kl.MeshMatern(V, T[:, :2])
    with pytest.raises(ValueError, match="vertices"):
        kl.MeshMatern(V[:, :1], T)
    with pytest.raises(ValueError, match="belong to a triangle"):
        kl.MeshMatern(jnp.concatenate([V, V[:1]]), T)
    with pytest.raises(ValueError, match="method"):
        kl.MeshHeat(V, T, method="arpack")
    with pytest.raises(ValueError, match="nu"):
        kl.MeshMatern(V, T, nu=0.0)
    with pytest.raises(ValueError, match="n_components"):
        kl.MeshHeat(V, T, n_components=0)
    k = kl.MeshHeat(V, T, n_components=500)
    assert k.n_components == V.shape[0]  # capped at V
    with pytest.raises(ValueError, match="vertex indices"):
        k(jnp.zeros((3, 2)), jnp.zeros((3, 2)))
