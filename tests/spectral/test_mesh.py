"""MeshLaplaceFeatures: FEM eigenvector features of the mesh kernels (GEO21)."""

from __future__ import annotations

import einx
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from geonnax.basis import icosphere

import kernellib as kl
from kernellib._einx import einsum


@pytest.fixture(scope="module")
def mesh() -> tuple[jax.Array, jax.Array]:
    return icosphere(2)


@pytest.fixture(scope="module")
def kernel(mesh) -> kl.MeshMatern:
    V, T = mesh
    return kl.MeshMatern(V, T, lengthscale=0.4, nu=1.5, variance=1.3, n_components=81)


def _gram(Phi: jax.Array) -> jax.Array:
    return einsum(Phi, Phi, "n f, m f -> n m")


def _vertices(n: int) -> jax.Array:
    return einx.id("n -> n 1", jnp.arange(n, dtype=float))


def test_features_reproduce_the_kernel(kernel) -> None:
    # Exact: Phi Phi^T and the kernel are the same finite eigen-expansion.
    X = _vertices(162)
    mf = kl.MeshLaplaceFeatures().fit(kernel, X)
    assert mf.n_features == 81
    Phi = mf(X)
    assert Phi.shape == (162, 81)
    np.testing.assert_allclose(_gram(Phi), kernel(X, X), atol=1e-12)


def test_truncation_drops_the_highest_modes(kernel) -> None:
    X = _vertices(162)
    full = kl.MeshLaplaceFeatures().fit(kernel, X)(X)
    part = kl.MeshLaplaceFeatures(n_components=49).fit(kernel, X)(X)
    np.testing.assert_allclose(part, full[:, :49])
    # The dropped variance is the truncated weights' share of sigma^2.
    w = kernel.spectral_weights()
    mean_var = jnp.sum(kernel.mass * jnp.diag(_gram(part))) / jnp.sum(kernel.mass)
    expected = jnp.sum(w[:49]) / jnp.sum(kernel.mass)
    np.testing.assert_allclose(mean_var, expected, rtol=1e-10)
    assert float(mean_var) < 1.3


@pytest.mark.slow
def test_points_features(mesh, kernel) -> None:
    V, T = mesh
    P = einx.mean("t [3] d -> t d", V[T[:6]])
    mf = kl.MeshLaplaceFeatures(inputs="points").fit(kernel, P)
    np.testing.assert_allclose(_gram(mf(P)), kernel(P, P), atol=1e-12)
    # At a vertex, the points path equals the vertex path.
    mv = kl.MeshLaplaceFeatures().fit(kernel, _vertices(1))
    np.testing.assert_allclose(mf(V[7:8]), mv(_vertices(8)[7:]), atol=1e-12)


@pytest.mark.slow
def test_gradients_reach_hyperparameters(kernel) -> None:
    X = _vertices(20)
    mf = kl.MeshLaplaceFeatures().fit(kernel, X)
    g = eqx.filter_grad(lambda m: jnp.sum(_gram(m(X))))(mf)
    g_kernel = eqx.filter_grad(lambda k: jnp.sum(k(X, X)))(kernel)
    np.testing.assert_allclose(g.kernel.lengthscale, g_kernel.lengthscale, rtol=1e-8)
    np.testing.assert_allclose(g.kernel.variance, g_kernel.variance, rtol=1e-8)


def test_jit_over_vertex_indices(kernel) -> None:
    X = _vertices(10)
    mf = kl.MeshLaplaceFeatures().fit(kernel, X)
    np.testing.assert_allclose(eqx.filter_jit(lambda m, x: m(x))(mf, X), mf(X))


def test_errors(mesh, kernel) -> None:
    V, _ = mesh
    with pytest.raises(NotImplementedError, match="mesh kernel"):
        kl.MeshLaplaceFeatures().fit(kl.RBF(), V)
    with pytest.raises(ValueError, match="vertex indices"):
        kl.MeshLaplaceFeatures().fit(kernel, V)
    with pytest.raises(ValueError, match=r"\(N, 3\) points"):
        kl.MeshLaplaceFeatures(inputs="points").fit(kernel, _vertices(3))
    with pytest.raises(ValueError, match="exceeds"):
        kl.MeshLaplaceFeatures(n_components=100).fit(kernel, _vertices(3))
    with pytest.raises(ValueError, match="inputs"):
        kl.MeshLaplaceFeatures(inputs="edges")  # ty: ignore[invalid-argument-type]
    with pytest.raises(ValueError, match="n_components"):
        kl.MeshLaplaceFeatures(n_components=0)
    with pytest.raises(RuntimeError, match="not fitted"):
        kl.MeshLaplaceFeatures()(_vertices(3))
    with pytest.raises(RuntimeError, match="unknown"):
        _ = kl.MeshLaplaceFeatures().n_features
