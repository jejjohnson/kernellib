"""GraphMatern / GraphHeat: kernel objects on node indices."""

from __future__ import annotations

import einx
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import kernellib as kl


def _graph():
    X = jax.random.uniform(jax.random.key(0), (30, 2))
    return kl.knn_graph(X, 5, ensure_connected=True)


def _nodes(n):
    return einx.id("n -> n 1", jnp.arange(n))


IDX = jnp.array([[3], [0], [17], [3], [29]])
IDX2 = jnp.array([[1], [8], [22]])


@pytest.mark.parametrize("normalization", ["symmetric", "unnormalized"])
def test_matern_matches_dense_function(normalization):
    g = _graph()
    k = kl.GraphMatern(
        g, nu=1.5, lengthscale=1.7, variance=2.0, normalization=normalization
    )
    K = kl.matern_graph_kernel(
        g, nu=1.5, lengthscale=1.7, variance=2.0, normalization=normalization
    )
    i1 = einx.id("n 1 -> n", IDX)
    i2 = einx.id("n 1 -> n", IDX2)
    assert np.allclose(k(IDX, IDX2), K[i1][:, i2], atol=1e-12)
    assert np.allclose(k.diag(IDX), jnp.diag(K)[i1], atol=1e-12)


@pytest.mark.parametrize("normalization", ["symmetric", "unnormalized"])
def test_heat_is_normalised_diffusion(normalization):
    g = _graph()
    ell, var = 1.3, 0.7
    k = kl.GraphHeat(g, lengthscale=ell, variance=var, normalization=normalization)
    D = kl.diffusion_kernel(g, beta=ell**2 / 2, normalization=normalization)
    D = var * D / (jnp.trace(D) / g.n_nodes)
    nodes = _nodes(g.n_nodes)
    assert np.allclose(k(nodes, nodes), D, atol=1e-12)


def test_dense_adjacency_input():
    g = _graph()
    W = g.to_dense()
    nodes = _nodes(g.n_nodes)
    a = kl.GraphMatern(W, nu=2.0, lengthscale=1.0)(nodes, nodes)
    b = kl.GraphMatern(g, nu=2.0, lengthscale=1.0)(nodes, nodes)
    assert np.allclose(a, b, atol=1e-12)


def test_psd_symmetric_and_average_variance():
    g = _graph()
    nodes = _nodes(g.n_nodes)
    for k in (
        kl.GraphMatern(g, nu=0.5, lengthscale=2.0, variance=1.5),
        kl.GraphHeat(g, lengthscale=2.0, variance=1.5),
    ):
        K = k(nodes, nodes)
        assert np.allclose(K, einx.id("i j -> j i", K))
        assert float(jnp.min(jnp.linalg.eigvalsh(K))) > -1e-10
        assert np.isclose(float(jnp.mean(k.diag(nodes))), 1.5)


def test_gram_only():
    k = kl.GraphHeat(_graph())
    assert not k.is_pointwise
    with pytest.raises(TypeError, match="Gram-only"):
        k.pairwise(jnp.array([0]), jnp.array([1]))


@pytest.mark.slow
def test_gradients_finite():
    g = _graph()
    nodes = _nodes(g.n_nodes)
    for k in (
        kl.GraphMatern(g, nu=1.5, lengthscale=1.2, variance=0.8),
        kl.GraphHeat(g, lengthscale=1.2, variance=0.8),
    ):
        grads = eqx.filter_grad(lambda k: jnp.sum(k(nodes, IDX2)))(k)
        assert np.isfinite(float(grads.lengthscale))
        assert np.isfinite(float(grads.variance))
        assert float(jnp.abs(grads.lengthscale)) > 0


def test_jit():
    k = kl.GraphMatern(_graph(), nu=1.5)
    assert np.allclose(eqx.filter_jit(k)(IDX, IDX2), k(IDX, IDX2))


def test_validation():
    g = _graph()
    with pytest.raises(ValueError, match="nu"):
        kl.GraphMatern(g, nu=0.0)
    with pytest.raises(ValueError, match="normalization"):
        kl.GraphHeat(g, normalization="random_walk")  # ty: ignore[invalid-argument-type]
    with pytest.raises(ValueError, match=r"\(N, 1\)"):
        kl.GraphHeat(g)(jnp.arange(3), jnp.arange(3))
