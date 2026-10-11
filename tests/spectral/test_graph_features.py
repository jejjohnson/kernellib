"""GraphEigenFeatures: rank-M features of GraphMatern / GraphHeat."""

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


def _gram(Phi):
    return einx.dot("n m, k m -> n k", Phi, Phi)


KERNELS = [
    pytest.param(
        lambda g, norm: kl.GraphMatern(
            g, nu=1.5, lengthscale=1.4, variance=2.0, normalization=norm
        ),
        id="matern",
    ),
    pytest.param(
        lambda g, norm: kl.GraphHeat(
            g, lengthscale=1.4, variance=2.0, normalization=norm
        ),
        id="heat",
    ),
]


@pytest.mark.parametrize("make", KERNELS)
@pytest.mark.parametrize("normalization", ["symmetric", "unnormalized"])
def test_full_rank_recovers_dense(make, normalization):
    g = _graph()
    k = make(g, normalization)
    nodes = _nodes(g.n_nodes)
    Phi = kl.GraphEigenFeatures(g.n_nodes).fit(k, nodes)(nodes)
    assert Phi.shape == (g.n_nodes, g.n_nodes)
    assert float(jnp.max(jnp.abs(_gram(Phi) - k(nodes, nodes)))) < 1e-10


@pytest.mark.parametrize("make", KERNELS)
@pytest.mark.parametrize("m", [3, 30])
def test_truncated_average_variance(make, m):
    g = _graph()
    nodes = _nodes(g.n_nodes)
    Phi = kl.GraphEigenFeatures(m).fit(make(g, "symmetric"), nodes)(nodes)
    var = einx.dot("n m, n m -> n", Phi, Phi)
    assert np.isclose(float(jnp.mean(var)), 2.0, atol=1e-10)


def test_subset_rows():
    g = _graph()
    k = kl.GraphMatern(g, nu=2.0, lengthscale=1.0)
    nodes = _nodes(g.n_nodes)
    idx = jnp.array([[4], [0], [25]])
    fm = kl.GraphEigenFeatures(3).fit(k, idx)
    i = einx.id("n 1 -> n", idx)
    assert np.allclose(fm(idx), fm(nodes)[i])


@pytest.mark.slow
def test_kronecker_matches_dense_on_grid():
    g = kl.grid_graph((30, 30))
    k = kl.GraphMatern(g, nu=1.5, lengthscale=4.0, normalization="unnormalized")
    nodes = _nodes(g.n_nodes)
    # Truncate at a gap of the spectrum: inside a repeated eigenvalue the two
    # solvers may return different bases of the eigenspace.
    lam, _ = kl.laplacian_eigpairs(g, 60, method="kronecker")
    gaps = np.flatnonzero(np.diff(np.asarray(lam)) > 1e-6) + 1
    m = int(gaps[gaps >= 40][0])
    dense = kl.GraphEigenFeatures(m, method="dense").fit(k, nodes)(nodes)
    kron = kl.GraphEigenFeatures(m, method="kronecker").fit(k, nodes)(nodes)
    assert float(jnp.max(jnp.abs(_gram(dense) - _gram(kron)))) < 1e-8


@pytest.mark.slow
def test_gradients_finite():
    g = _graph()
    nodes = _nodes(g.n_nodes)
    for k in (
        kl.GraphMatern(g, nu=1.5, lengthscale=1.2, variance=0.8),
        kl.GraphHeat(g, lengthscale=1.2, variance=0.8),
    ):
        fm = kl.GraphEigenFeatures(10).fit(k, nodes)

        def loss(fm):
            return jnp.sum(_gram(fm(nodes))[0])

        grads = eqx.filter_grad(loss)(fm)
        assert grads.kernel is not None
        assert np.isfinite(float(grads.kernel.lengthscale))
        assert np.isfinite(float(grads.kernel.variance))
        assert float(jnp.abs(grads.kernel.lengthscale)) > 0


@pytest.mark.slow
def test_underflowing_modes_have_finite_gradients():
    # A long heat lengthscale underflows the high modes' weights to zero.
    g = _graph()
    nodes = _nodes(g.n_nodes)
    k = kl.GraphHeat(g, lengthscale=60.0, normalization="unnormalized")
    fm = kl.GraphEigenFeatures(g.n_nodes).fit(k, nodes)
    assert float(jnp.min(fm.weights)) == 0.0
    grads = eqx.filter_grad(lambda fm: jnp.sum(fm(nodes)))(fm)
    assert grads.kernel is not None
    assert np.isfinite(float(grads.kernel.lengthscale))


@pytest.mark.slow
def test_operator_and_jit():
    g = _graph()
    nodes = _nodes(g.n_nodes)
    fm = kl.GraphEigenFeatures(8).fit(kl.GraphHeat(g), nodes)
    assert np.allclose(fm.operator(nodes).as_matrix(), _gram(fm(nodes)))
    assert np.allclose(eqx.filter_jit(lambda f, x: f(x))(fm, nodes), fm(nodes))


@pytest.mark.slow
def test_lanczos_needs_key_and_runs():
    g = _graph()
    nodes = _nodes(g.n_nodes)
    k = kl.GraphMatern(g, nu=1.5, normalization="unnormalized")
    fm = kl.GraphEigenFeatures(5, method="lanczos")
    with pytest.raises(ValueError, match="key"):
        fm.fit(k, nodes)
    dense = kl.GraphEigenFeatures(5, method="dense").fit(k, nodes)
    lanczos = fm.fit(k, nodes, key=jax.random.key(0))
    assert np.allclose(lanczos.weights, dense.weights, atol=1e-8)


def test_errors():
    g = _graph()
    nodes = _nodes(g.n_nodes)
    with pytest.raises(ValueError, match="n_components"):
        kl.GraphEigenFeatures(0)
    with pytest.raises(TypeError, match="GraphMatern or GraphHeat"):
        kl.GraphEigenFeatures(3).fit(kl.RBF(), nodes)
    with pytest.raises(ValueError, match="n must be"):
        kl.GraphEigenFeatures(31).fit(kl.GraphHeat(g), nodes)
    with pytest.raises(RuntimeError, match="not fitted"):
        kl.GraphEigenFeatures(3)(nodes)
