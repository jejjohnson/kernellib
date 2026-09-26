"""Neighbour graphs, Laplacians and graph kernels."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsl
import numpy as np
import pytest

import kernellib as kl


def _X(n=50, d=3, seed=0):
    return jax.random.normal(jax.random.key(seed), (n, d))


def _brute(X, k):
    X = np.asarray(X)
    D = np.sqrt(((X[:, None] - X[None]) ** 2).sum(-1))
    np.fill_diagonal(D, np.inf)
    idx = np.argsort(D, axis=1)[:, :k]
    return idx, np.take_along_axis(D, idx, axis=1)


class TestNearestNeighbors:
    @pytest.mark.parametrize("batch_size", [7, 16, 1024])
    def test_exact_matches_brute_force(self, batch_size):
        X = _X()
        g = kl.nearest_neighbors(X, 5, batch_size=batch_size)
        idx, dist = _brute(X, 5)
        assert np.array_equal(np.asarray(g.indices), idx)
        assert np.allclose(np.asarray(g.distances), dist, atol=1e-6)
        assert g.n_points == 50 and g.n_neighbors == 5

    def test_self_is_never_a_neighbour_but_duplicates_are(self):
        X = jnp.array([[0.0], [0.0], [5.0]])
        g = kl.nearest_neighbors(X, 1)
        assert g.indices.tolist() == [[1], [0], [0]] or g.indices.tolist() == [
            [1],
            [0],
            [1],
        ]
        assert float(g.distances[0, 0]) == 0.0

    def test_sklearn_backend_matches_exact(self):
        X = _X()
        exact = kl.nearest_neighbors(X, 4)
        sk = kl.nearest_neighbors(X, 4, backend="sklearn")
        assert np.array_equal(np.asarray(sk.indices), np.asarray(exact.indices))
        assert np.allclose(sk.distances, exact.distances, atol=1e-6)

    @pytest.mark.slow
    def test_pynndescent_backend_recovers_the_graph(self):
        # NN-descent is approximate, but on 300 points in 3-D it recovers
        # essentially every true neighbour.
        X = _X(n=300)
        exact = np.asarray(kl.nearest_neighbors(X, 5).indices)
        approx = np.asarray(
            kl.nearest_neighbors(X, 5, backend="pynndescent", random_state=0).indices
        )
        recall = np.mean(
            [len(set(a) & set(e)) / 5 for a, e in zip(approx, exact, strict=True)]
        )
        assert recall > 0.95
        assert not np.any(approx == np.arange(300)[:, None])

    def test_errors(self):
        with pytest.raises(ValueError, match="n_neighbors"):
            kl.nearest_neighbors(_X(n=5), 5)
        with pytest.raises(ValueError, match="backend"):
            kl.nearest_neighbors(_X(), 3, backend="annoy")


class TestAdjacencyAndLaplacian:
    def test_heat_weights_and_symmetry(self):
        X = _X()
        g = kl.nearest_neighbors(X, 6)
        W = kl.adjacency_matrix(g, bandwidth=0.7)
        assert jnp.allclose(W, W.T) and jnp.all(jnp.diag(W) == 0)
        i, j = 0, int(g.indices[0, 0])
        expected = jnp.exp(-(g.distances[0, 0] ** 2) / (2 * 0.7**2))
        assert jnp.allclose(W[i, j], expected)

    def test_default_bandwidth_is_the_median_distance(self):
        g = kl.nearest_neighbors(_X(), 6)
        sigma = jnp.median(g.distances)
        assert jnp.allclose(
            kl.adjacency_matrix(g), kl.adjacency_matrix(g, bandwidth=sigma)
        )

    def test_symmetrization_modes(self):
        X = jnp.array([[0.0], [1.0], [1.5], [10.0]])
        g = kl.nearest_neighbors(X, 1)
        union = kl.adjacency_matrix(g, weighting="connectivity", symmetrize="max")
        mutual = kl.adjacency_matrix(g, weighting="connectivity", symmetrize="min")
        mean = kl.adjacency_matrix(g, weighting="connectivity", symmetrize="mean")
        # 0 -> 1, 1 -> 2, 2 -> 1, 3 -> 2: only 1 <-> 2 is mutual.
        assert float(union[0, 1]) == 1.0 and float(mutual[0, 1]) == 0.0
        assert float(mutual[1, 2]) == 1.0 and float(mean[0, 1]) == 0.5

    def test_laplacians(self):
        W = kl.adjacency_matrix(kl.nearest_neighbors(_X(), 6))
        L = kl.graph_laplacian(W)
        assert jnp.allclose(L.sum(axis=1), 0.0, atol=1e-6)
        Ls = kl.graph_laplacian(W, "symmetric")
        lam = jnp.linalg.eigvalsh(Ls)
        assert lam[0] > -1e-6 and lam[-1] < 2 + 1e-6
        Lrw = kl.graph_laplacian(W, "random_walk")
        assert jnp.allclose(Lrw.sum(axis=1), 0.0, atol=1e-6)

    def test_isolated_node(self):
        W = jnp.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        Ls = kl.graph_laplacian(W, "symmetric")
        assert jnp.all(jnp.isfinite(Ls)) and jnp.all(Ls[2] == 0)

    def test_errors(self):
        g = kl.nearest_neighbors(_X(), 3)
        with pytest.raises(ValueError, match="weighting"):
            kl.adjacency_matrix(g, weighting="cosine")
        with pytest.raises(ValueError, match="symmetrize"):
            kl.adjacency_matrix(g, symmetrize="or")
        with pytest.raises(ValueError, match="normalization"):
            kl.graph_laplacian(jnp.eye(3), "left")


class TestGraphKernels:
    @staticmethod
    def _W():
        return kl.adjacency_matrix(kl.nearest_neighbors(_X(n=30), 5))

    def test_closed_forms(self):
        W = self._W()
        L = kl.graph_laplacian(W, "symmetric")
        eye = jnp.eye(30)
        assert jnp.allclose(kl.diffusion_kernel(W, 0.7), jsl.expm(-0.7 * L), atol=1e-8)
        assert jnp.allclose(
            kl.regularized_laplacian_kernel(W, 2.0),
            jnp.linalg.inv(eye + 4.0 * L),
            atol=1e-8,
        )
        assert jnp.allclose(
            kl.random_walk_kernel(W, p=3, a=2.5),
            jnp.linalg.matrix_power(2.5 * eye - L, 3),
            atol=1e-8,
        )

    @pytest.mark.parametrize(
        "fn",
        [
            kl.diffusion_kernel,
            kl.regularized_laplacian_kernel,
            kl.random_walk_kernel,
            kl.cosine_graph_kernel,
            kl.commute_time_kernel,
        ],
    )
    def test_symmetric_psd(self, fn):
        K = fn(self._W())
        assert jnp.allclose(K, K.T, atol=1e-10)
        assert jnp.linalg.eigvalsh(K)[0] > -1e-8

    def test_commute_time_is_the_pseudo_inverse(self):
        W = self._W()
        L = kl.graph_laplacian(W)
        K = kl.commute_time_kernel(W)
        assert jnp.allclose(L @ K @ L, L, atol=1e-8)

    def test_diffusion_is_differentiable_in_beta(self):
        W = self._W()
        g = jax.grad(lambda b: jnp.trace(kl.diffusion_kernel(W, b)))(0.5)
        L = kl.graph_laplacian(W, "symmetric")
        assert jnp.allclose(g, -jnp.trace(L @ jsl.expm(-0.5 * L)), rtol=1e-6)

    def test_errors(self):
        with pytest.raises(ValueError, match="symmetric Laplacian"):
            kl.diffusion_kernel(self._W(), normalization="random_walk")
        with pytest.raises(ValueError, match="p must be"):
            kl.random_walk_kernel(self._W(), p=0)
