"""Graph builders: neighbours, radii, lattices, adjacency matrices, edge lists."""

from __future__ import annotations

import einx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components, minimum_spanning_tree

import kernellib as kl
from kernellib._einx import einsum, rearrange
from kernellib._graph._construct import _dense_adjacency


def _X(n=40, d=2, seed=0):
    return jax.random.normal(jax.random.key(seed), (n, d))


def _pairs(graph):
    top = graph.topology
    return set(zip(top.senders.tolist(), top.receivers.tolist(), strict=True))


def _brute_distances(X):
    X = np.asarray(X)
    return np.sqrt(((X[:, None] - X[None]) ** 2).sum(-1))


class TestAdjacencyMatrix:
    @pytest.mark.parametrize("weighting", ["heat", "connectivity"])
    @pytest.mark.parametrize("symmetrize", ["max", "min", "mean"])
    @pytest.mark.parametrize("bandwidth", [None, 0.7])
    def test_bit_identical_to_the_dense_construction(
        self, weighting, symmetrize, bandwidth
    ):
        knn = kl.nearest_neighbors(_X(), 5)
        dense = _dense_adjacency(knn, weighting, bandwidth, symmetrize)
        sparse = kl.adjacency_matrix(
            knn, weighting=weighting, bandwidth=bandwidth, symmetrize=symmetrize
        )
        assert np.array_equal(np.asarray(sparse), np.asarray(dense))

    def test_bit_identical_with_duplicate_points(self):
        X = jnp.array([[0.0], [0.0], [1.0], [1.0], [3.0]])
        knn = kl.nearest_neighbors(X, 2)
        for symmetrize in ("max", "min", "mean"):
            dense = _dense_adjacency(knn, "heat", None, symmetrize)
            sparse = kl.adjacency_matrix(knn, symmetrize=symmetrize)
            assert np.array_equal(np.asarray(sparse), np.asarray(dense))

    def test_traced_indices_fall_back_to_the_dense_path(self):
        knn = kl.nearest_neighbors(_X(), 4)
        eager = kl.adjacency_matrix(knn, symmetrize="mean")
        jitted = jax.jit(lambda g: kl.adjacency_matrix(g, symmetrize="mean"))(knn)
        assert np.allclose(jitted, eager)

    def test_bandwidth_is_ignored_for_connectivity(self):
        knn = kl.nearest_neighbors(_X(), 4)
        W = kl.adjacency_matrix(knn, weighting="connectivity", bandwidth=0.3)
        assert set(np.unique(np.asarray(W)).tolist()) == {0.0, 1.0}

    def test_invalid_options(self):
        knn = kl.nearest_neighbors(_X(), 4)
        with pytest.raises(ValueError, match="weighting"):
            kl.adjacency_matrix(knn, weighting="cosine")
        with pytest.raises(ValueError, match="symmetrize"):
            kl.adjacency_matrix(knn, symmetrize="sum")


class TestGraphFromNeighbors:
    def test_to_dense_agrees_with_adjacency_matrix(self):
        knn = kl.nearest_neighbors(_X(), 5)
        g = kl.graph_from_neighbors(knn)
        assert np.array_equal(np.asarray(g.to_dense()), kl.adjacency_matrix(knn))

    def test_min_keeps_only_mutual_neighbours(self):
        knn = kl.nearest_neighbors(_X(), 3)
        idx = np.asarray(knn.indices)
        mutual = {
            (min(i, j), max(i, j))
            for i in range(idx.shape[0])
            for j in idx[i]
            if i in idx[j]
        }
        assert _pairs(kl.graph_from_neighbors(knn, symmetrize="min")) == mutual

    def test_local_bandwidth(self):
        X = _X(20)
        knn = kl.nearest_neighbors(X, 3)
        g = kl.graph_from_neighbors(knn, bandwidth="local")
        sigma = np.asarray(knn.distances)[:, -1]
        D = _brute_distances(X)
        s, r = g.topology.senders, g.topology.receivers
        expected = np.exp(-(D[s, r] ** 2) / (sigma[s] * sigma[r]))
        assert np.allclose(g.weights, expected)

    def test_stationary_kernel_on_distances_equals_heat(self):
        knn = kl.nearest_neighbors(_X(), 4)
        heat = kl.graph_from_neighbors(knn, bandwidth=0.8)
        rbf = kl.graph_from_neighbors(knn, weighting=kl.RBF(lengthscale=0.8))
        assert heat.topology == rbf.topology
        assert np.allclose(heat.weights, rbf.weights)

    def test_options_that_need_the_points(self):
        knn = kl.nearest_neighbors(_X(), 4)
        with pytest.raises(ValueError, match="cosine"):
            kl.graph_from_neighbors(knn, weighting="cosine")
        with pytest.raises(ValueError, match="non-stationary"):
            kl.graph_from_neighbors(knn, weighting=kl.Linear())
        with pytest.raises(ValueError, match="ARD"):
            kl.graph_from_neighbors(
                knn, weighting=kl.RBF(lengthscale=jnp.array([1.0, 2.0]))
            )

    def test_bandwidth_needs_heat(self):
        knn = kl.nearest_neighbors(_X(), 4)
        with pytest.raises(ValueError, match="bandwidth"):
            kl.graph_from_neighbors(knn, weighting="connectivity", bandwidth=1.0)
        with pytest.raises(ValueError, match="bandwidth"):
            kl.graph_from_neighbors(knn, bandwidth="widest")

    def test_traced_indices_are_rejected(self):
        knn = kl.nearest_neighbors(_X(), 4)
        with pytest.raises(TypeError, match="eagerly"):
            jax.jit(lambda g: kl.graph_from_neighbors(g).weights)(knn)


class TestKnnGraph:
    def test_kernel_weighting_with_rbf_equals_heat(self):
        X = _X()
        heat = kl.knn_graph(X, 5, bandwidth=0.6)
        rbf = kl.knn_graph(X, 5, weighting=kl.RBF(lengthscale=0.6))
        assert heat.topology == rbf.topology
        assert np.allclose(heat.weights, rbf.weights)

    def test_cosine_weights_are_clipped_at_zero(self):
        X = jnp.array([[1.0, 0.0], [0.9, 0.1], [-1.0, 0.05], [-0.9, -0.1]])
        g = kl.knn_graph(X, 2, weighting="cosine")
        Xn = np.asarray(X) / np.linalg.norm(X, axis=1, keepdims=True)
        s, r = g.topology.senders, g.topology.receivers
        expected = np.clip((Xn[s] * Xn[r]).sum(-1), 0.0, None)
        assert np.allclose(g.weights, expected)
        assert np.all(np.asarray(g.weights) >= 0.0)

    def test_matches_graph_from_neighbors(self):
        X = _X()
        g = kl.knn_graph(X, 4, symmetrize="mean")
        h = kl.graph_from_neighbors(kl.nearest_neighbors(X, 4), symmetrize="mean")
        assert g.topology == h.topology
        assert np.array_equal(g.weights, h.weights)

    @pytest.mark.parametrize("n_clusters", [2, pytest.param(5, marks=pytest.mark.slow)])
    def test_ensure_connected_adds_mst_edges(self, n_clusters):
        centres = einx.multiply(
            "c, d -> c d", 20.0 * jnp.arange(n_clusters), jnp.array([1.0, 0.3])
        )
        noise = jax.random.normal(jax.random.key(1), (n_clusters, 8, 2))
        X = rearrange(
            einx.add("c m d, c d -> c m d", noise, centres), "c m d -> (c m) d"
        )
        plain = kl.knn_graph(X, 3)
        joined = kl.knn_graph(X, 3, ensure_connected=True)

        def n_components(g):
            top = g.topology
            A = sp.coo_matrix(
                (np.ones(top.n_edges), (top.senders, top.receivers)),
                shape=(top.n_nodes, top.n_nodes),
            )
            return connected_components(A, directed=False)[0]

        assert n_components(plain) == n_clusters
        assert n_components(joined) == 1
        added = _pairs(joined) - _pairs(plain)
        assert len(added) == n_clusters - 1
        mst = minimum_spanning_tree(_brute_distances(X)).tocoo()
        mst_pairs = {
            (min(i, j), max(i, j)) for i, j in zip(mst.row, mst.col, strict=True)
        }
        assert added <= mst_pairs
        # Bridges are weighted like the rest: heat at the k-NN median.
        sigma = jnp.median(kl.nearest_neighbors(X, 3).distances)
        D = _brute_distances(X)
        for i, j in added:
            e = int(
                np.flatnonzero(
                    (joined.topology.senders == i) & (joined.topology.receivers == j)
                )[0]
            )
            expected = np.exp(-(D[i, j] ** 2) / (2 * float(sigma) ** 2))
            assert np.isclose(joined.weights[e], expected)

    @pytest.mark.slow
    def test_ensure_connected_is_a_no_op_when_connected(self):
        X = _X()
        g = kl.knn_graph(X, 8)
        assert kl.knn_graph(X, 8, ensure_connected=True).topology == g.topology


class TestRadius:
    def test_radius_neighbors_pads_beyond_the_radius(self):
        X = _X(30)
        g = kl.radius_neighbors(X, 0.5, max_neighbors=6)
        idx = np.asarray(g.indices)
        dist = np.asarray(g.distances)
        assert np.all((idx == -1) == np.isinf(dist))
        assert np.all(dist[idx >= 0] <= 0.5)

    def test_radius_graph_is_the_pairs_within_radius(self):
        X = _X(30)
        g = kl.radius_graph(X, 0.6, max_neighbors=29, weighting="connectivity")
        D = _brute_distances(X)
        expected = {
            (i, j) for i in range(30) for j in range(i + 1, 30) if D[i, j] <= 0.6
        }
        assert _pairs(g) == expected

    def test_padding_never_becomes_an_edge(self):
        X = jnp.array([[0.0], [5.0], [10.0]])  # nobody within the radius
        knn = kl.radius_neighbors(X, 1.0, max_neighbors=2)
        assert kl.graph_from_neighbors(knn).topology.n_edges == 0
        assert np.array_equal(kl.adjacency_matrix(knn), np.zeros((3, 3)))

    def test_median_bandwidth_is_over_kept_edges(self):
        X = _X(30)
        g = kl.radius_graph(X, 0.6, max_neighbors=29)
        D = _brute_distances(X)
        s, r = g.topology.senders, g.topology.receivers
        knn = kl.radius_neighbors(X, 0.6, max_neighbors=29)
        kept = np.asarray(knn.distances)[np.asarray(knn.indices) >= 0]
        sigma = np.median(kept)
        assert np.allclose(g.weights, np.exp(-(D[s, r] ** 2) / (2 * sigma**2)))


class TestGridGraph:
    def test_spacing_sets_inverse_square_axis_weights(self):
        g = kl.grid_graph((3, 4), spacing=(0.5, 2.0), periodic=(False, True))
        assert np.allclose(g.axis_weights, [4.0, 0.25])
        assert g.periodic == (False, True)
        assert isinstance(g, kl.GridGraph)

    def test_scalar_and_default_spacing(self):
        assert np.allclose(kl.grid_graph((3, 3), spacing=0.5).axis_weights, [4.0, 4.0])
        assert np.allclose(kl.grid_graph((3, 3)).axis_weights, [1.0, 1.0])

    def test_rw1_is_the_path_laplacian(self):
        # gaussx's RW1 structure matrix is the 1-D grid Laplacian.
        L = kl.grid_graph((5,)).laplacian_operator().as_matrix()
        eye = jnp.eye(5)
        D = eye[1:] - eye[:-1]  # first differences
        assert np.allclose(L, einsum(D, D, "e i, e j -> i j"))

    @pytest.mark.parametrize(
        ("spacing", "match"), [((1.0,), "one entry per axis"), (0.0, "positive")]
    )
    def test_invalid_spacing(self, spacing, match):
        with pytest.raises(ValueError, match=match):
            kl.grid_graph((3, 3), spacing=spacing)


class TestGraphFromAdjacency:
    def test_round_trip(self):
        W = kl.adjacency_matrix(kl.nearest_neighbors(_X(), 4))
        g = kl.graph_from_adjacency(W)
        assert np.array_equal(np.asarray(g.to_dense()), np.asarray(W))

    def test_atol_and_diagonal(self):
        W = jnp.array([[5.0, 1.0, 1e-8], [1.0, 5.0, 0.0], [1e-8, 0.0, 5.0]])
        assert _pairs(kl.graph_from_adjacency(W)) == {(0, 1), (0, 2)}
        assert _pairs(kl.graph_from_adjacency(W, atol=1e-6)) == {(0, 1)}

    def test_validation(self):
        with pytest.raises(ValueError, match="symmetric"):
            kl.graph_from_adjacency(jnp.array([[0.0, 1.0], [0.0, 0.0]]))
        with pytest.raises(ValueError, match="square"):
            kl.graph_from_adjacency(jnp.ones((2, 3)))
        with pytest.raises(TypeError, match="eagerly"):
            jax.jit(lambda W: kl.graph_from_adjacency(W).weights)(jnp.eye(2))


class TestGraphFromEdges:
    def test_equals_graph_from_adjacency(self):
        W = kl.adjacency_matrix(kl.nearest_neighbors(_X(), 4))
        s, r = np.nonzero(np.triu(np.asarray(W), k=1))
        g = kl.graph_from_edges(s, r, W.shape[0], weights=W[s, r])
        h = kl.graph_from_adjacency(W)
        assert g.topology == h.topology
        assert np.array_equal(g.weights, h.weights)

    @pytest.mark.parametrize(
        ("symmetrize", "expected"), [("max", 4.0), ("min", 1.0), ("mean", 7.0 / 3.0)]
    )
    def test_duplicate_and_reversed_pairs_merge(self, symmetrize, expected):
        g = kl.graph_from_edges(
            [0, 2, 0, 1],
            [2, 0, 2, 1],
            3,
            weights=jnp.array([1.0, 4.0, 2.0, 9.0]),  # 1 -> 1 is a self-loop
            symmetrize=symmetrize,
        )
        assert _pairs(g) == {(0, 2)}
        assert np.isclose(g.weights[0], expected)

    def test_heat_distances_equal_knn_heat_weights(self):
        X = _X()
        knn_g = kl.knn_graph(X, 4)
        sigma = jnp.median(kl.nearest_neighbors(X, 4).distances)
        top = knn_g.topology
        d = jnp.asarray(_brute_distances(X)[top.senders, top.receivers])
        g = kl.graph_from_edges(
            top.senders,
            top.receivers,
            top.n_nodes,
            distances=d,
            weighting="heat",
            bandwidth=sigma,
        )
        assert g.topology == top
        assert np.allclose(g.weights, knn_g.weights)

    def test_distances_through_a_kernel_and_connectivity(self):
        d = jnp.array([0.5, 1.0])
        g = kl.graph_from_edges([0, 1], [1, 2], 3, distances=d, weighting=kl.RBF())
        assert np.allclose(g.weights, np.exp(-0.5 * np.asarray(d) ** 2))
        c = kl.graph_from_edges([0, 1], [1, 2], 3, distances=d)
        assert np.array_equal(c.weights, [1.0, 1.0])

    def test_median_bandwidth(self):
        d = jnp.array([1.0, 2.0, 3.0])
        g = kl.graph_from_edges([0, 1, 2], [1, 2, 3], 4, distances=d, weighting="heat")
        assert np.allclose(g.weights, np.exp(-(np.asarray(d) ** 2) / (2 * 2.0**2)))

    def test_no_weights_is_connectivity(self):
        g = kl.graph_from_edges([0], [1], 2)
        assert np.array_equal(g.weights, [1.0])

    @pytest.mark.parametrize(
        ("kwargs", "error", "match"),
        [
            ({"weights": [1.0], "distances": [1.0]}, ValueError, "not both"),
            ({"weights": [1.0], "weighting": "heat"}, ValueError, "used as given"),
            ({"weighting": "heat"}, ValueError, "needs distances"),
            ({"distances": [1.0], "weighting": "cosine"}, ValueError, "cosine"),
            (
                {"distances": [1.0], "weighting": "heat", "bandwidth": "local"},
                ValueError,
                "local",
            ),
            ({"weights": [1.0, 2.0]}, ValueError, "shape"),
            ({"symmetrize": "sum"}, ValueError, "symmetrize"),
        ],
    )
    def test_validation(self, kwargs, error, match):
        with pytest.raises(error, match=match):
            kl.graph_from_edges([0], [1], 2, **kwargs)

    def test_index_validation(self):
        with pytest.raises(ValueError, match="out of range"):
            kl.graph_from_edges([0], [2], 2)
        with pytest.raises(TypeError, match="integers"):
            kl.graph_from_edges([0.0], [1.0], 2)
        with pytest.raises(ValueError, match="equal length"):
            kl.graph_from_edges([0, 1], [1], 2)


class TestEdgeWeights:
    def test_kernel_on_the_endpoints(self):
        X = _X(12)
        g = kl.edge_weights(kl.grid_graph((3, 4)), X, kl.RBF(lengthscale=0.7))
        top = g.topology
        D = _brute_distances(X)
        expected = np.exp(-(D[top.senders, top.receivers] ** 2) / (2 * 0.7**2))
        assert np.allclose(g.weights, expected)

    def test_rbf_on_a_knn_topology_equals_heat(self):
        X = _X()
        heat = kl.knn_graph(X, 4, bandwidth=0.9)
        assert np.allclose(
            kl.edge_weights(heat, X, kl.RBF(lengthscale=0.9)).weights, heat.weights
        )

    def test_jit_and_grad_in_the_features(self):
        X = _X(12)
        grid = kl.grid_graph((3, 4))
        f = jnp.arange(12.0)

        @jax.jit
        def energy(X):
            return kl.edge_weights(grid, X, kl.RBF()).dirichlet_energy(f)

        grad = jax.grad(energy)(X)
        eps = 1e-6
        E = jnp.zeros_like(X).at[5, 1].set(1.0)
        fd = (energy(X + eps * E) - energy(X - eps * E)) / (2 * eps)
        assert np.isclose(grad[5, 1], fd, rtol=1e-5)

    def test_row_count_is_checked(self):
        with pytest.raises(ValueError, match="one row per node"):
            kl.edge_weights(kl.grid_graph((2, 2)), jnp.ones((3, 1)), kl.RBF())
