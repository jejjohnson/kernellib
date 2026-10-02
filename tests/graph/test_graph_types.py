"""Sparse graph types: topology, operators and the lattice Kronecker sum."""

from __future__ import annotations

import functools as ft
import itertools

import equinox as eqx
import gaussx as gx
import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
import pytest

import kernellib as kl
from kernellib._einx import einsum, rearrange


NORMALIZATIONS = ["unnormalized", "symmetric", "random_walk"]


def _graph_from_dense(W) -> kl.Graph:
    """The `Graph` of a dense symmetric adjacency matrix (test oracle only)."""
    W = np.asarray(W)
    s, r = np.nonzero(np.triu(W, k=1))
    return kl.Graph(kl.GraphTopology(s, r, W.shape[0]), jnp.asarray(W[s, r]))


@ft.cache
def _knn_graph() -> tuple[kl.Graph, np.ndarray]:
    """A heat-weighted 4-NN graph on 30 points, and its dense adjacency."""
    X = np.asarray(jax.random.normal(jax.random.key(0), (30, 2)))
    D = np.sqrt(((X[:, None] - X[None]) ** 2).sum(-1))
    np.fill_diagonal(D, np.inf)
    idx = np.argsort(D, axis=1)[:, :4]
    knn = kl.KNNGraph(jnp.asarray(idx), jnp.asarray(np.take_along_axis(D, idx, 1)))
    W = np.asarray(kl.adjacency_matrix(knn))
    return _graph_from_dense(W), W


def _dense_grid_laplacian(shape, periodic, axis_weights, connectivity="face"):
    """Dense Laplacian of a lattice from explicit neighbour loops."""
    n = int(np.prod(shape))
    W = np.zeros((n, n))
    for here in itertools.product(*(range(m) for m in shape)):
        for delta in itertools.product((-1, 0, 1), repeat=len(shape)):
            steps = np.flatnonzero(delta)
            if steps.size == 0 or (connectivity == "face" and steps.size > 1):
                continue
            there = []
            for k, (c, d) in enumerate(zip(here, delta, strict=True)):
                t = c + d
                if periodic[k]:
                    t %= shape[k]
                elif not 0 <= t < shape[k]:
                    break
                there.append(t)
            else:
                i = np.ravel_multi_index(here, shape)
                j = np.ravel_multi_index(tuple(there), shape)
                W[i, j] = np.mean(np.asarray(axis_weights)[steps])
    return np.diag(W.sum(1)) - W


def _vertical_energy(image):
    return jnp.sum((image[1:] - image[:-1]) ** 2)


def _horizontal_energy(image):
    return jnp.sum((image[:, 1:] - image[:, :-1]) ** 2)


class TestGraphTopology:
    def test_validates_edges(self):
        with pytest.raises(ValueError, match="senders < receivers"):
            kl.GraphTopology([1], [0], 2)
        with pytest.raises(ValueError, match="senders < receivers"):
            kl.GraphTopology([1], [1], 2)
        with pytest.raises(ValueError, match="duplicates"):
            kl.GraphTopology([0, 0], [1, 1], 2)
        with pytest.raises(ValueError, match="out of range"):
            kl.GraphTopology([0], [2], 2)
        with pytest.raises(ValueError, match="equal length"):
            kl.GraphTopology([0, 1], [1], 3)
        with pytest.raises(TypeError, match="integers"):
            kl.GraphTopology([0.0], [1.0], 2)

    def test_traced_indices_are_rejected(self):
        with pytest.raises(TypeError, match="concrete host arrays"):
            jax.jit(lambda s: kl.GraphTopology(s, s + 1, 3))(jnp.array([0]))

    def test_hash_and_equality_by_content(self):
        a = kl.GraphTopology(np.array([0, 1]), np.array([1, 2]), 3)
        b = kl.GraphTopology([0, 1], [1, 2], 3)
        assert a == b and hash(a) == hash(b)
        assert a != kl.GraphTopology([0, 1], [1, 2], 4)
        assert a != kl.GraphTopology([0], [1], 3)
        with pytest.raises(AttributeError):
            a.n_nodes = 4
        assert not a.senders.flags.writeable

    def test_no_edges(self):
        g = kl.Graph(kl.GraphTopology([], [], 3), jnp.zeros(0))
        assert np.array_equal(g.to_dense(), np.zeros((3, 3)))
        assert np.array_equal(g.laplacian_operator().as_matrix(), np.zeros((3, 3)))
        assert g.incidence_operator().as_matrix().shape == (0, 3)


class TestGraph:
    def test_weights_shape_is_checked(self):
        with pytest.raises(ValueError, match="weights must have shape"):
            kl.Graph(kl.GraphTopology([0], [1], 2), jnp.ones(2))

    def test_integer_weights_become_float(self):
        g = kl.Graph(kl.GraphTopology([0], [1], 2), jnp.array([2]))
        assert jnp.issubdtype(g.weights.dtype, jnp.floating)

    def test_dense_and_bcoo_agree_with_adjacency_matrix(self):
        g, W = _knn_graph()
        assert np.allclose(g.to_dense(), W)
        assert np.allclose(g.to_bcoo().todense(), W)
        assert np.allclose(g.adjacency_operator().as_matrix(), W)
        assert np.allclose(g.degree(), np.diag(kl.graph_laplacian(W)))

    def test_edges_round_trip(self):
        g, _ = _knn_graph()
        s, r, w = g.edges()
        assert np.all(np.asarray(s) < np.asarray(r))
        assert np.array_equal(w, g.weights)

    @pytest.mark.parametrize("normalization", NORMALIZATIONS)
    def test_laplacian_matches_dense(self, normalization):
        g, W = _knn_graph()
        L = g.laplacian_operator(normalization)
        dense = kl.graph_laplacian(W, normalization)
        assert np.allclose(L.as_matrix(), dense)
        f = jax.random.normal(jax.random.key(1), (30,))
        assert np.allclose(L.mv(f), dense @ f)

    @pytest.mark.parametrize("normalization", NORMALIZATIONS)
    def test_isolated_node_matches_dense(self, normalization):
        W = jnp.zeros((4, 4)).at[0, 1].set(2.0).at[1, 0].set(2.0)
        W = W.at[1, 2].set(1.0).at[2, 1].set(1.0)  # node 3 is isolated
        L = _graph_from_dense(W).laplacian_operator(normalization)
        assert np.allclose(L.as_matrix(), kl.graph_laplacian(W, normalization))

    def test_laplacian_tags(self):
        g, _ = _knn_graph()
        for normalization in ("unnormalized", "symmetric"):
            L = g.laplacian_operator(normalization)
            assert lx.is_symmetric(L) and lx.is_positive_semidefinite(L)
        L_rw = g.laplacian_operator("random_walk")
        assert not lx.is_symmetric(L_rw)
        assert not lx.is_positive_semidefinite(L_rw)

    def test_unknown_normalization(self):
        g, _ = _knn_graph()
        with pytest.raises(ValueError, match="normalization"):
            g.laplacian_operator("bogus")

    def test_pattern_is_shared_across_reweightings(self):
        g, _ = _knn_graph()
        h = g.reweight(2.0 * g.weights)
        assert h.topology is g.topology
        assert h.laplacian_operator().pattern is g.laplacian_operator().pattern
        assert np.allclose(
            h.laplacian_operator().as_matrix(),
            2.0 * g.laplacian_operator().as_matrix(),
        )

    def test_incidence_gram_is_laplacian(self):
        g, _ = _knn_graph()
        B = g.incidence_operator().as_matrix()
        assert B.shape == (g.topology.n_edges, 30)
        assert np.allclose(
            einsum(B, B, "e i, e j -> i j"), g.laplacian_operator().as_matrix()
        )

    def test_dirichlet_energy(self):
        g, _ = _knn_graph()
        L = g.laplacian_operator().as_matrix()
        f = jax.random.normal(jax.random.key(2), (30,))
        assert np.allclose(g.dirichlet_energy(f), f @ L @ f)
        F = jax.random.normal(jax.random.key(3), (30, 2, 3))
        expected = einsum(F, L, F, "n a b, n m, m a b -> a b")
        assert np.allclose(g.dirichlet_energy(F), expected)

    def test_energy_gradient_matches_finite_differences(self):
        g, _ = _knn_graph()
        f = jax.random.normal(jax.random.key(4), (30,))

        def energy(w):
            return g.reweight(w).dirichlet_energy(f)

        def quadratic(w):
            # Through the sparse operator, to differentiate its values too.
            return f @ g.reweight(w).laplacian_operator().mv(f)

        w = g.weights
        eps = 1e-6
        e = jnp.zeros_like(w).at[3].set(1.0)
        fd = (energy(w + eps * e) - energy(w - eps * e)) / (2 * eps)
        grad = jax.jit(jax.grad(energy))(w)
        assert np.isclose(grad[3], fd, rtol=1e-6)
        assert np.allclose(jax.jit(jax.grad(quadratic))(w), grad)

    def test_jit_and_vmap_over_weights(self):
        g, _ = _knn_graph()
        f = jnp.ones(30).at[0].set(2.0)
        weights = jnp.stack([g.weights, 3.0 * g.weights])

        @jax.jit
        def energy(graph):
            return graph.laplacian_operator().mv(f) @ f

        batched = jax.vmap(lambda w: energy(g.reweight(w)))(weights)
        assert np.allclose(batched[1], 3.0 * batched[0])
        assert np.isclose(batched[0], g.dirichlet_energy(f))

    def test_topology_is_static(self):
        g, _ = _knn_graph()
        leaves = jax.tree_util.tree_leaves(g)
        assert len(leaves) == 1 and leaves[0] is g.weights
        assert eqx.tree_equal(g, g.reweight(g.weights))


class TestGridGraph:
    @pytest.mark.parametrize(
        ("shape", "periodic", "axis_weights"),
        [
            ((4, 5), (False, False), (1.0, 1.0)),
            ((4, 5), (True, True), (1.0, 1.0)),
            ((4, 5), (False, True), (1.0, 1.0)),
            ((4, 5), (True, False), (0.5, 2.0)),
            ((3, 4, 3), (False, True, False), (1.0, 0.25, 3.0)),
            ((6,), (True,), (2.0,)),
        ],
    )
    def test_kronecker_laplacian_matches_dense(self, shape, periodic, axis_weights):
        g = kl.GridGraph(shape, periodic=periodic, axis_weights=jnp.array(axis_weights))
        L = g.laplacian_operator()
        dense = _dense_grid_laplacian(shape, periodic, axis_weights)
        if len(shape) > 1:
            assert isinstance(L, gx.KroneckerSum)
        assert lx.is_symmetric(L) and lx.is_positive_semidefinite(L)
        f = jax.random.normal(jax.random.key(0), (g.n_nodes,))
        matrix, mv = jax.jit(lambda L: (L.as_matrix(), L.mv(f)))(L)
        assert np.allclose(matrix, dense)
        assert np.allclose(mv, dense @ f)
        # The sparse path over the explicit edge list agrees.
        sparse = kl.AbstractGraph.laplacian_operator(g)
        assert isinstance(sparse, gx.SparseOperator)
        assert np.allclose(sparse.as_matrix(), dense)

    @pytest.mark.parametrize("periodic", [False, (True, False)])
    def test_full_connectivity_matches_dense(self, periodic):
        g = kl.GridGraph(
            (3, 4), connectivity="full", periodic=periodic, axis_weights=[1.0, 3.0]
        )
        L = g.laplacian_operator()
        assert isinstance(L, gx.SparseOperator)
        flags = (periodic,) * 2 if isinstance(periodic, bool) else periodic
        dense = _dense_grid_laplacian((3, 4), flags, (1.0, 3.0), "full")
        assert np.allclose(L.as_matrix(), dense)

    def test_full_connectivity_neighbour_counts(self):
        assert np.array_equal(
            kl.GridGraph((3, 3), connectivity="full").degree(),
            [3, 5, 3, 5, 8, 5, 3, 5, 3],
        )
        assert kl.GridGraph((3, 3, 3), connectivity="full").degree()[13] == 26

    @pytest.mark.parametrize("normalization", ["symmetric", "random_walk"])
    def test_normalized_laplacians_are_sparse(self, normalization):
        g = kl.GridGraph((3, 4), periodic=(False, True))
        L = g.laplacian_operator(normalization)
        assert isinstance(L, gx.SparseOperator)
        assert np.allclose(
            L.as_matrix(), kl.graph_laplacian(g.to_dense(), normalization)
        )

    def test_row_major_node_order(self):
        # Node (i, j) of an (H, W) grid is i * W + j: right neighbours are +1.
        g = kl.GridGraph((2, 3))
        s, r, _ = g.edges()
        pairs = set(zip(np.asarray(s).tolist(), np.asarray(r).tolist(), strict=True))
        assert pairs == {(0, 1), (1, 2), (3, 4), (4, 5), (0, 3), (1, 4), (2, 5)}

    def test_dirichlet_energy_of_an_image(self):
        image = jax.random.normal(jax.random.key(0), (5, 6))
        g = kl.GridGraph(image.shape)
        expected = _vertical_energy(image) + _horizontal_energy(image)
        f = rearrange(image, "h w -> (h w)")
        assert np.isclose(g.dirichlet_energy(f), expected)

    def test_reweight_returns_a_graph(self):
        g = kl.GridGraph((3, 3))
        h = g.reweight(2.0 * g.weights)
        assert isinstance(h, kl.Graph)
        assert h.topology is g.topology
        assert np.allclose(h.to_dense(), 2.0 * g.to_dense())

    def test_axis_weights_are_differentiable(self):
        f = jax.random.normal(jax.random.key(0), (12,))

        def quadratic(a):
            return f @ kl.GridGraph((3, 4), axis_weights=a).laplacian_operator().mv(f)

        a = jnp.array([0.5, 2.0])
        grad = jax.jit(jax.grad(quadratic))(a)
        image = rearrange(f, "(h w) -> h w", h=3)
        expected = [_vertical_energy(image), _horizontal_energy(image)]
        assert np.allclose(grad, expected)

    def test_single_cell_axis(self):
        g = kl.GridGraph((1, 4))
        dense = _dense_grid_laplacian((1, 4), (False, False), (1.0, 1.0))
        assert np.allclose(g.laplacian_operator().as_matrix(), dense)

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"shape": ()}, "non-empty"),
            ({"shape": (0, 3)}, "positive"),
            ({"shape": (3, 3), "connectivity": "edge"}, "connectivity"),
            ({"shape": (3, 3), "periodic": (True,)}, "one entry per axis"),
            ({"shape": (2, 3), "periodic": True}, "at least 3 cells"),
            ({"shape": (3, 3), "axis_weights": [1.0]}, "axis_weights"),
        ],
    )
    def test_validation(self, kwargs, match):
        shape = kwargs.pop("shape")
        with pytest.raises(ValueError, match=match):
            kl.GridGraph(shape, **kwargs)

    @pytest.mark.slow
    def test_large_lattice_stays_implicit(self):
        # 10^6 nodes: the Kronecker sum needs only the two 1-D factors.
        g = kl.GridGraph((1000, 1000))
        L = g.laplacian_operator()
        y = L.mv(jnp.ones(g.n_nodes))
        assert np.allclose(y, 0.0)
