"""RootedTree / TreeSites and the tail-up / tail-down kernels (GEO16)."""

from __future__ import annotations

import math

import einx
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import kernellib as kl


# Y-shaped tree: root 0 - (length 2) - junction 1 - branches 2 (3) and 3 (4).
PARENT = [-1, 0, 1, 1]
LENGTH = [0.0, 2.0, 3.0, 4.0]
# Six sites: stem x2, branch 2 x2, branch 3 x2 (edge, offset from the node end).
EDGE = [1, 1, 2, 2, 3, 3]
OFFSET = [1.0, 0.0, 1.0, 2.0, 1.0, 3.0]


def _y_sites():
    return kl.RootedTree.from_parents(PARENT, LENGTH).sites(EDGE, OFFSET)


def _binary_tree(n_leaves, seed):
    rng = np.random.default_rng(seed)
    parent, leaves = [-1], [0]
    for _ in range(n_leaves - 1):
        v = leaves.pop(int(rng.integers(len(leaves))))
        for _ in range(2):
            parent.append(v)
            leaves.append(len(parent) - 1)
    return np.array(parent), rng.uniform(0.2, 2.0, len(parent))


def _multiway_tree(n, seed):
    rng = np.random.default_rng(seed)
    parent = np.array([-1] + [int(rng.integers(0, v)) for v in range(1, n)])
    return parent, rng.uniform(0.2, 2.0, n)


def _random_sites(parent, length, n_sites, seed, weights="leaf_count"):
    rng = np.random.default_rng(seed)
    tree = kl.RootedTree.from_parents(parent, length, weights)
    edge = rng.integers(1, len(parent), n_sites)
    offset = rng.uniform(0.0, 1.0, n_sites) * np.asarray(tree.length)[edge]
    return tree.sites(edge, offset)


def _idx(n):
    return einx.id("n -> n 1", jnp.arange(n))


class TestGeometry:
    def test_y_tree_hand_values(self):
        s = _y_sites()
        # Site depths t = 1, 2, 4, 3, 5, 3; junction at depth 2.
        h = np.array(
            [
                [0, 1, 3, 2, 4, 2],
                [1, 0, 2, 1, 3, 1],
                [3, 2, 0, 1, 5, 3],
                [2, 1, 1, 0, 4, 2],
                [4, 3, 5, 4, 0, 2],
                [2, 1, 3, 2, 2, 0],
            ],
            float,
        )
        conn = np.ones((6, 6), bool)
        for i in (2, 3):
            for j in (4, 5):
                conn[i, j] = conn[j, i] = False
        # unconnected pairs: (a, b) = distances to the junction, a <= b
        a = np.zeros((6, 6))
        b = h.copy()
        for (i, j), (aa, bb) in {
            (2, 4): (2, 3),
            (2, 5): (1, 2),
            (3, 4): (1, 3),
            (3, 5): (1, 1),
        }.items():
            a[i, j] = a[j, i] = aa
            b[i, j] = b[j, i] = bb
        r = math.sqrt(0.5)
        pi = np.where(conn, 1.0, 0.0)
        pi[:2, 2:] = r  # stem sites vs branch sites
        pi[2:, :2] = r
        assert np.allclose(s.h, h)
        assert np.allclose(s.a, a)
        assert np.allclose(s.b, b)
        assert np.array_equal(np.asarray(s.connected), conn)
        assert np.allclose(s.weight, pi)
        assert np.allclose(s.h, s.a + s.b)

    def test_ordering_a_le_b(self):
        parent, length = _binary_tree(30, 0)
        s = _random_sites(parent, length, 60, 1)
        assert bool(jnp.all(s.a <= s.b + 1e-12))
        assert np.allclose(s.h, einx.id("i j -> j i", s.h))

    def test_leaf_count_is_shreve_over_leaves(self):
        # Shreve order: leaves = 1, a junction = sum of its children.
        parent = [-1, 0, 0, 1, 1, 2, 5, 5, 5]
        t = kl.RootedTree.from_parents(parent, np.ones(9))
        shreve = np.array([5, 2, 3, 1, 1, 3, 1, 1, 1], float)
        # leaves: 3, 4, 6, 7, 8 -> 5 leaves
        assert np.allclose(t.weights, shreve / 5)
        assert float(t.weights[0]) == 1.0

    def test_uniform_weights(self):
        t = kl.RootedTree.from_parents([-1, 0, 0, 1, 1], np.ones(5), "uniform")
        assert np.allclose(t.weights, [1, 0.5, 0.5, 0.25, 0.25])

    def test_user_weights_accepted(self):
        t = kl.RootedTree.from_parents(PARENT, LENGTH, [1.0, 1.0, 0.25, 0.75])
        assert np.allclose(t.weights, [1.0, 1.0, 0.25, 0.75])

    def test_from_graph_equals_from_parents(self):
        parent, length = _multiway_tree(25, 3)
        senders = np.arange(1, 25)
        g = kl.graph_from_edges(
            senders, parent[1:], 25, weights=jnp.asarray(length[1:])
        )
        t_graph = kl.RootedTree.from_graph(g, root=0)
        t_par = kl.RootedTree.from_parents(parent, length)
        assert np.array_equal(t_graph.parent, t_par.parent)
        assert np.allclose(t_graph.length, t_par.length)
        assert np.allclose(t_graph.weights, t_par.weights)
        rng = np.random.default_rng(0)
        edge = rng.integers(1, 25, 20)
        off = rng.uniform(0, 1, 20) * length[edge]
        s1, s2 = t_graph.sites(edge, off), t_par.sites(edge, off)
        for name in ("h", "a", "b", "connected", "weight"):
            assert np.allclose(getattr(s1, name), getattr(s2, name))

    def test_from_graph_roots_elsewhere(self):
        g = kl.graph_from_edges([0, 1, 1], [1, 2, 3], 4)
        t = kl.RootedTree.from_graph(g, root=2)
        assert t.parent.tolist() == [1, 2, -1, 1]
        assert np.allclose(t.weights, [1 / 2, 1.0, 1.0, 1 / 2])


class TestInvalidInputs:
    def test_non_additive_weights(self):
        with pytest.raises(ValueError, match="not additive"):
            kl.RootedTree.from_parents(PARENT, LENGTH, [1.0, 1.0, 0.4, 0.4])

    @pytest.mark.parametrize(
        "w",
        [[0.9, 0.9, 0.45, 0.45], [1.0, 1.0, 0.0, 1.0], [1.0, 1.5, 0.75, 0.75]],
    )
    def test_weights_range_and_root(self, w):
        with pytest.raises(ValueError):
            kl.RootedTree.from_parents(PARENT, LENGTH, w)

    def test_weights_shape(self):
        with pytest.raises(ValueError, match="shape"):
            kl.RootedTree.from_parents(PARENT, LENGTH, [1.0, 1.0])

    def test_unknown_weight_rule(self):
        with pytest.raises(ValueError, match="leaf_count"):
            kl.RootedTree.from_parents(PARENT, LENGTH, "area")

    def test_two_roots(self):
        with pytest.raises(ValueError, match="exactly one root"):
            kl.RootedTree.from_parents([-1, -1, 0], [0.0, 0.0, 1.0])

    def test_no_root(self):
        with pytest.raises(ValueError, match="exactly one root"):
            kl.RootedTree.from_parents([1, 0], [1.0, 1.0])

    def test_cycle_in_parents(self):
        with pytest.raises(ValueError, match="cycle"):
            kl.RootedTree.from_parents([-1, 2, 1], [0.0, 1.0, 1.0])

    def test_negative_length(self):
        with pytest.raises(ValueError, match="non-negative"):
            kl.RootedTree.from_parents(PARENT, [0.0, -1.0, 1.0, 1.0])

    def test_from_graph_cycle(self):
        g = kl.graph_from_edges([0, 1, 2], [1, 2, 0], 3)
        with pytest.raises(ValueError, match=r"cycle.*matern_graph_kernel"):
            kl.RootedTree.from_graph(g, root=0)

    def test_from_graph_two_components(self):
        g = kl.graph_from_edges([0, 2], [1, 3], 4)
        with pytest.raises(ValueError, match="2 connected components"):
            kl.RootedTree.from_graph(g, root=0)

    def test_from_graph_grid_with_loops(self):
        with pytest.raises(ValueError, match="cycle"):
            kl.RootedTree.from_graph(kl.grid_graph((3, 3)), root=0)

    @pytest.mark.parametrize(
        "edge,offset",
        [([0], [0.0]), ([4], [0.0]), ([1], [2.5]), ([1], [-0.5])],
    )
    def test_bad_sites(self, edge, offset):
        t = kl.RootedTree.from_parents(PARENT, LENGTH)
        with pytest.raises(ValueError):
            t.sites(edge, offset)

    def test_bad_model(self):
        with pytest.raises(ValueError, match="model"):
            kl.TailUp(_y_sites(), "gaussian")
        with pytest.raises(ValueError, match="model"):
            kl.TailDown(_y_sites(), "mariah")


# Pinned values of Ver Hoef & Peterson (2010) Table 1 (and SSN2's cov_matrix.R),
# computed by hand at (h, a, b, alpha) with variance 1.
class TestPinnedValues:
    def _tu(self, model, h, alpha, pi=1.0):
        sites = kl.TreeSites(
            h=jnp.array([[0.0, h], [h, 0.0]]),
            a=jnp.zeros((2, 2)),
            b=jnp.array([[0.0, h], [h, 0.0]]),
            connected=jnp.ones((2, 2), bool),
            weight=jnp.array([[1.0, pi], [pi, 1.0]]),
        )
        return float(kl.TailUp(sites, model, range=alpha)(_idx(2), _idx(2))[0, 1])

    def _td(self, model, a, b, alpha):
        sites = kl.TreeSites(
            h=jnp.array([[0.0, a + b], [a + b, 0.0]]),
            a=jnp.array([[0.0, a], [a, 0.0]]),
            b=jnp.array([[0.0, b], [b, 0.0]]),
            connected=jnp.zeros((2, 2), bool),
            weight=jnp.eye(2),
        )
        return float(kl.TailDown(sites, model, range=alpha)(_idx(2), _idx(2))[0, 1])

    def test_tail_up(self):
        assert self._tu("exponential", 3.0, 4.0) == pytest.approx(math.exp(-0.75))
        assert self._tu("linear_sill", 3.0, 4.0) == pytest.approx(0.25)
        assert self._tu("linear_sill", 5.0, 4.0) == 0.0
        assert self._tu("spherical", 2.0, 4.0) == pytest.approx(0.3125)
        assert self._tu("spherical", 4.0, 4.0) == pytest.approx(0.0)
        assert self._tu("spherical", 4.5, 4.0) == 0.0
        assert self._tu("mariah", 3.0, 9.0) == pytest.approx(0.11446624, rel=1e-6)
        assert self._tu("mariah", 0.0, 9.0) == 1.0
        assert self._tu("spherical", 2.0, 4.0, pi=0.5) == pytest.approx(0.15625)

    def test_tail_down_connected_limit(self):
        # connected pair stored as (a, b) = (0, h): the connected formulas.
        assert self._td("exponential", 0.0, 3.0, 4.0) == pytest.approx(math.exp(-0.75))
        assert self._td("linear_sill", 0.0, 3.0, 4.0) == pytest.approx(0.25)
        assert self._td("spherical", 0.0, 2.0, 4.0) == pytest.approx(0.3125)

    def test_tail_down_unconnected(self):
        assert self._td("exponential", 1.0, 2.0, 4.0) == pytest.approx(math.exp(-0.75))
        assert self._td("linear_sill", 1.0, 2.0, 4.0) == pytest.approx(0.5)
        assert self._td("linear_sill", 1.0, 5.0, 4.0) == 0.0
        # (1 - 3*1/8 + 2/8) (1 - 2/4)^2 = 0.875 * 0.25
        assert self._td("spherical", 1.0, 2.0, 4.0) == pytest.approx(0.21875)
        assert self._td("spherical", 1.0, 4.5, 4.0) == 0.0

    def test_tail_down_continuity_at_a_equals_b_and_a_zero(self):
        # a = b = x: (1 - x/alpha)^3 ; a = 0: the connected spherical.
        assert self._td("spherical", 1.0, 1.0, 4.0) == pytest.approx(0.75**3)
        alpha, b = 4.0, 2.5
        assert self._td("spherical", 0.0, b, alpha) == pytest.approx(
            1 - 1.5 * b / alpha + 0.5 * (b / alpha) ** 3
        )


class TestKernelBehaviour:
    @pytest.mark.parametrize(
        "model", ["exponential", "linear_sill", "spherical", "mariah"]
    )
    def test_tail_up_zero_unconnected_and_diag(self, model):
        s = _y_sites()
        k = kl.TailUp(s, model, variance=2.5, range=6.0)
        K = k(_idx(6), _idx(6))
        assert np.all(np.asarray(K)[~np.asarray(s.connected)] == 0.0)
        assert np.allclose(jnp.diag(K), 2.5)
        assert np.allclose(k.diag(_idx(6)), 2.5)
        assert np.allclose(K, einx.id("i j -> j i", K))

    @pytest.mark.parametrize("model", ["exponential", "linear_sill", "spherical"])
    def test_tail_down_nonzero_unconnected_and_diag(self, model):
        s = _y_sites()
        k = kl.TailDown(s, model, variance=2.5, range=12.0)
        K = k(_idx(6), _idx(6))
        assert np.all(np.asarray(K)[~np.asarray(s.connected)] > 0.0)
        assert np.allclose(jnp.diag(K), 2.5)
        assert np.allclose(k.diag(_idx(6)), 2.5)
        assert np.allclose(K, einx.id("i j -> j i", K))

    def test_pairwise_matches_gram(self):
        s = _y_sites()
        for k in (
            kl.TailUp(s, "spherical", range=5.0),
            kl.TailDown(s, "spherical", range=5.0),
        ):
            K = k(_idx(6), _idx(6))
            assert float(k.pairwise(jnp.array([2]), jnp.array([4]))) == pytest.approx(
                float(K[2, 4])
            )
            assert k.is_pointwise

    def test_float_indices_and_rectangular(self):
        s = _y_sites()
        k = kl.TailUp(s, "exponential", range=4.0)
        K = k(jnp.array([[0.0], [2.0]]), jnp.array([[1.0], [3.0], [5.0]]))
        assert K.shape == (2, 3)
        with pytest.raises(ValueError, match="N, 1"):
            k(jnp.arange(3), jnp.arange(3))

    def test_gradients_finite(self):
        s = _y_sites()
        for k in (
            kl.TailUp(s, "mariah", range=4.0),
            kl.TailDown(s, "spherical", range=9.0),
        ):
            g = jax.grad(lambda kk: jnp.sum(kk(_idx(6), _idx(6))), allow_int=True)(k)
            assert np.isfinite(float(g.variance)) and np.isfinite(float(g.range))

    def test_jit(self):
        s = _y_sites()
        k = kl.TailUp(s, "spherical", range=5.0)
        out = eqx.filter_jit(lambda kk: kk(_idx(6), _idx(6)))(k)
        assert np.allclose(out, k(_idx(6), _idx(6)))

    def test_sum_with_white(self):
        s = _y_sites()
        k = kl.TailUp(s, "exponential", range=4.0) + kl.TailDown(
            s, "exponential", range=4.0
        )
        assert k(_idx(6), _idx(6)).shape == (6, 6)


def _min_eig(K):
    return float(jnp.linalg.eigvalsh(K)[0])


TU_MODELS = ["exponential", "linear_sill", "spherical", "mariah"]
TD_MODELS = ["exponential", "linear_sill", "spherical"]


class TestPositiveDefinite:
    @pytest.mark.parametrize("tree", ["binary", "multiway"])
    @pytest.mark.parametrize("weights", ["leaf_count", "uniform"])
    @pytest.mark.parametrize("rng", [0.7, 3.0, 25.0])
    def test_random_trees(self, tree, weights, rng):
        parent, length = (
            _binary_tree(100, 7) if tree == "binary" else _multiway_tree(120, 7)
        )
        s = _random_sites(parent, length, 200, 11, weights)
        if tree == "multiway":
            kids = np.bincount(parent[1:])
            assert kids.max() >= 3  # junctions with three or more children
        X = _idx(200)
        for m in TU_MODELS:
            K = kl.TailUp(s, m, variance=1.3, range=rng)(X, X)
            assert _min_eig(K) >= -1e-10 * 200, (m, "up")
        for m in TD_MODELS:
            K = kl.TailDown(s, m, variance=1.3, range=rng)(X, X)
            assert _min_eig(K) >= -1e-10 * 200, (m, "down")

    def test_sum_is_pd(self):
        parent, length = _binary_tree(60, 2)
        s = _random_sites(parent, length, 120, 5)
        X = _idx(120)
        K = kl.TailUp(s, "spherical", range=4.0)(X, X) + kl.TailDown(
            s, "exponential", range=4.0
        )(X, X)
        assert _min_eig(K) >= -1e-10 * 120
