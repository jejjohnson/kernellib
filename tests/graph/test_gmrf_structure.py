"""GMRF structure: null spaces, scaled structure matrices, mesh graphs, SPDE link."""

from __future__ import annotations

import gaussx as gx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import kernellib as kl
from kernellib._einx import einsum, reduce


def _constrained_variances(R, V):
    """diag of R^+ restricted to V^T x = 0 (the intrinsic GMRF's covariance)."""
    n = R.shape[0]
    P = jnp.eye(n) - einsum(V, V, "i c, j c -> i j")
    S = jnp.linalg.pinv(R)
    return jnp.diag(P @ S @ P)


def _triangular_lattice(rows=6, cols=6):
    """A mesh of near-equilateral triangles: every angle is 60 degrees."""
    pts = np.array(
        [
            [c + 0.5 * (r % 2), r * np.sqrt(3) / 2]
            for r in range(rows)
            for c in range(cols)
        ]
    )
    tris = []
    for r in range(rows - 1):
        for c in range(cols - 1):
            a, b = r * cols + c, r * cols + c + 1
            d, e = (r + 1) * cols + c, (r + 1) * cols + c + 1
            if r % 2 == 0:
                tris += [[a, b, d], [b, e, d]]
            else:
                tris += [[a, b, e], [a, e, d]]
    return jnp.asarray(pts), np.array(tris)


class TestNullSpace:
    @pytest.mark.parametrize(
        ("senders", "receivers", "n", "n_comp"),
        [
            ([0, 1, 2], [1, 2, 3], 4, 1),
            ([0, 2], [1, 3], 4, 2),
            ([0, 1, 3], [1, 2, 4], 7, 4),  # two blocks and two isolated nodes
        ],
    )
    def test_orthonormal_basis_of_the_kernel(self, senders, receivers, n, n_comp):
        g = kl.graph_from_edges(
            senders, receivers, n, weights=jnp.arange(1.0, 1.0 + len(senders))
        )
        V = kl.graph_null_space(g)
        assert V.shape == (n, n_comp) == (n, kl.n_components_graph(g))
        assert np.allclose(einsum(V, V, "i a, i b -> a b"), np.eye(n_comp))
        L = g.laplacian_operator().as_matrix()
        assert np.allclose(L @ V, 0.0)
        # It spans the whole kernel: rank(L) = n - n_comp.
        assert np.linalg.matrix_rank(np.asarray(L)) == n - n_comp

    def test_zero_weight_edges_split_components(self):
        g = kl.graph_from_edges(
            [0, 1, 2], [1, 2, 3], 4, weights=jnp.array([1.0, 0.0, 2.0])
        )
        V = kl.graph_null_space(g)
        assert V.shape == (4, 2)
        assert np.allclose(g.laplacian_operator().as_matrix() @ V, 0.0)

    def test_grid_dtype_follows_the_axis_weights(self):
        g = kl.GridGraph((3, 3), axis_weights=jnp.ones(2, dtype=jnp.float32))
        assert kl.graph_null_space(g).dtype == jnp.float32

    def test_grid(self):
        V = kl.graph_null_space(kl.grid_graph((3, 4)))
        assert np.allclose(V, 1.0 / np.sqrt(12))


class TestStructureMatrix:
    def test_unscaled_is_the_laplacian(self):
        g = kl.knn_graph(jax.random.normal(jax.random.key(0), (30, 2)), 4)
        R = kl.structure_matrix(g)
        assert isinstance(R, gx.SparseOperator)
        assert np.allclose(R.as_matrix(), g.laplacian_operator().as_matrix())
        assert R.T is R or np.allclose(R.as_matrix(), R.as_matrix().T)

    @pytest.mark.slow
    def test_scaled_has_unit_generalized_variance(self):
        g = kl.knn_graph(
            jax.random.normal(jax.random.key(1), (40, 2)), 5, ensure_connected=True
        )
        R = kl.structure_matrix(g, scaled=True).as_matrix()
        var = _constrained_variances(R, kl.graph_null_space(g))
        assert np.isclose(float(jnp.exp(jnp.mean(jnp.log(var)))), 1.0, rtol=1e-6)

    @pytest.mark.slow
    def test_each_component_is_scaled_separately(self):
        # A path of 3 and a triangle-plus-tail of 4, and an isolated node.
        g = kl.graph_from_edges(
            [0, 1, 3, 4, 3, 5], [1, 2, 4, 5, 5, 6], 8, weights=jnp.arange(1.0, 7.0)
        )
        R = kl.structure_matrix(g, scaled=True).as_matrix()
        V = kl.graph_null_space(g)
        var = _constrained_variances(R, V)
        for nodes in ([0, 1, 2], [3, 4, 5, 6]):
            gm = jnp.exp(jnp.mean(jnp.log(var[jnp.array(nodes)])))
            assert np.isclose(float(gm), 1.0, rtol=1e-6)
        assert np.allclose(R[7], 0.0)

    @pytest.mark.slow
    def test_zero_weight_edges_and_many_components(self):
        # Ten disjoint weighted pairs, a zero-weight edge joining two of
        # them, and an isolated node: every component is scaled to 1.
        s = [*range(0, 20, 2), 1]
        r = [*range(1, 20, 2), 2]
        w = jnp.concatenate([jnp.arange(1.0, 11.0), jnp.zeros(1)])
        g = kl.graph_from_edges(s, r, 21, weights=w)
        assert kl.n_components_graph(g) == 11
        R = kl.structure_matrix(g, scaled=True).as_matrix()
        var = _constrained_variances(R, kl.graph_null_space(g))
        for k in range(10):
            pair = jnp.array([2 * k, 2 * k + 1])
            gm = jnp.exp(jnp.mean(jnp.log(var[pair])))
            assert np.isclose(float(gm), 1.0, rtol=1e-6)

    def test_disconnected_grid_is_scaled_per_component(self):
        g = kl.GridGraph((3, 4), axis_weights=jnp.array([0.0, 1.0]))
        R = kl.structure_matrix(g, scaled=True)
        assert isinstance(R, gx.SparseOperator)  # rows: not one Kronecker block
        assert np.allclose(reduce(R.as_matrix(), "i j -> i", "sum"), 0.0)

    def test_grid_keeps_its_kronecker_structure(self):
        g = kl.grid_graph((5, 6), periodic=(False, True))
        R = kl.structure_matrix(g, scaled=True)
        assert isinstance(R, gx.KroneckerSum)
        var = _constrained_variances(R.as_matrix(), kl.graph_null_space(g))
        assert np.isclose(float(jnp.exp(jnp.mean(jnp.log(var)))), 1.0, rtol=1e-6)

    def test_rw1_matches_gaussx(self):
        R = kl.structure_matrix(kl.grid_graph((7,)))
        assert np.allclose(R.as_matrix(), gx.rw1_structure(7).as_matrix())


class TestMeshGraph:
    def test_connectivity(self):
        V, T = _triangular_lattice(3, 3)
        g = kl.mesh_graph(V, T)
        assert np.all(np.asarray(g.weights) == 1.0)
        # 3 + 3 + 3 horizontal/sloped rows of edges: every triangle side once.
        sides = {
            tuple(sorted(e))
            for t in T
            for e in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0]))
        }
        assert g.topology.n_edges == len(sides)

    def test_cotangent_weights_on_equilateral_triangles(self):
        V, T = _triangular_lattice(3, 3)
        g = kl.mesh_graph(V, T, weighting="cotangent")
        cot60 = 1.0 / np.sqrt(3)
        w = np.asarray(g.weights)
        # Interior edges see two 60-degree angles, boundary edges one.
        assert np.all(np.isclose(w, cot60) | np.isclose(w, 0.5 * cot60))

    def test_raises_on_a_non_delaunay_mesh(self):
        # Two flat triangles on the edge (0, 1): both opposite angles obtuse.
        V = jnp.array([[0.0, 0.0], [2.0, 0.0], [1.0, 0.2], [1.0, -0.2]])
        T = np.array([[0, 1, 2], [0, 3, 1]])
        with pytest.raises(ValueError, match=r"not Delaunay.*fem_matrices"):
            kl.mesh_graph(V, T, weighting="cotangent")
        # Connectivity weights do not care.
        assert kl.mesh_graph(V, T).topology.n_edges == 5

    def test_right_angles_give_exact_zeros(self):
        V = jnp.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        g = kl.mesh_graph(V, np.array([[0, 1, 2], [0, 2, 3]]), weighting="cotangent")
        diagonal = int(
            np.flatnonzero((g.topology.senders == 0) & (g.topology.receivers == 2))[0]
        )
        assert float(g.weights[diagonal]) == 0.0

    def test_checks_run_under_jit(self):
        V = jnp.array([[0.0, 0.0], [2.0, 0.0], [1.0, 0.2], [1.0, -0.2]])
        T = np.array([[0, 1, 2], [0, 3, 1]])

        @jax.jit
        def weights(V):
            return kl.mesh_graph(V, T, weighting="cotangent").weights

        with pytest.raises(Exception, match="not Delaunay"):
            jax.block_until_ready(weights(V))
        flat = jnp.zeros((4, 2))
        with pytest.raises(Exception, match="non-degenerate"):
            jax.block_until_ready(weights(flat))

    def test_float32_right_angles_are_not_rejected(self):
        # A translated right-triangle mesh in float32: the diagonal weights are
        # zero in exact arithmetic and rounding noise in float32.
        rows = np.array([[c, r] for r in range(4) for c in range(4)], dtype=np.float32)
        V = jnp.asarray(rows * 0.37 + np.float32(123.456), dtype=jnp.float32)
        tris = []
        for r in range(3):
            for c in range(3):
                a, b, d, e = (
                    4 * r + c,
                    4 * r + c + 1,
                    4 * (r + 1) + c,
                    4 * (r + 1) + c + 1,
                )
                tris += [[a, b, e], [a, e, d]]
        g = kl.mesh_graph(V, np.array(tris), weighting="cotangent")
        assert g.weights.dtype == jnp.float32
        assert np.all(np.asarray(g.weights) >= 0.0)

    def test_validation(self):
        V = jnp.zeros((3, 2))
        with pytest.raises(ValueError, match=r"\(T, 3\)"):
            kl.mesh_graph(V, np.array([[0, 1]]))
        with pytest.raises(ValueError, match="out of range"):
            kl.mesh_graph(V, np.array([[0, 1, 3]]))
        with pytest.raises(ValueError, match="degenerate"):
            kl.mesh_graph(V, np.array([[0, 1, 2]]), weighting="cotangent")
        with pytest.raises(ValueError, match="weighting"):
            kl.mesh_graph(jnp.eye(3)[:, :2], np.array([[0, 1, 2]]), weighting="uniform")

    def test_weights_are_differentiable_in_the_vertices(self):
        V, T = _triangular_lattice(3, 3)

        def total(V):
            return jnp.sum(kl.mesh_graph(V, T, weighting="cotangent").weights)

        grad = jax.grad(total)(V)
        E = jnp.zeros_like(V).at[4, 0].set(1.0)
        fd = (total(V + 1e-6 * E) - total(V - 1e-6 * E)) / 2e-6
        assert np.isclose(grad[4, 0], fd, rtol=1e-5, atol=1e-8)


@pytest.mark.integration
def test_cotangent_laplacian_is_the_fem_stiffness():
    V, T = _triangular_lattice(6, 7)
    g = kl.mesh_graph(V, T, weighting="cotangent")
    _, G = gx.fem_matrices(V, T)
    assert np.allclose(g.laplacian_operator().as_matrix(), G.as_matrix(), atol=1e-12)


@pytest.mark.integration
def test_graph_matern_matches_the_grid_spde():
    # nu_graph = alpha and 2 nu / l^2 = kappa^2, both normalised to average
    # marginal variance 1, on the unnormalised Laplacian (roadmap 5.2).
    alpha, kappa = 2, 0.5
    grid = kl.grid_graph((32, 32))
    K = kl.matern_graph_kernel(
        grid,
        nu=alpha,
        lengthscale=np.sqrt(2 * alpha) / kappa,
        normalization="unnormalized",
    )
    graph_var = jnp.diag(K)
    Q = gx.spde_precision_grid((32, 32), kappa, 1.0, alpha)
    spde_var = Q.diag_inv()
    spde_var = spde_var / jnp.mean(spde_var)
    assert np.allclose(graph_var, spde_var, rtol=1e-8)
