"""laplacian_eigpairs (dense, Kronecker, Lanczos, ARPACK) and n_components_graph."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import kernellib as kl
from kernellib._einx import einsum


def _projector(U):
    return einsum(U, U, "i k, j k -> i j")


def _gap_n(lam, n):
    """The largest n' <= n with lam[n' - 1] < lam[n'] (no split eigenspace)."""
    lam = np.asarray(lam)
    while n > 1 and np.isclose(lam[n - 1], lam[n], atol=1e-9):
        n -= 1
    return n


def _geometric_graph(n=300, seed=0):
    X = jax.random.uniform(jax.random.key(seed), (n, 2))
    return kl.knn_graph(X, 8, weighting="connectivity", ensure_connected=True)


def _assert_signs_fixed(U):
    U = np.asarray(U)
    peak = np.argmax(np.abs(U), axis=0)
    assert np.all(U[peak, np.arange(U.shape[1])] > 0)


class TestKronecker:
    @pytest.mark.parametrize(
        ("shape", "periodic", "axis_weights"),
        [
            ((6, 7), False, None),
            ((5, 5), (False, True), None),
            ((4, 6), True, (0.5, 2.0)),
            ((3, 4, 5), (True, False, False), (1.0, 2.0, 0.5)),
        ],
    )
    def test_agrees_with_dense(self, shape, periodic, axis_weights):
        g = kl.GridGraph(shape, periodic=periodic, axis_weights=axis_weights)
        N = g.n_nodes
        full_lam, _ = kl.laplacian_eigpairs(g, N, method="dense")
        n = _gap_n(full_lam, 12)
        lam_k, U_k = kl.laplacian_eigpairs(g, n)
        lam_d, U_d = kl.laplacian_eigpairs(g, n, method="dense")
        assert np.allclose(lam_k, lam_d, atol=1e-10)
        # Degenerate eigenvalues: compare the spanned subspaces, not vectors.
        assert np.allclose(_projector(U_k), _projector(U_d), atol=1e-8)
        assert np.allclose(einsum(U_k, U_k, "i a, i b -> a b"), np.eye(n), atol=1e-10)

    def test_closed_form_path_eigenvalues(self):
        lam, _ = kl.laplacian_eigpairs(kl.grid_graph((50,)), 5)
        k = np.arange(5)
        assert np.allclose(lam, 2 - 2 * np.cos(np.pi * k / 50), atol=1e-12)

    def test_is_the_default_for_face_grids_only(self):
        g = kl.grid_graph((6, 6))
        lam_default, _ = kl.laplacian_eigpairs(g, 3)
        lam_kron, _ = kl.laplacian_eigpairs(g, 3, method="kronecker")
        assert np.array_equal(lam_default, lam_kron)
        full = kl.grid_graph((4, 4), connectivity="full")
        lam, _ = kl.laplacian_eigpairs(full, 3)  # dense by default
        assert lam.shape == (3,)

    @pytest.mark.parametrize(
        ("graph", "normalization"),
        [
            (kl.grid_graph((4, 4)), "symmetric"),
            (kl.grid_graph((4, 4), connectivity="full"), "unnormalized"),
            (kl.graph_from_edges([0], [1], 2), "unnormalized"),
        ],
    )
    def test_kronecker_rejects_other_graphs(self, graph, normalization):
        with pytest.raises(ValueError, match="kronecker"):
            kl.laplacian_eigpairs(
                graph, 1, normalization=normalization, method="kronecker"
            )


class TestDenseAndSparse:
    @pytest.mark.parametrize("normalization", ["unnormalized", "symmetric"])
    def test_dense_matches_eigh_of_graph_laplacian(self, normalization):
        g = _geometric_graph(60)
        lam, U = kl.laplacian_eigpairs(g, 6, normalization=normalization)
        L = kl.graph_laplacian(g.to_dense(), normalization)
        assert np.allclose(lam, np.linalg.eigvalsh(np.asarray(L))[:6], atol=1e-10)
        assert np.allclose(L @ U, U * lam, atol=1e-9)

    def test_dense_adjacency_array_input(self):
        g = _geometric_graph(40)
        lam_a, U_a = kl.laplacian_eigpairs(g.to_dense(), 5)
        lam_g, U_g = kl.laplacian_eigpairs(g, 5)
        assert np.allclose(lam_a, lam_g)
        assert np.allclose(U_a, U_g, atol=1e-8)

    @pytest.mark.parametrize("normalization", ["unnormalized", "symmetric"])
    def test_arpack_agrees_with_dense(self, normalization):
        g = _geometric_graph(200)
        n = _gap_n(kl.laplacian_eigpairs(g, 9, normalization=normalization)[0], 8)
        lam_a, U_a = kl.laplacian_eigpairs(
            g, n, normalization=normalization, method="arpack"
        )
        lam_d, U_d = kl.laplacian_eigpairs(g, n, normalization=normalization)
        assert np.allclose(lam_a, lam_d, atol=1e-8)
        assert np.allclose(_projector(U_a), _projector(U_d), atol=1e-6)

    @pytest.mark.slow
    def test_lanczos_agrees_with_dense_on_a_geometric_graph(self):
        g = _geometric_graph(1500)
        n = _gap_n(kl.laplacian_eigpairs(g, 11)[0], 10)
        lam_l, U_l = kl.laplacian_eigpairs(
            g, n, method="lanczos", key=jax.random.key(0)
        )
        lam_d, U_d = kl.laplacian_eigpairs(g, n)
        assert np.allclose(lam_l, lam_d, rtol=1e-4, atol=1e-6)
        assert np.allclose(_projector(U_l), _projector(U_d), atol=1e-3)

    @pytest.mark.slow
    def test_lanczos_small_graph(self):
        g = _geometric_graph(80)
        lam_l, _ = kl.laplacian_eigpairs(g, 4, method="lanczos", key=jax.random.key(0))
        lam_d, _ = kl.laplacian_eigpairs(g, 4)
        assert np.allclose(lam_l, lam_d, atol=1e-8)

    @pytest.mark.slow
    def test_dense_is_differentiable_in_the_weights(self):
        g = _geometric_graph(30)

        def fiedler(w):
            return kl.laplacian_eigpairs(g.reweight(w), 2)[0][1]

        w = g.weights
        grad = jax.grad(fiedler)(w)
        e = jnp.zeros_like(w).at[0].set(1.0)
        fd = (fiedler(w + 1e-6 * e) - fiedler(w - 1e-6 * e)) / 2e-6
        assert np.isclose(grad[0], fd, rtol=1e-4, atol=1e-8)


class TestConventions:
    @pytest.mark.parametrize("method", ["dense", "kronecker", "arpack"])
    def test_sign_convention(self, method):
        g = kl.grid_graph((6, 5))
        _, U = kl.laplacian_eigpairs(g, 6, method=method)
        _assert_signs_fixed(U)
        _, U2 = kl.laplacian_eigpairs(g, 6, method=method)
        assert np.array_equal(U, U2)

    @pytest.mark.parametrize("method", ["dense", "arpack"])
    def test_zero_eigenvalues_count_components(self, method):
        # Three blocks plus two isolated nodes: 5 components.
        g = kl.graph_from_edges([0, 1, 0, 3, 4, 6, 7], [1, 2, 2, 4, 5, 7, 8], 11)
        assert kl.n_components_graph(g) == 5
        lam, _ = kl.laplacian_eigpairs(g, 7, method=method)
        assert int(np.sum(np.abs(np.asarray(lam)) < 1e-9)) == 5
        assert np.asarray(lam)[5] > 1e-3


class TestNComponents:
    def test_graphs_arrays_and_grids(self):
        assert kl.n_components_graph(kl.grid_graph((30, 30))) == 1
        g = kl.knn_graph(
            jnp.array([[0.0], [0.1], [10.0], [10.1]]), 1, weighting="connectivity"
        )
        assert kl.n_components_graph(g) == 2
        assert kl.n_components_graph(g.to_dense()) == 2
        assert kl.n_components_graph(jnp.zeros((4, 4))) == 4


class TestValidation:
    def test_invalid_arguments(self):
        g = _geometric_graph(30)
        with pytest.raises(ValueError, match="n must be"):
            kl.laplacian_eigpairs(g, 0)
        with pytest.raises(ValueError, match="n must be"):
            kl.laplacian_eigpairs(g, 31)
        with pytest.raises(ValueError, match="normalization"):
            kl.laplacian_eigpairs(g, 2, normalization="random_walk")
        with pytest.raises(ValueError, match="method"):
            kl.laplacian_eigpairs(g, 2, method="lobpcg")
        with pytest.raises(ValueError, match="PRNG key"):
            kl.laplacian_eigpairs(g, 2, method="lanczos")
        with pytest.raises(ValueError, match="n < N"):
            kl.laplacian_eigpairs(g, 30, method="arpack")
