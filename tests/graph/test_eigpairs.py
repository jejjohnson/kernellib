"""laplacian_eigpairs (dense, Kronecker, Lanczos, ARPACK) and n_components_graph."""

from __future__ import annotations

import einx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import kernellib as kl
from kernellib._einx import einsum, rearrange, reduce


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

    @pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
    @pytest.mark.parametrize(
        ("shape", "periodic", "axis_weights"),
        [
            ((7, 9), False, (1.0, 1.0)),
            ((6, 8), (False, True), (0.5, 3.0)),
            ((4, 5, 6), (True, False, True), (1.0, 2.0, 0.5)),
            ((3, 6, 4), False, (1.0, 1.0, 1.0)),
        ],
    )
    def test_closed_form_matches_dense_eigh(self, shape, periodic, axis_weights, dtype):
        g = kl.GridGraph(
            shape, periodic=periodic, axis_weights=jnp.asarray(axis_weights, dtype)
        )
        N = g.n_nodes
        lam, U = kl.laplacian_eigpairs(g, N, method="kronecker")
        assert lam.dtype == dtype
        assert U.dtype == dtype
        L = np.asarray(g.laplacian_operator().as_matrix(), np.float64)
        lam_ref, U_ref = np.linalg.eigh(L)
        tol = 1e-5 if dtype == jnp.float32 else 1e-12
        lam, U = np.asarray(lam, np.float64), np.asarray(U, np.float64)
        assert np.allclose(lam, lam_ref, atol=tol * lam_ref[-1])
        # Compare the eigenspace of each cluster of equal eigenvalues.
        breaks = np.flatnonzero(np.diff(lam_ref) > 1e-8 * lam_ref[-1]) + 1
        for idx in np.split(np.arange(N), breaks):
            assert np.allclose(
                _projector(U[:, idx]), _projector(U_ref[:, idx]), atol=10 * tol
            )

    def test_closed_form_large_path(self):
        # n (2n + 1) >= 2**31: the phase is reduced by an exact mulmod.
        n = 40_000
        lam, U = kl.laplacian_eigpairs(kl.grid_graph((n,)), 4)
        k = np.arange(4)
        assert np.allclose(lam, 4 * np.sin(np.pi * k / (2 * n)) ** 2, rtol=1e-12)
        i = np.arange(n)
        ref = np.sqrt(2.0 / n) * np.cos(
            np.pi * einx.multiply("i, k -> i k", i + 0.5, k) / n
        )
        ref[:, 0] = 1.0 / np.sqrt(n)
        assert np.allclose(np.abs(U), np.abs(ref), atol=1e-10)

    @pytest.mark.slow
    @pytest.mark.parametrize("periodic", [True, False])
    def test_float32_large_axis_is_accurate(self, periodic):
        # Cycle: k = n - 1 is among the 3 smallest; unfolded, its float32
        # eigenvalue was off by 1e-3 and its phase reached ~1.6e9.
        n = 40_000
        g = kl.GridGraph((n,), periodic=periodic, axis_weights=jnp.ones(1, jnp.float32))
        lam, U = kl.laplacian_eigpairs(g, 3)
        assert lam.dtype == U.dtype == jnp.float32
        k = np.arange(n)
        if periodic:
            ref_all = 4 * np.sin(np.pi * np.minimum(k, n - k) / n) ** 2
        else:
            ref_all = 4 * np.sin(np.pi * k / (2 * n)) ** 2
        ref = np.sort(ref_all)[:3]
        lam = np.asarray(lam, np.float64)
        assert abs(lam[0]) < 1e-12
        assert np.max(np.abs(lam[1:] - ref[1:]) / ref[1:]) <= 1e-5
        # Residual against the float64 operator, at float32 rounding level.
        U = np.asarray(U, np.float64)
        L = kl.GridGraph((n,), periodic=periodic).laplacian_operator()
        LU = np.asarray(jax.vmap(L.mv, in_axes=1, out_axes=1)(jnp.asarray(U)))
        err = LU - einx.multiply("i m, m -> i m", U, ref)
        assert np.sqrt(np.max(reduce(err**2, "i m -> m", "sum"))) < 1e-6
        assert np.allclose(einsum(U, U, "i a, i b -> a b"), np.eye(3), atol=1e-5)

    @pytest.mark.slow
    @pytest.mark.parametrize("periodic", [True, False])
    def test_high_frequency_columns_in_int32(self, periodic):
        # The highest frequencies of a long axis, whose products k (2i + 1)
        # overflow int32, against a float64 reference.
        from kernellib._graph._eigpairs import _lattice_eigvecs

        n = 40_000
        k = np.array([1, n // 2 - 1, n // 2 + 1, n - 2, n - 1])
        U = np.asarray(
            _lattice_eigvecs(n, periodic, jnp.asarray(k, jnp.int32), jnp.float32),
            np.float64,
        )
        i = np.arange(n)
        if periodic:
            m = np.minimum(k, n - k)
            phase = 2 * np.pi * np.mod(einx.multiply("i, m -> i m", i, m), n) / n
            ref = np.where(2 * k > n, np.sin(phase), np.cos(phase))
        else:
            ref = np.cos(np.pi * einx.multiply("i, m -> i m", i + 0.5, k) / n)
        ref = ref * np.sqrt(2.0 / n)
        assert np.allclose(U, ref, atol=1e-6)

    def test_mulmod_is_exact_without_overflow(self):
        from kernellib._graph._eigpairs import _mulmod

        p = 4 * 50_000_000
        rng = np.random.default_rng(0)
        a = rng.integers(0, p, 64)
        b = rng.integers(0, p, 16)
        got = _mulmod(jnp.asarray(a, jnp.int32), jnp.asarray(b, jnp.int32), p)
        a_obj, b_obj = a.astype(object), b.astype(object)
        ref = np.mod(einx.multiply("i, m -> i m", a_obj, b_obj), p).astype(np.int64)
        assert got.dtype == jnp.int32
        assert np.array_equal(np.asarray(got, np.int64), ref)

    @pytest.mark.slow
    def test_float32_1000_by_1000_grid_is_accurate(self):
        g = kl.GridGraph((1000, 1000), axis_weights=jnp.ones(2, jnp.float32))
        lam, U = kl.laplacian_eigpairs(g, 20)
        assert lam.dtype == jnp.float32
        assert U.shape == (1_000_000, 20)
        path = 2 - 2 * np.cos(np.pi * np.arange(1000) / 1000)
        ref = np.sort(rearrange(einx.add("a, b -> a b", path, path), "a b -> (a b)"))
        ref = ref[:20]
        assert abs(float(lam[0])) < 1e-6
        rel = np.abs(np.asarray(lam[1:], np.float64) - ref[1:]) / ref[1:]
        assert np.max(rel) <= 1e-5

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

    def test_zero_weight_edges_do_not_connect(self):
        # One edge of weight 0: two components, and two zero eigenvalues.
        g = kl.graph_from_edges([0, 1], [1, 2], 3, weights=jnp.array([0.0, 1.0]))
        assert kl.n_components_graph(g) == 2
        lam, _ = kl.laplacian_eigpairs(g, 3)
        assert int(np.sum(np.abs(np.asarray(lam)) < 1e-9)) == 2

    def test_zero_axis_weight_disconnects_a_grid(self):
        g = kl.GridGraph((3, 4), axis_weights=jnp.array([0.0, 1.0]))
        assert kl.n_components_graph(g) == 3  # three unconnected rows


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
        with pytest.raises(ValueError, match="oversample"):
            kl.laplacian_eigpairs(
                g, 5, method="lanczos", key=jax.random.key(0), oversample=-1
            )
