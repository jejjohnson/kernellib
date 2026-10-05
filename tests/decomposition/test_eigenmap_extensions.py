"""K5: graph inputs and every eigensolver for the eigenmaps, combine_potentials,
spatial_spectral_graph, and Schrödinger eigenmap projections."""

from __future__ import annotations

import einx
import jax
import jax.numpy as jnp
import lineax as lx
import pytest

import kernellib as kl
from kernellib._einx import einsum, reduce


def _same_subspace(A, B, atol=1e-6):
    """The column spaces agree: every principal-angle cosine is ~1. Robust to
    sign flips and rotations within degenerate eigenspaces."""
    Qa, _ = jnp.linalg.qr(A)
    Qb, _ = jnp.linalg.qr(B)
    cos = jnp.linalg.svd(einsum(Qa, Qb, "n i, n j -> i j"), compute_uv=False)
    return bool(jnp.min(cos) > 1 - atol)


def _knn(n=60, k=8, seed=0):
    X = jax.random.normal(jax.random.key(seed), (n, 2))
    knn = kl.nearest_neighbors(X, k)
    return X, kl.adjacency_matrix(knn), kl.graph_from_neighbors(knn)


def _graph(n=60, seed=0):
    """Points and a connected weighted graph on them, dense and sparse: a ring
    plus random chords, heat-weighted. No k-NN search (keeps tests fast)."""
    X = jax.random.normal(jax.random.key(seed), (n, 2))
    i = jnp.arange(n)
    chords = jax.random.randint(jax.random.key(seed + 1), (n,), 0, n)
    A = jnp.zeros((n, n)).at[i, (i + 1) % n].set(1.0).at[i, chords].set(1.0)
    A = jnp.maximum(A, einx.id("i j -> j i", A)) * (1.0 - jnp.eye(n))
    d2 = reduce(einx.subtract("i d, j d -> i j d", X, X) ** 2, "i j d -> i j", "sum")
    W = A * jnp.exp(-d2 / 2.0)
    return X, W, kl.graph_from_adjacency(W)


def _spread(Y, m):
    """Mean distance of the first ``m`` points to their centre, relative to
    that of all points."""

    def radius(Z):
        centred = einx.subtract("n k, k -> n k", Z, reduce(Z, "n k -> k", "mean"))
        return jnp.mean(jnp.sqrt(reduce(centred**2, "n k -> n", "sum")))

    return radius(Y[:m]) / radius(Y)


def _labels(n, m=10):
    return kl.label_potential(jnp.where(jnp.arange(n) < m, 0, -1))


_LANCZOS = pytest.param("lanczos", marks=pytest.mark.slow)  # Lanczos compiles


class TestGraphInputs:
    @pytest.mark.slow
    def test_graph_from_neighbors_matches_the_dense_adjacency(self):
        _, W, g = _knn()
        assert jnp.allclose(g.to_dense(), W)

    @pytest.mark.parametrize("method", [None, "dense", _LANCZOS, "arpack"])
    @pytest.mark.parametrize("constraint", ["degree", "identity"])
    def test_laplacian_eigenmap_graph_matches_dense(self, method, constraint):
        _, W, g = _graph()
        lam, Y = kl.laplacian_eigenmap(W, 3, constraint=constraint)
        lam_g, Y_g = kl.laplacian_eigenmap(
            g, 3, constraint=constraint, method=method, key=jax.random.key(0)
        )
        assert jnp.allclose(lam_g, lam, atol=1e-8)
        assert _same_subspace(Y_g, Y)
        if constraint == "degree":  # Y^T D Y = I on the graph path too
            D = jnp.diag(g.degree())
            assert jnp.allclose(einsum(Y_g, D, Y_g, "n a, n m, m b -> a b"), jnp.eye(3))

    @pytest.mark.parametrize("method", [_LANCZOS, "arpack"])
    def test_dense_array_with_a_sparse_method(self, method):
        _, W, _ = _graph()
        lam, Y = kl.laplacian_eigenmap(W, 2)
        lam_m, Y_m = kl.laplacian_eigenmap(W, 2, method=method, key=jax.random.key(0))
        assert jnp.allclose(lam_m, lam, atol=1e-8) and _same_subspace(Y_m, Y)

    @pytest.mark.slow
    def test_kronecker_on_a_grid(self):
        grid = kl.grid_graph((5, 7))
        lam, Y = kl.laplacian_eigenmap(grid, 3, constraint="identity")
        lam_d, Y_d = kl.laplacian_eigenmap(grid.to_dense(), 3, constraint="identity")
        assert jnp.allclose(lam, lam_d, atol=1e-10) and _same_subspace(Y, Y_d)
        with pytest.raises(ValueError, match="constraint='identity'"):
            kl.laplacian_eigenmap(grid, 3, method="kronecker")

    @pytest.mark.parametrize("method", [None, _LANCZOS, "arpack"])
    @pytest.mark.parametrize("kind", ["labels", "barrier", "operator"])
    def test_schrodinger_graph_matches_dense(self, method, kind):
        X, W, g = _graph()
        if kind == "labels":
            V = V_dense = _labels(60)
        elif kind == "barrier":
            V = V_dense = kl.barrier_potential(60, jnp.arange(5))
        else:  # a sparse operator potential, and its dense matrix
            V = kl.spatial_spectral_graph(X, g).laplacian_operator()
            V_dense = V.as_matrix()
        drop = kind != "barrier"
        lam, Y = kl.schrodinger_eigenmap(W, V_dense, 3, alpha=5.0, drop_first=drop)
        lam_g, Y_g = kl.schrodinger_eigenmap(
            g, V, 3, alpha=5.0, drop_first=drop, method=method, key=jax.random.key(0)
        )
        assert jnp.allclose(lam_g, lam, atol=1e-8)
        assert _same_subspace(Y_g, Y)

    @pytest.mark.slow
    def test_schrodinger_identity_constraint_on_a_grid(self):
        grid = kl.grid_graph((4, 6))
        V = kl.barrier_potential(24, jnp.array([0, 23]))
        kw = {"alpha": 2.0, "constraint": "identity", "drop_first": False}
        lam, Y = kl.schrodinger_eigenmap(grid.to_dense(), V, 3, **kw)
        for method in ("dense", "arpack"):  # arpack converts the KroneckerSum
            lam_g, Y_g = kl.schrodinger_eigenmap(grid, V, 3, method=method, **kw)
            assert jnp.allclose(lam_g, lam, atol=1e-8) and _same_subspace(Y_g, Y)

    def test_schrodinger_errors(self):
        _, _, g = _graph()
        with pytest.raises(ValueError, match="does not apply"):
            kl.schrodinger_eigenmap(g, _labels(60), method="kronecker")
        with pytest.raises(ValueError, match="needs a PRNG key"):
            kl.schrodinger_eigenmap(g, _labels(60), method="lanczos")
        with pytest.raises(ValueError, match="method must be"):
            kl.schrodinger_eigenmap(g, _labels(60), method="lobpcg")


class TestEstimators:
    @pytest.mark.slow
    def test_laplacian_eigenmaps_lanczos_and_precomputed_graph(self):
        X, _, g = _knn(k=10)
        dense = kl.LaplacianEigenmaps(n_components=2).fit(X)
        lanczos = kl.LaplacianEigenmaps(n_components=2, eigen_solver="lanczos").fit(X)
        assert jnp.allclose(lanczos.eigenvalues, dense.eigenvalues, atol=1e-8)
        assert _same_subspace(lanczos.embedding, dense.embedding)
        on_graph = kl.LaplacianEigenmaps(n_components=2).fit(X, graph=g)
        _, Y = kl.laplacian_eigenmap(g, 2)
        assert jnp.allclose(on_graph.embedding, Y) and on_graph.graph is g

    @pytest.mark.slow
    def test_schrodinger_eigenmaps_solvers_agree(self):
        X, _, g = _knn(k=10)
        V = _labels(60)
        V_op = lx.MatrixLinearOperator(V, lx.symmetric_tag)
        dense = kl.SchrodingerEigenmaps(alpha=5.0).fit(X, V)
        for solver, potential in (("lanczos", V), ("arpack", V_op), ("dense", V_op)):
            model = kl.SchrodingerEigenmaps(alpha=5.0, eigen_solver=solver)
            model = model.fit(X, potential)
            assert jnp.allclose(model.eigenvalues, dense.eigenvalues, atol=1e-8)
            assert _same_subspace(model.embedding, dense.embedding)
        on_graph = kl.SchrodingerEigenmaps(alpha=5.0).fit(X, V, graph=g)
        assert _same_subspace(on_graph.embedding, dense.embedding)

    def test_estimator_errors(self):
        X, _, g = _graph()
        with pytest.raises(ValueError, match="one node per point"):
            kl.LaplacianEigenmaps().fit(X[:50], graph=g)
        with pytest.raises(ValueError, match="potential must have shape"):
            kl.SchrodingerEigenmaps().fit(X, lx.DiagonalLinearOperator(jnp.ones(5)))


class TestCombinePotentials:
    def test_one_term_equals_the_alpha_normalisation(self):
        _, W, _ = _graph()
        V = _labels(60)
        lam, Y = kl.schrodinger_eigenmap(W, V, 2, alpha=3.0)
        combined = kl.combine_potentials(W, [(V, 3.0)])
        lam_c, Y_c = kl.schrodinger_eigenmap(
            W, combined, 2, alpha=1.0, normalize_potential=False
        )
        assert jnp.allclose(lam_c, lam) and jnp.allclose(Y_c, Y)

    def test_each_weight_stays_with_its_potential(self):
        # The old 'sspl' mode weighted the label potential by the spatial
        # alpha and vice versa.
        X, W, g = _graph()
        spatial = kl.spatial_spectral_graph(X, g).laplacian_operator()
        labels = _labels(60)
        tr_l = jnp.sum(g.degree())
        V = kl.combine_potentials(g, [(spatial, 2.0), (labels, 0.5)])
        S = spatial.as_matrix()
        expected = (
            2.0 * tr_l / jnp.trace(S) * S + 0.5 * tr_l / jnp.trace(labels) * labels
        )
        assert isinstance(V, lx.AbstractLinearOperator)  # stays sparse
        assert jnp.allclose(V.as_matrix(), expected)
        plain = kl.combine_potentials(W, [(S, 2.0), (labels, 0.5)], normalize=False)
        assert jnp.allclose(plain, 2.0 * S + 0.5 * labels)

    def test_diagonals_stay_diagonal_and_errors(self):
        _, W, _ = _graph()
        a, b = kl.barrier_potential(60, jnp.arange(3)), kl.barrier_potential(60, 5)
        V = kl.combine_potentials(W, [(a, 1.0), (b, 1.0)], normalize=False)
        assert V.shape == (60,) and jnp.allclose(V, a + b)
        with pytest.raises(ValueError, match="at least one"):
            kl.combine_potentials(W, [])


class TestSpatialSpectralGraph:
    @pytest.mark.slow
    def test_grid_topology_with_heat_weights(self):
        X = jax.random.normal(jax.random.key(1), (20, 3))
        grid = kl.grid_graph((4, 5))
        g = kl.spatial_spectral_graph(X, grid, bandwidth=0.7)
        assert g.topology == grid.topology
        s, r, w = g.edges()
        d2 = reduce((X[s] - X[r]) ** 2, "e d -> e", "sum")
        assert jnp.allclose(w, jnp.exp(-d2 / (2 * 0.7**2)))
        # Default bandwidth: the median spectral distance over the edges.
        default = kl.spatial_spectral_graph(X, grid)
        sigma = jnp.median(jnp.sqrt(d2))
        assert jnp.allclose(default.weights, jnp.exp(-d2 / (2 * sigma**2)))

    @pytest.mark.slow
    def test_matches_spatial_spectral_potential_on_the_same_edges(self):
        # A 1-D "image": the 2 nearest pixels of each are its lattice
        # neighbours (ends reach one further), so the potentials agree away
        # from the ends.
        X = jax.random.normal(jax.random.key(2), (12, 2))
        coords = einx.id("n -> n 1", jnp.arange(12.0))
        dense = kl.spatial_spectral_potential(X, coords, 2, bandwidth=1.0)
        sparse = kl.spatial_spectral_graph(X, kl.grid_graph((12,)), bandwidth=1.0)
        L = sparse.laplacian_operator().as_matrix()

        def off(M):  # the edge weights; degrees include the end effects
            return M[2:-2, 2:-2] - jnp.diag(jnp.diag(M[2:-2, 2:-2]))

        assert jnp.allclose(off(L), off(dense))


def _old_lpp(X, W, k, regularization=1e-8):
    """The pre-K5 LPP solve (inline Cholesky whitening), for regression."""
    degree = reduce(W, "i j -> i", "sum")
    mu = degree @ X / jnp.sum(degree)
    Xc = einx.subtract("n d, d -> n d", X, mu)
    A = einsum(Xc, kl.graph_laplacian(W), Xc, "n a, n m, m b -> a b")
    B = einsum(einx.multiply("n a, n -> n a", Xc, degree), Xc, "n a, n b -> a b")
    d = X.shape[1]
    B = B + regularization * jnp.trace(B) / d * jnp.eye(d)
    Ci = jnp.linalg.inv(jnp.linalg.cholesky(B))
    lam, U = jnp.linalg.eigh(einsum(Ci, A, Ci, "a i, i j, b j -> a b"))
    return lam[:k], einsum(Ci, U[:, :k], "a i, a k -> i k"), mu


class TestProjections:
    @pytest.mark.slow
    def test_lpp_matches_the_previous_solver(self):
        X, W, _ = _graph(n=80)
        X = jnp.concatenate([X, jnp.sin(X)], axis=1)
        lpp = kl.LocalityPreservingProjections(n_components=3).fit(X, graph=W)
        lam, P, mu = _old_lpp(X, W, 3)
        assert jnp.allclose(lpp.eigenvalues, lam, rtol=1e-10)
        assert jnp.allclose(lpp.mean, mu)
        # eigh fixes each eigenvector up to sign only.
        signs = jnp.sign(reduce(lpp.projection * P, "d k -> k", "sum"))
        assert jnp.allclose(einx.multiply("d k, k -> d k", P, signs), lpp.projection)

    @pytest.mark.slow
    def test_sep_with_zero_alpha_is_lpp_exactly(self):
        X, W, _ = _graph(n=70)
        X = jnp.concatenate([X, jnp.sin(X), X**2], axis=1)
        lpp = kl.LocalityPreservingProjections(n_components=2).fit(X, graph=W)
        sep = kl.SchrodingerEigenmapProjections(n_components=2, alpha=0.0)
        sep = sep.fit(X, _labels(70), graph=W)
        assert jnp.array_equal(sep.projection, lpp.projection)
        assert jnp.array_equal(sep.eigenvalues, lpp.eigenvalues)

    @pytest.mark.slow
    def test_sep_solves_its_generalised_eigenproblem(self):
        X = jax.random.normal(jax.random.key(9), (70, 4))
        V = _labels(70, 15)
        sep = kl.SchrodingerEigenmapProjections(
            n_components=2, alpha=5.0, regularization=0.0
        ).fit(X, V)
        W = kl.adjacency_matrix(kl.nearest_neighbors(X, 10))
        degree = reduce(W, "i j -> i", "sum")
        L = kl.graph_laplacian(W)
        alpha = 5.0 * jnp.trace(L) / jnp.trace(V)  # normalised in N space
        Xc = einx.subtract("n d, d -> n d", X, sep.mean)
        A = einsum(Xc, L + alpha * V, Xc, "n a, n m, m b -> a b")
        B = einsum(einx.multiply("n a, n -> n a", Xc, degree), Xc, "n a, n b -> a b")
        P = sep.projection
        assert jnp.allclose(
            A @ P, einx.multiply("d k, k -> d k", B @ P, sep.eigenvalues)
        )
        # The labelled points are pulled together, also out of sample.
        Z, Z0 = sep.transform(X), kl.LocalityPreservingProjections().fit(X).transform(X)

        assert _spread(Z, 15) < _spread(Z0, 15)

    @pytest.mark.slow
    def test_precomputed_graph_and_operator_potential(self):
        X, W, g = _graph(n=50)
        X = jnp.concatenate([X, X**2], axis=1)
        V = _labels(50)
        dense = kl.SchrodingerEigenmapProjections(n_components=2, alpha=2.0).fit(
            X, V, graph=W
        )
        sparse = kl.SchrodingerEigenmapProjections(n_components=2, alpha=2.0).fit(
            X, lx.MatrixLinearOperator(V, lx.symmetric_tag), graph=g
        )
        assert jnp.allclose(sparse.eigenvalues, dense.eigenvalues)
        assert _same_subspace(sparse.projection, dense.projection)
        lpp = kl.LocalityPreservingProjections(n_components=2).fit(X, graph=g)
        assert lpp.projection.shape == (4, 2)

    def test_errors(self):
        X = jnp.ones((20, 3))
        with pytest.raises(ValueError, match="potential must have shape"):
            kl.SchrodingerEigenmapProjections().fit(X, jnp.ones(5))
        with pytest.raises(ValueError, match="exceeds the input dimension"):
            kl.SchrodingerEigenmapProjections(n_components=4).fit(X, jnp.ones(20))
        with pytest.raises(RuntimeError, match="not fitted"):
            kl.SchrodingerEigenmapProjections().transform(X)
