"""#91: eigen_solver="lobpcg", the JAX-native sparse eigensolver of the
Laplacian and Schrödinger eigenmaps."""

from __future__ import annotations

import einx
import jax
import jax.numpy as jnp
import lineax as lx
import pytest

import kernellib as kl
from kernellib._decomposition._eigenmaps import _lobpcg_eigenmap
from kernellib._einx import einsum


def _swiss_roll(n, seed=0):
    u, v = jax.random.uniform(jax.random.key(seed), (2, n))
    t = 1.5 * jnp.pi * (1.0 + 2.0 * u)
    return einx.id("d n -> n d", jnp.stack([t * jnp.cos(t), 21.0 * v, t * jnp.sin(t)]))


def _overlap(A, B, weight=None):
    """Cosines of the principal angles between the column spaces, after
    orthonormalising in the ``diag(weight)`` inner product (``D`` for the
    degree constraint); 1 means the same subspace."""
    if weight is not None:
        A = einx_scale(jnp.sqrt(weight), A)
        B = einx_scale(jnp.sqrt(weight), B)
    Qa, _ = jnp.linalg.qr(A)
    Qb, _ = jnp.linalg.qr(B)
    return jnp.linalg.svd(einsum(Qa, Qb, "n i, n j -> i j"), compute_uv=False)


def einx_scale(w, Y):
    return einx.multiply("n, n k -> n k", w, Y)


@pytest.fixture(scope="module")
def roll():
    """A 500-point Swiss roll and its 10-NN graph, dense and sparse."""
    X = _swiss_roll(500)
    knn = kl.nearest_neighbors(X, 10)
    g = kl.graph_from_neighbors(knn)
    return X, g.to_dense(), g


KEY = jax.random.key(0)


class TestAgreesWithDense:
    def test_laplacian(self, roll):
        _, W, g = roll
        lam_d, Y_d = kl.laplacian_eigenmap(W, 2)
        lam, Y, n_iter = _lobpcg_eigenmap(g, None, 2, key=KEY)
        assert jnp.allclose(lam, lam_d, atol=1e-8)
        assert jnp.min(_overlap(Y, Y_d, g.degree())) > 0.999
        assert 0 < int(n_iter) <= 1000
        # Y^T D Y = I, eigenvalues ascending and >= 0.
        gram = einsum(Y, einx_scale(g.degree(), Y), "n i, n j -> i j")
        assert jnp.allclose(gram, jnp.eye(2), atol=1e-8)
        assert bool(jnp.all(jnp.diff(lam) >= 0)) and bool(jnp.all(lam >= -1e-8))

    def test_laplacian_identity_constraint(self, roll):
        _, W, g = roll
        lam_d, Y_d = kl.laplacian_eigenmap(W, 2, constraint="identity")
        lam, Y = kl.laplacian_eigenmap(
            g, 2, constraint="identity", method="lobpcg", key=KEY
        )
        assert jnp.allclose(lam, lam_d, atol=1e-8)
        assert jnp.min(_overlap(Y, Y_d)) > 0.999

    @pytest.mark.parametrize(
        "kind",
        [
            "barrier",
            pytest.param("labels", marks=pytest.mark.slow),
            pytest.param("sparse", marks=pytest.mark.slow),
        ],
    )
    def test_schrodinger(self, roll, kind):
        _, W, g = roll
        n = W.shape[0]
        drop_first = kind != "barrier"
        if kind == "barrier":
            V = kl.barrier_potential(n, jnp.arange(10))
            V_dense = V
        elif kind == "labels":
            V = kl.label_potential(jnp.where(jnp.arange(n) < 20, 0, -1))
            V_dense = V
        else:  # a sparse lineax potential: another graph's Laplacian, scaled
            other = kl.graph_from_neighbors(
                kl.nearest_neighbors(jax.random.normal(KEY, (n, 2)), 4)
            )
            V = 0.5 * lx.TaggedLinearOperator(
                other.laplacian_operator(), lx.symmetric_tag
            )
            V_dense = V.as_matrix()
        kw = {"alpha": 1.0, "drop_first": drop_first}
        lam_d, Y_d = kl.schrodinger_eigenmap(W, V_dense, 2, **kw)
        lam, Y = kl.schrodinger_eigenmap(g, V, 2, method="lobpcg", key=KEY, **kw)
        assert jnp.allclose(lam, lam_d, atol=1e-8)
        assert jnp.min(_overlap(Y, Y_d, g.degree())) > 0.999

    @pytest.mark.slow
    def test_estimators_and_n_iter(self, roll):
        X, W, g = roll
        V = kl.barrier_potential(X.shape[0], jnp.arange(10))
        le = kl.LaplacianEigenmaps(eigen_solver="lobpcg", random_state=0).fit(X)
        ref = kl.LaplacianEigenmaps().fit(X)
        assert jnp.allclose(le.eigenvalues, ref.eigenvalues, atol=1e-8)
        assert jnp.min(_overlap(le.embedding, ref.embedding)) > 0.999
        assert le.n_iter is not None and int(le.n_iter) > 0
        assert ref.n_iter is None
        se = kl.SchrodingerEigenmaps(
            drop_first=False, eigen_solver="lobpcg", max_iter=2000
        ).fit(X, V, graph=g)
        se_ref = kl.SchrodingerEigenmaps(drop_first=False).fit(X, V, graph=W)
        assert se.graph is g and int(se.n_iter) > 0
        assert jnp.allclose(se.eigenvalues, se_ref.eigenvalues, atol=1e-8)


class TestJit:
    def test_solve_traces_under_jit(self, roll):
        _, _, g = roll

        @jax.jit
        def embed(graph):
            return _lobpcg_eigenmap(graph, None, 2, key=KEY)

        lam, Y, n_iter = embed(g)
        lam_e, Y_e, n_iter_e = _lobpcg_eigenmap(g, None, 2, key=KEY)
        assert jnp.allclose(lam, lam_e, atol=1e-10)
        assert jnp.min(_overlap(Y, Y_e)) > 1 - 1e-8
        # Compiled and eager rounding differ, so the stopping step may too.
        assert abs(int(n_iter) - int(n_iter_e)) <= 5

    def test_check_fires_under_jit(self, roll):
        _, _, g = roll
        embed = jax.jit(
            lambda graph: _lobpcg_eigenmap(graph, None, 2, key=KEY, max_iter=2)
        )
        with pytest.raises(Exception, match=r"'lobpcg', max_iter=2\) did not converge"):
            jax.block_until_ready(embed(g))


class TestErrors:
    def test_needs_x64(self, roll):
        _, _, g = roll
        with (
            jax.enable_x64(False),
            pytest.raises(ValueError, match=r"needs float64.*jax_enable_x64"),
        ):
            kl.laplacian_eigenmap(g, 2, method="lobpcg", key=KEY)

    def test_needs_5k_below_n(self):
        X = _swiss_roll(30)
        g = kl.graph_from_neighbors(kl.nearest_neighbors(X, 5))
        with pytest.raises(
            ValueError, match=r"5 \* 6 eigenpairs < N = 30.*method='dense'"
        ):
            kl.laplacian_eigenmap(g, 5, method="lobpcg", key=KEY)
        model = kl.LaplacianEigenmaps(n_components=5, eigen_solver="lobpcg")
        with pytest.raises(ValueError, match="eigen_solver='dense'"):
            model.fit(X, graph=g)

    def test_unconverged_raises(self, roll):
        # The message names the caller's own solver parameter: method= for
        # the functions, eigen_solver= for the estimators.
        X, _, g = roll
        with pytest.raises(RuntimeError) as info:
            kl.laplacian_eigenmap(g, 2, method="lobpcg", key=KEY, max_iter=2)
        message = str(info.value)
        assert (
            "laplacian_eigenmap(method='lobpcg', max_iter=2) did not converge"
            in message
        )
        assert "use method='arpack'" in message and "eigen_solver" not in message
        model = kl.LaplacianEigenmaps(eigen_solver="lobpcg", max_iter=2)
        with pytest.raises(RuntimeError) as info:
            model.fit(X, graph=g)
        message = str(info.value)
        assert (
            "LaplacianEigenmaps(eigen_solver='lobpcg', max_iter=2) did not converge"
            in message
        )
        assert "use eigen_solver='arpack'" in message and "method=" not in message

    def test_key_potential_and_max_iter(self, roll):
        _, _, g = roll
        with pytest.raises(ValueError, match="needs a PRNG key"):
            kl.schrodinger_eigenmap(g, jnp.ones(500), method="lobpcg")
        V = kl.grid_graph((20, 25)).laplacian_operator()  # a gaussx.KroneckerSum
        with pytest.raises(ValueError, match="KroneckerSum"):
            kl.schrodinger_eigenmap(g, V, method="lobpcg", key=KEY)
        with pytest.raises(ValueError, match="max_iter must be >= 1"):
            kl.LaplacianEigenmaps(eigen_solver="lobpcg", max_iter=0)


@pytest.mark.slow
@pytest.mark.parametrize("kind", ["laplacian", "schrodinger"])
def test_matches_arpack_at_n_5000(kind):
    X = _swiss_roll(5000)
    kw = {"n_components": 2, "random_state": 0}
    if kind == "laplacian":
        arpack = kl.LaplacianEigenmaps(eigen_solver="arpack", **kw).fit(X)
        lobpcg = kl.LaplacianEigenmaps(eigen_solver="lobpcg", **kw).fit(X)
    else:
        # A potential widens the spectrum (c ~ 20 here) and slows LOBPCG,
        # which needs ~2100 iterations; at alpha = 1 (c ~ 195) it needs ~5000.
        V = kl.barrier_potential(5000, jnp.arange(50))
        kw = {**kw, "alpha": 0.1, "drop_first": False}
        arpack = kl.SchrodingerEigenmaps(eigen_solver="arpack", **kw).fit(X, V)
        lobpcg = kl.SchrodingerEigenmaps(
            eigen_solver="lobpcg", max_iter=3000, **kw
        ).fit(X, V)
    assert jnp.allclose(lobpcg.eigenvalues, arpack.eigenvalues, rtol=1e-6, atol=1e-9)
    degree = kl.graph_from_neighbors(lobpcg.graph).degree()
    assert jnp.min(_overlap(lobpcg.embedding, arpack.embedding, degree)) > 0.999
