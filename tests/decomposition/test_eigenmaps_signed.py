"""#180: signed-weight graphs (a cotangent `mesh_graph` with
``on_negative="allow"``) in the eigenmaps.

The degree constraint, $L y = \\lambda D y$, needs non-negative weights, as
the symmetric Laplacian does: every solver rejects a signed graph, eagerly
and under ``jit``. The identity constraint accepts it.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.spatial import Delaunay

import kernellib as kl
import kernellib._decomposition._eigenmaps as eigenmaps


KEY = jax.random.key(0)
SOLVERS = ["dense", "lanczos", "arpack", "lobpcg"]
JITTABLE = ["dense", "lanczos", pytest.param("lobpcg", marks=pytest.mark.slow)]
SIGNED = r"constraint='degree'\) needs non-negative edge weights"


def _mesh(on_negative):
    """Cotangent graph of a Delaunay mesh of 40 points: six of its hull
    edges see an obtuse angle, so their cotangent weights are negative."""
    V = np.random.default_rng(0).uniform(size=(40, 2))
    T = Delaunay(V).simplices
    g = kl.mesh_graph(jnp.asarray(V), T, weighting="cotangent", on_negative=on_negative)
    return jnp.asarray(V), g


@pytest.fixture(scope="module")
def signed():
    V, g = _mesh("allow")
    assert np.min(np.asarray(g.weights)) < 0.0
    return V, g


def _potential(n):
    return kl.barrier_potential(n, jnp.array([0, 5]))


def _embed(function, W, method, **kwargs):
    if function == "laplacian":
        return kl.laplacian_eigenmap(W, 2, method=method, key=KEY, **kwargs)
    return kl.schrodinger_eigenmap(
        W, _potential(40), 2, method=method, key=KEY, **kwargs
    )


@pytest.mark.parametrize("function", ["laplacian", "schrodinger"])
class TestDegreeConstraintRejects:
    @pytest.mark.parametrize("method", SOLVERS)
    def test_every_solver(self, signed, function, method):
        _, g = signed
        with pytest.raises(ValueError, match=SIGNED) as e:
            _embed(function, g, method)
        assert f"{function}_eigenmap(constraint='degree')" in str(e.value)
        assert "constraint='identity'" in str(e.value)

    @pytest.mark.parametrize("method", [None, "lobpcg"])
    def test_dense_adjacency(self, signed, function, method):
        _, g = signed
        with pytest.raises(ValueError, match=SIGNED):
            _embed(function, g.to_dense(), method)

    @pytest.mark.parametrize("method", JITTABLE)
    def test_under_jit(self, signed, function, method):
        _, g = signed
        embed = jax.jit(lambda g: _embed(function, g, method))
        with pytest.raises(Exception, match="non-negative edge weights"):
            jax.block_until_ready(embed(g))

    def test_dense_adjacency_under_jit(self, signed, function):
        _, g = signed
        embed = jax.jit(lambda W: _embed(function, W, None))
        with pytest.raises(Exception, match="non-negative edge weights"):
            jax.block_until_ready(embed(g.to_dense()))


class TestEstimatorsReject:
    @pytest.mark.parametrize("solver", SOLVERS)
    def test_laplacian_eigenmaps(self, signed, solver):
        X, g = signed
        model = kl.LaplacianEigenmaps(n_components=2, eigen_solver=solver)
        with pytest.raises(ValueError, match=SIGNED) as e:
            model.fit(X, graph=g)
        assert "LaplacianEigenmaps(constraint='degree')" in str(e.value)

    @pytest.mark.parametrize("solver", SOLVERS)
    def test_schrodinger_eigenmaps(self, signed, solver):
        X, g = signed
        model = kl.SchrodingerEigenmaps(n_components=2, eigen_solver=solver)
        with pytest.raises(ValueError, match=SIGNED) as e:
            model.fit(X, _potential(40), graph=g)
        assert "SchrodingerEigenmaps(constraint='degree')" in str(e.value)

    @pytest.mark.parametrize(
        "model",
        [
            kl.LocalityPreservingProjections(n_components=1),
            kl.SchrodingerEigenmapProjections(n_components=1),
            kl.KernelLocalityPreservingProjections(kl.RBF(), n_components=1),
            kl.KernelSchrodingerProjections(kl.RBF(), n_components=1),
        ],
        ids=lambda m: type(m).__name__,
    )
    def test_projections(self, signed, model):
        # Their constraint X^T D X (F^T D F) is the degree constraint.
        X, g = signed
        schrodinger = "Schrodinger" in type(model).__name__
        args = (X, _potential(40)) if schrodinger else (X,)
        with pytest.raises(ValueError, match="non-negative edge weights") as e:
            model.fit(*args, graph=g)
        assert str(e.value).startswith(type(model).__name__)


@pytest.mark.parametrize("method", SOLVERS)
class TestIdentityConstraintAccepts:
    """$(L + \\alpha V) y = \\lambda y$ is a symmetric eigenproblem for any
    weights; the cotangent Laplacian is even PSD (the FEM stiffness)."""

    def test_laplacian(self, signed, method):
        _, g = signed
        L = np.asarray(g.laplacian_operator().as_matrix())
        expected = np.linalg.eigvalsh(L)[1:3]
        lam, Y = kl.laplacian_eigenmap(
            g, 2, constraint="identity", method=method, key=KEY
        )
        assert np.allclose(lam, expected, atol=1e-8)
        assert Y.shape == (40, 2)

    def test_schrodinger(self, signed, method):
        _, g = signed
        L = g.laplacian_operator().as_matrix()
        V = _potential(40)
        alpha = jnp.sum(g.degree()) / jnp.sum(V)  # normalize_potential
        expected = np.linalg.eigvalsh(np.asarray(L + alpha * jnp.diag(V)))[1:3]
        lam, _ = kl.schrodinger_eigenmap(
            g, V, 2, constraint="identity", method=method, key=KEY
        )
        assert np.allclose(lam, expected, atol=1e-8)


@pytest.mark.slow
def test_lobpcg_identity_shift_is_signed_safe(monkeypatch):
    """The identity-constraint LOBPCG shift is the signed-safe Gershgorin
    bound, not 2 max(d), which a signed graph can put below lambda_max."""
    # A ring with weight 0.5, and a hub (node 0) joined with +1 to ten nodes
    # and with -0.9 to ten others: d_0 = 1.0, but its Gershgorin row is 20.
    n = 30
    senders = [*range(n), *([0] * 20)]
    receivers = [*((i + 1) % n for i in range(n)), *range(2, 22)]
    weights = [0.5] * n + [1.0] * 10 + [-0.9] * 10
    g = kl.graph_from_edges(
        jnp.array(senders), jnp.array(receivers), n, weights=jnp.array(weights)
    )
    L = np.asarray(g.laplacian_operator().as_matrix())
    spectrum = np.linalg.eigvalsh(L)
    assert 2.0 * np.max(np.asarray(g.degree())) < spectrum[-1]

    shifts = []
    solve = eigenmaps._shifted_lobpcg

    def spy(*args):
        shifts.append(args[5])
        return solve(*args)

    monkeypatch.setattr(eigenmaps, "_shifted_lobpcg", spy)
    lam, _ = kl.laplacian_eigenmap(
        g, 3, constraint="identity", drop_first=False, method="lobpcg", key=KEY
    )
    assert float(shifts[0]) >= spectrum[-1]
    assert np.allclose(lam, spectrum[:3], atol=1e-8)


@pytest.mark.parametrize("method", SOLVERS)
def test_non_negative_mesh_is_unaffected(method):
    """The same mesh with its negative weights clipped: the degree constraint
    runs, and every solver matches the dense generalised problem."""
    _, g = _mesh("clip")
    assert np.min(np.asarray(g.weights)) >= 0.0
    lam_d, _ = kl.laplacian_eigenmap(g.to_dense(), 2)
    lam, _ = kl.laplacian_eigenmap(g, 2, method=method, key=KEY)
    assert np.allclose(lam, lam_d, atol=1e-8)
    lam_d, _ = kl.schrodinger_eigenmap(g.to_dense(), _potential(40), 2)
    lam, _ = kl.schrodinger_eigenmap(g, _potential(40), 2, method=method, key=KEY)
    assert np.allclose(lam, lam_d, atol=1e-8)
