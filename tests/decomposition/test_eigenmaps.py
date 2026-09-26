"""Laplacian eigenmaps, Schrödinger eigenmaps and LPP."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import kernellib as kl


def _swiss_roll(n=400, seed=0):
    rng = np.random.default_rng(seed)
    t = 1.5 * np.pi * (1 + 2 * rng.uniform(size=n))
    h = 10 * rng.uniform(size=n)
    X = np.stack([t * np.cos(t), h, t * np.sin(t)], axis=1)
    return jnp.asarray(X), t


def _spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return np.corrcoef(ra, rb)[0, 1]


def _W(X, k=10):
    return kl.adjacency_matrix(kl.nearest_neighbors(X, k))


class TestLaplacianEigenmap:
    def test_solves_the_generalised_eigenproblem(self):
        X = jax.random.normal(jax.random.key(0), (60, 3))
        W = _W(X)
        lam, Y = kl.laplacian_eigenmap(W, 3)
        L, D = kl.graph_laplacian(W), jnp.diag(W.sum(1))
        assert jnp.allclose(L @ Y, D @ Y * lam, atol=1e-8)
        assert jnp.allclose(Y.T @ D @ Y, jnp.eye(3), atol=1e-8)
        # D-orthogonal to the trivial constant solution that was dropped.
        assert jnp.allclose(W.sum(1) @ Y, 0.0, atol=1e-8)
        assert jnp.all(lam > 0) and jnp.all(jnp.diff(lam) >= 0)

    def test_identity_constraint(self):
        W = _W(jax.random.normal(jax.random.key(1), (40, 2)))
        lam, Y = kl.laplacian_eigenmap(W, 2, constraint="identity")
        assert jnp.allclose(kl.graph_laplacian(W) @ Y, Y * lam, atol=1e-8)
        assert jnp.allclose(Y.T @ Y, jnp.eye(2), atol=1e-8)

    @pytest.mark.slow
    def test_unrolls_a_swiss_roll(self):
        X, t = _swiss_roll()
        le = kl.LaplacianEigenmaps(n_components=2, n_neighbors=10).fit(X)
        assert abs(_spearman(np.asarray(le.embedding[:, 0]), t)) > 0.95

    def test_arpack_matches_dense(self):
        X, _ = _swiss_roll(n=250)
        dense = kl.LaplacianEigenmaps(n_components=2).fit(X)
        sparse = kl.LaplacianEigenmaps(n_components=2, eigen_solver="arpack").fit(X)
        assert jnp.allclose(sparse.eigenvalues, dense.eigenvalues, rtol=1e-6)
        for j in range(2):
            a, b = dense.embedding[:, j], sparse.embedding[:, j]
            cos = jnp.abs(a @ b) / (jnp.linalg.norm(a) * jnp.linalg.norm(b))
            assert cos > 1 - 1e-6


class TestSchrodinger:
    def test_zero_alpha_is_laplacian_eigenmaps(self):
        X = jax.random.normal(jax.random.key(2), (50, 2))
        W = _W(X)
        V = kl.label_potential(jnp.where(jnp.arange(50) < 10, 0, -1))
        lam_le, Y_le = kl.laplacian_eigenmap(W, 2)
        lam_se, Y_se = kl.schrodinger_eigenmap(W, V, 2, alpha=0.0)
        assert jnp.allclose(lam_se, lam_le)
        assert jnp.allclose(jnp.abs(Y_se), jnp.abs(Y_le), atol=1e-8)

    def test_label_potential_pulls_a_class_together(self):
        X = jax.random.normal(jax.random.key(3), (80, 2))
        labels = jnp.where(jnp.arange(80) < 15, 0, -1)
        W = _W(X)

        def spread(Y):
            inside = jnp.linalg.norm(Y[:15] - Y[:15].mean(0), axis=1).mean()
            return inside / jnp.linalg.norm(Y - Y.mean(0), axis=1).mean()

        _, Y_le = kl.laplacian_eigenmap(W, 2)
        _, Y_se = kl.schrodinger_eigenmap(W, kl.label_potential(labels), 2, alpha=10.0)
        assert spread(Y_se) < 0.25 * spread(Y_le)

    def test_barrier_pins_points(self):
        X = jax.random.normal(jax.random.key(4), (60, 2))
        pinned = jnp.arange(5)
        V = kl.barrier_potential(60, pinned)
        _, Y = kl.schrodinger_eigenmap(_W(X), V, 2, alpha=50.0, drop_first=False)
        assert jnp.max(jnp.abs(Y[pinned])) < 0.1 * jnp.max(jnp.abs(Y))

    def test_normalization_makes_the_potential_scale_free(self):
        X = jax.random.normal(jax.random.key(5), (40, 2))
        W = _W(X)
        V = kl.label_potential(jnp.where(jnp.arange(40) < 8, 0, -1))
        a = kl.schrodinger_eigenmap(W, V, 2, alpha=3.0)[0]
        b = kl.schrodinger_eigenmap(W, 7.0 * V, 2, alpha=3.0)[0]
        assert jnp.allclose(a, b)

    def test_label_potential(self):
        V = kl.label_potential(jnp.array([0, 1, 0, -1, 1]))
        assert jnp.allclose(V.sum(1), 0.0)
        assert float(V[0, 2]) == -1.0 and float(V[0, 1]) == 0.0
        assert jnp.all(V[3] == 0)

    def test_spatial_spectral_potential(self):
        # A 6 x 6 "image": pixels on a grid with 2 spectral bands.
        yy, xx = jnp.meshgrid(jnp.arange(6.0), jnp.arange(6.0), indexing="ij")
        coords = jnp.stack([yy.ravel(), xx.ravel()], axis=1)
        X = jax.random.normal(jax.random.key(6), (36, 2))
        V = kl.spatial_spectral_potential(X, coords, n_neighbors=4)
        assert jnp.allclose(V, V.T) and jnp.allclose(V.sum(1), 0.0, atol=1e-6)
        assert jnp.linalg.eigvalsh(V)[0] > -1e-8
        # An interior pixel's 4 nearest pixels are its grid neighbours, and no
        # other pixel lists it (boundary pixels reach diagonals, never (2, 2)).
        pixel = 2 * 6 + 2
        joined = {int(j) for j in jnp.nonzero(V[pixel])[0] if int(j) != pixel}
        assert joined == {pixel - 6, pixel - 1, pixel + 1, pixel + 6}

    def test_estimator_and_arpack(self):
        X, _ = _swiss_roll(n=200)
        V = kl.label_potential(jnp.where(jnp.arange(200) < 20, 0, -1))
        dense = kl.SchrodingerEigenmaps(alpha=5.0).fit(X, V)
        sparse = kl.SchrodingerEigenmaps(alpha=5.0, eigen_solver="arpack").fit(X, V)
        assert dense.embedding.shape == (200, 2)
        assert jnp.allclose(sparse.eigenvalues, dense.eigenvalues, rtol=1e-5)

    def test_potential_shape_check(self):
        with pytest.raises(ValueError, match="potential must have shape"):
            kl.SchrodingerEigenmaps().fit(jnp.ones((10, 2)), jnp.ones(9))


class TestLPP:
    def test_transform_and_generalised_eigenproblem(self):
        X = jax.random.normal(jax.random.key(7), (80, 4))
        lpp = kl.LocalityPreservingProjections(n_components=2).fit(X)
        Z = lpp.transform(X)
        assert Z.shape == (80, 2)
        W = kl.adjacency_matrix(kl.nearest_neighbors(X, 10))
        d = W.sum(1)
        Xc = X - lpp.mean
        A = Xc.T @ kl.graph_laplacian(W) @ Xc
        P = lpp.projection
        # Aa = lambda B a (regularisation is 1e-8 relative, so tolerance 1e-6).
        B = (Xc * d[:, None]).T @ Xc
        assert jnp.allclose(A @ P, B @ P * lpp.eigenvalues, rtol=1e-5, atol=1e-6)

    def test_recovers_a_plane_in_five_dimensions(self):
        # Points spread over a 2-D plane, rotated into 5-D with tiny noise
        # off the plane. LPP's 2-D projection keeps almost every 8-NN
        # neighbourhood of the plane coordinates; a random projection keeps
        # under half.
        k1, k2, k3 = jax.random.split(jax.random.key(8), 3)
        P = jax.random.uniform(k1, (150, 2), minval=-3.0, maxval=3.0)
        R, _ = jnp.linalg.qr(jax.random.normal(k2, (5, 5)))
        X = jnp.concatenate([P, 0.01 * jax.random.normal(k3, (150, 3))], 1) @ R.T
        lpp = kl.LocalityPreservingProjections(n_components=2, n_neighbors=8).fit(X)

        def overlap(A, B):
            a = np.asarray(kl.nearest_neighbors(A, 8).indices)
            b = np.asarray(kl.nearest_neighbors(B, 8).indices)
            return np.mean(
                [len(set(x) & set(y)) / 8 for x, y in zip(a, b, strict=True)]
            )

        assert overlap(P, lpp.transform(X)) > 0.9
        assert overlap(P, X @ jax.random.normal(jax.random.key(1), (5, 2))) < 0.6

    def test_errors(self):
        with pytest.raises(ValueError, match="exceeds the input dimension"):
            kl.LocalityPreservingProjections(n_components=3).fit(jnp.ones((20, 2)))
        with pytest.raises(RuntimeError, match="not fitted"):
            kl.LocalityPreservingProjections().transform(jnp.ones((2, 2)))
        with pytest.raises(ValueError, match="eigen_solver"):
            kl.LaplacianEigenmaps(eigen_solver="lobpcg")
        with pytest.raises(ValueError, match="constraint"):
            kl.LaplacianEigenmaps(constraint="similarity")
