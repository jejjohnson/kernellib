"""KernelPCA(eigen_solver="randomized"): gaussx.randomized_eigh on H K H."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import kernellib as kl
from kernellib._decomposition._kpca import _doubly_centred
from kernellib._einx import einsum
from kernellib._operators._bridge import to_operator


def _X(n=200, d=3, seed=0):
    return jax.random.normal(jax.random.key(seed), (n, d))


def _principal_cosines(U, V):
    """Cosines of the principal angles between the column spans of U and V."""
    Qu, _ = jnp.linalg.qr(U)
    Qv, _ = jnp.linalg.qr(V)
    return jnp.linalg.svd(einsum(Qu, Qv, "n a, n b -> a b"), compute_uv=False)


def _fit(kernel, X, solver, key=0, **kwargs):
    return kl.KernelPCA(kernel, n_components=5, eigen_solver=solver, **kwargs).fit(
        X, key=jax.random.key(key)
    )


class TestRandomizedKernelPCA:
    def test_agrees_with_dense_on_a_fast_decaying_spectrum(self):
        # A smooth RBF on 3-D data: eigenvalues decay fast, so the leading
        # subspace is well separated and the range finder captures it.
        X = _X()
        kernel = kl.RBF(lengthscale=3.0)
        dense = _fit(kernel, X, "dense")
        rand = _fit(kernel, X, "randomized")
        assert np.allclose(rand.eigenvalues, dense.eigenvalues, rtol=1e-6)
        cosines = _principal_cosines(rand.embedding, dense.embedding)
        assert np.all(np.asarray(cosines) > 1.0 - 1e-8)

    def test_eigenvalues_are_descending_and_centring_stats_match(self):
        X = _X()
        kernel = kl.RBF(lengthscale=3.0)
        dense = _fit(kernel, X, "dense")
        rand = _fit(kernel, X, "randomized")
        lam = np.asarray(rand.eigenvalues)
        assert np.all(np.diff(lam) <= 0)
        assert np.allclose(rand.gram_column_means, dense.gram_column_means)
        assert np.isclose(rand.gram_mean, dense.gram_mean)

    def test_transform_reproduces_the_embedding(self):
        X = _X()
        rand = _fit(kl.RBF(lengthscale=3.0), X, "randomized")
        assert np.allclose(rand.transform(X), rand.embedding, atol=1e-6)
        X_new = _X(20, seed=4)
        dense = _fit(kl.RBF(lengthscale=3.0), X, "dense")
        # Equal up to the sign of each component.
        a, b = np.asarray(rand.transform(X_new)), np.asarray(dense.transform(X_new))
        assert np.allclose(np.abs(a), np.abs(b), atol=1e-6)

    def test_doubly_centred_operator_is_hkh(self):
        # A Gram that is far from centred (large mean), so HK, KH and HKH
        # all differ.
        X = _X(40) + 5.0
        K_op = to_operator(kl.RBF(lengthscale=4.0), X, implicit=True)
        K = K_op.as_matrix()
        n = 40
        H = jnp.eye(n) - jnp.full((n, n), 1.0 / n)
        HKH = H @ K @ H
        C = _doubly_centred(K_op, n)
        assert not np.allclose(H @ K, HKH)
        assert np.allclose(C.as_matrix(), HKH, atol=1e-10)
        v = jax.random.normal(jax.random.key(1), (n,))
        w = jax.random.normal(jax.random.key(2), (n,))
        assert np.isclose(w @ C.mv(v), v @ C.mv(w))
        assert np.allclose(C.mv(jnp.ones(n)), 0.0, atol=1e-10)

    @pytest.mark.slow
    def test_gram_only_kernels(self):
        X = _X(60)
        kernel = kl.RBF(lengthscale=3.0) + kl.Linear()
        rand = _fit(kernel, X, "randomized")
        dense = _fit(kernel, X, "dense")
        assert np.allclose(rand.eigenvalues, dense.eigenvalues, rtol=1e-5)

    def test_oversample_is_capped(self):
        X = _X(12)
        rand = kl.KernelPCA(
            kl.RBF(lengthscale=3.0),
            n_components=5,
            eigen_solver="randomized",
            oversample=50,
        ).fit(X, key=jax.random.key(0))
        assert rand.eigenvalues.shape == (5,)

    def test_validation(self):
        X = _X(30)
        with pytest.raises(ValueError, match="PRNG key"):
            kl.KernelPCA(kl.RBF(), eigen_solver="randomized").fit(X)
        with pytest.raises(ValueError, match="eigen_solver"):
            kl.KernelPCA(kl.RBF(), eigen_solver="arpack")
        with pytest.raises(ValueError, match="approx"):
            kl.KernelPCA(
                kl.RBF(),
                eigen_solver="randomized",
                approx=kl.NystromFeatures(10, jax.random.key(0)),
            )
        with pytest.raises(ValueError, match=">= 0"):
            kl.KernelPCA(kl.RBF(), n_power_iter=-1)
        with pytest.raises(ValueError, match="plain kernel PCA only"):
            kl.KernelPCA(kl.RBF(), eigen_solver="randomized", target_weight=1.0).fit(
                X, target=X[:, 0], key=jax.random.key(0)
            )

    def test_dense_path_ignores_the_key(self):
        X = _X(30)
        a = kl.KernelPCA(kl.RBF(), n_components=3).fit(X)
        b = kl.KernelPCA(kl.RBF(), n_components=3).fit(X, key=jax.random.key(7))
        assert np.array_equal(a.eigenvalues, b.eigenvalues)

    @pytest.mark.slow
    def test_sklearn_adapter(self):
        pytest.importorskip("sklearn")
        from kernellib.sklearn import KernelPCA

        X = np.asarray(_X())
        model = KernelPCA(
            n_components=3,
            kernel=kl.RBF(lengthscale=3.0),
            eigen_solver="randomized",
            random_state=0,
        ).fit(X)
        assert model.model_.eigen_solver == "randomized"
        dense = KernelPCA(n_components=3, kernel=kl.RBF(lengthscale=3.0)).fit(X)
        assert np.allclose(model.eigenvalues_, dense.eigenvalues_, rtol=1e-6)


@pytest.mark.slow
def test_power_iterations_help_on_a_slowly_decaying_spectrum():
    # Matern-1/2 at a short lengthscale: a flat spectrum, where the one-pass
    # range finder (q = 0) misses the leading eigenvalues. Averaged over 10
    # keys, the eigenvalue error must fall as q grows, and q = 3 must be much
    # closer than q = 0 (the gap is many sampling standard deviations).
    X = _X(400, d=2)
    kernel = kl.Matern(nu=0.5, lengthscale=0.2)
    dense = kl.KernelPCA(kernel, n_components=10).fit(X)

    def error(q, seed):
        rand = kl.KernelPCA(
            kernel,
            n_components=10,
            eigen_solver="randomized",
            n_power_iter=q,
            oversample=5,
        ).fit(X, key=jax.random.key(seed))
        rel = (dense.eigenvalues - rand.eigenvalues) / dense.eigenvalues
        return float(jnp.max(jnp.abs(rel)))

    errors = {q: np.array([error(q, s) for s in range(10)]) for q in (0, 1, 3)}
    means = {q: e.mean() for q, e in errors.items()}
    assert means[0] > means[1] > means[3]
    spread = errors[0].std(ddof=1) + errors[3].std(ddof=1)
    assert means[0] - means[3] > 4.0 * spread / np.sqrt(10)
