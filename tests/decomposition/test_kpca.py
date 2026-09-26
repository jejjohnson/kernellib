"""Kernel PCA."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from sklearn.decomposition import KernelPCA as SkKernelPCA

import kernellib as kl


K = kl.RBF(lengthscale=1.5)


def _X(n=60, seed=0):
    return jax.random.normal(jax.random.key(seed), (n, 3))


def _match_signs(a, b):
    return b * np.sign(np.sum(a * b, axis=0))


def test_matches_sklearn_with_a_precomputed_kernel():
    X, X_new = _X(), _X(n=10, seed=1)
    ours = kl.KernelPCA(K, n_components=3).fit(X)
    sk = SkKernelPCA(n_components=3, kernel="precomputed").fit(np.asarray(K(X, X)))
    assert np.allclose(ours.eigenvalues, sk.eigenvalues_, rtol=1e-8)
    emb = np.asarray(ours.embedding)
    assert np.allclose(
        emb, _match_signs(emb, sk.transform(np.asarray(K(X, X)))), atol=1e-8
    )
    new = np.asarray(ours.transform(X_new))
    sk_new = sk.transform(np.asarray(K(X_new, X)))
    assert np.allclose(new, _match_signs(new, sk_new), atol=1e-8)


def test_transform_of_training_points_is_the_embedding():
    X = _X()
    kpca = kl.KernelPCA(K, n_components=4).fit(X)
    assert jnp.allclose(kpca.transform(X), kpca.embedding, atol=1e-8)
    assert jnp.all(jnp.diff(kpca.explained_variance) <= 0)


def test_linear_kernel_is_pca():
    X = _X()
    kpca = kl.KernelPCA(kl.Linear(), n_components=2).fit(X)
    Xc = np.asarray(X - X.mean(0))
    _, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    pcs = Xc @ Vt[:2].T
    emb = np.asarray(kpca.embedding)
    assert np.allclose(emb, _match_signs(emb, pcs), atol=1e-8)
    assert np.allclose(kpca.eigenvalues, s[:2] ** 2, rtol=1e-8)


def test_full_rank_nystrom_recovers_the_exact_components():
    X = _X(n=40)
    exact = kl.KernelPCA(K, n_components=3).fit(X)
    approx = kl.KernelPCA(
        K,
        n_components=3,
        approx=kl.NystromFeatures(40, jax.random.key(0), jitter=1e-12),
    ).fit(X)
    assert jnp.allclose(approx.eigenvalues, exact.eigenvalues, rtol=1e-5)
    emb = np.asarray(approx.embedding)
    assert np.allclose(emb, _match_signs(emb, np.asarray(exact.embedding)), atol=1e-5)
    assert jnp.allclose(approx.transform(X), approx.embedding, atol=1e-8)


def test_gradients_reach_the_kernel():
    X = _X(n=30)

    def top_variance(ell):
        return (
            kl.KernelPCA(kl.RBF(lengthscale=ell), n_components=1).fit(X).eigenvalues[0]
        )

    assert jnp.isfinite(jax.grad(top_variance)(1.0))


def test_errors():
    with pytest.raises(ValueError, match="n_components"):
        kl.KernelPCA(K, n_components=0)
    with pytest.raises(ValueError, match="exceeds the number of points"):
        kl.KernelPCA(K, n_components=10).fit(_X(n=5))
    with pytest.raises(RuntimeError, match="not fitted"):
        kl.KernelPCA(K).transform(_X(n=2))
