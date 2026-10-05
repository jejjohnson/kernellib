"""K5: kernel locality preserving projections and kernel Schrödinger
projections, exact and with feature maps."""

from __future__ import annotations

import einx
import jax
import jax.numpy as jnp
import pytest

import kernellib as kl
from kernellib._einx import einsum, reduce


def _same_subspace(A, B, atol=1e-6):
    """Smallest cosine of the principal angles between the column spaces."""
    Qa, _ = jnp.linalg.qr(A)
    Qb, _ = jnp.linalg.qr(B)
    cos = jnp.linalg.svd(einsum(Qa, Qb, "n i, n j -> i j"), compute_uv=False)
    return float(jnp.min(cos)) > 1 - atol


def _data(n=60, d=3, seed=0):
    X = jax.random.normal(jax.random.key(seed), (n, d))
    W = kl.adjacency_matrix(kl.nearest_neighbors(X, 8))
    return X, W


def _labels(n, m=10):
    return kl.label_potential(jnp.where(jnp.arange(n) < m, 0, -1))


@pytest.mark.slow
class TestKernelLPP:
    def test_linear_kernel_spans_the_lpp_embedding(self):
        # y = K̄ α with K̄ = X̄ X̄ᵀ is linear in the (degree-centred) inputs.
        X, W = _data(n=80, d=4)
        lpp = kl.LocalityPreservingProjections(n_components=2).fit(X, graph=W)
        klpp = kl.KernelLocalityPreservingProjections(
            kl.Linear(), n_components=2, regularization=1e-10
        ).fit(X, graph=W)
        assert _same_subspace(klpp.embedding, lpp.transform(X), atol=1e-6)
        assert jnp.allclose(klpp.eigenvalues, lpp.eigenvalues, rtol=1e-4)

    def test_embedding_is_degree_centred_and_d_orthonormal(self):
        X, W = _data()
        klpp = kl.KernelLocalityPreservingProjections(kl.RBF(1.0), n_components=3)
        Y = klpp.fit(X, graph=W).embedding
        degree = reduce(W, "i j -> i", "sum")
        assert jnp.allclose(degree @ Y, 0.0, atol=1e-8)  # D-orthogonal to 1
        gram = einsum(einx.multiply("n a, n -> n a", Y, degree), Y, "n a, n b -> a b")
        assert jnp.allclose(gram, jnp.eye(3), atol=1e-8)  # Y^T D Y = I, as LPP
        assert jnp.all(jnp.diff(klpp.fit(X, graph=W).eigenvalues) >= 0)

    def test_transform_reproduces_the_training_embedding(self):
        X, _ = _data()
        klpp = kl.KernelLocalityPreservingProjections(kl.RBF(1.0)).fit(X)
        assert jnp.allclose(klpp.transform(X), klpp.embedding, atol=1e-8)
        approx = kl.KernelLocalityPreservingProjections(
            kl.RBF(1.0), approx=kl.RandomFourierFeatures(50, jax.random.key(0))
        ).fit(X)
        assert jnp.allclose(approx.transform(X), approx.embedding, atol=1e-8)

    def test_more_regularisation_is_smoother(self):
        # The ridge penalises ||f||_H: a larger one trades graph fit (larger
        # eigenvalues) for smoothness.
        X, W = _data()
        lam = [
            kl.KernelLocalityPreservingProjections(kl.RBF(1.0), regularization=r)
            .fit(X, graph=W)
            .eigenvalues[0]
            for r in (1e-4, 1e-2, 1.0)
        ]
        assert lam[0] < lam[1] < lam[2]

    def test_errors(self):
        X, _ = _data(d=2)
        with pytest.raises(ValueError, match="rank"):
            kl.KernelLocalityPreservingProjections(kl.Linear(), n_components=3).fit(X)
        with pytest.raises(ValueError, match="features"):
            kl.KernelLocalityPreservingProjections(
                kl.RBF(),
                n_components=6,
                approx=kl.NystromFeatures(5, jax.random.key(0)),
            ).fit(X)
        with pytest.raises(RuntimeError, match="not fitted"):
            kl.KernelLocalityPreservingProjections(kl.RBF()).transform(X)
        with pytest.raises(ValueError, match="one node per point"):
            kl.KernelLocalityPreservingProjections(kl.RBF()).fit(
                X, graph=jnp.ones((5, 5))
            )

    def test_nystrom_converges_to_exact(self):
        # Nyström on M landmarks drawn without replacement: at M = N it
        # reproduces K (up to jitter), and the ridge, relative to tr(F^T D F)
        # over the feature dimension, then matches the exact path's.
        X, W = _data(n=60)
        kernel = kl.RBF(1.5)
        exact = kl.KernelLocalityPreservingProjections(kernel, n_components=2)
        Y = exact.fit(X, graph=W).embedding
        errors = []
        for m in (8, 20, 60):
            approx = kl.KernelLocalityPreservingProjections(
                kernel, n_components=2, approx=kl.NystromFeatures(m, jax.random.key(1))
            ).fit(X, graph=W)
            Qa, _ = jnp.linalg.qr(approx.embedding)
            Qb, _ = jnp.linalg.qr(Y)
            cos = jnp.linalg.svd(einsum(Qa, Qb, "n i, n j -> i j"), compute_uv=False)
            errors.append(1.0 - float(jnp.min(cos)))
        assert errors[0] > errors[1] > errors[2]
        assert errors[2] < 1e-6


@pytest.mark.slow
class TestKernelSchrodinger:
    def test_zero_alpha_is_kernel_lpp(self):
        X, W = _data()
        klpp = kl.KernelLocalityPreservingProjections(kl.RBF(1.0)).fit(X, graph=W)
        ksep = kl.KernelSchrodingerProjections(kl.RBF(1.0), alpha=0.0)
        ksep = ksep.fit(X, _labels(60), graph=W)
        assert jnp.array_equal(ksep.embedding, klpp.embedding)

    def test_label_potential_pulls_a_class_together(self):
        X, W = _data(n=80)

        def spread(Y):
            def radius(Z):
                c = einx.subtract("n k, k -> n k", Z, reduce(Z, "n k -> k", "mean"))
                return jnp.mean(jnp.sqrt(reduce(c**2, "n k -> n", "sum")))

            return radius(Y[:15]) / radius(Y)

        V = _labels(80, 15)
        base = kl.KernelLocalityPreservingProjections(kl.RBF(1.0)).fit(X, graph=W)
        for approx in (None, kl.NystromFeatures(40, jax.random.key(0))):
            ksep = kl.KernelSchrodingerProjections(
                kl.RBF(1.0), alpha=10.0, approx=approx
            ).fit(X, V, graph=W)
            assert spread(ksep.transform(X)) < 0.5 * spread(base.embedding)


def test_kernel_schrodinger_potential_shape_check():
    X = jnp.ones((6, 2))
    with pytest.raises(ValueError, match="potential must have shape"):
        kl.KernelSchrodingerProjections(kl.RBF()).fit(X, jnp.ones(3))
