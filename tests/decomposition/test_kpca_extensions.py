"""KernelPCA extensions: centring helper, supervised / fair KPCA, pre-images."""

from __future__ import annotations

import itertools

import einx
import jax
import jax.numpy as jnp
import lineax as lx
import pytest

import kernellib as kl
from kernellib import functional as F


def _data(n=80, key=0):
    k1, k2 = jax.random.split(jax.random.key(key))
    X = jax.random.normal(k1, (n, 3))
    T = X[:, :1] + 0.3 * jax.random.normal(k2, (n, 1))
    return X, T


K = kl.RBF(lengthscale=1.5)


def _abs_cos(A, B):
    """|cosine| between matching columns: 1 when equal up to sign."""
    num = jnp.abs(einx.dot("n k, n k -> k", A, B))
    return num / jnp.sqrt(
        einx.dot("n k, n k -> k", A, A) * einx.dot("n k, n k -> k", B, B)
    )


def _centred_gram(X):
    G = K(X, X)
    return F.center_kernel(lx.MatrixLinearOperator(G, lx.symmetric_tag)).as_matrix()


def test_center_cross_kernel_on_the_training_points_is_center_kernel():
    X, _ = _data()
    G = K(X, X)
    got = F.center_cross_kernel(G, einx.mean("i j -> j", G), jnp.mean(G))
    assert jnp.allclose(got, _centred_gram(X), atol=1e-12)


def test_zero_weight_is_plain_kernel_pca():
    X, T = _data()
    plain = kl.KernelPCA(K, n_components=3).fit(X)
    with_target = kl.KernelPCA(K, n_components=3).fit(X, target=T)
    assert bool(jnp.all(plain.embedding == with_target.embedding))
    assert bool(jnp.all(plain.alphas == with_target.alphas))


def test_vanishing_weight_matches_plain_kernel_pca():
    # A traced weight takes the supervised branch; at gamma -> 0 it must agree.
    X, T = _data()
    plain = kl.KernelPCA(K, n_components=3).fit(X)
    tiny = kl.KernelPCA(K, n_components=3, target_weight=jnp.asarray(1e-12)).fit(
        X, target=T
    )
    assert jnp.allclose(_abs_cos(plain.embedding, tiny.embedding), 1.0, atol=1e-8)
    assert jnp.allclose(plain.eigenvalues, tiny.eigenvalues, rtol=1e-8)


@pytest.mark.parametrize("gamma", [-30.0, 50.0])
def test_solves_the_generalised_eigenproblem(gamma):
    # A^T K~ A = I, and K~ G K~ A = K~ A diag(rho) with G = I/n + gamma H K_T H / n^2.
    X, T = _data()
    n = X.shape[0]
    m = kl.KernelPCA(K, n_components=3, target_weight=gamma).fit(X, target=T)
    Kc = _centred_gram(X)
    KA = Kc @ m.alphas
    gram = einx.dot("n a, n b -> a b", m.alphas, KA)
    assert jnp.allclose(gram, jnp.eye(3), atol=1e-8)
    KT = einx.dot("i p, j p -> i j", T, T)
    Tc = F.center_kernel(lx.MatrixLinearOperator(KT, lx.symmetric_tag)).as_matrix()
    G = jnp.eye(n) / n + gamma * Tc / n**2
    lhs = Kc @ G @ KA
    rhs = einx.multiply("n k, k -> n k", KA, m.eigenvalues / n)
    assert float(jnp.max(jnp.abs(lhs - rhs))) < 1e-10


def test_dependence_moves_monotonically_with_the_weight():
    X, T = _data()
    lin = kl.Linear()

    def dependence(gamma):
        m = kl.KernelPCA(K, n_components=2, target_weight=gamma).fit(X, target=T)
        return float(kl.hsic(lin, lin, m.embedding, T))

    up = [dependence(g) for g in (0.0, 10.0, 100.0, 1000.0)]
    down = [dependence(g) for g in (0.0, -10.0, -100.0, -1000.0)]
    assert all(a <= b for a, b in itertools.pairwise(up))
    assert all(a >= b for a, b in itertools.pairwise(down))
    assert down[-1] < 1e-3 * down[0]


def test_approx_path_matches_the_exact_one_at_full_rank():
    X, T = _data(n=40)
    exact = kl.KernelPCA(K, n_components=2, target_weight=20.0).fit(X, target=T)
    approx = kl.KernelPCA(
        K,
        n_components=2,
        target_weight=20.0,
        approx=kl.NystromFeatures(40, jax.random.key(0), jitter=1e-12),
    ).fit(X, target=T)
    assert jnp.allclose(_abs_cos(exact.embedding, approx.embedding), 1.0, atol=1e-5)
    assert jnp.allclose(
        _abs_cos(exact.transform(X), approx.transform(X)), 1.0, atol=1e-5
    )


def test_pre_image_reconstructs_the_training_points():
    X, _ = _data(n=60)
    m = kl.KernelPCA(
        K, n_components=59, fit_inverse_transform=True, inverse_regularization=1e-10
    ).fit(X)
    rel = jnp.linalg.norm(m.inverse_transform(m.embedding) - X) / jnp.linalg.norm(X)
    assert float(rel) < 1e-3


def test_errors():
    X, _ = _data()
    with pytest.raises(ValueError, match="no target"):
        kl.KernelPCA(K, target_weight=1.0).fit(X)
    with pytest.raises(RuntimeError, match="fit_inverse_transform"):
        kl.KernelPCA(K).fit(X).inverse_transform(jnp.zeros((2, 2)))


@pytest.mark.integration
def test_pre_image_matches_scikit_learn():
    sklearn_decomposition = pytest.importorskip("sklearn.decomposition")
    X, _ = _data(n=60)
    X_test = X[:10] + 0.05
    lengthscale, alpha = 1.5, 0.1
    theirs = sklearn_decomposition.KernelPCA(
        n_components=5,
        kernel="rbf",
        gamma=float(kl.lengthscale_to_gamma(lengthscale)),
        fit_inverse_transform=True,
        alpha=alpha,
    ).fit(jax.device_get(X))
    ours = kl.KernelPCA(
        kl.RBF(lengthscale),
        n_components=5,
        fit_inverse_transform=True,
        inverse_kernel=kl.RBF(lengthscale),
        inverse_regularization=alpha / X.shape[0],  # KRR scales the ridge by n
    ).fit(X)
    expected = theirs.inverse_transform(theirs.transform(jax.device_get(X_test)))
    got = ours.inverse_transform(ours.transform(X_test))
    assert jnp.allclose(got, expected, atol=1e-6)


def _rank3():
    # A Linear kernel on 3-D inputs: the centred Gram has rank exactly 3.
    k1, k2 = jax.random.split(jax.random.key(7))
    X = jax.random.normal(k1, (30, 3))
    T = X[:, :1] + 0.1 * jax.random.normal(k2, (30, 1))
    return X, T


def test_fair_kpca_never_selects_null_space_directions():
    # With a strongly negative weight, a valid direction scores below 0; the
    # zero-scored null-space directions must still not be chosen (Codex P1).
    X, T = _rank3()
    m = kl.KernelPCA(kl.Linear(), n_components=3, target_weight=-1e4).fit(X, target=T)
    Kc = F.center_kernel(
        lx.MatrixLinearOperator(einx.dot("n d, m d -> n m", X, X), lx.symmetric_tag)
    )
    Kc = Kc.as_matrix()
    gram = einx.dot("n a, n b -> a b", m.alphas, Kc @ m.alphas)
    assert jnp.allclose(gram, jnp.eye(3), atol=1e-8)
    assert bool(jnp.all(einx.sum("n k -> k", m.embedding**2) > 1e-6))
    assert float(m.eigenvalues[-1]) < 0.0  # the valid negative direction is kept


def test_fair_kpca_approx_never_selects_null_space_directions():
    X, T = _rank3()
    m = kl.KernelPCA(
        kl.Linear(),
        n_components=3,
        target_weight=-1e4,
        approx=kl.NystromFeatures(10, jax.random.key(0), jitter=1e-12),
    ).fit(X, target=T)
    assert jnp.allclose(
        einx.dot("r a, r b -> a b", m.components, m.components), jnp.eye(3), atol=1e-8
    )
    assert bool(jnp.all(einx.sum("n k -> k", m.embedding**2) > 1e-6))


@pytest.mark.parametrize("approx", [False, True])
def test_more_components_than_the_rank_raise(approx):
    X, T = _rank3()
    kw = {"approx": kl.NystromFeatures(10, jax.random.key(0))} if approx else {}
    with pytest.raises(ValueError, match="exceeds the rank 3"):
        kl.KernelPCA(kl.Linear(), n_components=4, target_weight=-1.0, **kw).fit(
            X, target=T
        )


def test_concrete_array_zero_weight_needs_no_target():
    X, _ = _data()
    plain = kl.KernelPCA(K, n_components=3).fit(X)
    zero = kl.KernelPCA(K, n_components=3, target_weight=jnp.asarray(0.0)).fit(X)
    assert bool(jnp.all(plain.embedding == zero.embedding))
