"""KRR with a quadratic penalty: fair KRR (HSIC) and LapRLS (graph)."""

from __future__ import annotations

import itertools

import equinox as eqx
import gaussx as gx
import jax
import jax.numpy as jnp
import lineax as lx
import pytest

import kernellib as kl
from kernellib._einx import rearrange


def _data(n=60, key=0):
    k1, k2, k3 = jax.random.split(jax.random.key(key), 3)
    X = jax.random.normal(k1, (n, 2))
    S = X[:, :1] + 0.1 * jax.random.normal(k2, (n, 1))  # protected, correlated
    y = jnp.sin(X[:, 0]) + X[:, 1] + 0.05 * jax.random.normal(k3, (n,))
    return X, y, S


KERNEL = kl.RBF(lengthscale=1.0)


def _krr(mu, **kw):
    return kl.KRR(KERNEL, regularization=1e-3, penalty_weight=mu, **kw)


def _objective(alpha, X, y, M, mu, lam, mask):
    K = KERNEL(X, X)
    J = mask.astype(X.dtype)
    n_lab = jnp.sum(J)
    r = J * (y - K @ alpha)
    return r @ r / n_lab + lam * alpha @ K @ alpha + mu * alpha @ K @ M @ K @ alpha


def test_zero_weight_is_plain_krr():
    X, y, S = _data()
    plain = kl.KRR(KERNEL, regularization=1e-3).fit(X, y)
    zero = _krr(0.0).fit(X, y, penalty=kl.hsic_penalty(kl.Linear(), S))
    assert bool(jnp.all(plain.alpha == zero.alpha))  # the same code path
    all_true = _krr(0.0).fit(X, y, mask=jnp.ones(X.shape[0], dtype=bool))
    assert jnp.allclose(all_true.alpha, plain.alpha, rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize("penalty", ["hsic_rbf", "laplacian"])
@pytest.mark.parametrize("masked", [False, True])
def test_solution_is_stationary(penalty, masked):
    X, y, S = _data()
    n = X.shape[0]
    if penalty == "hsic_rbf":
        M = kl.hsic_penalty(kl.RBF(lengthscale=0.5), S)
    else:
        M = kl.laplacian_penalty(kl.adjacency_matrix(kl.nearest_neighbors(X, 8)))
    mask = (jnp.arange(n) % 3 != 0) if masked else jnp.ones(n, dtype=bool)
    model = _krr(5.0).fit(X, y, penalty=M, mask=mask if masked else None)
    grad = jax.grad(_objective)(model.alpha, X, y, M.as_matrix(), 5.0, 1e-3, mask)
    assert float(jnp.max(jnp.abs(grad))) < 1e-6


def test_woodbury_matches_dense_and_every_strategy():
    X, y, S = _data()
    M = kl.hsic_penalty(kl.Linear(variance=2.0), S)
    assert isinstance(M, gx.LowRankUpdate) and M.U.shape == (X.shape[0], 1)
    woodbury = _krr(10.0).fit(X, y, penalty=M)
    dense = _krr(10.0).fit(X, y, penalty=M, mask=jnp.ones(X.shape[0], dtype=bool))
    assert jnp.allclose(woodbury.alpha, dense.alpha, rtol=1e-8, atol=1e-10)
    cg = _krr(
        10.0, solver=gx.CGSolver(rtol=1e-12, atol=1e-12, max_steps=2000), implicit=True
    ).fit(X, y, penalty=M)
    assert jnp.allclose(cg.alpha, dense.alpha, rtol=1e-6, atol=1e-8)


def test_low_rank_penalty_matches_its_dense_form():
    _, _, S = _data()
    low = kl.hsic_penalty(kl.Linear(variance=2.0, bias=0.3), S).as_matrix()
    dense_kernel = kl.Linear(variance=2.0, bias=0.3)
    K = dense_kernel(S, S)
    n = S.shape[0]
    H = jnp.eye(n) - 1.0 / n
    assert jnp.allclose(low, H @ K @ H / n**2, atol=1e-12)


def test_multi_output_equals_per_column():
    X, y, S = _data()
    Y = jnp.stack([y, -2.0 * y + 1.0], axis=1)
    M = kl.hsic_penalty(kl.Linear(), S)
    both = _krr(10.0).fit(X, Y, penalty=M).alpha
    for c in range(2):
        col = _krr(10.0).fit(X, Y[:, c], penalty=M).alpha
        assert jnp.allclose(both[:, c], col, rtol=1e-10)


def test_continuous_at_zero_weight():
    # keras-fairkl#15: a vanishing fairness weight must not change the ridge.
    X, y, S = _data()
    M = kl.hsic_penalty(kl.Linear(), S)
    at_zero = kl.KRR(KERNEL, regularization=1e-3).fit(X, y).predict(X)
    near_zero = _krr(1e-8).fit(X, y, penalty=M).predict(X)
    assert jnp.allclose(near_zero, at_zero, rtol=1e-6, atol=1e-8)


def test_dependence_decreases_with_the_weight():
    X, y, S = _data()
    M = kl.hsic_penalty(kl.Linear(), S)
    lin = kl.Linear()
    values = [
        float(
            kl.hsic(
                lin,
                lin,
                rearrange(_krr(mu).fit(X, y, penalty=M).predict(X), "n -> n 1"),
                S,
            )
        )
        for mu in (0.0, 1.0, 10.0, 100.0, 1000.0)
    ]
    assert all(a > b for a, b in itertools.pairwise(values))


def test_implicit_normal_form_matches_dense():
    X, y, _ = _data()
    n = X.shape[0]
    M = kl.laplacian_penalty(kl.adjacency_matrix(kl.nearest_neighbors(X, 8)))
    mask = jnp.arange(n) % 2 == 0
    dense = _krr(2.0).fit(X, y, penalty=M, mask=mask)
    implicit = _krr(
        2.0, solver=gx.CGSolver(rtol=1e-12, atol=1e-12, max_steps=5000), implicit=True
    ).fit(X, y, penalty=M, mask=mask)
    assert jnp.allclose(implicit.predict(X), dense.predict(X), rtol=1e-5, atol=1e-7)


def test_unlabelled_targets_may_be_nan():
    X, y, _ = _data()
    mask = jnp.arange(X.shape[0]) < 30
    M = kl.laplacian_penalty(kl.adjacency_matrix(kl.nearest_neighbors(X, 8)))
    clean = _krr(1.0).fit(X, y, penalty=M, mask=mask)
    nan_y = jnp.where(mask, y, jnp.nan)
    dirty = _krr(1.0).fit(X, nan_y, penalty=M, mask=mask)
    assert jnp.allclose(dirty.alpha, clean.alpha)


def test_jit_fit_with_penalty():
    X, y, S = _data()
    M = kl.hsic_penalty(kl.Linear(), S)

    @eqx.filter_jit
    def fit(model, X, y, M):
        return model.fit(X, y, penalty=M).alpha

    assert jnp.allclose(fit(_krr(10.0), X, y, M), _krr(10.0).fit(X, y, penalty=M).alpha)


def test_errors():
    X, y, S = _data()
    with pytest.raises(ValueError, match="mask"):
        _krr(1.0).fit(X, y, mask=jnp.ones(3, dtype=bool))
    with pytest.raises(ValueError, match="no labelled points"):
        _krr(1.0).fit(X, y, mask=jnp.zeros(X.shape[0], dtype=bool))
    bad = gx.LowRankUpdate(
        base=lx.DiagonalLinearOperator(jnp.ones(X.shape[0])),
        U=S,
        d=jnp.ones(1),
        V=S,
    )
    with pytest.raises(ValueError, match="zero diagonal base"):
        _krr(1.0).fit(X, y, penalty=bad)


def _two_moons(n, key, noise=0.05):
    k1, k2 = jax.random.split(key)
    t = jnp.pi * jax.random.uniform(k1, (n,))
    upper = jnp.arange(n) < n // 2
    x = jnp.where(upper, jnp.cos(t), 1.0 - jnp.cos(t))
    z = jnp.where(upper, jnp.sin(t), 0.5 - jnp.sin(t))
    X = jnp.stack([x, z], axis=1) + noise * jax.random.normal(k2, (n, 2))
    return X, jnp.where(upper, 1.0, -1.0)


@pytest.mark.slow
def test_laprls_classifies_two_moons_from_two_labels():
    X, labels = _two_moons(200, jax.random.key(0))
    labelled = jnp.array([0, 199])  # one label per moon
    mask = jnp.zeros(200, dtype=bool).at[labelled].set(True)
    W = kl.adjacency_matrix(kl.nearest_neighbors(X, 10))
    laprls = kl.KRR(kl.RBF(0.3), regularization=1e-4, penalty_weight=100.0).fit(
        X, labels, mask=mask, penalty=kl.laplacian_penalty(W)
    )
    plain = kl.KRR(kl.RBF(0.3), regularization=1e-4).fit(X[labelled], labels[labelled])
    accuracy = jnp.mean(jnp.sign(laprls.predict(X)) == labels)
    baseline = jnp.mean(jnp.sign(plain.predict(X)) == labels)
    assert float(accuracy) > 0.95
    assert float(baseline) < 0.85


@pytest.mark.parametrize("dtype", [jnp.int32, jnp.bool_])
def test_hsic_penalty_keeps_a_fractional_variance_for_discrete_attributes(dtype):
    S = (jnp.arange(40) % 2).astype(dtype)[:, None]
    M = kl.hsic_penalty(kl.Linear(variance=0.5), S)
    assert jnp.issubdtype(M.U.dtype, jnp.floating)
    assert jnp.allclose(M.d, 0.5)
    dense = kl.hsic_penalty(kl.Linear(variance=0.5), S.astype(float))
    assert jnp.allclose(M.as_matrix(), dense.as_matrix())


def test_concrete_array_zero_weight_is_plain_krr():
    X, y, S = _data()
    plain = kl.KRR(KERNEL, regularization=1e-3).fit(X, y)
    zero = kl.KRR(KERNEL, regularization=1e-3, penalty_weight=jnp.asarray(0.0)).fit(
        X, y, penalty=kl.hsic_penalty(kl.Linear(), S)
    )
    assert bool(jnp.all(plain.alpha == zero.alpha))
