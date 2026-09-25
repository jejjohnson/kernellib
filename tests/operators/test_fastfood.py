"""Tests for FastFood random features (gaussx#62, implemented here).

Deterministic tests pin their keys: the draw is incidental to what they check.
The two statistical tests are bounded by the estimator's own spread across
independent draws, not by a fixed tolerance; each says where its bound comes
from.
"""

from __future__ import annotations

import gaussx as gx
import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
import numpy as np
import pytest
import scipy.linalg

import kernellib as kl


# --- Walsh-Hadamard transform ----------------------------------------------


@pytest.mark.parametrize("d", [1, 2, 4, 8, 64])
def test_hadamard_matches_dense_sylvester_matrix(d):
    x = jr.normal(jr.key(0), (3, d))
    expected = x @ jnp.asarray(scipy.linalg.hadamard(d), dtype=x.dtype).T
    assert jnp.allclose(kl.hadamard_transform(x), expected, atol=1e-10)


def test_hadamard_is_its_own_inverse_up_to_d():
    x = jr.normal(jr.key(1), (5, 16))
    assert jnp.allclose(kl.hadamard_transform(kl.hadamard_transform(x)), 16 * x)


def test_hadamard_jits_and_batches():
    x = jr.normal(jr.key(2), (2, 3, 8))
    expected = kl.hadamard_transform(x)
    assert jnp.allclose(jax.jit(kl.hadamard_transform)(x), expected)


@pytest.mark.parametrize("d", [3, 6, 12])
def test_hadamard_rejects_non_power_of_two(d):
    with pytest.raises(ValueError, match="power-of-two"):
        kl.hadamard_transform(jnp.ones(d))


# --- parameters -------------------------------------------------------------


def test_params_shapes_and_structure():
    p = kl.fastfood_params(d=5, n_components=20, lengthscale=0.7, key=jr.key(0))
    assert (p.d_padded, p.n_stacks) == (8, 3)
    for leaf in (p.B, p.G, p.P, p.S):
        assert leaf.shape == (3, 8)
    assert set(np.unique(np.asarray(p.B)).tolist()) <= {-1.0, 1.0}
    for row in np.asarray(p.P):
        assert sorted(row.tolist()) == list(range(8))
    assert bool(jnp.all(p.S > 0))


def test_params_are_deterministic_under_a_key():
    a = kl.fastfood_params(d=4, n_components=16, lengthscale=1.0, key=jr.key(3))
    b = kl.fastfood_params(d=4, n_components=16, lengthscale=1.0, key=jr.key(3))
    assert all(
        bool(jnp.array_equal(x, y))
        for x, y in zip(
            jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b), strict=True
        )
    )


def test_params_reject_bad_sizes():
    with pytest.raises(ValueError, match="positive"):
        kl.fastfood_params(d=0, n_components=4, lengthscale=1.0, key=jr.key(0))
    with pytest.raises(ValueError, match="positive"):
        kl.fastfood_params(d=3, n_components=0, lengthscale=1.0, key=jr.key(0))


# --- the structured product -------------------------------------------------


def _dense_v(p):
    """V = S H G Pi H B / (lengthscale sqrt(dp)) built from dense matrices."""
    dp = p.d_padded
    H = jnp.asarray(scipy.linalg.hadamard(dp), dtype=p.G.dtype)
    blocks = []
    for s in range(p.n_stacks):
        Pi = jnp.eye(dp)[p.P[s]]
        M = H @ jnp.diag(p.G[s]) @ Pi @ H @ jnp.diag(p.B[s])
        M = jnp.diag(p.S[s] / (jnp.sqrt(dp) * jnp.linalg.norm(p.G[s]))) @ M
        blocks.append(M)
    V = jnp.concatenate(blocks, axis=0)[: p.n_components, : p.d]
    return V / p.lengthscale


@pytest.mark.parametrize("lengthscale", [0.7, jnp.array([0.5, 1.0, 2.0, 0.8, 1.3])])
def test_frequencies_match_dense_product(lengthscale):
    p = kl.fastfood_params(d=5, n_components=20, lengthscale=lengthscale, key=jr.key(0))
    assert jnp.allclose(kl.fastfood_frequencies(p), _dense_v(p), atol=1e-10)


def test_row_lengths_are_s_over_lengthscale_before_padding_truncation():
    """With d a power of two nothing is truncated: every row has length S / l."""
    p = kl.fastfood_params(d=8, n_components=16, lengthscale=0.5, key=jr.key(1))
    norms = jnp.linalg.norm(kl.fastfood_frequencies(p), axis=1)
    expected = (p.S / 0.5).reshape(-1)
    assert jnp.allclose(norms, expected, rtol=1e-10)


def test_features_equal_cos_sin_of_dense_frequencies():
    p = kl.fastfood_params(d=5, n_components=20, lengthscale=0.9, key=jr.key(2))
    X = jr.normal(jr.key(3), (7, 5))
    Z = X @ kl.fastfood_frequencies(p).T
    expected = jnp.concatenate([jnp.cos(Z), jnp.sin(Z)], axis=-1) / jnp.sqrt(20.0)
    assert jnp.allclose(kl.fastfood_features(X, p), expected, atol=1e-10)


def test_features_reject_wrong_input_dimension():
    p = kl.fastfood_params(d=5, n_components=8, lengthscale=1.0, key=jr.key(0))
    with pytest.raises(ValueError, match="d=5"):
        kl.fastfood_features(jnp.ones((2, 4)), p)


def test_constant_ard_lengthscale_matches_scalar():
    kw = dict(d=3, n_components=16, key=jr.key(4))
    X = jr.normal(jr.key(5), (6, 3))
    a = kl.fastfood_features(X, kl.fastfood_params(lengthscale=0.8, **kw))
    b = kl.fastfood_features(X, kl.fastfood_params(lengthscale=jnp.full(3, 0.8), **kw))
    assert jnp.allclose(a, b, atol=1e-12)


# --- operator ---------------------------------------------------------------


def test_operator_is_psd_low_rank_update_and_solves():
    p = kl.fastfood_params(d=3, n_components=64, lengthscale=1.0, key=jr.key(6))
    X = jr.normal(jr.key(7), (40, 3))
    y = jr.normal(jr.key(8), (40,))
    K = kl.fastfood_operator(X, p)
    assert isinstance(K, gx.LowRankUpdate)
    assert lx.is_symmetric(K)
    assert lx.is_positive_semidefinite(K)
    Phi = kl.fastfood_features(X, p)
    assert jnp.allclose(K.as_matrix(), Phi @ Phi.T, atol=1e-12)

    noisy = gx.LowRankUpdate(
        lx.DiagonalLinearOperator(jnp.full(40, 0.1)), K.U, K.d, K.V
    )
    dense = K.as_matrix() + 0.1 * jnp.eye(40)
    assert jnp.allclose(gx.solve(noisy, y), jnp.linalg.solve(dense, y), atol=1e-8)


def test_gradient_through_lengthscale_is_finite():
    X = jr.normal(jr.key(9), (10, 3))

    def loss(ls):
        p = kl.fastfood_params(d=3, n_components=32, lengthscale=ls, key=jr.key(10))
        return jnp.sum(kl.fastfood_features(X, p) ** 2 * jnp.arange(64.0))

    g = jax.grad(loss)(0.7)
    assert jnp.isfinite(g)
    assert g != 0.0


# --- statistical behaviour ----------------------------------------------------


def _rbf(X, lengthscale):
    return kl.RBF(lengthscale=lengthscale)(X, X)


def _estimates(fn, n_draws):
    return jnp.stack([fn(jr.key(100 + i)) for i in range(n_draws)])


@pytest.mark.slow
def test_fastfood_gram_has_no_detectable_bias():
    """Average the Gram estimate over independent draws and compare with the
    exact kernel entry by entry. The bound is 7 standard errors of that
    average, estimated from the draws themselves (the rule in CLAUDE.md), plus
    a small floor for entries where every draw agrees."""
    X = jr.normal(jr.key(11), (12, 6))
    exact = _rbf(X, 1.3)
    n_draws = 40

    def one(key):
        p = kl.fastfood_params(d=6, n_components=64, lengthscale=1.3, key=key)
        Phi = kl.fastfood_features(X, p)
        return Phi @ Phi.T

    draws = _estimates(one, n_draws)
    se = jnp.std(draws, axis=0, ddof=1) / jnp.sqrt(n_draws)
    assert bool(jnp.all(jnp.abs(draws.mean(axis=0) - exact) <= 7 * se + 1e-6))


@pytest.mark.slow
def test_fastfood_error_is_comparable_to_rff():
    """At the same number of feature columns, FastFood's mean relative
    Frobenius error stays within a factor of 1.5 of RFF's (Le et al. 2013
    prove the same concentration rate). Averaged over draws so the comparison
    is between the two estimators, not two samples."""
    d, n_feat, ls = 16, 256, 2.0
    X = jr.normal(jr.key(12), (60, d))
    exact = _rbf(X, ls)
    norm = jnp.linalg.norm(exact)

    def ff_err(key):
        p = kl.fastfood_params(d=d, n_components=n_feat // 2, lengthscale=ls, key=key)
        Phi = kl.fastfood_features(X, p)
        return jnp.linalg.norm(Phi @ Phi.T - exact) / norm

    def rff_err(key):
        k1, k2 = jr.split(key)
        omega = jr.normal(k1, (n_feat, d)) / ls
        b = jr.uniform(k2, (n_feat,), maxval=2 * jnp.pi)
        return jnp.linalg.norm(kl.rff_operator(X, omega, b).as_matrix() - exact) / norm

    ff = _estimates(ff_err, 20).mean()
    rff = _estimates(rff_err, 20).mean()
    assert ff < 1.5 * rff
