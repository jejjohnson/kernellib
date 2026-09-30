"""Mini-batch CKA (`CKAAccumulator`)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

import kernellib as kl
from kernellib._einx import rearrange


def _data(n=2000, key=0):
    k1, k2 = jax.random.split(jax.random.key(key))
    X = jax.random.normal(k1, (n, 3))
    f = jnp.sin(2.0 * X[:, :1]) + 0.3 * jax.random.normal(k2, (n, 1))
    return X[:, :1], f


K = kl.RBF(lengthscale=1.0)


def _accumulate(X, Y, batch):
    acc = kl.CKAAccumulator(K, K)
    for start in range(0, X.shape[0], batch):
        acc = acc.update(X[start : start + batch], Y[start : start + batch])
    return acc


def test_one_batch_is_unbiased_cka():
    X, Y = _data(200)
    acc = kl.CKAAccumulator(K, K).update(X, Y)
    expected = kl.cka(K, K, X, Y, estimator="unbiased")
    assert jnp.allclose(acc.result(), expected, rtol=1e-12)
    assert int(acc.n_batches) == 1


def test_batch_size_does_not_bias_the_estimate():
    # Unbiased per-batch HSIC keeps the ratio consistent: batch 10 and batch
    # 1000 agree, where a biased accumulator drifts by O(1/B) (keras-fairkl#19
    # measured +16 % at B = 10).
    X, Y = _data()
    small = float(_accumulate(X, Y, 10).result())
    large = float(_accumulate(X, Y, 1000).result())
    assert abs(small - large) < 0.01


def test_runs_under_scan():
    X, Y = _data(400)
    batches_x = rearrange(X, "(b n) d -> b n d", b=8)
    batches_y = rearrange(Y, "(b n) d -> b n d", b=8)

    def step(acc, xy):
        return acc.update(*xy), None

    acc, _ = jax.lax.scan(step, kl.CKAAccumulator(K, K), (batches_x, batches_y))
    assert jnp.allclose(acc.result(), _accumulate(X, Y, 50).result(), rtol=1e-12)
    assert int(acc.n_batches) == 8


def test_degenerate_is_zero_and_empty_is_zero():
    X, _ = _data(100)
    assert float(kl.CKAAccumulator(K, K).result()) == 0.0
    acc = kl.CKAAccumulator(K, K).update(X, jnp.zeros_like(X))
    assert abs(float(acc.result())) < 1e-12  # 0 up to U-centring rounding


def test_small_batches_raise():
    X, Y = _data(3)
    with pytest.raises(ValueError, match="n >= 4"):
        kl.CKAAccumulator(K, K).update(X, Y)
