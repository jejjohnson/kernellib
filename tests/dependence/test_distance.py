"""Distance covariance / correlation and energy distance against references.

The references are Székely's definitions on double-centred (or U-centred)
distance matrices and pairwise distance means, computed directly in numpy,
so the tests check the HSIC / MMD identities rather than restating them.
Inputs use pinned keys: the randomness is incidental.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import kernellib as kl


def _pair(n=40, key=0):
    kx, ky = jax.random.split(jax.random.key(key))
    X = jax.random.normal(kx, (n, 2))
    Y = jnp.sin(X[:, :1]) + 0.3 * jax.random.normal(ky, (n, 1))
    return X, Y


def _dist(A, B, a):
    A, B = np.asarray(A), np.asarray(B)
    return np.linalg.norm(A[:, None, :] - B[None, :, :], axis=-1) ** a


def _double_centred(D):
    return D - D.mean(0, keepdims=True) - D.mean(1, keepdims=True) + D.mean()


def _u_centred(D):
    n = D.shape[0]
    U = (
        D
        - D.sum(0, keepdims=True) / (n - 2)
        - D.sum(1, keepdims=True) / (n - 2)
        + D.sum() / ((n - 1) * (n - 2))
    )
    np.fill_diagonal(U, 0.0)
    return U


def _dcov2(X, Y, a=1.0, estimator="biased"):
    Dx, Dy = _dist(X, X, a), _dist(Y, Y, a)
    if estimator == "biased":
        return np.mean(_double_centred(Dx) * _double_centred(Dy))
    n = Dx.shape[0]
    return np.sum(_u_centred(Dx) * _u_centred(Dy)) / (n * (n - 3))


@pytest.mark.parametrize("exponent", [0.5, 1.0, 1.5])
@pytest.mark.parametrize("estimator", ["biased", "unbiased"])
def test_distance_covariance_matches_szekely(exponent, estimator):
    X, Y = _pair()
    got = kl.distance_covariance_squared(X, Y, exponent=exponent, estimator=estimator)
    assert np.allclose(got, _dcov2(X, Y, exponent, estimator), rtol=1e-9)


@pytest.mark.parametrize("estimator", ["biased", "unbiased"])
def test_distance_correlation_matches_szekely(estimator):
    X, Y = _pair()
    expected = _dcov2(X, Y, 1.0, estimator) / np.sqrt(
        _dcov2(X, X, 1.0, estimator) * _dcov2(Y, Y, 1.0, estimator)
    )
    got = kl.distance_correlation_squared(X, Y, estimator=estimator)
    assert np.allclose(got, expected, rtol=1e-9)


@pytest.mark.parametrize("exponent", [0.5, 1.0])
def test_energy_distance_matches_pairwise_means(exponent):
    X = jax.random.normal(jax.random.key(0), (30, 2))
    Y = jax.random.normal(jax.random.key(1), (20, 2)) + 0.5
    Dxy, Dxx, Dyy = _dist(X, Y, exponent), _dist(X, X, exponent), _dist(Y, Y, exponent)
    biased = 2 * Dxy.mean() - Dxx.mean() - Dyy.mean()
    m, n = Dxx.shape[0], Dyy.shape[0]
    unbiased = 2 * Dxy.mean() - Dxx.sum() / (m * (m - 1)) - Dyy.sum() / (n * (n - 1))
    assert np.allclose(kl.energy_distance(X, Y, exponent=exponent), biased)
    assert np.allclose(
        kl.energy_distance(X, Y, exponent=exponent, estimator="unbiased"), unbiased
    )


def test_energy_distance_linear_estimator():
    X = jax.random.normal(jax.random.key(0), (20, 1))
    Y = jax.random.normal(jax.random.key(1), (20, 1))
    x1, x2, y1, y2 = (np.asarray(A)[:, 0] for A in (X[0::2], X[1::2], Y[0::2], Y[1::2]))
    h = np.abs(x1 - y2) + np.abs(x2 - y1) - np.abs(x1 - x2) - np.abs(y1 - y2)
    got = kl.energy_distance(X, Y, estimator="linear")
    assert np.allclose(got, h.mean())


@pytest.mark.slow
def test_distance_correlation_detects_nonlinear_dependence():
    X = jax.random.normal(jax.random.key(0), (300, 1))
    Z = jax.random.normal(jax.random.key(1), (300, 1))
    Y = X**2
    # Pearson misses it: X and X^2 are uncorrelated for symmetric X.
    assert abs(float(jnp.corrcoef(X[:, 0], Y[:, 0])[0, 1])) < 0.2
    assert kl.distance_correlation_squared(X, Y) > 5 * kl.distance_correlation_squared(
        X, Z
    )


def test_distance_correlation_invariances():
    X, Y = _pair()
    base = kl.distance_correlation_squared(X, Y)
    assert jnp.isclose(kl.distance_correlation_squared(X, X), 1.0)
    # Translation, scaling and rotation of either argument leave dCor unchanged.
    R = jnp.array([[0.6, -0.8], [0.8, 0.6]])
    moved = kl.distance_correlation_squared(3.0 * X @ R + 1.0, Y - 2.0)
    assert jnp.isclose(moved, base, rtol=1e-9)


@pytest.mark.slow
def test_full_nystrom_recovers_the_dense_value():
    X, Y = _pair(n=25)
    approx = kl.NystromFeatures(25, jax.random.key(0), jitter=1e-12)
    for fn in (kl.distance_covariance_squared, kl.distance_correlation_squared):
        assert jnp.allclose(fn(X, Y, approx=approx), fn(X, Y), rtol=1e-5)


@pytest.mark.slow
def test_gradients_are_finite_at_coincident_points():
    X = jnp.array([[0.0], [0.0], [1.0], [2.0], [2.0]])
    Y = jnp.array([[1.0], [1.0], [0.0], [3.0], [2.0]])
    for fn in (kl.distance_covariance_squared, kl.distance_correlation_squared):
        g = jax.grad(fn)(X, Y)
        assert bool(jnp.all(jnp.isfinite(g)))
    g = jax.grad(lambda x: kl.Distance().pairwise(x, x))(jnp.zeros(2))
    assert bool(jnp.all(jnp.isfinite(g)))


@pytest.mark.slow
def test_exponent_two_is_the_linear_kernel():
    X = jax.random.normal(jax.random.key(0), (6, 3))
    K = kl.Distance(variance=0.7, exponent=2.0)(X, X)
    assert jnp.allclose(K, kl.Linear(variance=0.7)(X, X), atol=1e-12)


@pytest.mark.parametrize("exponent", [0.3, 1.0, 1.9])
def test_psd(exponent):
    X = jax.random.normal(jax.random.key(0), (30, 3))
    eigs = jnp.linalg.eigvalsh(kl.Distance(exponent=exponent).gram(X))
    assert float(eigs.min()) > -1e-9


@pytest.mark.parametrize("exponent", [0.0, -1.0, 2.5])
def test_rejects_exponent_outside_zero_two(exponent):
    with pytest.raises(ValueError, match="exponent"):
        kl.Distance(exponent=exponent)
    with pytest.raises(ValueError, match="exponent"):
        kl.functional.distance_kernel(jnp.ones((2, 1)), jnp.ones((2, 1)), 1.0, exponent)


def test_float32_separations_survive_a_large_offset():
    # Adjacent float32 values at 2**24: the norm expansion rounds their
    # squared distance to zero, and the |x|^a anchor terms would bury it.
    x = jnp.array([[2.0**24]], dtype=jnp.float32)
    y = x + 2.0
    assert float(kl.energy_distance(x, y)) == 4.0
    kx, ky = jax.random.split(jax.random.key(0))
    X = jax.random.normal(kx, (60, 1), dtype=jnp.float32)
    Y = X**2 + 0.3 * jax.random.normal(ky, (60, 1), dtype=jnp.float32)
    base = kl.distance_correlation_squared(X, Y)
    shifted = kl.distance_correlation_squared(X + 1e3, Y - 1e3)
    assert jnp.allclose(shifted, base, rtol=2e-3)


@pytest.mark.parametrize("estimator", ["biased", "unbiased"])
def test_constant_input_has_zero_distance_correlation(estimator):
    X, _ = _pair()
    const = jnp.full((X.shape[0], 1), 3.0)
    for A, B in [(X, const), (const, X)]:
        got = kl.distance_correlation_squared(A, B, estimator=estimator)
        assert float(got) == 0.0


def test_nan_inputs_propagate():
    k = kl.Distance()
    assert jnp.isnan(k.pairwise(jnp.array([jnp.nan]), jnp.array([1.0])))
    X = jnp.array([[0.0], [jnp.nan], [2.0]])
    assert jnp.isnan(kl.energy_distance(X, X + 1.0))
    assert jnp.isnan(kl.distance_covariance_squared(X, X))


def test_nystrom_is_finite_with_a_zero_norm_landmark():
    # Every Distance landmark at the origin makes K_zz exactly zero; the
    # jitter floor keeps the Cholesky finite.
    X = jnp.zeros((5, 2))
    Phi = kl.NystromFeatures(1, jax.random.key(0)).fit(kl.Distance(), X)(X)
    assert bool(jnp.all(jnp.isfinite(Phi)))


def test_chunked_distances_match_a_direct_computation(monkeypatch):
    from kernellib.functional import _nonstationary

    monkeypatch.setattr(_nonstationary, "_CHUNK_ELEMENTS", 6)  # 1-row chunks
    X1 = jax.random.normal(jax.random.key(0), (7, 3))
    X2 = jax.random.normal(jax.random.key(1), (5, 3))
    got = kl.functional.distance_kernel(X1, X2, jnp.array(1.0), 0.8)
    D = _dist(X1, X2, 0.8)
    n1 = np.linalg.norm(np.asarray(X1), axis=1) ** 0.8
    n2 = np.linalg.norm(np.asarray(X2), axis=1) ** 0.8
    assert np.allclose(got, 0.5 * (n1[:, None] + n2[None, :] - D), rtol=1e-12)
