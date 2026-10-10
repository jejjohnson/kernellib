"""Exact Bessel-series features of the Periodic kernel (GEO22)."""

from __future__ import annotations

import itertools
import math

import einx
import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import ive

import kernellib as kl
from kernellib._einx import einsum
from kernellib._spectral._periodic import _scaled_bessel_i


def _X(n: int = 50) -> jnp.ndarray:
    return einx.id("n -> n 1", jnp.linspace(-2.0, 5.0, n))


def _gram(Phi: jnp.ndarray) -> jnp.ndarray:
    return einsum(Phi, Phi, "n f, m f -> n m")


def _smallest_k(kernel: kl.Periodic, target: float) -> int:
    """Smallest K whose truncation tail is below ``target``.

    The tail is ``2 sum_{k > K} I~_k(z)``, so it is read off one long
    recurrence instead of refitting for every K.
    """
    z = 1.0 / float(kernel.lengthscale) ** 2
    i_k = np.asarray(_scaled_bessel_i(jnp.asarray(z), 400, 400 + 32))
    tails = 2.0 * (np.sum(i_k) - np.cumsum(i_k))
    return int(np.argmax(tails < target))


@pytest.mark.parametrize("z", [0.01, 1.0, 100.0, 1e4])
def test_scaled_bessel_matches_scipy_ive(z):
    # Start the recurrence 8 sqrt(z) + 32 above the largest order: its
    # truncation error falls like exp(-(N^2 - k^2) / z) (see the helper).
    n_start = 60 + 32 + math.ceil(8.0 * math.sqrt(z))
    got = np.asarray(_scaled_bessel_i(jnp.asarray(z), 60, n_start))[:61]
    np.testing.assert_allclose(got, ive(np.arange(61), z), rtol=1e-12, atol=0.0)


@pytest.mark.parametrize("lengthscale", [2.0, 0.7, 0.2])
def test_gram_matches_the_periodic_kernel(lengthscale):
    k = kl.Periodic(lengthscale=lengthscale, variance=1.7, period=1.3)
    K = _smallest_k(k, 1e-12)
    X = _X()
    pf = kl.PeriodicFeatures(K).fit(k, X)
    assert float(pf.truncation_tail()) < 1e-12
    Phi = pf(X)
    assert Phi.shape == (50, 2 * K + 1)
    # Entrywise error <= variance * tail (< 1.7e-12), plus rounding.
    np.testing.assert_allclose(_gram(Phi), k(X, X), rtol=0.0, atol=1e-10)


def test_coefficients_are_non_negative_and_sum_to_the_variance():
    k = kl.Periodic(lengthscale=0.4, variance=2.5, period=3.0)
    pf = kl.PeriodicFeatures(40).fit(k, _X())
    q = pf.coefficients
    assert q.shape == (41,)
    assert bool(jnp.all(q >= 0.0))
    np.testing.assert_allclose(float(jnp.sum(q)), 2.5, rtol=0.0, atol=1e-12)
    # sum q / variance + tail = 1, with the tail computed independently.
    np.testing.assert_allclose(
        float(jnp.sum(q)) / 2.5 + float(pf.truncation_tail()), 1.0, atol=1e-14
    )


def test_truncation_tail_decreases_in_k():
    k = kl.Periodic(lengthscale=0.3)
    X = _X(4)
    tails = [
        float(kl.PeriodicFeatures(K, n_recurrence=120).fit(k, X).truncation_tail())
        for K in range(1, 40, 3)
    ]
    assert all(t >= 0.0 for t in tails)
    assert all(a > b for a, b in itertools.pairwise(tails))


def test_columns_are_constant_then_cosines_then_sines():
    k = kl.Periodic(lengthscale=1.0, variance=1.0, period=2.0)
    X = _X(7)
    pf = kl.PeriodicFeatures(3).fit(k, X)
    Phi, sq = pf(X), jnp.sqrt(pf.coefficients)
    x = X[:, 0]
    np.testing.assert_allclose(Phi[:, 0], sq[0] * jnp.ones_like(x), atol=1e-14)
    for j in range(1, 4):
        np.testing.assert_allclose(
            Phi[:, j], sq[j] * jnp.cos(j * jnp.pi * x), atol=1e-14
        )
        np.testing.assert_allclose(
            Phi[:, 3 + j], sq[j] * jnp.sin(j * jnp.pi * x), atol=1e-14
        )


@pytest.mark.slow
@pytest.mark.parametrize("lengthscale", [0.8, 0.05])
def test_gradients_match_the_exact_kernel(lengthscale):
    # z = 400 at lengthscale 0.05: the large-z regime of the recurrence.
    k = kl.Periodic(lengthscale=lengthscale, variance=1.3, period=1.7)
    X = _X(20)
    pf = kl.PeriodicFeatures(_smallest_k(k, 1e-14)).fit(k, X)

    # A non-uniform weighting, so the loss sees every Gram entry differently.
    x = X[:, 0]
    W = jnp.cos(einx.subtract("n, m -> n m", x, 0.3 * x))

    def approx(kernel):
        Phi = eqx.tree_at(lambda m: m.kernel, pf, kernel)(X)
        return jnp.sum(_gram(Phi) * W)

    def exact(kernel):
        return jnp.sum(kernel(X, X) * W)

    g, g_ref = eqx.filter_grad(approx)(k), eqx.filter_grad(exact)(k)
    for name in ("lengthscale", "variance", "period"):
        got, ref = getattr(g, name), getattr(g_ref, name)
        assert bool(jnp.isfinite(got)), name
        np.testing.assert_allclose(got, ref, rtol=1e-8, atol=1e-8, err_msg=name)


@pytest.mark.slow
def test_gradient_is_finite_at_very_large_z():
    # z = 1e4: the tail is not small at K = 20, but the derivative of the
    # recurrence must still be finite.
    k = kl.Periodic(lengthscale=0.01)
    X = _X(5)
    pf = kl.PeriodicFeatures(20).fit(k, X)

    def loss(kernel):
        m = eqx.tree_at(lambda f: f.kernel, pf, kernel)
        return jnp.sum(m(X)) + m.truncation_tail()

    g = eqx.filter_grad(loss)(k)
    assert all(bool(jnp.isfinite(getattr(g, n))) for n in ("lengthscale", "period"))


def test_jit_and_traced_fit():
    k = kl.Periodic(lengthscale=0.5)
    X = _X(10)
    eager = kl.PeriodicFeatures(16).fit(k, X)(X)

    @eqx.filter_jit
    def fit_and_map(kernel):
        return kl.PeriodicFeatures(16).fit(kernel, X)(X)

    # A traced lengthscale falls back to the 2K + 32 recurrence start.
    np.testing.assert_allclose(fit_and_map(k), eager, rtol=0.0, atol=1e-13)


def test_operator_is_the_low_rank_gram():
    k = kl.Periodic(lengthscale=0.6, variance=0.9)
    X = _X(12)
    pf = kl.PeriodicFeatures(20).fit(k, X)
    np.testing.assert_allclose(pf.operator(X).as_matrix(), k(X, X), atol=1e-10)


def test_rejects_multi_dimensional_inputs():
    X2 = jnp.zeros((4, 2))
    with pytest.raises(ValueError, match="Periodised"):
        kl.PeriodicFeatures(5).fit(kl.Periodic(), X2)
    pf = kl.PeriodicFeatures(5).fit(kl.Periodic(), _X(4))
    with pytest.raises(ValueError, match="GEO24"):
        pf(X2)


def test_rejects_other_kernels_and_bad_config():
    with pytest.raises(TypeError, match="Periodic kernel"):
        kl.PeriodicFeatures(5).fit(kl.RBF(), _X(4))
    with pytest.raises(ValueError, match="n_harmonics"):
        kl.PeriodicFeatures(0)
    with pytest.raises(ValueError, match="n_recurrence"):
        kl.PeriodicFeatures(10, n_recurrence=5)
    with pytest.raises(RuntimeError, match="not fitted"):
        kl.PeriodicFeatures(5)(_X(4))


def test_gradient_is_finite_when_high_harmonics_underflow():
    # At lengthscale 2 (z = 0.25), q_k underflows to 0 well before k = 200.
    k = kl.Periodic(lengthscale=2.0)
    X = _X(5)
    pf = kl.PeriodicFeatures(200).fit(k, X)
    assert float(pf.coefficients[-1]) == 0.0

    def loss(kernel):
        return jnp.sum(eqx.tree_at(lambda f: f.kernel, pf, kernel)(X))

    g = eqx.filter_grad(loss)(k)
    for name in ("lengthscale", "variance", "period"):
        assert bool(jnp.isfinite(getattr(g, name))), name
