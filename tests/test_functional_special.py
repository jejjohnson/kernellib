"""Tests for the private special functions (`log_bessel_kv`)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.special as sp

from kernellib.functional._special import log_bessel_kv


NUS = np.array([0.0, 0.1, 0.3, 0.5, 0.75, 1.0, 1.5, 2.3, 5.0, 10.0, 25.0])
XS = np.logspace(-4, 2.5, 60)


@pytest.mark.slow
def test_matches_scipy_over_a_grid():
    got = log_bessel_kv(NUS[:, None], XS[None, :])
    # Compare the exponentially scaled kve: kv itself underflows at large x.
    ref = np.log(sp.kve(NUS[:, None], XS[None, :])) - XS[None, :]
    rel = np.abs(np.expm1(np.asarray(got) - ref))
    assert rel.max() < 1e-10


@pytest.mark.slow
def test_negative_order_is_symmetric():
    x = jnp.asarray(XS)
    assert jnp.allclose(log_bessel_kv(-1.7, x), log_bessel_kv(1.7, x), rtol=1e-14)


def test_half_integer_closed_form():
    x = jnp.asarray(XS)
    exact = 0.5 * jnp.log(jnp.pi / (2 * x)) - x
    assert jnp.allclose(log_bessel_kv(0.5, x), exact, rtol=1e-12)


@pytest.mark.slow
def test_gradient_in_x_matches_the_recurrence():
    # dK_nu/dx = -(K_{nu-1} + K_{nu+1}) / 2.
    grad = jax.vmap(jax.vmap(jax.grad(log_bessel_kv, 1), (None, 0)), (0, None))
    got = np.asarray(grad(jnp.asarray(NUS), jnp.asarray(XS)))
    nu, x = NUS[:, None], XS[None, :]
    ref = -(sp.kve(nu - 1, x) + sp.kve(nu + 1, x)) / (2 * sp.kve(nu, x))
    assert np.allclose(got, ref, rtol=1e-10)


@pytest.mark.slow
def test_gradient_in_order_matches_finite_differences():
    # No closed form in nu; central differences of scipy's kve, O(eps^2).
    nus, eps = NUS[1:], 1e-5
    grad = jax.vmap(jax.vmap(jax.grad(log_bessel_kv, 0), (None, 0)), (0, None))
    got = np.asarray(grad(jnp.asarray(nus), jnp.asarray(XS)))
    nu, x = nus[:, None], XS[None, :]
    ref = (np.log(sp.kve(nu + eps, x)) - np.log(sp.kve(nu - eps, x))) / (2 * eps)
    assert np.allclose(got, ref, rtol=1e-6, atol=1e-6)


def test_broadcasts_and_jits():
    f = jax.jit(log_bessel_kv)
    assert f(jnp.ones((3, 1)), jnp.ones((1, 4))).shape == (3, 4)


@pytest.mark.parametrize("x", [1e-8, 1e3])
def test_finite_in_float32_where_kv_itself_is_not(x):
    # K_10(1e-8) overflows float32 and K_10(1e3) underflows; the log does not.
    value = log_bessel_kv(jnp.float32(10.0), jnp.float32(x))
    assert value.dtype == jnp.float32
    assert jnp.isfinite(value)
    ref = np.log(sp.kve(10.0, x)) - x
    assert np.isclose(float(value), ref, rtol=1e-5)
