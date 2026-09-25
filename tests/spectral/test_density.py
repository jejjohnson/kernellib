"""Spectral densities and frequency samplers on the stationary kernels."""

from __future__ import annotations

import math

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

import kernellib as kl


def _matern_1d_closed_form(omega, variance, lengthscale, nu):
    # The 1-D Matern density from pyrox_gp._basis._spectral_density, written
    # in terms of alpha = 2 nu / l^2, as an independent reference.
    alpha = 2.0 * nu / lengthscale**2
    c = 2.0 * math.sqrt(math.pi) * math.gamma(nu + 0.5) / math.gamma(nu)
    return variance * c * alpha**nu * (alpha + omega**2) ** (-(nu + 0.5))


@pytest.mark.parametrize("nu", [0.5, 1.5, 2.5])
def test_matern_density_matches_1d_closed_form(nu):
    k = kl.Matern(lengthscale=0.7, variance=1.3, nu=nu)
    omega = jnp.linspace(-5.0, 5.0, 11)
    got = k.spectral_density(omega[:, None])
    assert jnp.allclose(got, _matern_1d_closed_form(omega, 1.3, 0.7, nu))


def test_rbf_density_matches_1d_closed_form():
    k = kl.RBF(lengthscale=0.7, variance=1.3)
    omega = jnp.linspace(-5.0, 5.0, 11)
    expected = 1.3 * 0.7 * jnp.sqrt(2 * jnp.pi) * jnp.exp(-0.5 * (0.7 * omega) ** 2)
    assert jnp.allclose(k.spectral_density(omega[:, None]), expected)


@pytest.mark.parametrize(
    "kernel",
    [
        kl.RBF(lengthscale=0.7, variance=1.3),
        kl.Matern(lengthscale=0.7, variance=1.3, nu=1.5),
        kl.Matern(lengthscale=0.7, variance=1.3, nu=2.5),
    ],
    ids=["rbf", "matern15", "matern25"],
)
def test_density_inverts_to_kernel_1d(kernel):
    # Bochner: k(tau) = (2 pi)^-1 int S(w) cos(w tau) dw. Riemann sum on a
    # grid wide enough that the truncated tail is below the tolerance.
    omega = jnp.linspace(-80.0, 80.0, 160_001)
    dw = omega[1] - omega[0]
    S = kernel.spectral_density(omega[:, None])
    for tau in (0.0, 0.3, 1.0):
        k_tau = jnp.sum(S * jnp.cos(omega * tau)) * dw / (2 * jnp.pi)
        expected = kernel.pairwise(jnp.zeros(1), jnp.array([tau]))
        assert jnp.allclose(k_tau, expected, atol=1e-4)


@pytest.mark.parametrize("d", [2, 3])
@pytest.mark.parametrize(
    "kernel",
    [kl.RBF(lengthscale=0.7, variance=1.3), kl.Matern(0.7, 1.3, nu=2.5)],
    ids=["rbf", "matern25"],
)
def test_density_integrates_to_variance(kernel, d):
    # (2 pi)^-d int S = k(0) = variance. The density is radial for an
    # isotropic kernel, so integrate over shells: area of S^{d-1} times r^{d-1}.
    r = jnp.linspace(0.0, 200.0, 400_001)
    dr = r[1] - r[0]
    omega = jnp.zeros((r.size, d)).at[:, 0].set(r)
    area = 2 * math.pi ** (d / 2) / math.gamma(d / 2)
    total = jnp.sum(area * r ** (d - 1) * kernel.spectral_density(omega)) * dr
    assert jnp.allclose(total / (2 * jnp.pi) ** d, 1.3, rtol=1e-4)


def test_rbf_ard_density_factorises():
    ell = jnp.array([0.5, 1.2, 2.0])
    k = kl.RBF(lengthscale=ell, variance=0.8)
    omega = jax.random.normal(jax.random.key(0), (7, 3))
    per_dim = ell * jnp.sqrt(2 * jnp.pi) * jnp.exp(-0.5 * (ell * omega) ** 2)
    assert jnp.allclose(k.spectral_density(omega), 0.8 * jnp.prod(per_dim, axis=-1))


def test_matern_ard_density_is_rescaled_isotropic():
    # S_ARD(w) = det(L) S_unit(L w) for x -> x / l applied per dimension.
    ell = jnp.array([0.5, 1.2])
    ard = kl.Matern(lengthscale=ell, variance=0.8, nu=1.5)
    unit = kl.Matern(lengthscale=1.0, variance=0.8, nu=1.5)
    omega = jax.random.normal(jax.random.key(1), (5, 2))
    expected = jnp.prod(ell) * unit.spectral_density(omega * ell)
    assert jnp.allclose(ard.spectral_density(omega), expected)


def test_density_batch_shape():
    k = kl.RBF()
    assert k.spectral_density(jnp.zeros((4, 5, 2))).shape == (4, 5)


# Characteristic-function check of the samplers:
# E[cos(w . tau)] = k(tau) / variance. cos is bounded, so the sample mean has
# a finite standard error; the bound is 7 of those (the gaussx rule for tests
# on sampling behaviour), estimated from the same draws. tau is not along an
# axis, so a coordinate-wise Student-t draw for Matern (which gives the
# product of 1-D densities, exp(-|t1| - |t2| - |t3|) for nu = 1/2) is caught.
_TAU = jnp.array([0.4, -0.3, 0.2])


@pytest.mark.parametrize(
    "kernel",
    [
        kl.RBF(lengthscale=jnp.array([0.7, 1.2, 0.5]), variance=1.3),
        kl.Matern(lengthscale=0.7, variance=1.3, nu=0.5),
        kl.Matern(lengthscale=0.7, variance=1.3, nu=1.5),
        kl.Matern(lengthscale=jnp.array([0.7, 1.2, 0.5]), variance=1.3, nu=2.5),
        kl.RationalQuadratic(lengthscale=0.7, variance=1.3, alpha=0.8),
    ],
    ids=["rbf-ard", "matern05", "matern15", "matern25-ard", "rq"],
)
def test_sampled_frequencies_reproduce_kernel(kernel):
    n = 100_000
    omega = kernel.sample_frequencies(jax.random.key(0), n, 3)
    c = jnp.cos(omega @ _TAU)
    se = jnp.std(c) / jnp.sqrt(n)
    expected = kernel.pairwise(jnp.zeros(3), _TAU) / kernel.variance
    assert jnp.abs(jnp.mean(c) - expected) < 7 * se


@pytest.mark.slow
def test_coordinatewise_matern_draw_would_fail_the_check():
    # Guards the test above: the coordinate-wise t draw pyrox-gp used before
    # is far outside the bound for nu = 1/2 in 3-D.
    n = 100_000
    omega = jax.random.t(jax.random.key(0), 1.0, (n, 3)) / 0.7
    c = jnp.cos(omega @ _TAU)
    se = jnp.std(c) / jnp.sqrt(n)
    k = kl.Matern(lengthscale=0.7, nu=0.5)
    assert jnp.abs(jnp.mean(c) - k.pairwise(jnp.zeros(3), _TAU)) > 7 * se


def test_sample_frequencies_shape_and_dtype():
    k = kl.Matern(nu=1.5)
    omega = k.sample_frequencies(jax.random.key(0), 5, 2, dtype=jnp.float32)
    assert omega.shape == (5, 2)
    assert omega.dtype == jnp.float32


def test_sample_frequencies_is_deterministic_in_key():
    k = kl.RBF()
    a = k.sample_frequencies(jax.random.key(3), 4, 2)
    b = k.sample_frequencies(jax.random.key(3), 4, 2)
    assert jnp.array_equal(a, b)


def test_ard_size_mismatch_raises():
    k = kl.RBF(lengthscale=jnp.ones(3))
    with pytest.raises(ValueError, match="ARD lengthscale of size 3"):
        k.spectral_density(jnp.zeros((2, 2)))
    with pytest.raises(ValueError, match="ARD lengthscale of size 3"):
        k.sample_frequencies(jax.random.key(0), 2, 2)


def test_rational_quadratic_has_no_density():
    with pytest.raises(NotImplementedError, match="RationalQuadratic"):
        kl.RationalQuadratic().spectral_density(jnp.zeros((1, 1)))


class _Bare(kl.AbstractStationaryKernel):
    lengthscale: jax.Array = eqx.field(default=1.0, converter=jnp.asarray)
    variance: jax.Array = eqx.field(default=1.0, converter=jnp.asarray)

    def shape(self, r2):
        return jnp.exp(-r2)


def test_subclass_without_hooks_raises():
    with pytest.raises(NotImplementedError, match="_Bare"):
        _Bare().spectral_density(jnp.zeros((1, 1)))
    with pytest.raises(NotImplementedError, match="_Bare"):
        _Bare().sample_frequencies(jax.random.key(0), 1, 1)


def test_density_gradient_and_jit():
    omega = jnp.linspace(0.0, 3.0, 5)[:, None]

    def total(ell):
        return jnp.sum(kl.Matern(lengthscale=ell, nu=1.5).spectral_density(omega))

    g = jax.grad(total)(0.8)
    assert jnp.isfinite(g)
    assert jnp.allclose(jax.jit(total)(0.8), total(0.8))


def test_frequency_draw_is_differentiable_in_lengthscale():
    # sample_frequencies divides unit draws by the lengthscale, so the draw
    # is reparameterised: d omega / d l = -omega / l.
    def f(ell):
        return kl.RBF(lengthscale=ell).sample_frequencies(jax.random.key(0), 3, 2)

    omega = f(0.5)
    J = jax.jacobian(f)(0.5)
    assert jnp.allclose(J, -omega / 0.5)
