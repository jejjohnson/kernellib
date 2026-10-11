"""`Advected` and `RandomAdvectionRBF`: the frozen-field property, the limits
v = 0 and Σ_V → 0, the closed form against Monte Carlo over the velocity,
PSD, gradients and the random Fourier feature path."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

import kernellib as kl
from kernellib._einx import einsum, rearrange, reduce


V = jnp.array([1.5, -0.7])
L_V = jnp.array([[0.6, 0.0], [0.3, 0.4]])
RBF = kl.RBF(lengthscale=0.8, variance=1.3)
MATERN = kl.Matern(nu=2.5, lengthscale=0.9, variance=0.7)


def _points(n, d=3, key=0, scale=2.0):
    return jax.random.uniform(jax.random.key(key), (n, d), minval=-scale, maxval=scale)


def _random_advection(cov_tril=L_V, mean=V):
    return kl.RandomAdvectionRBF(mean, cov_tril, variance=1.3, lengthscale=0.8)


# -- Advected -----------------------------------------------------------------


@pytest.mark.parametrize("base", [RBF, MATERN])
def test_zero_velocity_is_the_spatial_kernel(base):
    Z1, Z2 = _points(6), _points(6, key=1)
    k = kl.Advected(base, jnp.zeros(2))
    assert jnp.allclose(k(Z1, Z2), base(Z1[:, :2], Z2[:, :2]))
    # Time is ignored altogether.
    shifted = Z2.at[:, 2].add(7.0)
    assert jnp.allclose(k(Z1, shifted), k(Z1, Z2))


@pytest.mark.parametrize("delta", [-3.0, 0.5, 10.0])
def test_frozen_field_property(delta):
    k = kl.Advected(MATERN, V)
    z = jnp.array([0.2, -0.4, 1.1])
    moved = z + jnp.concatenate([delta * V, jnp.array([delta])])
    assert jnp.allclose(k.pairwise(z, moved), MATERN.variance)


def test_matches_warped_and_its_formula():
    Z1, Z2 = _points(6), _points(6, key=1)
    k = kl.Advected(MATERN, V)
    warped = kl.Warped(MATERN, lambda z: z[:2] - V * z[2])
    assert jnp.allclose(k(Z1, Z2), warped(Z1, Z2))
    assert jnp.allclose(k.diag(Z1), MATERN.variance)
    assert k.is_stationary and k.is_pointwise and k.space_dims == 2
    pointwise = jax.vmap(lambda x: jax.vmap(lambda y: k.pairwise(x, y))(Z2))(Z1)
    assert jnp.allclose(pointwise, k(Z1, Z2))


def test_scalar_velocity_is_one_dimensional():
    k = kl.Advected(kl.RBF(), 2.0)
    assert k.space_dims == 1
    z = jnp.array([0.0, 0.0])
    assert jnp.allclose(k.pairwise(z, jnp.array([2.0, 1.0])), 1.0)


def test_advected_validation():
    with pytest.raises(TypeError, match="stationary"):
        kl.Advected(kl.Linear(), V)
    with pytest.raises(ValueError, match="vector"):
        kl.Advected(RBF, jnp.ones((2, 2)))
    with pytest.raises(ValueError, match="2x3 transform"):
        kl.Advected(RBF, V)(_points(3, d=2), _points(3, d=2))
    with pytest.raises(NotImplementedError, match="singular"):
        kl.Advected(RBF, V).spectral_density(jnp.zeros((1, 3)))


def test_advected_frequencies():
    k = kl.Advected(kl.RBF(lengthscale=0.5), V)
    omega = k.sample_frequencies(jax.random.key(0), 64, 3)
    assert omega.shape == (64, 3)
    # Full frequencies (ω₀, -v·ω₀): the time frequency is fixed by the space one.
    assert jnp.allclose(omega[:, 2], -einsum(omega[:, :2], V, "n d, d -> n"))
    with pytest.raises(ValueError, match="2x3 transform"):
        k.sample_frequencies(jax.random.key(0), 4, 2)
    with pytest.raises(NotImplementedError, match="anisotropic"):
        kl.OrthogonalRandomFeatures(8, jax.random.key(0)).fit(k, _points(4))


@pytest.mark.slow
def test_advected_random_fourier_features():
    k = kl.Advected(kl.RBF(lengthscale=0.7, variance=1.3), V)
    Z = _points(20)
    rff = kl.RandomFourierFeatures(20_000, jax.random.key(1)).fit(k, Z)
    Phi = rff(Z)
    approx = einsum(Phi, Phi, "n r, m r -> n m")
    # Each entry averages 20 000 bounded cosine pairs; the per-pair variance
    # is at most σ⁴ / 2, so the Monte Carlo error is about
    # 1.3 / sqrt(2 * 20 000) ≈ 0.0065. A bound of 0.05 is ~8 of those.
    assert jnp.max(jnp.abs(approx - k(Z, Z))) < 0.05


# -- RandomAdvectionRBF ---------------------------------------------------------


def test_vanishing_velocity_covariance_is_the_frozen_rbf():
    Z1, Z2 = _points(7), _points(6, key=1)
    k = _random_advection(cov_tril=1e-6 * jnp.eye(2))  # Σ_V = 1e-12 I
    frozen = kl.Advected(RBF, V)
    assert jnp.allclose(k(Z1, Z2), frozen(Z1, Z2), rtol=0.0, atol=1e-8)


def test_zero_time_lag_is_the_spatial_rbf():
    Z1, Z2 = _points(7), _points(6, key=1)
    Z2 = Z2.at[:, 2].set(0.0)
    Z1 = Z1.at[:, 2].set(0.0)
    assert jnp.allclose(_random_advection()(Z1, Z2), RBF(Z1[:, :2], Z2[:, :2]))


def test_closed_form_at_the_mean_displacement():
    # At h = μ u the exponent vanishes: C = σ² |I + u² Σ_V / ℓ²|^{-1/2}.
    k = _random_advection()
    u = 2.0
    z = jnp.zeros(3)
    zp = jnp.concatenate([u * V, jnp.array([u])])
    sigma = einsum(L_V, L_V, "i k, j k -> i j")
    det = jnp.linalg.det(jnp.eye(2) + u**2 * sigma / 0.8**2)
    assert jnp.allclose(k.pairwise(z, zp), 1.3 / jnp.sqrt(det))
    assert jnp.allclose(k.velocity_cov, sigma)


def test_random_advection_structure():
    k = _random_advection()
    Z1, Z2 = _points(6), _points(6, key=1)
    K = k(Z1, Z1)
    assert jnp.allclose(K, rearrange(K, "i j -> j i"))
    assert jnp.allclose(k.diag(Z1), 1.3)
    assert jnp.allclose(jnp.diag(K), k.diag(Z1))
    assert jnp.allclose(k(Z2, Z1), rearrange(k(Z1, Z2), "i j -> j i"))
    assert k.is_stationary and k.is_pointwise and k.space_dims == 2


def test_random_advection_validation():
    with pytest.raises(ValueError, match="vector"):
        kl.RandomAdvectionRBF(jnp.ones((2, 2)), jnp.eye(2))
    with pytest.raises(ValueError, match=r"shape \(2, 2\)"):
        kl.RandomAdvectionRBF(V, jnp.eye(3))
    with pytest.raises(ValueError, match="scalar"):
        kl.RandomAdvectionRBF(V, jnp.eye(2), lengthscale=jnp.ones(2))
    with pytest.raises(ValueError, match="3 columns"):
        _random_advection()(_points(3, d=2), _points(3, d=2))


@pytest.mark.slow
def test_closed_form_matches_monte_carlo():
    k = _random_advection()
    n = 100_000
    eps = jax.random.normal(jax.random.key(3), (n, 2))
    draws = V + einsum(L_V, eps, "i j, n j -> n i")  # V_i ~ N(μ, L Lᵀ)
    lags = [
        (jnp.array([0.4, -0.2]), 0.0),
        (jnp.array([1.0, 0.5]), 0.7),
        (jnp.array([3.0, -1.4]), 2.0),
        (jnp.array([-0.5, 1.0]), -1.5),
    ]
    for h, u in lags:
        samples = RBF.elwise(h - u * draws, jnp.zeros_like(draws))
        mean = reduce(samples, "n ->", "mean")
        se = jnp.std(samples) / jnp.sqrt(n)
        exact = k.pairwise(jnp.concatenate([h, jnp.array([u])]), jnp.zeros(3))
        # The Monte Carlo mean is within 5 of its own standard errors of the
        # expectation (a 5-sigma bound from the CLT); at u = 0 every draw is
        # the same, so allow float round-off there.
        assert jnp.abs(mean - exact) <= 5.0 * se + 1e-12


@pytest.mark.slow
@pytest.mark.parametrize(
    "kernel",
    [kl.Advected(MATERN, V), _random_advection()],
    ids=["Advected", "RandomAdvectionRBF"],
)
def test_psd_on_300_points(kernel):
    Z = _points(300, key=4)
    K = kernel(Z, Z)
    eigs = jnp.linalg.eigvalsh(K)
    assert jnp.min(eigs) > -1e-9 * jnp.max(eigs)


@pytest.mark.slow
@pytest.mark.parametrize(
    "kernel",
    [kl.Advected(MATERN, V), _random_advection()],
    ids=["Advected", "RandomAdvectionRBF"],
)
def test_velocity_gradients_are_finite(kernel):
    Z = _points(12, key=5)

    def loss(k):
        return jnp.sum(k(Z, Z))

    grads = eqx.filter_jit(eqx.filter_grad(loss))(kernel)
    if isinstance(kernel, kl.Advected):
        leaves = [grads.velocity]
    else:
        leaves = [grads.velocity_mean, grads.velocity_cov_tril]
    for g in leaves:
        assert jnp.all(jnp.isfinite(g))
        assert jnp.any(g != 0.0)
    # Finite on the diagonal too, where h = 0 and u = 0.
    z = Z[0]
    g = jax.grad(lambda zz: kernel.pairwise(zz, z))(z)
    assert jnp.all(jnp.isfinite(g))
