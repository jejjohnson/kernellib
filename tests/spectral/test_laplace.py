"""Laplace-eigenfunction (HSGP) features."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from geonnax.basis import fourier_basis

import kernellib as kl


def _X(n=40, d=2, key=0):
    return jax.random.uniform(jax.random.key(key), (n, d), minval=-1.0, maxval=1.0)


def test_rbf_1d_converges_inside_a_wide_box():
    X = jnp.linspace(-1.0, 1.0, 50)[:, None]
    k = kl.RBF(lengthscale=0.3, variance=1.7)
    Phi = kl.LaplaceEigenfunctionFeatures(64, boundary_factor=2.5).fit(k, X)(X)
    assert Phi.shape == (50, 64)
    assert jnp.allclose(Phi @ Phi.T, k(X, X), atol=1e-6)


@pytest.mark.slow
@pytest.mark.parametrize(
    ("kernel", "n_per_dim", "atol"),
    [
        (kl.RBF(lengthscale=jnp.array([0.3, 0.8]), variance=1.7), (60, 20), 1e-6),
        (kl.Matern(lengthscale=jnp.array([0.3, 0.8]), nu=2.5), (80, 30), 1e-3),
    ],
    ids=["rbf-ard", "matern25-ard"],
)
def test_ard_2d(kernel, n_per_dim, atol):
    # ARD needs the per-axis frequency vectors, not just the summed
    # eigenvalue; an isotropic density at sqrt(lambda) would be wrong here.
    X = _X()
    lap = kl.LaplaceEigenfunctionFeatures(n_per_dim, boundary_factor=4.0)
    Phi = lap.fit(kernel, X)(X)
    assert Phi.shape == (40, n_per_dim[0] * n_per_dim[1])
    assert jnp.allclose(Phi @ Phi.T, kernel(X, X), atol=atol)


@pytest.mark.slow
def test_matern_error_falls_with_basis_size():
    X = jnp.linspace(-1.0, 1.0, 30)[:, None]
    k = kl.Matern(lengthscale=0.3, nu=0.5)

    def err(m):
        Phi = kl.LaplaceEigenfunctionFeatures(m, boundary_factor=2.5).fit(k, X)(X)
        return jnp.max(jnp.abs(Phi @ Phi.T - k(X, X)))

    assert err(128) < err(32) < err(8)


def test_frequencies_follow_the_basis_order():
    # |omega_j|^2 is geonnax's summed eigenvalue, in the same row-major order.
    X = _X(n=3)
    lap = kl.LaplaceEigenfunctionFeatures((3, 4), L=(1.5, 2.0)).fit(kl.RBF(), X)
    _, lam = fourier_basis(X, (3, 4), (1.5, 2.0))
    assert lap.frequencies.shape == (12, 2)
    assert jnp.allclose(jnp.sum(lap.frequencies**2, axis=-1), lam)


def test_isotropic_matches_the_summed_eigenvalue_form():
    # pyrox-gp's form: basis * sqrt(S_radial(sqrt(lambda))).
    X = _X(n=5, d=3)
    k = kl.Matern(lengthscale=0.6, variance=0.9, nu=1.5)
    lap = kl.LaplaceEigenfunctionFeatures(4, L=2.0).fit(k, X)
    basis, lam = fourier_basis(X, 4, 2.0)
    omega = jnp.zeros((lam.size, 3)).at[:, 0].set(jnp.sqrt(lam))
    assert jnp.allclose(lap(X), basis * jnp.sqrt(k.spectral_density(omega)))


def test_half_widths_from_data_or_given():
    X = _X()
    auto = kl.LaplaceEigenfunctionFeatures(4).fit(kl.RBF(), X)
    assert jnp.allclose(jnp.array(auto.half_widths), 1.5 * jnp.max(jnp.abs(X), 0))
    given = kl.LaplaceEigenfunctionFeatures(4, L=3.0).fit(kl.RBF(), X)
    assert given.half_widths == (3.0, 3.0)


def test_operator_and_grad():
    X = _X()
    lap = kl.LaplaceEigenfunctionFeatures(6).fit(kl.RBF(lengthscale=0.5), X)
    Phi = lap(X)
    assert jnp.allclose(lap.operator(X).as_matrix(), Phi @ Phi.T)

    @eqx.filter_jit
    def loss(m):
        return jnp.sum(m(X))

    g = eqx.filter_grad(loss)(lap)
    assert jnp.isfinite(g.kernel.lengthscale)


@pytest.mark.parametrize(
    ("kernel", "exc", "match"),
    [
        (kl.RationalQuadratic(), NotImplementedError, "RationalQuadratic"),
        (kl.Periodic(), NotImplementedError, "spectral density"),
        (kl.RBF(lengthscale=jnp.ones(3)), ValueError, "ARD lengthscale"),
    ],
)
def test_fit_rejects_unsupported_kernels(kernel, exc, match):
    with pytest.raises(exc, match=match):
        kl.LaplaceEigenfunctionFeatures(4).fit(kernel, _X())


def test_config_errors():
    with pytest.raises(ValueError, match="n_per_dim"):
        kl.LaplaceEigenfunctionFeatures(0)
    with pytest.raises(ValueError, match="boundary_factor"):
        kl.LaplaceEigenfunctionFeatures(4, boundary_factor=1.0)
    with pytest.raises(ValueError, match="n_per_dim has 3 entries"):
        kl.LaplaceEigenfunctionFeatures((4, 4, 4)).fit(kl.RBF(), _X())
    with pytest.raises(ValueError, match="L has 1 entries"):
        kl.LaplaceEigenfunctionFeatures(4, L=(1.0,)).fit(kl.RBF(), _X())
    with pytest.raises(ValueError, match="positive"):
        kl.LaplaceEigenfunctionFeatures(4).fit(kl.RBF(), jnp.zeros((3, 2)))
    fitted = kl.LaplaceEigenfunctionFeatures(4).fit(kl.RBF(), _X())
    with pytest.raises(ValueError, match="2-dimensional"):
        fitted(_X(d=3))
