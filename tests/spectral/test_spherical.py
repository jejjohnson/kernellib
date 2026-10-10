"""SphericalHarmonicFeatures: weighted real spherical harmonics (GEO10)."""

from __future__ import annotations

import math

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from geonnax.basis import real_spherical_harmonics

import kernellib as kl
from kernellib._einx import einsum
from kernellib._spectral._spherical import _real_spherical_harmonics
from kernellib._testing import fibonacci_lonlat
from kernellib.functional import legendre_series
from kernellib.functional._geo import lonlat_to_unit


def _gram(Phi: jnp.ndarray) -> jnp.ndarray:
    return einsum(Phi, Phi, "n f, m f -> n m")


@pytest.mark.parametrize(
    "kernel",
    [
        pytest.param(
            kl.SphereMatern(lengthscale=0.4, nu=1.5, variance=1.3),  # L = 64
            marks=pytest.mark.slow,
        ),
        kl.SphereHeat(lengthscale=0.3, max_degree=40),
    ],
    ids=["matern-L64", "heat"],
)
def test_features_reproduce_the_series_gram(kernel):
    # Exact by the addition theorem: Phi Phi^T and the kernel are the same
    # finite Legendre series, so only float64 round-off separates them.
    X = fibonacci_lonlat(200)
    sh = kl.SphericalHarmonicFeatures().fit(kernel, X)
    assert sh.max_degree == kernel.max_degree
    Phi = sh(X)
    assert Phi.shape == (200, sh.n_features)
    assert float(jnp.max(jnp.abs(_gram(Phi) - kernel(X, X)))) < 1e-10


def test_unweighted_harmonics_are_orthonormal():
    # Pins geonnax's normalisation: (4 pi / n) Y^T Y approximates the
    # integral of Y_i Y_j over the sphere, which is the identity for
    # orthonormal harmonics. The only error is the equal-weight Fibonacci
    # quadrature, O(1/n) up to log factors for these degree <= 16
    # polynomials; measured ~1e-4 at n = 20000, inside the 1e-3 bound.
    n, L = 20000, 8
    Y = real_spherical_harmonics(lonlat_to_unit(fibonacci_lonlat(n)), L)
    G = 4.0 * math.pi / n * einsum(Y, Y, "n i, n j -> i j")
    eye = jnp.eye((L + 1) ** 2, dtype=G.dtype)
    assert float(jnp.max(jnp.abs(G - eye))) < 1e-3


def test_harmonics_match_geonnax():
    # Same functions, normalisation and column order as geonnax, including
    # at the poles, where the (x + iy)^m form avoids the 0 / 0.
    poles = jnp.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])
    u = jnp.concatenate([lonlat_to_unit(fibonacci_lonlat(50)), poles])
    ours = _real_spherical_harmonics(u, 8)
    assert float(jnp.max(jnp.abs(ours - real_spherical_harmonics(u, 8)))) < 1e-12


@pytest.mark.slow
def test_harmonics_stay_orthonormal_at_high_degree():
    # geonnax's unnormalised recursion overflows near L = 85; the normalised
    # one does not. Fibonacci quadrature of these degree <= 2L polynomials
    # needs n >> L^2 points; at n = 20000, L = 40 the error measured 6e-4.
    n, L = 20000, 40
    Y = _real_spherical_harmonics(lonlat_to_unit(fibonacci_lonlat(n)), L)
    G = 4.0 * math.pi / n * einsum(Y, Y, "n i, n j -> i j")
    eye = jnp.eye((L + 1) ** 2, dtype=G.dtype)
    assert float(jnp.max(jnp.abs(G - eye))) < 2e-3
    big = _real_spherical_harmonics(lonlat_to_unit(fibonacci_lonlat(10)), 150)
    assert bool(jnp.all(jnp.isfinite(big)))


@pytest.mark.slow
def test_chordal_rbf_error_is_the_funk_hecke_tail():
    # |Phi Phi^T - K| <= sum_{l > L} (2l+1)/(4 pi) a_l since |P_l| <= 1; the
    # tail is computed to degree 60, beyond which the RBF spectrum at
    # R / ell = 2 is below 1e-30. 1e-10 of slack covers the quadrature.
    L = 6
    c = kl.Chordal(kl.RBF(lengthscale=0.5, variance=1.1))
    X = fibonacci_lonlat(100)
    sh = kl.SphericalHarmonicFeatures(max_degree=L).fit(c, X)
    a = kl.funk_hecke_coefficients(c, 60, convention="legendre")
    tail = float(jnp.sum(a[L + 1 :]))
    err = float(jnp.max(jnp.abs(_gram(sh(X)) - c(X, X))))
    assert tail > 1e-4  # the truncation is visible at L = 6
    assert err <= tail + 1e-10
    # At u = v the series error equals the tail exactly.
    diag_err = jnp.abs(einsum(sh(X), sh(X), "n f, n f -> n") - c.diag(X))
    assert float(jnp.max(jnp.abs(diag_err - tail))) < 1e-8


def test_bare_kernel_on_a_sphere_of_given_radius():
    X = fibonacci_lonlat(50)
    k = kl.Matern(nu=2.5, lengthscale=2.0)
    bare = kl.SphericalHarmonicFeatures(max_degree=10, radius=3.0).fit(k, X)
    chordal = kl.SphericalHarmonicFeatures(max_degree=10).fit(
        kl.Chordal(k, radius=3.0), X
    )
    assert jnp.allclose(bare(X), chordal(X), atol=1e-12)


@pytest.mark.slow
def test_truncating_a_series_kernel_keeps_its_leading_coefficients():
    k = kl.SphereMatern(lengthscale=0.5, max_degree=30)
    X = fibonacci_lonlat(60)
    sh = kl.SphericalHarmonicFeatures(max_degree=10).fit(k, X)
    assert sh.n_features == 121
    cos = einsum(lonlat_to_unit(X), lonlat_to_unit(X), "n d, m d -> n m")
    expected = legendre_series(cos, k.degree_spectrum()[:11])
    assert float(jnp.max(jnp.abs(_gram(sh(X)) - expected))) < 1e-10


@pytest.mark.slow
def test_cartesian_and_radian_inputs():
    X = fibonacci_lonlat(30)
    k = kl.SphereHeat(lengthscale=0.5, max_degree=12)
    sh = kl.SphericalHarmonicFeatures().fit(k, X)
    xyz = 6.0 * lonlat_to_unit(X)  # any radius: (N, 3) inputs are normalised
    assert jnp.allclose(sh(xyz), sh(X), atol=1e-12)
    k_rad = kl.SphereHeat(lengthscale=0.5, max_degree=12, degrees=False)
    sh_rad = kl.SphericalHarmonicFeatures().fit(k_rad, X)
    assert sh_rad.degrees is False
    X_rad = fibonacci_lonlat(30, degrees=False)
    assert jnp.allclose(sh_rad(X_rad), sh(X), atol=1e-12)


@pytest.mark.slow
@pytest.mark.parametrize(
    "kernel",
    [
        kl.SphereMatern(lengthscale=0.5, nu=1.5, max_degree=12),
        kl.SphereHeat(lengthscale=0.2, max_degree=200),  # a_l underflows to 0
        kl.Chordal(kl.RBF(lengthscale=0.5)),
    ],
    ids=["matern", "heat-underflow", "chordal"],
)
def test_gradients_reach_the_lengthscale(kernel):
    X = fibonacci_lonlat(20)
    sh = kl.SphericalHarmonicFeatures(max_degree=12).fit(kernel, X)

    @eqx.filter_grad
    def grad(m):
        return jnp.sum(m(X) ** 2)

    g = grad(sh)
    inner = g.kernel.kernel if isinstance(kernel, kl.Chordal) else g.kernel
    assert bool(jnp.isfinite(inner.lengthscale))
    assert float(jnp.abs(inner.lengthscale)) > 0


@pytest.mark.slow
def test_jit_and_operator():
    X = fibonacci_lonlat(25)
    k = kl.SphereMatern(lengthscale=0.5, max_degree=8)
    sh = kl.SphericalHarmonicFeatures().fit(k, X)
    Phi = eqx.filter_jit(lambda m, x: m(x))(sh, X)
    assert jnp.allclose(Phi, sh(X))
    assert jnp.allclose(sh.operator(X).as_matrix(), k(X, X), atol=1e-10)


def test_n_features_before_fit():
    assert kl.SphericalHarmonicFeatures(max_degree=4).n_features == 25
    with pytest.raises(RuntimeError, match="max_degree"):
        _ = kl.SphericalHarmonicFeatures().n_features


def test_unfitted_call_raises():
    with pytest.raises(RuntimeError, match="not fitted"):
        kl.SphericalHarmonicFeatures(max_degree=2)(fibonacci_lonlat(3))


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"max_degree": -1}, "max_degree"),
        ({"radius": 0.0}, "radius"),
        ({"num_quadrature": 0}, "num_quadrature"),
    ],
)
def test_invalid_configuration(kwargs, match):
    with pytest.raises(ValueError, match=match):
        kl.SphericalHarmonicFeatures(**kwargs)


@pytest.mark.parametrize(
    "kernel",
    [
        kl.GreatCircleExponential(lengthscale=0.5),
        kl.GeometricAnisotropy(kl.RBF(), angles=0.3, ratios=0.5),
        kl.Scaled(kl.Chordal(kl.RBF()), 2.0),
        kl.ActiveDims(kl.RBF(), dims=(0, 1)),
    ],
    ids=["great-circle", "anisotropy", "nested-chordal", "active-dims"],
)
def test_non_zonal_kernels_are_rejected(kernel):
    with pytest.raises(NotImplementedError, match="zonal"):
        kl.SphericalHarmonicFeatures(max_degree=4).fit(kernel, fibonacci_lonlat(5))


def test_ard_kernel_is_rejected():
    k = kl.RBF(lengthscale=jnp.array([0.5, 1.0, 2.0]))
    with pytest.raises(NotImplementedError, match="per-dimension"):
        kl.SphericalHarmonicFeatures(max_degree=4).fit(k, fibonacci_lonlat(5))


def test_fit_errors():
    X = fibonacci_lonlat(5)
    with pytest.raises(ValueError, match="max_degree is required"):
        kl.SphericalHarmonicFeatures().fit(kl.Chordal(kl.RBF()), X)
    with pytest.raises(ValueError, match="exceeds"):
        kl.SphericalHarmonicFeatures(max_degree=9).fit(kl.SphereHeat(max_degree=8), X)
    with pytest.raises(ValueError, match="contradicts"):
        kl.SphericalHarmonicFeatures(degrees=False).fit(kl.SphereHeat(), X)
    with pytest.raises(ValueError, match="contradicts"):
        kl.SphericalHarmonicFeatures(max_degree=3, radius=2.0).fit(
            kl.Chordal(kl.RBF(), radius=1.0), X
        )
    with pytest.raises(ValueError, match="bare kernel"):
        kl.SphericalHarmonicFeatures(radius=2.0).fit(kl.SphereHeat(), X)
    with pytest.raises(ValueError, match="inputs"):
        kl.SphericalHarmonicFeatures(max_degree=2).fit(
            kl.SphereHeat(), jnp.zeros((5, 4))
        )


@pytest.mark.slow
def test_sphere_series_with_a_zero_spectrum_tail():
    # A band-limited user spectrum: degrees above 3 carry zero weight.
    k = kl.SphereSeries(lambda lam: jnp.where(lam <= 12.0, 1.0, 0.0), max_degree=6)
    X = fibonacci_lonlat(40)
    sh = kl.SphericalHarmonicFeatures().fit(k, X)
    w = sh.degree_weights()
    assert bool(jnp.all(w[4:] == 0.0))
    assert float(jnp.max(jnp.abs(_gram(sh(X)) - k(X, X)))) < 1e-10
    assert jax.numpy.all(jnp.isfinite(sh(X)))
