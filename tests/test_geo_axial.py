"""`LatitudeScaled`, `JunStein` and `LegendreLatitudeFn`: reductions to the base
kernel, axial symmetry, derivative terms against finite differences, and
positive definiteness on a Fibonacci lattice."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

import kernellib as kl
from kernellib._einx import einsum, rearrange
from kernellib._kernels._derivative import _value_gradient_block
from kernellib._testing import fibonacci_lonlat


BASE = kl.Chordal(kl.Matern(nu=2.5, lengthscale=0.6, variance=1.3))
SERIES = kl.SphereMatern(lengthscale=0.5, nu=2.5, max_degree=48)


def _smooth_fns(key=0, n_weights=4):
    """Random smooth ``a, b, c`` with ``a`` and ``b`` vanishing at the poles."""
    ka, kb, kc = jax.random.split(jax.random.key(key), 3)
    a = kl.LegendreLatitudeFn(jax.random.normal(ka, (n_weights,)), cos_factor=True)
    b = kl.LegendreLatitudeFn(jax.random.normal(kb, (n_weights,)), cos_factor=True)
    c = kl.LegendreLatitudeFn(1.0 + 0.3 * jax.random.normal(kc, (n_weights,)))
    return a, b, c


def _points(n=7, key=1):
    kl_, kp = jax.random.split(jax.random.key(key))
    lon = jax.random.uniform(kl_, (n,), minval=-180.0, maxval=180.0)
    lat = jax.random.uniform(kp, (n,), minval=-85.0, maxval=85.0)
    return jnp.column_stack([lon, lat])


def _min_eig(K):
    return float(jnp.linalg.eigvalsh(0.5 * (K + rearrange(K, "i j -> j i")))[0])


# -- LegendreLatitudeFn -------------------------------------------------------


def test_legendre_fn_values_and_cos_factor():
    w = jnp.array([0.5, 1.0, -2.0])
    phi = jnp.array([-1.0, 0.0, 0.7])
    t = jnp.sin(phi)
    expected = 0.5 + t - 2.0 * 0.5 * (3 * t**2 - 1)
    assert jnp.allclose(kl.LegendreLatitudeFn(w)(phi), expected)
    assert jnp.allclose(
        kl.LegendreLatitudeFn(w, cos_factor=True)(phi), jnp.cos(phi) * expected
    )
    assert float(
        kl.LegendreLatitudeFn(w, cos_factor=True)(jnp.pi / 2)
    ) == pytest.approx(0.0, abs=1e-12)


def test_legendre_fn_rejects_bad_weights():
    with pytest.raises(ValueError, match="1-D"):
        kl.LegendreLatitudeFn(jnp.ones((2, 2)))
    with pytest.raises(ValueError, match="non-empty"):
        kl.LegendreLatitudeFn(jnp.zeros((0,)))


def test_legendre_fn_is_differentiable_in_weights():
    g = jax.grad(lambda w: kl.LegendreLatitudeFn(w)(0.4))(jnp.ones(3))
    assert g.shape == (3,) and bool(jnp.all(jnp.isfinite(g)))


# -- LatitudeScaled -----------------------------------------------------------


@pytest.mark.parametrize("base", [BASE, SERIES], ids=["chordal", "series"])
def test_scaled_unit_scale_is_the_base_kernel(base):
    X = _points()
    k = kl.LatitudeScaled(base, lambda phi: jnp.ones_like(phi))
    assert jnp.allclose(k(X, X), base(X, X))
    assert jnp.allclose(k.diag(X), base.diag(X))


def test_scaled_matches_the_definition_and_pairwise():
    X, Y = _points(6, 2), _points(5, 3)
    s = lambda phi: 1.0 + 0.5 * jnp.cos(2.0 * phi)
    k = kl.LatitudeScaled(BASE, s)
    sx, sy = s(jnp.deg2rad(X[:, 1])), s(jnp.deg2rad(Y[:, 1]))
    expected = einsum(sx, BASE(X, Y), sy, "n, n m, m -> n m")
    assert jnp.allclose(k(X, Y), expected)
    assert jnp.allclose(k.pairwise(X[0], Y[1]), expected[0, 1])
    assert jnp.allclose(k.diag(X), jnp.diag(k(X, X)))
    assert k.is_pointwise


def test_scaled_radians_and_module_scale():
    X = _points()
    s = kl.LegendreLatitudeFn(jnp.array([1.0, 0.2]))
    deg = kl.LatitudeScaled(kl.Chordal(BASE.kernel), s)
    rad = kl.LatitudeScaled(kl.Chordal(BASE.kernel, degrees=False), s, degrees=False)
    Xr = jnp.deg2rad(X)
    assert jnp.allclose(deg(X, X), rad(Xr, Xr))


def test_scaled_is_axially_symmetric_and_varies_with_latitude():
    X = _points()
    k = kl.LatitudeScaled(BASE, lambda phi: 1.0 + 0.5 * jnp.sin(phi))
    shifted = X + jnp.array([33.0, 0.0])
    assert jnp.allclose(k(X, X), k(shifted, shifted), atol=1e-10)
    moved = X + jnp.array([0.0, 5.0])
    assert not jnp.allclose(k(X, X), k(moved, moved))


# -- JunStein -----------------------------------------------------------------


def _zero(phi):
    return 0.0 * phi


def _one(phi):
    return 1.0 + 0.0 * phi


@pytest.mark.parametrize("base", [BASE, SERIES], ids=["chordal", "series"])
def test_junstein_trivial_operator_is_the_base_kernel(base):
    X = _points()
    k = kl.JunStein(base, _zero, _zero, _one)
    assert jnp.allclose(k(X, X), base(X, X), atol=1e-10)
    assert jnp.allclose(k.diag(X), base.diag(X), atol=1e-10)


def test_junstein_is_axially_symmetric():
    a, b, c = _smooth_fns()
    k = kl.JunStein(BASE, a, b, c)
    X = _points()
    shifted = X + jnp.array([57.0, 0.0])
    assert jnp.allclose(k(X, X), k(shifted, shifted), atol=1e-10)
    moved = X + jnp.array([0.0, 4.0])
    assert not jnp.allclose(k(X, X), k(moved, moved))


def test_junstein_is_symmetric_and_pairwise_matches_gram():
    a, b, c = _smooth_fns(3)
    k = kl.JunStein(BASE, a, b, c)
    X = _points(6)
    K = k(X, X)
    assert jnp.allclose(K, rearrange(K, "i j -> j i"), atol=1e-10)
    assert jnp.allclose(K[1, 4], k.pairwise(X[1], X[4]))
    assert not k.is_stationary


def _fd_block(f, x, y, h=1e-5):
    """``[[k, d_y k], [d_x k, d_x d_y k]]`` by central differences, in (1, lon, lat)."""

    def at(p, i, s):
        return p.at[i].add(s)

    def dx(g, i):
        return lambda x, y: (g(at(x, i, h), y) - g(at(x, i, -h), y)) / (2 * h)

    def dy(g, j):
        return lambda x, y: (g(x, at(y, j, h)) - g(x, at(y, j, -h))) / (2 * h)

    rows = [[f, dy(f, 0), dy(f, 1)]]
    for i in (0, 1):
        rows.append([dx(f, i), dx(dy(f, 0), i), dx(dy(f, 1), i)])
    return jnp.array([[g(x, y) for g in row] for row in rows])


def test_junstein_derivative_terms_match_finite_differences():
    """The nine terms of the sum, against central differences of k0 (radians)."""
    base = kl.Chordal(BASE.kernel, degrees=False)
    x = jnp.array([0.3, 0.5])
    y = jnp.array([-0.4, -0.2])
    fd = _fd_block(base.pairwise, x, y)
    assert jnp.allclose(_value_gradient_block(base.pairwise, x, y), fd, atol=1e-6)

    a = lambda phi: 0.7 * jnp.cos(phi)
    b = lambda phi: 0.4 + 0.2 * jnp.sin(phi)
    c = lambda phi: 1.0 + 0.3 * jnp.sin(phi)
    k = kl.JunStein(base, a, b, c, degrees=False)
    ax = jnp.array([c(x[1]), b(x[1]), a(x[1])])
    ay = jnp.array([c(y[1]), b(y[1]), a(y[1])])
    assert jnp.allclose(k.pairwise(x, y), einsum(ax, fd, ay, "i, i j, j ->"), atol=1e-6)


def test_junstein_degrees_match_radians():
    a, b, c = _smooth_fns(5)
    deg = kl.JunStein(kl.Chordal(BASE.kernel), a, b, c)
    rad = kl.JunStein(kl.Chordal(BASE.kernel, degrees=False), a, b, c, degrees=False)
    X = _points()
    Xr = jnp.deg2rad(X)
    assert jnp.allclose(deg(X, X), rad(Xr, Xr), atol=1e-10)


def test_junstein_is_finite_at_the_poles():
    a, b, c = _smooth_fns()
    k = kl.JunStein(BASE, a, b, c)
    X = jnp.array([[0.0, 90.0], [120.0, 90.0], [0.0, -90.0], [10.0, 0.0]])
    K = k(X, X)
    assert bool(jnp.all(jnp.isfinite(K)))
    assert _min_eig(K) > -1e-10


@pytest.mark.parametrize(
    "base",
    [
        kl.Chordal(kl.Matern(nu=0.5, lengthscale=0.5)),
        kl.SphereMatern(lengthscale=0.5, nu=1.0),
        kl.GreatCircleExponential(),
    ],
    ids=["exp", "sphere-nu1", "great-circle"],
)
def test_junstein_rejects_rough_base_kernels(base):
    with pytest.raises(ValueError, match="differentiable"):
        kl.JunStein(base, _zero, _zero, _one)


def test_junstein_rejects_mismatched_units():
    with pytest.raises(ValueError, match="degrees"):
        kl.JunStein(kl.Chordal(BASE.kernel, degrees=False), _zero, _zero, _one)


# -- positive definiteness ----------------------------------------------------


def test_scaled_is_positive_definite():
    X = fibonacci_lonlat(400)
    k = kl.LatitudeScaled(BASE, kl.LegendreLatitudeFn(jnp.array([1.0, -0.4, 0.8])))
    assert _min_eig(k(X, X)) >= -1e-10 * X.shape[0]


@pytest.mark.parametrize("key", [0, 1])
@pytest.mark.parametrize("base", [BASE, SERIES], ids=["chordal", "series"])
def test_junstein_is_positive_definite(base, key):
    X = fibonacci_lonlat(400)
    a, b, c = _smooth_fns(key)
    K = kl.JunStein(base, a, b, c)(X, X)
    assert bool(jnp.all(jnp.isfinite(K)))
    assert _min_eig(K) >= -1e-10 * X.shape[0]
