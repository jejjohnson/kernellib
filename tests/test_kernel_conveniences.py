"""Transform methods, elwise and is_stationary (#60)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

import kernellib as kl


X = jax.random.normal(jax.random.key(0), (9, 3))
Y = jax.random.normal(jax.random.key(1), (9, 3))


class _GramOnly(kl.AbstractKernel):
    def __call__(self, A, B):
        return (A @ B.T) ** 2


@pytest.mark.slow
def test_stretch_is_a_lengthscale_for_stationary_kernels():
    k = kl.RBF(lengthscale=0.5).stretch(2.0)
    assert isinstance(k, kl.Warped) and isinstance(k.warp, kl.Stretch)
    assert jnp.allclose(k(X, Y), kl.RBF(lengthscale=1.0)(X, Y))
    ard = kl.Matern(nu=1.5).stretch(jnp.array([0.5, 1.0, 2.0]))
    expected = kl.Matern(nu=1.5, lengthscale=jnp.array([0.5, 1.0, 2.0]))
    assert jnp.allclose(ard(X, Y), expected(X, Y))


def test_stretch_scale_is_differentiable():
    g = jax.grad(lambda c: jnp.sum(kl.Linear().stretch(c)(X, Y)))(jnp.array(1.5))
    assert jnp.isfinite(g) and g != 0.0


def test_shift_is_a_noop_for_stationary_kernels():
    k = kl.RBF()
    assert k.shift(3.0) is k
    lin = kl.Linear().shift(jnp.array([1.0, 0.0, -1.0]))
    assert jnp.allclose(
        lin(X, Y), kl.Linear()(X - lin.warp.offset, Y - lin.warp.offset)
    )


def test_transform_select_periodic_build_the_constructors():
    w = jnp.tanh
    assert jnp.allclose(kl.RBF().transform(w)(X, Y), kl.Warped(kl.RBF(), w)(X, Y))
    sel = kl.Matern().select([0, 2])
    assert isinstance(sel, kl.ActiveDims) and sel.dims == (0, 2)
    assert jnp.allclose(sel(X, Y), kl.Matern()(X[:, [0, 2]], Y[:, [0, 2]]))
    per = kl.RBF().periodic(2.0)
    assert isinstance(per, kl.Periodised)
    assert jnp.allclose(per(X, Y), kl.Periodised(kl.RBF(), 2.0)(X, Y))


@pytest.mark.slow
@pytest.mark.parametrize(
    "kernel",
    [kl.RBF(), kl.Linear() + kl.Matern(), 2.0 * kl.Periodic(), _GramOnly()],
    ids=["rbf", "sum", "scaled", "gram-only"],
)
def test_elwise_is_the_diagonal_of_the_cross_gram(kernel):
    assert jnp.allclose(kernel.elwise(X, Y), jnp.diag(kernel(X, Y)))


def test_elwise_is_linear_in_n():
    n = 4096
    A = jnp.zeros((n, 2))
    for kernel in (kl.RBF(), _GramOnly()):
        assert f"{n},{n}" not in str(jax.make_jaxpr(kernel.elwise)(A, A))


STATIONARY = {
    "rbf": (kl.RBF(), True),
    "matern": (kl.Matern(), True),
    "periodic": (kl.Periodic(), True),
    "cosine": (kl.Cosine(), True),
    "white": (kl.White(), True),
    "constant": (kl.Constant(), True),
    "linear": (kl.Linear(), False),
    "polynomial": (kl.Polynomial(degree=2), False),
    "sum": (kl.RBF() + kl.Matern(), True),
    "sum-with-linear": (kl.RBF() + kl.Linear(), False),
    "product": (kl.RBF() * kl.Periodic(), True),
    "scaled": (3.0 * kl.RBF(), True),
    "active-dims": (kl.RBF().select([1]), True),
    "stretch": (kl.RBF().stretch(2.0), True),
    "shifted-linear": (kl.Linear().shift(1.0), False),
    "warped": (kl.RBF().transform(jnp.tanh), False),
    "periodised": (kl.Matern().periodic(2.0), True),
    "periodised-linear": (kl.Linear().periodic(2.0), False),
    "modulated": (kl.Modulated(kl.RBF(), lambda x: x[0]), False),
    "feature": (kl.FeatureKernel(lambda x: x), False),
    "gram-only": (_GramOnly(), False),
}


@pytest.mark.parametrize(
    ("kernel", "expected"), STATIONARY.values(), ids=STATIONARY.keys()
)
def test_is_stationary(kernel, expected):
    assert kernel.is_stationary is expected
