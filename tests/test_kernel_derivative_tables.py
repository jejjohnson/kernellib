"""Closed-form derivative tables of Johnson et al. (2020), pinned to autodiff.

Johnson, Laparra, Piles & Camps-Valls (2020), *Kernel methods and their
derivatives*, PLOS ONE 15(10): e0235885 (arXiv:2007.14706), Tables 1 and 2
give the first, second and mixed partials of the Linear, Polynomial, RBF,
Tanh and ARD kernels. Each test evaluates the closed form in the paper's
parameterisation and compares it with ``jax.grad`` / ``jax.hessian`` of
kernellib's ``pairwise`` (derivatives with respect to the first argument),
through this mapping:

| Paper | kernellib |
|---|---|
| Linear ``x.y`` | ``Linear(variance=1, bias=0)`` |
| Poly ``(g x.y + c0)^p`` | ``Polynomial(variance=g**p, bias=c0/g, degree=p)`` |
| RBF ``exp(-g |x-y|^2)`` | ``RBF(lengthscale=1/sqrt(2 g), variance=1)`` |
| Tanh ``tanh(g x.y + c0)`` | ``Tanh(variance=1, scale=g, bias=c0)`` |
| ARD ``nu^2 exp(-sum_d ((x_d-y_d)/lam_d)^2 / 2)`` | ``RBF(lam, nu**2)`` |

ERRATA (kernellib issue #271). The paper's ARD row has two sign errors: the
first partial is printed ``+(h_j/lam_j^2) k`` and the second partial
``(1/lam_j^2 + (h_j/lam_j^2)^2) k``, but the correct forms are
``-(h_j/lam_j^2) k`` and ``((h_j/lam_j^2)^2 - 1/lam_j^2) k`` (the paper's own
RBF row with ``g = 1/(2 lam^2)`` agrees with the corrected ones). The tests
below assert the CORRECTED forms and that the printed ones fail; do not "fix"
them to match the paper.
"""

from __future__ import annotations

import einx
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

import kernellib as kl
from kernellib._einx import rearrange
from kernellib._kernels._derivative import _check_differentiable


D = 3
N_PAIRS = 20
GAMMA, C0 = 0.8, 0.3
LAM = jnp.array([0.7, 1.3, 0.9])
NU2 = 1.5
XS = jr.normal(jr.key(0), (N_PAIRS, D))
YS = jr.normal(jr.key(1), (N_PAIRS, D))
OFF_DIAGONAL = ~jnp.eye(D, dtype=bool)
TOL = {"atol": 1e-12, "rtol": 1e-12}


def _outer(a, b):
    return einx.multiply("j, k -> j k", a, b)


# Each case returns (kernellib kernel, closed form (x, y) -> (grad, d2_jj, d2_jk)):
# grad (D,), d2_jj (D,), d2_jk (D, D) (off-diagonal entries are the mixed partials).


def _linear():
    def closed(x, y):
        z = jnp.zeros(D)
        return y, z, jnp.zeros((D, D))

    return kl.Linear(variance=1.0, bias=0.0), closed


def _poly(p):
    def closed(x, y):
        u = GAMMA * jnp.dot(x, y) + C0
        grad = GAMMA * p * y * u ** (p - 1)
        d2 = p * (p - 1) * (GAMMA * y) ** 2 * u ** (p - 2)
        mixed = p * (p - 1) * GAMMA**2 * _outer(y, y) * u ** (p - 2)
        return grad, d2, mixed

    return kl.Polynomial(variance=GAMMA**p, bias=C0 / GAMMA, degree=p), closed


def _rbf():
    def closed(x, y):
        h = x - y
        k = jnp.exp(-GAMMA * jnp.dot(h, h))
        grad = -2 * GAMMA * h * k
        d2 = 2 * GAMMA * (2 * GAMMA * h**2 - 1) * k
        mixed = 4 * GAMMA**2 * _outer(h, h) * k
        return grad, d2, mixed

    return kl.RBF(lengthscale=1.0 / jnp.sqrt(2 * GAMMA), variance=1.0), closed


def _tanh():
    def closed(x, y):
        u = GAMMA * jnp.dot(x, y) + C0
        k = jnp.tanh(u)
        sech2 = 1 - k**2
        grad = GAMMA * y * sech2
        d2 = -2 * (GAMMA * y) ** 2 * sech2 * k
        mixed = -2 * GAMMA**2 * _outer(y, y) * sech2 * k
        return grad, d2, mixed

    return kl.Tanh(variance=1.0, scale=GAMMA, bias=C0), closed


def _ard():
    # CORRECTED forms (see the module docstring).
    def closed(x, y):
        h = x - y
        k = NU2 * jnp.exp(-0.5 * jnp.sum((h / LAM) ** 2))
        grad = -h / LAM**2 * k
        d2 = (h**2 / LAM**4 - 1 / LAM**2) * k
        mixed = _outer(h / LAM**2, h / LAM**2) * k
        return grad, d2, mixed

    return kl.RBF(lengthscale=LAM, variance=NU2), closed


CASES = [
    pytest.param(_linear, id="linear"),
    pytest.param(lambda: _poly(2), id="poly2"),
    pytest.param(lambda: _poly(3), id="poly3"),
    pytest.param(_rbf, id="rbf"),
    pytest.param(_tanh, id="tanh"),
    pytest.param(_ard, id="ard"),
]


@pytest.mark.parametrize("make", CASES)
class TestTables:
    def test_first_partials(self, make):
        kernel, closed = make()
        auto = jax.vmap(jax.grad(kernel.pairwise, argnums=0))(XS, YS)
        table = jax.vmap(lambda x, y: closed(x, y)[0])(XS, YS)
        assert jnp.allclose(auto, table, **TOL)

    def test_second_partials(self, make):
        kernel, closed = make()
        hess = jax.vmap(jax.hessian(kernel.pairwise, argnums=0))(XS, YS)
        table = jax.vmap(lambda x, y: closed(x, y)[1])(XS, YS)
        auto = jax.vmap(jnp.diag)(hess)
        assert jnp.allclose(auto, table, **TOL)

    def test_mixed_partials(self, make):
        kernel, closed = make()
        hess = jax.vmap(jax.hessian(kernel.pairwise, argnums=0))(XS, YS)
        table = jax.vmap(lambda x, y: closed(x, y)[2])(XS, YS)
        mask = rearrange(OFF_DIAGONAL, "j k -> 1 j k")
        assert jnp.allclose(
            jnp.where(mask, hess, 0.0), jnp.where(mask, table, 0.0), **TOL
        )


class TestMapping:
    @pytest.mark.parametrize("p", [1, 2, 3])
    def test_polynomial(self, p):
        k = kl.Polynomial(variance=GAMMA**p, bias=C0 / GAMMA, degree=p)
        got = jax.vmap(k.pairwise)(XS, YS)
        want = jax.vmap(lambda x, y: (GAMMA * jnp.dot(x, y) + C0) ** p)(XS, YS)
        assert jnp.allclose(got, want, atol=1e-14, rtol=1e-14)

    def test_rbf(self):
        k = kl.RBF(lengthscale=1.0 / jnp.sqrt(2 * GAMMA))
        got = jax.vmap(k.pairwise)(XS, YS)
        want = jax.vmap(lambda x, y: jnp.exp(-GAMMA * jnp.sum((x - y) ** 2)))(XS, YS)
        assert jnp.allclose(got, want, atol=1e-14, rtol=1e-14)

    def test_tanh_is_the_sklearn_sigmoid(self):
        k = kl.Tanh(scale=GAMMA, bias=C0)
        want = jnp.tanh(GAMMA * einx.dot("n d, m d -> n m", XS, YS) + C0)
        assert jnp.allclose(k(XS, YS), want, atol=1e-14, rtol=1e-14)


# -- errata guard (kernellib issue #271) ---------------------------------------

X_ERR = jnp.array([0.3, -0.4])
Y_ERR = jnp.array([-0.2, 0.5])
LAM_ERR = jnp.array([0.7, 1.3])
NU2_ERR = 1.5


def _ard_autodiff():
    k = kl.RBF(lengthscale=LAM_ERR, variance=NU2_ERR)
    grad = jax.grad(k.pairwise, argnums=0)(X_ERR, Y_ERR)
    d2 = jnp.diag(jax.hessian(k.pairwise, argnums=0)(X_ERR, Y_ERR))
    return grad, d2


def _ard_kvalue():
    h = X_ERR - Y_ERR
    return NU2_ERR * jnp.exp(-0.5 * jnp.sum((h / LAM_ERR) ** 2))


def test_ard_corrected_forms_match_autodiff_at_the_measured_point():
    grad, d2 = _ard_autodiff()
    assert jnp.allclose(grad, jnp.array([-0.9333, 0.4871]), atol=1e-4)
    assert jnp.allclose(d2, jnp.array([-0.9142, -0.2818]), atol=1e-4)


def test_paper_printed_ard_forms_do_not_match_autodiff():
    """Pins the two sign errata so a "fix to match the paper" fails (#271)."""
    grad, d2 = _ard_autodiff()
    h, k = X_ERR - Y_ERR, _ard_kvalue()
    printed_grad = h / LAM_ERR**2 * k  # paper Table 1: missing minus sign
    printed_d2 = (1 / LAM_ERR**2 + (h / LAM_ERR**2) ** 2) * k  # paper Table 2
    assert not jnp.allclose(grad, printed_grad, atol=1e-6)
    assert jnp.allclose(grad, -printed_grad, atol=1e-12)
    assert not jnp.allclose(d2, printed_d2, atol=1e-6)
    assert jnp.allclose(printed_d2, jnp.array([2.8188, 0.8006]), atol=1e-4)


# -- higher order (Eq. 7, Appendix A): Hermite oracle --------------------------

HERMITE = {
    1: lambda z: z,
    2: lambda z: z**2 - 1,
    3: lambda z: z**3 - 3 * z,
    4: lambda z: z**4 - 6 * z**2 + 3,
}


@pytest.mark.parametrize("m", [1, 2, 3, 4])
def test_rbf_higher_order_derivatives_match_hermite(m):
    ell = 0.8
    k = kl.RBF(lengthscale=ell, variance=1.0)
    y = jnp.array([0.4])

    def f(x):
        return k.pairwise(rearrange(x, " -> 1"), y)

    g = f
    for _ in range(m):
        g = jax.grad(g)
    for h in (0.0, 0.3, -0.7, 1.9):
        x = jnp.asarray(0.4 + h)
        z = h / ell
        want = (-1 / ell) ** m * HERMITE[m](z) * jnp.exp(-0.5 * z**2)
        assert jnp.allclose(g(x), want, atol=1e-10, rtol=1e-10)


# -- Tanh specifics -------------------------------------------------------------


def test_tanh_gram_is_indefinite_on_the_documented_example():
    X = jnp.array([[1.0], [2.0]])
    eigs = jnp.linalg.eigvalsh(kl.Tanh()(X, X))
    assert abs(float(eigs[0]) - (-0.0909)) < 1e-4
    assert abs(float(eigs[1]) - 1.852) < 1e-3


def test_tanh_is_differentiable_and_non_psd_through_derivative_kernels():
    k = kl.Tanh()
    _check_differentiable(k, "Tanh")
    d = kl.Derivative(k, dx=0, dy=0)
    x = jnp.array([0.3, -0.2])
    assert jnp.isfinite(d.pairwise(x, x))
    assert not k.is_psd
    assert not d.is_psd
    assert not kl.DerivativeIndexed(k).is_psd


def test_is_psd_propagates_through_composites():
    assert kl.RBF().is_psd
    assert kl.Periodic().is_psd and kl.Cosine().is_psd
    assert not (kl.RBF() + kl.Tanh()).is_psd
    assert not (kl.RBF() * kl.Tanh()).is_psd
    assert not (2.0 * kl.Tanh()).is_psd
    assert not kl.Tanh().select([0]).is_psd
    assert not kl.Tanh().stretch(2.0).is_psd
    assert (2.0 * kl.RBF()).is_psd
    assert not (-2.0 * kl.RBF()).is_psd
