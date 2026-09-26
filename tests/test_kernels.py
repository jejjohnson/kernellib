"""Tests for the kernel classes and composition.

Every property in the design doc's testing section, parametrised over the
kernel zoo: symmetry, PSD, diag == diag(gram), pointwise == Gram, agreement
with `kernellib.functional`, finite hyperparameter gradients, jit and vmap.

Inputs use pinned keys: the randomness is incidental, so tolerances are
deterministic.
"""

from __future__ import annotations

from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

import kernellib as kl
from kernellib import functional as F


X1 = jr.normal(jr.key(0), (7, 3))
X2 = jr.normal(jr.key(1), (5, 3))


def _matern_reference(A, B, nu):
    return F.matern_kernel(A, B, jnp.array(1.7), jnp.array(0.9), nu)


# (kernel, functional reference, is PSD). Periodic and Cosine act on the
# Euclidean distance and are only PSD for 1-D inputs; see test_psd_in_one_dimension.
ZOO = [
    pytest.param(
        kl.RBF(lengthscale=0.7, variance=1.3),
        lambda A, B: F.rbf_kernel(A, B, jnp.array(1.3), jnp.array(0.7)),
        True,
        id="rbf",
    ),
    pytest.param(
        kl.RBF(lengthscale=jnp.array([0.5, 1.0, 2.0]), variance=1.3),
        lambda A, B: F.rbf_kernel(A, B, jnp.array(1.3), jnp.array([0.5, 1.0, 2.0])),
        True,
        id="rbf-ard",
    ),
    *[
        pytest.param(
            kl.Matern(lengthscale=0.9, variance=1.7, nu=nu),
            partial(_matern_reference, nu=nu),
            True,
            id=f"matern{nu}",
        )
        for nu in (0.5, 1.5, 2.5)
    ],
    pytest.param(
        kl.RationalQuadratic(lengthscale=0.8, variance=1.1, alpha=1.7),
        lambda A, B: F.rational_quadratic_kernel(
            A, B, jnp.array(1.1), jnp.array(0.8), jnp.array(1.7)
        ),
        True,
        id="rational_quadratic",
    ),
    pytest.param(
        kl.Periodic(lengthscale=0.6, variance=1.3, period=1.7),
        lambda A, B: F.periodic_kernel(
            A, B, jnp.array(1.3), jnp.array(0.6), jnp.array(1.7)
        ),
        False,
        id="periodic",
    ),
    pytest.param(
        kl.Cosine(variance=1.3, period=1.7),
        lambda A, B: F.cosine_kernel(A, B, jnp.array(1.3), jnp.array(1.7)),
        False,
        id="cosine",
    ),
    pytest.param(
        kl.Linear(variance=0.8, bias=0.3),
        lambda A, B: F.linear_kernel(A, B, jnp.array(0.8), jnp.array(0.3)),
        True,
        id="linear",
    ),
    pytest.param(
        kl.Polynomial(variance=0.8, bias=0.3, degree=3),
        lambda A, B: F.polynomial_kernel(A, B, jnp.array(0.8), jnp.array(0.3), 3),
        True,
        id="polynomial",
    ),
    pytest.param(
        kl.White(variance=0.5),
        lambda A, B: F.white_kernel(A, B, jnp.array(0.5)),
        True,
        id="white",
    ),
    pytest.param(
        kl.Constant(variance=1.8),
        lambda A, B: F.constant_kernel(A, B, jnp.array(1.8)),
        True,
        id="constant",
    ),
]


@pytest.mark.parametrize("kernel,reference,psd", ZOO)
class TestZoo:
    def test_gram_matches_functional(self, kernel, reference, psd):
        assert jnp.allclose(kernel(X1, X2), reference(X1, X2), rtol=0, atol=1e-12)

    def test_symmetric_under_argument_swap(self, kernel, reference, psd):
        assert jnp.allclose(kernel(X1, X2), kernel(X2, X1).T, atol=1e-12)

    def test_psd(self, kernel, reference, psd):
        if not psd:
            pytest.skip("only PSD for 1-D inputs")
        eigs = jnp.linalg.eigvalsh(kernel.gram(X1))
        assert float(eigs.min()) > -1e-9

    def test_diag_matches_gram_diagonal(self, kernel, reference, psd):
        assert jnp.allclose(kernel.diag(X1), jnp.diag(kernel.gram(X1)), atol=1e-12)

    def test_pointwise_matches_gram(self, kernel, reference, psd):
        assert kernel.is_pointwise
        K_pw = jax.vmap(lambda x: jax.vmap(lambda y: kernel.pairwise(x, y))(X2))(X1)
        assert jnp.allclose(K_pw, kernel(X1, X2), atol=1e-10)

    def test_hyperparameter_grad_is_finite(self, kernel, reference, psd):
        params, static = eqx.partition(kernel, eqx.is_inexact_array)

        def loss(p):
            return jnp.sum(eqx.combine(p, static).gram(X1))

        grads = jax.grad(loss)(params)
        leaves = jax.tree_util.tree_leaves(grads)
        assert leaves
        assert all(bool(jnp.all(jnp.isfinite(g))) for g in leaves)

    def test_jit_and_vmap(self, kernel, reference, psd):
        K = eqx.filter_jit(lambda k, A: k(A, A))(kernel, X1)
        assert jnp.allclose(K, kernel(X1, X1), atol=1e-12)
        batch = jnp.stack([X1, X1 + 1.0])
        Ks = jax.vmap(lambda A: kernel(A, A))(batch)
        assert Ks.shape == (2, 7, 7)


@pytest.mark.parametrize(
    "kernel",
    [kl.Periodic(lengthscale=0.6, variance=1.3, period=1.7), kl.Cosine(period=1.7)],
    ids=["periodic", "cosine"],
)
def test_psd_in_one_dimension(kernel):
    X = jnp.linspace(-3.0, 3.0, 25)[:, None]
    assert float(jnp.linalg.eigvalsh(kernel.gram(X)).min()) > -1e-9


# --- defaults and validation ----------------------------------------------


def test_defaults_match_pyrox_gp():
    assert float(kl.RBF().lengthscale) == 1.0
    assert float(kl.RBF().variance) == 1.0
    assert kl.Matern().nu == 2.5
    assert kl.Polynomial().degree == 2
    assert float(kl.Linear().bias) == 0.0
    assert float(kl.RationalQuadratic().alpha) == 1.0
    assert float(kl.Periodic().period) == 1.0


def test_matern_rejects_unsupported_nu():
    with pytest.raises(ValueError, match="nu"):
        kl.Matern(nu=1.0)


def test_polynomial_rejects_degree_zero():
    with pytest.raises(ValueError, match="degree"):
        kl.Polynomial(degree=0)


def test_ard_pairwise_rejects_mismatched_features():
    k = kl.RBF(lengthscale=jnp.ones(3))
    with pytest.raises(ValueError, match="many features"):
        k.pairwise(jnp.ones(1), jnp.ones(1))


def test_matern_pairwise_grad_finite_at_zero_distance():
    x = jnp.array([0.5, 0.5])
    g = jax.grad(lambda ls: kl.Matern(lengthscale=ls, nu=1.5).pairwise(x, x))(0.7)
    assert jnp.isfinite(g)


# --- composition ----------------------------------------------------------


def test_add_builds_flat_sum():
    k = kl.RBF() + kl.Linear() + kl.White()
    assert isinstance(k, kl.Sum)
    assert len(k.kernels) == 3


def test_mul_builds_flat_product_and_scaled():
    k = kl.RBF() * kl.Linear() * kl.Periodic()
    assert isinstance(k, kl.Product)
    assert len(k.kernels) == 3
    assert isinstance(2.0 * kl.RBF(), kl.Scaled)
    assert isinstance(kl.RBF() * 2.0, kl.Scaled)


def test_sum_product_scaled_values():
    a, b = kl.RBF(lengthscale=0.7), kl.Linear(variance=0.4)
    assert jnp.allclose((a + b)(X1, X2), a(X1, X2) + b(X1, X2))
    assert jnp.allclose((a * b)(X1, X2), a(X1, X2) * b(X1, X2))
    assert jnp.allclose((3.0 * a)(X1, X2), 3.0 * a(X1, X2))


@pytest.mark.parametrize(
    "kernel",
    [
        kl.RBF(0.7) + kl.White(0.1),
        kl.RBF(0.7) * kl.Linear(),
        2.5 * kl.Matern(nu=1.5),
        kl.ActiveDims(kl.RBF(0.9), dims=(0, 2)),
        kl.Warped(kl.RBF(0.9), warp=jnp.tanh),
    ],
    ids=["sum", "product", "scaled", "active_dims", "warped"],
)
def test_composites_pointwise_diag_and_grad(kernel):
    assert kernel.is_pointwise
    K_pw = jax.vmap(lambda x: jax.vmap(lambda y: kernel.pairwise(x, y))(X2))(X1)
    assert jnp.allclose(K_pw, kernel(X1, X2), atol=1e-10)
    assert jnp.allclose(kernel.diag(X1), jnp.diag(kernel.gram(X1)), atol=1e-12)
    params, static = eqx.partition(kernel, eqx.is_inexact_array)
    grads = jax.grad(lambda p: jnp.sum(eqx.combine(p, static).gram(X1)))(params)
    assert all(bool(jnp.all(jnp.isfinite(g))) for g in jax.tree_util.tree_leaves(grads))


def test_active_dims_selects_columns():
    k = kl.ActiveDims(kl.RBF(0.9), dims=(0, 2))
    assert jnp.allclose(k(X1, X2), kl.RBF(0.9)(X1[:, [0, 2]], X2[:, [0, 2]]))


def test_warped_applies_warp():
    k = kl.Warped(kl.RBF(0.9), warp=jnp.tanh)
    assert jnp.allclose(k(X1, X2), kl.RBF(0.9)(jnp.tanh(X1), jnp.tanh(X2)))


class _GramOnly(kl.AbstractKernel):
    def __call__(self, A, B):
        return A @ B.T


def test_gram_only_kernel():
    k = _GramOnly()
    assert not k.is_pointwise
    assert jnp.allclose(k.diag(X1), jnp.sum(X1 * X1, axis=-1))
    with pytest.raises(TypeError, match="Gram-only"):
        k.pairwise(X1[0], X2[0])


def test_composite_with_gram_only_child_is_not_pointwise():
    k = kl.RBF() + _GramOnly()
    assert not k.is_pointwise
    assert jnp.allclose(k(X1, X2), kl.RBF()(X1, X2) + X1 @ X2.T)


def test_composites_reject_empty_and_non_kernels():
    with pytest.raises(ValueError, match="at least one"):
        kl.Sum()
    with pytest.raises(TypeError, match="takes kernels"):
        kl.Product(kl.RBF(), 3.0)
    with pytest.raises(ValueError, match="at least one dimension"):
        kl.ActiveDims(kl.RBF(), dims=())


# -- derivatives at coincident points ----------------------------------------
# jax.hessian of pairwise at x = y used to be zero for the kernels computed
# through a sqrt guard (Matern 3/2 and 5/2, Periodic, Cosine), because the
# guard clips r^2 and zeroes the gradient of r. The closed forms below are
# d^2 k / dx dy^T at x = y, a multiple of the identity.

_ELL, _VAR, _PERIOD = 0.7, 1.5, 2.0


@pytest.mark.parametrize(
    ("kernel", "dim", "expected"),
    [
        (kl.RBF(_ELL, _VAR), 2, _VAR / _ELL**2),
        (kl.Matern(_ELL, _VAR, nu=1.5), 2, 3.0 * _VAR / _ELL**2),
        (kl.Matern(_ELL, _VAR, nu=2.5), 2, 5.0 * _VAR / (3.0 * _ELL**2)),
        (
            kl.Periodic(_ELL, _VAR, period=_PERIOD),
            1,
            4 * jnp.pi**2 * _VAR / (_PERIOD * _ELL) ** 2,
        ),
        (kl.Cosine(_VAR, period=_PERIOD), 1, _VAR * (2 * jnp.pi / _PERIOD) ** 2),
    ],
    ids=["rbf", "matern15", "matern25", "periodic", "cosine"],
)
def test_cross_hessian_at_coincident_points(kernel, dim, expected):
    x = jnp.linspace(0.1, 0.4, dim)
    H = jax.jacfwd(jax.grad(kernel.pairwise, argnums=0), argnums=1)(x, x)
    assert jnp.allclose(H, expected * jnp.eye(dim), rtol=1e-10)


@pytest.mark.parametrize("nu", [1.5, 2.5])
def test_derivative_gram_is_psd(nu):
    k = kl.Matern(lengthscale=0.8, nu=nu)
    X = jax.random.normal(jax.random.key(0), (6, 2))
    d2 = jax.jacfwd(jax.grad(k.pairwise, 0), 1)
    blocks = jax.vmap(lambda x: jax.vmap(lambda y: d2(x, y))(X))(X)
    G = blocks.transpose(0, 2, 1, 3).reshape(12, 12)
    assert jnp.linalg.eigvalsh(G)[0] > -1e-10


@pytest.mark.parametrize(
    "kernel",
    [
        kl.Matern(nu=1.5),
        kl.Matern(nu=2.5),
        kl.Periodic(period=_PERIOD),
        kl.Cosine(period=_PERIOD),
    ],
    ids=["matern15", "matern25", "periodic", "cosine"],
)
def test_taylor_branch_is_continuous(kernel):
    # Values and gradients agree on both sides of the switch to the expansion.
    x = jnp.zeros(1)
    below = jnp.array([jnp.sqrt(1e-10) * (1 - 1e-6)])
    above = jnp.array([jnp.sqrt(1e-10) * (1 + 1e-6)])
    assert jnp.allclose(
        kernel.pairwise(x, below), kernel.pairwise(x, above), rtol=1e-12
    )
    g = jax.grad(kernel.pairwise, argnums=1)
    assert jnp.allclose(g(x, below), g(x, above), rtol=1e-4)


def test_matern_half_gradient_is_finite_at_zero():
    g = jax.grad(kl.Matern(nu=0.5).pairwise, argnums=0)(jnp.zeros(2), jnp.zeros(2))
    assert jnp.all(jnp.isfinite(g))
