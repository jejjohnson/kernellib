"""Tests for the pure kernel functions in `kernellib.functional`.

Pin the math against direct closed-form computations on small hand-checkable
arrays. Ported verbatim from pyrox-gp's ``tests/gp/test_kernels.py`` (only the
import changed), so passing here means kernel values are unchanged for
pyrox-gp when it switches to kernellib.
"""

from __future__ import annotations

import einx
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from kernellib.functional import (
    constant_kernel,
    cosine_kernel,
    kernel_add,
    kernel_mul,
    linear_kernel,
    matern_kernel,
    periodic_kernel,
    polynomial_kernel,
    rational_quadratic_kernel,
    rbf_kernel,
    white_kernel,
)


# The ARD tests below pin the math to 1e-12 tolerances / bitwise equality.
jax.config.update("jax_enable_x64", True)


# --- rbf -------------------------------------------------------------------


def test_rbf_diagonal_equals_variance():
    X = jnp.array([[0.0], [1.0], [2.0]])
    K = rbf_kernel(X, X, jnp.array(2.5), jnp.array(1.0))
    assert jnp.allclose(jnp.diag(K), 2.5)


def test_rbf_matches_closed_form_scalar_pair():
    x1 = jnp.array([[0.0]])
    x2 = jnp.array([[1.5]])
    var = jnp.array(0.7)
    ls = jnp.array(0.4)
    expected = var * jnp.exp(-0.5 * (1.5 / ls) ** 2)
    K = rbf_kernel(x1, x2, var, ls)
    assert jnp.allclose(K, expected)


def test_rbf_is_symmetric():
    X = jax.random.normal(jax.random.PRNGKey(0), (5, 3))
    K = rbf_kernel(X, X, jnp.array(1.0), jnp.array(0.7))
    assert jnp.allclose(K, K.T)


def test_rbf_gram_is_psd():
    """Min eigenvalue should be nonnegative up to float32 roundoff."""
    X = jax.random.normal(jax.random.PRNGKey(1), (8, 2))
    K = rbf_kernel(X, X, jnp.array(1.3), jnp.array(0.5))
    eigs = jnp.linalg.eigvalsh(K + 1e-8 * jnp.eye(8))
    assert float(eigs.min()) > -1e-6


# --- matern ---------------------------------------------------------------


@pytest.mark.parametrize("nu", [0.5, 1.5, 2.5])
def test_matern_diagonal_equals_variance(nu):
    X = jnp.array([[0.0], [1.0], [2.0]])
    K = matern_kernel(X, X, jnp.array(1.7), jnp.array(0.9), nu)
    assert jnp.allclose(jnp.diag(K), 1.7, atol=1e-5)


def test_matern_half_matches_exponential():
    x1 = jnp.array([[0.0]])
    x2 = jnp.array([[1.2]])
    var = jnp.array(0.4)
    ls = jnp.array(0.6)
    K = matern_kernel(x1, x2, var, ls, 0.5)
    expected = var * jnp.exp(-1.2 / ls)
    assert jnp.allclose(K, expected)


def test_matern_three_halves_matches_closed_form():
    x1 = jnp.array([[0.0]])
    x2 = jnp.array([[0.8]])
    var = jnp.array(1.1)
    ls = jnp.array(0.5)
    a = jnp.sqrt(3.0) * 0.8 / ls
    expected = var * (1.0 + a) * jnp.exp(-a)
    K = matern_kernel(x1, x2, var, ls, 1.5)
    assert jnp.allclose(K, expected)


def test_matern_five_halves_matches_closed_form():
    x1 = jnp.array([[0.0]])
    x2 = jnp.array([[0.7]])
    var = jnp.array(0.9)
    ls = jnp.array(0.4)
    a = jnp.sqrt(5.0) * 0.7 / ls
    expected = var * (1.0 + a + (a * a) / 3.0) * jnp.exp(-a)
    K = matern_kernel(x1, x2, var, ls, 2.5)
    assert jnp.allclose(K, expected)


def test_matern_unsupported_nu_raises():
    X = jnp.zeros((1, 1))
    with pytest.raises(ValueError, match="nu"):
        matern_kernel(X, X, jnp.array(1.0), jnp.array(1.0), 1.0)


def test_matern_grad_is_finite_at_zero_distance():
    """Sqrt clipping must keep grad finite when X1 == X2."""

    def loss(ls):
        X = jnp.array([[0.5], [0.5]])
        K = matern_kernel(X, X, jnp.array(1.0), ls, 1.5)
        return jnp.sum(K)

    g = jax.grad(loss)(jnp.array(0.7))
    assert jnp.isfinite(g)


# --- periodic --------------------------------------------------------------


def test_periodic_repeats_with_period():
    """k(0, p) should equal k(0, 0) since the period brings us back."""
    var = jnp.array(1.3)
    ls = jnp.array(0.5)
    period = jnp.array(2.0)
    x1 = jnp.array([[0.0]])
    x2_zero = jnp.array([[0.0]])
    x2_period = jnp.array([[2.0]])  # one full period away
    k_zero = periodic_kernel(x1, x2_zero, var, ls, period)
    k_period = periodic_kernel(x1, x2_period, var, ls, period)
    assert jnp.allclose(k_zero, k_period, atol=1e-5)


def test_periodic_diagonal_equals_variance():
    X = jnp.array([[0.0], [1.0], [2.0]])
    K = periodic_kernel(X, X, jnp.array(2.0), jnp.array(0.7), jnp.array(1.5))
    assert jnp.allclose(jnp.diag(K), 2.0)


def test_periodic_grad_is_finite_at_zero_distance():
    """Sqrt clipping must keep grad finite when X1 == X2."""

    def loss(ls):
        X = jnp.array([[0.5], [0.5]])
        K = periodic_kernel(X, X, jnp.array(1.0), ls, jnp.array(1.0))
        return jnp.sum(K)

    g = jax.grad(loss)(jnp.array(0.7))
    assert jnp.isfinite(g)


# --- linear ----------------------------------------------------------------


def test_linear_matches_dot_product():
    X1 = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    X2 = jnp.array([[0.5, 0.5]])
    var = jnp.array(2.0)
    bias = jnp.array(0.3)
    K = linear_kernel(X1, X2, var, bias)
    expected = var * (X1 @ X2.T) + bias
    assert jnp.allclose(K, expected)


# --- rational quadratic ----------------------------------------------------


def test_rational_quadratic_large_alpha_matches_rbf():
    """As alpha -> infty, RQ converges to RBF. Moderate alpha + bounded distances
    keep the check inside float32 precision."""
    X = jnp.array([[0.0], [0.1], [0.2], [0.3]])
    var = jnp.array(1.0)
    ls = jnp.array(1.0)
    K_rq = rational_quadratic_kernel(X, X, var, ls, jnp.array(1e4))
    K_rbf = rbf_kernel(X, X, var, ls)
    assert jnp.allclose(K_rq, K_rbf, atol=1e-3)


def test_rational_quadratic_diagonal_equals_variance():
    X = jnp.array([[0.0], [1.0], [2.0]])
    K = rational_quadratic_kernel(X, X, jnp.array(1.5), jnp.array(0.5), jnp.array(2.0))
    assert jnp.allclose(jnp.diag(K), 1.5)


# --- polynomial ------------------------------------------------------------


def test_polynomial_degree_one_matches_shifted_dot_product():
    X1 = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    X2 = jnp.array([[0.5, 0.5]])
    var = jnp.array(2.0)
    bias = jnp.array(0.3)
    K = polynomial_kernel(X1, X2, var, bias, 1)
    expected = var * (X1 @ X2.T + bias)
    assert jnp.allclose(K, expected)


def test_polynomial_degree_two_matches_closed_form():
    X = jnp.array([[1.0], [2.0]])
    var = jnp.array(1.0)
    bias = jnp.array(0.5)
    K = polynomial_kernel(X, X, var, bias, 2)
    expected = (X @ X.T + bias) ** 2
    assert jnp.allclose(K, expected)


def test_polynomial_rejects_degree_zero():
    X = jnp.zeros((2, 1))
    with pytest.raises(ValueError, match="degree"):
        polynomial_kernel(X, X, jnp.array(1.0), jnp.array(0.0), 0)


# --- cosine ----------------------------------------------------------------


def test_cosine_equals_variance_at_zero_distance():
    X = jnp.array([[0.5], [1.5]])
    K = cosine_kernel(X, X, jnp.array(1.7), jnp.array(2.0))
    assert jnp.allclose(jnp.diag(K), 1.7)


def test_cosine_negates_at_half_period():
    x1 = jnp.array([[0.0]])
    x2 = jnp.array([[1.0]])  # half of period 2 -> cos(pi) = -1
    K = cosine_kernel(x1, x2, jnp.array(1.0), jnp.array(2.0))
    assert jnp.allclose(K, -1.0, atol=1e-5)


def test_cosine_grad_is_finite_at_zero_distance():
    """Sqrt clipping must keep grad finite when X1 == X2."""

    def loss(period):
        X = jnp.array([[0.5], [0.5]])
        K = cosine_kernel(X, X, jnp.array(1.0), period)
        return jnp.sum(K)

    g = jax.grad(loss)(jnp.array(1.0))
    assert jnp.isfinite(g)


# --- white -----------------------------------------------------------------


def test_white_is_diagonal_when_X1_is_X2():
    X = jnp.array([[0.0], [1.0], [2.0]])
    K = white_kernel(X, X, jnp.array(0.5))
    assert jnp.allclose(K, 0.5 * jnp.eye(3))


def test_white_is_zero_between_distinct_points():
    X1 = jnp.array([[0.0]])
    X2 = jnp.array([[1.0]])
    K = white_kernel(X1, X2, jnp.array(2.0))
    assert jnp.allclose(K, 0.0)


# --- constant --------------------------------------------------------------


def test_constant_is_uniform():
    X1 = jax.random.normal(jax.random.PRNGKey(0), (3, 2))
    X2 = jax.random.normal(jax.random.PRNGKey(1), (4, 2))
    K = constant_kernel(X1, X2, jnp.array(1.8))
    assert K.shape == (3, 4)
    assert jnp.allclose(K, 1.8)


# --- composition -----------------------------------------------------------


def test_kernel_add_is_pointwise_sum():
    X = jnp.array([[0.0], [1.0]])
    K1 = rbf_kernel(X, X, jnp.array(1.0), jnp.array(1.0))
    K2 = linear_kernel(X, X, jnp.array(0.5), jnp.array(0.0))
    assert jnp.allclose(kernel_add(K1, K2), K1 + K2)


def test_kernel_mul_is_pointwise_product():
    X = jnp.array([[0.0], [1.0]])
    K1 = rbf_kernel(X, X, jnp.array(1.0), jnp.array(1.0))
    K2 = periodic_kernel(X, X, jnp.array(1.0), jnp.array(0.5), jnp.array(1.0))
    assert jnp.allclose(kernel_mul(K1, K2), K1 * K2)


# --- jit / grad smoke ------------------------------------------------------


def test_rbf_jits_and_grads():
    X = jax.random.normal(jax.random.PRNGKey(2), (4, 2))

    @jax.jit
    def loss(ls):
        return jnp.sum(rbf_kernel(X, X, jnp.array(1.0), ls))

    g = jax.grad(loss)(jnp.array(0.7))
    assert jnp.isfinite(g)


# --- ARD (per-dimension) lengthscales --------------------------------------


def _sq_dist_reference(X1, X2, lengthscale):
    """Explicit ``(N1, N2, D)`` difference-tensor reference."""
    diff = (X1[:, None, :] - X2[None, :, :]) / lengthscale
    return jnp.sum(diff**2, axis=-1)


def _ard_kernel_cases():
    """The three ARD-capable kernels as ``(X1, X2, lengthscale) -> K``."""
    var = jnp.array(1.3)
    return [
        pytest.param(
            lambda X1, X2, ls: rbf_kernel(X1, X2, var, ls),
            lambda sq: var * jnp.exp(-0.5 * sq),
            id="rbf",
        ),
        pytest.param(
            lambda X1, X2, ls: matern_kernel(X1, X2, var, ls, 0.5),
            lambda sq: var * jnp.exp(-jnp.sqrt(sq)),
            id="matern12",
        ),
        pytest.param(
            lambda X1, X2, ls: matern_kernel(X1, X2, var, ls, 1.5),
            lambda sq: var * (1.0 + jnp.sqrt(3.0 * sq)) * jnp.exp(-jnp.sqrt(3.0 * sq)),
            id="matern32",
        ),
        pytest.param(
            lambda X1, X2, ls: matern_kernel(X1, X2, var, ls, 2.5),
            lambda sq: (
                var
                * (1.0 + jnp.sqrt(5.0 * sq) + 5.0 * sq / 3.0)
                * jnp.exp(-jnp.sqrt(5.0 * sq))
            ),
            id="matern52",
        ),
        pytest.param(
            lambda X1, X2, ls: rational_quadratic_kernel(
                X1, X2, var, ls, jnp.array(1.7)
            ),
            lambda sq: var * (1.0 + sq / (2.0 * 1.7)) ** (-1.7),
            id="rational_quadratic",
        ),
    ]


@pytest.mark.parametrize("kernel_fn,closed_form", _ard_kernel_cases())
def test_scalar_lengthscale_matches_difference_tensor_reference(kernel_fn, closed_form):
    """Back-compat guard: a non-unit *scalar* lengthscale must match a
    hand-written reference — this fails loudly if the lengthscale were
    applied twice (inside the distance primitive and again post hoc)."""
    X1 = jax.random.normal(jax.random.PRNGKey(0), (5, 3))
    X2 = jax.random.normal(jax.random.PRNGKey(1), (4, 3))
    ls = jnp.array(0.7)
    K = kernel_fn(X1, X2, ls)
    expected = closed_form(_sq_dist_reference(X1, X2, ls))
    assert jnp.allclose(K, expected, rtol=0.0, atol=1e-12)


@pytest.mark.parametrize("kernel_fn,closed_form", _ard_kernel_cases())
def test_ard_lengthscale_matches_difference_tensor_reference(kernel_fn, closed_form):
    X1 = jax.random.normal(jax.random.PRNGKey(0), (5, 3))
    X2 = jax.random.normal(jax.random.PRNGKey(1), (4, 3))
    ls = jnp.array([0.5, 1.0, 2.0])
    K = kernel_fn(X1, X2, ls)
    expected = closed_form(_sq_dist_reference(X1, X2, ls))
    assert jnp.allclose(K, expected, rtol=0.0, atol=1e-12)


@pytest.mark.parametrize("kernel_fn,closed_form", _ard_kernel_cases())
def test_ard_constant_vector_matches_scalar(kernel_fn, closed_form):
    """A constant ARD vector must agree with the scalar lengthscale.

    Not bitwise: the isotropic path expands on raw coordinates and divides
    the squared distance afterwards (keeping it bit-identical to the
    pre-ARD implementation, and exact-zero on self-distances), while the
    ARD path centres and scales the inputs first. The two round
    differently in the last few ulps by construction.
    """
    del closed_form
    X1 = jax.random.normal(jax.random.PRNGKey(0), (5, 3))
    X2 = jax.random.normal(jax.random.PRNGKey(1), (4, 3))
    K_scalar = kernel_fn(X1, X2, jnp.array(0.7))
    K_ard = kernel_fn(X1, X2, jnp.full((3,), 0.7))
    assert jnp.allclose(K_scalar, K_ard, rtol=0.0, atol=1e-12)


@pytest.mark.parametrize("kernel_fn,closed_form", _ard_kernel_cases())
def test_ard_grad_is_finite_on_self_gram(kernel_fn, closed_form):
    """Grad w.r.t. a ``(D,)`` lengthscale must stay finite on a self-Gram,
    where the diagonal has ``r = 0``."""
    del closed_form
    X = jax.random.normal(jax.random.PRNGKey(2), (6, 4))

    def loss(ls):
        return jnp.sum(kernel_fn(X, X, ls))

    g = jax.grad(loss)(jnp.full((4,), 0.7))
    assert g.shape == (4,)
    assert jnp.all(jnp.isfinite(g))


def test_ard_jit_vmap_over_per_latent_lengthscales():
    """Per-latent ARD: vmap a kernel over a ``(Q, D)`` lengthscale array."""
    X = jax.random.normal(jax.random.PRNGKey(3), (17, 4))
    ells = jax.random.uniform(jax.random.PRNGKey(4), (6, 4), minval=0.5, maxval=2.0)

    @jax.jit
    def grams(ells):
        return jax.vmap(lambda ls: rbf_kernel(X, X, jnp.array(1.0), ls))(ells)

    K = grams(ells)
    assert K.shape == (6, 17, 17)
    assert jnp.all(jnp.isfinite(K))


def test_ard_gram_never_builds_n1_n2_d_intermediate():
    """N = 2000, D = 50 ARD Gram stays O(N^2): every jaxpr intermediate is
    rank <= 2 (an ``(N, N, D)`` difference tensor would be 1.6 GB)."""
    X = jax.random.normal(jax.random.PRNGKey(5), (2000, 50))
    ls = jnp.linspace(0.5, 2.0, 50)
    jaxpr = jax.make_jaxpr(rbf_kernel)(X, X, jnp.array(1.0), ls)
    ranks = [len(v.aval.shape) for eqn in jaxpr.eqns for v in eqn.outvars]
    assert max(ranks) <= 2
    K = rbf_kernel(X, X, jnp.array(1.0), ls)
    assert K.shape == (2000, 2000)
    assert jnp.all(jnp.isfinite(K))


def test_periodic_and_cosine_bit_identical_to_unscaled_distance():
    """Periodic and Cosine are excluded from ARD: their Grams must stay
    bit-identical to the unscaled-distance formulation."""
    X1 = jax.random.normal(jax.random.PRNGKey(6), (5, 3))
    X2 = jax.random.normal(jax.random.PRNGKey(7), (4, 3))
    n1 = einx.dot("n1 d, n1 d -> n1", X1, X1)
    n2 = einx.dot("n2 d, n2 d -> n2", X2, X2)
    cross = einx.dot("n1 d, n2 d -> n1 n2", X1, X2)
    sq = jnp.clip(einx.add("n1, n2 -> n1 n2", n1, n2) - 2.0 * cross, min=0.0)
    r = jnp.sqrt(jnp.clip(sq, min=1e-30))
    var = jnp.array(1.3)
    ls = jnp.array(0.6)
    period = jnp.array(1.7)
    sinsq = jnp.sin(jnp.pi * r / period) ** 2
    expected_periodic = var * jnp.exp(-2.0 * sinsq / (ls * ls))
    expected_cosine = var * jnp.cos(2.0 * jnp.pi * r / period)
    K_periodic = periodic_kernel(X1, X2, var, ls, period)
    K_cosine = cosine_kernel(X1, X2, var, period)
    assert (K_periodic == expected_periodic).all()
    assert (K_cosine == expected_cosine).all()


def test_ard_kernel_is_symmetric_under_argument_swap():
    """``K(X1, X2) == K(X2, X1).T`` must hold for ARD. A data-dependent
    centring offset would break this, which is why the ARD path scales
    without centring."""
    X1 = jnp.asarray([[0.0], [1e8]], dtype=jnp.float32)
    X2 = jnp.asarray([[1.0]], dtype=jnp.float32)
    ell = jnp.full((1,), 1.0, dtype=jnp.float32)
    var = jnp.asarray(1.0, dtype=jnp.float32)
    a = rbf_kernel(X1, X2, var, ell)
    b = rbf_kernel(X2, X1, var, ell)
    assert jnp.array_equal(a, b.T)
    # The nearby pair is genuinely separated by 1, not collapsed to 0.
    assert float(a[0, 0]) == pytest.approx(float(jnp.exp(-0.5)), rel=1e-5)


def test_ard_rejects_mismatched_feature_dimensions():
    """A singleton feature axis would broadcast against a ``(D,)``
    lengthscale and silently repeat one coordinate across every dimension."""
    X_train = jr.normal(jr.PRNGKey(0), (5, 3))
    X_pred = jr.normal(jr.PRNGKey(1), (4, 1))
    ell = jnp.full((3,), 0.7)
    with pytest.raises(ValueError, match="many features"):
        rbf_kernel(X_pred, X_train, jnp.asarray(1.0), ell)
    with pytest.raises(ValueError, match="many features"):
        rbf_kernel(X_train, X_pred, jnp.asarray(1.0), ell)


def test_ard_gram_is_translation_invariant():
    """The shared centring offset must not change any Gram value."""
    X = jr.normal(jr.PRNGKey(0), (6, 3))
    ell = jnp.asarray([0.5, 1.0, 2.0])
    base = rbf_kernel(X, X, jnp.asarray(1.0), ell)
    shifted = rbf_kernel(X + 17.0, X + 17.0, jnp.asarray(1.0), ell)
    assert jnp.allclose(base, shifted, atol=1e-10)
