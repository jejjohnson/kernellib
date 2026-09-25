"""Tests for the matrix-level kernel statistics.

The centering, HSIC (biased) and MMD tests are moved from gaussx
``tests/kernels/test_kernel_approx.py`` with only the import changed.
The unbiased-HSIC and CKA tests are new and use pinned keys: the inputs are
incidental, and the unbiased estimator is checked against an independent
brute-force U-statistic rather than against a sampling bound.
"""

import itertools

import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
import numpy as np
import pytest

from kernellib.functional import (
    center_kernel,
    centering_operator,
    cka,
    hsic,
    mmd_squared,
)


def _rbf_kernel(x, y, lengthscale=1.0):
    diff = x - y
    return jnp.exp(-0.5 * jnp.sum(diff**2) / lengthscale**2)


class TestCenteringOperator:
    def test_shape(self):
        """Should be n x n."""
        n = 5
        op = centering_operator(n)
        assert op.as_matrix().shape == (n, n)

    def test_idempotent(self):
        """H^2 = H (centering is a projection)."""
        n = 6
        H = centering_operator(n).as_matrix()
        assert jnp.allclose(H @ H, H, atol=1e-6)

    def test_centers_vector(self):
        """H @ x should have zero mean."""
        n = 5
        H = centering_operator(n).as_matrix()
        x = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        centered = H @ x
        assert jnp.allclose(jnp.mean(centered), 0.0, atol=1e-10)


class TestCenterKernel:
    def test_row_col_mean_zero(self, getkey):
        """Centered kernel should have zero row and column means."""
        N = 6
        K = jax.random.normal(getkey(), (N, N))
        K = K @ K.T
        K_op = lx.MatrixLinearOperator(K, lx.symmetric_tag)

        K_c = center_kernel(K_op).as_matrix()
        assert jnp.allclose(jnp.mean(K_c, axis=0), 0.0, atol=1e-10)
        assert jnp.allclose(jnp.mean(K_c, axis=1), 0.0, atol=1e-10)

    def test_symmetric(self, getkey):
        """Centered kernel should be symmetric if input is."""
        N = 5
        K = jax.random.normal(getkey(), (N, N))
        K = K @ K.T
        K_op = lx.MatrixLinearOperator(K, lx.symmetric_tag)

        K_c = center_kernel(K_op)
        assert lx.is_symmetric(K_c)


class TestHSIC:
    def test_independent_near_zero(self, getkey):
        """HSIC of independent features should be near zero."""
        N = 50
        X = jax.random.normal(getkey(), (N, 1))
        Y = jax.random.normal(getkey(), (N, 1))
        K_f = jax.vmap(lambda x: jax.vmap(lambda y: _rbf_kernel(x, y))(X))(X)
        K_q = jax.vmap(lambda x: jax.vmap(lambda y: _rbf_kernel(x, y))(Y))(Y)
        K_f_op = lx.MatrixLinearOperator(K_f, lx.symmetric_tag)
        K_q_op = lx.MatrixLinearOperator(K_q, lx.symmetric_tag)

        h = hsic(K_f_op, K_q_op)
        assert jnp.abs(h) < 0.1

    def test_self_hsic_positive(self, getkey):
        """HSIC(K, K) should be positive."""
        N = 20
        K = jax.random.normal(getkey(), (N, N))
        K = K @ K.T + 0.1 * jnp.eye(N)
        K_op = lx.MatrixLinearOperator(K, lx.symmetric_tag)

        h = hsic(K_op, K_op)
        assert h > 0

    def test_scalar(self, getkey):
        """Should return a scalar."""
        N = 10
        K = jnp.eye(N)
        K_op = lx.MatrixLinearOperator(K, lx.symmetric_tag)
        h = hsic(K_op, K_op)
        assert h.shape == ()


class TestMMDSquared:
    def test_same_distribution_zero(self, getkey):
        """MMD^2 of identical distributions should be near zero."""
        N = 20
        X = jax.random.normal(getkey(), (N, 2))
        K_xx = jax.vmap(lambda x: jax.vmap(lambda y: _rbf_kernel(x, y))(X))(X)
        m = mmd_squared(K_xx, K_xx, K_xx)
        assert jnp.allclose(m, 0.0, atol=1e-6)

    def test_different_distributions_positive(self, getkey):
        """MMD^2 of different distributions should be positive."""
        N = 20
        X = jax.random.normal(getkey(), (N, 2))
        Y = jax.random.normal(getkey(), (N, 2)) + 3.0
        K_xx = jax.vmap(lambda x: jax.vmap(lambda y: _rbf_kernel(x, y))(X))(X)
        K_yy = jax.vmap(lambda x: jax.vmap(lambda y: _rbf_kernel(x, y))(Y))(Y)
        K_xy = jax.vmap(lambda x: jax.vmap(lambda y: _rbf_kernel(x, y))(Y))(X)
        m = mmd_squared(K_xx, K_yy, K_xy)
        assert m > 0

    def test_scalar(self, getkey):
        """Should return a scalar."""
        N = 5
        K_xx = jnp.eye(N)
        K_yy = jnp.eye(N)
        K_xy = jnp.zeros((N, N))
        m = mmd_squared(K_xx, K_yy, K_xy)
        assert m.shape == ()


# ---------------------------------------------------------------------------
# New: unbiased HSIC and CKA
# ---------------------------------------------------------------------------


def _gram(key, n, d=2):
    X = jr.normal(key, (n, d))
    return jax.vmap(lambda x: jax.vmap(lambda y: _rbf_kernel(x, y))(X))(X)


def _op(K):
    return lx.MatrixLinearOperator(K, lx.symmetric_tag)


def _hsic_u_statistic(K, L):
    """Brute-force U-statistic over ordered distinct quadruples (i, j, q, r).

    Song et al. (2012), eq. 5: averaging the kernel
    ``k_ij l_ij + k_ij l_qr - 2 k_ij l_iq`` over ordered distinct index
    quadruples. Shares no code with the closed form under test.
    """
    K, L = np.asarray(K), np.asarray(L)
    n = K.shape[0]
    total = 0.0
    count = 0
    for i, j, q, r in itertools.permutations(range(n), 4):
        total += K[i, j] * L[i, j] + K[i, j] * L[q, r] - 2.0 * K[i, j] * L[i, q]
        count += 1
    return total / count


class TestUnbiasedHSIC:
    @pytest.mark.parametrize("n", [4, 5, 7])
    def test_matches_brute_force_u_statistic(self, n):
        k1, k2 = jr.split(jr.key(0))
        K, L = _gram(k1, n), _gram(k2, n)
        h = hsic(_op(K), _op(L), estimator="unbiased")
        assert jnp.allclose(h, _hsic_u_statistic(K, L), rtol=1e-10, atol=1e-12)

    def test_ignores_diagonal(self):
        """The estimator zeroes the diagonal, so changing it has no effect."""
        k1, k2 = jr.split(jr.key(2))
        K, L = _gram(k1, 8), _gram(k2, 8)
        h = hsic(_op(K), _op(L), estimator="unbiased")
        h_shifted = hsic(_op(K + 5.0 * jnp.eye(8)), _op(L), estimator="unbiased")
        assert jnp.allclose(h, h_shifted, rtol=1e-12)

    def test_rejects_n_below_four(self):
        K = _op(jnp.eye(3))
        with pytest.raises(ValueError, match="n >= 4"):
            hsic(K, K, estimator="unbiased")

    def test_rejects_unknown_estimator(self):
        K = _op(jnp.eye(5))
        with pytest.raises(ValueError, match="estimator"):
            hsic(K, K, estimator="fancy")

    def test_biased_default_unchanged(self):
        k1, k2 = jr.split(jr.key(3))
        K, L = _op(_gram(k1, 10)), _op(_gram(k2, 10))
        assert hsic(K, L) == hsic(K, L, estimator="biased")


class TestCKA:
    def test_self_alignment_is_one(self):
        K = _op(_gram(jr.key(4), 12))
        assert jnp.allclose(cka(K, K), 1.0, rtol=1e-12)
        assert jnp.allclose(cka(K, K, estimator="unbiased"), 1.0, rtol=1e-12)

    def test_scale_invariant(self):
        k1, k2 = jr.split(jr.key(5))
        K, L = _gram(k1, 12), _gram(k2, 12)
        base = cka(_op(K), _op(L))
        assert jnp.allclose(cka(_op(3.0 * K), _op(0.5 * L)), base, rtol=1e-12)

    def test_biased_is_in_unit_interval_for_psd_kernels(self):
        k1, k2 = jr.split(jr.key(6))
        c = cka(_op(_gram(k1, 15)), _op(_gram(k2, 15)))
        assert 0.0 <= float(c) <= 1.0

    def test_symmetric_in_arguments(self):
        k1, k2 = jr.split(jr.key(7))
        K, L = _op(_gram(k1, 10)), _op(_gram(k2, 10))
        assert jnp.allclose(cka(K, L), cka(L, K), rtol=1e-12)
