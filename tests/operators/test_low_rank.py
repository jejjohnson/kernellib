"""Tests for the low-rank kernel operators: Nystrom and RFF.

Moved from gaussx ``tests/kernels/test_kernel_approx.py``; the centering, HSIC
and MMD tests from that file move with the statistics module.
"""

import jax
import jax.numpy as jnp
import lineax as lx

from kernellib import nystrom_operator, rff_operator


def _rbf_kernel(x, y, lengthscale=1.0):
    diff = x - y
    return jnp.exp(-0.5 * jnp.sum(diff**2) / lengthscale**2)


class TestNystromOperator:
    def test_shape(self, getkey):
        """Output should be N x N."""
        N, M = 10, 3
        K_XZ = jax.random.normal(getkey(), (N, M))
        K_ZZ = jax.random.normal(getkey(), (M, M))
        K_ZZ = K_ZZ @ K_ZZ.T + 0.1 * jnp.eye(M)
        K_ZZ_op = lx.MatrixLinearOperator(K_ZZ, lx.positive_semidefinite_tag)

        op = nystrom_operator(K_XZ, K_ZZ_op)
        mat = op.as_matrix()
        assert mat.shape == (N, N)

    def test_approximation(self, getkey):
        """Nystrom should approximate K_XZ K_ZZ^{-1} K_ZX."""
        N, M, D = 8, 3, 2
        X = jax.random.normal(getkey(), (N, D))
        Z = jax.random.normal(getkey(), (M, D))

        K_XZ = jax.vmap(lambda x: jax.vmap(lambda z: _rbf_kernel(x, z))(Z))(X)
        K_ZZ = jax.vmap(lambda z1: jax.vmap(lambda z2: _rbf_kernel(z1, z2))(Z))(Z)
        K_ZZ = K_ZZ + 0.01 * jnp.eye(M)
        K_ZZ_op = lx.MatrixLinearOperator(K_ZZ, lx.positive_semidefinite_tag)

        op = nystrom_operator(K_XZ, K_ZZ_op)
        approx = op.as_matrix()

        # Reference
        ref = K_XZ @ jnp.linalg.solve(K_ZZ, K_XZ.T)
        assert jnp.allclose(approx, ref, atol=1e-4)

    def test_symmetric_psd(self, getkey):
        """Result should be symmetric and PSD."""
        N, M = 6, 3
        K_XZ = jax.random.normal(getkey(), (N, M))
        K_ZZ = jax.random.normal(getkey(), (M, M))
        K_ZZ = K_ZZ @ K_ZZ.T + 0.1 * jnp.eye(M)
        K_ZZ_op = lx.MatrixLinearOperator(K_ZZ, lx.positive_semidefinite_tag)

        op = nystrom_operator(K_XZ, K_ZZ_op)
        assert lx.is_symmetric(op)
        assert lx.is_positive_semidefinite(op)


class TestRFFOperator:
    def test_shape(self, getkey):
        """Output should be N x N."""
        N, D, D_rff = 10, 3, 20
        X = jax.random.normal(getkey(), (N, D))
        omega = jax.random.normal(getkey(), (D_rff, D))
        b = jax.random.uniform(getkey(), (D_rff,), maxval=2 * jnp.pi)

        op = rff_operator(X, omega, b)
        mat = op.as_matrix()
        assert mat.shape == (N, N)

    def test_symmetric_psd(self, getkey):
        """Result should be symmetric and PSD."""
        N, D, D_rff = 8, 2, 15
        X = jax.random.normal(getkey(), (N, D))
        omega = jax.random.normal(getkey(), (D_rff, D))
        b = jax.random.uniform(getkey(), (D_rff,), maxval=2 * jnp.pi)

        op = rff_operator(X, omega, b)
        mat = op.as_matrix()
        assert jnp.allclose(mat, mat.T, atol=1e-6)
        eigvals = jnp.linalg.eigvalsh(mat)
        assert jnp.all(eigvals >= -1e-6)
