"""Tests for `stable_rbf_kernel`, moved from gaussx
``tests/linalg/test_mixed_precision.py``.

The `stable_squared_distances` tests stay in gaussx with the function.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr

from kernellib._testing import tree_allclose
from kernellib.functional import stable_rbf_kernel


# ---------------------------------------------------------------------------
# stable_rbf_kernel
# ---------------------------------------------------------------------------


class TestStableRBFKernel:
    def test_matches_naive(self, getkey):
        """Should match naive RBF computation in float64."""
        X = jr.normal(getkey(), (6, 3)).astype(jnp.float64)
        Z = jr.normal(getkey(), (4, 3)).astype(jnp.float64)
        ls = 1.5
        var = 2.0
        dist_sq = jnp.sum((X[:, None, :] - Z[None, :, :]) ** 2, axis=-1)
        expected = var * jnp.exp(-0.5 * dist_sq / ls**2)
        result = stable_rbf_kernel(
            X,
            Z,
            ls,
            var,
            compute_dtype=jnp.float64,
            accumulate_dtype=jnp.float64,
        )
        assert tree_allclose(result, expected, rtol=1e-10)

    def test_output_shape(self, getkey):
        X = jr.normal(getkey(), (10, 3))
        Z = jr.normal(getkey(), (6, 3))
        K = stable_rbf_kernel(X, Z, lengthscale=1.0)
        assert K.shape == (10, 6)

    def test_psd_high_dim(self, getkey):
        """RBF kernel eigenvalues should all be >= 0 for D=500."""
        D = 500
        X = jr.normal(getkey(), (30, D)).astype(jnp.float32)
        K = stable_rbf_kernel(X, X, lengthscale=1.0)
        eigvals = jnp.linalg.eigvalsh(K.astype(jnp.float64))
        assert jnp.all(eigvals > -1e-5)

    def test_cholesky_float32_high_dim(self, getkey):
        """Cholesky should succeed in float32 for D=500."""
        D = 500
        X = jr.normal(getkey(), (20, D)).astype(jnp.float32)
        K = stable_rbf_kernel(X, X, lengthscale=1.0)
        K = K + 1e-4 * jnp.eye(20)  # small jitter for numerics
        L = jnp.linalg.cholesky(K)
        assert jnp.all(jnp.isfinite(L))

    def test_grad_lengthscale(self, getkey):
        """Gradient should flow through lengthscale."""
        X = jr.normal(getkey(), (6, 3))
        Z = jr.normal(getkey(), (4, 3))

        def loss(ls):
            K = stable_rbf_kernel(X, Z, lengthscale=ls)
            return jnp.sum(K)

        g = jax.grad(loss)(jnp.array(1.0))
        assert jnp.isfinite(g)
        assert g != 0.0
