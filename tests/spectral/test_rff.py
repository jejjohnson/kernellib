"""Random Fourier feature prior draws, moved from ``pyrox_gp._basis._rff``."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from geonnax.randfeat import rff_cosine_forward

import kernellib as kl


def _paths(kernel, X, *, n_paths, n_features, key=0):
    v, ell, omega, phase, w = kl.draw_rff_cosine_basis(
        kernel,
        jax.random.key(key),
        n_paths=n_paths,
        n_features=n_features,
        in_features=X.shape[1],
    )
    return kl.evaluate_rff_cosine_paths(
        X, variance=v, lengthscale=ell, omega=omega, phase=phase, weights=w
    )


@pytest.mark.slow
@pytest.mark.parametrize(
    "kernel",
    [
        kl.RBF(lengthscale=0.4, variance=1.3),
        kl.Matern(lengthscale=0.5, nu=1.5),
        kl.Matern(lengthscale=jnp.array([0.5, 1.5]), nu=2.5),
        kl.RationalQuadratic(lengthscale=0.6, alpha=2.0),
    ],
    ids=["rbf", "matern15", "matern25-ard", "rq"],
)
def test_path_covariance_matches_kernel(kernel):
    # Each path draws its own frequencies, so across paths E[f(x) f(x')] is
    # the exact kernel. Bound each entry of the empirical covariance by 7
    # standard errors of the products f_s(x) f_s(x'), estimated from the draws.
    X = jax.random.uniform(jax.random.key(9), (5, 2), minval=-1.0, maxval=1.0)
    paths = _paths(kernel, X, n_paths=20_000, n_features=16)
    prods = paths[:, :, None] * paths[:, None, :]
    se = jnp.std(prods, axis=0) / jnp.sqrt(paths.shape[0])
    assert jnp.all(jnp.abs(jnp.mean(prods, axis=0) - kernel(X, X)) < 7 * se)


def test_shapes_and_dtype():
    v, ell, omega, phase, w = kl.draw_rff_cosine_basis(
        kl.RBF(lengthscale=jnp.ones(3)),
        jax.random.key(0),
        n_paths=2,
        n_features=8,
        in_features=3,
        dtype=jnp.float32,
    )
    assert v.shape == () and ell.shape == (3,)
    assert omega.shape == (2, 3, 8)
    assert phase.shape == w.shape == (2, 8)
    assert all(a.dtype == jnp.float32 for a in (v, ell, omega, phase, w))
    assert jnp.all((phase >= 0) & (phase <= 2 * jnp.pi))


def test_frequencies_are_unit_lengthscale():
    # The lengthscale is applied at evaluation, not baked into omega.
    a = kl.draw_rff_cosine_basis(
        kl.RBF(lengthscale=0.1),
        jax.random.key(0),
        n_paths=1,
        n_features=4,
        in_features=2,
    )[2]
    b = kl.draw_rff_cosine_basis(
        kl.RBF(lengthscale=5.0),
        jax.random.key(0),
        n_paths=1,
        n_features=4,
        in_features=2,
    )[2]
    assert jnp.array_equal(a, b)


def test_matches_geonnax_cosine_forward():
    # One path is sqrt(variance) * phi(x) . w with phi geonnax's single-cosine
    # map; the ARD lengthscale is folded into the frequency matrix.
    ell = jnp.array([0.5, 2.0])
    v, ell, omega, phase, w = kl.draw_rff_cosine_basis(
        kl.Matern(lengthscale=ell, variance=0.7, nu=1.5),
        jax.random.key(4),
        n_paths=1,
        n_features=6,
        in_features=2,
    )
    X = jax.random.normal(jax.random.key(5), (3, 2))
    got = kl.evaluate_rff_cosine_paths(
        X, variance=v, lengthscale=ell, omega=omega, phase=phase, weights=w
    )
    W = omega[0] / ell[:, None]
    phi = jax.vmap(lambda x: rff_cosine_forward(W, phase[0], 1.0, 6, x))(X)
    assert jnp.allclose(got[0], jnp.sqrt(v) * phi @ w[0])


@pytest.mark.parametrize("bad_field", ["n_paths", "n_features"])
def test_rejects_nonpositive_counts(bad_field):
    kwargs = {"n_paths": 2, "n_features": 4, "in_features": 1}
    kwargs[bad_field] = 0
    with pytest.raises(ValueError, match=bad_field):
        kl.draw_rff_cosine_basis(kl.RBF(), jax.random.key(0), **kwargs)


@pytest.mark.parametrize("kernel", [kl.Periodic(), kl.Linear(), kl.RBF() + kl.White()])
def test_rejects_non_stationary_kernels(kernel):
    with pytest.raises(NotImplementedError, match="stationary"):
        kl.draw_rff_cosine_basis(
            kernel, jax.random.key(0), n_paths=1, n_features=4, in_features=1
        )


def test_ard_mismatch_raises():
    v, ell, omega, phase, w = kl.draw_rff_cosine_basis(
        kl.RBF(lengthscale=jnp.ones(2)),
        jax.random.key(0),
        n_paths=1,
        n_features=4,
        in_features=2,
    )
    with pytest.raises(ValueError, match="ARD lengthscale of size 2"):
        kl.evaluate_rff_cosine_paths(
            jnp.zeros((3, 1)),
            variance=v,
            lengthscale=ell,
            omega=omega,
            phase=phase,
            weights=w,
        )


def test_evaluate_is_differentiable_in_hyperparameters():
    v, _, omega, phase, w = kl.draw_rff_cosine_basis(
        kl.RBF(), jax.random.key(0), n_paths=2, n_features=8, in_features=1
    )
    X = jnp.linspace(-1.0, 1.0, 4)[:, None]

    @jax.jit
    def loss(ell):
        return jnp.sum(
            kl.evaluate_rff_cosine_paths(
                X, variance=v, lengthscale=ell, omega=omega, phase=phase, weights=w
            )
            ** 2
        )

    assert jnp.isfinite(jax.grad(loss)(0.7))
