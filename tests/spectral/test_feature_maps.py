"""Kernel-level feature maps: RFF, ORF, FastFood, Nyström."""

from __future__ import annotations

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
import pytest

import kernellib as kl


RANDOM_MAPS = [
    kl.RandomFourierFeatures,
    kl.OrthogonalRandomFeatures,
    kl.FastFoodFeatures,
]
KERNELS = [
    kl.RBF(lengthscale=jnp.array([0.7, 1.2, 0.5]), variance=1.3),
    kl.Matern(lengthscale=0.7, variance=1.3, nu=0.5),
    kl.RationalQuadratic(lengthscale=0.7, variance=1.3, alpha=0.8),
]
KERNEL_IDS = ["rbf-ard", "matern05", "rq"]


def _X(n=6, d=3, key=9):
    return jax.random.uniform(jax.random.key(key), (n, d), minval=-1.0, maxval=1.0)


def _gram_error_draws(cls, kernel, X, *, n_features, n_draws):
    def one(key):
        Phi = cls(n_features, key).fit(kernel, X)(X)
        return Phi @ Phi.T - kernel(X, X)

    return jnp.stack([one(jax.random.key(s)) for s in range(n_draws)])


@pytest.mark.slow
@pytest.mark.parametrize("kernel", KERNELS, ids=KERNEL_IDS)
@pytest.mark.parametrize("cls", RANDOM_MAPS, ids=lambda c: c.__name__)
def test_random_maps_are_unbiased(cls, kernel):
    # Each draw's Gram error has mean zero. Average it over independent keys
    # and bound every entry by 7 standard errors of that average, estimated
    # from the spread across draws.
    errs = _gram_error_draws(cls, kernel, _X(), n_features=256, n_draws=40)
    se = jnp.std(errs, axis=0) / jnp.sqrt(errs.shape[0])
    assert jnp.all(jnp.abs(jnp.mean(errs, axis=0)) <= 7 * se + 1e-12)


@pytest.mark.slow
def test_fastfood_rbf_row_lengths_would_bias_matern():
    # Guards the test above: keeping fastfood_params' chi row lengths (the
    # RBF law) for a Matern-1/2 kernel is far outside the bound.
    kernel = kl.Matern(lengthscale=0.7, variance=1.3, nu=0.5)
    X = _X()

    def one(key):
        k_ff, _ = jax.random.split(key)
        params = kl.fastfood_params(3, 256, kernel.lengthscale, k_ff)
        Phi = jnp.sqrt(kernel.variance) * kl.fastfood_features(X, params)
        return Phi @ Phi.T - kernel(X, X)

    errs = jnp.stack([one(jax.random.key(s)) for s in range(40)])
    se = jnp.std(errs, axis=0) / jnp.sqrt(errs.shape[0])
    assert jnp.any(jnp.abs(jnp.mean(errs, axis=0)) > 7 * se)


@pytest.mark.parametrize("cls", RANDOM_MAPS, ids=lambda c: c.__name__)
def test_random_maps_are_exact_on_the_diagonal(cls):
    # cos^2 + sin^2 = 1 per frequency, so phi(x).phi(x) = variance exactly.
    X = _X()
    Phi = cls(16, jax.random.key(0)).fit(KERNELS[0], X)(X)
    assert Phi.shape == (6, 32)
    assert jnp.allclose(jnp.sum(Phi**2, axis=1), 1.3)


def test_orf_blocks_are_orthogonal():
    X = _X(d=4)
    orf = kl.OrthogonalRandomFeatures(10, jax.random.key(0)).fit(kl.Matern(), X)
    assert orf.omega.shape == (10, 4)
    for block in (orf.omega[:4], orf.omega[4:8], orf.omega[8:]):
        G = block @ block.T
        assert jnp.allclose(G - jnp.diag(jnp.diag(G)), 0.0, atol=1e-10)


def test_fastfood_row_lengths_come_from_the_kernel():
    X = _X(d=5)
    ff = kl.FastFoodFeatures(20, jax.random.key(0)).fit(kl.Matern(nu=0.5), X)
    assert ff.params.S.shape == (ff.params.n_stacks, 8)
    assert jnp.all(ff.params.S > 0)


@pytest.mark.parametrize("cls", RANDOM_MAPS, ids=lambda c: c.__name__)
def test_hyperparameters_are_read_at_call_time(cls):
    # Swapping the kernel inside a fitted map equals refitting with the same
    # key: the draw is stored at unit lengthscale and unit variance.
    X = _X()
    a = kl.RBF(lengthscale=0.5, variance=2.0)
    b = kl.RBF(lengthscale=1.5, variance=0.3)
    fitted = cls(8, jax.random.key(0)).fit(a, X)
    swapped = eqx.tree_at(lambda m: m.kernel, fitted, b)
    refit = cls(8, jax.random.key(0)).fit(b, X)
    assert jnp.allclose(swapped(X), refit(X))


@pytest.mark.parametrize(
    "cls",
    [*RANDOM_MAPS, lambda n, key: kl.NystromFeatures(n, key)],
    ids=[c.__name__ for c in RANDOM_MAPS] + ["NystromFeatures"],
)
def test_jit_and_grad_through_a_fitted_map(cls):
    X = _X()
    fitted = cls(4, jax.random.key(0)).fit(kl.RBF(lengthscale=0.8), X)

    # filter_* skips the non-float leaves (the PRNG key, FastFood's
    # permutation indices).
    @eqx.filter_jit
    def loss(m):
        return jnp.sum(m(X))

    g = eqx.filter_grad(loss)(fitted)
    assert jnp.isfinite(g.kernel.lengthscale)
    assert jnp.allclose(loss(fitted), jnp.sum(fitted(X)))


def test_operator_is_low_rank_psd():
    X = _X()
    m = kl.RandomFourierFeatures(4, jax.random.key(0)).fit(kl.RBF(), X)
    op = m.operator(X)
    Phi = m(X)
    assert jnp.allclose(op.as_matrix(), Phi @ Phi.T)
    assert lx.is_symmetric(op)
    assert lx.is_positive_semidefinite(op)
    assert op.U.shape == (6, 8)


def test_nystrom_is_exact_with_every_point_a_landmark():
    X = _X(n=8)
    k = kl.Matern(nu=1.5, lengthscale=0.6)
    Phi = kl.NystromFeatures(8, jax.random.key(0), jitter=1e-10).fit(k, X)(X)
    assert jnp.allclose(Phi @ Phi.T, k(X, X), atol=1e-6)


def test_nystrom_is_exact_on_its_landmarks():
    X = _X(n=30)
    k = kl.RBF(lengthscale=0.4) + kl.Linear()
    nys = kl.NystromFeatures(5, jax.random.key(1), jitter=1e-10).fit(k, X)
    Z = nys.landmarks
    assert Z.shape == (5, 3)
    # Landmarks are distinct rows of X.
    assert len({tuple(z.tolist()) for z in Z}) == 5
    assert all(any(jnp.array_equal(z, x) for x in X) for z in Z)
    Phi_Z = nys(Z)
    assert jnp.allclose(Phi_Z @ Phi_Z.T, k(Z, Z), atol=1e-6)


def test_unfitted_map_raises():
    m = kl.RandomFourierFeatures(4, jax.random.key(0))
    assert not m.is_fitted
    with pytest.raises(RuntimeError, match="not fitted"):
        m(_X())
    assert m.fit(kl.RBF(), _X()).is_fitted


@pytest.mark.parametrize(
    "make",
    [
        lambda: kl.RandomFourierFeatures(0, jax.random.key(0)),
        lambda: kl.OrthogonalRandomFeatures(0, jax.random.key(0)),
        lambda: kl.FastFoodFeatures(0, jax.random.key(0)),
        lambda: kl.NystromFeatures(0, jax.random.key(0)),
    ],
)
def test_nonpositive_size_raises(make):
    with pytest.raises(ValueError, match=">= 1"):
        make()


def test_nystrom_config_errors():
    with pytest.raises(ValueError, match="uniform"):
        kl.NystromFeatures(2, jax.random.key(0), selection="leverage")
    with pytest.raises(ValueError, match="at least as many inputs"):
        kl.NystromFeatures(10, jax.random.key(0)).fit(kl.RBF(), _X(n=4))


@pytest.mark.parametrize("cls", RANDOM_MAPS, ids=lambda c: c.__name__)
@pytest.mark.parametrize("kernel", [kl.Periodic(), kl.Linear(), kl.RBF() * kl.RBF()])
def test_random_maps_need_a_spectral_kernel(cls, kernel):
    with pytest.raises(NotImplementedError, match="NystromFeatures"):
        cls(4, jax.random.key(0)).fit(kernel, _X())


@pytest.mark.parametrize("cls", RANDOM_MAPS, ids=lambda c: c.__name__)
def test_ard_mismatch_raises_at_fit(cls):
    with pytest.raises(ValueError, match="ARD lengthscale of size 2"):
        cls(4, jax.random.key(0)).fit(kl.RBF(lengthscale=jnp.ones(2)), _X())


@pytest.mark.parametrize("cls", RANDOM_MAPS, ids=lambda c: c.__name__)
def test_input_dimension_mismatch_raises_at_call(cls):
    m = cls(4, jax.random.key(0)).fit(kl.RBF(), _X(d=3))
    with pytest.raises(ValueError):
        m(_X(d=2))


def test_fit_returns_a_new_module():
    m = kl.RandomFourierFeatures(4, jax.random.key(0))
    fitted = m.fit(kl.RBF(), _X())
    assert m.omega is None
    assert dataclasses.replace(fitted, kernel=None, omega=None).omega is None
