"""FeatureKernel and Modulated (#57)."""

from __future__ import annotations

import equinox as eqx
import gaussx as gx
import jax
import jax.numpy as jnp
import pytest

import kernellib as kl


X = jax.random.normal(jax.random.key(0), (25, 3))


@pytest.mark.slow
def test_identity_features_are_the_linear_kernel():
    k = kl.FeatureKernel(lambda x: x)
    assert jnp.allclose(k(X, X), kl.Linear()(X, X))
    assert jnp.allclose(k.diag(X), kl.Linear().diag(X))
    assert jnp.allclose(k.pairwise(X[1], X[4]), X[1] @ X[4])


def test_scalar_features_are_an_outer_product():
    f = lambda x: jnp.sin(x[0]) + x[1]
    fx = jax.vmap(f)(X)
    assert jnp.allclose(kl.FeatureKernel(f)(X, X), jnp.outer(fx, fx))


@pytest.mark.slow
@pytest.mark.parametrize(
    "make",
    [
        lambda: kl.RandomFourierFeatures(64, jax.random.key(1)),
        lambda: kl.NystromFeatures(10, jax.random.key(1)),
    ],
    ids=["rff", "nystrom"],
)
def test_a_fitted_feature_map_is_its_approximation_as_a_kernel(make):
    fmap = make().fit(kl.Matern(nu=1.5), X)
    k = kl.FeatureKernel(fmap)
    Phi = fmap(X)
    assert jnp.allclose(k(X, X), Phi @ Phi.T, atol=1e-10)
    assert jnp.allclose(k.pairwise(X[0], X[3]), Phi[0] @ Phi[3], atol=1e-10)


@pytest.mark.slow
def test_feature_kernel_stays_low_rank_in_to_operator():
    k = kl.FeatureKernel(lambda x: jnp.concatenate([x, x**2])) + kl.White(0.1)
    op = kl.to_operator(k, X)
    assert isinstance(op, gx.LowRankUpdate)
    assert op.U.shape == (25, 6)
    assert jnp.allclose(op.as_matrix(), k(X, X), atol=1e-12)


@pytest.mark.slow
def test_modulated_is_a_congruence():
    a = lambda x: 1.0 + x[0] ** 2
    k = kl.Modulated(kl.RBF(lengthscale=0.7), amplitude=a)
    ax = jax.vmap(a)(X)
    K = k(X, X)
    assert jnp.allclose(K, ax[:, None] * kl.RBF(lengthscale=0.7)(X, X) * ax[None, :])
    assert jnp.allclose(k.diag(X), ax**2)
    assert jnp.allclose(K[2, 7], k.pairwise(X[2], X[7]))
    assert jnp.linalg.eigvalsh(K).min() > -1e-10
    assert k.is_pointwise


def test_constant_amplitude_is_scaling():
    k = kl.Modulated(kl.Matern(), amplitude=lambda x: 3.0)
    assert jnp.allclose(k(X, X), 9.0 * kl.Matern()(X, X))


def test_modulated_keeps_structure():
    k = kl.Modulated(kl.Linear() + kl.White(0.2), amplitude=lambda x: jnp.exp(x[0]))
    op = kl.to_operator(k, X)
    assert isinstance(op, gx.LowRankUpdate)
    assert jnp.allclose(op.as_matrix(), k(X, X), atol=1e-10)


@pytest.mark.slow
def test_gradients_reach_neural_features_and_amplitudes():
    mlp = eqx.nn.MLP(3, 4, width_size=8, depth=1, key=jax.random.key(2))
    amp = eqx.nn.MLP(3, "scalar", width_size=8, depth=1, key=jax.random.key(3))
    kernel = kl.Modulated(kl.FeatureKernel(mlp) + kl.RBF(), amplitude=amp)

    @eqx.filter_jit
    @eqx.filter_grad
    def grad(k):
        op = kl.to_operator(k, X, noise=0.1)
        return gx.logdet(op)

    g = grad(kernel)
    leaves = jax.tree.leaves(eqx.filter(g, eqx.is_inexact_array))
    assert all(jnp.all(jnp.isfinite(leaf)) for leaf in leaves)
    assert jnp.any(g.amplitude.layers[0].weight != 0.0)
    assert jnp.any(g.kernel.kernels[0].features.layers[0].weight != 0.0)


# ---------------------------------------------------------------------------
# Nyström and residual kernels (#59)
# ---------------------------------------------------------------------------

Z = X[:6]


@pytest.mark.slow
def test_nystrom_kernel_matches_the_nystrom_operator():
    k = kl.RBF(lengthscale=0.8)
    kz = kl.nystrom_kernel(k, Z, jitter=1e-10)
    expected = kl.nystrom_operator(k(X, Z), kl.to_operator(k, Z)).as_matrix()
    assert jnp.allclose(kz(X, X), expected, atol=1e-6)


@pytest.mark.slow
def test_from_landmarks_equals_a_fit_with_those_landmarks():
    k = kl.Matern(nu=2.5)
    given = kl.NystromFeatures.from_landmarks(k, Z)
    fitted = kl.NystromFeatures(6, jax.random.key(0)).fit(k, X)
    fitted = eqx.tree_at(lambda m: m.landmarks, fitted, Z)
    assert jnp.allclose(given(X), fitted(X))


@pytest.mark.slow
def test_nystrom_residual_is_psd_and_vanishes_on_the_landmarks():
    k = kl.RBF(lengthscale=0.8)
    r = kl.Residual(k, kl.nystrom_kernel(k, Z))
    R = r(X, X)
    assert jnp.linalg.eigvalsh(R).min() > -1e-6
    assert jnp.all(jnp.abs(r.diag(Z)) < 1e-5)
    assert jnp.allclose(r.diag(X), jnp.diag(R), atol=1e-10)
    assert jnp.allclose(R[3, 9], r.pairwise(X[3], X[9]), atol=1e-10)


def test_residual_diag_never_forms_the_gram():
    n = 2048
    Xn = jax.random.normal(jax.random.key(5), (n, 2))
    k = kl.RBF()
    r = kl.Residual(k, kl.nystrom_kernel(k, Xn[:32]))
    assert f"{n},{n}" not in str(jax.make_jaxpr(r.diag)(Xn))


def test_nystrom_kernel_plus_noise_is_low_rank_with_dense_logdet():
    k = kl.nystrom_kernel(kl.RBF(lengthscale=0.8), Z) + kl.White(0.1)
    op = kl.to_operator(k, X)
    assert isinstance(op, gx.LowRankUpdate)
    dense = k(X, X)
    assert jnp.allclose(gx.logdet(op), jnp.linalg.slogdet(dense)[1], rtol=1e-8)


@pytest.mark.slow
def test_nystrom_kernel_hyperparameters_are_differentiable():
    def loss(ell):
        k = kl.nystrom_kernel(kl.RBF(lengthscale=ell), Z) + kl.White(0.1)
        return gx.logdet(kl.to_operator(k, X))

    g = jax.grad(loss)(jnp.array(0.8))
    assert jnp.isfinite(g) and g != 0.0
