"""Hilbert-space features for 2-D vector fields (GEO23, #223).

Every column is exactly divergence- or curl-free (autodiff), the geonnax
``divfree_basis`` convention is pinned, ΦΦᵀ converges to the GEO15 kernels,
and gradients, ``jit`` and the error paths work.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from geonnax.basis import divfree_basis

import kernellib as kl
from kernellib._einx import einsum, rearrange


K_RBF = kl.RBF(lengthscale=0.5, variance=1.3)
R = jnp.array([[0.0, 1.0], [-1.0, 0.0]])


def _points(n, key=0, scale=1.0):
    return jax.random.uniform(jax.random.key(key), (n, 2), minval=-scale, maxval=scale)


def _jacobian(feat, X):
    # (N, 2, M, 2): d Phi[(n i), m] / d x_a.
    return jax.vmap(jax.jacfwd(lambda x: feat(x[None])))(X)


def _divergence(J):
    # sum_i d_i F_i for every column: contract the component and derivative axes.
    return einsum(J, jnp.eye(2), "n i m a, i a -> n m")


def _curl(J):
    # d_0 F_1 - d_1 F_0
    return J[:, 1, :, 0] - J[:, 0, :, 1]


# Exact invariants ---------------------------------------------------------


@pytest.mark.parametrize("boundary", ["dirichlet", "neumann"])
def test_divergence_free_columns_have_zero_divergence(boundary):
    X = _points(8)
    feat = kl.DivergenceFreeFeatures(7, L=2.5, boundary=boundary).fit(K_RBF, X)
    J = _jacobian(feat, X)
    assert jnp.max(jnp.abs(_divergence(J))) < 1e-10
    # Control: the curl of these columns is not zero.
    assert jnp.max(jnp.abs(_curl(J))) > 0.1


@pytest.mark.parametrize("boundary", ["dirichlet", "neumann"])
def test_curl_free_columns_have_zero_curl(boundary):
    X = _points(8, key=1)
    feat = kl.CurlFreeFeatures(7, L=2.5, boundary=boundary).fit(K_RBF, X)
    J = _jacobian(feat, X)
    assert jnp.max(jnp.abs(_curl(J))) < 1e-10
    assert jnp.max(jnp.abs(_divergence(J))) > 0.1


def test_helmholtz_is_the_concatenation_and_has_both_parts():
    X = _points(6, key=2)
    k2 = kl.RBF(lengthscale=0.8)
    cf, df = kl.CurlFreeFeatures(5, L=2.0), kl.DivergenceFreeFeatures(6, L=2.0)
    feat = kl.HelmholtzFeatures(cf, df).fit(K_RBF, X, stream_kernel=k2)
    Phi = feat(X)
    assert Phi.shape == (12, 25 + 36)
    assert jnp.allclose(Phi[:, :25], cf.fit(K_RBF, X)(X))
    assert jnp.allclose(Phi[:, 25:], df.fit(k2, X)(X))
    # Single kernel for both parts by default.
    same = kl.HelmholtzFeatures(cf, df).fit(K_RBF, X)
    assert jnp.allclose(same(X)[:, 25:], df.fit(K_RBF, X)(X))


# Conventions --------------------------------------------------------------


def test_row_order_is_point_major_and_matches_gradients():
    X = _points(5, key=3)
    feat = kl.CurlFreeFeatures(4, L=2.0).fit(K_RBF, X)
    Phi = feat(X)
    assert Phi.shape == (10, 16)
    G = (
        feat._gradients(X)
        * jnp.sqrt(K_RBF.spectral_density(feat.frequencies))[None, :, None]
    )
    assert jnp.allclose(Phi, rearrange(G, "n m i -> (n i) m"))
    # Row 2n + i is component i of location n: single-point calls agree.
    assert jnp.allclose(feat(X[3:4]), Phi[6:8])


def test_matches_geonnax_divfree_basis_up_to_sign():
    # geonnax's (u, v) = (-d_y psi, d_x psi) = -R grad(phi); ours is +R grad(phi).
    X = _points(7, key=4)
    feat = kl.DivergenceFreeFeatures((4, 5), L=(2.0, 3.0)).fit(K_RBF, X)
    ours = einsum(R, feat._gradients(X), "i a, n m a -> n m i")
    theirs, lam = divfree_basis(X, (4, 5), (2.0, 3.0))
    assert jnp.allclose(ours, -theirs, atol=1e-10)
    assert jnp.allclose(jnp.sum(feat.frequencies**2, axis=-1), lam)


def test_resolved_box_and_defaults():
    X = _points(10, key=5)
    feat = kl.DivergenceFreeFeatures(4, boundary_factor=2.0).fit(K_RBF, X)
    assert feat.half_widths == pytest.approx(
        tuple(float(v) for v in 2.0 * jnp.max(jnp.abs(X), axis=0))
    )
    assert feat.is_fitted


# Convergence to the GEO15 kernels ------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize("boundary", ["dirichlet", "neumann"])
@pytest.mark.parametrize(
    ("features", "kernel"),
    [
        (kl.DivergenceFreeFeatures, kl.DivergenceFree),
        (kl.CurlFreeFeatures, kl.CurlFree),
    ],
    ids=["divfree", "curlfree"],
)
def test_gram_converges_to_the_vector_kernel(features, kernel, boundary):
    X = _points(60, key=0)  # central quarter-ish of the box [-4, 4]^2
    Xv = kl.vector_inputs(X)
    target = kernel(K_RBF)(Xv, Xv)
    errs = []
    for m in (8, 16, 32):
        Phi = features(m, L=4.0, boundary=boundary).fit(K_RBF, X)(X)
        errs.append(float(jnp.max(jnp.abs(Phi @ Phi.T - target))))
    assert errs[0] > errs[1] > errs[2]
    # Measured once: 5.9e-8 (Dirichlet) and 2.0e-7 (Neumann) at m = 32, L = 4.
    assert errs[2] < 1e-6


@pytest.mark.slow
def test_helmholtz_gram_converges_to_the_helmholtz_kernel():
    X = _points(40, key=1)
    k1, k2 = kl.RBF(lengthscale=0.5, variance=0.3), kl.RBF(lengthscale=0.7)
    Xv = kl.vector_inputs(X)
    feat = kl.HelmholtzFeatures(
        kl.CurlFreeFeatures(32, L=4.0), kl.DivergenceFreeFeatures(32, L=4.0)
    ).fit(k1, X, stream_kernel=k2)
    Phi = feat(X)
    target = kl.Helmholtz(potential=k1, stream=k2)(Xv, Xv)
    assert jnp.max(jnp.abs(Phi @ Phi.T - target)) < 1e-5


def test_matern_gram_gets_closer_with_more_features():
    # Fast-tier sanity check for a non-RBF density (Matern 5/2).
    k = kl.Matern(nu=2.5, lengthscale=0.6)
    X = _points(20, key=2)
    Xv = kl.vector_inputs(X)
    target = kl.DivergenceFree(k)(Xv, Xv)
    errs = []
    for m in (8, 20):
        Phi = kl.DivergenceFreeFeatures(m, L=4.0).fit(k, X)(X)
        errs.append(float(jnp.max(jnp.abs(Phi @ Phi.T - target))))
    assert errs[1] < errs[0]


# Gradients, jit, operator ---------------------------------------------------


@pytest.mark.parametrize(
    "cls", [kl.DivergenceFreeFeatures, kl.CurlFreeFeatures], ids=["df", "cf"]
)
def test_gradients_reach_the_hyperparameters(cls):
    X = _points(10, key=6)
    feat = cls(6, L=3.0).fit(K_RBF, X)

    def loss(ell, var):
        f = feat.fit(kl.RBF(lengthscale=ell, variance=var), X)
        return jnp.sum(f(X) @ f(X).T)

    g_ell, g_var = jax.grad(loss, argnums=(0, 1))(0.5, 1.3)
    assert jnp.isfinite(g_ell) and jnp.abs(g_ell) > 0
    assert jnp.isfinite(g_var) and jnp.abs(g_var) > 0


def test_jit_matches_eager():
    X = _points(10, key=7)
    feat = kl.HelmholtzFeatures(
        kl.CurlFreeFeatures(5, L=3.0), kl.DivergenceFreeFeatures(5, L=3.0)
    ).fit(K_RBF, X)
    assert jnp.allclose(jax.jit(lambda f, x: f(x))(feat, X), feat(X))


def test_operator_is_the_low_rank_gram():
    X = _points(6, key=8)
    feat = kl.DivergenceFreeFeatures(5, L=3.0).fit(K_RBF, X)
    Phi = feat(X)
    assert jnp.allclose(feat.operator(X).as_matrix(), Phi @ Phi.T)


# Invalid inputs ---------------------------------------------------------------


def test_fit_rejects_non_2d_inputs():
    X3 = jax.random.uniform(jax.random.key(0), (5, 3))
    for cls in (kl.DivergenceFreeFeatures, kl.CurlFreeFeatures):
        with pytest.raises(ValueError, match="needs \\(N, 2\\) inputs"):
            cls(4).fit(K_RBF, X3)


def test_features_reject_non_2d_inputs():
    feat = kl.CurlFreeFeatures(4, L=2.0).fit(K_RBF, _points(5))
    with pytest.raises(ValueError, match="needs \\(N, 2\\) inputs"):
        feat(jnp.zeros((5, 3)))


def test_unfitted_map_raises():
    X = _points(5)
    with pytest.raises(RuntimeError, match="not fitted"):
        kl.DivergenceFreeFeatures(4)(X)
    unfitted = kl.HelmholtzFeatures(
        kl.CurlFreeFeatures(3), kl.DivergenceFreeFeatures(3)
    )
    with pytest.raises(RuntimeError, match="not fitted"):
        unfitted(X)


def test_kernel_without_spectral_density_raises():
    with pytest.raises(NotImplementedError, match="spectral density"):
        kl.DivergenceFreeFeatures(4).fit(kl.Linear(), _points(5))


def test_invalid_configuration_raises():
    with pytest.raises(ValueError, match="num_basis_per_dim"):
        kl.CurlFreeFeatures(0)
    with pytest.raises(ValueError, match="boundary_factor"):
        kl.CurlFreeFeatures(4, boundary_factor=1.0)
    with pytest.raises(ValueError, match="boundary must be"):
        kl.CurlFreeFeatures(4, boundary="periodic")
    with pytest.raises(ValueError, match="positive"):
        kl.CurlFreeFeatures(4, L=-1.0).fit(K_RBF, _points(5))
    with pytest.raises(ValueError, match="entries"):
        kl.CurlFreeFeatures(4, L=(1.0, 2.0, 3.0)).fit(K_RBF, _points(5))
