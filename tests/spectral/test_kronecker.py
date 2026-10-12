"""KroneckerFeatures: separable space x time feature products (GEO24)."""

from __future__ import annotations

import einx
import equinox as eqx
import gaussx as gx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import kernellib as kl
from kernellib._einx import einsum, rearrange


NS, NT = 7, 6


def _grid(ns: int = NS, nt: int = NT):
    Xs = einx.id("n -> n 1", jnp.linspace(-1.0, 1.0, ns))
    Xt = einx.id("n -> n 1", jnp.linspace(-1.0, 1.0, nt) ** 3)
    # Site-major (site, time) pairs.
    X = jnp.stack([jnp.repeat(Xs[:, 0], nt), jnp.tile(Xt[:, 0], ns)], axis=1)
    return Xs, Xt, X


def _kernel(ls_s=0.6, ls_t=0.8, var_s=1.3, var_t=0.9):
    return kl.Product(
        kl.ActiveDims(kl.RBF(lengthscale=ls_s, variance=var_s), dims=(0,)),
        kl.ActiveDims(kl.Matern(nu=1.5, lengthscale=ls_t, variance=var_t), dims=(1,)),
    )


def _laplace_kf(m: int = 20):
    lap = kl.LaplaceEigenfunctionFeatures(m, boundary_factor=3.0)
    return kl.KroneckerFeatures(lap, lap, space_dims=1)


def _gram(Phi):
    return einsum(Phi, Phi, "n f, m f -> n m")


def test_gram_is_product_of_factor_grams():
    _, _, X = _grid()
    k = _kernel()
    key_s = jax.random.key(0)
    rff = kl.RandomFourierFeatures(64, key_s)
    lap = kl.LaplaceEigenfunctionFeatures(12, boundary_factor=3.0)
    kf = kl.KroneckerFeatures(rff, lap, space_dims=1).fit(k, X)
    Phi = kf(X)
    assert Phi.shape == (NS * NT, 128 * 12)
    Gs = _gram(kf.space_map(X[:, :1]))
    Gt = _gram(kf.time_map(X[:, 1:]))
    np.testing.assert_allclose(_gram(Phi), Gs * Gt, rtol=0, atol=1e-12)


def test_grid_matches_features_on_meshgrid_and_dense():
    Xs, Xt, X = _grid()
    kf = _laplace_kf().fit(_kernel(), X)
    op = kf.features_grid(Xs, Xt)
    assert isinstance(op, gx.Kronecker)
    Phi = kf(X)
    np.testing.assert_allclose(op.as_matrix(), Phi, rtol=0, atol=1e-13)
    # Row-major vec: (Phi_s kron Phi_t) vec(W) = vec(Phi_s W Phi_t^T).
    W = jax.random.normal(jax.random.key(1), (20, 20))
    F = einsum(kf.space_map(Xs), W, kf.time_map(Xt), "n r, r s, m s -> n m")
    np.testing.assert_allclose(
        op.mv(rearrange(W, "r s -> (r s)")),
        rearrange(F, "n m -> (n m)"),
        rtol=0,
        atol=1e-12,
    )


def test_grid_ridge_solve_matches_dense():
    Xs, Xt, X = _grid()
    kf = _laplace_kf(8).fit(_kernel(), X)
    Fs, Ft = kf.space_map(Xs), kf.time_map(Xt)
    Y = jax.random.normal(jax.random.key(2), (NS, NT))
    lam = 0.3
    # Dense reference: (Phi^T Phi + lam I)^{-1} Phi^T y.
    Phi = kf(X)
    y = rearrange(Y, "n m -> (n m)")
    w_dense = jnp.linalg.solve(
        einsum(Phi, Phi, "n f, n g -> f g") + lam * jnp.eye(Phi.shape[1]),
        einsum(Phi, y, "n f, n -> f"),
    )
    # Grid path: eigendecompose the two small Grams only.
    ds, Vs = jnp.linalg.eigh(einsum(Fs, Fs, "n r, n s -> r s"))
    dt, Vt = jnp.linalg.eigh(einsum(Ft, Ft, "n r, n s -> r s"))
    B = einsum(Fs, Y, Ft, "n r, n m, m s -> r s")
    B = einsum(Vs, B, Vt, "r a, r s, s b -> a b")
    B = B / (einx.multiply("a, b -> a b", ds, dt) + lam)
    W = einsum(Vs, B, Vt, "r a, a b, s b -> r s")
    np.testing.assert_allclose(rearrange(W, "r s -> (r s)"), w_dense, atol=1e-9)
    # Same eigenstructure through gaussx's Kronecker eig / matvec.
    G = gx.Kronecker(
        lx_matrix(einsum(Fs, Fs, "n r, n s -> r s")),
        lx_matrix(einsum(Ft, Ft, "n r, n s -> r s")),
    )
    vals = gx.eigvals(G)
    np.testing.assert_allclose(
        jnp.sort(vals), jnp.sort(einx.multiply("a, b -> (a b)", ds, dt)), atol=1e-10
    )
    np.testing.assert_allclose(
        einsum(Phi, Phi, "n f, n g -> f g"), G.as_matrix(), atol=1e-10
    )


def lx_matrix(M):
    import lineax as lx

    return lx.MatrixLinearOperator(M, lx.positive_semidefinite_tag)


def test_fit_validation():
    _, _, X = _grid()
    kf = _laplace_kf()
    with pytest.raises(TypeError, match="Product"):
        kf.fit(kl.RBF(), X)
    with pytest.raises(TypeError, match="Product"):
        kf.fit(kl.Product(kl.RBF(), kl.RBF()), X)
    ad = kl.ActiveDims
    with pytest.raises(ValueError, match="ActiveDims must be"):
        kf.fit(kl.Product(ad(kl.RBF(), dims=(1,)), ad(kl.RBF(), dims=(0,))), X)
    with pytest.raises(ValueError, match="space_dims"):
        kf.fit(_kernel(), X[:, :1])
    with pytest.raises(RuntimeError, match="not fitted"):
        kf(X)
    with pytest.raises(RuntimeError, match="not fitted"):
        kf.features_grid(X[:, :1], X[:, 1:])
    assert not kf.is_fitted
    assert kf.fit(_kernel(), X).is_fitted


def test_gradients_reach_both_factors():
    _, _, X = _grid()

    def loss(hyp):
        k = _kernel(*hyp)
        kf = _laplace_kf(10).fit(k, X)
        return jnp.sum(kf(X) ** 2)

    g = jax.grad(loss)(jnp.array([0.6, 0.8, 1.3, 0.9]))
    assert bool(jnp.all(jnp.isfinite(g)))
    assert bool(jnp.all(jnp.abs(g) > 0))

    # Through a fitted module with eqx.filter_grad: both maps' kernels.
    kf = _laplace_kf(10).fit(_kernel(), X)
    grads = eqx.filter_grad(lambda m: jnp.sum(m(X) ** 2))(kf)
    assert bool(jnp.all(jnp.isfinite(grads.space_map.kernel.lengthscale)))
    assert float(jnp.abs(grads.time_map.kernel.lengthscale)) > 0


def test_jit():
    Xs, Xt, X = _grid()
    kf = _laplace_kf(10).fit(_kernel(), X)
    np.testing.assert_allclose(eqx.filter_jit(kf)(X), kf(X), atol=1e-12)
    grid = eqx.filter_jit(lambda m: m.features_grid(Xs, Xt).as_matrix())(kf)
    np.testing.assert_allclose(grid, kf(X), atol=1e-12)


@pytest.mark.slow
@pytest.mark.parametrize("seed", [0])
def test_converges_to_separable_kernel_as_rank_grows(seed):
    # Deterministic factors: Laplace features in time and space, error is the
    # factors' truncation error (the Matern-3/2 time factor is the slow one);
    # it must shrink monotonically with rank and end below 1e-2.
    _, _, X = _grid(15, 12)
    k = _kernel()
    errs = []
    for m in (4, 8, 16, 32):
        kf = _laplace_kf(m).fit(k, X)
        errs.append(float(jnp.max(jnp.abs(_gram(kf(X)) - k(X, X)))))
    assert errs[0] > errs[1] > errs[2] > errs[3]
    assert errs[-1] < 1e-2

    # Random space factor: Hoeffding. Each RFF Gram entry is a mean of F terms
    # bounded in [-v, v] (v = variance), so |err| <= v sqrt(2 ln(2/delta)/F)
    # with prob >= 1 - delta; the product multiplies by the (accurate) time
    # factor with sup <= var_t. Union bound over the N^2 entries, delta=1e-3/N^2.
    key = jax.random.key(seed)
    n_feat = 4096
    kf = kl.KroneckerFeatures(
        kl.RandomFourierFeatures(n_feat, key),
        kl.LaplaceEigenfunctionFeatures(32, boundary_factor=3.0),
        space_dims=1,
    ).fit(k, X)
    n = X.shape[0]
    delta = 1e-3 / n**2
    bound = 2 * 1.3 * 0.9 * np.sqrt(2 * np.log(2 / delta) / n_feat) + 1e-2
    err = float(jnp.max(jnp.abs(_gram(kf(X)) - k(X, X))))
    assert err < bound
