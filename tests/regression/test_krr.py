"""Kernel ridge regression."""

from __future__ import annotations

import equinox as eqx
import gaussx as gx
import jax
import jax.numpy as jnp
import pytest

import kernellib as kl


def _data(n=30, key=0):
    k1, k2 = jax.random.split(jax.random.key(key))
    X = jax.random.uniform(k1, (n, 2), minval=-1.0, maxval=1.0)
    y = jnp.sin(3.0 * X[:, 0]) * jnp.cos(2.0 * X[:, 1])
    y = y + 0.05 * jax.random.normal(k2, (n,))
    return X, y


def test_matches_the_closed_form():
    X, y = _data()
    k = kl.Matern(nu=1.5, lengthscale=0.5)
    lam = 1e-2
    model = kl.KRR(k, regularization=lam).fit(X, y)
    alpha = jnp.linalg.solve(k(X, X) + lam * 30 * jnp.eye(30), y)
    assert jnp.allclose(model.alpha, alpha)
    X_test = X[:5] + 0.1
    assert jnp.allclose(model.predict(X_test), k(X_test, X) @ alpha)


def test_implicit_cg_matches_dense():
    X, y = _data()
    k = kl.RBF(lengthscale=0.5)
    dense = kl.KRR(k, regularization=1e-2).fit(X, y)
    implicit = kl.KRR(
        k,
        regularization=1e-2,
        solver=gx.CGSolver(rtol=1e-10, atol=1e-10, max_steps=500),
        implicit=True,
    ).fit(X, y)
    assert jnp.allclose(implicit.alpha, dense.alpha, atol=1e-6)
    assert jnp.allclose(implicit.predict(X), dense.predict(X), atol=1e-6)


def test_multi_output_equals_per_column_fits():
    X, y = _data()
    Y = jnp.stack([y, 2.0 * y + 1.0], axis=1)
    k = kl.RBF(lengthscale=0.4)
    model = kl.KRR(k, regularization=1e-3).fit(X, Y)
    assert model.alpha.shape == (30, 2)
    assert model.predict(X).shape == (30, 2)
    for c in range(2):
        single = kl.KRR(k, regularization=1e-3).fit(X, Y[:, c])
        assert jnp.allclose(model.alpha[:, c], single.alpha)


def test_regularization_scales_with_n():
    # The same lambda on duplicated data gives the same function: the ridge
    # is lambda * n, so the objective is a mean, not a sum.
    X, y = _data(n=12)
    k = kl.RBF(lengthscale=0.4)
    once = kl.KRR(k, regularization=1e-2).fit(X, y)
    twice = kl.KRR(k, regularization=1e-2).fit(
        jnp.concatenate([X, X]), jnp.concatenate([y, y])
    )
    X_test = X + 0.05
    assert jnp.allclose(once.predict(X_test), twice.predict(X_test), atol=1e-8)


def test_fits_a_smooth_function():
    X, y = _data(n=80)
    ell = kl.estimate_lengthscale(X)
    model = kl.KRR(kl.RBF(lengthscale=ell), regularization=1e-4).fit(X, y)
    X_test, y_test = _data(n=40, key=1)
    assert model.loss(X_test, y_test) < 0.02


def test_validation_loss_gradient():
    X, y = _data()
    X_val, y_val = _data(n=10, key=2)

    def val_loss(ell):
        return kl.KRR(kl.RBF(lengthscale=ell), 1e-2).fit(X, y).loss(X_val, y_val)

    g = jax.grad(val_loss)(0.5)
    eps = 1e-5
    fd = (val_loss(0.5 + eps) - val_loss(0.5 - eps)) / (2 * eps)
    assert jnp.allclose(g, fd, rtol=1e-4)


def test_jit_fit_and_predict():
    X, y = _data()
    fit = eqx.filter_jit(lambda m, X, y: m.fit(X, y))
    model = fit(kl.KRR(kl.RBF()), X, y)
    assert jnp.allclose(model.predict(X), kl.KRR(kl.RBF()).fit(X, y).predict(X))


def test_unfitted_and_shape_errors():
    model = kl.KRR(kl.RBF())
    assert not model.is_fitted
    with pytest.raises(RuntimeError, match="not fitted"):
        model.predict(jnp.zeros((2, 2)))
    with pytest.raises(ValueError, match="y must have shape"):
        model.fit(jnp.zeros((5, 2)), jnp.zeros(4))
    assert model.fit(
        jnp.zeros((5, 2)) + jnp.arange(5.0)[:, None], jnp.ones(5)
    ).is_fitted
