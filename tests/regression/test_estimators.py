"""Falkon and EigenPro estimators."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

import kernellib as kl


def _data(n=300, key=0):
    X = jax.random.uniform(jax.random.key(key), (n, 2), minval=-1.0, maxval=1.0)
    y = jnp.sin(3.0 * X[:, 0]) * jnp.cos(2.0 * X[:, 1])
    return X, y


K = kl.RBF(lengthscale=0.3)


class TestFalkon:
    @pytest.mark.slow
    def test_every_point_a_centre_is_krr(self):
        # Falkon's system with M = N reduces to (K + lambda n I) alpha = y.
        X, y = _data(n=60)
        falkon = kl.Falkon(
            K, n_inducing=60, regularization=1e-3, max_iter=200, tol=1e-12
        ).fit(X, y)
        krr = kl.KRR(K, regularization=1e-3).fit(X, y)
        X_test, _ = _data(n=20, key=1)
        assert jnp.allclose(falkon.predict(X_test), krr.predict(X_test), atol=1e-6)

    @pytest.mark.slow
    @pytest.mark.parametrize("implicit", [True, False])
    def test_generalises_with_few_centres(self, implicit):
        X, y = _data()
        # lambda of order 1/n is Falkon's regime: the default 20 iterations
        # converge. Far smaller lambda needs a larger max_iter.
        model = kl.Falkon(
            K, n_inducing=100, regularization=1e-4, implicit=implicit
        ).fit(X, y, key=jax.random.key(0))
        assert model.landmarks.shape == (100, 2)
        X_test, y_test = _data(n=100, key=1)
        assert model.loss(X_test, y_test) < 1e-3

    @pytest.mark.slow
    def test_implicit_and_dense_agree(self):
        X, y = _data(n=120)

        def fit(implicit):
            # Compared at convergence: truncated CG iterates differ at
            # rounding, and CG run far past convergence drifts.
            return kl.Falkon(
                K,
                n_inducing=40,
                regularization=1e-4,
                implicit=implicit,
                batch_size=32,
                max_iter=100,
                tol=1e-10,
            ).fit(X, y, key=jax.random.key(0))

        assert jnp.allclose(fit(True).alpha, fit(False).alpha, atol=1e-7)

    @pytest.mark.slow
    def test_multi_output(self):
        X, y = _data(n=100)
        Y = jnp.stack([y, -y], axis=1)
        model = kl.Falkon(K, n_inducing=30, regularization=1e-4).fit(
            X, Y, key=jax.random.key(0)
        )
        assert model.alpha.shape == (30, 2)
        assert jnp.allclose(model.predict(X)[:, 1], -model.predict(X)[:, 0])

    @pytest.mark.slow
    def test_gram_only_kernel_uses_the_dense_path(self):
        X, y = _data(n=80)
        model = kl.Falkon(K * 1.0, n_inducing=20).fit(X, y, key=jax.random.key(0))
        assert jnp.all(jnp.isfinite(model.predict(X)))

    def test_needs_a_key_to_subsample(self):
        X, y = _data(n=50)
        with pytest.raises(ValueError, match="PRNG key"):
            kl.Falkon(K, n_inducing=10).fit(X, y)

    @pytest.mark.parametrize("lam", [0.0, -1e-3, jnp.asarray(0.0)])
    def test_regularization_must_be_positive(self, lam):
        with pytest.raises(ValueError, match="regularization must be positive"):
            kl.Falkon(K, regularization=lam)

    def test_tol_must_be_positive(self):
        with pytest.raises(ValueError, match="tol must be positive"):
            kl.Falkon(K, tol=0.0)

    @pytest.mark.slow
    def test_default_tol_resolves_per_dtype(self):
        X, y = _data(n=40)
        model = kl.Falkon(K, n_inducing=40)
        assert model.tol is None
        assert model.fit(X, y).tol == 1e-6
        X32, y32 = X.astype(jnp.float32), y.astype(jnp.float32)
        fitted = model.fit(X32, y32)
        assert fitted.alpha.dtype == jnp.float32
        assert fitted.tol == pytest.approx(float(jnp.finfo(jnp.float32).eps) ** 0.5)
        # An explicit tol is kept as given.
        assert kl.Falkon(K, n_inducing=40, tol=1e-3).fit(X32, y32).tol == 1e-3

    @pytest.mark.slow
    def test_float32_converges_at_the_default_tol(self):
        # #162: in float32 the preconditioned residual stagnates above 1e-6,
        # so the old fixed tol reported converged=False on an accurate fit.
        X = jax.random.uniform(jax.random.key(0), (1000, 2), dtype=jnp.float32)
        y = jnp.sin(6.0 * X[:, 0])

        def fit(tol):
            return kl.Falkon(
                kl.RBF(lengthscale=0.2), n_inducing=200, regularization=1e-3, tol=tol
            ).fit(X, y, key=jax.random.key(3))

        default, tight = fit(None), fit(1e-6)
        assert default.alpha.dtype == jnp.float32
        assert bool(default.converged) and int(default.n_iter) < default.max_iter
        assert not bool(tight.converged)
        assert default.loss(X, y) < 1e-3
        assert default.loss(X, y) == pytest.approx(tight.loss(X, y), rel=1e-2)

    @pytest.mark.slow
    def test_float32_tiny_regularization_warns(self):
        X, y = _data(n=200)
        X32, y32 = X.astype(jnp.float32), y.astype(jnp.float32)
        model = kl.Falkon(K, n_inducing=50, regularization=1e-9)
        with pytest.warns(RuntimeWarning, match="float32 precision floor"):
            model.fit(X32, y32, key=jax.random.key(0))


class TestEigenPro:
    @pytest.mark.slow
    def test_converges_to_the_interpolant(self):
        X, y = _data()
        model = kl.EigenPro(
            K, epochs=10, batch_size=64, subsample_size=150, n_components=20
        ).fit(X, y, key=jax.random.key(1))
        assert model.alpha.shape == (300,)
        assert model.loss(X, y) < 1e-3
        X_test, y_test = _data(n=100, key=2)
        assert model.loss(X_test, y_test) < 1e-3

    @pytest.mark.slow
    def test_preconditioning_speeds_up_sgd(self):
        # One component barely changes plain kernel SGD; twenty damp the top
        # of the spectrum, allow a much larger step and fit faster.
        X, y = _data()

        def loss(k):
            return (
                kl.EigenPro(
                    K, epochs=2, batch_size=64, subsample_size=150, n_components=k
                )
                .fit(X, y, key=jax.random.key(1))
                .loss(X, y)
            )

        assert loss(20) < 0.5 * loss(1)

    @pytest.mark.slow
    def test_loss_decreases_with_epochs(self):
        X, y = _data()

        def loss(epochs):
            return (
                kl.EigenPro(
                    K, epochs=epochs, batch_size=64, subsample_size=150, n_components=20
                )
                .fit(X, y, key=jax.random.key(1))
                .loss(X, y)
            )

        assert loss(8) < loss(2) < loss(1)

    @pytest.mark.slow
    def test_multi_output_and_jit(self):
        X, y = _data(n=150)
        Y = jnp.stack([y, 2.0 * y], axis=1)
        model = kl.EigenPro(
            K, epochs=3, batch_size=50, subsample_size=100, n_components=10
        )
        fitted = eqx.filter_jit(lambda m: m.fit(X, Y, key=jax.random.key(0)))(model)
        assert fitted.alpha.shape == (150, 2)
        # Linear in the targets: same batches, same preconditioner.
        assert jnp.allclose(fitted.alpha[:, 1], 2.0 * fitted.alpha[:, 0])

    @pytest.mark.slow
    def test_every_point_is_visited_each_epoch(self):
        # #162: 101 = 3 * 32 + 5, so the old N // b batches skipped five
        # points; points outside the subsample then kept a zero weight.
        X, y = _data(n=101)
        model = kl.EigenPro(
            K, epochs=1, batch_size=32, subsample_size=20, n_components=5
        ).fit(X, y, key=jax.random.key(1))
        assert bool(jnp.all(model.alpha != 0.0))

    def test_argument_errors(self):
        X, y = _data(n=50)
        with pytest.raises(ValueError, match="PRNG key"):
            kl.EigenPro(K, subsample_size=20, n_components=5).fit(X, y)
        with pytest.raises(ValueError, match="n_components"):
            kl.EigenPro(K, subsample_size=20, n_components=20).fit(
                X, y, key=jax.random.key(0)
            )
        with pytest.raises(ValueError, match="epochs"):
            kl.EigenPro(K, epochs=0)
