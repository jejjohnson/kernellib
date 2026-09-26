"""Behaviour of the scikit-learn adapters beyond the generic checks."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from sklearn.base import clone
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import kernellib as kl
from kernellib.sklearn import (
    HSIC,
    MMD,
    EigenProRegressor,
    FalkonRegressor,
    FastFoodFeatures,
    KernelRidge,
    LaplaceEigenfunctionFeatures,
    NystromFeatures,
    OrthogonalRandomFeatures,
    RandomFourierFeatures,
)


def _data(n=80, d=2, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, size=(n, d))
    return X, np.sin(3 * X[:, 0]) + 0.5 * X[:, -1]


class TestParams:
    def test_nested_kernel_params(self):
        model = KernelRidge(kernel=kl.RBF(lengthscale=0.5) + kl.White(0.1))
        params = model.get_params()
        assert float(params["kernel__kernels__0__lengthscale"]) == 0.5
        assert float(params["kernel__kernels__1__variance"]) == pytest.approx(0.1)
        model.set_params(kernel__kernels__0__lengthscale=2.0)
        assert float(model.kernel.kernels[0].lengthscale) == 2.0
        assert isinstance(model.kernel, kl.Sum)

    def test_set_params_runs_converters_and_checks(self):
        model = KernelRidge(kernel=kl.Matern(nu=1.5))
        model.set_params(kernel__nu=0.5, kernel__lengthscale=3.0)
        assert model.kernel.nu == 0.5
        assert isinstance(model.kernel.lengthscale, jax.Array)
        with pytest.raises(ValueError, match="nu"):
            model.set_params(kernel__nu=0.7)

    def test_set_kernel_then_its_fields_in_one_call(self):
        model = KernelRidge().set_params(kernel=kl.RBF(), kernel__lengthscale=0.3)
        assert float(model.kernel.lengthscale) == pytest.approx(0.3)

    def test_invalid_nested_params(self):
        with pytest.raises(ValueError, match="valid fields"):
            KernelRidge(kernel=kl.RBF()).set_params(kernel__period=1.0)
        with pytest.raises(ValueError, match="set kernel to a kernel first"):
            KernelRidge().set_params(kernel__lengthscale=1.0)

    def test_clone_keeps_the_kernel(self):
        model = FalkonRegressor(kl.Matern(nu=2.5, lengthscale=0.3), n_inducing=10)
        copy = clone(model)
        assert float(copy.kernel.lengthscale) == pytest.approx(0.3)
        assert copy.kernel.nu == 2.5


class TestRegressors:
    def test_kernel_ridge_matches_krr(self):
        X, y = _data()
        k = kl.RBF(lengthscale=0.4)
        model = KernelRidge(kernel=k, regularization=1e-3).fit(X, y)
        expected = kl.KRR(k, regularization=1e-3).fit(jnp.asarray(X), jnp.asarray(y))
        assert np.allclose(model.predict(X), np.asarray(expected.predict(X)))
        assert isinstance(model.alpha_, np.ndarray)
        assert isinstance(model.model_, kl.KRR)

    def test_default_kernel_is_the_median_heuristic(self):
        X, y = _data()
        model = KernelRidge().fit(X, y)
        assert isinstance(model.kernel_, kl.RBF)
        assert np.isclose(
            float(model.kernel_.lengthscale), float(kl.estimate_lengthscale(X))
        )
        assert model.kernel is None  # the parameter is untouched

    def test_grid_search_over_a_composite_kernel(self):
        X, y = _data()
        search = GridSearchCV(
            KernelRidge(kernel=kl.RBF() + kl.Linear()),
            {"kernel__kernels__0__lengthscale": [0.01, 0.5], "regularization": [1e-4]},
            cv=3,
        ).fit(X, y)
        assert search.best_params_["kernel__kernels__0__lengthscale"] == 0.5

    def test_falkon_with_every_point_a_centre_is_kernel_ridge(self):
        X, y = _data(n=50)
        k = kl.RBF(lengthscale=0.5)
        falkon = FalkonRegressor(
            k, n_inducing=100, regularization=1e-3, max_iter=100, tol=1e-12
        ).fit(X, y)
        ridge = KernelRidge(kernel=k, regularization=1e-3).fit(X, y)
        assert falkon.landmarks_.shape == (50, 2)
        assert np.allclose(falkon.predict(X), ridge.predict(X), atol=1e-6)

    def test_random_state_reproducibility(self):
        X, y = _data(n=120)
        make = lambda seed: EigenProRegressor(
            epochs=2, subsample_size=60, n_components=8, random_state=seed
        ).fit(X, y)
        assert np.array_equal(make(0).alpha_, make(0).alpha_)
        assert not np.array_equal(make(0).alpha_, make(1).alpha_)

    def test_cross_val_score_and_multi_output(self):
        X, y = _data()
        scores = cross_val_score(
            FalkonRegressor(n_inducing=30, regularization=1e-4, random_state=0),
            X,
            y,
            cv=3,
        )
        assert np.all(scores > 0.95)
        Y = np.stack([y, -y], axis=1)
        pred = KernelRidge().fit(X, Y).predict(X)
        assert pred.shape == (80, 2)
        assert np.allclose(pred[:, 1], -pred[:, 0])

    def test_eigenpro_needs_three_samples(self):
        with pytest.raises(ValueError, match="n_samples = 2"):
            EigenProRegressor().fit(np.zeros((2, 1)), np.zeros(2))


class TestTransformers:
    @pytest.mark.parametrize(
        ("transformer", "width"),
        [
            (RandomFourierFeatures(16, random_state=0), 32),
            (OrthogonalRandomFeatures(16, random_state=0), 32),
            (FastFoodFeatures(16, random_state=0), 32),
            (NystromFeatures(16, random_state=0), 16),
            (LaplaceEigenfunctionFeatures(4), 16),
        ],
        ids=lambda v: type(v).__name__ if not isinstance(v, int) else str(v),
    )
    def test_output_width_and_feature_names(self, transformer, width):
        X, _ = _data()
        Z = transformer.fit_transform(X)
        assert Z.shape == (80, width)
        names = transformer.get_feature_names_out()
        assert len(names) == width
        assert names[0].startswith(type(transformer).__name__.lower())

    def test_features_match_the_kernellib_map(self):
        X, _ = _data()
        t = RandomFourierFeatures(8, kernel=kl.Matern(nu=1.5), random_state=3).fit(X)
        assert np.allclose(t.transform(X), np.asarray(t.feature_map_(jnp.asarray(X))))

    def test_pipeline(self):
        X, y = _data(n=200, d=3)
        pipe = make_pipeline(
            StandardScaler(),
            NystromFeatures(100, kernel=kl.RBF(lengthscale=1.0), random_state=0),
            Ridge(alpha=1e-4),
        ).fit(X, y)
        assert pipe.score(X, y) > 0.99

    def test_nystrom_caps_the_landmarks(self):
        X, _ = _data(n=10)
        with pytest.warns(UserWarning, match="n_components=50 > n_samples=10"):
            Z = NystromFeatures(50, random_state=0).fit_transform(X)
        assert Z.shape == (10, 10)

    def test_laplace_handles_a_zero_dimension(self):
        X, _ = _data()
        X[:, 1] = 0.0
        t = LaplaceEigenfunctionFeatures(3).fit(X)
        assert t.feature_map_.half_widths[1] == 1.0


class TestHSIC:
    def test_matches_kernellib(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(60, 2))
        y = X[:, 0] ** 2
        est = HSIC(estimator="unbiased").fit(X, y)
        expected = kl.hsic(
            est.kernel_x_,
            est.kernel_y_,
            jnp.asarray(X),
            jnp.asarray(y)[:, None],
            estimator="unbiased",
        )
        assert est.statistic_ == pytest.approx(float(expected))
        assert est.p_value_ is None
        assert np.isclose(
            float(est.kernel_x_.lengthscale), float(kl.estimate_lengthscale(X))
        )

    def test_normalize_is_cka(self):
        rng = np.random.default_rng(1)
        X = rng.normal(size=(50, 2))
        est = HSIC(normalize=True).fit(X, X)
        assert est.statistic_ == pytest.approx(1.0)

    def test_score_uses_the_fitted_kernels_and_nested_params(self):
        rng = np.random.default_rng(2)
        X = rng.normal(size=(50, 1))
        est = HSIC(kl.RBF(), kl.RBF()).fit(X, np.cos(X))
        est.set_params(kernel_x__lengthscale=5.0)
        assert float(est.kernel_x.lengthscale) == 5.0
        refit = clone(est).fit(X, np.cos(X))
        assert refit.statistic_ != est.statistic_
        assert est.score(X, np.cos(X)) == pytest.approx(est.statistic_)

    def test_randomised(self):
        rng = np.random.default_rng(3)
        X = rng.normal(size=(100, 2))
        dense = HSIC().fit(X, X[:, :1] ** 2)
        approx = HSIC(
            approx=kl.NystromFeatures(100, jax.random.key(0), jitter=1e-12)
        ).fit(X, X[:, :1] ** 2)
        assert approx.statistic_ == pytest.approx(dense.statistic_, rel=1e-4)

    def test_unpaired_raises(self):
        with pytest.raises(ValueError, match="paired"):
            HSIC().fit(np.zeros((5, 1)), np.zeros(4))


class TestMMD:
    def test_matches_kernellib_and_takes_different_sizes(self):
        rng = np.random.default_rng(0)
        X, Y = rng.normal(size=(40, 2)), rng.normal(size=(30, 2)) + 0.5
        est = MMD(estimator="unbiased").fit(X, Y)
        expected = kl.mmd_squared(
            est.kernel_, jnp.asarray(X), jnp.asarray(Y), estimator="unbiased"
        )
        assert est.statistic_ == pytest.approx(float(expected))
        pooled = np.concatenate([X, Y])
        assert np.isclose(
            float(est.kernel_.lengthscale), float(kl.estimate_lengthscale(pooled))
        )

    def test_p_value_under_the_null(self):
        rng = np.random.default_rng(1)
        X, Y = rng.normal(size=(50, 2)), rng.normal(size=(50, 2))
        est = MMD(estimator="linear", n_permutations=99, random_state=0).fit(X, Y)
        assert est.p_value_ > 0.05

    def test_feature_mismatch_raises(self):
        with pytest.raises(ValueError, match="same features"):
            MMD().fit(np.zeros((5, 2)), np.zeros((5, 3)))
