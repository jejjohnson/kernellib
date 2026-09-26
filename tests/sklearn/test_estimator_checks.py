"""scikit-learn's own estimator checks on every regressor and transformer."""

from __future__ import annotations

import pytest
from sklearn.utils.estimator_checks import parametrize_with_checks

from kernellib.sklearn import (
    EigenProRegressor,
    FalkonRegressor,
    FastFoodFeatures,
    KernelPCA,
    KernelRidge,
    LaplaceEigenfunctionFeatures,
    LaplacianEigenmaps,
    LocalityPreservingProjections,
    NystromFeatures,
    OrthogonalRandomFeatures,
    RandomFourierFeatures,
    SchrodingerEigenmaps,
)


ESTIMATORS = [
    KernelRidge(),
    FalkonRegressor(n_inducing=20, random_state=0),
    EigenProRegressor(
        epochs=2, batch_size=16, subsample_size=20, n_components=4, random_state=0
    ),
    RandomFourierFeatures(8, random_state=0),
    OrthogonalRandomFeatures(8, random_state=0),
    FastFoodFeatures(8, random_state=0),
    NystromFeatures(8, random_state=0),
    # One basis function per dimension keeps n_per_dim ** d small for the
    # wide inputs the checks use.
    LaplaceEigenfunctionFeatures(1),
]


@pytest.mark.slow
@parametrize_with_checks(ESTIMATORS)
def test_sklearn_compatible(estimator, check):
    check(estimator)


DECOMPOSITION = [
    KernelPCA(n_components=2),
    LocalityPreservingProjections(n_components=2, n_neighbors=5),
    LaplacianEigenmaps(n_components=2, n_neighbors=5),
    SchrodingerEigenmaps(n_components=2, n_neighbors=5, alpha=5.0),
]


@pytest.mark.slow
@parametrize_with_checks(DECOMPOSITION)
def test_decomposition_sklearn_compatible(estimator, check):
    check(estimator)
