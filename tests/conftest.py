"""Shared pytest configuration.

float64 is enabled for the whole suite, as in gaussx: the moved operator
tests compare against dense references at float64 tolerances.
"""

from __future__ import annotations

import equinox.internal as eqxi
import jax
import pytest


jax.config.update("jax_enable_x64", True)


@pytest.fixture
def getkey() -> eqxi.GetKey:
    """Fresh PRNG keys, seeded from ``EQX_GETKEY_SEED`` when set."""
    return eqxi.GetKey()


# Doctests cannot take decorators, so they are tiered here: the scikit-learn
# adapters' examples are integration tests, and the examples below measured
# over a second each (mostly jit compilation) join the slow tier.
_SLOW_DOCTESTS = frozenset(
    {
        "kernellib._decomposition._eigenmaps.LaplacianEigenmaps",
        "kernellib._decomposition._projections.LocalityPreservingProjections",
        "kernellib._decomposition._eigenmaps.SchrodingerEigenmaps",
        "kernellib._decomposition._eigenmaps.laplacian_eigenmap",
        "kernellib._decomposition._eigenmaps.schrodinger_eigenmap",
        "kernellib._graph._construct.adjacency_matrix",
        "kernellib._decomposition._kpca.KernelPCA",
        "kernellib._dependence._distance.distance_correlation_squared",
        "kernellib._dependence._distance.energy_distance",
        "kernellib._dependence._hsic.hsic",
        "kernellib._dependence._permutation.permutation_test",
        "kernellib._dependence._streaming.CKAAccumulator",
        "kernellib._dependence._taylor.taylor_statistics",
        "kernellib._heuristics.estimate_lengthscale",
        "kernellib._kernels._derivative.DerivativeIndexed",
        "kernellib._operators._fastfood.fastfood_features",
        "kernellib._operators._fastfood.fastfood_frequencies",
        "kernellib._operators._fastfood.fastfood_operator",
        "kernellib._operators._fastfood.hadamard_transform",
        "kernellib._regression._estimators.EigenPro",
        "kernellib._regression._estimators.Falkon",
        "kernellib._spectral._feature_maps.FastFoodFeatures",
        "kernellib._spectral._feature_maps.NystromFeatures",
        "kernellib._spectral._feature_maps.OrthogonalRandomFeatures",
        "kernellib._spectral._rff.draw_rff_cosine_basis",
        "kernellib._spectral._rff.evaluate_rff_cosine_paths",
        "kernellib.functional._special.log_bessel_kv",
    }
)


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        if not isinstance(item, pytest.DoctestItem):
            continue
        if item.name.startswith("kernellib.sklearn."):
            item.add_marker(pytest.mark.integration)
        elif item.name in _SLOW_DOCTESTS:
            item.add_marker(pytest.mark.slow)
