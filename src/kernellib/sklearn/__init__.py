"""scikit-learn adapters for kernellib (optional: ``pip install kernellib[sklearn]``).

The core library never imports scikit-learn; this module is imported only
when asked for. Each adapter wraps a kernellib object in scikit-learn's
estimator conventions (mutable, ``fit`` returns ``self``, fitted attributes
end in ``_``, NumPy in and out), so ``GridSearchCV``, ``Pipeline``,
``cross_val_score`` and ``clone`` work. Kernel hyperparameters are nested
parameters (``kernel__lengthscale``), so they can be searched over. The
fitted kernellib object is kept (``model_``, ``feature_map_``) for everything
JAX can do with it.

- Regressors: `KernelRidge`, `FalkonRegressor`, `EigenProRegressor`.
- Transformers: `RandomFourierFeatures`, `OrthogonalRandomFeatures`,
  `FastFoodFeatures`, `NystromFeatures`, `LaplaceEigenfunctionFeatures`.
- Dependence: `HSIC` (and CKA), `MMD`.
"""

from kernellib.sklearn._dependence import HSIC, MMD
from kernellib.sklearn._regressors import (
    EigenProRegressor,
    FalkonRegressor,
    KernelRidge,
)
from kernellib.sklearn._transformers import (
    FastFoodFeatures,
    LaplaceEigenfunctionFeatures,
    NystromFeatures,
    OrthogonalRandomFeatures,
    RandomFourierFeatures,
)


__all__ = [
    "HSIC",
    "MMD",
    "EigenProRegressor",
    "FalkonRegressor",
    "FastFoodFeatures",
    "KernelRidge",
    "LaplaceEigenfunctionFeatures",
    "NystromFeatures",
    "OrthogonalRandomFeatures",
    "RandomFourierFeatures",
]
