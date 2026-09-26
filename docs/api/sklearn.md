# scikit-learn

Optional adapters that put kernellib's estimators, feature maps and
dependence measures behind scikit-learn's estimator API. Install the extra,
then import the module explicitly; `import kernellib` never loads
scikit-learn.

```bash
uv add "kernellib[sklearn] @ git+https://github.com/jejjohnson/kernellib.git"
```

```python
import kernellib as kl
from kernellib.sklearn import KernelRidge, RandomFourierFeatures, HSIC
from sklearn.model_selection import GridSearchCV

search = GridSearchCV(
    KernelRidge(kernel=kl.Matern(nu=2.5)),
    {"kernel__lengthscale": [0.1, 0.3, 1.0], "regularization": [1e-4, 1e-3]},
).fit(X, y)
search.best_estimator_.model_  # the fitted kl.KRR, a JAX PyTree
```

Conventions:

- **Nested kernel parameters.** A kernel's fields are parameters of the
  estimator: `kernel__lengthscale`, `kernel__nu`, and
  `kernel__kernels__0__variance` inside a `Sum`. Setting one rebuilds the
  immutable kernel with that field replaced (running its converters and
  checks), so `GridSearchCV` and `RandomizedSearchCV` search over them.
- **Default kernel.** `kernel=None` means an `RBF` at the median-heuristic
  lengthscale of the training inputs (`estimate_lengthscale`); the kernel used
  is stored as `kernel_`.
- **Fitted kernellib objects.** `model_` (regressors) and `feature_map_`
  (transformers) hold the fitted kernellib object, so gradients and JAX
  transforms still work on it.
- **Randomness.** `random_state` follows scikit-learn (an int for
  reproducibility) and is mapped to a JAX key.
- **Precision.** Arrays are converted to JAX arrays; enable
  `jax_enable_x64` for float64 computation.

The regressors, transformers and decomposition adapters pass scikit-learn's
`check_estimator` suite.
`HSIC` and `MMD` follow the same conventions but are not predictors.

## Regressors

::: kernellib.sklearn.KernelRidge

::: kernellib.sklearn.FalkonRegressor

::: kernellib.sklearn.EigenProRegressor

## Transformers

::: kernellib.sklearn.RandomFourierFeatures

::: kernellib.sklearn.OrthogonalRandomFeatures

::: kernellib.sklearn.FastFoodFeatures

::: kernellib.sklearn.NystromFeatures

::: kernellib.sklearn.LaplaceEigenfunctionFeatures

## Decomposition

`KernelPCA` and `LocalityPreservingProjections` are transformers.
`LaplacianEigenmaps` and `SchrodingerEigenmaps` are transductive, like
``sklearn.manifold.SpectralEmbedding``: ``fit`` / ``fit_transform`` only.
`SchrodingerEigenmaps.fit(X, y)` takes partial labels, ``-1`` for unlabelled.

::: kernellib.sklearn.KernelPCA

::: kernellib.sklearn.LocalityPreservingProjections

::: kernellib.sklearn.LaplacianEigenmaps

::: kernellib.sklearn.SchrodingerEigenmaps

## Dependence

::: kernellib.sklearn.HSIC

::: kernellib.sklearn.MMD
