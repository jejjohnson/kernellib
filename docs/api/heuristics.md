# Heuristics

Data-driven starting points for kernel lengthscales, reimplemented from
pysim's `estimate_sigma` with its scaling bugs fixed (see the module
docstring).

```python
import kernellib as kl

ell = kl.estimate_lengthscale(X)  # median heuristic
ell = kl.estimate_lengthscale(X, "median", percent=0.1)  # local: 10% neighbour
ell = kl.estimate_lengthscale(X, subsample=2000, key=key)  # O(n^2) -> O(2000^2)
ell = kl.estimate_lengthscale(X, "silverman", ard=True)  # one per dimension
k = kl.RBF(lengthscale=ell)

grid = kl.lengthscale_grid(ell, decades=2.0, n_points=20)
gammas = kl.lengthscale_to_gamma(grid)  # scikit-learn's exp(-gamma r^2)
```

::: kernellib.estimate_lengthscale

::: kernellib.lengthscale_to_gamma

::: kernellib.gamma_to_lengthscale

::: kernellib.lengthscale_grid
