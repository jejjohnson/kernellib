---
name: add-estimator
description: Add a kernel estimator to kernellib — an immutable AbstractEstimator (KRR / Falkon / EigenPro style) or eqx.Module decomposition whose fit returns a new module and whose solves go through gaussx strategies and preconditioners — together with its mutable scikit-learn adapter that passes check_estimator. Use when asked to add, port or implement a kernel regressor, solver-backed estimator, kernel PCA / projection, or a scikit-learn wrapper in src/kernellib/_regression, _decomposition or kernellib.sklearn.
---

# Add an estimator

Read "3. Estimators" under "The contracts" in `AGENTS.md` first.

## 1. Make sure it does not exist yet

`KRR` already accepts any gaussx solver strategy and Nyström / RP-Cholesky
preconditioners (and penalties); `Falkon` is inducing-point
preconditioned CG; `EigenPro` is preconditioned SGD. A new **solver** for
kernel ridge regression is usually a gaussx strategy passed as
`KRR(solver=...)`, and belongs in gaussx — not a new estimator here.
Decomposition: `KernelPCA`, the eigenmaps and the (kernel) projections.

## 2. Write it

Regression (`src/kernellib/_regression/`), exemplar `KRR` (`_krr.py`):

- Subclass `AbstractEstimator`; implement `fit(X, y, *, key=None)` and
  `_predict(X)`; the base gives `predict` (raises when unfitted), `loss`
  (MSE, differentiable) and `is_fitted` (`alpha is not None`).
- Configuration fields first (the kernel; numeric settings you want to
  differentiate as array leaves via `eqx.field(converter=jnp.asarray)`,
  which `KRR.regularization` predates;
  settings that pick a code path as `eqx.field(static=True)`), validated in
  `__check_init__`; then fitted fields defaulting to `None`, `alpha`
  among them. `fit` returns `dataclasses.replace(self, ...)` and never
  mutates.
- Targets `(N,)` or `(N, C)`, checked with `_check_targets`; predictions
  shaped like the targets.
- Linear algebra through gaussx: `kl.to_operator(kernel, X, noise=...)`
  then `gx.solve` / a strategy's `solve`; preconditioners from gaussx; no
  hand-written CG, Cholesky or Woodbury.
- Require `key` wherever the fit depends on randomness (inducing points,
  minibatches) and raise when it is missing; never default to a fixed key.
- Iterative solves raise when they do not converge (gaussx's `throw=True`
  or `eqx.error_if`); a user-supplied strategy keeps its own tolerances.
- `jax.grad` must reach the kernel's hyperparameters through `fit`
  (that is the point of the immutable design) — test it.
- Docstring with the objective, the solve path, costs, and an executable
  `Examples:` block (list it in `_SLOW_DOCTESTS` in `tests/conftest.py` if it
  takes over a second).

Decomposition (`src/kernellib/_decomposition/`): an `eqx.Module` with the
same `fit` → new module / fitted fields `None` pattern (`KernelPCA`); graph
embeddings subclass `_GraphEmbedding`.

## 3. The scikit-learn adapter (`src/kernellib/sklearn/`)

- Regressors: subclass `_Regressor` (`_regressors.py`) and implement
  `_build(kernel, n) -> AbstractEstimator`; the base handles
  `validate_data`, the median-heuristic default kernel, the key split and
  `model_` / `kernel_` / `alpha_`. Decomposition adapters follow the
  classes in `_decomposition.py`.
- Constructor arguments stored verbatim (scikit-learn's rule), kernel
  hyperparameters searchable through `_KernelParamsMixin`
  (`kernel__lengthscale`), `random_state` for every key.
- Add an instance to `ESTIMATORS` (or `DECOMPOSITION`) in
  `tests/sklearn/test_estimator_checks.py`; behaviour tests in
  `tests/sklearn/test_adapters.py`.
- `import kernellib` never loads scikit-learn (`tests/test_imports.py`):
  the adapter imports the core, never the other way round.

## 4. Export, docs, tests

- Export from the subpackage, `src/kernellib/__init__.py` (sorted
  `__all__`) and `kernellib.sklearn.__init__`; `:::` entries on
  `docs/api/regression.md` / `decomposition.md` / `sklearn.md`; the name in its module's row of
  `docs/api/index.md`; `make capabilities`.
- Tests (`tests/regression/` or `tests/decomposition/`): agreement with a
  dense closed form (exact KRR on the same data), multi-output targets,
  the unfitted error, missing-key error, gradients through `fit`, `jit`;
  the non-convergence error for iterative paths.

## 5. Verify

`make test`, `uv run pytest -n auto -m "slow and not integration" tests/<area>`,
`uv run pytest -n auto -m integration tests/sklearn` for the adapter checks,
then `pre-pr-check`.
