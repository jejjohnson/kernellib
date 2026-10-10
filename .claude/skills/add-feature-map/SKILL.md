---
name: add-feature-map
description: Add a feature map or kernel approximation to kernellib — an AbstractFeatureMap (random Fourier / orthogonal / FastFood / Nyström / eigenfunction style) with fit(kernel, X), unit-scale draws, composite support, its scikit-learn transformer adapter, tests and docs; or a landmark-selection method. Use when asked to add, port or implement random features, a low-rank kernel approximation, or a Nyström landmark rule in src/kernellib/_spectral.
---

# Add a feature map

Read "2. Operators and feature maps" in `AGENTS.md`'s contracts first.

## 1. Make sure it does not exist yet

- `docs/api/capabilities.md`: the kernellib feature maps
  (`RandomFourierFeatures`, `OrthogonalRandomFeatures`, `FastFoodFeatures`,
  `NystromFeatures`, `LaplaceEigenfunctionFeatures`), `select_landmarks`,
  the low-rank operators (`nystrom_operator`, `rff_operator`,
  `fastfood_operator`), and geonnax's `randfeat` / `basis` arithmetic.
- A new landmark rule is a method of `select_landmarks`
  (`_spectral/_landmarks.py`, `_METHODS`), not a new map. RP-Cholesky comes
  from `gaussx.rp_cholesky`.

## 2. Write it (`src/kernellib/_spectral/`)

- Subclass `AbstractFeatureMap` (`_spectral/_base.py`) and implement
  `fit(kernel, X) -> AbstractFeatureMap` and `features(X) -> (N, R)`; the
  base provides `is_fitted`, `__call__` (raises unfitted) and `operator(X)`
  (a zero-base PSD `gx.LowRankUpdate`).
- Configuration (`n_features`, method names, jitter) is
  `eqx.field(static=True)`; fitted state are leaves, with a `kernel` field
  declared after the required fields and defaulting to `None`. `fit` returns
  `dataclasses.replace(self, kernel=kernel, ...)`.
- Random maps: name the PRNG field **`key`** (dependence measures split
  `approx.key` for independent X / Y fits); store draws at **unit**
  lengthscale and variance and read the kernel's hyperparameters at call
  time, so gradients flow to them; get frequencies from the kernel's own
  `sample_unit_frequencies` (don't hard-code the RBF law); accept `Sum` /
  `Scaled` composites through `_require_spectral` and split features with
  the existing helpers in `_feature_maps.py`.
- Arithmetic (cos / sin features, orthogonal blocks, Fourier bases) comes
  from `geonnax.randfeat` / `geonnax.basis`; a Hadamard transform from
  gaussx.
- Docstring: the approximation, its error / cost, the required kernel
  support, an executable `Examples:` block. A fitted map's key is a leaf,
  so differentiate with `eqx.filter_grad`.

## 3. Export, adapter, docs

- `_spectral/__init__.py`, `src/kernellib/__init__.py` (import and sorted
  `__all__`).
- A scikit-learn transformer in `src/kernellib/sklearn/_transformers.py`
  subclassing `_FeatureMap` with `_build(key, X)`; export it from
  `kernellib.sklearn` and add it to `ESTIMATORS` in
  `tests/sklearn/test_estimator_checks.py`.
- `::: kernellib.NewFeatures` and its row in the table on `docs/api/spectral.md`
  (and on `docs/api/sklearn.md` for the adapter), the name in its module's row of
  `docs/api/index.md`, `make capabilities`.

## 4. Tests (`tests/spectral/test_feature_maps.py`)

- Add it to `RANDOM_MAPS` (random maps) so the shared checks run.
- Φ Φᵀ approximates the exact Gram **within a bound derived from the
  estimator's own variance** (e.g. a number of standard errors), with the
  derivation in a comment; deterministic maps match exactly where they
  should (full-rank Nyström recovers the Gram).
- Gradients reach the kernel's hyperparameters; composites work; an
  unsupported kernel raises a clear error.
- The adapter passes `check_estimator` (integration tier).

## 5. Verify

`make test`, `uv run pytest -m slow tests/spectral`,
`uv run pytest -m integration tests/sklearn`, then `pre-pr-check`.
