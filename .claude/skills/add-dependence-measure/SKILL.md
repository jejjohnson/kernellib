---
name: add-dependence-measure
description: Add a kernel dependence or two-sample statistic to kernellib — an HSIC / CKA / MMD / distance-covariance style measure with its kernellib.functional Gram-level twin, biased and unbiased estimators, the approx= feature-map path, permutation-test support, its scikit-learn adapter, tests and docs. Use when asked to add, port or implement an independence test, similarity index, two-sample statistic or representation-similarity measure in src/kernellib/_dependence.
---

# Add a dependence measure

## 1. Make sure it does not exist yet

`hsic`, `cka`, `kernel_alignment`, `mmd_squared`, `energy_distance`,
`distance_covariance_squared`, `distance_correlation_squared`,
`taylor_statistics`, `CKAAccumulator` (streaming), `permutation_test`;
Gram-level `kernellib.functional.hsic`, `center_kernel`,
`centering_operator`, `center_cross_kernel`. A p-value is
`permutation_test(statistic, X, Y, key=..., kind=...)` around the
statistic, not a new function.

## 2. Two levels

- **Gram level** (`src/kernellib/functional/_statistics.py`): takes
  evaluated Gram matrices (dense, or a gaussx low-rank operator) and
  returns a scalar; centring through `center_kernel` /
  `centering_operator`, never an explicit `H = I - 11ᵀ/n` matrix product.
- **Kernel level** (`src/kernellib/_dependence/`): takes kernels and
  samples, `name(kernel_x, kernel_y, X, Y, *, estimator=..., approx=None)`
  (one kernel for a two-sample statistic). With `approx` (an unfitted
  feature map) it computes from features in O(N R²): a paired measure uses
  `_fit_pair`, which splits the map's `key` so X and Y get independent
  draws; a two-sample statistic fits one map on `concatenate([X, Y])`, as
  `mmd_squared` does (independent draws would bias the discrepancy).
- `estimator` is a `Literal` (`"biased"`, `"unbiased"`, …), validated with
  a clear error; cite the estimator (paper, equation) in the docstring.
- Differentiable in the kernels' hyperparameters and the inputs (bandwidth
  selection by gradient is a supported use); degenerate inputs (a constant
  sample) return a finite value and NaN inputs still propagate, as `cka`
  does.

## 3. Adapter, export, docs

- A scikit-learn estimator in `src/kernellib/sklearn/_dependence.py`
  following `HSIC` / `MMD` (`statistic_`, optional `p_value_` from
  `n_permutations`, `random_state`).
- Exports in `_dependence/__init__.py`, `functional/__init__.py`,
  `src/kernellib/__init__.py`, `kernellib.sklearn`. The kernel-level and
  functional versions may share a name (`hsic`, `cka`, `mmd_squared`): the
  capability index allows a top-level name shared with one
  `kernellib.functional` or `kernellib.sklearn` name.
- `:::` entries on `docs/api/dependence.md` (and `sklearn.md`; the
  functional page renders the whole module, so update its prose list), the name in its module's row of `docs/api/index.md`, `make capabilities`.

## 4. Tests (`tests/dependence/`)

- Kernel level matches the functional version on the same Grams; the
  feature path is exact for its own Gram, and full-rank Nyström recovers
  the dense value; random features approximate it within a stated
  sampling bound.
- The unbiased estimator is unbiased (mean over seeds, bound from the
  estimator's variance, derivation in a comment); known values (zero under
  independence in expectation, a closed form for a linear kernel —
  linear CKA is the RV coefficient).
- Detects dependence with `permutation_test`; gradients finite; errors for
  unpaired samples and unknown estimators.

## 5. Verify

`make test`, `uv run pytest -n auto -m "slow and not integration" tests/dependence`,
`uv run pytest -n auto -m integration tests/sklearn` (adapter), then
`pre-pr-check`.
