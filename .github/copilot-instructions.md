# Copilot Instructions

Read [`AGENTS.md`](../AGENTS.md) at the repository root first: it is the
single source of truth for every coding agent working here (the layer map,
the boundaries with gaussx / geonnax / pyrox, "reuse before you write", the
contracts, the tests that enforce them, commands, the pre-commit checklist,
the two-tool docs, git and PR rules).

The essentials, in case you only read this file:

- One package, `src/kernellib/`, in layers 0 → 1 → 2 (`functional/`,
  `_kernels/` → `_operators/`, `_spectral/`, `_graph/`, `_heuristics.py` →
  `_regression/`, `_dependence/`, `_decomposition/`), plus the opt-in
  `kernellib.sklearn` adapters. Search
  [`docs/api/capabilities.md`](../docs/api/capabilities.md) before writing a
  helper.
- `import kernellib` never loads numpyro, and the core never imports
  scikit-learn (only `kernellib.sklearn` and the optional neighbour backends
  do, lazily).
- Keep the contracts in `AGENTS.md`:
  - **kernels**: hyperparameters are `converter=jnp.asarray` array fields
    (no priors); pointwise-ness is the `is_pointwise` property; spectral
    support through the unit-density / unit-sampler hooks; join `ZOO`;
  - **operators and feature maps**: register new operators in
    `_KERNEL_OPERATORS` plus `lx.diagonal`; `to_operator` is the bridge;
    `fit(kernel, X)` returns a new map;
  - **estimators**: immutable `eqx.Module`s whose `fit` returns a new
    module; solves through gaussx; adapters pass `check_estimator`;
  - **einx** for every dense-array transpose, reshape, axis reduction and
    inserted-axis broadcast (not lint-enforced: grep your diff).
- Every docstring `Examples:` block runs (`--doctest-modules`).
- Before committing, from the repo root: `make test`,
  `uv run --group lint ruff check .`, `uv run --group lint ruff format --check .`,
  `make typecheck`; `make capabilities` after a public API change.
- Path-scoped standards live in `.github/instructions/`; code review follows
  [`CODE_REVIEW.md`](../CODE_REVIEW.md).
