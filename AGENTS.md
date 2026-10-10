# AGENTS.md

Standing instructions for **every** coding agent working in this repository
(Claude Code, Copilot, Codex, Gemini, …). This is the single source of truth:
`CLAUDE.md` and `.github/copilot-instructions.md` point here.

## What this repo is

kernellib is kernels and scalable kernel methods for JAX. It sits between
[gaussx](https://github.com/jejjohnson/gaussx) (structured operators,
solvers, preconditioners) and [pyrox-gp](https://github.com/jejjohnson/pyrox)
(GP models with NumPyro priors), and owns everything with a kernel in it:
kernel functions and composition, spectral densities and feature maps,
kernel operators, kernel ridge regression (KRR, Falkon, EigenPro),
dependence measures (HSIC, CKA, MMD), graphs and graph kernels, kernel
decompositions and kernel derivatives. pyrox-gp and pyrox-lgm build on it,
so **kernellib is a library of primitives**: new code composes what is here,
and anything genuinely new lands where the next person will find it.

The full design is `design_docs/kernellib/architecture.md`; read it before
adding a module, but where it disagrees with the code, the code (and the
decision log at the end of the design doc) wins — several early sketches
there (`sample_frequencies(key, n)`, `fit(kernel)`, a richer `_testing.py`)
were superseded.

It is one package, `src/kernellib/`; the public API is `kernellib.__all__`,
plus `kernellib.functional` and the opt-in `kernellib.sklearn`.

| Layer | Path | Owns |
|---|---|---|
| 0 | `functional/` | Pure kernel functions on arrays; matrix-level `hsic`, `cka`, `mmd_squared`, `center_kernel`, graph spectra |
| 0 | `_kernels/` | `AbstractKernel` ⊃ `AbstractPointwiseKernel` ⊃ `AbstractStationaryKernel`, concrete kernels, composition, feature / Nyström / derivative kernels |
| 1 | `_operators/` | `KernelOperator`, `ImplicitKernelOperator`, `ImplicitCrossKernelOperator`, Nyström / RFF / FastFood low-rank operators; the `to_operator` / `to_cross_operator` bridge |
| 1 | `_spectral/` | Feature maps (RFF, ORF, FastFood, Nyström, Laplace eigenfunctions), `select_landmarks`, RFF prior draws |
| 1 | `_heuristics.py` | Bandwidth heuristics |
| 1 | `_graph/` | Graph types, neighbour search, builders, Laplacians, eigenpairs, graph kernels, GMRF structure |
| 2 | `_regression/` | `KRR`, `Falkon`, `EigenPro`, penalties, the Falkon / EigenPro primitives |
| 2 | `_dependence/` | HSIC, CKA, MMD, distance covariance, permutation tests, streaming CKA |
| 2 | `_decomposition/` | Kernel PCA, Laplacian / Schrödinger eigenmaps, LPP / SEP and their kernel forms |
| adapter | `sklearn/` | scikit-learn estimators wrapping the core objects (extra `kernellib[sklearn]`) |
| shared | `_einx.py`, `_testing.py` | einx wrappers; `tree_allclose`, `random_pd_matrix` |

Dependencies point one way, 0 → 1 → 2. The two exceptions are lazy
in-function imports in `_kernels/_feature.py` and `_kernels/_residual.py`
(layer-0 kernels built from layer-1 feature maps); don't add more.

### Boundaries

- **kernellib never imports `numpyro`**, and `import kernellib` loads none of
  scikit-learn, pynndescent, numba, pipekit, pipekit-train or optax
  (`tests/test_imports.py`, slow tier). The opt-in exceptions:
  `kernellib.sklearn` (extra `kernellib[sklearn]`; never imported from the
  core) and the neighbour backends of
  `nearest_neighbors(backend="pynndescent" | "sklearn")` (extras
  `kernellib[neighbors]` / `kernellib[sklearn]`), which import their library
  inside the function, only when chosen, and name the extra if it is missing.
- **gaussx never imports kernellib.** gaussx is kernel-agnostic linear
  algebra; anything with a kernel in it lives here. Use gaussx's public API
  only — copy a small private helper rather than import a gaussx `_` module.
- **geonnax** supplies random-feature arithmetic and basis functions
  (`geonnax.randfeat`, `geonnax.basis`); don't duplicate them.
- **pyrox owns priors, guides and NumPyro.** Hyperparameters here are plain
  array fields with no transforms, priors or constraints; kernel defaults
  match pyrox-gp's (`tests/test_kernels.py::test_defaults_match_pyrox_gp`),
  and breaking changes go through a deprecation, because the pin cascade
  runs gaussx → kernellib → pyrox-gp → pyrox-nn.

## Reuse before you write

1. **Search the capability index.** [`docs/api/capabilities.md`](docs/api/capabilities.md)
   lists every public name in `kernellib`, `kernellib.functional` and
   `kernellib.sklearn`, grouped by defining module, with a one-line summary;
   then gaussx and geonnax's random features and bases. It is generated
   (`make capabilities`) and checked by `tests/test_capabilities.py`.
2. **If it is missing, add it at its layer**, in the subpackage that owns the
   concept; a helper two modules need goes in the owning subpackage (or
   `_einx.py` / `_testing.py`), not inline in one caller.
3. **One object, one name.** The deliberate exceptions are the matrix-level
   `kernellib.functional.hsic` / `mmd_squared` next to the kernels-and-data
   `kernellib.hsic` / `mmd_squared`, and the `kernellib.sklearn` adapters named
   like the core objects they wrap.

| You are about to write… | Use instead |
|---|---|
| A Gram matrix by hand (`exp(-‖x−y‖²/2ℓ²)`) | a kernel class (`RBF`, `Matern`, …) or `kernellib.functional` |
| Pairwise squared distances | the stationary kernel base (`functional/_distances._pairwise_sq_dist`); for mixed precision, `functional.stable_rbf_kernel` / `gaussx.stable_squared_distances` |
| `K + σ²I` then `jnp.linalg.solve` / `cholesky` | `to_operator(kernel, X, noise=σ²)` → `gaussx.solve` / `logdet` (PSD-tagged, structure-aware) |
| A matrix-free kernel matvec | `to_operator(..., implicit=True)`, `ImplicitKernelOperator`, `batched_kernel_matvec` |
| Random Fourier / orthogonal / FastFood / Nyström features | `RandomFourierFeatures`, `OrthogonalRandomFeatures`, `FastFoodFeatures`, `NystromFeatures`, `LaplaceEigenfunctionFeatures` |
| Choosing Nyström landmarks | `select_landmarks` (`rpcholesky` recommended) |
| A bandwidth / lengthscale rule | `estimate_lengthscale`, `lengthscale_to_gamma`, `lengthscale_grid` |
| HSIC / CKA / MMD / distance covariance, a permutation test | `hsic`, `cka`, `mmd_squared`, `distance_covariance_squared`, `permutation_test` (matrix level: `kernellib.functional`) |
| Kernel ridge regression, a CG / preconditioned solve | `KRR` (any gaussx solver strategy), `Falkon`, `EigenPro` |
| Centering a Gram | `functional.center_kernel` / `centering_operator` |
| A k-NN graph, a Laplacian, its eigenpairs | `knn_graph`, `graph_from_edges`, `laplacian_eigpairs`, the graph types' `laplacian_operator` |
| A graph kernel or GMRF structure matrix | `diffusion_kernel`, `matern_graph_kernel`, …; `structure_matrix`, `graph_null_space` |
| Kernel PCA / eigenmaps / LPP | `KernelPCA`, `LaplacianEigenmaps`, `LocalityPreservingProjections`, … |
| GP derivative covariances | `Derivative`, `DerivativeIndexed`, `derivative_inputs` |
| A trace of a product, a Hadamard transform, an RP-Cholesky | gaussx (`trace_product`, `hadamard_transform`, `rp_cholesky`) |
| RFF arithmetic, Fourier bases | `geonnax.randfeat`, `geonnax.basis` |
| A transpose, reshape, axis reduction or inserted-axis broadcast | `kernellib._einx` (`einsum`, `rearrange`, `reduce`, `repeat`) and `einx.<op>` |

## The contracts

### 1. Kernels

- **Pick the level.** `AbstractKernel` needs only the Gram-level
  `__call__(X1, X2) -> (N1, N2)`; `AbstractPointwiseKernel` adds
  `pairwise(x, y) -> ()` (the single differentiation point: derivative
  kernels, the implicit operators' custom JVP, HSIC input gradients);
  `AbstractStationaryKernel` needs `lengthscale` and `variance` fields and
  `shape(r2)`, and gets the closed-form Gram and `diag` for free. Whether a
  kernel is pointwise is the **`is_pointwise` property**, never an
  `isinstance` check (composites are `AbstractKernel`s).
- **Hyperparameters are plain array fields**:
  `eqx.field(default=..., converter=jnp.asarray)`. A Python float would
  become static under `eqx.partition(kernel, eqx.is_array)` and silently stop
  being differentiable through `to_operator(implicit=True)`. Settings that
  pick a code path (`Matern.nu`, `Polynomial.degree`) are
  `eqx.field(static=True)`; validation goes in `__check_init__`.
- **Override cheap paths** where they exist: `diag`, and the private
  `_gram_structure(X) -> GramParts | None` that lets `to_operator` keep a
  diagonal or low-rank Gram structured.
- **Spectral support is opt-in**: `unit_spectral_density(omega_sq, d)` and
  `sample_unit_frequencies(key, shape, dtype)` on a stationary kernel; the
  base turns them into `spectral_density`, `sample_frequencies(key, n, d)`
  and `spectral_variance`. Multivariate frequencies are drawn jointly
  (Matérn's multivariate Student-t), never coordinate-wise.
- **Numerics.** A kernel smooth in r² uses `_smooth_in_r2` (a double-where
  Taylor branch) so `jax.hessian` of `pairwise` is exact at x = y; a rough
  kernel joins the rejection list in `_derivative._check_differentiable`.
  Composites (`Sum`, `Product`, `Scaled`, `ActiveDims`, `Warped`,
  `Periodised`) build from each child's own Gram path.
- **`kernellib.functional`** is arrays in, arrays out, no kernel objects and
  no keys. Its kernel functions take **variance before lengthscale**
  (`rbf_kernel(X1, X2, variance, lengthscale)`), the classes declare
  lengthscale first.
- **Join the zoo.** A new kernel gets a `pytest.param` in `ZOO`
  (`tests/test_kernels.py`): agreement with `functional`, symmetry, PSD,
  `diag`, pointwise-vs-Gram, gradients, `filter_jit` / `vmap`. Spectral
  kernels also go through `tests/spectral/test_density.py`.

### 2. Operators and feature maps

- **Kernel operators** subclass `lx.AbstractLinearOperator`; a new class is
  appended to `_KERNEL_OPERATORS` in `_operators/__init__.py` (which
  registers lineax's predicates and `linearise`) **and** gets an
  `lx.diagonal` registration (gaussx's partial-Cholesky and preconditioned
  CG need the O(N) diagonal). **Tags are claims**: an operator built from a
  raw callable needs `tags={symmetric, positive_semidefinite}` by hand or
  gaussx's fast paths are skipped; a PSD tag on a non-square operator raises.
- **The bridge.** `to_operator(kernel, X, *, noise=None, implicit=False,
  structure="auto")` is the one way from a kernel to a gaussx-ready operator.
  `implicit=True` needs a pointwise kernel and a concrete `noise` (the fused
  `noise_var` is static); a Gram-only kernel raises rather than densifies.
- **Feature maps** subclass `AbstractFeatureMap`: `fit(kernel, X)` returns a
  new map (`dataclasses.replace`), `features(X) -> (N, R)`; configuration is
  static, fitted state (including a trailing `kernel=None` field) are leaves;
  random maps store draws at **unit** lengthscale and variance and read the
  hyperparameters at call time (so gradients flow), accept `Sum` / `Scaled`
  composites via `_require_spectral`, and name their PRNG field `key`.

### 3. Estimators

- **Core estimators are immutable `eqx.Module`s.** Configuration in the
  constructor (static when it is an int, bool or string); fitted fields
  default to `None`, declared after the required ones; `fit(...)` returns a
  **new** module via `dataclasses.replace`; `predict` / `transform` raise
  `RuntimeError` when unfitted. Regressors subclass `AbstractEstimator`
  (`_regression/_base.py`) and implement `_predict`.
- **Solves go through gaussx**: `KRR` takes any `gx.AbstractSolverStrategy`
  and builds preconditioned CG from `gx.NystromPreconditioner` /
  `gx.PartialCholeskyPreconditioner`; never hand-roll CG.
- **Keys.** A `key` is required wherever the result depends on it
  (randomized fits, sampling); `key=None` only where a key merely seeds a
  solver (roadmap decision 6, `docs/roadmap/roadmap.md`).
- **scikit-learn adapters** (`kernellib.sklearn`) are the mutable mirror:
  `__init__` stores parameters verbatim, `fit` returns `self`, fitted
  attributes end in `_`, NumPy in and out, `_KernelParamsMixin` before
  `BaseEstimator`, and every adapter passes `check_estimator`
  (`tests/sklearn/test_estimator_checks.py`, integration tier).

### 4. JAX numerics and the einx convention

- Pure functions; explicit keys; no Python control flow on traced values;
  builders that need concrete indices (graph construction) run eagerly and
  say so.
- The test suite runs with x64 on (`tests/conftest.py`); build constants with
  the input's dtype.
- **einx for every operation on a dense array**, in `src/` and `tests/`:
  contractions and transposes `einsum(A, x, "j i, j -> i")` (never `A.T @ x`,
  `jnp.einsum`, `jnp.transpose`); reshapes `rearrange` (never `.reshape`);
  axis reductions `reduce(K, "i j -> j", "mean")` (never `jnp.mean(K,
  axis=0)`); inserted-axis broadcasts `einx.subtract("i j, j -> i j", K,
  col)` (never `K - col[None, :]`). Use the `kernellib._einx` wrappers
  (tensor first, pattern last for `einsum`; raw `einx.<op>` takes the pattern
  first). Fine as they are: `L @ z` without a transpose, full reductions,
  `jnp.eye` / `jnp.diag`, `jnp.concatenate` / `jnp.stack`, lineax operator
  methods, and `axis=` on scipy / numpy sparse objects. **No test or lint
  rule enforces this** (older tests still break it), so grep your diff for
  `axis=`, `[:, None]`, `[None, :]`, `.T` and `reshape` before committing.

### The public API

- `__all__` in `src/kernellib/__init__.py` is the contract: sorted in RUF022
  order and unique (`tests/test_public_api.py`); a new non-package top-level
  module joins `SUBMODULES` there with its own sorted `__all__`.
- Each public name gets `::: kernellib.<Name>` on its page in `docs/api/` and
  a row in `docs/api/index.md` (no test checks this), and
  `make capabilities`.
- Google-style docstrings with jaxtyping shapes and an **executable**
  `Examples:` block: `--doctest-modules` runs every one in the fast tier, so
  verify the expected output against what the code prints. A doctest over
  ~1 s goes into `_SLOW_DOCTESTS` in `tests/conftest.py`; `kernellib.sklearn`
  doctests are integration automatically.

## What enforces them

| Test | Enforces |
|---|---|
| `tests/test_kernels.py` (`ZOO`, `TestZoo`) | Every kernel vs `functional`, symmetry, PSD, `diag`, pointwise-vs-Gram, gradients, `jit` / `vmap`; defaults match pyrox-gp |
| `tests/spectral/test_density.py`, `test_feature_maps.py` | Spectral densities (closed form, Bochner, sampler), feature maps (`RANDOM_MAPS`) |
| `tests/test_public_api.py` | `__all__` sorted, unique, importable; `SUBMODULES`; `py.typed`; version |
| `tests/test_imports.py` (slow) | `import kernellib` loads no numpyro / sklearn / pynndescent / numba / pipekit / optax |
| `tests/sklearn/test_estimator_checks.py` (integration) | Every adapter passes scikit-learn's `check_estimator` |
| `tests/test_capabilities.py` (integration) | `docs/api/capabilities.md` is current; no name bound to two objects |
| `tests/test_build_docs.py` | The docs pipeline's link rewriting and checks; `API_PORT` matches `docs/myst.yml` |
| `--doctest-modules` | Every docstring example runs |

## Recipes

Step-by-step recipes for the common jobs live as plain Markdown in
`.claude/skills/<name>/SKILL.md` (Claude Code loads them automatically; any
agent can read and follow them):

| Job | Recipe |
|---|---|
| Add a kernel (stationary, pointwise or Gram-level; spectral hooks) | `add-kernel` |
| Add a feature map, kernel approximation or landmark rule | `add-feature-map` |
| Add a kernel linear operator or extend `to_operator` | `add-kernel-operator` |
| Add an estimator or decomposition, with its scikit-learn adapter | `add-estimator` |
| Add a dependence or two-sample statistic | `add-dependence-measure` |
| Add a graph builder, weigher, graph kernel or embedding | `add-graph-component` |
| Add or update an example notebook | `add-notebook` |
| Bump gaussx / geonnax | `bump-upstream-pins` |
| Verify before a PR | `pre-pr-check` |
| Review a change | `kernellib-review` (+ the read-only `.claude/agents/reuse-reviewer.md` and `numerics-reviewer.md`) |
| Write a squash commit message | `squash-commit` |
| Open or link GitHub issues | `create-gh-issue`, `link-gh-issues` (templates in `.github/ISSUE_TEMPLATE/`) |

## Working in the repo

Always run Python tools through `uv run` (never the system Python); `git`,
`ls` and other non-Python commands need no `uv run`.

```bash
make install              # uv sync --all-groups + pre-commit hooks
make test                 # fast tier (+ doctests): uv run pytest -n auto
make test-slow            # -m "slow and not integration"
make test-integration     # -m integration (sklearn checks, optional backends, cross-library)
make test-all             # every tier: -m ""
make test-cov             # every tier with coverage
make lint                 # ruff check .   (entire repo, incl. notebook cells)
make format               # ruff format . && ruff check --fix .
make typecheck            # ty check src/kernellib scripts
make capabilities         # regenerate docs/api/capabilities.md
make docs-api             # API reference only (MkDocs, strict; fast, no Node)
make docs                 # both halves + link check (needs mystmd: npm install -g mystmd)
```

Run one test with `uv run pytest tests/test_kernels.py -k RBF -v` (from the
root, so `tests/conftest.py` applies; `pytest src/kernellib/...` alone skips
it).

### Test tiers

- **Unmarked (fast, the default):** unit tests and doctests, under ~1 s each.
- **`@pytest.mark.slow`:** over ~1 s (heavy numerics, `jit`+`grad`+`vmap`
  sweeps, Monte Carlo / RFF-convergence checks).
- **`@pytest.mark.integration`:** end-to-end workflows — scikit-learn
  `check_estimator`, optional backends, cross-library pipelines.

`addopts` selects `-m "not slow and not integration"`; the last `-m` wins, so
`-m slow`, `-m integration` or `-m ""` select the other tiers. CI runs the
three tiers as parallel jobs and gates coverage (90 %) on their union, so no
single tier has to reach it.

### Tests that assert on random draws

- **Incidental randomness** (any draw would do): pin the key
  (`jax.random.key(0)`) or use the `getkey` fixture.
- **Sampling behaviour under test** (an RFF Gram converging to the exact
  Gram, a randomized HSIC converging to the dense one): bound the estimator
  by its own sampling distribution and say in a comment where the bound came
  from (see `tests/spectral/test_feature_maps.py`,
  `tests/dependence/test_permutation.py`).

### Before every commit

All of these must pass, from the repo root:

1. `make test` (fast tier + doctests); `make test-slow` /
   `make test-integration` too when your change touches code those tiers
   cover.
2. `uv run --group lint ruff check .` — the **entire** repo, which includes
   `tests/`, `scripts/` and the code cells of `docs/notebooks/*.ipynb`.
3. `uv run --group lint ruff format --check .`
4. `make typecheck` (`src/kernellib` and `scripts`).
5. After changing a public API: `make capabilities`, the `docs/api` page and
   the `docs/api/index.md` row.
6. After changing docs: `make docs-api` at least; `make docs` when prose,
   notebooks or cross-references changed.
7. After changing a dependency: `uv lock`, and commit `uv.lock`.

## Coding principles

1. **Think before coding.** State assumptions; if a request has several
   readings, name them instead of picking one silently; ask when unsure.
2. **Simplicity first.** The minimum code that solves the problem: no
   speculative features, no single-use abstractions.
3. **Surgical changes.** Touch only what the task needs; match the existing
   style; don't refactor or add docstrings to code you didn't change; remove
   only what your change made unused.
4. **Goal-driven.** Turn the task into a check (a failing test, a reproduced
   bug, a dense reference to match) and loop until it passes.

Also: type hints on every public function, Google-style docstrings,
`eqx.Module` (not dataclasses) for anything that flows through JAX
(NamedTuples are fine for results such as `PermutationTestResult`).

## Git, commits and pull requests

- Never push to or merge into `main` unless explicitly told to ("push to
  main", "merge to main"). Work on a feature branch, commit locally, and push
  only when asked. "Merge the branch" means push the feature branch.
- Commit messages and PR titles follow
  [Conventional Commits](https://www.conventionalcommits.org/) with a
  lowercase subject (`feat(spectral): add …`); CI validates PR titles. Types:
  `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `build`, `ci`,
  `chore`, `revert`. Breaking changes use `!` and a `BREAKING CHANGE:` footer.
- Releases are cut by release-please; don't bump versions by hand.
- **Never replace or remove an existing PR title or description.** Read it
  first; only append checklist items or update their status (or fix a
  Conventional Commits violation in the title).
- Code review follows [`CODE_REVIEW.md`](CODE_REVIEW.md); issues follow the
  label taxonomy and epic model in [`docs/contributing.md`](docs/contributing.md).

### Pull Request Review Comments

After fixing a review comment, resolve its thread. Don't resolve threads you
didn't address.

```bash
# 1. List the review threads and their IDs
gh api graphql -f query='
  query($owner: String!, $repo: String!, $pr: Int!) {
    repository(owner: $owner, name: $repo) {
      pullRequest(number: $pr) {
        reviewThreads(first: 100) {
          nodes { id isResolved comments(first: 1) { nodes { body path line } } }
        }
      }
    }
  }' -f owner=OWNER -f repo=REPO -F pr=PR_NUMBER

# 2. Resolve an addressed thread
gh api graphql -f query='mutation($threadId: ID!) {
  resolveReviewThread(input: {threadId: $threadId}) { thread { isResolved } } }' \
  -f threadId=THREAD_ID
```

When the `gh` CLI is unavailable, use the GitHub MCP tools for the same
operations.

### Automated reviewers: hide addressed comments

This applies **only** to the GitHub Copilot and ChatGPT Codex bots (authors `copilot-pull-request-reviewer` and `chatgpt-codex-connector`). Human reviewers' comments are never hidden.

Address bot comments with code changes; **do not reply to them**. Once a bot comment is addressed (or deliberately declined; say why in the PR description or to the maintainer, not in a reply), resolve its thread as above, then hide the comment as resolved:

```bash
# 3a. Review-thread comment IDs (paginated: --paginate follows endCursor)
gh api graphql --paginate -f query='
  query($owner: String!, $repo: String!, $pr: Int!, $endCursor: String) {
    repository(owner: $owner, name: $repo) {
      pullRequest(number: $pr) {
        reviewThreads(first: 100, after: $endCursor) {
          pageInfo { hasNextPage endCursor }
          nodes { isResolved comments(first: 50) { nodes { id isMinimized author { login } } } }
        }
      }
    }
  }' -f owner=OWNER -f repo=REPO -F pr=PR_NUMBER

# 3b. Plain PR comment IDs (paginated: PRs can carry more than 100 comments)
gh api graphql --paginate -f query='
  query($owner: String!, $repo: String!, $pr: Int!, $endCursor: String) {
    repository(owner: $owner, name: $repo) {
      pullRequest(number: $pr) {
        comments(first: 100, after: $endCursor) {
          pageInfo { hasNextPage endCursor }
          nodes { id isMinimized author { login } body }
        }
      }
    }
  }' -f owner=OWNER -f repo=REPO -F pr=PR_NUMBER

# 4. Hide an addressed bot comment as resolved
gh api graphql -f query='mutation($id: ID!) { minimizeComment(input: {subjectId: $id, classifier: RESOLVED}) { minimizedComment { isMinimized } } }' -f id=COMMENT_ID
```

- **Review-thread comments:** hide the bot's comments in resolved threads only.
- **Plain PR comments:** on stacked PRs, Codex posts its review as one plain comment ("💡 Codex Review") rather than as threads. Address every finding in it, then hide the bot comment.
- Skip comments that are already minimized.


## Documentation

The docs are built by **two tools** and deployed as one site
(`docs/README.md` has the rationale); `docs.yml` builds them on every PR and
`pages.yml` deploys from `main`.

| Half | Tool | Source | Deployed at |
|---|---|---|---|
| Prose — home, guides, notebooks, roadmap | mystmd | `docs/*.md`, `docs/guide/`, `docs/notebooks/`, `docs/roadmap/` | `/` |
| API reference | MkDocs + mkdocstrings | `docs/api/` | `/reference/` |

`scripts/build_docs.py` (`make docs`) builds the API half, serves it on port
8910 so mystmd can read its `objects.inv`, builds the prose half with
`--strict`, assembles both into `public/`, rewrites the API URLs and checks
every internal link. mystmd is a Node CLI (`npm install -g mystmd`), not a uv
dependency; `make docs-api` needs neither.

- **Prose is MyST Markdown**, not MkDocs-Material: `:::{note}` /
  `:::{tab-set}` / `:::{dropdown}` directives. Cross-reference the API with
  `xref:` and the **top-level** name: [`RBF`](xref:api#kernellib.RBF), never
  `kernellib._kernels.RBF`. A missing target fails `myst build --strict`.
- **URLs are flat**: mystmd derives a page's URL from its basename
  (`guide/architecture.md` → `/architecture/`, underscores → hyphens), so keep
  basenames unique across `docs/guide/` and `docs/notebooks/`.
- **API pages** (`docs/api/*.md`) are MkDocs: hand-written prose plus one
  `::: kernellib.<Name>` per object, `!!!` admonitions.
- **Notebooks** in `docs/notebooks/` are executed `.ipynb` files with outputs
  committed (mystmd does not re-execute them). Author in jupytext percent
  format, execute, delete the `.py`, and add the notebook to the `toc` in
  `docs/myst.yml`. Full standards:
  `.github/instructions/docs-examples.instructions.md`.

## Plans

Plans go in `.plans/` (gitignored, never committed); track work in GitHub
issues. Long-lived design references the code must agree with go in
`design_docs/`. The one exception is the published cross-repo roadmap in
`docs/roadmap/`, which the maintainer chose to render in the docs (PR #92):
it holds proposals, not shipped features, and says so on every page; work
from it is still tracked as GitHub issues.
