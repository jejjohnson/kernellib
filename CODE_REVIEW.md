# Code Review Agent Instructions

Standing instructions for **all** agents performing code reviews on this
repository. kernellib is a JAX kernel library: most defects worth finding are
about **kernels and numerics** (a hyperparameter that silently stopped being
differentiable, a Gram densified where a structured operator was available,
a frequency sampler that is wrong in d > 1, a sampling test with a
tolerance tuned on one draw) or **boundaries** (a core import of
scikit-learn, a re-implemented gaussx primitive), not style. Read
"Boundaries", "Reuse before you write" and "The contracts" in
[`AGENTS.md`](AGENTS.md) first; this file is the checklist and the report
format.

---

## How to Obtain the Diff

Use the following command to get the diff for review:

```bash
BASE_BRANCH="$(git rev-parse --verify main >/dev/null 2>&1 && echo main || echo master)"
git --no-pager diff --no-prefix --unified=100000 --minimal $(git merge-base --fork-point "$BASE_BRANCH")...HEAD
```

If that fails (e.g. detached HEAD, shallow clone), fall back to:

```bash
git --no-pager diff --no-prefix --unified=100000 --minimal "$BASE_BRANCH"...HEAD
```

### Reading the diff

| Prefix | Meaning |
|--------|---------|
| `+` | Added line |
| `-` | Removed line |
| ` ` (space) | Unchanged context |
| `@@` | Hunk header |

---

## Review Checklist

Skip anything ruff, ty or the tests already enforce (formatting, import
order, `__all__` ordering, doctest output); review what they cannot see —
including the einx convention, which no test enforces.

### 1. Reuse and boundaries

- Every function, class or module the diff **adds** has been checked against
  [`docs/api/capabilities.md`](docs/api/capabilities.md) (kernellib, gaussx,
  geonnax). A re-implemented kernel, distance, feature map, landmark rule,
  CG loop, preconditioner, centering, HSIC / MMD or graph builder is a
  **High** finding, with the existing name to use.
- No `numpyro` anywhere; no scikit-learn / pynndescent / optax import
  reachable from `import kernellib` (only `kernellib.sklearn` and the
  in-function neighbour backends); core code never imports `kernellib.sklearn`.
- gaussx used through its public API only; geonnax for random-feature
  arithmetic and bases; layers 0 → 1 → 2 without new upward imports.
- Kernel defaults still match pyrox-gp; a breaking change carries a
  deprecation.

### 2. Kernels

- The right level (`AbstractKernel` / `AbstractPointwiseKernel` /
  `AbstractStationaryKernel`); pointwise-ness read from `is_pointwise`, not
  `isinstance`.
- Hyperparameters are `eqx.field(default=..., converter=jnp.asarray)` array
  leaves (no priors, transforms or constraints); code-path settings are
  static; validation in `__check_init__`.
- `diag` / `_gram_structure` overridden where a cheaper or structured form
  exists.
- Smooth-in-r² kernels use `_smooth_in_r2`; rough kernels are rejected by
  `_check_differentiable`.
- Spectral hooks (`unit_spectral_density`, `sample_unit_frequencies`) are
  consistent with each other (the sampler reproduces the kernel) and draw
  multivariate frequencies jointly.
- `functional` signatures keep variance before lengthscale; the class
  agrees with its `functional` twin; the kernel joins `ZOO`.

### 3. Operators and feature maps

- A new operator is in `_KERNEL_OPERATORS` and has an `lx.diagonal`
  registration; its tags are true (symmetric / PSD only when they are).
- Kernel → operator goes through `to_operator` / `to_cross_operator`; a
  Gram-only kernel is not silently densified on an implicit path.
- Feature maps: `fit(kernel, X)` returns a new map; static configuration;
  unit-scale draws with hyperparameters read at call time; composites
  accepted through `_require_spectral`; the PRNG field is `key`; the map
  joins `RANDOM_MAPS` and gets a `kernellib.sklearn` adapter.

### 4. Estimators

- Immutable `eqx.Module`; fitted fields default to `None`; `fit` returns a
  new module; unfitted `predict` / `transform` raise `RuntimeError`.
- Solves through gaussx strategies and preconditioners; convergence
  reported (`n_iter`, `converged`), not assumed.
- A `key` is required wherever the result depends on it.
- The scikit-learn adapter (if any) stores parameters verbatim, returns
  `self` from `fit`, ends fitted attributes in `_`, and is in the
  `check_estimator` lists.

### 5. JAX numerics and einx

- No Python control flow on traced values; eager-only builders say so.
- Dtypes follow the input (`jnp.eye(n, dtype=X.dtype)`); a bare Python
  scalar combined with an array is weakly typed and fine.
- No `jnp.linalg.solve` / `inv` / `cholesky` on a kernel matrix where
  `to_operator` + gaussx applies; no explicit inverse.
- Gradients: `jnp.where` branches safe for NaN gradients (the double-where
  in `_smooth_in_r2`); no hyperparameter turned static.
- **einx**: contractions / transposes via `einsum`, reshapes via
  `rearrange`, axis reductions via `reduce`, inserted-axis broadcasts via
  `einx.<op>` — grep the diff for `axis=`, `[:, None]`, `[None, :]`, `.T`
  and `reshape`.

### 6. Public API and documentation

- New names: in `__all__` (sorted, RUF022), on their `docs/api` page with a
  `::: kernellib.<Name>` entry and a row in `docs/api/index.md`,
  `docs/api/capabilities.md` regenerated; a new non-package module joins
  `SUBMODULES`.
- Docstrings: Google style, jaxtyping shapes, the formula, a reference for a
  published method, an executable `Examples:` block whose expected output is
  what the code prints (slow ones listed in `_SLOW_DOCTESTS`).
- Prose pages use MyST directives and `xref:api#kernellib.<TopLevelName>`;
  notebook basenames stay unique; new notebooks join the `docs/myst.yml` toc.

### 7. Tests

- Against a closed form, `functional`, a dense reference or a published
  value; structured and dense paths agree.
- Incidental randomness pinned; sampling behaviour bounded by its own
  distribution, with the bound's source in a comment.
- Tier markers right: unmarked under ~1 s, `slow` above, `integration` for
  scikit-learn checks, optional backends and cross-library workflows.

### 8. Modern Python and dependencies

- Type hints on every public function; `X | None`; specific exceptions with
  `raise ... from ...`; guard clauses over deep nesting.
- No new runtime dependency without discussion; optional ones behind an
  extra and an in-function import that names it; `uv.lock` updated.

---

## kernellib-Specific Checks

### Hyperparameters are array leaves

```python
# ❌ A Python float: static under eqx.partition, so to_operator(implicit=True)
#    silently drops its gradient
class MyKernel(AbstractStationaryKernel):
    lengthscale: float = 1.0
    variance: float = 1.0


# ✅ Array leaves, differentiable everywhere
class MyKernel(AbstractStationaryKernel):
    lengthscale: Float[Array, "*D"] = eqx.field(default=1.0, converter=jnp.asarray)
    variance: Float[Array, ""] = eqx.field(default=1.0, converter=jnp.asarray)
```

### Through the bridge, not dense LAPACK

```python
# ❌ Dense, untagged, no structure
alpha = jnp.linalg.solve(kernel(X, X) + noise * jnp.eye(n), y)

# ✅ PSD-tagged and structure-aware; swap in a CG strategy without touching the math
alpha = gx.solve(kl.to_operator(kernel, X, noise=noise), y)
```

### Pointwise-ness is a property

```python
# ❌ Composites (Sum, Scaled, ...) are AbstractKernels even when pointwise
if isinstance(kernel, kl.AbstractPointwiseKernel):
    ...

# ✅
if kernel.is_pointwise:
    ...
```

### The einx convention

```python
# ❌
K_centered = K - K.mean(axis=0)[None, :]
v = A.T @ x

# ✅
K_centered = einx.subtract("i j, j -> i j", K, reduce(K, "i j -> j", "mean"))
v = einsum(A, x, "j i, j -> i")
```

---

## Output Format

Format each review using this structure:

````
# Code Review for ${feature_description}

Overview of the changes, including the purpose, context, and files involved.

## Suggestions

### ${emoji} ${Summary of suggestion with necessary context}

* **Priority**: ${priority_emoji} ${priority_label}
* **File**: `${relative/path/to/file.py}`
* **Line(s)**: ${line_numbers}
* **Details**: Explanation of the issue and why it matters
* **Current Code**:
  ```python
  # problematic code
  ```
* **Suggested Change**:
  ```python
  # improved code with explanation
  ```

### (additional suggestions…)

## Summary

Brief summary of overall code quality and key action items.
````

---

## Priority Levels

| Emoji | Level | Use when |
|-------|-------|----------|
| 🔥 | **Critical** | Bugs, security issues, or code that will fail |
| ⚠️ | **High** | Significant issues affecting maintainability or correctness |
| 🟡 | **Medium** | Improvements for code quality or consistency |
| 🟢 | **Low** | Minor polish or optional enhancements |

## Suggestion Type Emojis

Prefix each suggestion title with a type indicator:

| Emoji | Type |
|-------|------|
| 🐛 | Bug or potential bug |
| 🔒 | Security concern |
| 🔧 | Change request (must fix) |
| ♻️ | Refactor suggestion |
| 📝 | Documentation improvement |
| 🎨 | Style / formatting issue |
| ⚡ | Performance consideration |
| 🧪 | Testing suggestion |
| ❓ | Question or clarification needed |
| ⛏️ | Nitpick (very minor) |
| 💭 | Design consideration |
| 👍 | Positive feedback (highlight good patterns) |
| 🌱 | Future consideration (not blocking) |

---

## Review Tone

- Be **constructive** and **specific**
- **Acknowledge** good patterns and decisions (use 👍 liberally)
- Explain the *why* behind every suggestion
- Offer **concrete alternatives**, not just criticism
- Recognize that context matters — ask clarifying questions when needed
- Keep feedback **actionable**: every suggestion should have a clear next step
