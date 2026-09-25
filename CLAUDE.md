# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

kernellib: kernels and scalable kernel methods for JAX. It sits between
[gaussx](https://github.com/jejjohnson/gaussx) (structured operators,
solvers, preconditioners) and [pyrox-gp](https://github.com/jejjohnson/pyrox)
(GP models with NumPyro priors). It owns kernel functions and composition,
spectral densities and feature maps, kernel ridge regression, dependence
measures (HSIC, CKA, MMD), kernel embeddings, and kernel derivatives. Built
with Python 3.12+, uv, pytest, mystmd, and MkDocs.

The full design lives in `design_docs/kernellib/architecture.md`. Read it
before adding a module: it fixes the kernel contract, the package layout, and
what belongs in gaussx versus here.

### Boundaries

- **kernellib never imports `numpyro`** (or scikit-learn). `tests/test_imports.py` enforces it.
- **gaussx never imports kernellib.** Anything with a kernel in it lives here, from the kernel operators up; gaussx is kernel-agnostic linear algebra (structured operators, solvers, preconditioners, `trace_product`, `stable_squared_distances`). gaussx's current kernel layer moves here in gaussx 0.2.0.
- **`kernellib.functional`** is arrays in, arrays out (pure kernel functions, matrix-level HSIC / CKA / MMD / centering). The same names at the top level take kernels and data.
- **geonnax** supplies basis functions and random-feature arithmetic; do not duplicate them.

## Common Commands

```bash
make install              # Install all deps (uv sync --all-groups) + pre-commit hooks
make test                 # Run tests: uv run pytest -v
make format               # Auto-fix: ruff format . && ruff check --fix .
make lint                 # Lint code: ruff check .
make typecheck            # Type check: ty check src/kernellib scripts
make precommit            # Run pre-commit on all files
make docs-serve           # Local docs server
```

### Running a single test

```bash
uv run pytest tests/test_example.py::TestClass::test_method -v
```

### Pre-commit checklist (all four must pass)

```bash
uv run pytest -v                              # Tests + doctests
uv run --group lint ruff check .              # Lint — ENTIRE repo, not just src/kernellib/
uv run --group lint ruff format --check .     # Format — ENTIRE repo
uv run --group typecheck ty check src/kernellib scripts  # Typecheck
```

**Critical**: Always lint/format with `.` (repo root), not `src/kernellib/`. CI runs `ruff check .` which includes `tests/`, `scripts/`, **and the code cells of `docs/notebooks/*.ipynb`**.

`--doctest-modules` is in `addopts` and `src/kernellib` is a `testpath`, so every `Examples:` block in a docstring is executed on each run. When you change behaviour, update the examples — and verify the expected output against what the code actually prints.

### Building the docs

```bash
make docs        # what CI runs: builds both halves and verifies every link
make docs-api    # API reference only — fast, and needs no Node
```

See the [Documentation](#documentation) section below; `mkdocs build` alone
covers only the API half.

## Architecture

### Package structure

All implementation lives in `src/kernellib/`. The public API is re-exported through `src/kernellib/__init__.py`, and `__all__` there is the contract — `tests/test_public_api.py` enforces it. Every new top-level module must be added to `SUBMODULES` in that test and given an API doc page.

The planned layout (from the design doc; directories appear as their phase lands):

| Path | Contents | Layer |
|---|---|---|
| `functional/` | Pure kernel functions on arrays; matrix-level `hsic`, `cka`, `mmd_squared`, `center_kernel` | 0 |
| `_kernels/` | `AbstractKernel` ⊃ `AbstractPointwiseKernel` ⊃ `AbstractStationaryKernel`, concrete kernels, composition | 0 |
| `_operators/` | `KernelOperator`, `ImplicitKernelOperator`, `ImplicitCrossKernelOperator`, Nyström / RFF / FastFood low-rank operators (moved from gaussx); `to_operator` / `to_cross_operator` bridge | 1 |
| `_spectral/` | Spectral densities, feature maps (RFF, ORF, Nyström, FastFood, Laplace eigenfunctions) | 1 |
| `_heuristics.py` | Bandwidth heuristics | 1 |
| `_regression/` | `KRR`, `Falkon`, `EigenPro` estimators, with the Falkon / EigenPro primitives moved from gaussx | 2 |
| `_dependence/` | HSIC, CKA, MMD, permutation tests | 2 |
| `_decomposition/` | Kernel PCA, graph kernels | 2 |
| `_derivatives.py` | Kernel derivatives | 2 |

Dependency direction is strictly one-way, layer 0 → 1 → 2.

### Key directories

| Path | Purpose |
|------|---------|
| `src/kernellib/` | Main package source code |
| `tests/` | Test suite |
| `design_docs/` | Committed design references |
| `docs/guide/` | Hand-written guide pages |
| `docs/api/` | mkdocstrings API reference, one page per module |
| `docs/notebooks/` | Executed example notebooks (outputs committed) |
| `scripts/` | Build tooling, incl. `build_docs.py` (the two-tool docs pipeline) |

## Test Speed Tiers

- Unmarked (default): unit tests, < ~1 s each.
- `@pytest.mark.slow`: individually expensive tests (> ~1.5 s — heavy numerics, `jit`+`grad`+`vmap` sweeps, RFF-convergence checks).
- `@pytest.mark.integration`: end-to-end workflows across gaussx / geonnax / pyrox-gp. Usually combined with `slow`.

Run a subset with `uv run pytest -m "not slow and not integration"`.

## Tests That Assert On Random Draws

- **If the randomness is incidental** — the test checks a correctness property and any draw would do — pin the key (`jax.random.key(0)`). Deterministic makes the tolerance mean something.
- **If the test is genuinely about sampling behaviour** (an RFF Gram converging to the exact Gram, a randomized HSIC converging to the dense one), bound the estimator by its own sampling distribution rather than a fixed `atol`, and say in a comment where the bound came from.

## Documentation

The docs are built by **two tools** and deployed as one site — see
`docs/README.md` for the full rationale.

| Half | Tool | Source | Deployed at |
|---|---|---|---|
| Prose — home, guides, notebooks | mystmd | `docs/*.md`, `docs/guide/`, `docs/notebooks/` | `/` |
| API reference | MkDocs + mkdocstrings | `docs/api/` | `/reference/` |

```bash
make docs          # build both halves, assemble into public/, verify links
make docs-api      # API reference only (fast; no Node needed)
make docs-serve    # build, then serve the assembled site at :8000
```

`scripts/build_docs.py` orchestrates this. It serves the freshly built
`site/` on port 8910 so mystmd can read the `objects.inv`, rewrites the
resulting localhost URLs to `/reference/`, repairs anchors broken by the
mystmd `$`-expansion bug, and then verifies that every internal link in the
assembled site resolves. Its pure functions are covered by
`tests/test_build_docs.py`.

**mystmd is a Node CLI**: `npm install -g mystmd`. It is not a uv dependency.

### Writing prose

Prose pages are **MyST Markdown**, not MkDocs-Material Markdown. Use
`:::{note}` / `:::{tab-set}` / `:::{dropdown}` directives, not `!!!` / `===`
/ `???` blocks.

Cross-reference the API with the `xref:` protocol and the **top-level**
exported name:

```markdown
[`RBF`](xref:api#kernellib.RBF)              <!-- correct -->
[`RBF`](xref:api#kernellib._kernels.RBF)     <!-- avoid: see docs/README.md -->
```

A target missing from the inventory fails `myst build --strict`.

### URLs are flat

mystmd derives a page's URL from its **basename**, so `guide/architecture.md`
is served at `/architecture/`, not `/guide/architecture/`. Keep basenames
unique across `docs/guide/` and `docs/notebooks/`. Frontmatter `slug:` is
ignored.

## Documentation Examples

Example notebooks live in `docs/notebooks/` as executed `.ipynb` files with
their outputs committed; mystmd renders them without re-executing. Author
them in jupytext percent format, execute, then delete the `.py`.

See `.github/instructions/docs-examples.instructions.md` for full standards.

## Coding Conventions

- Kernels, feature maps and estimators are `equinox.Module` subclasses (immutable, PyTree-compatible); hyperparameters are plain array fields with no transforms or priors
- Dependence measures and helpers are pure functions
- Use `jaxtyping` annotations for array shapes
- Use `einx` for tensor reshaping/contraction — no raw `jnp.reshape`/`jnp.transpose`/`jnp.einsum`
- Google-style docstrings with executable `Examples:` blocks
- Type hints on all public functions and methods
- Surgical changes only — don't refactor adjacent code or add docstrings to unchanged code

## Plans

Plans go in `.plans/` (gitignored, never committed). Track work via GitHub
issues. Long-lived design references that the code must agree with go in
`design_docs/`.

## PR Review Comments

When addressing PR review comments, always resolve each review thread after fixing it via the GitHub GraphQL API (`resolveReviewThread` mutation). Do not leave addressed comments unresolved. To obtain the required `threadId`, first list the pull request's review threads via the GitHub GraphQL API (see the "Pull Request Review Comments" section in `AGENTS.md` for a minimal query and end-to-end workflow).

## Code Review

Follow the guidance in `/CODE_REVIEW.md` for all code review tasks.
