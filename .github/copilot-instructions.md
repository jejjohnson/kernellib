# Copilot Instructions

## Project Overview

- **Python**: 3.12+
- **Package Manager**: uv
- **Domain**: kernels and scalable kernel methods for JAX, built on gaussx, equinox, lineax, einx, geonnax
- **Layout**: `src/` layout (`src/kernellib/`)
- **Testing**: pytest
- **Docs**: mystmd (prose, notebooks) + MkDocs/mkdocstrings (API reference), assembled by `scripts/build_docs.py`
- **Design**: `design_docs/kernellib/architecture.md` fixes the kernel contract, the layout, and the gaussx / pyrox-gp / geonnax boundaries

## Build & Test Commands

```bash
make install     # Install all dependencies (uv sync --all-groups)
make test        # Run tests (uv run pytest -v)
make lint        # Lint code (ruff check)
make format      # Format code (ruff format + ruff check --fix)
make typecheck   # Type check (ty check)
make precommit   # Run pre-commit on all files
make docs-serve  # Serve docs locally
```

## Before Every Commit — Mandatory Checklist

**All four checks must pass before any commit.** CI runs them on the entire repo (`ruff check .`), not just `src/kernellib/`, so always run the commands below from the repo root.

```bash
# 1. Tests + doctests — zero failures required
#    `--doctest-modules` is in addopts, so every `Examples:` block in a
#    docstring is executed. A stale example is a failing build.
uv run pytest -v

# 2. Lint — run on the ENTIRE repo (includes tests/ and scripts/)
uv run --group lint ruff check .

# 3. Format check — run on the ENTIRE repo
uv run --group lint ruff format --check .

# 4. Type check — on the package only
uv run --group typecheck ty check src/kernellib scripts
```

> **Common pitfall**: Running `ruff check src/kernellib/` instead of `ruff check .` misses import-sorting errors in `tests/` and `scripts/`. The CI workflow runs `ruff check .`. Always use `.` (repo root), not a subdirectory.

## Key Directories

| Path | Purpose |
|------|---------|
| `src/kernellib/` | Main package source code |
| `tests/` | Test suite |
| `design_docs/` | Committed design references |
| `docs/guide/` | Hand-written guide pages (**MyST** Markdown, not MkDocs-Material) |
| `docs/api/` | MkDocs API reference — the only half MkDocs builds |
| `docs/notebooks/` | Executed example notebooks (`.ipynb`, outputs committed) |
| `scripts/` | Build tooling, incl. the two-tool docs pipeline |

## Boundaries

- kernellib never imports `numpyro` or scikit-learn (`tests/test_imports.py`).
- Anything with a kernel in it lives here, from the kernel operators up; gaussx is kernel-agnostic linear algebra (structured operators, solvers, preconditioners, `trace_product`, `stable_squared_distances`).
- `kernellib.functional` is arrays in, arrays out; the top level takes kernels and data.
- Basis functions and random-feature arithmetic come from geonnax.

## Behavioral Guidelines

### Do Not Nitpick
- Ignore style issues that linters/formatters catch (formatting, import order, quote style)
- Don't suggest changes to code you weren't asked to modify
- Match existing patterns even if you'd do it differently

### Docstrings Are Tested
`Examples:` blocks in Google-style docstrings run under `--doctest-modules`.
When you change a function's behaviour, update its examples — and make sure
the expected output is what the code actually prints, not what it ought to.

### Always Propose Tests
When implementing features or fixing bugs:
1. Write a test that verifies the expected behavior
2. Implement the change
3. Verify the test passes

### Never Suggest Without a Proposal
Bad: "You should add validation here"
Good: "Add validation here. Proposed implementation:"
```python
if value < 0:
    raise ValueError("Value must be non-negative")
```

### Simplicity First
- No abstractions for single-use code
- No speculative features beyond what was asked
- If 200 lines could be 50, propose the simpler version

### Surgical Changes
- Only modify lines directly related to the request
- Don't refactor adjacent code
- Don't add docstrings/comments to code you didn't change
- Remove only imports/functions that YOUR changes made unused

## Plans

Plans go in `.plans/` (gitignored, never committed). Track work via GitHub issues. Long-lived design references go in `design_docs/`.

## PR Review Comments

When addressing PR review comments, always resolve each review thread after fixing it via the GitHub GraphQL API (`resolveReviewThread` mutation). Do not leave addressed comments unresolved. See the "Pull Request Review Comments" section in `AGENTS.md` for the exact GraphQL queries and workflow.

## Code Review

For all code review tasks, follow the guidance in `/CODE_REVIEW.md`.
