---
name: pre-pr-check
description: Run kernellib's full pre-PR verification — ruff lint and format on the whole repo, ty on src/kernellib and scripts, the fast tier with doctests, the slow and integration tiers of what changed, the capability index, the lockfile and the docs build. Use before committing, pushing or opening a pull request, and after any change to public API, dependencies, docstrings or docs.
---

# Pre-PR check

Run from the repo root and fix what fails before committing. Report results
honestly: what ran, what passed, what was skipped and why.

## Always

```bash
uv run --group lint ruff check .          # entire repo: tests/, scripts/, notebook cells
uv run --group lint ruff format --check .
make typecheck                            # ty on src/kernellib and scripts (what CI runs)
make test                                 # fast tier + the fast doctests (--doctest-modules)
```

## The tiers of what you touched

CI runs the slow and integration tiers too, as separate jobs, and gates
coverage (90 %) on the union of all three; run them locally for the areas
you changed so CI is not the first to see them:

```bash
uv run pytest -n auto -m "slow and not integration" tests/<area>
uv run pytest -n auto -m integration tests/<area>   # scikit-learn checks, capability index
```

A kernel change runs `tests/test_kernels.py`, `tests/spectral` and
`tests/test_derivative_kernels.py`; an operator change `tests/operators`
and `tests/regression`; anything in `kernellib.sklearn`
`tests/sklearn`. `make test-cov` runs every tier with the coverage report.

## When the public API changed

```bash
make capabilities
uv run pytest -m "" tests/test_capabilities.py tests/test_public_api.py   # every tier
```

…plus the `::: kernellib.Name` entry on its `docs/api/*.md` page and the
name in its module's row of `docs/api/index.md`. `__all__` stays in RUF022 order.

## When dependencies changed

`uv lock` and commit `uv.lock`; for gaussx / geonnax, the
`bump-upstream-pins` skill. `import kernellib` loads none of numpyro,
scikit-learn, pynndescent, numba, pipekit and optax
(`tests/test_imports.py`, slow); optional backends are imported inside the
function that needs them, with an install hint.

## When docs or docstrings changed

- Every `Examples:` block runs: under `make test`, except those listed in
  `_SLOW_DOCTESTS` (`tests/conftest.py`, slow tier; list one that takes
  over a second) and those in `kernellib.sklearn` (integration tier).
- `make docs-api` (MkDocs, strict) for docstrings and `docs/api/`;
  `make docs-check` (strict, both halves) and `make docs` (the assembled
  site with link checking) for prose and notebooks — both need the mystmd
  CLI; if the theme download is blocked, see `docs/README.md`.

## Before pushing

- `git status` shows no stray files (`.plans/`, notebook `.py` drafts).
- Conventional Commits title with a lowercase subject; `!` and a
  `BREAKING CHANGE:` footer for a breaking change.
- Push only to your feature branch, only when asked.
