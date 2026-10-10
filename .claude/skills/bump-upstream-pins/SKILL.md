---
name: bump-upstream-pins
description: Bump kernellib's git-pinned upstreams — gaussx (tag in [tool.uv.sources] plus a >= floor) and geonnax (a direct git URL) — across pyproject.toml, uv.lock and the capability index, then fix what the new versions break. Use when asked to upgrade gaussx or geonnax, after an upstream release, or when a new upstream feature is needed.
---

# Bump gaussx / geonnax

Neither is on PyPI, so each is pinned to a release tag and a bump touches
several places that must agree.

## 1. Where the pins live

| Upstream | Where |
|---|---|
| gaussx | `pyproject.toml`: the `gaussx>=X.Y.Z` floor in `dependencies` **and** the `tag` in `[tool.uv.sources]` |
| geonnax | `pyproject.toml`: the direct reference `geonnax @ git+https://github.com/jejjohnson/geonnax.git@vX.Y.Z` |
| both | `uv.lock`; `docs/api/capabilities.md` (records the versions it was generated against) |

Downstream, pyrox pins kernellib and overrides gaussx to one tag; after a
gaussx bump here, mention in the PR that pyrox's override must follow.

## 2. Bump

1. Read the upstream changelog between the old and new tags; note
   removals, renames and deprecations (gaussx warns with
   `GaussxDeprecationWarning` and names the replacement).
2. Update the pins (floor = new version, tag = new tag).
3. `uv lock`, then `uv sync --all-groups --all-extras`.
4. `make capabilities`: the upstream sections and their versions change; a
   new upstream name may now shadow a kernellib one — rename, or add it to
   `ALLOWED_SHARED_NAMES` in `scripts/capabilities.py` with the reason.

## 3. Fix and verify

- `make test`, then
  `uv run pytest -n auto -m "" tests/operators tests/regression tests/spectral`
  (every tier of those directories) (the gaussx
  strategies, preconditioners and `LowRankUpdate`) and the feature maps
  (geonnax `randfeat`).
- Replace deprecated upstream calls rather than silencing the warning.
- If the upstream now provides something kernellib hand-rolls, note it in
  the PR; converting it is a separate change unless it is a one-liner.
- `make typecheck`, lint and format, then the `pre-pr-check` skill.

The PR title is `build(deps): bump gaussx to vX.Y.Z` (or both together).
