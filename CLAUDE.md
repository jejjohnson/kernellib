# CLAUDE.md

The rules for every agent live in `AGENTS.md`; this file adds only what is
specific to Claude Code.

@AGENTS.md

## Claude Code specifics

- **Reuse first.** Search `docs/api/capabilities.md` (every public kernellib
  name, plus gaussx and geonnax's random features and bases) before writing a
  helper, a kernel, a feature map or any linear algebra; the "Reuse before
  you write" table in `AGENTS.md` maps the usual hand-rolled code to what
  already exists.
- **Skills** in `.claude/skills/` load on their own when a task matches
  their description (or run them as `/<name>`):
  - building: `add-kernel`, `add-feature-map`, `add-kernel-operator`,
    `add-estimator`, `add-dependence-measure`, `add-graph-component`,
    `add-notebook`, `bump-upstream-pins`;
  - shipping: `pre-pr-check`, `kernellib-review`, `squash-commit`;
  - GitHub housekeeping: `create-gh-issue`, `link-gh-issues`.
- **Subagents** (`.claude/agents/`), both read-only, both used by
  `kernellib-review`; run them on any diff that adds code, before
  committing:
  - `reuse-reviewer`: does the diff re-implement something in
    `docs/api/capabilities.md` (kernellib, gaussx, geonnax)?
  - `numerics-reviewer`: gradients at coincident points, traced control
    flow, dtypes, densified matrix-free paths, false tags, PRNG misuse,
    silent non-convergence.
- **GitHub.** When the `gh` CLI is unavailable, use the GitHub MCP tools for
  the same operations (PRs, issues, review threads, check runs).
