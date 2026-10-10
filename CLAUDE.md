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
- **GitHub.** When the `gh` CLI is unavailable, use the GitHub MCP tools for
  the same operations (PRs, issues, review threads, check runs).
