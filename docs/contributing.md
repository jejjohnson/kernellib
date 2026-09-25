# Contributing

This page documents the label taxonomy, epic model, and issue conventions used in this project.

---

## Label Taxonomy

Every issue carries exactly one `type:*`, one or more `area:*`, at most one `layer:*`, one `wave:*`, and one `priority:*`.

| Scope | Labels |
|---|---|
| **Type** | `type:epic-wave`, `type:epic-theme`, `type:feature`, `type:design`, `type:chore`, `type:docs`, `type:bug`, `type:research` |
| **Area** | `area:engineering`, `area:testing`, `area:docs`, `area:code` — extend per project (e.g. `area:algorithmic`, `area:integration`) |
| **Layer** | `layer:0-primitives`, `layer:1-components`, `layer:2-models` — only if the project has a formal layer stack |
| **Wave** | `wave:0`, `wave:1`, `wave:2`, … — release-scoped phases (add `wave:N-<slug>` variants for descriptive labels) |
| **Priority** | `priority:p0` (blocker), `priority:p1` (high), `priority:p2` (normal) |

Bootstrap the standard set on a fresh repo:

```bash
make gh-labels
```

The script lives at `.github/scripts/create-labels.sh`. Edit the hard-coded `create …` entries in the script to customise `area:*`, `layer:*`, and `wave:*` for the project, then re-run — the script is idempotent.

---

## Two-Layer Epic Model

Work is organised as **Wave → Theme → Issue**:

```
[EPIC] Wave N   (L1)   —   release-scoped mega-epic, owns a milestone
  ├── [EPIC] <theme>   (L2)   —   parallel-safe group of issues
  │     ├── feature / design / chore / bug issue
  │     └── …
  └── [EPIC] <theme>   (L2)
        └── …
```

- **Wave epic** (`type:epic-wave`) maps one-to-one with a milestone (`vX.Y-<slug>`). Groups several Theme epics that can run in parallel.
- **Theme epic** (`type:epic-theme`) groups concrete issues that ship together as a coherent slice. Themes inside a wave are parallel-safe unless the body says otherwise.
- **Issue** (feature / design / chore / bug) is the leaf — a single substantial deliverable that rolls up to a Theme epic.

Stragglers are discouraged: any `type:feature` issue should attach to a Theme epic. If no suitable theme exists, create one first.

---

## Issue Format

Issue templates live in `.github/ISSUE_TEMPLATE/`:

| Template | When to use |
|---|---|
| `Epic — Wave (L1)` | Opening a new release-scoped wave |
| `Epic — Theme (L2)` | Grouping related issues inside a wave |
| `Feature / Enhancement` | One substantial deliverable |
| `Design / ADR` | Resolve an open design question for a new API |
| `Bug report` | Something isn't working |
| `Research / Comparative Analysis` | Study prior art (external repo, paper) and produce a prioritized roadmap of follow-up issues |

Feature + Design templates include two **optional** sections for context-heavy issues:

- **Design Snapshot** — paste API sketches, code examples, or excerpts from private / external design docs so the issue is self-contained.
- **Mathematical Notes** — equations, sign conventions, numerical considerations.

Delete either section if not relevant. Both exist so that an implementer (human or AI agent) can work on the issue without opening other repos or channels.

---

## Creating issues

New issues should use one of the templates in `.github/ISSUE_TEMPLATE/` so that labels, required sections, and title conventions are consistent.

### From the UI

Click **New issue** on the repo's Issues tab and pick a template. The body is pre-filled with the section scaffolding — fill in, apply labels and milestone, submit.

### From the CLI

For drafted content, write the body to a file (minus the template's YAML frontmatter), then:

```bash
gh issue create \
  --title "feat(primitives): diagonal BLR update step" \
  --body-file /tmp/issue-body.md \
  --label "type:feature,area:algorithmic,layer:0-primitives,wave:1,priority:p1" \
  --milestone "v0.1-diagonal"
```

To open the UI in a browser with a template pre-selected:

```bash
gh issue create --web
```

The web flow opens `/issues/new/choose` so you can pick the template in the UI. `gh issue create` does not support pre-selecting a template for the web path; the `--template` flag is only used as starting body text in the non-web flow.

After the issue is created, apply native parent / blocked-by links via the Makefile targets documented below in the Relationships section (or `make gh-sub` / `make gh-block`).

### From Claude Code

The `create-gh-issue` skill (at `.claude/commands/create-gh-issue.md`) guides Claude through:

- **Picking the right template** (decision tree covering feature / design / bug / research / epic-wave / epic-theme)
- **Drafting the body** with required sections + rename guidance for optional sections (`Design Snapshot` → `Demo To Implement`, etc.)
- **Applying the correct label set** (which `type:*` + `area:*` + `layer:*` + `wave:*` + `priority:*` combos are valid for each work type)
- **Setting the milestone** (which `vX.Y-<slug>` corresponds to the wave)
- **Applying relationships** (chains to `link-gh-issues` skill for native sub-issue + blocked-by links)
- **Style conventions** — unicode math, `text` code fences, code-first-prose-last
- **Bulk workflow** — publishing a `.plans/<wave>-backlog.md` file into real GitHub issues (multi-pass: wave epic → theme epics → leaf issues → relationships), with draft-ID → GH-number mapping

Example prompts:

- "Open a feature issue for the diagonal BLR update step, child of theme epic #27"
- "File a design issue for the per-parameter prior question"
- "Publish the drafts in `.plans/wave-1-backlog.md` as real issues and wire up the sub-issue links"
- "Open a research issue comparing pof vs this project, based on my notes in `<path>`"

---

## Drafting a wave backlog

For large planning exercises (new wave, new release, large refactor), draft the whole backlog as one markdown file **before** opening GitHub issues. A template lives at [`docs/templates/wave-backlog.md`](https://github.com/jejjohnson/kernellib/blob/main/docs/templates/wave-backlog.md).

Why:

- **Review the whole wave in one scroll** instead of clicking through 15 half-drafted issues
- **Share context once** at the top (Shared Context · Design Snapshot · Intended Package Layout · Runtime Boundary) rather than duplicating across every child issue
- **Stable draft IDs** (`<PREFIX>-01`, `<PREFIX>-02`, …) let child issues reference each other before GitHub issue numbers exist

Workflow:

1. Copy `docs/templates/wave-backlog.md` into your project's `.plans/` directory (gitignored). Rename to describe the wave — e.g. `.plans/wave-1-backlog.md`.
2. Pick a short project prefix (e.g. `PYX` for pyrox, `OBX` for optax_bayes). Number drafts sequentially.
3. Fill in shared context at the top, then draft each issue body.
4. When the file is ready, open each draft as a real GitHub issue using the matching `.github/ISSUE_TEMPLATE/`. Copy the body verbatim.
5. Record GitHub issue numbers next to draft IDs in the backlog file, or replace draft IDs throughout. Update cross-references.
6. Either delete the backlog file or archive it in `.plans/archive/`.

---

## Relationships

Use an explicit `## Relationships` block at the bottom of each issue / epic body:

```markdown
## Relationships
- Parent: #<theme-epic>
- Blocked by: #
- Blocks: #
- Related: #
```

GitHub's task-list feature links bidirectionally from the parent, so checklist items in a Theme epic body auto-show in the referenced issues.

### Applying native GitHub relationships

The prose `## Relationships` block above is a human-readable record. GitHub also exposes two **native** relationship features that power the UI sub-issue panel and the dependency graph — **apply both**:

| Prose line | Native feature | Mutation |
|---|---|---|
| `Parent:` / `Theme Epics` / `Issues` checklist | Sub-issues (parent ↔ child hierarchy) | `addSubIssue` / `removeSubIssue` |
| `Blocked by:` | Typed dependency | `addBlockedBy` |
| `Blocks:` | Inverse — applied on the OTHER issue | `addBlockedBy` (on the blocked side) |
| `Related:` | No native feature | prose only |

#### From the UI

- **Sub-issues** — open the parent issue, side-panel → *Sub-issues* → *Create sub-issue* or *Add existing sub-issue*.
- **Blocked by** — open the blocked issue, side-panel → *Development* → *Mark as blocked by*.

#### From the CLI (Makefile targets)

For scripted work, the repo ships a helper script wrapped by three `make` targets:

```bash
# Link children as sub-issues under a parent
make gh-sub PARENT=7 CHILDREN="42 43 44"

# Mark issue 44 as blocked by 43
make gh-block ISSUE=44 BLOCKED_BY=43

# Inspect parent / sub-issues / blocking / blocked-by for an issue
make gh-show ISSUE=44
```

Direct script usage (same functionality plus unlink commands):

```bash
bash .github/scripts/link-issues.sh sub     <parent> <child> [<child> ...]
bash .github/scripts/link-issues.sh block   <issue> <blocker>
bash .github/scripts/link-issues.sh unsub   <parent> <child>
bash .github/scripts/link-issues.sh unblock <issue> <blocker>
bash .github/scripts/link-issues.sh show    <issue>
```

The script resolves issue numbers to GraphQL node IDs automatically, and treats "already linked" as a no-op so re-runs are safe.

#### From Claude Code

The `link-gh-issues` skill (at `.claude/commands/link-gh-issues.md`) guides Claude through the same operations — useful for bulk-applying relationships from a drafted wave backlog, parsing the `Issues` checklist out of a theme epic body, or verifying that the native links match the prose.

Example prompts:

- "Apply the relationships from epic #29"
- "Link #42, #43, #44 as sub-issues of #7"
- "Mark #44 as blocked by #43"
- "What's blocking #44?" (Claude runs `make gh-show ISSUE=44` or the underlying GraphQL query)

#### Raw GraphQL (for reference)

Under the hood the helper runs:

```graphql
mutation { addSubIssue(input: { issueId: "<parent-node-id>", subIssueId: "<child-node-id>" }) { subIssue { number } } }
mutation { addBlockedBy(input: { issueId: "<this-node-id>", blockingIssueId: "<blocker-node-id>" }) { issue { number } } }
```

Issue node IDs are resolved via `gh api repos/:owner/:repo/issues/<N> --jq .node_id`.

---

## Pre-commit checklist

Run these locally before opening a PR:

```bash
make format       # ruff format . + ruff check --fix .   (applies changes)
make lint         # ruff check .                         (CI-style check)
make typecheck    # ty check
make test         # pytest
```

Note that `make format` **mutates files** — it formats and applies autofixes. `make lint` is the CI-parity read-only check. Run `make format` first, commit the result, then run `make lint` / `make test` to verify.

Pre-commit hooks run ruff on every commit. Run `make precommit` to apply them to all files manually.

Commit messages follow the [Conventional Commits](https://www.conventionalcommits.org/) specification — enforced on PR titles by `.github/workflows/conventional-commits.yml`.
