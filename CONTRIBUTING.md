# Contributing

See the full contributor guide at [docs/contributing.md](docs/contributing.md) for:

- Label taxonomy (`type:*`, `area:*`, `layer:*`, `wave:*`, `priority:*`)
- Two-layer epic model (**Wave → Theme → Issue**)
- Issue format conventions (`.github/ISSUE_TEMPLATE/`)
- Relationships syntax (`Parent:`, `Blocked by:`, `Blocks:`, `Related:`)
- Pre-commit checklist and quality gates

Bootstrap the standard label set for a fresh clone of this template:

```bash
make gh-labels
```

Commit messages follow the [Conventional Commits](https://www.conventionalcommits.org/) spec — enforced on PR titles by CI.
