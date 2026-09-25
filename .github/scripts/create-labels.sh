#!/usr/bin/env bash
#
# Bootstrap the standard label taxonomy for this template.
#
# Creates labels for: type / area / layer / wave / priority.
# Idempotent — uses `gh label create --force` so re-runs are safe.
#
# Usage:
#   bash .github/scripts/create-labels.sh
#   or:
#   make gh-labels
#
# Requires: `gh` CLI authenticated with repo scope.
#
# Customise by editing the `create` calls below; run again to apply.

set -euo pipefail

if ! command -v gh >/dev/null 2>&1; then
  echo "error: gh CLI not found on PATH. Install from https://cli.github.com/" >&2
  exit 1
fi

# Fail early with a clear message if gh is not authenticated.
if ! gh auth status >/dev/null 2>&1; then
  echo "error: gh is not authenticated. Run 'gh auth login' first." >&2
  exit 1
fi

create() {
  local name="$1"
  local color="$2"
  local desc="$3"
  gh label create "$name" --color "$color" --description "$desc" --force >/dev/null
  printf '  ✓ %s\n' "$name"
}

echo "Creating type:* labels..."
create "type:epic-wave"   "5319e7" "Release-scoped mega-epic (L1)"
create "type:epic-theme"  "8b5cf6" "Parallel-safe theme under a wave (L2)"
create "type:feature"     "a2eeef" "New feature or enhancement"
create "type:bug"         "d73a4a" "Something isn't working"
create "type:design"      "fbca04" "Design / ADR issue for a new API"
create "type:chore"       "c5def5" "Engineering maintenance"
create "type:docs"        "0075ca" "Documentation work"
create "type:research"    "bfe5bf" "Comparative analysis / prior-art survey / investigation"

echo "Creating area:* labels..."
create "area:engineering" "0e8a16" "Build, packaging, CI, tooling"
create "area:testing"     "bfdadc" "Test suite"
create "area:docs"        "1d76db" "Docs + notebooks"
create "area:code"        "c2e0c6" "Source-code changes (catch-all fallback)"
create "area:algorithmic" "0052cc" "Numerical algorithms and their correctness"
# Edit / extend for your project. Common domain-specific areas:
#   area:algorithmic, area:integration, area:performance, area:security

echo "Creating Dependabot / misc labels..."
create "dependencies" "ededed" "Pull requests that update a dependency file"

echo "Creating layer:* labels (edit to match your project's layer stack)..."
create "layer:0-primitives" "fef2c0" "Layer 0 — kernel math, abstractions, composition"
create "layer:1-components" "fbca04" "Layer 1 — gaussx bridge, spectral side, heuristics"
create "layer:2-models"     "d4c5f9" "Layer 2 — estimators, dependence, embeddings"

echo "Creating wave:* labels (edit to match your release plan)..."
# Plain `wave:N` scheme — simple and matches the label format referenced in
# the epic issue templates. If you prefer descriptive slugs, add extras like
# `wave:0-bootstrap`, `wave:1-mvp` alongside the numeric base.
create "wave:0" "c2e0c6" "Wave 0"
create "wave:1" "bfd4f2" "Wave 1"
create "wave:2" "d4c5f9" "Wave 2"
create "wave:3" "f9d0c4" "Wave 3"
create "wave:4" "fef2c0" "Wave 4"

echo "Creating priority:* labels..."
create "priority:p0" "b60205" "Blocker for current wave"
create "priority:p1" "d93f0b" "High priority"
create "priority:p2" "fbca04" "Normal priority"

echo
echo "Done. View labels: gh label list"
