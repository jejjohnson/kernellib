#!/usr/bin/env bash
#
# Apply native GitHub issue relationships via the GraphQL API.
#
# Supports two relationship types:
#   - sub-issues (parent <-> child hierarchy)
#   - blocked-by / blocks (typed dependency)
#
# Usage:
#   bash .github/scripts/link-issues.sh sub <parent> <child> [<child> ...]
#   bash .github/scripts/link-issues.sh block <issue> <blocking-issue>
#   bash .github/scripts/link-issues.sh unsub <parent> <child>
#   bash .github/scripts/link-issues.sh unblock <issue> <blocking-issue>
#   bash .github/scripts/link-issues.sh show <issue>
#
# All issue arguments are numeric (e.g. 42), resolved to node IDs automatically.
#
# Or via Makefile:
#   make gh-sub PARENT=7 CHILDREN="42 43 44"
#   make gh-block ISSUE=44 BLOCKED_BY=43
#
# Requires: `gh` CLI authenticated with repo scope.

set -euo pipefail

if ! command -v gh >/dev/null 2>&1; then
  echo "error: gh CLI not found on PATH. Install from https://cli.github.com/" >&2
  exit 1
fi

if ! gh auth status >/dev/null 2>&1; then
  echo "error: gh is not authenticated. Run 'gh auth login' first." >&2
  exit 1
fi

# Resolve issue number -> node ID in the current repo.
node_id_for() {
  local number="$1"
  gh api "repos/:owner/:repo/issues/${number}" --jq '.node_id' 2>/dev/null \
    || { echo "error: could not resolve issue #${number} — wrong repo or number?" >&2; return 1; }
}

# Link one child under a parent. Idempotent: reports no-op if already linked.
add_sub_issue() {
  local parent="$1" child="$2"
  local parent_id child_id
  parent_id=$(node_id_for "$parent")
  child_id=$(node_id_for "$child")

  local out
  if out=$(gh api graphql \
      -f query='mutation($p: ID!, $c: ID!) { addSubIssue(input: {issueId: $p, subIssueId: $c}) { subIssue { number } } }' \
      -f p="$parent_id" -f c="$child_id" 2>&1); then
    printf '  ✓ sub-issue: #%s -> parent #%s\n' "$child" "$parent"
  else
    if grep -qiE "already|duplicate" <<<"$out"; then
      printf '  • no-op: #%s is already a sub-issue of #%s\n' "$child" "$parent"
    else
      printf '  ✗ failed #%s -> #%s: %s\n' "$child" "$parent" "$out" >&2
      return 1
    fi
  fi
}

remove_sub_issue() {
  local parent="$1" child="$2"
  local parent_id child_id
  parent_id=$(node_id_for "$parent")
  child_id=$(node_id_for "$child")
  gh api graphql \
    -f query='mutation($p: ID!, $c: ID!) { removeSubIssue(input: {issueId: $p, subIssueId: $c}) { issue { number } } }' \
    -f p="$parent_id" -f c="$child_id" >/dev/null
  printf '  ✓ removed sub-issue link: #%s -/> #%s\n' "$child" "$parent"
}

# Mark `issue` as blocked by `blocker`.
add_blocked_by() {
  local issue="$1" blocker="$2"
  local issue_id blocker_id
  issue_id=$(node_id_for "$issue")
  blocker_id=$(node_id_for "$blocker")

  local out
  if out=$(gh api graphql \
      -f query='mutation($i: ID!, $b: ID!) { addBlockedBy(input: {issueId: $i, blockingIssueId: $b}) { issue { number } } }' \
      -f i="$issue_id" -f b="$blocker_id" 2>&1); then
    printf '  ✓ #%s is now blocked by #%s\n' "$issue" "$blocker"
  else
    if grep -qiE "already|duplicate" <<<"$out"; then
      printf '  • no-op: #%s is already blocked by #%s\n' "$issue" "$blocker"
    else
      printf '  ✗ failed #%s blocked-by #%s: %s\n' "$issue" "$blocker" "$out" >&2
      return 1
    fi
  fi
}

remove_blocked_by() {
  local issue="$1" blocker="$2"
  local issue_id blocker_id
  issue_id=$(node_id_for "$issue")
  blocker_id=$(node_id_for "$blocker")
  gh api graphql \
    -f query='mutation($i: ID!, $b: ID!) { removeBlockedBy(input: {issueId: $i, blockingIssueId: $b}) { issue { number } } }' \
    -f i="$issue_id" -f b="$blocker_id" >/dev/null
  printf '  ✓ removed blocked-by: #%s no longer blocked by #%s\n' "$issue" "$blocker"
}

# Print parent / sub-issues / blocking / blocked-by for an issue.
show_issue() {
  local number="$1"
  local owner repo
  owner=$(gh repo view --json owner --jq '.owner.login')
  repo=$(gh repo view --json name --jq '.name')
  gh api graphql \
    -f query='
query($o: String!, $n: String!, $num: Int!) {
  repository(owner: $o, name: $n) {
    issue(number: $num) {
      number title state
      parent { number title state }
      subIssues(first: 50) { nodes { number title state } }
      subIssuesSummary { total completed percentCompleted }
      blockedBy(first: 50) { nodes { number title state } }
      blocking(first: 50) { nodes { number title state } }
    }
  }
}' -f o="$owner" -f n="$repo" -F num="$number"
}

usage() {
  sed -n '2,/^$/p' "$0" | sed 's/^# \{0,1\}//'
  exit 1
}

cmd="${1:-}"; shift || true
case "$cmd" in
  sub)
    [[ $# -ge 2 ]] || usage
    parent="$1"; shift
    for child in "$@"; do
      add_sub_issue "$parent" "$child"
    done
    ;;
  unsub)
    [[ $# -eq 2 ]] || usage
    remove_sub_issue "$1" "$2"
    ;;
  block)
    [[ $# -eq 2 ]] || usage
    add_blocked_by "$1" "$2"
    ;;
  unblock)
    [[ $# -eq 2 ]] || usage
    remove_blocked_by "$1" "$2"
    ;;
  show)
    [[ $# -eq 1 ]] || usage
    show_issue "$1"
    ;;
  *)
    usage
    ;;
esac
