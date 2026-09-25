# =============================================================================
# kernellib Makefile
# =============================================================================
#
# PREREQUISITES:
#   - uv installed  (https://github.com/astral-sh/uv)
#   - git available in PATH
#   - Copy .env.example to .env and fill in any overrides (optional)
#
# QUICK START:
#   make help          # Show all available commands
#   make install       # Install all dependency groups
#   make test          # Run tests
#   make lint          # Lint with ruff
#   make format        # Format with ruff
#
# =============================================================================

# ---------------------------------------------------------------------------
# .env support
# Silently include the .env file; missing file is fine - guard targets catch it.
# ---------------------------------------------------------------------------
-include .env
ifneq (,$(wildcard .env))
ENV_VARS := $(shell grep -E '^[A-Za-z_][A-Za-z0-9_]*=' .env | cut -d= -f1 | xargs)
export $(ENV_VARS)
endif

# ---------------------------------------------------------------------------
# Calculated variables
# ---------------------------------------------------------------------------
GIT_HASH := $(shell git rev-parse --short HEAD 2>/dev/null || echo "unknown")
PKG_VERSION := $(shell grep -E '^version\s*=' pyproject.toml 2>/dev/null \
	| sed -E 's/.*"([^"]+)".*/\1/' || echo "unknown")

# ---------------------------------------------------------------------------
# Paths (override via .env or command line)
# ---------------------------------------------------------------------------
PKGROOT ?= src/kernellib
# Additional paths to type-check alongside the package.
TYPECHECK_EXTRA ?= scripts

# ---------------------------------------------------------------------------
# ANSI colours
# ---------------------------------------------------------------------------
BLUE   := \033[36m
YELLOW := \033[33m
GREEN  := \033[32m
RED    := \033[31m
RESET  := \033[0m

# ---------------------------------------------------------------------------
# Guard pattern — usage: add check-env-VARNAME as a prerequisite
# Example:  my-target: check-env-MY_VAR
# ---------------------------------------------------------------------------
check-env-%:
	@if [ -z "$($*)" ]; then \
		printf "$(RED)❌  $* is not set.$(RESET)\n"; \
		printf "$(YELLOW)   Add it to .env or pass inline: make <target> $*=value$(RESET)\n"; \
		exit 1; \
	fi

# ---------------------------------------------------------------------------
# Phony declarations
# ---------------------------------------------------------------------------
.PHONY: help install lint format typecheck test test-cov \
        precommit build clean version docs docs-check docs-api docs-serve \
        gh-labels gh-sub gh-block gh-show

.DEFAULT_GOAL := help

# ===========================================================================
##@ Meta
# ===========================================================================

help: ## 📚 Show this help menu
	@printf "$(YELLOW)🐍 kernellib$(RESET)\n"
	@printf "%s\n" "-----------------------------------------------------------"
	@awk 'BEGIN {FS = ":.*##"; printf ""} \
	     /^[a-zA-Z_-]+:.*?##/ { printf "  $(BLUE)%-18s$(RESET) %s\n", $$1, $$2 } \
	     /^##@/ { printf "\n$(YELLOW)%s$(RESET)\n", substr($$0, 5) } ' \
	     $(MAKEFILE_LIST)

version: ## 📋 Display package version and git hash
	@printf "$(YELLOW)📋 Version Info$(RESET)\n"
	@printf "%s\n" "-----------------------------------------------------------"
	@printf "$(GREEN)Package : $(PKG_VERSION)$(RESET)\n"
	@printf "$(BLUE)Git hash: $(GIT_HASH)$(RESET)\n"

# ===========================================================================
##@ Setup
# ===========================================================================

install: ## 📦 Install all dependency groups via uv + pre-commit hooks
	@printf "$(YELLOW)>>> Installing all dependencies...$(RESET)\n"
	uv sync --all-groups
	uv run pre-commit install
	@printf "$(GREEN)>>> ✅ Installation complete!$(RESET)\n"

# Convenience: copy .env.example → .env if .env is missing
init: ## 🔧 Bootstrap .env from .env.example (skip if .env already exists)
	@if [ -f .env ]; then \
		printf "$(YELLOW)>>> .env already exists — skipping.$(RESET)\n"; \
	else \
		cp .env.example .env; \
		printf "$(GREEN)>>> ✅ .env created from .env.example$(RESET)\n"; \
	fi

# ===========================================================================
##@ Quality
# ===========================================================================

lint: ## 🧹 Lint code with ruff (no auto-fix) — entire repo
	@printf "$(YELLOW)>>> Running ruff check...$(RESET)\n"
	uv run --group lint ruff check .
	@printf "$(GREEN)>>> ✅ Lint passed!$(RESET)\n"

format: ## 🖊️  Format code with ruff (format + auto-fix) — entire repo
	@printf "$(YELLOW)>>> Running ruff format + fix...$(RESET)\n"
	uv run --group lint ruff format .
	uv run --group lint ruff check --fix .
	@printf "$(GREEN)>>> ✅ Format complete!$(RESET)\n"

typecheck: ## 🔬 Type-check with ty
	@printf "$(YELLOW)>>> Running type checks...$(RESET)\n"
	uv run --group typecheck ty check $(PKGROOT) $(TYPECHECK_EXTRA)
	@printf "$(GREEN)>>> ✅ Type check passed!$(RESET)\n"

# ===========================================================================
##@ Testing
# ===========================================================================

test: ## 🧪 Run tests with pytest (no coverage)
	@printf "$(YELLOW)>>> Running tests (no coverage)...$(RESET)\n"
	uv run pytest -v -o addopts=--doctest-modules
	@printf "$(GREEN)>>> ✅ Tests passed!$(RESET)\n"

test-cov: ## 📊 Run tests with coverage report
	@printf "$(YELLOW)>>> Running tests with coverage...$(RESET)\n"
	uv run pytest -v
	@printf "$(GREEN)>>> ✅ Coverage report generated!$(RESET)\n"

# ===========================================================================
##@ Pre-commit
# ===========================================================================

precommit: ## 🪝 Run pre-commit hooks on all files
	@printf "$(YELLOW)>>> Running pre-commit...$(RESET)\n"
	uv run pre-commit run --all-files
	@printf "$(GREEN)>>> ✅ Pre-commit passed!$(RESET)\n"

# ===========================================================================
##@ Build
# ===========================================================================

build: ## 🏗️  Build Python wheel and sdist
	@printf "$(YELLOW)>>> Building package...$(RESET)\n"
	uv build
	@printf "$(GREEN)>>> ✅ Build complete — see dist/$(RESET)\n"

clean: ## 🗑️  Remove build artefacts and cache directories
	@printf "$(YELLOW)>>> Cleaning up...$(RESET)\n"
	rm -rf dist/ build/ .eggs/ *.egg-info
	rm -rf .pytest_cache/ .ruff_cache/ .mypy_cache/
	rm -rf site/ public/ docs/_build/
	rm -f .coverage coverage.xml
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@printf "$(GREEN)>>> ✅ Clean complete!$(RESET)\n"

# ===========================================================================
##@ Docs
# ===========================================================================

docs: ## 📖 Build the full site (MyST prose + MkDocs API) into public/
	uv run python scripts/build_docs.py

docs-check: ## ✅ Validate docs sources without rendering the MyST theme
	uv run python scripts/build_docs.py --check

docs-api: ## 📚 Build only the MkDocs API reference into site/
	uv run --group docs mkdocs build --strict

docs-serve: docs ## 🌐 Build, then serve the assembled site at :8000
	@printf "$(GREEN)>>> http://127.0.0.1:8000$(RESET)\n"
	uv run python -m http.server 8000 --directory public

gh-labels: ## 🏷️  Bootstrap the GitHub label taxonomy (type / area / layer / wave / priority)
	bash .github/scripts/create-labels.sh

gh-sub: ## 🔗 Link CHILDREN as sub-issues of PARENT (e.g. make gh-sub PARENT=7 CHILDREN="42 43 44")
	@test -n "$(PARENT)"   || { echo "error: PARENT=<issue-number> required"   >&2; exit 1; }
	@test -n "$(CHILDREN)" || { echo "error: CHILDREN=\"<a> <b> ...\" required" >&2; exit 1; }
	bash .github/scripts/link-issues.sh sub $(PARENT) $(CHILDREN)

gh-block: ## 🚧 Mark ISSUE as blocked by BLOCKED_BY (e.g. make gh-block ISSUE=44 BLOCKED_BY=43)
	@test -n "$(ISSUE)"      || { echo "error: ISSUE=<issue-number> required"      >&2; exit 1; }
	@test -n "$(BLOCKED_BY)" || { echo "error: BLOCKED_BY=<issue-number> required" >&2; exit 1; }
	bash .github/scripts/link-issues.sh block $(ISSUE) $(BLOCKED_BY)

gh-show: ## 🔍 Show parent / sub-issues / blocking / blocked-by for ISSUE
	@test -n "$(ISSUE)" || { echo "error: ISSUE=<issue-number> required" >&2; exit 1; }
	bash .github/scripts/link-issues.sh show $(ISSUE)
