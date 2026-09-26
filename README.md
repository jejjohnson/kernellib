# kernellib

[![Tests](https://github.com/jejjohnson/kernellib/actions/workflows/ci.yml/badge.svg)](https://github.com/jejjohnson/kernellib/actions/workflows/ci.yml)
[![Lint](https://github.com/jejjohnson/kernellib/actions/workflows/lint.yml/badge.svg)](https://github.com/jejjohnson/kernellib/actions/workflows/lint.yml)
[![Type Check](https://github.com/jejjohnson/kernellib/actions/workflows/typecheck.yml/badge.svg)](https://github.com/jejjohnson/kernellib/actions/workflows/typecheck.yml)
[![Deploy Docs](https://github.com/jejjohnson/kernellib/actions/workflows/pages.yml/badge.svg)](https://github.com/jejjohnson/kernellib/actions/workflows/pages.yml)
[![codecov](https://codecov.io/gh/jejjohnson/kernellib/branch/main/graph/badge.svg)](https://codecov.io/gh/jejjohnson/kernellib)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://pre-commit.com/)

Author: J. Emmanuel Johnson
Repo: [https://github.com/jejjohnson/kernellib](https://github.com/jejjohnson/kernellib)
Website: [jejjohnson.netlify.com](https://jejjohnson.netlify.com)

Kernels and scalable kernel methods for JAX. kernellib owns kernel functions
and their composition, spectral densities and feature maps, kernel ridge
regression, dependence measures (HSIC, CKA, MMD), kernel embeddings, and
kernel derivatives. It does **not** own Gaussian processes, priors, or
inference: those live in [pyrox-gp](https://github.com/jejjohnson/pyrox).

Scale is inherited rather than reimplemented. Every kernel can be turned into a
[gaussx](https://github.com/jejjohnson/gaussx) linear operator, and every
algorithm here solves through gaussx's solver strategies (dense, CG, BBMM,
preconditioned CG).

```
lineax · matfree · equinox · einx          geonnax
          │                                   │
          ▼                                   │
        gaussx   ◄────────────────────────────┤   structured operators, solvers
          │                                   │
          ▼                                   │
       kernellib ◄────────────────────────────┘   kernels, kernel methods   ← this package
          │
          ▼
       pyrox-gp                                   GP models with NumPyro priors
```

Two rules hold the chain together: gaussx never imports kernellib, and
kernellib never imports NumPyro.

> **Status: scaffold.** This repository was reset from a 2018 numpy /
> scikit-learn package to a fresh JAX package with the tooling below. The
> public API lands in phases described in the architecture design document
> (`design_docs/kernellib/architecture.md`). Nothing beyond `__version__` is
> exported yet.

---

## 📂 Repository Layout

```
kernellib/
├── src/kernellib/                    # Main package code (src layout)
│   ├── __init__.py                   # Public API re-exports (the contract)
│   └── py.typed                      # PEP 561 typing marker
├── tests/                            # pytest test suite
├── docs/                             # Documentation source
│   ├── myst.yml                      # MyST project (prose half)
│   ├── README.md                     # How the two-tool docs build works
│   ├── guide/                        # Guide pages (MyST Markdown)
│   └── api/                          # mkdocstrings API reference (MkDocs)
├── design_docs/                      # Committed design references
├── .github/
│   ├── workflows/                    # GitHub Actions CI/CD workflows
│   ├── instructions/                 # Copilot custom instructions
│   ├── ISSUE_TEMPLATE/               # Issue templates + epic hierarchy
│   ├── copilot-instructions.md       # Copilot behavioural config
│   ├── dependabot.yml                # Automated dependency PRs
│   └── labeler.yml                   # Automatic PR labelling rules
├── scripts/build_docs.py             # Builds, assembles & link-checks the docs
├── pyproject.toml                    # Single source of truth for project metadata & tools
├── uv.lock                           # Fully reproducible lockfile
├── Makefile                          # Self-documenting task runner
├── mkdocs.yml                        # API-reference site configuration
├── .pre-commit-config.yaml           # Git hook definitions
├── release-please-config.json        # Automated release & changelog config
├── .release-please-manifest.json     # Tracks the current released version
├── .env.example                      # Template for local environment variables
├── AGENTS.md                         # Standing instructions for AI coding agents
├── CLAUDE.md                         # Claude Code guidance
├── CODE_REVIEW.md                    # Code review standards
└── CHANGELOG.md                      # Auto-generated changelog
```

---

## 🚀 Quick Start

```bash
# Prerequisites: uv (https://github.com/astral-sh/uv)
git clone https://github.com/jejjohnson/kernellib.git
cd kernellib
make install      # install all dependency groups + pre-commit hooks
make test         # run tests + doctests
make docs-serve   # preview docs locally
```

kernellib is not on PyPI yet. Until it is, install from the repository:

```bash
uv add "kernellib @ git+https://github.com/jejjohnson/kernellib.git"
```

`gaussx` is resolved from its GitHub release tag and `geonnax` from a pinned
git tag (see `[tool.uv.sources]` and `dependencies` in `pyproject.toml`), so
both install without any extra configuration.

The scikit-learn adapters (`kernellib.sklearn`) are an optional extra:

```bash
uv add "kernellib[sklearn] @ git+https://github.com/jejjohnson/kernellib.git"
```

---

## 🧭 Where things go

The rule in one sentence: **anything with a kernel in it lives in kernellib,
from the operators up; gaussx is kernel-agnostic linear algebra; geonnax is
kernel-agnostic feature-map arithmetic.**

| Family | `kernellib.functional` / operators (arrays in) | `kernellib` top level (kernels in) |
|---|---|---|
| Kernels | `rbf_kernel`, ..., `KernelOperator`, `ImplicitKernelOperator` | `RBF`, `Matern`, ..., composition, `to_operator` |
| Approximation | `nystrom_operator`, `rff_operator`, `fastfood_operator` | `NystromFeatures`, `RandomFourierFeatures`, `FastFoodFeatures` |
| Dependence | `functional.hsic`, `functional.cka`, `functional.mmd_squared` | `hsic`, `cka`, `mmd` on kernels and data, randomized variants |
| Regression | Falkon and EigenPro primitives | `KRR`, `Falkon`, `EigenPro` estimators |

gaussx keeps the kernel-agnostic pieces these build on: structured operators
such as `LowRankUpdate`, the solver strategies, preconditioners,
`trace_product`, and `stable_squared_distances`. gaussx's current kernel
layer moves here in gaussx 0.2.0.

---

## 🛠️ Development

```bash
make help         # every target, grouped
make format       # ruff format . + ruff check --fix .
make lint         # ruff check .
make typecheck    # ty check src/kernellib scripts
make test         # pytest (tests + doctests)
make test-cov     # pytest with the coverage gate
make docs         # build both halves of the docs and verify links
```

Commit messages and PR titles follow
[Conventional Commits](https://www.conventionalcommits.org/); releases are
cut by Release Please. See [CONTRIBUTING.md](CONTRIBUTING.md) for the label
taxonomy and issue conventions.

---

## 📚 Related projects

| Project | Role |
|---|---|
| [gaussx](https://github.com/jejjohnson/gaussx) | Structured linear algebra, solvers, Gaussian primitives |
| [pyrox](https://github.com/jejjohnson/pyrox) | Equinox-NumPyro bridge, GP models (`pyrox-gp`), Bayesian NN layers (`pyrox-nn`) |
| [geonnax](https://github.com/jejjohnson/geonnax) | Basis functions and random-feature primitives |

## License

MIT. See [LICENSE](LICENSE).
