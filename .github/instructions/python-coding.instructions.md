---
applyTo: "src/**/*.py,tests/**/*.py,scripts/**/*.py"
---

# Python Coding Standards

## Modern Python (3.12+)

- `from __future__ import annotations` at the top of every module
- Type hints on **all** public functions, methods, and module-level variables
- Modern union syntax: `X | None` not `Optional[X]`, `X | Y` not `Union[X, Y]`
- Built-in generics: `list[int]`, `dict[str, Any]` not `List[int]`, `Dict[str, Any]`
- `pathlib.Path` over `os.path`
- f-strings for string formatting
- `equinox.Module` for kernels, feature maps, operators and estimators (immutable pytrees; a dataclass is not one); NamedTuples for small results
- `Enum` for fixed sets of constants
- Context managers (`with` statements) for resource handling
- Specific exception types (never bare `except:`)
- Proper exception chaining (`raise ... from ...`)
- Early returns / guard clauses to reduce nesting

## Package Preferences

No new runtime dependency without discussion; build on what kernellib
already depends on (see "Boundaries" in `AGENTS.md`).

| Purpose | Preferred Package |
|---------|-------------------|
| Modules / pytrees | `equinox` |
| Linear operators, solvers, preconditioners | `lineax`, `gaussx` |
| Random-feature arithmetic, bases | `geonnax` |
| Axis-naming array ops | `einx`, through `kernellib._einx` |
| Shape annotations | `jaxtyping` |
| Path handling | `pathlib` (stdlib) |
| Testing | `pytest` |

## Documentation

- Module-level docstrings explaining purpose
- Function/method docstrings for all public APIs (Google style)
- Inline comments explaining *why*, not *what*
- Scientific algorithms should include Unicode equations in docstrings (e.g. `# σ² = Σ(xᵢ − μ)² / N`)
- Public classes and functions include an executable `Examples:` block (run by `--doctest-modules`; verify the expected output against what the code prints)
