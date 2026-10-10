---
name: add-kernel
description: Add a kernel to kernellib — a stationary, pointwise or Gram-level AbstractKernel subclass (or a composite), with array hyperparameters, its kernellib.functional twin, optional spectral hooks, the ZOO tests, docs and capability index. Use when asked to add, port or implement a covariance function / kernel, or to give an existing kernel a spectral density, in src/kernellib/_kernels or kernellib.functional.
---

# Add a kernel

Read "1. Kernels" under "The contracts" in `AGENTS.md` first; this is the
step-by-step.

## 1. Make sure it does not exist yet

- Search `docs/api/capabilities.md` for the kernel and its synonyms (Matérn
  ν = 1/2 is the exponential / Laplace kernel; rational quadratic is a scale
  mixture of RBFs). A composition of existing kernels (`k1 + k2`, `c * k`,
  `.select(dims)`, `.stretch`, `.periodic(p)`, `Warped`) needs no new class.
- pyrox-gp wraps kernellib kernels with priors; a prior is **not** a new
  kernel here.

## 2. Pick the level (`src/kernellib/_kernels/`)

| It is… | Subclass | Implement | Exemplar |
|---|---|---|---|
| A function of the scaled squared distance r² | `AbstractStationaryKernel` | `shape(r2)`; fields `lengthscale`, `variance` | `RBF`, `Matern` (`_stationary.py`) |
| Defined point by point (non-stationary, periodic) | `AbstractPointwiseKernel` | `pairwise(x, y) -> ()` | `Linear`, `Polynomial` (`_nonstationary.py`), `Periodic` |
| Only defined on whole Grams | `AbstractKernel` | `__call__(X1, X2) -> (N1, N2)` | `Residual` (`_residual.py`), `Modulated` (`_feature.py`) |

## 3. Write it

- Hyperparameters: `eqx.field(default=..., converter=jnp.asarray)` array
  fields, defaults matching pyrox-gp's if pyrox-gp has the kernel. Settings
  that choose a code path are `eqx.field(static=True)`; validate them in
  `__check_init__` with a `ValueError`.
- Smooth in r²: compute through `_smooth_in_r2(r2, exact, taylor)` so the
  Hessian of `pairwise` is exact at x = y (derivative kernels rely on it).
  Rough (not twice differentiable at 0): add it to the rejection list in
  `_derivative._check_differentiable`.
- Override `diag(X)` when a closed form is cheaper than the base's default
  (stationary: `variance * ones`; pointwise: `vmap` of `pairwise(x, x)`;
  Gram-only: the whole Gram, so always override it there), and the private
  `_gram_structure(X)` if the Gram is diagonal or low rank (see `White`,
  `Constant`, `Linear`) so `to_operator` keeps it structured.
- Spectral support (stationary only, optional): implement
  `unit_spectral_density(omega_sq, d)` (Bochner, with (2π)^-D on the
  inverse, at unit lengthscale and variance) and
  `sample_unit_frequencies(key, shape, dtype)` drawing **joint** multivariate
  frequencies. The base class supplies `spectral_density`,
  `sample_frequencies(key, n, d)` and `spectral_variance`.
- If PSD only in 1-D (as `Periodic` and `Cosine` are), say so in the
  docstring and point at `Periodised` for D > 1.
- The `kernellib.functional` twin (`functional/_stationary.py` or
  `_nonstationary.py`): `name_kernel(X1, X2, variance, lengthscale, ...)` —
  **variance first** — on 2-D arrays; export it from `functional/__init__.py`.
- Docstring: the formula, the hyperparameters, PSD conditions, a reference,
  and an executable `Examples:` block (it runs under `--doctest-modules`;
  print rounded values or shapes, and check the output is what the code
  prints).
- einx for every dense transpose, reshape, axis reduction and inserted-axis
  broadcast.

## 4. Export and document

- `_kernels/__init__.py`, then `src/kernellib/__init__.py` (import and
  `__all__`, kept in RUF022 order — `tests/test_public_api.py` checks it).
- `::: kernellib.NewKernel` on `docs/api/kernels.md` (and the functional
  twin on `docs/api/functional.md` if it is listed there), the name in its module's row of
  `docs/api/index.md`, and the spectral table on `docs/api/spectral.md` if
  it has spectral support.
- `make capabilities`.

## 5. Tests

- Pointwise and stationary kernels with a `functional` twin: add a
  `pytest.param(kernel, reference, psd)` to `ZOO` in `tests/test_kernels.py`
  (with `psd` set honestly): `TestZoo` then checks agreement with `functional`,
  symmetry, PSD, `diag`, `is_pointwise` and pointwise-vs-Gram, gradients
  (slow), `filter_jit` and `vmap`. A Gram-only kernel cannot pass `ZOO` (it
  asserts `is_pointwise`); test it on its own, as `test_gram_only_kernel`
  does.
- Kernel-specific values against a hand computation or a published
  formula, and the validation errors.
- Spectral: in `tests/spectral/test_density.py`, the closed form, Bochner
  inversion against the kernel, (2π)^-D × total mass = variance, and the sampler
  reproducing the kernel within a sampling bound stated in a comment.
- Derivative kernels work (or reject it, if rough) — `tests/test_derivative_kernels.py`.

## 6. Verify

`make test`, `uv run pytest -m slow tests/test_kernels.py tests/spectral
tests/test_derivative_kernels.py`,
then the `pre-pr-check` skill.
