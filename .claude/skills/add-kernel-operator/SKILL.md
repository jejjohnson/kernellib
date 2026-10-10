---
name: add-kernel-operator
description: Add a kernel linear operator to kernellib — a lineax AbstractLinearOperator over a kernel (matrix-free, cross, low-rank, batched), registered with lineax's predicates and lx.diagonal so gaussx solvers and preconditioners accept it — or extend the to_operator / to_cross_operator bridge. Use when asked to add a matrix-free or structured kernel operator, change how kernels become operators, or make a kernel path work with a gaussx solver in src/kernellib/_operators.
---

# Add a kernel operator

Read "2. Operators and feature maps" in `AGENTS.md`'s contracts.

## 1. Make sure it does not exist yet

`KernelOperator`, `ImplicitKernelOperator`, `ImplicitCrossKernelOperator`,
`batched_kernel_matvec`, `nystrom_operator`, `rff_operator`,
`fastfood_operator`, and gaussx's own operators (`LowRankUpdate`,
`Kronecker`, …). A structured Gram usually needs a `_gram_structure`
override on the kernel, not a new operator. Name a new operator after the
part of its contract that differs, not "implicit" or "lazy".

## 2. Write it (`src/kernellib/_operators/`)

- Subclass `lx.AbstractLinearOperator`; implement `mv`, `as_matrix`,
  `transpose`, `in_structure`, `out_structure`; static metadata as static
  fields; leading batch dimensions through `vmap_over_batch_dims`
  (`_utils.py`).
- Tags are claims: accept `tags` and normalise them (`_to_frozenset`); a PSD
  tag implies symmetric; a PSD tag on a non-square operator raises.
- Gradients: route hyperparameters through `params` and a custom JVP (see
  `_make_kernel_mv` in `_kernel.py`); without it, differentiate only what
  the scan supports.
- Register it: append the class to `_KERNEL_OPERATORS` in
  `_operators/__init__.py` (lineax's `is_*` predicates, `linearise`) and add
  an `lx.diagonal` registration (an O(N) diagonal, which gaussx's partial
  Cholesky and preconditioned CG need).
- If kernels should reach it through the bridge, extend `to_operator` /
  `to_cross_operator` in `_bridge.py`: Gram-only kernels raise rather than
  densify on an implicit path, and traced values never become static
  fields.

## 3. Export and document

`_operators/__init__.py`, `src/kernellib/__init__.py` (sorted `__all__`),
`::: kernellib.NewOperator` on `docs/api/operators.md` (mind its naming note),
the name in the Operators row of `docs/api/index.md`, `make capabilities`.

## 4. Tests (`tests/operators/`)

- `mv`, `transpose` and `as_matrix` against the dense Gram; batch
  dimensions.
- The lineax predicates and `lx.diagonal`; a solve through `gaussx.solve`
  with `gx.CGSolver` and `gx.PreconditionedCGSolver` matches the dense
  solve (`test_gaussx_integration.py`).
- Gradients with respect to the kernel's hyperparameters match the dense
  path; `jit` and `vmap` work.

## 5. Verify

`make test`, `uv run pytest -m slow tests/operators`,
`uv run pytest -m integration tests/operators` (the gaussx solver checks
in `test_gaussx_integration.py` are integration-tier), then `pre-pr-check`.
