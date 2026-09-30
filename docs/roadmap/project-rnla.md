---
date: 2026-09-30
---

# Project: randomized numerical linear algebra (RandNLA)

Randomized numerical linear algebra (RandNLA) means sketching operators, the
randomized range finder and the factorisations built on it (QB, SVD,
symmetric eigendecomposition, Nyström), randomly pivoted Cholesky,
sketch-and-precondition least squares, and stochastic trace estimation.
This project turns the survey in
[gaussx#156](https://github.com/jejjohnson/gaussx/issues/156) ("research:
RNLA Port") into work across the stack.

#156 was written in May 2026, before the kernellib split, and its module
map is out of date. This project re-scopes it against the code as of
2026-09-30 (gaussx 0.4.0, kernellib 0.0.11, pyrox-gp 0.1.7):

| #156 assumed | Actually |
|---|---|
| The kernel Nyström code lives in gaussx `_kernels/_kernel_approx.py` | The kernel layer moved to kernellib in gaussx 0.2.0. `nystrom_operator` and `NystromFeatures` are kernellib's |
| There is no randomized Nyström | gaussx has a `NystromPreconditioner`, but it is a randomized **Rayleigh–Ritz** projection (`QᵀAQ` on an orthonormalised Gaussian block), not a Nyström sketch, and it gets *worse* below full rank ([gaussx#354](https://github.com/jejjohnson/gaussx/issues/354)) |
| The preconditioner is pivoted Cholesky from matfree | gaussx has its own guarded **greedy** pivoted Cholesky (`guarded_pivoted_cholesky`, `_primitives/_root.py:251`). Randomly pivoted Cholesky is a change to how the pivot is picked |
| An LSQR strategy is needed | gaussx has `LSMRSolver`. Right preconditioning is just LSMR on the composed operator `A·M`, so there is no need for a second Krylov least-squares solver |
| SRTT needs a hand-rolled DCT or Hadamard | kernellib already has an O(d log d) fast Walsh–Hadamard transform, `hadamard_transform` (`_operators/_fastfood.py:55`), used by FastFood |
| Leverage-score landmark selection is new | kernellib already has it: `NystromFeatures(selection="leverage")` (kernellib#34, closed) |

The plan answers #156's three open questions:

1. **SRHT or SJLT?** Both. SparseSign (SJLT) is the default. SRHT is cheap
   to add because the Hadamard transform exists; it moves to gaussx (see
   [kernellib.md](roadmap-kernellib.md)). The DCT variant is dropped.
2. **A sub-package or a sibling `sketchx` library?** A private
   sub-package in gaussx, flat top-level names, like every other gaussx
   module. The whole surface is about 15 symbols, which does not justify a
   library, a release train and another pin in every downstream repo.
3. **JAX-native, or wrap PARLA?** JAX-native. The point is `jit`, `vmap`,
   GPU, and composing with lineax operators. PARLA and SciPy serve only as
   numerical references in tests, never as dependencies.

## Where the work lives

| Repo | Phases |
|---|---|
| gaussx | [G11–G17](roadmap-gaussx.md) (Part B): sketches, range finder / QB / randomized SVD and eigh, randomized Nyström and the preconditioner rewrite (#354), RPCholesky, sketch-and-precondition LSMR, XDiag, tier 2; plus G2 (`eigh_generalized`, shared with manifold) |
| kernellib | [K7–K10](roadmap-kernellib.md): Hadamard transform moves to gaussx, `select_landmarks`, preconditioned KRR, randomized `KernelPCA` |
| pyrox | [P3–P5](roadmap-pyrox.md): `init_inducing`, the large-`n` exact-GP recipe, LogFalkon centres |
| plumax | [X1](roadmap-plumax.md): `gaussx.randomized_svd` |

## What you can solve with it

From the [examples gallery](roadmap-examples.md):

- [kernel regression and GPs at scale](roadmap-examples.md#ex-scale);
- [tall nonlinear least squares](roadmap-examples.md#ex-lsq);
- [background covariances and EOFs](roadmap-examples.md#ex-eofs);
- [uncertainty on a 2M-node global mesh](roadmap-examples.md#ex-mesh) (the fallback for fields too large to factor).

**Repos not touched:**

- **somax** has no linear algebra of its own. It reaches gaussx only
  through filterax and vardax, and uses `LowRankUpdate` in one tutorial.
- **filterax** already works in the ensemble's low-rank space
  (`ensemble_covariance`, a `LowRankUpdate` innovation covariance,
  `solve_rows`). Its dense problems are ensemble-sized (`N_e × N_e`), so
  randomization gains nothing today. #156 §6.5's link between ensembles
  and range finders is real, but it is a research direction (for example
  singular-vector or bred perturbations from a range finder of the
  tangent-linear model), not a port.
- **geonnax** is not touched. It has no Hadamard transform or random
  projection that overlaps.

---


## Existing issues this project absorbs

| Issue | Relation |
|---|---|
| gaussx#156 | This project. Close it once the phase issues are filed |
| gaussx#354 | Closed by G13 |
| gaussx#371 | Folded into G14 (build-once preconditioners) |
| gaussx#345 | G13 and G14 must not reintroduce it: the preconditioner is built from `K` and takes the noise shift `σ²` explicitly, so the noise is never counted twice |
| gaussx#312 | Not fixed here, but it blocks P4 (hyperparameter gradients through preconditioned CG). It belongs to epic #283 |
| gaussx#413 | G12 documents which end of the spectrum each method targets: randomized methods target the **top** |
| kernellib#34 (closed) | Leverage-score landmarks already exist; K8 generalises them into `select_landmarks` |
| kernellib#91 | Not absorbed. The small end of a graph Laplacian's spectrum is LOBPCG / Lanczos territory; randomized range finders target the top |
| pyrox#50 | P5 supplies LogFalkon's centre selection |

## Non-goals

These come from #156, and are confirmed here:

- RandBLAS's sparse-matrix machinery. Sparse-sign sketches store their
  nonzero indices and apply via `segment_sum`, with no sparse matrix
  library.
- Counter-based reproducible RNGs. `jax.random` keys already give this.
- PARLA's object hierarchy. Here these are Equinox modules and functions.
- Tensor or Khatri–Rao sketches, and CholeskyQR with random
  preconditioning. Defer until there is a user.
- Replacing matfree's Hutchinson and SLQ. Hutch++ and XTrace sit next to
  them as tier-2 options.

## Open questions

| # | Question | Proposed answer |
|---|---|---|
| 1 | Should `svd` / `eig` default to `method="randomized"` when `rank` is given? | No. Lanczos stays the default: it is already wired in, and it has no oversampling parameter to explain. Randomized is opt-in, and recommended in docstrings for slowly decaying spectra with `n_power_iter ≥ 2` |
| 2 | Should the Nyström preconditioner need the shift `μ` (the noise variance)? | Yes. The Frangella–Tropp–Udell preconditioner is built from `K` and `μ` separately. That is also what fixes the #345 class of bugs |
| 3 | Sketch-and-solve as a solver strategy, or only a function? | A function. It is an approximation, not a solve, and a strategy would imply solver-level accuracy |
| 4 | Keep kernellib's `hadamard_transform` public? | Yes, as a re-export of gaussx's, because FastFood users already call it |

## Decisions log

| Date | Decision |
|---|---|
| 2026-09-30 | Plan drafted from gaussx#156, re-scoped to the post-split stack. RandNLA lives in gaussx as `_sketching/` and `_randomized/`, not in a new library. Implementation is JAX-native. SparseSign is the default sketch, with SRHT on a Hadamard transform moved from kernellib. No LSQR: LSMR on `A·M`. The first deliverable is fixing the Nyström preconditioner (#354) |
| 2026-09-30 | Reconciled with the INLA project. RandNLA estimators are the fallback backend for precision-form fields too large to factor; `diag_inv(method="xdiag")` (G16) is promoted out of "on demand". Nyström and RPCholesky preconditioners are documented as covariance-form only |
| 2026-09-30 | Fused with the other two projects into the per-repo [roadmap](roadmap.md), with one phase numbering per repo (the mapping from old ids is in the roadmap README) |
