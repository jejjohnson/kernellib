---
date: 2026-09-30
---

# Project: INLA (latent Gaussian models)

This project turns [gaussx#155](https://github.com/jejjohnson/gaussx/issues/155)
("research: R-INLA port") into work across the stack. #155 splits R-INLA
into five layers:

1. sparse precision linear algebra;
2. GMRF components;
3. SPDE priors;
4. the Laplace-plus-integration inference engine;
5. a modelling DSL.

It assigns layers 1–4 to gaussx and layer 5 to pyrox. This project keeps that
split, re-scopes it against the code as of 2026-09-30, and **fuses it with
the [manifold](project-manifold.md) and [RandNLA](project-rnla.md) projects** in the
[roadmap](roadmap.md), because they overlap heavily.

What #155 missed about the current code:

| #155 assumed | Actually |
|---|---|
| gaussx needs a Laplace inner loop from scratch | gaussx has `newton_update` (site natural parameters, with an `O(N)` diagonal-Hessian path), `cavity_distribution`, `GaussHermiteIntegrator`, and a likelihood family (`AbstractLikelihood`: Gaussian, Bernoulli, Poisson, Student-t). pyrox-gp builds `LaplaceInference`, `GaussNewtonInference`, EP and posterior linearisation on them (`_inference_nongauss.py`), plus Kalman versions (`_inference_nongauss_markov.py`). All are **covariance-form** (dense `K`) or Kalman. None is precision-form |
| RW / AR components need sparse Cholesky | Their precision matrices are banded. gaussx's `BlockTriDiag` already has `O(N d³)` `cholesky`, `solve` and `logdet` (RW1 and AR(1) with `d = 1`; RW2 and AR(2) with `d = 2`). Only the **selected inverse** (marginal variances) is missing: `diag_inv` falls back to dense Cholesky or Hutchinson |
| Grids need a mesh | A regular grid is a mesh whose FEM matrices are known: `C̃ = h²I`, and `G` is the grid Laplacian, a `KroneckerSum` of path Laplacians. So SPDE Matérn on a raster has exact eigenvalues, `logdet` and sampling through gaussx's existing `KroneckerSum` dispatch. This is the satellite gap-filling and SST case (#155 §5.3) |
| SPDE-FEM is a new design | pyrox has one: `design_docs/pyrox/features/gp/spde_fem.md` (draft, 2026-04-03). It puts FEM assembly and the solver in `pyrox.gp` and wraps CHOLMOD. This project moves the math to gaussx and the component to pyrox, and supersedes that doc |
| `gpyroX` | The old name for pyrox-gp |

The project's answers to #155's open questions:

1. **Which sparse Cholesky backend?**
   - Symbolic analysis on the host, once per sparsity pattern.
   - Numeric factorisation in JAX over static index arrays: jittable,
     vmappable over values, and differentiable through a selected-inverse
     VJP.
   - CHOLMOD through `pure_callback` as an opt-in backend.
   - Banded and Kronecker structure never reach either backend; they
     dispatch structurally first ([gaussx.md §3](roadmap-gaussx.md)).
2. **Which package is the PPL home?** A new workspace member,
   **`pyrox-lgm`** (`packages/pyrox-lgm`), next to `pyrox-gp`. LGMs are a
   strict superset of GPs, so it depends on gaussx and kernellib, not on
   pyrox-gp.
3. **Where do meshes come from?** Users bring the mesh (vertices,
   triangles) from fmesher, pygmsh or meshio. gaussx assembles P1 FEM
   `(C, G)` from it (about 100 lines) but never generates meshes. Rasters
   need no mesh (the grid path above).
4. **Which likelihoods in v1?** Those in gaussx's `AbstractLikelihood`
   family (Gaussian, Bernoulli, Poisson, Student-t), plus negative
   binomial and binomial. Tweedie and Beta follow the plumax / somax use
   cases.
5. **What does success look like?** Keep #155's target (Bayesian POD with
   covariates in under 5 s on CPU, posterior means within 1e-3 of R-INLA).
   Add a second target: Scotland lip cancer BYM2 matching R-INLA's
   hyperparameter posteriors, and an SPDE on a 10⁵-cell raster fitted
   through the Kronecker path on GPU.

## Where the work lives

| Repo | Phases |
|---|---|
| gaussx | [G1, G3–G10](roadmap-gaussx.md) (Part A): static-pattern `SparseOperator`, structured selected inverses, sparse Cholesky + Takahashi, `pseudo_logdet`, GMRF distributions, precision builders / SPDE / FEM, precision-form Laplace, θ-designs, VB correction; G16 (XDiag) as the fallback for fields too large to factor |
| kernellib | [K6](roadmap-kernellib.md): null spaces, structure matrices, mesh graphs, the graph-Matérn ↔ SPDE test; built on K2–K4 |
| pyrox | [P6–P10](roadmap-pyrox.md): the new `pyrox-lgm` package (components, PC priors, `inla()`), and the retarget of pyrox-gp's `spde_fem.md` |

The structure-dispatch table that organises the gaussx work is in
[gaussx.md §3](roadmap-gaussx.md).

## What you can solve with it

From the [examples gallery](roadmap-examples.md):

- [disease mapping with BYM2](#ex-bym2);
- [gap-filling sea-surface temperature](#ex-sst);
- [probability of detection for methane plumes](#ex-pod);
- [uncertainty on a 2M-node global mesh](#ex-mesh).

## Use cases

#155 §5 stands. Each use case below is tied to the path it runs on:

| Use case | Components | Path |
|---|---|---|
| Methane POD with covariates (plumax / MARS) | rw2 (season) + SPDE on the covariate plane + fixed effects, Bernoulli | banded + grid Kronecker, or mesh sparse Cholesky |
| Source emissions, disease-map style | bym2 on a basin adjacency (a kernellib graph), Poisson | sparse Cholesky + Takahashi |
| SST / SSS gap-filling (somax) | AR(1) ⊗ SPDE on the model grid, Gaussian | Kronecker(`BlockTriDiag`, grid SPDE): exact prior; `Q + σ⁻²I` on the observed mask needs sparse Cholesky or CG |
| Altimetry on the sphere | SPDE on an icosahedral surface mesh + rw1 bias | FEM on a surface mesh (users bring the mesh) |
| Plume retrieval residuals | SPDE on the image plane | grid Kronecker |

## Non-goals

These follow #155 §2.3, with two additions:

- Mesh generation.
- inlabru-style nonlinear predictors (use NumPyro).
- A port of R-INLA's C constants.
- `rgeneric`.
- The long tail of likelihoods.
- Smart-gradient finite differences: JAX gives exact gradients and
  Hessians of `log π̃(θ | y)` through the implicit Laplace mode and the
  selected-inverse logdet VJP.
- Wrapping PyINLA: it wraps the C engine, the opposite of this project.

## Open questions

| # | Question | Proposed answer |
|---|---|---|
| 1 | Fill-reducing ordering without new dependencies? | Reverse Cuthill–McKee from `scipy.sparse.csgraph` by default (SciPy is already a JAX dependency). AMD / METIS come through the optional CHOLMOD backend. Measure fill on the SPDE reference meshes before deciding whether a pure-Python AMD is worth vendoring |
| 2 | Hard or soft sum-to-zero constraints? | Both, per use. `inla()` uses hard constraints (conditioning by kriging, exact). The NumPyro face uses soft constraints (a tight Gaussian), because NUTS cannot handle a hard one |
| 3 | xarray summaries? | Behind an extra, `pyrox-lgm[xarray]`. The core returns an Equinox module of arrays |
| 4 | Should pyrox-gp's covariance-form `LaplaceInference` share code with the precision-form `laplace_mode`? | Share the site maths (already `newton_update`) and the likelihood family. Keep the two solvers separate: one factors `K`, the other `Q`. pyrox-gp can route a GMRF prior through `laplace_mode` later |

## Decisions log

| Date | Decision |
|---|---|
| 2026-09-30 | Plan drafted from gaussx#155 and merged with the manifold and RandNLA projects. There is one GMRF family in gaussx (G6). Spatial priors go to a new `pyrox-lgm` (P7). There is a JAX sparse Cholesky after all (G4), reversing the manifold project's earlier position. Banded and Kronecker structure dispatch before any sparse factorisation. RandNLA estimators are the fallback for fields too large to factor. pyrox's `spde_fem.md` is superseded |
| 2026-09-30 | Fused with the other two projects into the per-repo [roadmap](roadmap.md), with one phase numbering per repo (the mapping from old ids is in the roadmap README) |
