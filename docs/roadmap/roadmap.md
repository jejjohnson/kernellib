---
date: 2026-09-30
---

# Roadmap: graphs, GMRFs, INLA, randomized linear algebra and dependence penalties

:::{note}
**Status: draft (v0.2.0, 2026-09-30; updated 2026-10-08).** This is a plan,
not documentation of shipped features. The APIs below are proposals; each
phase lands as its own PR and issue in the repo it touches. Most of it has
now shipped: kernellib K1–K15 (all of it), gaussx G1–G14 (G15–G17 merged,
not yet released), pyrox-gp P1–P3 and part of P4, and pyrox-lgm P6–P9
(releases in the [implementation plan](roadmap-implementation.md#implementation-status); live
status in the [tracker](https://github.com/jejjohnson/kernellib/issues/125)).
manipy M0–M5 and plumax X1 have not started.
Where implementation changed an API, the repo pages say so in "As built" or
"As implemented" notes.
:::

## Summary

Four projects touch the same libraries, and overlap heavily:

| Project | Source | In one line |
|---|---|---|
| [Manifold learning and graphs](project-manifold.md) | The author's 2016–2018 `manipy` / `manifold_learning` research code | Graph primitives in kernellib; manipy becomes the JAX dimensionality-reduction library |
| [RandNLA](project-rnla.md) | [gaussx#156](https://github.com/jejjohnson/gaussx/issues/156) | Sketching, randomized factorisations and better preconditioners in gaussx; kernel uses in kernellib |
| [INLA](project-inla.md) | [gaussx#155](https://github.com/jejjohnson/gaussx/issues/155) | Structured precision and GMRFs in gaussx; latent Gaussian models and `inla()` in a new pyrox-lgm |
| [Dependence penalties](project-fairkl.md) | The author's [keras-fairkl](https://github.com/jejjohnson/keras-fairkl) and 2017 fair-learning notebooks | No new library: stable and gradient-safe HSIC / CKA, quadratic-penalty KRR (fair KRR, LapRLS), supervised / fair kernel PCA and pre-images in kernellib |

Every repo file mixes the API with **maths notes** (where each operation comes from) and **pseudocode examples**; the [examples gallery](roadmap-examples.md) shows end-to-end problems. This directory specifies the work **per repo**, with one phase numbering
per repo. The project pages keep what is project-specific: motivation, the
audit of the source issue or old code, use cases, non-goals, and open
questions.

| File | Phases | Serves |
|---|---|---|
| [gaussx.md](roadmap-gaussx.md) | G1–G10 (Part A: structured precision, GMRFs, INLA kernels), G11–G17 (Part B: randomized) | manifold, RandNLA, INLA |
| [kernellib.md](roadmap-kernellib.md) | K1–K5 (graphs, embeddings), K6 (GMRF structure), K7–K10 (randomized kernel methods), K11 (docs), K12–K14 (dependence penalties), K15 (proximity graphs) | all four |
| [pyrox.md](roadmap-pyrox.md) | P1–P5 (pyrox-gp), P6–P10 (pyrox-lgm, repo-level) | manifold, RandNLA, INLA |
| [manipy.md](roadmap-manipy.md) | M0–M5 | manifold |
| [plumax.md](roadmap-plumax.md) | X1 | RandNLA |
| [examples.md](roadmap-examples.md) | — | Thirteen end-to-end problems (model, maths, pseudocode, phases needed) |
| [implementation.md](roadmap-implementation.md) | all | The implementation plan: dependency graph, critical paths, release gates and the PR sequence per repo |

**Out of scope:**

- **somax and filterax.** somax has no linear algebra of its own, and
  filterax's dense problems are ensemble-sized. See the
  [RandNLA project](project-rnla.md#where-the-work-lives).
- **geonnax** keeps its `graph_laplacian_eigpairs` as a basis primitive
  ([manifold open question 3](project-manifold.md#open-questions)).

---

## 1. Placement rule and ownership

> If a spatial model or a GP would import it, it goes in **kernellib**,
> or in **gaussx** when there is no kernel or graph in it. NumPyro models,
> hyperpriors and inference drivers go in **pyrox**: GPs in pyrox-gp,
> latent Gaussian models in pyrox-lgm. If only dimensionality-reduction
> users would import it, it goes in **manipy**.

| Capability | Home | Phases |
|---|---|---|
| Sparse operator with a static pattern | gaussx | G1 |
| Generalised symmetric eigensolver | gaussx | G2 |
| Selected inverses: banded, Kronecker, sparse (Takahashi); sparse Cholesky | gaussx | G3, G4 |
| Pseudo-log-determinants | gaussx | G5 |
| GMRF distributions (constraints, marginal variances, sampling) | gaussx | G6 |
| Precision builders: iid, RW, AR, Besag, BYM2 scaling, SPDE (grid, FEM), FEM assembly | gaussx | G7 |
| Precision-form Laplace, θ-designs, VB correction | gaussx | G8–G10 |
| Sketches, range finder, randomized SVD / eigh / Nyström, RPCholesky, preconditioners, sketch-and-precondition LSMR, XDiag and tier-2 estimators | gaussx | G11–G17 |
| Graph construction (k-NN, radius, grid, mesh, edge lists from external tools, Delaunay / Gabriel / RNG), Laplacians, spectral graph kernels, graph Matérn, Laplacian eigenpairs | kernellib | K1–K4, K6, K15 |
| Laplacian / Schrödinger eigenmaps, LPP, SEP, kernel LPP / SEP | kernellib | K5 |
| GMRF structure: null spaces, structure matrices | kernellib | K6 |
| Landmark selection, preconditioned KRR, randomized kernel PCA | kernellib | K7–K10 |
| Gradient-safe CKA, stable unbiased HSIC, mini-batch CKA, Gaussian bandwidth | kernellib | K12 |
| Quadratic-penalty KRR (fair KRR, LapRLS), supervised / fair kernel PCA, pre-images | kernellib | K13, K14 |
| Graph Matérn inducing features, latent and inducing-point initialisation, exact-GP recipe | pyrox-gp | P1–P5 |
| Latent components (including the CAR / ICAR / Leroux / BYM2 spatial priors), PC priors, `LGM`, `inla()` | pyrox-lgm | P6–P9 |
| Manifold alignment, Isomap / LLE / diffusion maps / t-SNE, out-of-sample extension, HSI workflows, metrics, datasets | manipy | M0–M5 |

## 2. Dependency graph

```
             lineax · matfree · jax.experimental.sparse
                              │
geonnax ──────────┐           ▼
                  │        gaussx          (never imports kernellib)
                  │           │
                  ▼           ▼
                   kernellib               (never imports numpyro)
             ┌─────┼──────────┐
             ▼     ▼          ▼
         manipy  pyrox-gp  pyrox-lgm       (manipy never imports numpyro;
                                            pyrox-lgm does not import pyrox-gp)
plumax ──► gaussx
```

---

## 3. Cross-cutting decisions

These are the points where the projects meet. Every repo file follows
them.

1. **Static sparsity, traced values.**
   - gaussx's `SparseOperator` (G1) carries a static `SparsityPattern`,
     and only `values` is traced.
   - kernellib's `Graph` (K2) correspondingly has a static
     `GraphTopology`, and only `weights` is traced.
   - This is what lets a symbolic sparse Cholesky (G4) be computed once,
     and reused across every Newton step, hyperparameter point,
     reweighting and `vmap`.
   - Builders run eagerly, outside `jit`.
2. **Dispatch on structure before factorising.** Banded (`BlockTriDiag`),
   Kronecker and grid (`KroneckerSum`, `SpectralFunction`) structure take
   exact, cheap paths first. Then sparse Cholesky with Takahashi. Then the
   iterative fallback: CG, SLQ and XDiag (G16). The table is in
   [gaussx.md §3](roadmap-gaussx.md).
3. **Covariance form versus precision form.**
   - The Nyström and RPCholesky preconditioners (G13, G14) and
     `NystromLogdet` (G17) target covariance-form `K + σ²I`, whose hard
     directions are at the top of the spectrum.
   - GMRF precision systems `Q + AᵀWA` are hard at the bottom, so they use
     exact factorisations or Jacobi-preconditioned CG. Docstrings say so.
   - pyrox-gp (GPs) works in covariance form, and pyrox-lgm (LGMs) in
     precision form.
4. **One GMRF family.** `GaussianMRF` and `IntrinsicGMRF` (G6) serve the
   manifold project's spatial priors and INLA alike.
   - Hard constraints (by kriging) are for `inla()`.
   - Soft constraints are for NUTS. pyrox-lgm's NumPyro face defaults to
     `s = 1e-2` (gaussx's default `1e-3` hits NUTS's tree-depth cap).
   - BYM2 is `BYM2GMRF`, an `IntrinsicGMRF` subclass with its exact
     density, because the null vector of its joint precision depends on θ
     (gaussx#508).
   - Priors with hyperpriors live only in pyrox-lgm (P7).
5. **The noise shift is explicit.** Every preconditioner is built from the
   PSD part (`K`, or `AᵀA`) with `shift=σ²` passed separately (G13, G14,
   K9, P4). This fixes the gaussx#345 class of bugs by construction.
6. **Boundaries:**
   - gaussx never imports kernellib;
   - kernellib never imports NumPyro, and its core never imports
     scikit-learn;
   - distributions subclass NumPyro's `Distribution`, never bare Equinox
     modules;
   - einx for every contraction;
   - **PRNG keys:**
     - **gaussx:** randomized primitives take `key`, with `key=None`
       meaning `PRNGKey(0)`, as elsewhere in gaussx, stated in each
       docstring.
     - **kernellib, pyrox and manipy:** a `key` is **required** wherever
       the result depends on it (`select_landmarks`, `init_inducing`,
       `Falkon` centres, randomized `KernelPCA`), following kernellib's
       existing rule (`estimate_lengthscale(subsample=...)` raises
       without one).
     - **The exception:** `key=None` is allowed where the key only seeds
       an iterative solver whose answer does not depend on it, up to
       tolerance, such as the ARPACK start vector of
       `laplacian_eigpairs(method="arpack")` (seed 0 without a key).
       kernellib's JAX iterative paths (`"lanczos"`, and the eigenmaps'
       `"lobpcg"`) require a key; the eigenmap estimators derive it from
       `random_state`.
7. **External tools in examples: training loops and data sources.**
   - **Networks and SVI: `pipekit-train`.** Examples that train a network
     (an Equinox module, over mini-batches) use
     [pipekit](https://github.com/jejjohnson/pipekit)'s `TrainingLoop`
     with its Equinox backend, rather than a hand-rolled loop. The
     objective is a pipekit `TrainTask` (`loss_fn(model, batch, key)`).
     pyrox's SVI examples may use pipekit's `numpyro-svi` adapter the same
     way.
   - **Small full-batch fits: optax directly.** A few scalar
     hyperparameters, or `jax.grad` through `fit`, use optax directly.
     `TrainingLoop` is built for datasets and batches, and would hide the
     point of such examples.
   - **Docs-only dependency.** pipekit is a dependency of each repo's
     **docs group only**, pinned by git tag (it is not on PyPI; 0.0.2 at
     the time of writing). Library code never imports it, and the import
     tests are unchanged. Notebook outputs are committed, so the pin only
     matters when a notebook is re-executed.
   - **Real spatial data: city2graph.** Examples that start from polygons,
     road networks or origin–destination (OD) tables use
     [city2graph](https://github.com/c2g-dev/city2graph) (contiguity,
     proximity and OD graphs), also as a docs-only dependency. Its
     output enters through `kl.graph_from_edges` as integer arrays, with
     its distance-valued `weight` column converted to affinities (K2).
     GIS loaders and GNN converters stay in city2graph.

---

## 4. Order of work

Phases are specified, with tests, in the repo files. "Needs" means a
released version, and every phase is one PR. A wave is the **earliest**
a phase can start, computed from the "Needs" columns: within a wave,
everything can run in parallel. The
[implementation plan](roadmap-implementation.md) turns this into a
sequence, with slack, release gates and PR order.

| Wave | Phases |
|---|---|
| 1 | G1, G2, G3, G9, G11, G14, K1, K12, K13, K14, P1, P6, P10, M0 |
| 2 | G4 (G1), G12 (G11), G15 (G11), K2 (K1, G1), K7 (G11), K8 (G14) |
| 3 | G5 (G3, G4), G7 (G1, G3, G4), G13 (G12), G16 (G12), K3 (K2), K10 (G12), X1 (G12), P3 (K8), P5 (K8), M2 (K2) |
| 4 | G6 (G1, G3, G5), K4 (K3), K5 (K3, G2), K9 (G13, G14), P4 (G13; also gaussx#312), G17 on demand |
| 5 | G8 (G6), K6 (K2–K4; G7 for its test), P2 (K3, K4), M1 (K2, K5, G2) |
| 6 | G10 (G8), K15 (K2, K6), P7 (G6, G7, K6), M3 (M1, M2) |
| 7 | P8 (G8–G10, P7), K11 (K5, K6, K8–K10, K12–K15), M4 (M3), M5 (M3) |
| 8 | P9 (P8). P10 is also updated alongside every pyrox phase |

**Highest-value early deliverables:**

- **G13** fixes a live performance bug (#354).
- **G3** alone makes temporal INLA (RW / AR priors with any likelihood)
  exact and `O(N)`.
- **K8 → P3** gives kernel-aware landmark and inducing-point selection.
- **P1** removes a phantom `pca_init` from pyrox's design docs.
- **K12** fixes two live numerics bugs, kernellib#93 (NaN CKA) and #94
  (float32 unbiased HSIC), that make CKA unusable as a training penalty
  today. It needs nothing else.

**Critical path to a first end-to-end `inla()`:**
G1 → G4 → G5 → G6 → G8 → G10 → P8, with G7, K6 and P7 alongside. The
graph chain K1 → K2 → K3 → K4 → K6 → P7 has zero slack too: a delay in
either chain delays `inla()`.

### Old phase ids

For reading notes or issues written against the separate project plans:

| Old | New |
|---|---|
| manifold G1, G2, G3, G4 | G1, G2, G5, G6 |
| manifold K1–K5, K6 | K1–K5, K11 |
| manifold P1, P2, P3 | P2, P1, P7 |
| inla G1, G2 … G9 | G1, G3 … G10 (each +1 from G2 on) |
| inla K1 | K6 |
| inla L0 … L4 | P6 … P10 |
| rnla G1 … G6 | G11 … G16 (G6's tier-2 remainder is G17) |
| rnla K1 … K4 | K7 … K10 |
| rnla P1, P2, P3 | P3, P4, P5 |
| manipy M0–M5, plumax X1 | unchanged |

---

(roadmap-open-questions)=
## 5. Open questions

These are cross-cutting. Project-specific questions are on the project
pages.

| # | Question | Proposed answer |
|---|---|---|
| 1 | `jax.experimental.sparse` is experimental | Accept it, isolated behind `gaussx.SparseOperator` (G1), `Graph.to_bcoo()` (K2) and kernellib's LOBPCG eigenmap path (`lobpcg_standard` on a BCOO, kernellib v0.0.17), so a move to a stable API touches three places |
| 2 | Where do the CAR / ICAR distributions live? | **Resolved.** The distributions go in gaussx (G6). The priors with hyperpriors go in pyrox-lgm (P7), not pyrox-gp |
| 3 | Where does this roadmap live? | kernellib's `docs/roadmap/`, rendered in the MyST site, because gaussx's CLAUDE.md keeps design documents out of the gaussx repo (`.plans/`, gitignored). File the phases as issues in each repo, and link them here |
| 4 | Should the `key=None → PRNGKey(0)` default stay for new randomized code? | In gaussx, yes, for consistency with gaussx today. kernellib, pyrox and manipy require a key wherever the result depends on it (§3, decision 6). Revisit if silent determinism causes a bug |
| 5 | Should `laplacian_eigpairs(method="lanczos")` run the same residual check and Krylov growth as the eigenmaps? pyrox-gp calls it directly ([example 5](roadmap-examples.md)) | **Resolved.** Yes: the eigenmaps' residual check and Krylov-growing restarts moved into `_graph/_eigpairs.py`, so `laplacian_eigpairs(method="lanczos")` and every caller of it (pyrox-gp included) get them, and the eigenmaps reuse them (kernellib#181; [kernellib.md §4.3](roadmap-kernellib.md)) |

## 6. Decisions log

| Date | Decision |
|---|---|
| 2026-09-30 | The three project plans (manifold, RandNLA, INLA) are fused into this per-repo roadmap, with one phase numbering per repo and the old ids mapped in §4. Project pages keep motivation, audits, use cases, non-goals and open questions. Cross-cutting decisions are in §3 |
| 2026-09-30 | A fourth project, [dependence penalties](project-fairkl.md), from the keras-fairkl audit. It is kernellib-only (K12–K14) and has no cross-repo dependencies; its fast paths reuse gaussx's existing `LowRankUpdate` Woodbury solve. Bugs found are filed as kernellib#93, #94 and keras-fairkl#15–#19 |
| 2026-09-30 | Cross-cutting decision 7: examples that train networks use pipekit-train's `TrainingLoop` (a docs-only dependency, git-pinned); small full-batch fits use optax directly |
| 2026-09-30 | city2graph reviewed. Not a dependency; it becomes the docs-only data source for real spatial examples (decision 7). kernellib gains `graph_from_edges`, distance-to-weight conversion and `knn_graph(ensure_connected=)` in K2, and proximity graphs as K15. Gallery example 13 (OD flows) added |
| 2026-09-30 | The waves are recomputed as earliest-start levels from the "Needs" columns (G7, K5, M1, M3–M5 and P4 move earlier), and the [implementation plan](roadmap-implementation.md) is added |
| 2026-10-05 | Implementation findings from gaussx G4–G10, G13 and pyrox-lgm P6–P7 are folded back into the repo pages (kernellib#122): odd-`n` `rw2_structure` padding (use `n + 1` nodes and a row-selection projector); `pseudo_logdet(structure="laplacian")`; `laplace_mode` and `vb_mean_correction` take no `y` (the likelihood holds it), `BinomialLikelihood(y, n_trials)`, NB `concentration`; `theta_design(method=None)` and its derived CCD weights; lower-triangle sparse storage, `cholesky(SparseOperator)` → `SparseCholeskyFactor`, reverse-over-reverse Hessians; matrix-free conditioning of a spectral prior; `randomized_eigh(n_power_iter=0)` is more accurate than the old Rayleigh–Ritz, and there is no lazy Nyström path; `BYM2GMRF` (gaussx#508); pyrox-lgm's pins, `PCAR1Rho` on \|ρ\|, `PCBYM2Phi`'s null space and deflated SLQ, `BYM2`'s `(tau, phi)`, `SPDE`'s `range_sigma`, `Kronecker`'s fixed group τ, the NUTS soft-constraint scale and odd-`n` `RW2`. K5 and K15 are recorded as built, and the implementation plan gains a status section |
| 2026-10-08 | Status refresh. kernellib v0.0.17 ships K11 and post-phase fixes that changed APIs, recorded as "As implemented" notes in [kernellib.md](roadmap-kernellib.md): `mesh_graph(on_negative="raise" \| "clip" \| "allow")`, with signed-weight graphs rejected where weights must be non-negative (kernellib#157); closed-form Kronecker eigenpairs (#156); residual-checked, restarting Lanczos eigenmaps (#159) and `eigen_solver="lobpcg"` (#91); KRR `tol` / `max_steps` / `throw`, fitted `n_iter` / `converged`, a true-residual check and a small-λ warning, and Falkon's dtype-aware `tol` (#160, #162); `laplacian_penalty` on sparse graphs (#153); centred KernelPCA pre-images (#154). gaussx G15–G17 are merged (gaussx#511) but unreleased after v0.6.4, and their "As built" notes are in [gaussx.md](roadmap-gaussx.md). gaussx#312 (gradients through preconditioned CG) was fixed in gaussx v0.6.1, so P4 has no external blocker. pyrox-gp 0.1.8 ships P1–P3 and P4's solver; P4's matrix-free path is pyrox#277 |
