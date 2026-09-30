---
date: 2026-09-30
---

# Roadmap: graphs, GMRFs, INLA, randomized linear algebra and dependence penalties

:::{note}
**Status: draft (v0.2.0, 2026-09-30).** This is a plan, not documentation of
shipped features. The APIs below are proposals; each phase lands as its own
PR and issue in the repo it touches.
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
| [kernellib.md](roadmap-kernellib.md) | K1–K5 (graphs, embeddings), K6 (GMRF structure), K7–K10 (randomized kernel methods), K11 (docs), K12–K14 (dependence penalties) | all four |
| [pyrox.md](roadmap-pyrox.md) | P1–P5 (pyrox-gp), P6–P10 (pyrox-lgm, repo-level) | manifold, RandNLA, INLA |
| [manipy.md](roadmap-manipy.md) | M0–M5 | manifold |
| [plumax.md](roadmap-plumax.md) | X1 | RandNLA |
| [examples.md](roadmap-examples.md) | — | Twelve end-to-end problems (model, maths, pseudocode, phases needed) |

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
| Graph construction (k-NN, radius, grid, mesh), Laplacians, spectral graph kernels, graph Matérn, Laplacian eigenpairs | kernellib | K1–K4, K6 |
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
   - Soft constraints are for NUTS.
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
   - randomized functions take `key`, with `key=None` meaning
     `PRNGKey(0)`, as elsewhere in gaussx, stated in each docstring.

---

## 4. Order of work

Phases are specified, with tests, in the repo files. "Needs" means a
released version, and every phase is one PR. The waves below respect
every cross-repo dependency; within a wave, everything can run in
parallel.

| Wave | Phases |
|---|---|
| 1 | G1, G2, G3, G9, G11, G14, K1, K12, K13, K14, P1, P6, M0 |
| 2 | G4 (G1), G12 (G11), G15 (G11), K2 (K1, G1), K7 (G11), K8 (G14) |
| 3 | G5 (G3, G4), G13 (G12), G16 (G12), K3 (K2), K10 (G12), X1 (G12), P3 (K8), P5 (K8), M2 (K2) |
| 4 | G6 (G1, G3, G5), G7 (G1, G3, G4), K4 (K3), K9 (G13, G14), G17 on demand |
| 5 | G8 (G6), K5 (K3, G2), K6 (K2–K4; G7 for its test), P2 (K3, K4) |
| 6 | G10 (G8), P7 (G6, G7, K6), M1 (K2, K5, G2) |
| 7 | P8 (G8–G10, P7), K11 (K5, K6, K8–K10, K12–K14), M3 (M1, M2), M5 |
| 8 | P9, M4. P4 as soon as gaussx#312 is fixed (and G13 is out). P10 alongside every pyrox phase |

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
G1 → G4 → G5 → G6 → G8 → G10 → P8, with G7, K6 and P7 alongside.

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
| 1 | `jax.experimental.sparse` is experimental | Accept it, isolated behind `gaussx.SparseOperator` (G1) and `Graph.to_bcoo()` (K2), so a move to a stable API touches two places |
| 2 | Where do the CAR / ICAR distributions live? | **Resolved.** The distributions go in gaussx (G6). The priors with hyperpriors go in pyrox-lgm (P7), not pyrox-gp |
| 3 | Where does this roadmap live? | kernellib's `docs/roadmap/`, rendered in the MyST site, because gaussx's CLAUDE.md keeps design documents out of the gaussx repo (`.plans/`, gitignored). File the phases as issues in each repo, and link them here |
| 4 | Should the `key=None → PRNGKey(0)` default stay for new randomized code? | Yes, for consistency with gaussx today. Revisit repo-wide if silent determinism causes a bug |

## 6. Decisions log

| Date | Decision |
|---|---|
| 2026-09-30 | The three project plans (manifold, RandNLA, INLA) are fused into this per-repo roadmap, with one phase numbering per repo and the old ids mapped in §4. Project pages keep motivation, audits, use cases, non-goals and open questions. Cross-cutting decisions are in §3 |
| 2026-09-30 | A fourth project, [dependence penalties](project-fairkl.md), from the keras-fairkl audit. It is kernellib-only (K12–K14) and has no cross-repo dependencies; its fast paths reuse gaussx's existing `LowRankUpdate` Woodbury solve. Bugs found are filed as kernellib#93, #94 and keras-fairkl#15–#19 |
