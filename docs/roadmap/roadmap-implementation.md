---
date: 2026-09-30
---

# Implementation plan

:::{note}
**Status: in progress (updated 2026-10-08; drafted 2026-09-30).** Waves
1–7 are mostly shipped; see [what has shipped](#implementation-status).
The live status is the tracker,
[kernellib#125](https://github.com/jejjohnson/kernellib/issues/125). This page sequences the work that the
[roadmap](roadmap.md) specifies. The roadmap's repo pages are the
specification (API, maths, tests). This page only says **in what order**,
**what blocks what**, and **which release unblocks whom**. Everything on it
is derived from the "Needs" columns of the repo pages. If a dependency
changes there, regenerate the tables here.
:::

(implementation-status)=
## Status: what has shipped

The plan below is unchanged. This section records which release carried
each phase, as of 2026-10-08. The tracker
[kernellib#125](https://github.com/jejjohnson/kernellib/issues/125) is the
live status (epics, phase issues, "blocked by" links, and what is ready
next); this table is only a snapshot.

| Repo | Release | Phases |
|---|---|---|
| gaussx | 0.5.0 | G1, G2, G3, G9, G11, G14 |
| gaussx | 0.6.0 | G4–G8, G10, G12, G13 |
| gaussx | 0.6.1–0.6.2 | Fixes, including `BYM2GMRF` for the BYM2 density (gaussx#508, #518), exact structured paths in `AutoSolver` and `inv_quad_logdet`, Takahashi on diagonal-only columns, and gradients through data-dependent CG preconditioners (gaussx#312, fixed by #513) |
| gaussx | 0.6.3–0.6.4 | Fixes from the v0.2.0 review epics (dtype hygiene, dispatch conformance, solver strategies, GP recipes); API changes that touch the roadmap are noted in [gaussx.md](roadmap-gaussx.md) |
| gaussx | merged, unreleased (after v0.6.4) | G15 (#625), G16 (#628), G17 (#629, #631, #632); epic gaussx#511 closed |
| kernellib | 0.0.12–0.0.13 | K1, K12–K14 (0.0.13: review follow-ups) |
| kernellib | 0.0.14 | K2, K7–K10 |
| kernellib | 0.0.15 | K3, K4, K6 (K4 is missing from that release's notes) |
| kernellib | 0.0.16 | K5, K15; gaussx source raised to v0.6.1 (kernellib#142) |
| kernellib | 0.0.17 | K11 (kernellib#119); follow-ups that changed APIs, recorded in [kernellib.md](roadmap-kernellib.md): `mesh_graph(on_negative=)`, closed-form Kronecker eigenpairs, checked Lanczos and `eigen_solver="lobpcg"` eigenmaps, KRR iteration statistics and float32 checks, `laplacian_penalty` on graphs, centred pre-images |
| pyrox-lgm | 0.1.0–0.1.1 | P6–P9 (golden R-INLA fixtures in pyrox#275, merged after 0.1.1) |
| pyrox-gp | 0.1.8 | P1–P3 and P4's solver (`preconditioned_cg_solver`); P5 is a design-doc change only (pyrox#281). The release notes list only P2 and P3 |

- **Done:** every kernellib phase, K1–K15. gaussx G1–G17 are all merged;
  G15–G17 wait for the next gaussx release.
- **In progress:** P4's matrix-free path and its n = 20 000 tests
  (pyrox#277; epic pyrox#257 stays open for it), P10 (pyrox#256: pins
  and P2's eigenpairs done; the boundaries and SPDE pages not yet), and
  the θ-integration gap to R-INLA (pyrox#274).
- **Not started:** manipy M0–M5 (manipy#1–#7), plumax X1 (plumax#127).
- **Pins behind:** kernellib still pins gaussx v0.6.1 (latest v0.6.4);
  pyrox pins kernellib v0.0.15 (latest v0.0.17) and gaussx v0.6.1.
- **Milestones:**
  - **Reached:** A (numerics fixed), D (graphs: K1–K6, K15 and P2 in
    pyrox-gp 0.1.8), and E (first `inla()`; the remaining gap to R-INLA
    is in the integration over θ, pyrox#274).
  - **Partly reached:** B (everything but X1), C (G13, K9 and P4's solver;
    #312 is fixed, P4's matrix-free path is pyrox#277), G (K11 done; P10
    in progress) and H (P9 done; G17 merged but unreleased; M4 not
    started).
  - **Not started:** F (manipy 0.1).
- **What implementation changed in the specification** is folded back
  into the repo pages (kernellib#122, and the 2026-10-08 refresh), as the
  definition of done in §1 asks: look for the "As built" and "As
  implemented" notes there.

## 1. Rules of the road

- **One phase = one issue = one PR**, in the repo that owns it. The issue
  links to the phase's section in the roadmap, and uses GitHub's
  "blocked by" links for every entry in "Needs".
- **"Needs" means a released tag.** pyrox-gp, pyrox-lgm and manipy pin
  kernellib and gaussx by git tag, and kernellib pins gaussx. A
  cross-repo dependency therefore costs a release. §4 batches them, so
  that each repo releases at most once per wave.
- **A wave is one release cycle.** Everything in a wave can run in
  parallel. A phase may start in any wave between its earliest and latest
  wave (§3) without delaying the end.
- **Definition of done for a phase:**
  - the tests listed in its roadmap section pass, in the right speed tier;
  - API pages and docstrings (with executable examples) are updated;
  - the release-please note names the phase id;
  - the roadmap page is updated if the API changed during implementation.
    The roadmap stays the source of truth.
- **Worktrees and branches** follow each repo's conventions. gaussx and
  kernellib work goes in `geo_ml/.worktrees/<name>`.

## 2. Dependency graph

Thick red arrows are the zero-slack chains. Any delay on them delays the
last deliverable (P9, `inla()` diagnostics).

```{mermaid}
flowchart LR
  subgraph gaussx
    G1
    G2
    G3
    G4
    G5
    G6
    G7
    G8
    G9
    G10
    G11
    G12
    G13
    G14
    G15
    G16
    G17
  end
  subgraph kernellib
    K1
    K2
    K3
    K4
    K5
    K6
    K7
    K8
    K9
    K10
    K11
    K12
    K13
    K14
    K15
  end
  subgraph pyrox
    P1
    P2
    P3
    P4
    P5
    P6
    P7
    P8
    P9
    P10
  end
  subgraph manipy
    M0
    M1
    M2
    M3
    M4
    M5
  end
  subgraph plumax
    X1
  end
  G1 ==> G4
  G3 --> G5
  G4 ==> G5
  G1 ==> G6
  G3 --> G6
  G5 ==> G6
  G1 --> G7
  G3 --> G7
  G4 --> G7
  G6 ==> G8
  G8 ==> G10
  G11 --> G12
  G12 --> G13
  G11 --> G15
  G12 --> G16
  G12 --> G17
  G13 --> G17
  G1 ==> K2
  K1 ==> K2
  K2 ==> K3
  K3 ==> K4
  G2 --> K5
  K3 --> K5
  K2 ==> K6
  K3 ==> K6
  K4 ==> K6
  G11 --> K7
  G14 --> K8
  G13 --> K9
  G14 --> K9
  G12 --> K10
  K5 --> K11
  K6 --> K11
  K8 --> K11
  K9 --> K11
  K10 --> K11
  K12 --> K11
  K13 --> K11
  K14 --> K11
  K15 --> K11
  K2 --> K15
  K6 --> K15
  K3 --> P2
  K4 --> P2
  K8 --> P3
  G13 --> P4
  K8 --> P5
  G6 ==> P7
  G7 --> P7
  K6 ==> P7
  P6 --> P7
  G8 ==> P8
  G9 --> P8
  G10 ==> P8
  P7 ==> P8
  P8 ==> P9
  G2 --> M1
  K2 --> M1
  K5 --> M1
  M0 --> M1
  K2 --> M2
  M0 --> M2
  M1 --> M3
  M2 --> M3
  M3 --> M4
  M1 --> M5
  M2 --> M5
  M3 --> M5
  G12 --> X1
  classDef crit fill:#f9d6d5,stroke:#b03a2e,stroke-width:2px
  class G1,G4,G5,G6,G8,G10,K1,K2,K3,K4,K6,P7,P8,P9 crit
```

Soft dependencies (a test or an optional path only) are left out of the
graph. They don't block merging:

- K4 needs K3 only for its tests;
- K6's graph-Matérn ↔ SPDE test needs G7 (integration tier);
- K13 accepts K2 `Graph`s in `laplacian_penalty` once K2 has shipped (done: #153);
- G6's sparse-Cholesky sampling path needs G4;
- P4 is also blocked outside this plan, by gaussx#312 (fixed by gaussx#513 in v0.6.1).

## 3. Critical paths and slack

There are two zero-slack chains, and they meet at P7:

1. **INLA:** G1 → G4 → G5 → G6 → G8 → G10 → P8 → P9 (8 waves, the
   longest chain).
2. **Graphs:** K1 → K2 → K3 → K4 → K6 → P7 → P8. It depends on G1 through
   K2, so **G1 heads both chains**.

The other long chains have one wave of slack: manipy
(K2 → K3 → K5 → M1 → M3 → M4, M5) and docs (K6 → K15 → K11).

Sizes are rough (S about a week, M two to three, L four or more, for one
person) and only there to spot overloaded waves.

| Phase | Repo | What | Size | Needs | Earliest wave | Latest wave | Slack |
|---|---|---|---|---|---|---|---|
| G1 | [gaussx](roadmap-gaussx.md) | `SparseOperator`, static pattern | M | — | 1 | 1 | **0** |
| G2 | [gaussx](roadmap-gaussx.md) | `eigh_generalized` | M | — | 1 | 4 | 3 |
| G3 | [gaussx](roadmap-gaussx.md) | Structured selected inverses, shifted Kronecker; #344 | M | — | 1 | 2 | 1 |
| G4 | [gaussx](roadmap-gaussx.md) | Sparse Cholesky, Takahashi, VJPs | L | G1 | 2 | 2 | **0** |
| G5 | [gaussx](roadmap-gaussx.md) | `pseudo_logdet` | S | G3, G4 | 3 | 3 | **0** |
| G6 | [gaussx](roadmap-gaussx.md) | `GaussianMRF`, `IntrinsicGMRF` | M | G1, G3, G5 | 4 | 4 | **0** |
| G7 | [gaussx](roadmap-gaussx.md) | Precision builders, SPDE, FEM | L | G1, G3, G4 | 3 | 5 | 2 |
| G8 | [gaussx](roadmap-gaussx.md) | `laplace_mode`; binomial, neg. binomial | L | G6 | 5 | 5 | **0** |
| G9 | [gaussx](roadmap-gaussx.md) | `theta_design` | S | — | 1 | 6 | 5 |
| G10 | [gaussx](roadmap-gaussx.md) | `vb_mean_correction` | M | G8 | 6 | 6 | **0** |
| G11 | [gaussx](roadmap-gaussx.md) | Sketches, `hadamard_transform` | M | — | 1 | 4 | 3 |
| G12 | [gaussx](roadmap-gaussx.md) | Range finder, randomized SVD / eigh | M | G11 | 2 | 5 | 3 |
| G13 | [gaussx](roadmap-gaussx.md) | `randomized_nystrom`; #354 | M | G12 | 3 | 6 | 3 |
| G14 | [gaussx](roadmap-gaussx.md) | `rp_cholesky`; #371 | M | — | 1 | 6 | 5 |
| G15 | [gaussx](roadmap-gaussx.md) | Sketch-and-precondition LSMR | M | G11 | 2 | 8 | 6 |
| G16 | [gaussx](roadmap-gaussx.md) | `diag_inv(method="xdiag")` | M | G12 | 3 | 8 | 5 |
| G17 | [gaussx](roadmap-gaussx.md) | Tier 2 (on demand) | M | G12, G13 | 4 | 8 | 4 |
| K1 | [kernellib](roadmap-kernellib.md) | Move-only `_graph/` refactor | S | — | 1 | 1 | **0** |
| K2 | [kernellib](roadmap-kernellib.md) | `Graph` types and builders, `graph_from_edges` | L | G1, K1 | 2 | 2 | **0** |
| K3 | [kernellib](roadmap-kernellib.md) | `laplacian_eigpairs` | M | K2 | 3 | 3 | **0** |
| K4 | [kernellib](roadmap-kernellib.md) | Graph spectra, graph Matérn | S | K3 | 4 | 4 | **0** |
| K5 | [kernellib](roadmap-kernellib.md) | Eigenmap extensions, SEP, kernel LPP | M | G2, K3 | 4 | 5 | 1 |
| K6 | [kernellib](roadmap-kernellib.md) | GMRF structure, `mesh_graph` | M | K2, K3, K4 | 5 | 5 | **0** |
| K7 | [kernellib](roadmap-kernellib.md) | `hadamard_transform` → gaussx | S | G11 | 2 | 8 | 6 |
| K8 | [kernellib](roadmap-kernellib.md) | `select_landmarks` | M | G14 | 2 | 7 | 5 |
| K9 | [kernellib](roadmap-kernellib.md) | Preconditioned `KRR` | M | G13, G14 | 4 | 7 | 3 |
| K10 | [kernellib](roadmap-kernellib.md) | Randomized `KernelPCA` | S | G12 | 3 | 7 | 4 |
| K11 | [kernellib](roadmap-kernellib.md) | Docs | M | K5, K6, K8, K9, K10, K12, K13, K14, K15 | 7 | 8 | 1 |
| K12 | [kernellib](roadmap-kernellib.md) | Dependence numerics (#93, #94), `CKAAccumulator` | S | — | 1 | 7 | 6 |
| K13 | [kernellib](roadmap-kernellib.md) | Quadratic-penalty KRR | M | — | 1 | 7 | 6 |
| K14 | [kernellib](roadmap-kernellib.md) | `KernelPCA` supervision, pre-images | M | — | 1 | 7 | 6 |
| K15 | [kernellib](roadmap-kernellib.md) | Delaunay / Gabriel / RNG graphs | S | K2, K6 | 6 | 7 | 1 |
| P1 | [pyrox](roadmap-pyrox.md) | `latent_init` | S | — | 1 | 8 | 7 |
| P2 | [pyrox](roadmap-pyrox.md) | Graph Matérn inducing features | M | K3, K4 | 5 | 8 | 3 |
| P3 | [pyrox](roadmap-pyrox.md) | `init_inducing` | S | K8 | 3 | 8 | 5 |
| P4 | [pyrox](roadmap-pyrox.md) | Large-n exact-GP recipe (gaussx#312) | M | G13 | 4 | 8 | 4 |
| P5 | [pyrox](roadmap-pyrox.md) | LogFalkon centres | S | K8 | 3 | 8 | 5 |
| P6 | [pyrox](roadmap-pyrox.md) | pyrox-lgm scaffold | S | — | 1 | 5 | 4 |
| P7 | [pyrox](roadmap-pyrox.md) | LGM components, spatial priors, PC priors | L | G6, G7, K6, P6 | 6 | 6 | **0** |
| P8 | [pyrox](roadmap-pyrox.md) | `LGM`, `inla()` | L | G8, G9, G10, P7 | 7 | 7 | **0** |
| P9 | [pyrox](roadmap-pyrox.md) | Diagnostics, sugar, hybrids | M | P8 | 8 | 8 | **0** |
| P10 | [pyrox](roadmap-pyrox.md) | Repo-level docs and pins | S | — | 1 | 8 | 7 |
| M0 | [manipy](roadmap-manipy.md) | Scaffold, SSH remote | S | — | 1 | 5 | 4 |
| M1 | [manipy](roadmap-manipy.md) | Manifold alignment | M | G2, K2, K5, M0 | 5 | 6 | 1 |
| M2 | [manipy](roadmap-manipy.md) | HSI workflow, metrics, datasets | M | K2, M0 | 3 | 6 | 3 |
| M3 | [manipy](roadmap-manipy.md) | Isomap, LLE, diffusion maps | L | M1, M2 | 6 | 7 | 1 |
| M4 | [manipy](roadmap-manipy.md) | KEMA, t-SNE (on demand) | M | M3 | 7 | 8 | 1 |
| M5 | [manipy](roadmap-manipy.md) | Docs, reproduction notebooks | M | M1, M2, M3 | 7 | 8 | 1 |
| X1 | [plumax](roadmap-plumax.md) | `randomized_svd` backgrounds | S | G12 | 3 | 8 | 5 |

## 4. Release gates

Only cross-repo edges need a release. Each row is a release that must be
cut at the end of the given wave, and it must contain the listed phases.
Intra-repo dependencies (G4 on G1, for example) only need a merge.

| After wave | Release | Must contain | Unblocks |
|---|---|---|---|
| 1 | gaussx | G1, G2, G9, G11, G14 | G1 → K2; G2 → K5, M1; G9 → P8; G11 → K7; G14 → K8, K9 |
| 2 | gaussx | G12 | G12 → K10, X1 |
| 2 | kernellib | K2, K8 | K2 → M1, M2; K8 → P3, P5 |
| 3 | gaussx | G7, G13 | G7 → P7; G13 → K9, P4 |
| 3 | kernellib | K3 | K3 → P2 |
| 4 | gaussx | G6 | G6 → P7 |
| 4 | kernellib | K4, K5 | K4 → P2; K5 → M1 |
| 5 | gaussx | G8 | G8 → P8 |
| 5 | kernellib | K6 | K6 → P7 |
| 6 | gaussx | G10 | G10 → P8 |

In practice:

- **gaussx** releases after waves 1–6;
- **kernellib** after waves 2–5, plus a wave-7 release for the docs and
  anything else merged;
- **pyrox** releases as its phases land. Nothing downstream pins it.

## 5. Sequence per repo

Within a wave, start the zero-slack phases first. The rest can slide as
far as their latest wave (§3).

### gaussx

| Wave | PRs, least slack first (bold: zero slack) |
|---|---|
| 1 | **G1**, G3, G2, G11, G9, G14 |
| 2 | **G4**, G12, G15 |
| 3 | **G5**, G7, G13, G16 |
| 4 | **G6**, G17 |
| 5 | **G8** |
| 6 | **G10** |

- **Start G4 early.** A JAX sparse Cholesky with symbolic analysis is the
  largest technical risk on the critical path. Prototype the symbolic
  factorisation and the Takahashi sweep during wave 1, next to G1, even
  though G4 formally starts in wave 2. The opt-in CHOLMOD backend in the
  same phase is the fallback, if the pure-JAX numeric factorisation is
  slow.
- **G13** (#354) and **G3** are the highest-value early fixes.

### kernellib

| Wave | PRs, least slack first (bold: zero slack) |
|---|---|
| 1 | **K1**, K12, K13, K14 |
| 2 | **K2**, K8, K7 |
| 3 | **K3**, K10 |
| 4 | **K4**, K5, K9 |
| 5 | **K6** |
| 6 | K15 |
| 7 | K11 |

- **K12** (#93, #94) ships in wave 1: it fixes live bugs and needs nothing.
- **K13 and K14** have six waves of slack. Schedule them to fill gaps
  around the K1 → K6 chain, not in front of it.
- **K2** is the largest kernellib phase and sits on the critical path.
  Split its PR into types and operators first, then builders
  (`graph_from_edges`, `knn_graph(ensure_connected=)`, `radius_graph`,
  `grid_graph`). Both land before the wave-2 release.

### pyrox

| Wave | PRs, least slack first (bold: zero slack) |
|---|---|
| 1 | P6, P1, P10 |
| 3 | P3, P5 |
| 4 | P4 |
| 5 | P2 |
| 6 | **P7** |
| 7 | **P8** |
| 8 | **P9** |

- **P6** (the pyrox-lgm scaffold) has four waves of slack, but it is cheap.
  Do it in wave 1, so that P7 can start the moment G6, G7 and K6 are
  released.
- **P4** waits for gaussx#312 as well as G13. Track #312 as an external
  blocker. (#312 was fixed by gaussx#513, v0.6.1.)

### manipy

| Wave | PRs, least slack first (bold: zero slack) |
|---|---|
| 1 | M0 |
| 3 | M2 |
| 5 | M1 |
| 6 | M3 |
| 7 | M4, M5 |

### plumax

| Wave | PRs, least slack first (bold: zero slack) |
|---|---|
| 3 | X1 |

## 6. Milestones

| Milestone | Phases | Earliest | What it proves |
|---|---|---|---|
| **A. Numerics fixed** | K12 | wave 1 | CKA and HSIC are safe as training penalties (#93, #94) |
| **B. RandNLA core** | G11–G14, K7, K8, K10, X1, P3, P5 | wave 3 | Randomized factorisations, landmark and inducing selection; plumax off scikit-learn |
| **C. Preconditioned kernels** | G13, K9 (and P4 once #312 is fixed; it was, in gaussx v0.6.1) | wave 4 | KRR and exact GPs at n = 10⁵ without forming K |
| **D. Graphs** | K1–K6, K15, P2 | wave 6 | Sparse graphs end to end: builders, eigenpairs, graph Matérn, GMRF structure |
| **E. First `inla()`** | G1–G10, P6–P8 | wave 7 | The Scotland BYM2 and POD fixtures match R-INLA ([gallery 1](roadmap-examples.md#ex-bym2), [3](roadmap-examples.md#ex-pod)) |
| **F. manipy 0.1** | M0–M3, M5 | wave 7 | Alignment and classical DR in JAX, and the Indian Pines reproduction |
| **G. Docs** | K11, P10 | wave 7 | Notebooks for graphs, dependence penalties and landmark selection |
| **H. Complete** | P9, M4, G17 | wave 8 | Diagnostics and the on-demand extras |

## 7. First moves

In order:

1. **File the issues.** File one issue per phase in the owning repo, each
   linking to its roadmap section, with "blocked by" links from §2. Add
   one epic per repo. kernellib's `make gh-sub` and `make gh-block`
   helpers (and the same scripts in gaussx and pyrox) set the native
   links.
2. **Ship K12** (kernellib#93, #94): small, independent, fixes live bugs.
3. **Start the critical heads in parallel:** G1, and K1 (move-only, so
   quick). G1 is the head of both zero-slack chains.
4. **Fill wave 1 in gaussx:** G3 (with #344), G11, G14, then G2 and G9.
   Spike G4's symbolic analysis alongside.
5. **Cheap wave-1 scaffolding:** P1, P6, P10 in pyrox; M0 in manipy.
6. **Cut gaussx release 1** (G1, G2, G9, G11, G14; §4). That unblocks
   K2, K7 and K8 in wave 2.

## 8. Schedule risks

| Risk | Where | Mitigation |
|---|---|---|
| Sparse Cholesky in JAX is slow or hard to make traceable | G4, critical path | Spike in wave 1; the CHOLMOD backend (same phase) is the fallback; level scheduling and supernodes are follow-ups, not blockers |
| Release latency on cross-repo edges | every gate in §4 | Batch each repo's release per wave; never release per phase |
| K2 grows too large for one PR | critical path | Split into two PRs inside wave 2 (types and operators, then builders) |
| gaussx#312 stays open | P4 | P4 has four waves of slack and nothing depends on it. Did not happen: #312 was fixed in gaussx v0.6.1 |
| On-demand phases creep into the plan | G17, M4 | They start only when a user asks; they are off every critical path |
| A roadmap API changes during implementation | any | Update the repo page in the same PR, and regenerate this page's tables if "Needs" changed |
