---
date: 2026-09-30
---

# Project: manifold learning and graphs

The graph and manifold-learning code from the author's 2016–2018 research
(`manipy/manilearn`, `manifold_learning/src/python`,
`manifold_learning/src/matlab/main_functions`) becomes:

- reusable graph primitives in kernellib;
- the linear algebra they need at scale, in gaussx;
- graph GP features and spatial priors, in pyrox;
- a JAX dimensionality-reduction library, manipy.

The work itself is specified per repo in the [roadmap](roadmap.md).
This page keeps the project's motivation, the audit of the old code, and
its open questions.

**The core port is already done.** kernellib #43 (2026-09-26) shipped:

- kNN graphs, adjacency matrices, graph Laplacians and five spectral graph
  kernels;
- `KernelPCA`;
- `LaplacianEigenmaps` and `SchrodingerEigenmaps` (barrier, label and
  spatial-spectral potentials, with Cahill's `tr(L)/tr(V)` scaling);
- `LocalityPreservingProjections`;
- scikit-learn adapters for all of them.

This project covers what is left.

## Where the work lives

| Repo | Phases |
|---|---|
| kernellib | [K1–K5](roadmap-kernellib.md): `_graph/` package, sparse `Graph` with static topology, `grid_graph`, `graph_from_edges` (the seam for city2graph / libpysal / NetworkX), `knn_graph(ensure_connected=)`, scalable eigenpairs, graph Matérn, eigenmap extensions, SEP, kernel LPP / SEP; K15 proximity graphs (Delaunay, Gabriel, RNG); K11 docs |
| gaussx | [G1, G2, G5, G6](roadmap-gaussx.md): `SparseOperator`, `eigh_generalized`, `pseudo_logdet`, GMRF distributions (shared with INLA) |
| pyrox | [P1, P2](roadmap-pyrox.md): `latent_init`, graph Matérn inducing features; [P7](roadmap-pyrox.md): the CAR / ICAR / Leroux / BYM2 spatial priors, as pyrox-lgm components |
| manipy | [M0–M5](roadmap-manipy.md): fresh JAX rewrite; manifold alignment, more DR methods, hyperspectral workflows, metrics, datasets |

## What you can solve with it

From the [examples gallery](roadmap-examples.md):

- [hyperspectral classification and cross-sensor transfer](roadmap-examples.md#ex-hsi);
- [a GP on a river or road network](roadmap-examples.md#ex-graph-gp);
- [a GPLVM started on the data manifold](roadmap-examples.md#ex-gplvm);
- [disease mapping with BYM2](roadmap-examples.md#ex-bym2) (the spatial priors).

## Old code: what not to port

| Old | Replacement |
|---|---|
| `KnnSolver` (annoy, flann, `LSHForest`) | `nearest_neighbors(backend="pynndescent" \| "sklearn")`; `LSHForest` is gone from scikit-learn |
| megaman / pyamg eigensolver switch, `r_svd`, ARPACK `n + 15` workaround | `laplacian_eigpairs(method=...)` over `gaussx.eig` (K3) |
| `sim_potential` (pandas `groupby`) | `label_potential` |
| MATLAB `Adjacency` disk caching (`saved == 1`) | None; cache in user code |
| scikit-learn estimator classes | `kernellib.sklearn` / `manipy.sklearn` adapters over Equinox modules |
| `external_toolboxes/` (DRToolBox, DengCai, KEMA, SSSE, ...) | Third-party. Read as a menu and for references, never copied |

## Known bugs in the old code

Anyone comparing new results with old ones needs these. None of them are
reproduced in kernellib.

| Where | Bug | Effect |
|---|---|---|
| `PartialLabelsPotential.m:78` | `find(y(y==i))` returns `1:n_i` instead of the positions of class `i` | Must-link constraints joined the *first* `n_i` samples, not the labelled ones. Every SEPL / SSSEPL result from the MATLAB code is affected |
| `GraphEmbedding.m:105-116` (`'sspl'`) | Potentials swapped: `Vss` holds the label potential, `Vpl` the spatial one | α weighted the label potential and β the spatial one |
| `SchroedingerEigenmaps.m` option parser | Sets `spatialsigma` / `weightsigma`; the potential reads `clusterSigma` / `weightSigma` | Both potential bandwidths were always 1 |
| `GraphEmbedding.m` | Always drops the first eigenvector | Wrong with a potential, where the constant vector is not a null vector. kernellib has `drop_first` |
| `SpatialSpectralPotential.m:63` | `'angle'` kernel divides by `‖x₁‖·‖x₁‖` | Cosine weights wrong; the option was also rejected by validation (L129) |
| `manifoldalignmentprojections.m:66-67` | Non-cumulative slice indices | Correct for two domains only |
| `manipy/.../schroedinger.py` | `eigen_solver` undefined in `__init__`; `maximum` not imported in `lpp.py` | The 2018 Python did not run |

**Convention change:** the old heat kernel was `exp(-d²/σ²)` with σ = 1.
kernellib's is `exp(-d²/2σ²)` with the median neighbour distance as the
default σ. Old settings translate as `σ_new = σ_old / √2`.

---

## Open questions

| # | Question | Proposed answer |
|---|---|---|
| 1 | Name clash: kernellib's `KNNGraph` (search result: indices + distances) vs the new weighted `Graph` | Keep `KNNGraph` for 0.x; builders return `Graph`. Revisit a rename (`Neighbors`) before 1.0 |
| 2 | Public `kernellib.graph` namespace or flat top-level names? | Flat, like every other kernellib module. `functional` is the only public submodule and has a different contract (arrays in, arrays out) |
| 3 | geonnax's `graph_laplacian_eigpairs` duplicates kernellib's | geonnax keeps it as a dense basis primitive (it is a basis library). kernellib's `laplacian_eigpairs` is the graph-aware front door; pyrox-gp switches to it in P2 |
| 6 | Reproduce the old Indian Pines / Pavia numbers? | Yes, as a manipy notebook, with the bug fixes and the σ convention called out. Expect different numbers for the partial-label variants |


The questions about `jax.experimental.sparse` and about where the CAR /
ICAR distributions live are cross-project: see the
[roadmap](roadmap.md#roadmap-open-questions).

## Decisions log

| Date | Decision |
|---|---|
| 2026-09-30 | Plan drafted. Graph construction and graph spectral methods stay in kernellib (they serve spatial models and GPs, not just DR); kernel-free sparse linear algebra and GMRFs go to gaussx; manipy is rewritten in JAX on top of kernellib; the old code is a feature list, not code to copy |
| 2026-09-30 | Merged with the INLA project. The sparse operator is shared and gains a static sparsity pattern (so kernellib's `Graph` topology becomes static). The GMRF distributions and the CAR / ICAR / Leroux / BYM2 priors move into the INLA work (now G6 and P7, the priors in the new pyrox-lgm). The BYM2 scaling constant becomes exact through the selected inverse. The position that there would be no sparse Cholesky in JAX is withdrawn |
| 2026-09-30 | Fused with the other two projects into the per-repo [roadmap](roadmap.md), with one phase numbering per repo (the mapping from old ids is in the roadmap README) |
