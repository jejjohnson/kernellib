---
date: 2026-09-30
---

# manipy: roadmap

manipy's share of the [fused roadmap](roadmap.md). manipy becomes the JAX
dimensionality-reduction library. It depends on kernellib for kernels,
graphs and the spectral graph embeddings, and owns everything that only DR
users need. It serves the [manifold](project-manifold.md) project.

> Maths notes (**The maths.**) say where each operation comes from; **Example.** blocks are pseudocode against the *planned* API (`gx` = gaussx, `kl` = kernellib, `px` = pyrox-gp, `lgm` = pyrox-lgm). End-to-end problems are in the [examples gallery](roadmap-examples.md).

| Phase | What | Needs |
|---|---|---|
| M0 | Scaffold, legacy tag, branch rename, SSH remote | — |
| M1 | Manifold alignment (Wang, SSMA, SEMA) | K2, K5, G2 |
| M2 | Hyperspectral workflow, metrics, datasets | K2 |
| M3 | Isomap, the LLE family, diffusion maps, Nyström out-of-sample | M1, M2 |
| M4 | KEMA, t-SNE (on demand) | M3 |
| M5 | Documentation and reproduction notebooks | M1, M2 |

---

## 1. Current state

| Repo | Contents | Fate |
|---|---|---|
| `jejjohnson/manipy` (branch `master`, last commit 2018-02-25) | `manilearn/`: LE, LPP, Schrödinger eigenmaps, graph and neighbour utilities, and a megaman / pyamg eigensolver switch; numpy / scikit-learn / annoy / pyflann | Its algorithms are all in kernellib now (#43), cleaner. Several modules never ran (`eigen_solver` undefined, `maximum` not imported, `linear_graph_embedding` returns `None`). **Nothing is carried forward as code** |
| `jejjohnson/manifold_learning` (2017) | Python LE / LPP / SE, `ssma.py` (manifold alignment), HSI data helpers; MATLAB `main_functions/` (the reference implementations) and experiment scripts; vendored `external_toolboxes/` | Manifold alignment and the HSI workflow are the only pieces not yet ported anywhere. They are specified below from the MATLAB, **with its bugs fixed** (see [the manifold project](project-manifold.md#known-bugs-in-the-old-code)) |

Old code is a feature list and a numerical reference, not code to copy.
This is the same rule kernellib applied to its own 2018 code.

---

## 2. Scope

**In:**

- manifold alignment;
- non-spectral-graph DR methods;
- out-of-sample extension;
- hyperspectral (HSI) workflows;
- embedding-quality and classification metrics;
- dataset loaders;
- scikit-learn adapters for all of the above.

**Out (lives in kernellib):**

- graphs;
- Laplacians;
- graph kernels;
- Laplacian / Schrödinger eigenmaps, LPP, SEP, and kernel LPP / SEP;
- kernel PCA.

manipy **uses** these and never re-exports them. Users import
`kernellib.LaplacianEigenmaps` directly, and the docs say so.

**Boundaries:** manipy never imports NumPyro. The core never imports
scikit-learn (adapters only, behind an extra). It never imports pyrox.

---

## 3. Package layout

```
src/manipy/
├── __init__.py                 # flat public API; __all__ is the contract
├── _alignment/
│   ├── _domains.py             # Domain container; joint block assembly (Z, block-diagonal graphs)
│   ├── _class_graphs.py        # same-/different-label quadratic forms across domains; SSMA rescaling
│   ├── _linear.py              # ManifoldAlignment: "wang", "ssma", "sema"
│   └── _kernel.py              # KernelManifoldAlignment (KEMA): phase M4
├── _embeddings/
│   ├── _isomap.py              # Isomap, landmark Isomap
│   ├── _lle.py                 # LLE, modified LLE, Hessian LLE, LTSA
│   ├── _diffusion_maps.py      # diffusion maps with anisotropic normalisation α
│   └── _tsne.py                # phase M4
├── _out_of_sample.py           # Nyström extension for LE / SE / diffusion maps
├── _hsi/
│   ├── _image.py               # image_to_array / array_to_image (row-major), pixel_coordinates, pixel_graph
│   ├── _potentials.py          # image-level Schrödinger potentials on top of kernellib
│   └── _splits.py              # stratified_split(labels, *, fraction | count, key)
├── _metrics/
│   ├── _quality.py             # trustworthiness, continuity, LCMC, knn_preservation
│   └── _classification.py      # overall_accuracy, average_accuracy, cohen_kappa (+ variance), per_class_accuracy
├── _datasets/
│   ├── _synthetic.py           # swiss_roll, s_curve, severed_sphere (JAX, keyed)
│   └── _hsi.py                 # indian_pines, pavia_university, salinas (pooch download, checksums)
└── sklearn/                    # adapters, extra manipy[sklearn]
```

**Conventions** (identical to kernellib):

- Estimators are `eqx.Module`s: configuration in the constructor, `fit`
  returns the fitted module, and fitted fields are `None` before `fit`.
- Pure functions for helpers; jaxtyping shapes; einx for every reshape and
  contraction.
- Google docstrings with executable `Examples:`; doctests on.
- `tests/test_public_api.py` and `tests/test_imports.py` from the template.

**Dependencies:**

- **Core:** `jax`, `equinox`, `jaxtyping`, `einx`, `kernellib` (pinned by
  git tag until it is on PyPI), and `gaussx` (transitively, also used
  directly for `eigh_generalized` and `SparseOperator`).
- **Extras:**
  - `sklearn` (adapters);
  - `neighbors` (pynndescent, forwarded to kernellib);
  - `data` (`pooch`, plus `scipy.io` for `.mat` files).

---

## 4. Manifold alignment (M1)

**The maths.** **The problem.** Two sensors (say HyMap and AVIRIS) see the same kind of
scene through different bands, so their data live in different spaces
$\mathbb R^{D_1}$ and $\mathbb R^{D_2}$.

**The approach.** Alignment looks for projections $f_1, f_2$ into one
shared space in which three things hold at once:

- each domain's geometry is kept, through $\operatorname{tr}(F^\top Z^\top L_gZF)$;
- same-class samples meet, across domains, through
  $\operatorname{tr}(F^\top Z^\top L_sZF)$;
- different classes are pushed apart, as the constraint
  $F^\top Z^\top L_dZF = I$.

That is one generalised eigenproblem (gaussx G2) of size
$\sum_i D_i$, whatever the number of pixels.

### 4.1 Problem

There are `m` domains. Domain `i` has data `X_i` (`N_i × D_i`), labels
`y_i` (`N_i`, with `-1` for unlabelled), and optionally a spatial graph
`S_i` (for example `kl.grid_graph(image_i.shape[:2])`).

Joint quantities:

- `Z = blkdiag(X̄_1, …, X̄_m)`, of size `ΣN × ΣD`, where `X̄_i` is `X_i`
  centred per domain.
- Geometry: `W_g = blkdiag(W_1, …, W_m)`, with `W_i` the k-NN graph of
  `X_i` (`kl.knn_graph`). There are no cross-domain geometric edges.
  `L_g` and `D_g` are its Laplacian and degree matrix.
- Class graphs over all labelled samples, within and across domains:
  - `W_s[a, b] = 1` if `y_a = y_b`;
  - `W_d[a, b] = 1` if `y_a ≠ y_b`;
  - `L_s` and `L_d` are their Laplacians.

The alignment solves the generalised problem `A f = λ B f` in the `ΣD`
feature space. `f` stacks the projection vectors of all domains.

| `method` | `A` | `B` | Reference |
|---|---|---|---|
| `"wang"` | `Zᵀ(L_g + μ L_s)Z` | `Zᵀ D_g Z` | Wang & Mahadevan (2011) |
| `"ssma"` | `Zᵀ((1−μ) L_g + μ L_s)Z` | `Zᵀ L_d Z` | Tuia et al. (2014) |
| `"sema"` | `Zᵀ((1−μ)(L_g + α̃ P) + μ L_s)Z` | `Zᵀ D_g Z` | The author's Schrödinger alignment |

Details:

- **Ridge.** A ridge `λ_r` is added to both sides, `A + λ_r ZᵀZ` and
  `B + λ_r ZᵀZ`. This matches the MATLAB, which added `λI` in `N` space
  before projection.
- **SEMA's potential.** `P = blkdiag(P_1, …, P_m)`, with
  `P_i = kl.spatial_spectral_graph(X_i, S_i).laplacian_operator()`.
  `α̃ = α · tr(L_g)/tr(P)` when `normalize_potential=True`, which is the
  default. The MATLAB did not normalise, so pass
  `normalize_potential=False` to reproduce it.
- **SSMA rescaling.** Each class graph is rescaled so that its total weight
  equals that of `W_g`, after adding the identity (Tuia's reference code).
  The identity adds self-loops, which cancel in `L = D − W`. So it changes
  only the rescaling constant, not the Laplacian's structure. Implement it
  as a scalar and say so in a comment.
- **Solver.** `gaussx.eigh_generalized(A, B, rank=n_components)`, smallest
  first. No trivial solution is dropped, because the projected problem has
  none. `n_components ≤ ΣD`.

### 4.2 Efficient assembly

The class graphs are never formed at `ℓ × ℓ` size, where `ℓ` is the
number of labelled samples.

- For a class `c` with labelled rows `Z_c`,
  `Zᵀ L_{K_c} Z = n_c Z_cᵀZ_c − (Z_cᵀ1)(Z_cᵀ1)ᵀ`. Here `K_c` is the
  complete graph on class `c`.
- Summing over classes gives `Zᵀ L_s Z`.
- `L_d = L_{K_ℓ} − L_s`, where `K_ℓ` is the complete graph on all
  labelled samples, and `Zᵀ L_{K_ℓ} Z` has the same closed form.
- This costs `O(ℓ · (ΣD)²)` time and `O((ΣD)²)` memory. The MATLAB used
  `O(ℓ²)`.
- The geometric terms `Zᵀ L_g Z` and `Zᵀ D_g Z` are computed per domain,
  as `X̄_iᵀ L_i X̄_i` and so on, using the sparse `laplacian_operator()`.

### 4.3 API

```python
class ManifoldAlignment(eqx.Module):
    method: Literal["wang", "ssma", "sema"] = "ssma"
    n_components: int = 10
    mu: float = 0.5
    alpha: float = 1.0  # "sema" only
    normalize_potential: bool = True  # "sema" only
    ridge: float = 1e-6  # relative to tr(ZᵀZ)/ΣD
    n_neighbors: int = 10
    weighting: kl.Weighting = "heat"
    bandwidth: float | None = None
    standardize: bool = (
        False  # SSMA post-projection z-scoring, fitted on labelled embeddings
    )
    projections: tuple[Float[Array, "D_i n"], ...] | None = None
    means: tuple[Float[Array, " D_i"], ...] | None = None
    eigenvalues: Float[Array, " n"] | None = None

    def fit(
        self,
        X: Sequence[Array],
        y: Sequence[Array],
        spatial_graphs: Sequence[kl.AbstractGraph] | None = None,
    ) -> ManifoldAlignment: ...
    def transform(self, X: Array, *, domain: int) -> Float[Array, "M n"]: ...
```

Per-domain projections are slices of the stacked eigenvectors at
**cumulative** offsets. This fixes the MATLAB bug in
`manifoldalignmentprojections.m`, which worked for two domains only.

**Example.**

```python
ma = manipy.ManifoldAlignment(method="ssma", n_components=15, mu=0.5)
ma = ma.fit([X_hymap, X_aviris], [y_hymap, y_aviris])  # −1 marks unlabelled pixels
Z_train = ma.transform(X_hymap_labelled, domain=0)
Z_test = ma.transform(X_aviris_scene, domain=1)  # the other sensor, in the same space
pred = (
    SVC().fit(Z_train, y_hymap_labelled).predict(Z_test)
)  # train on one sensor, map the other
```

### 4.4 Tests

- Two domains that are rotations of the same data, with a few labels:
  aligned embeddings of matching points are close (the Procrustes error
  falls as `μ` grows).
- Three domains with different `D_i`: `transform` shapes are right, and
  the offsets are cumulative (this is the regression test for the MATLAB
  bug).
- The closed-form class quadratic forms equal the dense `Zᵀ L_s Z` and
  `Zᵀ L_d Z` on a small problem.
- The SSMA rescaling equals the MATLAB formula on a fixture (a hand
  computation, not a MATLAB run).
- `"sema"` with `alpha=0` and weight `μ` gives the same eigenvectors as
  `"wang"` with `μ' = μ/(1−μ)`. The eigenvalues are scaled by `1−μ`,
  because `A_sema = (1−μ)·A_wang` and `B` is the same.

---

## 5. Other methods (M3)

**The maths.** - **Isomap.** Double-centre the squared geodesic distances,
  $B = -\tfrac12 JD^{(2)}J$ with $J = I-\tfrac1n\mathbf 1\mathbf 1^\top$.
  The embedding is the top eigenvectors, scaled by $\sqrt\lambda$
  (classical MDS).
- **LLE.** The weights
  $W = \arg\min\sum_i\|x_i-\sum_jW_{ij}x_j\|^2$, with rows summing to 1
  over neighbours, need one regularised $k\times k$ solve per point. The
  embedding is the bottom eigenvectors of $(I-W)^\top(I-W)$, dropping the
  constant.
- **Diffusion maps.** $K^{(\alpha)} = D^{-\alpha}KD^{-\alpha}$, then
  $P = D_\alpha^{-1}K^{(\alpha)}$. The coordinates are
  $\lambda_k^t\psi_k$. With $\alpha = 1$ the density's influence is
  removed, and $P$ approximates the Laplace–Beltrami operator (Coifman &
  Lafon, 2006).

| Method | Notes |
|---|---|
| Isomap | Geodesics on `kl.knn_graph`, via `scipy.sparse.csgraph` shortest paths (CPU, not traced; documented), then classical MDS. Landmark Isomap (landmark MDS) for large `N` |
| LLE, modified LLE | Reconstruction weights: one regularised `k × k` solve per point, `vmap`ped over neighbourhoods. Embedding: the smallest eigenvectors of `(I−W)ᵀ(I−W)` as a `gaussx.SparseOperator`, through `gaussx.eig(rank=)` or kernellib's ARPACK path |
| Hessian LLE, LTSA | Same pattern: local SVDs `vmap`ped, a global sparse alignment matrix, smallest eigenpairs |
| Diffusion maps | Coifman & Lafon: anisotropic normalisation `α ∈ {0, ½, 1}` on a kernel Gram or k-NN graph, eigenpairs of the Markov matrix through the symmetric conjugate, diffusion time `t` |
| Out-of-sample (Nyström) | Bengio et al. (2004): for a random-walk-normalised embedding, `y_k(x) = (1/(1−λ_k)) Σ_j [w(x, x_j)/d(x)] y_jk`. It works with kernellib's LE / SE and with diffusion maps. Test: in-sample points reproduce their embedding |

Later (M4):

- **KEMA** (Tuia & Camps-Valls, 2016): the alignment of §4 with `Z`
  replaced by `blkdiag(K_1, …, K_m)` from kernellib kernels, in `ΣN`
  space. `approx=` feature maps make it scale, as `KernelPCA` does.
- **t-SNE:** exact for small `N`, and Barnes-Hut or FFT-based only if there
  is demand.

**Diffusion maps stay here** rather than in kernellib. They are a DR
method, not a kernel or graph primitive, and no spatial model or GP needs
them. Move them if one does.

---

**Example.**

```python
roll, colour = manipy.datasets.swiss_roll(2000, key=key)
Y_iso = manipy.Isomap(n_components=2, n_neighbors=12).fit(roll).embedding
Y_dm = manipy.DiffusionMaps(n_components=2, alpha=1.0, t=4).fit(roll).embedding
Y_le = (
    kl.LaplacianEigenmaps(n_components=2, n_neighbors=12).fit(roll).embedding
)  # from kernellib
scores = {
    name: manipy.metrics.trustworthiness(roll, Y, k=10)
    for name, Y in {"isomap": Y_iso, "diffusion": Y_dm, "eigenmaps": Y_le}.items()
}
```

## 6. Hyperspectral workflow (M2)

- **`image_to_array(image) -> (X, shape)` and `array_to_image(X, shape)`**
  - Row-major, matching `kl.GridGraph`'s node order.
  - The docstring warns that the 2016–2018 code was column-major, so
    indices from old experiments do not carry over.
- **`pixel_coordinates(shape)`** and **`pixel_graph(shape, *,
  connectivity="face")`**, a thin wrapper over `kl.grid_graph`.
- **`spatial_spectral_potential_image(image, *, bandwidth=None)`**:
  `kl.spatial_spectral_graph(X, pixel_graph(shape)).laplacian_operator()`.
  This is the grid version of Cahill's potential, with no coordinate
  search.
- **`partial_label_potential(labels)`**: forwards to `kl.label_potential`,
  and to a sparse variant once kernellib has one. It links the labelled
  pixels, which is what the MATLAB `PartialLabelsPotential.m` intended.
- **`stratified_split(labels, *, fraction=None, count=None, key)`**: per
  class, ignores the background label (`0` by default; configurable).
- **Metrics:** overall accuracy, average accuracy, Cohen's κ and its
  variance (Congalton & Green), per-class accuracy, and the confusion
  matrix.
- **Datasets:** `indian_pines()`, `pavia_university()`, `salinas()`
  - Downloaded with `pooch` from the EHU GIC mirror, with pinned
    checksums. Returns `(image, ground_truth)`.
  - Needs the `data` extra and is never imported by the core.
  - Tests are marked `integration` and skipped offline.

---

**Example.**

```python
# Indian Pines: spatial-spectral Schrödinger eigenmaps, then an SVM with 10 % of labels
cube, gt = manipy.datasets.indian_pines()
X, shape = manipy.hsi.image_to_array(cube)
V = manipy.hsi.spatial_spectral_potential_image(cube)  # grid topology, spectral weights
Y = (
    kl.SchrodingerEigenmaps(n_components=30, n_neighbors=20, alpha=17.8)
    .fit(X, V)
    .embedding
)
train, test = manipy.hsi.stratified_split(
    einx.rearrange("h w -> (h w)", gt), fraction=0.1, key=key
)
pred = SVC().fit(Y[train], gt_flat[train]).predict(Y[test])
oa, aa, kappa = (
    manipy.metrics.overall_accuracy(gt_flat[test], pred),
    manipy.metrics.average_accuracy(gt_flat[test], pred),
    manipy.metrics.cohen_kappa(gt_flat[test], pred),
)
```

## 7. Phases

### M0: scaffold

1. Tag the current `master` as `v0.0.0-legacy` and keep it on a `legacy`
   branch.
2. Rename the default branch to `main` (owner's decision; the plan assumes
   it).
3. Scaffold a fresh tree from `pypackage_template`, the same stack as
   kernellib:
   - uv and hatchling;
   - ruff and ty;
   - pytest with doctests and `slow` / `integration` tiers;
   - the two-tool docs pipeline (mystmd and MkDocs);
   - release-please;
   - `CLAUDE.md`, `AGENTS.md` and `CODE_REVIEW.md`, adapted from kernellib.
4. Check the PyPI name `manipy`. Fallback: `manipy-jax`.
5. **`manifold_learning`:** add a README pointer to manipy and archive the
   repo on GitHub. Owner's decision; nothing in it is needed after M1 and
   M2 except as a reference.
6. The remote URL in the local clone embeds a token. Switch it to SSH
   (`git@github.com:jejjohnson/manipy.git`) before the first push.

### M1: manifold alignment

- Needs kernellib K2 and K5, and gaussx G2.
- §4, plus a scikit-learn adapter. The adapter's `fit(X, y)` takes lists
  of per-domain arrays and exposes `transform(X, domain=)`.
- Multi-domain input cannot satisfy `check_estimator`, so the adapter is
  tested directly and documented as a partial fit of the scikit-learn
  contract.

### M2: HSI workflow, metrics, datasets

- Needs kernellib K2 for `grid_graph`.
- §6 and the `_quality.py` metrics.

### M3: Isomap, LLE family, diffusion maps, out-of-sample

- §5 first table. Each estimator lands in its own PR, with its adapter
  and `check_estimator`.

### M4: KEMA, t-SNE

- Only on demand.

### M5: documentation and reproduction

- Notebooks:
  - alignment on synthetic multi-view data;
  - Schrödinger eigenmaps on Indian Pines with a pixel-grid potential
    (kernellib estimators, manipy workflow);
  - a comparison of DR methods on the swiss roll with the quality
    metrics.
- **Reproduction notebook:** re-run the 2016–2017 Indian Pines / Pavia
  experiments, reporting OA / AA / κ against dimension with an SVM (the
  `sklearn` extra). The fixes to call out:
  - partial labels (every SEPL / SSSEPL number changes);
  - the swapped α/β;
  - the σ option names;
  - the heat-kernel convention, `σ_new = σ_old/√2`.

M0 can start immediately. M1 is the priority: it is the author's own
research and nothing else in the stack has it.
