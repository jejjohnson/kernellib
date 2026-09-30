---
date: 2026-09-30
---

# kernellib: roadmap

kernellib's share of the [fused roadmap](roadmap.md). kernellib owns
everything with a kernel or a graph in it:

- graph construction, Laplacians, spectral graph kernels and Laplacian
  eigenpairs;
- the canonical graph embeddings (Laplacian / Schrödinger eigenmaps, LPP,
  SEP, kernel LPP / SEP);
- the structure matrices that GMRF priors are built from;
- the kernel-specific uses of randomized linear algebra (landmark
  selection, preconditioned KRR, randomized kernel PCA).

It serves the [manifold](project-manifold.md),
[RandNLA](project-rnla.md) and [INLA](project-inla.md) projects.
The baseline is kernellib 0.0.11.

> Maths notes (**The maths.**) say where each operation comes from; **Example.** blocks are pseudocode against the *planned* API (`gx` = gaussx, `kl` = kernellib, `px` = pyrox-gp, `lgm` = pyrox-lgm). End-to-end problems are in the [examples gallery](roadmap-examples.md).

| Phase | What | Projects | Needs |
|---|---|---|---|
| K1 | Move-only refactor into `_graph/` (and LPP into `_projections.py`) | manifold | — |
| K2 | `AbstractGraph` / `Graph` (static topology) / `GridGraph`; builders; Laplacian and incidence operators | manifold, INLA | K1, G1 |
| K3 | `laplacian_eigpairs` (dense / Kronecker / Lanczos / ARPACK), `n_components_graph` | manifold, INLA | K2 |
| K4 | Graph spectra (`graph_heat_spectrum`, `graph_matern_spectrum`), `matern_graph_kernel` | manifold, INLA | K3 (tests only) |
| K5 | Eigenmap extensions, `combine_potentials`, SEP, kernel LPP / SEP, sklearn adapters | manifold | K3, G2 |
| K6 | GMRF structure: `graph_null_space`, `structure_matrix`, `mesh_graph`; the graph-Matérn ↔ SPDE test | INLA | K2–K4 (G7 for the test) |
| K7 | `hadamard_transform` moves to gaussx; the public name stays | RandNLA | G11 |
| K8 | `select_landmarks` (uniform / leverage / rpcholesky / greedy), used by `NystromFeatures`, `Falkon`, `EigenPro` | RandNLA | G14 |
| K9 | Preconditioned `KRR` (`preconditioner="nystrom" \| "rpcholesky"`) | RandNLA | G13, G14 |
| K10 | `KernelPCA(eigen_solver="randomized")` | RandNLA | G12 |
| K11 | Docs: API pages, notebooks, `architecture.md` updates | all | K5, K6, K8–K10 |

---

## 1. Current state

Shipped in #43 (`_decomposition/`, all names top-level):

| Module | Symbols | Limits |
|---|---|---|
| `_neighbors.py` | `KNNGraph`, `nearest_neighbors(backend="exact" \| "pynndescent" \| "sklearn")` | k-NN only |
| `_graph.py` | `adjacency_matrix`, `graph_laplacian`, `diffusion_kernel`, `regularized_laplacian_kernel`, `random_walk_kernel`, `cosine_graph_kernel`, `commute_time_kernel` | Dense `N × N` only; heat or connectivity weights only |
| `_eigenmaps.py` | `laplacian_eigenmap`, `schrodinger_eigenmap`, `barrier_potential`, `label_potential`, `spatial_spectral_potential`, `LaplacianEigenmaps`, `SchrodingerEigenmaps`, `LocalityPreservingProjections` | Dense JAX path, or a private SciPy ARPACK path inside the estimators; one potential at a time; spatial potential finds image neighbours by brute-force k-NN on pixel coordinates |
| `_kpca.py` | `KernelPCA` (exact and `approx=`) | None relevant here |
| `sklearn/_decomposition.py` | Adapters for all four estimators | None relevant here |

Outside `_decomposition/`:

| Where | What | Limits |
|---|---|---|
| `NystromFeatures.fit` (`_spectral/_feature_maps.py`) | `selection="uniform"`, or `"leverage"`: approximate ridge leverage scores from a uniform pilot, mixed with uniform sampling (kernellib#34) | No RPCholesky; the selection logic is private to this class |
| `nystrom_operator(K_XZ, ...)` (`_operators/_low_rank.py:22`) | Column Nyström given landmark columns | No selection step, by design |
| `Falkon` (`_regression/_estimators.py:121`) | Centres chosen uniformly with `jax.random.choice`; its own Cholesky-pair preconditioner (`_falkon.py:136`) | Uniform centres only |
| `EigenPro` (`_eigenpro.py:96-105`, `:209`) | Uniform subsample, then dense `eigh` of the subsample Gram. It deliberately avoids Lanczos, because the tail eigenvalue sets every correction weight (comment at `:101-104`) | Uniform subsample only. The dense `eigh` is correct as is |
| `KRR` (`_regression/_krr.py:59`) | `solver: gx.AbstractSolverStrategy = gx.DenseSolver()`; `implicit=True` gives a matrix-free operator | No preconditioner wired in; a plain `gx.CGSolver()` on `K + λnI` stalls as `n` grows |
| `KernelPCA` (`_decomposition/_kpca.py:100,119`) | Dense `eigh` of the centred Gram, or `approx=` feature maps | No matrix-free exact-kernel path |
| `hadamard_transform` (`_operators/_fastfood.py:55`) | O(d log d) fast Walsh–Hadamard transform, used by FastFood, public | Needed by gaussx's SRHT sketch; gaussx cannot import kernellib |


---

## 2. Gaps

1. **Sparse graphs as a public type.** Rasters with 10⁵–10⁶ pixels, and the
   spatial models built on them, cannot use a dense `N × N` adjacency.
2. **Lattice graphs.** A 4-connected grid is known analytically, and its
   Laplacian is a Kronecker sum of 1-D path Laplacians. gaussx already has
   exact `eig`, `logdet` and `solve` for `KroneckerSum`.
3. **Radius graphs, cosine weights, and edge weights from any kernellib
   kernel.** The old code had the first two. The third makes adjacency
   construction a kernel operation, which was the original objection in
   `architecture.md` (open question 9).
4. **Scalable Laplacian eigenpairs** (Kronecker, Lanczos, ARPACK) behind one
   function, usable by pyrox-gp's inducing features.
5. **Graph Matérn** (Borovitskiy et al., 2021), both as a dense kernel and as
   a spectrum on eigenvalues.
6. **Eigenmap extensions:** sparse inputs, several potentials at once,
   linear Schrödinger eigenmap projections (SEP, from the MATLAB
   `SchroedingerEigenmapProjections.m`), and kernel LPP / kernel SEP (the
   2018 `kernel_graph_embedding` stub).
7. **GMRF structure.** gaussx's GMRF builders need null spaces per connected
   component, a scaled structure matrix, and graphs built from meshes.
8. **Randomized kernel methods.** Landmark and centre selection is uniform
   everywhere except `NystromFeatures`. `KRR` has no preconditioner.
   `KernelPCA` has no matrix-free exact path. `hadamard_transform` has to
   move so that gaussx's SRHT sketch can use it.

---

## 3. Package layout

A new layer-1 package `_graph/` takes the graph code out of
`_decomposition/`. It sits at layer 1 because kernel-weighted edges use
`_kernels/` (layer 0), and the decompositions (layer 2) use it.

```
src/kernellib/
├── functional/
│   └── _graph.py              # NEW  graph_heat_spectrum, graph_matern_spectrum            (K4)
├── _graph/                    # NEW package, layer 1                                       (K1)
│   ├── __init__.py
│   ├── _neighbors.py          # MOVED KNNGraph, nearest_neighbors; NEW radius_neighbors     (K2)
│   ├── _types.py              # NEW  GraphTopology, AbstractGraph, Graph, GridGraph         (K2)
│   ├── _construct.py          # NEW  graph_from_neighbors, knn_graph, radius_graph, grid_graph,
│   │                          #      graph_from_adjacency (K2), mesh_graph (K6); MOVED adjacency_matrix
│   ├── _weights.py            # NEW  heat, connectivity, cosine, local scaling, any kernel  (K2)
│   ├── _laplacian.py          # MOVED graph_laplacian; NEW laplacian_operator (K2),
│   │                          #      graph_null_space, structure_matrix (K6)
│   ├── _eigpairs.py           # NEW  laplacian_eigpairs, n_components_graph; ARPACK moves here (K3)
│   └── _kernels.py            # MOVED the five graph kernels; NEW matern_graph_kernel       (K4)
├── _spectral/
│   └── _landmarks.py          # NEW  select_landmarks                                       (K8)
├── _operators/
│   └── _fastfood.py           # hadamard_transform imported from gaussx                     (K7)
├── _regression/
│   └── _krr.py                # KRR.preconditioner                                          (K9)
└── _decomposition/
    ├── _kpca.py               # eigen_solver="randomized"                                   (K10)
    ├── _eigenmaps.py          # LE, SE, potentials; graph inputs; combine_potentials,
    │                          # spatial_spectral_graph                                      (K5)
    ├── _projections.py        # MOVED LocalityPreservingProjections (K1); NEW SchrodingerEigenmapProjections (K5)
    └── _kernel_projections.py # NEW  KernelLocalityPreservingProjections, KernelSchrodingerProjections (K5)
```

Every existing top-level name keeps its import path (`kernellib.X`) and its
behaviour. `_decomposition/__init__.py` stops re-exporting the moved names,
and `kernellib/__init__.py` imports them from `_graph` instead.

Graph names stay flat at the top level
([manifold open question 2](project-manifold.md#open-questions)). The
spectra go in `kernellib.functional` because they are pure array
functions, like the rest of that namespace.

---

## 4. API — graphs and embeddings (K2–K5)

### 4.1 Graph types (`_graph/_types.py`)

**The maths.** For symmetric weights $W$ and degrees $D = \operatorname{diag}(W\mathbf 1)$,
the Laplacian $L = D - W$ is the matrix of the **Dirichlet energy**:

$$
f^\top Lf = \tfrac12\sum_{i,j}W_{ij}(f_i-f_j)^2 = \sum_{e=(i,j)}w_e(f_i-f_j)^2 = \|Bf\|^2,
\qquad L = B^\top B,\quad B_{e,:} = \sqrt{w_e}\,(e_i-e_j)^\top .
$$

So $L$ is PSD, its null space is the constants on each connected
component, and storing each edge once gives the incidence matrix $B$
directly.

- **Normalised forms.** $L_{\text{sym}} = D^{-1/2}LD^{-1/2}$ (spectrum in
  $[0,2]$) and $L_{\text{rw}} = D^{-1}L$.
- **Grids.** On a grid with 4-neighbour edges,
  $L = L_H\otimes I_W + I_H\otimes L_W = L_H\oplus L_W$, where $L_n$ is
  the path Laplacian. That is why `GridGraph` never stores its edges.

Undirected, weighted, no self-loops. Each edge is stored **once**, with
`senders < receivers`. Matrix-vector products symmetrise on the fly
(`segment_sum` in both directions), so the representation cannot drift out
of symmetry and the incidence matrix falls out directly.

Following Equinox's abstract-or-final rule, there is one abstract base and
two final classes:

```python
class AbstractGraph(eqx.Module):
    n_nodes: AbstractVar[int]  # static

    @abc.abstractmethod
    def edges(
        self,
    ) -> tuple[Int[Array, " E"], Int[Array, " E"], Float[Array, " E"]]: ...

    # concrete, shared
    def degree(self) -> Float[Array, " N"]: ...
    def to_bcoo(self) -> jsparse.BCOO: ...  # symmetric (N, N)
    def to_dense(self) -> Float[Array, "N N"]: ...
    def adjacency_operator(self) -> gaussx.SparseOperator: ...
    def laplacian_operator(
        self, normalization: Normalization = "unnormalized"
    ) -> lx.AbstractLinearOperator: ...  # tags: symmetric, PSD
    def incidence_operator(
        self,
    ) -> gaussx.SparseOperator: ...  # (E, N), rows √w_e (e_i − e_j)ᵀ
    def dirichlet_energy(
        self, f: Float[Array, "N ..."]
    ) -> Float[Array, "..."]: ...  # Σ_e w_e (f_i − f_j)², i.e. fᵀLf
    def reweight(self, weights: Float[Array, " E"]) -> Graph: ...


class GraphTopology:  # host-side, hashable; the graph's gaussx SparsityPattern derives from it
    senders: np.ndarray  # int32, (E,), senders < receivers
    receivers: np.ndarray
    n_nodes: int


class Graph(AbstractGraph):  # final
    topology: GraphTopology = eqx.field(static=True)
    weights: Float[Array, " E"]  # the only traced field


class GridGraph(AbstractGraph):  # final
    shape: tuple[int, ...] = eqx.field(static=True)  # (H, W) or (H, W, D, ...)
    connectivity: Literal["face", "full"] = eqx.field(
        static=True
    )  # 4/6-neighbour or 8/26-neighbour
    periodic: bool = eqx.field(static=True)
    axis_weights: Float[Array, " d"]  # per-axis edge weight (anisotropic spacing)
    n_nodes: int = eqx.field(static=True)
```

Notes:

- `GridGraph.edges()` builds its edge list lazily, so a grid never stores
  `E` explicitly unless asked.
- `GridGraph.laplacian_operator("unnormalized")` with `connectivity="face"`
  returns a nested `gaussx.KroneckerSum` of `axis_weights[k] * L_k`, where
  `L_k` is the path Laplacian (`periodic=False`, free boundary) or the cycle
  Laplacian (`periodic=True`).
  - This is exact, including border degrees.
  - Every other combination falls back to `SparseOperator`. The symmetric
    normalisation is not a Kronecker sum, because degrees vary at the
    border, and neither are diagonal edges.
- **Node order is row-major (C order)**, matching
  `rearrange("h w c -> (h w) c", image)`. The old MATLAB and Python code
  used column-major order. Say so in the docstring.
- **Topology is static; weights are traced.** This follows gaussx's static
  `SparsityPattern` ([G1](roadmap-gaussx.md)).
  - `laplacian_operator()` and `incidence_operator()` produce
    `gaussx.SparseOperator`s whose pattern is derived once from the
    topology, so a symbolic sparse Cholesky is reused across every
    reweighting, θ-point and `vmap`.
  - Builders run eagerly (outside `jit`). That was already true for
    `radius_graph` and `graph_from_adjacency`, and now holds for all of
    them.
- Weights are the only array field, so everything except the ARPACK path
  is differentiable in the edge weights. This covers `to_bcoo`, the
  operators, `dirichlet_energy`, and dense and Lanczos eigenpairs.

`Normalization` remains `"unnormalized" | "symmetric" | "random_walk"`.
`laplacian_operator("random_walk")` returns an operator with no symmetric
tag.

**Example.**

```python
g = kl.grid_graph((1024, 1024))  # 10⁶ nodes, no edge list stored
L = g.laplacian_operator()  # gx.KroneckerSum(L_1024, L_1024)
roughness = g.dirichlet_energy(
    einx.rearrange("h w -> (h w)", image)
)  # Σ over edges of (f_i − f_j)²
B = kl.knn_graph(
    X, 10
).incidence_operator()  # (E, N): the precision factor of an ICAR on X
```

### 4.2 Builders (`_graph/_construct.py`, `_graph/_weights.py`)

**The maths.** Heat weights are the RBF kernel evaluated on edges,
$w_{ij} = k(x_i,x_j) = \exp(-\|x_i-x_j\|^2/2\sigma^2)$. So any kernellib
kernel can weight a graph: $w_{ij} = k(x_i,x_j)$, restricted to a sparse
topology (k-NN, radius or grid). That is a sparse, local approximation of
the full Gram matrix.

**Self-tuning** replaces $\sigma^2$ with $\sigma_i\sigma_j$, where
$\sigma_i$ is the distance to the $k$-th neighbour. The bandwidth then
adapts to local density (Zelnik-Manor & Perona, 2004), which matters for
data with clusters of very different spread.

```python
def graph_from_neighbors(
    knn: KNNGraph,
    *,
    weighting: Weighting = "heat",
    bandwidth: float | Float[Array, ""] | Literal["median", "local"] | None = None,
    symmetrize: Literal["max", "min", "mean"] = "max",
) -> Graph: ...


def knn_graph(
    X,
    n_neighbors,
    *,
    weighting="heat",
    bandwidth=None,
    symmetrize="max",
    backend="exact",
    random_state=None,
) -> Graph: ...  # nearest_neighbors + graph_from_neighbors


def radius_graph(
    X, radius, *, max_neighbors: int, weighting="heat", bandwidth=None
) -> Graph: ...


def grid_graph(
    shape, *, connectivity="face", periodic=False, spacing=None
) -> GridGraph: ...


def graph_from_adjacency(W: Float[Array, "N N"], *, atol: float = 0.0) -> Graph: ...


def edge_weights(
    graph: AbstractGraph, X: Float[Array, "N D"], kernel: AbstractKernel
) -> Graph: ...


Weighting = Literal["heat", "connectivity", "cosine"] | AbstractKernel
```

- **`weighting`**
  - `"heat"` gives `exp(-d²/2σ²)`, the current convention.
  - `"cosine"` gives the cosine similarity of the endpoints, clipped at 0.
  - A kernellib kernel evaluates `kernel(x_i, x_j)` on each edge, vmapped
    over the edges. `"heat"` is `RBF(lengthscale=σ)`, spelled out for
    convenience.
- **`bandwidth`**
  - `None` or `"median"` is the current median-neighbour-distance
    heuristic.
  - `"local"` is Zelnik-Manor & Perona's self-tuning weights,
    `w_ij = exp(-d_ij² / σ_i σ_j)`, with `σ_i` the distance to the k-th
    neighbour. It reuses `_heuristics`.
- **`radius_graph`** needs static shapes. It runs a k-NN search with
  `max_neighbors`, then drops edges longer than `radius`. The final pruning
  step is eager: the edge count depends on the data, so the function is not
  jittable. Say so in the docstring.
- **`edge_weights`** replaces the edge weights of an existing topology.
  That generalises Cahill's spatial-spectral potential: grid topology,
  spectral weights. It is also the extension point for any spatial model
  that wants covariate-dependent adjacency.
- **`adjacency_matrix(knn, ...)`** stays, as
  `graph_from_neighbors(knn, ...).to_dense()`. It must stay bit-identical:
  the current doctests pin its output.

`radius_neighbors(X, radius, *, max_neighbors, backend)` goes next to
`nearest_neighbors` in `_neighbors.py`. It returns a `KNNGraph` padded with
index `-1` and distance `inf`.

**Example.**

```python
X = einx.rearrange("h w b -> (h w) b", cube)  # hyperspectral cube with B bands
spectral = kl.knn_graph(
    X, 10, weighting=kl.RBF(lengthscale=0.5)
)  # pixels that look alike
spatial = kl.grid_graph(cube.shape[:2])  # pixels that are adjacent
cahill = kl.edge_weights(spatial, X, kl.RBF(lengthscale=0.5))  # adjacent AND alike
stations = kl.radius_graph(
    station_xy, radius=5.0, max_neighbors=16, weighting="connectivity"
)
```

### 4.3 Laplacian eigenpairs (`_graph/_eigpairs.py`)

**The maths.** **Path Laplacians have closed-form eigenpairs: the DCT-II.**
$\lambda_k = 2-2\cos(\pi k/n)$ and $u_k(j) = \cos\!\big(\pi k(j+\tfrac12)/n\big)$,
for $k = 0,\dots,n-1$.

**Kronecker sums add eigenvalues.** Since
$(A\oplus B)(u\otimes v) = (\lambda+\mu)(u\otimes v)$, a grid's eigenpairs
are all the sums $\lambda^H_i+\lambda^W_j$, with outer-product
eigenvectors. Picking the $n$ smallest means sorting $HW$ scalars; no
$HW\times HW$ matrix ever exists.

**General graphs.** Lanczos converges fastest at the extremes of the
spectrum, and the smallest eigenvalues of $L$ are the largest of $cI-L$
for $c \ge \lambda_{\max}$. Gershgorin gives $c = 2\max_i d_i$.

```python
def laplacian_eigpairs(
    graph: AbstractGraph | Float[Array, "N N"],
    n: int,
    *,
    normalization: Literal["unnormalized", "symmetric"] = "unnormalized",
    method: Literal["dense", "kronecker", "lanczos", "arpack"] | None = None,
    key: PRNGKeyArray | None = None,
) -> tuple[Float[Array, " n"], Float[Array, "N n"]]:
    """Smallest n eigenpairs, ascending, ℓ²-orthonormal, sign-fixed."""
```

| `method` | How | Traced / differentiable | When |
|---|---|---|---|
| `"dense"` | `jnp.linalg.eigh` of the dense Laplacian | yes | Small graphs; the default for `Graph` and arrays |
| `"kronecker"` | `gaussx.eig` of each 1-D factor, then all `∏ n_k` eigenvalue sums, `argsort`, the `n` smallest; eigenvectors as outer products of factor columns (`einx.multiply("h, w -> (h w)")`) | yes | The default for a `GridGraph` with `connectivity="face"` and `"unnormalized"`; an error for anything else |
| `"lanczos"` | `gaussx.eig(c·I − L, rank=n + oversample, key=key)`, with `c` a Gershgorin bound (`2·max(degree)`, or `2` for `"symmetric"`), then `λ = c − μ` | yes | Large sparse graphs in JAX |
| `"arpack"` | The current `_smallest_sparse`, moved here unchanged | no (SciPy, CPU) | Large graphs where Lanczos converges badly |

- The default is picked by **type**, not by size, in line with
  `architecture.md` open question 7 ("the caller knows their `N`").
- **Sign fix:** each eigenvector is flipped so that its largest-magnitude
  entry is positive. Tests and downstream inducing features are then
  deterministic.
- **Disconnected graphs** have one zero eigenvalue per component. Callers
  that drop "the trivial" solution must drop all of them. Add
  `n_components_graph(graph)` (a connected-components count by label
  propagation) so they can.
- geonnax's `graph_laplacian_eigpairs` is left alone. It is a dense basis
  primitive, and this function is the graph-aware entry point ([manifold open
  question 3](project-manifold.md#open-questions)).

**Example.**

```python
lam, U = kl.laplacian_eigpairs(
    kl.grid_graph((2000, 2000)), 256
)  # Kronecker path: seconds
lam, U = kl.laplacian_eigpairs(
    river_network, 128, method="lanczos", key=key
)  # 10⁵ nodes, in JAX
```

### 4.4 Spectral graph kernels

**The maths.** A spectral graph kernel is $K = U\,\Phi(\Lambda)\,U^\top$, for
$L = U\Lambda U^\top$ and a decreasing $\Phi\ge0$. Smooth eigenvectors
(small $\lambda$) get large prior variance.

- **Diffusion.** $\Phi(\lambda) = e^{-t\lambda}$ is the heat equation
  $\partial_t u = -Lu$, run for time $t$.
- **Matérn.** $\Phi(\lambda) = (2\nu/\ell^2+\lambda)^{-\nu}$ is the graph
  analogue of the SPDE $(\kappa^2-\Delta)^{\nu/2}f = \mathcal W$
  (Borovitskiy et al., 2021). Since
  $(2\nu/\ell^2+\lambda)^{-\nu}\propto(1+\ell^2\lambda/2\nu)^{-\nu}\to e^{-\ell^2\lambda/2}$,
  it tends to diffusion with $t = \ell^2/2$ as $\nu\to\infty$.
- **Truncation.** Keeping the first $M$ eigenpairs gives a rank-$M$ prior.
  Its average-variance normalisation needs only
  $\frac1N\sum_{k\le M}\Phi(\lambda_k)$.

In `functional/_graph.py` (arrays in, arrays out; what pyrox-gp needs for
inducing features):

```python
def graph_heat_spectrum(
    eigvals, *, lengthscale, variance=1.0, n_nodes=None
) -> Float[Array, " M"]:
    ...
    # Φ(λ) = exp(−ℓ² λ / 2)


def graph_matern_spectrum(
    eigvals, *, nu, lengthscale, variance=1.0, n_nodes=None
) -> Float[Array, " M"]:
    ...
    # Φ(λ) = (2ν/ℓ² + λ)^(−ν)
```

With `n_nodes` given, Φ is scaled so that the average marginal variance of
`U diag(Φ) Uᵀ` equals `variance`, following Borovitskiy et al. For
orthonormal `U` that average is `Σ_k Φ_k / n_nodes`, so the eigenvectors
are not needed.

In `_graph/_kernels.py`, the dense kernel:

```python
def matern_graph_kernel(
    W: Float[Array, "N N"] | AbstractGraph,
    *,
    nu,
    lengthscale,
    variance=1.0,
    normalization="symmetric",
) -> Float[Array, "N N"]: ...
```

This uses the existing `_spectral` helper. As `ν → ∞` it tends to the
diffusion kernel with `beta = ℓ²/2`, which is a test. The five existing
graph kernels also start accepting an `AbstractGraph`, converted with
`to_dense()`.

**Example.**

```python
# A Matérn(ν = 3/2) prior draw on a traffic-sensor graph, from 200 eigenpairs
lam, U = kl.laplacian_eigpairs(sensors, 200, normalization="symmetric")
phi = kl.functional.graph_matern_spectrum(
    lam, nu=1.5, lengthscale=2.0, n_nodes=sensors.n_nodes
)
f = einx.dot("n m, m -> n", U, jnp.sqrt(phi) * jax.random.normal(key, (200,)))
```

### 4.5 Eigenmaps (`_decomposition/_eigenmaps.py`)

**The maths.** **Laplacian eigenmaps** solve

$$
\min_Y\ \operatorname{tr}(Y^\top LY)\ \ \text{s.t.}\ \ Y^\top DY = I \quad\Longrightarrow\quad Ly = \lambda Dy .
$$

Neighbours stay close; the constraint rules out the collapsed solution
and weights nodes by degree.

**Schrödinger eigenmaps** add a potential,
$\operatorname{tr}\big(Y^\top(L+\alpha V)Y\big)$:

- a **diagonal** $V$ (a barrier) pins chosen nodes towards 0;
- a **non-diagonal**
  $V = \sum_{(i,j)\in\mathcal C}(e_i-e_j)(e_i-e_j)^\top$ adds
  $\alpha\sum_{\mathcal C}\|y_i-y_j\|^2$, pulling chosen pairs together
  (same label, or spatial neighbours that also look alike).

Cahill's scaling, $\alpha\cdot\operatorname{tr}L/\operatorname{tr}V$,
makes $\alpha$ unit-free.

- **Graph inputs.** `laplacian_eigenmap`, `schrodinger_eigenmap` and both
  estimators accept `W: Array | AbstractGraph`.
  - Under the degree constraint, `L y = λ D y` is equivalent to
    `L_sym u = λ u` with `y = D^{-1/2} u`. So the graph path goes through
    `laplacian_eigpairs(normalization="symmetric")`, and every method in
    §4.3 becomes available to the eigenmaps.
  - For Schrödinger, the operator is `D^{-1/2}(L + αV)D^{-1/2}`, built as
    a lineax composition. Only the `"dense"`, `"lanczos"` and `"arpack"`
    methods apply.
  - The estimators' `eigen_solver` field widens to the §4.3 methods.
    `"dense"` and `"arpack"` keep their current meaning.
- **Several potentials.** Add

  ```python
  def combine_potentials(
      W, terms: Sequence[tuple[V, alpha]], *, normalize: bool = True
  ) -> V:
      ...
      # Σ_k α_k · tr(L)/tr(V_k) · V_k
  ```

  Then pass the result with `alpha=1.0, normalize_potential=False`. This
  gives the old `'sspl'` mode with the α/β swap fixed, and leaves
  `schrodinger_eigenmap`'s signature unchanged.
- **Grid-aware spatial potential.** Add
  `spatial_spectral_graph(X, spatial_graph, *, bandwidth=None) -> Graph`,
  which is `edge_weights(spatial_graph, X, RBF(bandwidth))`.
  - The potential is then that graph's `laplacian_operator()`.
  - `spatial_spectral_potential(X, coordinates, ...)` keeps its signature
    and dense output.
  - For images, the docs point to
    `spatial_spectral_graph(X, grid_graph(image.shape[:2]))`, which
    replaces an `O(N²)` coordinate search with a known topology.

**Example.**

```python
V = kl.combine_potentials(
    spectral,
    [(cahill.laplacian_operator(), 1.0), (kl.label_potential(partial_labels), 0.1)],
)
se = kl.SchrodingerEigenmaps(
    n_components=20,
    n_neighbors=20,
    alpha=1.0,
    normalize_potential=False,
    eigen_solver="lanczos",
)
Y = se.fit(X, V).embedding  # (H·W, 20) features for a downstream classifier
```

### 4.6 Linear projections (`_decomposition/_projections.py`)

**The maths.** Restrict the embedding to a linear map of the centred inputs,
$y = \bar Xa$. The eigenmap objective then becomes

$$
\bar X^\top L\bar X\,a = \lambda\,\bar X^\top D\bar X\,a ,
$$

a $D\times D$ problem, not an $N\times N$ one. New points embed as
$(x-\mu)A$. SEP adds $\alpha\bar X^\top V\bar X$ to the left-hand side.

`LocalityPreservingProjections` moves here unchanged. Add:

```python
class SchrodingerEigenmapProjections(_GraphEmbedding):
    """Linear Schrödinger eigenmaps: X̄ᵀ(L + αV)X̄ a = λ X̄ᵀ D X̄ a."""

    # fields: as LocalityPreservingProjections, plus alpha, normalize_potential
    def fit(self, X, potential) -> SchrodingerEigenmapProjections: ...
    def transform(self, X) -> Float[Array, "M n"]: ...
```

- It uses the same degree-weighted centring and relative ridge as LPP.
- The trace normalisation of α is done in `N` space, before projection,
  matching Cahill and the MATLAB code.
- Both classes solve through `gaussx.eigh_generalized` (gaussx G2), which
  replaces LPP's inline Cholesky whitening.

**Example.**

```python
sep = kl.SchrodingerEigenmapProjections(n_components=10, alpha=5.0).fit(
    X_train, V_train
)
Z_test = sep.transform(X_test)  # out-of-sample, which eigenmaps cannot do
```

### 4.7 Kernel projections (`_decomposition/_kernel_projections.py`)

**The maths.** By the representer theorem, $y = K\alpha$, which gives
$KLK\alpha = \lambda\,KDK\alpha$. With a feature map $\Phi$
($K\approx\Phi\Phi^\top$, `approx=`), this is linear LPP on $\Phi$, at
$O(NM^2)$ cost.

```python
class KernelLocalityPreservingProjections(eqx.Module):
    kernel: AbstractKernel

    # graph fields as LPP; regularization; approx: feature map | None (as KernelPCA)
    def fit(self, X) -> ...: ...  # K L K a = λ (K D K + ε I) a
    def transform(self, X_new) -> ...: ...  # k(X_new, X_train) @ A


class KernelSchrodingerProjections(
    eqx.Module
): ...  # K (L + αV) K a = λ (K D K + ε I) a
```

- **Graph source.** The graph is built on `X` by default. An optional
  `graph=` argument to `fit` accepts a precomputed graph, for example a
  spatial one.
- **Regularisation.** `ε` is relative to `tr(KDK)/N`, as in LPP.
- **`approx=`.** With a feature map (Nyström, RFF, ...), the problem
  becomes linear LPP/SEP in feature space, `Φ` of shape `N × M`, and
  reuses the §4.6 code. This is the same pattern as `KernelPCA`.
- **Relation to linear LPP.** With a `Linear` kernel and no regularisation,
  the embedding spans the same space as uncentred LPP. That is a test.

**Example.**

```python
klpp = kl.KernelLocalityPreservingProjections(
    kl.RBF(1.0), n_components=5, approx=kl.NystromFeatures(500, key)
).fit(X)
Z = klpp.transform(X_new)
```

### 4.8 scikit-learn adapters

`kernellib.sklearn` gains:

- `SchrodingerEigenmapProjections`, which takes partial labels as `y`
  (`-1` means unlabelled) like the existing Schrödinger adapter, and builds
  `label_potential(y)`;
- `KernelLocalityPreservingProjections`.

Both pass `check_estimator` (integration tier).

---

---

## 5. API — GMRF structure (K6)

**The maths.** - **Besag structure.** The Besag / ICAR structure matrix is the graph
  Laplacian itself, $R = L$. Its null space is spanned by the component
  indicators $\mathbf 1_C/\sqrt{|C|}$.
- **BYM2 scaling.** It rescales $R^\ast = sR$ so that the geometric mean
  of the marginal variances is 1:
  $\operatorname{diag}\big((sR)^{+}\big) = \operatorname{diag}(R^{+})/s$,
  hence $s = \exp\big(\frac1N\sum_i\log[R^{+}]_{ii}\big)$.
- **Cotangent weights.** For P1 elements, the stiffness entries are
  $G_{ij} = -\tfrac12(\cot\alpha_{ij}+\cot\beta_{ij})$ on each edge, where
  $\alpha_{ij}$ and $\beta_{ij}$ are the angles opposite the edge. So
  $G$ is a graph Laplacian with cotangent weights, which is what
  `mesh_graph(weighting="cotangent")` builds.

```python
def graph_null_space(graph: AbstractGraph) -> Float[Array, "N c"]:
    ...
    # orthonormal indicator vectors of the connected components: the null space of L


def structure_matrix(
    graph: AbstractGraph, *, scaled: bool = False
) -> lx.AbstractLinearOperator:
    ...
    # L (or s·L with gaussx.generalized_variance_scale when scaled=True), with PSD tags


def mesh_graph(
    vertices,
    triangles,
    *,
    weighting: Literal["connectivity", "cotangent"] = "connectivity",
) -> Graph: ...
```

- **`graph_null_space`.** The null space an `IntrinsicGMRF` needs for its
  constraints, one normalised indicator per component. It builds on
  `n_components_graph`.
- **`structure_matrix`.** A convenience that pairs the Laplacian with
  gaussx's scaling. It is the only place kernellib calls a GMRF-flavoured
  gaussx function (allowed: kernellib depends on gaussx).
- **`mesh_graph`.** The edge graph of a triangulation. With
  `"cotangent"` weights its Laplacian equals the P1 FEM stiffness matrix
  `G`, which gives:
  - a test tying `gaussx.fem_matrices` to kernellib's graph code;
  - Besag models on mesh nodes;
  - Laplacian / Schrödinger eigenmaps on surfaces (K5).


**Example.**

```python
counties = kl.graph_from_adjacency(queen_contiguity)  # e.g. from a shapefile
R = kl.structure_matrix(counties, scaled=True)  # s·L, ready for BYM2
N0 = kl.graph_null_space(counties)  # one column per island group
mesh = kl.mesh_graph(
    vertices, triangles, weighting="cotangent"
)  # Laplacian == gx.fem_matrices(...)[1]
```

### 5.1 What gaussx's GMRF builders take from K2–K4

| Phase | Symbol | Use in gaussx / pyrox-lgm |
|---|---|---|
| K2 | `Graph.laplacian_operator()` | The Besag / ICAR structure matrix `R`, passed to `gaussx.besag_structure` and `gaussx.generalized_variance_scale` |
| K2 | `GridGraph.laplacian_operator()` (a `KroneckerSum`) | The same operator gaussx's `spde_precision_grid` builds internally; a test oracle for it |
| K2 | `Graph.incidence_operator()` | The precision factor `F` for the perturbation–optimisation sampler (a `precision_factors` entry) |
| K2 | `knn_graph`, `radius_graph`, `graph_from_adjacency` | Adjacency for areal models built from point data or shapefile contiguity |
| K3 | `n_components_graph` | Rank deficiency of an ICAR |


### 5.2 The graph-Matérn ↔ SPDE correspondence (test only)

kernellib's `graph_matern_spectrum(λ, nu, lengthscale)` (K4)
is `(2ν/ℓ² + λ)^−ν`. gaussx's grid SPDE has covariance spectrum
`τ⁻² (κ² + λ)^−α` on the same Laplacian eigenvalues (unit spacing, lumped
mass). They are the same family with

- `ν_graph = α`, and `2ν_graph/ℓ² = κ²`;
- equal after each is normalised to average marginal variance 1.

The difference in exponent conventions (graph Matérn has no dimension,
while SPDE uses `α = ν + d/2`) is exactly what pyrox-gp's
`LaplacianInducingFeatures` got wrong ([P2](roadmap-pyrox.md)).

K6 adds an integration test on a 32 × 32 grid:
`diag(kl.matern_graph_kernel(grid, nu=α, lengthscale=√(2α)/κ))` equals
`gaussx.spde_precision_grid(..., alpha=α).diag_inv()` after normalisation.
The docs state the mapping, so users can move between covariance form (GP
inducing features) and precision form (GMRF / INLA) on one graph.

---

## 6. API — randomized kernel methods (K7–K10)

### 6.1 K7: `hadamard_transform` moves to gaussx

- The implementation moves to `gaussx._sketching._hadamard`.
- `kernellib.hadamard_transform` stays public, as a re-export
  (`from gaussx import hadamard_transform`), and FastFood imports it from
  there.
- Acceptance: FastFood's tests pass unchanged; `__all__` is unchanged.


### 6.2 K8: `select_landmarks`

**The maths.** Nyström on landmarks $Z$ gives $\hat K = K_{XZ}K_{ZZ}^{+}K_{ZX}$. Its
trace error is a sum of conditional variances,

$$
\operatorname{tr}(K-\hat K) = \sum_i\Big(k(x_i,x_i) - k_{iZ}K_{ZZ}^{-1}k_{Zi}\Big),
$$

which is exactly the residual diagonal RPCholesky samples from. Ridge
leverage scores $\ell_i(\lambda) = [K(K+\lambda I)^{-1}]_{ii}$ sum to the
effective dimension $d_{\text{eff}}(\lambda)$, the number of landmarks
that matter.

A new public function in `_spectral/_landmarks.py` (layer 1, next to the
feature maps that use it):

```python
def select_landmarks(
    kernel: AbstractKernel,
    X: Float[Array, "N D"],
    n_landmarks: int,
    *,
    method: Literal["uniform", "leverage", "rpcholesky", "greedy"] = "uniform",
    key: PRNGKeyArray,
    regularization: float | None = None,  # "leverage" only
    uniform_mixing: float = 0.5,  # "leverage" only
) -> Int[Array, " M"]:
    """Indices into X of the chosen landmarks."""
```

- **`"uniform"`** and **`"leverage"`** are the existing `NystromFeatures`
  code, moved here unchanged.
- **`"rpcholesky"`** is `gaussx.rp_cholesky` with
  `diagonal = kernel.diag(X)` and
  `column(k) = kernel.pairwise(X, X[k][None])[:, 0]`.
  - Cost: `O(N · M)` kernel evaluations and `O(N · M²)` arithmetic, and
    the `N × N` Gram is never formed.
  - It needs no regularisation parameter and no pilot sample. For
    Nyström, its approximation error is within a small factor of the
    best rank-`M` approximation in expectation (Chen, Epperly, Tropp &
    Webber, 2023). It becomes the recommended default in the docs, while
    the code default stays `"uniform"` for compatibility.
- **`"greedy"`** is `gaussx.rp_cholesky(pivoting="greedy")`: the
  conditional-variance rule, which Burt, Rasmussen & van der Wilk (2020)
  recommend for inducing points. It is deterministic.

Consumers gain a `selection` field with the same four methods:

- `NystromFeatures(selection=...)`, which now delegates to
  `select_landmarks`, with identical output for `"uniform"` and
  `"leverage"`;
- `Falkon(centers=...)`, default `"uniform"`;
- `EigenPro(subsample=...)`, default `"uniform"`. The dense `eigh` of the
  subsample stays, for the reason in the comment at `_eigenpro.py:101`.

Tests:

- **Pivots.** `"rpcholesky"` returns distinct indices.
- **Nyström error.** On clustered data the trace error `tr(K − K̂)` is
  lower than `"uniform"`'s in expectation (slow tier, many keys, bounded
  by the empirical distribution).
- **Determinism.** `"greedy"` gives the same indices for any key.
- **Delegation.** `NystromFeatures` output is bit-identical before and
  after for `"uniform"` and `"leverage"`.


**Example.**

```python
idx = kl.select_landmarks(
    kl.Matern(nu=1.5, lengthscale=0.3), X, 2000, method="rpcholesky", key=key
)
falkon = kl.Falkon(kernel, n_inducing=2000, centers="rpcholesky").fit(X, y, key=key)
```

### 6.3 K9: preconditioned KRR

**The maths.** KRR solves $(K+\lambda nI)\alpha = y$. CG needs
$O(\sqrt\kappa\,\log(1/\text{tol}))$ iterations, with
$\kappa = (\lambda_1+\lambda n)/\lambda n$, which explodes for small
$\lambda$. A Nyström or RPCholesky preconditioner of rank
$\ell\approx d_{\text{eff}}(\lambda n) = \sum_i\lambda_i/(\lambda_i+\lambda n)$
makes $\kappa = O(1)$ (gaussx G13).

```python
class KRR(AbstractEstimator):
    ...
    preconditioner: Literal["none", "nystrom", "rpcholesky"] = eqx.field(
        default="none", static=True
    )
    preconditioner_rank: int = eqx.field(default=200, static=True)
```

- **Routing.** When `preconditioner != "none"`, `KRR` solves
  `(K + λnI)α = y` with `gx.PreconditionedCGSolver`, preconditioned by:
  - `"nystrom"`:
    `gx.NystromPreconditioner.from_operator(K_op, rank, shift=λn, key=...)`,
    built on the implicit kernel operator (`rank` matvecs);
  - `"rpcholesky"`:
    `gx.PartialCholeskyPreconditioner.from_operator(K_op, rank, shift=λn, pivoting="random", key=...)`.
    This is "robust randomized preconditioning for KRR" (Díaz, Epperly,
    Frangella, Tropp & Webber, 2023). It uses `O(N·r)` kernel evaluations
    rather than `r` full matvecs, so it is the better choice when kernel
    evaluations are expensive.
- **Interface.** The preconditioner is always built from `K` with
  `shift=λn` passed separately, which is the #345 rule. `solver=` still
  works and takes precedence if the caller passes a strategy explicitly.
- **Why a convenience field and not only documentation.** Building a
  preconditioner needs `K` and `λn` separately, and only `KRR` knows both.
  A user passing a strategy cannot do that without re-deriving `λn`.

Tests:

- The CG iteration count with `"nystrom"` at `r = 200` on a Matérn-3/2
  problem with `n = 20 000` is below 10 % of the unpreconditioned count
  (slow tier).
- Predictions match `DenseSolver` at `n = 2000`.
- Update the notebook from kernellib#83 (KRR vs Falkon vs EigenPro) to add
  a preconditioned-KRR line.


**Example.**

```python
krr = kl.KRR(
    kl.Matern(nu=2.5, lengthscale=0.2),
    regularization=1e-6,
    implicit=True,
    preconditioner="rpcholesky",
    preconditioner_rank=1000,
)
krr = krr.fit(X, y, key=key)  # n = 10⁵, never materialises K
```

### 6.4 K10: randomized kernel PCA

**The maths.** Kernel PCA diagonalises the centred Gram matrix $HKH$, with
$H = I - \frac1n\mathbf 1\mathbf 1^\top$. Randomized Nyström (G13) gets
its top eigenpairs from $\ell$ matvecs with the implicit operator:
$O(n^2\ell)$ kernel work and $O(n\ell)$ memory, never $O(n^2)$ memory.

```python
class KernelPCA(eqx.Module):
    ...
    eigen_solver: Literal["dense", "randomized"] = eqx.field(
        default="dense", static=True
    )
    n_power_iter: int = eqx.field(default=2, static=True)
    oversample: int = eqx.field(default=10, static=True)
```

- **Algorithm.** `"randomized"` runs `gx.randomized_nystrom` (the centred
  Gram is PSD) on the centred implicit kernel operator
  (`centering_operator` composed with `to_operator(kernel, X, implicit=True)`).
  It never materialises the `N × N` Gram.
- **Out-of-sample `transform`** keeps its current formula; only `alphas`
  and `eigenvalues` change source.
- **Relation to `approx=`.** The feature-map path approximates the
  *kernel*; this path approximates the *eigendecomposition* of the exact
  kernel. The docs describe both and when to use each.
- **Tests.** Agreement with `"dense"` on the leading components (principal
  angles) for a fast-decaying spectrum, and improvement with
  `n_power_iter` on a slow one.


**Example.**

```python
kpca = kl.KernelPCA(kl.RBF(1.0), n_components=20, eigen_solver="randomized").fit(
    X
)  # n = 5·10⁴
```

### 6.5 Deliberately not changed

- **Graph eigenmaps** need the *smallest* Laplacian eigenpairs.
  Randomized range finders target the top, so eigenmaps stay on dense,
  Lanczos, ARPACK and LOBPCG (kernellib#91).
- **Falkon's own preconditioner.** Its Cholesky pair on the centres is the
  published algorithm. Only the centre selection changes (K8).
- **`nystrom_operator`.** It stays a column Nyström given landmarks, fed
  by `select_landmarks`. The sketch-based `gx.randomized_nystrom` needs no
  kernellib wrapper: it takes `to_operator(kernel, X, implicit=True)`
  directly, and the docs show it.
- **No GMRF maths.** GMRF distributions, precision builders and FEM
  assembly are gaussx's (G6, G7). Priors with hyperpriors are pyrox-lgm's
  (P7). Mesh generation is external.

---

## 7. Phases and tests

Each phase is one PR, released by release-please. pyrox-gp and manipy pin
kernellib by git tag, so any phase they consume needs a release.

### K1: move-only refactor

- Create `_graph/` and move `_neighbors.py`, the graph half of
  `_graph.py`, and the ARPACK helper.
- Move LPP to `_projections.py`.
- **Acceptance:** `kernellib.__all__` is unchanged, every doctest and test
  passes untouched, and `ruff`, `ty` and `test_public_api.py` are green.
  No gaussx dependency.

### K2: graph types and builders (needs gaussx G1 released)

- `AbstractGraph`, `Graph`, `GridGraph`, all builders from §4.2,
  `radius_neighbors`, and `laplacian_operator` / `incidence_operator`.
- `adjacency_matrix` is reimplemented on top of `graph_from_neighbors`.
- Tests:
  - `Graph.to_dense()` agrees with `adjacency_matrix`;
  - the Laplacian operator's matvec agrees with the dense `graph_laplacian`
    for all three normalisations;
  - `GridGraph`'s `KroneckerSum` Laplacian agrees with the dense one for
    both `periodic` settings and anisotropic `axis_weights`;
  - `incidence_operatorᵀ @ incidence_operator == laplacian`;
  - `dirichlet_energy(f) == f @ L @ f`;
  - kernel weighting with `RBF` equals `"heat"`;
  - gradients of the Dirichlet energy with respect to the edge weights
    match finite differences.

### K3: Laplacian eigenpairs (needs K2)

- `laplacian_eigpairs` with all four methods, and `n_components_graph`.
- Tests:
  - the Kronecker method agrees with the dense method on 2-D and 3-D grids,
    including degenerate eigenvalues (compare subspaces, not vectors);
  - Lanczos agrees with dense to a loose tolerance on a random geometric
    graph (slow tier);
  - sign convention;
  - on a disconnected graph, the number of zero eigenvalues equals the
    number of components.

### K4: graph spectra and graph Matérn (needs K3 only for tests)

- `graph_heat_spectrum`, `graph_matern_spectrum`, `matern_graph_kernel`,
  and `AbstractGraph` inputs for the existing graph kernels.
- Tests:
  - PSD;
  - the average-variance normalisation;
  - `matern_graph_kernel` tends to `diffusion_kernel` as `ν → ∞`;
  - the dense kernel equals `U diag(Φ) Uᵀ` built from the full
    eigendecomposition.

### K5: eigenmap extensions (needs K3 and gaussx G2 released)

- The §4.5–4.8 changes.
- Tests:
  - dense and graph inputs give the same embedding (up to sign and
    rotation within degenerate eigenspaces);
  - `combine_potentials` with one term equals the current α
    normalisation;
  - SEP with `alpha=0` equals LPP exactly;
  - kernel LPP with a `Linear` kernel spans the same subspace as uncentred
    LPP;
  - kernel LPP with `approx=` converges towards the exact version as the
    rank grows (slow tier);
  - scikit-learn `check_estimator` (integration tier).

### K6: GMRF structure (needs K2–K4)

- `graph_null_space` is orthonormal and spans the Laplacian's kernel on
  graphs with 1, 2 and isolated-node components.
- `mesh_graph(weighting="cotangent")`'s Laplacian equals
  `gaussx.fem_matrices`' `G` on a reference mesh (integration tier).
- The graph-Matérn ↔ SPDE test of §5.2 (integration tier, once G7 has
  shipped).

### K7–K10

The tests are given with each API section (§6.1–6.4).

### K11: documentation

- `docs/api/graph.md`: a new mkdocstrings page for the graph names.
  `docs/api/spectral.md` gains `select_landmarks`. `docs/api/regression.md`
  covers `KRR.preconditioner`.
- A new notebook, `docs/notebooks/graphs_and_spatial.ipynb`:
  - a grid graph on a synthetic raster, with Kronecker eigenpairs at
    `N = 10⁶`;
  - graph Matérn prior samples on a k-NN graph;
  - Schrödinger eigenmaps with a grid-based spatial-spectral potential on a
    small synthetic hyperspectral cube.
- The kernel-approximations notebook gains a section comparing landmark
  selections (uniform, leverage, RPCholesky, greedy) by Nyström error
  against rank. The KRR vs Falkon vs EigenPro notebook (kernellib#83)
  gains a preconditioned-KRR line.
- Update `graph_embeddings.ipynb` only if its code paths change (they
  should not).
- Update `design_docs/kernellib/architecture.md`:
  - add `_graph/`, `_projections.py`, `_kernel_projections.py` and
    `_spectral/_landmarks.py` to the package tree;
  - add graph and GMRF-structure rows to the ownership map, plus a note
    that randomized primitives are gaussx's;
  - mark open question 9 resolved (it still says "Deferred", which has been
    stale since the 2026-09-26 decision);
  - add a decisions-log entry pointing to this roadmap.

---

## 8. Import boundaries

- Nothing here imports NumPyro or scikit-learn from the core.
  `test_imports.py` keeps passing unchanged.
- `jax.experimental.sparse` is used only through `gaussx.SparseOperator`
  and `Graph.to_bcoo()`, so a change in that experimental API touches two
  places.
- SciPy stays confined to the `"arpack"` method, as today.

- K6's `structure_matrix` is the only call into gaussx's GMRF layer
  (allowed: kernellib depends on gaussx; the reverse never happens).

## 9. Risks

| Risk | Mitigation |
|---|---|
| Lanczos converges slowly at the small end of a clustered Laplacian spectrum | Oversample, use a loose tolerance in tests, document it, and keep `"arpack"` as the robust fallback |
| BCOO matvec is slow on some backends | Benchmark against a `segment_sum` matvec in K2 and pick per backend inside `SparseOperator` (gaussx), not here |
| Degenerate eigenvalues make eigenvector tests flaky | Compare subspaces (principal angles), never individual vectors |
| Row-major vs column-major confusion when porting old notebooks | Stated in `GridGraph`'s docstring, and in the manipy HSI helpers |
