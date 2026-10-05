# Decomposition

Kernel PCA, neighbourhood graphs and graph kernels, and graph embeddings:
Laplacian eigenmaps, Schrödinger eigenmaps and locality preserving
projections.

## Kernel PCA

PCA in a kernel's feature space. The exact path eigendecomposes the centred
Gram matrix ($O(N^3)$); pass `approx=` a feature map for ordinary PCA of
$\phi(X)$ in $O(N R^2)$.

```python
import kernellib as kl

kpca = kl.KernelPCA(kl.RBF(lengthscale=0.5), n_components=2).fit(X)
Z = kpca.transform(X_new)
kpca = kl.KernelPCA(k, n_components=10, approx=kl.NystromFeatures(500, key)).fit(X)
```

**Supervised and fair kernel PCA.** Pass targets to `fit` and a
``target_weight`` $\gamma$: the components maximise the variance plus
$\gamma$ times their linear HSIC with the targets. $\gamma > 0$ keeps
target-relevant structure (supervised KPCA, Barshan et al., 2011);
$\gamma < 0$ removes it (fair KPCA). It is still one eigenproblem, and
$\gamma = 0$ is plain kernel PCA.

```python
supervised = kl.KernelPCA(k, n_components=5, target_weight=100.0).fit(X, target=y)
fair = kl.KernelPCA(
    k, n_components=5, target_kernel=kl.Linear(), target_weight=-100.0
).fit(X, target=S)  # embedding (nearly) independent of S
```

**Pre-images.** With ``fit_inverse_transform=True``, a `KRR` from the
embedding back to the inputs is fitted (Bakir, Weston & Schölkopf, 2004), so
components map back to input space, e.g. for denoising:

```python
kpca = kl.KernelPCA(k, n_components=40, fit_inverse_transform=True).fit(X_clean)
X_denoised = kpca.inverse_transform(kpca.transform(X_noisy))
```

Three ways to compute kernel PCA, by what they approximate:

| Path | Kernel | Eigendecomposition | Cost | When |
|---|---|---|---|---|
| default (`eigen_solver="dense"`) | exact | exact `eigh` | $O(N^3)$ time, $O(N^2)$ memory | up to a few thousand points |
| `eigen_solver="randomized"` | exact | `gaussx.randomized_eigh` on the matrix-free $HKH$ | $O(N^2(k+p)q)$ kernel work, $O(N(k+p))$ memory | tens of thousands of points, few components |
| `approx=` a feature map | approximated by $\phi$ | exact, of $\phi(X)$ | $O(N R^2)$ | when even $N^2$ kernel evaluations are too many |

The randomized error decays like $(\lambda_{k+1}/\lambda_k)^{2q}$: raise
`n_power_iter` ($q$) for rough kernels or short lengthscales.

```python
kpca = kl.KernelPCA(
    kl.Matern(nu=1.5, lengthscale=0.3), n_components=20, eigen_solver="randomized"
).fit(X, key=key)  # n = 5e4: the N x N Gram is never formed
```

::: kernellib.KernelPCA

## Neighbourhood graphs

`nearest_neighbors` finds each point's k nearest other points. The default is
an exact, batched brute-force search in JAX. For large ``N`` choose
``backend="pynndescent"`` (approximate NN-descent, extra `kernellib[neighbors]`)
or ``backend="sklearn"`` (extra `kernellib[sklearn]`); both are imported only
when chosen. `adjacency_matrix` turns the graph into symmetric heat-kernel or
0/1 weights, and `graph_laplacian` gives $L = D - W$ or a normalised form.

```python
graph = kl.nearest_neighbors(X, 10, backend="pynndescent", random_state=0)
W = kl.adjacency_matrix(graph, weighting="heat")  # sigma: median distance
L = kl.graph_laplacian(W, "symmetric")
```

::: kernellib.nearest_neighbors

::: kernellib.KNNGraph

::: kernellib.adjacency_matrix

::: kernellib.graph_laplacian

## Sparse graphs

`Graph` stores a weighted, undirected graph as a static edge list
(`GraphTopology`, each edge once) and one traced weight per edge. Its
Laplacian, adjacency and incidence matrices are `gaussx.SparseOperator`s whose
pattern is built once from the topology, so a sparse Cholesky's symbolic
analysis is reused across reweightings, `jit` and `vmap`. `GridGraph` is a
lattice in row-major node order that never stores its edges; with
`connectivity="face"` its Laplacian is a nested `gaussx.KroneckerSum` of 1-D
path (or, per periodic axis, cycle) Laplacians.

```python
g = kl.GridGraph((1024, 1024))  # 10^6 nodes, no edge list stored
L = g.laplacian_operator()  # gx.KroneckerSum(L_1024, L_1024)
roughness = g.dirichlet_energy(einx.id("h w -> (h w)", image))
B = graph.incidence_operator()  # (E, N), B^T B = L
```

### Builders

Builders run eagerly (outside `jit`) and return a `Graph` or `GridGraph`
whose weights are differentiable. `weighting` is `"heat"`
($\exp(-d^2/2\sigma^2)$, the default), `"connectivity"`, `"cosine"` or any
kernellib kernel; `bandwidth="local"` gives self-tuning heat weights.
`graph_from_edges` is the seam for edge lists from city2graph, libpysal,
NetworkX or OpenStreetMap: pass lengths as `distances`, not as `weights`.

```python
spectral = kl.knn_graph(X, 10, weighting=kl.RBF(lengthscale=0.5))
spatial = kl.grid_graph(cube.shape[:2])  # pixels that are adjacent
cahill = kl.edge_weights(spatial, X, kl.RBF(lengthscale=0.5))  # adjacent AND alike
roads = kl.graph_from_edges(src, dst, n_nodes, distances=length_m, weighting="heat")
connected = kl.knn_graph(X, 5, ensure_connected=True)  # Borůvka bridges
```

::: kernellib.knn_graph

::: kernellib.graph_from_neighbors

::: kernellib.radius_graph

::: kernellib.radius_neighbors

::: kernellib.grid_graph

::: kernellib.graph_from_adjacency

::: kernellib.graph_from_edges

::: kernellib.edge_weights

### Proximity graphs

For spatial points in 2-D or 3-D, the Delaunay, Gabriel and relative
neighbourhood graphs need no `k`: they are nested,
EMST ⊆ RNG ⊆ GG ⊆ DT, so each is connected, with $O(N)$ edges. The
triangulation is `scipy.spatial.Delaunay` (imported lazily), and `weighting`
applies to the edge lengths as in `graph_from_edges`. For higher dimensions
use `knn_graph`.

```python
stations = kl.gabriel_graph(station_xy, weighting="connectivity")  # no k to tune
prior = kl.structure_matrix(stations)  # an ICAR on irregular monitoring sites
```

::: kernellib.delaunay_graph

::: kernellib.gabriel_graph

::: kernellib.relative_neighborhood_graph

### Laplacian eigenpairs

`laplacian_eigpairs(graph, n)` returns the `n` smallest eigenpairs, ascending
and sign-fixed (each eigenvector's largest-magnitude entry is positive). The
method follows the graph's type: closed-form Kronecker eigenpairs for a
face-connected `GridGraph` (a `2000 x 2000` grid in seconds), dense `eigh`
otherwise. Pass `method="lanczos"` (JAX, differentiable; needs a `key`) or
`method="arpack"` (SciPy, CPU) for large sparse graphs. A disconnected graph
has one zero eigenvalue per component: `n_components_graph` counts them.

```python
lam, U = kl.laplacian_eigpairs(kl.grid_graph((2000, 2000)), 256)  # Kronecker
lam, U = kl.laplacian_eigpairs(river_network, 128, method="lanczos", key=key)
```

::: kernellib.laplacian_eigpairs

::: kernellib.n_components_graph

### GMRF structure

The graph side of intrinsic GMRFs (Besag / ICAR, BYM2) in gaussx and
pyrox-lgm. `structure_matrix(graph)` is the Besag structure $R = L$ (a
`gaussx.SparseOperator`, or a `gaussx.KroneckerSum` on a grid), and with
`scaled=True` it is BYM2-scaled per connected component.
`graph_null_space` gives the sum-to-zero constraints, one per component.
`mesh_graph(vertices, triangles, weighting="cotangent")` has the P1 FEM
stiffness matrix as its Laplacian.

```python
counties = kl.graph_from_adjacency(queen_contiguity)
R = kl.structure_matrix(counties, scaled=True)  # s * L, ready for BYM2
N0 = kl.graph_null_space(counties)  # one column per island group
mesh = kl.mesh_graph(vertices, triangles, weighting="cotangent")
```

**Covariance form and precision form on one graph.** The graph Matérn kernel
$(2\nu/\ell^2 + \lambda)^{-\nu}$ and gaussx's SPDE precision
$\tau^2(\kappa^2 + \lambda)^{\alpha}$ are the same prior, with $\nu = \alpha$ and
`lengthscale = sqrt(2 * alpha) / kappa`, once both are normalised to average
variance 1. Use `normalization="unnormalized"` on the graph side.

::: kernellib.structure_matrix

::: kernellib.graph_null_space

::: kernellib.mesh_graph

### Graph types

::: kernellib.AbstractGraph

::: kernellib.Graph

::: kernellib.GraphTopology

::: kernellib.GridGraph

## Graph kernels

Spectral functions of the Laplacian, $K = U f(\Lambda) U^\top$: positive
semidefinite similarities between the nodes of a graph (Smola & Kondor, 2003).

| Kernel | $f(\lambda)$ |
|---|---|
| `diffusion_kernel` | $e^{-\beta\lambda}$ |
| `regularized_laplacian_kernel` | $(1 + \sigma^2\lambda)^{-1}$ |
| `random_walk_kernel` | $(a - \lambda)^p$ |
| `cosine_graph_kernel` | $\cos(\pi\lambda/4)$ |
| `commute_time_kernel` | $\lambda^{+}$ (pseudo-inverse) |
| `matern_graph_kernel` | $(2\nu/\ell^2 + \lambda)^{-\nu}$, normalised to average variance $\sigma^2$ |

Every graph kernel takes an adjacency matrix or any graph (`Graph`,
`GridGraph`). They are dense: an `N x N` eigendecomposition. For a large graph,
take the smallest eigenpairs from `laplacian_eigpairs` and weight them with
`kernellib.functional.graph_matern_spectrum` or `graph_heat_spectrum`, which
is a truncated, rank-`M` prior:

```python
lam, U = kl.laplacian_eigpairs(sensors, 200, normalization="symmetric")
phi = kl.functional.graph_matern_spectrum(
    lam, nu=1.5, lengthscale=2.0, n_nodes=sensors.n_nodes
)
f = einx.dot("n m, m -> n", U, jnp.sqrt(phi) * jax.random.normal(key, (200,)))
```

The graph Matérn kernel's $\nu$ is the SPDE exponent ($\kappa^2 = 2\nu/\ell^2$):
no dimension enters it. As $\nu \to \infty$ it tends to the diffusion kernel
with $\beta = \ell^2/2$.

::: kernellib.matern_graph_kernel

::: kernellib.diffusion_kernel

::: kernellib.regularized_laplacian_kernel

::: kernellib.random_walk_kernel

::: kernellib.cosine_graph_kernel

::: kernellib.commute_time_kernel

## Graph embeddings

All three solve a generalised eigenproblem on the neighbourhood graph. The
dense path is JAX (differentiable in the edge weights); ``eigen_solver="arpack"``
keeps the graph sparse for large ``N``.

| Embedding | Problem | New points |
|---|---|---|
| `LaplacianEigenmaps` | $L y = \lambda D y$ | no |
| `SchrodingerEigenmaps` | $(L + \alpha V) y = \lambda D y$ | no |
| `LocalityPreservingProjections` | $X^\top L X a = \lambda X^\top D X a$ | `transform` |

A Schrödinger potential $V$ steers the embedding: `barrier_potential` pins
points near the origin, `label_potential` pulls points sharing a label
together (semi-supervised), and `spatial_spectral_potential` pulls spatially
adjacent, spectrally similar pixels together (hyperspectral images).

```python
le = kl.LaplacianEigenmaps(n_components=2, n_neighbors=10).fit(X)
le.embedding

labels = ...  # -1 for unlabelled
se = kl.SchrodingerEigenmaps(n_components=2, alpha=10.0)
se = se.fit(X, kl.label_potential(labels))
```

::: kernellib.LaplacianEigenmaps

::: kernellib.SchrodingerEigenmaps

::: kernellib.LocalityPreservingProjections

::: kernellib.laplacian_eigenmap

::: kernellib.schrodinger_eigenmap

::: kernellib.barrier_potential

::: kernellib.label_potential

::: kernellib.spatial_spectral_potential
