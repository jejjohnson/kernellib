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
