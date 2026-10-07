# Decomposition

Kernel PCA and graph embeddings: Laplacian eigenmaps, Schrödinger
eigenmaps, locality preserving and Schrödinger eigenmap projections, and
their kernel versions. The graphs they are built on (neighbour search,
builders, Laplacians, eigenpairs, graph kernels) are on the
[Graphs](graph.md) page.

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

## Graph embeddings

All of them solve a generalised eigenproblem on a neighbourhood graph. The
dense path is JAX (differentiable in the edge weights); ``eigen_solver`` (or
``method=`` in the functions) also takes the sparse methods of
`laplacian_eigpairs`: ``"lanczos"`` (JAX), ``"arpack"`` (SciPy, CPU) and, for
Laplacian eigenmaps of a lattice under the identity constraint,
``"kronecker"``; and ``"lobpcg"`` (sparse JAX, see below).

| Embedding | Problem | New points |
|---|---|---|
| `LaplacianEigenmaps` | $L y = \lambda D y$ | no |
| `SchrodingerEigenmaps` | $(L + \alpha V) y = \lambda D y$ | no |
| `LocalityPreservingProjections` | $\bar X^\top L \bar X a = \lambda \bar X^\top D \bar X a$ | `transform` |
| `SchrodingerEigenmapProjections` | $\bar X^\top (L + \alpha V) \bar X a = \lambda \bar X^\top D \bar X a$ | `transform` |
| `KernelLocalityPreservingProjections` | $F^\top L F \beta + \epsilon \beta = \lambda F^\top D F \beta$, $\bar K = F F^\top$ | `transform` |
| `KernelSchrodingerProjections` | $F^\top (L + \alpha V) F \beta + \epsilon \beta = \lambda F^\top D F \beta$ | `transform` |

A Schrödinger potential $V$ steers the embedding: `barrier_potential` pins
points near the origin, `label_potential` pulls points sharing a label
together (semi-supervised), and `spatial_spectral_potential` pulls spatially
adjacent, spectrally similar pixels together (hyperspectral images). A
potential is a diagonal, a dense matrix or a sparse lineax operator.

```python
le = kl.LaplacianEigenmaps(n_components=2, n_neighbors=10).fit(X)
le.embedding

labels = ...  # -1 for unlabelled
se = kl.SchrodingerEigenmaps(n_components=2, alpha=10.0)
se = se.fit(X, kl.label_potential(labels))
```

**Sparse graphs and images.** The functions take an `AbstractGraph` as well
as an adjacency matrix, and the estimators' `fit` takes a precomputed
`graph=`. For an image, build the spatial-spectral potential on the pixel
lattice, `spatial_spectral_graph(X, grid_graph(image.shape[:2]))`, instead of
the `O(N^2)` coordinate search of `spatial_spectral_potential`.
`combine_potentials` weights several potentials into one, each by its own
$\alpha_k \operatorname{tr}L / \operatorname{tr}V_k$:

```python
spectral = kl.knn_graph(X, 20)
cahill = kl.spatial_spectral_graph(X, kl.grid_graph(cube.shape[:2]))
V = kl.combine_potentials(
    spectral,
    [(cahill.laplacian_operator(), 1.0), (kl.label_potential(partial_labels), 0.1)],
)
se = kl.SchrodingerEigenmaps(
    n_components=20, alpha=1.0, normalize_potential=False, eigen_solver="lanczos"
)
Y = se.fit(X, V, graph=spectral).embedding  # (H·W, 20), sparse throughout
```

`"lanczos"` stays in JAX, but a strong potential widens the spectrum of
$A = D^{-1/2}(L + \alpha V)D^{-1/2}$ and crowds the wanted eigenvalues near
0, where a fixed Krylov space may not converge. So each returned pair is
checked: its residual $\|A u - \lambda u\|$ must be at most
$\sqrt{\varepsilon}\,c$, with $\varepsilon$ the machine epsilon and
$c \approx \lambda_{\max}(A)$ the largest Ritz value. A failing run is
repeated with twice the Krylov oversample, from 200 up to 1600, and then
raises a `RuntimeError` rather than return a wrong embedding. Each
restart costs time and memory ($N \times$ the Krylov dimension), so for a
large cube with a large $\alpha$, `eigen_solver="arpack"` (SciPy, on the
CPU) is the robust choice.

**LOBPCG.** ``eigen_solver="lobpcg"`` (``method="lobpcg"``) is the
sparse solver that stays in JAX: the graph is a
``jax.experimental.sparse.BCOO``, and
``jax.experimental.sparse.linalg.lobpcg_standard`` runs on it with block
matvecs and small ``eigh`` calls, on the device and under ``jit``, with no
SciPy and no $N \times N$ matrix.

| `eigen_solver` | Graph | Runs | Memory | `jit` / gradients | dtype |
|---|---|---|---|---|---|
| `"dense"` | dense $N \times N$ | JAX `eigh` | $O(N^2)$ | yes / yes | any |
| `"lanczos"` | sparse operator | JAX, `gaussx.eig` | $O(N \cdot$ Krylov dim$)$ | yes / no | any |
| `"arpack"` | SciPy CSR | SciPy `eigsh`, CPU | $O(N k)$ | no / no | any |
| `"lobpcg"` | BCOO | JAX, `lobpcg_standard` | $O(N k + \lvert E \rvert)$ | yes / no | float64 only |

LOBPCG finds the *largest* eigenpairs of a standard problem, so it runs on
the shift-and-flip of $S = D^{-1/2}(L + \alpha V)D^{-1/2}$: if
$\operatorname{spec}(S) \subset [0, c]$, the top $k$ eigenpairs
$(\theta_i, u_i)$ of $cI - S$ are the bottom $k$ of $S$, with
$\lambda_i = c - \theta_i$ and $y_i = D^{-1/2} u_i$. For Laplacian
eigenmaps $S = L_{\mathrm{sym}}$ and $c = 2$ is exact, so the operator is
$I + D^{-1/2} W D^{-1/2}$; a potential adds $\alpha$ times the Gershgorin
bound $\max_i \sum_j |(D^{-1/2} V D^{-1/2})_{ij}|$ to $c$.

- **float64 only.** The wanted eigenvalues of $L_{\mathrm{sym}}$ are tiny
  (about $3 \times 10^{-5}$ for a 20 000-point Swiss roll), and after the
  shift they sit at $2 - \lambda$, next to the bulk. float32 LOBPCG cannot
  separate them and returns a *wrong* embedding (subspace overlap 0.09 with
  ARPACK), not an imprecise one, so without
  `jax.config.update("jax_enable_x64", True)` it raises a `ValueError`.
- **Iterations.** It stops when every pair meets LOBPCG's own eps-level
  tolerance, or at `max_iter` (default 1000); `n_iter` on the fitted model
  records which. The count grows with $N$ and with $c$, because the relative
  gap $(\theta_k - \theta_{k+1}) / (\theta_1 - \theta_{\min})$ shrinks:
  a 10-NN Swiss roll takes about 260 iterations at $N = 1000$, 420 at
  $N = 5000$ and 840 at $N = 20\,000$. A strong potential (large $c$) can
  take thousands; there `"arpack"` is better. The returned pairs get the
  same residual check as `"lanczos"` and raise a `RuntimeError` if it fails
  (raise `max_iter`). Shift-invert or a preconditioner would cut the
  iterations, but `lobpcg_standard` supports neither.
- **Limits.** It needs $5 k < N$, with $k$ the number of pairs (one more
  than `n_components` when the first is dropped); use `"dense"` below that.
  It has no gradients. A dense $(N, N)$ potential (`label_potential`) makes
  each matvec $O(N^2)$; pass a sparse lineax potential (e.g.
  `spatial_spectral_graph(...).laplacian_operator()`) to stay sparse.

```python
jax.config.update("jax_enable_x64", True)
le = kl.LaplacianEigenmaps(n_components=2, eigen_solver="lobpcg").fit(X)
le.embedding, le.n_iter
```

The [spatial-spectral Schrödinger eigenmaps example](../../spatial-spectral-eigenmaps/)
runs this on a synthetic hyperspectral cube, and the
[graphs and spatial models example](../../graphs-and-spatial/) covers grid
graphs, graph Matérn priors, proximity graphs and ICAR structure matrices.

**Out of sample.** The projections embed new points. SEP is LPP with a
potential (``alpha=0`` is LPP exactly); the kernel versions work in a
kernel's feature space, exactly ($O(N^3)$) or with ``approx=`` a feature map
($O(N M^2)$), and their ridge penalises the RKHS norm of the embedding.

```python
sep = kl.SchrodingerEigenmapProjections(n_components=10, alpha=5.0)
Z_test = sep.fit(X_train, V_train).transform(X_test)
klpp = kl.KernelLocalityPreservingProjections(
    kl.RBF(1.0), n_components=5, approx=kl.NystromFeatures(500, key)
).fit(X)
Z = klpp.transform(X_new)
```

::: kernellib.LaplacianEigenmaps

::: kernellib.SchrodingerEigenmaps

::: kernellib.LocalityPreservingProjections

::: kernellib.SchrodingerEigenmapProjections

::: kernellib.KernelLocalityPreservingProjections

::: kernellib.KernelSchrodingerProjections

::: kernellib.laplacian_eigenmap

::: kernellib.schrodinger_eigenmap

::: kernellib.combine_potentials

::: kernellib.barrier_potential

::: kernellib.label_potential

::: kernellib.spatial_spectral_graph

::: kernellib.spatial_spectral_potential
