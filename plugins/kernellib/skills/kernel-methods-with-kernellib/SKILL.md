---
name: kernel-methods-with-kernellib
description: Write kernel methods in JAX on kernellib — kernels as Equinox modules (RBF, Matérn, periodic, composites, derivative kernels), Gram matrices as gaussx operators, matrix-free and preconditioned kernel solves, random Fourier / Nyström / FastFood features, kernel ridge regression at scale (KRR, Falkon, EigenPro), HSIC / CKA / MMD dependence tests, graph kernels and kernel eigenmaps, and scikit-learn adapters. Use whenever a task computes a kernel or Gram matrix, solves a kernel system, approximates a kernel with features, fits a kernel regressor, or measures dependence between samples, in a project that uses (or could use) kernellib.
---

# Kernel methods on kernellib

kernellib turns a kernel into an Equinox module whose hyperparameters are
array leaves, a Gram matrix into a gaussx linear operator, and a kernel fit
into a pytree you can `jit`, `vmap` and differentiate through. Before
writing a distance matrix, a kernel formula, a Cholesky of `K + σ²I`, a CG
loop, random features or an HSIC estimator, look it up:

1. **The capability index** lists every public name with a one-line
   summary, plus gaussx and geonnax's random-feature and basis APIs:
   <https://jejjohnson.github.io/kernellib/reference/capabilities/>. Or
   list the installed version:

   ```python
   import importlib, inspect

   for name in ("kernellib", "kernellib.functional", "kernellib.sklearn"):
       try:
           module = importlib.import_module(name)
       except ImportError:
           continue  # kernellib.sklearn needs the [sklearn] extra
       for attr in getattr(module, "__all__", []):
           doc = (inspect.getdoc(getattr(module, attr)) or "").split("\n")[0]
           print(f"{name}.{attr}: {doc}")
   ```

2. Compose what exists. kernellib is not on PyPI yet:
   `uv add "kernellib @ git+https://github.com/jejjohnson/kernellib.git"`
   (extras: `[sklearn]` for the adapters, `[neighbors]` for approximate
   nearest neighbours).

## What lives where

| You need… | Use |
|---|---|
| A kernel object | `kl.RBF`, `Matern`, `RationalQuadratic`, `Periodic`, `Cosine`, `Linear`, `Polynomial`, `Distance`, `White`, `Constant`; compose with `+`, `*`, a scalar, `.select(dims)`, `.stretch(scale)`, `.periodic(period)`, `Warped`, `Periodised`, `Modulated` |
| A kernel as a pure function on arrays | `kernellib.functional` (`rbf_kernel(X1, X2, variance, lengthscale)`, …; variance before lengthscale) |
| Derivative observations | `kl.Derivative`, `DerivativeIndexed`, `derivative_inputs` |
| A starting lengthscale | `kl.estimate_lengthscale(X, method=...)` (median, Silverman, Scott, …), `lengthscale_grid` |
| `K + σ²I` for a solver | `kl.to_operator(kernel, X, noise=σ²)` (PSD-tagged; `implicit=True` for matrix-free), `to_cross_operator`; then `gaussx.solve`, `gaussx.logdet` |
| Matrix-free matvecs in batches | `kl.ImplicitKernelOperator`, `batched_kernel_matvec`, `implicit_cross_kernel` |
| A low-rank kernel approximation | `kl.RandomFourierFeatures`, `OrthogonalRandomFeatures`, `FastFoodFeatures`, `NystromFeatures` (+ `select_landmarks`), `LaplaceEigenfunctionFeatures`; as operators, `nystrom_operator`, `rff_operator`, `fastfood_operator` |
| Spectral densities / frequency samples | `kernel.spectral_density`, `kernel.sample_frequencies` |
| Kernel ridge regression | `kl.KRR` (any gaussx solver strategy; Nyström / RP-Cholesky preconditioned CG), `Falkon`, `EigenPro` |
| Dependence and two-sample tests | `kl.hsic`, `cka`, `kernel_alignment`, `mmd_squared`, `energy_distance`, `distance_correlation_squared`, `permutation_test`, `CKAAccumulator` (streaming) |
| Kernel PCA, graph embeddings | `kl.KernelPCA`, `LaplacianEigenmaps`, `SchrodingerEigenmaps`, `LocalityPreservingProjections` |
| Graphs and graph kernels | `kl.knn_graph`, `grid_graph`, `mesh_graph`, `graph_laplacian`, `diffusion_kernel`, `matern_graph_kernel`, … |
| scikit-learn pipelines | `kernellib.sklearn.KernelRidge`, `FalkonRegressor`, the feature-map transformers, `HSIC`, `MMD`, `KernelPCA` |

Priors on hyperparameters and GP models live in pyrox-gp, which wraps
these kernels; structured linear algebra (Kronecker, low-rank, solvers,
preconditioners) lives in gaussx.

## The rules your code must keep

- **Hyperparameters are leaves.** `kl.RBF(lengthscale=ell)` stores an
  array, so `jax.grad` reaches it; rebuild a kernel with new values
  (`kl.RBF(lengthscale=jnp.exp(log_ell))`) or `eqx.tree_at`, never mutate
  it. Settings that pick a code path (`Matern(nu=...)`) are static.
- **Solves through gaussx.** Build the operator with `to_operator` and call
  `gaussx.solve` / `logdet` (or an estimator's `solver=`); don't
  `jnp.linalg.solve` / `cho_solve` a Gram. For large N use
  `implicit=True` and an iterative strategy or `KRR(preconditioner=...)`.
- **Fits return new objects.** `KRR(...).fit(X, y)` and
  `RandomFourierFeatures(R, key).fit(kernel, X)` return fitted copies;
  keep the result.
- **Explicit keys.** Random features, landmark selection, preconditioners
  and permutation tests take a `key`; split it, never reuse it.
- **Pointwise-ness is a property.** Check `kernel.is_pointwise`, not
  `isinstance`: composites and wrappers inherit it from their parts. A
  kernel that is not pointwise (only defined on whole Grams) cannot go
  matrix-free: `to_operator(..., implicit=True)` raises for it.
- **JAX rules.** Inputs are 2-D `(N, D)`; float32 in gives float32 out;
  no Python control flow on traced values.

## Worked example

```python
import einx
import gaussx as gx
import jax
import jax.numpy as jnp
import jax.random as jr

import kernellib as kl


k_x, k_e, k_t, k_fit, k_rff, k_perm = jr.split(jr.key(0), 6)
X = jr.uniform(k_x, (400, 2), minval=-2.0, maxval=2.0)  # (N, D)
# y = sin 2x₁ · cos x₂ + ε,  ε ~ 𝒩(0, 0.1²)
y = jnp.sin(2.0 * X[:, 0]) * jnp.cos(X[:, 1]) + 0.1 * jr.normal(k_e, (400,))  # (N,)
X_test = jr.uniform(k_t, (100, 2), minval=-2.0, maxval=2.0)  # (T, D)
f_test = jnp.sin(2.0 * X_test[:, 0]) * jnp.cos(X_test[:, 1])  # (T,)

# 1. A kernel with a data-driven lengthscale (median heuristic)
ell = kl.estimate_lengthscale(X)  # () — scalar lengthscale
kernel = kl.RBF(lengthscale=ell)  # an Equinox module; hyperparameters are leaves

# 2. GP posterior mean: (K + σ²I)⁻¹ y through gaussx on a PSD-tagged operator
K_op = kl.to_operator(kernel, X, noise=0.01)  # (N, N) lineax operator
alpha = gx.solve(K_op, y)  # (N,)
mean = kernel(X_test, X) @ alpha  # (T, N) @ (N,) → (T,)

# 3. Kernel ridge regression, matrix-free, Nyström-preconditioned CG
krr = kl.KRR(
    kernel,
    regularization=1e-4,
    implicit=True,
    preconditioner="nystrom",
    preconditioner_rank=50,
).fit(X, y, key=k_fit)
pred = krr.predict(X_test)  # (T,)


# 4. The fit is a pytree: differentiate a validation loss through it
def val_loss(log_ell):
    k = kl.RBF(lengthscale=jnp.exp(log_ell))
    return kl.KRR(k, regularization=1e-4).fit(X, y).loss(X_test, f_test)


grad = jax.grad(val_loss)(jnp.log(ell))  # ∂loss/∂log ℓ, ()

# 5. Random Fourier features: Φ Φᵀ ≈ K in O(N R)
rff = kl.RandomFourierFeatures(512, k_rff).fit(kernel, X)
Phi = rff(X)  # (N, R)
gram_err = jnp.max(jnp.abs(Phi @ Phi.T - kernel(X, X)))  # ()

# 6. Is y independent of x₁? HSIC with a permutation p-value
x1 = einx.id("n -> n 1", X[:, 0])  # (N, 1)
Y = einx.id("n -> n 1", y)  # (N, 1)
test = kl.permutation_test(
    lambda A, B: kl.hsic(kl.RBF(), kl.RBF(), A, B),
    x1,
    Y,
    key=k_perm,
    n_permutations=199,
)
```

With x64 on, the GP posterior mean reaches an RMSE of about 0.10 against
the noise-free function, the preconditioned CG converges in a handful of
iterations, the positive gradient says the median-heuristic lengthscale
over-smooths, 512 random features match the Gram to about 0.1 everywhere,
and the permutation test rejects independence (p = 0.005, the smallest
attainable with 199 permutations).

## Don't write it — use kernellib

| Don't write… | Use |
|---|---|
| `jnp.exp(-0.5 * cdist(X1, X2) ** 2 / ell**2)` | `kl.RBF(lengthscale=ell)(X1, X2)` or `kernellib.functional.rbf_kernel` |
| `jnp.linalg.solve(K + s2 * jnp.eye(n), y)`, `cho_solve` | `gaussx.solve(kl.to_operator(kernel, X, noise=s2), y)` |
| A CG loop, a Nyström preconditioner | `kl.KRR(kernel, implicit=True, preconditioner="nystrom")`, or a gaussx strategy |
| `cos(X @ W + b)` random features | `kl.RandomFourierFeatures(R, key).fit(kernel, X)` (draws from the kernel's own spectral density) |
| Inducing-point / Nyström KRR | `kl.Falkon`, `kl.NystromFeatures` |
| `trace(K H L H) / n**2` | `kl.hsic(kernel_x, kernel_y, X, Y)`; normalised, `kl.cka` |
| An MMD two-sample test | `kl.mmd_squared` + `kl.permutation_test(..., kind="two_sample")` |
| A median-distance bandwidth | `kl.estimate_lengthscale(X)` |
| A kNN graph Laplacian and its eigenvectors | `kl.knn_graph`, `kl.graph_laplacian`, `kl.LaplacianEigenmaps` |
| A scikit-learn wrapper around a JAX kernel model | `kernellib.sklearn` |

## Self-check before you finish

- No dense Gram is formed on a path meant to scale (look for `kernel(X, X)`
  with large N, or `.as_matrix()`); solves go through gaussx.
- `jax.grad` of your loss with respect to the kernel's hyperparameters is
  finite (including at coincident points).
- Every random routine got its own key.
- A float32 input stays float32.

If kernellib lacks what you need, keep your addition small and shaped like
kernellib (a kernel subclassing `kl.AbstractStationaryKernel` or
`kl.AbstractPointwiseKernel`, a feature map subclassing
`kl.AbstractFeatureMap`) and consider proposing it upstream at
<https://github.com/jejjohnson/kernellib/issues>.
