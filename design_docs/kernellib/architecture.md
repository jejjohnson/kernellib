---
status: draft
version: 0.1.0
date: 2026-09-25
---

# kernellib: A Scalable Kernel Library Between gaussx and pyrox-gp

## Summary

kernellib becomes the home for **kernels and kernel methods that are not
Gaussian processes**: kernel functions and their composition, spectral
densities and feature maps, kernel ridge regression, dependence measures
(HSIC, CKA, MMD), kernel embeddings, and kernel derivatives. It sits between
[gaussx](https://github.com/jejjohnson/gaussx) (structured linear algebra,
solvers, operators) and
[pyrox-gp](https://github.com/jejjohnson/pyrox) (GP models with NumPyro
priors and guides), and it inherits scale from gaussx rather than
reimplementing it: every kernel can be turned into a gaussx linear operator,
and every algorithm in kernellib solves through gaussx's solver strategies.

gaussx's own kernel layer moves here too: the kernel operators, the Nyström /
RFF operators, the matrix-level HSIC / MMD statistics, and the Falkon and
EigenPro primitives. After the move gaussx is kernel-agnostic linear
algebra, which is what its vision document already claims, and **everything
with a kernel in it lives in kernellib**.

The existing kernellib code (2018, numpy / scikit-learn / numba) is retired.
Its *scope* survives; its code does not.

The dependency chain after the refactor:

```
lineax · matfree · equinox · einx          geonnax
          │                                   │
          ▼                                   │
        gaussx   ◄────────────────────────────┤   (gaussx never imports kernellib)
          │                                   │
          ▼                                   │
       kernellib ◄────────────────────────────┘   (kernellib never imports numpyro)
          │
          ▼
       pyrox-gp   (Parameterized wrappers, priors, guides, GP models)
          │
          ▼
       pyrox-nn
```

---

## Motivation: what exists today

The survey below is of `gaussx@0.1.0`, `pyrox-gp@0.1.5`, the current
`kernellib@master`, and `pysim`.

**gaussx already contains the scalable half of a kernel library.** It is all
kernel-agnostic: each entry point takes a callable `k(x, x') -> scalar` (or
`k(params, x, x')`) and never a concrete kernel. Every row below except the
grid helpers, the two preconditioners and `stable_squared_distances` moves to
kernellib (see *Boundaries*). Nothing else in gaussx depends on them.

| gaussx symbol | What it is |
|---|---|
| `KernelOperator`, `ImplicitKernelOperator`, `ImplicitCrossKernelOperator` | Matrix-free kernel matvecs with custom JVPs; `O(N)` memory via `lax.scan` |
| `nystrom_operator`, `rff_operator` | Low-rank kernel approximations as `LowRankUpdate` (Woodbury solves for free) |
| `center_kernel`, `centering_operator`, `hsic`, `mmd_squared` | Kernel statistics on *matrices / operators* |
| `falkon_preconditioner`, `falkon_solve`, `falkon_predict` | Falkon preconditioned CG for Nyström KRR |
| `eigenpro_preconditioner`, `eigenpro_step_size`, `eigenpro_correction` | EigenPro spectral preconditioning primitives for kernel SGD |
| `NystromPreconditioner`, `PartialCholeskyPreconditioner` | Preconditioners for CG on kernel systems |
| `stable_rbf_kernel`, `stable_squared_distances`, `batched_kernel_matvec` | Mixed-precision Gram assembly |
| `create_grid`, `grid_data`, `cubic_interpolation_weights` | KISS-GP grids |

**pyrox-gp owns the concrete kernels.** Two files, plus the spectral helpers:

| pyrox-gp module | Contents | Depends on pyrox / numpyro? |
|---|---|---|
| `pyrox_gp._src.kernels` | `rbf_kernel`, `matern_kernel`, `periodic_kernel`, `linear_kernel`, `rational_quadratic_kernel`, `polynomial_kernel`, `cosine_kernel`, `white_kernel`, `constant_kernel`, `kernel_add`, `kernel_mul` on Gram matrices | **No** (einx, jax, jaxtyping only) |
| `pyrox_gp._protocols.Kernel` | `eqx.Module` with abstract `__call__(X1, X2) -> Gram`, default `gram`, `diag` | **No** |
| `pyrox_gp._kernels` | `RBF`, `Matern`, ... as `Parameterized` wrappers with `get_param`, priors, autoguides | **Yes** |
| `pyrox_gp._basis._spectral_density`, `_rff` | Spectral density and RFF prior draws; dispatch by `isinstance(kernel, RBF | Matern)` | Yes (numpyro.distributions, concrete classes) |
| `pyrox_gp._context` | Per-call context so several kernel evaluations share one hyperparameter draw | Yes |

pyrox-nn imports `spectral_density`, `fourier_basis`, `_kernel_context` and
`Kernel` from pyrox-gp.

**The two halves are not connected.** pyrox-gp never constructs a
`KernelOperator` or `ImplicitKernelOperator` in source (one docstring mention).
Its kernels return Gram matrices; gaussx's operators want pointwise scalars.
There is no object today that is both "an RBF kernel with a lengthscale" and
"something `gaussx.solve` can be handed".

**Three HSIC implementations, no shared kernel.** `kernellib.dependence.hsic`,
`pysim.kernel.hsic` (HSIC / KA / CKA, randomized Nyström and RFF variants,
bandwidth heuristics) and `gaussx.hsic` (matrix-level) all exist. None of
them can be fed a kernel object.

**Old kernellib is not reusable code.** numpy, scikit-learn estimators,
numexpr, numba, a Python 2 era `setup.py`, and 1000 lines of hand-derived
RBF / ARD derivative kernels. Under JAX the derivatives are `jax.jacfwd` of a
pointwise kernel. What is worth keeping is the feature list: KRR, RFF,
Nyström, FastFood, HSIC / RHSIC and their input-gradients, kernel PCA style
embeddings, graph kernels from adjacency matrices, Laplacian and Schrödinger
eigenmaps, LPP.

---

## Goals and non-goals

### Goals

1. One kernel abstraction, usable without NumPyro, that pyrox-gp wraps rather
   than duplicates.
2. Scale inherited from gaussx: any kernel → gaussx operator; any algorithm
   → any gaussx solver strategy (`DenseSolver`, `CGSolver`, `BBMMSolver`,
   `PreconditionedCGSolver`, `AutoSolver`).
3. Kernel methods that are not GPs as first-class citizens: KRR (dense,
   Nyström / Falkon, EigenPro), dependence measures, kernel embeddings and
   PCA, derivatives, feature maps.
4. Spectral side of kernels (densities, RFF / ORF / Laplace-eigenfunction
   feature maps) in one place, shared by pyrox-gp pathwise sampling and
   pyrox-nn spectral layers.
5. Zero behaviour change for pyrox-gp and pyrox-nn users across the migration.
6. gaussx ends up kernel-agnostic: no module, symbol or docs page in gaussx
   has a kernel in it.

### Non-goals

| Not this | Lives in |
|---|---|
| Priors, guides, sample sites, `Parameterized` | pyrox-gp |
| GP models, ELBOs, conditionals, Markov / sparse GPs | pyrox-gp (modelling), gaussx (`_gp` recipes) |
| Structured operators, solvers, preconditioners, logdets | gaussx |
| Basis-function zoo (Laplace eigenfunctions, spherical harmonics, Slepian, wavelets, Gabor) | geonnax |
| Information theory (entropy, MI estimators, KDE, kNN) | pysim, unchanged |
| Multi-output kernels (LMC, ICM, OILMM) | pyrox-gp, for now (see open questions) |
| Positivity transforms / constrained optimisation of hyperparameters | caller (optax / pyrox-gp), see open questions |
| numpy / PyTorch backends | never |

---

## Ownership map

The rule in one sentence: **anything with a kernel in it lives in kernellib,
from the operators up; gaussx is kernel-agnostic linear algebra; geonnax is
kernel-agnostic feature-map arithmetic.**

| Responsibility | Owner | Notes |
|---|---|---|
| Pointwise kernel math, Gram assembly, composition | **kernellib** | Moved from `pyrox_gp._src.kernels` |
| Kernel abstraction (`AbstractKernel`, `AbstractPointwiseKernel`, `AbstractStationaryKernel`) | **kernellib** | `pyrox_gp.Kernel` becomes an alias |
| Kernel *operators* (`KernelOperator`, `ImplicitKernelOperator`, `ImplicitCrossKernelOperator`) and the `to_operator` bridge | **kernellib** | Moved from gaussx; lineax operators with custom JVPs, so they plug into every gaussx solver unchanged |
| Low-rank kernel operators (`nystrom_operator`, `rff_operator`, `fastfood_operator`, `FastFoodOperator`) | **kernellib** | Nyström / RFF moved from gaussx; FastFood new |
| Matrix-level kernel statistics (`hsic`, `cka`, `mmd_squared`, `center_kernel`, `centering_operator`) | **kernellib** | Moved from gaussx into `kernellib.functional`; unbiased HSIC and CKA new |
| Falkon and EigenPro primitives (preconditioners, one CG solve, step size, correction) | **kernellib** | Moved from gaussx, next to the estimators that use them |
| `stable_rbf_kernel`, `batched_kernel_matvec`, `batched_kernel_rmatvec` | **kernellib** | Moved from gaussx |
| Spectral densities, frequency sampling | **kernellib** | Method on stationary kernels; replaces isinstance dispatch in `pyrox_gp._basis` |
| Feature maps (`NystromFeatures`, `RandomFourierFeatures`, `OrthogonalRandomFeatures`, `FastFoodFeatures`, `LaplaceEigenfunctionFeatures`) | **kernellib** | Draw landmarks / frequencies / scalings from the kernel; use `geonnax.randfeat` and `geonnax.basis` for the arithmetic |
| Bandwidth heuristics (median / mean / Silverman / Scott, subsampled, k-th neighbour) | **kernellib** | Ported from pysim |
| `KRR`, `Falkon`, `EigenPro` estimators (fit / predict, landmark selection, training loops) | **kernellib** | |
| HSIC / CKA / kernel alignment / MMD over *kernels and data*, randomized variants, permutation tests, input gradients | **kernellib** | Ported from pysim; every path ends in the matrix-level functions above |
| Kernel PCA, kernel embeddings, graph kernels from adjacency | **kernellib** | Over `gaussx.eig`, `root_decomposition`, `LowRankUpdate` |
| Kernel derivatives (∂k/∂x, Gram blocks of derivatives, derivative of a KRR predictor) | **kernellib** | Autodiff of `pairwise` |
| Structured operators (`LowRankUpdate`, `Kronecker`, ...), solvers and strategies, preconditioners (`NystromPreconditioner`, `PartialCholeskyPreconditioner`), `trace_product`, `stable_squared_distances`, grids for interpolated operators, Gaussians, GP recipes, SSMs | **gaussx** | Kernel-agnostic. `stable_squared_distances` stays because gaussx's ensemble localization uses it and gaussx cannot import kernellib |
| `Parameterized` kernel wrappers, priors, context scoping | **pyrox-gp** | Delegates the math to kernellib |
| Basis functions on boxes, spheres, graphs; RFF forward helpers | **geonnax** | kernellib depends on it |

---

## The kernel contract

This is the one design decision that everything else hangs on. gaussx wants a
pointwise scalar function; pyrox-gp and every closed-form fast path want a Gram
matrix. Both must be first-class, and multi-output / structured kernels in
pyrox-gp only have a Gram form. So the hierarchy has three levels:

```python
class AbstractKernel(eqx.Module):
    """Anything that produces a Gram matrix. Same surface as today's pyrox_gp.Kernel."""

    @abstractmethod
    def __call__(self, X1: Float[Array, "N1 D"], X2: Float[Array, "N2 D"]) -> Float[Array, "N1 N2"]: ...

    def gram(self, X: Float[Array, "N D"]) -> Float[Array, "N N"]:
        return self(X, X)

    def diag(self, X: Float[Array, "N D"]) -> Float[Array, " N"]:
        return jnp.diag(self(X, X))            # overridden by every concrete kernel


class AbstractPointwiseKernel(AbstractKernel):
    """A kernel defined by k(x, x'). Gains the implicit-operator path."""

    @abstractmethod
    def pairwise(self, x: Float[Array, " D"], y: Float[Array, " D"]) -> Float[Array, ""]: ...

    def __call__(self, X1, X2):
        return jax.vmap(lambda x: jax.vmap(lambda y: self.pairwise(x, y))(X2))(X1)


class AbstractStationaryKernel(AbstractPointwiseKernel):
    """k(x, x') = variance * shape(r), r = ||(x - x') / lengthscale||."""

    lengthscale: Float[Array, ""] | Float[Array, " D"]
    variance: Float[Array, ""]

    @abstractmethod
    def shape(self, r2: Float[Array, ""]) -> Float[Array, ""]: ...
    @abstractmethod
    def spectral_density(self, omega: Float[Array, " D"]) -> Float[Array, ""]: ...
    def sample_frequencies(self, key, n: int) -> Float[Array, "n D"]: ...   # for RFF

    def pairwise(self, x, y): ...                 # variance * shape(scaled r²)
    def __call__(self, X1, X2): ...               # closed-form Gram via stable squared distances
    def diag(self, X): return self.variance * jnp.ones(X.shape[0])
```

Rules:

- **Hyperparameters are plain array fields.** No transforms, no priors, no
  contexts. A kernel is a frozen PyTree of positive numbers; constraining
  them is the optimiser's or pyrox-gp's job.
- **Stationary kernels override `__call__`** with the closed-form Gram path.
  A double `vmap` of RBF is both slower and less stable than
  `gaussx.stable_squared_distances`; the pointwise form exists for autodiff
  and implicit operators, not as the default Gram path.
- **`pairwise` is the differentiation point.** Kernel derivatives, HSIC
  input-gradients and the custom-JVP implicit operators all go through it.
- **`AbstractKernel` keeps only `__call__` abstract**, so `pyrox_gp.Kernel`
  can be replaced by an alias without touching `LMCKernel`, `ICMKernel`,
  `OILMMKernel`, warped kernels or any user subclass.

### Composition

Composition happens at the kernel level, not on evaluated matrices as
`kernel_add` / `kernel_mul` do today:

```python
class Sum(AbstractPointwiseKernel):      kernels: tuple[AbstractKernel, ...]
class Product(AbstractPointwiseKernel):  kernels: tuple[AbstractKernel, ...]
class Scaled(AbstractPointwiseKernel):   kernel: AbstractKernel; scale: Float[Array, ""]
class ActiveDims(AbstractPointwiseKernel): kernel: AbstractKernel; dims: tuple[int, ...]   # static
class Warped(AbstractPointwiseKernel):   kernel: AbstractKernel; warp: Callable           # input transform
```

`Sum` / `Product` of pointwise kernels are pointwise (so they still get the
implicit path); of Gram-only kernels they fall back to Gram arithmetic.
`__add__`, `__mul__`, `__rmul__` are sugar over these. The existing
`kernel_add` / `kernel_mul` matrix helpers stay in `kernellib.functional` for
callers that already hold matrices.

### The bridge to gaussx

```python
def to_operator(
    kernel: AbstractKernel,
    X: Float[Array, "N D"],
    *,
    noise: float | Float[Array, ""] | None = None,
    implicit: bool = False,
) -> lx.AbstractLinearOperator:
    """K(X, X) (+ noise I) as a gaussx operator, PSD/symmetric tagged."""

def to_cross_operator(
    kernel: AbstractKernel,
    X1: Float[Array, "N D"],
    X2: Float[Array, "M D"],
    *,
    implicit: bool = False,
    batch_size: int = 1024,
) -> lx.AbstractLinearOperator:
    """K(X1, X2) as a gaussx operator."""
```

Mapping:

| `implicit` | Kernel type | Returns |
|---|---|---|
| `False` | any | `lx.MatrixLinearOperator(kernel(X, X) [+ noise I], tags={symmetric, psd})` |
| `True`  | pointwise | `kernellib.ImplicitKernelOperator(kernel_fn, X, noise_var, params=params)` |
| `True`  | Gram-only | `TypeError` naming the kernel; no silent densification |
| cross, `True` | pointwise | `kernellib.ImplicitCrossKernelOperator(kernel_fn, X1, X2, batch_size, params=params)` |

`kernel_fn` and `params` come from `eqx.partition(kernel, eqx.is_array)`:
the array leaves are `params` (so the operator's custom JVP differentiates
through hyperparameters), the static remainder is closed over in
`kernel_fn = lambda p, x, y: eqx.combine(p, static).pairwise(x, y)`.

This is the whole "inherit scale" mechanism. The operators are lineax
`AbstractLinearOperator`s carrying `symmetric` / `positive_semidefinite` tags,
and gaussx's strategies dispatch on those tags rather than on operator type,
so after the bridge everything else is gaussx:

```python
K = kl.to_operator(kernel, X, noise=1e-2, implicit=True)
alpha = gaussx.solve(K, y, solver=gaussx.PreconditionedCGSolver(preconditioner_rank=100))
```

---

## Package layout

```
src/kernellib/
├── __init__.py            # public API, flat, re-exports only
├── _einx.py               # einx wrappers (copied pattern from gaussx; D9 rule: no raw reshape/einsum)
├── _testing.py            # PSD checks, pointwise-vs-gram, finite-difference derivative asserts
├── functional/            # arrays in, arrays out — no kernel objects, no keys
│   ├── _stationary.py     #   rbf, matern, rational_quadratic, periodic, cosine, white, constant; stable_rbf_kernel (moved)
│   ├── _nonstationary.py  #   linear, polynomial
│   ├── _distances.py      #   lengthscale-scaled distances, thin over gaussx.stable_squared_distances
│   ├── _compose.py        #   kernel_add, kernel_mul on matrices (kept for matrix callers)
│   └── _statistics.py     #   center_kernel, centering_operator, hsic (biased / unbiased), cka, mmd_squared (moved from gaussx, extended)
├── _kernels/
│   ├── _base.py           # AbstractKernel, AbstractPointwiseKernel, AbstractStationaryKernel
│   ├── _stationary.py     # RBF, Matern(nu static ∈ {0.5,1.5,2.5}), RationalQuadratic, Periodic, Cosine, White, Constant
│   ├── _nonstationary.py  # Linear, Polynomial(degree static)
│   └── _compose.py        # Sum, Product, Scaled, ActiveDims, Warped + operator sugar
├── _operators/
│   ├── _kernel.py         # KernelOperator (moved from gaussx)
│   ├── _implicit.py       # ImplicitKernelOperator (moved)
│   ├── _implicit_cross.py # ImplicitCrossKernelOperator (moved)
│   ├── _batched.py        # batched_kernel_matvec / rmatvec (moved)
│   ├── _low_rank.py       # nystrom_operator, rff_operator (moved); fastfood_operator, FastFoodOperator (new, Walsh–Hadamard matvec)
│   ├── _utils.py          # vmap_over_batch_dims, _to_frozenset (copied from gaussx private helpers)
│   └── _bridge.py         # to_operator, to_cross_operator
├── _spectral/
│   ├── _density.py        # spectral densities per stationary kernel (moved from pyrox_gp._basis)
│   ├── _base.py           # AbstractFeatureMap: fit(kernel[, X]) -> fitted map; __call__(X) -> Φ; operator(X) -> LowRankUpdate
│   ├── _rff.py            # draw_rff_cosine_basis, evaluate_rff_cosine_paths (moved); RandomFourierFeatures, OrthogonalRandomFeatures over geonnax.randfeat
│   ├── _fastfood.py       # FastFoodFeatures: draws B, Pi, G and the kernel-dependent S; operator(X) -> fastfood_operator
│   ├── _nystrom.py        # NystromFeatures: landmark selection (uniform; leverage-score later), Φ = K_xz L_zz^{-T}; operator(X) -> nystrom_operator
│   └── _laplace.py        # HSGP-style Laplace-eigenfunction approximation over geonnax.basis.fourier_basis / fourier_eigenvalues
├── _heuristics.py         # estimate_lengthscale(X, method=median|mean|silverman|scott, subsample, kth), sigma<->gamma, grids (pysim port)
├── _regression/
│   ├── _base.py           # AbstractEstimator: config in the constructor, fit(X, y[, key]) -> fitted module, predict(X)
│   ├── _krr.py            # KRR: any gaussx solver strategy, dense or implicit operator
│   ├── _falkon.py         # falkon_preconditioner, falkon_solve, falkon_predict, FalkonPreconditioner (moved) + Falkon estimator
│   └── _eigenpro.py       # eigenpro_preconditioner, eigenpro_step_size, eigenpro_correction (moved) + EigenPro estimator (the lax.scan loop)
├── _dependence/
│   ├── _hsic.py           # hsic / cka / kernel_alignment on kernels + data; approx= for Nyström / RFF / FastFood; hsic_input_gradient
│   ├── _mmd.py            # mmd over one kernel + two samples; linear-time estimator
│   └── _permutation.py    # permutation_test(statistic, ...) for any of the above
├── _decomposition/
│   ├── _kpca.py           # kernel PCA / kernel embeddings via gaussx.eig on centered operators
│   └── _graph.py          # graph kernels from adjacency: diffusion, regularized Laplacian, random-walk, cosine (old kernellib.decomposition.graph, JAX)
└── _derivatives.py        # kernel_jacobian, derivative Gram blocks, predictor_gradient for KRR
```

`functional/` is the array-level namespace: pure kernel functions and the
matrix-level statistics that used to be `gaussx.hsic` and friends. The same
names at the top level (`kernellib.hsic`) are the kernel-and-data versions.

Manifold learning from the old `decomposition/` (Laplacian eigenmaps,
Schrödinger eigenmaps, LPP, neighbour graphs) is **not** in the first
milestone. Adjacency construction is a nearest-neighbours problem, not a
kernel one; it can come back later on top of `graph_laplacian_eigpairs`
from geonnax if there is demand.

Dependencies:

```toml
dependencies = [
  "jax>=0.10", "jaxlib>=0.10", "equinox>=0.13.8", "jaxtyping>=0.3",
  "einx>=0.4.3", "lineax>=0.1.1",
  "gaussx>=0.1.0",          # >=0.2.0 once the kernel layer is removed there
  "geonnax @ git+https://github.com/jejjohnson/geonnax.git@v0.0.5",   # until it ships to PyPI
]
```

No numpyro. No scikit-learn. No numpy beyond what JAX brings.

---

## Boundaries in detail

### gaussx: the kernel layer moves out

The rule: **anything with a kernel in it lives in kernellib, from the
operators up. gaussx is kernel-agnostic linear algebra.** An earlier draft
split by "matrices in, matrices out → gaussx", which left a `_kernels/`
package and a kernels docs page in gaussx next to a library called kernellib.
That was a boundary drawn for tidiness of *signatures*, not of *concepts*,
and it is gone.

The move is clean. Nothing inside gaussx depends on its kernel pieces: no
module outside `_kernels/` and the three kernel-operator files imports them,
and the solver strategies dispatch on lineax tags rather than on operator
type, so a kernel operator defined in kernellib plugs into `gaussx.solve`
exactly as one defined in gaussx does. pyrox-gp uses none of these symbols
in source. The one internal consumer is `stable_squared_distances`, used by
`euclidean_distance` in the ensemble-localization code.

| Moves to kernellib (~2000 lines + tests) | Stays in gaussx |
|---|---|
| `KernelOperator`, `ImplicitKernelOperator`, `ImplicitCrossKernelOperator` (+ `implicit_cross_kernel`) | `LowRankUpdate`, `Kronecker`, `BlockDiag`, `Toeplitz`, `Circulant`, ... every kernel-agnostic operator |
| `nystrom_operator`, `rff_operator` | `NystromPreconditioner`, `PartialCholeskyPreconditioner` (probe *any* operator) |
| `center_kernel`, `centering_operator`, `hsic`, `mmd_squared` | `trace_product`, `solve`, `logdet`, `eig`, `root_decomposition`, all primitives and strategies |
| `falkon_preconditioner`, `falkon_solve`, `falkon_predict`, `FalkonPreconditioner` | `create_grid`, `grid_data`, `cubic_interpolation_weights` (belong to `InterpolatedOperator` / KISS-GP) |
| `eigenpro_preconditioner`, `eigenpro_step_size`, `eigenpro_correction`, `EigenProPreconditioner` | `stable_squared_distances` |
| `stable_rbf_kernel`, `batched_kernel_matvec`, `batched_kernel_rmatvec` | `_gp/` recipes, `_ssm/`, `_quadrature/`, `_inference/`, distributions |
| `tests/kernels/*`, `tests/operators/test_kernel*.py`, `tests/operators/test_implicit*.py`, `tests/linalg/test_batched_matvec.py`, the `stable_rbf_kernel` tests | everything else |
| `docs/api/kernels.md` and the kernel rows of `docs/api/operators.md` | |

**Why `stable_squared_distances` stays (option 1).** It is a distance
primitive, not a kernel: the `‖x‖² + ‖z‖² − 2x·z` expansion with float64
accumulation and a clamp at zero, in `_linalg/_mixed_precision.py` next to
the other mixed-precision helpers. Kernels are one consumer; Gaspari–Cohn
localization in gaussx's ensemble Kalman code is another, and gaussx cannot
import kernellib. The alternatives were a private copy of 48 lines of subtle
numerics inside gaussx (two implementations that drift) or moving the
localization code, which is ensemble-filtering territory that gaussx's vision
already assigns to filterax. kernellib's `functional/_distances.py` wraps it
and adds the lengthscale scaling.

**Private helpers.** The three operator files use `vmap_over_batch_dims` and
`_to_frozenset` from gaussx's private `_operators/` internals. kernellib copies
both (a few lines each) into `_operators/_utils.py` rather than importing
private names.

**gaussx changes:**

1. Remove the kernel layer in **gaussx 0.2.0**: the files, their tests, the
   `__init__` exports, `docs/api/kernels.md`, and the `_kernels/` row in
   `docs/architecture.md`. No deprecation shims: gaussx cannot re-export from
   kernellib without creating the cycle, and there are no external users to
   protect. The CHANGELOG entry names the new home of every symbol.
2. Add a `LowRankUpdate × LowRankUpdate` fast path to `trace_product`
   (`tr(UUᵀVVᵀ) = ‖UᵀV‖²_F`). This is generic linear algebra and stays a
   gaussx contribution; it is what makes randomized HSIC `O(n m²)`.
3. Documentation: the ecosystem diagrams in `docs/vision.md` and
   `docs/architecture.md` gain a kernellib node between gaussx and pyrox-gp,
   and the "not this, go here" table gains a row for kernels and kernel
   methods.

gaussx must never import kernellib. gaussx tests that need a kernel (e.g.
for `NystromPreconditioner`) use inline lambdas as they do now.

### pyrox-gp: wrappers stay, math moves

| Today | After |
|---|---|
| `pyrox_gp._src.kernels.rbf_kernel` etc. | Module body becomes `from kernellib.functional import *` re-exports for one deprecation cycle, then the module is removed |
| `pyrox_gp._protocols.Kernel` | `Kernel = kernellib.AbstractKernel` (alias). Same abstract method, same `gram` / `diag` defaults, so every existing subclass is unaffected |
| `pyrox_gp._kernels.RBF.__call__` calls `_k.rbf_kernel(X1, X2, self.get_param(...), ...)` | Unchanged in phase 2. In phase 4 gains `frozen() -> kernellib.RBF`, which resolves the params once inside the current context and returns a plain kernellib kernel |
| `pyrox_gp._basis._spectral_density.spectral_density(kernel, omega)` with `isinstance(kernel, RBF | Matern)` | `kernel.frozen().spectral_density(omega)` for pyrox kernels; kernellib kernels pass through. Same public signature, so pyrox-nn is unaffected |
| `pyrox_gp._basis._rff.draw_rff_cosine_basis` | Thin wrapper over `kernellib.spectral.draw_rff_cosine_basis(kernel.frozen(), ...)`; numpyro sampling of the frequencies stays here if it is a sample site, otherwise it moves entirely |
| `pyrox_gp._context._kernel_context` | Unchanged. `frozen()` is called inside it |

`pyrox_gp.SDEKernel` continues to be re-exported from `gaussx._ssm`; it is a
state-space object, not a covariance function, and is out of kernellib's
scope.

pyrox-gp adds `kernellib>=0.1` to its dependencies. pyrox-nn is untouched:
its imports of `spectral_density`, `fourier_basis`, `_kernel_context` and
`Kernel` from pyrox-gp all keep resolving.

### geonnax: bases and feature-map primitives

kernellib depends on geonnax for the things geonnax already does well and
kernellib should not duplicate:

| geonnax symbol | kernellib use |
|---|---|
| `randfeat.rff_forward`, `rff_cosine_forward`, `orthogonal_blocks`, `OrthogonalRandomFeatures` | RFF / ORF feature maps; kernellib supplies the frequencies from `sample_frequencies` |
| `basis.fourier_basis`, `fourier_eigenvalues` | Laplace-eigenfunction (HSGP) approximation of any stationary kernel on a box: `Φ diag(S(√λ)) Φᵀ` as a `LowRankUpdate` |
| `basis.graph_laplacian_eigpairs` | Graph kernels and future eigenmaps |
| `basis.real_spherical_harmonics`, `harmonic_degrees` | Kernels on the sphere via spherical-harmonic expansions (later) |
| `basis.rbf_basis`, `wendland_c2`, `wendland_c4` | Compactly supported kernels (later) |

geonnax depends only on jax, equinox, jaxtyping, einops, einx, so there is
no cycle. It is pinned by git tag exactly as pyrox-gp pins it until it is
on PyPI.

### pysim: reimplemented, not modified

pysim is not touched. The following is reimplemented in kernellib in JAX with
the same semantics, and pysim may later import from kernellib if wanted:

| pysim | kernellib |
|---|---|
| `HSIC(center, bias, kernel, gamma)` with `score(normalize)`: HSIC (centred, unnormalised), KA (uncentred, normalised), CKA (centred, normalised) | `hsic(kx, ky, X, Y, *, center=True, estimator="biased" \| "unbiased")`, `cka(...)`, `kernel_alignment(...)` |
| `RandomizedHSIC` (Nyström) and `RFFHSIC` | `hsic(..., approx=NystromFeatures(...) \| RandomFourierFeatures(...) \| FastFoodFeatures(...))`: the feature map yields `LowRankUpdate` operators for `X` and `Y`, and `kernellib.functional.hsic` evaluates `‖Φ̃xᵀ Φ̃y‖²_F / n²` in `O(n m²)` through gaussx's low-rank `trace_product` fast path |
| `estimate_sigma` / `estimate_gamma` (mean, median, Silverman, Scott; subsample; k-th percentile neighbour), `sigma_to_gamma`, `get_sigma_grid` | `estimate_lengthscale(X, *, method, subsample, percent, key)`, `lengthscale_to_gamma`, `lengthscale_grid` |
| `RandomFourierFeatures` (sklearn transformer) | `RandomFourierFeatures(n_features, key).fit(kernel)` eqx.Module with `__call__(X) -> Φ` and `operator(X)` |
| `information/*` (entropy, MI, KDE, kNN) | out of scope |

Old kernellib's `hsic_rbf_derivative` / `rhsic_rff_derivative` become
`jax.grad(hsic, argnums=...)` with respect to `X`; a convenience
`hsic_input_gradient` is exposed for discoverability.

---

## Public API sketch

Two conventions throughout. Dependence measures are plain functions
returning scalars. Estimators and feature maps are immutable equinox
modules: the constructor holds configuration, `fit` returns a *new* fitted
module, `predict` / `__call__` act on the fitted one, and everything is
`jit` / `grad` / `vmap` compatible because fitted state is plain fields.

Two namespaces. `kernellib.functional` is arrays in, arrays out: pure kernel
functions and the matrix-level statistics. The top level is kernels and data
in.

### Kernels and the bridge

```python
import gaussx as gx
import jax
import jax.numpy as jnp
import kernellib as kl

k = kl.RBF(lengthscale=jnp.array([1.0, 0.5]), variance=1.0) + kl.White(1e-3)
K = k(X1, X2)                    # Gram, closed-form path
kd = k.diag(X)
s = k.spectral_density(omega)    # stationary kernels only
ell = kl.estimate_lengthscale(X, method="median", subsample=2000, key=key)

# operator level (moved from gaussx): callable + points in, lineax operator out
K_op = kl.ImplicitKernelOperator(kernel_fn, X, noise_var=0.1, params=params)
K_nm = kl.ImplicitCrossKernelOperator(kernel_fn, X, Z, batch_size=1024)

# the bridge builds those from a kernel object
K_op = kl.to_operator(k, X, noise=0.1, implicit=True)
alpha = gx.solve(K_op, y, solver=gx.PreconditionedCGSolver(preconditioner_rank=100))
```

### Dependence

The dense path and the randomized path end in the same matrix-level
function; only what it is handed differs.

```python
# matrix level: matrices or operators in, scalar out
h = kl.functional.hsic(K, L)                              # biased (moved from gaussx)
h = kl.functional.hsic(K, L, estimator="unbiased")
c = kl.functional.cka(K, L)
m = kl.functional.mmd_squared(K_xx, K_yy, K_xy)
Kc = kl.functional.center_kernel(K)                       # LowRankUpdate in, LowRankUpdate out

# kernel level: kernels and data in
kx = kl.RBF(lengthscale=kl.estimate_lengthscale(X))
ky = kl.RBF(lengthscale=kl.estimate_lengthscale(Y))
h = kl.hsic(kx, ky, X, Y)                                  # -> functional.hsic(kx.gram(X), ky.gram(Y))
h = kl.hsic(kx, ky, X, Y, estimator="unbiased")
c = kl.cka(kx, ky, X, Y)
a = kl.kernel_alignment(kx, ky, X, Y)                      # uncentred, normalised
m = kl.mmd(kx, X, Y)                                       # one kernel, two samples
p = kl.permutation_test(kl.hsic, kx, ky, X, Y, n_perms=500, key=key)
g = kl.hsic_input_gradient(kx, ky, X, Y)                   # jax.grad through pairwise

# randomized: the feature map owns the randomness
h = kl.hsic(kx, ky, X, Y, approx=kl.NystromFeatures(n_components=300, key=key))
h = kl.hsic(kx, ky, X, Y, approx=kl.RandomFourierFeatures(n_features=1024, key=key))
h = kl.hsic(kx, ky, X, Y, approx=kl.FastFoodFeatures(n_features=1024, key=key))
# internally: functional.hsic(approx.fit(kx, X).operator(X), approx.fit(ky, Y).operator(Y))
# both operands are LowRankUpdate; gaussx.trace_product uses ||U^T V||_F^2
```

### Approximation

A feature map is built from a kernel and a key. It can give you the
feature matrix, or a gaussx `LowRankUpdate` that already solves and takes
logdets through Woodbury.

```python
# operator level: arrays in, LowRankUpdate out
K_low = kl.nystrom_operator(K_XZ, K_ZZ_op)          # moved from gaussx
K_low = kl.rff_operator(X, omega, b)                # moved from gaussx
K_low = kl.fastfood_operator(X, B, Pi, G, S)        # new; matvec via kl.FastFoodOperator

# kernel level
k = kl.Matern(nu=1.5, lengthscale=0.7)

nys = kl.NystromFeatures(n_components=500, key=key, selection="uniform").fit(k, X)
phi = nys(X_test)                     # (N, 500) features  K_xz L_zz^{-T}
K_low = nys.operator(X)               # kl.nystrom_operator under the hood
nys.landmarks                         # the Z that was chosen

rff = kl.RandomFourierFeatures(n_features=2048, key=key).fit(k)    # omega ~ k.sample_frequencies
orf = kl.OrthogonalRandomFeatures(n_features=2048, key=key).fit(k) # geonnax.randfeat.orthogonal_blocks
ff  = kl.FastFoodFeatures(n_features=2048, key=key).fit(k)         # B, Pi, G drawn; S from k's radial density
lap = kl.LaplaceEigenfunctionFeatures(bounds=(-L, L), n_per_dim=32).fit(k)   # geonnax.basis.fourier_basis
phi = rff(X)
K_low = ff.operator(X)

# any of them plugs into gaussx directly
alpha = gx.solve(K_low + 1e-2 * gx.identity(N), y)   # Woodbury, O(N m^2)
```

### Regression

The primitives and the estimators live side by side; the estimator adds
landmark selection, operator construction from the kernel, the training loop
and `predict`.

```python
# primitive level (moved from gaussx)
P = kl.falkon_preconditioner(K_mm, regularization=lam)
alpha = kl.falkon_solve(K_nm_op, y, P, regularization=lam, max_iter=20)
y_hat = kl.falkon_predict(kernel_fn, Z, alpha, X_test)

# estimator level
model = kl.KRR(k, regularization=1e-2, solver=gx.AutoSolver()).fit(X, y)
model = kl.Falkon(k, n_inducing=2000, regularization=1e-3, max_iter=20).fit(X, y, key=key)
model = kl.EigenPro(k, epochs=10, batch_size=512, subsample_size=4000, n_components=100).fit(X, y, key=key)

y_hat = model.predict(X_test)
dy = kl.predictor_gradient(model, X_test)          # ∂f̂/∂x via pairwise
model.alpha, model.landmarks                        # fitted state is plain fields

# the fitted model is a PyTree, so hyperparameter gradients just work
loss = jax.grad(lambda ell: kl.KRR(kl.RBF(ell), 1e-2).fit(X, y).loss(X_val, y_val))(1.0)
```

### Derivatives and decomposition

```python
J = kl.kernel_jacobian(k, x, y)                 # ∂k(x, y)/∂x
dK = kl.derivative_gram(k, X1, X2)              # (N1, N2, D)
emb = kl.kernel_pca(k, X, n_components=10, solver=gx.DenseSolver())
Kg = kl.diffusion_kernel(adjacency, beta=0.5)
```

### pyrox-gp on top

Unchanged shape: it freezes its parameterized kernel once inside the
context and hands a plain kernellib kernel down.

```python
k_frozen = gp_kernel.frozen()                              # inside _kernel_context
rff = kl.RandomFourierFeatures(1024, key).fit(k_frozen)    # pathwise prior draws
```

---

## Migration plan

Each phase is one or more PRs, independently shippable, with no user-facing
breakage. Repo names in bold. The gaussx removal comes *after* kernellib's
first release so there is never a window in which the kernel operators exist
nowhere.

### Phase 0: scaffold (**kernellib**)

1. Tag the current `master` as `v0.1.0-legacy` and keep it on a `legacy`
   branch. The old code is not carried forward.
2. Rename the default branch to `main` to match gaussx and pyrox
   (decision for the owner; the rest of the plan assumes `main`).
3. Fresh tree scaffolded from `pypackage_template`: `src/` layout with
   hatchling, `uv` groups (`dev`, `lint`, `typecheck`, `docs`), ruff, `ty`,
   pytest with doctests and the coverage gate plus `slow` / `integration`
   markers, pre-commit, release-please with manifest, the template's CI
   workflows, the two-tool docs pipeline, `CLAUDE.md` / `AGENTS.md` /
   `CODE_REVIEW.md`, `Makefile`.
4. Version `0.0.1`. The `kernellib` name is free on PyPI (checked
   2026-09-25).
5. Import-guard test: `import kernellib` must not put `numpyro` or `sklearn`
   in `sys.modules`.

### Phase 1: kernels, operators and the bridge (**kernellib**)

- `functional/`: copy `pyrox_gp._src.kernels` and its tests verbatim, then
  route distances through `gaussx.stable_squared_distances`; add
  `stable_rbf_kernel` (moved) and `_statistics.py` (`center_kernel`,
  `centering_operator`, `hsic`, `mmd_squared` moved from gaussx, plus the
  unbiased estimator and `cka`).
- `_operators/`: copy `KernelOperator`, `ImplicitKernelOperator`,
  `ImplicitCrossKernelOperator`, `batched_kernel_matvec`, `nystrom_operator`,
  `rff_operator` from gaussx with their tests; copy the two private helpers.
- `_kernels/`: the three abstract classes, the concrete kernels, composition.
- `_operators/_bridge.py`: `to_operator`, `to_cross_operator`.
- Tests: see the testing section. Includes one integration test that solves
  a kernel system through `gaussx.CGSolver` via the implicit bridge and
  matches the dense solve.
- Release `kernellib 0.1.0`.

### Phase 2: pyrox-gp consumes kernellib (**pyrox**)

- Add `kernellib>=0.1.0` to `packages/pyrox-gp/pyproject.toml`.
- `pyrox_gp._src.kernels` becomes re-exports with a `DeprecationWarning` on
  import; its tests move to kernellib (done in phase 1) and are deleted here.
- `pyrox_gp._protocols.Kernel = kernellib.AbstractKernel`.
- Replace the two docstring references to `gaussx.ImplicitKernelOperator`
  and `gaussx.stable_rbf_kernel` with their kernellib names.
- Full pyrox test suite green with no other change. This is the proof the
  alias is safe.

### Phase 3: remove the kernel layer from gaussx (**gaussx**)

- Delete `_kernels/` except `_grid.py`, the three kernel-operator files,
  `batched_kernel_matvec`, `stable_rbf_kernel`, their tests, their
  `__init__` exports, and `docs/api/kernels.md`. Keep
  `stable_squared_distances`, the grid helpers and the preconditioners.
- Add the `LowRankUpdate × LowRankUpdate` fast path to `trace_product` with
  a test against the dense value.
- Update `docs/vision.md` and `docs/architecture.md` (kernellib node, "go
  here instead" row, `_kernels/` row removed).
- Release **gaussx 0.2.0**; CHANGELOG names the new home of every removed
  symbol. kernellib raises its pin to `gaussx>=0.2.0` in phase 4.

### Phase 4: spectral side (**kernellib**, then **pyrox**)

- kernellib: `spectral_density` and `sample_frequencies` on every stationary
  kernel; `_spectral/_rff.py` with the moved draw / evaluate helpers;
  `AbstractFeatureMap` with `fit` / `__call__` / `operator`;
  `RandomFourierFeatures`, `OrthogonalRandomFeatures` wrappers over geonnax;
  `NystromFeatures`; `LaplaceEigenfunctionFeatures`; `FastFoodOperator`,
  `fastfood_operator` and `FastFoodFeatures`.
- pyrox: `frozen()` on `_ParameterizedKernel`; `_basis._spectral_density`
  and `_basis._rff` delegate. pyrox-nn tests green with no change.
- Release `kernellib 0.2.0`, `pyrox-gp` patch.

### Phase 5: algorithms (**kernellib**)

In order of value, each its own PR:

1. `_heuristics.py` (pysim port) — small, needed by everything below.
2. `_regression/_krr.py` and the `AbstractEstimator` base.
3. `_dependence/_hsic.py`, `_mmd.py`, `_permutation.py` (pysim port, plus
   gradients); the randomized path over the phase 4 feature maps.
4. `_regression/_falkon.py`, `_eigenpro.py`: the moved primitives plus the
   `Falkon` and `EigenPro` estimators.
5. `_derivatives.py`.
6. `_decomposition/_kpca.py`, `_graph.py`.

Release `kernellib 0.3.0`.

### Phase 6: documentation (**pyrox**)

- pyrox `design_docs/pyrox/boundaries.md`: ownership rows for kernels move
  from `pyrox.gp` to kernellib.
- No code changes.

---

## Testing strategy

Following the gaussx rules: unmarked tests under a second, `slow` for
sweeps, pinned keys when randomness is incidental, sampling bounds stated
in a comment when it is not.

Per concrete kernel (parametrised over the kernel zoo):

- Symmetry: `k(X1, X2) == k(X2, X1).T`.
- PSD: smallest eigenvalue of `k.gram(X)` ≥ `-tol`.
- `diag(X)` equals `jnp.diag(gram(X))`.
- Pointwise-vs-Gram: `vmap²(pairwise)` equals the overridden `__call__`.
- Stationary: shift invariance; `spectral_density` integrates to
  `variance` (Bochner) on a grid; an RFF Gram with many features converges
  to the exact Gram (`slow`).
- `jax.grad` through hyperparameters is finite; `jit` and `vmap` over a
  batch of inputs both work.

Bridge:

- `to_operator(implicit=True)` matvec equals dense Gram matvec.
- CG solve through the implicit operator matches dense solve.
- Gradient of `solve(to_operator(k, X), y)` w.r.t. `k.lengthscale` matches
  the dense gradient (exercises the custom JVP with kernellib params).

Algorithms:

- KRR matches the closed-form solution for small `n`, for every gaussx
  solver strategy.
- Falkon and EigenPro converge to KRR on small problems (`slow`).
- `hsic` on dense kernels equals `functional.hsic` on the same matrices;
  randomized HSIC converges to dense as `m → n`; CKA of a matrix with itself
  is one; unbiased HSIC of independent draws has mean zero within its
  sampling bound.
- `kernel_jacobian` matches finite differences; `derivative_gram` matches
  the closed-form RBF derivative.
- Every reimplemented pysim function is compared against a numpy
  reference implementation written inline in the test (not against pysim,
  which is not a dependency).

Moved code:

- gaussx's kernel tests move verbatim with the code (`tests/kernels/*`,
  the kernel-operator and batched-matvec tests, the `stable_rbf_kernel`
  tests) and keep passing before anything else is touched.
- Unbiased HSIC matches the Song et al. formula written out densely; `cka`
  of a matrix with itself is one and is scale invariant.
- `center_kernel` of a `LowRankUpdate` stays a `LowRankUpdate` and its
  `as_matrix()` equals `H K H`.
- `FastFoodOperator` matvec equals the dense `S H G Pi H B` product;
  `fastfood_operator` Gram agrees with `rff_operator` Gram in expectation
  for the RBF kernel (`slow`, sampling bound stated in the test).

gaussx (its own suite): `trace_product` on two `LowRankUpdate` operands
equals the dense value.

---

## Versioning and release

- kernellib: semver `0.x` via release-please, conventional commits, same as
  gaussx. Public API is only what `kernellib/__init__.py` exports.
- Pins: kernellib `gaussx>=0.1.0`, raised to `>=0.2.0` once the kernel
  layer is removed there; pyrox-gp `kernellib>=0.1.0` (then `>=0.2.0` after
  phase 4). Never upper-bound within `0.x` unless a break is
  known.
- The pin cascade (gaussx → kernellib → pyrox-gp → pyrox-nn) is the main
  ongoing cost of a separate package. Mitigations: keep kernellib's public
  surface small; land breaking changes in kernellib behind a deprecation
  cycle so pyrox-gp can update on its own cadence; run pyrox's test suite
  against kernellib `main` in a weekly extended workflow.

---

## Risks and open questions

| # | Question | Proposed answer |
|---|---|---|
| 1 | PyPI name `kernellib` may be taken | Check before phase 1's release; fallback `kernellib-jax` |
| 2 | Default branch `master` → `main` | Yes, in phase 0 |
| 3 | Where do positivity constraints live? A plain optax user needs softplus somewhere | Not in kernellib v0. Document the pattern (`eqx.tree_at` + `jax.nn.softplus` in the user's loss). Revisit a `kernellib.transforms` module if it is requested |
| 4 | Multi-output kernels (LMC, ICM, OILMM) | Stay in pyrox-gp. They are Gram-only `AbstractKernel` subclasses, so they still type-check against kernellib; moving them is a later decision |
| 5 | `Matern` with general `nu` vs the three closed forms | Static `nu ∈ {0.5, 1.5, 2.5}` as pyrox-gp does today; general `nu` via Bessel is a later addition |
| 6 | float64 assumptions in stable Gram paths | Same as gaussx: accumulate in float64 when x64 is enabled, otherwise degrade gracefully; test both |
| 7 | `to_operator(implicit=...)` default | Explicit `False`. No size heuristic; the caller knows their `N` |
| 8 | pyrox-gp `frozen()` inside NumPyro tracing | Params are resolved through the existing per-call context, so a `frozen()` call inside `_kernel_context` sees one draw. Needs a regression test in pyrox-gp under `handlers.trace` and `handlers.seed` |
| 9 | Manifold learning (eigenmaps, LPP, Schrödinger) from old kernellib | Deferred; not in scope for 0.1–0.3 |
| 10 | gaussx 0.2.0 removes public symbols with no shims | Accepted: no external users, pyrox-gp does not use them, and a shim would need the forbidden gaussx → kernellib import. The CHANGELOG maps every symbol to its new name |
| 11 | FastFood scaling `S` for non-RBF kernels | RBF: `s_i ~ χ(d)` scaled by `‖G‖_F⁻¹`. Other stationary kernels: draw `s_i` from the kernel's radial spectral distribution via `sample_frequencies` norms. Documented per kernel; tested by RFF-vs-FastFood Gram agreement |
| 12 | Should `stable_squared_distances` follow the kernels? | No (option 1): it is a distance primitive with a non-kernel consumer inside gaussx; a private copy or moving localization were both worse |

---

## Decisions log

| Date | Decision |
|---|---|
| 2026-09-24 | kernellib is a standalone repo and package, not a fourth workspace member in pyrox, so non-Bayesian users get it without NumPyro |
| 2026-09-24 | Old kernellib code is retired to a `legacy` branch; nothing is ported line by line |
| 2026-09-24 | ~~gaussx keeps all Falkon / EigenPro / HSIC / MMD / Nyström / RFF *operations*; kernellib owns the *algorithms*~~ Superseded 2026-09-25 |
| 2026-09-24 | ~~Placement test: needs a kernel object or a random key → kernellib; matrices / operators in, matrices / operators out → gaussx~~ Superseded 2026-09-25 |
| 2026-09-24 | ~~gaussx gains four additions (unbiased HSIC, CKA, low-rank centering, FastFood)~~ Superseded 2026-09-25: all four are kernellib work except the `trace_product` fast path |
| 2026-09-24 | kernellib API conventions: dependence measures are plain functions; estimators and feature maps are immutable eqx modules with config in the constructor, `fit` returning a new module, and `operator(X)` returning a gaussx `LowRankUpdate` |
| 2026-09-24 | Randomized HSIC / CKA / MMD reuse the matrix-level `hsic` on `LowRankUpdate` operands rather than a separate feature-matrix code path |
| 2026-09-24 | pysim's HSIC family and bandwidth heuristics are reimplemented in kernellib; pysim itself is not changed |
| 2026-09-24 | geonnax is a dependency for basis functions and random-feature primitives; kernellib does not duplicate them |
| 2026-09-24 | Kernel contract: `AbstractKernel` (Gram-only, abstract `__call__`) ⊃ `AbstractPointwiseKernel` (abstract `pairwise`) ⊃ `AbstractStationaryKernel` (shape + spectral density). `pyrox_gp.Kernel` becomes an alias of the first |
| 2026-09-24 | kernellib never imports numpyro; gaussx never imports kernellib |
| 2026-09-25 | The whole kernel layer of gaussx moves to kernellib: kernel operators, Nyström / RFF operators, matrix-level HSIC / MMD / centering, Falkon and EigenPro primitives, `stable_rbf_kernel`, batched kernel matvecs. Rule: anything with a kernel in it lives in kernellib, from the operators up |
| 2026-09-25 | gaussx 0.2.0 removes those symbols outright, no deprecation shims. The only gaussx additions are the low-rank `trace_product` fast path and docs |
| 2026-09-25 | `stable_squared_distances` stays in gaussx (option 1); kernellib wraps it |
| 2026-09-25 | Matrix-level functions live in `kernellib.functional`; the same names at the top level take kernels and data |
| 2026-09-25 | Phase order: kernellib 0.1.0 (with the moved code) → pyrox-gp consumes → gaussx 0.2.0 removal → spectral → algorithms |
