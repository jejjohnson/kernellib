---
date: 2026-09-30
---

# pyrox: roadmap

pyrox's share of the [fused roadmap](roadmap.md). Two workspace members
change:

- **pyrox-gp** gets graph Matérn inducing features, latent and
  inducing-point initialisation, and a large-`n` exact-GP recipe.
- **pyrox-lgm** is a new member (`packages/pyrox-lgm`) for latent Gaussian
  models: components, PC priors and `inla()`. It includes the CAR / ICAR /
  Leroux / BYM2 spatial priors.

pyrox never implements linear algebra or kernels. It picks defaults, adds
hyperpriors and NumPyro sites, and composes kernellib and gaussx. It serves
the [manifold](project-manifold.md), [RandNLA](project-rnla.md) and
[INLA](project-inla.md) projects. The baseline is pyrox-gp 0.1.7,
which pins kernellib v0.0.5 and `gaussx>=0.2.0`.

> Maths notes (**The maths.**) say where each operation comes from; **Example.** blocks are pseudocode against the *planned* API (`gx` = gaussx, `kl` = kernellib, `px` = pyrox-gp, `lgm` = pyrox-lgm). End-to-end problems are in the [examples gallery](roadmap-examples.md).

| Phase | Package | What | Projects | Needs |
|---|---|---|---|---|
| P1 | pyrox-gp | `latent_init` (replaces the missing `pca_init`) | manifold | kernellib pin ≥ v0.0.8 |
| P2 | pyrox-gp | Graph Matérn `LaplacianInducingFeatures` | manifold, INLA | K3, K4 |
| P3 | pyrox-gp | `init_inducing` | RandNLA | K8 |
| P4 | pyrox-gp | Large-`n` exact-GP recipe (preconditioned CG + SLQ) | RandNLA | G13, and gaussx#312 fixed |
| P5 | pyrox-gp | LogFalkon centres (pyrox#50) | RandNLA | K8 |
| P6 | pyrox-lgm | Scaffold `packages/pyrox-lgm` | INLA | — |
| P7 | pyrox-lgm | Components with a NumPyro face (including the spatial priors), PC priors | INLA, manifold | G6, G7, K6 |
| P8 | pyrox-lgm | `LGM`, `inla()`, `INLAResult` | INLA | G8–G10, P7 |
| P9 | pyrox-lgm | `f(...)` sugar, diagnostics, simplified Laplace, MCMC-INLA hybrid | INLA | P8 |
| P10 | repo | `boundaries.md`, `spde_fem.md` retarget, pyrox#50, dependency pins | all | — |

**Order:**

1. **P1 and P6 first:** both are small and unblocked. P1 removes a
   phantom function from the design docs.
2. **P2, P3, P5** as their kernellib phases are tagged.
3. **P7 → P8 → P9** follow gaussx's Part A.
4. **P4 waits on gaussx#312.**
5. **P10** is updated alongside each phase.

---

## 1. Current state

### Graphs and latent variables (pyrox-gp)

| Where | What | Issue |
|---|---|---|
| `_inducing.py:497` `LaplacianInducingFeatures` | `fit(adjacency, num_basis, normalized=True)` calls geonnax's dense `graph_laplacian_eigpairs`; `K_uu` / `k_ux` use `spectral_density(kernel, eigvals, D=1)` | See below |
| `_basis/_spectral_density.py` | `spectral_density` evaluates the Euclidean density at `√λ` | Correct for HSGP on a box, but only approximately right on a graph |
| `design_docs/pyrox/features/gp/models.md:378,430` | GPLVM (Gap 7) and Bayesian GPLVM (Gap 8) demos call `pca_init(Y, Q)` | `pca_init` does not exist anywhere |
| `_latent_factor_models.py:22-30` | `Z_T` is initialised with `init_to_sample` to avoid the saddle at zero | A data-driven start would converge faster and more reproducibly |

**What the graph inducing features compute today:**

- **RBF:** `S(√λ) = σ² √(2π) ℓ · exp(−ℓ²λ/2)`. That is the graph heat
  (diffusion) spectrum with the right shape, but a Euclidean prefactor.
  The variance is therefore not the average marginal variance on the graph.
- **Matérn:** `S(√λ) ∝ (2ν/ℓ² + λ)^−(ν + 1/2)`. The exponent comes from
  the Euclidean `d = 1` density. Borovitskiy et al.'s graph Matérn uses
  `(2ν/ℓ² + λ)^−ν`, so today's features are a graph Matérn with smoothness
  `ν + ½`, again with a Euclidean prefactor.
- The class docstring says only the heat family is supported, but
  `_check_stationary` accepts Matérn anyway.
- Eigenpairs are dense `O(V³)`, so graphs of more than a few thousand
  nodes are out of reach.

### Solvers and inducing points (pyrox-gp)

- **Inducing points.** The library never initialises them: `SparseGPPrior`
  takes a user `Z` or `InducingFeatures` (`_sparse.py:84-115`). There is no
  k-means or random-choice helper.
- **Solvers.** Every model takes `solver: AbstractSolverStrategy | None`
  and defaults to `DenseSolver` (`_models.py:80-91`, `_sparse.py:87-99`,
  `_sparse_markov.py`, `_multi_output_models.py`). Docstrings mention
  `CGSolver`, `BBMMSolver` and `ComposedSolver`, but nothing constructs a
  preconditioner.
- **Design docs** already describe the target:
  - `api/gp/moments.md:228-295` has CG with a Hutchinson + Lanczos
    log-determinant, and BBMM "preconditioning: pivoted Cholesky partial
    factorization rank-r";
  - `features/gp/logfalkon.md:26,59` chooses centres "uniformly at random
    or via leverage-score".
- **The pin** is kernellib v0.0.5 (git tag), which predates everything
  here.

### Laplace-type inference and latent Gaussian models

- **Laplace-type inference exists, but only in covariance form.**
  - pyrox-gp's `LaplaceInference`, `GaussNewtonInference`,
    `PosteriorLinearization`, `ExpectationPropagation` and
    `QuasiNewtonInference` (`_inference_nongauss.py:363-758`) factor a
    dense `K`.
  - The `*Markov` versions (`_inference_nongauss_markov.py`) run Kalman /
    RTS smoothing.
  - All are scalar-latent only (`_check_scalar_latent`, `:182`), and all
    point-estimate the hyperparameters. None integrates over θ.
- **`design_docs/pyrox/features/gp/spde_fem.md`** (draft, 2026-04-03)
  puts FEM assembly (`pyrox.gp._src.fem`), an `SPDESolver` and
  `spde_gp_factor` in pyrox-gp, with CHOLMOD. Nothing of it is
  implemented.
- **Missing everywhere:** GMRF components, sum-to-zero constraints, PC
  priors, θ-integration, the VB correction, an LGM notion, and summaries.
- **Issues:** pyrox#50 (SPDE-FEM among other integrations), pyrox#43 (the
  wave-5 epic).

---

## 2. pyrox-gp

### 2.1 P1: `latent_init`

**The maths.** **Probabilistic PCA**, $y = Wx+\varepsilon$ with
$x\sim\mathcal N(0,I_Q)$, has the maximum-likelihood solution
$W = U_Q(\Lambda_Q-\sigma^2I)^{1/2}R$ (Tipping & Bishop, 1999).

**Dually**, a GPLVM with a linear kernel, $K = XX^\top + \sigma^2 I$, has
maximum-likelihood latents spanned by the top-$Q$ principal components of
$Y$ (Lawrence, 2005). So PCA is the exact optimum of the linear-kernel
GPLVM, which makes it the natural start for a nonlinear one. Kernel PCA
and Laplacian eigenmaps start it on the data manifold instead, which helps
when the manifold is curled (swiss rolls, periodic motion).

The current pin, v0.0.5, predates the eigenmaps. `KernelPCA` and
`LaplacianEigenmaps` first shipped in v0.0.8, so bump the pin to at least
that tag (the latest is v0.0.11).

`pyrox_gp/_latent_init.py`, exported at the top level:

```python
def latent_init(
    Y: Float[Array, "N D"],
    n_latent: int,
    *,
    method: Literal["pca", "kernel_pca", "laplacian_eigenmaps"] = "pca",
    kernel: kl.AbstractKernel | None = None,  # for "kernel_pca"
    n_neighbors: int = 10,  # for "laplacian_eigenmaps"
    standardize: bool = True,
) -> Float[Array, "N Q"]: ...
```

| `method` | How |
|---|---|
| `"pca"` | SVD of the centred `Y` (einx for the centring). No kernellib call is needed |
| `"kernel_pca"` | `kl.KernelPCA(kernel, n_components=n_latent).fit(Y).embedding` |
| `"laplacian_eigenmaps"` | `kl.LaplacianEigenmaps(n_components=n_latent, n_neighbors=n_neighbors).fit(Y).embedding` |

- **`standardize=True`** scales each column to unit variance, which
  matches the GPLVM prior `X ~ N(0, I_Q)`. Eigenmap coordinates otherwise
  have degree-weighted scales that are far from 1.
- **Sign and rotation.** The result is deterministic up to kernellib's
  eigenvector sign convention, so no key is needed.

**Example.**

```python
def gplvm(Y):
    N, D = Y.shape
    X = numpyro.param(
        "X", px.latent_init(Y, 2, method="laplacian_eigenmaps", n_neighbors=15)
    )
    kernel = px.RBF(
        variance=numpyro.param("var", 1.0), lengthscale=numpyro.param("ls", jnp.ones(2))
    )
    prior = px.GPPrior(kernel=kernel, solver=gx.CholeskySolver(), X=X)
    for d in range(D):
        px.gp_factor(f"y_{d}", prior, Y[:, d], numpyro.param("noise", 0.1))
```

#### Wiring

- `models.md` Gaps 7 and 8: replace `pca_init(Y, Q)` with
  `latent_init(Y, Q)`.
- `_latent_factor_models.py`: add an optional `init_latents` argument to
  the init-strategy helper. When it is given, `Z_T` uses
  `init_to_value(values={"Z_T": init_latents.T})`, because the site is
  stored transposed as `(Q, N)`. Otherwise the current `init_to_sample`
  behaviour stays.

#### Tests

- Linear-Gaussian data `Y = X Wᵀ + ε`: the `"pca"` initialisation spans
  the true latent subspace (principal angles below a tolerance).
- Every method returns `(N, Q)` with unit column variance when
  `standardize=True`.
- An LFR fit started from `latent_init` reaches at least the ELBO of
  `init_to_sample` for the same step budget (slow tier, fixed key).

#### Later (optional, needs gaussx G6)

A graph-regularised GPLVM prior, `p(X) ∝ exp(−(α/2) tr(Xᵀ L X))`,
with `L` the Laplacian of `kl.knn_graph(Y)`. It is an `IntrinsicGMRF`
per latent column, plus the `N(0, I)` prior to fix the null space. Its
MAP limit ties Laplacian / Schrödinger eigenmaps to the GPLVM. Put it in
a design note first; don't build it before GPLVM (Gap 7) exists.


### 2.2 P2: graph Matérn inducing features

**The maths.** **Inter-domain features on a graph.** For
$f\sim\mathcal{GP}\big(0,\ U\Phi(\Lambda)U^\top\big)$ on the nodes, take
the projections onto Laplacian eigenvectors, $u_k = \phi_k^\top f$. Then

$$
\operatorname{Cov}(u_k,u_l) = \Phi(\lambda_k)\,\delta_{kl},\qquad
\operatorname{Cov}\big(u_k, f(v)\big) = \Phi(\lambda_k)\,\phi_k(v).
$$

$K_{uu}$ is diagonal, so an SVGP with $M$ such features costs $O(NM)$
per step, with no $M\times M$ Cholesky. The whole fix in this phase is to
use the graph's own $\Phi$ (heat or Matérn, kernellib K4) instead of a
Euclidean spectral density evaluated at $\sqrt\lambda$.

#### Changes to `LaplacianInducingFeatures`

```python
class LaplacianInducingFeatures(eqx.Module):
    eigvals: Float[Array, " M"]
    eigvecs: Float[Array, "V M"]
    n_nodes: int = eqx.field(static=True)  # NEW: for variance normalisation

    @classmethod
    def fit(
        cls,
        graph: kl.AbstractGraph | Float[Array, "V V"],
        num_basis: int,
        *,
        normalization: Literal["unnormalized", "symmetric"] = "symmetric",
        method: Literal["dense", "kronecker", "lanczos", "arpack"] | None = None,
        key: PRNGKeyArray | None = None,
        normalized: bool | None = None,  # deprecated alias
    ) -> LaplacianInducingFeatures: ...
```

- **Eigenpairs** come from `kernellib.laplacian_eigpairs`, so graphs (not
  just adjacency arrays), grid graphs with exact Kronecker eigenpairs, and
  Lanczos / ARPACK for large graphs all work. A dense adjacency array is
  still accepted.
- **Deprecated alias.** `normalized=True/False` maps to
  `"symmetric"` / `"unnormalized"` and warns. Remove it in 0.3.
- **Graph spectra.** A private `_graph_spectrum(kernel, eigvals, n_nodes)`
  replaces the `spectral_density` call in `K_uu` and `k_ux`:
  - `RBF` (or kernellib `RBF`) uses `kl.functional.graph_heat_spectrum`;
  - `Matern(nu)` uses `kl.functional.graph_matern_spectrum(nu=nu)`;
  - both normalise to average marginal variance `σ²` over `n_nodes`;
  - anything else raises `NotImplementedError` naming the two supported
    kernels.
- `spectral_density` itself is unchanged, because HSGP and VFF still need
  the Euclidean density.
- **Behaviour change.** `K_uu` changes scale for RBF, and shape and scale
  for Matérn. Release it as a `fix:` with a CHANGELOG entry spelling out
  both.

**Example.**

```python
# SVGP over 50k traffic sensors: X is a vector of node indices, not coordinates
sensors = kl.knn_graph(sensor_xy, 8)
feats = px.LaplacianInducingFeatures.fit(sensors, 256, method="lanczos", key=key)
prior = px.SparseGPPrior(px.Matern(nu=1.5, lengthscale=2.0), inducing=feats)
```

#### Tests (`tests/basis/test_laplacian.py`)

- With `num_basis = V` (the full basis), `k_ux K_uu⁻¹ k_uxᵀ` over all
  nodes equals `kernellib.matern_graph_kernel` / `diffusion_kernel`
  (integration tier: cross-library).
- The Kronecker method on a `grid_graph` matches the dense method.
- The average of `diag(k_ux K_uu⁻¹ k_uxᵀ)` equals `σ²` at `num_basis = V`.
- `normalized=` warns and maps correctly.

#### Docs

- `design_docs/pyrox/features/gp/inducing_features.md`, Gap 3: mark it
  implemented and point to the kernellib functions. Its spectral density
  formula uses the Euclidean `ν + d/2` exponent; correct it to the graph
  form.


### 2.3 P3: `init_inducing`

**The maths.** Choosing inducing inputs $Z$ from $X$ by **greedy conditional variance**
is pivoted Cholesky on $K_{ff}$. Burt, Rasmussen & van der Wilk (2019,
2020) bound the gap between the SVGP ELBO and the log marginal
likelihood by a quantity that grows with
$t = \operatorname{tr}(K_{ff} - Q_{ff})$, the Nyström trace error.
Choosing $Z$ to make $t$ small, which is RPCholesky's objective (gaussx
G14), therefore directly controls how loose the ELBO can be.

`pyrox_gp/_inducing_init.py`, exported at the top level:

```python
def init_inducing(
    X: Float[Array, "N D"],
    n_inducing: int,
    *,
    kernel: Kernel | None,  # required keyword; None only with method="uniform"
    method: Literal["uniform", "rpcholesky", "greedy", "leverage"] = "rpcholesky",
    key: PRNGKeyArray,
) -> Float[Array, "M D"]:
    """Inducing inputs chosen from X."""
```

- It calls `kernellib.select_landmarks` on the kernel frozen at its
  current parameters (`kernel.frozen()`, inside the kernel context), and
  returns `X[indices]`.
- **`kernel`** is a required keyword with no default, so the default
  call `init_inducing(X, M, kernel=k, key=key)` is always valid.
  `kernel=None` is accepted only with `method="uniform"`; any other
  method raises.
- **The default is `"rpcholesky"`.** It is kernel-aware and
  parameter-free, and at initialisation it is as good as or better than
  k-means for sparse variational GPs.
  - `"greedy"` is Burt, Rasmussen & van der Wilk's (2020)
    conditional-variance rule, with their convergence argument.
  - `"rpcholesky"` is its randomized counterpart. It avoids greedy's
    tendency to pick outliers on heavy-tailed inputs.
- **Documentation.** Selection happens once, before training, from a
  fixed kernel. If the hyperparameters move a lot during training, the
  user can call it again, and the docstring shows the pattern.

**Wiring:**

- the SVGP examples in the docs switch from hand-sliced `X[:M]` to
  `init_inducing`;
- `design_docs/pyrox/features/gp/` gains a short note in the sparse-GP
  page.

**Tests:**

- the output has shape `(M, D)`, and every row is a row of `X`;
- `"greedy"` is deterministic;
- the SVGP ELBO after a fixed number of steps from `"rpcholesky"` is at
  least that from `"uniform"` on a clustered 2-D problem (slow tier, fixed
  keys).


**Example.**

```python
Z = px.init_inducing(X, 512, kernel=kernel, method="rpcholesky", key=key)
prior = px.SparseGPPrior(kernel, Z=Z)
```

### 2.4 P4: large-`n` exact-GP recipe

**The maths.** With $\hat K = K+\sigma^2I$ and $\alpha = \hat K^{-1}y$:

$$
\log p(y) = -\tfrac12\,y^\top\alpha - \tfrac12\log|\hat K| - \tfrac n2\log 2\pi,
\qquad
\partial_\theta\log p(y) = \tfrac12\,\alpha^\top(\partial_\theta\hat K)\,\alpha - \tfrac12\operatorname{tr}\big(\hat K^{-1}\partial_\theta\hat K\big).
$$

- the solve for $\alpha$ is preconditioned CG (G13);
- the trace is Hutchinson,
  $\frac1s\sum_i z_i^\top\hat K^{-1}\partial_\theta\hat K z_i$, reusing the
  same batched solves;
- the log-determinant is SLQ.

Everything is matvec-bound, at $O(n^2)$ per matvec with an implicit
kernel operator and $O(n)$ memory.

This is a documented recipe plus a small constructor helper, not a new
model:

```python
def preconditioned_cg_solver(
    *,
    preconditioner: Literal["nystrom", "rpcholesky"] = "nystrom",
    rank: int = 200,
    logdet: Literal["slq", "nystrom"] = "slq",
    key: PRNGKeyArray,
) -> gx.AbstractSolverStrategy:
    """A gaussx solver strategy for exact GPs at n ≈ 10⁵–10⁶."""
```

- **The noise term.** The exact-GP model already builds `K + σ²I` as a
  sum. The helper returns a strategy that builds the preconditioner from
  `K` with `shift=σ²` (the #345 rule, [G13](roadmap-gaussx.md)).
- **`logdet="nystrom"`** uses gaussx's tier-2 `NystromLogdet` (Wenger et
  al., 2022) once it exists. Until then only `"slq"` is accepted.
- **Blocker.** Hyperparameter learning needs gradients through the
  preconditioned solve, which raises today (gaussx#312, under epic #283).
  P4 does not start until #312 is closed. The preconditioner itself is
  `stop_gradient`ed and needs no derivative.
- **Tests (integration tier):**
  - on `n = 20 000` with a Matérn-3/2 kernel, the marginal likelihood and
    its gradient match the dense solver at `n = 3000` (a subset) to SLQ
    tolerance;
  - the CG iteration count stays below a fixed budget as `n` grows.
- **Docs:** update `api/gp/moments.md` to point at this recipe in place
  of its pivoted-Cholesky-only BBMM description.


**Example.**

```python
# 10⁵ weather stations, Matérn-3/2 plus noise; hyperparameters by gradient ascent
solver = px.preconditioned_cg_solver(preconditioner="nystrom", rank=500, key=key)
prior = px.GPPrior(kernel=px.Matern(nu=1.5), X=stations_xyz, solver=solver)
# inside the model: px.gp_factor("temp", prior, temp_obs, noise_var); fit with SVI, whose gradients use CG
```

### 2.5 P5: LogFalkon centres

LogFalkon's design (`features/gp/logfalkon.md`) chooses centres "uniformly
or via leverage score". When it is built, it takes a `centers` argument
with the same four methods and delegates to `kernellib.select_landmarks`.
This is a line in that feature's spec, not separate work.


---

## 3. pyrox-lgm

### 3.1 P6: package layout

```
packages/pyrox-lgm/                      # NEW workspace member (root pyproject already globs packages/*)
├── pyproject.toml                        # deps: pyrox, gaussx[numpyro], kernellib, numpyro, optax; extra: xarray
└── src/pyrox_lgm/
    ├── _components/
    │   ├── _base.py                      # AbstractComponent
    │   ├── _temporal.py                  # IID, RW1, RW2, AR1
    │   ├── _areal.py                     # Besag, BYM2, CAR, Leroux        (the spatial priors)
    │   ├── _spde.py                      # SPDE (mesh or grid)
    │   ├── _generic.py                   # Generic(structure) — the rgeneric escape hatch, without the leak
    │   └── _combinators.py               # Kronecker(a, b) (space-time "group"), Replicate
    ├── _priors/_pc.py                    # PCPrecision, PCMatern, PCBYM2Phi, PCAR1Rho
    ├── _model.py                         # FixedEffects, LGM
    ├── _inla.py                          # inla()
    ├── _result.py                        # INLAResult, marginals, sampling, predict
    ├── _numpyro.py                       # the NumPyro face: component.sample(), lgm_model() for NUTS
    ├── _diagnostics.py                   # P9: DIC, WAIC, CPO, PIT
    └── _formula.py                       # P9: f(...) sugar
```

**Why a separate package** (#155 open question 2): LGMs are a strict
superset of GPs. The package needs gaussx and kernellib but nothing from
pyrox-gp, and keeping it separate keeps pyrox-gp's GP-centred API
unchanged. The package boundary is enforced by the root
`pyproject.toml`'s workspace, with no import from pyrox-gp.

:::{note} As built (pyrox-lgm 0.1, pyrox#260)
**Pins.** A uv workspace resolves one version per package, and
kernellib v0.0.15's own `[tool.uv.sources]` pins gaussx to `v0.6.0`, which
uv honours for a git dependency. pyrox-lgm needs gaussx `v0.6.1`
(`BYM2GMRF`, gaussx#518), so pyrox's root `pyproject.toml` carries a
`[tool.uv] override-dependencies` entry on gaussx `v0.6.1`, which keeps one
gaussx for every member. kernellib v0.0.16 sources gaussx `v0.6.1`
(kernellib#142), so the override can go once pyrox moves to kernellib
≥ 0.0.16. Until then, move it in step with pyrox-gp's gaussx source.
:::


### 3.2 P7: components

```python
class AbstractComponent(eqx.Module):
    name: str = eqx.field(static=True)

    def theta_spec(
        self,
    ) -> dict[
        str, tuple[dist.Distribution, Transform]
    ]: ...  # hyperpriors, unconstrained transforms
    def prior(
        self,
        theta: dict[str, Array],
        *,
        constraint: Literal["hard", "soft", "none"] = "hard",
        soft_constraint_scale: float = 1e-3,  # gaussx's default
    ) -> gx.GaussianMRF | gx.IntrinsicGMRF: ...
    @property
    def n_nodes(self) -> int: ...  # latent size, including any padding node
    @property
    def n_index(self) -> int: ...  # addressable nodes; < n_nodes for BYM2, odd RW2
    def projector(
        self, index: Array
    ) -> gx.SparseOperator: ...  # (n_obs, n_nodes): which node(s) each row touches
    def sample(
        self, index: Array | None = None, *, soft_constraint_scale: float = 1e-2
    ) -> Array: ...  # NumPyro face: θ ~ hyperpriors, x ~ prior, soft constraints
```

| Component | θ (default prior) | gaussx builder | Notes |
|---|---|---|---|
| `IID(n)` | τ (`PCPrecision(1, 0.01)`) | `iid_precision` | |
| `RW1(n)`, `RW2(n)` | τ (`PCPrecision`) | `rw1_structure`, `rw2_structure` | `scale_model=True` by default (Sørbye & Rue, 2014); hard sum-to-zero in `inla()`, soft in `.sample()`. Odd-`n` `RW2` carries gaussx's padding node (`n_nodes = n + 1`, `n_index = n`) |
| `AR1(n)` | τ, ρ (`PCAR1Rho`) | `ar1_precision` | |
| `Besag(graph)` | τ | `besag_structure(kl.structure_matrix(graph))` | One constraint per connected component (`kl.graph_null_space`) |
| `BYM2(graph)` | τ (`PCPrecision`), φ (`PCBYM2Phi`) | `BYM2GMRF` (on `bym2_precision`), `generalized_variance_scale` | The scaling constant is exact and sparse (G7); it replaces the earlier "dense or Hutchinson" idea. The field is `(b, u*)`, `n_nodes = 2n`, `n_index = n` |
| `CAR(graph)` (proper) | τ, ρ | `SparseOperator` `τ(D − ρW)` | `log|Q|` from the precomputed `L_sym` spectrum (dense for `N ≲ 10⁴`), otherwise the sparse Cholesky logdet |
| `Leroux(graph)` | τ, ρ | `τ(ρR + (1−ρ)I)` | On a `GridGraph` it is a `KroneckerSum`, with exact everything |
| `SPDE(mesh=(V, T) \| grid=shape, alpha=2)` | `range_sigma` = `[range, σ]` (joint `PCMatern`) | `fem_matrices` + `spde_precision`, or `spde_precision_grid`; `matern_spde_params` | The projector is `fem_projector` (mesh) or index selection (grid) |
| `Generic(structure, null_space=None)` | τ | the user's operator | |
| `Kronecker(main, group)` | the union of both, with the group's τ fixed at 1 | `gx.Kronecker` | Separable space-time, R-INLA's `group` |

**The NumPyro face** (`.sample()`) registers NumPyro sites, draws θ from the default (or user) hyperpriors,
and draws `x` from the gaussx GMRF with soft constraints. So these
components are usable inside any NumPyro model under NUTS, with or
without `inla()`. Each areal component keeps a private cache (spectra, null spaces, scaling), built once at
construction.

:::{note} As built (pyrox-lgm 0.1, pyrox#262–#264)
- **`BYM2`'s θ is `(tau, phi)`**, not `(σ, φ)`. `PCPrecision` on τ is the
  same prior as an exponential on σ = τ^{-1/2}, and τ keeps one convention
  across components. The field is `(b, u*)` with `n_nodes = 2n`, and
  observations see `b` (`n_index = n`). Its prior is gaussx's `BYM2GMRF`,
  because an `IntrinsicGMRF` on `bym2_precision` has the wrong density
  (gaussx#508).
- **`SPDE`'s joint `PCMatern`** sits under one θ key, `range_sigma`
  = `[range, σ]`.
- **`Kronecker(main, group)` fixes the group's τ at 1**, because only the
  product of the two τ's is identified (R-INLA's `group`). An `AR1` group
  keeps its ρ.
- **The NumPyro face's soft constraint defaults to `s = 1e-2`**, not
  gaussx's `1e-3`. On a 12 × 12 BYM2 Poisson model, `1e-3` hits NUTS's
  tree-depth cap (1023 leapfrog steps per iteration), while `1e-2` takes
  255, with the same posterior. Every `prior()` takes
  `soft_constraint_scale` (default `1e-3`, as in gaussx).
- **Odd-`n` `RW2`** carries gaussx's padding node (G7). Its `scale_model`
  is recovered over the `n` real nodes as `s_N^{N/n}`, from the scale
  `s_N` over all `N = n + 1` nodes (the padding node has unit variance and
  pulls the geometric mean towards 1), and the node never enters the
  projector. Inside `inla()` only the sum-to-zero constraint is imposed,
  as R-INLA's `rw2` does.
:::

**Example.**

```python
# The NumPyro face: BYM2 disease mapping under NUTS, with no INLA involved
def disease_map(E, y, counties):
    beta0 = numpyro.sample("beta0", dist.Normal(0.0, 10.0))
    b = lgm.BYM2(
        counties, name="region"
    ).sample()  # τ, φ from PC priors; soft sum-to-zero (s = 1e-2)
    numpyro.sample("y", dist.Poisson(E * jnp.exp(beta0 + b)), obs=y)


mcmc = MCMC(NUTS(disease_map), num_warmup=500, num_samples=1000)
mcmc.run(key, expected, cases, counties)
```

### 3.3 P7: PC priors (`_priors/_pc.py`)

**The maths.** **Penalised-complexity priors** (Simpson et al., 2017) start from a base
model $\xi_0$ (no effect, no spatial structure, infinite range). They
measure a component's complexity by
$d(\xi) = \sqrt{2\,\mathrm{KLD}(\pi_\xi\,\|\,\pi_{\xi_0})}$, and put an
exponential prior on $d$:

$$
\pi(\xi) = \lambda\,e^{-\lambda d(\xi)}\,\Big|\frac{\partial d}{\partial\xi}\Big| .
$$

- **Precision τ of a Gaussian effect.** $d\propto\tau^{-1/2} = \sigma$,
  so $\pi(\sigma) = \lambda e^{-\lambda\sigma}$, with
  $\lambda = -\log\alpha/U$ calibrated from $P(\sigma>U) = \alpha$.
- **BYM2's φ.**

  $$
  \mathrm{KLD}(\phi) = \tfrac12\Big[n\phi\big(\tfrac1n\operatorname{tr}R_\ast^{+} - 1\big) - \sum_i\log\big(1-\phi+\phi\gamma_i\big)\Big],
  $$

  with $\gamma_i$ the eigenvalues of $R_\ast^{+}$. Hence the need for a
  spectrum, and SLQ on large graphs.

All are `numpyro.distributions.Distribution` subclasses:

| Prior | Density | Calibration |
|---|---|---|
| `PCPrecision(U, alpha)` | on τ: exponential on `σ = τ^{-1/2}`, with the Jacobian | `P(σ > U) = α` |
| `PCMatern(range0, alpha_range, sigma0, alpha_sigma, d)` | joint on `(ρ, σ)`: `(d/2)λ_ρ ρ^{−d/2−1} e^{−λ_ρ ρ^{−d/2}} · λ_σ e^{−λ_σ σ}` (Fuglstad et al., 2019) | `P(ρ < ρ₀) = α_ρ`, `P(σ > σ₀) = α_σ` |
| `PCBYM2Phi(U, alpha, structure_spectrum)` | exponential on `d(φ) = √(2 KLD(φ))` (Riebler et al., 2016) | `P(φ < U) = α` |
| `PCAR1Rho(U, alpha)` | Sørbye & Rue (2017), base model `ρ = 0` (only this base is implemented), symmetric in ρ | `P(\|ρ\| > U) = α`, as R-INLA's `pc.cor0` |

**`PCBYM2Phi` needs the spectrum of the scaled structure**, because
`KLD(φ)` involves `log|(1−φ)I + φR*⁺|`. The spectrum must contain the
**null space** (the γ = 0 eigenvalues): only they send `d(φ) → ∞` as
`φ → 1`, which makes the prior proper. `lgm.structure_spectrum(R*,
null_space)` computes it once:

- dense `eigh` for `n ≤ 5000`;
- exact factor eigenvalues on a `GridGraph` (a `KroneckerSum`);
- otherwise, a **deflated stochastic Lanczos quadrature spectral
  density**. One Lanczos run keeps up to `deflate` of the largest
  eigenvalues of `R*⁺` exactly (they dominate `tr R*⁺`); probes projected
  off the null space and those eigenvectors give the rest. Without
  deflation, 64 probes gave `tr(R*⁺)` about 2 % off on a 25 × 20 grid
  (γ_max = 80 against tr = 518); with it, the PC distance is within about
  1e-4. It is an [RandNLA](project-rnla.md) consumer. The density is
  then evaluated in closed form for any φ.

**Example.**

```python
sigma_prior = lgm.PCPrecision(U=1.0, alpha=0.01)  # P(σ > 1) = 0.01
range_sd = lgm.PCMatern(
    range0=10.0, alpha_range=0.05, sigma0=2.0, alpha_sigma=0.05, d=2
)
# PCBYM2Phi: the spectrum of R*⁺, null space included, computed once per graph
R_star = kl.structure_matrix(counties, scaled=True)
spectrum = lgm.structure_spectrum(R_star, kl.graph_null_space(counties))
phi_prior = lgm.PCBYM2Phi(
    U=0.5, alpha=2 / 3, structure_spectrum=spectrum
)  # P(φ < U) = α
rho_prior = lgm.PCAR1Rho(U=0.5, alpha=0.1)  # P(|ρ| > 0.5) = 0.1
```

### 3.4 P8: `LGM` and `inla()`

**The maths.** `inla()` is three formulas run in sequence:

1. The Laplace marginal (gaussx G8):
   $\log\tilde\pi(\theta\mid y) = \log\tilde\pi(y\mid\theta) + \log\pi(\theta)$.
   It is maximised with exact gradients.
2. A design $\{\theta_k,\Delta_k\}$ in the Hessian's eigenbasis (G9).
3. Mixtures over the design points:
   $\tilde\pi(x_i\mid y) = \sum_k \mathcal N\big(x_i;\ \hat x_i(\theta_k),\ \Sigma_{ii}(\theta_k)\big)\,\tilde\pi(\theta_k\mid y)\,\Delta_k$,
   with $\Sigma_{ii}$ from Takahashi (G3, G4) and, under `strategy="vb"`,
   the means corrected by G10.

The same weights give the marginal likelihood,
$\tilde\pi(y) \approx \sum_k\tilde\pi(y\mid\theta_k)\,\pi(\theta_k)\,\Delta_k$.

:::{note} As built (pyrox-lgm 0.1, pyrox#266)
The code below is the shipped API, which differs from the plan in a few places:

- `likelihood` takes pyrox-lgm's own observation models (`lgm.Gaussian`, `Poisson`, `Bernoulli`, `Binomial`, `NegativeBinomial`), not a bare `gx.AbstractLikelihood`. They own the likelihood's hyperparameters and priors, so those can join θ.
- The marginal likelihood is the Gaussian approximation over θ, corrected by the design's ratio to that Gaussian. The plain weighted sum above was 0.15 off on the RW2 test; the corrected value is within 0.02 of brute force.
- The inner fits are a loop over design points through module-level compiled functions, not a `vmap`. A point that needs a larger Newton budget is retried and reweighted on its own.
:::

```python
class FixedEffects(eqx.Module):
    names: tuple[str, ...] = eqx.field(static=True)
    prior_precision: float = 1e-3


class LGM(eqx.Module):
    components: tuple[AbstractComponent, ...]
    fixed: FixedEffects | None
    likelihood: AbstractObservation  # lgm.Gaussian(), lgm.Poisson(), …; its θ joins

    def latent_prior(
        self, theta
    ) -> gx.GaussianMRF: ...  # block-diagonal over components (+ fixed effects)
    def projector(
        self, data
    ) -> gx.SparseOperator: ...  # [A_1 | … | A_k | X], fixed-effect columns last
    def log_posterior_theta(
        self, u, data, *, projector=None, max_newton=50
    ) -> Array: ...  # gx.laplace_mode(...).log_marginal + log π(u), u unconstrained


def inla(
    model: LGM,
    data: Mapping[str, Array],
    *,
    strategy: Literal["vb", "gaussian", "sla"] = "vb",  # "sla" added in P9
    integration: Literal["auto", "eb", "grid", "ccd"] = "auto",
    key: PRNGKeyArray | None = None,  # reserved: the design is deterministic
    max_newton: int = 50,
    theta_init: Array | None = None,
    max_theta_iter: int = 200,
    theta_tol: float = 1e-5,
    verbose: bool = False,
) -> INLAResult: ...
```

The pipeline:

1. **Build once.** Build the latent prior's pattern, the pattern of
   `Q + AᵀWA`, and the symbolic Cholesky, all on the host. Fixed-effect
   columns are dense. They are ordered last so they add `p` dense rows
   without destroying fill, as R-INLA does.
2. **θ-mode.** Run optax L-BFGS on `−log_posterior_theta`, with exact
   gradients through `laplace_mode`'s implicit differentiation and the
   selected-inverse logdet VJP.
3. **Design.** Call `gx.theta_design` with `"auto"`: `"eb"` if the user
   asks, `"grid"` for `m ≤ 2`, `"ccd"` otherwise.
4. **Inner fits.** Run `laplace_mode` at each design point: same
   pattern, same symbolic factorisation, new values. As built, this is a
   loop through module-level compiled functions, so a repeat fit reuses
   every compilation (the POD toy runs in 1.4 s warm).
5. **Per-θ marginals.** For each θ-point, the Gaussian marginals of `x`
   are the mode and the Takahashi variances. Apply
   `gx.vb_mean_correction` when `strategy="vb"`, which is R-INLA's
   default since 22.11.
6. **Mix.** Mix across θ-points with the design weights to get the
   latent marginals (mean, sd, quantiles by root-finding on the mixture
   CDF). The hyperparameter marginals come from the design points in `z`,
   and the log marginal likelihood from the weighted sum.

`INLAResult` provides:

- `.fixed`, `.random[name]`, `.hyperpar`: arrays of summary statistics;
- `.linear_predictor`, `.predictor_means`, `.predictor_variances`: the
  linear predictor's marginals (added in P9);
- `.latent_skewness`: zero, except under `strategy="sla"` (added in P9);
- `.diagnostics()`: DIC, WAIC, CPO and PIT (added in P9);
- `.log_marginal_likelihood`;
- `.sample_latent(key, n)`: mixture sampling via each θ-point's factor;
- `.predict(new_data)`;
- `.to_xarray()` behind the `xarray` extra (#155 open question 3).

**Failure handling.** θ-points whose inner Newton did not converge are
re-run with four times the iterations, then dropped with a warning. The
result records how many were dropped. As built, the θ-mode search is
stricter:

- it rejects an unconverged inner fit at any iterate;
- if the mode's own fit fails, the whole fit re-runs at the larger budget, and then raises;
- missing `theta_tol` within `max_theta_iter` also raises.

**Example.**

```python
# Lip-cancer-style disease mapping (Poisson, BYM2, one covariate)
model = lgm.LGM(
    components=(lgm.BYM2(counties, name="region"),),
    fixed=lgm.FixedEffects(("intercept", "pct_agri")),
    likelihood=lgm.Poisson(),
)
data = {"y": cases, "offset": jnp.log(expected), "region": region_idx, "pct_agri": agri}
res = lgm.inla(model, data)
res.fixed["pct_agri"]  # posterior mean, sd, quantiles of the covariate effect
res.hyperpar["region.phi"]  # how much of the variation is spatially structured

# SST gap-filling: space-time AR(1) ⊗ SPDE on a 1° grid, Gaussian likelihood
sst_model = lgm.LGM(
    components=(
        lgm.Kronecker(
            lgm.AR1(n_days, name="t"), lgm.SPDE(grid=(180, 360), alpha=2, name="s")
        ),
    ),
    fixed=lgm.FixedEffects(("intercept",)),
    likelihood=lgm.Gaussian(),
)
```

### 3.5 P9: ergonomics and extensions

:::{note} As built (pyrox-lgm, pyrox#267, #269, #270)
Shipped as planned, with three changes, all described below:

- CPO and PIT use a Gaussian cavity rather than the raw harmonic identity.
- `strategy="sla"` produces skew-normal marginals.
- `FixedEffects` covariates can be JAX values, which the hybrid needs.
:::

**The maths.** **CPO** is leave-one-out without refitting:

$$
\mathrm{CPO}_i = \pi(y_i\mid y_{-i}) = \Big(\int\frac{\pi(\eta_i\mid y)}{\pi(y_i\mid\eta_i)}\,d\eta_i\Big)^{-1}.
$$

As built, dividing a Gaussian marginal by $\pi(y_i\mid\eta_i)$ itself
diverges for most likelihoods: $1/\pi$ grows like $e^{e^\eta}$ for a
Poisson. So the site is removed at the order of the Laplace approximation,
as R-INLA does, using a Gaussian cavity. With $g_i$ and $W_i$ the gradient
and negative Hessian of $\log\pi(y_i\mid\eta)$ at the mean $m_i$, and
$v_i$ the variance of $\eta_i$ from Takahashi:

$$
\pi(\eta_i\mid y_{-i}) \approx \mathcal N\big(m_i - g_i c_i,\ c_i\big),
\qquad c_i = \frac{v_i}{1 - W_i v_i}.
$$

CPO is the predictive density of $y_i$ under that cavity, integrated on
the marginal's own Gauss–Hermite nodes. It is exact for a Gaussian
likelihood at fixed θ.

**PIT** is $P(Y_i\le y_i\mid y_{-i})$ under the same cavity. Both mix over
the design with leave-one-out weights $\propto w_k/\mathrm{CPO}_{ik}$.

**WAIC** needs the posterior mean and variance of
$\log\pi(y_i\mid\eta_i)$ under the predictor marginals; **DIC** needs the
mean deviance and the deviance at the posterior mean.

**Simplified Laplace.** The Laplace marginal
$\pi(x_i) \propto \pi(x,y)/\pi_G(x_{-i}\mid x_i)$ is taken at the Gaussian
conditional mean and expanded to third order in the standardised $z$ (Rue,
Martino & Chopin, 2009, §3.2.3):

$$
-\tfrac12 z^2 + \gamma_1 z + \tfrac16\gamma_3 z^3,\qquad
\gamma_1 = \tfrac12\sum_j d_j\,(s_j^2 b_{ij} - b_{ij}^3),\qquad
\gamma_3 = \sum_j d_j\, b_{ij}^3.
$$

Here $d_j$ is the third derivative of $\log\pi(y_j\mid\eta_j)$ at the mode,
$b_{ij} = \operatorname{cov}(x_i,\eta_j)/\sigma_i$ and $s_j^2 = \operatorname{var}(\eta_j)$.
The mean shifts by $\sigma_i(\gamma_1 + \gamma_3/2)$, and the skewness is
$\gamma_3$. Each marginal becomes the skew-normal with those moments.

- **`f(...)` sugar.**
  `f(column, model, *, n=None, graph=None, hyper=None, **kwargs)` builds a
  component named after the data column it indexes:
  - `"iid"`, `"rw1"`, `"rw2"`, `"ar1"` (sized by `n`);
  - `"besag"`, `"bym2"`, `"car"`, `"leroux"` (on `graph`);
  - `"spde"` and `"generic"`.

  `hyper` keys map onto the constructors' prior arguments. It is sugar over
  the constructors, not a string-formula parser.
- **Diagnostics.** `lgm.diagnostics(result)` or `result.diagnostics()`
  returns `Diagnostics(cpo, pit, log_score, waic, p_waic, dic, p_d)`, with
  no refits. Observation models gain `site_log_prob` / `site_cdf`, with
  closed-form CDFs for the count likelihoods. A custom observation model
  needs `site_distribution` only for the diagnostics.
- **Simplified Laplace.** `inla(..., strategy="sla")`:
  - costs one sparse solve per observation, for `cov(x, η) = Σ Aᵀ`;
  - produces skew-normal mixture summaries and `INLAResult.latent_skewness`;
  - leaves a Gaussian likelihood unchanged.

  Full Laplace is only a slow validation reference, in tests.
- **MCMC-INLA hybrid.** The notebook `lgm_mcmc_inla.ipynb` runs NUTS over
  a non-LGM parameter and θ, with
  `numpyro.factor(model.log_posterior_theta(u, data_at(phi)))` as the
  potential (Gómez-Rubio & Rue, 2018). This needs that function's exact
  gradients, and covariates that may depend on the sampled parameter. It
  agrees with INLA run on a grid of that parameter and averaged by
  marginal likelihood.

---

## 4. P10: repo-level changes

- **`design_docs/pyrox/boundaries.md`:**
  - add rows: graph construction, Laplacians, graph spectra, GMRF
    structure → kernellib; sparse operators, GMRF distributions,
    precision builders, INLA kernels, randomized linear algebra → gaussx;
    graph inducing features, latent and inducing-point initialisation,
    exact-GP recipes → pyrox-gp; latent components, PC priors, `inla()` →
    pyrox-lgm;
  - state the pyrox-gp / pyrox-lgm split: GPs in covariance form versus
    LGMs in precision form;
  - add geonnax, which the table does not mention today although pyrox-gp
    re-exports `graph_laplacian_eigpairs` and `eof_basis` from it.
- **`design_docs/pyrox/features/gp/spde_fem.md`** gets a
  `status: superseded` note mapping its pieces:
  - Layer 0 (`pyrox.gp._src.fem`) becomes `gaussx.fem_matrices`,
    `fem_projector` and `spde_precision` (G7);
  - Layer 1 (`SPDESolver`) becomes `gaussx.GaussianMRF` +
    `SparseCholeskySolver` (G4, G6);
  - Layer 2 (`spde_gp_factor`) becomes the `pyrox_lgm.SPDE` component's
    NumPyro face (P7).
  - Its non-stationary and rational-α sections stay, as gaussx follow-ups.
- **pyrox#50.** Its SPDE-FEM item is retargeted to P7 and P8, and its
  LogFalkon item gets P5.
- **`pyrox_gp._basis` still re-exports geonnax's
  `graph_laplacian_eigpairs`.** Keep the re-export for compatibility, but
  stop using it internally after P2.
- **pyrox-gp's covariance-form `LaplaceInference` is unchanged**
  ([INLA open question 4](project-inla.md#open-questions)). A later
  option: route GMRF priors in pyrox-gp models through `gx.laplace_mode`.
- **Training loops in docs** ([roadmap decision 7](roadmap.md)). New
  examples that fit by SVI may use pipekit-train's `numpyro-svi` adapter
  instead of a hand-written `svi.update` loop, with pipekit in the docs
  dependency group only. Existing notebooks change only when re-executed
  for another reason.
- **Real spatial data in docs** (decision 7). The pyrox-lgm BYM2 notebook
  starts from polygons through city2graph's `contiguity_graph` and
  `kl.graph_from_edges` (K2), instead of a pre-built adjacency matrix.
  The OD-flow example ([gallery 13](roadmap-examples.md#ex-od)) is a
  candidate second notebook after P8.
- **Dependency pins:**
  - pyrox-gp: kernellib ≥ v0.0.8 for P1; the tag containing K3 and K4 for
    P2; the tag containing K8 for P3 and P5; gaussx with G13, after #312,
    for P4.
  - pyrox-lgm: gaussx with G6–G10, and kernellib with K6.
- No changes to `pyrox` (core) or `pyrox-nn`.

---

## 5. Phases and tests

P1–P5's tests are listed with each section above. P6–P9:

| Phase | Content | Tests |
|---|---|---|
| P6 | Scaffold `packages/pyrox-lgm` (pyproject, `__init__`, API test, import guard, docs stub) | Workspace `uv sync`; the import guard (no pyrox-gp import) |
| P7 | Components, NumPyro face, PC priors | Each PC prior's calibration statement holds (`P(σ > U) = α` by quadrature); `PCBYM2Phi`'s SLQ spectrum agrees with dense at `n = 500`; `.sample()` under `handlers.seed` / `trace` has the right sites; NUTS on a 12 × 12 grid BYM2 Poisson has no divergences (integration) |
| P8 | `LGM`, `inla()`, `INLAResult` | Golden R-INLA fixtures (generated offline, as in gaussx.md §6): RW2 Gaussian on a sine (exact), AR(1), Scotland BYM2 Poisson, SPDE Poisson on a small mesh, Bernoulli POD toy (rw2 + fixed effects). Latent means within 1e-3 relative; hyperparameter posterior medians within 5 %; log marginal likelihood within 0.1. Benchmark: POD toy under 5 s on CPU (#155's target) |
| P9 | Sugar, diagnostics, SLA, hybrid notebook | CPO equals brute-force leave-one-out on a tiny problem; SLA moves marginal skewness towards the full-Laplace reference. **Done:** CPO / PIT equal the exact leave-one-out at fixed θ (Gaussian) and match Poisson refits within 0.3 %; SLA means within 0.03 sd and skewness within 0.05 of full Laplace (pyrox#267, #269, #270) |
| P10 | pyrox-gp doc retarget, `boundaries.md`, pyrox#50 | Docs build |
