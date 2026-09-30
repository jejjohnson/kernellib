---
date: 2026-09-30
---

# gaussx: roadmap

gaussx's share of the [fused roadmap](roadmap.md). Everything here is
kernel-free and graph-free linear algebra, used by the three projects
([manifold](project-manifold.md), [RandNLA](project-rnla.md),
[INLA](project-inla.md)). kernellib builds graphs and meshes and hands
gaussx operators. gaussx never imports kernellib. The baseline is gaussx
0.4.0.

| Phase | What | Projects | Needs |
|---|---|---|---|
| G1 | `SparseOperator` with a static `SparsityPattern` | manifold, INLA | — |
| G2 | `eigh_generalized` | manifold, RandNLA | — |
| G3 | Structured selected inverses: `BlockTriDiag`, (Sum)Kronecker `diag_inv`; fix #344 | INLA | — |
| G4 | Sparse Cholesky, Takahashi selected inverse, VJPs, opt-in CHOLMOD | INLA | G1 |
| G5 | `pseudo_logdet` | manifold, INLA | G3 (grid path), G4 (cofactor path) |
| G6 | `GaussianMRF`, `IntrinsicGMRF` | manifold, INLA | G1, G3, G5 (G4 for sparse Cholesky sampling) |
| G7 | Precision builders (iid, rw1, rw2, ar1, besag, bym2, `generalized_variance_scale`), SPDE (grid and FEM), `fem_matrices`, `fem_projector` | INLA | G1, G3 (G4 for meshes) |
| G8 | `laplace_mode`: precision-form Newton, implicit differentiation, Laplace log-marginal | INLA | G6 |
| G9 | `theta_design` (`eb`, `grid`, `ccd`) | INLA | — |
| G10 | `vb_mean_correction` | INLA | G8 |
| G11 | Sketching operators, `hadamard_transform` (moved from kernellib) | RandNLA | — |
| G12 | `range_finder`, `qb`, `randomized_svd`, `randomized_eigh`; `method="randomized"` | RandNLA | G11 |
| G13 | `randomized_nystrom`; `NystromPreconditioner` rewrite (closes #354) | RandNLA | G12 |
| G14 | `rp_cholesky`; `PartialCholeskyPreconditioner(pivoting=...)` built once (#371) | RandNLA | — |
| G15 | `SketchAndPrecondLSMR`, `sketch_and_solve` | RandNLA | G11 |
| G16 | `diag_inv(method="xdiag")`: the marginal-variance fallback for fields too large to factor | RandNLA, INLA | G12 |
| G17 | Tier 2, on demand: Hutch++ / XTrace, `NystromLogdet`, interpolative / CUR | RandNLA | G12, G13 |

> Maths notes (**The maths.**) say where each operation comes from; **Example.** blocks are pseudocode against the *planned* API (`gx` = gaussx, `kl` = kernellib, `px` = pyrox-gp, `lgm` = pyrox-lgm). End-to-end problems are in the [examples gallery](roadmap-examples.md).

The document is in two parts:

- **Part A (G1–G10)** is the structured-precision stack: graphs, GMRFs
  and INLA.
- **Part B (G11–G17)** is the randomized stack.

They meet in two places:

- G16 is Part A's fallback for fields too large to factor;
- G2's generalised eigensolver serves both manifold alignment and LPP.

---

## 1. Current state

| Symbol | Relevance |
|---|---|
| `KroneckerSum` with exact `eig`, `logdet`, `solve`, `sqrt_matmul` | 4-connected grid Laplacians (`L_H ⊕ L_W`), SPDE on rasters, Leroux on grids. Missing: `diag_inv`. A Leroux precision `τ(ρL + (1−ρ)I)` is a `KroneckerSum` as is, because `(A + cI) ⊕ B = A ⊕ B + cI` |
| `BlockTriDiag` (+ lower / upper) with `cholesky`, `solve`, `logdet`, `diag`, `trace` dispatch, `O(N d³)` | Every banded GMRF (RW, AR, temporal SDE priors). Missing: the selected inverse. #344: its `symmetric_tag` is wrong |
| `diag_inv(op, method="auto" \| "cholesky" \| "solve" \| "hutchinson")` | No structural dispatch: dense Cholesky for `N ≤ 2048`, Hutchinson above |
| `eig(op, rank=)`, `svd(op, rank=)` via matfree Krylov (Lanczos, Golub–Kahan) | Matrix-free partial eigendecompositions. No block randomized option for slowly decaying spectra |
| `AutoSolver` (CG for large PSD), `PreconditionedCGSolver`, `JacobiPreconditioner`, `LSMRSolver` | Iterative solves. LSMR has no preconditioning |
| `SLQLogdet`, stochastic `trace` / `diag` / `frobenius_norm` (matfree Hutchinson; signs / normal / sphere probes) | The iterative log-determinant; no Hutch++ / XTrace / XDiag variance reduction |
| `NystromPreconditioner` | A randomized **Rayleigh–Ritz** projection (`QᵀAQ` on an orthonormalised Gaussian block), not a Nyström sketch. It raises CG iteration counts 1.2–20× below full rank (#354) |
| `guarded_pivoted_cholesky`, `PartialCholeskyPreconditioner` | **Greedy** pivoted partial Cholesky, matrix-free, with a `pstrf`-style guard; the pivots are not returned. The preconditioner is rebuilt inside every solve (#371) and counts the noise twice (#345) |
| `LowRankUpdate(base, U, d, V, orthonormal=)` | Low-rank algebra with structural `solve` / `logdet`; the return type of every randomized factorisation |
| `MultivariateNormalPrecision` (NumPyro subclass) | The pattern for the GMRF distributions |
| `newton_update`, `cavity_distribution`, `damped_natural_update` | Site-level Newton / EP algebra, used by pyrox-gp's Laplace and EP |
| `_quadrature/`: `GaussHermiteIntegrator`, unscented, fifth-order; `AbstractLikelihood` (Gaussian, Bernoulli, Poisson, Student-t) | Expectations for the VB correction; the likelihood family for Laplace; the home for θ-designs |
| `_ssm/` (Kalman, RTS, UDL for block-tridiagonal precision, SpInGP) | Temporal GMRFs as state-space models. The RTS recursion is the block selected inverse in disguise |

---

## 2. Package layout

```
src/gaussx/
├── _operators/
│   ├── _sparse.py                   # SparseOperator + SparsityPattern                    (G1)
│   └── _spectral_function.py        # f(A ⊕ B) with shared eigenvectors (grid SPDE)       (G7)
├── _sparse/                         # NEW                                                 (G4)
│   ├── _symbolic.py                 # SymbolicCholesky: ordering, elimination tree, pattern of L (host)
│   ├── _numeric.py                  # sparse_cholesky (JAX, static indices), triangular solves
│   ├── _takahashi.py                # selected_inverse on pattern(L + Lᵀ)
│   ├── _vjp.py                      # custom VJPs: logdet (via Takahashi), solve
│   └── _cholmod.py                  # opt-in backend (pure_callback, extra gaussx[cholmod])
├── _sketching/                      # NEW                                                 (G11)
│   ├── _base.py                     # AbstractSketch
│   ├── _dense.py                    # GaussianSketch, OrthonormalSketch
│   ├── _sparse_sign.py              # SparseSignSketch
│   ├── _srht.py                     # SRHTSketch
│   ├── _sampling.py                 # RowSamplingSketch
│   └── _hadamard.py                 # hadamard_transform (moved from kernellib)
├── _randomized/                     # NEW
│   ├── _range_finder.py             # range_finder, qb                                    (G12)
│   ├── _svd.py                      # randomized_svd, randomized_eigh                     (G12)
│   ├── _nystrom.py                  # randomized_nystrom                                  (G13)
│   ├── _rpcholesky.py               # rp_cholesky                                         (G14)
│   ├── _trace.py                    # hutchpp, xtrace                                     (G17)
│   └── _interpolative.py            # column_id, cur                                      (G17)
├── _linalg/
│   ├── _diag_inv.py                 # structural dispatch (G3, G4); method="xdiag" (G16)
│   └── _selected_inverse.py         # block Takahashi for BlockTriDiag                    (G3)
├── _primitives/
│   ├── _eig.py                      # eigh_generalized (G2); method="randomized" (G12)
│   ├── _svd.py                      # method="randomized"                                 (G12)
│   ├── _logdet.py                   # pseudo_logdet                                       (G5)
│   ├── _root.py                     # guarded_pivoted_cholesky shares its loop with rp_cholesky (G14)
│   └── _trace.py                    # method="hutchpp" | "xtrace"                         (G17)
├── _preconditioners/
│   ├── _nystrom.py                  # REWRITTEN on randomized_nystrom                     (G13)
│   └── _partial_cholesky.py         # pivoting="greedy" | "random"; from_operator         (G14)
├── _strategies/
│   ├── _sparse_cholesky.py          # SparseCholeskySolver                                (G4)
│   ├── _sketch_precond.py           # SketchAndPrecondLSMR, sketch_and_solve              (G15)
│   └── _nystrom_logdet.py           # NystromLogdet                                       (G17)
├── _distributions/_gmrf.py          # GaussianMRF, IntrinsicGMRF                          (G6)
├── _gmrf/                           # NEW                                                 (G7)
│   ├── _temporal.py                 # iid, rw1, rw2, ar1
│   ├── _areal.py                    # besag_structure, bym2_precision, generalized_variance_scale
│   ├── _spde.py                     # spde_precision, spde_precision_grid, matern_spde_params
│   └── _fem.py                      # fem_matrices, fem_projector
├── _inference/
│   ├── _laplace.py                  # laplace_mode, LaplaceResult                         (G8)
│   └── _vb_correction.py            # vb_mean_correction                                  (G10)
└── _quadrature/_theta_design.py     # theta_design                                        (G9)
```

All new names are exported flat at the top level, with one API page per
new area.

**Conventions**, as in the rest of gaussx:

- Equinox modules and pure functions, with jaxtyping shapes.
- einx for every contraction and rearrangement, including scan bodies and
  tests.
- Distributions subclass NumPyro's `Distribution`, following the `LGSSM` /
  `MultivariateNormalPrecision` pattern, never bare Equinox modules.
- Every randomized function takes `key`, and follows the existing
  convention (`key=None` means `PRNGKey(0)`), stated in each docstring.

---

# Part A — structured precision, GMRFs, INLA

## 3. The organising idea: dispatch on precision structure

**The maths.** A GMRF is a Gaussian $x \sim \mathcal N(\mu, Q^{-1})$ with a sparse
precision, and sparsity *is* conditional independence (Rue & Held 2005,
Thm 2.2):

$$
Q_{ij} = 0 \iff x_i \perp x_j \mid x_{-\{i,j\}},\qquad
\mathbb E[x_i \mid x_{-i}] = \mu_i - \frac{1}{Q_{ii}}\sum_{j\neq i}Q_{ij}(x_j-\mu_j),\qquad
\operatorname{Prec}(x_i \mid x_{-i}) = Q_{ii}.
$$

Local models (random walks, neighbours on a map, finite elements) therefore
give sparse `Q`. Every operation below reduces to a Cholesky factor
$Q = LL^\top$:

- $\log|Q| = 2\sum_i \log L_{ii}$;
- $Q^{-1}b$ is two triangular solves;
- a sample is $x = \mu + L^{-\top}z$;
- marginal variances come from the selected inverse (G3, G4).

What changes with the structure is the cost of $L$:

- bandwidth $b$: $O(Nb^2)$;
- a 2-D mesh under nested dissection: $O(N^{3/2})$ time and
  $O(N\log N)$ fill;
- a 3-D mesh: $O(N^2)$;
- a Kronecker sum: eigendecompose the factors instead, $O(\sum_k n_k^3)$.

The table sends each case to its cheapest exact path.

Every INLA primitive needs four operations on a precision `Q`, or on the
Laplace Hessian `H = Q + Aᵀ W A`: `solve`, `logdet`, the diagonal of the
inverse (`diag_inv`, the marginal variances), and `sample`. gaussx already
routes primitives by operator type. Part A adds the missing rows, so
each INLA component lands on the cheapest exact path:

| Structure | Where it arises | solve / logdet | diag of inverse | sample |
|---|---|---|---|---|
| `BlockTriDiag` | rw1, rw2, ar1, ar(p), temporal SDE priors | exists | **new** (block Takahashi recursion, O(N d³)) | exists (Cholesky) |
| `KroneckerSum` / eigen-factored | SPDE on a raster (`K = κ²I + L_H ⊕ L_W`), Leroux on a grid | exists | **new** (from the factor eigenvectors) | exists |
| `Kronecker` | separable space-time `Q_t ⊗ Q_s` (prior only) | exists | **new** (`diag_inv(Q_t) ⊗ diag_inv(Q_s)`) | exists |
| `SparseOperator` + sparse Cholesky | besag / bym2 on graphs, SPDE on meshes, any `H = Q + AᵀWA` | **new** | **new** (Takahashi) | **new** |
| `SparseOperator`, too large to factor | 10⁶-node meshes on GPU | CG + SLQ (existing) | `diag_inv(method="xdiag")` (G16), or Hutchinson (existing) | perturbation–optimisation (G6 factors) |

The Laplace Hessian of a structured prior with a diagonal likelihood keeps
the structure only when `A` is the identity (or a row selection on a grid
with Gaussian noise, `Q + σ⁻²I`). Otherwise it becomes a
`SparseOperator` whose sparsity pattern is fixed across all θ. That is why
symbolic analysis happens once.


## 4. API

### 4.1 G1: `SparseOperator` with a static pattern

**The maths.** The pattern of every matrix INLA factorises is known before any value is.
The Laplace Hessian adds $A^\top W A$ to $Q$, and

$$
(A^\top W A)_{ij} = \sum_k A_{ki}\,w_k\,A_{kj} \neq 0 \;\Rightarrow\; i, j \in \operatorname{supp}(A_{k,:}) \text{ for some row } k,
$$

so
$\operatorname{pattern}(Q + A^\top WA) = \operatorname{pattern}(Q)\cup\bigcup_k \operatorname{supp}(A_{k,:})^2$.
It is computed once, on the host.

- **FEM projector.** Each row touches the three vertices of one triangle,
  which are already neighbours in $Q$, so the pattern does not grow at
  all.
- **Fixed effects.** Each adds one dense row and column.

**The sparsity pattern is static.** A symbolic Cholesky (G4) is then
computed once per pattern, and reused across every Newton step, θ-point,
reweighting and `vmap`ped dataset. Only `values` is traced.
```python
class SparsityPattern:  # host-side, hashable (content hash of the index arrays)
    rows: np.ndarray  # int32, (nnz,), canonical order, diagonal always present
    cols: np.ndarray
    shape: tuple[int, int]
    symmetric: bool  # store the lower triangle only when True


class SparseOperator(lx.AbstractLinearOperator):
    values: Float[Array, " nnz"]  # traced
    pattern: SparsityPattern = eqx.field(static=True)
    tags: frozenset[object] = eqx.field(static=True)

    @classmethod
    def from_coo(
        cls, rows, cols, values, shape, *, symmetric=False, tags=frozenset()
    ): ...
    def to_bcoo(self) -> jsparse.BCOO: ...
    def add_diagonal(self, d: Float[Array, " n"]) -> SparseOperator: ...  # same pattern
    def union(
        self, other: SparseOperator
    ) -> SparseOperator: ...  # pattern union (host), values added
    def congruence(
        self, A: SparseOperator, w: Float[Array, " m"]
    ) -> SparseOperator: ...  # Aᵀ diag(w) A on a precomputed pattern
```

- **Lineax interface:**
  - `mv` uses `segment_sum` over the pattern, or a BCOO matvec, whichever
    wins the G1 benchmark on CPU and GPU (internal, invisible to callers);
  - `as_matrix` is dense;
  - `transpose` swaps the index arrays;
  - `from_coo(..., symmetric=True)` takes each off-diagonal entry once and
    mirrors it. This matches kernellib's edge-once graph storage.
- **`congruence`** forms `Aᵀ diag(w) A`. Its pattern is computed on the host
  once, and its values in JAX with `segment_sum` over precomputed index
  triples. It is what makes the Laplace Hessian `Q + AᵀWA` cheap to rebuild
  at every Newton step.
- **Primitive dispatch:**

  | Primitive | Behaviour |
  |---|---|
  | `diag` | Exact, read from the pattern |
  | `solve` | `SparseCholeskySolver` (G4) when the caller passes it; otherwise `AutoSolver` (CG when tagged PSD and large, dense below the threshold) |
  | `logdet` | `SparseCholeskySolver` (G4) when passed; otherwise `SLQLogdet` for large PSD |
  | `diag_inv` | Takahashi through the factor (G4); otherwise the existing Hutchinson, or XDiag (G16) |
  | `eig(rank=)` | The existing Lanczos path, which needs only `mv` |
  | `cholesky` | A `SparseCholeskyFactor` (G4). Before G4 lands: a dense fallback below `AutoSolver.size_threshold`, an informative `NotImplementedError` above |

- **Preconditioning.** The existing `JacobiPreconditioner` works through
  the exact `diag`. It is usually enough for graph Laplacians plus a
  diagonal shift.
- `jax.experimental.sparse` is imported only in this module.

**Example.**

```python
# An ICAR structure matrix from an edge list (e.g. county contiguity), host-side indices
N = n_counties
deg = np.bincount(senders, w, N) + np.bincount(receivers, w, N)
R = gx.SparseOperator.from_coo(
    np.r_[np.arange(N), senders],
    np.r_[np.arange(N), receivers],
    jnp.r_[deg, -w],
    (N, N),
    symmetric=True,
    tags=frozenset({lx.positive_semidefinite_tag}),
)
Q = eqx.tree_at(
    lambda op: op.values, R, tau * R.values
)  # new values, same static pattern
# Newton step t of a Laplace fit: rebuild H on the precomputed pattern, values only
H = Q.union(Q.congruence(A, w_t))  # Q + Aᵀ diag(w_t) A
```

### 4.2 G2: `eigh_generalized`

**The maths.** Graph embeddings and alignment all minimise a trace under a quadratic
constraint:

$$
\min_Y \operatorname{tr}(Y^\top A Y)\ \ \text{s.t.}\ \ Y^\top B Y = I
\quad\Longrightarrow\quad A y = \lambda B y
$$

(stationarity of the Lagrangian).

- **`B = CCᵀ`.** It becomes $C^{-1}AC^{-\top}u = \lambda u$ with
  $y = C^{-\top}u$.
- **Diagonal `B`.** $C = B^{1/2}$, and everything stays matrix-free.
- **Singular `B`.** The constraint only sees $\operatorname{range}(B)$.
  With $B = U_+S_+U_+^\top$, solve
  $S_+^{-1/2}U_+^\top A U_+S_+^{-1/2}u = \lambda u$. Directions in
  $\ker B$ are unconstrained, and must be dropped, not assigned an
  infinite eigenvalue.

```python
def eigh_generalized(
    A: lx.AbstractLinearOperator,
    B: lx.AbstractLinearOperator,
    *,
    rank: int | None = None,
    which: Literal["smallest", "largest"] = "smallest",
    rcond: float | None = None,
    key: PRNGKeyArray | None = None,
) -> tuple[Float[Array, " K"], Float[Array, "N K"]]:
    """Solve A v = λ B v for symmetric A and symmetric PSD B. Returns B-orthonormal v."""
```

Dispatch on `B`:

| `B` | Method |
|---|---|
| `DiagonalLinearOperator` with positive entries | Scale: `S = B^{-1/2} A B^{-1/2}`, `eig(S)`, `v = B^{-1/2} u`. This stays matrix-free, so `rank=` goes through Lanczos. It is kernellib's current degree-constraint trick |
| Positive definite (tagged) | Cholesky whitening `C⁻¹ A C⁻ᵀ`, which is LPP's current code path |
| PSD, possibly singular (the default for dense) | Eigendecompose `B`; directions with eigenvalue `≤ rcond · max` are dropped from the problem; whiten the rest. This is the old `eigh_robust`, but it drops the null directions instead of setting them to `∞`. The returned `K` shrinks if `B`'s rank is below the requested count, and a warning says so |

With `which="smallest"` and `rank=` on a matrix-free path, use the shift
`c·I − S` with a Gershgorin bound, as kernellib's ARPACK path does now.
Einx for every contraction and rearrangement, following gaussx convention.

Consumers:

- kernellib's eigenmaps and linear projections (K5): the degree
  constraint, and LPP / SEP whitening;
- manipy's manifold alignment (M1), whose `B` can be singular.

**Example.**

```python
# Laplacian eigenmaps: L y = λ D y with D diagonal, so Lanczos stays matrix-free
lam, Y = gx.eigh_generalized(
    L_op, lx.DiagonalLinearOperator(degree), rank=3, which="smallest"
)
Y = Y[:, 1:]  # drop the constant solution
# Manifold alignment: B = Zᵀ L_d Z is dense and often singular (few labels)
lam, F = gx.eigh_generalized(
    lx.MatrixLinearOperator(ZtAZ),
    lx.MatrixLinearOperator(ZtBZ),
    which="smallest",
    rcond=1e-10,
)
```

### 4.3 G3: structured selected inverses

**The maths.** **Takahashi's identity** (Takahashi, Fagan & Chen, 1973). From
$Q = LL^\top$, the inverse $\Sigma = Q^{-1}$ satisfies
$L^\top\Sigma = L^{-1}$. The right-hand side is lower triangular with
diagonal $1/L_{ii}$, so reading off the upper triangle ($j \ge i$) gives

$$
\Sigma_{ij} = \frac{\delta_{ij}}{L_{ii}^2} - \frac{1}{L_{ii}}\sum_{k>i}L_{ki}\,\Sigma_{kj}.
$$

Running this backwards from $i = N$, each entry needs only entries below
it, and only where $L_{ki}\neq 0$. For block-tridiagonal $L$ the sum has
a single term ($k = i+1$), which gives the two-line block recursion below:
the RTS smoother, written in precision form.

**Kronecker sums.** Since
$f(A\oplus B) = (U_A\otimes U_B)\,f(\Lambda_A\oplus\Lambda_B)\,(U_A\otimes U_B)^\top$,

$$
\big[f(A\oplus B)^{-1}\big]_{(h,w),(h,w)} = \sum_{i,j}\frac{U_A[h,i]^2\,U_B[w,j]^2}{f(\lambda^A_i+\lambda^B_j)}
= \big[(U_A\circ U_A)\,M\,(U_B\circ U_B)^\top\big]_{hw},\qquad M_{ij} = \frac{1}{f(\lambda^A_i+\lambda^B_j)}.
$$

```python
def selected_inverse(op: BlockTriDiag) -> BlockTriDiag: ...  # the band of op⁻¹
```

- **`BlockTriDiag`.** The block Takahashi recursion, which is the RTS
  smoother in precision form. Factor `op = L Lᵀ` by block Cholesky, with
  diagonal blocks `L_k` and sub-diagonal blocks `B_k` (block `(k+1, k)`).
  Then sweep backwards from `Σ_NN = L_N⁻ᵀ L_N⁻¹`:

  ```
  Σ_{k+1,k} = −Σ_{k+1,k+1} B_k L_k⁻¹
  Σ_kk      = L_k⁻ᵀ (L_k⁻¹ − B_kᵀ Σ_{k+1,k})
  ```

  With `d = 1` this is Rue & Held's scalar recursion. All blocks are
  `(d, d)`, the cost is `O(N d³)`, and it runs as a `lax.scan`. It returns
  both diagonal and off-diagonal blocks, because the VB correction and
  predictor variances need neighbouring covariances.
- **`diag_inv` dispatch:**
  - `BlockTriDiag` takes the diagonal of `selected_inverse`.
  - `Kronecker(A, B)` gives `diag_inv(A) ⊗ diag_inv(B)`.
  - `KroneckerSum(A, B)` (and G7's spectral functions of it): with
    `A = U_A Λ_A U_Aᵀ` and `B = U_B Λ_B U_Bᵀ`,
    `diag(f(A ⊕ B)⁻¹) = (U_A ∘ U_A) · M · (U_B ∘ U_B)ᵀ` with
    `M_ij = 1/f(λ^A_i + λ^B_j)`. That is two matrix products, `O(H²W + HW²)`
    for an `H × W` grid, and it is exact.
  - A `pinv=True` flag zeroes the entries at `f = 0`, for intrinsic priors
    on grids. That is the exact BYM2 and ICAR scaling constant on a raster.
- **Fix #344** (`BlockTriDiag`'s `symmetric_tag`) in the same PR: every
  banded GMRF depends on it.


**Example.**

```python
# Posterior sd of a daily RW2 trend under Gaussian noise: τ·Q + σ⁻² I stays block-tridiagonal
H = gx.add_diagonal(tau * gx.rw2_structure(n_days), jnp.full(n_days, sigma**-2))
trend = gx.solve(H, y / sigma**2)
sd = jnp.sqrt(gx.diag_inv(H))  # block Takahashi: O(N), not O(N³)

# Prior sd of a Matérn field on a 2048 × 2048 raster: two small matrix products
Q = gx.spde_precision_grid((2048, 2048), kappa=0.05, tau=1.0, alpha=2)
sd = jnp.sqrt(
    gx.diag_inv(Q)
)  # shows the boundary inflation to be removed by domain extension
```

### 4.4 G4: sparse Cholesky, Takahashi, VJPs

**The maths.** Cholesky is Gaussian elimination. Eliminating node $j$ connects all of its
not-yet-eliminated neighbours, so the column patterns follow the
**elimination tree**:

$$
\operatorname{struct}(L_{:,j}) = \operatorname{struct}(Q_{j:,j})\ \cup \bigcup_{c:\,\operatorname{parent}(c)=j}\operatorname{struct}(L_{:,c})\setminus\{c\},
\qquad \operatorname{parent}(j) = \min\{i>j : L_{ij}\neq 0\}.
$$

- **Why the analysis runs once.** This depends only on the pattern and
  the ordering, so the symbolic analysis runs once on the host.
- **The ordering decides the fill.** RCM minimises bandwidth, while AMD and
  nested dissection minimise fill (`O(N log N)` on planar meshes).

**Differentiating.**

$$
d\log|Q| = \operatorname{tr}(Q^{-1}\,dQ) = \sum_{(i,j)\in\operatorname{pattern}(Q)}\Sigma_{ij}\,dQ_{ij},
$$

and $\operatorname{pattern}(Q)\subseteq\operatorname{pattern}(L+L^\top)$
is exactly where Takahashi evaluates $\Sigma$. So the log-determinant's
gradient costs one selected-inversion sweep, and never forms $Q^{-1}$.

For $x = Q^{-1}b$: $\bar b = Q^{-1}\bar x$ and $\bar Q = -\bar b\,x^\top$,
restricted to the pattern.

```python
def symbolic_cholesky(
    pattern: SparsityPattern,
    *,
    ordering: Literal["rcm", "natural", "amd"] = "rcm",
    backend: Literal["jax", "cholmod"] = "jax",
) -> SymbolicCholesky: ...  # host, cacheable


class SparseCholeskyFactor(eqx.Module):
    values: Float[Array, " nnz_L"]
    symbolic: SymbolicCholesky = eqx.field(
        static=True
    )  # permutation, etree, CSC pattern of L, index plans

    def solve(self, b): ...  # P ᵀ L⁻ᵀ L⁻¹ P b
    def logdet(self): ...  # 2 Σ log L_jj
    def solve_lower_transpose(self, z): ...  # sampling: x = Pᵀ L⁻ᵀ z
    def selected_inverse(
        self,
    ) -> SparseOperator: ...  # (Q⁻¹) on pattern(L + Lᵀ), in the original order
    def diag_inv(self) -> Float[Array, " n"]: ...


def sparse_cholesky(
    op: SparseOperator, symbolic: SymbolicCholesky | None = None
) -> SparseCholeskyFactor: ...


class SparseCholeskySolver(AbstractSolverStrategy):  # solve / logdet through the factor
    ordering: ...
    backend: ...
```

- **Symbolic analysis** runs on the host, in NumPy and SciPy:
  - Reverse Cuthill–McKee ordering (`scipy.sparse.csgraph`) by default.
    `"amd"` is available through the CHOLMOD backend ([INLA open
    question 1](project-inla.md#open-questions)).
  - It computes the elimination tree, the column counts and the row
    pattern of `L`, and index plans for the numeric phase. Each column's
    row list and update list is padded to static shapes.
  - Results are cached per `SparsityPattern` hash.
- **Numeric factorisation** is left-looking, as a `lax.scan` over columns
  in elimination order. For each column `j`, it gathers
  `L[j:, k] · L[j, k]` for `k` in the row structure of `j`, subtracts,
  takes a square root and scales.
  - Everything is traced, so it `jit`s, and it `vmap`s over `values` for
    batched θ or datasets.
  - It is sequential over columns. Level scheduling over the elimination
    tree (independent subtrees in parallel) is a follow-up for GPU.
    Supernodes (dense blocks) are a follow-up for fill-heavy 3-D meshes.
- **Takahashi.** The backward recursion over columns uses
  `Z_ij = δ_ij / L_jj² − (1/L_jj) Σ_{k > j, k ∈ struct(L_{:,j})} L_kj Z_ki`.
  It evaluates exactly the entries of `pattern(L + Lᵀ)`, which is closed
  under the recursion. Cost is `O(Σ_j |struct(L_{:,j})|²)`: about the cost
  of the factorisation.
- **VJPs** (`_vjp.py`), so that `jax.grad` of `log π̃(θ | y)` is exact and
  cheap:
  - `logdet(Q)`: `Q̄ = Z` restricted to `pattern(Q) ⊆ pattern(L + Lᵀ)`.
    That is one Takahashi sweep, never `Q⁻¹`.
  - `solve(Q, b) = x`: `b̄ = Q⁻¹ x̄`, and `Q̄_ij = −b̄_i x_j` on
    `pattern(Q)`, symmetrised.
  - Sampling uses the triangular-solve adjoint.
- **CHOLMOD backend** (`gaussx[cholmod]`, via scikit-sparse):
  - `analyze` on the host gives the symbolic factor and AMD / METIS
    ordering. The numeric factorisation runs through
    `jax.pure_callback(..., vmap_method="sequential")` and returns `L`'s
    values on the simplicial pattern.
  - Takahashi, the VJPs and the solves are the same JAX code, so both
    backends give identical gradients.
  - It is CPU-only, and `vmap` over it is sequential.
- **Primitive dispatch:**
  - `cholesky(SparseOperator)` returns a `SparseCholeskyFactor`;
  - `solve`, `logdet` and `diag_inv` go through `SparseCholeskySolver`
    when the operator is tagged PSD and the caller passes the strategy.
    There is no size heuristic: the caller knows `N`.
  - `AutoSolver` keeps choosing CG for large PSD operators. That is the
    iterative backend of §3, fed by
    [G16](roadmap-gaussx.md).
- **Scale targets:**
  - 2-D SPDE meshes up to about 10⁵ nodes on CPU with RCM ordering.
  - Beyond that, AMD / METIS through CHOLMOD.
  - Beyond that, the iterative backend.
  - Stated in the docs with measured fill.


**Example.**

```python
# Matérn SPDE on a 40k-node coastline mesh: analyse once, factor for many κ
C, G = gx.fem_matrices(vertices, triangles)
sym = gx.symbolic_cholesky(gx.spde_precision(C, G, 1.0, 1.0, 2).pattern)  # host, once


def logdet(log_kappa):
    Q = gx.spde_precision(C, G, kappa=jnp.exp(log_kappa), tau=1.0, alpha=2)
    return gx.sparse_cholesky(Q, sym).logdet()


jax.vmap(logdet)(
    jnp.linspace(-1.0, 1.0, 16)
)  # 16 factorisations, one symbolic analysis
jax.grad(logdet)(0.0)  # d log|Q| / d log κ, through one Takahashi sweep
```

### 4.5 G5: `pseudo_logdet`

**The maths.** **The matrix-tree theorem.** For the weighted Laplacian of a connected
graph, every principal $(N-1)$-minor equals the weighted spanning-tree
count, and the pseudo-determinant is $N$ times it:

$$
\det L_{(-k,-k)} = \tau_w(G) = \sum_{T\ \text{spanning tree}}\ \prod_{e\in T}w_e\quad\forall k,
\qquad \operatorname{pdet}(L) = \prod_{\lambda_i>0}\lambda_i = N\,\tau_w(G).
$$

(The coefficient of $\lambda$ in $\det(\lambda I - L)$ is both
$\pm\prod_{\lambda_i>0}\lambda_i$ and the sum of the $N$ principal
$(N-1)$-minors.) So $\log\operatorname{pdet}(L)$ is one sparse Cholesky of
a matrix with one row and one column deleted.

**The null-space path.** It uses
$\operatorname{pdet}(A) = \det(A + NN^\top)$ for orthonormal $N$ spanning
$\ker A$: each zero eigenvalue becomes 1, and the range is untouched.

```python
def pseudo_logdet(
    operator: lx.AbstractLinearOperator,
    *,
    null_space: Float[Array, "N c"] | None = None,
    rcond: float | None = None,
    strategy: AbstractLogdetStrategy | None = None,
) -> Float[Array, ""]:
    """log of the product of the non-zero eigenvalues of a symmetric PSD operator."""
```

| Operator | Method |
|---|---|
| `KroneckerSum` | Factor eigenvalues, all pairwise sums, drop those `≤ rcond · max`, sum the logs |
| Dense | `eigvalsh` with the same threshold |
| Any operator with `null_space` given (orthonormal columns spanning the kernel) | `logdet(A + N Nᵀ)`, because adding `N Nᵀ` replaces each zero eigenvalue with 1. The sum is a lineax sum operator with matvec only, so `SLQLogdet` applies. The matrix determinant lemma does **not** apply (`A` is singular), so `LowRankUpdate`'s logdet cannot be reused |

For a connected graph Laplacian, `null_space = 1/√N`. kernellib computes
per-component null spaces (`graph_null_space`, K6); gaussx only takes the
matrix. Two more paths:

- **Laplacian cofactor.** For a weighted graph Laplacian, every cofactor
  equals the weighted spanning-tree count (the matrix-tree theorem). So
  `log|L|₊ = Σ_c (log n_c + logdet(L_c with one node removed))`, summing
  over connected components `c`. This takes one sparse or banded Cholesky
  per component (G3, G4), and keeps sparsity, unlike `L + N Nᵀ`, which is
  dense. Selected by `structure="laplacian"` (besag, rw1).
- **Constants are constant.** An intrinsic GMRF's `log|R|₊` does not
  depend on τ, so `IntrinsicGMRF(include_normalizer=False)` (the default)
  never needs it. When a marginal likelihood for model comparison does
  need it, it is computed once, on the cheapest path above, and cached.
  For RW1 on regular spacing the eigenvalues are known in closed form
  (`2 − 2 cos(πk/n)`, `k = 0 … n−1`), which is also a test oracle. RW2's
  structure `D₂ᵀD₂` is *not* the square of RW1's (the free ends differ), so
  it is checked against dense.

**Example.**

```python
# ICAR normalising constant, for comparing models by marginal likelihood (computed once, cached)
half_log_pdet = 0.5 * gx.pseudo_logdet(R, structure="laplacian")
# On a grid there is no factorisation at all: sum the logs of the non-zero λ^H_i + λ^W_j
half_log_pdet_grid = 0.5 * gx.pseudo_logdet(
    kl.grid_graph((512, 512)).laplacian_operator()
)
```

### 4.6 G6: `GaussianMRF`, `IntrinsicGMRF`

**The maths.** Three identities carry the whole class.

- **Perturbation–optimisation.** If $Q = \sum_k F_k^\top F_k$ and the
  $z_k \sim \mathcal N(0, I)$ are independent, then
  $r = \sum_k F_k^\top z_k$ has $\operatorname{Cov}(r) = Q$. So
  $x = Q^{-1}r$ has $\operatorname{Cov}(x) = Q^{-1}QQ^{-1} = Q^{-1}$: one
  solve per sample, and no factorisation.
- **Conditioning by kriging.** For $x\sim\mathcal N(\mu, Q^{-1})$ and the
  constraint $A_cx = e$,

  $$
  x^\ast = x - Q^{-1}A_c^\top\,(A_cQ^{-1}A_c^\top)^{-1}(A_cx - e)
  $$

  is an exact draw from $x\mid A_cx = e$ (a Gaussian projection; Rue &
  Held, eq. 2.30), with covariance
  $Q^{-1} - Q^{-1}A_c^\top S^{-1}A_cQ^{-1}$, where
  $S = A_cQ^{-1}A_c^\top$.
- **Observations.** For $y = Ax + \varepsilon$ with
  $\varepsilon\sim\mathcal N(0,\Lambda^{-1})$:

  $$
  Q_{\text{post}} = Q + A^\top\Lambda A,\qquad
  \mu_{\text{post}} = \mu + Q_{\text{post}}^{-1}A^\top\Lambda\,(y - A\mu).
  $$

  The posterior of a GMRF is a GMRF, on the pattern of G1.

Both classes are NumPyro `Distribution` subclasses following the
`MultivariateNormalPrecision` / `LGSSM` pattern: `pytree_data_fields`, the
optional `numpyro` extra, and einx throughout.

```python
class GaussianMRF(dist.Distribution):
    loc: Float[Array, " N"]
    precision: (
        lx.AbstractLinearOperator
    )  # SparseOperator | BlockTriDiag | (Sum)Kronecker | SpectralFunction | dense
    precision_factors: tuple[lx.AbstractLinearOperator, ...] | None = (
        None  # Q = Σ F_kᵀ F_k, for the CG sampler
    )
    solver: AbstractSolverStrategy | None = (
        None  # e.g. SparseCholeskySolver(symbolic=...)
    )
    logdet_strategy: AbstractLogdetStrategy | None = None
    log_det_precision: Float[Array, ""] | None = (
        None  # known log|Q|, overrides the strategy
    )

    def log_prob(self, x): ...
    def sample(self, key, sample_shape=()): ...
    def marginal_variances(self) -> Float[Array, " N"]: ...  # diag_inv dispatch
    def condition_on_observations(
        self, A, noise_precision, y
    ) -> GaussianMRF: ...  # Q + AᵀΛA; mean via one solve
    def condition_on_constraints(
        self, A_c, e
    ) -> ConstrainedGMRF: ...  # hard linear constraints


class IntrinsicGMRF(dist.Distribution):
    """Improper N(loc, (τ R)⁺), R the structure matrix, on the complement of null(R)."""

    loc: Float[Array, " N"]
    precision_scale: Float[Array, ""]  # τ
    structure: lx.AbstractLinearOperator  # R
    null_space: Float[Array, "N c"]
    precision_factors: tuple[lx.AbstractLinearOperator, ...] | None = (
        None  # R = Σ F_kᵀ F_k
    )
    constraint: Literal["none", "soft", "hard"] = "hard"
    soft_constraint_scale: float = 1e-3
    include_normalizer: bool = False  # the τ-free constant ½ log|R|₊
```

**Why factors.** Graph precision matrices come factored:

- the unnormalised Laplacian is `Bᵀ diag(w) B`, with `B` the signed
  incidence matrix (kernellib's `incidence_operator()`, already scaled by
  `√w`);
- proper CAR is `ρ·(incidence)ᵀ(incidence) + (1−ρ)·D`;
- Leroux is `ρ·(incidence)ᵀ(incidence) + (1−ρ)·I`.

Given the factors, sampling needs no Cholesky.

- **Sampling dispatch**, in this order:
  1. A factorisable precision (dense, `BlockTriDiag`, or `SparseOperator`
     with `SparseCholeskySolver`): `μ + L⁻ᵀz`.
  2. `Kronecker` / `KroneckerSum` / `SpectralFunction` structure: the
     factor-eigenvector path (`sqrt_matmul`).
  3. Otherwise, perturbation–optimisation (Papandreou & Yuille, 2010):
     draw `z_k ~ N(0, I)` for each factor, then solve
     `Q x = Σ_k F_kᵀ z_k` with one CG solve. That gives
     `x ~ N(0, Q⁻¹)`. For `IntrinsicGMRF`, the right-hand side lies in
     `range(R)`, so CG converges on the singular system, and the result is
     projected orthogonally to `null_space`.
- **`log_prob`:**
  - `GaussianMRF`: `½ log|Q| − ½ (x−μ)ᵀQ(x−μ) − (N/2) log 2π`.
    `log|Q|` is `log_det_precision` if given (callers that know it in
    closed form, such as pyrox-lgm's `CAR`), otherwise `logdet` through
    `solver` or `logdet_strategy` (Kronecker, banded, sparse Cholesky, or
    SLQ).
  - `IntrinsicGMRF`: `((N−c)/2) log τ − (τ/2)(x−μ)ᵀR(x−μ)`, plus
    `½ pseudo_logdet(R)` when `include_normalizer`. The `τ` term is always
    included, which is what makes `τ` identifiable under a hyperprior.
- **Constraints:**
  - `"hard"` (the default, and what INLA needs) is conditioning by
    kriging (Rue & Held 2005, §2.3.3):
    `x ← x − Q⁻¹A_cᵀ(A_c Q⁻¹ A_cᵀ)⁻¹(A_c x − e)`. That costs `c` extra
    solves with the existing factor. Marginal variances subtract
    `diag(Q⁻¹A_cᵀ S⁻¹ A_c Q⁻¹)`, where `S = A_c Q⁻¹ A_cᵀ`. An intrinsic
    structure is shifted by a tiny `ε·I` (relative to its mean diagonal)
    before kriging, as R-INLA does. `log_prob` is the constrained density
    (Rue & Held, eq. 2.30).
  - `"soft"` is a tight Gaussian on `null_spaceᵀx` (the Stan / Morris et
    al., 2019 approach). It is for NUTS users, because a hard constraint
    does not work with NUTS. pyrox-lgm's NumPyro face (P7) uses it.

**Example.**

```python
# Gap-fill a cloudy sea-surface-temperature snapshot on an H × W grid
prior = gx.GaussianMRF(
    jnp.zeros(H * W), gx.spde_precision_grid((H, W), kappa, tau, alpha=2)
)
A = gx.SparseOperator.from_coo(
    np.arange(n_clear), clear_idx, jnp.ones(n_clear), (n_clear, H * W)
)
post = prior.condition_on_observations(A, noise_precision=1 / 0.2**2, y=sst[clear_idx])
filled = einx.rearrange(
    "(h w) -> h w", post.loc, h=H
)  # the posterior is sparse (13-point stencil) → G4
sd = einx.rearrange("(h w) -> h w", jnp.sqrt(post.marginal_variances()), h=H)
draws = post.sample(key, (20,))  # 20 plausible gap-filled fields

# An ICAR field with a hard sum-to-zero constraint, one per island group
icar = gx.IntrinsicGMRF(jnp.zeros(N), 2.0, R, null_space=kl.graph_null_space(counties))
u = icar.sample(key)  # null_spaceᵀ u == 0 to machine precision
```

### 4.7 G7: precision builders, SPDE, FEM

**The maths.** Each builder is a quadratic form $\tfrac{\tau}{2}\|Dx\|^2$ for a sparse
difference operator $D$, so $Q = \tau D^\top D$:

- **RW1.** Increments $x_{i+1}-x_i\sim\mathcal N(0,\tau^{-1})$: $D = D_1$,
  $Q$ is tridiagonal, and $\ker Q = \operatorname{span}\{\mathbf 1\}$.
- **RW2.** $x_{i+1}-2x_i+x_{i-1}\sim\mathcal N(0,\tau^{-1})$: $D = D_2$,
  $Q$ is pentadiagonal, and $\ker Q = \operatorname{span}\{\mathbf 1, t\}$.
  This is the discrete cubic smoothing spline.
- **AR(1).** $x_t = \rho x_{t-1}+\varepsilon_t$, started from the
  stationary law:
  $Q = \frac{\tau}{1-\rho^2}\operatorname{tridiag}(-\rho,\ 1+\rho^2,\ -\rho)$,
  with 1 in the two corners, so the marginal precision is $\tau$.
- **Besag.**
  $\tfrac{\tau}{2}\sum_{i\sim j}w_{ij}(x_i-x_j)^2 = \tfrac{\tau}{2}x^\top Lx$,
  with $D$ the weighted incidence matrix (kernellib's
  `incidence_operator`).
- **SPDE** (Lindgren, Rue & Lindström, 2011).
  $(\kappa^2-\Delta)^{\alpha/2}(\tau x) = \mathcal W$ has Matérn
  covariance with $\nu = \alpha - d/2$.
  - **The weak form.** Expand $x = \sum_i w_i\psi_i$ in P1 hat
    functions and test the weak form against each $\psi_j$:

    $$
    C_{ij} = \int\psi_i\psi_j,\qquad G_{ij} = \int\nabla\psi_i\cdot\nabla\psi_j,\qquad
    K = \kappa^2 C + G,\qquad Q_1 = \tau^2 K,\quad Q_2 = \tau^2 K\tilde C^{-1}K,\quad
    Q_{\alpha} = K\tilde C^{-1}Q_{\alpha-2}\tilde C^{-1}K.
    $$

    $\tilde C$ is the lumped (diagonal) mass matrix, which keeps $Q$
    sparse.
  - **Per triangle**, with $e_i$ the edge opposite vertex $i$:
    $C^T = \tfrac{|T|}{12}(\mathbf 1\mathbf 1^\top + I)$,
    $\tilde C^T = \tfrac{|T|}{3}I$, and
    $G^T_{ij} = \tfrac{e_i\cdot e_j}{4|T|}$. These are vectorised with
    einx over triangles and scattered with `segment_sum`.
  - **Parameters.** The marginal variance is
    $\sigma^2 = \frac{\Gamma(\nu)}{\Gamma(\alpha)(4\pi)^{d/2}\kappa^{2\nu}\tau^2}$,
    and the practical range is $\rho = \sqrt{8\nu}/\kappa$.
- **SPDE on a grid.** With spacing $h$ and the right-triangle mesh,
  $\tilde C = h^2 I$ and $G$ is the 5-point Laplacian
  $L_H\oplus L_W$. So $K = h^2(\kappa^2 I + h^{-2}(L_H\oplus L_W))$ and
  $Q_\alpha = \tau^2 h^2(\kappa^2 I + h^{-2}(L_H\oplus L_W))^\alpha$: a
  function of one Kronecker sum.

All builders return operators, and none imports kernellib (a graph's
structure matrix arrives as an operator, from kernellib's
`Graph.laplacian_operator()`).

| Builder | Returns | Notes |
|---|---|---|
| `iid_precision(n, tau)` | `DiagonalLinearOperator` | |
| `rw1_structure(n, *, spacing=None, cyclic=False)` | `BlockTriDiag` (`d = 1`), null space `1` | Irregular spacing uses weights `1/h`. This is the path-graph Laplacian: a test against kernellib's `grid_graph((n,))` |
| `rw2_structure(n, *, cyclic=False)` | `BlockTriDiag` with `d = 2` blocks (pentadiagonal); null space `{1, t}` | Odd `n` is padded with one decoupled unit-precision node, and results are stripped. Irregular spacing (crw2, Lindgren & Rue 2008) is a follow-up |
| `ar1_precision(n, rho, tau)` | `BlockTriDiag` (`d = 1`) | Marginal-precision parameterisation `τ/(1−ρ²)`. AR(p) as `d = p` blocks is a follow-up |
| `besag_structure(laplacian_op)` | the operator, tagged PSD; the null space comes from kernellib (`graph_null_space`) | Just validation and tags |
| `generalized_variance_scale(structure, null_space)` | scalar `s` | The geometric mean of `diag(R⁺)` under the sum-to-zero constraint (Sørbye & Rue, 2014). Uses G3 (grids, exact), or the G4 selected inverse on `R + εI` followed by the kriging correction (graphs, exact and sparse). It replaces the earlier "dense or Hutchinson" `bym2_scaling` idea; very large graphs use G16 |
| `bym2_precision(structure_scaled, tau, phi)` | `SparseOperator` on the stacked `(b, u*)` | Riebler et al. (2016): `Q = [[τ/(1−φ)·I, −√(τφ)/(1−φ)·I], [−√(τφ)/(1−φ)·I, R* + φ/(1−φ)·I]]`. Sparse, with `R*`'s pattern plus two diagonals, fixed across `(τ, φ)` |
| `spde_precision(C_lumped, G, kappa, tau, alpha: int)` | `SparseOperator` | `K = κ² C̃ + G`; `Q₁ = τ²K`, `Q₂ = τ² K C̃⁻¹ K`, then `Q_α = K C̃⁻¹ Q_{α−2} C̃⁻¹ K`. The pattern (the α-ring neighbourhood) is computed once on the host |
| `spde_precision_grid(shape, kappa, tau, alpha, *, spacing=1.0, periodic=False)` | `SpectralFunction(KroneckerSum(L₁, L₂, …), f)` with `f(λ) = τ² h² (κ² + λ/h²)^α` | Exact solve, logdet, `diag_inv` and sampling via factor eigenvectors: `O(Σ n_k³ + N log N)`. Boundary effects are handled by domain extension, as with meshes, and documented |
| `matern_spde_params(range, sigma, nu, d)` | `(kappa, tau, alpha)` | `κ = √(8ν)/ρ`, `α = ν + d/2`, `τ` set for unit marginal variance (#155 appendix A) |
| `fem_matrices(vertices, triangles)` | `(C_lumped: Diagonal, G: SparseOperator)` | P1 on planar (`V × 2`) or surface (`V × 3`, e.g. an icosahedral sphere) triangulations. Per-triangle local matrices use einx; assembly is a `segment_sum` into a pattern built on the host from `triangles` |
| `fem_projector(vertices, triangles, points, *, triangle_index=None)` | `SparseOperator` `(n_obs, V)`, barycentric weights | Point location is blocked brute force for planar meshes, or the caller passes `triangle_index`. No mesh generation ([INLA non-goals](project-inla.md#non-goals)) |

`SpectralFunction` (`_operators/_spectral_function.py`) is the small
operator behind the grid SPDE: `f(A₁ ⊕ … ⊕ A_d)`, stored as factor
eigendecompositions plus `f`, with `solve`, `logdet`, `diag_inv`,
`sqrt_matmul` and `mv` all through the shared eigenvectors. It reuses
`_linalg/_eigen_factorization.py`.

Non-stationary `κ(s), τ(s)` (`spde_fem.md` §8) and rational non-integer
`α` (Bolin & Kirchner, 2020) are follow-ups, once a user needs them.


**Example.**

```python
# Temporal: a daily RW2 trend and an AR(1) nuisance
R_trend = gx.rw2_structure(365)  # null space {1, t}
Q_ar = gx.ar1_precision(365, rho=0.8, tau=10.0)
# Areal: BYM2 on a county graph from kernellib
R = kl.structure_matrix(counties)
s = gx.generalized_variance_scale(R, kl.graph_null_space(counties))
Q_bym2 = gx.bym2_precision(s * R, tau=1.5, phi=0.7)  # sparse (b, u*) stack
# Continuous space: Matérn ν = 1 on a coastline mesh, range 50 km, sd 2
C, G = gx.fem_matrices(vertices_km, triangles)
kappa, tau, alpha = gx.matern_spde_params(range=50.0, sigma=2.0, nu=1.0, d=2)
Q_spde = gx.spde_precision(C, G, kappa, tau, alpha)
A = gx.fem_projector(
    vertices_km, triangles, station_xy
)  # (n_stations, n_nodes), 3 non-zeros per row
# ...or on a ¼° global raster, with no mesh at all
Q_grid = gx.spde_precision_grid(
    (720, 1440), kappa, tau, alpha, spacing=0.25, periodic=True
)
```

### 4.8 G8: precision-form Laplace

**The maths.** Let $\eta = Ax$ and $f(\eta) = \sum_i\log p(y_i\mid\eta_i,\theta)$, with
$g = f'$ and $W = -\operatorname{diag}(f'')$. $W$ is diagonal because the
likelihood factorises over sites. The log posterior
$\ell(x) = f(Ax) - \tfrac12(x-\mu)^\top Q(x-\mu)$ has

$$
\nabla\ell = A^\top g - Q(x-\mu),\qquad -\nabla^2\ell = Q + A^\top WA =: H,
$$

so the Newton step $x_{t+1} = x_t + H^{-1}\nabla\ell$ rearranges to

$$
H\,x_{t+1} = Q\mu + A^\top(g + W\eta_t),
$$

one solve with a matrix of fixed pattern.

**The Laplace marginal.** Evaluate Bayes' identity at the mode $\hat x$,
with the Gaussian $\tilde\pi_G(x\mid y,\theta) = \mathcal N(\hat x, H^{-1})$
in the denominator:

$$
\log\tilde\pi(y\mid\theta) = f(A\hat x) - \tfrac12(\hat x-\mu)^\top Q(\hat x-\mu) + \tfrac12\log|Q| - \tfrac12\log|H|.
$$

**θ-gradients.** $\hat x(\theta)$ is defined implicitly by
$\nabla\ell(\hat x,\theta) = 0$, so
$\partial_\theta\hat x = H^{-1}\,\partial_\theta\nabla\ell$ (the implicit
function theorem). That is one more solve with the final factor. The
log-determinants differentiate through Takahashi (G4).

```python
class LaplaceResult(eqx.Module):
    mode: Float[Array, " N"]
    hessian: (
        lx.AbstractLinearOperator
    )  # H = Q + Aᵀ W A at the mode, same structure class as Q where possible
    factor: Any  # the factorisation of H, reused for marginals and sampling
    log_marginal: Float[Array, ""]  # log π̃(y | θ) up to θ-free constants
    n_iter: Int[Array, ""]
    converged: Bool[Array, ""]


def laplace_mode(
    prior: GaussianMRF | IntrinsicGMRF,
    likelihood: AbstractLikelihood,
    y,
    *,
    projector: lx.AbstractLinearOperator | None = None,
    offset=None,
    init=None,
    max_iter: int = 50,
    tol: float = 1e-8,
    damping: float = 1.0,
) -> LaplaceResult: ...
```

- **Newton on precision form.** At iterate `x_t`, with `η_t = A x_t + o`:
  - `g_t = ∂ log p(y|η)`, and `W_t = diag(max(−∂² log p(y|η), floor))`
    (reusing `newton_update`'s diagonal path);
  - `H_t = Q + Aᵀ W_t A`, via `SparseOperator.congruence` on the fixed
    pattern;
  - solve `H_t x_{t+1} = Q μ + Aᵀ(g_t + W_t η_t − W_t o)`, one
    factorisation per step on a symbolic analysis shared by every step
    and every θ;
  - with damping or a backtracking line search, as pyrox-gp's
    `LaplaceInference` does.
- **Structure.** When `A` is the identity or a row selection, and `Q` is
  `BlockTriDiag` or `SpectralFunction`, `H` keeps banded or diagonal-plus
  structure where it can: `BlockTriDiag + diag` stays `BlockTriDiag`. A
  `SpectralFunction` plus a non-constant diagonal does not, and falls back
  to sparse or iterative.
- **Implicit differentiation.** The mode satisfies
  `∇ₓ[log p(y|Ax) − ½(x−μ)ᵀQ(x−μ)] = 0`. The gradient with respect to θ
  flows through `lax.custom_root` with the Newton system as the tangent
  solve: one extra solve with the final factor (Margossian, 2023).
- **Log marginal:**
  `log p(y|x̂) − ½(x̂−μ)ᵀQ(x̂−μ) + ½ log|Q| − ½ log|H|`, with the
  constraint corrections for intrinsic priors (Rue et al., 2009, eq. 3).
  Its θ-gradient is exact through the G4 VJPs.
- **Gaussian likelihood.** One step, and exact.
- **Out of scope:** likelihoods with a non-diagonal Hessian in `η` (outside
  the LGM class; use NumPyro), and multi-output latents per site.


**Example.**

```python
# Poisson counts with an RW2 seasonal effect; θ = log τ
def log_marginal(log_tau):
    prior = gx.IntrinsicGMRF(
        jnp.zeros(365),
        jnp.exp(log_tau),
        gx.rw2_structure(365),
        null_space=rw2_null,
        constraint="hard",
    )
    return gx.laplace_mode(
        prior, gx.PoissonLikelihood(), counts
    ).log_marginal  # H stays BlockTriDiag


value, grad = jax.value_and_grad(log_marginal)(0.0)  # exact: implicit diff + Takahashi
```

### 4.9 G9: θ-designs (`_quadrature/_theta_design.py`)

**The maths.** Take $\theta^\ast$ as the mode of $\log\tilde\pi(\theta\mid y)$, with
$-\nabla^2\log\tilde\pi(\theta^\ast) = V\Lambda V^\top$. Reparameterise
$\theta(z) = \theta^\ast + V\Lambda^{-1/2}z$, so that the posterior is
approximately standard normal in $z$. Integration then needs only a few
points in $z$:

$$
\tilde\pi(x_i\mid y) \approx \sum_{k=1}^K \tilde\pi(x_i\mid\theta_k, y)\ \tilde\pi(\theta_k\mid y)\,\Delta_k .
$$

**CCD** uses a resolution-V fractional factorial (about $2^{m-p}$
points), $2m$ axial points, and the centre, all non-centre points on the
sphere of radius $f_0\sqrt m$. That is $O(m^2)$ points, for example 15 at
$m = 3$ and 27 at $m = 5$, against $3^m$ for a grid. It integrates a
quadratic log-density well, because the design is rotatable (Rue et al.,
2009, §6.5).

```python
def theta_design(
    log_post: Callable[[Array], Array],
    mode: Float[Array, " m"],
    *,
    method: Literal["eb", "grid", "ccd"] = "ccd",
    hessian=None,
    grid_step: float = 1.0,
    grid_threshold: float = 2.5,
    ccd_f0: float = 1.1,
) -> tuple[Float[Array, "K m"], Float[Array, " K"]]:  # points, normalised log-weights
    ...
```

- **The z-parameterisation.** `θ = θ* + V Λ^{-1/2} z`, where
  `−∇²log π̃(θ*) = V Λ Vᵀ`. The Hessian comes from `jax.hessian` of
  `log_post` if not given. That is exact through G8's implicit
  differentiation: no smart-gradient finite differences.
- **`"eb"`**: the mode alone.
- **`"grid"`**: axis-wise exploration in steps of `grid_step` in `z`,
  keeping points within `grid_threshold` log-units of the mode (Rue et
  al., 2009, §6.5). The default for `m ≤ 2`.
- **`"ccd"`**: the centre, `2m` axial points, and a resolution-V
  fractional factorial `2^{m−p}`. All non-centre points lie on the sphere
  of radius `f0·√m` in `z`, and the centre and shell weights come from Rue
  et al. (2009), §6.5. The default for `m > 2`.
- Weights are corrected by the evaluated `log_post` at each point, then
  normalised.
- Asymmetric per-direction scaling (R-INLA's "stdev.corr") is a
  follow-up.
- The design is pure: `log_post` is any callable. The pyrox driver does
  mode-finding (L-BFGS on `jax.grad`).


**Example.**

```python
theta_star = lbfgs_minimise(
    lambda th: -log_post(th), jnp.zeros(3)
)  # jax.grad through G8
pts, logw = gx.theta_design(log_post, theta_star, method="ccd")  # (15, 3) and (15,)
fits = jax.vmap(lambda th: gx.laplace_mode(prior_at(th), lik, y))(
    pts
)  # one symbolic analysis, batched values
post_mean = einx.dot("k, k n -> n", jnp.exp(logw), fits.mode)
```

### 4.10 G10: low-rank VB mean correction

**The maths.** Keep the Laplace covariance, and move the mean within a $p$-dimensional
subspace: $q_\delta = \mathcal N(\hat x + H^{-1}S\delta,\ H^{-1})$, with
$S \in \mathbb R^{N\times p}$. Maximise

$$
\mathcal L(\delta) = \sum_i\mathbb E_{\eta_i\sim\mathcal N(m_i(\delta),\,v_i)}\big[\log p(y_i\mid\eta_i)\big]
- \tfrac12\big(\bar x(\delta)-\mu\big)^\top Q\big(\bar x(\delta)-\mu\big),
$$

where $\bar x(\delta) = \hat x + H^{-1}S\delta$,
$m(\delta) = A\bar x(\delta)$, and $v_i = [AH^{-1}A^\top]_{ii}$. The
KL term's covariance part does not depend on $\delta$.

- **Cost.** The expectations are 1-D Gauss–Hermite. The derivatives in
  $\delta$ are $p$-dimensional, so a few Newton steps cost $p$ solves with
  the existing factor.
- **What it fixes.** The Laplace approximation's main failure is a biased
  mean under skewed likelihoods (Poisson with small counts, Bernoulli).
  This removes most of that bias at almost no cost.

```python
def vb_mean_correction(
    result: LaplaceResult,
    prior,
    likelihood,
    y,
    *,
    projector=None,
    subspace: Int[Array, " p"] | Float[Array, "N p"],
    n_iter: int = 5,
    integrator=GaussHermiteIntegrator(order=20),
) -> Float[Array, " N"]: ...
```

This is Van Niekerk & Rue (2024). It keeps the Laplace covariance `H⁻¹`
and corrects the mean, `x̂ + H⁻¹ S δ`, for a `p`-dimensional shift `δ`
in the span of `subspace`. By default that span is the fixed effects plus
a few latent directions. `δ` maximises the variational objective
`E_q[log p(y | η)] − KL(q ‖ prior)`.

- **Predictor variances.** The expectations need the marginal variances
  of `η = A x`: `diag(A H⁻¹ Aᵀ)`. For a sparse `A` whose rows touch only
  nodes that are mutually adjacent in `Q` (true for FEM projectors, where
  a row's nodes share a triangle, and for identity or selection
  projectors), every needed entry lies in `pattern(L + Lᵀ)`. So the
  variances come from the same Takahashi sweep, not from solves.
- **Cost.** `p` solves with the existing factor per iteration, plus
  per-site Gauss–Hermite expectations. About free next to the Laplace fit.


**Example.**

```python
res = gx.laplace_mode(prior, gx.BernoulliLikelihood(), detected, projector=A)
mean_vb = gx.vb_mean_correction(
    res,
    prior,
    gx.BernoulliLikelihood(),
    detected,
    projector=A,
    subspace=fixed_effect_idx,
)
```

---

# Part B — randomized linear algebra

## 5. API

### 5.1 G11: sketching operators

**The maths.** A matrix $S\in\mathbb R^{d\times m}$ is an **$\varepsilon$-subspace
embedding** for $\operatorname{range}(A)$, with
$A\in\mathbb R^{m\times n}$, if

$$
(1-\varepsilon)\|Ax\| \le \|SAx\| \le (1+\varepsilon)\|Ax\|\qquad\forall x .
$$

Sketch sizes that suffice with constant failure probability, and the cost
of applying each:

| Sketch | Size `d` | Cost to apply |
|---|---|---|
| Gaussian | $O(n/\varepsilon^2)$ | $O(dmn)$ |
| Sparse sign, $O(\log n)$ non-zeros per column (Cohen, 2016) | $O(n\log n/\varepsilon^2)$ | $O(\text{nnz}\cdot n)$ |
| SRHT (Tropp, 2011) | $O((n+\log m)\log n/\varepsilon^2)$ | $O(mn\log m)$ |

The SRHT works because the randomised Hadamard transform $HD$ spreads
every vector's mass evenly over coordinates. It flattens the leverage, so
uniform row sampling afterwards is safe.

A sketch `S ∈ ℝ^{d×m}` is **sampled once** and then applied as many times
as needed. #156's strawman took a key in `materialize(key)`, which makes it
easy to apply one `S` and the transpose of a different one. Here the random
draws live in the module:

```python
class AbstractSketch(eqx.Module):
    in_size: AbstractVar[int]  # m (static)
    out_size: AbstractVar[int]  # d (static)

    @abc.abstractmethod
    def apply(self, A: Float[Array, "m *rest"]) -> Float[Array, "d *rest"]: ...  # S A
    @abc.abstractmethod
    def apply_transpose(
        self, Y: Float[Array, "d *rest"]
    ) -> Float[Array, "m *rest"]: ...  # Sᵀ Y

    def sketch_operator(
        self, op: lx.AbstractLinearOperator
    ) -> Float[Array, "d n"]: ...  # S A for a matrix-free A
    def as_operator(self) -> lx.AbstractLinearOperator: ...  # S as a (d, m) operator


class GaussianSketch(AbstractSketch):  # entries N(0, 1/d); stores the (d, m) matrix
    @classmethod
    def sample(cls, key, d: int, m: int) -> GaussianSketch: ...


class OrthonormalSketch(AbstractSketch):  # Gaussian with orthonormal rows (thin QR)
    ...


class SparseSignSketch(
    AbstractSketch
):  # nnz entries ±1/√nnz per column, at random rows
    rows: Int[Array, "nnz m"]
    signs: Float[Array, "nnz m"]

    @classmethod
    def sample(cls, key, d: int, m: int, *, nnz: int = 8) -> SparseSignSketch: ...


class SRHTSketch(
    AbstractSketch
):  # S = √(m₂/d) · R H D P, with m padded to m₂ = 2^⌈log₂ m⌉
    permutation: Int[Array, " m"]
    signs: Float[Array, " m"]
    rows: Int[Array, " d"]
    ...


class RowSamplingSketch(AbstractSketch):  # rows i_k ~ p, weights 1/√(d p_{i_k})
    rows: Int[Array, " d"]
    weights: Float[Array, " d"]

    @classmethod
    def sample(
        cls, key, d: int, m: int, *, probabilities=None
    ) -> RowSamplingSketch: ...
```

- **Sparse-sign `apply`.** A single `segment_sum` of
  `signs · A[column]` into `rows`, so no sparse-matrix library is needed.
  Its cost is `O(nnz · m · ncols)`.
- **SRHT `apply`.** Pad, apply signs and the permutation, apply
  `hadamard_transform` along the leading axis, take the rows, rescale. Its
  cost is `O(m log m · ncols)`.
- **`sketch_operator`** forms `S A` with `d` transpose-matvecs of `A`, as
  `(Aᵀ Sᵀ)ᵀ`, vmapped. For a dense `A`, it calls `apply` directly.
- **`hadamard_transform(x)`** moves here unchanged from kernellib (the
  unnormalised Sylvester form, O(d log d), power-of-two length). The
  transform is kernel-free linear algebra, and gaussx cannot import
  kernellib.


**Example.**

```python
# Sketch a tall Jacobian (10⁶ residuals × 200 parameters) down to 800 rows
S = gx.SparseSignSketch.sample(key, d=800, m=1_000_000, nnz=8)
SJ = S.sketch_operator(J_op)  # (800, 200), matrix-free
sv = jnp.linalg.svd(SJ, compute_uv=False)  # J's singular values, within (1 ± ε)
```

### 5.2 G12: range finder, QB, SVD, eigh

**The maths.** For a Gaussian test matrix $\Omega\in\mathbb R^{n\times(k+p)}$ and
$Q = \operatorname{orth}(A\Omega)$ (Halko, Martinsson & Tropp, 2011,
Thm 10.6):

$$
\mathbb E\,\|A - QQ^\top A\|_2 \le \Big(1+\sqrt{\tfrac{k}{p-1}}\Big)\sigma_{k+1} + \frac{e\sqrt{k+p}}{p}\Big(\sum_{j>k}\sigma_j^2\Big)^{1/2}.
$$

- **The tail term** is what hurts for slowly decaying spectra (Matérn-½
  Gram matrices, most geophysical fields).
- **Power iterations fix it.** $q$ of them apply the same bound to
  $(AA^\top)^qA$, whose singular values are $\sigma_j^{2q+1}$, which
  crushes the tail.
- **The price** is $2q$ more passes over $A$, with re-orthonormalisation
  after each so that small directions are not lost to round-off.

```python
def range_finder(
    op, rank, *, oversample=10, n_power_iter=2, sketch=None, key=None
) -> Float[Array, "m l"]: ...
def qb(
    op, rank, *, oversample=10, n_power_iter=2, key=None
) -> tuple[Float[Array, "m l"], Float[Array, "l n"]]: ...
def randomized_svd(
    op, rank, *, oversample=10, n_power_iter=2, key=None
) -> tuple[U, s, Vt]: ...
def randomized_eigh(
    op, rank, *, oversample=10, n_power_iter=2, which="largest", key=None
) -> tuple[vals, vecs]: ...
```

- **`range_finder`** is Halko, Martinsson & Tropp (2011), Algorithm 4.4
  (randomized subspace iteration):
  - `Y = AΩ` with `ℓ = rank + oversample`, then `Q = qr(Y)`;
  - each power step does `Q̂ = qr(AᵀQ)` then `Q = qr(AQ̂)`, orthonormalising
    every half-step;
  - `Ω` is Gaussian by default. Pass `sketch=` to use a sparse-sign or SRHT
    test matrix instead, applied as its transpose (`n × ℓ`).
- **`qb`** returns `Q` and `B = QᵀA`.
- **`randomized_svd`** is the SVD of `B`, lifted as `U = Q·U_B`, then
  truncated.
- **`randomized_eigh`** is for symmetric, possibly indefinite, operators.
  It computes the Rayleigh–Ritz matrix `QᵀAQ`, runs `eigh` on it, and keeps
  the top `rank` by `which` (`"largest"`, or `"magnitude"`). For PSD
  operators, `randomized_nystrom` is strictly more accurate for the same
  number of matvecs (Tropp et al., 2017), and the docstring says so.
- **Primitive dispatch.**
  - `svd(op, rank=k, method="randomized", ...)` and
    `eig(op, rank=k, method="randomized", ...)` route here.
  - `method="lanczos"` stays the default ([RandNLA open question 1](project-rnla.md#open-questions)).
  - Docstrings recommend `n_power_iter ≥ 2` for slowly decaying spectra
    (Matérn-½, most geophysical kernels).
  - They also state that randomized methods target the **top** of the
    spectrum (#413).


**Example.**

```python
# 50 EOFs of a (100k pixels × 3650 days) anomaly matrix, available only as a matvec
U, s, Vt = gx.randomized_svd(anomalies_op, 50, oversample=10, n_power_iter=2, key=key)
eofs, pcs = U, einx.multiply("k, k t -> k t", s, Vt)
```

### 5.3 G13: randomized Nyström and its preconditioner (closes #354)

**The maths.** For a PSD $A$ and a test matrix $\Omega$, the **Nyström approximation**

$$
\hat A = (A\Omega)\,(\Omega^\top A\Omega)^{+}\,(A\Omega)^\top
$$

satisfies $0\preceq\hat A\preceq A$ and needs one pass ($\ell$ matvecs).
For the same $\ell$ it is more accurate than the Rayleigh–Ritz
approximation $QQ^\top AQQ^\top$ that today's preconditioner uses (Tropp
et al., 2017). The shift $\nu$ in the algorithm only stabilises the small
Cholesky.

**Why the preconditioner works** (Frangella, Tropp & Udell, 2023). With
$P^{-1}$ as below:

- on $\operatorname{range}(U)$, $P^{-1}(A+\mu I) \approx \hat\lambda_\ell + \mu$;
- on the complement, it is $A + \mu I$ itself, with eigenvalues in
  $[\mu,\ \lambda_{\ell+1}+\mu]$.

So

$$
\kappa\big(P^{-1/2}(A+\mu I)P^{-1/2}\big) \le \frac{\hat\lambda_\ell + \mu + \|A-\hat A\|}{\mu},
$$

which is $O(1)$ once $\hat\lambda_\ell \lesssim \mu$, that is once
$\ell \gtrsim d_{\text{eff}}(\mu) = \sum_i \lambda_i/(\lambda_i+\mu)$.

```python
def randomized_nystrom(op, rank, *, oversample=0, key=None) -> LowRankUpdate: ...
```

This is Tropp, Yurtsever, Udell & Cevher (2017), Algorithm 3. Given a PSD
`A`:

1. `Ω = qr(randn(n, ℓ))`.
2. `Y = AΩ`, and the shift `ν = √n · eps · ‖Y‖₂`.
3. `Y_ν = Y + νΩ`, `C = chol(ΩᵀY_ν)`, `B = Y_ν C⁻¹`.
4. `U, Σ, _ = svd(B)`, `Λ̂ = max(Σ² − ν, 0)`.

It returns
`LowRankUpdate(base=zeros, U=U, d=Λ̂, V=U, orthonormal=True)`, so
`solve(Â + σ²I)`, `logdet(Â + σ²I)` and the rest dispatch through the
existing Woodbury rules.

The rewritten `NystromPreconditioner` follows Frangella, Tropp & Udell
(2023):

```python
class NystromPreconditioner(AbstractPreconditioner):
    basis: Float[Array, "n l"]  # U
    eigenvalues: Float[Array, " l"]  # Λ̂, descending
    shift: Float[Array, ""]  # μ

    @classmethod
    def from_operator(
        cls, operator, rank, *, shift, oversample=0, key=None
    ) -> NystromPreconditioner:
        """operator is the PSD part A (e.g. K), NOT A + μI; shift is μ (e.g. σ²)."""

    def as_operator(self, operator=None) -> lx.AbstractLinearOperator:
        ...
        # P⁻¹ x = (λ̂_ℓ + μ) U (Λ̂ + μI)⁻¹ Uᵀx + (x − U Uᵀx)
```

**Guarantee.** With rank `ℓ = 2⌈1.5 · d_eff(μ)⌉ + 1`, the preconditioned
`A + μI` has expected condition number below 28 (Frangella, Tropp & Udell,
2023). Here `d_eff(μ) = tr(A(A + μI)⁻¹)` is the effective dimension. So CG
converges in a number of iterations that does not grow with `n`, which is
exactly what #354 measured failing for the current Rayleigh–Ritz version.

**Covariance form only.** The preconditioner approximates the *top* of
`A`'s spectrum, which is right for `K + σ²I`. It is not a good
preconditioner for a precision-form GMRF system `Q + AᵀWA`, whose hard
directions are its smallest eigenvalues. There, use Part A's
exact factorisations or Jacobi-preconditioned CG
(§3). The docstring says so.

**Breaking change.** `shift` becomes required, and `operator` now means
the PSD part only, not the system operator. This is accepted pre-1.0,
with a CHANGELOG entry, following the same precedent as gaussx 0.2.0.

**`PreconditionedCGSolver` wiring.** The solver must pass `K` and `σ²`
separately wherever it knows them (`SumOperator(K, σ²I)`). It must never
build from `K + σ²I` and then add `σ²` again, which is the #345 bug. If the
operator arrives unsplit, the solver raises and asks for an explicit
preconditioner.

The Rayleigh–Ritz algorithm survives as `randomized_eigh(n_power_iter=0)`,
which is what it always was.


**Example.**

```python
# GP regression with n = 200k and a Matérn-3/2 kernel: preconditioned CG instead of Cholesky
K_op = kl.to_operator(kernel, X, implicit=True)
P = gx.NystromPreconditioner.from_operator(K_op, rank=500, shift=noise_var, key=key)
A_op = K_op + lx.DiagonalLinearOperator(jnp.full(n, noise_var))
alpha = gx.PreconditionedCGSolver(preconditioner=P).solve(
    A_op, y
)  # tens of iterations, not thousands
```

### 5.4 G14: randomly pivoted Cholesky

**The maths.** RPCholesky picks pivot $s$ with probability proportional to the residual
diagonal, $d_s = [A - FF^\top]_{ss}$: the variance not yet explained. It
samples the diagonal of the Schur complement, a cheap stand-in for a
determinantal point process.

**The guarantee** (Chen, Epperly, Tropp & Webber, 2023). With
$k \ge r/\varepsilon + r\log\!\big(1/(\varepsilon\eta)\big)$ pivots,

$$
\mathbb E\,\operatorname{tr}(A - FF^\top) \le (1+\varepsilon)\,\operatorname{tr}\big(A - [\![A]\!]_r\big),
\qquad \eta = \operatorname{tr}\big(A - [\![A]\!]_r\big)/\operatorname{tr}A,
$$

where $[\![A]\!]_r$ is the best rank-$r$ approximation.

- **The alternatives.** Greedy pivoting has no such guarantee and chases
  outliers. Uniform sampling ignores the geometry.
- **Cost.** $k$ columns, that is $O(Nk)$ kernel evaluations and
  $O(Nk^2)$ flops.

```python
def rp_cholesky(
    diagonal, column, rank, *, pivoting="random", block_size=1, key=None
) -> tuple[Float[Array, "N k"], Int[Array, " k"]]: ...
```

This is Chen, Epperly, Tropp & Webber (2023).

- **Pivot rule.**
  - `"random"`: at step `i`, sample the pivot `s ∝ max(d_res, 0)` with
    `jax.random.categorical(log d_res)`, where `d_res` is the residual
    diagonal.
  - `"greedy"`: take `argmax(d_res)`, which is today's behaviour.
- **Update.** `g = column(s) − F F[s]ᵀ`, `F[:, i] = g/√g_s`, then
  `d_res −= F[:, i]²`.
- **Guard.** The existing `pstrf`-style guard is kept. Once the residual
  is exhausted, surplus columns are exactly zero.
- **Pivots are returned.** kernellib needs them as landmark indices, and
  `F Fᵀ = A[:, S] A[S, S]⁺ A[S, :]` is the column Nyström approximation on
  those pivots.
- **Blocks.** `block_size > 1` is the accelerated blocked variant (Epperly,
  Tropp & Webber, 2024). It is a follow-up if the one-column loop is too
  slow. It samples `b` pivots, then rejection-cleans.

`guarded_pivoted_cholesky` becomes a thin wrapper, `pivoting="greedy"`,
dropping the pivots. Its behaviour is bit-identical, including gh-236 and
gh-237.

`PartialCholeskyPreconditioner` gains `pivoting` and `key`, plus the
build-once `from_operator(operator, rank, *, shift, pivoting, key)` from
#371, with the same `shift` semantics as G13. So #345 cannot recur.


**Example.**

```python
# 10⁶ points: pick 1000 landmarks without ever forming K
F, pivots = gx.rp_cholesky(
    kernel.diag(X), lambda j: kernel.pairwise(X, X[j][None])[:, 0], rank=1000, key=key
)
Z = X[pivots]  # landmarks for Nyström / Falkon / SVGP inducing points
```

### 5.5 G15: sketch-and-precondition least squares

**The maths.** If $S$ is an $\varepsilon$-embedding for $\operatorname{range}(A)$ and
$SA = QR$, set $M = R^{-1}$. Then $\|SAMy\| = \|Qy\| = \|y\|$, and so

$$
\|AMy\| \in \Big[\tfrac{1}{1+\varepsilon},\ \tfrac{1}{1-\varepsilon}\Big]\,\|y\|
\quad\Longrightarrow\quad \kappa(AM) \le \frac{1+\varepsilon}{1-\varepsilon}.
$$

- **Iteration count.** LSMR / LSQR contract the error by
  $\frac{\kappa-1}{\kappa+1}$ per iteration. At $\varepsilon = 0.2$
  ($d \approx 4n$) that is $\kappa\le1.5$ and a factor of 5 per iteration,
  so about 15 iterations reach 1e-10, however large $m$ is.
- **Ridge.** Stack $\tilde A = [A;\ \delta I]$. Only the $A$ block needs
  sketching.

```python
class SketchAndPrecondLSMR(AbstractSolverStrategy):
    """min ‖Ax − b‖² + δ²‖x‖² for tall A (m ≫ n), in O(log 1/η) LSMR steps."""

    sketch: Literal["sparse_sign", "srht", "gaussian"] = eqx.field(
        static=True, default="sparse_sign"
    )
    sampling_factor: float = eqx.field(static=True, default=4.0)  # d = ⌈γ n⌉
    nnz: int = eqx.field(static=True, default=8)
    damp: float = eqx.field(static=True, default=0.0)  # δ
    atol: float = ...
    btol: float = ...
    maxiter: int = 100
    seed: int = eqx.field(static=True, default=0)


def sketch_and_solve(
    operator, vector, *, sketch: AbstractSketch, damp=0.0
) -> Float[Array, " n"]: ...
```

`solve(A, b)` works on the augmented system `Ã = [A; δI]`,
`b̃ = [b; 0]`, which is the plain system when `δ = 0`:

1. Sample `S` (`d × m`) and form `SA` (`d × n`) with `sketch_operator`.
2. QR of `[SA; δI] = QR`, so the preconditioner is `M = R⁻¹`, applied by
   a triangular solve.
3. Warm start: `x₀ = R⁻¹ Q_{1:d}ᵀ (S b)`. This is sketch-and-solve, for
   free.
4. Solve `min ‖Ã M y − (b̃ − Ã x₀)‖` with **undamped** `LSMRSolver`
   (lineax's path, with its implicit differentiation) on the composed
   operator `Ã ∘ M`. Then `x = x₀ + M y`.

- **Convergence.** A subspace embedding with distortion `ε` gives
  `κ(ÃM) ≤ (1+ε)/(1−ε)`, so iteration counts do not depend on `m`
  (Blendenpik; LSRN; RandLAPACK's `SPO`). Right preconditioning is just
  operator composition, so no LSQR is needed.
- **`logdet`** is not meaningful for a rectangular least-squares strategy.
  It delegates to `LSMRSolver`'s SLQ path for protocol completeness, and
  the docstring says to use a square-system strategy instead.
- **Scope.** The QR of the `d × n` sketch is dense, `O(d n²)`, so this
  targets `n ≲ 10⁴`. For larger `n`, solve the normal equations
  `(AᵀA + δ²I)x = Aᵀb` with `PreconditionedCGSolver` and G13's
  `NystromPreconditioner`, built on `AᵀA` with `shift=δ²` (Frangella,
  Tropp & Udell's ridge setting).
- **`sketch_and_solve`** returns `R⁻¹ Qᵀ S b` alone. It is a function, not
  a strategy, because it is an approximation ([RandNLA open question 3](project-rnla.md#open-questions)).


**Example.**

```python
# Gauss–Newton step for a plume inversion: 10⁶ pixels, 300 parameters, Tikhonov damping δ
step = gx.SketchAndPrecondLSMR(sampling_factor=4.0, damp=delta).solve(J_op, -residual)
```

### 5.6 G16: `diag_inv(method="xdiag")`

**The maths.** Split $\operatorname{diag}(B)$ for $B = A^{-1}$ into a low-rank part and a
remainder. With $Q = \operatorname{orth}(B\Omega)$, obtained from $k$
solves,

$$
\operatorname{diag}(B) = \operatorname{diag}(QQ^\top B) + \operatorname{diag}\big((I-QQ^\top)B\big).
$$

- **The first term is exact.**
- **The second is estimated** by Hutchinson,
  $\operatorname{diag}(M)\approx\frac1s\sum_i\omega_i\odot M\omega_i$.
- **XDiag reuses every probe** both to build $Q$ and to estimate the
  remainder (leave-one-out "exchangeability"), so each solve counts
  twice.
- **Why this suits GMRFs.** For a GMRF, $B = \Sigma$ has a few
  large-variance directions (smooth, large-scale modes), where Hutchinson
  alone is noisy. The low-rank part captures exactly those.

A variance-reduced estimator of `diag(A⁻¹)` (Epperly, Tropp & Webber,
2024). It combines:

- a low-rank part from a randomized range finder (G12) of `A⁻¹`, applied
  through solves with `A`;
- exchangeable Hutchinson on the remainder.

It is Part A's marginal-variance fallback for precision matrices too large
to factor (§3). It is also how `generalized_variance_scale` (G7) scales
to such graphs.

**Example.**

```python
# Marginal sd on a 2M-node global mesh, where Cholesky fill is too large
sd = jnp.sqrt(
    gx.diag_inv(
        H,
        method="xdiag",
        num_probes=64,
        key=key,
        solver=gx.PreconditionedCGSolver(preconditioner=gx.JacobiPreconditioner()),
    )
)
```

### 5.7 G17: tier 2, on demand

**The maths.** **Hutch++** splits
$\operatorname{tr}(A) = \operatorname{tr}(Q^\top AQ) + \operatorname{tr}\big((I-QQ^\top)A(I-QQ^\top)\big)$,
with $Q$ from a range finder. That reaches relative error $\varepsilon$
in $O(1/\varepsilon)$ matvecs, against Hutchinson's $O(1/\varepsilon^2)$.

**`NystromLogdet`** uses
$\log|A+\mu I| = \log|P| + \log|P^{-1/2}(A+\mu I)P^{-1/2}|$. The first
term is exact from the Nyström factors. The second is SLQ on a
near-identity operator, so its variance is small.

| Symbol | What | Reference |
|---|---|---|
| `trace(op, method="hutchpp" \| "xtrace")` | Low-rank deflation plus Hutchinson on the remainder: `O(1/ε)` matvecs against Hutchinson's `O(1/ε²)` | Meyer, Musco, Musco & Woodruff (2021); Epperly, Tropp & Webber (2024) |
| `NystromLogdet` | `log|A + μI| = log|P| + log|P^{-½}(A+μI)P^{-½}|`: the first term exact from G13's factors, the second by SLQ on a near-identity operator, so SLQ's variance collapses. **Covariance form only** (P4's exact-GP recipe), not for GMRF precision matrices | Wenger et al. (2022) |
| `column_id(op, rank, key)`, `cur(op, rank, key)` | Interpolative and CUR decompositions from a sketch plus pivoted QR, returning actual column and row indices | Voronin & Martinsson (2017) |

---

## 6. Phases and tests

Reference problems for G6–G10 are those from gaussx#155's Phase 1. Their
R-INLA posterior summaries are generated **offline** by an R script under
`scripts/golden/inla/` and committed as JSON fixtures. CI never runs R.

| Phase | Tests |
|---|---|
| G1 | `mv`, `as_matrix`, `transpose` and `diag` against dense; CG `solve` against dense on a PSD Laplacian plus a shift; SLQ `logdet` within its own error bound (slow); Lanczos `eig(rank=)` against dense; `jit` and `grad` through `values`; `vmap` over `values`; pattern hashing is stable across processes; `add_diagonal` and `congruence` preserve the pattern; the BCOO-vs-`segment_sum` benchmark (slow) |
| G2 | Agreement with `scipy.linalg.eigh(a, b)` for a positive definite `B`; the diagonal-`B` path equals kernellib's current `_smallest_generalized` output; a singular `B` drops directions and warns; `rank=` with Lanczos against dense |
| G3 | The `BlockTriDiag` selected inverse equals the dense inverse's band on random SPD blocks (`d ∈ {1, 2, 3}`); `KroneckerSum` `diag_inv` equals dense, including `pinv=True` on a singular grid Laplacian; #344 regression |
| G4 | The factor equals dense Cholesky of the permuted matrix; Takahashi equals the dense inverse on `pattern(L + Lᵀ)`; logdet and solve VJPs match `jax.grad` through dense; `vmap` over values; RCM fill on a reference 2-D mesh is recorded and bounded; CHOLMOD and JAX backends agree (integration tier, skipped without scikit-sparse) |
| G5 | Dense against `KroneckerSum` against `null_space` + SLQ on a grid Laplacian; two connected components (`c = 2`); invariance to the choice of null-space basis; the cofactor path equals dense on connected and disconnected graphs; RW1 closed form; RW2 against dense |
| G6 | `log_prob` against a dense MVN from `Q⁻¹`, with and without `log_det_precision`; the intrinsic `log_prob` against the closed form on a path graph, including `(N−c)/2 · log τ`; each sampling-dispatch branch has the right covariance on a 10 × 10 grid, bounded by the estimator's own sampling distribution (slow); intrinsic samples are orthogonal to `null_space`; hard-constrained samples satisfy `A_c x = e` to machine precision; constrained marginal variances equal dense; `grad` of `log_prob` with respect to a factor scale; both classes work under `numpyro.handlers.seed` and `trace` |
| G7 | `rw1_structure` equals the path Laplacian; RW2 null space; AR(1) marginal variance `1/τ` in the interior; `bym2_precision`'s marginal of `b` has the BYM2 covariance (dense check at `n = 30`); `generalized_variance_scale` equals the dense definition and R-INLA's scaled value for the Scotland graph (golden); FEM `C` and `G` on a single triangle and a unit square equal the hand values; grid SPDE equals the FEM SPDE on a regular right-triangle mesh with lumped mass; `matern_spde_params` gives marginal variance ≈ σ² away from the boundary (slow) |
| G8 | Gaussian likelihood: the mode and `log_marginal` equal the exact conjugate result; Poisson on RW2: the mode matches dense Newton, and `jax.grad` of `log_marginal` with respect to `log τ` matches finite differences; Scotland BYM2 Poisson: the mode matches R-INLA's (golden, 1e-4) |
| G9 | CCD point count and symmetry for `m = 3..6`; on a Gaussian `log_post` the weighted design recovers its mean and covariance; the grid threshold is respected |
| G10 | On a Bernoulli / RW2 problem with skewed posteriors, the corrected means move towards R-INLA's `"vb"` means (golden) and away from the plain Gaussian approximation |

### G11: sketches and the Hadamard transform

- **Subspace embedding.** On a random `m × n` matrix with `d = 4n`, the
  singular values of `S Q_A` lie in `[1−ε, 1+ε]`. Use a fixed key, with a
  tolerance taken from the distribution's own concentration bound, noted
  in a comment.
- **Adjoint consistency.** `⟨Sx, y⟩ = ⟨x, Sᵀy⟩` for every sketch.
- **Matrix-free agreement.** `sketch_operator(op)` equals
  `apply(op.as_matrix())`.
- **SRHT** equals its dense `R H D P` construction.
- **`hadamard_transform`** is bit-identical to kernellib's (kernellib's
  FastFood tests run against it in K7).

### G12: range finder, QB, randomized SVD and eigh

- The range-finder error on a matrix with a known spectrum is within
  Halko–Martinsson–Tropp's expected bound.
- `randomized_svd` equals dense SVD on a fast-decaying spectrum.
- `n_power_iter` monotonically improves accuracy on a Matérn-½ Gram
  matrix (slow tier).
- `svd(method="randomized")` dispatch works.

### G13: randomized Nyström and the preconditioner

- Nyström is exact for a rank-`k` PSD matrix at `ℓ ≥ k`.
- The #354 regression test: CG iterations on `K + σ²I` for a Matérn-3/2
  kernel, `n = 5000`, are **non-increasing in rank** and below the
  unpreconditioned count at every tested rank. The same test fails on the
  current implementation (slow tier).
- The factors plug into `LowRankUpdate` solve and logdet unchanged.

### G14: RPCholesky and the preconditioner

- `pivoting="greedy"` is bit-identical to today's
  `guarded_pivoted_cholesky`, including the gh-236 and gh-237 cases.
- `"random"` pivots are distinct, and the trace error
  `tr(A − FFᵀ)` beats greedy on a clustered data set in expectation (slow
  tier, many keys).
- `from_operator` builds once.
- #345 regression: at full rank the preconditioner is exactly
  `(K + σ²I)⁻¹`.

### G15: sketch-and-precondition LSMR

- The LSMR iteration count is ≤ 30 and flat across
  `m ∈ {10⁴, 10⁵}` at fixed `n` (slow tier).
- The result matches `jnp.linalg.lstsq`, and the ridge solution for
  `δ > 0`.
- `sketch_and_solve` error is `O(ε)`.

### G16 and G17

- **G16** lands with Part A's iterative backend. It is tested against the
  Takahashi selected inverse (G4) on meshes that still factor, and its
  error must beat Hutchinson's at equal solve counts.
- **Each G17 item** lands only with its first user, with its reference
  paper's headline experiment as the test.

**Release cadence.** Release after G1 (kernellib K2 waits on it), after G2
(K5, M1), after G11 (K7), after G12 (K10, X1), after G13 and G14 (K8, K9),
and after G6 and G7 (P7).

---

## 7. Boundaries

- gaussx never imports kernellib. No function here takes a graph, an
  adjacency matrix, a mesh object or node coordinates. It takes
  operators, factors, null-space bases, and vertex / triangle arrays
  (G7's FEM assembly).
- Hyperpriors, PC priors and NumPyro sites live in pyrox (P7), not here.
  The BYM2 scaling constant is computed here (`generalized_variance_scale`,
  G7), because it is linear algebra.
- Kernel-specific uses of Part B (landmark selection, preconditioned KRR,
  randomized kernel PCA) live in kernellib (K8–K10).

## 8. Risks

| Risk | Mitigation |
|---|---|
| CG converges slowly for badly conditioned precision matrices (small `1−ρ`, fine grids) | Sparse Cholesky (G4) where it fits; Jacobi preconditioning by default otherwise; document the conditioning; the Kronecker fast path for grids avoids CG entirely |
| SLQ variance inside an MCMC log density adds noise to the acceptance ratio | For CAR, the priors use the exact spectral log-determinant precomputed once (pyrox-lgm `CAR`, P7). SLQ is the fallback for precision matrices without such a structure |
| The `jax.experimental.sparse` API changes | Confined to `_operators/_sparse.py` |
| A JAX left-looking Cholesky is slow for large or 3-D meshes (sequential columns, padded gathers) | Measure early (G4 benchmark); supernodes and level scheduling as follow-ups; CHOLMOD opt-in; the iterative backend beyond that |
| RCM fill is much worse than AMD or nested dissection on 2-D meshes | Record fill on the reference meshes; if it is prohibitive, CHOLMOD becomes the recommended path for meshes, or vendor a pure-Python AMD |
| `pure_callback` with CHOLMOD breaks under some transformations | Only the numeric factorisation crosses the callback; everything differentiated (Takahashi, the VJPs) is JAX, so `grad` is unaffected. `vmap` is sequential and documented |
| Implicit differentiation through a non-converged mode gives wrong gradients | `converged` is returned; the pyrox driver refuses θ-points whose inner Newton did not converge, and re-runs them with more iterations |
| The Nyström preconditioner's `shift` requirement ripples through `PreconditionedCGSolver` and pyrox-gp | Land G13 together with the #345 wiring fix; pyrox-gp does not construct preconditioners today, so there are no downstream breaks yet |
| Gradients through preconditioned CG (#312) | Out of scope. Preconditioners are built outside the solve and `stop_gradient`ed (they do not need derivatives), which keeps them compatible with whatever #312 settles on |
| One-column RPCholesky loops are slow on GPU | The blocked variant (`block_size > 1`) as a follow-up |
| SRHT padding to a power of two wastes up to 2× memory | Documented; SparseSign stays the default |
