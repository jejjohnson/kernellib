---
date: 2026-09-30
---

# Project: dependence penalties (from keras-fairkl)

[keras-fairkl](https://github.com/jejjohnson/keras-fairkl) (`fairkl`
0.1.5) is a Keras 3 library for fairness-constrained kernel learning. It
comes with the 2017 TensorFlow fair-learning notebooks that preceded it
(`.data/scripts/B1*`, `B3*`). The goal here is **not** a second fairness
library. It is to move the useful primitives underneath fairkl into
kernellib, where they serve every dependence-measure user: fairness,
semi-supervised learning, supervised embeddings and representation
comparison.

## What fairkl has, against kernellib 0.0.11

About 80 % of fairkl already exists in kernellib, usually more generally:

| fairkl | kernellib today | Verdict |
|---|---|---|
| `rbf_kernel`, `linear_kernel`, `polynomial_kernel` | `RBF`, `Linear`, `Polynomial`, and the rest of the kernel family | Skip |
| `nystrom_approximate`, `random_fourier_features`, `random_kitchen_sinks` | `NystromFeatures`, `RandomFourierFeatures`, `OrthogonalRandomFeatures`, `nystrom_operator`, `rff_operator` | Skip |
| `center_kernel`, `centering_matrix` | `functional.center_kernel` (which also centres a low-rank operator without forming it), `centering_operator` | Skip |
| `hsic_biased` / `_rbf` / `_linear`, `cka_biased`, `cka_debiased`, `mmd_rbf` | `hsic`, `cka`, `mmd_squared`: biased and unbiased estimators (and linear-time MMD), `approx=` feature maps, permutation tests | Skip, except the numerics below |
| `center_gram_unbiased` (U-centring) | The unbiased HSIC uses the expanded Song et al. formula | **Port** as the numerically stable form (K12, #94) |
| `CKAMetric`'s component accumulation | None | **Port** as a consistent mini-batch `CKAAccumulator` (K12) |
| `max(denominator, 1e-6)` in the CKA functions | Unguarded division | **Port** the idea, as a gradient-safe guard (K12, #93) |
| The notebooks' `bandwidth(d)` | `estimate_lengthscale` has median / mean / Silverman / Scott | **Port** as `method="gaussian"` (K12) |
| `solve_cholesky`, `solve_cg` | gaussx solver strategies | Skip |
| Keras layers (trainable `log_sigma`, landmarks), `HSICLoss` / `CKALoss` / `MMDLoss`, the metrics | Equinox kernels are trainable pytrees; in JAX a loss is a function | Skip |
| `FairKernelRidge` (Adam on a CKA penalty) | `KRR` | **Port the closed form it lost**: quadratic-penalty KRR (K13) |
| `FairKernelPCA`, `FairPCA` (Adam, soft orthogonality) | `KernelPCA` | **Port** as a generalised eigenproblem, supervised or fair (K14) |
| `FairKernelPCA.inverse_transform` | None | **Port** as a learned pre-image with any kernel (K14) |
| `_center_cross_kernel` | Inline in `KernelPCA.transform` | **Port** as public `functional.center_cross_kernel` (K14) |
| `FairLinear`, `FairModelWrapper` | — | Skip: `loss + mu * kl.cka(...)` with `jax.grad` is the whole pattern. Shown in the docs (K11) |
| `sklearn_compat`, `tuning` (keras-tuner) | `kernellib.sklearn` | Skip |

The central finding is that fairkl's models are **all** trained by
gradient descent on a CKA penalty with an RBF kernel on the predictions.
With a linear kernel on the predictions, the fair KRR objective is
quadratic and has a closed form (Pérez-Suay et al., 2017, the original
fair kernel learning paper). That closed form is identical to
Laplacian-regularised least squares with a different penalty matrix. So
kernellib gains one estimator that does both, and fair and supervised
kernel PCA become one eigenproblem.

## Bugs found in the audit

Each is filed as a self-contained issue with a reproduction and measured
numbers:

| Issue | Bug | How the port avoids it |
|---|---|---|
| [kernellib#93](https://github.com/jejjohnson/kernellib/issues/93) | `cka` is `nan` / `-inf`, with a NaN gradient, for constant or near-constant inputs (float32 unbiased from scale 1e-2) | K12's gradient-safe ratio, with CKA = 0 when degenerate |
| [kernellib#94](https://github.com/jejjohnson/kernellib/issues/94) | The unbiased HSIC cancels catastrophically in float32: 18 % off at scale 1e-2, exactly 0 at 1e-3 | K12's U-centring; the low-rank path centres its factors |
| [keras-fairkl#15](https://github.com/jejjohnson/keras-fairkl/issues/15) | `FairKernelRidge` solves `(K+λI)` at μ = 0 but optimises a mean-MSE loss (whose optimum is `(K+nλI)`) for μ > 0, so training MSE is 10× worse at μ = 1e-8 | K13 uses KRR's mean convention in the closed form, and tests continuity at μ → 0 |
| [keras-fairkl#16](https://github.com/jejjohnson/keras-fairkl/issues/16) | A hard-coded `sigma_f=1.0` on the predictions: the same predictions score CKA 0.93 or 0.28 depending on the target's units | K13's closed form uses a linear kernel on the predictions, which is scale-free. The gradient recipe (K12) takes the bandwidth from the target, once |
| [keras-fairkl#17](https://github.com/jejjohnson/keras-fairkl/issues/17) | The models import `jax` (undeclared) despite "any Keras backend" | Not applicable: kernellib is JAX-native |
| [keras-fairkl#18](https://github.com/jejjohnson/keras-fairkl/issues/18) | `FairKernelPCA` is not KPCA at μ = 0: the wrong constraint (`VᵀV`, not `AᵀK_cA`), soft, with unidentifiable components and a linear pre-image | K14's closed form reproduces `KernelPCA` exactly at γ = 0, and the pre-image kernel is configurable |
| [keras-fairkl#19](https://github.com/jejjohnson/keras-fairkl/issues/19) | `CKAMetric` drifts with batch size by default (+16 % at batch 10) | `CKAAccumulator` uses unbiased per-batch HSIC only |

## Where the work lives

| Repo | Phases |
|---|---|
| kernellib | [K12–K14](roadmap-kernellib.md): dependence numerics and `CKAAccumulator`, quadratic-penalty KRR (`hsic_penalty`, `laplacian_penalty`), `KernelPCA` supervision, pre-images and `center_cross_kernel`; K11 gains a `dependence_penalties` notebook |
| gaussx | Nothing new. K13's fast path uses `gx.LowRankUpdate`'s general Woodbury solve, which exists already. G2 (`eigh_generalized`) is not needed, because K14 reduces to a standard eigenproblem |
| pyrox, manipy, plumax | Nothing required. manipy can use K14's `inverse_transform` for reconstruction errors |
| keras-fairkl | Issues #15–#19. Archiving is [open question 1](#open-questions-fairkl) |

## What you can solve with it

From the [examples gallery](roadmap-examples.md):

- [fair regression](roadmap-examples.md#ex-fair): predictions independent of protected
  attributes, with the whole trade-off curve;
- [semi-supervised regression on a graph](roadmap-examples.md#ex-laprls): a few hundred
  labels and many unlabelled points;
- comparing network representations over a whole dataset (kernellib#89),
  with `CKAAccumulator`.

## Non-goals

- A fairness library, or anything named "fair" in kernellib's API. The
  names are neutral (`penalty`, `target_weight`), and fairness appears in
  the docs as one use.
- Group-fairness metrics (demographic parity, equalised odds). These are
  classification metrics, not kernel methods, and belong to fairlearn.
- Keras layers, losses, metrics, or keras-tuner search spaces.
- Gradient-trained linear or kernel PCA. The eigenproblem is exact and
  cheaper.
- A kernellib estimator for nonlinear (RBF-on-predictions) penalties. It
  is a three-line `jax.grad` loss, shown in the docs.

(open-questions-fairkl)=
## Open questions

| # | Question | Proposed answer |
|---|---|---|
| 1 | What happens to keras-fairkl? | Archive it once K12–K14 are released, with a README pointing to kernellib and a migration table (this page's audit table). Close #15–#19 as superseded, unless someone needs the Keras models maintained |
| 2 | Should degenerate CKA be 0 or NaN? | 0, with a zero gradient. A constant is independent of everything, and NaN poisons optimiser state. Documented in both `cka` docstrings |
| 3 | Is `penalty` an estimator field or a `fit` argument? | A `fit` argument, because it is aligned with `X` like a sample weight. `penalty_weight` is the field, so it can be cross-validated |
| 4 | Is a μ-path helper worth adding (sharing the $r+1$ base solves across μ)? | Not in K13. `vmap` over `penalty_weight` recomputes the base solves. Add `fit_path` only if the docs notebook shows the cost matters |

## Decisions log

| Date | Decision |
|---|---|
| 2026-09-30 | Audited keras-fairkl 0.1.5 and the 2017 notebooks against kernellib 0.0.11. No new library. Port the closed forms (quadratic-penalty KRR, supervised / fair KPCA), the pre-image, the stable unbiased HSIC, the mini-batch CKA, the gradient-safe CKA and the Gaussian bandwidth as kernellib K12–K14. Filed kernellib#93, #94 and keras-fairkl#15–#19 |
