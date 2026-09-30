---
date: 2026-09-30
---

# plumax: roadmap

plumax's share of the [fused roadmap](roadmap.md). plumax already uses a
randomized SVD, but through scikit-learn. One phase swaps it for gaussx's.
It serves the [RandNLA](project-rnla.md) project.

> Maths notes (**The maths.**) say where each operation comes from; **Example.** blocks are pseudocode against the *planned* API (`gx` = gaussx, `kl` = kernellib, `px` = pyrox-gp, `lgm` = pyrox-lgm). End-to-end problems are in the [examples gallery](roadmap-examples.md).

| Phase | What | Needs |
|---|---|---|
| X1 | `gaussx.randomized_svd` in the matched-filter and assimilation backgrounds | G12 |

Future INLA use (plume retrieval residuals as an SPDE field, Bayesian POD
with covariates) is in the [INLA project](project-inla.md#use-cases).
It needs no plumax changes until pyrox-lgm (P8) ships.

---

## 1. Current state (on GitHub, `jejjohnson/plumax`)

- **`src/plumax/matched_filter/background.py:248-262`** fits
  `sklearn.decomposition.TruncatedSVD(algorithm="randomized", n_oversamples=..., random_state=...)`
  on the centred pixel spectra `Xc` (`n_samples × n_bands`).
  - It takes `V = components_` and
    `d = singular_values_² / n_samples` (MLE normalisation, chosen to
    match the empirical, Ledoit–Wolf and OAS estimators in the same
    module).
  - It wraps the result as a `gx.LowRankUpdate`.
  - It runs on the CPU in NumPy and is not traceable, so it cannot sit
    inside a jitted retrieval.
- **`src/plumax/assimilation/background.py:164`** runs a full
  `np.linalg.svd`, then returns a `gx.LowRankUpdate`. That is wasteful
  whenever the kept rank is well below `min(n_samples, n_bands)`.

## 2. X1: switch to `gaussx.randomized_svd` (needs G12 released)

**The maths.** **The matched filter.** Model background pixels as
$x\sim\mathcal N(\mu,\Sigma)$, and a plume as adding $\alpha t$ for a
known target signature $t$. The matched-filter estimate is

$$
\hat\alpha = \frac{t^\top\Sigma^{-1}(x-\mu)}{t^\top\Sigma^{-1}t}.
$$

**Why randomized SVD.**

- With $\Sigma\approx V\operatorname{diag}(d)V^\top + \epsilon I$, the
  rank-$r$ part coming from the SVD of the centred background pixels,
  Woodbury makes $\Sigma^{-1}$ cost $O(Br)$ per pixel.
- The randomized SVD gets $V$ and $d$ from $O(r)$ passes over the pixels.
  It never forms the $B\times B$ covariance, and never runs a full
  $O(NB\min(N,B))$ SVD.
- Because it is JAX, the whole filter jits and runs on GPU.

- **Matched-filter background:**
  - `U, s, Vt = gx.randomized_svd(Xc, rank, oversample=n_oversamples, n_power_iter=n_iter, key=key)`;
  - `V = Vt`, and `d = s**2 / n_samples`, with the normalisation
    unchanged.
- **Parameter mapping.** `n_oversamples` maps to `oversample`, and
  `TruncatedSVD`'s `n_iter` maps to `n_power_iter`. The current call does
  not pass `n_iter`, so it runs with `TruncatedSVD`'s default of 5 (the
  `"auto"` rule belongs to `sklearn.utils.extmath.randomized_svd`, not to
  `TruncatedSVD`). Use `n_power_iter=5` so results do not shift.
  scikit-learn also normalises between power iterations (by LU, under
  `power_iteration_normalizer="auto"`); gaussx re-orthonormalises by QR,
  which is at least as stable.
- **Seeds.** `random_state: int` becomes `key`. Keep accepting an int,
  converted with `jax.random.key(random_state)`.
- **Assimilation background:** use `gx.randomized_svd` when
  `rank < 0.5 · min(shape)`, and keep the full SVD otherwise.
- **Dependencies.** If this was the last scikit-learn use in
  `matched_filter/`, move scikit-learn to an optional extra.
  (Ledoit–Wolf and OAS may still need it; check first.)

**Example.**

```python
Xc = pixels - pixels.mean(axis=0)  # (n_pixels, n_bands)
_, s, Vt = gx.randomized_svd(
    Xc, 30, oversample=10, n_power_iter=5, key=jax.random.key(seed)
)
Sigma = gx.LowRankUpdate(
    lx.DiagonalLinearOperator(jnp.full(n_bands, eps)), Vt.T, s**2 / n_pixels
)
alpha_hat = jax.jit(jax.vmap(matched_filter, in_axes=(0, None, None)))(
    pixels, target, Sigma
)
```

### Tests

- **Parity with scikit-learn.** On a fixed synthetic hyperspectral cube,
  the covariance eigenvalues `d` agree within 1 % relative and the
  subspaces agree (principal angles under 1e-3). Both methods are
  randomized, so compare against the dense SVD as the oracle, and require
  gaussx to be at least as close as scikit-learn.
- `matched_filter_snr` and `detection_threshold` are unchanged within
  tolerance on the existing fixtures.
- **Jit.** The background builder can now be jitted: add a test that does
  so.

## 3. Not in scope

#156 §6.6 suggests sketch-and-precondition least squares for Gauss–Newton
steps in retrievals. plumax's 4DVar
(`assimilation/cost.py`, `control.py`) uses L-BFGS with `gx.solve` and
`gx.cholesky`, and has no Gauss–Newton inner loop today. Revisit when one
exists: `gx.SketchAndPrecondLSMR` (G15) is the tool.
