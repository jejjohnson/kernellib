---
date: 2026-09-30
---

# Examples gallery: what the roadmap lets you solve

Each example is a real problem, stated with its model and with pseudocode
against the *planned* API: `gx` = gaussx, `kl` = kernellib,
`px` = pyrox-gp, `lgm` = pyrox-lgm, `manipy`. Each lists the phases it
needs, so the gallery doubles as a check that the roadmap delivers
something usable at each wave.

| # | Problem | Domain | Key phases |
|---|---|---|---|
| 1 | [Disease mapping with BYM2](#ex-bym2) | epidemiology, emissions by region | G6, G7, G8–G10, K6, P7, P8 |
| 2 | [Gap-filling sea-surface temperature](#ex-sst) | oceanography | G1, G3, G7, G16, P8 |
| 3 | [Probability of detection for methane plumes](#ex-pod) | remote sensing | G3, G8, G10, P8 |
| 4 | [Hyperspectral classification and cross-sensor transfer](#ex-hsi) | remote sensing | K2, K5, M1, M2 |
| 5 | [A GP on a river or road network](#ex-graph-gp) | hydrology, traffic | K2–K4, P2 |
| 6 | [Kernel regression and GPs at scale](#ex-scale) | any | G13, G14, K8, K9, P3, P4 |
| 7 | [Tall nonlinear least squares](#ex-lsq) | retrievals, inverse problems | G11, G15 |
| 8 | [Background covariances and EOFs](#ex-eofs) | remote sensing, climate | G12, X1 |
| 9 | [A GPLVM started on the data manifold](#ex-gplvm) | latent-variable models | K5, P1 |
| 10 | [Uncertainty on a 2M-node global mesh](#ex-mesh) | geostatistics on the sphere | G1, G7, G16 |
| 11 | [Fair regression](#ex-fair) | credit, hiring, public-sector risk scores | K12, K13 (K9 at scale) |
| 12 | [Semi-supervised regression on a graph](#ex-laprls) | remote sensing with sparse in-situ labels | K13 (K2 for sparse graphs) |

---

(ex-bym2)=
## 1. Disease mapping with BYM2

**Problem.** Counts $y_i$ of cases (or detections) per area, with
expected counts $E_i$ and a covariate $z_i$. How much of the excess risk
is spatially structured, and what is the covariate's effect?

**Model** (Riebler et al., 2016):

$$
y_i\sim\operatorname{Poisson}\big(E_ie^{\eta_i}\big),\qquad
\eta_i = \beta_0 + \beta_1z_i + b_i,\qquad
b = \sigma\big(\sqrt{1-\phi}\,v + \sqrt{\phi}\,u^\ast\big),
$$

where $v\sim\mathcal N(0,I)$ and $u^\ast$ is an ICAR scaled to unit
generalised variance. The stacked $(b,u^\ast)$ has the sparse precision of
`gx.bym2_precision`. The PC priors are on $\sigma$, and on $\phi$, the
share of the variance that is spatially structured.

```python
counties = kl.graph_from_adjacency(queen_contiguity)
model = lgm.LGM(
    components=(lgm.BYM2(counties, name="region"),),
    fixed=lgm.FixedEffects(("intercept", "z")),
    likelihood=gx.PoissonLikelihood(),
)
res = lgm.inla(
    model, {"y": y, "offset": jnp.log(E), "region": jnp.arange(n), "z": z}, key=key
)
relative_risk = jnp.exp(res.random["region"].mean)
res.hyperpar["region.phi"].quantiles(0.025, 0.5, 0.975)
```

**What makes it fast.** The Hessian $Q + W$ keeps $Q$'s pattern, and it is
analysed once. The θ-design is a small grid in $(\sigma,\phi)$, a few dozen points at most. Each
point needs a few sparse factorisations, with marginal variances from one
Takahashi sweep. For NUTS instead of INLA, the same `lgm.BYM2(...).sample()`
goes inside a NumPyro model (see [pyrox.md](roadmap-pyrox.md)).

---

(ex-sst)=
## 2. Gap-filling sea-surface temperature (space-time)

**Problem.** A daily SST field on a 1° grid (180 × 360), with clouds
removing 60 % of pixels. The goal is posterior mean fields and
per-pixel uncertainty, for every day of a year.

**Model.**
$y_{st} = x_{st} + \varepsilon_{st}$, observed only on the cloud-free
mask $S$, with a separable prior

$$
x\sim\mathcal N\big(0,\ (Q_t\otimes Q_s)^{-1}\big),\qquad
Q_t = \text{AR(1)},\qquad
Q_s = \tau^2h^2\big(\kappa^2I + h^{-2}(L_{\text{lat}}\oplus L_{\text{lon}})\big)^2 .
$$

**The numerics.** The posterior precision
$Q_t\otimes Q_s + \sigma^{-2}S^\top S$ has 23.6 M unknowns. The mask
breaks the Kronecker structure, but
$P = Q_t\otimes Q_s + \bar s\,\sigma^{-2}I$ (with $\bar s$ the observed
fraction) is still exactly invertible through the factor eigenvectors
(the shifted-Kronecker solve of [gaussx.md](roadmap-gaussx.md), G3). As
a CG preconditioner, it makes the iteration count depend on how irregular
the mask is, not on the grid size.

```python
prior_s = gx.spde_precision_grid(
    (180, 360), kappa, tau, alpha=2, spacing=1.0, periodic=True
)
prior_t = gx.ar1_precision(365, rho=0.9, tau=1.0)
Q = gx.Kronecker(prior_t, prior_s)
H = Q + lx.DiagonalLinearOperator(
    mask_flat / 0.3**2
)  # Q + σ⁻² SᵀS (S selects the clear pixels)
# Q_t ⊗ Q_s + c·I, exact via the factor eigenbases (G3); the spatial factor stays spectral
c = mask_flat.mean() / 0.3**2
shift = gx.Kronecker(
    lx.DiagonalLinearOperator(jnp.full(365, c)),
    lx.DiagonalLinearOperator(jnp.ones(180 * 360)),
)
P = gx.SumOfKroneckers((Q, shift))
mean = gx.PreconditionedCGSolver(preconditioner=gx.OperatorPreconditioner(P)).solve(
    H, obs_rhs
)
sd = jnp.sqrt(gx.diag_inv(H, method="xdiag", num_probes=64, key=key))
```

Via `inla()`, the hyperparameters $(\rho,\kappa,\tau,\sigma)$ are
integrated rather than fixed (see [pyrox.md §3.4](roadmap-pyrox.md)).

---

(ex-pod)=
## 3. Probability of detection for methane plumes

**Problem.** Binary detections $d_j$ of known releases, with flux $q_j$,
wind speed $w_j$ and instrument covariates $c_j$. The target is the
detection probability as a smooth function of flux and wind, with
credible bands.

**Model.**

$$
d_j\sim\operatorname{Bernoulli}\big(\operatorname{logit}^{-1}(\eta_j)\big),\qquad
\eta_j = \beta^\top c_j + f_1(\log q_j) + f_2(w_j),
$$

with $f_1, f_2$ RW2 on binned values (cubic-spline priors,
block-tridiagonal precision).

```python
model = lgm.LGM(
    components=(lgm.RW2(n_bins=50, name="log_flux"), lgm.RW2(n_bins=30, name="wind")),
    fixed=lgm.FixedEffects(("intercept", "sza", "albedo")),
    likelihood=gx.BernoulliLikelihood(),
)
res = lgm.inla(
    model, data, strategy="vb", key=key
)  # the VB mean correction matters for Bernoulli
pod_curve = jax.nn.sigmoid(res.random["log_flux"].mean + res.fixed["intercept"].mean)
```

**What makes it fast.** Two RW2 blocks, plus a handful of dense
fixed-effect columns ordered last. The factorisation is essentially banded,
and the whole fit takes seconds on CPU (#155's target).

---

(ex-hsi)=
## 4. Hyperspectral classification and cross-sensor transfer

**Problem.** Classify every pixel of a hyperspectral image (Indian Pines,
145 × 145 × 200 bands) from 10 % of labels. Then reuse the labels from
one sensor to classify a scene from another.

**Model.** Schrödinger eigenmaps,
$(L + \alpha V)y = \lambda Dy$:

- $L$ comes from a spectral k-NN graph;
- $V$ is the Laplacian of the pixel grid, reweighted by spectral
  similarity (Cahill et al., 2014).

This embeds pixels so that neighbours which look alike collapse together.
Manifold alignment then solves one generalised eigenproblem across both
sensors (see [manipy.md §4](roadmap-manipy.md)).

```python
X, shape = manipy.hsi.image_to_array(cube)
V = manipy.hsi.spatial_spectral_potential_image(cube)
Y = (
    kl.SchrodingerEigenmaps(n_components=30, alpha=17.8, eigen_solver="lanczos")
    .fit(X, V)
    .embedding
)
clf = SVC().fit(Y[train], labels[train])

ma = manipy.ManifoldAlignment(method="sema", n_components=20).fit(
    [X_sensor_a, X_sensor_b],
    [y_a, y_b],
    spatial_graphs=[kl.grid_graph(sa), kl.grid_graph(sb)],
)
pred_b = (
    SVC()
    .fit(ma.transform(X_a_lab, domain=0), y_a_lab)
    .predict(ma.transform(X_sensor_b, domain=1))
)
```

---

(ex-graph-gp)=
## 5. A GP on a river or road network

**Problem.** Water temperature at 3,000 gauges on a river network, or
speeds at 50,000 traffic sensors. Correlation should follow the network,
not straight-line distance.

**Model.** $f\sim\mathcal{GP}(0, K)$ with the graph Matérn kernel
$K = U(2\nu/\ell^2 + \Lambda)^{-\nu}U^\top$ on the network's Laplacian.
Sparse variational inference uses $M$ Laplacian-eigenvector features,
whose $K_{uu}$ is diagonal.

```python
network = kl.graph_from_adjacency(reach_adjacency)  # or kl.knn_graph(sensor_xy, 8)
feats = px.LaplacianInducingFeatures.fit(network, 256, method="lanczos", key=key)
prior = px.SparseGPPrior(px.Matern(nu=1.5, lengthscale=3.0), inducing=feats)
# X is the vector of node indices; the ELBO costs O(N M) per step
```

---

(ex-scale)=
## 6. Kernel regression and GPs at scale

**Problem.** Fit KRR, or a GP's hyperparameters, on
$n = 2\times10^5$ points, where Cholesky ($O(n^3)$ time, $O(n^2)$ memory)
is out of reach.

**Model.** Solve $(K+\mu I)\alpha = y$ by CG, preconditioned with a
Nyström or RPCholesky approximation of rank
$\ell\approx d_{\text{eff}}(\mu)$. The condition number drops from
$\lambda_1/\mu$ to $O(1)$ (see [gaussx.md §5.3](roadmap-gaussx.md)).

```python
kernel = kl.Matern(nu=1.5, lengthscale=0.2)
krr = kl.KRR(
    kernel,
    regularization=1e-6,
    implicit=True,
    preconditioner="rpcholesky",
    preconditioner_rank=1000,
).fit(X, y, key=key)
Z = px.init_inducing(
    X, 1024, kernel=kernel, method="rpcholesky", key=key
)  # or: an SVGP with good Z
```

---

(ex-lsq)=
## 7. Tall nonlinear least squares (retrievals)

**Problem.** Invert a forward model $F(\theta)$, with 300 parameters, for
10⁶ observed pixels by Gauss–Newton, with Tikhonov regularisation.

**Model.** Each step solves
$\min_s\|J s + r\|^2 + \delta^2\|s\|^2$. Sketching $J$ to about $4n$ rows
gives a preconditioner under which LSMR converges in about 15 iterations,
whatever the number of pixels (see [gaussx.md §5.5](roadmap-gaussx.md)).

```python
def gauss_newton(theta, n_steps=10):
    for _ in range(n_steps):
        r, J_op = F(theta) - y_obs, jax.linearize(F, theta)[1]  # J as a matvec
        J = lx.FunctionLinearOperator(J_op, jax.eval_shape(lambda: theta))
        theta = theta + gx.SketchAndPrecondLSMR(sampling_factor=4.0, damp=delta).solve(
            J, -r
        )
    return theta
```

---

(ex-eofs)=
## 8. Background covariances and EOFs

**Problem.** A low-rank-plus-diagonal background covariance for a matched
filter over 10⁷ pixels × 400 bands; or the leading EOFs of a decade of
daily fields.

**Model.** A randomized range finder with power iterations gives the top
$r$ singular pairs from $O(r)$ passes over the data. The error is
controlled by $\sigma_{r+1}$, with the tail damped as
$\sigma^{2q+1}$ (see [gaussx.md §5.2](roadmap-gaussx.md)).

```python
_, s, Vt = gx.randomized_svd(Xc, 30, oversample=10, n_power_iter=5, key=key)
Sigma = gx.LowRankUpdate(
    lx.DiagonalLinearOperator(jnp.full(n_bands, eps)), Vt.T, s**2 / n_pixels
)
```

---

(ex-gplvm)=
## 9. A GPLVM started on the data manifold

**Problem.** Learn a 2-D latent space for high-dimensional observations
that lie on a curled manifold (motion capture, spectra over a gradient).
Starting from PCA folds it.

**Model.** GPLVM,
$y_{:,d}\sim\mathcal N\big(0, K(X,X)+\sigma^2I\big)$. Initialising $X$
with Laplacian eigenmaps starts it at the unfolded coordinates; PCA is
the linear-kernel optimum (see [pyrox.md §2.1](roadmap-pyrox.md)).

```python
X0 = px.latent_init(Y, 2, method="laplacian_eigenmaps", n_neighbors=15)
# numpyro.param("X", X0) inside the GPLVM model, then SVI / MAP as usual
```

---

(ex-mesh)=
## 10. Uncertainty on a 2M-node global mesh

**Problem.** A Matérn field on an icosahedral mesh of the sphere, with 2M
nodes and ¹⁄₁₀° resolution. The goal is the posterior marginal sd
everywhere. At this size, the Cholesky fill of a 2-D mesh
($O(N\log N)$ with nested dissection, far more with RCM) is large.

**Model.** SPDE on the sphere, from the FEM matrices of the surface mesh.
Posterior $H = Q + A^\top\Lambda A$. The fallback chain applies (see
[gaussx.md §3](roadmap-gaussx.md)): CG for the mean, and XDiag for
$\operatorname{diag}(H^{-1})$, whose low-rank part captures the
large-scale, high-variance modes.

```python
C, G = gx.fem_matrices(
    icosphere_vertices, icosphere_triangles
)  # 3-D vertices: a surface mesh
Q = gx.spde_precision(C, G, *gx.matern_spde_params(range=500.0, sigma=1.0, nu=1.0, d=2))
# the sphere is star-shaped, so G7 locates each point by a ray–triangle test
A = gx.fem_projector(icosphere_vertices, icosphere_triangles, obs_xyz)
H = Q.union(Q.congruence(A, jnp.full(n_obs, 1 / noise_var)))
cg = gx.PreconditionedCGSolver(preconditioner=gx.JacobiPreconditioner())
mean = cg.solve(H, A.T.mv(y_obs / noise_var))
sd = jnp.sqrt(gx.diag_inv(H, method="xdiag", num_probes=64, solver=cg, key=key))
```

---

(ex-fair)=
## 11. Fair regression

**Problem.** Predict income from the Adult census (about 30,000 rows)
without the predictions depending on sex or race, and show the full
accuracy–dependence trade-off rather than a single operating point.

**Model** (Pérez-Suay et al., 2017). Fair kernel ridge regression, with the
linear-kernel HSIC between the predictions $f = K\alpha$ and the protected
attributes $S$ as the penalty:

$$
\min_\alpha\ \tfrac1n\|y-K\alpha\|^2 + \lambda\,\alpha^\top K\alpha
+ \mu\,\underbrace{\tfrac1{n^2}\alpha^\top KHSS^\top HK\alpha}_{\operatorname{HSIC}_b(f,\,S)} .
$$

It is quadratic, so its solution is one linear system, a rank-2 update of
KRR's (see [kernellib.md](roadmap-kernellib.md), K13).

```python
S = jnp.asarray(adult[["sex", "race_white"]], dtype=float)
base = kl.KRR(
    kl.RBF(2.0),
    regularization=1e-4,
    implicit=True,
    preconditioner="rpcholesky",
    preconditioner_rank=1000,
)
penalty = kl.hsic_penalty(kl.Linear(), S)  # H S Sᵀ H / n², rank 2
dependence = partial(kl.cka, kl.RBF(1.0), kl.Linear(), estimator="unbiased")
front = []
for mu in jnp.logspace(-1, 3, 12):
    fit = dataclasses.replace(base, penalty_weight=mu).fit(
        X, y, penalty=penalty, key=key
    )
    pred = fit.predict(X_test)
    front.append((jnp.mean((pred - y_test) ** 2), dependence(pred[:, None], S_test)))
```

The same trade-off for a network needs gradients through a nonlinear
penalty. Only K12's gradient-safe `kl.cka` is required:
`loss = mse + mu * kl.cka(kl.RBF(ell_y), kl.RBF(1.0), pred[:, None], s)`.

**What makes it fast.** By Woodbury, each μ costs three ordinary KRR
solves: one for $y$, one per attribute. Each is preconditioned CG on the
implicit kernel operator (K9), so $K$ is never formed.

---

(ex-laprls)=
## 12. Semi-supervised regression on a graph

**Problem.** Soil moisture at 20,000 pixels, with only 300 in-situ labels.
The unlabelled pixels still show where the data manifold is.

**Model.** Laplacian-regularised least squares (Belkin, Niyogi &
Sindhwani, 2006). $f = K\alpha$ over all points, with a data term on the
labelled ones only ($J$ is the labelled mask, $l = \operatorname{tr}J$) and
a smoothness penalty along a k-NN graph:

$$
\min_\alpha\ \tfrac1l\|J(y-K\alpha)\|^2 + \lambda\,\alpha^\top K\alpha
+ \tfrac{\mu}{n^2} f^\top Lf
\quad\Longrightarrow\quad
\big(JK + l\lambda I + \tfrac{l\mu}{n^2}LK\big)\alpha = Jy .
$$

This is the same estimator as [fair regression](#ex-fair), with a
different penalty matrix.

```python
graph = kl.nearest_neighbors(X_all, 10)  # a K2 Graph once that lands
laprls = kl.KRR(kl.RBF(0.5), regularization=1e-4, penalty_weight=1.0).fit(
    X_all,
    y_all,  # any value where unlabelled
    mask=is_labelled,
    penalty=kl.laplacian_penalty(kl.adjacency_matrix(graph)),
)
moisture_map = laprls.predict(X_grid)
```

**What makes it work.** With a mask, K13 solves the symmetric normal form
$(KJK + l\lambda K + \tfrac{l\mu}{n^2}KLK)\alpha = KJy$ by CG. With a
sparse K2 Laplacian, each iteration costs three kernel matvecs and one
sparse matvec.
