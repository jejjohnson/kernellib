# Spectral

The spectral side of stationary kernels. By Bochner's theorem a stationary
kernel is the Fourier transform of a non-negative spectral density,

$$
k(\tau) = \frac{1}{(2\pi)^D} \int S(\omega)\, e^{i \omega^\top \tau}\, d\omega,
$$

so $S$ integrates to $(2\pi)^D \sigma^2$ and $S / ((2\pi)^D \sigma^2)$ is a
probability density over frequencies. Random Fourier features, FastFood and
Laplace-eigenfunction (HSGP) approximations all start from it.

## Densities and samplers

Every `AbstractStationaryKernel` has two methods:

- `spectral_density(omega)` evaluates $S$ at frequency vectors of shape
  ``(..., D)``, lengthscale (scalar or ARD) and variance included.
- `sample_frequencies(key, n, d)` draws ``n`` frequencies from the normalised
  density, so that $\mathbb{E}[\cos(\omega^\top \tau)] = k(\tau) / \sigma^2$.

A kernel opts in by implementing `unit_spectral_density` and
`sample_unit_frequencies` for its unit-variance, unit-lengthscale form; the
base class applies the hyperparameters.

| Kernel | `spectral_density` | `sample_frequencies` |
|---|---|---|
| `RBF` | Gaussian | $\mathcal{N}(0, I) / \ell$ |
| `Matern` | $(2\nu + \lVert\ell\omega\rVert^2)^{-(\nu + D/2)}$ | multivariate Student-t, $2\nu$ degrees of freedom |
| `RationalQuadratic` | Gamma mixture of Gaussians, closed form in a Bessel $K_{\alpha - D/2}$ (infinite at $\omega = 0$ when $\alpha \le D/2$) | Gamma scale mixture of Gaussians |

| `Scaled` (`c * k`) | $c\, S_k(\omega)$ | those of `k` |
| `Sum` (`k_1 + ... + k_J`) | $\sum_j S_j(\omega)$ | mixture: a part drawn with probability $\propto k_j(0)$, then its frequency |

`Scaled` and `Sum` have the methods when every part is a stationary kernel (or
another `Scaled` / `Sum` of them), and a `spectral_variance` ($k(0)$, the
density's total mass over $(2\pi)^D$). The random-feature maps give each part
an equal share of the frequencies and weight its features by
$\sqrt{c_j \sigma_j^2 / F_j}$, which is unbiased like mixture sampling but has
lower variance and keeps every part's hyperparameters differentiable after
`fit`. `Product` has a spectrum (the convolution of the parts) but no closed
form, so it has neither method.

`Periodic` and `Cosine` are not `AbstractStationaryKernel` subclasses and have
neither method.

```python
import jax
import jax.numpy as jnp
import kernellib as kl

k = kl.Matern(nu=1.5, lengthscale=jnp.array([0.5, 2.0]))
S = k.spectral_density(omega)  # omega: (M, 2)
W = k.sample_frequencies(jax.random.key(0), 1024, 2)  # (1024, 2)
```

## Feature maps

A feature map approximates a kernel by an explicit map,
$k(x, x') \approx \phi(x)^\top \phi(x')$. Configure it, `fit(kernel, X)` (which
returns a new module; `X` fixes the input dimension and, for Nyström, supplies
the landmarks), then call it for the feature matrix or use `operator(X)` for a
gaussx `LowRankUpdate` that solves and takes log-determinants through
Woodbury.

| Map | Kernels | Features | Cost per input |
|---|---|---|---|
| `RandomFourierFeatures` | stationary with a sampler | $2F$ | $O(FD)$ |
| `OrthogonalRandomFeatures` | stationary with a sampler | $2F$ | $O(FD)$, lower variance |
| `FastFoodFeatures` | stationary with a sampler | $2F$ | $O(F \log D)$, $O(F)$ storage |
| `NystromFeatures` | any | $M$ | $O(MD + M^2)$ |
| `LaplaceEigenfunctionFeatures` | stationary with a density | $\prod_d m_d$ | $O(D \prod_d m_d)$, deterministic |

The random maps store their draw at unit lengthscale and unit variance and read
the kernel's hyperparameters when called, so a fitted map differentiates with
respect to them (use `eqx.filter_grad`; the PRNG key is a leaf).

```python
import gaussx as gx

k = kl.Matern(nu=1.5, lengthscale=0.7)
rff = kl.RandomFourierFeatures(1024, key).fit(k, X)
Phi = rff(X)  # (N, 2048)
K_low = rff.operator(X)  # LowRankUpdate, rank 2048

nys = kl.NystromFeatures(300, key).fit(k, X)
nys.landmarks  # (300, D)

# Landmarks by approximate ridge leverage scores instead of uniformly
lev = kl.NystromFeatures(
    300, key, selection="leverage", leverage_regularization=1e-3
).fit(k, X)
```

Leverage-score landmarks help on unevenly spread data once `n_components` is
at least the effective dimension $d_{\mathrm{eff}}(\lambda)$; below it they can
all go to isolated points. `uniform_mixing` (default `0.5`) mixes in the
uniform distribution to guard against that; `0` gives pure leverage sampling.

::: kernellib.AbstractFeatureMap

::: kernellib.RandomFourierFeatures

::: kernellib.OrthogonalRandomFeatures

::: kernellib.FastFoodFeatures

::: kernellib.NystromFeatures

## Landmark selection

`select_landmarks` chooses Nyström landmarks (Falkon centres, EigenPro
subsamples, inducing points) from the data. `NystromFeatures(selection=)`,
`Falkon(centers=)` and `EigenPro(subsample=)` all take the same four methods.
The code default is `"uniform"` for compatibility; **`"rpcholesky"` is the
recommended choice**. It samples each landmark in proportion to the variance
the previous ones leave unexplained, needs `O(N M)` kernel evaluations and no
tuning, and its Nyström error is within a small factor of the best rank-`M`
approximation in expectation (Chen, Epperly, Tropp & Webber, 2023).

| `method` | Rule | Cost |
|---|---|---|
| `"uniform"` | uniformly without replacement | `O(M)` |
| `"leverage"` | approximate ridge leverage scores, mixed with uniform | `O(N M^2)` |
| `"rpcholesky"` | randomly pivoted Cholesky (`gaussx.rp_cholesky`) | `O(N M)` kernel evals, `O(N M^2)` |
| `"greedy"` | largest residual variance, deterministic | as `"rpcholesky"` |

```python
idx = kl.select_landmarks(
    kl.Matern(nu=1.5, lengthscale=0.3), X, 2000, method="rpcholesky", key=key
)
falkon = kl.Falkon(kernel, n_inducing=2000, centers="rpcholesky").fit(X, y, key=key)
```

### The Nyström approximation

Landmarks $S \subset \{1, \dots, N\}$, $|S| = M$, give the Nyström
approximation (Williams & Seeger, 2001)

$$
\hat K = K_{:,S}\,K_{S,S}^{+}\,K_{S,:}, \qquad
\operatorname{tr}(K - \hat K) = \sum_i \big(k(x_i, x_i)
- k_{iS} K_{SS}^{+} k_{Si}\big),
$$

the PSD approximation that reproduces $K$ exactly on the chosen rows and
columns. The trace error is a sum of conditional variances: $k(x_i, x_i) - k_{iS}
K_{SS}^{+} k_{Si}$ is the variance of a GP at $x_i$ after observing it
noise-free at the landmarks. Choosing landmarks well means making that
residual small with few columns. `NystromFeatures` evaluates
$\phi(x) = L^{-1}k(Z, x)$ with $K_{ZZ} + \epsilon I = LL^\top$, so that
$\phi(x)^\top\phi(x') \approx k(x, x')$.

### Ridge leverage scores (`"leverage"`)

The ridge leverage score of point $i$ is

$$
\ell_i(\lambda) = \big[K (K + \lambda n I)^{-1}\big]_{ii},
\qquad \sum_i \ell_i(\lambda) = d_{\mathrm{eff}}(\lambda),
$$

with the same $\lambda n$ ridge convention as `KRR` (``regularization``,
default $10^{-3}$). It measures how much point $i$ is needed to fit a ridge
regression at that $\lambda$; sampling $O(d_{\mathrm{eff}} \log
d_{\mathrm{eff}})$ landmarks in proportion to it gives a Nyström
approximation that preserves the KRR risk (Alaoui & Mahoney, 2015; Musco &
Musco, 2017; Rudi et al., 2018). Exact scores cost $O(N^3)$, so they are
approximated from a uniform pilot: Nyström features $\Phi$ on
$m_0 = \min(2M, N)$ uniform points give

$$
\tilde\ell_i = \phi_i^\top(\Phi^\top\Phi + \lambda n I)^{-1}\phi_i
+ \frac{k(x_i, x_i) - \|\phi_i\|^2}{\lambda n},
$$

the leverage of $\Phi\Phi^\top$ (by the push-through identity) plus the
pilot's residual variance over $\lambda n$, so points the pilot explains
badly are not missed. This is a single pilot
level, not Musco & Musco's recursive scheme. Landmarks are then drawn
without replacement from
$p_i = (1 - u)\,\tilde\ell_i / \sum_j \tilde\ell_j + u / N$, $u$ =
``uniform_mixing`` (default $0.5$).

### Randomly pivoted Cholesky (`"rpcholesky"`)

RPCholesky builds a partial Cholesky factor $F$, $FF^\top = \hat K$, one
pivot at a time, sampling each pivot in proportion to the current residual
diagonal $d = \operatorname{diag}(K - FF^\top)$ (Chen, Epperly, Tropp &
Webber, 2023). It touches $K$ only through its diagonal and one column per
landmark, never the $N \times N$ matrix:

```text
d = diag(K)                                  # k(x_i, x_i), N evaluations
F = zeros(N, M); S = []
for t in 1..M:
    s ~ Categorical(d / sum(d))              # greedy: s = argmax(d)
    g = K[:, s] - F[:, :t-1] F[s, :t-1]^T    # one kernel column, N evaluations
    F[:, t] = g / sqrt(g[s])
    d = max(d - F[:, t]^2, 0)
    S.append(s)
return S                                     # F F^T = K[:, S] K[S, S]^+ K[S, :]
```

With $k \ge r/\varepsilon + r\log(1/(\varepsilon\eta))$ pivots,
$\mathbb E\operatorname{tr}(K - \hat K) \le (1 + \varepsilon)
\operatorname{tr}(K - [\![K]\!]_r)$, where $[\![K]\!]_r$ is the best rank-$r$
approximation and $\eta = \operatorname{tr}(K - [\![K]\!]_r)/\operatorname{tr}K$:
near-optimal, with no parameter to tune. Cost: $O(NM)$ kernel evaluations
and $O(NM^2)$ flops.

### Greedy pivoting (`"greedy"`)

The same loop with $s = \arg\max_i d_i$: always the point with the largest
conditional variance (pivoted Cholesky; the inducing-point rule of Burt,
Rasmussen & van der Wilk, 2020). It is deterministic and has the same cost,
but no error guarantee: it picks isolated points and outliers first, and on
clustered data it can do worse than uniform sampling.

### Numerics

- The Cholesky methods run in the kernel's dtype. Once the largest residual
  falls below ``N * eps * max(diag K)`` (LAPACK `?pstrf`'s rule, in
  `gaussx.rp_cholesky`) the numerical rank is exhausted, for example with
  duplicated inputs or a long lengthscale. The remaining landmarks are then
  filled with unused points: uniformly at random for `"rpcholesky"`, in
  index order for `"greedy"`. The returned indices are always $M$ distinct
  points.
- `NystromFeatures` and the leverage pilot add a relative jitter
  $\epsilon = 10^{-6}$ times the mean of $\operatorname{diag}K_{ZZ}$ (an
  absolute $10^{-6}$ when that mean is zero) before the Cholesky of
  $K_{ZZ}$, so near-duplicate landmarks do not break it.
- Leverage scores need $M \gtrsim d_{\mathrm{eff}}(\lambda)$. With fewer
  landmarks than that, the scores concentrate on a few isolated points,
  which is what ``uniform_mixing`` guards against. A $\lambda$ that is too
  small pushes every score towards 1 (for a full-rank $K$), and the method
towards uniform.
- `"uniform"` evaluates no kernel; the others cost $O(NM)$ kernel
  evaluations (`"rpcholesky"`, `"greedy"`) or $O(N M)$ evaluations plus an
  $O(N M^2)$ solve for the pilot (`"leverage"`).

### References

- Williams & Seeger (2001). Using the Nyström method to speed up kernel
  machines. NeurIPS 13.
  [papers.nips.cc](https://papers.nips.cc/paper/1866-using-the-nystrom-method-to-speed-up-kernel-machines)
- Alaoui & Mahoney (2015). Fast randomized kernel methods with statistical
  guarantees. NeurIPS. [arXiv:1411.0306](https://arxiv.org/abs/1411.0306)
- Musco & Musco (2017). Recursive sampling for the Nyström method. NeurIPS.
  [arXiv:1605.07583](https://arxiv.org/abs/1605.07583)
- Rudi, Calandriello, Carratino & Rosasco (2018). On fast leverage score
  sampling and optimal learning. NeurIPS.
  [arXiv:1810.13258](https://arxiv.org/abs/1810.13258)
- Chen, Epperly, Tropp & Webber (2023). Randomly pivoted Cholesky: practical
  approximation of a kernel matrix with few entry evaluations.
  [arXiv:2207.06503](https://arxiv.org/abs/2207.06503)
- Burt, Rasmussen & van der Wilk (2020). Convergence of sparse variational
  inference in Gaussian processes regression. JMLR.
  [arXiv:2008.00323](https://arxiv.org/abs/2008.00323)

::: kernellib.select_landmarks

## Laplace eigenfunctions (HSGP)

`LaplaceEigenfunctionFeatures` is the Hilbert-space approximation of Solin &
Särkkä: on a box $[-L_1, L_1] \times \dots \times [-L_D, L_D]$,
$k(x, x') \approx \sum_j S(\omega_j) \phi_j(x) \phi_j(x')$ with the
Laplacian's Dirichlet eigenfunctions from `geonnax.basis.fourier_basis` and
$\omega_j$ their per-axis frequency vectors, so ARD lengthscales are exact. The
basis does not depend on the kernel; only the weights $S(\omega_j)$ do. The box
must be wide relative to the lengthscale; see the class docstring.

```python
lap = kl.LaplaceEigenfunctionFeatures(n_per_dim=(64, 32), boundary_factor=2.5)
lap = lap.fit(kl.RBF(lengthscale=jnp.array([0.3, 0.8])), X)
lap.half_widths  # the resolved box
Phi = lap(X)  # (N, 2048)
```

::: kernellib.LaplaceEigenfunctionFeatures

## Random Fourier feature prior draws

`draw_rff_cosine_basis` and `evaluate_rff_cosine_paths`, moved from
pyrox-gp, draw whole prior function paths
$\tilde f(x) = \sum_j w_j \sqrt{2\sigma^2/F} \cos(\omega_j^\top x / \ell + b_j)$.
The frequencies are drawn at unit lengthscale and the lengthscale is applied
at evaluation, so one draw can be paired with hyperparameters resolved
elsewhere; pyrox-gp's pathwise samplers rely on that.

```python
v, ell, omega, phase, w = kl.draw_rff_cosine_basis(
    k, key, n_paths=8, n_features=512, in_features=2
)
paths = kl.evaluate_rff_cosine_paths(
    X, variance=v, lengthscale=ell, omega=omega, phase=phase, weights=w
)  # (8, N)
```

::: kernellib.draw_rff_cosine_basis

::: kernellib.evaluate_rff_cosine_paths
