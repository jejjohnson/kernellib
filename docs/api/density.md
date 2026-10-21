# Kernel density estimation

`KernelDensity` is a weighted Parzen estimator with a stable score and Hessian
and three ways to choose the bandwidth: Scott, Silverman, and leave-one-out
(LOO) likelihood maximised with `jax.grad`. Everything is evaluated in log
space. A naive `mean(K(X*, X))` underflows to 0 a few lengthscales from the
data, and its log and score then become `-inf` / NaN. The log-space path stays
finite at any distance.

```python
import jax
import kernellib as kl

kde = kl.KernelDensity(kl.RBF(), bandwidth="scott").fit(X, weights=beta)
kde.log_density(Q)  # (M,)
kde.score(Q)  # (M, D): grad_x log p
kde.hessian_log_density(Q)  # (M, D, D)
kde.lengthscale  # the selected bandwidth
kl.gradient(kde.log_density, Q)  # log_density is a KD1 scalar field
kde.sample(jax.random.key(0), 1000)  # RBF only
```

!!! note "`score` is the score function"
    `KernelDensity.score` is $\nabla_x\log\hat p$, as in score matching and
    diffusion models. It is **not** scikit-learn's `score`. The adapter
    [`kernellib.sklearn.KernelDensity`][kernellib.sklearn.KernelDensity]
    follows scikit-learn: its `score` is the total log-likelihood and
    `score_samples` is the log-density.

## Estimator

Given training points $x_i\in\mathbb R^D$ and weights $\beta_i\ge 0$, `fit`
normalises the weights to $\sum_i\beta_i = 1$ (uniform $1/N$ by default).
The estimator is Johnson et al. (2020), §5.1, Eq 18 and its weighted form,
after Parzen (1962) and Rosenblatt (1956):

$$
\hat p(x) = \frac1c\sum_{i=1}^N\beta_i\,k(x, x_i),\qquad
c = \int_{\mathbb R^D}k(x, 0)\,dx .
$$

Let $\ell_i(x) = \log\beta_i + \log k(x, x_i)$. Then

$$
\log\hat p(x) = \operatorname{logsumexp}_i\,\ell_i(x) - \log c,\qquad
\pi_i(x) = \operatorname{softmax}_i\,\ell_i(x).
$$

$\log k$ comes from each stationary kernel's exact log-profile
`_log_shape(r²)`: $-r^2/2$ for RBF, $-\alpha\log(1 + r^2/2\alpha)$ for the
rational quadratic, and $\log\operatorname{poly}(r) - \sqrt{2\nu}\,r$ for
Matérn. It never computes $\log$ of an underflowed kernel value. Integer
weights are equivalent to duplicated rows. Uniform weights equal
`weights=None`. Negative, non-finite or all-zero weights raise
`ValueError`, or an `equinox` runtime error under `jit`.

## Normalisation

kernellib's Fourier convention is
$k(\tau) = (2\pi)^{-D}\int S(\omega)e^{i\omega^\top\tau}d\omega$, so
$c = \int k = S(0)$, the kernel's
[`spectral_density`][kernellib.AbstractStationaryKernel.spectral_density] at
zero. `fit` computes it in log space,
$\log c = \log\sigma^2 + \sum_d\log\ell_d + \log s(0)$, where $s$ is the
unit-lengthscale density. With $\Lambda = \operatorname{diag}(\ell_d^2)$:

| Kernel | $k(\tau)$ | $c$ |
|---|---|---|
| `RBF` | $\sigma^2 e^{-\frac12\tau^\top\Lambda^{-1}\tau}$ | $\sigma^2(2\pi)^{D/2}\prod_d\ell_d$ |
| `Matern(nu)` | Matérn-$\nu$ | $\sigma^2\prod_d\ell_d\,(2\pi/\nu)^{D/2}\,\Gamma(\nu + D/2)/\Gamma(\nu)$ |
| `RationalQuadratic(alpha)` | $\sigma^2(1 + r^2/2\alpha)^{-\alpha}$ | $\sigma^2\prod_d\ell_d\,(2\pi\alpha)^{D/2}\,\Gamma(\alpha - D/2)/\Gamma(\alpha)$, finite iff $\alpha > D/2$ |

Both the Matérn and the rational-quadratic constants tend to the RBF one as
$\nu, \alpha\to\infty$. The variance $\sigma^2$ cancels, so $\hat p$ does not
depend on it.

**`normalize=False`.** `normalize=True` (the default) needs an
`AbstractStationaryKernel` whose $S(0)$ is finite and positive. Otherwise
`fit` raises a `ValueError` that names `normalize=False`. This covers
`Linear`, `Polynomial`, `Periodic` and `Cosine`, composites, and
`RationalQuadratic` with $\alpha\le D/2$. With `normalize=False`, $c := 1$ and
`log_density` is $\log\sum_i\beta_ik(x, x_i)$, the kernel mean embedding at
$x$. That is a log-density up to an additive constant. `score`,
`hessian_log_density` and anything built on them (density ridges, mean shift)
are unchanged, because they do not see $c$. The kernel must still be
non-negative. `fit` checks $k(x_i, x_i) > 0$, and for a non-stationary kernel
`log_density` is `log(kernel(...))`.

## Score and Hessian

For the Gaussian (`RBF`) kernel, let $m(x) = \sum_i\pi_ix_i$ and
$C(x) = \sum_i\pi_i(x_i - m)(x_i - m)^\top$. Then

$$
s(x) = \nabla\log\hat p = \Lambda^{-1}\big(m(x) - x\big),\qquad
\nabla^2\log\hat p = \Lambda^{-1}C(x)\Lambda^{-1} - \Lambda^{-1}.
$$

$m(x) - x = \Lambda s$ is the mean-shift vector, the step that subspace-
constrained mean shift takes (Ozertem & Erdogmus, 2011). Because $\pi$ is a
softmax, $s$ stays finite arbitrarily far from the data, where it tends to
$\Lambda^{-1}(x_{\rm nearest} - x)$. With one training point,
$s = \Lambda^{-1}(x_1 - x)$ and $\nabla^2\log\hat p = -\Lambda^{-1}$ exactly.
$C$ is computed on points centred at the weighted mean
$\bar x = \sum_i\beta_ix_i$, so its cancellation error scales with the
spread of the data and not with its distance from the origin.

For every other kernel, `score` is KD1's
[`gradient`][kernellib.gradient] of `log_density`, and `hessian_log_density`
is KD1's [`hessian`][kernellib.hessian] of it. Autodiff through the
`logsumexp` of `_log_shape` is stable. The Hessian needs a kernel that is
twice differentiable at $x = x_i$. KD1's guard rejects `Matern(nu=0.5)`, whose
score still exists away from the training points. In every case,

$$
\nabla\hat p = \hat p\,s,\qquad
\nabla^2\hat p = \hat p\,\big(\nabla^2\log\hat p + s\,s^\top\big).
$$

## Streaming evaluation

`batch_size=None` builds the $(M, N)$ block of $\ell_{mi}$ at once.
`batch_size=B` never forms it. Queries are mapped with
`jax.lax.map(..., batch_size=B)`, and the training axis is scanned in chunks
of $B$ with an online log-sum-exp. The Gaussian path also carries the moment
sums. The running max $m$ is a stop-gradient shift, so its value cancels
exactly:

```text
carry: m = -inf, s = 0, S1 = 0, S2 = 0           # per query
for each training chunk (X_b, log beta_b)        # padded with log beta = -inf
    t    = log beta_b + log k(x, X_b)            # (B,)
    m'   = stop_gradient(max(m, max_b t))
    r    = exp(m - m')                           # 0 while m = -inf
    e    = exp(t - m')
    s    = r s  + sum_b e
    S1   = r S1 + sum_b e (x_b - xbar)           # Gaussian score
    S2   = r S2 + sum_b e (x_b - xbar)(x_b - xbar)^T   # Gaussian Hessian
log p(x) = m + log s - log c
m(x)     = xbar + S1 / s
C(x)     = S2 / s - (S1 / s)(S1 / s)^T
```

Each step is `jax.checkpoint`ed, so reverse mode (the generic-kernel score and
Hessian, and the LOO gradient) does not store every chunk. Peak memory is
$O(B^2 D^2)$ for the Gaussian Hessian and $O(B^2)$ for the log-density.
The chunked and dense paths agree to rounding.

## Bandwidth

`bandwidth=None` keeps the kernel's lengthscale, and a number sets it. A rule
selects the lengthscale in `fit` and stores it in the fitted kernel with
`dataclasses.replace`, so `kde.lengthscale` is the selected value. A rule is
ARD (one lengthscale per dimension) when the kernel's lengthscale is `(D,)`.

- **`"scott"` / `"silverman"`.** These are
  [`estimate_lengthscale`][kernellib.estimate_lengthscale]: Scott (1992)
  gives $\hat\sigma\,n^{-1/(D+4)}$ and Silverman (1986) gives
  $\hat\sigma\,(n(D+2)/4)^{-1/(D+4)}$. With weights, $n$ is Kish's effective
  sample size $n_{\rm eff} = (\sum_i\beta_i)^2/\sum_i\beta_i^2$, and
  $\hat\sigma_d$ is the weighted standard deviation with the
  $1/(1 - \sum_i\beta_i^2)$ correction, as in `scipy.stats.gaussian_kde`.
- **`"loo"`** is likelihood cross-validation (Duin, 1976). It maximises

  $$
  \mathcal L(\ell) = \sum_i\beta_i\log\hat p_{-i}(x_i),\qquad
  \hat p_{-i}(x_i) = \frac{\sum_{j\ne i}\beta_jk_\ell(x_i, x_j)}
      {c_\ell\sum_{j\ne i}\beta_j},
  $$

  in log space, with the $i$-th term masked to $-\infty$. It optimises
  $\theta = \log\ell$ (scalar, or `(D,)` for ARD) with
  `jax.scipy.optimize.minimize(method="BFGS")`, starting from Scott's value.
  It warns if BFGS stops without meeting its gradient tolerance within
  `loo_max_iter` steps. It always uses the true $c_\ell$, because without it
  $\mathcal L$ grows with $\ell$. Two coincident training points make
  $\mathcal L\to\infty$ as $\ell\to 0$, so `fit` rejects duplicated rows and
  points to Scott, Silverman or de-duplication.
  `KernelDensity.loo_log_likelihood` is smooth in $\ell$, so
  `jax.grad(lambda k: kl.KernelDensity(k).fit(X).loo_log_likelihood())` also
  works.

The tests pin each of these: agreement with
`sklearn.neighbors.KernelDensity` (with `sample_weight`) to $10^{-10}$; the
normalisation by quadrature and against the closed forms above; the closed
forms against KD1 autodiff; the LOO likelihood against a brute-force refit
loop; the LOO bandwidth as a stationary point and the argmax on a grid; and
Scott with weights against `scipy.stats.gaussian_kde`.

## Sampling

For `RBF` only: draw $i\sim\operatorname{Cat}(\beta)$, then
$x = x_i + \Lambda^{1/2}\varepsilon$ with $\varepsilon\sim\mathcal N(0, I)$.
Other kernels raise `NotImplementedError`.

## Reference

::: kernellib.KernelDensity

## References

- Johnson, Laparra, Pérez-Suay, Mahecha & Camps-Valls (2020). Kernel methods
  and their derivatives: concept and perspectives for the earth system
  sciences. PLOS ONE 15(10): e0235885.
  [doi:10.1371/journal.pone.0235885](https://doi.org/10.1371/journal.pone.0235885)
  (§5.1 Eqs 18–19, §5.2 Eq 20)
- Parzen (1962). On estimation of a probability density function and mode.
  Annals of Mathematical Statistics 33(3): 1065–1076.
  [doi:10.1214/aoms/1177704472](https://doi.org/10.1214/aoms/1177704472)
- Rosenblatt (1956). Remarks on some nonparametric estimates of a density
  function. Annals of Mathematical Statistics 27(3): 832–837.
  [doi:10.1214/aoms/1177728190](https://doi.org/10.1214/aoms/1177728190)
- Silverman (1986). *Density Estimation for Statistics and Data Analysis*.
  Chapman & Hall (reissued by Routledge, 2018).
  [doi:10.1201/9781315140919](https://doi.org/10.1201/9781315140919)
- Scott (1992). *Multivariate Density Estimation: Theory, Practice, and
  Visualization*. Wiley.
  [doi:10.1002/9780470316849](https://doi.org/10.1002/9780470316849)
- Duin (1976). On the choice of smoothing parameters for Parzen estimators of
  probability density functions. IEEE Transactions on Computers C-25(11):
  1175–1179. [doi:10.1109/TC.1976.1674577](https://doi.org/10.1109/TC.1976.1674577)
- Ozertem & Erdogmus (2011). Locally defined principal curves and surfaces.
  Journal of Machine Learning Research 12: 1249–1286.
  [jmlr.org/papers/v12/ozertem11a](https://jmlr.org/papers/v12/ozertem11a.html)
