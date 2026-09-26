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
| `RationalQuadratic` | not available (needs a Bessel $K$ JAX lacks) | Gamma scale mixture of Gaussians |

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
```

::: kernellib.AbstractFeatureMap

::: kernellib.RandomFourierFeatures

::: kernellib.OrthogonalRandomFeatures

::: kernellib.FastFoodFeatures

::: kernellib.NystromFeatures

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
