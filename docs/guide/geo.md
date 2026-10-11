---
title: Kernels on the Earth
short_title: Kernels on the Earth
description: Choosing distances and positive-definite kernels for (lon, lat) data, with geostatistics, space–time, spectral features and meshes.
---

Geographic data live on a sphere of radius $R \approx 6371$ km, not on the
$(\text{lon}, \text{lat})$ rectangle they are stored in. Two things go wrong
when that is forgotten: distances break at the dateline and the poles, and
the obvious fix, an RBF of the great-circle distance, is **not a valid
covariance**. This page is the map of kernellib's geo toolbox: which distance
to use, which kernel families are positive definite on the sphere, and where
the geostatistics, space–time, spectral and mesh tools fit. Every symbol links
to its API entry.

:::{tip} Worked example
The [Kernels on the Earth notebook](../notebooks/geo_kernels.ipynb) runs
everything on this page with physical units and figures: the
positive-definiteness trap on a Fibonacci lattice, variography and kriging,
anisotropy ellipses, $10^5$ satellite points with spherical-harmonic
features, a sparse tapered covariance, an advected plume, a Slepian regional
fit, Matérn on a coastline mesh and a seasonal cycle.
:::

## Conventions

| Convention | Value | Notes |
|---|---|---|
| Point layout | `(N, 2)` arrays of `(lon, lat)` | **longitude first**, as in GeoJSON |
| Angles | `degrees=True` (default) | `degrees=False` for radians |
| Sphere radius | `radius=1.0` (default) | the unit sphere: distances are angles in radians |
| Kilometres | `radius=kl.EARTH_RADIUS_KM` | [`EARTH_RADIUS_KM`](xref:api#kernellib.EARTH_RADIUS_KM) $= 6371.0088$ km, the IUGG mean radius |
| Lengthscales, ranges, taper ranges | units of `radius` | km with `EARTH_RADIUS_KM` |
| Variances | squared field units | e.g. K² for a temperature anomaly |

:::{warning} Unit pitfalls
- **Degrees vs radians.** Radians passed to a degree-default function shrink
  every distance by $180/\pi \approx 57$.
- **Unit sphere vs km.** With the default `radius=1.0`, `lengthscale=500.0`
  means 500 *radians*: a field constant over the globe. Set
  `radius=kl.EARTH_RADIUS_KM` and give lengthscales in km, or keep the unit
  sphere and give them in radians ($1\ \text{rad} \approx 6371$ km,
  $1° \approx 111.2$ km).
- **`SphereMatern`'s lengthscale** is the $\ell$ of the spectrum
  $(2\nu/\ell^2 + \lambda_l)^{-(\nu + 1)}$, in units of `radius`; for
  $\ell \ll R$ it matches the Euclidean `Matern` with the same $\nu$ and
  $\ell$. The graph Matérn uses the exponent $-\nu$ instead.
- **Radius-search arguments.** In `radius_neighbors` and `radius_graph`,
  `radius` is the *search* radius and the sphere's radius is `sphere_radius`.
:::

## Distances

Write a point as the unit vector
$u(x) = (\cos\phi\cos\lambda,\ \cos\phi\sin\lambda,\ \sin\phi)$
(longitude $\lambda$, latitude $\phi$) and let
$\theta = \arccos(u \cdot u') \in [0, \pi]$ be the central angle. Then

$$
d_{\mathrm{gc}} = R\,\theta, \qquad
d_{\mathrm{ch}} = \lVert R u - R u' \rVert = 2R\sin\frac{\theta}{2},
$$

the great-circle distance (along the surface) and the chordal distance
(through the Earth). The identity follows from
$\lVert u - u'\rVert^2 = 2 - 2\cos\theta = 4\sin^2(\theta/2)$. The chord is
monotone in $\theta$, so both rank neighbours identically, and they agree to
second order at short range, $d_{\mathrm{ch}} = d_{\mathrm{gc}}(1 - \theta^2/24 + \dots)$:
about 0.1 % apart at 1000 km.

| Function | Returns |
|---|---|
| [`great_circle_distance`](xref:api#kernellib.functional.great_circle_distance) | $R\theta$ with $\theta = 2\operatorname{atan2}(\lVert u - u'\rVert, \lVert u + u'\rVert)$, accurate near zero and near antipodes |
| [`chordal_distance`](xref:api#kernellib.functional.chordal_distance) | $2R\sin(\theta/2)$ |
| [`lonlat_to_unit`](xref:api#kernellib.functional.lonlat_to_unit) | the $(N, 3)$ unit vectors $u(x)$ |

Neighbour searches and bandwidth rules take `metric="great_circle"` (or
`"chordal"`) and run a Euclidean search on the unit vectors, with no dateline
or pole artefacts. The reduction is exact: the chord is monotone in $\theta$,
so the unit vectors order neighbours exactly as the great-circle distance
does. The search itself is exact with the default `backend="exact"` (and
`"sklearn"`); `backend="pynndescent"` is approximate. The geo metrics are available in
[`nearest_neighbors`](xref:api#kernellib.nearest_neighbors),
[`radius_neighbors`](xref:api#kernellib.radius_neighbors),
[`knn_graph`](xref:api#kernellib.knn_graph),
[`radius_graph`](xref:api#kernellib.radius_graph) and
[`estimate_lengthscale`](xref:api#kernellib.estimate_lengthscale).

```python
import kernellib as kl

R = kl.EARTH_RADIUS_KM
knn = kl.nearest_neighbors(X, 10, metric="great_circle", radius=R)  # km
ell = kl.estimate_lengthscale(X, "median", metric="great_circle", radius=R)
```

## The positive-definiteness rule

:::{important} Positive definiteness on the sphere
An isotropic function $\psi(\theta)$ is a valid covariance on the sphere
$S^2$ if and only if its Legendre coefficients are non-negative
{cite:p}`schoenberg1942positive`:

$$
\psi(\theta) = \sum_{l \ge 0} b_l\, P_l(\cos\theta), \qquad
b_l = \frac{2l + 1}{2}\int_{-1}^{1} \psi(\arccos t)\, P_l(t)\, dt \;\ge\; 0 .
$$

$b_l$ is the variance carried by the degree-$l$ spherical harmonics, the
sphere's power spectrum. Two consequences
{cite:p}`gneiting2013spheres`:

- **Chordal is always valid.** Any kernel positive definite on
  $\mathbb{R}^3$, evaluated at the chordal distance, is positive definite on
  the sphere: its Gram *is* a Euclidean Gram at the points $R u_i$.
- **Great-circle needs a proof.** $\theta$ is not a Euclidean distance. The
  Gaussian $e^{-\theta^2/2c^2}$ is never valid, the Matérn of $R\theta$ only
  for $\nu \le \tfrac12$, the powered exponential $e^{-(\theta/c)^\alpha}$ only
  for $\alpha \le 1$.

**Never use an RBF, or a Matérn with $\nu > \tfrac12$, of the great-circle
distance.** Its Grams can have negative eigenvalues (on 400 Fibonacci points
with $\ell = 8000$ km the RBF has 194), so a GP can report negative variances.
:::

## Choosing a kernel

| Your data | Distance | Kernel family | Use |
|---|---|---|---|
| Global or continental `(lon, lat)`, any smoothness | chordal | any Euclidean kernel in $\mathbb{R}^3$ | [`Chordal`](xref:api#kernellib.Chordal)`(Matern(...), radius=R)` |
| Global, geostatistics ranges in km, rough fields | great-circle | Gneiting's Table 1 families | [`GreatCircleExponential`](xref:api#kernellib.GreatCircleExponential), [`GreatCirclePoweredExponential`](xref:api#kernellib.GreatCirclePoweredExponential), [`GreatCircleCauchy`](xref:api#kernellib.GreatCircleCauchy) |
| Global, sparse Grams or tapers | great-circle, compact | compactly supported, $c \le \pi$ | [`GreatCircleSpherical`](xref:api#kernellib.GreatCircleSpherical), [`GreatCircleAskey`](xref:api#kernellib.GreatCircleAskey), [`GreatCircleWendland`](xref:api#kernellib.GreatCircleWendland) |
| Global, intrinsic Matérn or heat, exact spectrum | geodesic (spectral) | Legendre series in $\lambda_l = l(l+1)/R^2$ | [`SphereMatern`](xref:api#kernellib.SphereMatern), [`SphereHeat`](xref:api#kernellib.SphereHeat), [`SphereSeries`](xref:api#kernellib.SphereSeries) |
| Regional, $\lesssim 1000$ km, projected to km | Euclidean on a local plane | classic geostatistics models | [`Spherical`](xref:api#kernellib.Spherical), [`Cubic`](xref:api#kernellib.Cubic), [`Pentaspherical`](xref:api#kernellib.Pentaspherical), [`HoleEffect`](xref:api#kernellib.HoleEffect), [`Stable`](xref:api#kernellib.Stable), [`GeneralizedCauchy`](xref:api#kernellib.GeneralizedCauchy), [`Wendland`](xref:api#kernellib.Wendland) |
| Directional correlation (jets, valleys) | rotated, rescaled lag | geometric anisotropy | [`GeometricAnisotropy`](xref:api#kernellib.GeometricAnisotropy), [`LinearTransform`](xref:api#kernellib.LinearTransform) |
| Space–time `(x, y, t)` | space and time lags | non-separable Gneiting | [`Gneiting`](xref:api#kernellib.Gneiting) |
| A tracer in a wind or current | lag minus displacement $vu$ | frozen field | [`Advected`](xref:api#kernellib.Advected), [`RandomAdvectionRBF`](xref:api#kernellib.RandomAdvectionRBF) |
| A coastline, lake or other irregular domain | geodesic on a triangle mesh | FEM Matérn / heat | [`MeshMatern`](xref:api#kernellib.MeshMatern), [`MeshHeat`](xref:api#kernellib.MeshHeat) |
| A river or road network, areal units | graph | graph Matérn / heat | [`GraphMatern`](xref:api#kernellib.GraphMatern), [`GraphHeat`](xref:api#kernellib.GraphHeat) |
| A seasonal or diurnal cycle | time lag mod the period | periodic | [`Periodic`](xref:api#kernellib.Periodic) + [`PeriodicFeatures`](xref:api#kernellib.PeriodicFeatures) |

## Kernels on the sphere

::::{tab-set}
:::{tab-item} Chordal
[`Chordal`](xref:api#kernellib.Chordal) maps each `(lon, lat)` to
$R\,u(x) \in \mathbb{R}^3$ and applies any Euclidean kernel there:
$k(x, x') = k_0(R u(x), R u(x'))$. Valid whenever $k_0$ is, with $k_0$'s
smoothness: a Matérn-$\nu$ base gives a field $\lceil\nu\rceil - 1$ times
mean-square differentiable. A stationary base sees
$2R\sin(\theta/2) \approx R\theta$, so its lengthscale is in the units of
`radius` at local scales.

```python
k = kl.Chordal(kl.Matern(nu=1.5, lengthscale=1500.0), radius=kl.EARTH_RADIUS_KM)
```
:::

:::{tab-item} Great-circle
$k(x, x') = \sigma^2\,\psi(\theta / c)$ with $c = \ell / R$, so `lengthscale`
is a range in the units of `radius`. The families and their constraints are
Gneiting's {cite:p}`gneiting2013spheres`, checked at construction:

| Kernel | $\psi(t)$ | Constraints |
|---|---|---|
| [`GreatCircleExponential`](xref:api#kernellib.GreatCircleExponential) | $e^{-t}$ | $c > 0$ |
| [`GreatCirclePoweredExponential`](xref:api#kernellib.GreatCirclePoweredExponential) | $e^{-t^\alpha}$ | $\alpha \in (0, 1]$ |
| [`GreatCircleCauchy`](xref:api#kernellib.GreatCircleCauchy) | $(1 + t^\alpha)^{-\tau/\alpha}$ | $\alpha \in (0, 1]$, $\tau > 0$ |
| [`GreatCircleSpherical`](xref:api#kernellib.GreatCircleSpherical) | $(1 + t/2)(1 - t)_+^2$ | $c \le \pi$ |
| [`GreatCircleAskey`](xref:api#kernellib.GreatCircleAskey) | $(1 - t)_+^\tau$ | $c \le \pi$, $\tau \ge 2$ |
| [`GreatCircleWendland`](xref:api#kernellib.GreatCircleWendland) | $(1 + \tau t)(1 - t)_+^\tau$ (order 2) | $c \le \pi$, $\tau \ge 4$ |

All share [`AbstractGreatCircleKernel`](xref:api#kernellib.AbstractGreatCircleKernel).
The last three are exactly zero beyond $\theta = c$: sparse Grams and tapers.
:::

:::{tab-item} Sphere series
The intrinsic kernels are functions of the Laplace–Beltrami operator, with
eigenvalues $\lambda_l = l(l+1)/R^2$ of multiplicity $2l + 1$. By the
addition theorem they are Legendre series

$$
k(u, v) = \sum_{l=0}^{L} a_l\, P_l(u \cdot v), \qquad
a_l \propto (2l + 1)\,\Phi(\lambda_l), \qquad \sum_l a_l = \sigma^2,
$$

valid for every truncation $L$ because $a_l \ge 0$. The Matérn
{cite:p}`borovitskiy2020matern` solves the SPDE
$(2\nu/\ell^2 - \Delta)^{\nu/2 + 1/2} f = \mathcal{W}$
{cite:p}`lindgren2011spde`, so
$\Phi(\lambda) = (2\nu/\ell^2 + \lambda)^{-(\nu+1)}$, valid for every
$\nu > 0$; the heat kernel has $\Phi(\lambda) = e^{-\ell^2\lambda/2}$.
`truncation_tail()` estimates the variance beyond `max_degree`; rough kernels
at short $\ell$ need many degrees.

| Kernel | $\Phi(\lambda)$ |
|---|---|
| [`SphereMatern`](xref:api#kernellib.SphereMatern) | $(2\nu/\ell^2 + \lambda)^{-(\nu + 1)}$ |
| [`SphereHeat`](xref:api#kernellib.SphereHeat) | $\exp(-\ell^2\lambda/2)$ |
| [`SphereSeries`](xref:api#kernellib.SphereSeries) | any `spectrum_fn` |

All share [`AbstractSphereSeriesKernel`](xref:api#kernellib.AbstractSphereSeriesKernel).
Any isotropic kernel of $\mathbb{R}^3$ restricted to the sphere is zonal,
$\kappa(u \cdot v) = k(Ru, Rv)$, and
[`funk_hecke_coefficients`](xref:api#kernellib.funk_hecke_coefficients)
computes its spectrum in one of two conventions. Keep them apart:

| `convention=` | Returns | Formula | Meaning |
|---|---|---|---|
| `"funk_hecke"` (default) | the Funk–Hecke eigenvalue $\mu_l$ | $\mu_l = 2\pi\int_{-1}^{1}\kappa(t)\,P_l(t)\,dt$ | eigenvalue of the kernel's integral operator on every degree-$l$ harmonic |
| `"legendre"` | the Legendre series coefficient $a_l$ | $a_l = \frac{2l+1}{4\pi}\,\mu_l = \frac{2l+1}{2}\int_{-1}^{1}\kappa(t)\,P_l(t)\,dt$ | the $a_l$ of $\kappa(t) = \sum_l a_l P_l(t)$ above, comparable with `degree_spectrum()` |
:::
::::

## Geostatistics

**Variograms.** The semivariogram {cite:p}`matheron1963principles`

$$
\gamma(h) = \tfrac12\,\mathbb{E}\big[(Z(x + h) - Z(x))^2\big] = \tau^2 + \sigma^2 - C(h)
\qquad (h > 0)
$$

reads off the **nugget** $\tau^2$ (the jump at $0^+$: measurement noise and
sub-sampling-scale variability), the **sill** $\tau^2 + \sigma^2$ (the total
variance) and the **practical range**: the lag at which $\gamma$ reaches
$\tau^2 + 0.95\,\sigma^2$, i.e. 95 % of the rise from the nugget to the sill
(where the correlation $\rho(h)$ falls to 0.05), not 95 % of the total sill.
It is the decorrelation length, about $2.74\,\ell$ for a Matérn-3/2.
[`empirical_variogram`](xref:api#kernellib.empirical_variogram) bins pairs by
Euclidean or great-circle distance (Matheron or Cressie–Hawkins estimators
{cite:p}`cressie1993statistics`) into a
[`Variogram`](xref:api#kernellib.Variogram), and
[`fit_variogram`](xref:api#kernellib.fit_variogram) fits $\ell$, $\sigma^2$
and $\tau^2$ by weighted least squares:

```python
v = kl.empirical_variogram(X, y, metric="great_circle", radius=kl.EARTH_RADIUS_KM)
kernel, nugget = kl.fit_variogram(v, kl.Matern(nu=1.5))  # ranges in km
gp_kernel = kl.Chordal(kernel, radius=kl.EARTH_RADIUS_KM)  # valid on the sphere
```

**Anisotropy.** Zonal jets stretch correlation east–west; valleys and fronts
channel it. Geometric anisotropy measures the lag in a rotated, rescaled
frame, $k(x, x') = k_0(A(x - x'))$ with
$A = \operatorname{diag}(1, 1/a)\,R_\alpha^\top$: correlation ellipses with
major axis at angle $\alpha$ carrying $\ell$, minor axis $a\ell$.
[`GeometricAnisotropy`](xref:api#kernellib.GeometricAnisotropy) builds $A$ from
angles and ratios, [`LinearTransform`](xref:api#kernellib.LinearTransform)
takes any $A$; the spectral density stays closed-form, so random Fourier
features still apply.

**Classic models.** [`Spherical`](xref:api#kernellib.Spherical),
[`Cubic`](xref:api#kernellib.Cubic),
[`Pentaspherical`](xref:api#kernellib.Pentaspherical),
[`HoleEffect`](xref:api#kernellib.HoleEffect),
[`Stable`](xref:api#kernellib.Stable),
[`GeneralizedCauchy`](xref:api#kernellib.GeneralizedCauchy) and
[`Wendland`](xref:api#kernellib.Wendland) are the gstools / gstat models as
Euclidean kernels; several are valid only for $d \le 3$. On the sphere use
them through `Chordal`.

**Tapering.** Multiplying a covariance by a compactly supported taper,
$K_{\mathrm{tap}} = K \circ T$ {cite:p}`furrer2006tapering`, keeps positive
definiteness (Schur product theorem) and makes the matrix exactly zero beyond
the taper range $\rho$. The taper should be at least as smooth at the origin
as $K$ (Wendland C² for $\nu \le 3/2$, C⁴ for $\nu \le 5/2$), and $\rho$
should span the lags that dominate the kriging weights (several station
spacings); on the sphere the taper is `GreatCircleWendland`, so
$\rho \le \pi R$. [`Tapered`](xref:api#kernellib.Tapered) is the kernel,
[`tapered_operator`](xref:api#kernellib.tapered_operator) the sparse
`gaussx.SparseOperator` that `gaussx.SparseCholeskySolver` factorises:

```python
import gaussx as gx
import jax.numpy as jnp

K = kl.tapered_operator(
    kl.GreatCircleExponential(lengthscale=800.0, radius=kl.EARTH_RADIUS_KM),
    X,
    taper_range=600.0,
    metric="great_circle",
    radius=kl.EARTH_RADIUS_KM,
)
K = K.add_diagonal(jnp.full(X.shape[0], 0.25))  # nugget, same sparsity pattern
logdet = gx.SparseCholeskySolver().logdet(K)
```

## Space–time and transport

**Non-separable space–time.** A separable $C_s(h)\,C_t(u)$ keeps the spatial
pattern fixed at every time lag; physically, large structures usually outlive
small ones. [`Gneiting`](xref:api#kernellib.Gneiting)
{cite:p}`gneiting2002nonseparable` couples them:

$$
C(h, u) = \frac{\sigma^2}{\psi(u)^{\tau}}
\exp\!\left(-\frac{c\,\lVert h\rVert^{2\gamma}}{\psi(u)^{\beta\gamma}}\right),
\qquad \psi(u) = a|u|^{2\alpha} + 1,
$$

with $c = \ell_s^{-2\gamma}$, $a = \ell_t^{-2\alpha}$ and separability
$\beta \in [0, 1]$: the spatial range at time lag $u$ is
$\ell_s\,\psi(u)^{\beta/2}$. For global data, map `(lon, lat)` to 3-D points
in km, append the time column and use `space_dims=3`: the inputs are
$(N, 4)$, and the chordal distance is the spatial lag.

```python
import einx
import jax.numpy as jnp

R = kl.EARTH_RADIUS_KM
xyz = R * kl.functional.lonlat_to_unit(lonlat)  # (N, 3) km
X_st = jnp.concatenate([xyz, einx.id("n -> n 1", t_days)], axis=-1)  # (N, 4)
k = kl.Gneiting(space_dims=3, space_lengthscale=500.0, time_lengthscale=3.0, beta=0.7)
K = k(X_st, X_st)
```

**Transport.** A tracer in a steady wind $v$ is, by Taylor's frozen-turbulence
hypothesis {cite:p}`taylor1938spectrum`, a pattern carried along:
$Z(x, t) = Z_0(x - vt)$, so $k = k_s(h - vu)$.
[`Advected`](xref:api#kernellib.Advected) implements it for any stationary
$k_s$ (multiply by a kernel in time for decay). With an uncertain wind
$V \sim \mathcal N(\mu, \Sigma_V)$,
[`RandomAdvectionRBF`](xref:api#kernellib.RandomAdvectionRBF) is the
closed-form average {cite:p}`cox1988simple`: the correlation moves with $\mu$
and spreads as $\ell^2 I + u^2\Sigma_V$. Keep the units consistent: km and
hours give velocities in km/h.

```python
import jax.numpy as jnp
import kernellib as kl

wind = jnp.array([10.0, 0.0])  # km/h towards the east; inputs (x_km, y_km, t_h)
k = kl.Advected(kl.Matern(nu=1.5, lengthscale=4.0), wind) * kl.Matern(
    nu=0.5, lengthscale=6.0
).select((2,))
```

## Spectral features and scale

Dense GP algebra costs $O(N^3)$ time to factorise the $N \times N$ Gram and
$O(N^2)$ memory to store it. Deterministic feature maps make a GP a Bayesian
linear regression on the features, and stay differentiable in the kernel's
hyperparameters:

| Feature map | Kernel | Features | When |
|---|---|---|---|
| [`SphericalHarmonicFeatures`](xref:api#kernellib.SphericalHarmonicFeatures) | sphere-series or `Chordal` | $(L+1)^2$ weighted real harmonics, $\phi_{lm} = \sqrt{4\pi a_l/(2l+1)}\,Y_{lm}$ | global, smooth fields, $N \gg (L+1)^2$ |
| [`SlepianFeatures`](xref:api#kernellib.SlepianFeatures) | `Chordal` (zonal) | about the Shannon number $(L+1)^2(1 - \cos r)/2$ cap-concentrated modes {cite:p}`simons2006slepian` | one region (a cap of radius $r$) |
| [`PeriodicFeatures`](xref:api#kernellib.PeriodicFeatures) | `Periodic` | $2K+1$ harmonics of the exact Bessel series, truncated at $K$ (`truncation_tail()`) {cite:p}`solin2014periodic` | seasonal / diurnal cycles |
| [`MeshLaplaceFeatures`](xref:api#kernellib.MeshLaplaceFeatures) | `MeshMatern`, `MeshHeat` | $M$ FEM eigenfunctions | irregular domains |
| [`GraphEigenFeatures`](xref:api#kernellib.GraphEigenFeatures) | `GraphMatern`, `GraphHeat` | $M$ Laplacian eigenvectors | large graphs |

A degree $l$ resolves half-wavelengths of about $\pi R / l$ ($L = 32$:
about 625 km), so pick $L$ from the shortest scale you need, and check
`truncation_tail()`. When the field is rough or short-ranged and $(L+1)^2$
explodes, taper instead (above).

:::{dropdown} Features or the dense kernel?
With $F = (L+1)^2$ features, a fit costs $O(NF)$ to evaluate the features
($O(BF)$ memory when accumulated in batches of $B$ rows), $O(NF^2)$ to form
$\Phi^\top\Phi$ and $O(F^3)$ to factorise the posterior precision; a
prediction costs $O(F)$ per point for the mean and $O(F^2)$ for the variance.
The dense kernel costs $O(N^2)$ memory for the Gram and $O(N^3)$ time to
factorise it. With $L = 32$ (1089 features), $10^5$
observations fit in about 5 s on a 4-core CPU (the notebook's run); the
dense Gram would need 80 GB.
For $N$ of a few thousand, the dense kernel with `to_operator` and a gaussx
solver is simpler and exact.
:::

## Meshes and graphs

[`MeshMatern`](xref:api#kernellib.MeshMatern) and
[`MeshHeat`](xref:api#kernellib.MeshHeat) (base
[`AbstractMeshKernel`](xref:api#kernellib.AbstractMeshKernel)) discretise the
Laplace–Beltrami operator with P1 finite elements on a triangle mesh and use
the same spectra as the sphere kernels {cite:p}`borovitskiy2020matern`. Open
boundaries get natural (Neumann) conditions, so correlation flows around a
peninsula, not across it, which no distance-based kernel can do. On an
icosphere the FEM eigenvalues approach $l(l+1)/R^2$ and `MeshMatern` approaches
`SphereMatern`.

[`GraphMatern`](xref:api#kernellib.GraphMatern) and
[`GraphHeat`](xref:api#kernellib.GraphHeat) are the same construction on a
graph Laplacian (river networks, road networks, areal units); build the graph
with `knn_graph(..., metric="great_circle", radius=R)` for stations.

## All geo symbols

| Theme | Symbols |
|---|---|
| Distances | [`EARTH_RADIUS_KM`](xref:api#kernellib.EARTH_RADIUS_KM), [`great_circle_distance`](xref:api#kernellib.functional.great_circle_distance), [`chordal_distance`](xref:api#kernellib.functional.chordal_distance), [`lonlat_to_unit`](xref:api#kernellib.functional.lonlat_to_unit) |
| Neighbours | [`nearest_neighbors`](xref:api#kernellib.nearest_neighbors), [`radius_neighbors`](xref:api#kernellib.radius_neighbors), [`knn_graph`](xref:api#kernellib.knn_graph), [`radius_graph`](xref:api#kernellib.radius_graph), [`estimate_lengthscale`](xref:api#kernellib.estimate_lengthscale) |
| Sphere kernels | [`Chordal`](xref:api#kernellib.Chordal), [`AbstractGreatCircleKernel`](xref:api#kernellib.AbstractGreatCircleKernel), [`GreatCircleExponential`](xref:api#kernellib.GreatCircleExponential), [`GreatCirclePoweredExponential`](xref:api#kernellib.GreatCirclePoweredExponential), [`GreatCircleCauchy`](xref:api#kernellib.GreatCircleCauchy), [`GreatCircleSpherical`](xref:api#kernellib.GreatCircleSpherical), [`GreatCircleAskey`](xref:api#kernellib.GreatCircleAskey), [`GreatCircleWendland`](xref:api#kernellib.GreatCircleWendland), [`AbstractSphereSeriesKernel`](xref:api#kernellib.AbstractSphereSeriesKernel), [`SphereMatern`](xref:api#kernellib.SphereMatern), [`SphereHeat`](xref:api#kernellib.SphereHeat), [`SphereSeries`](xref:api#kernellib.SphereSeries), [`funk_hecke_coefficients`](xref:api#kernellib.funk_hecke_coefficients) |
| Geostatistics | [`Spherical`](xref:api#kernellib.Spherical), [`Cubic`](xref:api#kernellib.Cubic), [`Pentaspherical`](xref:api#kernellib.Pentaspherical), [`HoleEffect`](xref:api#kernellib.HoleEffect), [`Stable`](xref:api#kernellib.Stable), [`GeneralizedCauchy`](xref:api#kernellib.GeneralizedCauchy), [`Wendland`](xref:api#kernellib.Wendland), [`LinearTransform`](xref:api#kernellib.LinearTransform), [`GeometricAnisotropy`](xref:api#kernellib.GeometricAnisotropy), [`Variogram`](xref:api#kernellib.Variogram), [`empirical_variogram`](xref:api#kernellib.empirical_variogram), [`fit_variogram`](xref:api#kernellib.fit_variogram), [`Tapered`](xref:api#kernellib.Tapered), [`tapered_operator`](xref:api#kernellib.tapered_operator) |
| Space–time, transport | [`Gneiting`](xref:api#kernellib.Gneiting), [`Advected`](xref:api#kernellib.Advected), [`RandomAdvectionRBF`](xref:api#kernellib.RandomAdvectionRBF) |
| Features | [`SphericalHarmonicFeatures`](xref:api#kernellib.SphericalHarmonicFeatures), [`SlepianFeatures`](xref:api#kernellib.SlepianFeatures), [`PeriodicFeatures`](xref:api#kernellib.PeriodicFeatures), [`MeshLaplaceFeatures`](xref:api#kernellib.MeshLaplaceFeatures), [`GraphEigenFeatures`](xref:api#kernellib.GraphEigenFeatures) |
| Meshes, graphs | [`AbstractMeshKernel`](xref:api#kernellib.AbstractMeshKernel), [`MeshMatern`](xref:api#kernellib.MeshMatern), [`MeshHeat`](xref:api#kernellib.MeshHeat), [`GraphMatern`](xref:api#kernellib.GraphMatern), [`GraphHeat`](xref:api#kernellib.GraphHeat) |
