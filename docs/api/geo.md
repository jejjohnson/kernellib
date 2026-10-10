# Geo

Kernels and tools for data on the Earth. Geographic inputs are ``(N, 2)``
arrays of ``(lon, lat)``, longitude first, in degrees by default
(``degrees=False`` for radians). Distances come back in units of ``radius``:
``radius=1.0`` gives the great-circle angle in radians and
``radius=EARTH_RADIUS_KM`` kilometres.

!!! note "Positive-definiteness on the sphere"
    Feeding great-circle distance into an RBF kernel, or a Matérn with
    $\nu > \tfrac12$, does **not** give a positive-definite kernel on the sphere
    (Gneiting 2013). Use a Euclidean kernel of the **chordal** distance, or a
    kernel proven valid on great-circle distance.

## Distances

```python
import kernellib as kl

d_km = kl.functional.great_circle_distance(X1, X2, radius=kl.EARTH_RADIUS_KM)
c_km = kl.functional.chordal_distance(X1, X2, radius=kl.EARTH_RADIUS_KM)
U = kl.functional.lonlat_to_unit(X)  # (N, 3) unit vectors
```

The chordal distance $2R\sin(\theta/2)$ is a monotone function of the
great-circle angle $\theta$, so both order neighbours identically; any
Euclidean kernel of the chordal distance is positive definite on the sphere.

::: kernellib.EARTH_RADIUS_KM

::: kernellib.functional.great_circle_distance

::: kernellib.functional.chordal_distance

::: kernellib.functional.lonlat_to_unit

## Kernels on the sphere

`Chordal` maps each ``(lon, lat)`` point to $R\,u(x) \in \mathbb{R}^3$ and
applies any Euclidean kernel there, so the usual RBF, Matérn, rational
quadratic, sums and products all work on global data:

```python
k = kl.Chordal(kl.Matern(nu=1.5, lengthscale=500.0), radius=kl.EARTH_RADIUS_KM)
K = k(X, X)  # (N, N), positive definite
```

It is positive definite whenever the base kernel is: for points $x_i$, the
Gram $[k(x_i, x_j)]$ equals the base kernel's Gram on the points
$R\,u(x_i) \in \mathbb{R}^3$, which is positive semidefinite. A stationary
base kernel sees the chordal distance $2R\sin(\theta/2) \approx R\theta$, so
its lengthscale is in the units of ``radius`` at local scales. The smoothness
of the base kernel carries over to the sphere.

Random-feature maps take the 3-D points; `NystromFeatures` accepts the
`Chordal` kernel directly:

```python
U = k.to_cartesian(X)
rff = kl.RandomFourierFeatures(512, key).fit(k.kernel, U)
Phi = rff(U)  # Phi @ Phi.T ≈ k(X, X)
```

For ``(lon, lat, t)`` inputs, multiply by a kernel in time:
``kl.Product(kl.ActiveDims(k, dims=(0, 1)), kl.ActiveDims(kl.Matern(...), dims=(2,)))``.

::: kernellib.Chordal

## Great-circle kernels

Kernels of the great-circle angle $\theta \in [0, \pi]$ that are proven
positive definite on the sphere: $k(x, x') = \sigma^2\,\psi(\theta / c)$ with
$c = \ell / R$, so ``lengthscale`` is in the units of ``radius`` (kilometres
with ``radius=kl.EARTH_RADIUS_KM``) and ranges mean what they mean in
geostatistics. The families and constraints are those of Gneiting (2013),
Table 1; the constraints are checked at construction ($c \le \pi$ only when
``lengthscale`` is concrete, not under ``jit``). The last three are compactly
supported: exactly zero for $\theta \ge c$, which gives sparse Grams and
tapers.

```python
k = kl.GreatCircleWendland(lengthscale=1500.0, radius=kl.EARTH_RADIUS_KM)
K = k(X, X)  # X: (N, 2) lon/lat in degrees
```

| Kernel | $\psi(t)$, $t = \theta / c$ | Constraints |
|---|---|---|
| `GreatCircleExponential` | $e^{-t}$ | $c > 0$ |
| `GreatCirclePoweredExponential` | $e^{-t^\alpha}$ | $c > 0$, $\alpha \in (0, 1]$ |
| `GreatCircleCauchy` | $(1 + t^\alpha)^{-\tau/\alpha}$ | $c > 0$, $\alpha \in (0, 1]$, $\tau > 0$ |
| `GreatCircleSpherical` | $(1 + t/2)(1 - t)_+^2$ | $c \in (0, \pi]$ |
| `GreatCircleAskey` | $(1 - t)_+^\tau$ | $c \in (0, \pi]$, $\tau \ge 2$ |
| `GreatCircleWendland(order=2)` | $(1 + \tau t)(1 - t)_+^\tau$ | $c \in (0, \pi]$, $\tau \ge 4$ |
| `GreatCircleWendland(order=4)` | $(1 + \tau t + (\tau^2 - 1)t^2/3)(1 - t)_+^\tau$ | $c \in (0, \pi]$, $\tau \ge 6$ |

**Why not RBF on great-circle distance?** A kernel that is positive definite
on $\mathbb{R}^3$ stays so on the sphere when fed the *chordal* distance, but
not when fed the great-circle distance: $\theta$ is not a Euclidean distance.
For $\psi(\theta)$ to be positive definite on $S^2$, its Legendre coefficients
must all be non-negative, and for the Gaussian $e^{-\theta^2 / 2c^2}$ and the
Matérn with $\nu > \tfrac12$ some are negative (Gneiting 2013; Huang, Zhang &
Robeson 2011). The failure is not academic: on 100 Fibonacci points the Gram of
$e^{-\theta^2/2}$ ($c = 1$ rad) has an eigenvalue of about $-4 \times 10^{-3}$,
so a GP with it can report negative variances. Use one of the kernels above, or
a Euclidean kernel of the chordal distance.

::: kernellib.AbstractGreatCircleKernel

::: kernellib.GreatCircleExponential

::: kernellib.GreatCirclePoweredExponential

::: kernellib.GreatCircleCauchy

::: kernellib.GreatCircleSpherical

::: kernellib.GreatCircleAskey

::: kernellib.GreatCircleWendland

## Neighbours and bandwidths on the sphere

`nearest_neighbors`, `radius_neighbors`, `knn_graph`, `radius_graph` and
`estimate_lengthscale` take `metric="great_circle"` or `"chordal"` for
`(lon, lat)` inputs, with `radius` (the sphere's; `sphere_radius` in the two
radius-search functions, whose `radius` is the search radius) and `degrees`.
Distances, heat-kernel bandwidths and lengthscales then come back in units of
the sphere radius, km with `EARTH_RADIUS_KM`.

```python
import kernellib as kl

R = kl.EARTH_RADIUS_KM
knn = kl.nearest_neighbors(X, 10, metric="great_circle", radius=R)  # km
near = kl.radius_neighbors(
    X, 250.0, max_neighbors=64, metric="great_circle", sphere_radius=R
)
graph = kl.knn_graph(X, 10, weighting="heat", metric="great_circle", radius=R)
ell = kl.estimate_lengthscale(X, "median", metric="great_circle", radius=R)
```

The search needs no new code: the chordal distance $2R\sin(\theta/2)$ is
monotone in the great-circle angle $\theta$, so a Euclidean search on the 3-D
unit vectors (any backend) returns exactly the great-circle neighbours, with
no dateline or pole artefacts. The distances of the neighbours found are then
recomputed in the chosen metric.

## Geostatistics models

The standard covariance models of gstools, scikit-gstat and R's gstat, as
stationary kernels $k(x, x') = \sigma^2 \psi(r)$ with
$r = \|(x - x')/\ell\|$. For the compact models $\ell$ is the **range**:
$\psi = 0$ for $r \ge 1$, so their Gram matrices are sparse. The models valid
only for $d \le 3$ raise on wider inputs. None registers a spectral density,
so random Fourier features do not apply to them.

| Model | $\psi(r)$ | Valid $d$ | Smoothness at 0 | Compact? |
|---|---|---|---|---|
| `Spherical` | $1 - \tfrac32 r + \tfrac12 r^3$ | $\le 3$ | $C^0$ | yes |
| `Cubic` | $1 - 7r^2 + \tfrac{35}{4}r^3 - \tfrac72 r^5 + \tfrac34 r^7$ | $\le 3$ | $C^2$ | yes |
| `Pentaspherical` | $1 - \tfrac{15}{8} r + \tfrac54 r^3 - \tfrac38 r^5$ | $\le 3$ | $C^0$ | yes |
| `HoleEffect` | $\sin(r)/r$ | $\le 3$ | $C^\infty$ | no |
| `Stable` | $\exp(-r^\alpha)$, $0 < \alpha \le 2$ | all | $C^0$ ($C^\infty$ at $\alpha = 2$) | no |
| `GeneralizedCauchy` | $(1 + r^\alpha)^{-\beta/\alpha}$, $0 < \alpha \le 2$, $\beta > 0$ | all | $C^0$ ($C^\infty$ at $\alpha = 2$) | no |
| `Wendland` (`order=2`) | $(1 - r)_+^4 (4r + 1)$ | $\le 3$ | $C^2$ | yes |
| `Wendland` (`order=4`) | $(1 - r)_+^6 (35r^2 + 18r + 3)/3$ | $\le 3$ | $C^4$ | yes |

`Stable(alpha=2)` is `RBF` with lengthscale $\ell/\sqrt2$, `Stable(alpha=1)`
is `Matern(nu=0.5)`, and `GeneralizedCauchy(alpha=2, beta=b)` is
`RationalQuadratic` with `alpha=b/2` and lengthscale $\ell/\sqrt b$.

::: kernellib.Spherical

::: kernellib.Cubic

::: kernellib.Pentaspherical

::: kernellib.HoleEffect

::: kernellib.Stable

::: kernellib.GeneralizedCauchy

::: kernellib.Wendland

::: kernellib.functional.spherical_kernel

::: kernellib.functional.cubic_kernel

::: kernellib.functional.pentaspherical_kernel

::: kernellib.functional.hole_effect_kernel

::: kernellib.functional.stable_kernel

::: kernellib.functional.generalized_cauchy_kernel

::: kernellib.functional.wendland_kernel

## Anisotropy

`LinearTransform` evaluates a stationary kernel on a linearly mapped lag,
$k(x, x') = k_0(A(x - x'))$. `GeometricAnisotropy` builds $A$ from rotation
angles and minor/major axis ratios, $A = \operatorname{diag}(1, 1/a_1, \ldots)\,R^\top$:
the correlation contours become an ellipse (2-D) or ellipsoid (3-D) whose
major axis, at angle $\alpha$ counter-clockwise from the x-axis, carries the
base kernel's lengthscale $\ell$, and whose minor axes carry $a_i \ell$.
Unlike an ARD lengthscale, the axes need not be the coordinate axes. In 3-D
$R = R_z(\alpha) R_y(\beta) R_x(\gamma)$ (yaw, pitch, roll).

Both keep the spectral side in closed form,
$S_A(\omega) = |\det A|^{-1} S_0(A^{-\top}\omega)$, with frequencies
$\omega = A^\top \omega_0$, so `RandomFourierFeatures` and the RFF prior
paths accept them. `OrthogonalRandomFeatures` and `FastFoodFeatures` need an
isotropic density and refuse them.

```python
import jax.numpy as jnp
import kernellib as kl

# range 10 along a valley at 30°, 2.5 across it
k = kl.GeometricAnisotropy.from_angles(
    kl.Matern(nu=1.5, lengthscale=10.0), 30.0, 0.25, degrees=True
)
```

::: kernellib.LinearTransform

::: kernellib.GeometricAnisotropy

## Variograms

The empirical semivariogram $\hat\gamma(h)$ is the first step of a
geostatistics workflow: it shows the nugget, sill and range of a field, and
fitting a stationary kernel to it gives a cheap initialiser for GP
hyperparameters. `empirical_variogram` bins the pairs by Euclidean or
great-circle distance (Matheron's or the robust Cressie–Hawkins estimator),
streaming row blocks so memory stays $O(\text{batch}\cdot N)$;
`fit_variogram` fits $\gamma(h) = \tau^2 + \sigma^2 - k(h)$ by weighted least
squares.

```python
import kernellib as kl

v = kl.empirical_variogram(X, y, bins=20, estimator="cressie")
kernel, nugget = kl.fit_variogram(v, kl.Matern(nu=1.5))

# (lon, lat) data: bin by great-circle distance in kilometres
v_km = kl.empirical_variogram(
    X_lonlat, y, metric="great_circle", radius=kl.EARTH_RADIUS_KM
)
```

::: kernellib.Variogram

::: kernellib.empirical_variogram

::: kernellib.fit_variogram

## Spherical-harmonic spectra

An isotropic kernel on $\mathbb{R}^3$ restricted to a sphere of radius $R$ is
zonal, $\kappa(t) = k(Ru, Rv)$ with $t = u \cdot v$, so the Funk–Hecke theorem
gives it one spectral coefficient per spherical-harmonic degree. Two
conventions are available:

- ``convention="funk_hecke"`` (default, as in pyrox-gp):
  $a_l = 2\pi \int_{-1}^{1} \kappa(t) P_l(t)\, dt$, the eigenvalue on each
  degree-$l$ harmonic, with $\kappa(t) = \sum_l \frac{2l + 1}{4\pi} a_l P_l(t)$;
- ``convention="legendre"``: $c_l = \frac{2l + 1}{4\pi} a_l$, the Legendre
  series coefficients, $\kappa(t) = \sum_l c_l P_l(t)$.

```python
a = kl.funk_hecke_coefficients(kl.Matern(nu=1.5, lengthscale=0.3), l_max=40)
```

::: kernellib.funk_hecke_coefficients

## Tapering

Covariance tapering (Furrer, Genton & Nychka 2006) multiplies a covariance
elementwise by a compactly supported taper $T$,
$K_{tap} = K \circ T$. By the Schur product theorem the result is positive
semidefinite whenever $K$ and $T$ are; it keeps $K$'s short-range behaviour,
which dominates kriging weights, and it is exactly zero beyond the taper's
range. `tapered_operator` stores only the non-zeros, about $n \bar m$ of them
for $\bar m$ neighbours per point, as a `gaussx.SparseOperator` that
`gaussx.SparseCholeskySolver` factorises for exact solves and
log-determinants.

The sparsity pattern comes from `radius_neighbors` on the host (so `X` must be
concrete); the values are traced and differentiable in the kernel's
hyperparameters. Pass the `pattern` of an earlier result to skip the search,
which also makes the call `jit`-compatible.

```python
import gaussx as gx
import jax.numpy as jnp
import kernellib as kl

K = kl.tapered_operator(kl.Matern(lengthscale=0.2, nu=1.5), X, taper_range=0.3)
K = K.add_diagonal(jnp.full(X.shape[0], noise))  # same symmetric pattern
logdet = gx.SparseCholeskySolver().logdet(K)

# (lon, lat) stations: a great-circle Wendland taper, range in kilometres
K_geo = kl.tapered_operator(
    kl.GreatCircleExponential(lengthscale=500.0, radius=kl.EARTH_RADIUS_KM),
    X_lonlat,
    taper_range=1000.0,
    metric="great_circle",
    radius=kl.EARTH_RADIUS_KM,
)
```

!!! note "Choosing the taper"
    The taper should be at least as smooth at the origin as the covariance
    (Furrer et al. 2006, Thm 2.2 and §3). For a Matérn with smoothness $\nu$
    in $d \le 3$, use `taper="wendland2"` (Wendland C²) for $\nu \le 1.5$ and
    `taper="wendland4"` (C⁴) for $\nu \le 2.5$. The rule is documented, not
    enforced. On the sphere the taper is `GreatCircleWendland`, which needs
    `taper_range <= π · radius`.

A point with more than `max_neighbors` neighbours within `taper_range`
raises, naming the `max_neighbors` needed: silently dropping pairs would
break positive definiteness. The one- and two-taper likelihood bias
corrections (Kaufman, Schervish & Nychka 2008) are not included.

::: kernellib.Tapered

::: kernellib.tapered_operator
