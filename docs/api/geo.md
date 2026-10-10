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
