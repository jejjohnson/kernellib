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
