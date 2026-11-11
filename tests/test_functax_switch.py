"""kernellib's outputs pinned at c2d1fce, the last commit that imported geonnax.

kernellib#301 moves its bases, its sphere geometry and its Legendre series to
functax. Every code path that called a moved function is evaluated here on
fixed inputs and compared with ``data/functax_switch_c2d1fce.npz`` at
``atol=1e-12`` in float64: the Wendland, sphere-series, latitude, Slepian,
spherical-harmonic and Laplace kernels, the vector and random features, the
Funk-Hecke spectrum, the sphere distances and the Legendre series with its
gradient at the ends of [-1, 1]. Feature maps are compared through their Gram
matrices Phi Phi^T, which is the kernel they approximate.

The fixture was written by running this module as a script at c2d1fce
(``uv run python tests/test_functax_switch.py``); rerun it only to re-pin on
purpose. Slow tier: the eager spherical-harmonic and Slepian builds dominate.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from pathlib import Path

import jax


jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

import kernellib as kl
import kernellib.functional as F
from kernellib._einx import einsum
from kernellib._testing import fibonacci_lonlat


FIXTURE = Path(__file__).parent / "data" / "functax_switch_c2d1fce.npz"
ATOL = 1e-12

Outputs = dict[str, np.ndarray]


def _inputs() -> tuple[jax.Array, jax.Array, jax.Array]:
    rng = np.random.default_rng(0)
    X3 = jnp.asarray(rng.uniform(-1.0, 1.0, (12, 3)))
    X2 = jnp.asarray(rng.uniform(-1.5, 1.5, (12, 2)))
    lon, lat = rng.uniform(-180.0, 180.0, 12), rng.uniform(-89.0, 89.0, 12)
    return X3, X2, jnp.asarray(np.column_stack([lon, lat]))


def _gram(Phi: jax.Array) -> np.ndarray:
    """Phi Phi^T; for (N, M, 2) vector features, the (N, 2, N, 2) block Gram."""
    if Phi.ndim == 3:
        return np.asarray(einsum(Phi, Phi, "n m a, k m b -> n a k b"))
    return np.asarray(einsum(Phi, Phi, "n m, k m -> n k"))


def _geometry() -> Outputs:
    _, _, lonlat = _inputs()
    return {
        "unit_degrees": np.asarray(F.lonlat_to_unit(lonlat)),
        "unit_radians": np.asarray(
            F.lonlat_to_unit(jnp.deg2rad(lonlat), degrees=False)
        ),
        "unit_integer": np.asarray(F.lonlat_to_unit(jnp.asarray([[90, 0], [0, 45]]))),
        "great_circle": np.asarray(F.great_circle_distance(lonlat, lonlat)),
        "chordal": np.asarray(F.chordal_distance(lonlat, lonlat)),
        "fibonacci_degrees": np.asarray(fibonacci_lonlat(40)),
        "fibonacci_radians": np.asarray(fibonacci_lonlat(40, degrees=False)),
    }


def _legendre(series: Callable) -> Outputs:
    rng = np.random.default_rng(1)
    x = jnp.concatenate([jnp.linspace(-1.0, 1.0, 41), jnp.array([1.0 + 1e-12])])
    out = {}
    for L in (0, 1, 2, 7, 64):
        c = jnp.asarray(rng.standard_normal(L + 1))
        grad = jax.grad(lambda t, c=c: series(t, c))
        out[f"legendre_{L}"] = np.asarray(series(x, c))
        out[f"legendre_{L}_grad_at_1"] = np.asarray(grad(jnp.array(1.0)))
        out[f"legendre_{L}_grad_at_-1"] = np.asarray(grad(jnp.array(-1.0)))
    return out


def _wendland() -> Outputs:
    X3, _, _ = _inputs()
    out = {}
    for order in (2, 4):
        k = kl.Wendland(lengthscale=1.3, variance=0.8, order=order)
        out[f"wendland_{order}"] = np.asarray(k(X3, X3))
        out[f"wendland_kernel_{order}"] = np.asarray(
            F.wendland_kernel(X3, X3, jnp.array(0.8), jnp.array(1.3), order=order)
        )
    return out


def _sphere_series() -> Outputs:
    _, _, lonlat = _inputs()
    kernels = {
        "sphere_matern": kl.SphereMatern(lengthscale=0.5, nu=1.5, variance=2.0),
        "sphere_heat": kl.SphereHeat(lengthscale=0.4, variance=0.7),
        "sphere_series": kl.SphereSeries(lambda lam: (1.0 + lam) ** -2.0),
    }
    out = {}
    for name, k in kernels.items():
        out[name] = np.asarray(k(lonlat, lonlat))
        # the gradient runs through the Gram's diagonal, where u . u = 1
        out[f"{name}_grad"] = np.asarray(
            jax.grad(lambda X, k=k: jnp.sum(k(X, X)))(lonlat)
        )
    fn = kl.LegendreLatitudeFn(jnp.array([1.0, 0.3, -0.2, 0.1]))
    out["legendre_latitude_fn"] = np.asarray(jax.vmap(fn)(jnp.deg2rad(lonlat[:, 1])))
    scaled = kl.LatitudeScaled(kl.SphereMatern(lengthscale=0.5), fn)
    out["latitude_scaled"] = np.asarray(scaled(lonlat, lonlat))
    return out


def _sphere_spectral() -> Outputs:
    _, _, lonlat = _inputs()
    matern = kl.SphereMatern(lengthscale=0.5, nu=1.5, variance=2.0)
    chordal = kl.Chordal(kl.RBF(lengthscale=0.6))
    slepian = kl.SlepianFeatures(
        l_max=10, cap_radius=float(np.radians(30.0)), cap_centre=(20.0, -10.0)
    ).fit(matern, lonlat)
    assert slepian.basis is not None
    return {
        "funk_hecke_chordal_rbf": np.asarray(kl.funk_hecke_coefficients(chordal, 20)),
        "spherical_harmonic_matern": _gram(
            kl.SphericalHarmonicFeatures().fit(matern, lonlat)(lonlat)
        ),
        "spherical_harmonic_chordal": _gram(
            kl.SphericalHarmonicFeatures(max_degree=12).fit(chordal, lonlat)(lonlat)
        ),
        "slepian": _gram(slepian(lonlat)),
        "slepian_eigenvalues": np.asarray(slepian.basis.eigenvalues),
    }


def _box_spectral() -> Outputs:
    X3, X2, _ = _inputs()
    rbf = kl.RBF(lengthscale=0.7, variance=1.2)
    out = {
        "laplace_2d": _gram(kl.LaplaceEigenfunctionFeatures(6, L=2.5).fit(rbf, X2)(X2)),
        "laplace_3d": _gram(
            kl.LaplaceEigenfunctionFeatures((3, 4, 5)).fit(kl.Matern(nu=1.5), X3)(X3)
        ),
    }
    for bc in ("dirichlet", "neumann"):
        for name, cls in (
            ("divergence_free", kl.DivergenceFreeFeatures),
            ("curl_free", kl.CurlFreeFeatures),
        ):
            out[f"{name}_{bc}"] = _gram(cls(5, L=2.5, boundary=bc).fit(rbf, X2)(X2))
    return out


def _random_features() -> Outputs:
    X3, _, _ = _inputs()
    key = jax.random.key(7)
    out = {}
    for name, cls in (
        ("rff", kl.RandomFourierFeatures),
        ("orf", kl.OrthogonalRandomFeatures),
    ):
        for kname, k in (("rbf", kl.RBF(lengthscale=0.7)), ("matern", kl.Matern())):
            out[f"{name}_{kname}"] = _gram(cls(16, key).fit(k, X3)(X3))
    return out


def _legendre_series() -> Outputs:
    # kernellib.functional.legendre_series is the public path at c2d1fce; after
    # the switch it is functax's function behind a DeprecationWarning
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        series = F.legendre_series
    return _legendre(series)


GROUPS: dict[str, Callable[[], Outputs]] = {
    "geometry": _geometry,
    "legendre_series": _legendre_series,
    "wendland": _wendland,
    "sphere_series": _sphere_series,
    "box_spectral": _box_spectral,
    "random_features": _random_features,
    "sphere_spectral": _sphere_spectral,
}


@pytest.mark.slow
@pytest.mark.parametrize("group", list(GROUPS))
def test_matches_c2d1fce(group: str) -> None:
    pinned = np.load(FIXTURE)
    got = GROUPS[group]()
    for name, value in got.items():
        np.testing.assert_allclose(
            value, pinned[name], rtol=0.0, atol=ATOL, err_msg=f"{group}: {name}"
        )


if __name__ == "__main__":
    FIXTURE.parent.mkdir(exist_ok=True)
    arrays = {name: v for group in GROUPS.values() for name, v in group().items()}
    np.savez_compressed(FIXTURE, **arrays)
    print(f"wrote {len(arrays)} arrays to {FIXTURE}")
