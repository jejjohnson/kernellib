"""Fitting a feature-map template for the randomised dependence measures."""

from __future__ import annotations

import dataclasses

import jax
from jaxtyping import Array, Float

from kernellib._kernels import AbstractKernel
from kernellib._spectral import AbstractFeatureMap


def _fit_pair(
    approx: AbstractFeatureMap,
    kernel_x: AbstractKernel,
    X: Float[Array, "N Dx"],
    kernel_y: AbstractKernel,
    Y: Float[Array, "N Dy"],
) -> tuple[Float[Array, "N Rx"], Float[Array, "N Ry"]]:
    """Features of ``X`` and ``Y`` from two independent fits of ``approx``.

    A map with a ``key`` gets two keys split from it, so the two
    approximations are independent draws.
    """
    approx_x, approx_y = approx, approx
    key = getattr(approx, "key", None)
    if key is not None:
        key_x, key_y = jax.random.split(key)
        approx_x = dataclasses.replace(approx, key=key_x)
        approx_y = dataclasses.replace(approx, key=key_y)
    return approx_x.fit(kernel_x, X)(X), approx_y.fit(kernel_y, Y)(Y)
