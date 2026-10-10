r"""Spherical-harmonic spectrum of a zonal kernel by the Funk-Hecke theorem.

An isotropic Euclidean kernel restricted to a sphere of radius $R$ is zonal:
for unit vectors $u, v$ it depends on $t = u \cdot v$ alone,
$\kappa(t) = k(Ru, Rv)$. The Funk-Hecke theorem then diagonalises it in
spherical harmonics, with one eigenvalue per degree,

$$
a_l = 2\pi \int_{-1}^{1} \kappa(t)\, P_l(t)\, dt,
\qquad
\kappa(t) = \sum_{l} \frac{2l + 1}{4\pi}\, a_l\, P_l(t).
$$

The integral is computed by Gauss-Legendre quadrature, so any kernel works,
not only those with a closed-form series. Ported from pyrox-gp
(``pyrox_gp._inducing.funk_hecke_coefficients``), whose convention
``"funk_hecke"`` is the default.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Iterator
from typing import Literal

import equinox as eqx
import jax.numpy as jnp
from geonnax.basis import gauss_legendre, legendre_polynomials
from jaxtyping import Array, Float

from kernellib._einx import einsum, rearrange
from kernellib._kernels import AbstractKernel


__all__ = ["funk_hecke_coefficients"]


def _submodules(module: eqx.Module) -> Iterator[eqx.Module]:
    """``module`` and every `equinox.Module` nested in its fields."""
    yield module
    for field in dataclasses.fields(module):
        value = getattr(module, field.name, None)
        children = value if isinstance(value, (tuple, list)) else (value,)
        for child in children:
            if isinstance(child, eqx.Module):
                yield from _submodules(child)


def _reject_anisotropic(kernel: AbstractKernel) -> None:
    """Reject per-dimension (ARD) lengthscales anywhere in ``kernel``.

    The coefficients sample the kernel along a single meridian and reuse the
    value for every orientation, which is valid only for a zonal kernel.
    Unequal axis lengthscales make the kernel orientation dependent, so the
    spherical-harmonic covariance stops being diagonal and the coefficients
    would be silently wrong.
    """
    for module in _submodules(kernel):
        lengthscale = getattr(module, "lengthscale", None)
        if lengthscale is not None and jnp.size(lengthscale) > 1:
            raise NotImplementedError(
                "Funk-Hecke coefficients require a zonal (isotropic) kernel, "
                f"but {type(module).__name__} has a per-dimension lengthscale "
                f"of shape {jnp.shape(lengthscale)}. An ARD kernel is "
                "orientation dependent, so its spherical-harmonic covariance "
                "is not diagonal. Use a scalar lengthscale."
            )


def funk_hecke_coefficients(
    kernel: AbstractKernel,
    l_max: int,
    *,
    radius: float = 1.0,
    num_quadrature: int = 256,
    convention: Literal["funk_hecke", "legendre"] = "funk_hecke",
) -> Float[Array, " L"]:
    r"""Per-degree spherical-harmonic spectrum of a kernel on a sphere.

    The kernel must be isotropic on $\mathbb{R}^3$, so that on the sphere of
    radius $R$ it is zonal, $\kappa(t) = k(Ru, Rv)$ with $t = u \cdot v$.
    It is evaluated in one batched call at $R\,(0, 0, 1)$ against
    $R\,(\sqrt{1 - t^2}, 0, t)$ for the quadrature nodes $t$, so gradients
    reach the hyperparameters. Two conventions are returned:

    - ``"funk_hecke"`` (default, pyrox-gp's):
      $a_l = 2\pi \int_{-1}^{1} \kappa(t) P_l(t)\, dt$, with
      $\kappa(t) = \sum_l \frac{2l + 1}{4\pi} a_l P_l(t)$. $a_l$ is the
      eigenvalue of the kernel's integral operator on every degree-$l$
      spherical harmonic of the unit sphere.
    - ``"legendre"``: $c_l = \frac{2l + 1}{4\pi} a_l$, the coefficients of
      the Legendre series $\kappa(t) = \sum_l c_l P_l(t)$.

    The integral uses ``num_quadrature``-node Gauss-Legendre quadrature,
    exact for polynomials of degree $\le 2Q - 1$, so keep ``num_quadrature``
    well above ``l_max`` and above the kernel's effective degree (a short
    lengthscale relative to ``radius`` needs more nodes). A positive-definite
    kernel has $a_l \ge 0$ (Schoenberg 1942); negative values at the level of
    the quadrature error are clipped to zero.

    Args:
        kernel: Isotropic kernel on 3-D inputs. Any component with a
            per-dimension (ARD) lengthscale is rejected.
        l_max: Maximum degree, inclusive, ``>= 0``.
        radius: Sphere radius $R$, in the kernel's input units.
        num_quadrature: Number of Gauss-Legendre nodes.
        convention: ``"funk_hecke"`` for $a_l$, ``"legendre"`` for $c_l$.

    Returns:
        The ``(l_max + 1,)`` coefficients, indexed by degree.

    Raises:
        NotImplementedError: If the kernel has an ARD lengthscale.
        ValueError: If ``l_max`` is negative, ``num_quadrature`` is below 1
            or ``convention`` is unknown.

    Examples:
        >>> import math
        >>> import jax.numpy as jnp
        >>> import kernellib as kl
        >>> k = kl.RBF(lengthscale=0.5, variance=2.0)
        >>> a = kl.funk_hecke_coefficients(k, 30)
        >>> a.shape
        (31,)
        >>> bool(jnp.all(a >= 0.0))
        True
        >>> # sum_l (2l + 1) / (4 pi) a_l = kappa(1) = variance
        >>> c = kl.funk_hecke_coefficients(k, 30, convention="legendre")
        >>> bool(abs(float(jnp.sum(c)) - 2.0) < 1e-4)
        True
        >>> bool(jnp.allclose(c, (2 * jnp.arange(31) + 1) / (4 * math.pi) * a))
        True
    """
    if l_max < 0:
        raise ValueError(f"l_max must be >= 0, got {l_max}.")
    if num_quadrature < 1:
        raise ValueError(f"num_quadrature must be >= 1, got {num_quadrature}.")
    if convention not in ("funk_hecke", "legendre"):
        raise ValueError(
            f"convention must be 'funk_hecke' or 'legendre', got {convention!r}."
        )
    _reject_anisotropic(kernel)

    nodes, weights = gauss_legendre(num_quadrature)  # host-side constants
    return _coefficients(
        kernel,
        jnp.asarray(radius),
        jnp.asarray(nodes),
        jnp.asarray(weights),
        l_max,
        legendre=convention == "legendre",
    )


@eqx.filter_jit
def _coefficients(
    kernel: AbstractKernel,
    radius: Float[Array, ""],
    t: Float[Array, " Q"],
    w: Float[Array, " Q"],
    l_max: int,
    *,
    legendre: bool,
) -> Float[Array, " L"]:
    """Quadrature of ``2 pi int kappa(t) P_l(t) dt``, compiled once per shape."""
    sin_t = jnp.sqrt(jnp.maximum(1.0 - t**2, 0.0))
    n0 = jnp.array([[0.0, 0.0, 1.0]], dtype=t.dtype)  # (1, 3)
    n_t = rearrange(jnp.stack([sin_t, jnp.zeros_like(t), t]), "c q -> q c")
    # One batched (1, Q) call keeps autodiff edges to the hyperparameters.
    kappa = kernel(radius * n0, radius * n_t)[0]  # (Q,)

    P = legendre_polynomials(t, l_max)  # (Q, L)
    a = 2.0 * math.pi * einsum(w * kappa, P, "q, q l -> l")
    # Schoenberg: a_l >= 0 for a PD kernel; drop quadrature noise below 0.
    a = jnp.maximum(a, 0.0)
    if legendre:
        ell = jnp.arange(l_max + 1, dtype=a.dtype)
        return (2.0 * ell + 1.0) / (4.0 * math.pi) * a
    return a
