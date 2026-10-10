"""Test utilities shared by kernellib's test suite.

Copied from gaussx's private ``_testing`` module so the moved tests keep
their assertions without importing gaussx internals.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from geonnax.basis import fibonacci_sphere
from jaxtyping import Array, Bool, Float

from kernellib._einx import einsum


__all__ = [
    "fibonacci_lonlat",
    "random_pd_matrix",
    "tree_allclose",
]


def tree_allclose(
    x: object, y: object, *, rtol: float = 1e-5, atol: float = 1e-8
) -> bool | Bool[Array, ""]:
    """PyTree-aware approximate equality check.

    Wraps ``eqx.tree_equal`` with tolerance support, matching the pattern
    used in the lineax and gaussx test suites.

    Examples:
        >>> import jax.numpy as jnp
        >>> from kernellib._testing import tree_allclose
        >>> bool(tree_allclose(jnp.array([1.0, 2.0]), jnp.array([1.0, 2.0 + 1e-9])))
        True
    """
    return eqx.tree_equal(x, y, typematch=True, rtol=rtol, atol=atol)


def random_pd_matrix(
    key: jax.Array,
    n: int,
    *,
    dtype: jnp.dtype = jnp.float64,
) -> Float[Array, "n n"]:
    """Generate a random positive-definite ``n x n`` matrix.

    Copied from gaussx's private ``_testing`` module.

    Examples:
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> from kernellib._testing import random_pd_matrix
        >>> A = random_pd_matrix(jr.key(0), 4, dtype=jnp.float32)
        >>> bool(jnp.linalg.eigvalsh(A).min() > 0)
        True
    """
    A = jr.normal(key, (n, n), dtype=dtype)
    return einsum(A, A, "i k, j k -> i j") + 0.1 * jnp.eye(n, dtype=dtype)


def fibonacci_lonlat(n: int, *, degrees: bool = True) -> Float[Array, "n 2"]:
    """``n`` near-uniform ``(lon, lat)`` points on the sphere (Fibonacci lattice).

    A thin wrapper over `geonnax.basis.fibonacci_sphere`, for
    positive-definiteness tests of kernels on the sphere.

    Examples:
        >>> from kernellib._testing import fibonacci_lonlat
        >>> fibonacci_lonlat(100).shape
        (100, 2)
    """
    points, _ = fibonacci_sphere(
        n, output="lonlat", output_unit="degrees" if degrees else "radians"
    )
    return points
