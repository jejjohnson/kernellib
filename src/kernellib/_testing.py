"""Test utilities shared by kernellib's test suite.

Copied from gaussx's private ``_testing`` module so the moved tests keep
their assertions without importing gaussx internals.
"""

from __future__ import annotations

import equinox as eqx
from jaxtyping import Array, Bool


__all__ = [
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
