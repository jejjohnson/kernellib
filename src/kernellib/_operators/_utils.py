"""Private utilities shared by the kernel operators.

Copied from gaussx's private ``_operators`` helpers rather than imported, so
kernellib does not depend on gaussx internals.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

import jax


def vmap_over_batch_dims(fn: Callable, num_batch_dims: int) -> Callable:
    """Apply ``jax.vmap`` repeatedly over the leading batch dimensions."""
    for _ in range(num_batch_dims):
        fn = jax.vmap(fn)
    return fn


def _to_frozenset(x: object | frozenset[object]) -> frozenset[object]:
    """Convert a single tag or frozenset of tags to a frozenset."""
    if isinstance(x, frozenset):
        return x
    if isinstance(x, Iterable):
        return frozenset(x)
    return frozenset([x])
