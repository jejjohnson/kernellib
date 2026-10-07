"""Graph Laplacians: unnormalised, symmetric and random-walk."""

from __future__ import annotations

from typing import Literal

import einx
import jax.numpy as jnp
from jaxtyping import Array, Float

from kernellib._einx import reduce


__all__ = ["Normalization", "graph_laplacian"]

Normalization = Literal["unnormalized", "symmetric", "random_walk"]


def graph_laplacian(
    W: Float[Array, "N N"], normalization: Normalization = "unnormalized"
) -> Float[Array, "N N"]:
    r"""Graph Laplacian of a symmetric adjacency matrix.

    - ``"unnormalized"``: $L = D - W$.
    - ``"symmetric"``: $L = I - D^{-1/2} W D^{-1/2}$, eigenvalues in $[0, 2]$.
    - ``"random_walk"``: $L = I - D^{-1} W$ (not symmetric).

    Isolated nodes (zero degree) get a zero row in the normalised forms.

    Examples:
        >>> import jax.numpy as jnp
        >>> import kernellib as kl
        >>> W = jnp.array([[0.0, 1.0], [1.0, 0.0]])
        >>> kl.graph_laplacian(W).tolist()
        [[1.0, -1.0], [-1.0, 1.0]]
    """
    degree = reduce(W, "i j -> i", "sum")
    if normalization == "unnormalized":
        return jnp.diag(degree) - W
    safe = jnp.where(degree > 0, degree, 1.0)
    connected = (degree > 0).astype(W.dtype)
    if normalization == "symmetric":
        d_isqrt = connected / jnp.sqrt(safe)
        scaled = einx.multiply("i, i j -> i j", d_isqrt, W)
        return jnp.diag(connected) - einx.multiply("i j, j -> i j", scaled, d_isqrt)
    if normalization == "random_walk":
        return jnp.diag(connected) - einx.multiply("i, i j -> i j", connected / safe, W)
    raise ValueError(
        "normalization must be 'unnormalized', 'symmetric' or 'random_walk', got "
        f"{normalization!r}."
    )
