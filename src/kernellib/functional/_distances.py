"""Lengthscale-scaled pairwise squared distances.

Private helper shared by every distance-based kernel in `kernellib.functional`.
Ported verbatim from ``pyrox_gp._src.kernels`` so kernel values are
bit-identical to pyrox-gp's. Routing through `gaussx.stable_squared_distances`
was evaluated and deferred: that function casts to ``float32`` by default,
and called with the input dtype it performs the same expansion as below.
"""

from __future__ import annotations

import einx
import jax.numpy as jnp
from jaxtyping import Array, Float


def _pairwise_sq_dist(
    X1: Float[Array, "N1 D"],
    X2: Float[Array, "N2 D"],
    lengthscale: Float[Array, ""] | Float[Array, " D"] | float = 1.0,
) -> Float[Array, "N1 N2"]:
    """Squared Euclidean distance matrix, optionally lengthscale-scaled.

    Two paths, chosen by the (statically known) rank of ``lengthscale``:

    * **Isotropic.** Expand on the raw coordinates and scale the resulting
      squared distance — bit-identical to the pre-ARD implementation, and
      robust to a small lengthscale because the division happens after the
      cancellation, so a self-distance of exactly zero stays zero.
    * **ARD.** Scale the inputs before the expansion. This is what supports
      a per-dimension lengthscale while keeping every intermediate at
      ``(N1,)``, ``(N2,)``, or ``(N1, N2)``; no ``(N1, N2, D)`` broadcast
      tensor is ever built.

    Both clip at zero to absorb the small negative values that arise from
    float cancellation on near-identical points.

    Note:
        The ARD path expands on scaled coordinates, so — like any
        norm-expansion distance — it loses separations that are small
        relative to the scaled magnitudes, and overflows once
        ``spread / lengthscale`` exceeds the dtype's range. Centring on a
        data-dependent offset was considered and rejected: it makes the
        result depend on which argument is ``X1`` (breaking
        ``K(X1, X2) == K(X2, X1).T``) and can erase locally representable
        differences when one input set has a large spread. The isotropic
        path divides after the expansion and is unaffected by all of this.

    Args:
        X1: ``(N1, D)`` inputs.
        X2: ``(N2, D)`` inputs.
        lengthscale: Scalar (isotropic) or ``(D,)`` (ARD) lengthscale.

    Returns:
        ``(N1, N2)`` scaled squared distances.
    """
    if jnp.ndim(lengthscale) == 0:
        n1 = einx.dot("n1 d, n1 d -> n1", X1, X1)
        n2 = einx.dot("n2 d, n2 d -> n2", X2, X2)
        cross = einx.dot("n1 d, n2 d -> n1 n2", X1, X2)
        # ‖xᵢ‖² + ‖x′ⱼ‖² broadcast to (N1, N2) via a named outer sum.
        norm_sum = einx.add("n1, n2 -> n1 n2", n1, n2)
        sq = jnp.clip(norm_sum - 2.0 * cross, min=0.0)
        return sq / lengthscale**2

    d = jnp.size(lengthscale)
    if X1.shape[-1] != d or X2.shape[-1] != d:
        raise ValueError(
            f"ARD lengthscale of size {d} requires both inputs to have that "
            f"many features; got X1 with {X1.shape[-1]} and X2 with "
            f"{X2.shape[-1]}. A singleton feature axis would broadcast "
            "silently and repeat one coordinate across every dimension."
        )
    X1 = X1 / lengthscale
    X2 = X2 / lengthscale
    n1 = einx.dot("n1 d, n1 d -> n1", X1, X1)
    n2 = einx.dot("n2 d, n2 d -> n2", X2, X2)
    cross = einx.dot("n1 d, n2 d -> n1 n2", X1, X2)
    norm_sum = einx.add("n1, n2 -> n1 n2", n1, n2)
    return jnp.clip(norm_sum - 2.0 * cross, min=0.0)
