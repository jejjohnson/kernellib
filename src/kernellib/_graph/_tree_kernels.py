r"""Tail-up and tail-down moving-average kernels on rooted trees.

Ver Hoef & Peterson (2010) build valid covariances on a stream network by
moving-averaging white noise along the network with a kernel whose "tail"
points toward the leaves (tail-up: covariance only between ancestor--
descendant pairs) or toward the root (tail-down: all pairs). Positive
definiteness holds by construction for additive weights, on any rooted tree.
"""

from __future__ import annotations

from typing import Literal

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from kernellib._graph._kernel_classes import _indices
from kernellib._graph._tree import TreeSites
from kernellib._kernels import AbstractPointwiseKernel


__all__ = ["TailDown", "TailUp"]

TailUpModel = Literal["exponential", "linear_sill", "spherical", "mariah"]
TailDownModel = Literal["exponential", "linear_sill", "spherical"]

_MARIAH_SCALE = 90.0  # SSN2 / Ver Hoef & Peterson: range is the effective range


def _tail_up_profile(model: str, h, rng):
    """Unit-variance profile $C_1(h)$ (finite and smooth in ``rng``)."""
    r = h / rng
    if model == "exponential":
        return jnp.exp(-r)
    if model == "linear_sill":
        return jnp.where(r <= 1.0, 1.0 - r, 0.0)
    if model == "spherical":
        return jnp.where(r <= 1.0, 1.0 - 1.5 * r + 0.5 * r**3, 0.0)
    x = _MARIAH_SCALE * r  # mariah
    pos = x > 0
    safe = jnp.where(pos, x, 1.0)
    return jnp.where(pos, jnp.log1p(safe) / safe, 1.0)


def _tail_down_profile(model: str, a, b, rng):
    """Unit-variance profile of the distances $a \\le b$ to the junction."""
    ra, rb = a / rng, b / rng
    if model == "exponential":
        return jnp.exp(-(ra + rb))
    if model == "linear_sill":
        return jnp.where(rb <= 1.0, 1.0 - rb, 0.0)
    return jnp.where(  # spherical
        rb <= 1.0, (1.0 - 1.5 * ra + 0.5 * rb) * (1.0 - rb) ** 2, 0.0
    )


class TailUp(AbstractPointwiseKernel):
    r"""Tail-up covariance on a rooted tree (Ver Hoef & Peterson, 2010).

    Non-zero only between **connected** pairs, i.e. when one site lies on the
    other's path to the root (stream networks: flow-connected sites). With the
    path distance $h_{ij}$ and the weight
    $\pi_{ij} = \sqrt{\Omega(\text{leafward}) / \Omega(\text{rootward})} \le 1$,

    $$
    C_{TU}(s_i, s_j) = \pi_{ij}\, C_1(h_{ij}),
    $$

    with $C_1$, for range $\alpha$ and variance $\sigma^2$:

    - `"exponential"`: $C_1(h) = \sigma^2 e^{-h/\alpha}$
    - `"linear_sill"`: $C_1(h) = \sigma^2 (1 - h/\alpha)\,\mathbf 1\{h \le \alpha\}$
    - `"spherical"`:
      $C_1(h) = \sigma^2 \bigl(1 - \tfrac{3h}{2\alpha}
      + \tfrac{h^3}{2\alpha^3}\bigr)\,\mathbf 1\{h \le \alpha\}$
    - `"mariah"`: $C_1(h) = \sigma^2 \log(90h/\alpha + 1) / (90h/\alpha)$,
      $C_1(0) = \sigma^2$

    The Mariah form uses $\alpha$ as the *effective* range (the factor 90).
    Mixing hydrology's terms in: root / outlet, toward the leaves /
    upstream, connected / flow-connected, weights / watershed area or
    Shreve order.

    **Positive definiteness** holds by the moving-average construction for
    valid (additive) weights, on any rooted tree (binary or not). It does
    **not** extend to graphs with loops (braided channels, meshed grids);
    `RootedTree.from_graph` raises on those. A Euclidean-distance kernel
    (spherical, ...) is *not* valid on path distance, which is why these
    models exist.

    Pseudocode::

        sites = tree.sites(edge, offset)  # host, once
        for i, j in pairs:
            C[i, j] = sites.weight[i, j] * C1(
                sites.h[i, j]
            )  # 0 if not connected

    **Numerics.** Inputs are ``(N, 1)`` site indices into ``sites`` (integers
    or integral floats), like the graph kernels; the Gram gathers rows and
    columns of precomputed ``N x N`` geometry, so the cost is ``O(N1 N2)`` and
    only ``variance`` and ``range`` are traced (both differentiable). The
    Mariah form evaluates $\log(1+x)/x$ with a guarded $x = 0$ branch, so its
    gradients are finite at coincident sites. Full models are sums,
    ``TailUp(...) + TailDown(...) + White(...)``; to add a Euclidean part, put
    the site index and the coordinates in separate columns and use
    `AbstractKernel.select` on each.

    Attributes:
        sites: Precomputed geometry, from `RootedTree.sites`.
        model: One of ``"exponential"``, ``"linear_sill"``, ``"spherical"``,
            ``"mariah"``.
        variance: $\sigma^2$.
        range: $\alpha > 0$ (the effective range for ``"mariah"``).

    References:
        Ver Hoef, J. M. and Peterson, E. E. (2010). A moving average approach
        for spatial statistical models of stream networks. *JASA* 105(489),
        6--18 (Table 1). Ver Hoef, Peterson and Theobald (2006). *Environ.
        Ecol. Stat.* 13, 449--464. R package SSN2 (reference behaviour).

    Examples:
        >>> import jax.numpy as jnp
        >>> import kernellib as kl
        >>> tree = kl.RootedTree.from_parents([-1, 0, 1, 1], [0.0, 2.0, 3.0, 4.0])
        >>> sites = tree.sites([1, 2, 3], [1.0, 1.0, 1.0])
        >>> k = kl.TailUp(sites, "exponential", variance=2.0, range=4.0)
        >>> idx = jnp.array([[0], [1], [2]])
        >>> K = k(idx, idx)
        >>> bool(K[1, 2] == 0.0)  # the two branches are not connected
        True
        >>> round(float(K[0, 1]), 3)  # sqrt(0.5) * 2 * exp(-3/4)
        0.668
    """

    sites: TreeSites
    model: TailUpModel = eqx.field(default="exponential", static=True)
    variance: Float[Array, ""] = eqx.field(default=1.0, converter=jnp.asarray)
    range: Float[Array, ""] = eqx.field(default=1.0, converter=jnp.asarray)

    def __check_init__(self) -> None:
        if self.model not in ("exponential", "linear_sill", "spherical", "mariah"):
            raise ValueError(
                "model must be 'exponential', 'linear_sill', 'spherical' or "
                f"'mariah', got {self.model!r}."
            )

    def _gram(self, i, j):
        h = self.sites.h[i, j]
        w = self.sites.weight[i, j]
        return self.variance * w * _tail_up_profile(self.model, h, self.range)

    def pairwise(self, x: Int[Array, " 1"], y: Int[Array, " 1"]) -> Float[Array, ""]:
        return self._gram(
            jnp.asarray(x).astype(jnp.int32)[0], jnp.asarray(y).astype(jnp.int32)[0]
        )

    def __call__(
        self, X1: Int[Array, "N1 1"], X2: Int[Array, "N2 1"]
    ) -> Float[Array, "N1 N2"]:
        i, j = _indices(X1), _indices(X2)
        h = self.sites.h[i][:, j]
        w = self.sites.weight[i][:, j]
        return self.variance * w * _tail_up_profile(self.model, h, self.range)

    def diag(self, X: Int[Array, "N 1"]) -> Float[Array, " N"]:
        i = _indices(X)
        return (
            self.variance
            * jnp.diag(self.sites.weight)[i]
            * _tail_up_profile(self.model, jnp.diag(self.sites.h)[i], self.range)
        )


class TailDown(AbstractPointwiseKernel):
    r"""Tail-down covariance on a rooted tree (Ver Hoef & Peterson, 2010).

    Non-zero for **every** pair. For a connected pair it is a function of the
    path distance $h$; for any other pair, of the distances $a \le b$ from the
    two sites to their closest common ancestor (junction; stream networks:
    confluence). `TreeSites` stores $a = 0$, $b = h$ for connected pairs, so
    one formula in $(a, b)$ covers all pairs. For range $\alpha$ and variance
    $\sigma^2$:

    - `"exponential"`: $\sigma^2 e^{-h/\alpha}$ if connected, else
      $\sigma^2 e^{-(a+b)/\alpha}$
    - `"linear_sill"`: $\sigma^2 (1 - h/\alpha)\,\mathbf 1\{h \le \alpha\}$ if
      connected, else $\sigma^2 (1 - b/\alpha)\,\mathbf 1\{b \le \alpha\}$
    - `"spherical"`: $\sigma^2 \bigl(1 - \tfrac{3h}{2\alpha}
      + \tfrac{h^3}{2\alpha^3}\bigr)\,\mathbf 1\{h \le \alpha\}$ if connected,
      else $\sigma^2 \bigl(1 - \tfrac{3a}{2\alpha} + \tfrac{b}{2\alpha}\bigr)
      \bigl(1 - \tfrac{b}{\alpha}\bigr)^2\,\mathbf 1\{b \le \alpha\}$

    The two cases agree at $a = 0$, $b = h$, and the covariance is
    continuous across $a = b$. The tail-down Mariah form is not implemented.
    Terms: toward the root / downstream, junction / confluence.

    **Positive definiteness** holds for the moving-average construction on any
    rooted tree; it needs no weights. It does **not** extend to graphs with
    loops; use `matern_graph_kernel` or another graph kernel there.

    Pseudocode::

        sites = tree.sites(edge, offset)  # host, once
        for i, j in pairs:
            C[i, j] = C2(sites.a[i, j], sites.b[i, j])

    **Numerics.** As for `TailUp`: ``(N, 1)`` site indices, gathered from
    precomputed ``N x N`` geometry; ``variance`` and ``range`` differentiable.

    Attributes:
        sites: Precomputed geometry, from `RootedTree.sites`.
        model: One of ``"exponential"``, ``"linear_sill"``, ``"spherical"``.
        variance: $\sigma^2$.
        range: $\alpha > 0$.

    References:
        Ver Hoef, J. M. and Peterson, E. E. (2010). *JASA* 105(489), 6--18
        (Table 1). Peterson and Ver Hoef (2010). *Ecology* 91(3), 644--651.
        R package SSN2 (reference behaviour).

    Examples:
        >>> import jax.numpy as jnp
        >>> import kernellib as kl
        >>> tree = kl.RootedTree.from_parents([-1, 0, 1, 1], [0.0, 2.0, 3.0, 4.0])
        >>> sites = tree.sites([2, 3], [1.0, 1.0])  # one site on each branch
        >>> k = kl.TailDown(sites, "spherical", variance=1.0, range=8.0)
        >>> idx = jnp.array([[0], [1]])
        >>> K = k(idx, idx)
        >>> round(float(K[0, 1]), 5)  # a = 2, b = 3: unconnected, still correlated
        0.31738
    """

    sites: TreeSites
    model: TailDownModel = eqx.field(default="exponential", static=True)
    variance: Float[Array, ""] = eqx.field(default=1.0, converter=jnp.asarray)
    range: Float[Array, ""] = eqx.field(default=1.0, converter=jnp.asarray)

    def __check_init__(self) -> None:
        if self.model not in ("exponential", "linear_sill", "spherical"):
            raise ValueError(
                "model must be 'exponential', 'linear_sill' or 'spherical', "
                f"got {self.model!r}."
            )

    def pairwise(self, x: Int[Array, " 1"], y: Int[Array, " 1"]) -> Float[Array, ""]:
        i = jnp.asarray(x).astype(jnp.int32)[0]
        j = jnp.asarray(y).astype(jnp.int32)[0]
        return self.variance * _tail_down_profile(
            self.model, self.sites.a[i, j], self.sites.b[i, j], self.range
        )

    def __call__(
        self, X1: Int[Array, "N1 1"], X2: Int[Array, "N2 1"]
    ) -> Float[Array, "N1 N2"]:
        i, j = _indices(X1), _indices(X2)
        a = self.sites.a[i][:, j]
        b = self.sites.b[i][:, j]
        return self.variance * _tail_down_profile(self.model, a, b, self.range)

    def diag(self, X: Int[Array, "N 1"]) -> Float[Array, " N"]:
        i = _indices(X)
        return self.variance * _tail_down_profile(
            self.model, jnp.diag(self.sites.a)[i], jnp.diag(self.sites.b)[i], self.range
        )
