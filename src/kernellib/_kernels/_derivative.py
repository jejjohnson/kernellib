r"""Derivative kernels: covariances of a GP's partial derivatives.

For $f \sim \mathcal{GP}(0, k)$,
$\operatorname{cov}(\partial_i f(x), \partial_j f(x')) =
\partial^2 k / \partial x_i \partial x'_j$. These kernels are autodiff of
`pairwise`, so they compose with `+`, `*`, `to_operator` and the solvers like
any other kernel. Helpers that only *evaluate* derivatives (a derivative
Gram, a predictor's gradient) remain one-line `jax.grad` compositions; see
the "Kernels and JAX" tutorial.
"""

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from kernellib._kernels._base import AbstractKernel, AbstractPointwiseKernel


__all__ = ["Derivative", "DerivativeIndexed", "derivative_inputs"]


def _check_differentiable(kernel: AbstractKernel, name: str) -> None:
    """Reject kernels whose GP is not mean-square differentiable."""
    from kernellib._kernels._nonstationary import Distance
    from kernellib._kernels._stationary import Matern, White

    if not kernel.is_pointwise:
        raise TypeError(
            f"{name} differentiates pairwise(x, y); {type(kernel).__name__} is "
            "Gram-only."
        )
    rough = [
        leaf
        for leaf in jax.tree.leaves(
            kernel, is_leaf=lambda node: isinstance(node, Matern | White | Distance)
        )
        if isinstance(leaf, White)
        or (isinstance(leaf, Matern) and leaf.nu < 1.0)
        or (isinstance(leaf, Distance) and leaf.exponent < 2.0)
    ]
    if rough:
        leaf = rough[0]
        detail = ""
        if isinstance(leaf, Matern):
            detail = f"(nu={leaf.nu})"
        elif isinstance(leaf, Distance):
            detail = f"(exponent={leaf.exponent})"
        raise ValueError(
            f"{name} needs a mean-square differentiable GP, but the kernel "
            f"contains {type(leaf).__name__}{detail}, whose sample paths have "
            "no derivative. Use Matern(nu >= 1.5), RBF or RationalQuadratic."
        )


class Derivative(AbstractPointwiseKernel):
    r"""$\partial_{x_i} \partial_{x'_j} k(x, x')$ for fixed input dimensions.

    ``dx`` differentiates the first argument, ``dy`` the second; ``None``
    leaves that argument undifferentiated. So
    ``Derivative(k, dx=i)`` is $\operatorname{cov}(\partial_i f(x), f(x'))$
    and ``Derivative(k, dx=i, dy=j)`` is
    $\operatorname{cov}(\partial_i f(x), \partial_j f(x'))$ (mlkernels'
    ``k.diff(i, j)``). Only ``dx = dy`` gives a PSD kernel on its own; the
    mixed blocks are cross-covariances for assembling a joint model (see
    `DerivativeIndexed`).

    The kernel must be pointwise and its GP mean-square differentiable
    (`RBF`, `RationalQuadratic`, `Matern` with ``nu >= 1.5``, `Periodic`,
    ...); `Matern(nu=0.5)` and `White` raise.

    Attributes:
        kernel: The kernel to differentiate.
        dx: Input dimension of the first argument, or ``None``.
        dy: Input dimension of the second argument, or ``None``.

    Examples:
        >>> import jax.numpy as jnp
        >>> import kernellib as kl
        >>> k = kl.Derivative(kl.RBF(lengthscale=0.5), dx=0, dy=0)
        >>> x = jnp.zeros(1)
        >>> float(k.pairwise(x, x))  # variance of f': sigma^2 / l^2
        4.0
    """

    kernel: AbstractKernel
    dx: int | None = eqx.field(default=None, static=True)
    dy: int | None = eqx.field(default=None, static=True)

    def __check_init__(self) -> None:
        if self.dx is None and self.dy is None:
            raise ValueError("Derivative needs dx, dy or both.")
        _check_differentiable(self.kernel, "Derivative")

    def pairwise(
        self, x: Float[Array, " D"], y: Float[Array, " D"]
    ) -> Float[Array, ""]:
        for dim in (self.dx, self.dy):
            if dim is not None and not 0 <= dim < x.shape[-1]:
                raise ValueError(
                    f"Derivative dimension {dim} is out of range for "
                    f"{x.shape[-1]}-dimensional inputs."
                )
        f = self.kernel.pairwise
        if self.dx is not None:
            f = _partial(f, argnum=0, dim=self.dx)
        if self.dy is not None:
            f = _partial(f, argnum=1, dim=self.dy)
        return f(x, y)

    @property
    def is_stationary(self) -> bool:
        return self.kernel.is_stationary


def _partial(f, argnum: int, dim: int):
    grad = jax.grad(f, argnums=argnum)
    return lambda x, y: grad(x, y)[dim]


class DerivativeIndexed(AbstractPointwiseKernel):
    r"""Joint covariance of a GP's values and partial derivatives.

    Each input row is ``[x, i]``: a location and what is observed there,
    ``i = -1`` for $f(x)$ and ``i = d`` for $\partial_d f(x)$. The kernel is

    $$
    k([x, i], [x', j]) = \operatorname{cov}(\partial_i f(x), \partial_j f(x')),
    $$

    with $\partial_{-1}$ the identity, so one Gram over mixed rows is the
    covariance of any combination of value and gradient observations (all
    partials, some partials, or none at some points). Build the rows with
    `derivative_inputs`. It keeps the ``(N1, N2)`` Gram contract, so it
    composes with `to_operator`, the solvers and noise like any kernel.

    Each entry evaluates the ``(D + 1) x (D + 1)`` block of value, gradient
    and cross-Hessian and picks one element, ``O(D^2)`` per pair.

    Attributes:
        kernel: The kernel of $f$; pointwise, mean-square differentiable.

    Examples:
        >>> import jax.numpy as jnp
        >>> import kernellib as kl
        >>> X = jnp.array([[0.0], [0.5]])
        >>> Xa = kl.derivative_inputs(X)  # values, then d/dx_0
        >>> Xa.shape
        (4, 2)
        >>> K = kl.DerivativeIndexed(kl.RBF(lengthscale=0.5))(Xa, Xa)
        >>> float(K[2, 2])  # var f'(0) = 1 / l^2
        4.0
    """

    kernel: AbstractKernel

    def __check_init__(self) -> None:
        _check_differentiable(self.kernel, "DerivativeIndexed")

    def pairwise(
        self, x: Float[Array, " D1"], y: Float[Array, " D1"]
    ) -> Float[Array, ""]:
        block = _value_gradient_block(self.kernel.pairwise, x[:-1], y[:-1])
        i = jnp.round(x[-1]).astype(jnp.int32) + 1
        j = jnp.round(y[-1]).astype(jnp.int32) + 1
        return block[i, j]


def _value_gradient_block(f, x: Float[Array, " D"], y: Float[Array, " D"]):
    """``[[k, ∂_y k], [∂_x k, ∂_x ∂_y k]]``, shape ``(D + 1, D + 1)``."""
    value = f(x, y)
    gx = jax.grad(f, argnums=0)(x, y)
    gy = jax.grad(f, argnums=1)(x, y)
    hxy = jax.jacfwd(jax.grad(f, argnums=0), argnums=1)(x, y)
    top = jnp.concatenate([value[None], gy])
    bottom = jnp.concatenate([gx[:, None], hxy], axis=1)
    return jnp.concatenate([top[None, :], bottom], axis=0)


def derivative_inputs(
    X: Float[Array, "N D"],
    dims: Sequence[int] | None = None,
    *,
    values: bool = True,
) -> Float[Array, "M D1"]:
    """Rows ``[x, i]`` for `DerivativeIndexed`: values, then partials.

    Args:
        X: Locations, shape ``(N, D)``.
        dims: Partial derivatives observed at every location; all ``D`` by
            default, ``()`` for none.
        values: Whether the values $f(x)$ are observed too.

    Returns:
        ``(M, D + 1)`` rows, blocked by kind: the ``N`` value rows (if
        ``values``) first, then ``N`` rows per entry of ``dims``, in order.
        Order observations the same way.

    Examples:
        >>> import jax.numpy as jnp
        >>> import kernellib as kl
        >>> kl.derivative_inputs(jnp.zeros((3, 2)), dims=[1]).shape
        (6, 3)
    """
    n, d = X.shape
    dims = tuple(range(d)) if dims is None else tuple(dims)
    if any(not 0 <= i < d for i in dims):
        raise ValueError(f"dims must lie in [0, {d}), got {dims}.")
    kinds = ((-1,) if values else ()) + dims
    if not kinds:
        raise ValueError("derivative_inputs needs values=True or some dims.")
    return jnp.concatenate(
        [
            jnp.concatenate([X, jnp.full((n, 1), i, dtype=X.dtype)], axis=1)
            for i in kinds
        ]
    )
