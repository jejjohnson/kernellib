"""The bridge from kernel objects to gaussx-compatible linear operators.

This is how kernellib inherits scale: after ``to_operator``, everything is
lineax / gaussx (``gaussx.solve``, ``gaussx.logdet``, the solver strategies).
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Float

from kernellib._kernels._base import AbstractKernel
from kernellib._operators._implicit import ImplicitKernelOperator
from kernellib._operators._implicit_cross import ImplicitCrossKernelOperator


__all__ = [
    "to_cross_operator",
    "to_operator",
]

_PSD_TAGS = frozenset({lx.symmetric_tag, lx.positive_semidefinite_tag})


def _split(kernel: AbstractKernel):
    """``(kernel_fn, params)`` for the implicit operators.

    The array leaves become ``params``, so the operators' custom JVPs
    differentiate through the hyperparameters; the static rest is closed over.
    """
    if not kernel.is_pointwise:
        raise TypeError(
            f"implicit=True needs a pointwise kernel; {type(kernel).__name__} "
            "is Gram-only. Use implicit=False to materialise the Gram matrix."
        )
    params, static = eqx.partition(kernel, eqx.is_array)

    def kernel_fn(p, x, y):
        return eqx.combine(p, static).pairwise(x, y)

    return kernel_fn, params


def to_operator(
    kernel: AbstractKernel,
    X: Float[Array, "N D"],
    *,
    noise: float | Float[Array, ""] | None = None,
    implicit: bool = False,
) -> lx.AbstractLinearOperator:
    r"""``K(X, X) (+ noise * I)`` as a symmetric, PSD-tagged linear operator.

    Args:
        kernel: Any kernel. ``implicit=True`` needs a pointwise one.
        X: Inputs, shape ``(N, D)``.
        noise: Optional diagonal noise variance added to the kernel.
        implicit: ``False`` (default) materialises the Gram matrix as a
            `lineax.MatrixLinearOperator`. ``True`` returns a matrix-free
            `ImplicitKernelOperator` with ``O(N)`` memory per matvec and the
            noise fused in; gradients reach the kernel's hyperparameters
            through its custom JVP.

    Returns:
        A square ``(N, N)`` operator for use with `gaussx.solve` and friends.

    Raises:
        TypeError: If ``implicit=True`` and the kernel is Gram-only, or if
            ``implicit=True`` and ``noise`` is a traced value. The implicit
            operator stores its noise as a static Python float, so it cannot
            be jitted over or differentiated; use ``implicit=False`` for that.

    Examples:
        >>> import gaussx as gx
        >>> import jax.numpy as jnp
        >>> import kernellib as kl
        >>> X = jnp.linspace(0.0, 1.0, 5)[:, None]
        >>> K = kl.to_operator(kl.RBF(lengthscale=0.3), X, noise=0.1, implicit=True)
        >>> gx.solve(K, jnp.ones(5)).shape
        (5,)
    """
    if not implicit:
        K = kernel(X, X)
        if noise is not None:
            K = K + noise * jnp.eye(X.shape[0], dtype=K.dtype)
        return lx.MatrixLinearOperator(K, _PSD_TAGS)
    kernel_fn, params = _split(kernel)
    try:
        noise_var = 0.0 if noise is None else float(noise)
    except jax.errors.ConcretizationTypeError as err:
        raise TypeError(
            "implicit=True needs a concrete noise value: ImplicitKernelOperator "
            "stores noise_var as a static float. Use implicit=False to trace "
            "or differentiate the noise."
        ) from err
    return ImplicitKernelOperator(
        kernel_fn, X, noise_var, params=params, tags=_PSD_TAGS
    )


def to_cross_operator(
    kernel: AbstractKernel,
    X1: Float[Array, "N D"],
    X2: Float[Array, "M D"],
    *,
    implicit: bool = False,
    batch_size: int = 1024,
) -> lx.AbstractLinearOperator:
    """``K(X1, X2)`` as a rectangular linear operator.

    Args:
        kernel: Any kernel. ``implicit=True`` needs a pointwise one.
        X1: Row inputs, shape ``(N, D)``.
        X2: Column inputs, shape ``(M, D)``.
        implicit: ``False`` (default) materialises the matrix. ``True``
            returns a matrix-free `ImplicitCrossKernelOperator` processing
            ``batch_size`` rows per step.
        batch_size: Rows per step for the implicit operator.

    Returns:
        An ``(N, M)`` operator.

    Raises:
        TypeError: If ``implicit=True`` and the kernel is Gram-only.

    Examples:
        >>> import jax.numpy as jnp
        >>> import kernellib as kl
        >>> X, Z = jnp.zeros((6, 2)), jnp.ones((3, 2))
        >>> kl.to_cross_operator(kl.RBF(), X, Z, implicit=True).mv(
        ...     jnp.ones(3)
        ... ).shape
        (6,)
    """
    if not implicit:
        return lx.MatrixLinearOperator(kernel(X1, X2))
    kernel_fn, params = _split(kernel)
    return ImplicitCrossKernelOperator(kernel_fn, X1, X2, batch_size, params=params)
