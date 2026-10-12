r"""Derivative operators of callables: gradient, Jacobian, Hessian, Laplacian,
divergence and curl.

A **scalar field** is any callable ``f: (N, D) -> (N,)`` and a **vector
field** any callable ``F: (N, D) -> (N, M)``: a fitted ``KRR`` / ``Falkon`` /
``EigenPro`` ``predict``, a closure over a kernel expansion, or any JAX
function. Every operator returns exact (autodiff) per-point derivatives,
batched over ``N``.

Contract:

- **Pointwise.** ``f`` is evaluated one row at a time,
  $f_n(x) := f(x[\text{None}])[0]$, and the operators differentiate that
  single-point function. When ``f(X)[n]`` depends only on ``X[n]`` (every
  predictor), this is the derivative of ``f(X)[n]`` with respect to
  ``X[n]``. For a functional of the whole dataset (HSIC as a function of
  $X$), use `jax.grad` directly.
- **Batching.** ``batch_size=None`` maps the per-point derivative over all
  ``N`` rows with `jax.vmap`; an integer uses
  ``jax.lax.map(..., batch_size=batch_size)``, so peak memory is
  ``O(batch_size * cost(f))`` (a ``(batch_size, N_train)`` kernel block for
  KRR).
- **Guard.** With ``check=True``, a bound method whose owner has a ``kernel``
  field that is an `AbstractKernel` (``KRR.predict``, ``Falkon.predict``,
  ``EigenPro.predict``, an adapter's ``model_.predict``) is refused when that
  kernel is rough at ``x = x_i``: ``Matern(nu=0.5)``, ``White`` and
  ``Distance(exponent < 2)``. The test is conservative for ``Distance`` with
  ``1 < exponent < 2``, whose expansion is $C^1$ but not $C^2$. Other
  callables are not checked; ``check=False`` opts out.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from kernellib._einx import rearrange
from kernellib._kernels._base import AbstractKernel
from kernellib._kernels._derivative import _reject_rough


__all__ = ["curl", "divergence", "gradient", "hessian", "jacobian", "laplacian"]


ScalarField = Callable[[Float[Array, "N D"]], Float[Array, " N"]]
VectorField = Callable[[Float[Array, "N D"]], Float[Array, "N M"]]


def _pointwise(f: Callable) -> Callable:
    """``x (D,) -> f(x[None])[0]``: one row at a time (the pointwise contract)."""
    return lambda x: f(rearrange(x, "d -> 1 d"))[0]


def _over_points(fn: Callable, X: Float[Array, "N D"], batch_size: int | None) -> Array:
    """Map a per-point function over the rows of ``X``."""
    if batch_size is None:
        return jax.vmap(fn)(X)
    if batch_size < 1:
        raise ValueError(f"batch_size must be a positive integer, got {batch_size}.")
    return jax.lax.map(fn, X, batch_size=batch_size)


def _maybe_check(f: Callable, name: str, check: bool) -> None:
    """Refuse a bound method of an estimator whose kernel is rough at ``x_i``.

    Duck typed (``f.__self__.kernel``), so no import from ``_regression``.
    """
    if not check:
        return
    kernel = getattr(getattr(f, "__self__", None), "kernel", None)
    if isinstance(kernel, AbstractKernel):
        _reject_rough(kernel, name)


def _output_shape(f: Callable, X: Float[Array, "N D"], name: str) -> tuple[int, ...]:
    """Validate ``X`` and return the per-row output shape of ``f``."""
    if X.ndim != 2:
        raise ValueError(f"{name} needs X of shape (N, D), got {X.shape}.")
    out = jax.eval_shape(f, jax.ShapeDtypeStruct((1, X.shape[1]), X.dtype))
    if len(out.shape) == 0 or out.shape[0] != 1:
        raise ValueError(
            f"{name}: f must map (N, D) to (N, ...), but a (1, {X.shape[1]}) "
            f"input gave {out.shape}."
        )
    return tuple(out.shape[1:])


def _scalar_field(f: Callable, X: Float[Array, "N D"], name: str) -> None:
    shape = _output_shape(f, X, name)
    if shape:
        raise ValueError(
            f"{name} needs a scalar field f: (N, D) -> (N,), but f returns "
            f"(N, {', '.join(map(str, shape))}); use jacobian for vector fields."
        )


def _vector_field(F: Callable, X: Float[Array, "N D"], name: str) -> int:
    shape = _output_shape(F, X, name)
    if len(shape) != 1:
        raise ValueError(
            f"{name} needs a vector field F: (N, D) -> (N, M), but F returns "
            f"(N{''.join(f', {s}' for s in shape)})."
        )
    return shape[0]


def _square_field(F: Callable, X: Float[Array, "N D"], name: str) -> None:
    m = _vector_field(F, X, name)
    if m != X.shape[1]:
        raise ValueError(
            f"{name} needs F: (N, D) -> (N, D), but F maps D = {X.shape[1]} "
            f"inputs to M = {m} outputs."
        )


def _resolve_mode(
    mode: Literal["auto", "forward", "reverse"], d: int, m: int
) -> Literal["forward", "reverse"]:
    """``auto``: forward (``D`` JVPs) iff ``D <= M``, else reverse (``M`` VJPs)."""
    if mode == "auto":
        return "forward" if d <= m else "reverse"
    if mode not in ("forward", "reverse"):
        raise ValueError(f"mode must be 'auto', 'forward' or 'reverse', got {mode!r}.")
    return mode


def gradient(
    f: ScalarField,
    X: Float[Array, "N D"],
    *,
    batch_size: int | None = None,
    check: bool = True,
) -> Float[Array, "N D"]:
    r"""Per-point gradient $\nabla f(x_n)$ of a scalar field.

    For a kernel expansion $\hat f(x) = \sum_i \alpha_i k(x, x_i)$ this is
    $\sum_i \alpha_i \nabla_x k(x, x_i)$ (Johnson et al. 2020, Eq. 8). One
    reverse-mode pass (VJP) per point.

    Args:
        f: Scalar field, ``(N, D) -> (N,)``, evaluated one row at a time.
        X: Points, shape ``(N, D)``.
        batch_size: ``None`` vmaps over all points; an integer maps over
            them in chunks of this size (bounded memory).
        check: Refuse an estimator's bound ``predict`` whose kernel is not
            differentiable at the training points (see the module docstring).

    Returns:
        ``(N, D)`` gradients, row ``n`` is $\nabla f(x_n)$.

    Raises:
        ValueError: If ``X`` is not 2-D, ``f`` does not return ``(N,)``, or
            the owner's kernel is rough.

    Examples:
        >>> import jax.numpy as jnp
        >>> import kernellib as kl
        >>> f = lambda X: X[:, 0] ** 2 + X[:, 1] ** 2
        >>> kl.gradient(f, jnp.array([[1.0, 2.0], [0.5, -1.0]])).tolist()
        [[2.0, 4.0], [1.0, -2.0]]
    """
    _scalar_field(f, X, "gradient")
    _maybe_check(f, "gradient", check)
    return _over_points(jax.grad(_pointwise(f)), X, batch_size)


def jacobian(
    F: VectorField,
    X: Float[Array, "N D"],
    *,
    mode: Literal["auto", "forward", "reverse"] = "auto",
    batch_size: int | None = None,
    check: bool = True,
) -> Float[Array, "N M D"]:
    r"""Per-point Jacobian $J_n[i, j] = \partial F_i(x_n) / \partial x^j$.

    ``mode="forward"`` pushes ``D`` JVPs per point (`jax.jacfwd`),
    ``"reverse"`` pulls ``M`` VJPs (`jax.jacrev`); ``"auto"`` picks forward
    iff ``D <= M``, so the cost is $\min(D, M)$ passes.

    Args:
        F: Vector field, ``(N, D) -> (N, M)``, evaluated one row at a time.
        X: Points, shape ``(N, D)``.
        mode: ``"auto"``, ``"forward"`` or ``"reverse"``.
        batch_size: ``None`` vmaps over all points; an integer maps over
            them in chunks of this size.
        check: Refuse an estimator's bound ``predict`` whose kernel is not
            differentiable at the training points.

    Returns:
        ``(N, M, D)`` Jacobians.

    Raises:
        ValueError: If ``X`` is not 2-D, ``F`` does not return ``(N, M)``,
            ``mode`` is unknown, or the owner's kernel is rough.

    Examples:
        >>> import jax.numpy as jnp
        >>> import kernellib as kl
        >>> F = lambda X: jnp.stack([X[:, 0] * X[:, 1], X[:, 0]], axis=-1)
        >>> kl.jacobian(F, jnp.array([[2.0, 3.0]])).tolist()
        [[[3.0, 2.0], [1.0, 0.0]]]
    """
    m = _vector_field(F, X, "jacobian")
    _maybe_check(F, "jacobian", check)
    jac = jax.jacfwd if _resolve_mode(mode, X.shape[1], m) == "forward" else jax.jacrev
    return _over_points(jac(_pointwise(F)), X, batch_size)


def hessian(
    f: ScalarField,
    X: Float[Array, "N D"],
    *,
    batch_size: int | None = None,
    check: bool = True,
) -> Float[Array, "N D D"]:
    r"""Per-point Hessian $H_n[j, k] = \partial^2 f(x_n) / \partial x^j \partial x^k$.

    For a kernel expansion this is $\sum_i \alpha_i \partial_j \partial_k
    k(x, x_i)$ (Johnson et al. 2020, Eq. 10). Forward-over-reverse: ``D``
    Hessian-vector products per point.

    Args:
        f: Scalar field, ``(N, D) -> (N,)``, evaluated one row at a time.
        X: Points, shape ``(N, D)``.
        batch_size: ``None`` vmaps over all points; an integer maps over
            them in chunks of this size.
        check: Refuse an estimator's bound ``predict`` whose kernel is not
            twice differentiable at the training points.

    Returns:
        ``(N, D, D)`` symmetric Hessians.

    Raises:
        ValueError: If ``X`` is not 2-D, ``f`` does not return ``(N,)``, or
            the owner's kernel is rough.

    Examples:
        >>> import jax.numpy as jnp
        >>> import kernellib as kl
        >>> f = lambda X: X[:, 0] ** 2 * X[:, 1]
        >>> kl.hessian(f, jnp.array([[1.0, 2.0]])).tolist()
        [[[4.0, 2.0], [2.0, 0.0]]]
    """
    _scalar_field(f, X, "hessian")
    _maybe_check(f, "hessian", check)
    return _over_points(jax.jacfwd(jax.grad(_pointwise(f))), X, batch_size)


def laplacian(
    f: ScalarField,
    X: Float[Array, "N D"],
    *,
    batch_size: int | None = None,
    check: bool = True,
) -> Float[Array, " N"]:
    r"""Per-point Laplacian $\Delta f(x_n) = \operatorname{tr} H_n$.

    $\Delta f = \sum_j \partial_j^2 f$ (Johnson et al. 2020, Eq. 9). Exact:
    the trace of the per-point Hessian, ``D`` Hessian-vector products per
    point.

    Args:
        f: Scalar field, ``(N, D) -> (N,)``, evaluated one row at a time.
        X: Points, shape ``(N, D)``.
        batch_size: ``None`` vmaps over all points; an integer maps over
            them in chunks of this size.
        check: Refuse an estimator's bound ``predict`` whose kernel is not
            twice differentiable at the training points.

    Returns:
        ``(N,)`` Laplacians.

    Raises:
        ValueError: If ``X`` is not 2-D, ``f`` does not return ``(N,)``, or
            the owner's kernel is rough.

    Examples:
        >>> import jax.numpy as jnp
        >>> import kernellib as kl
        >>> f = lambda X: X[:, 0] ** 2 + X[:, 1] ** 2 + X[:, 2] ** 2
        >>> kl.laplacian(f, jnp.zeros((2, 3))).tolist()
        [6.0, 6.0]
    """
    _scalar_field(f, X, "laplacian")
    _maybe_check(f, "laplacian", check)
    hess = jax.jacfwd(jax.grad(_pointwise(f)))
    return _over_points(lambda x: jnp.trace(hess(x)), X, batch_size)


def divergence(
    F: VectorField,
    X: Float[Array, "N D"],
    *,
    batch_size: int | None = None,
    check: bool = True,
) -> Float[Array, " N"]:
    r"""Per-point divergence $\nabla \cdot F(x_n) = \operatorname{tr} J_n$.

    $\nabla \cdot F = \sum_i \partial_i F_i$. Needs ``M == D``. Exact: the
    trace of the forward-mode Jacobian, ``D`` JVPs per point.

    Args:
        F: Vector field, ``(N, D) -> (N, D)``, evaluated one row at a time.
        X: Points, shape ``(N, D)``.
        batch_size: ``None`` vmaps over all points; an integer maps over
            them in chunks of this size.
        check: Refuse an estimator's bound ``predict`` whose kernel is not
            differentiable at the training points.

    Returns:
        ``(N,)`` divergences.

    Raises:
        ValueError: If ``X`` is not 2-D, ``F`` does not return ``(N, D)``, or
            the owner's kernel is rough.

    Examples:
        >>> import jax.numpy as jnp
        >>> import kernellib as kl
        >>> F = lambda X: X  # the identity field (x, y, z)
        >>> kl.divergence(F, jnp.ones((2, 3))).tolist()
        [3.0, 3.0]
    """
    _square_field(F, X, "divergence")
    _maybe_check(F, "divergence", check)
    jac = jax.jacfwd(_pointwise(F))
    return _over_points(lambda x: jnp.trace(jac(x)), X, batch_size)


def _curl_from_jacobian(J: Float[Array, "D D"]) -> Array:
    """``J[i, j] = dF_i/dx_j``; scalar curl in 2-D, vector curl in 3-D."""
    if J.shape[0] == 2:
        return J[1, 0] - J[0, 1]
    return jnp.stack([J[2, 1] - J[1, 2], J[0, 2] - J[2, 0], J[1, 0] - J[0, 1]])


def curl(
    F: VectorField,
    X: Float[Array, "N D"],
    *,
    batch_size: int | None = None,
    check: bool = True,
) -> Float[Array, " N"] | Float[Array, "N 3"]:
    r"""Per-point curl of a 2-D or 3-D vector field.

    With $J_{ij} = \partial F_i / \partial x^j$ (0-indexed): in 2-D the
    scalar curl (vorticity) $\partial_0 F_1 - \partial_1 F_0 = J_{10} -
    J_{01}$; in 3-D the vector curl $(J_{21} - J_{12},\ J_{02} - J_{20},\
    J_{10} - J_{01})$. ``D`` JVPs per point.

    Args:
        F: Vector field, ``(N, D) -> (N, D)`` with ``D`` 2 or 3.
        X: Points, shape ``(N, D)``.
        batch_size: ``None`` vmaps over all points; an integer maps over
            them in chunks of this size.
        check: Refuse an estimator's bound ``predict`` whose kernel is not
            differentiable at the training points.

    Returns:
        ``(N,)`` scalar curls for ``D = 2``, ``(N, 3)`` curls for ``D = 3``.

    Raises:
        ValueError: If ``X`` is not 2-D, ``D`` is not 2 or 3, ``F`` does not
            return ``(N, D)``, or the owner's kernel is rough.

    Examples:
        >>> import jax.numpy as jnp
        >>> import kernellib as kl
        >>> F = lambda X: jnp.stack([-X[:, 1], X[:, 0]], axis=-1)  # rotation
        >>> kl.curl(F, jnp.ones((2, 2))).tolist()
        [2.0, 2.0]
    """
    if X.ndim == 2 and X.shape[1] not in (2, 3):
        raise ValueError(
            "curl is defined for D = 2 (scalar) and D = 3 (vector), got "
            f"X of shape {X.shape}."
        )
    _square_field(F, X, "curl")
    _maybe_check(F, "curl", check)
    jac = jax.jacfwd(_pointwise(F))
    return _over_points(lambda x: _curl_from_jacobian(jac(x)), X, batch_size)
