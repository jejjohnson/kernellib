r"""Sensitivity maps: per-feature and per-sample relevance from derivatives.

Johnson et al. (2020), *Kernel methods and their derivatives*, PLOS ONE
15(10): e0235885 (https://doi.org/10.1371/journal.pone.0235885), Eqs 11-13.
Both summaries reduce one ``(N, D)``
derivative matrix $G$ with a transform $\tau(u) = u^2$ (default) or $|u|$:

- `sensitivity`: $\hat s_j = \frac1N \sum_n \tau(G_{nj})$, shape ``(D,)``;
- `point_sensitivity`: $q_n = \frac1D \sum_j \tau(G_{nj})$, shape ``(N,)``.

$G$ comes from one of two kinds of callable:

- a **pointwise predictor** ``f: (N, D) -> (N,)`` (``model.predict``, a
  classifier margin, any JAX function): $G_{n\cdot} = \nabla f(x_n)$, the
  KD1 `gradient` under the pointwise contract;
- a **dataset functional** ``F: (N, D) -> ()`` (HSIC, MMD, a loss as a
  function of the sample): $G = \nabla_X F$, row $n$ being how $F$ changes
  when sample $n$ moves.

``wrt="auto"`` picks the mode from the output shape of ``f(X)``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from kernellib._derivatives._operators import _maybe_check, gradient, hessian
from kernellib._einx import einsum, reduce


__all__ = ["point_sensitivity", "sensitivity"]


Wrt = Literal["auto", "point", "dataset"]
Order = Literal[1, 2]
Transform = Literal["square", "abs"]
Normalize = Literal["sum", "max"] | None
PredictorOrFunctional = Callable[
    [Float[Array, "N D"]], Float[Array, " N"] | Float[Array, ""]
]


def _resolve_wrt(
    f: Callable, X: Float[Array, "N D"], wrt: Wrt, name: str
) -> Literal["point", "dataset"]:
    """Read ``jax.eval_shape(f, X)``: ``(N,)`` is "point", ``()`` is "dataset"."""
    if X.ndim != 2:
        raise ValueError(f"{name} needs X of shape (N, D), got {X.shape}.")
    if wrt not in ("auto", "point", "dataset"):
        raise ValueError(f"wrt must be 'auto', 'point' or 'dataset', got {wrt!r}.")
    shape = tuple(jax.eval_shape(f, X).shape)
    n = X.shape[0]
    if len(shape) == 2 and shape[0] == n:
        raise ValueError(
            f"{name}: f(X) has shape {shape}, a multi-output predictor; select "
            "one output first, e.g. `lambda X: f(X)[:, c]`."
        )
    modes: dict[tuple[int, ...], Literal["point", "dataset"]] = {
        (): "dataset",
        (n,): "point",
    }
    detected = modes.get(shape)
    if wrt == "auto":
        if detected is None:
            raise ValueError(
                f"{name}: f(X) has shape {shape}; expected ({n},) for a pointwise "
                "predictor or () for a dataset functional."
            )
        return detected
    if detected != wrt:
        expected = f"({n},)" if wrt == "point" else "()"
        raise ValueError(
            f"{name}: wrt={wrt!r} needs f(X) of shape {expected}, but f(X) has "
            f"shape {shape}."
        )
    return wrt


def _derivative_matrix(
    f: PredictorOrFunctional,
    X: Float[Array, "N D"],
    *,
    wrt: Wrt = "auto",
    order: Order = 1,
    batch_size: int | None = None,
    check: bool = True,
    name: str = "sensitivity",
) -> Float[Array, "N D"]:
    r"""The ``(N, D)`` derivative matrix both sensitivities reduce.

    "point": $\partial_j f(x_n)$ (``order=1``, KD1 `gradient`) or
    $\partial_j^2 f(x_n)$ (``order=2``, the diagonal of KD1 `hessian`).
    "dataset": $\partial F / \partial X_{nj}$ (`jax.grad`), ``order=1`` only.
    """
    if order not in (1, 2):
        raise ValueError(f"order must be 1 or 2, got {order!r}.")
    mode = _resolve_wrt(f, X, wrt, name)
    if mode == "dataset":
        if order != 1:
            raise ValueError(
                f"{name}: order=2 with a dataset functional needs the "
                "(N D) x (N D) Hessian of F; only order=1 is supported."
            )
        _maybe_check(f, name, check)
        return jax.grad(f)(X)
    if order == 1:
        return gradient(f, X, batch_size=batch_size, check=check)
    H = hessian(f, X, batch_size=batch_size, check=check)
    return einsum(H, jnp.eye(X.shape[1], dtype=H.dtype), "n d e, d e -> n d")


def _transform(G: Float[Array, "N D"], transform: Transform) -> Float[Array, "N D"]:
    if transform == "square":
        return G * G
    if transform == "abs":
        return jnp.abs(G)
    raise ValueError(f"transform must be 'square' or 'abs', got {transform!r}.")


def _normalize(v: Float[Array, " K"], normalize: Normalize) -> Float[Array, " K"]:
    """``v / sum(v)`` or ``v / max(v)``, 0 (not NaN) when the divisor is 0.

    Double ``where``: the masked branch divides by 1, so the gradient of the
    result stays finite at a zero divisor.
    """
    if normalize is None:
        return v
    if normalize not in ("sum", "max"):
        raise ValueError(f"normalize must be 'sum', 'max' or None, got {normalize!r}.")
    denom = reduce(v, "k ->", normalize)
    positive = denom > 0
    safe = jnp.where(positive, denom, jnp.ones_like(denom))
    return jnp.where(positive, v / safe, jnp.zeros_like(v))


def sensitivity(
    f: PredictorOrFunctional,
    X: Float[Array, "N D"],
    *,
    wrt: Wrt = "auto",
    order: Order = 1,
    transform: Transform = "square",
    normalize: Normalize = None,
    batch_size: int | None = None,
    check: bool = True,
) -> Float[Array, " D"]:
    r"""Per-feature sensitivity map $\hat s_j = \frac1N \sum_n \tau(\partial_j f(x_n))$.

    The population sensitivity of feature $j$ (Johnson et al. 2020, Eq. 11,
    written with the joint density since $\partial_j f$ depends on all of
    $x$) is $s_j = \mathbb E_{x \sim p}[(\partial_j f(x))^2]$; this returns
    its Monte-Carlo estimate over the rows of ``X`` (Eq. 12), with
    $\tau(u) = u^2$ (``"square"``) or $|u|$ (``"abs"``). For a dataset
    functional $F$, $\partial_j f(x_n)$ is replaced by
    $\partial F / \partial X_{nj}$, the change of $F$ when sample $n$ moves
    along feature $j$ (the paper's HSIC sensitivity, §6).

    ``order=2`` uses the curvature $\partial_j^2 f(x_n)$ instead (Eq. 6),
    from the diagonal of the exact per-point `hessian` (pointwise only;
    ``(N, D, D)`` memory, so use ``batch_size`` for large ``D``).

    Args:
        f: A pointwise predictor ``(N, D) -> (N,)`` or a dataset functional
            ``(N, D) -> ()``.
        X: Points, shape ``(N, D)``.
        wrt: ``"auto"`` reads the output shape of ``f(X)``: ``(N,)`` is
            ``"point"``, ``()`` is ``"dataset"``. Forcing a mode checks the
            shape agrees.
        order: ``1`` (gradients) or ``2`` (Hessian diagonals, ``"point"``
            only).
        transform: ``"square"`` (the paper's default) or ``"abs"``.
        normalize: ``None``; ``"sum"`` returns $\hat s / \sum_j \hat s_j$;
            ``"max"`` returns $\hat s / \max_j \hat s_j$. A zero divisor
            (constant ``f``) gives zeros, not NaN.
        batch_size: Pointwise mode: ``None`` vmaps over all points, an
            integer maps over them in chunks (see `gradient`). Ignored for
            a dataset functional.
        check: Refuse an estimator's bound ``predict`` whose kernel is not
            differentiable at the training points (see `gradient`).

    Returns:
        ``(D,)`` per-feature sensitivities.

    Raises:
        ValueError: If ``X`` is not 2-D, ``f(X)`` is neither ``(N,)`` nor
            ``()`` (for a multi-output ``(N, C)`` predictor, select one
            output), ``wrt`` disagrees with the output shape, ``order=2``
            is asked of a dataset functional, an option is unknown, or the
            owner's kernel is rough.

    Examples:
        >>> import jax.numpy as jnp
        >>> import kernellib as kl
        >>> X = jnp.array([[1.0, 2.0], [0.5, -1.0], [0.0, 3.0]])
        >>> f = lambda X: 3.0 * X[:, 0] - X[:, 1]  # pointwise, (N,)
        >>> kl.sensitivity(f, X).tolist()
        [9.0, 1.0]
        >>> kl.sensitivity(f, X, normalize="sum").round(4).tolist()
        [0.9, 0.1]
        >>> F = lambda X: jnp.mean(X[:, 0] ** 2)  # dataset functional, ()
        >>> kl.sensitivity(F, X, wrt="dataset").round(4).tolist()
        [0.1852, 0.0]
    """
    G = _derivative_matrix(
        f, X, wrt=wrt, order=order, batch_size=batch_size, check=check
    )
    return _normalize(reduce(_transform(G, transform), "n d -> d", "mean"), normalize)


def point_sensitivity(
    f: PredictorOrFunctional,
    X: Float[Array, "N D"],
    *,
    wrt: Wrt = "auto",
    order: Order = 1,
    transform: Transform = "square",
    normalize: Normalize = None,
    batch_size: int | None = None,
    check: bool = True,
) -> Float[Array, " N"]:
    r"""Per-sample sensitivity $q_n = \frac1D \sum_j \tau(\partial_j f(x_n))$.

    Johnson et al. (2020), Eq. 13: the **mean** over features, so with
    $\tau(u) = u^2$, $q_n = \|\nabla f(x_n)\|_2^2 / D$. It ranks samples by
    how strongly the predictor varies around them or, for a dataset
    functional $F$, by how much moving each sample changes $F$. Same modes
    and options as `sensitivity`; the two satisfy
    $\frac1D \sum_j \hat s_j = \frac1N \sum_n q_n = \|G\|_F^2 / (N D)$.

    Args:
        f: A pointwise predictor ``(N, D) -> (N,)`` or a dataset functional
            ``(N, D) -> ()``.
        X: Points, shape ``(N, D)``.
        wrt: ``"auto"``, ``"point"`` or ``"dataset"`` (see `sensitivity`).
        order: ``1`` (gradients) or ``2`` (Hessian diagonals, ``"point"``
            only).
        transform: ``"square"`` (the paper's default) or ``"abs"``.
        normalize: ``None``, ``"sum"`` or ``"max"``, over the ``N`` samples.
        batch_size: Pointwise mode chunk size; ignored for a functional.
        check: Refuse an estimator's bound ``predict`` whose kernel is not
            differentiable at the training points.

    Returns:
        ``(N,)`` per-sample sensitivities.

    Raises:
        ValueError: As for `sensitivity`.

    Examples:
        >>> import jax.numpy as jnp
        >>> import kernellib as kl
        >>> X = jnp.array([[1.0, 2.0], [0.5, -1.0], [0.0, 3.0]])
        >>> f = lambda X: X[:, 0] ** 2  # gradient (2 x_0, 0)
        >>> kl.point_sensitivity(f, X).tolist()
        [2.0, 0.5, 0.0]
        >>> kl.point_sensitivity(f, X, normalize="max").tolist()
        [1.0, 0.25, 0.0]
    """
    G = _derivative_matrix(
        f,
        X,
        wrt=wrt,
        order=order,
        batch_size=batch_size,
        check=check,
        name="point_sensitivity",
    )
    return _normalize(reduce(_transform(G, transform), "n d -> n", "mean"), normalize)
