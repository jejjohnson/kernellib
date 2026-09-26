"""Shared plumbing for the scikit-learn adapters.

Kernel hyperparameters become nested scikit-learn parameters: ``kernel`` is
an ordinary constructor parameter, and ``kernel__lengthscale``,
``kernel__kernels__0__variance`` and so on reach inside it, so
``GridSearchCV`` can search over them. Setting one rebuilds the (immutable)
kernel with that field replaced.
"""

from __future__ import annotations

import copy
import dataclasses
from typing import Any

import jax
import numpy as np
from jaxtyping import Array, Float, PRNGKeyArray
from sklearn.utils import check_random_state

from kernellib._heuristics import estimate_lengthscale
from kernellib._kernels import RBF, AbstractKernel


# Median-heuristic subsample: O(n^2) distances are capped at this many points.
_HEURISTIC_SUBSAMPLE = 1000


class _KernelParamsMixin:
    """Expose the fields of kernel-valued parameters as nested parameters.

    Subclasses list their kernel-valued constructor parameters in
    ``_kernel_params``. Must come before ``BaseEstimator`` in the bases.
    """

    _kernel_params: tuple[str, ...] = ("kernel",)

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        params = super().get_params(deep=deep)  # ty: ignore[unresolved-attribute]
        if deep:
            for name in self._kernel_params:
                kernel = params.get(name)
                if isinstance(kernel, AbstractKernel):
                    params.update(_flatten(kernel, name))
        return params

    def set_params(self, **params: Any) -> Any:
        nested: dict[str, dict[str, Any]] = {}
        for key in list(params):
            head, sep, rest = key.partition("__")
            if sep and head in self._kernel_params:
                nested.setdefault(head, {})[rest] = params.pop(key)
        # Top-level first, so ``kernel=RBF(), kernel__lengthscale=...`` works.
        super().set_params(**params)  # ty: ignore[unresolved-attribute]
        for head, updates in nested.items():
            kernel = getattr(self, head)
            if not isinstance(kernel, AbstractKernel):
                raise ValueError(
                    f"Cannot set {head}__* parameters while {head} is {kernel!r}; "
                    f"set {head} to a kernel first."
                )
            for path, value in updates.items():
                kernel = _set_path(kernel, path.split("__"), value, prefix=head)
            setattr(self, head, kernel)
        return self


def _flatten(kernel: AbstractKernel, prefix: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for field in dataclasses.fields(kernel):
        value = getattr(kernel, field.name)
        name = f"{prefix}__{field.name}"
        out[name] = value
        if isinstance(value, AbstractKernel):
            out.update(_flatten(value, name))
        elif isinstance(value, tuple) and all(
            isinstance(v, AbstractKernel) for v in value
        ):
            for i, child in enumerate(value):
                out[f"{name}__{i}"] = child
                out.update(_flatten(child, f"{name}__{i}"))
    return out


def _set_path(kernel: AbstractKernel, path: list[str], value: Any, *, prefix: str):
    names = {f.name for f in dataclasses.fields(kernel)}
    head, rest = path[0], path[1:]
    if head not in names:
        raise ValueError(
            f"Invalid parameter {prefix}__{head} for {type(kernel).__name__}; "
            f"valid fields are {sorted(names)}."
        )
    if not rest:
        return _replace(kernel, head, value)
    child = getattr(kernel, head)
    if isinstance(child, tuple):
        index = int(rest[0])
        new_child = (
            value
            if len(rest) == 1
            else _set_path(
                child[index], rest[1:], value, prefix=f"{prefix}__{head}__{index}"
            )
        )
        return _replace(kernel, head, (*child[:index], new_child, *child[index + 1 :]))
    return _replace(
        kernel, head, _set_path(child, rest, value, prefix=f"{prefix}__{head}")
    )


def _replace(module: Any, name: str, value: Any) -> Any:
    """``module`` with field ``name`` replaced, running its converter."""
    try:
        # Runs converters and __check_init__ (e.g. Matern's nu check).
        return dataclasses.replace(module, **{name: value})
    except TypeError:
        # Custom __init__ (Sum(*kernels), Product(*kernels)): set on a copy.
        field = next(f for f in dataclasses.fields(module) if f.name == name)
        converter = field.metadata.get("converter")
        new = copy.copy(module)
        object.__setattr__(new, name, converter(value) if converter else value)
        return new


def _key(random_state: Any) -> PRNGKeyArray:
    """A JAX key from a scikit-learn ``random_state``."""
    seed = check_random_state(random_state).randint(np.iinfo(np.int32).max)
    return jax.random.key(int(seed))


def _default_kernel(
    kernel: AbstractKernel | None, X: Float[Array, "N D"], key: PRNGKeyArray
) -> AbstractKernel:
    """``kernel``, or an RBF at the median-heuristic lengthscale of ``X``."""
    if kernel is not None:
        return kernel
    if X.shape[0] < 2:
        return RBF()
    ell = estimate_lengthscale(
        X, subsample=min(X.shape[0], _HEURISTIC_SUBSAMPLE), key=key
    )
    # Duplicated or constant inputs give a zero median distance.
    ell = float(ell)
    return RBF(lengthscale=ell if np.isfinite(ell) and ell > 0 else 1.0)
