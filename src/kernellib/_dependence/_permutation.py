"""Permutation tests for any dependence or two-sample statistic."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal, NamedTuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray


__all__ = ["PermutationTestResult", "permutation_test"]


class PermutationTestResult(NamedTuple):
    """Outcome of `permutation_test`.

    Attributes:
        statistic: The statistic on the observed data.
        p_value: ``(1 + #{null >= statistic}) / (1 + n_permutations)``.
        null_distribution: The statistic on each permutation.
    """

    statistic: Float[Array, ""]
    p_value: Float[Array, ""]
    null_distribution: Float[Array, " n_permutations"]


def permutation_test(
    statistic: Callable[[Array, Array], Float[Array, ""]],
    X: Float[Array, "Nx Dx"],
    Y: Float[Array, "Ny Dy"],
    *,
    key: PRNGKeyArray,
    n_permutations: int = 500,
    kind: Literal["independence", "two_sample"] = "independence",
) -> PermutationTestResult:
    r"""Permutation p-value for a statistic that is large under the alternative.

    - ``kind="independence"`` (HSIC, CKA): ``X`` and ``Y`` are paired; the
      null is simulated by shuffling ``Y``'s rows, which breaks the pairing.
    - ``kind="two_sample"`` (MMD): the null is simulated by shuffling the
      pooled sample and splitting it back into sizes ``Nx`` and ``Ny``.

    The p-value counts the observed statistic as one of the permutations, so
    it is never zero and the test is exact at any level.

    Args:
        statistic: ``statistic(X, Y) -> scalar``, e.g.
            ``lambda X, Y: kl.hsic(kx, ky, X, Y)``. Must be traceable: the
            permutations run under `jax.lax.map`.
        X: First sample.
        Y: Second sample.
        key: PRNG key for the permutations.
        n_permutations: Number of permutations.
        kind: ``"independence"`` or ``"two_sample"``.

    Returns:
        The statistic, the p-value and the null distribution.

    Raises:
        ValueError: On an unknown ``kind``, ``n_permutations < 1``, or unpaired
            samples for an independence test.

    Examples:
        >>> import jax
        >>> import kernellib as kl
        >>> X = jax.random.normal(jax.random.key(0), (60, 1))
        >>> k = kl.RBF()
        >>> result = kl.permutation_test(
        ...     lambda X, Y: kl.hsic(k, k, X, Y),
        ...     X,
        ...     X**2,
        ...     key=jax.random.key(1),
        ...     n_permutations=99,
        ... )
        >>> float(result.p_value)  # nothing beats the observed pairing
        0.01
    """
    if n_permutations < 1:
        raise ValueError(f"n_permutations must be >= 1, got {n_permutations}.")
    keys = jax.random.split(key, n_permutations)
    if kind == "independence":
        if X.shape[0] != Y.shape[0]:
            raise ValueError(
                "An independence test needs paired samples of the same size, got "
                f"{X.shape[0]} and {Y.shape[0]}."
            )

        def one(k: PRNGKeyArray) -> Float[Array, ""]:
            return statistic(X, Y[jax.random.permutation(k, Y.shape[0])])

    elif kind == "two_sample":
        m = X.shape[0]
        Z = jnp.concatenate([X, Y])

        def one(k: PRNGKeyArray) -> Float[Array, ""]:
            Zp = Z[jax.random.permutation(k, Z.shape[0])]
            return statistic(Zp[:m], Zp[m:])

    else:
        raise ValueError(f"kind must be 'independence' or 'two_sample', got {kind!r}.")
    observed = statistic(X, Y)
    null = jax.lax.map(one, keys)
    p_value = (1.0 + jnp.sum(null >= observed)) / (1.0 + n_permutations)
    return PermutationTestResult(observed, p_value, null)
