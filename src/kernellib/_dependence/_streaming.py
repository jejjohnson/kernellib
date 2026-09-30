r"""Mini-batch CKA: accumulate unbiased HSIC over batches.

CKA over a whole dataset needs ``N x N`` Gram matrices. Nguyen, Raghu &
Kornblith (2021) instead average the unbiased HSIC of each mini-batch,

$$
\mathrm{CKA}_{\mathrm{mb}} = \frac{\sum_b \mathrm{HSIC}_u(K_b, L_b)}
    {\sqrt{\sum_b \mathrm{HSIC}_u(K_b, K_b)\,\sum_b \mathrm{HSIC}_u(L_b, L_b)}},
$$

which is consistent whatever the batch size, because each term is unbiased.
With biased per-batch terms the ratio drifts by ``O(1/B)``. Memory is
``O(B^2)`` per batch.

**The batches must be random draws** (shuffled, independent, preferably of
equal size). Each term only sees dependence *within* its batch, so ordered
or stratified batches bias the estimate. In the extreme, if both
representations are constant within every batch but their means vary
between batches, every term is zero and the result is ``0``, while the
full-data CKA is ``1``.
"""

from __future__ import annotations

import dataclasses

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from kernellib._dependence._hsic import _cka_parts
from kernellib._kernels import AbstractKernel
from kernellib._spectral import AbstractFeatureMap
from kernellib.functional._statistics import _cka_ratio


__all__ = ["CKAAccumulator"]


class CKAAccumulator(eqx.Module):
    r"""Running mini-batch CKA (Nguyen, Raghu & Kornblith, 2021).

    An immutable pytree: `update` returns a new accumulator with one more
    batch's unbiased ``HSIC(x, y)``, ``HSIC(x, x)`` and ``HSIC(y, y)`` added
    to the running sums, and `result` returns their CKA. It works inside
    ``jax.jit`` and as a ``jax.lax.scan`` carry. Degenerate sums give ``0``,
    as in `kernellib.cka`.

    Feed it **shuffled** batches: ordered or stratified batches bias the
    estimate (see the module docstring).

    Attributes:
        kernel_x: Kernel on the first representation.
        kernel_y: Kernel on the second representation.
        approx: Optional unfitted feature map, as in `kernellib.cka`.
        sums: Running sums of the unbiased ``HSIC(x, y)``, ``HSIC(x, x)`` and
            ``HSIC(y, y)``.
        n_batches: Number of batches added.

    Examples:
        >>> import jax
        >>> import kernellib as kl
        >>> X = jax.random.normal(jax.random.key(0), (400, 3))
        >>> Y = X @ jax.random.normal(jax.random.key(1), (3, 5))  # linear map of X
        >>> acc = kl.CKAAccumulator(kl.Linear(), kl.Linear())
        >>> for start in range(0, 400, 50):
        ...     acc = acc.update(X[start : start + 50], Y[start : start + 50])
        >>> int(acc.n_batches), bool(acc.result() > 0.5)
        (8, True)
    """

    kernel_x: AbstractKernel
    kernel_y: AbstractKernel
    approx: AbstractFeatureMap | None = None
    sums: Float[Array, 3] = eqx.field(default_factory=lambda: jnp.zeros(3))
    n_batches: Int[Array, ""] = eqx.field(
        default_factory=lambda: jnp.zeros((), dtype=jnp.int32)
    )

    def update(
        self, X: Float[Array, "B Dx"], Y: Float[Array, "B Dy"]
    ) -> CKAAccumulator:
        """Add one batch of paired samples (``B >= 4``).

        Args:
            X: Batch of the first representation, shape ``(B, Dx)``.
            Y: Paired batch of the second representation, shape ``(B, Dy)``.

        Returns:
            A new accumulator including this batch.

        Raises:
            ValueError: On mismatched batch sizes, or ``B < 4``.
        """
        xy, xx, yy = _cka_parts(
            self.kernel_x, self.kernel_y, X, Y, "unbiased", self.approx
        )
        batch = jnp.stack([xy, xx, yy]).astype(self.sums.dtype)
        return dataclasses.replace(
            self, sums=self.sums + batch, n_batches=self.n_batches + 1
        )

    def result(self) -> Float[Array, ""]:
        """Mini-batch CKA of the batches added so far (``0`` if degenerate)."""
        xy, xx, yy = self.sums[0], self.sums[1], self.sums[2]
        return _cka_ratio(xy, xx, yy, "unbiased")
