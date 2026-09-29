"""Approximations as kernels: the Nyström kernel and approximation residuals."""

from __future__ import annotations

from jaxtyping import Array, Float

from kernellib._kernels._base import AbstractKernel
from kernellib._kernels._feature import FeatureKernel


__all__ = ["Residual", "nystrom_kernel"]


def nystrom_kernel(
    kernel: AbstractKernel,
    landmarks: Float[Array, "M D"],
    *,
    jitter: float = 1e-6,
) -> FeatureKernel:
    r"""The Nyström kernel $k_Z(x, x') = k(x, Z) K_{ZZ}^{-1} k(Z, x')$.

    ``FeatureKernel(NystromFeatures.from_landmarks(kernel, landmarks))``:
    rank ``M``, kept low-rank by `to_operator`, with $K_{ZZ}$ formed from
    ``kernel`` at call time so its hyperparameters stay differentiable. For
    landmarks chosen from data (uniform or leverage scores), fit a
    `NystromFeatures` and wrap it in `FeatureKernel` instead.

    Args:
        kernel: The kernel to approximate.
        landmarks: Inducing points $Z$, shape ``(M, D)``.
        jitter: Relative diagonal jitter on $K_{ZZ}$.

    Returns:
        The Nyström kernel.

    Examples:
        >>> import jax.numpy as jnp
        >>> import kernellib as kl
        >>> Z = jnp.linspace(-1.0, 1.0, 5)[:, None]
        >>> kz = kl.nystrom_kernel(kl.RBF(lengthscale=0.5), Z)
        >>> bool(jnp.allclose(kz(Z, Z), kl.RBF(lengthscale=0.5)(Z, Z), atol=1e-4))
        True
    """
    from kernellib._spectral._feature_maps import NystromFeatures

    return FeatureKernel(
        NystromFeatures.from_landmarks(kernel, landmarks, jitter=jitter)
    )


class Residual(AbstractKernel):
    r"""What an approximation misses, $k(x, x') - \tilde k(x, x')$.

    With $\tilde k$ a Nyström kernel (`nystrom_kernel`, or `FeatureKernel`
    of a fitted `NystromFeatures`) this is the covariance of a GP given its
    values at the landmarks: positive semidefinite, zero (up to jitter) on
    the landmarks, and its diagonal is the FITC correction and the base of
    sparse-GP predictive variances. `diag` costs ``O(N M)`` and never forms
    the ``N x N`` Gram. The full Gram is dense.

    The residual of an *unbiased* approximation such as random Fourier
    features is not PSD in general (it is not a lower bound on $k$); it is
    still useful for error analysis.

    Attributes:
        kernel: The exact kernel.
        approx: The approximating kernel.

    Examples:
        >>> import jax.numpy as jnp
        >>> import kernellib as kl
        >>> k = kl.RBF(lengthscale=0.5)
        >>> Z = jnp.linspace(-1.0, 1.0, 5)[:, None]
        >>> r = kl.Residual(k, kl.nystrom_kernel(k, Z))
        >>> bool(jnp.all(r.diag(Z) < 1e-4))
        True
    """

    kernel: AbstractKernel
    approx: AbstractKernel

    def __call__(
        self, X1: Float[Array, "N1 D"], X2: Float[Array, "N2 D"]
    ) -> Float[Array, "N1 N2"]:
        return self.kernel(X1, X2) - self.approx(X1, X2)

    def diag(self, X: Float[Array, "N D"]) -> Float[Array, " N"]:
        return self.kernel.diag(X) - self.approx.diag(X)

    def pairwise(
        self, x: Float[Array, " D"], y: Float[Array, " D"]
    ) -> Float[Array, ""]:
        return self.kernel.pairwise(x, y) - self.approx.pairwise(x, y)

    @property
    def is_pointwise(self) -> bool:
        return self.kernel.is_pointwise and self.approx.is_pointwise
