"""Kernel-level feature maps: random Fourier, orthogonal, FastFood, Nyström.

Each is an `AbstractFeatureMap`: configure it, ``fit(kernel, X)``, then call
it for features or ``operator(X)`` for a gaussx `LowRankUpdate`. The random
maps draw frequencies from the kernel's spectral sampler at unit lengthscale
and apply the kernel's lengthscale and variance when called; Nyström picks
landmarks and evaluates the kernel against them.
"""

from __future__ import annotations

import dataclasses
import math

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsl
from geonnax.randfeat import orthogonal_blocks, rff_forward
from jaxtyping import Array, Float, PRNGKeyArray

from kernellib._kernels import AbstractKernel, AbstractStationaryKernel
from kernellib._operators._fastfood import (
    FastFoodParams,
    fastfood_features,
    fastfood_params,
)
from kernellib._spectral._base import AbstractFeatureMap, _require_spectral


__all__ = [
    "FastFoodFeatures",
    "NystromFeatures",
    "OrthogonalRandomFeatures",
    "RandomFourierFeatures",
]


def _check_positive(name: str, value: int) -> None:
    if value < 1:
        raise ValueError(f"{name} must be >= 1, got {value}.")


def _cos_sin_features(
    kernel: AbstractStationaryKernel,
    omega: Float[Array, "F D"],
    X: Float[Array, "N D"],
) -> Float[Array, "N two_F"]:
    """``sqrt(σ²/F) [cos(XWᵀ), sin(XWᵀ)]`` with ``W = omega / lengthscale``."""
    if X.shape[-1] != omega.shape[-1]:
        raise ValueError(
            f"The map was fitted on {omega.shape[-1]}-dimensional inputs, got "
            f"{X.shape[-1]}."
        )
    W = omega / kernel._lengthscale_vector(omega.shape[-1])
    n_features = omega.shape[0]
    Phi = jax.vmap(lambda x: rff_forward(W.T, 1.0, n_features, x))(X)
    return jnp.sqrt(kernel.variance) * Phi


class RandomFourierFeatures(AbstractFeatureMap):
    r"""Random Fourier features (Rahimi & Recht, 2007).

    $\phi(x) = \sqrt{\sigma^2 / F}\,[\cos(W x), \sin(W x)]$ with the ``F``
    rows of $W$ drawn from the kernel's spectral density, so
    $\phi(x)^\top\phi(x') = \frac{\sigma^2}{F}\sum_j \cos(w_j^\top(x - x'))$
    is an unbiased estimate of $k(x, x')$. The feature matrix has
    ``2 * n_features`` columns. The arithmetic is
    ``geonnax.randfeat.rff_forward``.

    Attributes:
        n_features: Number of frequencies ``F``.
        key: PRNG key for the frequency draw.
        kernel: The fitted kernel, ``None`` before `fit`.
        omega: Unit-lengthscale frequencies ``(F, D)``, ``None`` before `fit`.

    Examples:
        >>> import jax
        >>> import jax.numpy as jnp
        >>> import kernellib as kl
        >>> X = jnp.linspace(-1.0, 1.0, 6)[:, None]
        >>> rff = kl.RandomFourierFeatures(256, jax.random.key(0))
        >>> rff = rff.fit(kl.Matern(nu=1.5, lengthscale=0.5), X)
        >>> rff(X).shape
        (6, 512)
    """

    n_features: int = eqx.field(static=True)
    key: PRNGKeyArray
    kernel: AbstractStationaryKernel | None = None
    omega: Float[Array, "F D"] | None = None

    def __check_init__(self) -> None:
        _check_positive("n_features", self.n_features)

    def fit(
        self, kernel: AbstractKernel, X: Float[Array, "N D"]
    ) -> RandomFourierFeatures:
        """Draw ``n_features`` frequencies for ``kernel`` in ``X``'s dimension.

        Raises:
            NotImplementedError: If the kernel has no spectral sampler.
            ValueError: If an ARD lengthscale does not match ``X``.
        """
        kernel = _require_spectral(kernel, type(self).__name__)
        d = X.shape[-1]
        kernel._lengthscale_vector(d)
        omega = kernel.sample_unit_frequencies(self.key, (self.n_features, d), X.dtype)
        return dataclasses.replace(self, kernel=kernel, omega=omega)

    def features(self, X: Float[Array, "N D"]) -> Float[Array, "N two_F"]:
        assert self.kernel is not None and self.omega is not None
        return _cos_sin_features(self.kernel, self.omega, X)


class OrthogonalRandomFeatures(AbstractFeatureMap):
    r"""Orthogonal random features (Yu et al., 2016).

    Like `RandomFourierFeatures`, but the frequency directions within each
    block of ``D`` are exactly orthogonal (Haar blocks from
    ``geonnax.randfeat.orthogonal_blocks``). Their lengths are drawn from the
    kernel's own radial distribution, the norms of its unit-lengthscale
    spectral samples, so the construction holds for any isotropic unit
    density, not just the Gaussian. The estimate stays unbiased and its
    variance is lower than plain RFF at the same feature count.

    Attributes:
        n_features: Number of frequencies ``F``; the last block is truncated
            when ``F`` is not a multiple of ``D``.
        key: PRNG key for the frequency draw.
        kernel: The fitted kernel, ``None`` before `fit`.
        omega: Unit-lengthscale frequencies ``(F, D)``, ``None`` before `fit`.

    Examples:
        >>> import jax
        >>> import jax.numpy as jnp
        >>> import kernellib as kl
        >>> X = jnp.ones((3, 4))
        >>> orf = kl.OrthogonalRandomFeatures(8, jax.random.key(0)).fit(kl.RBF(), X)
        >>> W = orf.omega[:4]
        >>> G = W @ W.T  # one block: orthogonal rows
        >>> bool(jnp.allclose(G - jnp.diag(jnp.diag(G)), 0.0, atol=1e-6))
        True
    """

    n_features: int = eqx.field(static=True)
    key: PRNGKeyArray
    kernel: AbstractStationaryKernel | None = None
    omega: Float[Array, "F D"] | None = None

    def __check_init__(self) -> None:
        _check_positive("n_features", self.n_features)

    def fit(
        self, kernel: AbstractKernel, X: Float[Array, "N D"]
    ) -> OrthogonalRandomFeatures:
        """Draw orthogonal frequency blocks for ``kernel``.

        Raises:
            NotImplementedError: If the kernel has no spectral sampler.
            ValueError: If an ARD lengthscale does not match ``X``.
        """
        kernel = _require_spectral(kernel, type(self).__name__)
        d = X.shape[-1]
        kernel._lengthscale_vector(d)
        n_blocks = math.ceil(self.n_features / d)
        key_dir, key_len = jax.random.split(self.key)
        Q = orthogonal_blocks(d, n_blocks, key=key_dir).astype(X.dtype)
        directions = Q / jnp.linalg.norm(Q, axis=0, keepdims=True)  # (D, B*D)
        lengths = jnp.linalg.norm(
            kernel.sample_unit_frequencies(key_len, (n_blocks * d, d), X.dtype),
            axis=-1,
        )
        omega = (directions * lengths).T[: self.n_features]
        return dataclasses.replace(self, kernel=kernel, omega=omega)

    def features(self, X: Float[Array, "N D"]) -> Float[Array, "N two_F"]:
        assert self.kernel is not None and self.omega is not None
        return _cos_sin_features(self.kernel, self.omega, X)


class FastFoodFeatures(AbstractFeatureMap):
    r"""FastFood random features (Le, Sarlós & Smola, 2013) for any kernel
    with a spectral sampler.

    Wraps the operator-level `fastfood_params` / `fastfood_features`: the
    structured product $S H G \Pi H B$ gives ``O(F log D)`` feature
    evaluation and ``O(F)`` storage. `fastfood_params` draws the row lengths
    $S$ for the RBF kernel ($\chi$ with ``d_padded`` degrees of freedom);
    here they are the norms of the kernel's own unit-lengthscale spectral
    samples in ``d_padded`` dimensions, which is the same for RBF and the
    right radial law for `Matern` and `RationalQuadratic`. Their spectral
    laws are elliptical, so the marginal on the ``D`` unpadded coordinates is
    the ``D``-dimensional law. The feature matrix has ``2 * n_features``
    columns.

    Attributes:
        n_features: Number of frequencies ``F``.
        key: PRNG key for the structured draw.
        kernel: The fitted kernel, ``None`` before `fit`.
        params: The drawn `FastFoodParams`, ``None`` before `fit`.

    Examples:
        >>> import jax
        >>> import jax.numpy as jnp
        >>> import kernellib as kl
        >>> X = jnp.zeros((4, 5))
        >>> ff = kl.FastFoodFeatures(64, jax.random.key(0)).fit(kl.RBF(), X)
        >>> ff(X).shape, ff.params.d_padded
        ((4, 128), 8)
    """

    n_features: int = eqx.field(static=True)
    key: PRNGKeyArray
    kernel: AbstractStationaryKernel | None = None
    params: FastFoodParams | None = None

    def __check_init__(self) -> None:
        _check_positive("n_features", self.n_features)

    def fit(self, kernel: AbstractKernel, X: Float[Array, "N D"]) -> FastFoodFeatures:
        """Draw the FastFood structure for ``kernel`` in ``X``'s dimension.

        Raises:
            NotImplementedError: If the kernel has no spectral sampler.
            ValueError: If an ARD lengthscale does not match ``X``.
        """
        kernel = _require_spectral(kernel, type(self).__name__)
        d = X.shape[-1]
        kernel._lengthscale_vector(d)
        key_ff, key_s = jax.random.split(self.key)
        params = fastfood_params(d, self.n_features, kernel.lengthscale, key_ff)
        shape = (params.n_stacks, params.d_padded, params.d_padded)
        S = jnp.linalg.norm(
            kernel.sample_unit_frequencies(key_s, shape, params.G.dtype), axis=-1
        )
        return dataclasses.replace(
            self, kernel=kernel, params=dataclasses.replace(params, S=S)
        )

    def features(self, X: Float[Array, "N D"]) -> Float[Array, "N two_F"]:
        assert self.kernel is not None and self.params is not None
        params = dataclasses.replace(self.params, lengthscale=self.kernel.lengthscale)
        return jnp.sqrt(self.kernel.variance) * fastfood_features(X, params)


class NystromFeatures(AbstractFeatureMap):
    r"""Nyström features (Williams & Seeger, 2001) for any kernel.

    With landmarks $Z$ and $K_{ZZ} = L L^\top$,
    $\phi(x) = L^{-1} k(Z, x)$, so $\Phi\Phi^\top = K_{XZ} K_{ZZ}^{-1} K_{ZX}$,
    exact on the landmarks. Landmarks are drawn uniformly without
    replacement from the fitting inputs. $K_{ZZ}$ is formed from the kernel
    at call time, so hyperparameter gradients flow through it.

    Attributes:
        n_components: Number of landmarks ``M``.
        key: PRNG key for landmark selection.
        selection: Landmark rule; only ``"uniform"`` for now.
        jitter: Relative diagonal jitter on $K_{ZZ}$, scaled by its mean
            diagonal.
        kernel: The fitted kernel, ``None`` before `fit`.
        landmarks: The chosen ``(M, D)`` landmarks, ``None`` before `fit`.

    Examples:
        >>> import jax
        >>> import jax.numpy as jnp
        >>> import kernellib as kl
        >>> X = jax.random.normal(jax.random.key(1), (20, 2))
        >>> nys = kl.NystromFeatures(5, jax.random.key(0)).fit(kl.RBF(), X)
        >>> nys(X).shape, nys.landmarks.shape
        ((20, 5), (5, 2))
    """

    n_components: int = eqx.field(static=True)
    key: PRNGKeyArray
    selection: str = eqx.field(default="uniform", static=True)
    jitter: float = eqx.field(default=1e-6, static=True)
    kernel: AbstractKernel | None = None
    landmarks: Float[Array, "M D"] | None = None

    def __check_init__(self) -> None:
        _check_positive("n_components", self.n_components)
        if self.selection != "uniform":
            raise ValueError(f"selection must be 'uniform', got {self.selection!r}.")

    def fit(self, kernel: AbstractKernel, X: Float[Array, "N D"]) -> NystromFeatures:
        """Choose ``n_components`` landmarks from ``X``.

        Raises:
            ValueError: If ``X`` has fewer rows than ``n_components``.
        """
        n = X.shape[0]
        if self.n_components > n:
            raise ValueError(
                f"n_components={self.n_components} landmarks need at least as "
                f"many inputs; got {n}."
            )
        idx = jax.random.choice(self.key, n, (self.n_components,), replace=False)
        return dataclasses.replace(self, kernel=kernel, landmarks=X[idx])

    def features(self, X: Float[Array, "N D"]) -> Float[Array, "N M"]:
        assert self.kernel is not None and self.landmarks is not None
        Z = self.landmarks
        K_zz = self.kernel(Z, Z)
        eps = self.jitter * jnp.mean(jnp.diag(K_zz))
        L = jnp.linalg.cholesky(K_zz + eps * jnp.eye(Z.shape[0], dtype=K_zz.dtype))
        return jsl.solve_triangular(L, self.kernel(Z, X), lower=True).T
