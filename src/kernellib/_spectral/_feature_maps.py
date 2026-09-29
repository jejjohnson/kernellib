"""Kernel-level feature maps: random Fourier, orthogonal, FastFood, Nyström.

Each is an `AbstractFeatureMap`: configure it, ``fit(kernel, X)``, then call
it for features or ``operator(X)`` for a gaussx `LowRankUpdate`. The random
maps draw frequencies from the kernel's spectral sampler at unit lengthscale
and apply the kernel's lengthscale and variance when called; Nyström picks
landmarks and evaluates the kernel against them.

A `Scaled` or `Sum` of stationary kernels, ``k = sum_j c_j k_j``, gets an
equal share ``F_j`` of the frequencies per part and features
``sqrt(c_j σ_j² / F_j) [cos, sin]`` from that part's own spectrum. That is
unbiased for ``k`` like mixture sampling, with lower variance, and keeps
every part's hyperparameters (scales included) differentiable after `fit`.
"""

from __future__ import annotations

import dataclasses
import math

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsl
from geonnax.randfeat import orthogonal_blocks, rff_forward
from jaxtyping import Array, Float, Int, PRNGKeyArray

from kernellib._einx import einsum, reduce
from kernellib._kernels import AbstractKernel
from kernellib._kernels._compose import SpectralComponents, _spectral_components
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


def _split_features(n_features: int, n_parts: int) -> tuple[int, ...]:
    """Split ``n_features`` evenly over ``n_parts`` (remainder to the first)."""
    if n_features < n_parts:
        raise ValueError(
            f"n_features={n_features} is fewer than the kernel's {n_parts} "
            "stationary parts; each part needs at least one frequency."
        )
    base, extra = divmod(n_features, n_parts)
    return tuple(base + (j < extra) for j in range(n_parts))


def _part_keys(key: PRNGKeyArray, n_parts: int) -> list[PRNGKeyArray]:
    # A lone stationary kernel uses the key as is, so its draw is unchanged.
    return [key] if n_parts == 1 else list(jax.random.split(key, n_parts))


def _fitted_components(kernel: AbstractKernel | None) -> SpectralComponents:
    components = _spectral_components(kernel) if kernel is not None else None
    assert components is not None
    return components


def _cos_sin_features(
    kernel: AbstractKernel | None,
    omega: Float[Array, "F D"],
    sizes: tuple[int, ...],
    X: Float[Array, "N D"],
) -> Float[Array, "N two_F"]:
    """``sqrt(c_j σ_j²/F_j) [cos(XW_jᵀ), sin(XW_jᵀ)]`` per part, concatenated.

    ``W_j`` is part ``j``'s rows of ``omega`` over its lengthscale.
    """
    if X.shape[-1] != omega.shape[-1]:
        raise ValueError(
            f"The map was fitted on {omega.shape[-1]}-dimensional inputs, got "
            f"{X.shape[-1]}."
        )
    d = omega.shape[-1]
    blocks = []
    start = 0
    for (c, k), size in zip(_fitted_components(kernel), sizes, strict=True):
        W = omega[start : start + size] / k._lengthscale_vector(d)
        start += size
        Phi = jax.vmap(lambda x, W=W, size=size: rff_forward(W.T, 1.0, size, x))(X)
        blocks.append(jnp.sqrt(c * k.variance) * Phi)
    return blocks[0] if len(blocks) == 1 else jnp.concatenate(blocks, axis=-1)


class RandomFourierFeatures(AbstractFeatureMap):
    r"""Random Fourier features (Rahimi & Recht, 2007).

    $\phi(x) = \sqrt{\sigma^2 / F}\,[\cos(W x), \sin(W x)]$ with the ``F``
    rows of $W$ drawn from the kernel's spectral density, so
    $\phi(x)^\top\phi(x') = \frac{\sigma^2}{F}\sum_j \cos(w_j^\top(x - x'))$
    is an unbiased estimate of $k(x, x')$. The feature matrix has
    ``2 * n_features`` columns. The arithmetic is
    ``geonnax.randfeat.rff_forward``.

    Accepts a `Scaled` or `Sum` of stationary kernels (see the module
    docstring for how the frequencies are shared between the parts).

    Attributes:
        n_features: Number of frequencies ``F``.
        key: PRNG key for the frequency draw.
        kernel: The fitted kernel, ``None`` before `fit`.
        omega: Unit-lengthscale frequencies ``(F, D)``, the parts' blocks
            stacked in order, ``None`` before `fit`.
        sizes: Frequencies per stationary part, ``None`` before `fit`.

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
    kernel: AbstractKernel | None = None
    omega: Float[Array, "F D"] | None = None
    sizes: tuple[int, ...] | None = eqx.field(default=None, static=True)

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
        components = _require_spectral(kernel, type(self).__name__)
        d = X.shape[-1]
        sizes = _split_features(self.n_features, len(components))
        keys = _part_keys(self.key, len(components))
        blocks = []
        for (_, k), size, key in zip(components, sizes, keys, strict=True):
            k._lengthscale_vector(d)
            blocks.append(k.sample_unit_frequencies(key, (size, d), X.dtype))
        omega = jnp.concatenate(blocks, axis=0)
        return dataclasses.replace(self, kernel=kernel, omega=omega, sizes=sizes)

    def features(self, X: Float[Array, "N D"]) -> Float[Array, "N two_F"]:
        assert self.omega is not None and self.sizes is not None
        return _cos_sin_features(self.kernel, self.omega, self.sizes, X)


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
        omega: Unit-lengthscale frequencies ``(F, D)``, the parts' blocks
            stacked in order, ``None`` before `fit`.
        sizes: Frequencies per stationary part, ``None`` before `fit`.

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
    kernel: AbstractKernel | None = None
    omega: Float[Array, "F D"] | None = None
    sizes: tuple[int, ...] | None = eqx.field(default=None, static=True)

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
        components = _require_spectral(kernel, type(self).__name__)
        d = X.shape[-1]
        sizes = _split_features(self.n_features, len(components))
        keys = _part_keys(self.key, len(components))
        blocks = []
        for (_, k), size, key in zip(components, sizes, keys, strict=True):
            k._lengthscale_vector(d)
            n_blocks = math.ceil(size / d)
            key_dir, key_len = jax.random.split(key)
            Q = orthogonal_blocks(d, n_blocks, key=key_dir).astype(X.dtype)
            directions = Q / jnp.linalg.norm(Q, axis=0, keepdims=True)  # (D, B*D)
            lengths = jnp.linalg.norm(
                k.sample_unit_frequencies(key_len, (n_blocks * d, d), X.dtype),
                axis=-1,
            )
            blocks.append((directions * lengths).T[:size])
        omega = jnp.concatenate(blocks, axis=0)
        return dataclasses.replace(self, kernel=kernel, omega=omega, sizes=sizes)

    def features(self, X: Float[Array, "N D"]) -> Float[Array, "N two_F"]:
        assert self.omega is not None and self.sizes is not None
        return _cos_sin_features(self.kernel, self.omega, self.sizes, X)


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
        params: The drawn `FastFoodParams` (a tuple of them, one per
            stationary part, for a `Scaled` or `Sum`), ``None`` before `fit`.

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
    kernel: AbstractKernel | None = None
    params: FastFoodParams | tuple[FastFoodParams, ...] | None = None

    def __check_init__(self) -> None:
        _check_positive("n_features", self.n_features)

    def fit(self, kernel: AbstractKernel, X: Float[Array, "N D"]) -> FastFoodFeatures:
        """Draw the FastFood structure for ``kernel`` in ``X``'s dimension.

        Raises:
            NotImplementedError: If the kernel has no spectral sampler.
            ValueError: If an ARD lengthscale does not match ``X``.
        """
        components = _require_spectral(kernel, type(self).__name__)
        d = X.shape[-1]
        sizes = _split_features(self.n_features, len(components))
        keys = _part_keys(self.key, len(components))
        parts = []
        for (_, k), size, key in zip(components, sizes, keys, strict=True):
            k._lengthscale_vector(d)
            key_ff, key_s = jax.random.split(key)
            params = fastfood_params(d, size, k.lengthscale, key_ff)
            shape = (params.n_stacks, params.d_padded, params.d_padded)
            S = jnp.linalg.norm(
                k.sample_unit_frequencies(key_s, shape, params.G.dtype), axis=-1
            )
            parts.append(dataclasses.replace(params, S=S))
        return dataclasses.replace(
            self, kernel=kernel, params=parts[0] if len(parts) == 1 else tuple(parts)
        )

    def features(self, X: Float[Array, "N D"]) -> Float[Array, "N two_F"]:
        assert self.params is not None
        parts = self.params if isinstance(self.params, tuple) else (self.params,)
        blocks = [
            jnp.sqrt(c * k.variance)
            * fastfood_features(
                X, dataclasses.replace(params, lengthscale=k.lengthscale)
            )
            for (c, k), params in zip(
                _fitted_components(self.kernel), parts, strict=True
            )
        ]
        return blocks[0] if len(blocks) == 1 else jnp.concatenate(blocks, axis=-1)


class NystromFeatures(AbstractFeatureMap):
    r"""Nyström features (Williams & Seeger, 2001) for any kernel.

    With landmarks $Z$ and $K_{ZZ} = L L^\top$,
    $\phi(x) = L^{-1} k(Z, x)$, so $\Phi\Phi^\top = K_{XZ} K_{ZZ}^{-1} K_{ZX}$,
    exact on the landmarks. $K_{ZZ}$ is formed from the kernel at call time,
    so hyperparameter gradients flow through it.

    Landmarks are drawn without replacement from the fitting inputs:

    - ``selection="uniform"`` (default): uniformly.
    - ``selection="leverage"``: with probability proportional to approximate
      ridge leverage scores $\ell_i(\lambda) = [K (K + \lambda n I)^{-1}]_{ii}$
      (Alaoui & Mahoney, 2015; Rudi et al., 2018), which put landmarks where
      the kernel matrix has the most effective dimension, so fewer are
      needed on unevenly spread data. The scores come from a uniform pilot
      Nyström map $\Phi$ on ``m0 = min(2 M, N)`` points,

      $$
      \tilde\ell_i = \phi_i^\top (\Phi^\top\Phi + \lambda n I)^{-1} \phi_i
          + \frac{k(x_i, x_i) - \lVert\phi_i\rVert^2}{\lambda n},
      $$

      the second term adding back the pilot's residual so badly explained
      points are not under-sampled. The pilot costs
      ``O(N m0^2 + m0^3)`` time and ``O(N m0)`` memory, once, at `fit`;
      with ``m0 = N`` the scores are exact.

      Leverage sampling pays off when ``n_components`` is at least the
      effective dimension $d_{\mathrm{eff}}(\lambda) = \sum_i \ell_i$.
      Below it, points that are each alone in their neighbourhood (leverage
      near 1) can take every landmark and leave dense regions with none,
      which is worse than uniform. ``uniform_mixing`` guards against that:
      landmarks are drawn from
      $(1 - u)\, \tilde\ell / \sum \tilde\ell + u / N$. The default
      ``u = 0.5`` was never worse than uniform in our tests; ``u = 0`` is
      pure leverage sampling, much better once ``n_components`` exceeds
      $d_{\mathrm{eff}}$ (raise ``leverage_regularization`` to lower it).

    Attributes:
        n_components: Number of landmarks ``M``.
        key: PRNG key for landmark selection.
        selection: Landmark rule, ``"uniform"`` or ``"leverage"``.
        leverage_regularization: The $\lambda$ of the ridge leverage scores
            (``selection="leverage"`` only); the effective dimension, and so
            how concentrated the scores are, grows as it shrinks.
        uniform_mixing: Weight in ``[0, 1]`` of the uniform distribution
            mixed into the leverage-score distribution
            (``selection="leverage"`` only).
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
        >>> lev = kl.NystromFeatures(5, jax.random.key(0), selection="leverage")
        >>> lev.fit(kl.RBF(), X)(X).shape
        (20, 5)
    """

    n_components: int = eqx.field(static=True)
    key: PRNGKeyArray
    selection: str = eqx.field(default="uniform", static=True)
    leverage_regularization: float = eqx.field(default=1e-3, static=True)
    uniform_mixing: float = eqx.field(default=0.5, static=True)
    jitter: float = eqx.field(default=1e-6, static=True)
    kernel: AbstractKernel | None = None
    landmarks: Float[Array, "M D"] | None = None

    def __check_init__(self) -> None:
        _check_positive("n_components", self.n_components)
        if self.selection not in ("uniform", "leverage"):
            raise ValueError(
                f"selection must be 'uniform' or 'leverage', got {self.selection!r}."
            )
        if not 0.0 <= self.uniform_mixing <= 1.0:
            raise ValueError(
                f"uniform_mixing must be in [0, 1], got {self.uniform_mixing}."
            )
        if not self.leverage_regularization > 0.0:
            raise ValueError(
                "leverage_regularization must be positive, got "
                f"{self.leverage_regularization}."
            )

    @classmethod
    def from_landmarks(
        cls,
        kernel: AbstractKernel,
        landmarks: Float[Array, "M D"],
        *,
        jitter: float = 1e-6,
    ) -> NystromFeatures:
        """A fitted map with given landmarks, e.g. inducing points.

        No selection happens, so the ``key`` is a placeholder.

        Examples:
            >>> import jax.numpy as jnp
            >>> import kernellib as kl
            >>> Z = jnp.linspace(-1.0, 1.0, 4)[:, None]
            >>> nys = kl.NystromFeatures.from_landmarks(kl.RBF(), Z)
            >>> nys(jnp.zeros((3, 1))).shape
            (3, 4)
        """
        return cls(
            landmarks.shape[0],
            jax.random.key(0),
            jitter=jitter,
            kernel=kernel,
            landmarks=landmarks,
        )

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
        if self.selection == "uniform":
            idx = jax.random.choice(self.key, n, (self.n_components,), replace=False)
        else:
            key_pilot, key_draw = jax.random.split(self.key)
            m0 = min(2 * self.n_components, n)
            pilot = jax.random.choice(key_pilot, n, (m0,), replace=False)
            scores = _ridge_leverage_scores(
                kernel, X, pilot, self.leverage_regularization, self.jitter
            )
            idx = jax.random.choice(
                key_draw,
                n,
                (self.n_components,),
                replace=False,
                p=(1.0 - self.uniform_mixing) * scores / jnp.sum(scores)
                + self.uniform_mixing / n,
            )
        return dataclasses.replace(self, kernel=kernel, landmarks=X[idx])

    def features(self, X: Float[Array, "N D"]) -> Float[Array, "N M"]:
        assert self.kernel is not None and self.landmarks is not None
        return _nystrom_features(self.kernel, self.landmarks, X, self.jitter)


def _nystrom_features(
    kernel: AbstractKernel,
    Z: Float[Array, "M D"],
    X: Float[Array, "N D"],
    jitter: float,
) -> Float[Array, "N M"]:
    """``L^{-1} k(Z, X)``, transposed, with ``K_ZZ + jitter I = L Lᵀ``."""
    K_zz = kernel(Z, Z)
    eps = jitter * jnp.mean(jnp.diag(K_zz))
    L = jnp.linalg.cholesky(K_zz + eps * jnp.eye(Z.shape[0], dtype=K_zz.dtype))
    return jsl.solve_triangular(L, kernel(Z, X), lower=True).T


def _ridge_leverage_scores(
    kernel: AbstractKernel,
    X: Float[Array, "N D"],
    pilot: Int[Array, " m0"],
    regularization: float,
    jitter: float,
) -> Float[Array, " N"]:
    r"""Approximate ridge leverage scores from a pilot Nyström map.

    With pilot features $\Phi$ (``N x m0``, $K \approx \Phi\Phi^\top$) the
    push-through identity gives the leverage of $\Phi\Phi^\top$ as
    $\phi_i^\top (\Phi^\top\Phi + \lambda n I)^{-1} \phi_i$; the Nyström
    residual on the diagonal, over $\lambda n$, is added back. Non-negative;
    they sum to about the effective dimension $d_{\mathrm{eff}}(\lambda)$.
    """
    n = X.shape[0]
    Phi = _nystrom_features(kernel, X[pilot], X, jitter)  # (N, m0)
    ridge = regularization * n
    gram = einsum(Phi, Phi, "n a, n b -> a b")
    chol = jnp.linalg.cholesky(gram + ridge * jnp.eye(gram.shape[0], dtype=gram.dtype))
    solved = jsl.solve_triangular(chol, Phi.T, lower=True)  # (m0, N)
    explained = reduce(solved**2, "a n -> n", "sum")
    residual = jnp.maximum(kernel.diag(X) - reduce(Phi**2, "n a -> n", "sum"), 0.0)
    return explained + residual / ridge
