r"""Separable space-time features: the Kronecker product of two feature maps.

A separable kernel $k((x, t), (x', t')) = k_s(x, x')\,k_t(t, t')$ is
approximated by the product of two feature maps,
$k_s \approx \phi_s^\top\phi_s$ and $k_t \approx \phi_t^\top\phi_t$:

$$
k_s(x, x')\,k_t(t, t')
\approx (\phi_s(x)^\top\phi_s(x'))(\phi_t(t)^\top\phi_t(t'))
= (\phi_s(x) \otimes \phi_t(t))^\top(\phi_s(x') \otimes \phi_t(t')),
$$

the mixed-product rule $(a^\top a')(b^\top b') = (a \otimes b)^\top
(a' \otimes b')$. The product has $R_s R_t$ features and is PSD by
construction. On a full grid the feature matrix is itself a Kronecker product,
which `gaussx.Kronecker` keeps factored.
"""

from __future__ import annotations

import dataclasses

import einx
import equinox as eqx
import gaussx as gx
import lineax as lx
from jaxtyping import Array, Float

from kernellib._kernels import AbstractKernel, ActiveDims, Product
from kernellib._spectral._base import AbstractFeatureMap


__all__ = ["KroneckerFeatures"]


class KroneckerFeatures(AbstractFeatureMap):
    r"""Features of a separable kernel $k_s k_t$ as a product of two maps.

    Inputs are ``X = [space columns | time columns]``: the first
    ``space_dims`` columns go to ``space_map``, the rest to ``time_map``. The
    features of one point are the Kronecker product of the factors',

    $$
    \phi(x, t) = \phi_s(x) \otimes \phi_t(t) \in \mathbb{R}^{R_s R_t},
    \qquad
    \phi(x, t)^\top\phi(x', t') = (\phi_s(x)^\top\phi_s(x'))\,
    (\phi_t(t)^\top\phi_t(t')),
    $$

    so for scattered data `features` stacks the **face-splitting** (row-wise
    Kronecker) product of $\Phi_s(X_s)$ and $\Phi_t(X_t)$, in $O(N R_s R_t)$.
    Index $(r, s)$ of the product is column $r R_t + s$ (the row-major
    convention of ``jnp.kron``).

    **Grid path.** For $N_s$ sites and $N_t$ times, all pairs, in site-major
    order (row $i N_t + j$ is site $i$ at time $j$), the feature matrix is the
    Kronecker product
    $\Phi = \Phi_s \otimes \Phi_t \in \mathbb{R}^{N_s N_t \times R_s R_t}$.
    `features_grid` returns it as a `gaussx.Kronecker` of the two factors
    without forming it, which costs $O(N_s R_s + N_t R_t)$ memory instead of
    $O(N_s N_t R_s R_t)$. The identities used, with **row-major** $\mathrm{vec}$
    (stack the rows; this is ``rearrange(X, "a b -> (a b)")`` and matches
    ``jnp.kron``; in column-major $\mathrm{vec}$ the same identity reads
    $(A \otimes B)\mathrm{vec}(X) = \mathrm{vec}(B X A^\top)$):

    $$
    (A \otimes B)\,\mathrm{vec}(X) = \mathrm{vec}(A X B^\top),
    \qquad
    (A \otimes B)^\top = A^\top \otimes B^\top,
    \qquad
    (A \otimes B)(C \otimes D) = AC \otimes BD.
    $$

    - **Predictions.** For weights $W \in \mathbb{R}^{R_s \times R_t}$, the
      latent field on the grid is $F = \Phi_s W \Phi_t^\top
      \in \mathbb{R}^{N_s \times N_t}$, i.e. $\mathrm{vec}(F) = \Phi\,
      \mathrm{vec}(W)$, in $O(N_s R_s R_t + N_s N_t R_t)$.
    - **Normal equations.** With $G_s = \Phi_s^\top\Phi_s$ and $G_t =
      \Phi_t^\top\Phi_t$, $\Phi^\top\Phi = G_s \otimes G_t$ and $\Phi^\top
      \mathrm{vec}(Y) = \mathrm{vec}(\Phi_s^\top Y \Phi_t)$ for a grid of
      observations $Y \in \mathbb{R}^{N_s \times N_t}$.
    - **Ridge solve.** Eigendecompose the small factors, $G_s = V_s D_s V_s^\top$
      and $G_t = V_t D_t V_t^\top$. Then $G_s \otimes G_t + \lambda I
      = (V_s \otimes V_t)(D_s \otimes D_t + \lambda I)(V_s \otimes V_t)^\top$
      and the ridge weights $(\Phi^\top\Phi + \lambda I)^{-1}\Phi^\top y$ are
      computed in $O(R_s^3 + R_t^3 + N_s R_s R_t + N_t R_s R_t)$ with
      no $N_s N_t \times R_s R_t$ matrix. The same factorisation gives the
      Woodbury solves and log-determinants of a GP with noise $\lambda$
      (Saatçi, 2012; Wilson & Nickisch, 2015).

    Pseudocode of the ridge solve on a grid (``Y`` is ``(Ns, Nt)``):

    ```text
    Fs, Ft = space_map(Xs), time_map(Xt)         # (Ns, Rs), (Nt, Rt)
    ds, Vs = eigh(Fs.T @ Fs); dt, Vt = eigh(Ft.T @ Ft)
    B = Fs.T @ Y @ Ft                            # (Rs, Rt), = unvec(Phi.T vec(Y))
    B = Vs.T @ B @ Vt                            # rotate into the eigenbasis
    W = B / (outer(ds, dt) + lam)                # diagonal solve
    W = Vs @ W @ Vt.T                            # rotate back
    ```

    The fit contract is a separable kernel built as
    ``Product(ActiveDims(k_s, dims=space), ActiveDims(k_t, dims=time))``; each
    map is fitted to its own factor on its own columns, so the hyperparameters
    of both factors are read at call time and differentiable.

    Numerics. The Gram of the product is the elementwise product of the two
    Grams, so the error is that of the factors: for two random maps it is
    roughly $\epsilon_s\,\sigma_t^2 + \sigma_s^2\,\epsilon_t$ (variance times
    the factor's own error), no worse than the sum of the factors' errors.
    Deterministic factors (`PeriodicFeatures`, `LaplaceEigenfunctionFeatures`,
    `SphericalHarmonicFeatures`) give a deterministic product. The kernel's
    total variance is shared between the factors, so give one factor the
    variance and the other unit variance to keep the scale interpretable.
    The ridge solve squares the condition number through $G_s \otimes G_t$,
    so truncate tiny eigenvalues ($d_s d_t \ll \lambda$ is harmless, but
    $\lambda \to 0$ is not).

    Attributes:
        space_map: A feature map for the space columns.
        time_map: A feature map for the time columns.
        space_dims: Number of leading columns of ``X`` that are space; the
            remaining columns are time (static).

    References:
        Saatçi, Y. (2012). *Scalable Inference for Structured Gaussian Process
        Models.* PhD thesis, University of Cambridge.
        Wilson, A. G. & Nickisch, H. (2015). Kernel interpolation for scalable
        structured Gaussian processes (KISS-GP). *ICML*.

    Examples:
        >>> import einx
        >>> import jax.numpy as jnp
        >>> import kernellib as kl
        >>> Xs = einx.id("n -> n 1", jnp.linspace(-1.0, 1.0, 6))
        >>> Xt = einx.id("n -> n 1", jnp.linspace(-1.0, 1.0, 5))
        >>> ks, kt = kl.RBF(lengthscale=0.5), kl.RBF(lengthscale=0.7)
        >>> k = kl.Product(
        ...     kl.ActiveDims(ks, dims=(0,)), kl.ActiveDims(kt, dims=(1,))
        ... )
        >>> lap = kl.LaplaceEigenfunctionFeatures(24, boundary_factor=3.0)
        >>> kf = kl.KroneckerFeatures(lap, lap, space_dims=1)
        >>> X = jnp.stack(  # site-major grid of (site, time) pairs
        ...     [jnp.repeat(Xs[:, 0], 5), jnp.tile(Xt[:, 0], 6)], axis=1
        ... )
        >>> kf = kf.fit(k, X)
        >>> Phi = kf(X)
        >>> Phi.shape
        (30, 576)
        >>> G = einx.dot("n f, m f -> n m", Phi, Phi)
        >>> bool(jnp.max(jnp.abs(G - k(X, X))) < 1e-3)
        True
        >>> op = kf.features_grid(Xs, Xt)  # gaussx.Kronecker, never densified
        >>> op.out_size(), op.in_size()
        (30, 576)
        >>> bool(jnp.allclose(op.as_matrix(), Phi))
        True
    """

    space_map: AbstractFeatureMap
    time_map: AbstractFeatureMap
    space_dims: int = eqx.field(static=True)

    def __check_init__(self) -> None:
        if self.space_dims < 1:
            raise ValueError(f"space_dims must be >= 1, got {self.space_dims}.")

    @property
    def is_fitted(self) -> bool:
        """Whether both factor maps have been fitted."""
        return self.space_map.is_fitted and self.time_map.is_fitted

    def fit(self, kernel: AbstractKernel, X: Float[Array, "N D"]) -> KroneckerFeatures:
        """Return a copy with each map fitted to its factor of ``kernel``.

        Args:
            kernel: ``Product(ActiveDims(k_s, dims=space), ActiveDims(k_t,
                dims=time))`` with ``space = (0, ..., space_dims - 1)`` and
                ``time = (space_dims, ..., D - 1)``.
            X: ``(N, D)`` inputs, ``[space columns | time columns]``.

        Raises:
            TypeError: If ``kernel`` is not a two-factor `Product` of
                `ActiveDims`.
            ValueError: If ``X`` has no time column, or the `ActiveDims`
                ``dims`` are not the space and time column blocks.
        """
        if not (
            isinstance(kernel, Product)
            and len(kernel.kernels) == 2
            and all(isinstance(f, ActiveDims) for f in kernel.kernels)
        ):
            raise TypeError(
                "KroneckerFeatures needs Product(ActiveDims(k_s, space), "
                f"ActiveDims(k_t, time)), got {kernel!r:.80}."
            )
        k_s, k_t = kernel.kernels
        assert isinstance(k_s, ActiveDims) and isinstance(k_t, ActiveDims)
        if X.ndim != 2 or X.shape[1] <= self.space_dims:
            raise ValueError(
                f"X must be (N, D) with D > space_dims={self.space_dims}, got "
                f"shape {X.shape}."
            )
        space = tuple(range(self.space_dims))
        time = tuple(range(self.space_dims, X.shape[1]))
        if k_s.dims != space or k_t.dims != time:
            raise ValueError(
                f"ActiveDims must be dims={space} for space and dims={time} "
                f"for time, got {k_s.dims} and {k_t.dims}."
            )
        return dataclasses.replace(
            self,
            space_map=self.space_map.fit(k_s.kernel, X[:, : self.space_dims]),
            time_map=self.time_map.fit(k_t.kernel, X[:, self.space_dims :]),
        )

    def features(self, X: Float[Array, "N D"]) -> Float[Array, "N RsRt"]:
        r"""Row-wise Kronecker product $\phi_s(x_s) \otimes \phi_t(x_t)$."""
        phi_s = self.space_map(X[:, : self.space_dims])
        phi_t = self.time_map(X[:, self.space_dims :])
        return einx.multiply("n r, n s -> n (r s)", phi_s, phi_t)

    def features_grid(
        self, Xs: Float[Array, "Ns Ds"], Xt: Float[Array, "Nt Dt"]
    ) -> gx.Kronecker:
        r"""The grid feature matrix $\Phi_s \otimes \Phi_t$ as a `gaussx.Kronecker`.

        Rows are the ``Ns * Nt`` (site, time) pairs in site-major order, i.e.
        row $i N_t + j$ is ``(Xs[i], Xt[j])``; the ``Ns * Nt x Rs * Rt``
        matrix is never formed. Use ``op.mv`` / ``op.T.mv`` or the identity
        $(\Phi_s \otimes \Phi_t)\,\mathrm{vec}(W) = \mathrm{vec}(\Phi_s W
        \Phi_t^\top)$ (row-major) for products.

        Raises:
            RuntimeError: If the map has not been fitted.
        """
        if not self.is_fitted:
            raise RuntimeError(
                "KroneckerFeatures is not fitted; call .fit(kernel, X) first."
            )
        return gx.Kronecker(
            lx.MatrixLinearOperator(self.space_map(Xs)),
            lx.MatrixLinearOperator(self.time_map(Xt)),
        )
