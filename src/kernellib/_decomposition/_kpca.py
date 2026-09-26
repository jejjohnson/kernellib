r"""Kernel principal component analysis.

Kernel PCA (Schölkopf, Smola & Müller, 1998) is PCA in the feature space of a
kernel. With the centred Gram matrix $\tilde K = H K H$, $H = I - \tfrac1n
\mathbf 1 \mathbf 1^\top$, and its top eigenpairs
$\tilde K u_j = \lambda_j u_j$, the $j$-th component of a point $x$ is

$$
z_j(x) = \sum_i \alpha_{ij}\, \tilde k(x, x_i), \qquad
\alpha_{\cdot j} = u_j / \sqrt{\lambda_j},
$$

where $\tilde k(x, \cdot)$ is the kernel row centred with the training
statistics. On the training points $z_j = \sqrt{\lambda_j}\, u_j$. The
eigenvalues divided by $n$ are the feature-space variances.

The exact path costs $O(N^3)$. With ``approx``, a feature map
$\phi$ (Nyström, random Fourier, ...) replaces the kernel and kernel PCA
becomes ordinary PCA of $\phi(X)$, in $O(N R^2)$.
"""

from __future__ import annotations

import dataclasses

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float

from kernellib._kernels import AbstractKernel
from kernellib._spectral import AbstractFeatureMap


__all__ = ["KernelPCA"]


class KernelPCA(eqx.Module):
    r"""Kernel principal component analysis.

    Attributes:
        kernel: The kernel.
        n_components: Number of components.
        approx: Optional unfitted feature map for the ``O(N R^2)`` path.
        X_train: Training inputs (exact path), ``None`` before `fit`.
        alphas: Dual coefficients ``(N, n)`` (exact path).
        eigenvalues: Eigenvalues of the centred Gram matrix (or of
            $\bar\Phi^\top \bar\Phi$), ``(n,)``.
        explained_variance: ``eigenvalues / N``, the feature-space variances.
        embedding: The training points' components, ``(N, n)``.
        feature_map: The fitted feature map (approximate path).
        components: Principal directions in feature space ``(R, n)``
            (approximate path).

    Examples:
        >>> import jax
        >>> import jax.numpy as jnp
        >>> import kernellib as kl
        >>> X = jax.random.normal(jax.random.key(0), (50, 3))
        >>> kpca = kl.KernelPCA(kl.RBF(lengthscale=2.0), n_components=2).fit(X)
        >>> bool(jnp.allclose(kpca.transform(X), kpca.embedding, atol=1e-4))
        True
    """

    kernel: AbstractKernel
    n_components: int = eqx.field(default=2, static=True)
    approx: AbstractFeatureMap | None = None
    X_train: Float[Array, "N D"] | None = None
    alphas: Float[Array, "N n"] | None = None
    eigenvalues: Float[Array, " n"] | None = None
    explained_variance: Float[Array, " n"] | None = None
    embedding: Float[Array, "N n"] | None = None
    gram_column_means: Float[Array, " N"] | None = None
    gram_mean: Float[Array, ""] | None = None
    feature_map: AbstractFeatureMap | None = None
    feature_mean: Float[Array, " R"] | None = None
    components: Float[Array, "R n"] | None = None

    def __check_init__(self) -> None:
        if self.n_components < 1:
            raise ValueError(f"n_components must be >= 1, got {self.n_components}.")

    def fit(self, X: Float[Array, "N D"]) -> KernelPCA:
        """Fit the components on ``X``.

        Raises:
            ValueError: If ``n_components`` exceeds what the data (or the
                feature map) supports.
        """
        n = X.shape[0]
        if self.approx is not None:
            fmap = self.approx.fit(self.kernel, X)
            Phi = fmap(X)
            if self.n_components > min(Phi.shape):
                raise ValueError(
                    f"n_components={self.n_components} exceeds the rank bound "
                    f"{min(Phi.shape)} of the features."
                )
            mu = jnp.mean(Phi, axis=0)
            Pc = Phi - mu
            lam, U = jnp.linalg.eigh(Pc.T @ Pc)
            lam, U = lam[::-1][: self.n_components], U[:, ::-1][:, : self.n_components]
            return dataclasses.replace(
                self,
                eigenvalues=lam,
                explained_variance=lam / n,
                embedding=Pc @ U,
                feature_map=fmap,
                feature_mean=mu,
                components=U,
            )
        if self.n_components > n:
            raise ValueError(
                f"n_components={self.n_components} exceeds the number of points {n}."
            )
        K = self.kernel(X, X)
        col = jnp.mean(K, axis=0)
        total = jnp.mean(K)
        Kc = K - col[None, :] - col[:, None] + total
        lam, U = jnp.linalg.eigh(0.5 * (Kc + Kc.T))
        lam, U = lam[::-1][: self.n_components], U[:, ::-1][:, : self.n_components]
        safe = jnp.sqrt(jnp.clip(lam, min=jnp.finfo(lam.dtype).tiny))
        return dataclasses.replace(
            self,
            X_train=X,
            alphas=U / safe,
            eigenvalues=lam,
            explained_variance=lam / n,
            embedding=U * safe,
            gram_column_means=col,
            gram_mean=total,
        )

    def transform(self, X: Float[Array, "M D"]) -> Float[Array, "M n"]:
        """Components of new points.

        Raises:
            RuntimeError: If not fitted.
        """
        if self.components is not None:
            assert self.feature_map is not None and self.feature_mean is not None
            return (self.feature_map(X) - self.feature_mean) @ self.components
        if self.alphas is None:
            raise RuntimeError("KernelPCA is not fitted; call .fit(X) first.")
        assert self.X_train is not None and self.gram_column_means is not None
        assert self.gram_mean is not None
        Kt = self.kernel(X, self.X_train)
        Ktc = (
            Kt
            - self.gram_column_means[None, :]
            - jnp.mean(Kt, axis=1, keepdims=True)
            + self.gram_mean
        )
        return Ktc @ self.alphas
