r"""Nonstationary kernels with a location-dependent lengthscale or anisotropy.

- `Gibbs` has a lengthscale $\ell(x)$ (a scalar, or one per dimension) that
  varies with location (Gibbs, 1997).
- `PaciorekSchervish` has a full local anisotropy matrix $\Sigma(x)$
  (Paciorek & Schervish, 2006).

Both wrap a stationary isotropic correlation $R$ that is positive definite in
every dimension (RBF, Matérn, rational quadratic). The kernels are not
stationary: they depend on $x$ and $x'$, not on $x - x'$.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import equinox as eqx
import jax.numpy as jnp
import jax.scipy.linalg as jsl
from jaxtyping import Array, Float

from kernellib._einx import einsum, reduce, repeat
from kernellib._kernels._base import AbstractPointwiseKernel
from kernellib._kernels._stationary import RBF, Matern, RationalQuadratic


__all__ = ["Gibbs", "PaciorekSchervish"]

_BASES = ("rbf", "matern12", "matern32", "matern52", "rq")
_NU = {"matern12": 0.5, "matern32": 1.5, "matern52": 2.5}


def _check_base(base: str) -> None:
    if base not in _BASES:
        raise ValueError(
            f"base must be one of {_BASES} (an isotropic correlation that is PD in "
            f"every dimension; compact or d <= 3 models such as Spherical and "
            f"Wendland are not valid), got {base!r}."
        )


def _profile(
    base: str, alpha: Float[Array, ""], r2: Float[Array, ...]
) -> Float[Array, ...]:
    """Unit-variance, unit-lengthscale profile $R$ at squared distance ``r2``."""
    if base == "rbf":
        return RBF().shape(r2)
    if base == "rq":
        return RationalQuadratic(alpha=alpha).shape(r2)
    return Matern(nu=_NU[base]).shape(r2)


def _log_diag(M: Float[Array, "D D"]) -> Float[Array, " D"]:
    return jnp.log(jnp.abs(jnp.diagonal(M)))


class Gibbs(AbstractPointwiseKernel):
    r"""Gibbs kernel: a lengthscale that varies with location.

    With a lengthscale field $\ell(x) > 0$ (a scalar, or a vector
    $\ell_d(x)$ with one entry per input dimension),

    $$
    k(x, x') = \sigma^2 \prod_{d=1}^{D}
    \sqrt{\frac{2\,\ell_d(x)\,\ell_d(x')}{\ell_d(x)^2 + \ell_d(x')^2}}\;
    R\!\left(\sqrt{\sum_{d=1}^{D}
    \frac{2\,(x_d - x'_d)^2}{\ell_d(x)^2 + \ell_d(x')^2}}\right),
    $$

    where $R$ is the unit correlation selected by ``base``. For a scalar
    field this is the prefactor $(2\ell\ell'/(\ell^2+\ell'^2))^{D/2}$ and
    distance $\lVert x-x'\rVert\sqrt{2/(\ell^2+\ell'^2)}$. With
    ``base="rbf"`` it is Gibbs' $\exp(-\lVert x-x'\rVert^2/(\ell^2+\ell'^2))$.
    It is the `PaciorekSchervish` kernel with
    $\Sigma(x) = \operatorname{diag}(\ell_d(x)^2)$. A constant field gives
    the stationary kernel exactly, and $k(x, x) = \sigma^2$ for every
    field.

    **Positive definiteness.** $k$ is the Paciorek & Schervish (2004, 2006,
    Thm. 1) construction with diagonal $\Sigma(x)$, which is PD on
    $\mathbb{R}^D$ for any $\ell(\cdot) > 0$, however rough, provided $R$
    is PD in every dimension: the Gaussian scale-mixture representation
    $R(r) = \int e^{-r^2 \tau}\,dF(\tau)$ (RBF, Matérn, rational quadratic)
    turns $k$ into a mixture of kernels
    $\int_{\mathbb{R}^D} \phi(x-u;\,\Sigma(x))\,\phi(x'-u;\,\Sigma(x'))\,du$
    of Gaussian smoothing kernels, each of which is a Gram matrix
    $\int f(x,u) f(x',u)\,du$ (Gibbs, 1997; Higdon, Swall & Kern, 1999).
    Compact or $d \le 3$ models (Spherical, Wendland) are not valid
    choices, hence the ``base`` literal.

    **Pseudocode.**

    ```text
    l, l' = ell(x), ell(x')                      # (D,) after broadcasting
    s = l^2 + l'^2
    log_pref = 0.5 * sum(log 2 + log l + log l' - log s)
    Q = sum(2 (x - x')^2 / s)
    k = variance * exp(log_pref) * R(Q)          # R takes the squared distance
    ```

    **Numerics.** The prefactor is formed in log space,
    $\log s = \operatorname{logaddexp}(2\log\ell, 2\log\ell')$, so very
    small or very different lengthscales neither underflow in float32 nor
    lose the factor to cancellation. The fields are evaluated once per
    point and the ``(N, N)`` Gram is formed from broadcasts, so cost is
    $O(N^2 D)$. ``lengthscale_fn`` must return strictly positive values (for
    instance a ``softplus`` or ``exp`` output head); this is not
    enforced. Matérn-1/2 is not differentiable at coincident points, as in
    the stationary kernel.

    Attributes:
        lengthscale_fn: Callable (or `equinox.Module`, such as
            `equinox.nn.MLP`) mapping a point ``(D,)`` to ``l(x) > 0``:
            a scalar, or a ``(D,)`` vector. Differentiable with respect to
            its parameters.
        variance: Signal variance $\sigma^2$.
        base: ``"rbf"``, ``"matern12"``, ``"matern32"``, ``"matern52"`` or
            ``"rq"``. Static.
        alpha: Shape of the rational quadratic (used only for ``"rq"``).

    Raises:
        ValueError: If ``base`` is not one of the above, or the field does
            not return a scalar or a ``(D,)`` vector.

    References:
        Gibbs, M. N. (1997). *Bayesian Gaussian processes for regression and
        classification*. PhD thesis, University of Cambridge.
        Higdon, D., Swall, J. & Kern, J. (1999). Non-stationary spatial
        modeling. *Bayesian Statistics 6*.
        Paciorek, C. J. & Schervish, M. J. (2004). Nonstationary covariance
        functions for Gaussian process regression. *NIPS 16*.
        Paciorek, C. J. & Schervish, M. J. (2006). Spatial modelling using a
        new class of nonstationary covariance functions. *Environmetrics*,
        17(5).

    Examples:
        >>> import jax.numpy as jnp
        >>> import kernellib as kl
        >>> k = kl.Gibbs(lambda x: 0.5 + 0.0 * x[0])  # constant field
        >>> x, y = jnp.array([0.0, 0.0]), jnp.array([0.3, 0.4])
        >>> bool(jnp.allclose(k.pairwise(x, y), kl.RBF(lengthscale=0.5).pairwise(x, y)))
        True
        >>> # a field that grows with x1: shorter correlation on the left
        >>> k = kl.Gibbs(lambda x: 0.1 + 0.5 * jnp.exp(x[0]))
        >>> X = jnp.array([[-1.0, 0.0], [-0.9, 0.0], [1.0, 0.0], [1.1, 0.0]])
        >>> K = k(X, X)
        >>> bool(K[2, 3] > K[0, 1])
        True
        >>> bool(jnp.allclose(jnp.diag(K), 1.0))
        True
    """

    lengthscale_fn: Callable[[Float[Array, " D"]], Float[Array, ""]]
    variance: Float[Array, ""] = eqx.field(default=1.0, converter=jnp.asarray)
    base: Literal["rbf", "matern12", "matern32", "matern52", "rq"] = eqx.field(
        default="rbf", static=True
    )
    alpha: Float[Array, ""] = eqx.field(default=1.0, converter=jnp.asarray)

    def __check_init__(self) -> None:
        _check_base(self.base)

    def _lengthscales(self, x: Float[Array, " D"]) -> Float[Array, " D"]:
        ell = jnp.asarray(self.lengthscale_fn(x))
        if ell.ndim > 1 or (ell.ndim == 1 and ell.shape[0] not in (1, x.shape[-1])):
            raise ValueError(
                "lengthscale_fn must return a scalar or a vector of shape "
                f"({x.shape[-1]},), got {ell.shape}."
            )
        return jnp.broadcast_to(ell, x.shape[-1:])

    def pairwise(
        self, x: Float[Array, " D"], y: Float[Array, " D"]
    ) -> Float[Array, ""]:
        lx, ly = self._lengthscales(x), self._lengthscales(y)
        log_s = jnp.logaddexp(2.0 * jnp.log(lx), 2.0 * jnp.log(ly))
        log_pref = 0.5 * reduce(
            jnp.log(2.0) + jnp.log(lx) + jnp.log(ly) - log_s, "d ->", "sum"
        )
        diff = x - y
        q = reduce(2.0 * diff * diff * jnp.exp(-log_s), "d ->", "sum")
        return self.variance * jnp.exp(log_pref) * _profile(self.base, self.alpha, q)

    def diag(self, X: Float[Array, "N D"]) -> Float[Array, " N"]:
        return repeat(self.variance, "-> n", n=X.shape[0])


class PaciorekSchervish(AbstractPointwiseKernel):
    r"""Paciorek-Schervish kernel: a location-dependent anisotropy matrix.

    With a local covariance field $\Sigma(x) \succ 0$ (the shape and
    orientation of the correlation ellipse at $x$), and
    $\bar\Sigma = (\Sigma(x) + \Sigma(x'))/2$,

    $$
    k(x, x') = \sigma^2\,|\Sigma(x)|^{1/4}\,|\Sigma(x')|^{1/4}\,
    |\bar\Sigma|^{-1/2}\; R\!\left(\sqrt{Q}\right),\qquad
    Q = (x-x')^\top \bar\Sigma^{-1} (x-x'),
    $$

    with $R$ the unit correlation selected by ``base`` (its argument is the
    Mahalanobis distance $\sqrt Q$). A constant $\Sigma = \Lambda$ gives the
    stationary kernel with Mahalanobis metric $\Lambda^{-1}$, e.g.
    $\Sigma = \operatorname{diag}(\ell_d^2)$ the ARD kernel;
    $k(x, x) = \sigma^2$ always. `Gibbs` is the case
    $\Sigma(x) = \operatorname{diag}(\ell_d(x)^2)$.

    **Positive definiteness.** Paciorek & Schervish (2004; 2006, Thm. 1):
    if $R$ is an isotropic correlation that is PD on $\mathbb{R}^d$ for
    every $d$, then $k$ is PD on $\mathbb{R}^D$ for any $\Sigma(\cdot)
    \succ 0$, with no smoothness assumption on the field. The proof writes
    $R(r) = \int e^{-r^2\tau}\,dF(\tau)$ (a Gaussian scale mixture, by
    Schoenberg's theorem), and shows each Gaussian term, with the
    prefactor, to be the convolution
    $\int \phi(x-u;\,\Sigma(x)/(2\tau))\,\phi(x'-u;\,\Sigma(x')/(2\tau))\,du$
    up to a constant, which is a Gram matrix $\int f(x,u) f(x',u)\,du$,
    hence PSD; mixtures of PSD kernels are PSD. RBF, every Matérn and the
    rational quadratic qualify; compact or $d \le 3$ models (Spherical,
    Wendland) do not, hence the ``base`` literal.

    **Pseudocode.**

    ```text
    L, L' = cov_fn(x), cov_fn(x')                # Cholesky factors of Sigma
    S_bar = (L L^T + L' L'^T) / 2;  C = chol(S_bar)
    log_pref = 0.5 sum(log|diag L|) + 0.5 sum(log|diag L'|) - sum(log diag C)
    Q = || C^{-1} (x - x') ||^2                  # triangular solve
    k = variance * exp(log_pref) * R(Q)
    ```

    **Numerics.** The determinants come from the Cholesky factors,
    $\log|\Sigma| = 2\sum_i \log|L_{ii}|$, and the prefactor is formed in
    log space, which avoids the under- and overflow of $|\Sigma|$ for small
    or large lengthscales (the determinant of a $D \times D$ matrix with
    entries $\sim 10^{-2}$ is already below float32's useful range for
    $D \ge 8$). $Q$ uses one triangular solve, not an inverse. Each pair
    costs one $D \times D$ Cholesky, which suits $D \le 3$ (space, or
    chordal coordinates on the sphere). ``cov_fn`` must return a factor with
    a nonzero diagonal (use a ``softplus`` or ``exp`` head for it);
    this is not enforced.

    Attributes:
        cov_fn: Callable (or `equinox.Module`) mapping a point ``(D,)`` to a
            ``(D, D)`` lower-triangular Cholesky factor $L(x)$ with
            $\Sigma(x) = L L^\top$ (any nonsingular square ``L`` gives a
            valid $\Sigma$; only the lower triangle is used).
            Differentiable with respect to its parameters.
        variance: Signal variance $\sigma^2$.
        base: ``"rbf"``, ``"matern12"``, ``"matern32"``, ``"matern52"`` or
            ``"rq"``. Static.
        alpha: Shape of the rational quadratic (used only for ``"rq"``).

    Raises:
        ValueError: If ``base`` is not one of the above, or ``cov_fn`` does
            not return a ``(D, D)`` matrix.

    References:
        Paciorek, C. J. & Schervish, M. J. (2004). Nonstationary covariance
        functions for Gaussian process regression. *NIPS 16*.
        Paciorek, C. J. & Schervish, M. J. (2006). Spatial modelling using a
        new class of nonstationary covariance functions. *Environmetrics*,
        17(5).
        Higdon, D., Swall, J. & Kern, J. (1999). Non-stationary spatial
        modeling. *Bayesian Statistics 6*.

    Examples:
        >>> import jax.numpy as jnp
        >>> import kernellib as kl
        >>> L = jnp.diag(jnp.array([0.5, 2.0]))  # Sigma = diag(0.25, 4)
        >>> k = kl.PaciorekSchervish(lambda x: L, base="matern52")
        >>> ard = kl.Matern(nu=2.5, lengthscale=jnp.array([0.5, 2.0]))
        >>> x, y = jnp.array([0.0, 0.0]), jnp.array([0.3, 0.4])
        >>> bool(jnp.allclose(k.pairwise(x, y), ard.pairwise(x, y)))
        True
        >>> # lengthscale 0.2 on the left, 1.0 on the right, isotropic
        >>> def cov(x):
        ...     return jnp.eye(2) * (0.2 + 0.8 * (x[0] > 0))
        >>> kn = kl.PaciorekSchervish(cov)
        >>> X = jnp.array([[-1.0, 0.0], [-0.8, 0.0], [1.0, 0.0], [1.2, 0.0]])
        >>> bool(kn(X, X)[2, 3] > kn(X, X)[0, 1])
        True
    """

    cov_fn: Callable[[Float[Array, " D"]], Float[Array, "D D"]]
    variance: Float[Array, ""] = eqx.field(default=1.0, converter=jnp.asarray)
    base: Literal["rbf", "matern12", "matern32", "matern52", "rq"] = eqx.field(
        default="rbf", static=True
    )
    alpha: Float[Array, ""] = eqx.field(default=1.0, converter=jnp.asarray)

    def __check_init__(self) -> None:
        _check_base(self.base)

    def _factor(self, x: Float[Array, " D"]) -> Float[Array, "D D"]:
        L = jnp.asarray(self.cov_fn(x))
        d = x.shape[-1]
        if L.shape != (d, d):
            raise ValueError(f"cov_fn must return a ({d}, {d}) matrix, got {L.shape}.")
        return jnp.tril(L)

    def pairwise(
        self, x: Float[Array, " D"], y: Float[Array, " D"]
    ) -> Float[Array, ""]:
        Lx, Ly = self._factor(x), self._factor(y)
        half = 0.5 * (
            einsum(Lx, Lx, "i k, j k -> i j") + einsum(Ly, Ly, "i k, j k -> i j")
        )
        C = jnp.linalg.cholesky(half)
        log_pref = (
            0.5 * reduce(_log_diag(Lx), "d ->", "sum")
            + 0.5 * reduce(_log_diag(Ly), "d ->", "sum")
            - reduce(_log_diag(C), "d ->", "sum")
        )
        z = jsl.solve_triangular(C, x - y, lower=True)
        q = reduce(z * z, "d ->", "sum")
        return self.variance * jnp.exp(log_pref) * _profile(self.base, self.alpha, q)

    def diag(self, X: Float[Array, "N D"]) -> Float[Array, " N"]:
        return repeat(self.variance, "-> n", n=X.shape[0])
