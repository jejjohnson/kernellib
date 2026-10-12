r"""Weighted Parzen kernel density estimation (KD6).

With training points $x_i \in \mathbb R^D$ and weights $\beta_i \ge 0$,
$\sum_i \beta_i = 1$ (Johnson et al. 2020, §5.1, Eq 18 and its weighted form),

$$
\hat p(x) = \frac1c \sum_{i=1}^N \beta_i\, k(x, x_i), \qquad
c = \int_{\mathbb R^D} k(x, 0)\, dx = S(0),
$$

with $S$ the kernel's spectral density (`spectral_density`). Everything is
evaluated in log space: with $\ell_i(x) = \log\beta_i + \log k(x, x_i)$,
$\log\hat p(x) = \operatorname{logsumexp}_i \ell_i(x) - \log c$, and
$\log k$ comes from the kernel's exact ``_log_shape``, never from
``log(kernel(...))``, so the estimate is finite arbitrarily far from the data.

For the Gaussian (`RBF`) kernel the score and Hessian of $\log\hat p$ are
closed form in the posterior weights $\pi_i(x) = \operatorname{softmax}_i
\ell_i(x)$; for any other kernel they are the KD1 operators `gradient` /
`hessian` of `KernelDensity.log_density`. ``batch_size`` streams both the
query and the training axis (an online log-sum-exp), so no ``(M, N)`` Gram is
ever formed.
"""

from __future__ import annotations

import dataclasses
import warnings
from collections.abc import Callable
from typing import Literal

import einx
import equinox as eqx
import jax
import jax.numpy as jnp
from jax.scipy.optimize import minimize
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray

from kernellib._derivatives import gradient, hessian
from kernellib._einx import einsum, rearrange, reduce
from kernellib._heuristics import _weighted_rule_lengthscale
from kernellib._kernels import RBF, AbstractKernel, AbstractStationaryKernel
from kernellib._regression._krr import _require
from kernellib.functional._distances import _pairwise_sq_dist


__all__ = ["KernelDensity"]

Bandwidth = (
    float
    | Float[Array, ""]
    | Float[Array, " D"]
    | Literal["scott", "silverman", "loo"]
    | None
)

_LogKernel = Callable[[Float[Array, "M D"], Float[Array, "B D"]], Float[Array, "M B"]]


# -- the log-kernel and the normaliser -------------------------------------------


def _log_kernel(
    kernel: AbstractKernel, X1: Float[Array, "M D"], X2: Float[Array, "N D"]
) -> Float[Array, "M N"]:
    r"""$\log k(x, x')$: exact for stationary kernels, ``log(k)`` otherwise."""
    if isinstance(kernel, AbstractStationaryKernel):
        r2 = _pairwise_sq_dist(X1, X2, kernel.lengthscale)
        return jnp.log(kernel.variance) + kernel._log_shape(r2)
    return jnp.log(kernel(X1, X2))


_NORMALIZE_HINT = (
    "Pass normalize=False for the unnormalised estimate log sum_i beta_i "
    "k(x, x_i) (a log-density up to an additive constant; the score and "
    "Hessian are unchanged)."
)


def _log_normalizer(
    kernel: AbstractKernel, d: int, *, check: bool = True
) -> Float[Array, ""]:
    r"""$\log c = \log S(0) = \log\sigma^2 + \sum_d\log\ell_d + \log s(0)$.

    $S(0) = \int k$ by kernellib's Fourier convention (`spectral_density`).
    It is computed in log space so that $\prod_d \ell_d$ cannot underflow.

    Raises:
        ValueError: If the kernel is not an `AbstractStationaryKernel` with a
            spectral density, or (with ``check``) if $S(0)$ is not finite and
            positive (``RationalQuadratic`` with $\alpha \le D/2$).
    """
    name = type(kernel).__name__
    if not isinstance(kernel, AbstractStationaryKernel):
        raise ValueError(
            f"normalize=True needs an AbstractStationaryKernel with a finite "
            f"integral; {name} is not one. {_NORMALIZE_HINT}"
        )
    ell = kernel._lengthscale_vector(d)
    try:
        s0 = kernel.unit_spectral_density(jnp.zeros((), dtype=ell.dtype), d)
    except NotImplementedError as err:
        raise ValueError(
            f"normalize=True needs the integral of the kernel, its spectral "
            f"density at 0, but {name} has no spectral density. {_NORMALIZE_HINT}"
        ) from err
    log_c = jnp.log(kernel.variance) + jnp.sum(jnp.log(ell)) + jnp.log(s0)
    if not check:
        return log_c
    return _require(
        jnp.isfinite(log_c),
        f"normalize=True needs a kernel with a finite, positive integral, but "
        f"{name} has integral {{inf, 0 or NaN}} in D = {d} (RationalQuadratic "
        f"needs alpha > D / 2). {_NORMALIZE_HINT}",
        log_c,
    )


# -- streaming log-sum-exp and posterior moments ----------------------------------


def _block_terms(
    log_k: _LogKernel,
    Xq: Float[Array, "M D"],
    exclude: Int[Array, " M"] | None,
    Xb: Float[Array, "B D"],
    log_beta: Float[Array, " B"],
    index: Int[Array, " B"],
) -> Float[Array, "M B"]:
    r"""$\ell_{mb} = \log\beta_b + \log k(x_m, x_b)$; ``-inf`` at ``b = exclude[m]``."""
    t = einx.add("m b, b -> m b", log_k(Xq, Xb), log_beta)
    if exclude is not None:
        t = jnp.where(einx.equal("m, b -> m b", exclude, index), -jnp.inf, t)
    return t


def _outer(X: Float[Array, "N D"]) -> Float[Array, "N D D"]:
    """Row-wise outer products $x_n x_n^\top$."""
    return einx.multiply("n d, n e -> n d e", X, X)


def _moments(
    log_k: _LogKernel,
    Xq: Float[Array, "M D"],
    X: Float[Array, "N D"],
    log_beta: Float[Array, " N"],
    *,
    order: int,
    batch_size: int | None,
    center: Float[Array, " D"] | None = None,
    exclude: Int[Array, " M"] | None = None,
) -> tuple[Array, Array | None, Array | None]:
    r"""The log-sum-exp of $\ell_{mi}$ and the first ``order`` posterior moments.

    Returns $\operatorname{lse}_i \ell_{mi}$ ``(M,)`` and, for ``order >= 1``,
    $E_\pi[x_i - c]$ ``(M, D)``, for ``order >= 2``, also
    $E_\pi[(x_i - c)(x_i - c)^\top]$ ``(M, D, D)``, with
    $\pi_{mi} = \operatorname{softmax}_i \ell_{mi}$. ``center`` $c$ (the
    weighted mean) keeps the second moment free of cancellation when the data
    sit far from the origin.

    ``batch_size=None`` builds the ``(M, N)`` block at once. An integer
    streams it: queries go through ``jax.lax.map(..., batch_size)`` and the
    training axis through a checkpointed ``jax.lax.scan`` that carries the
    running max $m$, $s = \sum_i e^{\ell_i - m}$ and the moment sums
    (online softmax), so memory is $O(\text{batch\_size}^2 D^{\text{order}})$.
    """
    n, d = X.shape
    if center is None:
        center = jnp.zeros((d,), dtype=X.dtype)

    if batch_size is None:
        t = _block_terms(log_k, Xq, exclude, X, log_beta, jnp.arange(n))
        lse = reduce(t, "m n -> m", "logsumexp")
        if order == 0:
            return lse, None, None
        pi = jnp.exp(einx.subtract("m n, m -> m n", t, lse))
        Xc = einx.subtract("n d, d -> n d", X, center)
        mean = einsum(pi, Xc, "m n, n d -> m d")
        second = einsum(pi, _outer(Xc), "m n, n d e -> m d e") if order >= 2 else None
        return lse, mean, second

    bs = batch_size
    n_chunks = -(-n // bs)
    pad = n_chunks * bs - n
    # Padded rows get log beta = -inf, so they add exactly zero.
    chunks = (
        rearrange(jnp.pad(X, ((0, pad), (0, 0))), "(c b) d -> c b d", b=bs),
        rearrange(
            jnp.pad(log_beta, (0, pad), constant_values=-jnp.inf),
            "(c b) -> c b",
            b=bs,
        ),
        rearrange(jnp.arange(n_chunks * bs), "(c b) -> c b", b=bs),
    )

    def one_query(
        args: tuple[Float[Array, " D"], Int[Array, ""] | None],
    ) -> tuple[Array, Array | None, Array | None]:
        x, ex = args
        xq = rearrange(x, "d -> 1 d")
        exq = None if ex is None else rearrange(ex, " -> 1")

        def body(carry: tuple, chunk: tuple) -> tuple[tuple, None]:
            m, s, *sums = carry
            Xb, lb, ib = chunk
            t = _block_terms(log_k, xq, exq, Xb, lb, ib)
            # The shift only conditions exp: its value cancels and it carries
            # no gradient.
            m_new = jax.lax.stop_gradient(jnp.maximum(m, reduce(t, "m b -> m", "max")))
            shift = jnp.where(jnp.isfinite(m_new), m_new, 0.0)
            rescale = jnp.exp(m - shift)  # 0 while m is still -inf
            e = jnp.exp(einx.subtract("m b, m -> m b", t, shift))
            s = s * rescale + reduce(e, "m b -> m", "sum")
            new_sums = []
            if order >= 1:
                Xc = einx.subtract("b d, d -> b d", Xb, center)
                new_sums.append(
                    einx.multiply("m d, m -> m d", sums[0], rescale)
                    + einsum(e, Xc, "m b, b d -> m d")
                )
                if order >= 2:
                    new_sums.append(
                        einx.multiply("m d e, m -> m d e", sums[1], rescale)
                        + einsum(e, _outer(Xc), "m b, b d e -> m d e")
                    )
            return (m_new, s, *new_sums), None

        dtype = jnp.result_type(X.dtype, log_beta.dtype)
        init: list[Array] = [
            jnp.full((1,), -jnp.inf, dtype=dtype),
            jnp.zeros((1,), dtype=dtype),
        ]
        if order >= 1:
            init.append(jnp.zeros((1, d), dtype=dtype))
        if order >= 2:
            init.append(jnp.zeros((1, d, d), dtype=dtype))
        (m, s, *sums), _ = jax.lax.scan(jax.checkpoint(body), tuple(init), chunks)
        lse = jnp.where(jnp.isfinite(m), m, 0.0) + jnp.log(s)
        mean = einx.divide("m d, m -> m d", sums[0], s)[0] if order >= 1 else None
        second = einx.divide("m d e, m -> m d e", sums[1], s)[0] if order >= 2 else None
        return lse[0], mean, second

    return jax.lax.map(one_query, (Xq, exclude), batch_size=bs)


def _has_duplicates(X: Float[Array, "N D"]) -> Bool[Array, ""]:
    """Whether two rows of ``X`` are exactly equal (lexicographic sort)."""
    order = jnp.lexsort(rearrange(X, "n d -> d n"))
    Xs = X[order]
    return jnp.any(reduce(Xs[1:] == Xs[:-1], "n d -> n", "all"))


# -- the estimator ------------------------------------------------------------------


class KernelDensity(eqx.Module):
    r"""Weighted Parzen density $\hat p(x) = \frac1c\sum_i\beta_i k(x, x_i)$.

    Johnson et al. (2020), §5.1 Eq 18 and the weighted form below it; Parzen
    (1962), Rosenblatt (1956). The weights $\beta_i \ge 0$ are normalised in
    `fit` to sum to 1 (uniform $1/N$ by default), and
    $c = \int k(x, 0)\,dx = S(0)$ is the kernel's spectral density at zero
    (`spectral_density`), so the variance $\sigma^2$ cancels. Every method
    takes ``(M, D)`` and returns per-row values, so `log_density` is a KD1
    scalar field: ``kl.gradient(kde.log_density, X)`` works unchanged.

    **Closed forms (RBF).** With $\Lambda = \operatorname{diag}(\ell_d^2)$,
    posterior weights $\pi_i(x) = \operatorname{softmax}_i(\log\beta_i
    + \log k(x, x_i))$, $m(x) = \sum_i\pi_ix_i$ and
    $C(x) = \sum_i\pi_i(x_i - m)(x_i - m)^\top$,

    $$
    s(x) = \nabla\log\hat p = \Lambda^{-1}(m(x) - x), \qquad
    \nabla^2\log\hat p = \Lambda^{-1}C(x)\Lambda^{-1} - \Lambda^{-1}.
    $$

    $m(x) - x$ is the mean-shift vector. Because $\pi$ is a softmax, $s$
    stays finite far from the data, where it tends to
    $\Lambda^{-1}(x_{\rm nearest} - x)$. Other kernels use KD1's `gradient`
    and `hessian` of `log_density` (the Hessian refuses kernels that are not
    twice differentiable at $x = x_i$, such as Matérn-½). Always,
    $\nabla\hat p = \hat p\, s$ and
    $\nabla^2\hat p = \hat p\,(\nabla^2\log\hat p + s s^\top)$.

    **Normalisation.** ``normalize=True`` needs an
    `AbstractStationaryKernel` with a finite, positive $S(0)$: RBF, Matérn,
    and ``RationalQuadratic`` with $\alpha > D/2$. ``normalize=False`` sets
    $c = 1$: `log_density` is $\log\sum_i\beta_ik(x, x_i)$, the kernel mean
    embedding at $x$, a log-density up to an additive constant, and the
    score and Hessians are unchanged. The kernel must be non-negative;
    `fit` checks $k(x_i, x_i) > 0$.

    **Bandwidth.** ``None`` keeps the kernel's lengthscale, a number sets it,
    and a rule selects it in `fit`: ``"scott"`` / ``"silverman"``
    (`estimate_lengthscale`; with weights, Kish's
    $n_{\rm eff} = 1/\sum_i\beta_i^2$ and the weighted standard deviation, as
    ``scipy.stats.gaussian_kde``), or ``"loo"``, which maximises the
    leave-one-out likelihood (Duin 1976)

    $$
    \mathcal L(\ell) = \sum_i\beta_i\log\hat p_{-i}(x_i), \qquad
    \hat p_{-i}(x_i) = \frac{\sum_{j\ne i}\beta_jk_\ell(x_i, x_j)}
        {c_\ell\sum_{j\ne i}\beta_j},
    $$

    over $\theta = \log\ell$ (scalar, or ``(D,)`` for an ARD kernel) with
    BFGS from Scott's value. ARD is used when the kernel's lengthscale is
    ``(D,)``. ``"loo"`` always uses the true normaliser $c_\ell$ (without it
    $\mathcal L$ grows with $\ell$), and raises on duplicated training rows,
    where $\mathcal L \to \infty$ as $\ell \to 0$.

    **Naming.** `score` is $\nabla_x\log\hat p$ (the score-matching sense),
    not scikit-learn's ``score``; the adapter
    `kernellib.sklearn.KernelDensity` keeps scikit-learn's meaning.

    Attributes:
        kernel: Non-negative kernel; after `fit`, with the selected
            lengthscale. Default `RBF`.
        bandwidth: ``None`` (keep ``kernel.lengthscale``), a lengthscale
            (scalar or ``(D,)``), or ``"scott"``, ``"silverman"``, ``"loo"``.
        normalize: Divide by $c = \int k$ (default), or use $c = 1$.
        batch_size: ``None`` for dense evaluation, or the chunk size of both
            the query and the training axis (memory
            $O(\text{batch\_size}^2)$ per step).
        loo_max_iter: BFGS iterations for ``bandwidth="loo"``.
        X_train: Training points ``(N, D)``; ``None`` before `fit`.
        weights: $\beta$ ``(N,)``, non-negative, summing to 1.
        log_normalizer: $\log c$ (0 with ``normalize=False``).

    Examples:
        >>> import jax
        >>> import jax.numpy as jnp
        >>> import kernellib as kl
        >>> X = jax.random.normal(jax.random.key(0), (200, 2))
        >>> kde = kl.KernelDensity(kl.RBF(), bandwidth="scott").fit(X)
        >>> kde.log_density(X[:3]).shape, kde.score(X[:3]).shape
        ((3,), (3, 2))
        >>> kde.hessian_log_density(X[:3]).shape
        (3, 2, 2)
        >>> from kernellib._einx import rearrange
        >>> grid = jnp.linspace(-8.0, 8.0, 801)
        >>> kde_1d = kl.KernelDensity(kl.RBF(lengthscale=0.5)).fit(X[:, :1])
        >>> p = kde_1d.density(rearrange(grid, "g -> g 1"))
        >>> round(float(jnp.sum(p) * (grid[1] - grid[0])), 3)  # integrates to 1
        1.0
    """

    kernel: AbstractKernel = eqx.field(default_factory=RBF)
    bandwidth: Bandwidth = None
    normalize: bool = eqx.field(default=True, static=True)
    batch_size: int | None = eqx.field(default=None, static=True)
    loo_max_iter: int = eqx.field(default=100, static=True)
    X_train: Float[Array, "N D"] | None = None
    weights: Float[Array, " N"] | None = None
    log_normalizer: Float[Array, ""] | None = None

    def __check_init__(self) -> None:
        if self.batch_size is not None and self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}.")
        if self.loo_max_iter < 1:
            raise ValueError(f"loo_max_iter must be >= 1, got {self.loo_max_iter}.")
        if isinstance(self.bandwidth, str) and self.bandwidth not in (
            "scott",
            "silverman",
            "loo",
        ):
            raise ValueError(
                "bandwidth must be None, a lengthscale, 'scott', 'silverman' or "
                f"'loo'; got {self.bandwidth!r}."
            )

    # -- fitting --------------------------------------------------------------------

    def fit(
        self, X: Float[Array, "N D"], *, weights: Float[Array, " N"] | None = None
    ) -> KernelDensity:
        r"""Store the data and weights, select the bandwidth, compute $\log c$.

        Args:
            X: Training points ``(N, D)``.
            weights: Non-negative weights $\beta$ ``(N,)`` with a positive
                sum, normalised here to sum to 1; ``None`` is uniform. Integer
                weights are equivalent to duplicating rows.

        Returns:
            A fitted copy.

        Raises:
            ValueError: On a bad shape, negative / non-finite / all-zero
                weights (an ``equinox`` runtime error under ``jit``), a kernel
                that is not integrable with ``normalize=True``, a kernel with
                $k(x_i, x_i) \le 0$, a bandwidth rule on a kernel without a
                lengthscale, or ``"loo"`` with fewer than two points or
                duplicated rows.
        """
        X = jnp.asarray(X)
        if X.ndim != 2 or X.shape[0] < 1:
            raise ValueError(f"X must be (N, D) with N >= 1, got shape {X.shape}.")
        n, d = X.shape
        if weights is None:
            beta = jnp.full((n,), 1.0 / n, dtype=X.dtype)
        else:
            w = jnp.asarray(weights, dtype=X.dtype)
            if w.shape != (n,):
                raise ValueError(f"weights must have shape ({n},), got {w.shape}.")
            w = _require(jnp.all(jnp.isfinite(w)), "weights must be finite.", w)
            w = _require(jnp.all(w >= 0), "weights must be non-negative.", w)
            w = _require(jnp.sum(w) > 0, "weights must have a positive sum.", w)
            beta = w / jnp.sum(w)

        kernel = self._select_bandwidth(X, weights, beta)
        if self.normalize:
            log_c = _log_normalizer(kernel, d)
        else:
            log_c = jnp.zeros((), dtype=X.dtype)
        X = _require(
            jnp.all(kernel.diag(X) > 0),
            f"KernelDensity needs a non-negative kernel with k(x, x) > 0; "
            f"{type(kernel).__name__} gives k(x_i, x_i) <= 0.",
            X,
        )
        return dataclasses.replace(
            self, kernel=kernel, X_train=X, weights=beta, log_normalizer=log_c
        )

    def _select_bandwidth(
        self,
        X: Float[Array, "N D"],
        weights: Float[Array, " N"] | None,
        beta: Float[Array, " N"],
    ) -> AbstractKernel:
        bw, kernel = self.bandwidth, self.kernel
        if bw is None:
            return kernel
        if "lengthscale" not in {f.name for f in dataclasses.fields(kernel)}:
            raise ValueError(
                f"bandwidth={bw!r} sets a lengthscale, but {type(kernel).__name__} "
                "has none; use bandwidth=None."
            )
        ard = jnp.ndim(kernel.lengthscale) == 1  # ty: ignore[unresolved-attribute]
        if isinstance(bw, str) and X.shape[0] < 2:
            raise ValueError(
                f"bandwidth={bw!r} needs at least two points; got n_samples = "
                f"{X.shape[0]}."
            )
        if bw in ("scott", "silverman"):
            ell = _weighted_rule_lengthscale(
                X,
                bw,  # ty: ignore[invalid-argument-type]
                None if weights is None else beta,
                ard=ard,
            )
        elif bw == "loo":
            ell = self._loo_lengthscale(X, beta, weights is None, ard)
        else:
            ell = jnp.asarray(bw)
        return dataclasses.replace(kernel, lengthscale=ell)

    def _loo_lengthscale(
        self, X: Float[Array, "N D"], beta: Float[Array, " N"], uniform: bool, ard: bool
    ) -> Float[Array, ""] | Float[Array, " D"]:
        kernel = self.kernel
        d = X.shape[1]
        if not isinstance(kernel, AbstractStationaryKernel):
            raise ValueError(
                "bandwidth='loo' needs an AbstractStationaryKernel (the LOO "
                f"likelihood needs the kernel's integral); got {type(kernel).__name__}."
            )
        X = _require(
            ~_has_duplicates(X),
            "bandwidth='loo' on data with duplicated rows: the leave-one-out "
            "likelihood grows without bound as the lengthscale goes to 0. Use "
            "bandwidth='scott' or 'silverman', or de-duplicate (merging "
            "duplicates into weights).",
            X,
        )
        ell0 = _weighted_rule_lengthscale(
            X, "scott", None if uniform else beta, ard=ard
        )
        theta0 = jnp.log(jnp.atleast_1d(ell0))
        log_beta = jnp.log(beta)

        def objective(theta: Float[Array, " P"]) -> Float[Array, ""]:
            ell = jnp.exp(theta if ard else theta[0])
            k = dataclasses.replace(kernel, lengthscale=ell)
            log_c = _log_normalizer(k, d, check=False)
            return -_loo(k, X, beta, log_beta, log_c, self.batch_size)

        gtol = 1e-10 if theta0.dtype == jnp.float64 else 1e-5
        result = minimize(
            objective,
            theta0,
            method="BFGS",
            options={"maxiter": self.loo_max_iter, "gtol": gtol},
        )
        try:
            failed, nit = not bool(result.success), int(result.nit)
        except jax.errors.ConcretizationTypeError:  # under jit: no warning
            failed, nit = False, 0
        if failed:
            warnings.warn(
                f"bandwidth='loo': BFGS stopped after {nit} iterations "
                f"(loo_max_iter={self.loo_max_iter}) without meeting its "
                f"gradient tolerance (status {int(result.status)}); the "
                "lengthscale may not maximise the LOO likelihood.",
                RuntimeWarning,
                stacklevel=4,
            )
        ell = jnp.exp(result.x)
        return ell if ard else ell[0]

    # -- fitted state -----------------------------------------------------------------

    def _fitted(self) -> tuple[Float[Array, "N D"], Float[Array, " N"], Array]:
        if self.X_train is None or self.weights is None or self.log_normalizer is None:
            raise RuntimeError("KernelDensity is not fitted; call fit(X) first.")
        return self.X_train, self.weights, self.log_normalizer

    def _log_k(self, X1: Float[Array, "M D"], X2: Float[Array, "B D"]) -> Array:
        return _log_kernel(self.kernel, X1, X2)

    @property
    def lengthscale(self) -> Float[Array, ""] | Float[Array, " D"]:
        """The kernel's lengthscale: after `fit`, the selected bandwidth."""
        return self.kernel.lengthscale  # ty: ignore[unresolved-attribute]

    # -- densities -------------------------------------------------------------------

    def log_density(self, X: Float[Array, "M D"]) -> Float[Array, " M"]:
        r"""$\log\hat p(x) = \operatorname{logsumexp}_i(\log\beta_i
        + \log k(x, x_i)) - \log c$, finite far from the data.

        Examples:
            >>> import jax.numpy as jnp
            >>> import kernellib as kl
            >>> kde = kl.KernelDensity().fit(jnp.zeros((1, 1)))
            >>> log_p = kde.log_density(jnp.array([[100.0]]))  # p underflows to 0
            >>> round(float(log_p[0]), 2)  # -100**2 / 2 - log(sqrt(2 pi))
            -5000.92
        """
        X_train, beta, log_c = self._fitted()
        lse, _, _ = _moments(
            self._log_k,
            X,
            X_train,
            jnp.log(beta),
            order=0,
            batch_size=self.batch_size,
        )
        return lse - log_c

    def density(self, X: Float[Array, "M D"]) -> Float[Array, " M"]:
        r"""$\hat p(x) = \exp\log\hat p(x)$."""
        return jnp.exp(self.log_density(X))

    def log_likelihood(self, X: Float[Array, "M D"]) -> Float[Array, ""]:
        r"""Mean log-density $\frac1M\sum_m\log\hat p(x_m)$ (differentiable)."""
        return jnp.mean(self.log_density(X))

    def loo_log_likelihood(self) -> Float[Array, ""]:
        r"""$\sum_i\beta_i\log\hat p_{-i}(x_i)$, the leave-one-out log-likelihood.

        $\hat p_{-i}$ is the estimate refitted without $x_i$ (its weights
        renormalised by $1 - \beta_i$), evaluated in log space with the
        $i$-th term masked to $-\infty$. With ``normalize=False`` it uses
        $c = 1$ like `log_density`. Smooth in the lengthscale, so
        ``jax.grad`` through `fit` and this method works.

        Raises:
            ValueError: With fewer than two training points.
        """
        X_train, beta, log_c = self._fitted()
        if X_train.shape[0] < 2:
            raise ValueError("loo_log_likelihood needs at least two training points.")
        return _loo(self.kernel, X_train, beta, jnp.log(beta), log_c, self.batch_size)

    # -- derivatives -----------------------------------------------------------------

    def _gaussian(
        self, X: Float[Array, "M D"], order: int
    ) -> tuple[Array, Array, Array | None]:
        """Closed-form RBF log-density, score and (``order=2``) Hessian."""
        X_train, beta, log_c = self._fitted()
        center = einsum(beta, X_train, "n, n d -> d")
        lse, mean, second = _moments(
            self._log_k,
            X,
            X_train,
            jnp.log(beta),
            order=order,
            batch_size=self.batch_size,
            center=center,
        )
        assert mean is not None
        inv = 1.0 / self.kernel._lengthscale_vector(X.shape[1]) ** 2  # ty: ignore[unresolved-attribute]
        shift = einx.add("m d, d -> m d", mean, center) - X  # m(x) - x
        score = einx.multiply("m d, d -> m d", shift, inv)
        if order < 2:
            return lse - log_c, score, None
        assert second is not None
        C = second - einx.multiply("m d, m e -> m d e", mean, mean)
        H = einx.multiply("m d e, d, e -> m d e", C, inv, inv)
        return (
            lse - log_c,
            score,
            einx.subtract("m d e, d e -> m d e", H, jnp.diag(inv)),
        )

    def score(self, X: Float[Array, "M D"]) -> Float[Array, "M D"]:
        r"""$\nabla_x\log\hat p(x)$ (closed form for RBF, else KD1 `gradient`).

        Examples:
            >>> import jax.numpy as jnp
            >>> import kernellib as kl
            >>> kde = kl.KernelDensity(kl.RBF(lengthscale=2.0)).fit(jnp.zeros((1, 2)))
            >>> kde.score(jnp.array([[1.0, -4.0]])).tolist()  # (x_1 - x) / l^2
            [[-0.25, 1.0]]
        """
        if isinstance(self.kernel, RBF):
            return self._gaussian(X, 1)[1]
        return gradient(self.log_density, X, batch_size=self.batch_size, check=False)

    def hessian_log_density(self, X: Float[Array, "M D"]) -> Float[Array, "M D D"]:
        r"""$\nabla_x^2\log\hat p(x)$ (closed form for RBF, else KD1 `hessian`).

        Raises:
            ValueError: For a kernel that is not twice differentiable at
                $x = x_i$ (KD1's guard), such as ``Matern(nu=0.5)``.
        """
        if isinstance(self.kernel, RBF):
            H = self._gaussian(X, 2)[2]
            assert H is not None
            return H
        return hessian(self.log_density, X, batch_size=self.batch_size)

    def density_gradient(self, X: Float[Array, "M D"]) -> Float[Array, "M D"]:
        r"""$\nabla\hat p = \hat p\, \nabla\log\hat p$."""
        if isinstance(self.kernel, RBF):
            log_p, s, _ = self._gaussian(X, 1)
        else:
            log_p, s = self.log_density(X), self.score(X)
        return einx.multiply("m, m d -> m d", jnp.exp(log_p), s)

    def density_hessian(self, X: Float[Array, "M D"]) -> Float[Array, "M D D"]:
        r"""$\nabla^2\hat p = \hat p\,(\nabla^2\log\hat p + s s^\top)$."""
        if isinstance(self.kernel, RBF):
            log_p, s, H = self._gaussian(X, 2)
            assert H is not None
        else:
            log_p, s, H = (
                self.log_density(X),
                self.score(X),
                self.hessian_log_density(X),
            )
        outer = einx.multiply("m d, m e -> m d e", s, s)
        return einx.multiply("m, m d e -> m d e", jnp.exp(log_p), H + outer)

    # -- sampling --------------------------------------------------------------------

    def sample(self, key: PRNGKeyArray, n: int) -> Float[Array, "n D"]:
        r"""Draw ``n`` points: $i \sim \operatorname{Cat}(\beta)$,
        $x = x_i + \Lambda^{1/2}\varepsilon$, $\varepsilon \sim \mathcal N(0, I)$.

        Raises:
            NotImplementedError: For a kernel other than `RBF`.

        Examples:
            >>> import jax
            >>> import jax.numpy as jnp
            >>> import kernellib as kl
            >>> kde = kl.KernelDensity().fit(jnp.zeros((5, 3)))
            >>> kde.sample(jax.random.key(0), 4).shape
            (4, 3)
        """
        X_train, beta, _ = self._fitted()
        if not isinstance(self.kernel, RBF):
            raise NotImplementedError(
                "sample supports the RBF kernel only, got "
                f"{type(self.kernel).__name__}."
            )
        key_i, key_e = jax.random.split(key)
        idx = jax.random.categorical(key_i, jnp.log(beta), shape=(n,))
        eps = jax.random.normal(key_e, (n, X_train.shape[1]), dtype=X_train.dtype)
        ell = self.kernel._lengthscale_vector(X_train.shape[1])
        return X_train[idx] + einx.multiply("n d, d -> n d", eps, ell)


def _loo(
    kernel: AbstractKernel,
    X: Float[Array, "N D"],
    beta: Float[Array, " N"],
    log_beta: Float[Array, " N"],
    log_c: Float[Array, ""],
    batch_size: int | None,
) -> Float[Array, ""]:
    r"""$\sum_i\beta_i[\operatorname{lse}_{j\ne i}(\log\beta_j + \log k_{ij})
    - \log(1 - \beta_i) - \log c]$."""
    lse, _, _ = _moments(
        lambda A, B: _log_kernel(kernel, A, B),
        X,
        X,
        log_beta,
        order=0,
        batch_size=batch_size,
        exclude=jnp.arange(X.shape[0]),
    )
    return einsum(beta, lse - jnp.log1p(-beta) - log_c, "n, n ->")
