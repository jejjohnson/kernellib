"""KD6: KernelDensity, the weighted Parzen estimator, its score / Hessian and
bandwidth selection."""

from __future__ import annotations

import math

import einx
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.stats
from sklearn.neighbors import KernelDensity as SkKernelDensity

import kernellib as kl
from kernellib._density._kde import _log_normalizer
from kernellib._einx import einsum, reduce
from kernellib._heuristics import _weighted_rule_lengthscale


def _data(n=40, d=2, seed=0):
    return jax.random.normal(jax.random.key(seed), (n, d))


def _weights(n, seed=1):
    return jax.random.uniform(jax.random.key(seed), (n,), minval=0.1, maxval=2.0)


# -- the log-kernel hook -------------------------------------------------------

LOG_SHAPE_KERNELS = [
    pytest.param(kl.RBF(), id="rbf"),
    pytest.param(kl.Matern(nu=0.5), id="matern12"),
    pytest.param(kl.Matern(nu=1.5), id="matern32"),
    pytest.param(kl.Matern(nu=2.5), id="matern52"),
    pytest.param(kl.RationalQuadratic(alpha=0.7), id="rq0.7"),
    pytest.param(kl.RationalQuadratic(alpha=3.0), id="rq3"),
]


@pytest.mark.parametrize("kernel", LOG_SHAPE_KERNELS)
def test_log_shape_matches_log_of_shape(kernel):
    r2 = jnp.array([1e-12, 1e-9, 1e-6, 0.1, 1.0, 10.0, 50.0])
    np.testing.assert_allclose(
        kernel._log_shape(r2), jnp.log(kernel.shape(r2)), rtol=1e-12, atol=1e-14
    )


@pytest.mark.parametrize("kernel", LOG_SHAPE_KERNELS)
def test_log_shape_is_finite_where_shape_underflows(kernel):
    r2 = jnp.array([1e8])
    assert bool(jnp.isfinite(kernel._log_shape(r2))[0])


# -- agreement with scikit-learn -----------------------------------------------


@pytest.mark.parametrize("d", [1, 3])
@pytest.mark.parametrize("weighted", [False, True])
def test_gaussian_log_density_matches_sklearn(d, weighted):
    X, Q = _data(60, d), _data(25, d, seed=5)
    h = 0.4
    w = _weights(60) if weighted else None
    ours = kl.KernelDensity(kl.RBF(lengthscale=h)).fit(X, weights=w).log_density(Q)
    ref = SkKernelDensity(kernel="gaussian", bandwidth=h).fit(
        np.asarray(X), sample_weight=None if w is None else np.asarray(w)
    )
    np.testing.assert_allclose(ours, ref.score_samples(np.asarray(Q)), atol=1e-10)


# -- normalisation -------------------------------------------------------------


@pytest.mark.parametrize(
    ("kernel", "half_width"),
    [
        (kl.RBF(lengthscale=0.5), 40.0),
        (kl.Matern(nu=1.5, lengthscale=1.0), 60.0),
        # Tails ~ r^-4: the mass beyond |x| = 200 is ~1e-8.
        (kl.RationalQuadratic(alpha=2.0, lengthscale=0.5), 200.0),
    ],
)
def test_integrates_to_one_1d(kernel, half_width):
    X = _data(20, 1)
    grid = jnp.linspace(-half_width, half_width, 4001)
    p = kl.KernelDensity(kernel).fit(X).density(einx.id("g -> g 1", grid))
    total = jax.scipy.integrate.trapezoid(p, grid)
    assert abs(float(total) - 1.0) < 1e-6


def test_integrates_to_one_2d():
    X = _data(15, 2)
    g = jnp.linspace(-8.0, 8.0, 401)
    G = jnp.stack(jnp.meshgrid(g, g, indexing="ij"), axis=-1)
    p = (
        kl.KernelDensity(kl.RBF(lengthscale=0.6))
        .fit(X)
        .density(einx.id("a b d -> (a b) d", G))
    )
    total = reduce(p, "n -> ", "sum") * (g[1] - g[0]) ** 2
    assert abs(float(total) - 1.0) < 1e-4


def _closed_log_c(kernel, d):
    ell = np.broadcast_to(np.asarray(kernel.lengthscale), (d,))
    log_c = math.log(float(kernel.variance)) + float(np.sum(np.log(ell)))
    if isinstance(kernel, kl.RBF):
        return log_c + (d / 2) * math.log(2 * math.pi)
    if isinstance(kernel, kl.Matern):
        nu = kernel.nu
        return (
            log_c
            + (d / 2) * math.log(2 * math.pi / nu)
            + math.lgamma(nu + d / 2)
            - math.lgamma(nu)
        )
    a = float(kernel.alpha)
    return (
        log_c
        + (d / 2) * math.log(2 * math.pi * a)
        + math.lgamma(a - d / 2)
        - (math.lgamma(a))
    )


NORMALIZER_KERNELS = [
    pytest.param(kl.RBF(lengthscale=0.7, variance=1.7), id="rbf"),
    pytest.param(kl.Matern(nu=0.5, lengthscale=1.3, variance=0.4), id="matern12"),
    pytest.param(kl.Matern(nu=1.5, lengthscale=0.3), id="matern32"),
    pytest.param(kl.Matern(nu=2.5, lengthscale=2.0, variance=3.0), id="matern52"),
    pytest.param(kl.RationalQuadratic(alpha=1.5, lengthscale=0.8), id="rq1.5"),
    pytest.param(kl.RationalQuadratic(alpha=4.0, variance=2.0), id="rq4"),
]


@pytest.mark.parametrize("d", [1, 2, 3])
@pytest.mark.parametrize("kernel", NORMALIZER_KERNELS)
def test_log_normalizer_closed_forms(kernel, d):
    if isinstance(kernel, kl.RationalQuadratic) and float(kernel.alpha) <= d / 2:
        with pytest.raises(ValueError, match="alpha > D / 2"):
            _log_normalizer(kernel, d)
        return
    np.testing.assert_allclose(
        np.exp(_log_normalizer(kernel, d)),
        np.exp(_closed_log_c(kernel, d)),
        rtol=1e-12,
    )


@pytest.mark.parametrize("d", [1, 2, 3])
def test_log_normalizer_ard(d):
    kernel = kl.RBF(lengthscale=jnp.linspace(0.5, 2.0, d), variance=0.3)
    np.testing.assert_allclose(
        np.exp(_log_normalizer(kernel, d)),
        np.exp(_closed_log_c(kernel, d)),
        rtol=1e-12,
    )


@pytest.mark.parametrize(
    "kernel",
    [kl.Linear(), kl.Periodic(), kl.RationalQuadratic(alpha=1.0)],
    ids=["linear", "periodic", "rq-alpha<=D/2"],
)
def test_normalize_true_rejects_non_integrable_kernels(kernel):
    with pytest.raises(ValueError, match="normalize=False"):
        kl.KernelDensity(kernel).fit(_data(10, 2) + 3.0)


@pytest.mark.slow
@pytest.mark.parametrize(
    "kernel", [kl.RBF(lengthscale=0.5), kl.Matern(nu=2.5)], ids=["rbf", "matern52"]
)
def test_normalize_false_is_the_mean_embedding(kernel):
    X, Q, w = _data(), _data(12, seed=3), _weights(40)
    plain = kl.KernelDensity(kernel, normalize=False).fit(X, weights=w)
    normed = kl.KernelDensity(kernel).fit(X, weights=w)
    beta = w / jnp.sum(w)
    np.testing.assert_allclose(
        plain.log_density(Q), jnp.log(kernel(Q, X) @ beta), rtol=1e-12
    )
    assert float(plain.log_normalizer) == 0.0
    np.testing.assert_allclose(plain.score(Q), normed.score(Q), atol=1e-12)


# -- score and Hessian ---------------------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize(
    "lengthscale", [0.6, jnp.array([0.4, 1.1])], ids=["isotropic", "ard"]
)
def test_gaussian_closed_forms_match_kd1_autodiff(lengthscale):
    X, Q = _data(), _data(50, seed=7) * 1.5
    w = _weights(40)
    kde = kl.KernelDensity(kl.RBF(lengthscale=lengthscale)).fit(X, weights=w)
    np.testing.assert_allclose(
        kde.score(Q), kl.gradient(kde.log_density, Q), atol=1e-10
    )
    np.testing.assert_allclose(
        kde.hessian_log_density(Q), kl.hessian(kde.log_density, Q), atol=1e-10
    )


@pytest.mark.slow
@pytest.mark.parametrize(
    "kernel",
    [kl.Matern(nu=2.5, lengthscale=0.7), kl.RationalQuadratic(alpha=2.0)],
    ids=["matern52", "rq"],
)
def test_autodiff_score_matches_finite_differences(kernel):
    X, Q = _data(), _data(10, seed=4)
    kde = kl.KernelDensity(kernel).fit(X)
    h = 1e-5
    eye = jnp.eye(2)
    fd = jnp.stack(
        [
            (kde.log_density(Q + h * e) - kde.log_density(Q - h * e)) / (2 * h)
            for e in eye
        ],
        axis=-1,
    )
    np.testing.assert_allclose(kde.score(Q), fd, atol=1e-6)


@pytest.mark.slow
@pytest.mark.parametrize(
    "kernel", [kl.RBF(lengthscale=0.5), kl.Matern(nu=2.5)], ids=["rbf", "matern52"]
)
def test_density_derivative_identities(kernel):
    X, Q = _data(), _data(8, seed=2)
    kde = kl.KernelDensity(kernel).fit(X, weights=_weights(40))
    p, s, H = kde.density(Q), kde.score(Q), kde.hessian_log_density(Q)
    np.testing.assert_allclose(
        kde.density_gradient(Q), einx.multiply("m, m d -> m d", p, s), atol=1e-12
    )
    expected = einx.multiply(
        "m, m d e -> m d e", p, H + einx.multiply("m d, m e -> m d e", s, s)
    )
    np.testing.assert_allclose(kde.density_hessian(Q), expected, atol=1e-12)


@pytest.mark.slow
def test_gaussian_density_hessian_matches_jax_hessian():
    X, Q = _data(), _data(6, seed=2)
    kde = kl.KernelDensity(kl.RBF(lengthscale=jnp.array([0.5, 0.8]))).fit(X)

    def density_at(x):
        return kde.density(einx.id("d -> 1 d", x))[0]

    ref = jax.vmap(jax.hessian(density_at))(Q)
    np.testing.assert_allclose(kde.density_hessian(Q), ref, atol=1e-10)


@pytest.mark.parametrize("dtype", [jnp.float64, jnp.float32])
def test_score_is_finite_far_from_the_data(dtype):
    ell = jnp.array([0.5, 0.8], dtype=dtype)
    # Well-separated points: along each query direction the nearest one leads
    # the next by ~200 in squared scaled distance, so it carries the softmax.
    X = jnp.array([[0.0, 0.0], [1.0, 0.5], [-1.0, 2.0], [2.0, -1.0]], dtype=dtype)
    kde = kl.KernelDensity(kl.RBF(lengthscale=ell)).fit(X)
    # 50 lengthscales beyond every training point, along +x, -x and +y.
    reach = reduce(jnp.abs(X), "n d -> d", "max") + 50.0 * ell
    Q = jnp.stack([reach, -reach, reach]) * jnp.array(
        [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=dtype
    )
    log_p, s = kde.log_density(Q), kde.score(Q)
    assert bool(jnp.all(jnp.isfinite(log_p)))
    assert bool(jnp.all(jnp.isfinite(s)))
    diff = einx.divide(
        "m n d, d -> m n d", einx.subtract("m d, n d -> m n d", Q, X), ell
    )
    d2 = reduce(diff**2, "m n d -> m n", "sum")
    nearest = X[einx.id("m 1 -> m", einx.argmin("m [n]", d2))]
    expected = einx.divide("m d, d -> m d", nearest - Q, ell**2)
    rtol = 1e-8 if dtype == jnp.float64 else 1e-4
    np.testing.assert_allclose(s, expected, rtol=rtol)


def test_single_point_score_and_hessian_are_exact():
    ell = jnp.array([0.5, 2.0])
    x1 = jnp.array([[0.3, -1.2]])
    kde = kl.KernelDensity(kl.RBF(lengthscale=ell)).fit(x1)
    Q = _data(9, seed=8) * 3.0
    np.testing.assert_allclose(
        kde.score(Q), einx.divide("m d, d -> m d", x1 - Q, ell**2), atol=1e-14
    )
    H = kde.hessian_log_density(Q)
    np.testing.assert_allclose(
        H, einx.id("d e -> m d e", -jnp.diag(1 / ell**2), m=9), atol=1e-14
    )


@pytest.mark.slow
def test_kd1_operators_see_a_scalar_field():
    kde = kl.KernelDensity(kl.Matern(nu=0.5)).fit(_data())
    Q = _data(5, seed=9)
    assert kde.score(Q).shape == (5, 2)
    with pytest.raises(ValueError, match="Matern"):
        kde.hessian_log_density(Q)


# -- weights -------------------------------------------------------------------


def test_weights_none_equals_uniform():
    X, Q = _data(), _data(10, seed=3)
    a = kl.KernelDensity(kl.RBF(lengthscale=0.5)).fit(X)
    b = kl.KernelDensity(kl.RBF(lengthscale=0.5)).fit(X, weights=jnp.full(40, 3.0))
    np.testing.assert_allclose(a.log_density(Q), b.log_density(Q), atol=1e-12)
    np.testing.assert_allclose(b.weights, jnp.full(40, 1 / 40), atol=1e-15)


def test_integer_weights_equal_duplicated_rows():
    X, Q = _data(), _data(10, seed=3)
    w = jnp.ones(40).at[0].set(2.0)
    weighted = kl.KernelDensity(kl.RBF(lengthscale=0.5)).fit(X, weights=w)
    duplicated = kl.KernelDensity(kl.RBF(lengthscale=0.5)).fit(
        jnp.concatenate([X, X[:1]])
    )
    np.testing.assert_allclose(
        weighted.log_density(Q), duplicated.log_density(Q), atol=1e-12
    )


@pytest.mark.parametrize(
    ("w", "match"),
    [
        (jnp.ones(40).at[3].set(-1.0), "non-negative"),
        (jnp.zeros(40), "positive sum"),
        (jnp.ones(40).at[0].set(jnp.nan), "finite"),
    ],
    ids=["negative", "all-zero", "nan"],
)
def test_bad_weights_raise(w, match):
    X = _data()
    with pytest.raises(ValueError, match=match):
        kl.KernelDensity().fit(X, weights=w)
    fit = eqx.filter_jit(lambda kde, X, w: kde.fit(X, weights=w))
    with pytest.raises(eqx.EquinoxRuntimeError, match=match):
        fit(kl.KernelDensity(), X, w)


def test_weights_shape_is_checked():
    with pytest.raises(ValueError, match="shape"):
        kl.KernelDensity().fit(_data(), weights=jnp.ones(3))


# -- chunked path --------------------------------------------------------------


@pytest.mark.slow
def test_chunked_equals_dense():
    X = _data(1000, 2)
    Q = _data(300, 2, seed=11) * 2.0
    w = _weights(1000)
    kernel = kl.RBF(lengthscale=jnp.array([0.3, 0.5]))
    dense = kl.KernelDensity(kernel).fit(X, weights=w)
    chunked = kl.KernelDensity(kernel, batch_size=64).fit(X, weights=w)
    np.testing.assert_allclose(chunked.log_density(Q), dense.log_density(Q), atol=1e-12)
    np.testing.assert_allclose(chunked.score(Q), dense.score(Q), atol=1e-12)
    np.testing.assert_allclose(
        chunked.hessian_log_density(Q), dense.hessian_log_density(Q), atol=1e-12
    )


@pytest.mark.slow
def test_chunked_generic_kernel_equals_dense():
    X, Q = _data(70), _data(13, seed=2)
    kernel = kl.Matern(nu=2.5, lengthscale=0.6)
    dense = kl.KernelDensity(kernel).fit(X)
    chunked = kl.KernelDensity(kernel, batch_size=16).fit(X)
    np.testing.assert_allclose(chunked.log_density(Q), dense.log_density(Q), atol=1e-12)
    np.testing.assert_allclose(chunked.score(Q), dense.score(Q), atol=1e-12)
    np.testing.assert_allclose(
        chunked.loo_log_likelihood(), dense.loo_log_likelihood(), atol=1e-12
    )


# -- leave-one-out -------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize("kernel", [kl.RBF(lengthscale=0.4), kl.Matern(nu=1.5)])
def test_loo_log_likelihood_matches_brute_force(kernel):
    X, w = _data(25), _weights(25)
    kde = kl.KernelDensity(kernel).fit(X, weights=w)
    beta = w / jnp.sum(w)
    brute = 0.0
    for i in range(25):
        keep = jnp.arange(25) != i
        held_out = kl.KernelDensity(kernel).fit(X[keep], weights=w[keep])
        brute += float(beta[i]) * float(held_out.log_density(X[i : i + 1])[0])
    np.testing.assert_allclose(kde.loo_log_likelihood(), brute, atol=1e-12)


@pytest.mark.slow
def test_loo_bandwidth_is_a_stationary_point_and_the_grid_argmax():
    X = jax.random.normal(jax.random.key(0), (500, 1))
    kde = kl.KernelDensity(kl.RBF(), bandwidth="loo").fit(X)
    ell = kde.lengthscale

    def loo(log_ell):
        fitted = kl.KernelDensity(kl.RBF(lengthscale=jnp.exp(log_ell))).fit(X)
        return fitted.loo_log_likelihood()

    assert abs(float(jax.grad(loo)(jnp.log(ell)))) < 1e-6
    grid = kl.lengthscale_grid(ell, decades=1.0, n_points=200)
    values = jax.vmap(lambda g: loo(jnp.log(g)))(grid)
    best = grid[jnp.argmax(values)]
    step = float(jnp.log(grid[1] / grid[0]))
    assert abs(float(jnp.log(best / ell))) <= step


def test_loo_rejects_duplicated_rows():
    X = _data(20)
    X = X.at[5].set(X[2])
    with pytest.raises(ValueError, match="duplicated"):
        kl.KernelDensity(bandwidth="loo").fit(X)


@pytest.mark.slow
def test_loo_ard_returns_one_lengthscale_per_dimension():
    X = _data(60) * jnp.array([0.2, 3.0])
    kde = kl.KernelDensity(kl.RBF(lengthscale=jnp.ones(2)), bandwidth="loo").fit(X)
    assert kde.lengthscale.shape == (2,)
    assert float(kde.lengthscale[1]) > float(kde.lengthscale[0])


# -- bandwidth rules -----------------------------------------------------------


@pytest.mark.parametrize("rule", ["scott", "silverman"])
@pytest.mark.parametrize("ard", [False, True])
def test_rule_bandwidth_equals_estimate_lengthscale(rule, ard):
    X = _data(50, 3)
    kernel = kl.RBF(lengthscale=jnp.ones(3) if ard else 1.0)
    kde = kl.KernelDensity(kernel, bandwidth=rule).fit(X)
    assert bool(jnp.all(kde.lengthscale == kl.estimate_lengthscale(X, rule, ard=ard)))
    uniform = kl.KernelDensity(kernel, bandwidth=rule).fit(X, weights=jnp.ones(50))
    np.testing.assert_allclose(uniform.lengthscale, kde.lengthscale, rtol=1e-12)


@pytest.mark.parametrize("rule", ["scott", "silverman"])
def test_weighted_rule_matches_scipy_gaussian_kde(rule):
    x, w = _data(80, 1), _weights(80)
    kde = kl.KernelDensity(bandwidth=rule).fit(x, weights=w)
    ref = scipy.stats.gaussian_kde(
        np.asarray(x[:, 0]), bw_method=rule, weights=np.asarray(w)
    )
    np.testing.assert_allclose(
        kde.lengthscale, np.sqrt(ref.covariance[0, 0]), rtol=1e-12
    )


def test_weighted_rule_multivariate_formula():
    X, w = _data(80, 3), _weights(80)
    beta = np.asarray(w / jnp.sum(w))
    n_eff = 1.0 / np.sum(beta**2)
    std = np.sqrt(np.diag(np.cov(np.asarray(X), rowvar=False, aweights=beta)))
    expected = np.mean(std) * n_eff ** (-1.0 / 7.0)
    np.testing.assert_allclose(
        _weighted_rule_lengthscale(X, "scott", w), expected, rtol=1e-12
    )


def test_numeric_bandwidth_sets_the_lengthscale():
    kde = kl.KernelDensity(kl.Matern(), bandwidth=0.3).fit(_data())
    assert float(kde.lengthscale) == pytest.approx(0.3)
    assert isinstance(kde.kernel, kl.Matern)


def test_bandwidth_needs_a_lengthscale_field():
    with pytest.raises(ValueError, match="has none"):
        kl.KernelDensity(kl.Linear(), bandwidth="scott", normalize=False).fit(_data())


def test_bad_bandwidth_string():
    with pytest.raises(ValueError, match="bandwidth"):
        kl.KernelDensity(bandwidth="median")


# -- sampling ------------------------------------------------------------------


def test_sample_moments():
    X, w = _data(20), _weights(20)
    ell = jnp.array([0.3, 0.6])
    kde = kl.KernelDensity(kl.RBF(lengthscale=ell)).fit(X, weights=w)
    draws = kde.sample(jax.random.key(0), 100_000)
    beta = kde.weights
    mean = einsum(beta, X, "n, n d -> d")
    second = einsum(beta, einx.multiply("n d, n e -> n d e", X, X), "n, n d e -> d e")
    cov = second - einx.multiply("d, e -> d e", mean, mean) + jnp.diag(ell**2)
    centred = einx.subtract("n d, d -> n d", draws, reduce(draws, "n d -> d", "mean"))
    emp_cov = einsum(centred, centred, "n d, n e -> d e") / draws.shape[0]
    # Entries of cov are O(1); the Monte Carlo standard error of a mean and a
    # covariance entry with 1e5 draws is ~sqrt(var / 1e5) <~ 5e-3.
    np.testing.assert_allclose(reduce(draws, "n d -> d", "mean"), mean, atol=1e-2)
    np.testing.assert_allclose(emp_cov, cov, atol=1e-2)


def test_sample_needs_rbf():
    kde = kl.KernelDensity(kl.Matern()).fit(_data())
    with pytest.raises(NotImplementedError, match="RBF"):
        kde.sample(jax.random.key(0), 3)


# -- transformations and errors ------------------------------------------------


def test_log_likelihood_gradient_matches_finite_differences():
    X, Q = _data(), _data(20, seed=6)

    def ll(ell):
        return kl.KernelDensity(kl.RBF(lengthscale=ell)).fit(X).log_likelihood(Q)

    ell, h = jnp.asarray(0.5), 1e-6
    fd = (ll(ell + h) - ll(ell - h)) / (2 * h)
    np.testing.assert_allclose(jax.grad(ll)(ell), fd, rtol=1e-6)


def test_fit_under_filter_jit():
    X = _data()
    jitted = eqx.filter_jit(kl.KernelDensity.fit)(
        kl.KernelDensity(bandwidth="scott"), X
    )
    eager = kl.KernelDensity(bandwidth="scott").fit(X)
    np.testing.assert_allclose(jitted.lengthscale, eager.lengthscale, rtol=1e-12)
    np.testing.assert_allclose(
        jitted.log_density(X[:3]), eager.log_density(X[:3]), atol=1e-12
    )


def test_unfitted_raises():
    with pytest.raises(RuntimeError, match="not fitted"):
        kl.KernelDensity().log_density(_data(3))
