"""Vector-field kernels (#214): `CurlFree`, `DivergenceFree`, `Helmholtz`.

Closed forms against autodiff, the 2-D rotation identity, zero divergence /
curl of the kernel columns (autodiff) and of a posterior mean (finite
differences), PSD Grams, the point-major layout and the `to_operator` →
gaussx solve path.
"""

from __future__ import annotations

import gaussx as gx
import jax
import jax.numpy as jnp
import pytest

import kernellib as kl
from kernellib._einx import einsum, rearrange, reduce
from kernellib._geo._vector import _cross_hessian


ELL, VAR = 0.6, 1.3
R = jnp.array([[0.0, 1.0], [-1.0, 0.0]])


def _points(n, d=2, key=0):
    return jax.random.uniform(jax.random.key(key), (n, d), minval=-1.0, maxval=1.0)


def _autodiff_hessian(kernel, x, y):
    return jax.jacfwd(jax.grad(kernel.pairwise, argnums=0), argnums=1)(x, y)


def _rbf_closed_forms(x, y):
    # The issue's closed forms, isotropic RBF, h = x - y.
    d = x.shape[0]
    h = x - y
    r2 = jnp.sum(h**2) / ELL**2
    c = VAR / ELL**2 * jnp.exp(-0.5 * r2)
    hh = einsum(h, h, "i, j -> i j") / ELL**2
    eye = jnp.eye(d)
    return c * (eye - hh), c * (hh + (d - 1 - r2) * eye)


# Closed form vs autodiff --------------------------------------------------


@pytest.mark.parametrize("d", [2, 3])
@pytest.mark.parametrize("ard", [False, True], ids=["iso", "ard"])
def test_rbf_closed_form_matches_autodiff(d, ard):
    ell = jnp.linspace(0.4, 1.1, d) if ard else jnp.asarray(ELL)
    k = kl.RBF(lengthscale=ell, variance=VAR)
    X = _points(4, d)
    for x, y in [(X[0], X[1]), (X[2], X[3]), (X[0], X[0])]:
        H = _autodiff_hessian(k, x, y)
        assert jnp.allclose(_cross_hessian(k, x, y), H, rtol=0, atol=1e-10)
        assert jnp.allclose(kl.CurlFree(k).matrix(x, y), H, atol=1e-10)
        df = jnp.trace(H) * jnp.eye(d) - rearrange(H, "a b -> b a")
        assert jnp.allclose(kl.DivergenceFree(k).matrix(x, y), df, atol=1e-10)


@pytest.mark.parametrize("d", [2, 3])
def test_rbf_matches_the_documented_closed_forms(d):
    k = kl.RBF(lengthscale=ELL, variance=VAR)
    X = _points(3, d, key=1)
    for x, y in [(X[0], X[1]), (X[1], X[2]), (X[2], X[2])]:
        cf, df = _rbf_closed_forms(x, y)
        assert jnp.allclose(kl.CurlFree(k).matrix(x, y), cf, atol=1e-10)
        assert jnp.allclose(kl.DivergenceFree(k).matrix(x, y), df, atol=1e-10)


@pytest.mark.slow
@pytest.mark.parametrize(
    "kernel",
    [
        kl.RBF(lengthscale=ELL, variance=VAR),
        kl.Matern(nu=2.5, lengthscale=0.8),
        kl.RationalQuadratic(lengthscale=0.7, alpha=1.5),
        kl.Gibbs(lambda x: 0.4 + 0.3 * jax.nn.sigmoid(3.0 * x[0])),
    ],
    ids=["rbf", "matern52", "rq", "gibbs"],
)
def test_divergence_free_2d_is_the_rotated_curl_free(kernel):
    # R H Rᵀ = tr(H) I - Hᵀ, including the nonstationary case where H ≠ Hᵀ.
    X = _points(3, key=2)
    for x, y in [(X[0], X[1]), (X[1], X[2])]:
        H = kl.CurlFree(kernel).matrix(x, y)
        rotated = einsum(R, H, R, "i a, a b, j b -> i j")
        assert jnp.allclose(kl.DivergenceFree(kernel).matrix(x, y), rotated)


# Invariants: zero divergence / curl of the kernel columns -----------------


def _column_divergence(kernel, x, y):
    # sum_i d/dx_i K[i, j](x, y), one value per column j.
    J = jax.jacfwd(kernel.matrix, argnums=0)(x, y)  # (i, j, a) = dK_ij/dx_a
    return einsum(J, jnp.eye(x.shape[0]), "i j a, i a -> j")


@pytest.mark.parametrize("d", [2, 3])
@pytest.mark.parametrize(
    "kernel",
    [
        kl.RBF(lengthscale=ELL, variance=VAR),
        pytest.param(kl.Matern(nu=2.5, lengthscale=0.8), marks=pytest.mark.slow),
    ],
    ids=["rbf", "matern52"],
)
def test_divergence_free_columns_have_zero_divergence(kernel, d):
    X = _points(4, d, key=3)
    k = kl.DivergenceFree(kernel)
    for x, y in [(X[0], X[1]), (X[2], X[3])]:
        assert jnp.max(jnp.abs(_column_divergence(k, x, y))) < 1e-10
    # Control: the curl-free columns do have divergence.
    assert jnp.max(jnp.abs(_column_divergence(kl.CurlFree(kernel), X[0], X[1]))) > 0.1


@pytest.mark.parametrize(
    "kernel",
    [
        kl.RBF(lengthscale=ELL, variance=VAR),
        pytest.param(kl.Matern(nu=2.5, lengthscale=0.8), marks=pytest.mark.slow),
    ],
    ids=["rbf", "matern52"],
)
def test_curl_free_columns_have_zero_curl(kernel):
    X = _points(4, key=4)

    def curl(k, x, y):
        J = jax.jacfwd(k.matrix, argnums=0)(x, y)  # (i, j, a)
        return J[1, :, 0] - J[0, :, 1]

    for x, y in [(X[0], X[1]), (X[2], X[3])]:
        assert jnp.max(jnp.abs(curl(kl.CurlFree(kernel), x, y))) < 1e-10
    assert jnp.max(jnp.abs(curl(kl.DivergenceFree(kernel), X[0], X[1]))) > 0.1


# Positive definiteness ----------------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize(
    "kernel",
    [
        kl.CurlFree(kl.RBF(lengthscale=0.5)),
        kl.DivergenceFree(kl.RBF(lengthscale=0.5)),
        kl.Helmholtz(kl.RBF(lengthscale=0.5), kl.RBF(lengthscale=0.3, variance=2.0)),
    ],
    ids=["curl-free", "div-free", "helmholtz"],
)
def test_gram_is_psd(kernel):
    n, d = 150, 2
    X = _points(n, key=5)
    Xv = kl.vector_inputs(X)
    K = kernel(Xv, Xv)
    assert K.shape == (n * d, n * d)
    assert jnp.allclose(K, rearrange(K, "a b -> b a"), atol=1e-12)
    assert jnp.linalg.eigvalsh(K)[0] >= -1e-10 * n * d


@pytest.mark.slow
def test_gram_is_psd_autodiff_base():
    X = _points(40, d=3, key=6)
    Xv = kl.vector_inputs(X)
    for k in (
        kl.CurlFree(kl.Matern(nu=2.5, lengthscale=0.5)),
        kl.DivergenceFree(kl.RationalQuadratic(lengthscale=0.5)),
    ):
        assert jnp.linalg.eigvalsh(k(Xv, Xv))[0] >= -1e-10 * Xv.shape[0]


# Layout and composition ---------------------------------------------------


@pytest.mark.slow
def test_gram_is_the_point_major_flattening():
    X1, X2 = _points(5, key=7), _points(3, key=8)
    k = kl.Helmholtz(kl.RBF(lengthscale=0.5), kl.RBF(lengthscale=0.9))
    blocks = k.matrix_gram(X1, X2)
    assert blocks.shape == (5, 3, 2, 2)
    K = k(kl.vector_inputs(X1), kl.vector_inputs(X2))
    assert jnp.allclose(K, rearrange(blocks, "n m i j -> (n i) (m j)"))
    assert jnp.allclose(
        k.diag(kl.vector_inputs(X1)), jnp.diag(k.gram(kl.vector_inputs(X1)))
    )


def test_vector_inputs_reorders_derivative_inputs():
    X = _points(4, d=3, key=9)
    Xv = kl.vector_inputs(X)
    assert Xv.shape == (12, 4)
    assert jnp.allclose(Xv[5], jnp.append(X[1], 2.0))
    component_major = kl.derivative_inputs(X, values=False)
    assert jnp.allclose(rearrange(component_major, "(i n) c -> (n i) c", n=4), Xv)


@pytest.mark.slow
def test_curl_free_is_derivative_indexed_on_gradient_rows():
    # The same [x, i] convention: CurlFree is DerivativeIndexed without values.
    X = _points(5, key=10)
    Xa = kl.derivative_inputs(X, values=False)
    k = kl.RBF(lengthscale=ELL, variance=VAR)
    assert jnp.allclose(kl.CurlFree(k)(Xa, Xa), kl.DerivativeIndexed(k)(Xa, Xa))


def test_helmholtz_is_the_sum():
    X = _points(6, key=11)
    Xv = kl.vector_inputs(X)
    kp, ks = kl.RBF(lengthscale=0.5), kl.RBF(lengthscale=0.8, variance=0.5)
    expected = (kl.CurlFree(kp) + kl.DivergenceFree(ks))(Xv, Xv)
    assert jnp.allclose(kl.Helmholtz(kp, ks)(Xv, Xv), expected)
    same = kl.Helmholtz(kp, kp)(Xv, Xv)
    assert jnp.allclose(same, kl.CurlFree(kp)(Xv, Xv) + kl.DivergenceFree(kp)(Xv, Xv))


@pytest.mark.slow
def test_hyperparameter_gradients_are_finite():
    Xv = kl.vector_inputs(_points(5, key=12))

    def loss(ell):
        k = kl.Helmholtz(kl.RBF(lengthscale=ell), kl.RBF(lengthscale=2.0 * ell))
        return gx.logdet(kl.to_operator(k, Xv, noise=1e-3))

    g = jax.grad(loss)(jnp.asarray(0.5))
    assert jnp.isfinite(g) and g != 0.0


# Posterior mean through to_operator -> gaussx, checked by finite differences


def _posterior_mean_fn(kernel, implicit=False):
    X = _points(30, key=13)
    Y = jax.random.normal(jax.random.key(14), (30, 2))
    Xv = kl.vector_inputs(X)
    op = kl.to_operator(kernel, Xv, noise=1e-4, implicit=implicit)
    alpha = gx.solve(op, rearrange(Y, "n i -> (n i)"))

    def mean(x):  # (2,) -> (2,)
        K = kernel(kl.vector_inputs(rearrange(x, "d -> 1 d")), Xv)
        return K @ alpha

    return mean, alpha


def _fd_jacobian(f, x, eps=1e-5):
    # J[i, a] = d f_i / d x_a by central differences.
    e = jnp.eye(x.shape[0]) * eps
    cols = [(f(x + e[a]) - f(x - e[a])) / (2 * eps) for a in range(x.shape[0])]
    return rearrange(jnp.stack(cols), "a i -> i a")


@pytest.mark.slow
def test_posterior_means_are_divergence_and_curl_free():
    Xq = _points(5, key=15) * 0.8
    df_mean, _ = _posterior_mean_fn(kl.DivergenceFree(kl.RBF(lengthscale=0.5)))
    cf_mean, _ = _posterior_mean_fn(kl.CurlFree(kl.RBF(lengthscale=0.5)))
    for x in Xq:
        J_df, J_cf = _fd_jacobian(df_mean, x), _fd_jacobian(cf_mean, x)
        scale = jnp.max(jnp.abs(J_df))
        assert scale > 0.1  # the field is not trivially constant
        assert jnp.abs(jnp.trace(J_df)) < 1e-6 * scale  # divergence
        assert jnp.abs(J_cf[1, 0] - J_cf[0, 1]) < 1e-6 * jnp.max(jnp.abs(J_cf))
        # Controls: the other invariant is not zero.
        assert jnp.abs(J_df[1, 0] - J_df[0, 1]) > 1e-3 * scale


@pytest.mark.slow
def test_implicit_operator_solve_matches_dense():
    k = kl.DivergenceFree(kl.RBF(lengthscale=0.5))
    _, dense = _posterior_mean_fn(k)
    _, implicit = _posterior_mean_fn(k, implicit=True)
    assert jnp.allclose(dense, implicit, rtol=1e-6, atol=1e-6)


# Validation ---------------------------------------------------------------


@pytest.mark.parametrize("cls", [kl.CurlFree, kl.DivergenceFree])
@pytest.mark.parametrize(
    "kernel",
    [kl.Matern(nu=0.5), kl.RBF() + kl.White(0.1)],
    ids=["matern12", "white"],
)
def test_rough_bases_are_rejected(cls, kernel):
    with pytest.raises(ValueError, match="mean-square differentiable"):
        cls(kernel)


def test_helmholtz_rejects_a_rough_part():
    with pytest.raises(ValueError, match="mean-square differentiable"):
        kl.Helmholtz(kl.RBF(), kl.Matern(nu=0.5))


def test_gram_only_base_is_rejected():
    class _GramOnly(kl.AbstractKernel):
        def __call__(self, X1, X2):
            return reduce(einsum(X1, X2, "n d, m d -> n m d"), "n m d -> n m", "sum")

    with pytest.raises(TypeError, match="Gram-only"):
        kl.CurlFree(_GramOnly())


def test_divergence_free_needs_two_dimensions():
    with pytest.raises(ValueError, match="dimension >= 2"):
        kl.DivergenceFree(kl.RBF()).matrix(jnp.zeros(1), jnp.ones(1))
