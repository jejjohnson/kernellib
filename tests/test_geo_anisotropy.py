"""`LinearTransform` and `GeometricAnisotropy`: kernel values, the spectral
transform S_A(ω) = |det A|⁻¹ S₀(A⁻ᵀω), frequency sampling ω = Aᵀω₀, and the
random Fourier feature maps."""

from __future__ import annotations

import math

import einx
import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

import kernellib as kl
from kernellib._einx import einsum, rearrange, reduce


ELL = 1.7
MATERN = kl.Matern(nu=1.5, lengthscale=ELL, variance=1.3)
PSI1 = (1.0 + math.sqrt(3.0)) * math.exp(-math.sqrt(3.0))  # Matérn-3/2 at r = 1


def _ga2(alpha=0.6, a=0.4, base=MATERN):
    return kl.GeometricAnisotropy(base, jnp.array([alpha]), jnp.array([a]))


def _ga3(angles=(0.4, -0.7, 1.1), ratios=(0.5, 0.25), base=MATERN):
    return kl.GeometricAnisotropy(base, jnp.array(angles), jnp.array(ratios))


def _points(n, d, key=0, scale=2.0):
    return jax.random.uniform(jax.random.key(key), (n, d), minval=-scale, maxval=scale)


def _k0(kernel, h):
    """``k(0, h)`` for one lag vector ``h``."""
    return kernel.pairwise(jnp.zeros_like(h), h)


# -- kernel values ------------------------------------------------------------


@pytest.mark.parametrize("alpha", [0.0, 0.3, 1.2, 2.9, -2.0])
def test_ratio_one_is_the_base_kernel(alpha):
    X = _points(7, 2)
    assert jnp.allclose(_ga2(alpha, 1.0)(X, X), MATERN(X, X))
    X3 = _points(7, 3)
    k3 = _ga3((alpha, 0.5 * alpha, -alpha), (1.0, 1.0))
    assert jnp.allclose(k3(X3, X3), MATERN(X3, X3))


@pytest.mark.parametrize("alpha", [0.0, 0.6, 2.2])
def test_alpha_and_alpha_plus_pi_agree(alpha):
    X = _points(7, 2, key=1)
    assert jnp.allclose(_ga2(alpha)(X, X), _ga2(alpha + jnp.pi)(X, X))


def test_ranges_along_the_main_axes_2d():
    alpha, a = 0.6, 0.4
    k = _ga2(alpha, a)
    major = jnp.array([jnp.cos(alpha), jnp.sin(alpha)])
    minor = jnp.array([-jnp.sin(alpha), jnp.cos(alpha)])
    assert jnp.allclose(_k0(k, ELL * major), 1.3 * PSI1)
    assert jnp.allclose(_k0(k, a * ELL * minor), 1.3 * PSI1)
    # Not the same correlation at equal distance along the two axes.
    assert not jnp.allclose(_k0(k, ELL * minor), 1.3 * PSI1)


def test_ranges_along_the_main_axes_3d():
    k = _ga3()
    R = k.rotation
    assert jnp.allclose(einsum(R, R, "k i, k j -> i j"), jnp.eye(3))
    assert jnp.allclose(jnp.linalg.det(R), 1.0)
    axes = rearrange(R, "d k -> k d")  # columns of R: the main axes
    for axis, ratio in zip(axes, (1.0, 0.5, 0.25), strict=True):
        assert jnp.allclose(_k0(k, ratio * ELL * axis), 1.3 * PSI1)


def test_3d_yaw_pitch_roll_order():
    # R = R_z(alpha) R_y(beta) R_x(gamma): a pure yaw turns the major axis in the x-y
    # plane, and a pitch beta tilts it to (cos beta, 0, -sin beta).
    yaw = _ga3((0.5, 0.0, 0.0)).rotation
    pitch = _ga3((0.0, 0.5, 0.0)).rotation
    roll = _ga3((0.0, 0.0, 0.5)).rotation
    assert jnp.allclose(yaw[:, 0], jnp.array([jnp.cos(0.5), jnp.sin(0.5), 0.0]))
    assert jnp.allclose(pitch[:, 0], jnp.array([jnp.cos(0.5), 0.0, -jnp.sin(0.5)]))
    assert jnp.allclose(roll[:, 0], jnp.array([1.0, 0.0, 0.0]))
    full = _ga3((0.5, 0.5, 0.5)).rotation
    assert jnp.allclose(full, yaw @ pitch @ roll)


def test_3d_zero_angles_is_ard():
    k = _ga3((0.0, 0.0, 0.0), (0.5, 0.25))
    ard = kl.Matern(nu=1.5, lengthscale=ELL * jnp.array([1.0, 0.5, 0.25]), variance=1.3)
    X = _points(8, 3, key=2)
    assert jnp.allclose(k(X, X), ard(X, X))


def test_diagonal_linear_transform_is_ard():
    A = jnp.diag(jnp.array([2.0, 0.25]))
    k = kl.LinearTransform(kl.RBF(lengthscale=0.8), A)
    ard = kl.RBF(lengthscale=0.8 / jnp.array([2.0, 0.25]))
    X = _points(6, 2, key=3)
    assert jnp.allclose(k(X, X), ard(X, X))


def test_gram_pairwise_and_diag_agree():
    k = _ga2()
    X1, X2 = _points(5, 2, key=4), _points(3, 2, key=5)
    pw = jax.vmap(lambda x: jax.vmap(lambda y: k.pairwise(x, y))(X2))(X1)
    assert jnp.allclose(k(X1, X2), pw)
    assert jnp.allclose(k.diag(X1), jnp.diag(k(X1, X1)))
    assert jnp.allclose(k.diag(X1), 1.3)
    assert k.is_stationary and k.is_pointwise
    assert k._gram_structure(X1) is None


def test_general_transform_matches_its_definition():
    A = jnp.array([[1.0, 0.7], [-0.3, 1.5]])
    k = kl.LinearTransform(MATERN, A)
    X1, X2 = _points(5, 2, key=6), _points(4, 2, key=7)
    want = MATERN(einsum(A, X1, "i j, n j -> n i"), einsum(A, X2, "i j, n j -> n i"))
    assert jnp.allclose(k(X1, X2), want)


@pytest.mark.parametrize("d", [2, 3])
def test_positive_definite(d):
    n = 60
    k = _ga2() if d == 2 else _ga3()
    X = _points(n, d, key=8)
    assert jnp.min(jnp.linalg.eigvalsh(k(X, X))) >= -1e-10 * n


@pytest.mark.slow
def test_gradients_wrt_angles_and_ratios_are_finite():
    X = _points(6, 3, key=9)

    def loss(k):
        return jnp.sum(k(X, X) ** 2)

    g = eqx.filter_grad(loss)(_ga3())
    for leaf in (g.angles, g.ratios, g.kernel.lengthscale, g.kernel.variance):
        assert jnp.all(jnp.isfinite(leaf))
        assert jnp.any(leaf != 0.0)
    g2 = eqx.filter_grad(lambda k: jnp.sum(k(X[:, :2], X[:, :2])))(_ga2())
    assert jnp.isfinite(g2.angles).all() and jnp.isfinite(g2.ratios).all()
    gA = eqx.filter_grad(lambda k: jnp.sum(k(X, X)))(
        kl.LinearTransform(MATERN, jnp.eye(3))
    )
    assert jnp.isfinite(gA.A).all()


def test_jit():
    k = _ga2()
    X = _points(4, 2)
    assert jnp.allclose(eqx.filter_jit(lambda k: k(X, X))(k), k(X, X))


def test_from_angles_degrees():
    k = kl.GeometricAnisotropy.from_angles(MATERN, 30.0, 0.5, degrees=True)
    assert jnp.allclose(k.angles, jnp.array([jnp.pi / 6]))
    assert jnp.allclose(k.A, _ga2(jnp.pi / 6, 0.5).A)
    assert k.lengthscale == 1.0 and k.variance == 1.3


def test_bad_inputs_raise():
    with pytest.raises(ValueError, match="2-D"):
        kl.GeometricAnisotropy(MATERN, jnp.array([0.1, 0.2]), jnp.array([0.5]))
    with pytest.raises(ValueError, match="square"):
        kl.LinearTransform(MATERN, jnp.ones((2, 3)))
    with pytest.raises(TypeError, match="stationary"):
        kl.LinearTransform(kl.Linear(), jnp.eye(2))  # ty: ignore[invalid-argument-type]
    with pytest.raises(ValueError, match="transform"):
        _ga2()(_points(2, 3), _points(2, 3))
    with pytest.raises(ValueError, match="transform"):
        _ga2().spectral_density(jnp.zeros((1, 3)))
    with pytest.raises(ValueError, match="transform"):
        _ga2().sample_frequencies(jax.random.key(0), 4, 3)
    with pytest.raises(NotImplementedError, match="anisotropic"):
        _ga2().unit_spectral_density(jnp.zeros(()), 2)


# -- spectral side ------------------------------------------------------------


def test_rbf_density_is_the_gaussian_closed_form():
    # k(h) = σ² exp(-hᵀPh / 2) with P = AᵀA / ℓ² has density
    # S(ω) = σ² (2π)^{D/2} det(P)^{-1/2} exp(-ωᵀP⁻¹ω / 2).
    A = jnp.array([[1.0, 0.7], [-0.3, 1.5]])
    ell, var = 0.8, 1.4
    k = kl.LinearTransform(kl.RBF(lengthscale=ell, variance=var), A)
    P = einsum(A, A, "k i, k j -> i j") / ell**2
    omega = _points(9, 2, key=10, scale=3.0)
    quad = einsum(omega, jnp.linalg.inv(P), omega, "n i, i j, n j -> n")
    want = var * 2.0 * jnp.pi / jnp.sqrt(jnp.linalg.det(P)) * jnp.exp(-0.5 * quad)
    assert jnp.allclose(k.spectral_density(omega), want)


@pytest.mark.parametrize("base", [kl.RBF(lengthscale=0.9), kl.Matern(nu=2.5)])
def test_density_inverts_to_the_kernel(base):
    # k(τ) = (2π)^{-2} ∫ S(ω) cos(ωᵀτ) dω by the midpoint rule on a wide grid;
    # the Matérn-5/2 tail ~|ω|^{-7} truncated at |ω| = 40 costs < 1e-6.
    k = _ga2(0.6, 0.5, base=base)
    n, half = 801, 40.0
    w = jnp.linspace(-half, half, n)
    dw = w[1] - w[0]
    grid = rearrange(jnp.stack(jnp.meshgrid(w, w, indexing="ij")), "d a b -> (a b) d")
    S = k.spectral_density(grid)
    taus = jnp.array([[0.0, 0.0], [0.4, -0.2], [-0.3, 0.9], [1.0, 0.5]])
    phase = einsum(grid, taus, "m d, t d -> m t")
    approx = einsum(S, jnp.cos(phase), "m, m t -> t") * dw**2 / (2.0 * jnp.pi) ** 2
    exact = jax.vmap(lambda t: _k0(k, t))(taus)
    assert jnp.allclose(approx, exact, atol=1e-4)


def test_rbf_frequencies_have_covariance_AtA_over_ell2():
    A = jnp.array([[1.0, 0.7], [-0.3, 1.5]])
    ell = 0.8
    k = kl.LinearTransform(kl.RBF(lengthscale=ell), A)
    n = 20_000
    omega = k.sample_frequencies(jax.random.key(0), n, 2)
    P = einsum(A, A, "k i, k j -> i j") / ell**2
    cov = einsum(omega, omega, "n i, n j -> i j") / n
    # For a zero-mean Gaussian, Var[ω_i ω_j] = P_ij² + P_ii P_jj, so the
    # sample second moment has standard error sqrt((P_ij² + P_ii P_jj) / n);
    # bound every entry by 7 of them.
    p_diag = jnp.diag(P)
    se = jnp.sqrt((P**2 + einx.multiply("i, j -> i j", p_diag, p_diag)) / n)
    assert jnp.all(jnp.abs(cov - P) <= 7.0 * se)


@pytest.mark.parametrize("kernel", [_ga2(), _ga3()], ids=["2d", "3d"])
def test_frequencies_reproduce_the_kernel(kernel):
    # E[cos(ωᵀτ)] = k(τ) / σ² (Bochner). The Monte Carlo mean of n draws has
    # standard error std(cos) / sqrt(n), estimated from the draws themselves;
    # bound each lag by 7 of them.
    d = kernel.A.shape[0]
    n = 20_000
    omega = kernel.sample_frequencies(jax.random.key(1), n, d)
    taus = _points(5, d, key=11, scale=1.0)
    c = jnp.cos(einsum(omega, taus, "n d, t d -> n t"))
    mean = reduce(c, "n t -> t", "mean")
    se = reduce(c, "n t -> t", "std") / jnp.sqrt(n)
    exact = jax.vmap(lambda t: _k0(kernel, t))(taus) / kernel.variance
    assert jnp.all(jnp.abs(mean - exact) <= 7.0 * se + 1e-12)


def test_spectral_variance_and_dtype():
    k = _ga2()
    assert jnp.allclose(k.spectral_variance, 1.3)
    w = k.sample_frequencies(jax.random.key(0), 4, 2, dtype=jnp.float32)
    assert w.dtype == jnp.float32 and w.shape == (4, 2)


@pytest.mark.slow
def test_rff_paths_and_maps_accept_it():
    k = _ga2()
    X = _points(5, 2)
    Phi = kl.RandomFourierFeatures(16, jax.random.key(0)).fit(k, X)(X)
    assert Phi.shape == (5, 32)
    # Every RFF map is exact on the diagonal: ||φ(x)||² = σ².
    assert jnp.allclose(reduce(Phi**2, "n f -> n", "sum"), 1.3)
    v, ell, omega, phase, w = kl.draw_rff_cosine_basis(
        k, jax.random.key(0), n_paths=3, n_features=8, in_features=2
    )
    paths = kl.evaluate_rff_cosine_paths(
        X, variance=v, lengthscale=ell, omega=omega, phase=phase, weights=w
    )
    assert paths.shape == (3, 5) and jnp.all(jnp.isfinite(paths))
    # A sum with an isotropic kernel uses each part's own frequencies.
    s = k + kl.RBF(lengthscale=0.5)
    assert kl.RandomFourierFeatures(16, jax.random.key(0)).fit(s, X)(X).shape == (5, 32)


@pytest.mark.parametrize("cls", [kl.OrthogonalRandomFeatures, kl.FastFoodFeatures])
def test_length_only_maps_refuse_it(cls):
    X = _points(4, 2)
    with pytest.raises(NotImplementedError, match="isotropic"):
        cls(8, jax.random.key(0)).fit(_ga2(), X)
    with pytest.raises(NotImplementedError, match="isotropic"):
        cls(8, jax.random.key(0)).fit(kl.RBF() + _ga2(), X)


@pytest.mark.slow
@pytest.mark.parametrize("kernel", [_ga2(), _ga3()], ids=["2d", "3d"])
def test_rff_gram_converges_to_the_exact_gram(kernel):
    # Each draw's RFF Gram error has mean zero. Average it over independent
    # keys and bound every entry by 7 standard errors of that average,
    # estimated from the spread of the draws (the RFF sampling distribution).
    d = kernel.A.shape[0]
    X = _points(6, d, key=12, scale=1.0)
    K = kernel(X, X)

    def err(seed):
        Phi = kl.RandomFourierFeatures(256, jax.random.key(seed)).fit(kernel, X)(X)
        return einsum(Phi, Phi, "n f, m f -> n m") - K

    errs = jnp.stack([err(s) for s in range(40)])
    mean = reduce(errs, "s n m -> n m", "mean")
    se = reduce(errs, "s n m -> n m", "std") / jnp.sqrt(errs.shape[0])
    assert jnp.all(jnp.abs(mean) <= 7.0 * se + 1e-12)
    # Each entry is σ²/F Σ_j cos(ω_jᵀτ), of variance ≤ σ⁴/F, so the RMS error
    # is at most σ²/sqrt(F) in expectation; 40 x 36 entries concentrate it well
    # below 1.5 times that.
    rmse = jnp.sqrt(jnp.mean(errs**2))
    assert rmse < 1.5 * kernel.variance / jnp.sqrt(256.0)
