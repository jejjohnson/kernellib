"""Graph spectra (heat, Matérn) and the dense graph Matérn kernel."""

from __future__ import annotations

import einx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import kernellib as kl
from kernellib._einx import einsum
from kernellib.functional import graph_heat_spectrum, graph_matern_spectrum


def _graph():
    X = jax.random.uniform(jax.random.key(0), (40, 2))
    return kl.knn_graph(X, 5, ensure_connected=True)


def _normalised(K):
    return K / (jnp.trace(K) / K.shape[0])


class TestSpectra:
    def test_average_variance_normalisation(self):
        lam = jnp.linspace(0.0, 2.0, 30)
        for phi in (
            graph_matern_spectrum(
                lam, nu=2.5, lengthscale=1.3, variance=2.0, n_nodes=50
            ),
            graph_heat_spectrum(lam, lengthscale=1.3, variance=2.0, n_nodes=50),
        ):
            # Truncated to 30 of 50 eigenpairs: sum(phi) / N is the variance.
            assert np.isclose(float(jnp.sum(phi)) / 50, 2.0)

    def test_unnormalised_closed_forms(self):
        lam = jnp.array([0.0, 0.3, 1.7])
        assert np.allclose(
            graph_matern_spectrum(lam, nu=1.5, lengthscale=0.8, variance=3.0),
            3.0 * (2 * 1.5 / 0.8**2 + lam) ** -1.5,
        )
        assert np.allclose(
            graph_heat_spectrum(lam, lengthscale=0.8, variance=3.0),
            3.0 * jnp.exp(-0.5 * 0.8**2 * lam),
        )

    def test_decreasing_and_positive(self):
        lam = jnp.linspace(0.0, 4.0, 20)
        phi = graph_matern_spectrum(lam, nu=0.5, lengthscale=1.0)
        assert np.all(np.asarray(phi) > 0) and np.all(np.diff(phi) < 0)

    def test_matern_tends_to_heat(self):
        lam = jnp.linspace(0.0, 2.0, 25)
        heat = graph_heat_spectrum(lam, lengthscale=1.5, n_nodes=25)
        # Large nu: the raw spectrum underflows, the normalised one does not.
        far = graph_matern_spectrum(lam, nu=1e5, lengthscale=1.5, n_nodes=25)
        near = graph_matern_spectrum(lam, nu=2.0, lengthscale=1.5, n_nodes=25)
        assert np.all(np.isfinite(np.asarray(far)))
        assert float(jnp.max(jnp.abs(far - heat))) < 1e-3
        assert float(jnp.max(jnp.abs(near - heat))) > 1e-2

    def test_normalisation_in_float32(self):
        # Large common log offsets: the normalisation must still hold, and the
        # large-nu Matern spectrum must stay close to the heat spectrum.
        lam = jnp.linspace(0.0, 2.0, 25, dtype=jnp.float32)
        far = graph_matern_spectrum(lam, nu=1e7, lengthscale=1.5, n_nodes=25)
        heat = graph_heat_spectrum(lam, lengthscale=1.5, n_nodes=25)
        assert far.dtype == jnp.float32
        assert np.isclose(float(jnp.sum(far)), 25.0, rtol=1e-5)
        assert float(jnp.max(jnp.abs(far - heat))) < 1e-3
        big = graph_heat_spectrum(
            jnp.full(10, 1e6, dtype=jnp.float32), lengthscale=10.0, n_nodes=10
        )
        assert np.allclose(big, 1.0, rtol=1e-5)

    def test_differentiable_in_the_hyperparameters(self):
        lam = jnp.linspace(0.0, 2.0, 10)

        def total(ell):
            return jnp.sum(graph_matern_spectrum(lam, nu=1.5, lengthscale=ell) * lam)

        g = jax.grad(total)(1.0)
        fd = (total(1.0 + 1e-6) - total(1.0 - 1e-6)) / 2e-6
        assert np.isclose(g, fd, rtol=1e-5)


class TestMaternGraphKernel:
    @pytest.mark.parametrize("normalization", ["symmetric", "unnormalized"])
    def test_equals_u_phi_ut_from_the_full_eigendecomposition(self, normalization):
        g = _graph()
        K = kl.matern_graph_kernel(
            g, nu=1.5, lengthscale=1.2, variance=0.7, normalization=normalization
        )
        lam, U = jnp.linalg.eigh(kl.graph_laplacian(g.to_dense(), normalization))
        phi = graph_matern_spectrum(
            jnp.clip(lam, min=0.0), nu=1.5, lengthscale=1.2, variance=0.7, n_nodes=40
        )
        expected = einsum(einx.multiply("i k, k -> i k", U, phi), U, "i k, j k -> i j")
        assert np.allclose(K, expected, atol=1e-10)

    def test_positive_semidefinite_with_unit_average_variance(self):
        K = kl.matern_graph_kernel(_graph(), nu=0.5, lengthscale=0.7, variance=2.0)
        assert np.allclose(K, K.T)
        assert np.min(np.linalg.eigvalsh(np.asarray(K))) > -1e-10
        assert np.isclose(float(jnp.trace(K)) / 40, 2.0)

    def test_tends_to_the_diffusion_kernel(self):
        g = _graph()
        ell = 1.4
        diffusion = _normalised(kl.diffusion_kernel(g, beta=ell**2 / 2))
        far = kl.matern_graph_kernel(g, nu=1e5, lengthscale=ell)
        near = kl.matern_graph_kernel(g, nu=1.0, lengthscale=ell)
        assert float(jnp.max(jnp.abs(far - diffusion))) < 1e-3
        assert float(jnp.max(jnp.abs(near - diffusion))) > 1e-2

    def test_graph_and_array_inputs_agree(self):
        g = _graph()
        W = g.to_dense()
        assert np.allclose(
            kl.matern_graph_kernel(g, nu=1.5, lengthscale=1.0),
            kl.matern_graph_kernel(W, nu=1.5, lengthscale=1.0),
        )

    def test_random_walk_normalisation_is_rejected(self):
        with pytest.raises(ValueError, match="symmetric Laplacian"):
            kl.matern_graph_kernel(
                _graph(), nu=1.5, lengthscale=1.0, normalization="random_walk"
            )


@pytest.mark.parametrize(
    ("fn", "kwargs"),
    [
        (kl.diffusion_kernel, {"beta": 0.5}),
        (kl.regularized_laplacian_kernel, {"sigma": 0.7}),
        (kl.random_walk_kernel, {"p": 2}),
        (kl.cosine_graph_kernel, {}),
        (kl.commute_time_kernel, {}),
    ],
)
def test_existing_graph_kernels_accept_graphs(fn, kwargs):
    g = _graph()
    assert np.allclose(fn(g, **kwargs), fn(g.to_dense(), **kwargs))
