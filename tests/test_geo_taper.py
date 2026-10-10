"""Tests for covariance tapering into a sparse gaussx operator (GEO8).

Inputs come from pinned keys: the randomness is incidental, so the tolerances
are deterministic.
"""

from __future__ import annotations

import functools
import re

import einx
import equinox as eqx
import gaussx as gx
import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
import numpy as np
import pytest

import kernellib as kl
from kernellib import functional as F
from kernellib._einx import reduce


N = 200
X = jr.uniform(jr.key(0), (N, 2))
LONLAT = jr.uniform(
    jr.key(1),
    (N, 2),
    minval=jnp.array([-30.0, -20.0]),
    maxval=jnp.array([30.0, 20.0]),
)
KERNEL = kl.Matern(lengthscale=0.2, nu=1.5)
RANGE = 0.3
GC_KERNEL = kl.GreatCircleExponential(lengthscale=1000.0, radius=kl.EARTH_RADIUS_KM)
GC_RANGE = 1500.0


@functools.cache
def _euclidean() -> tuple[gx.SparseOperator, jax.Array, jax.Array]:
    K = kl.tapered_operator(KERNEL, X, taper_range=RANGE)
    dense = kl.Tapered(KERNEL, kl.Wendland(lengthscale=RANGE))(X, X)
    dist = jnp.sqrt(
        reduce(einx.subtract("i d, j d -> i j d", X, X) ** 2, "i j d -> i j", "sum")
    )
    return K, dense, dist


@functools.cache
def _great_circle() -> tuple[gx.SparseOperator, jax.Array, jax.Array]:
    K = kl.tapered_operator(
        GC_KERNEL,
        LONLAT,
        taper_range=GC_RANGE,
        metric="great_circle",
        radius=kl.EARTH_RADIUS_KM,
    )
    taper = kl.GreatCircleWendland(lengthscale=GC_RANGE, radius=kl.EARTH_RADIUS_KM)
    dense = kl.Tapered(GC_KERNEL, taper)(LONLAT, LONLAT)
    dist = F.great_circle_distance(LONLAT, LONLAT, radius=kl.EARTH_RADIUS_KM)
    return K, dense, dist


CASES = [
    pytest.param(_euclidean, RANGE, id="euclidean"),
    pytest.param(_great_circle, GC_RANGE, id="great_circle"),
]


@pytest.mark.parametrize("build,taper_range", CASES)
class TestTaperedOperator:
    def test_matches_dense_tapered_gram(self, build, taper_range):
        K, dense, _ = build()
        assert jnp.allclose(K.as_matrix(), dense, rtol=0, atol=1e-12)

    def test_pattern_is_the_support(self, build, taper_range):
        K, _, dist = build()
        p = K.pattern
        assert p.symmetric
        assert p.shape == (N, N)
        # The diagonal is stored.
        diag = p.rows == p.cols
        assert np.array_equal(np.sort(p.rows[diag]), np.arange(N))
        # Off-diagonal entries are exactly the pairs closer than the range.
        dist = np.asarray(dist)
        off = ~diag
        assert np.all(dist[p.rows[off], p.cols[off]] < taper_range)
        lower = np.tril(dist < taper_range, k=-1)
        assert int(np.count_nonzero(lower)) == int(np.count_nonzero(off))
        # Sparse: well below the N^2 / 2 entries of a dense lower triangle.
        assert p.nnz < N * N // 4

    def test_positive_semidefinite(self, build, taper_range):
        K, dense, _ = build()
        assert float(jnp.linalg.eigvalsh(K.as_matrix()).min()) >= -1e-10 * N
        assert float(jnp.linalg.eigvalsh(dense).min()) >= -1e-10 * N


# The widening searches compile one neighbour search per size: slow on the
# sphere, so only the Euclidean case runs in the fast tier.
@pytest.mark.parametrize(
    "build,taper_range",
    [
        pytest.param(_euclidean, RANGE, id="euclidean"),
        pytest.param(
            _great_circle, GC_RANGE, id="great_circle", marks=pytest.mark.slow
        ),
    ],
)
def test_too_few_neighbours_raises_with_the_count(build, taper_range):
    _, _, dist = build()
    m = 60  # a subset keeps the widening searches cheap
    euclid = build is _euclidean
    Xs = (X if euclid else LONLAT)[:m]
    kernel = KERNEL if euclid else GC_KERNEL
    kwargs = {} if euclid else {"metric": "great_circle", "radius": kl.EARTH_RADIUS_KM}
    # Neighbours of each point within range, itself excluded.
    counts = reduce(dist[:m, :m] < taper_range, "i j -> i", "sum") - 1
    needed = int(jnp.max(counts))
    assert needed > 1
    with pytest.raises(ValueError, match=re.escape(f"max_neighbors >= {needed}")):
        kl.tapered_operator(
            kernel, Xs, taper_range=taper_range, max_neighbors=needed - 1, **kwargs
        )
    # Exactly the reported count is enough.
    ok = kl.tapered_operator(
        kernel, Xs, taper_range=taper_range, max_neighbors=needed, **kwargs
    )
    assert ok.pattern.nnz == m + (int(jnp.sum(counts)) // 2)


def test_wendland4_taper_matches_dense():
    K = kl.tapered_operator(KERNEL, X, taper_range=RANGE, taper="wendland4")
    dense = kl.Tapered(KERNEL, kl.Wendland(lengthscale=RANGE, order=4))(X, X)
    assert jnp.allclose(K.as_matrix(), dense, rtol=0, atol=1e-12)


@pytest.mark.slow
def test_logdet_gradient_matches_dense():
    noise = 0.1
    pattern = kl.tapered_operator(KERNEL, X, taper_range=RANGE).pattern
    taper = kl.Wendland(lengthscale=RANGE)

    @jax.jit
    def sparse(ell):
        K = kl.tapered_operator(
            kl.Matern(lengthscale=ell, nu=1.5), X, taper_range=RANGE, pattern=pattern
        )
        K = K.add_diagonal(jnp.full(N, noise))
        return gx.SparseCholeskySolver().logdet(K)

    def dense(ell):
        K = kl.Tapered(kl.Matern(lengthscale=ell, nu=1.5), taper)(X, X)
        return jnp.linalg.slogdet(K + noise * jnp.eye(N))[1]

    ell = jnp.array(0.2)
    assert jnp.allclose(sparse(ell), dense(ell), rtol=0, atol=1e-8)
    assert jnp.allclose(jax.grad(sparse)(ell), jax.grad(dense)(ell), rtol=0, atol=1e-8)


def test_operator_is_tagged_psd_and_symmetric():
    K = kl.tapered_operator(KERNEL, X[:20], taper_range=RANGE)
    assert lx.is_symmetric(K)
    assert lx.is_positive_semidefinite(K)


def test_single_point():
    K = kl.tapered_operator(KERNEL, X[:1], taper_range=RANGE)
    assert K.as_matrix().tolist() == [[1.0]]


def test_invalid_options_raise():
    with pytest.raises(ValueError, match="taper must be"):
        kl.tapered_operator(KERNEL, X, taper_range=RANGE, taper="wendland6")  # ty: ignore[invalid-argument-type]
    with pytest.raises(ValueError, match="metric must be"):
        kl.tapered_operator(KERNEL, X, taper_range=RANGE, metric="chordal")  # ty: ignore[invalid-argument-type]


class TestTapered:
    k = kl.Tapered(KERNEL, kl.Wendland(lengthscale=RANGE))

    def test_is_the_product(self):
        expected = F.matern_kernel(
            X, X, jnp.array(1.0), jnp.array(0.2), 1.5
        ) * F.wendland_kernel(X, X, jnp.array(1.0), jnp.array(RANGE))
        assert jnp.allclose(self.k(X, X), expected, rtol=0, atol=1e-12)

    def test_diag_elwise_and_pairwise(self):
        G = self.k(X[:10], X[:10])
        assert jnp.allclose(self.k.diag(X[:10]), jnp.diag(G), atol=1e-12)
        assert jnp.allclose(
            self.k.elwise(X[:10], X[10:20]),
            jnp.diag(self.k(X[:10], X[10:20])),
            atol=1e-12,
        )
        assert jnp.allclose(self.k.pairwise(X[0], X[1]), G[0, 1], atol=1e-12)

    def test_flags(self):
        assert self.k.is_pointwise
        assert self.k.is_stationary
        gc = kl.Tapered(
            GC_KERNEL,
            kl.GreatCircleWendland(lengthscale=GC_RANGE, radius=kl.EARTH_RADIUS_KM),
        )
        assert gc.is_pointwise
        assert not gc.is_stationary

    def test_jit_and_grad(self):
        g = eqx.filter_grad(lambda k: jnp.sum(k(X[:10], X[:10])))(self.k)
        assert jnp.isfinite(g.kernel.lengthscale)
        out = eqx.filter_jit(lambda k, A: k(A, A))(self.k, X[:10])
        assert jnp.allclose(out, self.k(X[:10], X[:10]), atol=1e-12)


@pytest.mark.slow
def test_large_build_and_factorise_stay_sparse():
    n = 20_000
    Xl = jr.uniform(jr.key(3), (n, 2))
    # About n * pi * r^2 ~ 25 neighbours per point.
    K = kl.tapered_operator(
        kl.Matern(lengthscale=0.01, nu=0.5), Xl, taper_range=0.02, max_neighbors=64
    )
    assert K.pattern.nnz < 40 * n
    K = K.add_diagonal(jnp.full(n, 0.1))
    logdet = gx.SparseCholeskySolver().logdet(K)
    assert jnp.isfinite(logdet)
