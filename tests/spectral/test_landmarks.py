"""Landmark selection: select_landmarks and its consumers."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import kernellib as kl
from kernellib._einx import einsum, rearrange
from kernellib._spectral._landmarks import _ridge_leverage_scores


METHODS = ["uniform", "leverage", "rpcholesky", "greedy"]


def _X(n=200, d=2, seed=0):
    return jax.random.normal(jax.random.key(seed), (n, d))


def _clustered(n_per=60, seed=0):
    """Three clusters of very different size and spread."""
    k1, k2, k3 = jax.random.split(jax.random.key(seed), 3)
    return jnp.concatenate(
        [
            0.05 * jax.random.normal(k1, (4 * n_per, 2)),
            3.0 + 0.5 * jax.random.normal(k2, (n_per, 2)),
            jnp.array([-4.0, 4.0]) + 1.5 * jax.random.normal(k3, (n_per // 2, 2)),
        ]
    )


def _old_nystrom_indices(nys, kernel, X):
    """The selection code of `NystromFeatures.fit` before K8, verbatim."""
    n = X.shape[0]
    if nys.selection == "uniform":
        return jax.random.choice(nys.key, n, (nys.n_components,), replace=False)
    key_pilot, key_draw = jax.random.split(nys.key)
    m0 = min(2 * nys.n_components, n)
    pilot = jax.random.choice(key_pilot, n, (m0,), replace=False)
    scores = _ridge_leverage_scores(
        kernel, X, pilot, nys.leverage_regularization, nys.jitter
    )
    return jax.random.choice(
        key_draw,
        n,
        (nys.n_components,),
        replace=False,
        p=(1.0 - nys.uniform_mixing) * scores / jnp.sum(scores)
        + nys.uniform_mixing / n,
    )


def _trace_error(kernel, X, idx):
    """tr(K - K_XZ K_ZZ^+ K_ZX)."""
    Z = X[idx]
    K_zz = kernel(Z, Z)
    K_xz = kernel(X, Z)
    sol = jnp.linalg.lstsq(K_zz, rearrange(K_xz, "n m -> m n"))[0]
    return jnp.sum(kernel.diag(X)) - einsum(K_xz, sol, "n m, m n ->")


class TestSelectLandmarks:
    @pytest.mark.parametrize(
        "method",
        [
            "uniform",
            pytest.param("leverage", marks=pytest.mark.slow),
            "rpcholesky",
            "greedy",
        ],
    )
    def test_distinct_indices(self, method):
        idx = kl.select_landmarks(
            kl.RBF(lengthscale=0.5), _X(), 25, method=method, key=jax.random.key(1)
        )
        assert idx.shape == (25,)
        values = np.asarray(idx)
        assert len(set(values.tolist())) == 25
        assert values.min() >= 0 and values.max() < 200

    def test_greedy_is_deterministic(self):
        X = _X()
        draws = [
            kl.select_landmarks(
                kl.RBF(), X, 15, method="greedy", key=jax.random.key(seed)
            )
            for seed in range(3)
        ]
        for idx in draws[1:]:
            assert np.array_equal(idx, draws[0])

    def test_greedy_first_pivot_is_the_largest_variance(self):
        # Equal diagonals: argmax picks index 0; the next pivot is the point
        # least explained by it, the farthest one.
        X = jnp.array([[0.0], [0.1], [5.0]])
        idx = kl.select_landmarks(
            kl.RBF(), X, 2, method="greedy", key=jax.random.key(0)
        )
        assert idx.tolist() == [0, 2]

    def test_rpcholesky_varies_with_the_key(self):
        X = _X()
        a = kl.select_landmarks(
            kl.RBF(), X, 10, method="rpcholesky", key=jax.random.key(0)
        )
        b = kl.select_landmarks(
            kl.RBF(), X, 10, method="rpcholesky", key=jax.random.key(1)
        )
        assert not np.array_equal(a, b)

    @pytest.mark.parametrize("method", ["rpcholesky", "greedy"])
    def test_exhausted_rank_is_filled_with_unused_points(self, method):
        # Three distinct points, each repeated: the kernel matrix has rank 3.
        X = rearrange(jnp.tile(jnp.array([0.0, 2.0, 7.0]), 4), "n -> n 1")
        idx = kl.select_landmarks(kl.RBF(), X, 6, method=method, key=jax.random.key(3))
        values = np.asarray(idx).tolist()
        assert len(set(values)) == 6
        assert {float(X[i, 0]) for i in values[:3]} == {0.0, 2.0, 7.0}
        if method == "greedy":
            # Filled in index order with the lowest unused indices.
            unused = [i for i in range(12) if i not in values[:3]]
            assert values[3:] == unused[:3]

    def test_jittable(self):
        X = _X()

        @jax.jit
        def select(key):
            return kl.select_landmarks(kl.RBF(), X, 10, method="rpcholesky", key=key)

        assert select(jax.random.key(0)).shape == (10,)

    @pytest.mark.slow
    def test_works_for_gram_only_kernels(self):
        X = _X(50)
        kernel = kl.RBF() + kl.Linear()
        idx = kl.select_landmarks(
            kernel, X, 8, method="rpcholesky", key=jax.random.key(0)
        )
        assert len(set(np.asarray(idx).tolist())) == 8

    def test_leverage_regularization_default(self):
        X = _X()
        key = jax.random.key(4)
        a = kl.select_landmarks(kl.RBF(), X, 10, method="leverage", key=key)
        b = kl.select_landmarks(
            kl.RBF(), X, 10, method="leverage", key=key, regularization=1e-3
        )
        assert np.array_equal(a, b)

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"method": "kmeans"}, "method"),
            ({"n_landmarks": 0}, "n_landmarks"),
            ({"n_landmarks": 201}, "n_landmarks"),
            ({"uniform_mixing": 1.5}, "uniform_mixing"),
            ({"regularization": 0.0}, "regularization"),
        ],
    )
    def test_validation(self, kwargs, match):
        args = {"n_landmarks": 5, **kwargs}
        n_landmarks = args.pop("n_landmarks")
        with pytest.raises(ValueError, match=match):
            kl.select_landmarks(
                kl.RBF(), _X(), n_landmarks, key=jax.random.key(0), **args
            )

    @pytest.mark.slow
    def test_rpcholesky_beats_uniform_on_clustered_data(self):
        # tr(K - K_hat) over 40 keys per method. The bound is the empirical
        # sampling distribution of the difference in means: require the gap to
        # exceed 4 standard errors, not a fixed tolerance.
        X = _clustered()
        kernel = kl.RBF(lengthscale=0.3)

        def errors(method):
            return np.array(
                [
                    float(
                        _trace_error(
                            kernel,
                            X,
                            kl.select_landmarks(
                                kernel, X, 20, method=method, key=jax.random.key(s)
                            ),
                        )
                    )
                    for s in range(40)
                ]
            )

        uniform, rpc = errors("uniform"), errors("rpcholesky")
        gap = uniform.mean() - rpc.mean()
        stderr = np.sqrt(uniform.var(ddof=1) / 40 + rpc.var(ddof=1) / 40)
        assert gap > 4.0 * stderr


class TestConsumers:
    @pytest.mark.parametrize("selection", ["uniform", "leverage"])
    def test_nystrom_features_are_bit_identical(self, selection):
        X = _X(120)
        kernel = kl.RBF(lengthscale=0.7)
        nys = kl.NystromFeatures(12, jax.random.key(5), selection=selection)
        fitted = nys.fit(kernel, X)
        expected = X[_old_nystrom_indices(nys, kernel, X)]
        assert np.array_equal(np.asarray(fitted.landmarks), np.asarray(expected))

    @pytest.mark.parametrize("selection", ["rpcholesky", "greedy"])
    def test_nystrom_features_new_selections(self, selection):
        X = _X(120)
        nys = kl.NystromFeatures(12, jax.random.key(5), selection=selection)
        fitted = nys.fit(kl.RBF(), X)
        idx = kl.select_landmarks(
            kl.RBF(), X, 12, method=selection, key=jax.random.key(5)
        )
        assert np.array_equal(fitted.landmarks, X[idx])

    @pytest.mark.slow
    def test_falkon_uniform_centres_unchanged(self):
        X = _X(150, 1)
        y = jnp.sin(3.0 * X[:, 0])
        key = jax.random.key(2)
        model = kl.Falkon(kl.RBF(), n_inducing=20).fit(X, y, key=key)
        expected = X[jax.random.choice(key, 150, (20,), replace=False)]
        assert np.array_equal(model.landmarks, expected)

    @pytest.mark.parametrize("centers", ["rpcholesky", "greedy", "leverage"])
    @pytest.mark.slow
    def test_falkon_centres(self, centers):
        X = _X(150, 1)
        y = jnp.sin(3.0 * X[:, 0])
        key = jax.random.key(2)
        model = kl.Falkon(
            kl.RBF(), n_inducing=20, centers=centers, regularization=1e-6
        ).fit(X, y, key=key)
        idx = kl.select_landmarks(kl.RBF(), X, 20, method=centers, key=key)
        assert np.array_equal(model.landmarks, X[idx])
        assert float(model.loss(X, y)) < 1e-3

    def test_falkon_rejects_unknown_centres(self):
        X = _X(50, 1)
        with pytest.raises(ValueError, match="method"):
            kl.Falkon(kl.RBF(), n_inducing=10, centers="kmeans").fit(
                X, X[:, 0], key=jax.random.key(0)
            )

    @pytest.mark.slow
    def test_eigenpro_uniform_subsample_unchanged(self):
        X = jax.random.uniform(jax.random.key(0), (120, 1))
        y = jnp.sin(6.0 * X[:, 0])
        key = jax.random.key(1)
        model = kl.EigenPro(
            kl.RBF(lengthscale=0.2), epochs=1, subsample_size=40, n_components=5
        ).fit(X, y, key=key)
        key_pre, _ = jax.random.split(key)
        expected = jax.random.choice(key_pre, 120, (40,), replace=False)
        assert np.array_equal(model.preconditioner.subsample_indices, expected)

    @pytest.mark.slow
    def test_eigenpro_greedy_subsample(self):
        X = jax.random.uniform(jax.random.key(0), (120, 1))
        y = jnp.sin(6.0 * X[:, 0])
        model = kl.EigenPro(
            kl.RBF(lengthscale=0.2),
            epochs=3,
            subsample_size=40,
            n_components=5,
            subsample="greedy",
        ).fit(X, y, key=jax.random.key(1))
        expected = kl.select_landmarks(
            kl.RBF(lengthscale=0.2), X, 40, method="greedy", key=jax.random.key(0)
        )
        assert np.array_equal(model.preconditioner.subsample_indices, expected)
        assert np.isfinite(float(model.loss(X, y)))

    def test_eigenpro_preconditioner_checks_the_subsample_shape(self):
        X = _X(30, 1)
        op = kl.to_operator(kl.RBF(), X)
        with pytest.raises(ValueError, match="subsample_indices"):
            kl.eigenpro_preconditioner(
                op, subsample_size=10, n_components=2, subsample_indices=jnp.arange(9)
            )


@pytest.mark.parametrize("method", ["leverage", "rpcholesky", "greedy"])
def test_repeat_calls_with_the_same_shapes_do_not_retrace(method, monkeypatch):
    # Count traces of the jitted core: a cache hit runs no Python. The shapes
    # are unusual so that no other test has compiled them already.
    import kernellib._spectral._landmarks as landmarks

    traces = []
    original = landmarks._fill_exhausted
    monkeypatch.setattr(
        landmarks,
        "_fill_exhausted",
        lambda *args: traces.append(1) or original(*args),
    )
    original_scores = landmarks._ridge_leverage_scores
    monkeypatch.setattr(
        landmarks,
        "_ridge_leverage_scores",
        lambda *args: traces.append(1) or original_scores(*args),
    )
    kernel = kl.RBF(lengthscale=0.7)
    X = _X(n=37, d=3)
    first = kl.select_landmarks(kernel, X, 5, method=method, key=jax.random.key(0))
    n_traces = len(traces)
    second = kl.select_landmarks(
        kernel, X + 1.0, 5, method=method, key=jax.random.key(1)
    )
    assert len(traces) == n_traces
    assert n_traces == 1
    assert first.shape == second.shape == (5,)
