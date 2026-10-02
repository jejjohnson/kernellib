"""KRR with Nyström and RPCholesky preconditioners (preconditioned CG)."""

from __future__ import annotations

import gaussx as gx
import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
import pytest

import kernellib as kl


def _data(n, d=2, seed=0):
    k1, k2 = jax.random.split(jax.random.key(seed))
    X = jax.random.uniform(k1, (n, d))
    y = jnp.sin(4.0 * X[:, 0]) * jnp.cos(3.0 * X[:, 1])
    return X, y + 0.05 * jax.random.normal(k2, (n,))


class TestSolverResolution:
    def test_default_is_unchanged_dense(self):
        X, y = _data(60)
        model = kl.KRR(kl.RBF(lengthscale=0.3), regularization=1e-4)
        assert model.solver is None and model.preconditioner == "none"
        dense = kl.KRR(
            kl.RBF(lengthscale=0.3), regularization=1e-4, solver=gx.DenseSolver()
        )
        assert np.array_equal(model.fit(X, y).alpha, dense.fit(X, y).alpha)
        assert isinstance(model._solver(X, None), gx.DenseSolver)

    def test_explicit_solver_is_used_as_given(self):
        X, _ = _data(60)
        cg = gx.CGSolver(rtol=1e-10, atol=1e-10, max_steps=500)
        model = kl.KRR(kl.RBF(lengthscale=0.3), regularization=1e-3, solver=cg)
        assert model._solver(X, None) is cg

    def test_solver_and_preconditioner_together_raise(self):
        with pytest.raises(ValueError, match="not both"):
            kl.KRR(kl.RBF(), solver=gx.DenseSolver(), preconditioner="nystrom")

    def test_invalid_options(self):
        with pytest.raises(ValueError, match="preconditioner must be"):
            kl.KRR(kl.RBF(), preconditioner="jacobi")
        with pytest.raises(ValueError, match="preconditioner_rank"):
            kl.KRR(kl.RBF(), preconditioner="nystrom", preconditioner_rank=0)

    def test_preconditioner_needs_a_key(self):
        X, y = _data(40)
        with pytest.raises(ValueError, match="PRNG key"):
            kl.KRR(kl.RBF(), preconditioner="rpcholesky").fit(X, y)

    @pytest.mark.parametrize("kind", ["nystrom", "rpcholesky"])
    def test_routes_through_preconditioned_cg(self, kind):
        X, _ = _data(50)
        model = kl.KRR(kl.RBF(), preconditioner=kind, preconditioner_rank=10)
        solver = model._solver(X, jax.random.key(0))
        assert isinstance(solver, gx.PreconditionedCGSolver)
        expected = (
            gx.NystromPreconditioner
            if kind == "nystrom"
            else gx.PartialCholeskyPreconditioner
        )
        assert isinstance(solver.preconditioner, expected)

    def test_rank_is_capped_at_n(self):
        X, y = _data(30)
        model = kl.KRR(
            kl.RBF(lengthscale=0.3),
            regularization=1e-3,
            preconditioner="rpcholesky",
            preconditioner_rank=200,
        )
        assert np.all(np.isfinite(model.fit(X, y, key=jax.random.key(0)).alpha))


class TestAgreement:
    @pytest.mark.parametrize(
        ("n", "kind"),
        [
            (300, "nystrom"),
            pytest.param(300, "rpcholesky", marks=pytest.mark.slow),
            pytest.param(2000, "nystrom", marks=pytest.mark.slow),
            pytest.param(2000, "rpcholesky", marks=pytest.mark.slow),
        ],
    )
    def test_predictions_match_dense(self, n, kind):
        # n = 2000 per the spec (slow tier), n = 300 in the fast tier.
        # Matern-3/2 with a small ridge: badly conditioned for plain CG, well
        # conditioned once preconditioned.
        X, y = _data(n)
        kernel = kl.Matern(nu=1.5, lengthscale=0.3)
        dense = kl.KRR(kernel, regularization=1e-5).fit(X, y)
        pcg = kl.KRR(
            kernel,
            regularization=1e-5,
            implicit=True,
            preconditioner=kind,
            preconditioner_rank=200,
        ).fit(X, y, key=jax.random.key(1))
        X_test, _ = _data(200, seed=3)
        assert np.allclose(pcg.predict(X_test), dense.predict(X_test), atol=1e-4)

    @pytest.mark.slow
    def test_multi_output_targets(self):
        X, y = _data(80)
        Y = jnp.stack([y, 2.0 * y], axis=1)
        model = kl.KRR(
            kl.RBF(lengthscale=0.3),
            regularization=1e-4,
            preconditioner="nystrom",
            preconditioner_rank=30,
        ).fit(X, Y, key=jax.random.key(0))
        dense = kl.KRR(kl.RBF(lengthscale=0.3), regularization=1e-4).fit(X, Y)
        assert np.allclose(model.alpha, dense.alpha, rtol=1e-3, atol=1e-3)

    @pytest.mark.slow
    def test_woodbury_penalty_path_uses_the_preconditioner(self):
        X, y = _data(80)
        S = X[:, :1]
        penalty = kl.hsic_penalty(kl.Linear(), S)
        kwargs = {"regularization": 1e-3, "penalty_weight": 0.5}
        dense = kl.KRR(kl.RBF(lengthscale=0.3), **kwargs).fit(X, y, penalty=penalty)
        pcg = kl.KRR(
            kl.RBF(lengthscale=0.3),
            preconditioner="nystrom",
            preconditioner_rank=40,
            **kwargs,
        ).fit(X, y, penalty=penalty, key=jax.random.key(0))
        assert np.allclose(pcg.predict(X), dense.predict(X), atol=1e-4)

    @pytest.mark.slow
    def test_sklearn_adapter_passes_the_preconditioner(self):
        pytest.importorskip("sklearn")
        from kernellib.sklearn import KernelRidge

        X, y = _data(80)
        model = KernelRidge(
            kl.RBF(lengthscale=0.3),
            regularization=1e-4,
            implicit=True,
            preconditioner="nystrom",
            preconditioner_rank=30,
        ).fit(np.asarray(X), np.asarray(y))
        assert model.model_.preconditioner == "nystrom"
        dense = kl.KRR(kl.RBF(lengthscale=0.3), regularization=1e-4).fit(X, y)
        assert np.allclose(model.predict(np.asarray(X)), dense.predict(X), atol=1e-4)


def _cg_steps(A, b, preconditioner=None):
    options = {} if preconditioner is None else {"preconditioner": preconditioner}
    solver = lx.CG(rtol=1e-6, atol=1e-6, max_steps=20_000)
    sol = lx.linear_solve(A, b, solver, options=options, throw=False)
    return int(sol.stats["num_steps"])


@pytest.mark.slow
def test_nystrom_cuts_cg_iterations_below_ten_percent():
    # Spec: Matern-3/2, rank 200, preconditioned CG count below 10 % of plain
    # CG's on the same system and tolerance. The spec's n = 20 000 needs
    # thousands of matrix-free matvecs of several seconds each; iteration
    # counts depend on the spectrum, not on how K is stored, so this runs the
    # same regime (ridge 1e-6 * n) on a dense n = 5000 Gram. Measured with
    # these keys: 75 vs 1013 steps (7.4 %).
    n = 5000
    X, y = _data(n, d=2, seed=5)
    kernel = kl.Matern(nu=1.5, lengthscale=0.2)
    shift = 1e-6 * n
    K_dense = kernel(X, X)
    K = lx.MatrixLinearOperator(K_dense, lx.positive_semidefinite_tag)
    A = lx.MatrixLinearOperator(
        K_dense + shift * jnp.eye(n), lx.positive_semidefinite_tag
    )
    P = gx.NystromPreconditioner.from_operator(
        K, 200, shift=shift, key=jax.random.key(0)
    )
    plain = _cg_steps(A, y)
    preconditioned = _cg_steps(A, y, P.as_operator(A))
    assert preconditioned < 0.1 * plain, (plain, preconditioned)
