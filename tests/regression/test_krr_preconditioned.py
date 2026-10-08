"""KRR with Nyström and RPCholesky preconditioners (preconditioned CG)."""

from __future__ import annotations

import dataclasses
import warnings

import equinox as eqx
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


class TestIterationStatistics:
    def test_preconditioned_fit_records_n_iter_and_convergence(self):
        X, y = _data(200)
        model = kl.KRR(
            kl.RBF(lengthscale=0.3),
            regularization=1e-4,
            preconditioner="nystrom",
            preconditioner_rank=40,
            max_steps=500,
        ).fit(X, y, key=jax.random.key(0))
        assert model.n_iter is not None and model.n_iter.shape == ()
        assert 0 < int(model.n_iter) < model.max_steps
        assert bool(model.converged)

    def test_multi_output_statistics_are_per_column(self):
        X, y = _data(60)
        Y = jnp.stack([y, 2.0 * y, -y], axis=1)
        model = kl.KRR(
            kl.RBF(lengthscale=0.3), regularization=1e-3, solver=gx.CGSolver()
        ).fit(X, Y)
        assert model.n_iter.shape == (3,) and model.converged.shape == (3,)
        assert bool(jnp.all(model.converged))

    def test_direct_solve_has_no_statistics(self):
        X, y = _data(40)
        model = kl.KRR(kl.RBF(lengthscale=0.3), regularization=1e-3).fit(X, y)
        assert model.n_iter is None and model.converged is None

    def test_throw_false_returns_the_last_iterate_unconverged(self):
        # Plain CG at a tiny ridge needs far more than 20 steps.
        X, y = _data(100)
        kwargs = {
            "regularization": 1e-12,
            "solver": gx.CGSolver(rtol=1e-8, atol=1e-8, max_steps=20),
        }
        with pytest.raises(Exception, match=r"max_steps|maximum number of"):
            kl.KRR(kl.RBF(lengthscale=0.3), **kwargs).fit(X, y)
        with pytest.warns(RuntimeWarning, match="did not converge"):
            model = kl.KRR(kl.RBF(lengthscale=0.3), throw=False, **kwargs).fit(X, y)
        assert not bool(model.converged)
        assert int(model.n_iter) == 20
        assert np.all(np.isfinite(model.alpha))

    def test_tol_and_max_steps_reach_the_preconditioned_solver(self):
        X, _ = _data(50)
        model = kl.KRR(
            kl.RBF(),
            preconditioner="rpcholesky",
            preconditioner_rank=10,
            tol=1e-4,
            max_steps=7,
        )
        solver = model._solver(X, jax.random.key(0))
        assert (solver.rtol, solver.atol, solver.max_steps) == (1e-4, 1e-4, 7)

    def test_invalid_tol_and_max_steps(self):
        with pytest.raises(ValueError, match="tol"):
            kl.KRR(kl.RBF(), tol=0.0)
        with pytest.raises(ValueError, match="max_steps"):
            kl.KRR(kl.RBF(), max_steps=0)

    def test_woodbury_path_records_statistics(self):
        X, y = _data(60)
        penalty = kl.hsic_penalty(kl.Linear(), X[:, :1])
        model = kl.KRR(
            kl.RBF(lengthscale=0.3),
            regularization=1e-3,
            penalty_weight=0.5,
            solver=gx.CGSolver(),
        ).fit(X, y, penalty=penalty)
        assert int(model.n_iter) > 0 and bool(model.converged)

    def test_gmres_path_records_statistics(self):
        X, y = _data(40)
        mask = jnp.arange(40) % 2 == 0
        model = kl.KRR(kl.RBF(lengthscale=0.3), regularization=1e-3, implicit=True).fit(
            X, y, mask=mask
        )
        assert int(model.n_iter) > 0 and bool(model.converged)

    def test_gmres_takes_tol_and_max_steps(self, monkeypatch):
        built = []

        def spy(**kwargs):
            built.append(kwargs)
            return gmres(**kwargs)

        gmres = lx.GMRES
        monkeypatch.setattr(lx, "GMRES", spy)
        X, y = _data(40)
        mask = jnp.arange(40) % 2 == 0
        kwargs = {"regularization": 1e-3, "implicit": True}
        kl.KRR(kl.RBF(lengthscale=0.3), tol=1e-9, max_steps=30, **kwargs).fit(
            X, y, mask=mask
        )
        # An explicit solver with tolerances keeps them, as on the CG paths.
        solver = gx.CGSolver(rtol=1e-7, atol=1e-8, max_steps=40)
        kl.KRR(kl.RBF(lengthscale=0.3), solver=solver, max_steps=1, **kwargs).fit(
            X, y, mask=mask
        )
        got = [(b["rtol"], b["atol"], b["max_steps"]) for b in built]
        assert got == [(1e-9, 1e-9, 30), (1e-7, 1e-8, 40)]

    def test_gmres_stops_at_max_steps(self):
        # lineax's first GMRES step is a dummy pass and convergence needs a
        # small update, so two steps can never converge.
        X, y = _data(40)
        mask = jnp.arange(40) % 2 == 0
        kwargs = {"regularization": 1e-3, "implicit": True, "max_steps": 2}
        with pytest.raises(Exception, match=r"max_steps|maximum number of"):
            kl.KRR(kl.RBF(lengthscale=0.3), **kwargs).fit(X, y, mask=mask)
        with pytest.warns(RuntimeWarning, match="did not converge"):
            model = kl.KRR(kl.RBF(lengthscale=0.3), throw=False, **kwargs).fit(
                X, y, mask=mask
            )
        assert not bool(model.converged)
        assert int(model.n_iter) == 2
        assert np.all(np.isfinite(model.alpha))


def _float32_data(n):
    """1-D float32 data: the suite runs in x64, so cast explicitly."""
    X = jax.random.uniform(jax.random.key(0), (n, 1), dtype=jnp.float32)
    return X, jnp.sin(6.0 * X[:, 0])


def _relative_residual(model, X, y):
    lam_n = model.regularization * X.shape[0]
    r = model.kernel(X, X) @ model.alpha + lam_n * model.alpha - y
    return float(jnp.linalg.norm(r) / jnp.linalg.norm(y))


_RPCHOLESKY = {"preconditioner": "rpcholesky", "preconditioner_rank": 30}


class TestLowPrecision:
    """#162: float32 at a tiny ridge must not return garbage silently."""

    @pytest.mark.slow
    def test_regularization_below_the_dtype_floor_warns(self):
        X, y = _float32_data(30)
        # The warning comes before the solve, which here proves it right:
        # the float32 Cholesky of the numerically singular system fails.
        with (
            pytest.warns(RuntimeWarning, match="float32 precision floor"),
            pytest.raises(eqx.EquinoxRuntimeError, match="non-finite"),
        ):
            kl.KRR(kl.RBF(lengthscale=0.2), regularization=1e-9).fit(X, y)
        # The same ridge is far above float64's floor.
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            kl.KRR(kl.RBF(lengthscale=0.2), regularization=1e-9).fit(
                X.astype(jnp.float64), y.astype(jnp.float64)
            )

    @pytest.mark.slow
    def test_float32_tiny_ridge_raises_instead_of_returning_garbage(self):
        # Before #162, CG reported convergence here on weights whose
        # residual was of the order of ||y||.
        X, y = _float32_data(300)
        model = kl.KRR(kl.RBF(lengthscale=0.2), regularization=1e-9, **_RPCHOLESKY)
        with (
            pytest.warns(RuntimeWarning, match="precision floor"),
            pytest.raises(eqx.EquinoxRuntimeError, match="residual"),
        ):
            model.fit(X, y, key=jax.random.key(2))

    @pytest.mark.slow
    def test_float32_tiny_ridge_without_throw_reports_unconverged(self):
        X, y = _float32_data(300)
        model = kl.KRR(
            kl.RBF(lengthscale=0.2), regularization=1e-9, throw=False, **_RPCHOLESKY
        )
        with pytest.warns(RuntimeWarning) as record:
            fitted = model.fit(X, y, key=jax.random.key(2))
        messages = " ".join(str(w.message) for w in record)
        assert "precision floor" in messages and "did not converge" in messages
        assert not bool(fitted.converged)
        assert _relative_residual(fitted, X, y) > 1e-2

    @pytest.mark.slow
    def test_float32_moderate_ridge_is_residual_checked_and_converged(self):
        X, y = _float32_data(300)
        model = kl.KRR(kl.RBF(lengthscale=0.2), regularization=1e-3, **_RPCHOLESKY)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            fitted = model.fit(X, y, key=jax.random.key(2))
        assert fitted.alpha.dtype == jnp.float32
        assert bool(fitted.converged)
        assert _relative_residual(fitted, X, y) < 1e-4

    @pytest.mark.slow
    def test_residual_check_fires_on_a_bad_solve(self, monkeypatch):
        # A CG solve that reports success on a wrong solution: the residual
        # check must catch it, whatever CG's own bookkeeping says.
        real_solve = lx.linear_solve

        def bad_solve(*args, **kwargs):
            solution = real_solve(*args, **kwargs)
            return eqx.tree_at(lambda s: s.value, solution, solution.value + 1.0)

        monkeypatch.setattr(lx, "linear_solve", bad_solve)
        X, y = _data(40)
        solver = gx.CGSolver(rtol=1e-8, atol=1e-8, max_steps=500)
        model = kl.KRR(kl.RBF(lengthscale=0.3), regularization=1e-3, solver=solver)
        with pytest.raises(eqx.EquinoxRuntimeError, match="residual"):
            model.fit(X, y)
        with pytest.warns(RuntimeWarning, match="did not converge"):
            fitted = dataclasses.replace(model, throw=False).fit(X, y)
        assert not bool(fitted.converged)

    def test_residual_check_passes_a_sound_solve(self):
        X, y = _data(40)
        solver = gx.CGSolver(rtol=1e-8, atol=1e-8, max_steps=500)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            fitted = kl.KRR(
                kl.RBF(lengthscale=0.3), regularization=1e-3, solver=solver
            ).fit(X, y)
        assert bool(fitted.converged)
