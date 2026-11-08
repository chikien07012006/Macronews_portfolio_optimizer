import cvxpy as cp
import numpy as np
import pandas as pd
import pytest

from common.config import load_config
from optimization.equilibrium_prior import equilibrium_prior
from optimization.optimizer import check_kkt, closed_form_weights, solve_qp

CFG = load_config()
CORE = list(CFG.universe.core)
GAMMA = 2.5


def random_cov(assets, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(len(assets), len(assets)))
    cov = 0.004 * (a @ a.T / len(assets) + 0.5 * np.eye(len(assets)))
    return pd.DataFrame(cov, index=assets, columns=assets)


@pytest.fixture
def prior():
    cov = random_cov(CORE)
    pi, w_ref = equilibrium_prior(cov, CFG)
    return cov, pi, w_ref


def test_no_views_returns_reference_weights(prior):
    cov, pi, w_ref = prior
    w = solve_qp(pi, cov, None, GAMMA, 0.30, 0.0)
    np.testing.assert_allclose(w, w_ref, atol=1e-6)
    # already holding w_ref: optimal and nothing to trade
    w = solve_qp(pi, cov, w_ref, GAMMA, 0.30, 0.001)
    np.testing.assert_allclose(w, w_ref, atol=1e-6)


def test_matches_closed_form_when_no_bound_binds(prior):
    cov, pi, _ = prior
    rng = np.random.default_rng(1)
    mu = pi + pd.Series(rng.normal(0, 0.0002, len(CORE)), index=CORE)
    expected = closed_form_weights(mu, cov, GAMMA)
    assert expected.between(0.01, 0.29).all()          # interior: no bound binds
    w = solve_qp(mu, cov, None, GAMMA, 0.30, 0.0)
    np.testing.assert_allclose(w, expected, atol=1e-6)


def test_closed_form_formula():
    cov = pd.DataFrame(0.004 * np.eye(2), index=["A", "B"], columns=["A", "B"])
    mu = pd.Series([0.02, 0.0], index=["A", "B"])
    w = closed_form_weights(mu, cov, GAMMA)
    np.testing.assert_allclose(w, [1.5, -0.5])         # short B, 150% in A
    assert w.sum() == pytest.approx(1.0)


def test_binding_bounds_satisfy_kkt(prior):
    cov, pi, w_ref = prior
    mu = pi.copy()
    mu["XLK"] += 0.05                                  # strong bullish view
    mu["XLE"] -= 0.05                                  # strong bearish view
    w = solve_qp(mu, cov, w_ref, GAMMA, 0.30, 0.001)
    assert w["XLK"] == pytest.approx(0.30, abs=1e-6)
    assert w["XLE"] == pytest.approx(0.0, abs=1e-6)
    assert w.sum() == pytest.approx(1.0)
    assert (w >= 0).all() and (w <= 0.30 + 1e-9).all()
    assert check_kkt(w, mu, cov, w_ref, GAMMA, 0.30, 0.001)


def test_check_kkt_rejects_suboptimal(prior):
    cov, pi, w_ref = prior
    mu = pi.copy()
    mu["XLK"] += 0.05
    assert not check_kkt(w_ref, mu, cov, w_ref, GAMMA, 0.30, 0.001)   # ignores the view
    eq = pd.Series(1 / len(CORE), index=CORE)
    assert not check_kkt(eq, pi, cov, None, GAMMA, 0.30, 0.0)
    assert check_kkt(w_ref, pi, cov, None, GAMMA, 0.30, 0.0)


def test_no_trade_region(prior):
    cov, pi, w_ref = prior
    rng = np.random.default_rng(2)
    mu = pi + pd.Series(rng.normal(0, 0.0005, len(CORE)), index=CORE)
    w_big_lambda = solve_qp(mu, cov, w_ref, GAMMA, 0.30, 0.01)
    np.testing.assert_allclose(w_big_lambda, w_ref, atol=1e-6)        # gain < cost: hold
    w_no_lambda = solve_qp(mu, cov, w_ref, GAMMA, 0.30, 0.0)
    assert np.abs(w_no_lambda - w_ref).sum() > 1e-3                   # would trade without it
    assert check_kkt(w_big_lambda, mu, cov, w_ref, GAMMA, 0.30, 0.01)


def test_turnover_penalty_reduces_trading(prior):
    cov, pi, w_ref = prior
    mu = pi.copy()
    mu["XLK"] += 0.01
    turnover = [np.abs(solve_qp(mu, cov, w_ref, GAMMA, 0.30, lam) - w_ref).sum()
                for lam in (0.0, 0.001, 0.003)]
    assert turnover[0] > turnover[1] > turnover[2]


def test_first_week_is_plain_mean_variance(prior):
    cov, pi, _ = prior
    mu = pi.copy()
    mu["XLK"] += 0.02
    first = solve_qp(mu, cov, None, GAMMA, 0.30, 0.001)
    no_penalty = solve_qp(mu, cov, None, GAMMA, 0.30, 0.0)
    np.testing.assert_allclose(first, no_penalty, atol=1e-6)


def test_new_asset_and_label_alignment():
    old_cov = random_cov(CORE, seed=3)
    old = solve_qp(equilibrium_prior(old_cov, CFG)[0], old_cov)        # last week: no XLRE
    assets = CORE + ["XLRE"]
    cov = random_cov(assets, seed=3)
    pi, _ = equilibrium_prior(cov, CFG)
    w = solve_qp(pi, cov, old.iloc[::-1], GAMMA, 0.30, 0.001)          # shuffled labels
    assert list(w.index) == assets
    assert w["XLRE"] > 0
    assert check_kkt(w, pi, cov, old, GAMMA, 0.30, 0.001)


def test_solver_failure_keeps_current_weights(prior, monkeypatch):
    cov, pi, w_ref = prior

    def fail(self, *args, **kwargs):
        raise cp.SolverError("boom")

    monkeypatch.setattr(cp.Problem, "solve", fail)
    w_prev = w_ref.drop("GLD")                                         # GLD not held
    w = solve_qp(pi, cov, w_prev, GAMMA, 0.30, 0.001)
    assert w["GLD"] == 0.0
    pd.testing.assert_series_equal(w.drop("GLD"), w_prev.rename("weight"), check_names=False)
    with pytest.raises(RuntimeError):
        solve_qp(pi, cov, None, GAMMA, 0.30, 0.001)


def test_inaccurate_status_counts_as_failure(prior, monkeypatch):
    cov, pi, w_ref = prior

    def inaccurate(self, *args, **kwargs):
        self._status = cp.OPTIMAL_INACCURATE

    monkeypatch.setattr(cp.Problem, "solve", inaccurate)
    np.testing.assert_allclose(solve_qp(pi, cov, w_ref), w_ref)


def test_infeasible_cap_rejected():
    cov = random_cov(["A", "B", "C"])
    mu = pd.Series(0.0, index=cov.index)
    with pytest.raises(ValueError):
        solve_qp(mu, cov, None, GAMMA, 0.30, 0.001)


def test_mismatched_assets_rejected(prior):
    cov, pi, _ = prior
    with pytest.raises(ValueError):
        solve_qp(pi.drop("XLK"), cov)
