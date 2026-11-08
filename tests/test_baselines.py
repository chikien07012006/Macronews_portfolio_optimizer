import numpy as np
import pandas as pd
import pytest

from common.config import load_config
from optimization.baselines import (
    bl_no_views,
    equal_weight,
    inverse_volatility,
    markowitz,
    min_variance,
    sample_mean_returns,
)
from optimization.equilibrium_prior import equilibrium_prior
from optimization.optimizer import check_kkt, solve_qp

CFG = load_config()
CORE = list(CFG.universe.core)


def random_cov(assets, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(len(assets), len(assets)))
    cov = 0.004 * (a @ a.T / len(assets) + 0.5 * np.eye(len(assets)))
    return pd.DataFrame(cov, index=assets, columns=assets)


@pytest.fixture
def weekly_excess():
    rng = np.random.default_rng(5)
    dates = pd.date_range("2010-01-08", periods=200, freq="W-FRI")
    return pd.DataFrame(rng.normal(0.001, 0.02, (200, len(CORE))), index=dates, columns=CORE)


def test_equal_weight():
    w = equal_weight(CORE)
    assert list(w.index) == CORE
    np.testing.assert_allclose(w, 1 / 11)


def test_inverse_volatility_uncapped():
    vols = np.array([0.04, 0.05, 0.06, 0.07, 0.08])
    cov = pd.DataFrame(np.diag(vols**2), index=list("ABCDE"), columns=list("ABCDE"))
    w = inverse_volatility(cov, w_max=0.30)
    expected = (1 / vols) / (1 / vols).sum()
    np.testing.assert_allclose(w, expected)
    assert w.max() < 0.30


def test_inverse_volatility_capped_redistributes_proportionally():
    vols = np.array([0.01, 0.05, 0.05, 0.06, 0.08])         # A would get ~57%
    cov = pd.DataFrame(np.diag(vols**2), index=list("ABCDE"), columns=list("ABCDE"))
    w = inverse_volatility(cov, w_max=0.30)
    assert w["A"] == pytest.approx(0.30)
    assert w.sum() == pytest.approx(1.0)
    assert (w <= 0.30 + 1e-12).all()
    rest = w.drop("A")
    inv = 1 / vols[1:]
    np.testing.assert_allclose(rest, 0.70 * inv / inv.sum())   # still 1/sigma among the rest
    assert w["B"] > w["D"] > w["E"]


def test_inverse_volatility_ignores_correlation():
    cov = random_cov(CORE)
    diag_only = pd.DataFrame(np.diag(np.diag(cov)), index=CORE, columns=CORE)
    pd.testing.assert_series_equal(inverse_volatility(cov), inverse_volatility(diag_only))


def test_min_variance_matches_closed_form_when_interior():
    cov = random_cov(CORE)
    inv_one = np.linalg.solve(cov.to_numpy(), np.ones(len(CORE)))
    expected = inv_one / inv_one.sum()                       # Sigma^-1 1 / 1'Sigma^-1 1
    assert ((expected > 0.01) & (expected < 0.29)).all()
    w = min_variance(cov, None, 2.5, 0.30, 0.0)
    np.testing.assert_allclose(w, expected, atol=1e-6)


def test_min_variance_beats_others_on_risk():
    cov = random_cov(CORE)
    var = lambda w: float(w @ cov @ w)
    w_mv = min_variance(cov, None, 2.5, 0.30, 0.0)
    assert var(w_mv) <= var(equal_weight(CORE)) + 1e-12
    assert var(w_mv) <= var(inverse_volatility(cov)) + 1e-12
    mu0 = pd.Series(0.0, index=CORE)
    assert check_kkt(w_mv, mu0, cov, None, 2.5, 0.30, 0.0)


def test_sample_mean_returns_window_and_scaling(weekly_excess):
    as_of = weekly_excess.index[170]
    mu = sample_mean_returns(weekly_excess, as_of, CORE, 156, 4)
    expected = 4 * weekly_excess.iloc[15:171].mean()
    np.testing.assert_allclose(mu, expected)


def test_sample_mean_returns_is_point_in_time(weekly_excess):
    as_of = weekly_excess.index[170]
    changed = weekly_excess.copy()
    changed.iloc[171:] = 1.0                                  # future returns
    pd.testing.assert_series_equal(sample_mean_returns(weekly_excess, as_of, CORE),
                                   sample_mean_returns(changed, as_of, CORE))


def test_sample_mean_returns_short_history(weekly_excess):
    with pytest.raises(ValueError):
        sample_mean_returns(weekly_excess, weekly_excess.index[100], CORE)
    young = weekly_excess.assign(XLRE=np.nan)
    young.iloc[-60:, -1] = 0.002                              # only 60 weeks of XLRE
    mu = sample_mean_returns(young, young.index[-1], CORE + ["XLRE"])
    assert mu["XLRE"] == pytest.approx(4 * 0.002)


def test_markowitz_uses_sample_means(weekly_excess):
    cov = random_cov(CORE)
    as_of = weekly_excess.index[-1]
    w = markowitz(weekly_excess, cov, as_of, None, 2.5, 0.30, 0.001)
    mu = sample_mean_returns(weekly_excess, as_of, CORE)
    pd.testing.assert_series_equal(w, solve_qp(mu, cov, None, 2.5, 0.30, 0.001))
    assert w.sum() == pytest.approx(1.0) and (w <= 0.30 + 1e-9).all()


def test_bl_no_views_stays_near_reference():
    cov = random_cov(CORE)
    _, w_ref = equilibrium_prior(cov, CFG)
    w = bl_no_views(cov, CFG, w_ref)
    assert np.abs(w - w_ref).sum() < 0.05
    pi, _ = equilibrium_prior(cov, CFG)
    expected = solve_qp(pi, (1 + CFG.prior.tau) * cov, w_ref,
                        CFG.optimizer.gamma, CFG.optimizer.w_max, CFG.optimizer.turnover_penalty)
    pd.testing.assert_series_equal(w, expected)
