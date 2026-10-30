import numpy as np
import pandas as pd
import pytest
from sklearn.covariance import LedoitWolf

from optimization.ledoit_wolf import ledoit_wolf_cov

TICKERS = ["XLB", "XLE", "XLF", "XLK", "TLT"]


def returns(n=300, seed=0, tickers=TICKERS):
    rng = np.random.default_rng(seed)
    k = len(tickers)
    corr = 0.5 * np.ones((k, k)) + 0.5 * np.eye(k)
    vol = np.linspace(0.005, 0.02, k)
    x = rng.multivariate_normal(np.zeros(k), corr * np.outer(vol, vol), size=n)
    return pd.DataFrame(x, index=pd.bdate_range("2020-01-01", periods=n), columns=tickers)


def test_matches_sklearn():
    r = returns()
    res = ledoit_wolf_cov(r, horizon_days=1)
    ref = LedoitWolf().fit(r.iloc[-252:].to_numpy())
    np.testing.assert_allclose(res.cov.to_numpy(), ref.covariance_, rtol=1e-10)
    assert res.shrinkage == pytest.approx(ref.shrinkage_)


def test_components_are_consistent():
    res = ledoit_wolf_cov(returns())
    s = res.sample_cov.to_numpy()
    assert res.nu == pytest.approx(np.trace(s) / len(s))
    expected = (1 - res.shrinkage) * s + res.shrinkage * res.nu * np.eye(len(s))
    np.testing.assert_allclose(res.cov_daily.to_numpy(), expected)
    np.testing.assert_allclose(res.cov.to_numpy(), 20 * expected)
    assert 0 <= res.shrinkage <= 1
    # total variance is preserved, off-diagonal covariances shrink towards 0
    assert np.trace(res.cov_daily) == pytest.approx(np.trace(s))
    off = ~np.eye(len(s), dtype=bool)
    assert (np.abs(res.cov_daily.to_numpy()[off]) <= np.abs(s[off]) + 1e-18).all()


def test_symmetric_positive_definite_and_labelled():
    res = ledoit_wolf_cov(returns())
    assert list(res.cov.index) == TICKERS and list(res.cov.columns) == TICKERS
    np.testing.assert_allclose(res.cov, res.cov.T)
    np.linalg.cholesky(res.cov.to_numpy())


def test_as_of_ignores_later_data():
    r = returns()
    as_of = r.index[270]
    later_changed = r.copy()
    later_changed.loc[later_changed.index > as_of] *= 10
    a = ledoit_wolf_cov(r, as_of=as_of)
    b = ledoit_wolf_cov(later_changed, as_of=as_of)
    pd.testing.assert_frame_equal(a.cov, b.cov)
    expected_window = r.loc[:as_of].iloc[-252:]
    np.testing.assert_allclose(a.sample_cov, np.cov(expected_window.to_numpy(), rowvar=False, bias=True))


def test_only_last_252_days_used():
    r = returns()
    older_changed = r.copy()
    older_changed.iloc[:-252] *= 10
    pd.testing.assert_frame_equal(ledoit_wolf_cov(r).cov, ledoit_wolf_cov(older_changed).cov)


def test_asset_without_full_window_excluded():
    r = returns()
    r.loc[r.index[:-200], "XLK"] = np.nan  # only 200 days of history
    res = ledoit_wolf_cov(r)
    assert "XLK" not in res.cov.index and len(res.cov) == 4


def test_too_little_history_raises():
    with pytest.raises(ValueError):
        ledoit_wolf_cov(returns(n=100))


def test_more_shrinkage_and_better_conditioning_with_short_history():
    tickers = [f"A{i}" for i in range(25)]
    short = ledoit_wolf_cov(returns(n=40, tickers=tickers), window=40)
    long = ledoit_wolf_cov(returns(n=2000, tickers=tickers), window=2000)
    assert short.shrinkage > long.shrinkage
    assert np.linalg.cond(short.cov_daily) < np.linalg.cond(short.sample_cov)
