import numpy as np
import pandas as pd
import pytest

from common.config import load_config
from simulation.features import (
    cross_sectional_zscore,
    momentum,
    technical_features,
    volatility,
)

TICKERS = ["XLB", "XLE", "XLK", "TLT", "GLD"]


def market(n_days=400, seed=0, tickers=TICKERS):
    """Random daily returns and weekly excess returns on the Fridays of the same calendar."""
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2020-01-01", periods=n_days)
    vol = np.linspace(0.005, 0.02, len(tickers))
    daily = pd.DataFrame(rng.normal(0, vol, size=(n_days, len(tickers))),
                         index=days, columns=tickers)
    fridays = pd.DatetimeIndex(days[days.weekday == 4], name="decision_date")
    weekly = pd.DataFrame(rng.normal(0.001, 0.02, size=(len(fridays), len(tickers))),
                          index=fridays, columns=tickers)
    return weekly, daily


@pytest.fixture
def cfg():
    return load_config()


def test_momentum_matches_loop():
    weekly, _ = market()
    mom = momentum(weekly, weeks=12, skip=1)
    x = weekly.to_numpy()
    for p in range(len(weekly)):
        if p < 11:
            assert mom.iloc[p].isna().all()
        else:
            expected = np.prod(1 + x[p - 11:p], axis=0) - 1   # weeks t-11 .. t-1
            np.testing.assert_allclose(mom.iloc[p].to_numpy(), expected, rtol=1e-12, atol=1e-14)


def test_momentum_skips_latest_week():
    weekly, _ = market()
    t = weekly.index[30]
    bumped = weekly.copy()
    bumped.loc[t] = 0.5
    assert momentum(bumped).loc[t].equals(momentum(weekly).loc[t])
    assert not momentum(bumped).loc[weekly.index[31]].equals(momentum(weekly).loc[weekly.index[31]])


def test_volatility_matches_loop():
    weekly, daily = market()
    vol = volatility(daily, weekly.index, days=20)
    for t in weekly.index[5:10]:
        window = daily.loc[:t].iloc[-20:]
        assert window.index[-1] == t
        expected = window.std(ddof=1).to_numpy() * np.sqrt(252)
        np.testing.assert_allclose(vol.loc[t].to_numpy(), expected, rtol=1e-12)
    assert vol.loc[weekly.index[0]].isna().all()   # first Friday: fewer than 20 days


def test_zscore_rows():
    values = pd.DataFrame([[1.0, 2.0, 3.0], [5.0, 5.0, 5.0], [1.0, np.nan, 3.0]])
    z = cross_sectional_zscore(values)
    np.testing.assert_allclose(z.iloc[0], [-1.0, 0.0, 1.0])
    assert z.iloc[1].isna().all()                 # no dispersion
    np.testing.assert_allclose(z.iloc[2, [0, 2]], [-1 / np.sqrt(2), 1 / np.sqrt(2)])


def test_features_are_standardized_raw_features(cfg):
    weekly, daily = market()
    feats = technical_features(weekly, daily, cfg)
    raw = {"MOM": momentum(weekly), "REV": weekly, "VOL": volatility(daily, weekly.index)}
    for name, table in raw.items():
        expected = cross_sectional_zscore(table).stack()
        got = feats[name]
        np.testing.assert_allclose(got.to_numpy(), expected.reindex(got.index).to_numpy())


def test_shape_and_index(cfg):
    weekly, daily = market()
    feats = technical_features(weekly, daily, cfg)
    assert list(feats.columns) == ["MOM", "REV", "VOL"]
    assert feats.index.names == ["decision_date", "ticker"]
    assert not feats.isna().any().any()
    dates = feats.index.get_level_values("decision_date").unique()
    assert dates[0] == weekly.index[11]           # first week with 11 weeks of momentum
    assert set(feats.index.get_level_values("ticker")) == set(TICKERS)


def test_each_week_has_mean_zero_std_one(cfg):
    weekly, daily = market()
    feats = technical_features(weekly, daily, cfg)
    by_week = feats.groupby(level="decision_date")
    np.testing.assert_allclose(by_week.mean().to_numpy(), 0.0, atol=1e-12)
    np.testing.assert_allclose(by_week.std().to_numpy(), 1.0, rtol=1e-12)


def test_no_look_ahead(cfg):
    weekly, daily = market()
    t = weekly.index[40]
    before = technical_features(weekly, daily, cfg)

    rng = np.random.default_rng(99)
    weekly2, daily2 = weekly.copy(), daily.copy()
    weekly2.loc[weekly2.index > t] = rng.uniform(-0.3, 0.3, size=weekly2.loc[weekly2.index > t].shape)
    daily2.loc[daily2.index > t] = rng.uniform(-0.3, 0.3, size=daily2.loc[daily2.index > t].shape)
    after = technical_features(weekly2, daily2, cfg)

    pd.testing.assert_frame_equal(before.loc[:t], after.loc[:t])
    assert not before.loc[weekly.index[41]:].equals(after.loc[weekly.index[41]:])


def test_asset_enters_when_history_is_complete(cfg):
    weekly, daily = market()
    late = weekly.index[30]
    weekly.loc[weekly.index < late, "GLD"] = np.nan
    daily.loc[daily.index < late, "GLD"] = np.nan
    feats = technical_features(weekly, daily, cfg)
    gld_dates = feats.xs("GLD", level="ticker").index
    assert gld_dates[0] == weekly.index[30 + 11]   # MOM needs weeks t-11 .. t-1 to exist
    # weeks without GLD are still standardized over the other four assets
    week = feats.loc[weekly.index[35]]
    assert list(week.index) == ["XLB", "XLE", "XLK", "TLT"]
    np.testing.assert_allclose(week.std().to_numpy(), 1.0)


def test_windows_come_from_config():
    weekly, daily = market()
    cfg = load_config(overrides={"features.mom_weeks": 6, "features.vol_days": 10})
    feats = technical_features(weekly, daily, cfg)
    assert feats.index.get_level_values("decision_date")[0] == weekly.index[5]


def test_missing_daily_column_rejected(cfg):
    weekly, daily = market()
    with pytest.raises(ValueError, match="TLT"):
        technical_features(weekly, daily.drop(columns="TLT"), cfg)
