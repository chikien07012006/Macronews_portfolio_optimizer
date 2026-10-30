import numpy as np
import pandas as pd
import pytest

from data_pipeline import market_data as md
from data_pipeline.database import connect, upsert

# Trading days 2023-03-24 .. 2023-04-21, Good Friday 2023-04-07 closed
DAYS = pd.bdate_range("2023-03-24", "2023-04-21").drop(pd.Timestamp("2023-04-07"))
XLRE_START = pd.Timestamp("2023-04-03")  # stands in for an asset that starts trading mid-sample


@pytest.fixture
def conn(tmp_path):
    c = connect(tmp_path / "t.db")
    upsert(c, "assets", pd.DataFrame({"ticker": ["SPY", "XLK", "XLRE"],
                                      "asset_class": ["benchmark", "sector", "sector"]}))
    rng = np.random.default_rng(1)
    rows = []
    for ticker in ["SPY", "XLK", "XLRE"]:
        days = DAYS[DAYS >= XLRE_START] if ticker == "XLRE" else DAYS
        price = 100 * np.cumprod(1 + rng.normal(0, 0.01, len(days)))
        rows.append(pd.DataFrame({"date": days, "ticker": ticker, "adj_close": price}))
    upsert(c, "prices_daily", pd.concat(rows))

    # DTB3 on every weekday (incl. Good Friday), rising 0.01 per day, two days missing
    fred_days = pd.bdate_range("2023-03-24", "2023-04-21")
    value = 4.50 + 0.01 * np.arange(len(fred_days))
    dtb3 = pd.DataFrame({"date": fred_days, "series": "DTB3", "value": value})
    dtb3.loc[dtb3["date"].isin(pd.to_datetime(["2023-03-30", "2023-04-12"])), "value"] = np.nan
    upsert(c, "macro_daily", dtb3)
    yield c
    c.close()


def dtb3(conn, day):
    return md.query(conn, "SELECT value FROM macro_daily WHERE date = ?", [day])["value"].item()


class TestDaily:
    def test_daily_returns_are_price_ratios(self, conn):
        p = md.load_prices(conn)
        r = md.daily_returns(conn)
        d = pd.Timestamp("2023-03-29")
        prev = pd.Timestamp("2023-03-28")
        assert r.loc[d, "XLK"] == pytest.approx(p.loc[d, "XLK"] / p.loc[prev, "XLK"] - 1)

    def test_no_return_before_or_on_first_trading_day(self, conn):
        r = md.daily_returns(conn)
        assert r.loc[r.index <= XLRE_START, "XLRE"].isna().all()
        assert r.loc[r.index > XLRE_START, "XLRE"].notna().all()

    def test_risk_free_uses_previous_day_rate(self, conn):
        rf = md.risk_free_daily(conn)
        assert rf.index.equals(DAYS[1:])
        assert rf.loc["2023-03-29"] == pytest.approx(dtb3(conn, "2023-03-28") / 100 / 252)

    def test_risk_free_fills_missing_rate(self, conn):
        rf = md.risk_free_daily(conn)
        # 03-30 has no rate, so the return on 03-31 uses the 03-29 rate
        assert rf.loc["2023-03-31"] == pytest.approx(dtb3(conn, "2023-03-29") / 100 / 252)

    def test_risk_free_after_holiday_uses_last_trading_day(self, conn):
        rf = md.risk_free_daily(conn)
        # Monday after Good Friday: the previous trading day is Thursday 04-06
        assert rf.loc["2023-04-10"] == pytest.approx(dtb3(conn, "2023-04-06") / 100 / 252)


class TestWeekly:
    def test_decision_dates_include_good_friday_thursday(self, conn):
        w = md.weekly_returns(conn)
        assert pd.Timestamp("2023-04-06") in w.index
        assert pd.Timestamp("2023-04-07") not in w.index
        assert w.index[0] == pd.Timestamp("2023-03-31")  # first week (ending 03-24) dropped

    def test_weekly_return_is_compounded_daily_return(self, conn):
        w = md.weekly_returns(conn)
        r = md.daily_returns(conn)
        week = r.loc["2023-04-10":"2023-04-14", "XLK"]
        assert w.loc["2023-04-14", "XLK"] == pytest.approx((1 + week).prod() - 1)

    def test_weekly_risk_free_compounds_the_same_days(self, conn):
        rf = md.risk_free_daily(conn)
        wrf = md.weekly_risk_free(conn)
        holiday_week = rf.loc["2023-04-03":"2023-04-06"]  # 4 trading days
        assert wrf.loc["2023-04-06"] == pytest.approx((1 + holiday_week).prod() - 1)
        assert wrf.index.equals(md.weekly_returns(conn).index)

    def test_excess_is_return_minus_risk_free(self, conn):
        x = md.weekly_excess_returns(conn)
        w = md.weekly_returns(conn)
        wrf = md.weekly_risk_free(conn)
        pd.testing.assert_frame_equal(x, w.sub(wrf, axis=0))

    def test_asset_first_week_is_nan(self, conn):
        w = md.weekly_returns(conn)
        assert np.isnan(w.loc["2023-04-06", "XLRE"])  # no price at the previous decision date
        assert not np.isnan(w.loc["2023-04-14", "XLRE"])


@pytest.mark.parametrize("end", ["2023-04-06", "2023-04-12", "2023-04-14"])
def test_point_in_time(conn, end):
    """Data loaded up to ``end`` give the same values as the full data cut at ``end``."""
    cut = pd.Timestamp(end)
    pd.testing.assert_frame_equal(md.daily_returns(conn, end), md.daily_returns(conn).loc[:cut])
    pd.testing.assert_series_equal(md.risk_free_daily(conn, end), md.risk_free_daily(conn).loc[:cut])
    full = md.weekly_excess_returns(conn)
    pd.testing.assert_frame_equal(md.weekly_excess_returns(conn, end), full.loc[:cut])
