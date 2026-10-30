from datetime import date

import pandas as pd
import pytest

from common.config import load_config
from data_pipeline import download_market
from data_pipeline.database import connect, query


def test_asset_table():
    df = download_market.asset_table(load_config()).set_index("ticker")
    assert len(df) == 14
    assert df.loc["SPY", "asset_class"] == "benchmark"
    assert df.loc["TLT", "asset_class"] == "bond" and df.loc["GLD", "asset_class"] == "commodity"
    assert (df.loc[["XLK", "XLRE", "XLC"], "asset_class"] == "sector").all()
    assert df.loc["XLC", "inception"] == pd.Timestamp("2018-06-19")
    assert pd.isna(df.loc["XLK", "inception"])
    assert df["name"].notna().all()


class FakeTicker:
    """Stands in for yfinance.Ticker: returns a short tz-aware history like Yahoo does."""

    def __init__(self, ticker):
        self.ticker = ticker

    def history(self, start, end, auto_adjust, actions):
        assert auto_adjust is True
        idx = pd.DatetimeIndex(["2023-01-03", "2023-01-04"]).tz_localize("America/New_York")
        return pd.DataFrame({"Close": [100.0, 101.0]}, index=idx)


def test_download_prices_into_db(tmp_path, monkeypatch):
    monkeypatch.setattr(download_market.yf, "Ticker", FakeTicker)
    db = tmp_path / "t.db"
    summary = download_market.download_prices(load_config(), date(2023, 1, 1), date(2023, 1, 31), db)
    assert len(summary) == 14

    conn = connect(db)
    prices = query(conn, "SELECT * FROM prices_daily", parse_dates=["date"])
    assert len(prices) == 28
    assert set(prices["date"]) == {pd.Timestamp("2023-01-03"), pd.Timestamp("2023-01-04")}
    assert query(conn, "SELECT COUNT(*) AS n FROM assets")["n"].item() == 14


def test_empty_history_raises(monkeypatch):
    class Empty(FakeTicker):
        def history(self, **kwargs):
            return pd.DataFrame()

    monkeypatch.setattr(download_market.yf, "Ticker", Empty)
    monkeypatch.setattr(download_market.time, "sleep", lambda s: None)
    with pytest.raises(ValueError):
        download_market.fetch_prices("XLK", date(2023, 1, 1), date(2023, 1, 31))
