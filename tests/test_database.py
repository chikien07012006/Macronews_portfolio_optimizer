import pandas as pd
import pytest

from data_pipeline.database import connect, query, upsert


@pytest.fixture
def conn(tmp_path):
    c = connect(tmp_path / "test.db")
    upsert(c, "assets", pd.DataFrame({"ticker": ["XLK", "TLT"], "asset_class": ["sector", "bond"]}))
    yield c
    c.close()


def prices(values):
    return pd.DataFrame({
        "date": pd.to_datetime(["2023-01-03", "2023-01-03", "2023-01-04"]),
        "ticker": ["XLK", "TLT", "XLK"],
        "adj_close": values,
    })


def test_creates_schema(conn):
    tables = set(query(conn, "SELECT name FROM sqlite_master WHERE type = 'table'")["name"])
    assert {"assets", "prices_daily", "macro_daily"} <= tables


def test_upsert_round_trip_and_replace(conn):
    assert upsert(conn, "prices_daily", prices([100.0, 90.0, 101.0])) == 3
    upsert(conn, "prices_daily", prices([100.0, 90.0, 102.0]))  # same keys: replaced, not duplicated
    df = query(conn, "SELECT * FROM prices_daily ORDER BY date, ticker", parse_dates=["date"])
    assert len(df) == 3
    assert df["date"].iloc[0] == pd.Timestamp("2023-01-03")
    assert df.loc[df["date"] == "2023-01-04", "adj_close"].item() == 102.0


def test_point_in_time_filter_by_date_string(conn):
    upsert(conn, "prices_daily", prices([100.0, 90.0, 101.0]))
    df = query(conn, "SELECT * FROM prices_daily WHERE date <= ?", ["2023-01-03"])
    assert set(df["date"]) == {"2023-01-03"}


def test_missing_values_stored_as_null(conn):
    upsert(conn, "macro_daily", pd.DataFrame({
        "date": pd.to_datetime(["2023-01-02"]), "series": ["DTB3"], "value": [float("nan")]}))
    assert query(conn, "SELECT value FROM macro_daily")["value"].isna().all()


def test_tz_aware_timestamps_stored_in_utc(tmp_path):
    c = connect(tmp_path / "t.db")
    c.execute("CREATE TABLE docs (id INTEGER PRIMARY KEY, ts TEXT)")
    ts = pd.Series([pd.Timestamp("2023-03-10 16:00", tz="America/New_York")])
    upsert(c, "docs", pd.DataFrame({"id": [1], "ts": ts}))
    assert query(c, "SELECT ts FROM docs")["ts"].item() == "2023-03-10T21:00:00Z"
    back = query(c, "SELECT ts FROM docs", parse_dates=["ts"])["ts"].item()
    assert back == ts.item()


def test_foreign_key_enforced(conn):
    bad = pd.DataFrame({"date": ["2023-01-03"], "ticker": ["NOPE"], "adj_close": [1.0]})
    with pytest.raises(Exception):
        upsert(conn, "prices_daily", bad)
