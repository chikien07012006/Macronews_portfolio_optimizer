from __future__ import annotations

import sqlite3

import pandas as pd

from common.calendar import decision_dates
from data_pipeline.database import query

TRADING_DAYS_PER_YEAR = 252


def _end_clause(end: str | pd.Timestamp | None) -> tuple[str, list[str]]:
    if end is None:
        return "", []
    return " AND date <= ?", [pd.Timestamp(end).strftime("%Y-%m-%d")]


def trading_calendar(conn: sqlite3.Connection, benchmark: str = "SPY") -> pd.DatetimeIndex:
    df = query(conn, "SELECT date FROM prices_daily WHERE ticker = ? ORDER BY date", [benchmark],
               parse_dates=["date"])
    return pd.DatetimeIndex(df["date"], name="date")


def load_prices(conn: sqlite3.Connection, end: str | pd.Timestamp | None = None) -> pd.DataFrame:
    clause, params = _end_clause(end)
    df = query(conn, f"SELECT date, ticker, adj_close FROM prices_daily WHERE 1 = 1{clause}",
               params, parse_dates=["date"])
    prices = df.pivot(index="date", columns="ticker", values="adj_close").sort_index()
    prices.columns.name = None
    return prices


def daily_returns(conn: sqlite3.Connection, end: str | pd.Timestamp | None = None) -> pd.DataFrame:
    return load_prices(conn, end).pct_change(fill_method=None).iloc[1:]


def risk_free_daily(
    conn: sqlite3.Connection, end: str | pd.Timestamp | None = None, series: str = "DTB3"
) -> pd.Series:
    clause, params = _end_clause(end)
    df = query(conn, f"SELECT date, value FROM macro_daily WHERE series = ?{clause}",
               [series, *params], parse_dates=["date"])
    rate = df.set_index("date")["value"].dropna().sort_index()

    days = trading_calendar(conn)
    if end is not None:
        days = days[days <= pd.Timestamp(end)]
    as_of_day = rate.reindex(days.union(rate.index)).ffill().reindex(days)
    rf = (as_of_day.shift(1) / 100 / TRADING_DAYS_PER_YEAR).iloc[1:]
    rf.name = "rf"
    return rf


def _weekly_decisions(conn: sqlite3.Connection, end, frequency: str) -> pd.DatetimeIndex:
    dates = decision_dates(trading_calendar(conn), frequency=frequency)
    return dates if end is None else dates[dates <= pd.Timestamp(end)]


def weekly_returns(
    conn: sqlite3.Connection, end: str | pd.Timestamp | None = None, frequency: str = "weekly"
) -> pd.DataFrame:
    decisions = _weekly_decisions(conn, end, frequency)
    prices = load_prices(conn, end).reindex(decisions)
    out = prices.pct_change(fill_method=None).iloc[1:]
    out.index.name = "decision_date"
    return out


def weekly_risk_free(
    conn: sqlite3.Connection, end: str | pd.Timestamp | None = None, frequency: str = "weekly"
) -> pd.Series:
    decisions = _weekly_decisions(conn, end, frequency)
    rf = risk_free_daily(conn, end)
    week = decisions[decisions.searchsorted(rf.index[rf.index <= decisions[-1]])]
    rf = rf[rf.index <= decisions[-1]]
    out = (1 + rf).groupby(week.to_numpy()).prod() - 1
    out = out.reindex(decisions[1:])
    out.index.name = "decision_date"
    out.name = "rf"
    return out


def weekly_excess_returns(
    conn: sqlite3.Connection, end: str | pd.Timestamp | None = None, frequency: str = "weekly"
) -> pd.DataFrame:
    return weekly_returns(conn, end, frequency).sub(weekly_risk_free(conn, end, frequency), axis=0)
