from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from common.config import Settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS assets (
    ticker      TEXT PRIMARY KEY,
    name        TEXT,
    asset_class TEXT NOT NULL,          -- sector | bond | commodity | benchmark
    inception   TEXT                    -- first date the asset may be held
);

CREATE TABLE IF NOT EXISTS prices_daily (
    date      TEXT NOT NULL,
    ticker    TEXT NOT NULL REFERENCES assets(ticker),
    adj_close REAL NOT NULL,
    PRIMARY KEY (date, ticker)
);

CREATE TABLE IF NOT EXISTS macro_daily (
    date   TEXT NOT NULL,
    series TEXT NOT NULL,               -- FRED series id, e.g. DTB3
    value  REAL,
    PRIMARY KEY (date, series)
);
"""


def connect(path: str | Path | None = None) -> sqlite3.Connection:
    """Open the project database, creating the file and schema if needed."""
    path = Path(path) if path is not None else Settings.from_env().db_path
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA)
    return conn


def _to_sql_column(col: pd.Series) -> pd.Series:
    if isinstance(col.dtype, pd.DatetimeTZDtype):
        return col.dt.tz_convert("UTC").dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    if pd.api.types.is_datetime64_any_dtype(col):
        fmt = "%Y-%m-%d" if (col.dropna() == col.dropna().dt.normalize()).all() else "%Y-%m-%dT%H:%M:%S"
        return col.dt.strftime(fmt)
    return col


def upsert(conn: sqlite3.Connection, table: str, df: pd.DataFrame) -> int:
    """Insert rows, replacing existing rows with the same primary key. Returns row count."""
    if df.empty:
        return 0
    out = df.apply(_to_sql_column).astype(object).where(df.notna(), None)
    cols = ", ".join(out.columns)
    marks = ", ".join("?" * len(out.columns))
    with conn:
        conn.executemany(
            f"INSERT OR REPLACE INTO {table} ({cols}) VALUES ({marks})",
            out.itertuples(index=False, name=None),
        )
    return len(out)


def query(
    conn: sqlite3.Connection,
    sql: str,
    params: Sequence | dict = (),
    parse_dates: list[str] | None = None,
) -> pd.DataFrame:
    """Run a SELECT and return a DataFrame, parsing the given columns as dates."""
    df = pd.read_sql_query(sql, conn, params=params)
    for col in parse_dates or []:
        df[col] = pd.to_datetime(df[col], utc=df[col].str.endswith("Z").any())
    return df
