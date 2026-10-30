from __future__ import annotations

import argparse
import io
import time
from datetime import date

import pandas as pd
import requests

from common.config import load_config
from data_pipeline.database import connect, upsert

FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
DEFAULT_SERIES = ("DTB3",)  # 3-month Treasury bill rate, % per year: the risk-free rate


def fetch_series(series: str, start: date, end: date, retries: int = 3) -> pd.DataFrame:
    params = {"id": series, "cosd": start.isoformat(), "coed": end.isoformat()}
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(FRED_CSV_URL, params=params, timeout=30)
            resp.raise_for_status()
            break
        except requests.RequestException:
            if attempt == retries:
                raise
            time.sleep(2**attempt)

    raw = pd.read_csv(io.StringIO(resp.text), na_values=["."])
    if series not in raw.columns:
        raise ValueError(f"unexpected FRED response for {series}: columns {list(raw.columns)}")
    dates = pd.to_datetime(raw.iloc[:, 0])  # first column: observation_date (or DATE)
    df = pd.DataFrame({"date": dates, "series": series, "value": pd.to_numeric(raw[series])})
    df = df[(df["date"] >= pd.Timestamp(start)) & (df["date"] <= pd.Timestamp(end))]
    if df["value"].notna().sum() == 0:
        raise ValueError(f"no observations returned for {series}")
    return df.reset_index(drop=True)


def download_fred(
    series: tuple[str, ...], start: date, end: date, db_path: str | None = None
) -> pd.DataFrame:
    """Download FRED series into ``macro_daily`` and return a per-series summary."""
    conn = connect(db_path)
    try:
        summary = []
        for s in series:
            df = fetch_series(s, start, end)
            upsert(conn, "macro_daily", df)
            obs = df.dropna(subset=["value"])
            summary.append({
                "series": s,
                "rows": len(df),
                "missing": len(df) - len(obs),
                "first": obs["date"].min().date(),
                "last": obs["date"].max().date(),
            })
            row = summary[-1]
            print(f"  {s:<6} {row['rows']:>5} rows ({row['missing']} missing)  "
                  f"{row['first']} → {row['last']}  range {obs['value'].min():.2f}–{obs['value'].max():.2f}%")
        return pd.DataFrame(summary)
    finally:
        conn.close()


def main(argv: list[str] | None = None) -> None:
    cfg = load_config()
    parser = argparse.ArgumentParser(description="Download FRED series (risk-free rate) into SQLite.")
    parser.add_argument("--series", nargs="+", default=list(DEFAULT_SERIES))
    parser.add_argument("--start", type=date.fromisoformat, default=cfg.periods.data_start)
    parser.add_argument("--end", type=date.fromisoformat, default=cfg.periods.end)
    parser.add_argument("--db", default=None, help="SQLite path (default: data/macronews.db)")
    args = parser.parse_args(argv)

    print(f"Downloading FRED series {', '.join(args.series)}, {args.start} → {args.end}")
    download_fred(tuple(args.series), args.start, args.end, args.db)
    print("Done.")


if __name__ == "__main__":
    main()
