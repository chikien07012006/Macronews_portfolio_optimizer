from __future__ import annotations

import argparse
import time
import warnings
from datetime import date, timedelta

import pandas as pd
import yfinance as yf

from common.config import Config, load_config
from data_pipeline.database import connect, upsert

warnings.filterwarnings("ignore", module="yfinance")

ASSET_NAMES = {
    "XLB": "Materials Select Sector SPDR Fund",
    "XLE": "Energy Select Sector SPDR Fund",
    "XLF": "Financial Select Sector SPDR Fund",
    "XLI": "Industrial Select Sector SPDR Fund",
    "XLK": "Technology Select Sector SPDR Fund",
    "XLP": "Consumer Staples Select Sector SPDR Fund",
    "XLU": "Utilities Select Sector SPDR Fund",
    "XLV": "Health Care Select Sector SPDR Fund",
    "XLY": "Consumer Discretionary Select Sector SPDR Fund",
    "XLRE": "Real Estate Select Sector SPDR Fund",
    "XLC": "Communication Services Select Sector SPDR Fund",
    "TLT": "iShares 20+ Year Treasury Bond ETF",
    "GLD": "SPDR Gold Shares",
    "SPY": "SPDR S&P 500 ETF Trust",
}
ASSET_CLASSES = {"TLT": "bond", "GLD": "commodity"}


def asset_table(cfg: Config) -> pd.DataFrame:
    """One row per ticker for the ``assets`` table."""
    u = cfg.universe
    rows = []
    for ticker in u.all_tickers:
        if ticker == u.benchmark:
            asset_class = "benchmark"
        elif ticker in u.sectors or ticker in u.optional:
            asset_class = "sector"
        else:
            asset_class = ASSET_CLASSES[ticker]
        rows.append({
            "ticker": ticker,
            "name": ASSET_NAMES.get(ticker),
            "asset_class": asset_class,
            "inception": u.optional.get(ticker),  # None: available over the whole sample
        })
    df = pd.DataFrame(rows)
    df["inception"] = pd.to_datetime(df["inception"])
    return df


def fetch_prices(ticker: str, start: date, end: date, retries: int = 3) -> pd.DataFrame:
    """Daily adjusted closes for ``ticker`` on ``[start, end]`` as (date, ticker, adj_close)."""
    for attempt in range(1, retries + 1):
        try:
            hist = yf.Ticker(ticker).history(
                start=start.isoformat(),
                end=(end + timedelta(days=1)).isoformat(),  # yfinance's end is exclusive
                auto_adjust=True,
                actions=False,
            )
            if hist.empty:
                raise ValueError(f"no data returned for {ticker}")
            break
        except Exception:
            if attempt == retries:
                raise
            time.sleep(2**attempt)
    dates = hist.index.tz_localize(None).normalize() if hist.index.tz else hist.index.normalize()
    return pd.DataFrame({"date": dates, "ticker": ticker, "adj_close": hist["Close"].to_numpy()})


def download_prices(cfg: Config, start: date, end: date, db_path: str | None = None) -> pd.DataFrame:
    """Download all tickers into the database and return a per-ticker summary."""
    conn = connect(db_path)
    try:
        upsert(conn, "assets", asset_table(cfg))
        summary = []
        for ticker in cfg.universe.all_tickers:
            prices = fetch_prices(ticker, start, end)
            upsert(conn, "prices_daily", prices)
            summary.append({
                "ticker": ticker,
                "rows": len(prices),
                "first": prices["date"].min().date(),
                "last": prices["date"].max().date(),
            })
            print(f"  {ticker:<5} {len(prices):>5} rows  {summary[-1]['first']} → {summary[-1]['last']}")
        return pd.DataFrame(summary)
    finally:
        conn.close()


def main(argv: list[str] | None = None) -> None:
    cfg = load_config()
    parser = argparse.ArgumentParser(description="Download daily adjusted prices into SQLite.")
    parser.add_argument("--start", type=date.fromisoformat, default=cfg.periods.data_start)
    parser.add_argument("--end", type=date.fromisoformat, default=cfg.periods.end)
    parser.add_argument("--db", default=None, help="SQLite path (default: data/macronews.db)")
    args = parser.parse_args(argv)

    print(f"Downloading {len(cfg.universe.all_tickers)} tickers, {args.start} → {args.end}")
    download_prices(cfg, args.start, args.end, args.db)
    print("Done.")


if __name__ == "__main__":
    main()
