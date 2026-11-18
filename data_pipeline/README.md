# data_pipeline

Downloads, cleans and stores market data in a single SQLite file (default
`data/macronews.db`, git-ignored), and exposes it to the rest of the project as pandas
objects. Nothing here makes portfolio decisions.

## Commands

Run from the repository root. Both commands default to the dates in
[config.yaml](../config.yaml) (`periods.data_start` to `periods.end`) and are safe to re-run:
rows are upserted on their primary key.

```bash
python -m data_pipeline.download_market   # daily adjusted closes for the ETF universe + SPY (Yahoo Finance)
python -m data_pipeline.download_fred     # 3-month T-bill rate DTB3, the risk-free rate (FRED CSV export)
```

Shared options: `--start YYYY-MM-DD`, `--end YYYY-MM-DD`, `--db PATH`.
`download_fred` also takes `--series` to fetch other FRED series.

## Files

| File | Purpose |
|---|---|
| [database.py](database.py) | SQLite schema, `connect`, `upsert`, `query`. |
| [download_market.py](download_market.py) | Prices via `yfinance` with retries; also fills the `assets` table. |
| [download_fred.py](download_fred.py) | FRED series into `macro_daily`. |
| [market_data.py](market_data.py) | Read side: trading calendar, daily and weekly (excess) returns, risk-free rate. |

## Database tables

| Table | Content |
|---|---|
| `assets` | One row per ticker: name, asset class (sector, bond, commodity, benchmark) and inception date. |
| `prices_daily` | `(date, ticker, adj_close)`. |
| `macro_daily` | `(date, series, value)`, for example DTB3. |
| `backtest_runs`, `backtest_nav`, `backtest_weights` | Saved simulation runs with the configuration they used (written by [simulation](../simulation/)). |

Tables for FOMC documents and news will be added with the text-collection milestones (C5).

Tests: [test_database.py](../tests/test_database.py),
[test_download_market.py](../tests/test_download_market.py),
[test_download_fred.py](../tests/test_download_fred.py),
[test_market_data.py](../tests/test_market_data.py).
