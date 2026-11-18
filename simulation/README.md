# simulation

The week-by-week replay: it builds features and views from point-in-time data, calls
[optimization](../optimization/) for target weights, simulates execution and costs, and
computes metrics and report artifacts.

## Run it

From the repository root, after the data is downloaded (see [data_pipeline](../data_pipeline/)):

```bash
# Development / validation run: preserves the 2018-2023 holdout.
python -m simulation.experiment --start 2013-01-01 --end 2017-12-31 --out reports/validation
python -m simulation.report --out reports/validation
```

| Option (`simulation.experiment`) | Meaning |
|---|---|
| `--start`, `--end` | Backtest window. Defaults: `periods.validation_start` to `periods.end`. |
| `--out DIR` | Output directory; results go to `DIR/results/`. Default: `reports/`. |
| `--only NAME ...` | Run a subset of strategies (names as in the table below). |
| `--save-db` | Also store each run, its configuration, NAV and weights in SQLite. |

`simulation.report --out DIR` reads `DIR/results/` and writes figures (PDF and PNG) to
`DIR/figures/` and tables (CSV and LaTeX) to `DIR/tables/`.

Do not inspect or tune against 2018-2023 while developing: the default window includes it.

## Files

| File | Purpose |
|---|---|
| [features.py](features.py) | Momentum, short-term reversal and volatility, standardized across assets within each week. |
| [views.py](views.py) | Ridge view generator: trains on weeks whose 4-week label is already known at the decision date (purged, expanding window) and returns absolute views `(P, q, Ω)`. Shared by technical and, later, news features. |
| [strategies.py](strategies.py) | The main method and every baseline as a `strategy(t, w_current) -> weights` function, plus the `Inputs` object that caches the covariance per date so all strategies share one risk estimate. |
| [backtest.py](backtest.py) | Replay loop: decide at close of *t*, trade at the next trading-day close, charge costs on turnover (sum of absolute weight changes), mark to market daily; `save_run` persists a run. |
| [metrics.py](metrics.py) | Annualized return and volatility, Sharpe, Sortino, maximum drawdown, CVaR 95%, turnover. |
| [experiment.py](experiment.py) | Command-line runner over all strategies; writes CSVs. |
| [report.py](report.py) | Figures F1-F3 and tables T1-T2 from the experiment CSVs. |

## Strategies

`BL + technical views` (main), `BL (no views)`, `Markowitz`, `Min variance`,
`Inverse volatility`, `Equal weight`, `SPY buy-and-hold`. See the table in the
[root README](../README.md#methods-compared) for what each one isolates.

## Outputs in `results/`

| File | Content |
|---|---|
| `nav_daily.csv` | Daily net asset value, one column per strategy, starting at 1.0. |
| `returns_weekly.csv`, `rf_weekly.csv` | Weekly strategy returns and risk-free rate at decision dates. |
| `turnover.csv` | Turnover (sum of absolute weight changes) at each execution date; the first trade from cash is 1.0. |
| `weights_<strategy>.csv` | Target weights at each execution date. |
| `metrics.csv` | One row per strategy; the source of table T1. |

## Point-in-time rules

At decision date *t* only data up to the close of *t* is used. Ridge labels look ahead by
construction, so training weeks are chosen so that each label is fully realized by *t*. These
rules are covered by [test_backtest.py](../tests/test_backtest.py),
[test_features.py](../tests/test_features.py), [test_views.py](../tests/test_views.py) and
[test_metrics.py](../tests/test_metrics.py).
