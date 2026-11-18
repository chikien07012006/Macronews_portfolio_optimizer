# tests

pytest suite. Run from the repository root:

```bash
pytest                      # everything
pytest tests/test_views.py  # one module
```

The tests use synthetic data, temporary SQLite files and fakes for the Yahoo Finance and FRED
calls; they need no network access or downloaded data.

| Area | Modules |
|---|---|
| Configuration and calendar | `test_config.py`, `test_calendar.py` |
| Data pipeline | `test_database.py`, `test_download_market.py`, `test_download_fred.py`, `test_market_data.py` |
| Optimization | `test_ledoit_wolf.py`, `test_prior.py`, `test_black_litterman.py`, `test_optimizer.py`, `test_baselines.py` |
| Simulation | `test_features.py`, `test_views.py`, `test_backtest.py`, `test_metrics.py` |

Look-ahead bias is the main failure mode of a backtest, so anything that touches time is
tested for it: decision and execution dates, document week assignment, feature windows, Ridge
training windows and trade timing.
