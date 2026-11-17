"""Run the main method and all baselines through the weekly backtest and save the results.

    python -m simulation.experiment                       # 2013-01-01 .. 2023-12-31
    python -m simulation.experiment --start 2013-01-01 --end 2017-12-31 --out reports/validation

Writes to ``<out>/results/``: daily NAV, weekly returns at decision dates, weekly risk-free
rate, turnover, metrics and the target weights of every strategy. Figures and tables are
made from these files by ``python -m simulation.report``.
"""
from __future__ import annotations

import argparse
import logging
import re
import time
from pathlib import Path

import pandas as pd

from common.config import PROJECT_ROOT, Settings, load_config
from data_pipeline.database import connect
from data_pipeline.market_data import weekly_risk_free
from simulation.backtest import BacktestResult, run_backtest, save_run
from simulation.metrics import compute_metrics
from simulation.strategies import STRATEGIES, Inputs

log = logging.getLogger(__name__)


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def run_all(conn, cfg, start: str, end: str, names: list[str] | None = None
            ) -> dict[str, BacktestResult]:
    inputs = Inputs(conn, cfg)
    results = {}
    for name in names or list(STRATEGIES):
        t0 = time.perf_counter()
        results[name] = run_backtest(conn, STRATEGIES[name](inputs), cfg, start, end, name)
        log.info("%-22s %6.1f s", name, time.perf_counter() - t0)
    return results


def save_results(results: dict[str, BacktestResult], rf_weekly: pd.Series, out: Path
                 ) -> pd.DataFrame:
    out.mkdir(parents=True, exist_ok=True)
    first = next(iter(results.values()))
    decisions = first.decisions

    nav = pd.DataFrame({name: r.nav for name, r in results.items()})
    weekly = nav.reindex(decisions).pct_change().iloc[1:]
    turnover = pd.DataFrame({name: r.turnover for name, r in results.items()})
    metrics = pd.DataFrame({
        name: compute_metrics(r.nav, rf_weekly, r.turnover, r.decisions)
        for name, r in results.items()
    }).T

    nav.rename_axis("date").to_csv(out / "nav_daily.csv")
    weekly.rename_axis("decision_date").to_csv(out / "returns_weekly.csv")
    rf_weekly.reindex(decisions).rename("rf").rename_axis("decision_date").to_csv(out / "rf_weekly.csv")
    turnover.rename_axis("execution_date").to_csv(out / "turnover.csv")
    metrics.rename_axis("strategy").to_csv(out / "metrics.csv")
    for name, r in results.items():
        r.weights.to_csv(out / f"weights_{slug(name)}.csv")
    return metrics


def main(argv: list[str] | None = None) -> None:
    cfg = load_config()
    parser = argparse.ArgumentParser(description="Backtest the main method and the baselines.")
    parser.add_argument("--start", default=str(cfg.periods.validation_start))
    parser.add_argument("--end", default=str(cfg.periods.end))
    parser.add_argument("--out", default=str(PROJECT_ROOT / "reports"),
                        help="output folder (results/ is created inside)")
    parser.add_argument("--only", nargs="+", choices=list(STRATEGIES), metavar="NAME",
                        help="run only these strategies")
    parser.add_argument("--save-db", action="store_true", help="also store the runs in SQLite")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    conn = connect(Settings.from_env().db_path)
    log.info("backtest %s .. %s", args.start, args.end)
    results = run_all(conn, cfg, args.start, args.end, args.only)
    rf_weekly = weekly_risk_free(conn)
    metrics = save_results(results, rf_weekly, Path(args.out) / "results")

    if args.save_db:
        for name, r in results.items():
            save_run(conn, r, metrics.loc[name].to_dict())

    with pd.option_context("display.float_format", "{:.4f}".format, "display.width", 140):
        log.info("\n%s", metrics)
    log.info("results written to %s", Path(args.out) / "results")


if __name__ == "__main__":
    main()
