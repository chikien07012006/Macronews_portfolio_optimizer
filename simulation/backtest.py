from __future__ import annotations

import json
import sqlite3
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from common.calendar import decision_dates, execution_dates
from common.config import Config
from data_pipeline.database import upsert
from data_pipeline.market_data import daily_returns, trading_calendar

Strategy = Callable[[pd.Timestamp, pd.Series], pd.Series]


@dataclass(frozen=True)
class BacktestResult:
    nav: pd.Series
    daily_returns: pd.Series
    weights: pd.DataFrame
    turnover: pd.Series
    costs: pd.Series
    name: str
    config: dict[str, Any]
    decisions: pd.DatetimeIndex


def equal_weight_strategy(assets: Sequence[str]) -> Strategy:
    """Create a strategy that assigns equal weight to each supplied asset."""
    assets = tuple(assets)
    if not assets:
        raise ValueError("assets must not be empty")

    def strategy(as_of: pd.Timestamp, w_current: pd.Series) -> pd.Series:
        return pd.Series(1.0 / len(assets), index=assets, dtype=float)

    return strategy


def run_backtest(
    conn: sqlite3.Connection,
    strategy: Strategy,
    cfg: Config,
    start: str,
    end: str,
    name: str,
) -> BacktestResult:
    """Simulate daily portfolio returns and weekly close-to-close rebalances."""
    start_date, end_date = pd.Timestamp(start), pd.Timestamp(end)
    if start_date > end_date:
        raise ValueError("start must be on or before end")

    calendar = trading_calendar(conn)
    if calendar.empty:
        raise ValueError("the database has no trading days")

    lag = cfg.backtest.execution_lag_days
    if lag < 1:
        raise ValueError("execution_lag_days must be at least 1")
    if not np.isfinite(cfg.backtest.cost_bps) or cfg.backtest.cost_bps < 0:
        raise ValueError("cost_bps must be finite and non-negative")
    decisions = decision_dates(calendar, start_date, end_date)
    decision_positions = calendar.get_indexer(decisions)
    executable = decision_positions + lag < len(calendar)
    decisions = decisions[executable]
    if len(decisions):
        executions = execution_dates(calendar, decisions, lag=lag)
    else:
        executions = pd.DatetimeIndex([], name="execution_date")

    if not len(calendar[calendar >= start_date]):
        raise ValueError("start is after the last trading day in the database")
    last_day = max(end_date, executions[-1]) if len(executions) else min(end_date, calendar[-1])
    last_day = min(last_day, calendar[-1])
    days = calendar[(calendar >= start_date) & (calendar <= last_day)]
    if days.empty:
        raise ValueError("the requested range contains no trading days")

    returns = daily_returns(conn).reindex(days)
    assets = returns.columns
    current = pd.Series(0.0, index=assets, dtype=float)
    decision_to_execution = dict(zip(decisions, executions))
    decision_days = set(decisions)
    execution_days = set(executions)
    target_by_execution: dict[pd.Timestamp, pd.Series] = {}
    nav_values: list[float] = []
    daily_return_values: list[float] = []
    weight_rows: list[pd.Series] = []
    turnover_values: list[float] = []
    cost_values: list[float] = []
    turnover_dates: list[pd.Timestamp] = []
    value = 1.0
    cost_rate = cfg.backtest.cost_bps / 10_000.0

    for day in days:
        previous_value = value
        asset_returns = returns.loc[day]
        held = current != 0
        missing_held = held & asset_returns.isna()
        if missing_held.any():
            missing = list(current.index[missing_held])
            raise ValueError(f"missing returns for held assets on {day.date()}: {missing}")
        asset_returns = asset_returns.fillna(0.0)
        portfolio_return = float(current.dot(asset_returns))
        if not np.isfinite(portfolio_return) or 1.0 + portfolio_return <= 0.0:
            raise ValueError(f"invalid portfolio return on {day.date()}: {portfolio_return}")

        value *= 1.0 + portfolio_return
        current = current * (1.0 + asset_returns) / (1.0 + portfolio_return)

        if day in decision_days:
            target = strategy(pd.Timestamp(day), current.copy())
            if not isinstance(target, pd.Series):
                raise TypeError("strategy must return a pandas Series")
            if target.empty or target.index.has_duplicates:
                raise ValueError("strategy weights must have a non-empty, unique index")
            target = pd.to_numeric(target, errors="raise").astype(float)
            if not np.isfinite(target.to_numpy()).all():
                raise ValueError("strategy weights must be finite")
            unknown = target.index.difference(assets)
            if len(unknown):
                raise ValueError(f"strategy returned unknown assets: {list(unknown)}")
            if not np.isclose(target.sum(), 1.0, atol=1e-8):
                raise ValueError("strategy weights must sum to 1")
            target = target.reindex(assets, fill_value=0.0)
            target_by_execution[decision_to_execution[day]] = target

        if day in execution_days:
            target = target_by_execution.pop(day)
            turnover = float((target - current).abs().sum())
            fee = cost_rate * turnover
            if not np.isfinite(fee) or fee >= 1.0:
                raise ValueError(f"invalid transaction cost on {day.date()}: {fee}")
            value *= 1.0 - fee
            current = target
            weight_rows.append(target.rename(day))
            turnover_dates.append(pd.Timestamp(day))
            turnover_values.append(turnover)
            cost_values.append(fee)

        daily_return_values.append(value / previous_value - 1.0)
        nav_values.append(value)

    nav = pd.Series(nav_values, index=days, name="nav", dtype=float)
    daily_returns_out = pd.Series(daily_return_values, index=days, name="daily_returns", dtype=float)
    weights = pd.DataFrame(weight_rows, index=pd.DatetimeIndex(turnover_dates), columns=assets)
    weights.index.name = "execution_date"
    turnover = pd.Series(turnover_values, index=turnover_dates, name="turnover", dtype=float)
    costs = pd.Series(cost_values, index=turnover_dates, name="costs", dtype=float)

    return BacktestResult(
        nav=nav,
        daily_returns=daily_returns_out,
        weights=weights,
        turnover=turnover,
        costs=costs,
        name=name,
        config=cfg.to_dict(),
        decisions=decisions,
    )


def save_run(
    conn: sqlite3.Connection, result: BacktestResult, metrics: dict[str, float]
) -> str:
    """Persist a backtest run and its NAV and target weights."""
    run_id = uuid.uuid4().hex
    config_json = json.dumps(result.config, sort_keys=True)
    metrics_json = json.dumps(metrics, sort_keys=True, allow_nan=False)
    created_at = datetime.now(timezone.utc).isoformat()

    with conn:
        conn.execute(
            """INSERT INTO backtest_runs
               (run_id, name, created_at, config_json, metrics_json)
               VALUES (?, ?, ?, ?, ?)""",
            (run_id, result.name, created_at, config_json, metrics_json),
        )

    nav_rows = pd.DataFrame(
        {
            "run_id": run_id,
            "date": result.nav.index,
            "nav": result.nav.to_numpy(),
        }
    )
    weight_rows = (
        result.weights.rename_axis("date")
        .stack()
        .rename("weight")
        .reset_index()
        .rename(columns={"level_1": "ticker"})
    )
    weight_rows.insert(0, "run_id", run_id)
    upsert(conn, "backtest_nav", nav_rows)
    upsert(conn, "backtest_weights", weight_rows)
    return run_id
