import json

import numpy as np
import pandas as pd
import pytest

from common.config import load_config
from data_pipeline.database import connect, query, upsert
from simulation.backtest import equal_weight_strategy, run_backtest, save_run


@pytest.fixture
def prices_db(tmp_path):
    conn = connect(tmp_path / "t.db")
    days = pd.bdate_range("2023-01-02", "2023-01-24")
    assets = ["A", "B", "SPY"]
    upsert(
        conn,
        "assets",
        pd.DataFrame({"ticker": assets, "asset_class": ["sector", "sector", "benchmark"]}),
    )
    returns = pd.DataFrame(0.0, index=days, columns=assets)
    returns.loc["2023-01-10", "A"] = 0.10
    returns.loc["2023-01-16", "A"] = 0.10
    returns.loc["2023-01-17", "B"] = 0.20
    prices = (1 + returns).cumprod() * 100
    rows = prices.rename_axis("date").stack().rename("adj_close").reset_index()
    rows.columns = ["date", "ticker", "adj_close"]
    upsert(conn, "prices_daily", rows)
    yield conn
    conn.close()


def zero_cost_config():
    return load_config(overrides={"backtest.cost_bps": 0.0})


def test_equal_weight_strategy_and_fixed_single_asset_returns(prices_db):
    result = run_backtest(
        prices_db,
        equal_weight_strategy(["A"]),
        zero_cost_config(),
        "2023-01-06",
        "2023-01-13",
        "fixed",
    )

    assert result.nav.loc["2023-01-09"] == pytest.approx(1.0)
    assert result.nav.loc["2023-01-13"] == pytest.approx(1.1)
    assert result.weights.loc["2023-01-09", "A"] == 1.0
    assert result.daily_returns.loc["2023-01-10"] == pytest.approx(0.10)


def test_weights_drift_by_daily_asset_returns(prices_db):
    observed = []

    def strategy(as_of, w_current):
        observed.append((as_of, w_current.copy()))
        return pd.Series({"A": 0.5, "B": 0.5})

    run_backtest(
        prices_db, strategy, zero_cost_config(), "2023-01-06", "2023-01-13", "drift"
    )

    second_decision = next(weights for day, weights in observed if day == pd.Timestamp("2023-01-13"))
    assert second_decision["A"] == pytest.approx(0.55 / 1.05)
    assert second_decision["B"] == pytest.approx(0.50 / 1.05)


def test_execution_uses_old_weights_then_applies_fee_and_new_weights(prices_db):
    def strategy(as_of, w_current):
        ticker = "A" if as_of == pd.Timestamp("2023-01-06") else "B"
        return pd.Series({ticker: 1.0})

    result = run_backtest(
        prices_db,
        strategy,
        load_config(overrides={"backtest.cost_bps": 5.0}),
        "2023-01-06",
        "2023-01-17",
        "switch",
    )

    assert result.weights.loc["2023-01-09", "A"] == 1.0
    assert result.weights.loc["2023-01-16", "B"] == 1.0
    assert result.turnover.loc["2023-01-16"] == pytest.approx(2.0)
    assert result.costs.loc["2023-01-16"] == pytest.approx(0.001)
    assert result.daily_returns.loc["2023-01-16"] == pytest.approx(1.1 * 0.999 - 1.0)
    assert result.daily_returns.loc["2023-01-17"] == pytest.approx(0.20)


def test_strategy_is_only_called_on_decision_dates_and_execution_is_later(prices_db):
    called = []

    def strategy(as_of, w_current):
        called.append(as_of)
        return pd.Series({"A": 1.0})

    result = run_backtest(
        prices_db, strategy, zero_cost_config(), "2023-01-06", "2023-01-13", "no-lookahead"
    )

    assert called == list(result.decisions)
    assert all(execution > decision for execution, decision in zip(result.weights.index, result.decisions))


def test_save_run_persists_run_nav_and_weights(prices_db):
    result = run_backtest(
        prices_db,
        equal_weight_strategy(["A", "B"]),
        zero_cost_config(),
        "2023-01-06",
        "2023-01-13",
        "equal",
    )
    metrics = {"ann_return": 0.12, "max_drawdown": 0.1}

    run_id = save_run(prices_db, result, metrics)

    run = query(prices_db, "SELECT * FROM backtest_runs WHERE run_id = ?", [run_id]).iloc[0]
    assert run["name"] == "equal"
    assert json.loads(run["config_json"]) == result.config
    assert json.loads(run["metrics_json"]) == metrics
    assert len(query(prices_db, "SELECT * FROM backtest_nav WHERE run_id = ?", [run_id])) == len(result.nav)
    weights = query(prices_db, "SELECT * FROM backtest_weights WHERE run_id = ?", [run_id])
    assert len(weights) == result.weights.size
    assert np.allclose(weights.groupby("date")["weight"].sum().to_numpy(), 1.0)
