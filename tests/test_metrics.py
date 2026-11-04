import numpy as np
import pandas as pd
import pytest

from simulation.metrics import compute_metrics


def test_metrics_on_small_nav_example():
    dates = pd.to_datetime(["2023-01-06", "2023-01-13", "2023-01-20", "2023-01-27"])
    nav = pd.Series([1.0, 1.2, 0.9, 1.1], index=dates)
    rf = pd.Series(0.0, index=dates[1:])
    turnover = pd.Series([0.5, 1.0, 0.25], index=dates[1:])

    metrics = compute_metrics(nav, rf, turnover, dates)

    assert metrics["max_drawdown"] == pytest.approx(0.25)
    assert metrics["ann_return"] == pytest.approx(1.1 ** (52 / 3) - 1)
    assert metrics["ann_vol"] == pytest.approx(nav.pct_change().iloc[1:].std() * np.sqrt(52))
    assert metrics["avg_turnover"] == pytest.approx(0.5833333333)
    assert metrics["ann_turnover"] == pytest.approx(0.5833333333 * 52)
    assert metrics["cvar_95"] == pytest.approx(0.25)


def test_sharpe_uses_weekly_excess_returns():
    dates = pd.date_range("2023-01-06", periods=5, freq="W-FRI")
    returns = pd.Series([0.02, -0.01, 0.03, 0.01], index=dates[1:])
    nav = pd.Series([1.0, *(1 + returns).cumprod().to_list()], index=dates)
    rf = pd.Series([0.005, 0.005, 0.01, 0.005], index=dates[1:])

    metrics = compute_metrics(nav, rf, pd.Series(dtype=float), dates)
    excess = returns - rf

    assert metrics["sharpe"] == pytest.approx(excess.mean() / excess.std() * np.sqrt(52))


def test_requires_nav_at_each_decision():
    nav = pd.Series([1.0, 1.1], index=pd.to_datetime(["2023-01-06", "2023-01-20"]))
    decisions = pd.to_datetime(["2023-01-06", "2023-01-13", "2023-01-20"])

    with pytest.raises(ValueError, match="missing decision dates"):
        compute_metrics(nav, pd.Series(dtype=float), pd.Series(dtype=float), decisions)
