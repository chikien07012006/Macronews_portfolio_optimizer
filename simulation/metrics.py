from __future__ import annotations

import numpy as np
import pandas as pd


def compute_metrics(
    nav: pd.Series,
    rf_weekly: pd.Series,
    turnover: pd.Series,
    decisions: pd.DatetimeIndex,
) -> dict[str, float]:
    """Compute annualized performance and risk statistics from weekly NAV samples."""
    nav = nav.sort_index().astype(float)
    decisions = pd.DatetimeIndex(decisions).normalize().unique().sort_values()
    if len(nav) == 0:
        raise ValueError("nav must not be empty")
    if nav.index.has_duplicates or not nav.index.is_monotonic_increasing:
        raise ValueError("nav index must be unique and increasing")
    if not np.isfinite(nav.to_numpy()).all() or (nav <= 0).any():
        raise ValueError("nav values must be finite and positive")

    sampled_nav = nav.reindex(decisions)
    if sampled_nav.isna().any():
        missing = decisions[sampled_nav.isna()]
        raise ValueError(f"nav is missing decision dates: {list(missing)}")

    weekly_returns = sampled_nav.pct_change().iloc[1:]
    weekly_returns.name = "weekly_return"
    weekly_rf = rf_weekly.copy()
    weekly_rf.index = pd.DatetimeIndex(weekly_rf.index).normalize()
    excess = weekly_returns - weekly_rf.reindex(weekly_returns.index)
    excess = excess.dropna()
    periods = len(weekly_returns)

    if periods:
        first_value = float(sampled_nav.iloc[0])
        last_value = float(sampled_nav.iloc[-1])
        annual_return = (last_value / first_value) ** (52.0 / periods) - 1.0
        annual_volatility = (
            float(weekly_returns.std(ddof=1) * np.sqrt(52.0))
            if periods > 1
            else float("nan")
        )
        excess_std = float(excess.std(ddof=1)) if periods > 1 else float("nan")
        sharpe = (
            float(excess.mean() / excess_std * np.sqrt(52.0))
            if np.isfinite(excess_std) and excess_std > 0
            else float("nan")
        )
        downside_deviation = float(np.sqrt(np.square(excess.clip(upper=0.0)).mean()))
        sortino = (
            float(excess.mean() / downside_deviation * np.sqrt(52.0))
            if downside_deviation > 0
            else float("nan")
        )
        quantile = float(weekly_returns.quantile(0.05))
        cvar = float(-weekly_returns[weekly_returns <= quantile].mean())
    else:
        annual_return = annual_volatility = sharpe = sortino = cvar = float("nan")

    drawdown = 1.0 - nav / nav.cummax()
    avg_turnover = float(turnover.mean()) if not turnover.empty else 0.0
    return {
        "ann_return": float(annual_return),
        "ann_vol": float(annual_volatility),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "max_drawdown": float(drawdown.max()),
        "cvar_95": float(cvar),
        "avg_turnover": avg_turnover,
        "ann_turnover": avg_turnover * 52.0,
    }
