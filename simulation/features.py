"""Technical features: momentum, short-term reversal and volatility.

Each feature is computed per asset at every weekly decision date t from data up to t only,
then standardized cross-sectionally (across the assets of the same week):

    MOM(i,t) = prod_{s=t-11}^{t-1} (1 + X(i,s)) - 1      11 weeks, skipping the latest week
    REV(i,t) = X(i,t)                                     latest weekly excess return
    VOL(i,t) = sqrt(252) * std( r(i,d) : last 20 trading days d <= t )
    Z(i,t)   = (F(i,t) - mean_j F(j,t)) / std_j F(j,t)

X is the weekly excess return, r the daily return. Window lengths come from
``cfg.features`` (mom_weeks = 12, mom_skip_weeks = 1, vol_days = 20).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from common.config import Config

TRADING_DAYS_PER_YEAR = 252
FEATURE_NAMES = ("MOM", "REV", "VOL")


def momentum(weekly_excess: pd.DataFrame, weeks: int = 12, skip: int = 1) -> pd.DataFrame:
    """Compounded excess return over weeks t-(weeks-1) .. t-skip, i.e. ``weeks - skip`` weeks.

    NaN until a full window is available (or when any week in the window is missing).
    """
    window = weeks - skip
    if skip < 0 or window < 1:
        raise ValueError(f"need weeks > skip >= 0, got weeks={weeks}, skip={skip}")
    # product of (1 + X) as a sum of logs: same value, vectorized over the whole table
    log_growth = np.log1p(weekly_excess).shift(skip).rolling(window).sum()
    return np.expm1(log_growth)


def volatility(
    daily_returns: pd.DataFrame, dates: pd.DatetimeIndex, days: int = 20
) -> pd.DataFrame:
    """Annualized std of the last ``days`` daily returns up to and including each date."""
    if days < 2:
        raise ValueError(f"need at least 2 days for a standard deviation, got {days}")
    vol = daily_returns.rolling(days).std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    return vol.reindex(pd.DatetimeIndex(dates))


def cross_sectional_zscore(values: pd.DataFrame) -> pd.DataFrame:
    """Standardize each row (week) across columns (assets); NaN entries are ignored.

    A row with fewer than two values or no dispersion becomes NaN.
    """
    std = values.std(axis=1)
    return values.sub(values.mean(axis=1), axis=0).div(std.where(std > 0), axis=0)


def technical_features(
    weekly_excess: pd.DataFrame, daily_returns: pd.DataFrame, cfg: Config
) -> pd.DataFrame:
    """MOM, REV and VOL for every (decision date, asset), standardized within each week.

    ``weekly_excess``: weekly excess returns, index = decision dates, columns = tickers.
    ``daily_returns``: daily returns, index = trading days, must contain the same tickers.

    Each week is standardized over the assets that have all three features, so the three
    columns always compare the same set of assets. Rows with missing history are dropped.
    Returns a DataFrame indexed by (decision_date, ticker) with columns MOM, REV, VOL.
    """
    assets = weekly_excess.columns
    missing = assets.difference(daily_returns.columns)
    if len(missing):
        raise ValueError(f"daily_returns has no column for {list(missing)}")
    dates = pd.DatetimeIndex(weekly_excess.index)
    f = cfg.features

    raw = {
        "MOM": momentum(weekly_excess, f.mom_weeks, f.mom_skip_weeks),
        "REV": weekly_excess,
        "VOL": volatility(daily_returns[assets], dates, f.vol_days),
    }
    complete = raw["MOM"].notna() & raw["REV"].notna() & raw["VOL"].notna()
    z = {name: cross_sectional_zscore(table.where(complete)) for name, table in raw.items()}

    index = pd.MultiIndex.from_product([dates, assets], names=["decision_date", "ticker"])
    out = pd.DataFrame({name: z[name].to_numpy().ravel() for name in FEATURE_NAMES}, index=index)
    return out.dropna()
