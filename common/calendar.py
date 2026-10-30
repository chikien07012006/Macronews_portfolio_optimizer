"""Weekly decision calendar.

Single definition of "week t" shared by every component:

* decision date  - the last trading day of each Saturday-Friday week (normally Friday,
  Thursday when Friday is a holiday). Information up to that day's close is usable.
* execution date - the next trading day; target weights are traded at its close.
* document week  - a document belongs to the first decision date whose cut-off
  (16:00 ET on that day) is at or after its timestamp.

Trading days are taken from a price index (e.g. SPY), so no exchange-calendar library
is needed.
"""

from __future__ import annotations

import pandas as pd


def _trading_index(trading_days: pd.DatetimeIndex) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(trading_days)
    if idx.tz is not None:
        raise ValueError("trading_days must be tz-naive dates")
    idx = idx.normalize()
    if not idx.is_monotonic_increasing or idx.has_duplicates:
        raise ValueError("trading_days must be strictly increasing")
    return idx


def decision_dates(
    trading_days: pd.DatetimeIndex,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    frequency: str = "weekly",
) -> pd.DatetimeIndex:
    """Last trading day of each week (or month) within ``[start, end]``.

    The final period is dropped if the data stop before its last weekday (e.g. index ends on
    a Wednesday), because its last trading day is not yet known at that point. A period whose
    last weekday is a holiday at the very end of the data is therefore also dropped.
    """
    idx = _trading_index(trading_days)
    freq = {"weekly": "W-FRI", "monthly": "M"}[frequency]
    periods = idx.to_period(freq)
    last = pd.Series(idx, index=idx).groupby(periods).max()
    if len(last):
        last_weekday = pd.offsets.BDay().rollback(last.index[-1].end_time.normalize())
        if idx[-1] < last_weekday:
            last = last.iloc[:-1]
    dates = pd.DatetimeIndex(last.values, name="decision_date")
    if start is not None:
        dates = dates[dates >= pd.Timestamp(start)]
    if end is not None:
        dates = dates[dates <= pd.Timestamp(end)]
    return dates


def execution_dates(
    trading_days: pd.DatetimeIndex, decisions: pd.DatetimeIndex, lag: int = 1
) -> pd.DatetimeIndex:
    """Trading day ``lag`` sessions after each decision date.

    Raises if a decision date is not a trading day or has no trading day ``lag`` sessions later.
    """
    idx = _trading_index(trading_days)
    if lag < 0:
        raise ValueError("lag must be non-negative")
    pos = idx.get_indexer(pd.DatetimeIndex(decisions).normalize())
    if (pos < 0).any():
        raise ValueError("every decision date must be a trading day")
    target = pos + lag
    if (target >= len(idx)).any():
        raise ValueError("no trading day available after the last decision date")
    return pd.DatetimeIndex(idx[target], name="execution_date")


def decision_cutoffs(
    decisions: pd.DatetimeIndex, timezone: str = "America/New_York", cutoff: str = "16:00"
) -> pd.DatetimeIndex:
    """Tz-aware cut-off timestamps (decision date at ``cutoff`` local time)."""
    offset = pd.Timedelta(f"{cutoff}:00")
    return (pd.DatetimeIndex(decisions).normalize() + offset).tz_localize(timezone)


def assign_week(
    timestamps: pd.Series | pd.DatetimeIndex,
    decisions: pd.DatetimeIndex,
    timezone: str = "America/New_York",
    cutoff: str = "16:00",
) -> pd.Series:
    """Map document timestamps to the decision date whose information set includes them.

    ``timestamps`` must be tz-aware (convert or localize at ingestion, where the source's
    convention is known). A document stamped exactly at the cut-off counts for that week.
    Documents after the last cut-off get ``NaT``.
    """
    ts = pd.Series(timestamps)
    if ts.dt.tz is None:
        raise ValueError("timestamps must be tz-aware; localize them at ingestion")
    cutoffs = decision_cutoffs(decisions, timezone, cutoff)
    pos = cutoffs.searchsorted(ts.dt.tz_convert(timezone), side="left")
    out = pd.Series(pd.NaT, index=ts.index, dtype="datetime64[ns]", name="decision_date")
    valid = pos < len(cutoffs)
    out[valid] = pd.DatetimeIndex(decisions).normalize()[pos[valid]]
    return out
