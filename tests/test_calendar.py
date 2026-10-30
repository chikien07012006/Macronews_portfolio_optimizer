import numpy as np
import pandas as pd
import pytest

from common.calendar import assign_week, decision_cutoffs, decision_dates, execution_dates

# NYSE holidays in Jan-Apr 2023 (enough for the cases below)
HOLIDAYS = pd.to_datetime(["2023-01-02", "2023-01-16", "2023-02-20", "2023-04-07"])


def trading_days(start="2023-01-03", end="2023-04-28"):
    days = pd.bdate_range(start, end)
    return days[~days.isin(HOLIDAYS)]


def ts(s):
    return pd.Timestamp(s, tz="America/New_York")


class TestDecisionDates:
    def test_fridays_in_normal_weeks(self):
        d = decision_dates(trading_days(), end="2023-02-28")
        assert (d.day_of_week == 4).all()
        assert d[0] == pd.Timestamp("2023-01-06")

    def test_good_friday_falls_back_to_thursday(self):
        d = decision_dates(trading_days(), start="2023-04-01", end="2023-04-14")
        assert list(d) == [pd.Timestamp("2023-04-06"), pd.Timestamp("2023-04-14")]

    def test_one_per_week_and_all_trading_days(self):
        days = trading_days()
        d = decision_dates(days)
        assert d.isin(days).all()
        assert not d.to_period("W-FRI").has_duplicates

    def test_incomplete_final_week_dropped(self):
        d = decision_dates(trading_days(end="2023-04-26"))  # data stop on a Wednesday
        assert d[-1] == pd.Timestamp("2023-04-21")

    def test_complete_final_week_kept(self):
        d = decision_dates(trading_days(end="2023-04-28"))
        assert d[-1] == pd.Timestamp("2023-04-28")

    def test_monthly(self):
        d = decision_dates(trading_days(), frequency="monthly")
        assert list(d) == list(pd.to_datetime(["2023-01-31", "2023-02-28", "2023-03-31", "2023-04-28"]))

    def test_rejects_unsorted(self):
        with pytest.raises(ValueError):
            decision_dates(trading_days()[::-1])


class TestExecutionDates:
    def test_next_trading_day(self):
        days = trading_days()
        dec = pd.to_datetime(["2023-01-06", "2023-01-13", "2023-04-06"])
        ex = execution_dates(days, dec)
        # Mon; Tue after MLK day; Mon after Good Friday
        assert list(ex) == list(pd.to_datetime(["2023-01-09", "2023-01-17", "2023-04-10"]))

    def test_always_strictly_after_decision(self):
        days = trading_days()
        dec = decision_dates(days)[:-1]
        assert (execution_dates(days, dec) > dec).all()

    def test_non_trading_decision_raises(self):
        with pytest.raises(ValueError):
            execution_dates(trading_days(), pd.to_datetime(["2023-04-07"]))

    def test_no_day_after_end_raises(self):
        days = trading_days()
        with pytest.raises(ValueError):
            execution_dates(days, days[-1:])


class TestAssignWeek:
    dec = pd.to_datetime(["2023-03-03", "2023-03-10", "2023-03-17"])

    def test_cutoff_boundaries(self):
        docs = pd.Series([ts("2023-03-10 15:59"), ts("2023-03-10 16:00"), ts("2023-03-10 16:01"),
                          ts("2023-03-11 09:00")])
        out = assign_week(docs, self.dec)
        expected = pd.to_datetime(["2023-03-10", "2023-03-10", "2023-03-17", "2023-03-17"])
        assert list(out) == list(expected)

    def test_converts_from_utc_across_dst(self):
        # DST starts 2023-03-12: cut-off on 03-10 is 21:00 UTC, on 03-17 it is 20:00 UTC
        # so the same 20:30 UTC is before the cut-off in the first week and after it in the second
        docs = pd.Series(pd.to_datetime(["2023-03-10 20:30", "2023-03-17 19:30", "2023-03-17 20:30"])
                         .tz_localize("UTC"))
        out = assign_week(docs, self.dec)
        assert out.iloc[0] == pd.Timestamp("2023-03-10")
        assert out.iloc[1] == pd.Timestamp("2023-03-17")
        assert pd.isna(out.iloc[2])  # 16:30 EDT on the last decision date

    def test_after_last_cutoff_is_nat(self):
        out = assign_week(pd.Series([ts("2023-03-20 10:00")]), self.dec)
        assert out.isna().all()

    def test_naive_timestamps_rejected(self):
        with pytest.raises(ValueError):
            assign_week(pd.Series(pd.to_datetime(["2023-03-10 12:00"])), self.dec)

    def test_no_look_ahead(self):
        """A document is never used before it exists, and never later than necessary."""
        days = trading_days()
        dec = decision_dates(days)
        rng = np.random.default_rng(0)
        seconds = rng.integers(0, 80 * 86400, size=2000)
        docs = pd.Series(ts("2023-01-01") + pd.to_timedelta(seconds, unit="s"))
        week = assign_week(docs, dec)
        cut = decision_cutoffs(dec)
        cut_by_date = pd.Series(cut, index=dec)

        valid = week.notna()
        assert valid.all()
        assigned_cut = cut_by_date.loc[week[valid]].to_numpy()
        assert (docs[valid].to_numpy() <= assigned_cut).all()
        prev_pos = dec.get_indexer(week[valid]) - 1
        has_prev = prev_pos >= 0
        prev_cut = cut[prev_pos[has_prev]]
        assert (docs[valid][has_prev].to_numpy() > prev_cut.to_numpy()).all()
