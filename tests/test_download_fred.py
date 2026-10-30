from datetime import date

import pandas as pd
import pytest

from data_pipeline import download_fred
from data_pipeline.database import connect, query

CSV = """observation_date,DTB3
2023-01-02,.
2023-01-03,4.30
2023-01-04,4.35
2023-01-05,
"""


class FakeResponse:
    def __init__(self, text):
        self.text = text

    def raise_for_status(self):
        pass


@pytest.fixture
def fake_fred(monkeypatch):
    calls = []

    def get(url, params, timeout):
        calls.append(params)
        return FakeResponse(CSV)

    monkeypatch.setattr(download_fred.requests, "get", get)
    return calls


def test_fetch_series_parses_missing_values(fake_fred):
    df = download_fred.fetch_series("DTB3", date(2023, 1, 1), date(2023, 1, 31))
    assert list(df["date"]) == list(pd.to_datetime(["2023-01-02", "2023-01-03", "2023-01-04", "2023-01-05"]))
    assert df["value"].isna().tolist() == [True, False, False, True]
    assert df.loc[1, "value"] == 4.30
    assert fake_fred[0] == {"id": "DTB3", "cosd": "2023-01-01", "coed": "2023-01-31"}


def test_fetch_series_clips_to_period(fake_fred):
    df = download_fred.fetch_series("DTB3", date(2023, 1, 3), date(2023, 1, 4))
    assert len(df) == 2


def test_wrong_series_column_raises(fake_fred):
    with pytest.raises(ValueError):
        download_fred.fetch_series("CPIAUCSL", date(2023, 1, 1), date(2023, 1, 31))


def test_download_into_db(tmp_path, fake_fred):
    db = tmp_path / "t.db"
    summary = download_fred.download_fred(("DTB3",), date(2023, 1, 1), date(2023, 1, 31), db)
    assert summary.loc[0, "missing"] == 2
    stored = query(connect(db), "SELECT * FROM macro_daily ORDER BY date")
    assert len(stored) == 4
    assert stored["value"].isna().sum() == 2
