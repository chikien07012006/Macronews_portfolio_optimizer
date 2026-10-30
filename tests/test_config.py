from datetime import date

import pytest

from common.config import Settings, config_from_dict, load_config


def test_default_config_loads():
    cfg = load_config()
    assert len(cfg.universe.core) == 11
    assert cfg.universe.core[:9] == cfg.universe.sectors
    assert "SPY" in cfg.universe.all_tickers and "XLRE" in cfg.universe.all_tickers
    assert cfg.universe.optional["XLC"] == date(2018, 6, 19)
    assert cfg.periods.test_start == date(2018, 1, 1)
    assert cfg.prior.delta == 2.5 and cfg.optimizer.w_max == 0.30


def test_overrides():
    cfg = load_config(overrides={"optimizer.w_max": 0.25, "news.half_life_days": 7})
    assert cfg.optimizer.w_max == 0.25
    assert cfg.news.half_life_days == 7


@pytest.mark.parametrize("key", ["optimizer.wmax", "nope.w_max", "optimizer.w_max.x"])
def test_unknown_override_rejected(key):
    with pytest.raises(KeyError):
        load_config(overrides={key: 1})


def test_reference_weights_must_sum_to_one():
    with pytest.raises(ValueError):
        load_config(overrides={"reference.equity": 0.8})


def test_periods_must_be_ordered():
    with pytest.raises(ValueError):
        load_config(overrides={"periods.test_start": "2012-01-01"})


def test_to_dict_round_trip():
    cfg = load_config()
    assert config_from_dict(cfg.to_dict()) == cfg


def test_settings_from_env(monkeypatch, tmp_path):
    monkeypatch.setenv("MACRONEWS_DATA_DIR", str(tmp_path))
    monkeypatch.delenv("MACRONEWS_DB_PATH", raising=False)
    s = Settings.from_env(dotenv_path=tmp_path / "missing.env")
    assert s.data_dir == tmp_path
    assert s.db_path == tmp_path / "macronews.db"
