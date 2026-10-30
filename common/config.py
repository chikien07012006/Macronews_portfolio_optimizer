"""Experiment configuration (YAML) and environment settings (.env).

Experiment parameters live in ``config.yaml`` and are loaded into frozen dataclasses, so
every backtest can store the exact configuration it ran with (``Config.to_dict``).
Secrets and machine-specific values (API keys, data and database paths) come from
environment variables / ``.env`` via ``Settings``.
"""

from __future__ import annotations

import copy
import os
from dataclasses import asdict, dataclass, field, fields
from datetime import date
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def _as_date(value: date | str) -> date:
    return value if isinstance(value, date) else date.fromisoformat(value)


@dataclass(frozen=True)
class UniverseConfig:
    sectors: tuple[str, ...]
    others: tuple[str, ...]
    optional: dict[str, date] = field(default_factory=dict)
    benchmark: str = "SPY"

    def __post_init__(self) -> None:
        object.__setattr__(self, "sectors", tuple(self.sectors))
        object.__setattr__(self, "others", tuple(self.others))
        object.__setattr__(
            self, "optional", {k: _as_date(v) for k, v in self.optional.items()}
        )

    @property
    def core(self) -> tuple[str, ...]:
        """Assets available over the whole sample (sectors first, then others)."""
        return self.sectors + self.others

    @property
    def all_tickers(self) -> tuple[str, ...]:
        """Every ticker that has to be downloaded, including optional assets and benchmark."""
        return self.core + tuple(self.optional) + (self.benchmark,)


@dataclass(frozen=True)
class PeriodsConfig:
    data_start: date
    warmup_start: date
    validation_start: date
    test_start: date
    end: date

    def __post_init__(self) -> None:
        for f in fields(self):
            object.__setattr__(self, f.name, _as_date(getattr(self, f.name)))
        ordered = [getattr(self, f.name) for f in fields(self)]
        if ordered != sorted(ordered):
            raise ValueError(f"periods must be in chronological order: {ordered}")


@dataclass(frozen=True)
class CalendarConfig:
    timezone: str = "America/New_York"
    cutoff: str = "16:00"


@dataclass(frozen=True)
class RiskConfig:
    cov_window_days: int = 252
    horizon_days: int = 20


@dataclass(frozen=True)
class PriorConfig:
    delta: float = 2.5
    tau: float = 0.05


@dataclass(frozen=True)
class FeaturesConfig:
    mom_weeks: int = 12
    mom_skip_weeks: int = 1
    vol_days: int = 20


@dataclass(frozen=True)
class ViewsConfig:
    horizon_weeks: int = 4
    min_train_weeks: int = 156
    signal_threshold: float = 0.5
    ridge_alpha: float | None = None
    c_news: float = 1.0
    c_tech: float = 1.0
    kappa: float = 1.0


@dataclass(frozen=True)
class NewsConfig:
    half_life_days: float = 14.0
    lookback_weeks: int = 8
    zscore_weeks: int = 156


@dataclass(frozen=True)
class OptimizerConfig:
    gamma: float = 2.5
    w_max: float = 0.30
    turnover_penalty: float = 0.001
    solver: str = "CLARABEL"
    fallback_solver: str = "OSQP"


@dataclass(frozen=True)
class BacktestConfig:
    cost_bps: float = 5.0
    rebalance: str = "weekly"
    execution_lag_days: int = 1

    def __post_init__(self) -> None:
        if self.rebalance not in ("weekly", "monthly"):
            raise ValueError(f"rebalance must be 'weekly' or 'monthly', got {self.rebalance!r}")


@dataclass(frozen=True)
class Config:
    universe: UniverseConfig
    reference: dict[str, float]
    periods: PeriodsConfig
    calendar: CalendarConfig = field(default_factory=CalendarConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    prior: PriorConfig = field(default_factory=PriorConfig)
    features: FeaturesConfig = field(default_factory=FeaturesConfig)
    views: ViewsConfig = field(default_factory=ViewsConfig)
    news: NewsConfig = field(default_factory=NewsConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)

    def __post_init__(self) -> None:
        unknown = set(self.reference) - {"equity", *self.universe.others}
        if unknown:
            raise ValueError(f"reference weights for unknown assets: {sorted(unknown)}")
        total = sum(self.reference.values())
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"reference weights must sum to 1, got {total}")
        if self.optimizer.w_max * len(self.universe.core) < 1.0:
            raise ValueError("w_max too small: weights cannot sum to 1")

    def to_dict(self) -> dict[str, Any]:
        """Plain dict (dates as ISO strings), suitable for YAML/JSON next to a run's results."""
        return _jsonable(asdict(self))


_SECTIONS: dict[str, type] = {
    "universe": UniverseConfig,
    "periods": PeriodsConfig,
    "calendar": CalendarConfig,
    "risk": RiskConfig,
    "prior": PriorConfig,
    "features": FeaturesConfig,
    "views": ViewsConfig,
    "news": NewsConfig,
    "optimizer": OptimizerConfig,
    "backtest": BacktestConfig,
}


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, date):
        return obj.isoformat()
    return obj


def _apply_overrides(raw: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Apply dotted-key overrides, e.g. ``{"optimizer.w_max": 0.25}``."""
    raw = copy.deepcopy(raw)
    for dotted, value in overrides.items():
        *parents, leaf = dotted.split(".")
        node = raw
        for key in parents:
            if not isinstance(node.get(key), dict):
                raise KeyError(f"unknown config section in override {dotted!r}")
            node = node[key]
        if leaf not in node:
            raise KeyError(f"unknown config key in override {dotted!r}")
        node[leaf] = value
    return raw


def config_from_dict(raw: dict[str, Any]) -> Config:
    unknown = set(raw) - set(_SECTIONS) - {"reference"}
    if unknown:
        raise KeyError(f"unknown config sections: {sorted(unknown)}")
    kwargs: dict[str, Any] = {"reference": dict(raw["reference"])}
    for name, cls in _SECTIONS.items():
        if name in raw:
            try:
                kwargs[name] = cls(**raw[name])
            except TypeError as exc:
                raise KeyError(f"invalid keys in config section {name!r}: {exc}") from exc
    return Config(**kwargs)


def load_config(
    path: str | Path | None = None, overrides: dict[str, Any] | None = None
) -> Config:
    """Load a YAML config (default: ``config.yaml``) with optional dotted overrides."""
    with open(path or DEFAULT_CONFIG_PATH) as fh:
        raw = yaml.safe_load(fh)
    return config_from_dict(_apply_overrides(raw, overrides or {}))


@dataclass(frozen=True)
class Settings:
    """Machine-specific settings read from the environment (and ``.env`` if present)."""

    data_dir: Path
    db_path: Path

    @classmethod
    def from_env(cls, dotenv_path: str | Path | None = None) -> Settings:
        load_dotenv(dotenv_path or PROJECT_ROOT / ".env", override=False)
        env = os.environ
        data_dir = Path(env.get("MACRONEWS_DATA_DIR", PROJECT_ROOT / "data"))
        return cls(
            data_dir=data_dir,
            db_path=Path(env.get("MACRONEWS_DB_PATH", data_dir / "macronews.db")),
        )
