"""Typed configuration loaded from YAML."""

from __future__ import annotations

import os
from datetime import date, time
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator

from gridwise.battery import BatterySpec

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Paths(BaseModel):
    raw: Path
    interim: Path
    models: Path
    reports: Path
    site: Path

    def resolve(self, root: Path) -> Paths:
        def r(p: Path) -> Path:
            return p if p.is_absolute() else (root / p).resolve()

        return Paths(**{k: r(v) for k, v in self.model_dump().items()})


class Market(BaseModel):
    timezone: str = "Europe/Berlin"
    history_start: date
    decision_time_utc: time


class Smard(BaseModel):
    resolution: str = "hour"
    series: dict[str, tuple[int, str]]


class Weather(BaseModel):
    model: str
    lead_days: int = Field(ge=2, le=7)
    start: date
    variables: list[str]
    locations: dict[str, list[tuple[str, float, float]]]

    @field_validator("lead_days")
    @classmethod
    def _lead_is_safe(cls, v: int) -> int:
        # With a 1-day lead, the run for the last hours of delivery day D starts at about 18:00 UTC on
        # D-1, after the auction has closed. Two days is the shortest lead that is always available.
        if v < 2:
            raise ValueError("a weather lead under 2 days is not available before the auction closes")
        return v

    def sites(self) -> list[tuple[str, str, float, float]]:
        """Flat list of (group, name, latitude, longitude)."""
        return [(g, n, lat, lon) for g, items in self.locations.items() for n, lat, lon in items]


class Backtest(BaseModel):
    """One rolling-origin backtest: every test day is forecast with models that only saw earlier days."""

    feature_set: str
    test_start: date
    test_end: date | None = None  # None = the last day with known prices
    warmup_days: int = 120  # forecast days before test_start, used only to fill the calibration window
    min_train_days: int = 180
    # Score only the days that another backtest also scored, so two runs compare like with like.
    same_days_as: str | None = None


class Calibration(BaseModel):
    window_days: int = 90
    mode: str = "adaptive"
    gamma: float = 0.01


class Risk(BaseModel):
    alpha: float = Field(gt=0, lt=1, default=0.1)
    weight: float = Field(ge=0, le=1, default=0.5)
    scenarios: int = Field(gt=1, default=200)
    copula_days: int = 180


class Live(BaseModel):
    feature_set: str = "weather"
    models: list[str] = ["lear", "qgbm", "qnn"]
    history_days: int = 300  # past days re-forecast to fill the calibration window and the copula
    min_train_days: int = 180
    profile_days: int = 28


class Config(BaseModel):
    paths: Paths
    tracking_uri: str
    artifact_root: str = "../mlruns"
    experiment: str
    market: Market
    smard: Smard
    weather: Weather
    quantiles: list[float]
    backtests: dict[str, Backtest]
    calibration: Calibration
    battery: BatterySpec
    risk: Risk
    live: Live = Field(default_factory=Live)
    models: dict[str, Any] = Field(default_factory=dict)
    extra: dict[str, Any] = Field(default_factory=dict)

    @field_validator("quantiles")
    @classmethod
    def _quantiles_ok(cls, v: list[float]) -> list[float]:
        if v != sorted(v) or len(set(v)) != len(v) or not all(0 < q < 1 for q in v):
            raise ValueError("quantiles must be strictly increasing and inside (0, 1)")
        if 0.5 not in v:
            raise ValueError("quantiles must include the median 0.5")
        if any(abs((1 - q) - p) > 1e-9 for q, p in zip(v, reversed(v), strict=True)):
            raise ValueError("quantiles must be symmetric around 0.5 so that they pair into intervals")
        return v


def load_config(path: str | Path | None = None, root: Path = PROJECT_ROOT) -> Config:
    p = Path(path) if path else root / "configs" / "default.yaml"
    raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    known = set(Config.model_fields)
    extra = {k: raw.pop(k) for k in list(raw) if k not in known}
    cfg = Config.model_validate({**raw, "extra": extra})
    cfg.paths = cfg.paths.resolve(root)
    return cfg


def _abs(p: str, root: Path) -> Path:
    path = Path(p)
    return path if path.is_absolute() else (root / path).resolve()


def tracking_uri(cfg: Config, root: Path = PROJECT_ROOT) -> str:
    """SQLite tracking database (MLflow 3 no longer accepts a plain folder as the tracking store)."""
    env = os.environ.get("MLFLOW_TRACKING_URI")
    if env:
        return env
    db = _abs(cfg.tracking_uri, root)
    db.parent.mkdir(parents=True, exist_ok=True)
    return "sqlite:///" + db.as_posix()


def setup_mlflow(cfg: Config, root: Path = PROJECT_ROOT) -> str:
    """Point MLflow at the shared store and make sure the experiment exists. Returns the experiment id."""
    import mlflow

    mlflow.set_tracking_uri(tracking_uri(cfg, root))
    exp = mlflow.get_experiment_by_name(cfg.experiment)
    if exp is not None:
        mlflow.set_experiment(experiment_id=exp.experiment_id)
        return str(exp.experiment_id)
    artifacts = _abs(cfg.artifact_root, root)
    artifacts.mkdir(parents=True, exist_ok=True)
    exp_id = mlflow.create_experiment(cfg.experiment, artifact_location=artifacts.as_uri())
    mlflow.set_experiment(experiment_id=exp_id)
    return str(exp_id)
