from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from gridwise.config import Backtest, Config, load_config
from gridwise.features import OPERATOR, WEATHER, Features, build
from gridwise.frames import Daily, to_daily
from tests.synthetic import hourly_table

TZ = "Europe/Berlin"


@pytest.fixture(scope="session")
def hourly() -> pl.DataFrame:
    return hourly_table()


@pytest.fixture(scope="session")
def daily(hourly: pl.DataFrame) -> Daily:
    return to_daily(hourly, ["price", *WEATHER, *OPERATOR], TZ)


@pytest.fixture(scope="session")
def feats(daily: Daily) -> Features:
    return build(daily, "weather")


@pytest.fixture(scope="session")
def cfg(tmp_path_factory) -> Config:
    """The real configuration pointed at a temporary folder holding synthetic data, with small fast models."""
    root = tmp_path_factory.mktemp("proj")
    c = load_config()
    c.paths = c.paths.model_copy(
        update={k: root / k for k in ("raw", "interim", "models", "reports", "site")}
    )
    c.paths.interim.mkdir(parents=True)
    hourly_table(days=420).write_parquet(c.paths.interim / "hourly.parquet")
    c.backtests = {
        "t": Backtest(feature_set="weather", test_start=date(2024, 11, 1), warmup_days=60, min_train_days=120)
    }
    c.models = {
        "lear": {"train_days": 200, "refit_every": 14},
        "qgbm": {"train_days": 400, "refit_every": 60, "num_boost_round": 60, "params": {"num_threads": 2}},
        "qnn": {"train_days": 400, "refit_every": 90, "seeds": 1, "max_epochs": 30, "threads": 2},
    }
    c.risk = c.risk.model_copy(update={"scenarios": 40, "copula_days": 60})
    c.live = c.live.model_copy(
        update={"history_days": 120, "min_train_days": 120, "models": ["lear", "qgbm"]}
    )
    return c
