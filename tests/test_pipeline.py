"""End-to-end on synthetic data: config -> backtest -> battery study."""

from __future__ import annotations

import json

import numpy as np
import pytest

from gridwise import backtest, decide
from gridwise.config import Config, load_config
from gridwise.models.baselines import Lear, SimilarDayNaive


@pytest.fixture(scope="module")
def summary(cfg: Config) -> dict:
    return backtest.run(cfg, "t", ["naive", "lear", "qgbm", "qnn"])


def test_config_loads_and_validates():
    c = load_config()
    assert 0.5 in c.quantiles and c.weather.lead_days == 2
    assert {"main", "long", "leaky"} <= set(c.backtests)
    assert len(c.weather.sites()) == sum(len(v) for v in c.weather.locations.values())
    with pytest.raises(ValueError):
        type(c.weather)(**{**c.weather.model_dump(), "lead_days": 1})
    for bad in ([0.1, 0.9], [0.1, 0.5, 0.8], [0.5, 0.1, 0.9]):
        with pytest.raises(ValueError):
            Config(**{**c.model_dump(), "quantiles": bad})


def test_models_beat_the_naive_baseline(summary):
    m = summary["models"]
    assert set(m) == {"naive", "lear", "qgbm", "qnn", "ensemble"}
    assert m["naive"]["rmae"] == 1.0
    for name in ("lear", "qgbm", "qnn", "ensemble"):
        assert m[name]["rmae"] < 0.9, f"{name} should clearly beat the naive forecast on learnable data"
        assert m[name]["dm_p_mae_vs_naive"] < 0.01
    assert m["qgbm"]["days"] == m["naive"]["days"] > 50


def test_calibrated_intervals_cover(summary):
    for name, s in summary["models"].items():
        assert s["coverage_80"] == pytest.approx(0.80, abs=0.08), name
        assert s["coverage_90"] == pytest.approx(0.90, abs=0.07), name
        assert s["width_90"] > s["width_80"] > 0
    # a point forecast has no interval before calibration
    assert summary["models"]["naive"]["coverage_90_uncalibrated"] < 0.1


def test_forecast_files_round_trip(cfg, summary):
    out = cfg.paths.reports / "backtest" / "t"
    f = backtest.Forecasts.load("qgbm", out / "qgbm.npz")
    assert f.q.shape == (len(f.idx), 24, len(cfg.quantiles))
    assert (np.diff(f.q, axis=-1) >= -1e-9).all()  # quantiles do not cross
    assert f.is_test.sum() == summary["models"]["qgbm"]["days"]
    saved = json.loads((out / "summary.json").read_text())
    assert saved["backtest"] == "t" and set(saved["models"]["qgbm"]["by_year"]) == {"2024", "2025"}
    again = backtest.run(cfg, "t", ["naive", "lear", "qgbm", "qnn"], reuse=True)
    assert again["models"]["qgbm"]["mae"] == summary["models"]["qgbm"]["mae"]


def test_training_never_uses_the_forecast_day(cfg):
    """A model asked to forecast day d must be fitted on days before d only."""
    feats = backtest.load_features(cfg, "weather")
    seen: list[tuple[int, int]] = []

    class Spy(SimilarDayNaive):
        refit_every = 5

        def fit(self, feats, train_idx):
            seen.append((int(train_idx.max()), len(train_idx)))

        def predict(self, feats, idx):
            assert int(idx.min()) > seen[-1][0]
            return super().predict(feats, idx)

    idx = np.where(feats.ready)[0][-40:]
    out = backtest.rolling_forecast(Spy(), feats, idx, min_train_days=50)
    assert out.shape == (40, 24) and len(seen) in (8, 9)  # 40 days in calendar-anchored blocks of 5
    # the block a day belongs to does not depend on where the run starts
    later = backtest.rolling_forecast(Spy(), feats, idx[3:], min_train_days=50)
    assert np.array_equal(out[3:], later)
    starts = {backtest.block_start(feats, int(d), 5) for d in idx}
    assert all(int(feats.days[s].astype("int64")) % 5 == 0 for s in starts)
    with pytest.raises(ValueError):
        backtest.rolling_forecast(Lear(train_days=10), feats, idx, min_train_days=50)


def test_battery_study(cfg, summary):
    s = decide.run(cfg, "t")
    p = s["policies"]
    assert s["days"] > 30
    assert p["perfect"]["share_of_perfect"] == 1.0
    for name, v in p.items():
        assert v["eur_per_day"] <= p["perfect"]["eur_per_day"] + 1e-6, name
    assert p["ensemble_mean"]["eur_per_day"] > p["naive"]["eur_per_day"]
    assert "gain_vs_profile_ci95" in p["ensemble_mean"]
    decide.print_summary(s)
    backtest.print_summary(summary)


def test_unknown_model_name(cfg):
    with pytest.raises(ValueError):
        backtest.make_models(cfg, "weather", ["prophet"])


def test_comparison_runs_score_the_same_days(cfg, summary):
    from datetime import date

    from gridwise.config import Backtest

    cfg.backtests["shorter"] = Backtest(
        feature_set="weather", test_start=date(2024, 12, 15), warmup_days=60, min_train_days=120
    )
    backtest.run(cfg, "shorter", ["naive"])
    cfg.backtests["t_same"] = Backtest(feature_set="price", test_start=date(2024, 11, 1), warmup_days=60,
                                       min_train_days=120, same_days_as="shorter")  # fmt: skip
    s = backtest.run(cfg, "t_same", ["naive"])
    f = backtest.Forecasts.load("naive", cfg.paths.reports / "backtest" / "t_same" / "naive.npz")
    assert f.days[f.is_test][0] == np.datetime64("2024-12-15")  # not its own test_start
    assert s["models"]["naive"]["days"] == f.is_test.sum() < summary["models"]["naive"]["days"]
