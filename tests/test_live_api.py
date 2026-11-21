"""The daily job, the scoreboard, the API and MLflow tracking, on synthetic data."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import numpy as np
import polars as pl
import pytest
from fastapi.testclient import TestClient

from gridwise import backtest, decide, live, track
from gridwise.dataset import load_hourly
from gridwise.serve import api

NOW = datetime(2025, 2, 10, 7, 30, tzinfo=UTC)  # decision time for delivery day 2025-02-11
TARGET = "2025-02-11"
# local midnight of the target day (Europe/Berlin is UTC+1 in February)
TARGET_START = datetime(2025, 2, 10, 23, tzinfo=UTC)


@pytest.fixture(scope="module")
def before(cfg) -> pl.DataFrame:
    """The data as it looks in the morning before the auction: no price for the target day yet."""
    return load_hourly(cfg).with_columns(
        pl.when(pl.col("ts_utc") >= TARGET_START).then(None).otherwise(pl.col("price")).alias("price")
    )


@pytest.fixture(scope="module")
def doc(cfg, before) -> dict:
    return live.make_forecast(cfg, NOW, before)


def test_forecast_document(cfg, doc):
    assert doc["delivery_day"] == TARGET and doc["issued_after_prices"] is False
    assert doc["models"] == ["lear", "qgbm"] and doc["inputs"]["prices_until"] == "2025-02-10"
    assert len(doc["hours"]) == 24 and doc["hours"][0]["start_utc"].startswith("2025-02-10T23:00")
    q = np.array([h["quantiles"] for h in doc["hours"]])
    assert q.shape == (24, len(cfg.quantiles)) and (np.diff(q, axis=1) >= 0).all()
    for plan in doc["schedules"].values():
        c, d = np.array(plan["charge_mw"]), np.array(plan["discharge_mw"])
        assert len(c) == 24 and (c * d == 0).all() and c.max() <= cfg.battery.power_mw + 1e-9
    assert doc["expected_profit_eur"]["mean"] >= doc["expected_profit_eur"]["cvar"] - 0.01
    assert "SMARD" in doc["attribution"] and "Open-Meteo" in doc["attribution"]


def test_past_days_come_from_the_cache(cfg, before, doc, monkeypatch):
    def no_refit(*a, **k):
        raise AssertionError("nothing should be re-forecast")

    monkeypatch.setattr(live, "rolling_forecast", no_refit)
    again = live.make_forecast(cfg, NOW, before)
    assert again["hours"] == doc["hours"]  # same inputs, same forecast


def test_first_forecast_is_kept_and_scored_after_prices_publish(cfg, before, doc):
    path = live.write_forecast(cfg, doc)
    assert path.name == f"{TARGET}.json"
    changed = {**doc, "mean": [0.0] * 24}
    live.write_forecast(cfg, changed)
    assert json.loads(path.read_text())["mean"] == doc["mean"]  # never replaced

    board = live.update_scores(cfg, before)  # prices not published yet
    assert board["days_scored"] == 0
    board = live.update_scores(cfg, load_hourly(cfg))  # prices published
    assert board["days_scored"] == 1 and board["first_day"] == TARGET
    assert 0 <= board["coverage_90"] <= 1 and board["mae"] > 0
    p = board["profit_eur_per_day"]
    assert p["perfect"] >= max(p["mean"], p["cvar"], p["profile"]) - 1e-6
    assert board["share_of_perfect"]["perfect"] == 1
    scores = json.loads((cfg.paths.site / "live" / "scores.json").read_text())
    assert len(scores[TARGET]["actual"]) == 24 and scores[TARGET]["hours"] == 24
    assert json.loads((cfg.paths.site / "live" / "index.json").read_text())["forecast_days"] == [TARGET]
    assert live.update_scores(cfg, load_hourly(cfg))["days_scored"] == 1  # scoring twice changes nothing


def test_late_forecast_is_flagged_and_never_scored(cfg):
    late = live.make_forecast(cfg, datetime(2025, 2, 11, 7, 30, tzinfo=UTC), load_hourly(cfg))
    assert late["issued_after_prices"] is True
    live.write_forecast(cfg, late)
    board = live.update_scores(cfg, load_hourly(cfg))
    assert board["days_scored"] == 1 and board["flagged_not_scored"] == ["2025-02-12"]


def test_incomplete_inputs_stop_the_job(cfg, before):
    broken = before.with_columns(
        pl.when(pl.col("ts_utc") >= TARGET_START).then(None).otherwise(pl.col("wx_solar")).alias("wx_solar")
    )
    with pytest.raises(RuntimeError, match="incomplete"):
        live.make_forecast(cfg, NOW, broken)


def test_api(cfg, monkeypatch):
    monkeypatch.setenv("LIVE_DIR", str(cfg.paths.site / "live"))
    client = TestClient(api.app)
    h = client.get("/health").json()
    assert h["status"] == "ok" and h["latest_forecast_day"] == "2025-02-12" and h["days_scored"] == 1
    assert client.get("/forecast/latest").json()["delivery_day"] == "2025-02-12"
    assert client.get(f"/forecast/{TARGET}").json()["delivery_day"] == TARGET
    assert client.get("/forecast/2020-01-01").status_code == 404
    assert client.get("/forecast/not-a-date").status_code == 422
    assert client.get("/scoreboard").json()["days_scored"] == 1

    r = client.post("/schedule", json={"prices": [10, 10, 100, 100]}).json()
    assert r["hours"] == 4 and sum(r["charge_mw"][2:]) == 0 and r["planned_profit_eur"] > 0
    assert r["state_of_charge_mwh"][-1] == 0
    by_day = client.post("/schedule", json={"day": TARGET}).json()
    assert by_day["hours"] == 24 and TARGET in by_day["source"]
    big = client.post("/schedule", json={"prices": [10, 10, 10, 10, 100, 100, 100, 100], "battery": {
        "power_mw": 5, "capacity_mwh": 20, "eff_charge": 0.95, "eff_discharge": 0.95, "max_cycles_per_day": 1,
        "degradation_eur_per_mwh": 0}}).json()  # fmt: skip
    assert max(big["charge_mw"]) == 5 and big["battery"]["capacity_mwh"] == 20
    assert client.post("/schedule", json={}).status_code == 422
    assert client.post("/schedule", json={"prices": [1, 2], "day": TARGET}).status_code == 422
    assert client.post("/schedule", json={"day": "2020-01-01"}).status_code == 404


def test_api_before_anything_is_published(tmp_path, monkeypatch):
    monkeypatch.setenv("LIVE_DIR", str(tmp_path))
    client = TestClient(api.app)
    assert client.get("/health").json() == {"status": "ok", "version": api.__version__, "forecast_days": 0,
                                            "latest_forecast_day": None, "days_scored": 0}  # fmt: skip
    assert client.get("/forecast/latest").status_code == 404
    assert client.get("/scoreboard").status_code == 404


def test_tracking_records_the_run(cfg, tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "sqlite:///" + (tmp_path / "mlflow.db").as_posix())
    cfg.artifact_root = str(tmp_path / "mlruns")
    if not (cfg.paths.reports / "backtest" / "t" / "summary.json").exists():
        backtest.run(cfg, "t", ["naive", "lear", "qgbm"])
    decide.run(cfg, "t")
    run_id = track.log_backtest(cfg, "t")
    assert json.loads((cfg.paths.reports / "run_ids.json").read_text())["t"] == run_id
    import mlflow

    run = mlflow.get_run(run_id)
    assert run.data.metrics["fc.ensemble.mae"] > 0
    assert run.data.metrics["bat.perfect.share_of_perfect"] == 1.0
    assert "bat.ensemble_mean.gain_vs_profile_ci95_lo" in run.data.metrics
    assert run.data.params["feature_set"] == "weather"


def test_monitoring(cfg):
    from gridwise import monitor

    hourly = load_hourly(cfg)
    doc = monitor.write(cfg, hourly)
    names = {c["name"]: c for c in doc["checks"]}
    # one forecast was issued late in the tests above, so the timeliness check must flag it
    assert names["Forecast published on time"]["status"] == "warning"
    assert "1 of 2" in names["Forecast published on time"]["detail"]
    assert names["Weather inputs match the archive"]["status"] == "ok"
    assert doc["rolling"]["days"] == 1 and doc["overall"] == "warning"
    assert set(doc["position_in_history"]) == {"price level", "price spread"}
    assert all(0 <= v <= 1 for v in doc["position_in_history"].values())
    assert json.loads((cfg.paths.site / "live" / "monitoring.json").read_text())["checks"]

    # if the archive later disagreed with what a forecast used, the parity check must catch it
    changed = hourly.with_columns((pl.col("wx_solar") + 50.0).alias("wx_solar"))
    parity = {c["name"]: c for c in monitor.build(cfg, changed)["checks"]}["Weather inputs match the archive"]
    assert parity["status"] == "warning"


def test_position_in_history():
    from gridwise.monitor import percentile_in_history

    calm = np.r_[np.linspace(0, 100, 1000), np.full(30, 50.0)]
    assert percentile_in_history(calm) == pytest.approx(0.5, abs=0.02)
    spike = np.r_[np.linspace(0, 100, 1000), np.full(30, 500.0)]
    assert percentile_in_history(spike) == 1.0  # higher than anything seen: the monitor flags this
    assert percentile_in_history(np.arange(50.0)) is None  # too little history to say
    with_gaps = np.r_[np.full(10, np.nan), calm]
    assert percentile_in_history(with_gaps) == pytest.approx(0.5, abs=0.02)


def test_report_is_built_from_the_recorded_runs(cfg, tmp_path, monkeypatch):
    from gridwise import report

    monkeypatch.setenv("MLFLOW_TRACKING_URI", "sqlite:///" + (tmp_path / "mlflow.db").as_posix())
    cfg.artifact_root = str(tmp_path / "mlruns")
    decide.run(cfg, "t")
    run_id = track.log_backtest(cfg, "t")
    # the report reads fixed names; record the same run under each of them
    names = ("main", "main_price_only", "leaky", "long")
    (cfg.paths.reports / "run_ids.json").write_text(json.dumps(dict.fromkeys(names, run_id)))
    text, dash = report.build(cfg)
    assert run_id in text and "## 1. Forecast accuracy" in text and "## 5. What the forecast is worth" in text
    assert "| 2024 |" in text and "| 2025 |" in text  # the by-year table of the long run
    assert "p < 0.001" in text and "1 MW / 2 MWh, 94% charge" in text
    assert dash["run_id"] == run_id and dash["days"] > 50
    assert dash["models"]["ensemble"]["mae"] > 0 and dash["policies"]["perfect"]["share_of_perfect"] == 1.0
    assert dash["battery"]["capacity_mwh"] == cfg.battery.capacity_mwh
    assert "published after the auction closes" in dash["leak_note"]

    readme = report.render_readme(
        cfg, "__DAYS__ days; __MODEL_ROWS__; __POLICY_ROWS__; __LEAK_SENTENCE__; __GAP_CLOSED__"
    )
    assert "__" not in readme and "GridWise ensemble" in readme and "Perfect foresight" in readme
