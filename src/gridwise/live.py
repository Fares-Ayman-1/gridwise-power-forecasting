"""The daily live job.

    uv run python -m gridwise.live run        # score past forecasts, then forecast tomorrow

`forecast` runs in the morning of day D-1, before the 12:00 auction for delivery day D:
it refreshes the data, re-forecasts the recent past to fill the calibration window (cached), forecasts
day D, plans the battery and writes docs/live/forecasts/<D>.json. The file is committed before the
real prices exist, so the record cannot be back-filled.

`score` runs later: once the prices of a forecast day are published it writes the day's errors and the
money each schedule made, and rebuilds docs/live/scoreboard.json.

A forecast written after its prices were already published is kept but flagged and never scored.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np

from gridwise import metrics as M
from gridwise import monitor
from gridwise.backtest import Forecasts, ensemble, interval, make_models, rolling_forecast, to_forecasts
from gridwise.battery import Schedule, plan, plan_cvar, profit
from gridwise.config import PROJECT_ROOT, Config, load_config
from gridwise.dataset import load_hourly, refresh
from gridwise.features import OPERATOR, WEATHER, Features, build
from gridwise.frames import day_hours, hourly_lookup, to_daily
from gridwise.models.base import Forecaster
from gridwise.scenarios import copula_correlation, forecast_mean, pit, sample

ATTRIBUTION = "Prices: Bundesnetzagentur | SMARD.de (CC BY 4.0). Weather: Open-Meteo.com (CC BY 4.0)."


def git_sha(root: Path = PROJECT_ROOT) -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=root, capture_output=True, text=True
        )
        return out.stdout.strip() or "unknown"
    except OSError:
        return "unknown"


def _signature(model: Forecaster, feats: Features, cfg: Config) -> str:
    """Changes whenever anything that affects a raw forecast changes, which invalidates the cache."""
    spec = {
        "model": model.name,
        "settings": cfg.models.get(model.name, {}),
        "features": feats.names,
        "taus": cfg.quantiles,
        "weather": [cfg.weather.model, cfg.weather.lead_days, cfg.weather.sites()],
    }
    return hashlib.sha256(json.dumps(spec, sort_keys=True, default=str).encode()).hexdigest()[:16]


def cached_raw(model: Forecaster, feats: Features, idx: np.ndarray, cfg: Config) -> np.ndarray:
    """Raw forecasts for `idx`, computing only the days that are not in the on-disk cache."""
    path = cfg.paths.interim / "raw_cache" / f"{model.name}.npz"
    sig = _signature(model, feats, cfg)
    have: dict[str, np.ndarray] = {}
    if path.exists():
        z = np.load(path)
        if str(z["signature"]) == sig:
            have = dict(zip(z["days"].tolist(), z["raw"], strict=True))
    days = feats.days[idx].astype("datetime64[D]").astype(str)
    todo = np.array([i for i, d in zip(idx, days, strict=True) if d not in have], dtype=int)
    if len(todo):
        new = rolling_forecast(model, feats, todo, cfg.live.min_train_days)
        for d, r in zip(feats.days[todo].astype("datetime64[D]").astype(str), new, strict=True):
            have[d] = r
        keep = sorted(have)[-(cfg.live.history_days + 60) :]
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, signature=sig, days=np.array(keep), raw=np.stack([have[d] for d in keep]))
    return np.stack([have[d] for d in days])


def _schedule_json(s: Schedule) -> dict[str, list[float]]:
    return {"charge_mw": np.round(s.charge, 4).tolist(), "discharge_mw": np.round(s.discharge, 4).tolist()}


def make_forecast(cfg: Config, now: datetime, hourly: Any | None = None) -> dict[str, Any]:
    """Forecast and schedule for the delivery day after `now`'s local date. Does not write anything."""
    tz = cfg.market.timezone
    target: date = now.astimezone(ZoneInfo(tz)).date() + timedelta(days=1)
    hourly = load_hourly(cfg) if hourly is None else hourly
    daily = to_daily(hourly, ["price", *WEATHER, *OPERATOR], tz, last_day=target)
    feats = build(daily, cfg.live.feature_set)
    d = daily.index(target)
    if not feats.ready[d]:
        missing = [n for n, col in zip(feats.names, feats.X[d].T, strict=True) if np.isnan(col).any()]
        raise RuntimeError(f"inputs for {target} are incomplete: {missing[:8]}")
    prices_already_known = bool((~np.isnan(feats.price[d])).any())

    lo = np.datetime64(target - timedelta(days=cfg.live.history_days))
    past = np.where(
        feats.ready & (feats.days >= lo) & (np.arange(len(daily)) < d) & ~np.isnan(feats.y).any(axis=1)
    )[0]
    idx = np.r_[past, d]
    members: list[Forecasts] = []
    for model in make_models(cfg, cfg.live.feature_set, cfg.live.models):
        raw = cached_raw(model, feats, idx, cfg)
        members.append(to_forecasts(model.name, model.kind, feats, idx, raw, cfg, target))
    final = ensemble(members, feats, cfg) if len(members) > 1 else members[0]

    taus = np.asarray(cfg.quantiles)
    hours = day_hours(target, tz)
    slots = np.array([s for _, s in hours])
    q_day = final.q[-1][slots]  # (H, Q)
    mean = forecast_mean(q_day, taus)
    u = pit(final.q[:-1], taus, final.y[:-1])
    corr = copula_correlation(u[-cfg.risk.copula_days :])[np.ix_(slots, slots)]
    if len(slots) != 24:
        corr = 0.98 * corr + 0.02 * np.eye(len(slots))
    scen = sample(
        q_day, taus, corr, cfg.risk.scenarios, np.random.default_rng(int(target.strftime("%Y%m%d")))
    )
    plans = {
        "mean": plan(mean, cfg.battery),
        "cvar": plan_cvar(scen, cfg.battery, cfg.risk.alpha, cfg.risk.weight),
    }
    last_price_day = feats.days[np.where(~np.isnan(feats.price).any(axis=1))[0][-1]]
    return {
        "delivery_day": target.isoformat(),
        "issued_at_utc": now.astimezone(UTC).isoformat(timespec="seconds"),
        "issued_after_prices": prices_already_known,
        "code_version": git_sha(),
        "models": [m.name for m in members],
        "calibration": cfg.calibration.model_dump(),
        "history_days_used": len(past),
        "inputs": {
            "prices_until": str(last_price_day),
            "weather_model": cfg.weather.model,
            "weather_lead_days": cfg.weather.lead_days,
            # kept so the monitor can later confirm the archive holds the same values
            "weather_day_mean": {
                c: round(float(np.nanmean(daily[c][d])), 6)
                for c in WEATHER
                if cfg.live.feature_set == "weather"
            },
        },
        "quantile_levels": cfg.quantiles,
        "hours": [
            {
                "start_utc": t.isoformat(timespec="minutes"),
                "local_hour": int(s),
                "quantiles": np.round(q, 2).tolist(),
            }
            for (t, s), q in zip(hours, q_day, strict=True)
        ],
        "mean": np.round(mean, 2).tolist(),
        "member_medians": {
            m.name: np.round(m.q[-1][slots][:, cfg.quantiles.index(0.5)], 2).tolist() for m in members
        },
        "battery": cfg.battery.model_dump(),
        "schedules": {k: _schedule_json(s) for k, s in plans.items()},
        "expected_profit_eur": {k: round(profit(s, mean, cfg.battery), 2) for k, s in plans.items()},
        "attribution": ATTRIBUTION,
    }


def write_forecast(cfg: Config, doc: dict[str, Any]) -> Path:
    """Write the forecast file. An existing file is never replaced: a day's first forecast is the record."""
    path = cfg.paths.site / "live" / "forecasts" / f"{doc['delivery_day']}.json"
    if path.exists():
        print(f"{path.name} already exists; keeping the first forecast")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    return path


def _schedule(doc: dict[str, Any], key: str) -> Schedule:
    s = doc["schedules"][key]
    c, d = np.array(s["charge_mw"]), np.array(s["discharge_mw"])
    return Schedule(c, d, np.zeros_like(c), "stored")


def score_day(
    cfg: Config, doc: dict[str, Any], actual: dict[datetime, float], profile_grid: Any
) -> dict[str, Any] | None:
    """Score one forecast file. Returns None while the day's prices are not all published."""
    hours = [datetime.fromisoformat(h["start_utc"]) for h in doc["hours"]]
    if any(t not in actual for t in hours):
        return None
    taus = np.asarray(doc["quantile_levels"])
    y = np.array([actual[t] for t in hours])
    q = np.array([h["quantiles"] for h in doc["hours"]])
    slots = np.array([h["local_hour"] for h in doc["hours"]])
    spec = cfg.battery.model_copy(update=doc["battery"])
    day = date.fromisoformat(doc["delivery_day"])
    g = profile_grid.index(day)
    profile = np.nanmean(profile_grid["price"][g - cfg.live.profile_days : g], axis=0)[slots]
    out: dict[str, Any] = {
        "delivery_day": doc["delivery_day"],
        "actual": np.round(y, 2).tolist(),
        "mae": M.mae(q[None, :, list(taus).index(0.5)], y[None]),
        "crps": M.crps(q[None], y[None], taus),
        "hours": len(y),
        "profit_eur": {
            "perfect": profit(plan(y, spec), y, spec),
            "profile": profit(plan(profile, spec), y, spec),
            **{k: profit(_schedule(doc, k), y, spec) for k in doc["schedules"]},
        },
    }
    for nominal in (0.8, 0.9):
        lo, hi = interval(q, taus, nominal)
        out[f"inside_{round(nominal * 100)}"] = int(np.sum((y >= lo) & (y <= hi)))
    return out


def update_scores(cfg: Config, hourly: Any | None = None) -> dict[str, Any]:
    """Score every unscored forecast whose prices are published, then rebuild the scoreboard."""
    live = cfg.paths.site / "live"
    live.mkdir(parents=True, exist_ok=True)
    scores_path = live / "scores.json"
    scores: dict[str, Any] = (
        json.loads(scores_path.read_text(encoding="utf-8")) if scores_path.exists() else {}
    )
    hourly = load_hourly(cfg) if hourly is None else hourly
    actual = hourly_lookup(hourly, "price")
    grid = to_daily(hourly, ["price"], cfg.market.timezone)
    flagged = []
    for path in sorted((live / "forecasts").glob("*.json")):
        doc = json.loads(path.read_text(encoding="utf-8"))
        if doc["issued_after_prices"]:
            flagged.append(doc["delivery_day"])
            continue
        if doc["delivery_day"] not in scores:
            s = score_day(cfg, doc, actual, grid)
            if s is not None:
                scores[doc["delivery_day"]] = s
    scores = dict(sorted(scores.items()))
    scores_path.write_text(json.dumps(scores, indent=1), encoding="utf-8")

    board: dict[str, Any] = {
        "updated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "days_scored": len(scores),
        "first_day": next(iter(scores), None),
        "last_day": next(reversed(scores), None),
        "flagged_not_scored": flagged,
        "attribution": ATTRIBUTION,
    }
    if scores:
        hours = sum(s["hours"] for s in scores.values())
        policies = list(next(iter(scores.values()))["profit_eur"])
        total = {k: sum(s["profit_eur"][k] for s in scores.values()) for k in policies}
        board |= {
            "mae": sum(s["mae"] * s["hours"] for s in scores.values()) / hours,
            "crps": sum(s["crps"] * s["hours"] for s in scores.values()) / hours,
            "coverage_80": sum(s["inside_80"] for s in scores.values()) / hours,
            "coverage_90": sum(s["inside_90"] for s in scores.values()) / hours,
            "profit_eur_per_day": {k: v / len(scores) for k, v in total.items()},
            "share_of_perfect": {
                k: (v / total["perfect"] if total["perfect"] else None) for k, v in total.items()
            },
            "daily": [
                {
                    "day": d,
                    "mae": round(s["mae"], 2),
                    "crps": round(s["crps"], 2),
                    "profit_eur": {k: round(v, 2) for k, v in s["profit_eur"].items()},
                }
                for d, s in scores.items()
            ],
        }
    (live / "scoreboard.json").write_text(json.dumps(board, indent=1), encoding="utf-8")
    index = sorted(p.stem for p in (live / "forecasts").glob("*.json"))
    (live / "index.json").write_text(json.dumps({"forecast_days": index}), encoding="utf-8")
    return board


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=["run", "forecast", "score"])
    ap.add_argument("--config", default=None)
    ap.add_argument("--no-refresh", action="store_true", help="use the data already on disk")
    a = ap.parse_args()
    cfg = load_config(a.config)
    now = datetime.now(UTC)
    hourly = load_hourly(cfg) if a.no_refresh else refresh(cfg, now)
    if a.command in ("run", "score"):
        board = update_scores(cfg, hourly)
        print(f"scoreboard: {board['days_scored']} days scored")
    if a.command in ("run", "forecast"):
        doc = make_forecast(cfg, now, hourly)
        path = write_forecast(cfg, doc)
        flag = (
            " (issued after prices were published: flagged, will not be scored)"
            if doc["issued_after_prices"]
            else ""
        )
        print(f"forecast for {doc['delivery_day']} -> {path}{flag}")
        update_scores(cfg, hourly)
    health = monitor.write(cfg, hourly)
    for c in health["checks"]:
        print(f"  [{c['status']}] {c['name']}: {c['detail']}")


if __name__ == "__main__":
    main()
