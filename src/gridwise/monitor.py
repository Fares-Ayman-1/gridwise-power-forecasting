"""Health checks on the live system, written to docs/live/monitoring.json by the daily job.

Four questions, each answered from files already on disk:

1. Did a forecast get published for every day, and before its prices existed?
2. Do the weather inputs each forecast used match what the archive now holds for that day?
   (They should be identical; a difference would mean the live job saw provisional values.)
3. Are the recent errors and the recent interval coverage in line with the backtest?
4. Is the market still inside what the models were trained on? (where the last 30 days' price level and
   price spread sit among the three years before them)
"""

from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from typing import Any

import numpy as np
import polars as pl

from gridwise.config import Config
from gridwise.features import WEATHER, level_and_scale
from gridwise.frames import to_daily

RECENT_DAYS = 30
MIN_DAYS_TO_JUDGE = 14
PARITY_TOLERANCE = 1e-4  # stored values are rounded to six decimals
TRAINING_DAYS = 1095


def percentile_in_history(
    values: np.ndarray, recent_days: int = RECENT_DAYS, history_days: int = TRAINING_DAYS
) -> float | None:
    """Share of the earlier `history_days` values that lie below the median of the last `recent_days`.

    Near 0 or 1 means the market is at an edge of, or outside, what the models have seen.
    """
    known = values[~np.isnan(values)]
    if len(known) < recent_days + 180:
        return None
    history = known[-(recent_days + history_days) : -recent_days]
    return float(np.mean(history < np.median(known[-recent_days:])))


def _position_text(name: str, share: float) -> str:
    if share > 0.995:
        return f"the recent {name} is higher than at any time in the previous three years"
    if share < 0.005:
        return f"the recent {name} is lower than at any time in the previous three years"
    return f"the recent {name} is higher than on {share:.0%} of days in the previous three years"


def _check(name: str, status: str, detail: str) -> dict[str, str]:
    return {"name": name, "status": status, "detail": detail}


def build(cfg: Config, hourly: pl.DataFrame, today: date | None = None) -> dict[str, Any]:
    live = cfg.paths.site / "live"
    today = today or datetime.now(UTC).date()
    docs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((live / "forecasts").glob("*.json"))]
    scores_path, ref_path = live / "scores.json", live / "backtest.json"
    scores = json.loads(scores_path.read_text(encoding="utf-8")) if scores_path.exists() else {}
    ref = json.loads(ref_path.read_text(encoding="utf-8")) if ref_path.exists() else None
    checks: list[dict[str, str]] = []

    # 1. on time
    days = [date.fromisoformat(d["delivery_day"]) for d in docs]
    on_time = [d for d in docs if not d["issued_after_prices"]]
    if not docs:
        checks.append(_check("Forecast published on time", "warning", "No forecast has been published yet."))
    else:
        missing = [str(days[0] + timedelta(days=i)) for i in range((days[-1] - days[0]).days + 1)]
        missing = [d for d in missing if date.fromisoformat(d) not in set(days)]
        late = len(docs) - len(on_time)
        stale = days[-1] < today
        status = "warning" if (missing or late or stale) else "ok"
        detail = f"{len(on_time)} of {len(docs)} forecasts were issued before their prices."
        if missing:
            detail += f" Missing days: {', '.join(missing[:5])}."
        if stale:
            detail += f" The newest forecast is for {days[-1]}, which is already in the past."
        checks.append(_check("Forecast published on time", status, detail))

    # 2. input parity
    grid = to_daily(hourly, ["price", *WEATHER], cfg.market.timezone)
    worst, compared = 0.0, 0
    for d in docs:
        stored = d["inputs"].get("weather_day_mean")
        day = date.fromisoformat(d["delivery_day"])
        if not stored or day not in grid._index:
            continue
        row = grid.index(day)
        for col, v in stored.items():
            now_value = float(np.nanmean(grid[col][row]))
            if not np.isnan(now_value):
                worst = max(worst, abs(now_value - v) / max(abs(v), 1e-9))
                compared += 1
    if compared:
        ok = worst <= PARITY_TOLERANCE
        checks.append(
            _check(
                "Weather inputs match the archive",
                "ok" if ok else "warning",
                f"{compared} stored values compared; "
                + (
                    "all equal to the archive within rounding."
                    if ok
                    else f"largest relative difference {worst:.2%}."
                ),
            )
        )

    # 3. recent accuracy and coverage
    recent = list(scores.values())[-RECENT_DAYS:]
    rolling: dict[str, float] | None = None
    if recent:
        hours = sum(s["hours"] for s in recent)
        rolling = {
            "days": len(recent),
            "mae": sum(s["mae"] * s["hours"] for s in recent) / hours,
            "crps": sum(s["crps"] * s["hours"] for s in recent) / hours,
            "coverage_80": sum(s["inside_80"] for s in recent) / hours,
            "coverage_90": sum(s["inside_90"] for s in recent) / hours,
        }
        if len(recent) < MIN_DAYS_TO_JUDGE:
            msg = f"Only {len(recent)} scored days; at least {MIN_DAYS_TO_JUDGE} are needed to judge."
            checks.append(_check("Ranges hold as often as they should", "ok", msg))
        else:
            cov = rolling["coverage_90"]
            status = "ok" if 0.82 <= cov <= 0.97 else "warning"
            msg = f"90% range held {cov:.0%} of the time over the last {len(recent)} days."
            checks.append(_check("Ranges hold as often as they should", status, msg))
            if ref:
                base = ref["models"]["ensemble"]["mae"]
                status = "ok" if rolling["mae"] <= 1.5 * base else "warning"
                msg = f"Recent error {rolling['mae']:.1f} EUR/MWh; backtest {base:.1f}."
                checks.append(_check("Error in line with the backtest", status, msg))

    # 4. inside the training range?
    position: dict[str, float] = {}
    level, scale = level_and_scale(grid["price"])
    for name, v in {"price level": level, "price spread": scale}.items():
        pct = percentile_in_history(v)
        if pct is not None:
            position[name] = pct
    if position:
        edge = [k for k, v in position.items() if v < 0.01 or v > 0.99]
        detail = "; ".join(_position_text(n, v) for n, v in position.items())
        note = f" Outside the usual range: {', '.join(edge)}." if edge else ""
        detail = detail[0].upper() + detail[1:] + "." + note
        checks.append(_check("Market inside the range seen in training", "warning" if edge else "ok", detail))

    return {
        "updated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "overall": "warning" if any(c["status"] == "warning" for c in checks) else "ok",
        "checks": checks,
        "rolling": rolling,
        "position_in_history": position,
    }


def write(cfg: Config, hourly: pl.DataFrame) -> dict[str, Any]:
    doc = build(cfg, hourly)
    out = cfg.paths.site / "live" / "monitoring.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    return doc
