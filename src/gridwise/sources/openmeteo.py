"""Open-Meteo Previous Runs API: weather forecasts at a fixed lead time.

Weather data by Open-Meteo.com, CC BY 4.0; the free API is for non-commercial use.

`<variable>_previous_day2` at valid time t is the value from the model run started at
floor_6h(t - 48 h). That was checked on 2026-11-21 against the Single Runs API for ICON-EU, for past
hours and for tomorrow's hours. So for every hour of delivery day D the run started no later than
18:00 UTC on D-2 and its output is published many hours before the 07:30 UTC decision time on D-1.
For hours whose run does not exist yet the API returns a newer run as a placeholder; `final_until`
gives the last valid time that is already final.
"""

from __future__ import annotations

import json
import time
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import httpx
import polars as pl

from gridwise.sources.http import get_json, make_client

BASE = "https://previous-runs-api.open-meteo.com/v1/forecast"
RUN_AVAILABLE_AFTER = timedelta(hours=6)  # a run's output is online well within 6 h of its start


def final_until(now: datetime, lead_days: int) -> datetime:
    """Last valid hour whose lead-`lead_days` forecast is final at time `now`.

    The value for valid time t comes from the run started at floor_6h(t - lead). It is final once that
    run is published, taken here as RUN_AVAILABLE_AFTER after its start.
    """
    ready = now.astimezone(UTC) - RUN_AVAILABLE_AFTER
    last_run = ready.replace(hour=ready.hour // 6 * 6, minute=0, second=0, microsecond=0)
    return last_run + timedelta(days=lead_days, hours=5)


def _months(start: date, end: date) -> list[tuple[date, date]]:
    out, cur = [], start
    while cur <= end:
        nxt = (cur.replace(day=1) + timedelta(days=32)).replace(day=1)
        out.append((cur, min(end, nxt - timedelta(days=1))))
        cur = nxt
    return out


def fetch(
    sites: list[tuple[str, str, float, float]],
    variables: list[str],
    model: str,
    lead_days: int,
    start: date,
    end: date,
    cache_dir: Path,
    client: httpx.Client | None = None,
    pause: float = 4.0,
    today: date | None = None,
) -> pl.DataFrame:
    """Long table: ts_utc, site, group, one column per variable. Cached per calendar month."""
    client = client or make_client()
    today = today or datetime.now(UTC).date()
    cols = [f"{v}_previous_day{lead_days}" for v in variables]
    key = f"{model}_pd{lead_days}_{len(sites)}sites"
    frames = []
    for m_start, m_end in _months(start, end):
        path = cache_dir / key / f"{m_start:%Y-%m}_{m_start:%d}-{m_end:%d}.json"
        # a month is final once its last day is more than a week old
        frozen = m_end < today - timedelta(days=7)
        if path.exists() and frozen:
            doc = json.loads(path.read_text(encoding="utf-8"))
        else:
            doc = get_json(
                client,
                BASE,
                {
                    "latitude": ",".join(str(s[2]) for s in sites),
                    "longitude": ",".join(str(s[3]) for s in sites),
                    "hourly": ",".join(cols),
                    "start_date": m_start.isoformat(),
                    "end_date": m_end.isoformat(),
                    "models": model,
                    "timezone": "UTC",
                    "wind_speed_unit": "ms",
                },
            )
            if isinstance(doc, dict):  # a single site comes back as an object, not a list
                doc = [doc]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(doc), encoding="utf-8")
            time.sleep(pause)  # stay far below the free tier's 600 calls a minute
        if len(doc) != len(sites):
            raise ValueError(f"{path.name}: expected {len(sites)} sites, got {len(doc)}")
        for (group, name, _, _), d in zip(sites, doc, strict=True):
            h = d["hourly"]
            frames.append(
                pl.DataFrame(
                    {"time": h["time"], **{v: h[c] for v, c in zip(variables, cols, strict=True)}},
                    schema_overrides={v: pl.Float64 for v in variables},
                ).with_columns(pl.lit(name).alias("site"), pl.lit(group).alias("group"))
            )
    return (
        pl.concat(frames)
        .with_columns(
            pl.col("time").str.to_datetime("%Y-%m-%dT%H:%M", time_unit="ms").cast(pl.Datetime("ms", "UTC"))
        )
        .rename({"time": "ts_utc"})
        .select("ts_utc", "group", "site", *variables)
        .sort("ts_utc", "group", "site")
    )
