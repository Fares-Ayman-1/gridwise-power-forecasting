"""HTTP API over the live record, plus a battery planner.

    uv run uvicorn gridwise.serve.api:app --port 8000

GET  /health                 status and the latest forecast day
GET  /forecast/latest        the newest forecast file
GET  /forecast/{day}         the forecast for one delivery day (YYYY-MM-DD)
GET  /scoreboard             the live scoreboard
POST /schedule               plan a battery on your own prices, or on a stored forecast, with your own battery

The API reads the JSON files the daily job writes (LIVE_DIR, default docs/live); it trains nothing.
"""

from __future__ import annotations

import json
import os
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, model_validator

from gridwise import __version__
from gridwise.battery import BatterySpec, plan, profit
from gridwise.config import PROJECT_ROOT

app = FastAPI(
    title="GridWise",
    version=__version__,
    description="Day-ahead power price forecasts for Germany and battery schedules built on them.",
)

DEFAULT_BATTERY = BatterySpec(power_mw=1.0, capacity_mwh=2.0, eff_charge=0.94, eff_discharge=0.94,
                              max_cycles_per_day=2.0, degradation_eur_per_mwh=10.0)  # fmt: skip


def live_dir() -> Path:
    return Path(os.environ.get("LIVE_DIR", PROJECT_ROOT / "docs" / "live"))


def _read(path: Path) -> Any:
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"{path.name} not found")
    return json.loads(path.read_text(encoding="utf-8"))


def _forecast_days() -> list[str]:
    return sorted(p.stem for p in (live_dir() / "forecasts").glob("*.json"))


class ScheduleRequest(BaseModel):
    prices: list[float] | None = Field(
        default=None, min_length=2, max_length=100, description="EUR/MWh per hour"
    )
    day: date | None = Field(
        default=None, description="plan on the stored mean forecast of this delivery day"
    )
    battery: BatterySpec | None = None

    @model_validator(mode="after")
    def _one_source(self) -> ScheduleRequest:
        if (self.prices is None) == (self.day is None):
            raise ValueError("give exactly one of `prices` or `day`")
        return self


class ScheduleResponse(BaseModel):
    source: str
    hours: int
    charge_mw: list[float]
    discharge_mw: list[float]
    state_of_charge_mwh: list[float]
    planned_profit_eur: float
    battery: BatterySpec


@app.get("/health")
def health() -> dict[str, Any]:
    days = _forecast_days()
    board_path = live_dir() / "scoreboard.json"
    scored = json.loads(board_path.read_text(encoding="utf-8"))["days_scored"] if board_path.exists() else 0
    return {"status": "ok", "version": __version__, "forecast_days": len(days),
            "latest_forecast_day": days[-1] if days else None, "days_scored": scored}  # fmt: skip


@app.get("/forecast/latest")
def forecast_latest() -> Any:
    days = _forecast_days()
    if not days:
        raise HTTPException(status_code=404, detail="no forecast has been written yet")
    return _read(live_dir() / "forecasts" / f"{days[-1]}.json")


@app.get("/forecast/{day}")
def forecast_day(day: date) -> Any:
    return _read(live_dir() / "forecasts" / f"{day.isoformat()}.json")


@app.get("/scoreboard")
def scoreboard() -> Any:
    return _read(live_dir() / "scoreboard.json")


@app.post("/schedule", response_model=ScheduleResponse)
def schedule(req: ScheduleRequest) -> ScheduleResponse:
    spec = req.battery or DEFAULT_BATTERY
    if req.prices is not None:
        prices, source = np.asarray(req.prices, dtype=float), "prices in the request"
    else:
        assert req.day is not None
        doc = _read(live_dir() / "forecasts" / f"{req.day.isoformat()}.json")
        prices, source = np.asarray(doc["mean"], dtype=float), f"mean forecast for {req.day.isoformat()}"
    if not np.isfinite(prices).all():
        raise HTTPException(status_code=422, detail="prices must be finite numbers")
    s = plan(prices, spec)
    return ScheduleResponse(
        source=source,
        hours=len(prices),
        charge_mw=np.round(s.charge, 4).tolist(),
        discharge_mw=np.round(s.discharge, 4).tolist(),
        state_of_charge_mwh=np.round(s.soc, 4).tolist(),
        planned_profit_eur=round(profit(s, prices, spec), 2),
        battery=spec,
    )
