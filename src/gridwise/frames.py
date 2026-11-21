"""Hourly UTC series <-> a day x 24 grid in market local time.

The auction is held per local calendar day. Twice a year that day has 23 or 25 hours. The grid always
has 24 slots (local hours 0..23):

- on the 25-hour day the two hours with local hour 2 are averaged into slot 2;
- on the 23-hour day slot 2 does not exist; it is filled with the mean of slots 1 and 3 so that lag
  features stay complete, and `real` marks it as not a real hour so it is never scored.

Battery schedules use `day_hours`, which lists the real hours of a day, so money is always computed on
real hourly prices.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl

SLOTS = 24


def day_hours(day: date, tz: str) -> list[tuple[datetime, int]]:
    """The real hours of a local delivery day as (UTC start, slot). 23, 24 or 25 entries."""
    zone = ZoneInfo(tz)
    start = datetime.combine(day, datetime.min.time(), tzinfo=zone).astimezone(UTC)
    end = datetime.combine(day + timedelta(days=1), datetime.min.time(), tzinfo=zone).astimezone(UTC)
    out, t = [], start
    while t < end:
        out.append((t, t.astimezone(zone).hour))
        t += timedelta(hours=1)
    return out


@dataclass
class Daily:
    """Day x slot arrays. `days` are consecutive local dates; values are NaN where unknown."""

    days: np.ndarray  # datetime64[D], shape (n,)
    values: dict[str, np.ndarray]  # each float64, shape (n, 24)
    real: np.ndarray  # bool, shape (n, 24): False for the slot that does not exist on 23-hour days
    tz: str = "Europe/Berlin"
    _index: dict[date, int] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self._index = {d.astype(object): i for i, d in enumerate(self.days)}

    def __len__(self) -> int:
        return len(self.days)

    def index(self, day: date) -> int:
        return self._index[day]

    def day(self, i: int) -> date:
        return self.days[i].astype(object)

    def __getitem__(self, col: str) -> np.ndarray:
        return self.values[col]


def to_daily(hourly: pl.DataFrame, cols: list[str], tz: str, last_day: date | None = None) -> Daily:
    """Pivot hourly columns onto the day x slot grid. The grid runs from the first to the last local day."""
    local = pl.col("ts_utc").dt.convert_time_zone(tz)
    df = hourly.select("ts_utc", *cols).with_columns(
        local.dt.date().alias("day"), local.dt.hour().alias("slot")
    )
    agg = df.group_by("day", "slot").agg([pl.col(c).mean() for c in cols])
    first = agg["day"].min()
    last = last_day or agg["day"].max()
    assert isinstance(first, date) and isinstance(last, date)
    n = (last - first).days + 1
    days = np.arange(np.datetime64(first), np.datetime64(first) + n)
    values = {c: np.full((n, SLOTS), np.nan) for c in cols}
    di = (agg["day"].to_numpy().astype("datetime64[D]") - np.datetime64(first)).astype(int)
    si = agg["slot"].to_numpy().astype(int)
    keep = (di >= 0) & (di < n)
    for c in cols:
        values[c][di[keep], si[keep]] = agg[c].to_numpy().astype(float)[keep]

    real = np.ones((n, SLOTS), dtype=bool)
    for i, d in enumerate(days):
        hours = day_hours(d.astype(object), tz)
        if len(hours) == SLOTS - 1:  # spring forward: one local hour is skipped
            missing = (set(range(SLOTS)) - {s for _, s in hours}).pop()
            real[i, missing] = False
            lo, hi = max(missing - 1, 0), min(missing + 1, SLOTS - 1)
            for c in cols:
                values[c][i, missing] = np.nanmean([values[c][i, lo], values[c][i, hi]])
    return Daily(days=days, values=values, real=real, tz=tz)


def hourly_lookup(hourly: pl.DataFrame, col: str) -> dict[datetime, float]:
    """UTC hour -> value, for scoring schedules on real hourly prices."""
    sub = hourly.select("ts_utc", col).drop_nulls()
    return dict(zip(sub["ts_utc"].to_list(), sub[col].to_list(), strict=True))
