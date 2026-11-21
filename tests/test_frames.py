from datetime import UTC, date, datetime, timedelta

import numpy as np
import polars as pl

from gridwise.frames import day_hours, hourly_lookup, to_daily

TZ = "Europe/Berlin"


def test_day_lengths_follow_daylight_saving():
    assert len(day_hours(date(2025, 3, 30), TZ)) == 23
    assert len(day_hours(date(2025, 10, 26), TZ)) == 25
    assert len(day_hours(date(2025, 6, 1), TZ)) == 24
    slots = [s for _, s in day_hours(date(2025, 10, 26), TZ)]
    assert slots.count(2) == 2  # the repeated hour
    assert 2 not in [s for _, s in day_hours(date(2025, 3, 30), TZ)]  # the skipped hour


def _table(start: datetime, hours: int) -> pl.DataFrame:
    ts = [start + timedelta(hours=i) for i in range(hours)]
    return pl.DataFrame({"ts_utc": ts, "price": np.arange(hours, dtype=float)}).with_columns(
        pl.col("ts_utc").cast(pl.Datetime("ms", "UTC"))
    )


def test_grid_handles_the_25_hour_day():
    # 2025-10-25 22:00 UTC is local midnight of 2025-10-26
    df = _table(datetime(2025, 10, 25, 22, tzinfo=UTC), 25)
    g = to_daily(df, ["price"], TZ)
    row = g["price"][g.index(date(2025, 10, 26))]
    assert row[2] == 2.5  # mean of the two 02:00 hours (values 2 and 3)
    assert row[3] == 4 and row[23] == 24
    assert g.real[g.index(date(2025, 10, 26))].all()


def test_grid_handles_the_23_hour_day():
    df = _table(datetime(2025, 3, 29, 23, tzinfo=UTC), 23)
    g = to_daily(df, ["price"], TZ)
    i = g.index(date(2025, 3, 30))
    assert not g.real[i, 2] and g.real[i].sum() == 23
    assert g["price"][i, 2] == np.mean([g["price"][i, 1], g["price"][i, 3]])
    assert not np.isnan(g["price"][i]).any()


def test_grid_is_consecutive_and_lookup_matches(hourly):
    g = to_daily(hourly, ["price"], TZ)
    assert (np.diff(g.days).astype(int) == 1).all()
    look = hourly_lookup(hourly, "price")
    t, slot = day_hours(g.day(10), TZ)[5]
    assert g["price"][10, slot] == look[t]
