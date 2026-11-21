"""Synthetic hourly market data with the same columns as the real table. No real data is used in tests."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl


def hourly_table(
    start: datetime = datetime(2024, 1, 1, tzinfo=UTC), days: int = 420, seed: int = 0
) -> pl.DataFrame:
    """Prices driven by a daily shape, a weekly pattern, wind and sun, plus slowly changing noise."""
    rng = np.random.default_rng(seed)
    n = days * 24
    ts = [start + timedelta(hours=i) for i in range(n)]
    hour = np.arange(n) % 24
    dow = (np.arange(n) // 24) % 7
    wind = np.clip(0.4 + np.cumsum(rng.normal(0, 0.03, n)) % 0.5, 0, 1)
    sun = np.clip(np.sin((hour - 6) / 12 * np.pi), 0, None) * rng.uniform(0.3, 1.0, days).repeat(24)
    shape = 20 * np.sin((hour - 8) / 24 * 2 * np.pi) + 15 * np.sin((hour - 18) / 12 * 2 * np.pi)
    noise = np.zeros(n)
    for i in range(1, n):
        noise[i] = 0.9 * noise[i - 1] + rng.normal(0, 4)
    price = 90 + shape - 60 * wind - 50 * sun - 12 * (dow >= 5) + noise
    load = 55_000 + 8_000 * np.sin((hour - 9) / 24 * 2 * np.pi) - 6_000 * (dow >= 5)
    return pl.DataFrame(
        {
            "ts_utc": ts,
            "price": price,
            "load": load,
            "wind_onshore": 30_000 * wind,
            "wind_offshore": 6_000 * wind,
            "solar": 40_000 * sun,
            "load_fc": load + rng.normal(0, 500, n),
            "wind_onshore_fc": 30_000 * wind + rng.normal(0, 800, n),
            "wind_offshore_fc": 6_000 * wind + rng.normal(0, 200, n),
            "solar_fc": 40_000 * sun + rng.normal(0, 800, n),
            "wx_wind_on": 3 + 12 * wind,
            "wx_wind_on_pow": wind,
            "wx_wind_off": 4 + 14 * wind,
            "wx_wind_off_pow": np.clip(wind * 1.2, 0, 1),
            "wx_solar": 800 * sun,
            "wx_temp": 10 + 8 * np.sin((hour - 14) / 24 * 2 * np.pi),
        }
    ).with_columns(pl.col("ts_utc").cast(pl.Datetime("ms", "UTC")))
