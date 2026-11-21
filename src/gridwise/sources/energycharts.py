"""Energy-Charts (Fraunhofer ISE) day-ahead prices, used only to fill gaps in the SMARD price series.

The API states its price data as "CC BY 4.0 from Bundesnetzagentur | SMARD.de". It serves 15-minute
prices since the auction moved to 15-minute products; the hourly mean equals SMARD's hourly value
(checked on 2026-11-21: largest difference 0.005 EUR/MWh over three days).
"""

from __future__ import annotations

from datetime import date

import httpx
import polars as pl

from gridwise.sources.http import get_json, make_client

BASE = "https://api.energy-charts.info/price"


def hourly_prices(
    start: date, end: date, zone: str = "DE-LU", client: httpx.Client | None = None
) -> pl.DataFrame:
    """Hourly mean price for local days start..end as columns ts_utc, price. Incomplete hours are dropped
    unless the source itself is hourly."""
    doc = get_json(
        client or make_client(), BASE, {"bzn": zone, "start": start.isoformat(), "end": end.isoformat()}
    )
    df = pl.DataFrame(
        {"s": doc["unix_seconds"], "price": doc["price"]}, schema={"s": pl.Int64, "price": pl.Float64}
    )
    if df.height == 0:
        return pl.DataFrame(schema={"ts_utc": pl.Datetime("ms", "UTC"), "price": pl.Float64})
    step = int(df["s"].diff().drop_nulls().mode()[0]) if df.height > 1 else 3600
    per_hour = max(1, 3600 // step)
    return (
        df.drop_nulls()
        .with_columns((pl.col("s") // 3600 * 3600 * 1000).alias("ms"))
        .group_by("ms")
        .agg(pl.col("price").mean(), pl.len().alias("n"))
        .filter(pl.col("n") == per_hour)
        .with_columns(pl.from_epoch("ms", time_unit="ms").cast(pl.Datetime("ms", "UTC")).alias("ts_utc"))
        .select("ts_utc", "price")
        .sort("ts_utc")
    )
