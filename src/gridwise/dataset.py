"""Download the sources and assemble one hourly table.

    uv run python -m gridwise.dataset            # download or refresh, then build data/interim/hourly.parquet

Columns of hourly.parquet: ts_utc, the SMARD series (price in EUR/MWh, the others in MWh per hour) and
the weather aggregates wx_* built from forecasts at the configured lead time.
"""

from __future__ import annotations

import argparse
from datetime import UTC, date, datetime, timedelta
from zoneinfo import ZoneInfo

import polars as pl

from gridwise.config import Config, load_config
from gridwise.sources import energycharts, openmeteo, smard

# Simple wind power curve: nothing below 3 m/s, rated output from 12 m/s, shut down above 25 m/s.
CUT_IN, RATED, CUT_OUT = 3.0, 12.0, 25.0


def power_curve(speed: pl.Expr) -> pl.Expr:
    frac = ((speed**3 - CUT_IN**3) / (RATED**3 - CUT_IN**3)).clip(0.0, 1.0)
    return pl.when(speed > CUT_OUT).then(0.0).otherwise(frac)


def aggregate_weather(sites: pl.DataFrame) -> pl.DataFrame:
    """Site-level forecasts -> one row per hour with a few national proxies."""

    def group_mean(group: str, expr: pl.Expr, name: str) -> pl.DataFrame:
        return sites.filter(pl.col("group") == group).group_by("ts_utc").agg(expr.mean().alias(name))

    parts = [
        group_mean("wind_onshore", pl.col("wind_speed_100m"), "wx_wind_on"),
        group_mean("wind_onshore", power_curve(pl.col("wind_speed_100m")), "wx_wind_on_pow"),
        group_mean("wind_offshore", pl.col("wind_speed_100m"), "wx_wind_off"),
        group_mean("wind_offshore", power_curve(pl.col("wind_speed_100m")), "wx_wind_off_pow"),
        group_mean("solar", pl.col("shortwave_radiation"), "wx_solar"),
        group_mean("temperature", pl.col("temperature_2m"), "wx_temp"),
    ]
    out = parts[0]
    for p in parts[1:]:
        out = out.join(p, on="ts_utc", how="full", coalesce=True)
    return out.sort("ts_utc")


WX_COLS = ["wx_wind_on", "wx_wind_on_pow", "wx_wind_off", "wx_wind_off_pow", "wx_solar", "wx_temp"]


def fill_price_gaps(market: pl.DataFrame, now: datetime, lookback_days: int = 21) -> pl.DataFrame:
    """Fill missing recent prices from Energy-Charts. SMARD's chart data sometimes lags by a day or two."""
    recent = market.filter(pl.col("ts_utc") >= now - timedelta(days=lookback_days))
    gaps = recent.filter(pl.col("price").is_null() & (pl.col("ts_utc") <= now + timedelta(days=2)))
    if gaps.height == 0:
        return market
    first = gaps["ts_utc"].min()
    assert isinstance(first, datetime)
    try:
        other = energycharts.hourly_prices(first.date() - timedelta(days=1), now.date() + timedelta(days=1))
    except Exception as e:  # the fallback is best effort; the caller checks that the needed days exist
        print(f"price gap fill skipped: {e}")
        return market
    other = other.rename({"price": "price_fill"})
    out = market.join(other, on="ts_utc", how="full", coalesce=True).sort("ts_utc")
    filled = out.filter(pl.col("price").is_null() & pl.col("price_fill").is_not_null()).height
    print(f"filled {filled} missing price hours from Energy-Charts")
    return out.with_columns(pl.coalesce("price", "price_fill").alias("price")).drop("price_fill")


def refresh(cfg: Config, now: datetime | None = None, weather_end: date | None = None) -> pl.DataFrame:
    """Download what is missing and write the interim tables. Returns the hourly table."""
    now = now or datetime.now(UTC)
    tz = ZoneInfo(cfg.market.timezone)
    start = datetime.combine(cfg.market.history_start, datetime.min.time(), tzinfo=tz)
    cfg.paths.interim.mkdir(parents=True, exist_ok=True)

    market = smard.download(cfg.smard.series, cfg.smard.resolution, start, cfg.paths.raw / "smard")
    market = fill_price_gaps(market, now)
    market.write_parquet(cfg.paths.interim / "smard_hourly.parquet")

    # weather up to the day after tomorrow covers every hour of tomorrow's delivery day in UTC
    w_end = weather_end or (now.date() + timedelta(days=2))
    sites = openmeteo.fetch(
        cfg.weather.sites(),
        cfg.weather.variables,
        cfg.weather.model,
        cfg.weather.lead_days,
        cfg.weather.start,
        w_end,
        cfg.paths.raw / "openmeteo",
    )
    # drop placeholder values: hours whose run has not been published yet
    sites = sites.filter(pl.col("ts_utc") <= openmeteo.final_until(now, cfg.weather.lead_days))
    sites.write_parquet(cfg.paths.interim / "weather_sites.parquet")

    hourly = market.join(aggregate_weather(sites), on="ts_utc", how="full", coalesce=True).sort("ts_utc")
    hourly.write_parquet(cfg.paths.interim / "hourly.parquet")
    return hourly


def load_hourly(cfg: Config) -> pl.DataFrame:
    return pl.read_parquet(cfg.paths.interim / "hourly.parquet")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    cfg = load_config(ap.parse_args().config)
    df = refresh(cfg)
    print(f"{df.height} hourly rows from {df['ts_utc'].min()!s} to {df['ts_utc'].max()!s}")
    for name, nulls in df.null_count().row(0, named=True).items():
        print(f"  {name}: {df.height - nulls} values")


if __name__ == "__main__":
    main()
