"""SMARD.de (Bundesnetzagentur) market data: prices, load, generation and operator forecasts.

Data: Bundesnetzagentur | SMARD.de, CC BY 4.0. The chart API serves each series as weekly chunks:

    /app/chart_data/{filter}/{region}/index_{resolution}.json                -> chunk start times
    /app/chart_data/{filter}/{region}/{filter}_{region}_{resolution}_{t}.json -> [[ms, value], ...]

Chunks are cached on disk. Old chunks never change, so only the newest few are downloaded again.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import httpx
import polars as pl

from gridwise.sources.http import get_json, make_client

BASE = "https://www.smard.de/app/chart_data"


class Smard:
    def __init__(self, cache_dir: Path, client: httpx.Client | None = None, refresh_last: int = 3):
        self.cache_dir = cache_dir
        self.client = client or make_client()
        self.refresh_last = refresh_last

    def index(self, fid: int, region: str, resolution: str) -> list[int]:
        doc = get_json(self.client, f"{BASE}/{fid}/{region}/index_{resolution}.json")
        return sorted(int(t) for t in doc["timestamps"])

    def chunk(self, fid: int, region: str, resolution: str, start_ms: int, refresh: bool) -> list[list]:
        path = self.cache_dir / f"{fid}_{region}_{resolution}" / f"{start_ms}.json"
        if path.exists() and not refresh:
            return json.loads(path.read_text(encoding="utf-8"))
        doc = get_json(self.client, f"{BASE}/{fid}/{region}/{fid}_{region}_{resolution}_{start_ms}.json")
        series = doc["series"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(series), encoding="utf-8")
        return series

    def series(
        self, name: str, fid: int, region: str, resolution: str, start: datetime, workers: int = 8
    ) -> pl.DataFrame:
        """One series from `start` to the newest value, as columns ts_utc and `name`."""
        starts = self.index(fid, region, resolution)
        start_ms = int(start.timestamp() * 1000)
        # keep the chunk that contains `start`: the last chunk start at or before it
        first = max((i for i, t in enumerate(starts) if t <= start_ms), default=0)
        wanted = starts[first:]
        fresh = set(wanted[-self.refresh_last :])

        def load(t: int) -> list[list]:
            return self.chunk(fid, region, resolution, t, refresh=t in fresh)

        with ThreadPoolExecutor(max_workers=workers) as pool:
            chunks = list(pool.map(load, wanted))
        rows = [(int(ms), None if v is None else float(v)) for c in chunks for ms, v in c]
        df = pl.DataFrame(rows, schema={"ms": pl.Int64, name: pl.Float64}, orient="row")
        return (
            df.with_columns(
                pl.from_epoch("ms", time_unit="ms").cast(pl.Datetime("ms", "UTC")).alias("ts_utc")
            )
            .filter(pl.col("ts_utc") >= start.astimezone(UTC))
            .unique("ts_utc", keep="last")
            .sort("ts_utc")
            .select("ts_utc", name)
        )


def download(
    series: dict[str, tuple[int, str]], resolution: str, start: datetime, cache_dir: Path
) -> pl.DataFrame:
    """All configured series joined on ts_utc (outer join, sorted)."""
    client = Smard(cache_dir)
    out: pl.DataFrame | None = None
    for name, (fid, region) in series.items():
        df = client.series(name, fid, region, resolution, start)
        out = df if out is None else out.join(df, on="ts_utc", how="full", coalesce=True)
    assert out is not None, "no series configured"
    return out.sort("ts_utc")
