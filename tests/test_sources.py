import json
from datetime import UTC, date, datetime, timedelta

import httpx
import polars as pl
import pytest

from gridwise import dataset
from gridwise.sources import energycharts, openmeteo
from gridwise.sources.http import FetchError, get_json
from gridwise.sources.smard import Smard

WEEK_MS = 7 * 24 * 3600 * 1000
T0 = int(datetime(2025, 1, 5, 23, tzinfo=UTC).timestamp() * 1000)


def _smard_client(calls: list[str], n_chunks: int = 5) -> httpx.Client:
    starts = [T0 + i * WEEK_MS for i in range(n_chunks)]

    def handler(req: httpx.Request) -> httpx.Response:
        calls.append(req.url.path)
        if req.url.path.endswith("index_hour.json"):
            return httpx.Response(200, json={"timestamps": starts})
        t = int(req.url.path.rsplit("_", 1)[1].removesuffix(".json"))
        series = [[t + h * 3600_000, float(h) if t != starts[-1] or h < 100 else None] for h in range(168)]
        return httpx.Response(200, json={"series": series})

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_smard_series_and_cache(tmp_path):
    calls: list[str] = []
    s = Smard(tmp_path, client=_smard_client(calls), refresh_last=3)
    start = datetime.fromtimestamp((T0 + WEEK_MS + 5 * 3600_000) / 1000, UTC)  # inside the second chunk
    df = s.series("price", 4169, "DE-LU", "hour", start)
    assert df.columns == ["ts_utc", "price"]
    assert df["ts_utc"].min() == start and df["ts_utc"].is_sorted()
    assert df.height == 4 * 168 - 5 and df["price"].null_count() == 68
    assert len([c for c in calls if "index" not in c]) == 4  # the first chunk is before `start`

    calls.clear()
    s.series("price", 4169, "DE-LU", "hour", start)
    assert len([c for c in calls if "index" not in c]) == 3  # only the newest three are downloaded again


def test_get_json_retries_then_fails(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda _: None)
    attempts = {"n": 0}

    def flaky(req: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        return httpx.Response(503 if attempts["n"] < 3 else 200, json={"ok": True})

    assert get_json(httpx.Client(transport=httpx.MockTransport(flaky)), "https://x/y") == {"ok": True}
    assert attempts["n"] == 3
    dead = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(404, text="nope")))
    with pytest.raises(FetchError):
        get_json(dead, "https://x/y")


@pytest.mark.parametrize(
    "d_minus_1", [date(2026, 1, 14), date(2026, 3, 28), date(2026, 7, 1), date(2026, 10, 24)]
)
def test_weather_for_the_whole_delivery_day_is_final_at_decision_time(d_minus_1):
    """At 07:30 UTC on D-1 the 48-hour-lead forecast must already be final for every hour of day D."""
    from gridwise.frames import day_hours

    now = datetime.combine(d_minus_1, datetime.min.time(), tzinfo=UTC).replace(hour=7, minute=30)
    last_hour = day_hours(d_minus_1 + timedelta(days=1), "Europe/Berlin")[-1][0]
    assert openmeteo.final_until(now, lead_days=2) >= last_hour
    # with a one-day lead the last hours would not be final: that lead would leak
    assert openmeteo.final_until(now, lead_days=1) < last_hour


def test_openmeteo_fetch_parses_and_caches(tmp_path, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda _: None)
    calls = []

    def handler(req: httpx.Request) -> httpx.Response:
        calls.append(dict(req.url.params))
        start, end = (
            date.fromisoformat(req.url.params["start_date"]),
            date.fromisoformat(req.url.params["end_date"]),
        )
        times = [
            f"{start + timedelta(days=d)}T{h:02d}:00"
            for d in range((end - start).days + 1)
            for h in range(24)
        ]
        one = {"hourly": {"time": times, "wind_speed_100m_previous_day2": [5.0] * len(times),
                          "temperature_2m_previous_day2": [None] + [1.0] * (len(times) - 1)}}  # fmt: skip
        return httpx.Response(200, json=[one, one])

    client = httpx.Client(transport=httpx.MockTransport(handler))
    sites = [("wind_onshore", "a", 54.0, 9.0), ("temperature", "b", 52.5, 13.4)]
    kw = dict(sites=sites, variables=["wind_speed_100m", "temperature_2m"], model="icon_eu", lead_days=2,
              start=date(2025, 1, 30), end=date(2025, 2, 2), cache_dir=tmp_path, client=client)  # fmt: skip
    df = openmeteo.fetch(**kw, today=date(2025, 6, 1))
    assert df.height == 2 * 4 * 24 and set(df["site"]) == {"a", "b"}
    assert df["ts_utc"].dtype == pl.Datetime("ms", "UTC")
    assert df["temperature_2m"].null_count() == 4  # nulls survive as nulls
    assert len(calls) == 2 and calls[0]["models"] == "icon_eu"  # one request per month
    openmeteo.fetch(**kw, today=date(2025, 6, 1))
    assert len(calls) == 2  # both months are old: served from the cache
    openmeteo.fetch(**kw, today=date(2025, 2, 3))
    assert len(calls) == 4  # recent months are downloaded again


def test_energycharts_quarter_hours_become_hourly_means():
    t0 = int(datetime(2026, 11, 20, 0, tzinfo=UTC).timestamp())
    secs = [t0 + i * 900 for i in range(7)]  # 1 full hour + 3 quarters
    doc = {"unix_seconds": secs, "price": [10.0, 20.0, 30.0, 40.0, 1.0, 2.0, 3.0]}
    client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, json=doc)))
    df = energycharts.hourly_prices(date(2026, 11, 20), date(2026, 11, 20), client=client)
    assert df.height == 1 and df["price"][0] == 25.0  # the incomplete hour is dropped


def test_power_curve_and_weather_aggregation():
    df = pl.DataFrame({"v": [0.0, 3.0, 7.5, 12.0, 20.0, 26.0]}).select(
        dataset.power_curve(pl.col("v")).alias("p")
    )
    p = df["p"].to_list()
    assert p[0] == 0 and p[1] == 0 and 0 < p[2] < 1 and p[3] == 1 and p[4] == 1 and p[5] == 0
    ts = datetime(2025, 1, 1, tzinfo=UTC)
    sites = pl.DataFrame(
        {
            "ts_utc": [ts] * 4,
            "group": ["wind_onshore", "wind_onshore", "solar", "temperature"],
            "site": ["a", "b", "c", "d"],
            "wind_speed_100m": [12.0, 3.0, 99.0, 99.0],
            "shortwave_radiation": [0.0, 0.0, 400.0, 0.0],
            "temperature_2m": [0.0, 0.0, 0.0, 7.0],
        }
    )
    row = dataset.aggregate_weather(sites).row(0, named=True)
    assert row["wx_wind_on"] == 7.5 and row["wx_wind_on_pow"] == 0.5
    assert row["wx_solar"] == 400.0 and row["wx_temp"] == 7.0
    assert row["wx_wind_off"] is None  # no offshore site in this toy input


def test_price_gaps_are_filled_from_the_second_source(monkeypatch):
    now = datetime(2026, 11, 21, 8, tzinfo=UTC)
    ts = [now - timedelta(hours=h) for h in range(5, 0, -1)]
    market = pl.DataFrame({"ts_utc": ts, "price": [1.0, None, None, 4.0, None]}).with_columns(
        pl.col("ts_utc").cast(pl.Datetime("ms", "UTC"))
    )
    fill = market.select("ts_utc").with_columns(pl.lit(9.0).alias("price")).head(4)
    monkeypatch.setattr(energycharts, "hourly_prices", lambda *a, **k: fill)
    out = dataset.fill_price_gaps(market, now)
    assert out["price"].to_list() == [
        1.0,
        9.0,
        9.0,
        4.0,
        None,
    ]  # SMARD values win; unknown hours stay unknown

    def boom(*a, **k):
        raise FetchError("down")

    monkeypatch.setattr(energycharts, "hourly_prices", boom)
    assert dataset.fill_price_gaps(market, now).equals(market)  # the fallback failing is not fatal
    full = market.with_columns(pl.col("price").fill_null(0.0))
    assert dataset.fill_price_gaps(full, now).equals(full)
    assert json.dumps({"ok": 1})  # keep json import used
