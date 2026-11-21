# Data

No market or weather data is stored in this repository. The code downloads it from the sources below
and caches it under `data/` (git-ignored).

```
uv run python -m gridwise.dataset      # first run: about 3,800 small requests, 5 to 10 minutes
```

## Sources and terms

| Source | What is used | Terms (read on 2026-11-21) |
|---|---|---|
| [SMARD.de](https://www.smard.de/en) (Bundesnetzagentur) | Hourly day-ahead price for the DE-LU zone since October 2018; load, wind and solar generation; the grid operators' day-ahead forecasts | CC BY 4.0, credit "Bundesnetzagentur \| SMARD.de". The SMARD data-use page adds that the licensing of wholesale prices is under review. |
| [Energy-Charts](https://api.energy-charts.info/) (Fraunhofer ISE) | Day-ahead prices, only to fill hours missing from SMARD | The API states the price data as "CC BY 4.0 from Bundesnetzagentur \| SMARD.de". |
| [Open-Meteo](https://open-meteo.com/) Previous Runs API | ICON-EU forecasts of 100 m wind speed, solar radiation and temperature at 24 sites, as issued 48 hours before each hour | Data CC BY 4.0. The free API is for non-commercial use only, with daily and hourly call limits. |

The live record under `docs/live/` contains this project's own forecasts, plus the real price of each
scored hour so that the scores can be checked. Those prices carry the SMARD credit above.

Using this project commercially would need a paid Open-Meteo plan and a fresh read of SMARD's terms for
wholesale prices.

## What each input is and when it is known

The auction for delivery day D closes at 12:00 (Europe/Berlin) on D-1. The daily job runs at 06:30 UTC on D-1.

| Input | Known at 06:30 UTC on D-1? | Used |
|---|---|---|
| Prices of D-1 and earlier | Yes. D-1 prices were published around 13:00 on D-2. | Yes |
| Weather forecast for D issued 48 h ahead | Yes. The latest run needed started at 18:00 UTC on D-2. | Yes |
| Calendar (weekday, public holidays) | Yes | Yes |
| Grid operators' load forecast for D | Unclear: it was online at 15:15 on D-1 when checked, earlier times were not checked | No |
| Grid operators' wind and solar forecast for D | **No.** On 2026-11-21 at 15:15 local, three hours after the auction, it was still missing. | Only in the "leaky" comparison |
| Actual load and generation | Up to a few hours ago | No |

## Details worth knowing

- **Weather lead time.** `<variable>_previous_day2` at valid time t is the value from the model run that
  started at floor_6h(t - 48 h). This was checked hour by hour against Open-Meteo's Single Runs API for
  ICON-EU, both for a past day and for the next day. A one-day lead would be more accurate but its run
  for the evening hours of D starts after the auction, so the config refuses it.
- **Weather history starts on 2024-02-18.** Earlier forecasts at a fixed lead are not archived, which is
  why the main backtest starts in 2025 and the six-year backtest uses prices only.
- **Five days without weather forecasts** (2026-04-07 to 2026-04-11) are missing from the archive. Those
  days are skipped by every model in the weather backtests.
- **15-minute prices.** Since October 2025 the auction clears 15-minute products. SMARD's hourly series
  is the mean of the four quarters (checked against Energy-Charts: largest difference 0.005 EUR/MWh).
  A battery that trades the same volume in all four quarters of an hour is paid exactly that mean.
- **Daylight saving.** Delivery days have 23 or 25 hours twice a year. Features use a 24-slot grid
  (`gridwise/frames.py`); battery schedules and money always use the real hours.
- **SMARD gaps.** On 2026-11-21 SMARD's chart data had no prices for the two most recent days while
  Energy-Charts had them; the loader fills such gaps and prints how many hours it filled.
