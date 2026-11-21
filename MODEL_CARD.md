# Model card: GridWise day-ahead price forecaster

## What it is

An ensemble that forecasts the 24 hourly day-ahead electricity prices of the German-Luxembourg bidding
zone for the next delivery day, as 13 quantiles per hour (2.5% to 97.5%).

| Member | Type | Refit |
|---|---|---|
| LASSO baseline | One linear model per hour on the prices of D-1, D-2, D-3 and D-7, weekday and weather, in the style of LEAR (Lago et al., 2021); gives a point forecast | Every day, on the last 728 days |
| Quantile gradient boosting | LightGBM, one model per quantile level, all hours pooled, 39 features | Every 28 days, on up to 1,095 days |
| Quantile neural network | PyTorch MLP (256, 128) that outputs all 24 hours and 13 quantiles at once; non-crossing head; 5 seeds averaged | Every 28 days, on up to 1,095 days |

Each member is calibrated online with adaptive conformal inference (90-day window). The ensemble is the
level-by-level average of the calibrated members, calibrated once more.

- Version 0.1.0, November 2026. Author: Fares Ayman. Licence: MIT (code).
- No hyperparameter search was run; settings are in `configs/default.yaml`.

## Intended use

- Studying probabilistic price forecasting, calibration under regime change, and forecast-to-decision
  evaluation.
- A reference for a leak-safe live forecasting pipeline.

Not intended for real trading without the checks a trading desk would add (intraday and balancing
markets, bidding rules, position limits, data contracts). Nothing here is financial advice.

## Inputs and outputs

Inputs, all known by 06:30 UTC on the day before delivery: day-ahead prices up to that day; ICON-EU
forecasts of 100 m wind speed, solar radiation and temperature at 24 German sites as issued 48 hours
ahead; the calendar with German public holidays. See [DATA.md](DATA.md).

Outputs: 13 quantiles per hour in EUR/MWh, the implied mean, and a battery schedule.

## Training data

- Prices: SMARD.de, hourly, from October 2018.
- Weather: Open-Meteo Previous Runs API, from 18 February 2024 (the start of its fixed-lead archive).
  Models that use weather therefore train on at most the days since then.

## Evaluation

Numbers are in [RESULTS.md](RESULTS.md), generated from MLflow runs. In short, on the main rolling-origin
backtest the ensemble has the lowest error of all models and its 80%, 90% and 95% ranges hold at
close to their nominal rates. The Diebold-Mariano test against the LASSO baseline and the by-year table
for 2020 to 2026 are in the same file.

How to read the metrics:

- **MAE**: mean absolute error of the median forecast, EUR/MWh.
- **CRPS**: a score for the whole distribution, approximated on the 13 quantile levels; lower is better.
- **Coverage**: share of real prices inside a forecast range. It should equal the range's nominal level;
  higher is not better, it means ranges are wider than they need to be.

## Limits and known failure modes

- **Regime change.** The models see prices relative to the previous week, which lets them follow a new
  price level within days, but the first days after an abrupt shift will be poor. Calibration reacts
  within days to weeks, not hours.
- **Missing drivers.** Gas and carbon prices, plant outages and cross-border flows are not inputs.
- **Weather lead.** Weather forecasts are 48 to 53 hours old by design, to guarantee availability. A
  fresher forecast would be more accurate but could not be backtested honestly.
- **Short weather history.** Under three years, so rare weather (long windless cold spells) is thinly
  represented.
- **Extremes.** Quantiles stop at 2.5% and 97.5%. Price spikes and deeply negative prices beyond them
  are extrapolated linearly and should not be trusted.
- **Hourly resolution.** The market clears 15-minute products since October 2025; the model forecasts
  the hourly mean.
- **Daylight-saving days.** The skipped hour is not forecast; the repeated hour gets one forecast for
  both real hours.
- **Five days in April 2026** have no archived weather forecast and are not forecast by weather models.
- **Source outages.** If SMARD and Energy-Charts both lack yesterday's prices, or the weather archive
  lacks tomorrow, the daily job stops instead of publishing a forecast from incomplete inputs.

## Battery study assumptions

1 MW / 2 MWh, 94% charge and 94% discharge efficiency, at most 2 full cycles a day, 10 EUR per MWh
delivered as wear cost, empty at the start and end of each day, price taker, day-ahead market only.
These are stated assumptions, not measurements of a real battery.

## Monitoring

`docs/live/monitoring.json` is rewritten every day with four checks: forecasts published on time,
weather inputs equal to the archive, recent coverage and error against the backtest, and a population
stability index for the price level, the price spread and the wind forecast.

## People and data

No personal data is used. Sources and licences: [DATA.md](DATA.md).
