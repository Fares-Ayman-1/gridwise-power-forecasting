# Results

Generated from MLflow on 2026-11-21 15:29 UTC by `python -m gridwise.report`.
Every number below is a logged metric of the MLflow run named in its section.

Units: prices and errors in EUR/MWh. **MAE** is the mean absolute error of the median forecast.
**CRPS** scores the whole forecast distribution (lower is better; for a point forecast it equals MAE).
"Range holds" is how often the real price fell inside the forecast range; it should match the nominal level.

## 1. Forecast accuracy (main backtest)

Run `f76ab0473be441dc98d5ff890402c811`. 686 delivery days, 2025-01-01 to 2026-11-22. Each day was forecast by
models fitted on earlier days only, from inputs that exist before the auction closes, then calibrated
online from earlier errors only.

| Model | MAE | vs naive | CRPS | 80% range holds | 90% range holds | 95% range holds | 90% range width |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Ensemble** (average of the three models below, recalibrated) | 15.46 | -52.5% | 11.53 | 79.7% | 90.1% | 95.2% | 72.7 |
| Quantile neural network (all 24 hours at once) | 16.96 | -47.9% | 12.49 | 80.3% | 90.2% | 95.2% | 79.3 |
| Quantile gradient boosting (LightGBM) | 16.00 | -50.9% | 11.80 | 80.0% | 89.9% | 94.9% | 76.1 |
| LASSO baseline in the LEAR style, refitted daily | 18.33 | -43.7% | 14.25 | 80.5% | 90.5% | 95.3% | 91.5 |
| Naive: repeat a similar day | 32.57 | +0.0% | 25.12 | 80.2% | 90.1% | 94.9% | 166.8 |

Against the LASSO baseline the ensemble lowers MAE by **15.7%** and CRPS by **19.1%**.
One-sided Diebold-Mariano test on daily losses, ensemble vs LASSO: p < 0.001 (MAE), p < 0.001 (CRPS).

## 2. What calibration does

Share of real prices inside the 90% range, before and after the online conformal step:

| Model | Before | After | CRPS before | CRPS after |
|---|---:|---:|---:|---:|
| **Ensemble** (average of the three models below, recalibrated) | 93.8% | 90.1% | 11.49 | 11.53 |
| Quantile neural network (all 24 hours at once) | 78.7% | 90.2% | 12.58 | 12.49 |
| Quantile gradient boosting (LightGBM) | 76.5% | 89.9% | 11.94 | 11.80 |

The naive and LASSO models give one number per hour; their ranges come entirely from the same
calibration step applied to their recent errors. The ensemble row starts from members that are already
calibrated: their average is a little too wide, and the last step narrows it back to 90%.

## 3. What the inputs are worth, and what a leaky backtest would have claimed

Same models and the same days, different inputs. (The last row has one day fewer: the operators'
forecast is missing for it.)

| Inputs | Available before the auction? | Days | Ensemble MAE | Ensemble CRPS | Run |
|---|---|---:|---:|---:|---|
| Prices and calendar only | yes | 686 | 19.83 | 14.84 | `f1f3603e82a247a8851d7520fcadb52d` |
| Prices, calendar, weather forecasts made 48 h ahead (the real system) | yes | 686 | 15.46 | 11.53 | `f76ab0473be441dc98d5ff890402c811` |
| Prices, calendar, grid operators' load, wind and solar forecasts | **no** | 685 | 13.88 | 10.28 | `91d83e3717964dcda7e280463fc629ef` |

Weather forecasts lower the ensemble's MAE by **22.0%** compared with prices alone.
A backtest that used the grid operators' wind and solar forecasts would have reported an error 10% lower (13.88 instead of 15.46 EUR/MWh). Those forecasts are published after the auction closes, so they are not used.
Checked on 2026-11-21: at 15:15 local time, three hours after the auction for the next day had closed,
SMARD still had no wind or solar forecast for that day.

## 5. What the forecast is worth to a battery (main backtest)

Run `f76ab0473be441dc98d5ff890402c811`. 686 days. Battery: 1 MW / 2 MWh, 94% charge and 94% discharge efficiency, at most 2 full cycles a day, 10 EUR per MWh delivered as wear cost.
Each plan is fixed before prices are known and paid at the real hourly prices, net of the wear cost.

| Plan | EUR per day | Share of perfect | Days with a loss | Mean of worst 5% of days | Gain over no-model plan, EUR/day (95% CI) |
|---|---:|---:|---:|---:|---:|
| Perfect foresight (upper bound, not achievable) | 239.8 | 100.0% | 0.0% | 21.0 |  |
| **Plan on the ensemble's forecast mean** | 227.5 | 94.9% | 0.6% | 8.2 | +14.6 (+11.9 to +17.3) |
| Risk-averse plan on ensemble scenarios (CVaR) | 220.4 | 91.9% | 0.3% | -1.0 | +7.5 (+3.4 to +11.5) |
| Plan on the ensemble's median | 227.5 | 94.9% | 0.6% | 7.0 | +14.6 (+11.8 to +17.3) |
| Plan on the LASSO baseline | 223.1 | 93.0% | 1.5% | 3.5 | +10.1 (+7.3 to +13.1) |
| Plan on the naive forecast | 211.7 | 88.3% | 3.4% | -12.0 | -1.3 (-4.2 to +2.1) |
| No model: average price of each hour over the last 28 days | 212.9 | 88.8% | 2.5% | -8.2 |  |

Profit is linear in price, so planning on the forecast mean is already the best plan for expected profit;
sampling scenarios cannot beat it. The risk-averse plan is the one place the full distribution changes the
decision: it gives up 7.1 EUR a day on average and its worst 5% of days average -1.0 EUR against 8.2 EUR for the mean plan. Risk aversion bought nothing here:
a battery that only trades day-ahead rarely loses money, so there is little downside to insure against.

## 6. Setup recorded with the main run

- Features: 39 (weather set); weather: icon_eu, 2-day lead, 24 sites
- Quantile levels: [0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.975]
- Calibration: {'window_days': 90, 'mode': 'adaptive', 'gamma': 0.01}
- Models: LASSO {'train_days': 728, 'refit_every': 1}; boosting {'train_days': 1095, 'refit_every': 28, 'num_boost_round': 200, 'params': {'learning_rate': 0.1}}; network {'train_days': 1095, 'refit_every': 28, 'seeds': 5}
- Risk settings: {'alpha': 0.1, 'weight': 0.5, 'scenarios': 200, 'copula_days': 180}
- Code version: `unknown`

No hyperparameter search was run. Model settings were fixed before the backtests; the only earlier look at
the data was a 30-day smoke test at the end of the window.
