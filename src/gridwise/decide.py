"""Battery study: what each forecast is worth in money.

    uv run python -m gridwise.decide --name main

For every test day each policy fixes a schedule before prices are known; the schedule is then paid at
the real hourly prices. Policies differ only in the price information they plan on:

- perfect        the real prices (upper bound, not achievable)
- profile        the average price of each hour over the previous 28 days (no forecasting model)
- naive, lear    the point forecast of that baseline
- <model>_median the median forecast
- <model>_mean   the mean implied by the quantile forecast (the risk-neutral optimum, see battery.py)
- <model>_cvar   joint scenarios from the quantile forecast, objective mixing mean and CVaR
"""

from __future__ import annotations

import argparse
import json
from typing import Any

import numpy as np

from gridwise import metrics as M
from gridwise.backtest import Forecasts
from gridwise.battery import BatterySpec, plan, plan_cvar, profit
from gridwise.config import Config, load_config
from gridwise.dataset import load_hourly
from gridwise.frames import SLOTS, day_hours, hourly_lookup, to_daily
from gridwise.scenarios import copula_correlation, forecast_mean, pit, sample

PROFILE_DAYS = 28


def _load(cfg: Config, name: str, model: str) -> Forecasts:
    return Forecasts.load(model, cfg.paths.reports / "backtest" / name / f"{model}.npz")


def run(
    cfg: Config, name: str, model: str = "ensemble", baselines: tuple[str, ...] = ("naive", "lear")
) -> dict[str, Any]:
    taus = np.asarray(cfg.quantiles)
    i50 = cfg.quantiles.index(0.5)
    spec: BatterySpec = cfg.battery
    hourly = load_hourly(cfg)
    actual = hourly_lookup(hourly, "price")
    grid = to_daily(hourly, ["price"], cfg.market.timezone)
    main = _load(cfg, name, model)
    base = {b: _load(cfg, name, b) for b in baselines}
    u = pit(main.q, taus, main.y)  # (n, 24) probability level of each outcome under its own forecast
    rng = np.random.default_rng(7)

    names = ["perfect", "profile", *baselines, f"{model}_median", f"{model}_mean", f"{model}_cvar"]
    rows: list[int] = []
    daily: dict[str, list[float]] = {k: [] for k in names}
    skipped = 0
    for i in np.where(main.is_test)[0]:
        day = main.days[i].astype(object)
        hours = day_hours(day, cfg.market.timezone)
        if any(t not in actual for t, _ in hours) or i < cfg.risk.copula_days // 2:
            skipped += 1
            continue
        slots = np.array([s for _, s in hours])
        real_prices = np.array([actual[t] for t, _ in hours])
        g = grid.index(day)
        profile = np.nanmean(grid["price"][g - PROFILE_DAYS : g], axis=0)[slots]
        q_day = main.q[i][slots]  # (H, Q)
        corr24 = copula_correlation(u[max(0, i - cfg.risk.copula_days) : i])
        corr = corr24[np.ix_(slots, slots)]
        if len(slots) != SLOTS:  # 23/25-hour days: repeated slots must not make the matrix singular
            corr = 0.98 * corr + 0.02 * np.eye(len(slots))
        scen = sample(q_day, taus, corr, cfg.risk.scenarios, rng)
        plans = {
            "perfect": plan(real_prices, spec),
            "profile": plan(profile, spec),
            **{b: plan(base[b].q[i][slots][:, i50], spec) for b in baselines},
            f"{model}_median": plan(q_day[:, i50], spec),
            f"{model}_mean": plan(forecast_mean(q_day, taus), spec),
            f"{model}_cvar": plan_cvar(scen, spec, cfg.risk.alpha, cfg.risk.weight),
        }
        for k, schedule in plans.items():
            daily[k].append(profit(schedule, real_prices, spec))
        rows.append(int(i))

    out_dir = cfg.paths.reports / "backtest" / name
    arr = {k: np.array(v) for k, v in daily.items()}
    np.savez_compressed(out_dir / "decisions.npz", days=main.days[rows].astype(str), **arr)  # type: ignore[arg-type]

    perfect = arr["perfect"].mean()
    summary: dict[str, Any] = {
        "backtest": name,
        "model": model,
        "days": len(rows),
        "skipped_days": skipped,
        "battery": spec.model_dump(),
        "risk": cfg.risk.model_dump(),
        "policies": {},
    }
    for k, v in arr.items():
        worst = np.sort(v)[: max(1, len(v) // 20)]
        s: dict[str, Any] = {
            "eur_per_day": float(v.mean()),
            "share_of_perfect": float(v.mean() / perfect),
            "loss_day_share": float(np.mean(v < 0)),
            "worst_5pct_mean": float(worst.mean()),
            "std": float(v.std(ddof=1)),
        }
        for ref in ("profile", "lear", f"{model}_mean"):
            if ref in arr and ref != k and k != "perfect":
                lo, hi = M.paired_bootstrap(v - arr[ref])
                s[f"gain_vs_{ref}"] = float((v - arr[ref]).mean())
                s[f"gain_vs_{ref}_ci95"] = [lo, hi]
        summary["policies"][k] = s
    (out_dir / "decisions.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def print_summary(s: dict[str, Any]) -> None:
    b = s["battery"]
    print(
        f"\n{s['backtest']}: {s['days']} days, {b['power_mw']} MW / {b['capacity_mwh']} MWh battery, "
        f"model {s['model']}"
    )
    print(
        f"{'policy':18s} {'EUR/day':>8s} {'% perfect':>9s} {'loss days':>9s} {'worst 5%':>9s}  "
        f"gain vs profile [95% CI]"
    )
    for k, p in s["policies"].items():
        gain = ""
        if "gain_vs_profile" in p:
            lo, hi = p["gain_vs_profile_ci95"]
            gain = f"{p['gain_vs_profile']:+7.1f} [{lo:+.1f}, {hi:+.1f}]"
        print(f"{k:18s} {p['eur_per_day']:8.1f} {p['share_of_perfect']:9.1%} {p['loss_day_share']:9.1%} "
              f"{p['worst_5pct_mean']:9.1f}  {gain}")  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--name", default="main")
    ap.add_argument("--model", default="ensemble")
    a = ap.parse_args()
    print_summary(run(load_config(a.config), a.name, a.model))


if __name__ == "__main__":
    main()
