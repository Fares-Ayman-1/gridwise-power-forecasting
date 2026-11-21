"""Write RESULTS.md, README.md and docs/live/backtest.json from MLflow, so every published number traces to a run.

    uv run python -m gridwise.report

The runs are the ones recorded in reports/run_ids.json by `gridwise.track`.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from typing import Any

import mlflow

from gridwise.config import PROJECT_ROOT, Config, load_config, setup_mlflow

MODELS = ["ensemble", "qnn", "qgbm", "lear", "naive"]
MODEL_LABEL = {
    "ensemble": "**Ensemble** (average of the three models below, recalibrated)",
    "qnn": "Quantile neural network (all 24 hours at once)",
    "qgbm": "Quantile gradient boosting (LightGBM)",
    "lear": "LASSO baseline in the LEAR style, refitted daily",
    "naive": "Naive: repeat a similar day",
}
POLICY_LABEL = {
    "perfect": "Perfect foresight (upper bound, not achievable)",
    "ensemble_mean": "**Plan on the ensemble's forecast mean**",
    "ensemble_cvar": "Risk-averse plan on ensemble scenarios (CVaR)",
    "ensemble_median": "Plan on the ensemble's median",
    "lear": "Plan on the LASSO baseline",
    "naive": "Plan on the naive forecast",
    "profile": "No model: average price of each hour over the last 28 days",
}


class Run:
    def __init__(self, name: str, run_id: str):
        self.name, self.id = name, run_id
        data = mlflow.get_run(run_id).data
        self.m: dict[str, float] = data.metrics
        self.p: dict[str, str] = data.params

    def fc(self, model: str, key: str) -> float:
        return self.m[f"fc.{model}.{key}"]

    def bat(self, policy: str, key: str) -> float:
        return self.m[f"bat.{policy}.{key}"]

    def models(self) -> list[str]:
        return [m for m in MODELS if f"fc.{m}.mae" in self.m]

    def policies(self) -> list[str]:
        found = {k.split(".")[1] for k in self.m if k.startswith("bat.")}
        return [p for p in POLICY_LABEL if p in found]

    def years(self, model: str) -> list[str]:
        prefix = f"fc.{model}.by_year."
        return sorted({k[len(prefix) :].split(".")[0] for k in self.m if k.startswith(prefix)})


def load_runs(cfg: Config) -> dict[str, Run]:
    setup_mlflow(cfg)
    ids = json.loads((cfg.paths.reports / "run_ids.json").read_text(encoding="utf-8"))
    return {name: Run(name, rid) for name, rid in ids.items()}


def _change(ratio: float) -> str:
    """Relative change as a signed percentage, without a "-0.0%"."""
    return "0.0%" if abs(ratio) < 0.0005 else f"{ratio:+.1%}"


def _p(v: float) -> str:
    return "p < 0.001" if v < 0.001 else f"p = {v:.3f}"


def _battery(params: str) -> str:
    b = json.loads(params.replace("'", '"'))
    return (
        f"{b['power_mw']:g} MW / {b['capacity_mwh']:g} MWh, {b['eff_charge']:.0%} charge and "
        f"{b['eff_discharge']:.0%} discharge efficiency, at most {b['max_cycles_per_day']:g} full cycles a day, "
        f"{b['degradation_eur_per_mwh']:g} EUR per MWh delivered as wear cost"
    )


def forecast_table(r: Run) -> list[str]:
    L = [
        "| Model | MAE | vs naive | CRPS | 80% range holds | 90% range holds | 95% range holds | 90% range width |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for m in r.models():
        L.append(
            f"| {MODEL_LABEL[m]} | {r.fc(m, 'mae'):.2f} | {r.fc(m, 'rmae') - 1:+.1%} | {r.fc(m, 'crps'):.2f} | "
            f"{r.fc(m, 'coverage_80'):.1%} | {r.fc(m, 'coverage_90'):.1%} | {r.fc(m, 'coverage_95'):.1%} | "
            f"{r.fc(m, 'width_90'):.1f} |"
        )
    return L


def battery_table(r: Run) -> list[str]:
    L = [
        "| Plan | EUR per day | Share of perfect | Days with a loss | Mean of worst 5% of days | Gain over no-model plan, EUR/day (95% CI) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for p in r.policies():
        gain = "" if p in ("perfect", "profile") else (
            f"{r.bat(p, 'gain_vs_profile'):+.1f} ({r.bat(p, 'gain_vs_profile_ci95_lo'):+.1f} to "
            f"{r.bat(p, 'gain_vs_profile_ci95_hi'):+.1f})"
        )  # fmt: skip
        L.append(
            f"| {POLICY_LABEL[p]} | {r.bat(p, 'eur_per_day'):.1f} | {r.bat(p, 'share_of_perfect'):.1%} | "
            f"{r.bat(p, 'loss_day_share'):.1%} | {r.bat(p, 'worst_5pct_mean'):.1f} | {gain} |"
        )
    return L


def build(cfg: Config) -> tuple[str, dict[str, Any]]:
    runs = load_runs(cfg)
    main = runs["main"]
    days = int(main.fc("ensemble", "days"))
    L = [
        "# Results",
        "",
        f"Generated from MLflow on {datetime.now(UTC):%Y-%m-%d %H:%M} UTC by `python -m gridwise.report`.",
        "Every number below is a logged metric of the MLflow run named in its section.",
        "",
        "Units: prices and errors in EUR/MWh. **MAE** is the mean absolute error of the median forecast.",
        "**CRPS** scores the whole forecast distribution (lower is better; for a point forecast it equals MAE).",
        '"Range holds" is how often the real price fell inside the forecast range; it should match the nominal level.',
        "",
        "## 1. Forecast accuracy (main backtest)",
        "",
        f"Run `{main.id}`. {days} delivery days, {main.p['test_start']} to {main.p['test_end']}. Each day was forecast by",
        "models fitted on earlier days only, from inputs that exist before the auction closes, then calibrated",
        "online from earlier errors only.",
        "",
        *forecast_table(main),
        "",
    ]
    best = "ensemble"
    if "lear" in main.models():
        gain_mae = 1 - main.fc(best, "mae") / main.fc("lear", "mae")
        gain_crps = 1 - main.fc(best, "crps") / main.fc("lear", "crps")
        L += [
            f"Against the LASSO baseline the ensemble lowers MAE by **{gain_mae:.1%}** and CRPS by **{gain_crps:.1%}**.",
            f"One-sided Diebold-Mariano test on daily losses, ensemble vs LASSO: {_p(main.fc(best, 'dm_p_mae_vs_lear'))} "
            f"(MAE), {_p(main.fc(best, 'dm_p_crps_vs_lear'))} (CRPS).",
            "",
        ]
    L += [
        "## 2. What calibration does",
        "",
        "Share of real prices inside the 90% range, before and after the online conformal step:",
        "",
        "| Model | Before | After | CRPS before | CRPS after |",
        "|---|---:|---:|---:|---:|",
    ]
    for m in [m for m in main.models() if m in ("qnn", "qgbm", "ensemble")]:
        L.append(
            f"| {MODEL_LABEL[m]} | {main.fc(m, 'coverage_90_uncalibrated'):.1%} | {main.fc(m, 'coverage_90'):.1%} | "
            f"{main.fc(m, 'crps_uncalibrated'):.2f} | {main.fc(m, 'crps'):.2f} |"
        )
    L += [
        "",
        "The naive and LASSO models give one number per hour; their ranges come entirely from the same",
        "calibration step applied to their recent errors. The ensemble row starts from members that are already",
        "calibrated: their average is a little too wide, and the last step narrows it back to 90%.",
        "",
    ]
    doc_leak = None
    if "main_price_only" in runs or "leaky" in runs:
        L += [
            "## 3. What the inputs are worth, and what a leaky backtest would have claimed",
            "",
            "Same models and the same days, different inputs. (The last row has one day fewer: the operators'",
            "forecast is missing for it.)",
            "",
            "| Inputs | Available before the auction? | Days | Ensemble MAE | Ensemble CRPS | Run |",
            "|---|---|---:|---:|---:|---|",
        ]
        rows = [
            ("main_price_only", "Prices and calendar only", "yes"),
            ("main", "Prices, calendar, weather forecasts made 48 h ahead (the real system)", "yes"),
            ("leaky", "Prices, calendar, grid operators' load, wind and solar forecasts", "**no**"),
        ]
        for name, label, ok in rows:
            if name in runs:
                r = runs[name]
                L.append(
                    f"| {label} | {ok} | {int(r.fc('ensemble', 'days'))} | {r.fc('ensemble', 'mae'):.2f} | "
                    f"{r.fc('ensemble', 'crps'):.2f} | `{r.id}` |"
                )
        L.append("")
        if "main_price_only" in runs:
            po = runs["main_price_only"]
            L.append(
                f"Weather forecasts lower the ensemble's MAE by **{1 - main.fc('ensemble', 'mae') / po.fc('ensemble', 'mae'):.1%}** "
                "compared with prices alone."
            )
        if "leaky" in runs:
            lk = runs["leaky"]
            over = 1 - lk.fc("ensemble", "mae") / main.fc("ensemble", "mae")
            doc_leak = (
                f"A backtest that used the grid operators' wind and solar forecasts would have reported an error "
                f"{abs(over):.0%} {'lower' if over > 0 else 'higher'} ({lk.fc('ensemble', 'mae'):.2f} instead of "
                f"{main.fc('ensemble', 'mae'):.2f} EUR/MWh). Those forecasts are published after the auction closes, "
                "so they are not used."
            )
            L += [
                doc_leak,
                "Checked on 2026-11-21: at 15:15 local time, three hours after the auction for the next day had closed,",
                "SMARD still had no wind or solar forecast for that day.",
            ]
        L.append("")
    if "long" in runs:
        lg = runs["long"]
        years = lg.years("ensemble")
        L += [
            "## 4. Six years including the energy crisis (prices and calendar only)",
            "",
            f"Run `{lg.id}`. {int(lg.fc('ensemble', 'days'))} days, {lg.p['test_start']} to {lg.p['test_end']}. Weather forecasts at a",
            "safe lead time only exist from 2024, so this run uses prices and the calendar alone. It shows whether",
            "the ranges stay honest when the market changes regime.",
            "",
            "| Year | Naive MAE | LASSO MAE | Ensemble MAE | Ensemble vs LASSO | Ensemble 90% range holds |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for y in years:
            g = lambda m, k, y=y: lg.m[f"fc.{m}.by_year.{y}.{k}"]  # noqa: E731
            L.append(
                f"| {y} | {g('naive', 'mae'):.2f} | {g('lear', 'mae'):.2f} | {g('ensemble', 'mae'):.2f} | "
                f"{_change(g('ensemble', 'mae') / g('lear', 'mae') - 1)} | {g('ensemble', 'coverage_90'):.1%} |"
            )
        L += [
            f"| **All** | {lg.fc('naive', 'mae'):.2f} | {lg.fc('lear', 'mae'):.2f} | {lg.fc('ensemble', 'mae'):.2f} | "
            f"{_change(lg.fc('ensemble', 'mae') / lg.fc('lear', 'mae') - 1)} | {lg.fc('ensemble', 'coverage_90'):.1%} |",
            "",
        ]
        if "bat.ensemble_mean.eur_per_day" in lg.m:
            L += [
                f"Battery over the same {lg.p['decision_days']} days: the plan on the ensemble forecast captured "
                f"{lg.bat('ensemble_mean', 'share_of_perfect'):.1%} of the perfect-foresight profit, the no-model plan "
                f"{lg.bat('profile', 'share_of_perfect'):.1%} (gain {lg.bat('ensemble_mean', 'gain_vs_profile'):+.1f} EUR a day, "
                f"95% CI {lg.bat('ensemble_mean', 'gain_vs_profile_ci95_lo'):+.1f} to "
                f"{lg.bat('ensemble_mean', 'gain_vs_profile_ci95_hi'):+.1f}). Here the risk-averse plan did what it is meant to: "
                f"its worst 5% of days averaged {lg.bat('ensemble_cvar', 'worst_5pct_mean'):.1f} EUR against "
                f"{lg.bat('ensemble_mean', 'worst_5pct_mean'):.1f} EUR, and it lost money on "
                f"{lg.bat('ensemble_cvar', 'loss_day_share'):.1%} of days against {lg.bat('ensemble_mean', 'loss_day_share'):.1%}, "
                f"for {lg.bat('ensemble_mean', 'eur_per_day') - lg.bat('ensemble_cvar', 'eur_per_day'):.1f} EUR a day less on average.",
                "",
            ]
    L += [
        "## 5. What the forecast is worth to a battery (main backtest)",
        "",
        f"Run `{main.id}`. {main.p['decision_days']} days. Battery: {_battery(main.p['battery'])}.",
        "Each plan is fixed before prices are known and paid at the real hourly prices, net of the wear cost.",
        "",
        *battery_table(main),
        "",
    ]
    if "ensemble_cvar" in main.policies():
        L += [
            "Profit is linear in price, so planning on the forecast mean is already the best plan for expected profit;",
            "sampling scenarios cannot beat it. The risk-averse plan is the one place the full distribution changes the",
            f"decision: it gives up {main.bat('ensemble_mean', 'eur_per_day') - main.bat('ensemble_cvar', 'eur_per_day'):.1f} EUR a day "
            f"on average and its worst 5% of days average {main.bat('ensemble_cvar', 'worst_5pct_mean'):.1f} EUR against "
            f"{main.bat('ensemble_mean', 'worst_5pct_mean'):.1f} EUR for the mean plan. In this run risk aversion bought",
            "nothing: with a good forecast a battery that only trades day-ahead rarely loses money, so there was little",
            "downside to insure against. In the six-year run of section 4, with a weaker forecast and a wilder market, it",
            "did protect the worst days, at a cost in average profit.",
            "",
        ]
    L += [
        "## 6. Setup recorded with the main run",
        "",
        f"- Features: {main.p['n_features']} ({main.p['feature_set']} set); weather: {main.p['weather']}",
        f"- Quantile levels: {main.p['quantiles']}",
        f"- Calibration: {main.p['calibration']}",
        f"- Models: LASSO {main.p.get('model.lear')}; boosting {main.p.get('model.qgbm')}; network {main.p.get('model.qnn')}",
        f"- Risk settings: {main.p.get('risk')}",
        f"- Code version: `{mlflow.get_run(main.id).data.tags.get('code_version')}`",
        "",
        "No hyperparameter search was run. Model settings were fixed before the backtests; the only earlier look at",
        "the data was a 30-day smoke test at the end of the window.",
        "",
    ]
    dashboard = {
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "run_id": main.id,
        "test_start": main.p["test_start"],
        "test_end": main.p["test_end"],
        "days": days,
        "models": {
            m: {k: main.fc(m, k) for k in ("mae", "rmae", "crps", "coverage_80", "coverage_90", "width_90")}
            for m in main.models()
        },
        "policy_days": int(main.p["decision_days"]),
        "battery": json.loads(main.p["battery"].replace("'", '"')),
        "policies": {
            p: {
                k: main.bat(p, k)
                for k in ("eur_per_day", "share_of_perfect", "loss_day_share", "worst_5pct_mean")
            }
            for p in main.policies()
        },
        "leak_note": doc_leak,
    }
    return "\n".join(L), dashboard


README_MODEL = {
    "ensemble": "**GridWise ensemble**",
    "qgbm": "Quantile gradient boosting",
    "qnn": "Quantile neural network",
    "lear": "LASSO baseline (LEAR style)",
    "naive": "Naive (repeat a similar day)",
}
README_POLICY = {
    "perfect": "Perfect foresight (upper bound)",
    "ensemble_mean": "**Plan on the GridWise forecast**",
    "lear": "Plan on the LASSO baseline",
    "ensemble_cvar": "Risk-averse plan (CVaR)",
    "profile": "No model: 28-day average of each hour",
    "naive": "Plan on the naive forecast",
}


def render_readme(cfg: Config, template: str) -> str:
    """Fill the README template with numbers read from the recorded MLflow runs."""
    runs = load_runs(cfg)
    main = runs["main"]
    model_rows = "\n".join(
        f"| {README_MODEL[m]} | {main.fc(m, 'mae'):.2f} | {main.fc(m, 'rmae') - 1:+.0%} | {main.fc(m, 'crps'):.2f} | "
        f"{main.fc(m, 'coverage_90'):.1%} |"
        for m in main.models()
    )
    policies = sorted(
        (p for p in README_POLICY if p in main.policies()), key=lambda p: -main.bat(p, "share_of_perfect")
    )
    policy_rows = "\n".join(
        f"| {README_POLICY[p]} | {main.bat(p, 'eur_per_day'):.0f} | {main.bat(p, 'share_of_perfect'):.1%} |"
        for p in policies
    )
    profile, ours = main.bat("profile", "share_of_perfect"), main.bat("ensemble_mean", "share_of_perfect")
    leak = "the leaky comparison has not been run yet."
    if "leaky" in runs:
        lk = runs["leaky"].fc("ensemble", "mae")
        leak = (
            f"with the operators' forecasts the same models report an error of {lk:.2f} EUR/MWh "
            f"instead of {main.fc('ensemble', 'mae'):.2f}, {abs(1 - lk / main.fc('ensemble', 'mae')):.0%} "
            f"{'better' if lk < main.fc('ensemble', 'mae') else 'worse'} than the honest number."
        )
    values = {
        "__DAYS__": f"{int(main.fc('ensemble', 'days'))}",
        "__START__": main.p["test_start"],
        "__END__": main.p["test_end"],
        "__MODEL_ROWS__": model_rows,
        "__POLICY_ROWS__": policy_rows,
        "__RAW90__": f"{main.fc('qgbm', 'coverage_90_uncalibrated'):.1%}",
        "__CAL90__": f"{main.fc('qgbm', 'coverage_90'):.1%}",
        "__LEAK_SENTENCE__": leak,
        "__CVAR_EUR__": f"{main.bat('ensemble_cvar', 'eur_per_day'):.0f}",
        "__MEAN_EUR__": f"{main.bat('ensemble_mean', 'eur_per_day'):.0f}",
        "__CVAR_WORST__": f"{main.bat('ensemble_cvar', 'worst_5pct_mean'):.0f}",
        "__MEAN_WORST__": f"{main.bat('ensemble_mean', 'worst_5pct_mean'):.0f}",
        "__PROFILE_SHARE__": f"{profile:.1%}",
        "__GAP_CLOSED__": f"{(ours - profile) / (1 - profile):.0%}",
    }
    if "long" in runs:
        lg = runs["long"]
        cov = [lg.m[f"fc.ensemble.by_year.{y}.coverage_90"] for y in lg.years("ensemble")]
        values |= {
            "__LONG_YEARS__": f"{lg.years('ensemble')[0]} to {lg.years('ensemble')[-1]}",
            "__LONG_COV_MIN__": f"{min(cov):.1%}",
            "__LONG_COV_MAX__": f"{max(cov):.1%}",
            "__LONG_CVAR_WORST__": f"{lg.bat('ensemble_cvar', 'worst_5pct_mean'):.0f}",
            "__LONG_MEAN_WORST__": f"{lg.bat('ensemble_mean', 'worst_5pct_mean'):.0f}",
            "__LONG_CVAR_COST__": f"{lg.bat('ensemble_mean', 'eur_per_day') - lg.bat('ensemble_cvar', 'eur_per_day'):.0f}",
        }
    for k, v in values.items():
        template = template.replace(k, v)
    return template


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    cfg = load_config(ap.parse_args().config)
    text, dashboard = build(cfg)
    (PROJECT_ROOT / "RESULTS.md").write_text(text, encoding="utf-8")
    out = cfg.paths.site / "live" / "backtest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dashboard, indent=1), encoding="utf-8")
    print(f"wrote RESULTS.md and {out}")
    template = PROJECT_ROOT / "README.template.md"
    if template.exists():
        readme = render_readme(cfg, template.read_text(encoding="utf-8"))
        (PROJECT_ROOT / "README.md").write_text(readme, encoding="utf-8")
        print("wrote README.md from README.template.md")


if __name__ == "__main__":
    main()
