"""Draw the README figures from the saved backtest forecasts.

    uv run python scripts/make_figures.py

Colours and mark rules follow one small system: blue = this project's forecast, orange = the comparison,
grey = context; thin marks, one axis per chart, direct labels, quiet grid.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from gridwise.backtest import Forecasts
from gridwise.config import load_config

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, CONTEXT = "#2a78d6", "#eb6834", "#898781"
BAND80, BAND95 = "#bdd6f4", "#dfeafa"

plt.rcParams.update(
    {
        "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"],
        "font.size": 10.5,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK2,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.axisbelow": True,
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "axes.titlesize": 12.5,
        "axes.titleweight": "semibold",
        "axes.titlelocation": "left",
    }
)

MODEL_LABEL = {
    "ensemble": "Ensemble",
    "qgbm": "Gradient boosting",
    "qnn": "Neural network",
    "lear": "LASSO baseline",
    "naive": "Naive",
}
POLICY_LABEL = {
    "ensemble_mean": "Plan on the ensemble forecast",
    "ensemble_median": "Plan on the ensemble median",
    "ensemble_cvar": "Risk-averse plan",
    "lear": "Plan on the LASSO baseline",
    "naive": "Plan on the naive forecast",
    "profile": "No model (28-day hourly average)",
}


def finish(fig: plt.Figure, path: Path, subtitle: str | None = None) -> None:
    if subtitle:
        fig.axes[0].text(
            0, 1.02, subtitle, transform=fig.axes[0].transAxes, color=INK2, fontsize=10, va="bottom"
        )
    fig.savefig(path, dpi=160, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print(f"wrote {path}")


def bars(ax: plt.Axes, labels: list[str], values: list[float], fmt: str, highlight: int = 0) -> None:
    y = np.arange(len(labels))[::-1]
    colors = [BLUE if i == highlight else CONTEXT for i in range(len(labels))]
    ax.barh(y, values, height=0.3, color=colors)  # thin marks: about 20 px at this size
    for yi, v in zip(y, values, strict=True):
        ax.text(v, yi, "  " + fmt.format(v), va="center", ha="left", color=INK, fontweight="semibold")
    ax.set_yticks(y, labels, color=INK2)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    ax.spines["bottom"].set_visible(False)
    ax.margins(y=0.12)


def fig_accuracy(summary: dict, out: Path) -> None:
    order = [m for m in ("ensemble", "qgbm", "qnn", "lear", "naive") if m in summary["models"]]
    fig, ax = plt.subplots(figsize=(7.2, 0.52 * len(order) + 1.3))
    bars(ax, [MODEL_LABEL[m] for m in order], [summary["models"][m]["mae"] for m in order], "{:.1f}")
    ax.set_xlim(0, max(summary["models"][m]["mae"] for m in order) * 1.15)
    ax.set_title("Forecast error by model", pad=24)
    days = int(summary["models"]["ensemble"]["days"])
    sub = f"Mean absolute error, EUR/MWh. {days} days, {summary['test_start']} to {summary['test_end']}. Lower is better."
    finish(fig, out / "accuracy.png", sub)


def fig_battery(decisions: dict, out: Path) -> None:
    pol = {k: v for k, v in decisions["policies"].items() if k != "perfect"}
    order = sorted(pol, key=lambda k: -pol[k]["share_of_perfect"])
    fig, ax = plt.subplots(figsize=(7.2, 0.52 * len(order) + 1.3))
    bars(ax, [POLICY_LABEL.get(k, k) for k in order], [100 * pol[k]["share_of_perfect"] for k in order], "{:.1f}%",
         highlight=order.index("ensemble_mean"))  # fmt: skip
    ax.set_xlim(0, 108)
    ax.set_xticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
    ax.set_title("Share of perfect-foresight battery profit captured", pad=24)
    b = decisions["battery"]
    sub = f"{decisions['days']} days, {b['power_mw']:g} MW / {b['capacity_mwh']:g} MWh battery, plans fixed before prices were known."
    finish(fig, out / "battery.png", sub)


def fig_reliability(f: Forecasts, taus: np.ndarray, out: Path) -> None:
    r = f.is_test
    ok = f.real[r] & ~np.isnan(f.y[r])
    y = f.y[r][ok]
    before = [(y < f.q_uncal[r][ok][:, j]).mean() for j in range(len(taus))]
    after = [(y < f.q[r][ok][:, j]).mean() for j in range(len(taus))]
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    ax.plot([0, 1], [0, 1], color=AXIS, linewidth=1.2)
    ax.plot(
        taus,
        before,
        color=ORANGE,
        linewidth=2,
        marker="o",
        markersize=5,
        markeredgecolor=SURFACE,
        markeredgewidth=1.2,
    )
    ax.plot(
        taus,
        after,
        color=BLUE,
        linewidth=2,
        marker="o",
        markersize=5,
        markeredgecolor=SURFACE,
        markeredgewidth=1.2,
    )
    # labels sit on the side of each curve that is free of the other one
    ax.annotate("Before calibration", (taus[-4], before[-4]), xytext=(34, -34), textcoords="offset points",
                color=INK2, ha="left", va="center", arrowprops={"arrowstyle": "-", "color": MUTED, "linewidth": 0.8})  # fmt: skip
    ax.annotate("After calibration", (taus[3], after[3]), xytext=(34, -34), textcoords="offset points",
                color=INK2, ha="left", va="center", arrowprops={"arrowstyle": "-", "color": MUTED, "linewidth": 0.8})  # fmt: skip
    ax.text(0.93, 0.985, "perfect", color=MUTED, ha="right", va="top", fontsize=9.5)
    ticks = [0, 0.25, 0.5, 0.75, 1]
    ax.set_xticks(ticks, [f"{t:.0%}" for t in ticks])
    ax.set_yticks(ticks, [f"{t:.0%}" for t in ticks])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xlabel("Quantile level the model claimed")
    ax.set_ylabel("Share of real prices below it")
    ax.set_title("Calibration makes the quantiles mean what they say", pad=24)
    finish(fig, out / "calibration.png", f"{MODEL_LABEL[f.name]} model, {int(r.sum())} days.")


def fig_coverage_over_time(f: Forecasts, taus: np.ndarray, out: Path, window: int = 90) -> None:
    r = f.is_test
    lo, hi = list(taus).index(0.05), list(taus).index(0.95)
    ok = f.real[r] & ~np.isnan(f.y[r])

    def rolling(q: np.ndarray) -> np.ndarray:
        inside = ((f.y[r] >= q[r][..., lo]) & (f.y[r] <= q[r][..., hi]) & ok).sum(axis=1).astype(float)
        hours = ok.sum(axis=1).astype(float)
        k = np.ones(window)
        return np.convolve(inside, k, "valid") / np.convolve(hours, k, "valid")

    days = f.days[r][window - 1 :].astype("datetime64[D]").astype(object)
    after, before = rolling(f.q), rolling(f.q_uncal)
    fig, ax = plt.subplots(figsize=(8.2, 3.6))
    ax.axhline(0.9, color=AXIS, linewidth=1.2)
    ax.plot(days, before, color=ORANGE, linewidth=2)
    ax.plot(days, after, color=BLUE, linewidth=2)
    ax.text(days[-1], after[-1], "  After calibration", color=INK2, va="center")
    ax.text(days[-1], before[-1], "  Before", color=INK2, va="center")
    ax.text(days[0], 0.905, "target 90%", color=MUTED, va="bottom", fontsize=9.5)
    ax.set_ylim(min(0.5, float(before.min()) - 0.03), 1.0)
    ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0], ["50%", "60%", "70%", "80%", "90%", "100%"])
    ax.set_xlim(days[0], days[-1])
    ax.margins(x=0)
    ax.set_title("The 90% range keeps holding through the 2021-23 energy crisis", pad=24)
    sub = f"Share of real prices inside the 90% range, rolling {window} days. {MODEL_LABEL[f.name]}, prices and calendar only."
    fig.subplots_adjust(right=0.82)
    finish(fig, out / "coverage_over_time.png", sub)


def fig_example_day(f: Forecasts, taus: np.ndarray, out: Path) -> None:
    """The test day whose CRPS is the median of all test days: a typical day, chosen by rule, not by eye."""
    from gridwise.metrics import daily_pinball

    r = np.where(f.is_test)[0]
    loss = daily_pinball(f.q[r], f.y[r], taus, f.real[r])
    i = r[np.argsort(loss)[len(loss) // 2]]
    q, y, hours = f.q[i], f.y[i], np.arange(24)
    t = list(taus)
    fig, ax = plt.subplots(figsize=(8.2, 3.8))
    ax.fill_between(hours, q[:, t.index(0.025)], q[:, t.index(0.975)], color=BAND95, linewidth=0)
    ax.fill_between(hours, q[:, t.index(0.1)], q[:, t.index(0.9)], color=BAND80, linewidth=0)
    ax.plot(hours, q[:, t.index(0.5)], color=BLUE, linewidth=2)
    ax.plot(hours, y, color=ORANGE, linewidth=2)
    ax.plot(hours[-1], y[-1], "o", color=ORANGE, markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.5)
    gap = abs(q[-1, t.index(0.5)] - y[-1]) > 0.06 * (q.max() - q.min())
    ax.text(23.3, y[-1], "Real price", color=INK2, va="center")
    if gap:
        ax.text(23.3, q[-1, t.index(0.5)], "Forecast median", color=INK2, va="center")
    ax.text(12, q[12, t.index(0.975)], "95% range", color=MUTED, ha="center", va="bottom", fontsize=9.5)
    ax.set_xticks(range(0, 24, 3), [f"{h:02d}:00" for h in range(0, 24, 3)])
    ax.set_xlim(0, 23)
    day = f.days[i].astype("datetime64[D]").astype(object)
    ax.set_title(f"A typical day: forecast for {day:%A %d %B %Y}", pad=24)
    sub = "EUR/MWh by hour. Dark band: 80% range; light band: 95% range. The day with the median score of the backtest."
    fig.subplots_adjust(right=0.84)
    finish(fig, out / "example_day.png", sub)


def main() -> None:
    cfg = load_config()
    taus = np.asarray(cfg.quantiles)
    out = cfg.paths.site / "images"
    out.mkdir(parents=True, exist_ok=True)
    bt = cfg.paths.reports / "backtest"
    fig_accuracy(json.loads((bt / "main" / "summary.json").read_text(encoding="utf-8")), out)
    fig_battery(json.loads((bt / "main" / "decisions.json").read_text(encoding="utf-8")), out)
    fig_reliability(Forecasts.load("qgbm", bt / "main" / "qgbm.npz"), taus, out)
    fig_example_day(Forecasts.load("ensemble", bt / "main" / "ensemble.npz"), taus, out)
    if (bt / "long" / "ensemble.npz").exists():
        fig_coverage_over_time(Forecasts.load("ensemble", bt / "long" / "ensemble.npz"), taus, out)


if __name__ == "__main__":
    main()
