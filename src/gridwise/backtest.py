"""Rolling-origin backtest of the forecasters.

    uv run python -m gridwise.backtest --name main

Every test day is forecast by models fitted only on earlier days, then calibrated online with errors
from earlier days only. Raw and calibrated forecasts are saved under reports/backtest/<name>/ so the
battery study and the report can be rebuilt without refitting.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np

from gridwise import metrics as M
from gridwise.config import Backtest, Config, load_config
from gridwise.conformal import calibrate_sequence, point_to_quantiles
from gridwise.dataset import load_hourly
from gridwise.features import OPERATOR, WEATHER, Features, build
from gridwise.frames import to_daily
from gridwise.models.base import Forecaster
from gridwise.models.baselines import Lear, SimilarDayNaive
from gridwise.models.gbm import QuantileGBM

LEAR_EXOG = {
    "price": [],
    "weather": ["wx_wind_on_pow", "wx_wind_off_pow", "wx_solar", "wx_temp"],
    "operator_leaky": ["op_load", "op_wind", "op_solar"],
}


@dataclass
class Forecasts:
    """Forecasts of one model for consecutive scored days."""

    name: str
    idx: np.ndarray  # day indices into the feature arrays
    days: np.ndarray  # datetime64[D]
    raw: np.ndarray  # (n, 24, Q) normalised, before calibration
    q: np.ndarray  # (n, 24, Q) EUR/MWh, calibrated
    q_uncal: np.ndarray  # (n, 24, Q) EUR/MWh, before calibration
    y: np.ndarray  # (n, 24) EUR/MWh
    real: np.ndarray  # (n, 24) bool
    is_test: np.ndarray  # (n,) bool: False for warm-up days

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path, idx=self.idx, days=self.days.astype("datetime64[D]").astype(str), raw=self.raw, q=self.q,
            q_uncal=self.q_uncal, y=self.y, real=self.real, is_test=self.is_test,
        )  # fmt: skip

    @classmethod
    def load(cls, name: str, path: Path) -> Forecasts:
        z = np.load(path)
        return cls(name, z["idx"], z["days"].astype("datetime64[D]"), z["raw"], z["q"], z["q_uncal"], z["y"],
                   z["real"], z["is_test"])  # fmt: skip


def load_features(cfg: Config, feature_set: str) -> Features:
    hourly = load_hourly(cfg)
    cols = ["price", *WEATHER, *OPERATOR]
    return build(to_daily(hourly, cols, cfg.market.timezone), feature_set)


def last_priced_day(feats: Features) -> date:
    known = ~np.isnan(feats.price).any(axis=1)
    return feats.days[np.where(known)[0][-1]].astype(object)


def make_models(cfg: Config, feature_set: str, names: list[str]) -> list[Forecaster]:
    out: list[Forecaster] = []
    for n in names:
        if n == "naive":
            out.append(SimilarDayNaive())
        elif n == "lear":
            out.append(Lear(exog=LEAR_EXOG[feature_set], **cfg.models.get("lear", {})))  # type: ignore[arg-type]
        elif n == "qgbm":
            out.append(QuantileGBM(cfg.quantiles, **cfg.models.get("qgbm", {})))
        elif n == "qnn":
            from gridwise.models.nn import QuantileNN  # imported here: PyTorch is an optional extra

            out.append(QuantileNN(cfg.quantiles, feature_set=feature_set, **cfg.models.get("qnn", {})))
        else:
            raise ValueError(f"unknown model {n!r}")
    return out


def block_start(feats: Features, d: int, refit_every: int) -> int:
    """First day index of the refit block that contains day d.

    Blocks are anchored to the calendar (days since 1970-01-01), not to where a run happens to start.
    So the model used for a given day is the same in a long backtest and in the daily live job.
    """
    ordinal = int(feats.days[d].astype("datetime64[D]").astype("int64"))
    return d - ordinal % refit_every


def rolling_forecast(model: Forecaster, feats: Features, idx: np.ndarray, min_train_days: int) -> np.ndarray:
    """Raw normalised forecasts for the days `idx` (sorted).

    The model for a block of days is fitted on days before the block's first calendar day only.
    """
    ready = np.where(feats.ready & ~np.isnan(feats.y).any(axis=1))[0]
    out: np.ndarray | None = None
    i = 0
    while i < len(idx):
        start = block_start(feats, int(idx[i]), model.refit_every)
        train = ready[ready < start]
        if model.train_days is not None:
            train = train[train >= start - model.train_days]
        if len(train) < min_train_days:
            raise ValueError(
                f"{model.name}: only {len(train)} training days before {feats.days[int(idx[i])]}"
            )
        model.fit(feats, train)
        j = i
        while j < len(idx) and idx[j] < start + model.refit_every:
            j += 1
        pred = model.predict(feats, idx[i:j])
        if out is None:
            out = np.full((len(idx), *pred.shape[1:]), np.nan)
        out[i:j] = pred
        i = j
    assert out is not None
    return out


def to_forecasts(
    name: str, kind: str, feats: Features, idx: np.ndarray, raw: np.ndarray, cfg: Config, test_start: date
) -> Forecasts:
    """Calibrate raw forecasts day by day and convert them to EUR/MWh."""
    taus = np.asarray(cfg.quantiles)
    if kind == "point":
        raw = point_to_quantiles(raw, len(taus))
    cal = calibrate_sequence(
        raw,
        feats.y[idx],
        taus,
        feats.real[idx],
        cfg.calibration.window_days,
        cfg.calibration.mode,
        cfg.calibration.gamma,
    )
    return Forecasts(
        name=name,
        idx=idx,
        days=feats.days[idx],
        raw=raw,
        q=feats.to_price(cal, idx),
        q_uncal=feats.to_price(np.sort(raw, axis=-1), idx),
        y=feats.price[idx],
        real=feats.real[idx],
        is_test=feats.days[idx] >= np.datetime64(test_start),
    )


def run_model(model: Forecaster, feats: Features, bt: Backtest, cfg: Config, end: date) -> Forecasts:
    start = bt.test_start - timedelta(days=bt.warmup_days)
    lo, hi = np.datetime64(start), np.datetime64(end)
    idx = np.where(feats.ready & (feats.days >= lo) & (feats.days <= hi))[0]
    raw = rolling_forecast(model, feats, idx, bt.min_train_days)
    return to_forecasts(model.name, model.kind, feats, idx, raw, cfg, bt.test_start)


def ensemble(members: list[Forecasts], feats: Features, cfg: Config, name: str = "ensemble") -> Forecasts:
    """Average the members' calibrated quantiles level by level, then calibrate the average again."""
    base = members[0]
    for m in members[1:]:
        if not np.array_equal(m.idx, base.idx):
            raise ValueError("ensemble members cover different days")
    taus = np.asarray(cfg.quantiles)
    scale = feats.scale[base.idx][:, None, None]
    level = feats.level[base.idx][:, None, None]
    raw = np.mean([(m.q - level) / scale for m in members], axis=0)
    cal = calibrate_sequence(
        raw, feats.y[base.idx], taus, base.real, cfg.calibration.window_days, cfg.calibration.mode,
        cfg.calibration.gamma,
    )  # fmt: skip
    return Forecasts(
        name=name,
        idx=base.idx,
        days=base.days,
        raw=raw,
        q=feats.to_price(cal, base.idx),
        q_uncal=feats.to_price(raw, base.idx),
        y=base.y,
        real=base.real,
        is_test=base.is_test,
    )


def interval(q: np.ndarray, taus: np.ndarray, nominal: float) -> tuple[np.ndarray, np.ndarray]:
    lo_tau = (1 - nominal) / 2
    lo = int(np.argmin(np.abs(taus - lo_tau)))
    hi = int(np.argmin(np.abs(taus - (1 - lo_tau))))
    assert abs(taus[lo] - lo_tau) < 1e-9, f"no quantile level for a {nominal:.0%} interval"
    return q[..., lo], q[..., hi]


def score(f: Forecasts, taus: np.ndarray, rows: np.ndarray | None = None) -> dict[str, float]:
    """Forecast metrics on test days (or the given boolean row mask)."""
    r = f.is_test if rows is None else rows
    q, qu, y, mask = f.q[r], f.q_uncal[r], f.y[r], f.real[r]
    med = q[..., list(taus).index(0.5)]
    out = {
        "days": float(r.sum()),
        "mae": M.mae(med, y, mask),
        "rmse": M.rmse(med, y, mask),
        "crps": M.crps(q, y, taus, mask),
        "crps_uncalibrated": M.crps(qu, y, taus, mask),
    }
    for nominal in (0.8, 0.9, 0.95):
        lo, hi = interval(q, taus, nominal)
        ulo, uhi = interval(qu, taus, nominal)
        k = f"{round(nominal * 100)}"
        out[f"coverage_{k}"] = M.coverage(lo, hi, y, mask)
        out[f"coverage_{k}_uncalibrated"] = M.coverage(ulo, uhi, y, mask)
        out[f"width_{k}"] = M.width(lo, hi, y, mask)
    return out


def compare(a: Forecasts, b: Forecasts, taus: np.ndarray) -> dict[str, float]:
    """Is A more accurate than B on the test days? One-sided Diebold-Mariano p-values on daily losses."""
    r = a.is_test
    i50 = list(taus).index(0.5)
    _, p_mae = M.diebold_mariano(
        M.daily_abs_error(a.q[r][..., i50], a.y[r], a.real[r]),
        M.daily_abs_error(b.q[r][..., i50], b.y[r], b.real[r]),
    )
    _, p_crps = M.diebold_mariano(
        M.daily_pinball(a.q[r], a.y[r], taus, a.real[r]), M.daily_pinball(b.q[r], b.y[r], taus, b.real[r])
    )
    return {"dm_p_mae": p_mae, "dm_p_crps": p_crps}


def run(cfg: Config, name: str, model_names: list[str], reuse: bool = False) -> dict[str, Any]:
    bt = cfg.backtests[name]
    feats = load_features(cfg, bt.feature_set)
    end = bt.test_end or last_priced_day(feats)
    out_dir = cfg.paths.reports / "backtest" / name
    taus = np.asarray(cfg.quantiles)
    results: dict[str, Forecasts] = {}
    timing: dict[str, float] = {}
    for model in make_models(cfg, bt.feature_set, model_names):
        path = out_dir / f"{model.name}.npz"
        t0 = time.time()
        if reuse and path.exists():
            results[model.name] = Forecasts.load(model.name, path)
        else:
            results[model.name] = run_model(model, feats, bt, cfg, end)
            results[model.name].save(path)
        timing[model.name] = time.time() - t0
        print(f"[{name}] {model.name}: {timing[model.name]:.0f}s", flush=True)
    if bt.same_days_as:
        other = Forecasts.load("naive", cfg.paths.reports / "backtest" / bt.same_days_as / "naive.npz")
        shared = other.days[other.is_test]
        for mname, f in results.items():
            f.is_test = f.is_test & np.isin(f.days, shared)
            f.save(out_dir / f"{mname}.npz")
    members = [results[m] for m in ("lear", "qgbm", "qnn") if m in results]
    if len(members) >= 2:
        results["ensemble"] = ensemble(members, feats, cfg)
        results["ensemble"].save(out_dir / "ensemble.npz")

    summary: dict[str, Any] = {
        "backtest": name,
        "feature_set": bt.feature_set,
        "test_start": str(bt.test_start),
        "test_end": str(end),
        "n_features": len(feats.names),
        "models": {},
        "seconds": timing,
    }
    for mname, f in results.items():
        s: dict[str, Any] = dict(score(f, taus))
        if "naive" in results:
            s["rmae"] = s["mae"] / score(results["naive"], taus)["mae"]
        for ref in ("naive", "lear"):
            if ref in results and ref != mname:
                s |= {f"{k}_vs_{ref}": v for k, v in compare(f, results[ref], taus).items()}
        years = f.days.astype("datetime64[Y]").astype(int) + 1970
        s["by_year"] = {
            str(y): {k: v for k, v in score(f, taus, f.is_test & (years == y)).items() if "uncal" not in k}
            for y in sorted(set(years[f.is_test].tolist()))
        }
        summary["models"][mname] = s
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def print_summary(summary: dict[str, Any]) -> None:
    print(
        f"\n{summary['backtest']} ({summary['feature_set']}): "
        f"{summary['test_start']} .. {summary['test_end']}"
    )
    print(f"{'model':10s} {'days':>5s} {'MAE':>7s} {'rMAE':>6s} {'CRPS':>7s} {'cov80':>6s} {'cov90':>6s} "
          f"{'raw90':>6s} {'w90':>7s}")  # fmt: skip
    for n, s in summary["models"].items():
        print(f"{n:10s} {s['days']:5.0f} {s['mae']:7.2f} {s.get('rmae', float('nan')):6.3f} {s['crps']:7.2f} "
              f"{s['coverage_80']:6.3f} {s['coverage_90']:6.3f} {s['coverage_90_uncalibrated']:6.3f} "
              f"{s['width_90']:7.1f}")  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--name", default="main")
    ap.add_argument("--models", default="naive,lear,qgbm,qnn")
    ap.add_argument("--reuse", action="store_true", help="reuse saved forecasts instead of refitting")
    a = ap.parse_args()
    cfg = load_config(a.config)
    print_summary(run(cfg, a.name, a.models.split(","), a.reuse))


if __name__ == "__main__":
    main()
