"""Log a finished backtest to MLflow and remember its run id.

    uv run python -m gridwise.track --name main

Every number in RESULTS.md is read back from the MLflow run recorded in reports/run_ids.json, never
from "the latest run" (the clock on the development machine once jumped backwards).
"""

from __future__ import annotations

import argparse
import json
from typing import Any

import mlflow

from gridwise.config import Config, load_config, setup_mlflow
from gridwise.live import git_sha


def _flat(prefix: str, d: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for k, v in d.items():
        key = f"{prefix}.{k}"
        if isinstance(v, dict):
            out |= _flat(key, v)
        elif isinstance(v, int | float) and not isinstance(v, bool):
            out[key] = float(v)
    return out


def log_backtest(cfg: Config, name: str) -> str:
    out_dir = cfg.paths.reports / "backtest" / name
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    decisions_path = out_dir / "decisions.json"
    decisions = json.loads(decisions_path.read_text(encoding="utf-8")) if decisions_path.exists() else None
    setup_mlflow(cfg)
    bt = cfg.backtests[name]
    with mlflow.start_run(
        run_name=f"backtest-{name}", tags={"code_version": git_sha(), "backtest": name}
    ) as run:
        mlflow.log_params(
            {
                "feature_set": bt.feature_set,
                "test_start": summary["test_start"],
                "test_end": summary["test_end"],
                "n_features": summary["n_features"],
                "warmup_days": bt.warmup_days,
                "quantiles": cfg.quantiles,
                "calibration": cfg.calibration.model_dump(),
                "weather": f"{cfg.weather.model}, {cfg.weather.lead_days}-day lead, "
                f"{len(cfg.weather.sites())} sites",
                **{f"model.{k}": v for k, v in cfg.models.items()},
            }
        )
        metrics = _flat("fc", summary["models"])
        if decisions:
            mlflow.log_params(
                {
                    "battery": decisions["battery"],
                    "risk": decisions["risk"],
                    "decision_model": decisions["model"],
                    "decision_days": decisions["days"],
                }
            )
            metrics |= _flat("bat", decisions["policies"])
            for pol, p in decisions["policies"].items():
                for k, v in p.items():
                    if k.endswith("_ci95"):
                        metrics[f"bat.{pol}.{k}_lo"], metrics[f"bat.{pol}.{k}_hi"] = float(v[0]), float(v[1])
            mlflow.log_artifact(str(decisions_path))
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(out_dir / "summary.json"))
        run_id = run.info.run_id
    ids_path = cfg.paths.reports / "run_ids.json"
    ids = json.loads(ids_path.read_text(encoding="utf-8")) if ids_path.exists() else {}
    ids[name] = run_id
    ids_path.write_text(json.dumps(ids, indent=2), encoding="utf-8")
    return run_id


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--name", required=True)
    a = ap.parse_args()
    print(f"logged {a.name} as MLflow run {log_backtest(load_config(a.config), a.name)}")


if __name__ == "__main__":
    main()
