#!/usr/bin/env bash
# Backtest -> battery study -> MLflow, for one named backtest. Usage: scripts/run_backtest.sh main [--reuse]
set -euo pipefail
name="$1"; shift || true
export PYTHONUTF8=1 MLFLOW_DISABLE_AGENT_HINT=1
uv run python -W ignore -m gridwise.backtest --name "$name" "$@"
uv run python -W ignore -m gridwise.decide --name "$name"
uv run python -W ignore -m gridwise.track --name "$name"
