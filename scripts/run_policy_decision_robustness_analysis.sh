#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_PYTHON="/home/kai/anaconda3/envs/lunar-explorer/bin/python"
PYTHON_BIN="${PYTHON:-$DEFAULT_PYTHON}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="python3"
fi

has_config="false"
for arg in "$@"; do
  if [[ "$arg" == "--config" ]]; then
    has_config="true"
    break
  fi
done

if [[ "$has_config" == "true" ]]; then
  exec "$PYTHON_BIN" "$SCRIPT_DIR/run_policy_decision_robustness_analysis.py" "$@"
fi

exec "$PYTHON_BIN" "$SCRIPT_DIR/run_policy_decision_robustness_analysis.py" "$@" \
  --config "$SCRIPT_DIR/../configs/policy_decision_robustness_v1.json"
