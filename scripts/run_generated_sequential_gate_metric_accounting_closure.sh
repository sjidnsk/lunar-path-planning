#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DEFAULT_PYTHON="/home/kai/anaconda3/envs/lunar-explorer/bin/python"
PYTHON_BIN="${PYTHON:-$DEFAULT_PYTHON}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="python3"
fi

DIAGNOSIS="${DIAGNOSIS:-outputs/path_feedback_batch_quasi_real_generated_sequential_contract_compatibility_diagnosis_v1}"
AUDIT="${AUDIT:-outputs/path_feedback_batch_generated_sequential_gate_metric_accounting_audit_v1}"
CONFIG="${CONFIG:-configs/generated_sequential_gate_metric_accounting_audit_v1.json}"

if [[ ! -f "$REPO_ROOT/$DIAGNOSIS/quasi-real-generated-sequential-contract-compatibility-summary.json" ]]; then
  echo "missing compatibility diagnosis summary: $DIAGNOSIS/quasi-real-generated-sequential-contract-compatibility-summary.json" >&2
  exit 2
fi

rm -rf "$REPO_ROOT/$AUDIT"

PYTHON="$PYTHON_BIN" bash "$SCRIPT_DIR/run_generated_sequential_gate_metric_accounting_audit.sh" \
  --diagnosis-root "$DIAGNOSIS" \
  --output-root "$AUDIT" \
  --config "$REPO_ROOT/$CONFIG"
