#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DEFAULT_PYTHON="/home/kai/anaconda3/envs/lunar-explorer/bin/python"
PYTHON_BIN="${PYTHON:-$DEFAULT_PYTHON}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="python3"
fi

SRC="${SRC:-outputs/path_feedback_batch_guarded_ppo_rollout_clean_src_v1}"
BASE="${BASE:-outputs/path_feedback_batch_quasi_real_teacher_distillation_candidate_v1}"
UPDATE="${UPDATE:-outputs/path_feedback_batch_limited_quasi_real_ppo_update_smoke_v1}"
DIAGNOSIS="${DIAGNOSIS:-outputs/path_feedback_batch_quasi_real_generated_sequential_contract_compatibility_diagnosis_v1}"
CONFIG="${CONFIG:-configs/quasi_real_generated_sequential_contract_compatibility_diagnosis_v1.json}"

if [[ ! -f "$REPO_ROOT/$UPDATE/limited-quasi-real-ppo-update-smoke-summary.json" ]]; then
  echo "missing update smoke summary: $UPDATE/limited-quasi-real-ppo-update-smoke-summary.json" >&2
  exit 2
fi

rm -rf "$REPO_ROOT/$DIAGNOSIS"

PYTHON="$PYTHON_BIN" bash "$SCRIPT_DIR/run_quasi_real_generated_sequential_contract_compatibility_diagnosis.sh" \
  --update-smoke-root "$UPDATE" \
  --base-candidate-root "$BASE" \
  --source-root "$SRC" \
  --output-root "$DIAGNOSIS" \
  --config "$REPO_ROOT/$CONFIG"
