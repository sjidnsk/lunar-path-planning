#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DEFAULT_PYTHON="/home/kai/anaconda3/envs/lunar-explorer/bin/python"
PYTHON_BIN="${PYTHON:-$DEFAULT_PYTHON}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="python3"
fi

PILOT="${PILOT:-outputs/path_feedback_batch_quasi_real_guarded_ppo_rollout_pilot_v1}"
BATCH_ROOT="${BATCH_ROOT:-outputs/path_feedback_batch_guarded_ppo_rollout_clean_src_v1}"
UPDATE="${UPDATE:-outputs/path_feedback_batch_return_aligned_guarded_ppo_update_smoke_v1}"
QUASI_REAL="${QUASI_REAL:-outputs/path_feedback_batch_quasi_real_safe_better_opportunity_expansion_v1}"
FREEZE="${FREEZE:-outputs/path_feedback_batch_quasi_real_guarded_ppo_evidence_freeze_v1}"
CONFIG="${CONFIG:-configs/quasi_real_guarded_ppo_evidence_freeze_v1.json}"
PILOT_CONFIG="${PILOT_CONFIG:-configs/quasi_real_guarded_ppo_rollout_pilot_v1.json}"

rm -rf "$REPO_ROOT/$PILOT"

PYTHON="$PYTHON_BIN" bash "$SCRIPT_DIR/run_quasi_real_guarded_ppo_rollout_pilot.sh" \
  --update-smoke-root "$UPDATE" \
  --candidate-root "$UPDATE" \
  --quasi-real-root "$QUASI_REAL" \
  --output-root "$PILOT" \
  --config "$PILOT_CONFIG"

rm -rf "$REPO_ROOT/$FREEZE"

PYTHON="$PYTHON_BIN" bash "$SCRIPT_DIR/run_quasi_real_guarded_ppo_evidence_freeze.sh" \
  --pilot-root "$PILOT" \
  --batch-root "$BATCH_ROOT" \
  --update-root "$UPDATE" \
  --output-root "$FREEZE" \
  --config "$CONFIG"
