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
BASE="${BASE:-outputs/path_feedback_batch_quasi_real_iterative_ppo_mini_loop_stability_v1/round-02/update}"
GUARDED_COLLECTOR="${GUARDED_COLLECTOR:-outputs/path_feedback_batch_guarded_ppo_rollout_pilot_v1/pilot/collector}"
RETURN_ALIGNED="${RETURN_ALIGNED:-outputs/path_feedback_batch_return_aligned_guarded_multi_step_ppo_collector_expansion_v1}"
UPDATE="${UPDATE:-outputs/path_feedback_batch_return_aligned_guarded_ppo_update_smoke_v1}"
CONFIG="${CONFIG:-configs/return_aligned_guarded_ppo_update_smoke_v1.json}"
READINESS_CONFIG="${READINESS_CONFIG:-configs/policy_training_readiness_review_v1.json}"

PYTHON="$PYTHON_BIN" bash "$SCRIPT_DIR/run_return_aligned_guarded_multi_step_ppo_collector_closure.sh"

rm -rf "$REPO_ROOT/$UPDATE"

PYTHON="$PYTHON_BIN" bash "$SCRIPT_DIR/run_return_aligned_guarded_ppo_update_smoke.sh" \
  --source-root "$SRC" \
  --base-candidate-root "$BASE" \
  --guarded-collector-root "$GUARDED_COLLECTOR" \
  --return-aligned-root "$RETURN_ALIGNED" \
  --output-root "$UPDATE" \
  --config "$CONFIG"

PYTHON="$PYTHON_BIN" bash "$SCRIPT_DIR/run_policy_training_readiness_review.sh" \
  --batch-root "$SRC" \
  --config "$READINESS_CONFIG" \
  --return-aligned-guarded-ppo-update-smoke-summary "$UPDATE/return-aligned-guarded-ppo-update-smoke-summary.json" \
  --validate-only
