#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DEFAULT_PYTHON="/home/kai/anaconda3/envs/lunar-explorer/bin/python"
PYTHON_BIN="${PYTHON:-$DEFAULT_PYTHON}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="python3"
fi

SOURCE_SRC="${SOURCE_SRC:-outputs/path_feedback_batch_sequential_multi_step_opportunity_clean_src_v1}"
SOURCE_CAND="${SOURCE_CAND:-outputs/path_feedback_batch_sequential_multi_step_opportunity_candidate_v1}"
SOURCE_SEQ="${SOURCE_SEQ:-outputs/path_feedback_batch_policy_gated_sequential_multi_step_opportunity_rollout_v1}"

SRC="${SRC:-outputs/path_feedback_batch_ppo_collector_clean_src_v1}"
CAND="${CAND:-outputs/path_feedback_batch_ppo_collector_candidate_v1}"
SEQ="${SEQ:-outputs/path_feedback_batch_ppo_collector_sequential_v1}"
COLLECTOR="${COLLECTOR:-outputs/path_feedback_batch_ppo_rollout_collector_dry_run_v1}"

rm -rf "$SRC" "$CAND" "$SEQ" "$COLLECTOR"
cp -a "$SOURCE_SRC" "$SRC"
cp -a "$SOURCE_CAND" "$CAND"
cp -a "$SOURCE_SEQ" "$SEQ"

PYTHON="$PYTHON_BIN" bash "$SCRIPT_DIR/run_sequential_evidence_consistency_check.sh" \
  --batch-root "$SEQ" \
  --readiness-summary "$SRC/policy-training-readiness-review-summary.json" \
  --config "$REPO_ROOT/configs/sequential_evidence_consistency_v1.json"

PYTHON="$PYTHON_BIN" bash "$SCRIPT_DIR/run_ppo_rollout_collector_dry_run.sh" \
  --sequential-root "$SEQ" \
  --candidate-root "$CAND" \
  --output-root "$COLLECTOR" \
  --config "$REPO_ROOT/configs/ppo_rollout_collector_dry_run_v1.json"

PYTHON="$PYTHON_BIN" bash "$SCRIPT_DIR/run_policy_training_readiness_review.sh" \
  --batch-root "$SRC" \
  --config "$REPO_ROOT/configs/policy_training_readiness_review_v1.json" \
  --raw-policy-generalization-evaluation-summary "$CAND/raw-policy-generalization-evaluation-summary.json" \
  --policy-gated-sequential-canary-rollout-summary "$SEQ/policy-gated-sequential-canary-rollout-summary.json" \
  --ppo-rollout-collector-summary "$COLLECTOR/ppo-rollout-collector-summary.json"
