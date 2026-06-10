#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON:-/home/kai/anaconda3/envs/lunar-explorer/bin/python}"

SOURCE_ROOT="${SOURCE_ROOT:-outputs/path_feedback_batch_guarded_ppo_rollout_clean_src_v1}"
CANDIDATE_ROOT="${CANDIDATE_ROOT:-outputs/path_feedback_batch_guarded_ppo_rollout_pilot_v1/update}"
QUASI_REAL_ROOT="${QUASI_REAL_ROOT:-outputs/path_feedback_batch_quasi_real_map_domain_gap_v1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/path_feedback_batch_quasi_real_shadow_policy_behavior_v1}"
CONFIG="${CONFIG:-configs/quasi_real_shadow_policy_behavior_audit_v1.json}"

rm -rf "$REPO_ROOT/$OUTPUT_ROOT"

PYTHON="$PYTHON_BIN" bash "$SCRIPT_DIR/run_quasi_real_shadow_policy_behavior_audit.sh" \
  --source-root "$SOURCE_ROOT" \
  --candidate-root "$CANDIDATE_ROOT" \
  --quasi-real-root "$QUASI_REAL_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --config "$CONFIG"
