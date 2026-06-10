#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON:-/home/kai/anaconda3/envs/lunar-explorer/bin/python}"

SHADOW_ROOT="${SHADOW_ROOT:-outputs/path_feedback_batch_quasi_real_shadow_policy_behavior_v1}"
QUASI_REAL_ROOT="${QUASI_REAL_ROOT:-outputs/path_feedback_batch_quasi_real_map_domain_gap_v1}"
BASE_CANDIDATE_ROOT="${BASE_CANDIDATE_ROOT:-outputs/path_feedback_batch_guarded_ppo_rollout_pilot_v1/update}"
TAXONOMY_ROOT="${TAXONOMY_ROOT:-outputs/path_feedback_batch_quasi_real_shadow_failure_taxonomy_v1}"
DATASET_ROOT="${DATASET_ROOT:-outputs/path_feedback_batch_quasi_real_shadow_alignment_dataset_v1}"
PREFERENCE_ROOT="${PREFERENCE_ROOT:-outputs/path_feedback_batch_quasi_real_shadow_alignment_preference_v1}"
CANDIDATE_ROOT="${CANDIDATE_ROOT:-outputs/path_feedback_batch_quasi_real_shadow_alignment_candidate_v1}"

rm -rf "$REPO_ROOT/$TAXONOMY_ROOT" "$REPO_ROOT/$DATASET_ROOT" "$REPO_ROOT/$PREFERENCE_ROOT" "$REPO_ROOT/$CANDIDATE_ROOT"

PYTHON="$PYTHON_BIN" bash "$SCRIPT_DIR/run_quasi_real_shadow_failure_taxonomy.sh" \
  --shadow-root "$SHADOW_ROOT" \
  --quasi-real-root "$QUASI_REAL_ROOT" \
  --output-root "$TAXONOMY_ROOT" \
  --config configs/quasi_real_shadow_failure_taxonomy_v1.json

PYTHON="$PYTHON_BIN" bash "$SCRIPT_DIR/run_quasi_real_shadow_alignment_dataset.sh" \
  --taxonomy-root "$TAXONOMY_ROOT" \
  --output-root "$DATASET_ROOT" \
  --config configs/quasi_real_shadow_alignment_splits_v1.json

PYTHON="$PYTHON_BIN" bash "$SCRIPT_DIR/run_quasi_real_shadow_alignment_preference_mining.sh" \
  --taxonomy-root "$TAXONOMY_ROOT" \
  --dataset-root "$DATASET_ROOT" \
  --output-root "$PREFERENCE_ROOT" \
  --config configs/quasi_real_shadow_alignment_preference_v1.json

PYTHON="$PYTHON_BIN" bash "$SCRIPT_DIR/run_quasi_real_shadow_alignment_candidate.sh" \
  --taxonomy-root "$TAXONOMY_ROOT" \
  --dataset-root "$DATASET_ROOT" \
  --preference-root "$PREFERENCE_ROOT" \
  --base-candidate-root "$BASE_CANDIDATE_ROOT" \
  --quasi-real-root "$QUASI_REAL_ROOT" \
  --output-root "$CANDIDATE_ROOT" \
  --config configs/quasi_real_shadow_alignment_candidate_v1.json
