#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON:-/home/kai/anaconda3/envs/lunar-explorer/bin/python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

SOURCE_ROOT="${SOURCE_ROOT:-outputs/path_feedback_batch_guarded_ppo_rollout_clean_src_v1}"
CANDIDATE_ROOT="${CANDIDATE_ROOT:-outputs/path_feedback_batch_quasi_real_teacher_distillation_candidate_v1}"
QUASI_REAL_ROOT="${QUASI_REAL_ROOT:-outputs/path_feedback_batch_quasi_real_safe_better_opportunity_expansion_v1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/path_feedback_batch_quasi_real_guarded_teacher_following_pilot_v1}"

rm -rf "${REPO_ROOT}/${OUTPUT_ROOT}"

PYTHON="${PYTHON_BIN}" bash "${SCRIPT_DIR}/run_quasi_real_guarded_teacher_following_pilot.sh" \
  --source-root "${SOURCE_ROOT}" \
  --candidate-root "${CANDIDATE_ROOT}" \
  --quasi-real-root "${QUASI_REAL_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --config configs/quasi_real_guarded_teacher_following_pilot_v1.json
