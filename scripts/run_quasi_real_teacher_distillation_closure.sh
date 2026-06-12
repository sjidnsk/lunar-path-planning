#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON:-/home/kai/anaconda3/envs/lunar-explorer/bin/python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

TEACHER_ROOT="${TEACHER_ROOT:-outputs/path_feedback_batch_quasi_real_teacher_equivalent_validation_v1}"
QUASI_REAL_ROOT="${QUASI_REAL_ROOT:-outputs/path_feedback_batch_quasi_real_safe_better_opportunity_expansion_v1}"
SOURCE_ROOT="${SOURCE_ROOT:-outputs/path_feedback_batch_guarded_ppo_rollout_clean_src_v1}"
BASE_CANDIDATE_ROOT="${BASE_CANDIDATE_ROOT:-outputs/path_feedback_batch_quasi_real_shadow_alignment_candidate_v1}"
TAXONOMY_ROOT="${TAXONOMY_ROOT:-outputs/path_feedback_batch_quasi_real_teacher_distillation_taxonomy_v1}"
DATASET_ROOT="${DATASET_ROOT:-outputs/path_feedback_batch_quasi_real_teacher_distillation_dataset_v1}"
PREFERENCE_ROOT="${PREFERENCE_ROOT:-outputs/path_feedback_batch_quasi_real_teacher_distillation_preference_v1}"
CANDIDATE_ROOT="${CANDIDATE_ROOT:-outputs/path_feedback_batch_quasi_real_teacher_distillation_candidate_v1}"
VALIDATION_ROOT="${VALIDATION_ROOT:-outputs/path_feedback_batch_quasi_real_teacher_distillation_validation_v1}"

rm -rf \
  "${REPO_ROOT}/${TAXONOMY_ROOT}" \
  "${REPO_ROOT}/${DATASET_ROOT}" \
  "${REPO_ROOT}/${PREFERENCE_ROOT}" \
  "${REPO_ROOT}/${CANDIDATE_ROOT}" \
  "${REPO_ROOT}/${VALIDATION_ROOT}"

PYTHON="${PYTHON_BIN}" bash "${SCRIPT_DIR}/run_quasi_real_teacher_distillation_taxonomy.sh" \
  --teacher-root "${TEACHER_ROOT}" \
  --quasi-real-root "${QUASI_REAL_ROOT}" \
  --output-root "${TAXONOMY_ROOT}" \
  --config configs/quasi_real_teacher_distillation_taxonomy_v1.json

PYTHON="${PYTHON_BIN}" bash "${SCRIPT_DIR}/run_quasi_real_teacher_distillation_dataset.sh" \
  --taxonomy-root "${TAXONOMY_ROOT}" \
  --output-root "${DATASET_ROOT}" \
  --config configs/quasi_real_teacher_distillation_dataset_v1.json

PYTHON="${PYTHON_BIN}" bash "${SCRIPT_DIR}/run_quasi_real_teacher_distillation_preference_mining.sh" \
  --taxonomy-root "${TAXONOMY_ROOT}" \
  --dataset-root "${DATASET_ROOT}" \
  --output-root "${PREFERENCE_ROOT}" \
  --config configs/quasi_real_teacher_distillation_preference_v1.json

PYTHON="${PYTHON_BIN}" bash "${SCRIPT_DIR}/run_quasi_real_teacher_distillation_candidate.sh" \
  --taxonomy-root "${TAXONOMY_ROOT}" \
  --dataset-root "${DATASET_ROOT}" \
  --preference-root "${PREFERENCE_ROOT}" \
  --base-candidate-root "${BASE_CANDIDATE_ROOT}" \
  --source-root "${SOURCE_ROOT}" \
  --quasi-real-root "${QUASI_REAL_ROOT}" \
  --output-root "${CANDIDATE_ROOT}" \
  --validation-output-root "${VALIDATION_ROOT}" \
  --config configs/quasi_real_teacher_distillation_candidate_v1.json
