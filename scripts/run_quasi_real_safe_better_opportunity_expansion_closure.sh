#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DEFAULT_PYTHON="/home/kai/anaconda3/envs/lunar-explorer/bin/python"
PYTHON_BIN="${PYTHON:-$DEFAULT_PYTHON}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="python3"
fi

OUT="${OUT:-outputs/path_feedback_batch_quasi_real_safe_better_opportunity_expansion_v1}"
EXPANSION_CONFIG="${EXPANSION_CONFIG:-configs/quasi_real_safe_better_opportunity_expansion_v1.json}"
DOMAIN_GAP_CONFIG="${DOMAIN_GAP_CONFIG:-configs/quasi_real_map_domain_gap_evaluation_v1.json}"
DIAGNOSIS_CONFIG="${DIAGNOSIS_CONFIG:-configs/quasi_real_safe_alternative_opportunity_diagnosis_v1.json}"
DATA_MANIFEST="${DATA_MANIFEST:-model-explorer/data/manifests/lunar_south_pole_lro_lola_gdr_875s_20m.json}"
SOURCE_MATRIX="${SOURCE_MATRIX:-model-explorer/data/manifests/lunar_south_pole_lro_lola_selection_matrix_v1.json}"
EXPANSION_MATRIX="${EXPANSION_MATRIX:-model-explorer/data/manifests/lunar_south_pole_lro_lola_safe_better_opportunity_matrix_v1.json}"
GENERATED_REFERENCE="${GENERATED_REFERENCE:-outputs/path_feedback_batch_guarded_ppo_rollout_clean_src_v1/batch-evaluation-summary.json}"
GUARDED_ROOT="${GUARDED_ROOT:-outputs/path_feedback_batch_quasi_real_guarded_policy_pilot_v1}"
ALIGNMENT_ROOT="${ALIGNMENT_ROOT:-outputs/path_feedback_batch_quasi_real_shadow_alignment_candidate_v1}"

rm -rf "$REPO_ROOT/$OUT"

PYTHON="$PYTHON_BIN" bash "$SCRIPT_DIR/run_quasi_real_lola_data_prepare.sh" \
  --manifest "$DATA_MANIFEST" \
  --output-root "$OUT"

PYTHON="$PYTHON_BIN" bash "$SCRIPT_DIR/run_quasi_real_safe_better_opportunity_expansion.sh" \
  --matrix-manifest "$SOURCE_MATRIX" \
  --output-root "$OUT" \
  --config "$EXPANSION_CONFIG"

PYTHON="$PYTHON_BIN" bash "$SCRIPT_DIR/run_quasi_real_map_path_feedback_bridge.sh" \
  --matrix-manifest "$EXPANSION_MATRIX" \
  --output-root "$OUT" \
  --config "$EXPANSION_CONFIG"

PYTHONPATH="$REPO_ROOT/model-explorer/src" "$PYTHON_BIN" -m model_explorer path-feedback run \
  "$REPO_ROOT/$OUT/quasi-real-map-path-feedback-manifest.json"

PYTHON="$PYTHON_BIN" bash "$SCRIPT_DIR/run_quasi_real_map_domain_gap_evaluation.sh" \
  --bridge-summary "$OUT/quasi-real-map-path-feedback-bridge-summary.json" \
  --quasi-real-path-feedback-summary "$OUT/quasi-real-map-path-feedback-summary.json" \
  --generated-reference-summary "$GENERATED_REFERENCE" \
  --output-root "$OUT" \
  --config "$DOMAIN_GAP_CONFIG"

PYTHON="$PYTHON_BIN" bash "$SCRIPT_DIR/run_quasi_real_safe_alternative_opportunity_diagnosis.sh" \
  --quasi-real-root "$OUT" \
  --guarded-pilot-root "$GUARDED_ROOT" \
  --alignment-root "$ALIGNMENT_ROOT" \
  --output-root "$OUT" \
  --config "$DIAGNOSIS_CONFIG"

PYTHON="$PYTHON_BIN" bash "$SCRIPT_DIR/run_quasi_real_safe_better_opportunity_expansion.sh" \
  --matrix-manifest "$SOURCE_MATRIX" \
  --output-root "$OUT" \
  --config "$EXPANSION_CONFIG" \
  --diagnosis-summary "$OUT/quasi-real-safe-alternative-opportunity-summary.json"
