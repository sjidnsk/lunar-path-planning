#!/usr/bin/env bash
set -euo pipefail

DRY_RUN=0
OUTPUT_ROOT="outputs/path_feedback_validation"
TOP_K=3
SCENARIO_SET="smoke"
DIAGNOSTIC_PROFILE="baseline"
ANCHOR_PROJECTION_CANDIDATE_GENERATION=0
ANCHOR_PROJECTION_SELECTION_PATH_COST_BONUS="0.0"
ANCHOR_PROJECTION_MAX_SELECTION_PATH_COST_REGRESSION="6.0"
ANCHOR_PROJECTION_MAX_SELECTION_RISK_REGRESSION="0.5"
ANCHOR_PROJECTION_CONTRACT_AWARE_TRAINABLE_TARGET_GENERATION=0
ANCHOR_PROJECTION_PREFER_CONTRACT_SAFE_TRAINABLE_TARGETS=0
ANCHOR_PROJECTION_MAX_TRAINABLE_DISTANCE_CELLS="2"
ANCHOR_PROJECTION_MAX_TRAINABLE_DISTANCE_M="1.0"
ANCHOR_PROJECTION_PLANNER_VALIDATED_TRAINABLE_TARGET_MINING=0
ANCHOR_PROJECTION_ALLOW_PLANNER_VALIDATED_DISTANCE_EXCEPTION=0
ANCHOR_PROJECTION_MAX_PLANNER_VALIDATED_DISTANCE_CELLS="3"
ANCHOR_PROJECTION_MAX_PLANNER_VALIDATED_DISTANCE_M="1.5"
PLANNER_EXTRA_ARGS=()
MODULES=(path-planner model-explorer dev-platform-constraints)
DEFAULT_PYTHON="/home/kai/anaconda3/envs/lunar-explorer/bin/python"
PYTHON_BIN="${PYTHON:-$DEFAULT_PYTHON}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  if [[ -n "${PYTHON:-}" ]]; then
    echo "Configured PYTHON is not executable: $PYTHON_BIN" >&2
    exit 2
  fi
  PYTHON_BIN="python3"
fi

usage() {
  cat <<'USAGE'
Usage: bash scripts/run_path_feedback_validation.sh [options]

Run the reproducible semi-real path feedback validation chain:
dev-platform-constraints -> model-explorer -> path-planner.

Options:
  --output-root PATH    Output root for generated maps, sidecars, manifest, and reports.
                        Default: outputs/path_feedback_validation
  --top-k N            Number of candidate goals per scenario to evaluate. Default: 3
  --scenario-set NAME   Validation scenario set: smoke, stress, holdout,
                        raw_align_train, raw_align_val, raw_align_test,
                        policy_canary, policy_canary_diversity,
                        policy_canary_opportunity_quality,
                        policy_canary_dense_choke_opportunity,
                        policy_canary_value_stability, or all.
                        Default: smoke
  --diagnostic-profile NAME
                        Diagnostic profile: baseline, execution, iris, or all.
                        Default: baseline
  --simulate-tracking  Forward path-planner tracking simulation diagnostics.
  --optimize-trajectory
                        Forward fixed-corridor trajectory optimization diagnostics.
  --drake-iris-regions Forward optional Drake workspace Iris diagnostics.
  --gcs-trajectory-smoke
                        Forward optional Drake GCS corridor trajectory smoke diagnostics.
  --gcs-geometric-candidate
                        Forward optional Drake GCS sampled geometric candidate comparison.
  --gcs-motion-feasibility
                        Forward optional Drake GCS sampled trajectory curvature/heading diagnostics.
  --gcs-curvature-constrained-candidate
                        Forward optional curvature-constrained GCS sampled candidate repair diagnostics.
  --gcs-control-point-candidate
                        Forward optional control-point direction-cone GCS terrain-cost candidate diagnostics.
  --anchor-projection-candidate-generation
                        Enable model-explorer projected target candidate generation for
                        platform-inflated-goal-blocked policy targets.
  --anchor-projection-selection-path-cost-bonus VALUE
                        Opt-in source-selection path-cost bonus for generated
                        anchor-projection execution-target candidates. Only
                        used by model-explorer manifest generation.
  --anchor-projection-max-selection-path-cost-regression VALUE
                        Maximum raw path-cost regression versus the best
                        feasible alternative for a source-selected projected
                        candidate to remain trainable. Manifest-only.
  --anchor-projection-max-selection-risk-regression VALUE
                        Maximum risk regression versus the best feasible
                        alternative for a source-selected projected candidate
                        to remain trainable. Manifest-only.
  --anchor-projection-contract-aware-trainable-target-generation
                        Generate opt-in same-action execution substitutes that
                        keep the policy action index inside the current
                        model-explorer action mask.
  --anchor-projection-prefer-contract-safe-trainable-targets
                        Prefer contract-safe same-action substitutes before
                        ordinary path-cost ranking when quality gates allow it.
  --anchor-projection-max-trainable-distance-cells VALUE
                        Default trainable anchor-projection distance gate in cells.
  --anchor-projection-max-trainable-distance-m VALUE
                        Default trainable anchor-projection distance gate in meters.
  --anchor-projection-planner-validated-trainable-target-mining
                        Enable opt-in post-planner trainable target mining metadata.
  --anchor-projection-allow-planner-validated-distance-exception
                        Allow source-selected same-action substitutes within the
                        planner-validated exception gate to be mined later.
  --anchor-projection-max-planner-validated-distance-cells VALUE
                        Opt-in planner-validated distance exception gate in cells.
  --anchor-projection-max-planner-validated-distance-m VALUE
                        Opt-in planner-validated distance exception gate in meters.
  --gcs-control-point-terrain-weight VALUE
                        Forward explicit control-point terrain objective weight.
  --gcs-control-point-second-difference-weight VALUE
                        Forward explicit control-point second-difference objective weight.
  --gcs-control-point-high-cost-exposure-weight VALUE
                        Forward explicit control-point high-cost exposure proxy objective weight.
  --gcs-control-point-direction-cone-max-error-deg VALUE
                        Forward explicit control-point direction_cone tolerance in degrees.
  --gcs-control-point-direction-cone-rho-floor-m VALUE
                        Forward explicit control-point direction_cone rho floor in meters.
  --gcs-control-point-direction-cone-seed-rho-ratio VALUE
                        Forward explicit control-point direction_cone seed-distance rho ratio.
  --planning-backend NAME
                        Forward path-planner planning backend: astar, region_graph_guided,
                        or channel_aware_astar.
  --channel-aware-neighborhood-radius-cells N
                        Forward channel-aware A* local channel radius.
  --channel-aware-center-weight VALUE
                        Forward channel-aware A* center-cell cost weight.
  --channel-aware-neighborhood-mean-weight VALUE
                        Forward channel-aware A* neighborhood mean-cost weight.
  --channel-aware-neighborhood-max-weight VALUE
                        Forward channel-aware A* neighborhood max-cost weight.
  --channel-aware-high-cost-exposure-weight VALUE
                        Forward channel-aware A* high-cost exposure proxy weight.
  --channel-aware-blocked-nearby-weight VALUE
                        Forward channel-aware A* blocked-nearby penalty weight.
  --channel-aware-clearance-weight VALUE
                        Forward channel-aware A* clearance penalty weight.
  --channel-aware-smoothness-weight VALUE
                        Forward channel-aware A* smoothness/direction proxy weight.
  --channel-aware-high-cost-threshold VALUE
                        Forward channel-aware A* high-cost threshold.
  --dry-run            Print planned commands without writing validation outputs.
  -h, --help           Show this help.
USAGE
}

require_value() {
  local option="$1"
  local value="${2:-}"
  if [[ -z "$value" ]]; then
    echo "Missing value for $option" >&2
    exit 2
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output-root)
      require_value "$1" "${2:-}"
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    --top-k)
      require_value "$1" "${2:-}"
      TOP_K="$2"
      shift 2
      ;;
    --scenario-set)
      require_value "$1" "${2:-}"
      SCENARIO_SET="$2"
      shift 2
      ;;
    --diagnostic-profile)
      require_value "$1" "${2:-}"
      DIAGNOSTIC_PROFILE="$2"
      shift 2
      ;;
    --simulate-tracking|--optimize-trajectory|--drake-iris-regions|--gcs-trajectory-smoke|--gcs-geometric-candidate|--gcs-motion-feasibility|--gcs-curvature-constrained-candidate|--gcs-control-point-candidate)
      PLANNER_EXTRA_ARGS+=("$1")
      shift
      ;;
    --anchor-projection-candidate-generation)
      ANCHOR_PROJECTION_CANDIDATE_GENERATION=1
      shift
      ;;
    --anchor-projection-selection-path-cost-bonus)
      require_value "$1" "${2:-}"
      ANCHOR_PROJECTION_SELECTION_PATH_COST_BONUS="$2"
      shift 2
      ;;
    --anchor-projection-max-selection-path-cost-regression)
      require_value "$1" "${2:-}"
      ANCHOR_PROJECTION_MAX_SELECTION_PATH_COST_REGRESSION="$2"
      shift 2
      ;;
    --anchor-projection-max-selection-risk-regression)
      require_value "$1" "${2:-}"
      ANCHOR_PROJECTION_MAX_SELECTION_RISK_REGRESSION="$2"
      shift 2
      ;;
    --anchor-projection-contract-aware-trainable-target-generation)
      ANCHOR_PROJECTION_CONTRACT_AWARE_TRAINABLE_TARGET_GENERATION=1
      shift
      ;;
    --anchor-projection-prefer-contract-safe-trainable-targets)
      ANCHOR_PROJECTION_PREFER_CONTRACT_SAFE_TRAINABLE_TARGETS=1
      shift
      ;;
    --anchor-projection-max-trainable-distance-cells)
      require_value "$1" "${2:-}"
      ANCHOR_PROJECTION_MAX_TRAINABLE_DISTANCE_CELLS="$2"
      shift 2
      ;;
    --anchor-projection-max-trainable-distance-m)
      require_value "$1" "${2:-}"
      ANCHOR_PROJECTION_MAX_TRAINABLE_DISTANCE_M="$2"
      shift 2
      ;;
    --anchor-projection-planner-validated-trainable-target-mining)
      ANCHOR_PROJECTION_PLANNER_VALIDATED_TRAINABLE_TARGET_MINING=1
      shift
      ;;
    --anchor-projection-allow-planner-validated-distance-exception)
      ANCHOR_PROJECTION_ALLOW_PLANNER_VALIDATED_DISTANCE_EXCEPTION=1
      shift
      ;;
    --anchor-projection-max-planner-validated-distance-cells)
      require_value "$1" "${2:-}"
      ANCHOR_PROJECTION_MAX_PLANNER_VALIDATED_DISTANCE_CELLS="$2"
      shift 2
      ;;
    --anchor-projection-max-planner-validated-distance-m)
      require_value "$1" "${2:-}"
      ANCHOR_PROJECTION_MAX_PLANNER_VALIDATED_DISTANCE_M="$2"
      shift 2
      ;;
    --gcs-control-point-terrain-weight|--gcs-control-point-second-difference-weight|--gcs-control-point-high-cost-exposure-weight|--gcs-control-point-direction-cone-max-error-deg|--gcs-control-point-direction-cone-rho-floor-m|--gcs-control-point-direction-cone-seed-rho-ratio|--channel-aware-neighborhood-radius-cells|--channel-aware-center-weight|--channel-aware-neighborhood-mean-weight|--channel-aware-neighborhood-max-weight|--channel-aware-high-cost-exposure-weight|--channel-aware-blocked-nearby-weight|--channel-aware-clearance-weight|--channel-aware-smoothness-weight|--channel-aware-high-cost-threshold)
      require_value "$1" "${2:-}"
      PLANNER_EXTRA_ARGS+=("$1" "$2")
      shift 2
      ;;
    --planning-backend)
      require_value "$1" "${2:-}"
      case "$2" in
        astar|region_graph_guided|channel_aware_astar)
          ;;
        *)
          echo "--planning-backend must be one of: astar, region_graph_guided, channel_aware_astar" >&2
          exit 2
          ;;
      esac
      PLANNER_EXTRA_ARGS+=("$1" "$2")
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if ! [[ "$TOP_K" =~ ^[0-9]+$ ]] || [[ "$TOP_K" -lt 1 ]]; then
  echo "--top-k must be a positive integer" >&2
  exit 2
fi

if ! "$PYTHON_BIN" - "$ANCHOR_PROJECTION_SELECTION_PATH_COST_BONUS" <<'PY'
import math
import sys

try:
    value = float(sys.argv[1])
except ValueError:
    raise SystemExit(1)
raise SystemExit(0 if math.isfinite(value) and value >= 0.0 else 1)
PY
then
  echo "--anchor-projection-selection-path-cost-bonus must be a non-negative finite number" >&2
  exit 2
fi

for numeric_gate in \
  "$ANCHOR_PROJECTION_MAX_SELECTION_PATH_COST_REGRESSION" \
  "$ANCHOR_PROJECTION_MAX_SELECTION_RISK_REGRESSION" \
  "$ANCHOR_PROJECTION_MAX_TRAINABLE_DISTANCE_CELLS" \
  "$ANCHOR_PROJECTION_MAX_TRAINABLE_DISTANCE_M" \
  "$ANCHOR_PROJECTION_MAX_PLANNER_VALIDATED_DISTANCE_CELLS" \
  "$ANCHOR_PROJECTION_MAX_PLANNER_VALIDATED_DISTANCE_M"; do
  if ! "$PYTHON_BIN" - "$numeric_gate" <<'PY'
import math
import sys

try:
    value = float(sys.argv[1])
except ValueError:
    raise SystemExit(1)
raise SystemExit(0 if math.isfinite(value) and value >= 0.0 else 1)
PY
  then
    echo "anchor-projection source-selection regression gates must be non-negative finite numbers" >&2
    exit 2
  fi
done

if [[ "$ANCHOR_PROJECTION_CANDIDATE_GENERATION" -eq 0 ]]; then
  if ! "$PYTHON_BIN" - "$ANCHOR_PROJECTION_SELECTION_PATH_COST_BONUS" <<'PY'
import sys

raise SystemExit(0 if float(sys.argv[1]) == 0.0 else 1)
PY
  then
    echo "--anchor-projection-selection-path-cost-bonus requires --anchor-projection-candidate-generation" >&2
    exit 2
  fi
  if [[ "$ANCHOR_PROJECTION_CONTRACT_AWARE_TRAINABLE_TARGET_GENERATION" -ne 0 || "$ANCHOR_PROJECTION_PREFER_CONTRACT_SAFE_TRAINABLE_TARGETS" -ne 0 || "$ANCHOR_PROJECTION_PLANNER_VALIDATED_TRAINABLE_TARGET_MINING" -ne 0 || "$ANCHOR_PROJECTION_ALLOW_PLANNER_VALIDATED_DISTANCE_EXCEPTION" -ne 0 ]]; then
    echo "contract-aware anchor-projection options require --anchor-projection-candidate-generation" >&2
    exit 2
  fi
fi

case "$SCENARIO_SET" in
  smoke|stress|holdout|raw_align_train|raw_align_val|raw_align_test|policy_canary|policy_canary_diversity|policy_canary_opportunity_quality|policy_canary_dense_choke_opportunity|policy_canary_value_stability|all)
    ;;
  *)
    echo "--scenario-set must be one of: smoke, stress, holdout, raw_align_train, raw_align_val, raw_align_test, policy_canary, policy_canary_diversity, policy_canary_opportunity_quality, policy_canary_dense_choke_opportunity, policy_canary_value_stability, all" >&2
    exit 2
    ;;
esac

case "$DIAGNOSTIC_PROFILE" in
  baseline|execution|iris|all)
    ;;
  *)
    echo "--diagnostic-profile must be one of: baseline, execution, iris, all" >&2
    exit 2
    ;;
esac

append_planner_arg() {
  local candidate="$1"
  local arg
  for arg in "${PLANNER_EXTRA_ARGS[@]}"; do
    if [[ "$arg" == "$candidate" ]]; then
      return
    fi
  done
  PLANNER_EXTRA_ARGS+=("$candidate")
}

has_planner_arg() {
  local candidate="$1"
  local arg
  for arg in "${PLANNER_EXTRA_ARGS[@]}"; do
    if [[ "$arg" == "$candidate" ]]; then
      return 0
    fi
  done
  return 1
}

case "$DIAGNOSTIC_PROFILE" in
  execution)
    append_planner_arg "--simulate-tracking"
    append_planner_arg "--optimize-trajectory"
    ;;
  iris)
    append_planner_arg "--drake-iris-regions"
    append_planner_arg "--gcs-trajectory-smoke"
    append_planner_arg "--gcs-geometric-candidate"
    append_planner_arg "--gcs-motion-feasibility"
    append_planner_arg "--gcs-curvature-constrained-candidate"
    ;;
  all)
    append_planner_arg "--simulate-tracking"
    append_planner_arg "--optimize-trajectory"
    append_planner_arg "--drake-iris-regions"
    append_planner_arg "--gcs-trajectory-smoke"
    append_planner_arg "--gcs-geometric-candidate"
    append_planner_arg "--gcs-motion-feasibility"
    append_planner_arg "--gcs-curvature-constrained-candidate"
    ;;
esac

for calibration_arg in \
  --gcs-control-point-terrain-weight \
  --gcs-control-point-second-difference-weight \
  --gcs-control-point-high-cost-exposure-weight \
  --gcs-control-point-direction-cone-max-error-deg \
  --gcs-control-point-direction-cone-rho-floor-m \
  --gcs-control-point-direction-cone-seed-rho-ratio; do
  if has_planner_arg "$calibration_arg" && ! has_planner_arg "--gcs-control-point-candidate"; then
    echo "$calibration_arg requires --gcs-control-point-candidate" >&2
    exit 2
  fi
done

ACCEPTANCE_GATE="custom"
if [[ "$SCENARIO_SET" == "all" && "$DIAGNOSTIC_PROFILE" == "all" && "$TOP_K" == "3" ]]; then
  ACCEPTANCE_GATE="semi-real-closed-loop"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
if [[ "$OUTPUT_ROOT" != /* ]]; then
  OUTPUT_ROOT="$REPO_ROOT/$OUTPUT_ROOT"
fi

MAP_DIR="$OUTPUT_ROOT/maps"
SCENARIO_CONFIG="$OUTPUT_ROOT/npz_validation_scenarios.json"
EXPORT_DIR="$OUTPUT_ROOT/path_planner_sidecars"
MANIFEST_PATH="$OUTPUT_ROOT/path-feedback-manifest.json"
SUMMARY_PATH="$OUTPUT_ROOT/path-feedback-summary.json"
REPORT_PATH="$OUTPUT_ROOT/path-feedback-summary.md"

DEV_ROOT="$REPO_ROOT/dev-platform-constraints"
MODEL_ROOT="$REPO_ROOT/model-explorer"
PATH_PLANNER_ROOT="$REPO_ROOT/path-planner"

format_command() {
  local formatted=""
  local arg
  for arg in "$@"; do
    if [[ -n "$formatted" ]]; then
      formatted+=" "
    fi
    printf -v arg "%q" "$arg"
    formatted+="$arg"
  done
  printf '%s' "$formatted"
}

run_cmd() {
  local cwd="$1"
  shift
  local display
  display="$(format_command "$@")"
  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "[DRY RUN] (cd $cwd && $display)"
    return
  fi
  echo "==> (cd $cwd && $display)"
  (cd "$cwd" && "$@")
}

run_pythonpath_cmd() {
  local cwd="$1"
  shift
  local display
  display="$(format_command "$@")"
  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "[DRY RUN] (cd $cwd && PYTHONPATH=src $display)"
    return
  fi
  echo "==> (cd $cwd && PYTHONPATH=src $display)"
  (cd "$cwd" && env PYTHONPATH=src "$@")
}

ensure_submodules() {
  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "[DRY RUN] git submodule update --init --recursive ${MODULES[*]}"
    return
  fi

  if [[ -d "$REPO_ROOT/.git" ]]; then
    git -C "$REPO_ROOT" submodule update --init --recursive "${MODULES[@]}"
  fi

  local module
  for module in "${MODULES[@]}"; do
    if [[ ! -d "$REPO_ROOT/$module/src" ]]; then
      echo "Missing initialized submodule: $module" >&2
      echo "Run: git submodule update --init --recursive ${MODULES[*]}" >&2
      exit 1
    fi
  done
}

write_manifest() {
  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "[DRY RUN] write path-feedback manifest: $MANIFEST_PATH"
    return
  fi

  "$PYTHON_BIN" - "$SCENARIO_CONFIG" "$EXPORT_DIR" "$MANIFEST_PATH" "$SUMMARY_PATH" "$REPORT_PATH" "$TOP_K" "$SCENARIO_SET" "$DIAGNOSTIC_PROFILE" "$ACCEPTANCE_GATE" "$PATH_PLANNER_ROOT" "$PYTHON_BIN" "$ANCHOR_PROJECTION_CANDIDATE_GENERATION" "$ANCHOR_PROJECTION_SELECTION_PATH_COST_BONUS" "$ANCHOR_PROJECTION_MAX_SELECTION_PATH_COST_REGRESSION" "$ANCHOR_PROJECTION_MAX_SELECTION_RISK_REGRESSION" "$ANCHOR_PROJECTION_CONTRACT_AWARE_TRAINABLE_TARGET_GENERATION" "$ANCHOR_PROJECTION_PREFER_CONTRACT_SAFE_TRAINABLE_TARGETS" "$ANCHOR_PROJECTION_MAX_TRAINABLE_DISTANCE_CELLS" "$ANCHOR_PROJECTION_MAX_TRAINABLE_DISTANCE_M" "$ANCHOR_PROJECTION_PLANNER_VALIDATED_TRAINABLE_TARGET_MINING" "$ANCHOR_PROJECTION_ALLOW_PLANNER_VALIDATED_DISTANCE_EXCEPTION" "$ANCHOR_PROJECTION_MAX_PLANNER_VALIDATED_DISTANCE_CELLS" "$ANCHOR_PROJECTION_MAX_PLANNER_VALIDATED_DISTANCE_M" "${PLANNER_EXTRA_ARGS[@]}" <<'PY'
import json
import sys
from pathlib import Path

scenario_config = Path(sys.argv[1])
export_dir = Path(sys.argv[2])
manifest_path = Path(sys.argv[3])
summary_path = Path(sys.argv[4])
report_path = Path(sys.argv[5])
top_k = int(sys.argv[6])
scenario_set = sys.argv[7]
diagnostic_profile = sys.argv[8]
acceptance_gate = sys.argv[9]
path_planner_root = Path(sys.argv[10])
python_executable = sys.argv[11]
anchor_projection_candidate_generation = sys.argv[12] == "1"
anchor_projection_selection_path_cost_bonus = float(sys.argv[13])
anchor_projection_max_selection_path_cost_regression = float(sys.argv[14])
anchor_projection_max_selection_risk_regression = float(sys.argv[15])
anchor_projection_contract_aware_trainable_target_generation = sys.argv[16] == "1"
anchor_projection_prefer_contract_safe_trainable_targets = sys.argv[17] == "1"
anchor_projection_max_trainable_distance_cells = int(float(sys.argv[18]))
anchor_projection_max_trainable_distance_m = float(sys.argv[19])
anchor_projection_planner_validated_trainable_target_mining = sys.argv[20] == "1"
anchor_projection_allow_planner_validated_distance_exception = sys.argv[21] == "1"
anchor_projection_max_planner_validated_distance_cells = int(float(sys.argv[22]))
anchor_projection_max_planner_validated_distance_m = float(sys.argv[23])
extra_args = sys.argv[24:]

payload = json.loads(scenario_config.read_text(encoding="utf-8"))
scenarios = []
for item in payload["scenarios"]:
    scenario_id = item["scenario_id"]
    scenarios.append(
        {
            "scenario_id": scenario_id,
            "scenario_group": item.get("scenario_group", "unknown"),
            "scenario_seed": item.get("seed"),
            "scenario_variant_id": item.get("scenario_variant_id"),
            "contract": str(export_dir / f"{scenario_id}.contract.json"),
            "sidecar": str(export_dir / f"{scenario_id}.path-planner-sidecar.json"),
            "current_cell": item["start_cell"],
        }
    )

manifest = {
    "schema_version": "path-feedback-manifest/v1",
    "scenario_set": scenario_set,
    "diagnostic_profile": diagnostic_profile,
    "acceptance_gate": acceptance_gate,
    "top_k": top_k,
    "planner_extra_args": extra_args,
    "acceptance_metadata": {
        "schema_version": "path-feedback-acceptance-metadata/v1",
        "scenario_set": scenario_set,
        "diagnostic_profile": diagnostic_profile,
        "acceptance_gate": acceptance_gate,
        "top_k": top_k,
        "python_executable": python_executable,
        "planner_extra_args": extra_args,
        "anchor_projection_candidate_generation_enabled": anchor_projection_candidate_generation,
        "anchor_projection_selection_path_cost_bonus": anchor_projection_selection_path_cost_bonus,
        "anchor_projection_max_selection_path_cost_regression": anchor_projection_max_selection_path_cost_regression,
        "anchor_projection_max_selection_risk_regression": anchor_projection_max_selection_risk_regression,
        "anchor_projection_contract_aware_trainable_target_generation": anchor_projection_contract_aware_trainable_target_generation,
        "anchor_projection_prefer_contract_safe_trainable_targets": anchor_projection_prefer_contract_safe_trainable_targets,
        "anchor_projection_max_trainable_distance_cells": anchor_projection_max_trainable_distance_cells,
        "anchor_projection_max_trainable_distance_m": anchor_projection_max_trainable_distance_m,
        "anchor_projection_planner_validated_trainable_target_mining": anchor_projection_planner_validated_trainable_target_mining,
        "anchor_projection_allow_planner_validated_distance_exception": anchor_projection_allow_planner_validated_distance_exception,
        "anchor_projection_max_planner_validated_distance_cells": anchor_projection_max_planner_validated_distance_cells,
        "anchor_projection_max_planner_validated_distance_m": anchor_projection_max_planner_validated_distance_m,
        "open_grid_fallback_used": None,
        "open_grid_fallback_used_gate": {
            "status": "pending",
            "expected": False,
            "actual": None,
            "reason_codes": ["open_grid_fallback_gate_pending"],
        },
    },
    "open_grid_fallback_used_gate": {
        "status": "pending",
        "expected": False,
        "actual": None,
        "reason_codes": ["open_grid_fallback_gate_pending"],
    },
    "planner": {
        "backend": "path_planner_route",
        "path_planner_root": str(path_planner_root),
        "python_executable": python_executable,
    },
    "scenarios": scenarios,
    "outputs": {
        "summary": str(summary_path),
        "report": str(report_path),
    },
}
if extra_args:
    manifest["planner"]["extra_args"] = extra_args
if anchor_projection_candidate_generation:
    manifest["planner"]["anchor_projection_candidate_generation"] = {
        "enabled": True,
        "require_anchor_reachable": True,
        "source_selection_path_cost_bonus": anchor_projection_selection_path_cost_bonus,
        "max_source_selection_path_cost_regression": anchor_projection_max_selection_path_cost_regression,
        "max_source_selection_risk_regression": anchor_projection_max_selection_risk_regression,
        "contract_aware_trainable_target_generation": anchor_projection_contract_aware_trainable_target_generation,
        "prefer_contract_safe_trainable_targets": anchor_projection_prefer_contract_safe_trainable_targets,
        "max_trainable_projection_distance_cells": anchor_projection_max_trainable_distance_cells,
        "max_trainable_projection_distance_m": anchor_projection_max_trainable_distance_m,
        "planner_validated_trainable_target_mining": anchor_projection_planner_validated_trainable_target_mining,
        "allow_planner_validated_distance_exception": anchor_projection_allow_planner_validated_distance_exception,
        "max_planner_validated_distance_cells": anchor_projection_max_planner_validated_distance_cells,
        "max_planner_validated_distance_m": anchor_projection_max_planner_validated_distance_m,
    }
manifest_path.parent.mkdir(parents=True, exist_ok=True)
manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps({"manifest": str(manifest_path), "scenario_count": len(scenarios)}, ensure_ascii=False))
PY
}

assert_output_files() {
  local path
  for path in "$MANIFEST_PATH" "$SUMMARY_PATH" "$REPORT_PATH"; do
    if [[ ! -s "$path" ]]; then
      echo "Expected non-empty output file was not created: $path" >&2
      exit 1
    fi
  done
}

validate_summary() {
  "$PYTHON_BIN" - "$SUMMARY_PATH" "$MANIFEST_PATH" "$SCENARIO_CONFIG" "$SCENARIO_SET" "$DIAGNOSTIC_PROFILE" "$TOP_K" "$ACCEPTANCE_GATE" "$PYTHON_BIN" "${PLANNER_EXTRA_ARGS[@]}" <<'PY'
import json
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
manifest_path = Path(sys.argv[2])
scenario_config_path = Path(sys.argv[3])
scenario_set = sys.argv[4]
diagnostic_profile = sys.argv[5]
top_k = int(sys.argv[6])
acceptance_gate = sys.argv[7]
python_executable = sys.argv[8]
planner_extra_args = sys.argv[9:]
summary = json.loads(summary_path.read_text(encoding="utf-8"))
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
scenario_config = json.loads(scenario_config_path.read_text(encoding="utf-8"))
expected_ids = {item["scenario_id"] for item in scenario_config["scenarios"]}
expected_count = len(expected_ids)

required_values = {
    "schema_version": "path-feedback-summary/v1",
    "scenario_count": expected_count,
    "open_grid_fallback_used": False,
}
for key, expected in required_values.items():
    actual = summary.get(key)
    if actual != expected:
        raise SystemExit(f"{summary_path}: expected {key}={expected!r}, got {actual!r}")

expected_metadata = {
    "scenario_set": scenario_set,
    "diagnostic_profile": diagnostic_profile,
    "acceptance_gate": acceptance_gate,
    "top_k": top_k,
    "planner_extra_args": planner_extra_args,
}
for key, expected in expected_metadata.items():
    actual = summary.get(key)
    if actual != expected:
        raise SystemExit(f"{summary_path}: expected {key}={expected!r}, got {actual!r}")
    manifest_actual = manifest.get(key)
    if manifest_actual != expected:
        raise SystemExit(f"{manifest_path}: expected {key}={expected!r}, got {manifest_actual!r}")

planner = manifest.get("planner")
if not isinstance(planner, dict):
    raise SystemExit(f"{manifest_path}: planner must be an object")
if planner.get("python_executable") != python_executable:
    raise SystemExit(
        f"{manifest_path}: expected planner.python_executable={python_executable!r}, "
        f"got {planner.get('python_executable')!r}"
    )

acceptance_metadata = summary.get("acceptance_metadata")
if not isinstance(acceptance_metadata, dict):
    raise SystemExit(f"{summary_path}: acceptance_metadata must be an object")
for key, expected in expected_metadata.items():
    actual = acceptance_metadata.get(key)
    if actual != expected:
        raise SystemExit(f"{summary_path}: expected acceptance_metadata.{key}={expected!r}, got {actual!r}")
if acceptance_metadata.get("python_executable") != python_executable:
    raise SystemExit(
        f"{summary_path}: expected acceptance_metadata.python_executable={python_executable!r}, "
        f"got {acceptance_metadata.get('python_executable')!r}"
    )
open_grid_gate = acceptance_metadata.get("open_grid_fallback_used_gate")
if not isinstance(open_grid_gate, dict) or open_grid_gate.get("status") != "passed":
    raise SystemExit(f"{summary_path}: open_grid_fallback_used_gate must pass")
if summary.get("open_grid_fallback_used_gate") != open_grid_gate:
    raise SystemExit(f"{summary_path}: top-level open_grid_fallback_used_gate must mirror acceptance metadata")

manifest["open_grid_fallback_used_gate"] = dict(open_grid_gate)
manifest_metadata = manifest.get("acceptance_metadata")
manifest_metadata = manifest_metadata if isinstance(manifest_metadata, dict) else {}
manifest_metadata.update(
    {
        "open_grid_fallback_used": summary.get("open_grid_fallback_used"),
        "open_grid_fallback_used_gate": dict(open_grid_gate),
    }
)
manifest["acceptance_metadata"] = manifest_metadata
manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

candidate_count = summary.get("candidate_count")
if not isinstance(candidate_count, int) or candidate_count < 3:
    raise SystemExit(f"{summary_path}: expected candidate_count >= 3, got {candidate_count!r}")

required_keys = (
    "coverage_per_path_cost",
    "path_planning_failure_count",
    "replan_count",
    "tracking_safety_violation_count",
    "trajectory_optimization_fallback_count",
    "region_graph_disconnected_count",
    "iris_requested_count",
    "iris_report_count",
    "iris_status_counts",
    "iris_fallback_count",
    "iris_failure_count",
    "iris_region_count_total",
    "iris_fallback_reasons",
    "region_graph_source_counts",
    "region_graph_fallback_count",
    "region_graph_fallback_reasons",
    "region_graph_start_goal_disconnected_count",
    "scenario_group_summary",
    "diagnostic_interpretation",
    "selection_changed_count",
    "selection_changed_rate",
)
missing = [key for key in required_keys if key not in summary]
if missing:
    raise SystemExit(f"{summary_path}: missing required summary keys: {', '.join(missing)}")

diagnostic_interpretation = summary.get("diagnostic_interpretation")
if not isinstance(diagnostic_interpretation, dict):
    raise SystemExit(f"{summary_path}: diagnostic_interpretation must be an object")
if not isinstance(diagnostic_interpretation.get("scenario_group_interpretation"), dict):
    raise SystemExit(f"{summary_path}: diagnostic_interpretation.scenario_group_interpretation must be an object")

scenario_ids = {item.get("scenario_id") for item in summary.get("scenarios", [])}
if scenario_ids != expected_ids:
    raise SystemExit(f"{summary_path}: expected scenarios {sorted(expected_ids)}, got {sorted(scenario_ids)}")

groups_by_id = {
    item["scenario_id"]: item.get("scenario_group", "unknown")
    for item in scenario_config["scenarios"]
}
metadata_by_id = {
    item["scenario_id"]: {
        "scenario_seed": item.get("seed"),
        "scenario_variant_id": item.get("scenario_variant_id"),
    }
    for item in scenario_config["scenarios"]
}
for item in summary.get("scenarios", []):
    expected_group = groups_by_id.get(item.get("scenario_id"), "unknown")
    if item.get("scenario_group") != expected_group:
        raise SystemExit(
            f"{summary_path}: expected {item.get('scenario_id')} scenario_group={expected_group!r}, "
            f"got {item.get('scenario_group')!r}"
        )
    expected_metadata = metadata_by_id.get(item.get("scenario_id"), {})
    for metadata_key, expected_value in expected_metadata.items():
        if item.get(metadata_key) != expected_value:
            raise SystemExit(
                f"{summary_path}: expected {item.get('scenario_id')} {metadata_key}={expected_value!r}, "
                f"got {item.get(metadata_key)!r}"
            )
    scenario_interpretation = item.get("diagnostic_interpretation")
    if not isinstance(scenario_interpretation, dict):
        raise SystemExit(f"{summary_path}: {item.get('scenario_id')} missing diagnostic_interpretation")
    if "target_replacement_reason" not in scenario_interpretation:
        raise SystemExit(f"{summary_path}: {item.get('scenario_id')} missing target_replacement_reason")
    for candidate in item.get("path_feedback", {}).get("candidates", []):
        if not candidate.get("context_id"):
            raise SystemExit(
                f"{summary_path}: {item.get('scenario_id')} candidate {candidate.get('action_index')} "
                "missing policy context_id"
            )
        if candidate.get("legacy_identity_fallback_used") is not False:
            raise SystemExit(
                f"{summary_path}: {item.get('scenario_id')} candidate {candidate.get('action_index')} "
                "must not use legacy identity fallback"
            )
        candidate_interpretation = candidate.get("diagnostic_interpretation")
        if not isinstance(candidate_interpretation, dict):
            raise SystemExit(
                f"{summary_path}: {item.get('scenario_id')} candidate {candidate.get('action_index')} "
                "missing diagnostic_interpretation"
            )

if scenario_set in {"stress", "all"}:
    stress_items = [
        item
        for item in summary.get("scenarios", [])
        if str(item.get("scenario_id", "")).startswith("npz_")
        and item.get("scenario_id") not in {
            "npz_shadow_corridor",
            "npz_rock_field_multi_pose",
            "npz_low_confidence_risk_band",
        }
    ]
    stress_replan_or_failure = sum(
        int(item.get("path_feedback", {}).get("failure_count", 0))
        + int(item.get("path_feedback", {}).get("replan_count", 0))
        for item in stress_items
    )
    stress_sampled_region_decision_diagnostics = sum(
        int(item.get("sampled_region_path_diagnostics", {}).get("selected_count", 0))
        + int(item.get("sampled_region_path_diagnostics", {}).get("fallback_count", 0))
        + int(item.get("sampled_region_path_diagnostics", {}).get("terminal_adjusted_count", 0))
        + int(item.get("sampled_region_path_diagnostics", {}).get("reachable_terminal_rescue_count", 0))
        + int(item.get("sampled_region_path_diagnostics", {}).get("proxy_goal_anchor_selected_count", 0))
        for item in stress_items
    )
    if stress_sampled_region_decision_diagnostics < 1:
        group_summary = summary.get("scenario_group_summary", {})
        group_summary = group_summary if isinstance(group_summary, dict) else {}
        for group_name in ("stress", "mixed_stress"):
            group_payload = group_summary.get(group_name, {})
            if not isinstance(group_payload, dict):
                continue
            stress_sampled_region_decision_diagnostics += (
                int(group_payload.get("sampled_region_path_selected_count", 0))
                + int(group_payload.get("sampled_region_path_fallback_count", 0))
                + int(group_payload.get("sampled_region_path_terminal_adjusted_count", 0))
                + int(group_payload.get("sampled_region_path_reachable_terminal_rescue_count", 0))
                + int(group_payload.get("sampled_region_path_proxy_goal_anchor_selected_count", 0))
            )
    if stress_replan_or_failure + stress_sampled_region_decision_diagnostics < 1:
        raise SystemExit(
            f"{summary_path}: stress scenarios must produce failure, replan, or sampled-region diagnostics"
        )
    mixed_items = [
        item
        for item in summary.get("scenarios", [])
        if item.get("scenario_group") == "mixed_stress"
    ]
    if mixed_items:
        mixed_reachable = sum(int(item.get("path_feedback", {}).get("reachable_count", 0)) for item in mixed_items)
        mixed_replan_or_failure = sum(
            int(item.get("path_feedback", {}).get("failure_count", 0))
            + int(item.get("path_feedback", {}).get("replan_count", 0))
            for item in mixed_items
        )
        mixed_group_summary = summary.get("scenario_group_summary", {}).get("mixed_stress", {})
        if not isinstance(mixed_group_summary, dict):
            mixed_group_summary = {}
        mixed_sampled_region_decision_diagnostics = int(
            mixed_group_summary.get("sampled_region_path_selected_count", 0)
        ) + int(mixed_group_summary.get("sampled_region_path_terminal_adjusted_count", 0))
        if mixed_sampled_region_decision_diagnostics < 1:
            mixed_sampled_region_decision_diagnostics = sum(
                int(item.get("sampled_region_path_diagnostics", {}).get("selected_count", 0))
                + int(item.get("sampled_region_path_diagnostics", {}).get("terminal_adjusted_count", 0))
                for item in mixed_items
            )
        if mixed_reachable < 1:
            raise SystemExit(f"{summary_path}: mixed stress scenarios must include at least one reachable candidate")
        if mixed_replan_or_failure + mixed_sampled_region_decision_diagnostics < 1:
            raise SystemExit(
                f"{summary_path}: mixed stress scenarios must produce failure, replan, "
                "or sampled-region decision diagnostics"
            )

print(
    json.dumps(
        {
            "status": "valid",
            "summary": str(summary_path),
            "scenario_count": summary["scenario_count"],
            "candidate_count": summary["candidate_count"],
            "selection_changed_count": summary["selection_changed_count"],
            "open_grid_fallback_used": summary["open_grid_fallback_used"],
            "acceptance_gate": summary["acceptance_gate"],
            "scenario_set": summary["scenario_set"],
            "diagnostic_profile": summary["diagnostic_profile"],
        },
        ensure_ascii=False,
    )
)
PY
}

cat <<INFO
Repository: $REPO_ROOT
Output root: $OUTPUT_ROOT
Python executable: $PYTHON_BIN
Acceptance gate: $ACCEPTANCE_GATE
Top-K: $TOP_K
Scenario set: $SCENARIO_SET
Diagnostic profile: $DIAGNOSTIC_PROFILE
Planner extra args: ${PLANNER_EXTRA_ARGS[*]:-(none)}
Anchor projection candidate generation: $([[ "$ANCHOR_PROJECTION_CANDIDATE_GENERATION" -eq 1 ]] && echo enabled || echo disabled)
Anchor projection selection path-cost bonus: $ANCHOR_PROJECTION_SELECTION_PATH_COST_BONUS
Anchor projection max selection path-cost regression: $ANCHOR_PROJECTION_MAX_SELECTION_PATH_COST_REGRESSION
Anchor projection max selection risk regression: $ANCHOR_PROJECTION_MAX_SELECTION_RISK_REGRESSION
Anchor projection contract-aware trainable target generation: $([[ "$ANCHOR_PROJECTION_CONTRACT_AWARE_TRAINABLE_TARGET_GENERATION" -eq 1 ]] && echo enabled || echo disabled)
Anchor projection prefer contract-safe trainable targets: $([[ "$ANCHOR_PROJECTION_PREFER_CONTRACT_SAFE_TRAINABLE_TARGETS" -eq 1 ]] && echo enabled || echo disabled)
Anchor projection max trainable distance cells: $ANCHOR_PROJECTION_MAX_TRAINABLE_DISTANCE_CELLS
Anchor projection max trainable distance m: $ANCHOR_PROJECTION_MAX_TRAINABLE_DISTANCE_M
INFO

ensure_submodules

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "[DRY RUN] mkdir -p $OUTPUT_ROOT"
else
  mkdir -p "$OUTPUT_ROOT"
fi

run_cmd "$DEV_ROOT" \
  "$PYTHON_BIN" scripts/generate_npz_validation_maps.py \
  --scenario-set "$SCENARIO_SET" \
  --output-dir "$MAP_DIR" \
  --scenario-config "$SCENARIO_CONFIG"

run_pythonpath_cmd "$DEV_ROOT" \
  "$PYTHON_BIN" scripts/export_path_planner_sidecars.py \
  --scenario-config "$SCENARIO_CONFIG" \
  --output-dir "$EXPORT_DIR" \
  --top-k "$TOP_K"

write_manifest

run_pythonpath_cmd "$MODEL_ROOT" \
  "$PYTHON_BIN" -m model_explorer path-feedback validate "$MANIFEST_PATH"

run_pythonpath_cmd "$MODEL_ROOT" \
  "$PYTHON_BIN" -m model_explorer path-feedback run "$MANIFEST_PATH"

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "[DRY RUN] validate summary gates: $SUMMARY_PATH"
else
  assert_output_files
  validate_summary
fi
