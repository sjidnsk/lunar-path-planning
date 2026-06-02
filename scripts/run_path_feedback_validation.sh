#!/usr/bin/env bash
set -euo pipefail

DRY_RUN=0
OUTPUT_ROOT="outputs/path_feedback_validation"
TOP_K=3
SCENARIO_SET="smoke"
DIAGNOSTIC_PROFILE="baseline"
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
  --scenario-set NAME   Validation scenario set: smoke, stress, or all. Default: smoke
  --diagnostic-profile NAME
                        Diagnostic profile: baseline, execution, iris, or all.
                        Default: baseline
  --simulate-tracking  Forward path-planner tracking simulation diagnostics.
  --optimize-trajectory
                        Forward fixed-corridor trajectory optimization diagnostics.
  --drake-iris-regions Forward optional Drake workspace Iris diagnostics.
  --planning-backend NAME
                        Forward path-planner planning backend: astar or region_graph_guided.
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
    --simulate-tracking|--optimize-trajectory|--drake-iris-regions)
      PLANNER_EXTRA_ARGS+=("$1")
      shift
      ;;
    --planning-backend)
      require_value "$1" "${2:-}"
      case "$2" in
        astar|region_graph_guided)
          ;;
        *)
          echo "--planning-backend must be one of: astar, region_graph_guided" >&2
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

case "$SCENARIO_SET" in
  smoke|stress|all)
    ;;
  *)
    echo "--scenario-set must be one of: smoke, stress, all" >&2
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

case "$DIAGNOSTIC_PROFILE" in
  execution)
    append_planner_arg "--simulate-tracking"
    append_planner_arg "--optimize-trajectory"
    ;;
  iris)
    append_planner_arg "--drake-iris-regions"
    ;;
  all)
    append_planner_arg "--simulate-tracking"
    append_planner_arg "--optimize-trajectory"
    append_planner_arg "--drake-iris-regions"
    ;;
esac

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

  "$PYTHON_BIN" - "$SCENARIO_CONFIG" "$EXPORT_DIR" "$MANIFEST_PATH" "$SUMMARY_PATH" "$REPORT_PATH" "$TOP_K" "$SCENARIO_SET" "$DIAGNOSTIC_PROFILE" "$ACCEPTANCE_GATE" "$PATH_PLANNER_ROOT" "$PYTHON_BIN" "${PLANNER_EXTRA_ARGS[@]}" <<'PY'
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
extra_args = sys.argv[12:]

payload = json.loads(scenario_config.read_text(encoding="utf-8"))
scenarios = []
for item in payload["scenarios"]:
    scenario_id = item["scenario_id"]
    scenarios.append(
        {
            "scenario_id": scenario_id,
            "scenario_group": item.get("scenario_group", "unknown"),
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
for item in summary.get("scenarios", []):
    expected_group = groups_by_id.get(item.get("scenario_id"), "unknown")
    if item.get("scenario_group") != expected_group:
        raise SystemExit(
            f"{summary_path}: expected {item.get('scenario_id')} scenario_group={expected_group!r}, "
            f"got {item.get('scenario_group')!r}"
        )
    scenario_interpretation = item.get("diagnostic_interpretation")
    if not isinstance(scenario_interpretation, dict):
        raise SystemExit(f"{summary_path}: {item.get('scenario_id')} missing diagnostic_interpretation")
    if "target_replacement_reason" not in scenario_interpretation:
        raise SystemExit(f"{summary_path}: {item.get('scenario_id')} missing target_replacement_reason")
    for candidate in item.get("path_feedback", {}).get("candidates", []):
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
    if stress_replan_or_failure < 1:
        raise SystemExit(f"{summary_path}: stress scenarios must produce at least one failure or replan diagnostic")
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
        if mixed_reachable < 1:
            raise SystemExit(f"{summary_path}: mixed stress scenarios must include at least one reachable candidate")
        if mixed_replan_or_failure < 1:
            raise SystemExit(f"{summary_path}: mixed stress scenarios must produce failure or replan diagnostics")

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
