#!/usr/bin/env bash
set -euo pipefail

DRY_RUN=0
OUTPUT_ROOT="outputs/path_feedback_validation"
TOP_K=3
SCENARIO_SET="smoke"
DIAGNOSTIC_PROFILE="baseline"
PLANNER_EXTRA_ARGS=()
MODULES=(path-planner model-explorer dev-platform-constraints)

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

  python3 - "$SCENARIO_CONFIG" "$EXPORT_DIR" "$MANIFEST_PATH" "$SUMMARY_PATH" "$REPORT_PATH" "$TOP_K" "$PATH_PLANNER_ROOT" "${PLANNER_EXTRA_ARGS[@]}" <<'PY'
import json
import sys
from pathlib import Path

scenario_config = Path(sys.argv[1])
export_dir = Path(sys.argv[2])
manifest_path = Path(sys.argv[3])
summary_path = Path(sys.argv[4])
report_path = Path(sys.argv[5])
top_k = int(sys.argv[6])
path_planner_root = Path(sys.argv[7])
extra_args = sys.argv[8:]

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
    "top_k": top_k,
    "planner": {
        "backend": "path_planner_route",
        "path_planner_root": str(path_planner_root),
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
  python3 - "$SUMMARY_PATH" "$SCENARIO_CONFIG" "$SCENARIO_SET" <<'PY'
import json
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
scenario_config_path = Path(sys.argv[2])
scenario_set = sys.argv[3]
summary = json.loads(summary_path.read_text(encoding="utf-8"))
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
    "region_graph_source_counts",
    "region_graph_fallback_count",
    "region_graph_start_goal_disconnected_count",
    "scenario_group_summary",
    "selection_changed_count",
    "selection_changed_rate",
)
missing = [key for key in required_keys if key not in summary]
if missing:
    raise SystemExit(f"{summary_path}: missing required summary keys: {', '.join(missing)}")

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
        },
        ensure_ascii=False,
    )
)
PY
}

cat <<INFO
Repository: $REPO_ROOT
Output root: $OUTPUT_ROOT
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
  python3 scripts/generate_npz_validation_maps.py \
  --scenario-set "$SCENARIO_SET" \
  --output-dir "$MAP_DIR" \
  --scenario-config "$SCENARIO_CONFIG"

run_pythonpath_cmd "$DEV_ROOT" \
  python3 scripts/export_path_planner_sidecars.py \
  --scenario-config "$SCENARIO_CONFIG" \
  --output-dir "$EXPORT_DIR" \
  --top-k "$TOP_K"

write_manifest

run_pythonpath_cmd "$MODEL_ROOT" \
  python3 -m model_explorer path-feedback validate "$MANIFEST_PATH"

run_pythonpath_cmd "$MODEL_ROOT" \
  python3 -m model_explorer path-feedback run "$MANIFEST_PATH"

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "[DRY RUN] validate summary gates: $SUMMARY_PATH"
else
  assert_output_files
  validate_summary
fi
