from __future__ import annotations

import argparse
import json
import sys
from math import isfinite
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:
    from frontier_coverage_planner_common import enumerate_frontier_candidates, frontier_cells
    from git_provenance import git_snapshot
    from global_99_coverage_contract import (
        Cell,
        ConfigError,
        bfs_tree,
        cell_from_config,
        coverage_footprint,
        load_global_99_config,
        neighbors4,
        resolve_path,
        scenario_geometry,
        unique_sorted,
        utc_now,
        write_json,
        write_jsonl,
    )
    from global_99_governance_common import global_99_boundary_defaults
except ModuleNotFoundError:  # pragma: no cover
    from scripts.frontier_coverage_planner_common import enumerate_frontier_candidates, frontier_cells
    from scripts.git_provenance import git_snapshot
    from scripts.global_99_coverage_contract import (
        Cell,
        ConfigError,
        bfs_tree,
        cell_from_config,
        coverage_footprint,
        load_global_99_config,
        neighbors4,
        resolve_path,
        scenario_geometry,
        unique_sorted,
        utc_now,
        write_json,
        write_jsonl,
    )
    from scripts.global_99_governance_common import global_99_boundary_defaults


CONFIG_SCHEMA_VERSION = "xunce-topology-feature-extraction-audit-config/v1"
SUMMARY_SCHEMA_VERSION = "xunce-topology-feature-extraction-audit-summary/v1"
MANIFEST_SCHEMA_VERSION = "xunce-topology-feature-extraction-audit-manifest/v1"
CANDIDATE_ROW_SCHEMA_VERSION = "xunce-topology-feature-extraction-candidate-row/v1"
EDGE_ROW_SCHEMA_VERSION = "xunce-topology-feature-extraction-edge-row/v1"
MEMORY_SCHEMA_VERSION = "xunce-topology-feature-extraction-memory/v1"
FIELD_AUDIT_SCHEMA_VERSION = "xunce-topology-feature-extraction-field-audit/v1"
DETERMINISM_AUDIT_SCHEMA_VERSION = "xunce-topology-feature-extraction-determinism-audit/v1"
BOUNDARY_AUDIT_SCHEMA_VERSION = "xunce-topology-feature-extraction-boundary-audit/v1"
REJECTION_REPORT_SCHEMA_VERSION = "xunce-topology-feature-extraction-rejection-report/v1"

DEFAULT_CONFIG = "configs/xunce_topology_feature_extraction_audit_v1.json"
DEFAULT_OUTPUT_ROOT = "outputs/path_feedback_batch_xunce_topology_feature_extraction_audit_v1"

SUMMARY_FILE = "xunce-topology-feature-extraction-audit-summary.json"
MANIFEST_FILE = "xunce-topology-feature-extraction-audit-manifest.json"
CANDIDATES_FILE = "xunce-topology-feature-extraction-candidates.jsonl"
EDGES_FILE = "xunce-topology-feature-extraction-edges.jsonl"
MEMORY_FILE = "xunce-topology-feature-extraction-memory.json"
FIELD_AUDIT_FILE = "xunce-topology-feature-extraction-field-audit.json"
DETERMINISM_AUDIT_FILE = "xunce-topology-feature-extraction-determinism-audit.json"
BOUNDARY_AUDIT_FILE = "xunce-topology-feature-extraction-boundary-audit.json"
REJECTION_REPORT_FILE = "xunce-topology-feature-extraction-rejection-report.json"
REPORT_FILE = "xunce-topology-feature-extraction-audit-report.md"

PASS_NEXT_REQUIRED_CHANGE = "topology_aware_coverage_graph_proto"
FIX_STAGE3_NEXT_REQUIRED_CHANGE = "fix_xunce_topology_observation_contract"
FAIL_NEXT_REQUIRED_CHANGE = "fix_xunce_topology_feature_extraction_audit"

BOUNDARY_FIELDS = tuple(global_99_boundary_defaults()) + (
    "real_world_release_approved",
    "real_world_performance_claimed",
    "default_policy_replacement_approved",
    "real_executor_connection_approved",
    "starts_online_canary",
    "publishes_checkpoint",
    "replaces_default_policy",
    "connects_real_executor",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Xunce Topology Feature Extraction Audit v1.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--repo-root")
    args = parser.parse_args(argv)

    repo_root = Path(args.repo_root).resolve() if args.repo_root else Path(__file__).resolve().parents[1]
    try:
        summary = run_xunce_topology_feature_extraction_audit(
            config_path=resolve_path(Path(args.config), repo_root),
            output_root=resolve_path(Path(args.output_root), repo_root),
            repo_root=repo_root,
        )
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2

    print(
        json.dumps(
            {
                "status": summary["status"],
                "reason_codes": summary["reason_codes"],
                "candidate_count": summary["candidate_count"],
                "edge_count": summary["edge_count"],
                "next_required_change": summary["next_required_change"],
                "summary": summary["summary"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if summary["status"] == "passed" else 1


def run_xunce_topology_feature_extraction_audit(
    *,
    config_path: Path,
    output_root: Path,
    repo_root: Path,
) -> dict[str, Any]:
    repo_root = Path(repo_root)
    config_path = Path(config_path)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    config = _load_config(config_path, repo_root)
    paths = _artifact_paths(output_root)
    stage3_root = resolve_path(Path(config["source_topology_observation_contract_root"]), repo_root)
    stage3_summary_path = stage3_root / "xunce-topology-observation-contract-summary.json"
    stage3_contract_path = stage3_root / "xunce-topology-observation-contract.json"
    stage3_summary = _load_json(stage3_summary_path)
    stage3_contract = _load_json(stage3_contract_path)
    boundary_audit = _boundary_audit(stage3_summary)

    extraction_result = _empty_extraction_result()
    extraction_error_reasons: list[str] = []
    if _stage3_ready(stage3_summary):
        if not isinstance(stage3_contract, dict):
            extraction_error_reasons.append("missing_topology_observation_contract")
        else:
            extraction_result = _extract_topology_features(config, stage3_contract)
            extraction_error_reasons.extend(extraction_result["reason_codes"])

    field_audit = _field_audit(stage3_contract, extraction_result)
    determinism_audit = _determinism_audit(extraction_result)
    decision = _decision(
        stage3_summary=stage3_summary,
        stage3_contract=stage3_contract,
        boundary_audit=boundary_audit,
        field_audit=field_audit,
        determinism_audit=determinism_audit,
        extraction_error_reasons=extraction_error_reasons,
    )
    generated_at = utc_now()
    summary = _summary(
        generated_at=generated_at,
        config_path=config_path,
        output_root=output_root,
        paths=paths,
        stage3_summary_path=stage3_summary_path,
        stage3_contract_path=stage3_contract_path,
        extraction_result=extraction_result,
        field_audit=field_audit,
        determinism_audit=determinism_audit,
        boundary_audit=boundary_audit,
        decision=decision,
        repo_root=repo_root,
    )
    manifest = _manifest(generated_at, config_path, output_root, paths, summary)
    rejection_report = _rejection_report(decision, field_audit, determinism_audit, boundary_audit)

    write_jsonl(paths["candidates"], extraction_result["candidate_rows"])
    write_jsonl(paths["edges"], extraction_result["edge_rows"])
    write_json(paths["memory"], extraction_result["memory"])
    write_json(paths["field_audit"], field_audit)
    write_json(paths["determinism_audit"], determinism_audit)
    write_json(paths["boundary_audit"], boundary_audit)
    write_json(paths["rejection_report"], rejection_report)
    write_json(paths["manifest"], manifest)
    write_json(paths["summary"], summary)
    paths["report"].write_text(_render_report(summary, field_audit, rejection_report), encoding="utf-8")
    return summary


def _load_config(path: Path, repo_root: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"config file does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigError(f"config JSON is invalid: {exc}") from exc
    if not isinstance(payload, dict):
        raise ConfigError("config root must be an object")
    if payload.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ConfigError(f"schema_version must be {CONFIG_SCHEMA_VERSION!r}")
    if not isinstance(payload.get("source_topology_observation_contract_root"), str):
        raise ConfigError("source_topology_observation_contract_root must be a string")
    if not isinstance(payload.get("scenario"), dict):
        raise ConfigError("scenario must be an object")
    scenario_wrapper = {
        "schema_version": "global-99-coverage-benchmark-config/v1",
        "target_coverage_rate": _float_value(payload.get("target_coverage_rate"), "target_coverage_rate"),
        "path_budget_m": _float_value(payload.get("path_budget_m"), "path_budget_m"),
        "scenarios": [payload["scenario"]],
    }
    tmp_config_path = repo_root / ".tmp-xunce-feature-validation.json"
    # Use the existing validator without leaving a file behind.
    try:
        tmp_config_path.write_text(json.dumps(scenario_wrapper), encoding="utf-8")
        load_global_99_config(tmp_config_path)
    finally:
        if tmp_config_path.exists():
            tmp_config_path.unlink()
    normalized = dict(payload)
    for key in ("target_coverage_rate", "path_budget_m", "revisit_penalty_weight", "new_coverage_weight"):
        normalized[key] = _float_value(payload.get(key), key)
    normalized["coverage_radius_cells"] = _int_value(payload.get("coverage_radius_cells"), "coverage_radius_cells")
    if normalized["coverage_radius_cells"] < 0:
        raise ConfigError("coverage_radius_cells must be >= 0")
    return normalized


def _artifact_paths(output_root: Path) -> dict[str, Path]:
    return {
        "summary": output_root / SUMMARY_FILE,
        "manifest": output_root / MANIFEST_FILE,
        "candidates": output_root / CANDIDATES_FILE,
        "edges": output_root / EDGES_FILE,
        "memory": output_root / MEMORY_FILE,
        "field_audit": output_root / FIELD_AUDIT_FILE,
        "determinism_audit": output_root / DETERMINISM_AUDIT_FILE,
        "boundary_audit": output_root / BOUNDARY_AUDIT_FILE,
        "rejection_report": output_root / REJECTION_REPORT_FILE,
        "report": output_root / REPORT_FILE,
    }


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _empty_extraction_result() -> dict[str, Any]:
    return {
        "status": "failed",
        "reason_codes": [],
        "scenario_id": None,
        "candidate_rows": [],
        "edge_rows": [],
        "memory": {
            "schema_version": MEMORY_SCHEMA_VERSION,
            "scenario_id": None,
            "features": {},
            "missing_indicators": {},
        },
        "coverage_memory_cell_count": 0,
        "reachable_safe_cell_count": 0,
        "frontier_cell_count": 0,
    }


def _stage3_ready(stage3_summary: dict[str, Any] | None) -> bool:
    return (
        isinstance(stage3_summary, dict)
        and stage3_summary.get("status") == "passed"
        and stage3_summary.get("next_required_change") == "topology_feature_extraction_audit"
    )


def _extract_topology_features(config: dict[str, Any], contract: dict[str, Any]) -> dict[str, Any]:
    scenario = dict(config["scenario"])
    geometry = scenario_geometry(scenario)
    scenario_id = geometry["scenario_id"]
    width = int(geometry["width"])
    height = int(geometry["height"])
    resolution_m = float(geometry["resolution_m"])
    target_cells = set(geometry["reachable_safe_cells"])
    navigation_cells = set(geometry["navigation_cells"]) & target_cells
    start = geometry["start"]
    current_cell = cell_from_config(scenario.get("current_cell", list(start)), "current_cell")
    if current_cell not in navigation_cells:
        current_cell = start
    current_covered_cells = _cells_from_config(scenario.get("current_covered_cells", []), "current_covered_cells")
    previous_path_cells = _cells_from_config(scenario.get("previous_path_cells", []), "previous_path_cells")
    coverage_memory_cells = (
        (coverage_footprint(start, width, height, int(config["coverage_radius_cells"])) | current_covered_cells | previous_path_cells)
        & target_cells
    )
    frontier = frontier_cells(coverage_memory_cells, navigation_cells)
    candidates = enumerate_frontier_candidates(
        current_cell=current_cell,
        frontier=frontier,
        navigation_cells=navigation_cells,
        target_cells=target_cells,
        covered_target_cells=coverage_memory_cells,
        coverage_map_cells=coverage_memory_cells,
        width=width,
        height=height,
        resolution_m=resolution_m,
        coverage_radius_cells=int(config["coverage_radius_cells"]),
        revisit_penalty_weight=float(config["revisit_penalty_weight"]),
        new_coverage_weight=float(config["new_coverage_weight"]),
        component_best_only=False,
    )
    candidate_rows = _candidate_rows(
        scenario_id=scenario_id,
        candidates=candidates,
        contract_fields=contract.get("candidate_topology_fields", []),
        coverage_memory_cells=coverage_memory_cells,
        target_cells=target_cells,
        navigation_cells=navigation_cells,
        width=width,
        height=height,
        path_budget_m=float(config["path_budget_m"]),
    )
    edge_rows = _edge_rows(
        scenario_id=scenario_id,
        candidate_rows=candidate_rows,
        raw_candidates=candidates,
        contract_fields=contract.get("candidate_edge_fields", []),
        navigation_cells=navigation_cells,
        target_cells=target_cells,
    )
    memory = _memory_payload(
        scenario_id=scenario_id,
        contract_fields=contract.get("coverage_memory_fields", []),
        scenario=scenario,
        coverage_memory_cells=coverage_memory_cells,
        target_cells=target_cells,
        path_budget_m=float(config["path_budget_m"]),
        roi_groups=_roi_groups(target_cells, width, height),
    )
    reasons = []
    if not target_cells:
        reasons.append("coverage_denominator_invalid")
    if not frontier:
        reasons.append("no_frontier_cells")
    if not candidate_rows:
        reasons.append("no_frontier_candidates")
    if len(candidate_rows) > 1 and not edge_rows:
        reasons.append("no_candidate_edges")
    return {
        "status": "passed" if not reasons else "failed",
        "reason_codes": unique_sorted(reasons),
        "scenario_id": scenario_id,
        "candidate_rows": candidate_rows,
        "edge_rows": edge_rows,
        "memory": memory,
        "coverage_memory_cell_count": len(coverage_memory_cells),
        "reachable_safe_cell_count": len(target_cells),
        "frontier_cell_count": len(frontier),
    }


def _candidate_rows(
    *,
    scenario_id: str,
    candidates: list[dict[str, Any]],
    contract_fields: list[dict[str, Any]],
    coverage_memory_cells: set[Cell],
    target_cells: set[Cell],
    navigation_cells: set[Cell],
    width: int,
    height: int,
    path_budget_m: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates):
        waypoint = tuple(candidate["selected_waypoint"])
        event_cells = set(candidate["event_cells"])
        path = [tuple(cell) for cell in candidate["path"]]
        features = {
            "frontier_cluster_id": int(candidate["cluster_index"]),
            "roi_group_id": _roi_group_id(waypoint, width, height),
            "new_coverage_cell_count": int(candidate["new_target_cell_count"]),
            "coverage_overlap_count": len(event_cells & coverage_memory_cells),
            "bfs_distance_from_current": max(0, len(path) - 1),
            "path_bottleneck_score": _path_bottleneck_score(path, navigation_cells),
            "revisit_path_cell_count": int(candidate["revisited_path_cell_count"]),
            "budget_fraction_cost": _safe_div(float(candidate["path_cost_m"]), path_budget_m),
            "fallback_risk": 1.0 if int(candidate["new_target_cell_count"]) <= 0 or float(candidate["path_cost_m"]) > path_budget_m else 0.0,
        }
        feature_names = [str(field.get("name")) for field in contract_fields]
        missing_indicators = {str(field.get("missing_indicator")): name not in features for field, name in zip(contract_fields, feature_names)}
        rows.append(
            {
                "schema_version": CANDIDATE_ROW_SCHEMA_VERSION,
                "scenario_id": scenario_id,
                "candidate_index": index,
                "baseline_rank": int(candidate["baseline_rank"]),
                "selected_waypoint": list(waypoint),
                "path": [list(cell) for cell in path],
                "features": {name: features[name] for name in feature_names if name in features},
                "missing_indicators": missing_indicators,
                "event_target_cell_count": len(event_cells & target_cells),
            }
        )
    return rows


def _edge_rows(
    *,
    scenario_id: str,
    candidate_rows: list[dict[str, Any]],
    raw_candidates: list[dict[str, Any]],
    contract_fields: list[dict[str, Any]],
    navigation_cells: set[Cell],
    target_cells: set[Cell],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    feature_names = [str(field.get("name")) for field in contract_fields]
    for left_index, left in enumerate(candidate_rows):
        for right_index in range(left_index + 1, len(candidate_rows)):
            right = candidate_rows[right_index]
            left_cell = tuple(left["selected_waypoint"])
            right_cell = tuple(right["selected_waypoint"])
            overlap = set(raw_candidates[left_index]["event_cells"]) & set(raw_candidates[right_index]["event_cells"]) & target_cells
            union = (set(raw_candidates[left_index]["event_cells"]) | set(raw_candidates[right_index]["event_cells"])) & target_cells
            bfs_distance = _bfs_distance(left_cell, right_cell, navigation_cells)
            coverage_overlap_ratio = _safe_div(len(overlap), len(union))
            shared_bottleneck = (
                float(left["features"].get("path_bottleneck_score", 0.0)) >= 0.5
                and float(right["features"].get("path_bottleneck_score", 0.0)) >= 0.5
            )
            features = {
                "same_frontier_cluster": left["features"].get("frontier_cluster_id") == right["features"].get("frontier_cluster_id"),
                "same_roi_group": left["features"].get("roi_group_id") == right["features"].get("roi_group_id"),
                "bfs_distance_between_candidates": bfs_distance,
                "coverage_overlap_ratio": coverage_overlap_ratio,
                "shared_bottleneck": shared_bottleneck,
                "mutual_redundancy_score": coverage_overlap_ratio * (1.0 if shared_bottleneck else 0.5),
            }
            rows.append(
                {
                    "schema_version": EDGE_ROW_SCHEMA_VERSION,
                    "scenario_id": scenario_id,
                    "candidate_left_index": left_index,
                    "candidate_right_index": right_index,
                    "candidate_pair_key": f"{left_index:04d}-{right_index:04d}",
                    "features": {name: features[name] for name in feature_names if name in features},
                    "missing_indicators": {
                        str(field.get("missing_indicator")): str(field.get("name")) not in features for field in contract_fields
                    },
                }
            )
    rows.sort(key=lambda row: row["candidate_pair_key"])
    return rows


def _memory_payload(
    *,
    scenario_id: str,
    contract_fields: list[dict[str, Any]],
    scenario: dict[str, Any],
    coverage_memory_cells: set[Cell],
    target_cells: set[Cell],
    path_budget_m: float,
    roi_groups: dict[int, set[Cell]],
) -> dict[str, Any]:
    history = scenario.get("recent_history", []) if isinstance(scenario.get("recent_history"), list) else []
    previous_path_cells = _cells_from_config(scenario.get("previous_path_cells", []), "previous_path_cells")
    total_recent_cost = sum(float(item.get("path_cost_m", 0.0)) for item in history if isinstance(item, dict))
    first_new = float(history[0].get("new_coverage_cell_count", 0.0)) if history and isinstance(history[0], dict) else 0.0
    last_new = float(history[-1].get("new_coverage_cell_count", 0.0)) if history and isinstance(history[-1], dict) else 0.0
    fallback_count = sum(1 for item in history if isinstance(item, dict) and item.get("fallback_used") is True)
    group_rates = [
        _safe_div(len(cells & coverage_memory_cells), len(cells))
        for cells in roi_groups.values()
        if cells
    ]
    features = {
        "coverage_rate": _safe_div(len(coverage_memory_cells & target_cells), len(target_cells)),
        "remaining_budget_fraction": max(0.0, min(1.0, 1.0 - _safe_div(total_recent_cost, path_budget_m))),
        "recent_path_cost_trend": _safe_div(total_recent_cost, max(1, len(history)) * path_budget_m),
        "recent_new_coverage_trend": last_new - first_new,
        "revisit_rate": _safe_div(len(previous_path_cells & coverage_memory_cells), len(previous_path_cells)),
        "fallback_rate": _safe_div(fallback_count, len(history)),
        "roi_group_completion_ratio": _safe_div(sum(group_rates), len(group_rates)),
    }
    return {
        "schema_version": MEMORY_SCHEMA_VERSION,
        "scenario_id": scenario_id,
        "features": {str(field.get("name")): features[str(field.get("name"))] for field in contract_fields if str(field.get("name")) in features},
        "missing_indicators": {
            str(field.get("missing_indicator")): str(field.get("name")) not in features for field in contract_fields
        },
    }


def _field_audit(contract: dict[str, Any] | None, extraction_result: dict[str, Any]) -> dict[str, Any]:
    candidate_fields = contract.get("candidate_topology_fields", []) if isinstance(contract, dict) else []
    edge_fields = contract.get("candidate_edge_fields", []) if isinstance(contract, dict) else []
    memory_fields = contract.get("coverage_memory_fields", []) if isinstance(contract, dict) else []
    candidate_missing = _missing_field_count(extraction_result["candidate_rows"], candidate_fields)
    edge_missing = _missing_field_count(extraction_result["edge_rows"], edge_fields)
    memory_missing = _missing_field_count([extraction_result["memory"]], memory_fields)
    finite_violations = _finite_numeric_violations(extraction_result["candidate_rows"], candidate_fields)
    finite_violations += _finite_numeric_violations(extraction_result["edge_rows"], edge_fields)
    finite_violations += _finite_numeric_violations([extraction_result["memory"]], memory_fields)
    reasons = []
    if candidate_missing:
        reasons.append("candidate_topology_fields_missing")
    if edge_missing:
        reasons.append("candidate_edge_fields_missing")
    if memory_missing:
        reasons.append("coverage_memory_fields_missing")
    if finite_violations:
        reasons.append("non_finite_topology_numeric_features")
    return {
        "schema_version": FIELD_AUDIT_SCHEMA_VERSION,
        "status": "passed" if not reasons else "failed",
        "reason_codes": reasons,
        "candidate_topology_field_count": len(candidate_fields),
        "candidate_edge_field_count": len(edge_fields),
        "coverage_memory_field_count": len(memory_fields),
        "candidate_missing_field_count": candidate_missing,
        "edge_missing_field_count": edge_missing,
        "memory_missing_field_count": memory_missing,
        "finite_numeric_violation_count": finite_violations,
    }


def _determinism_audit(extraction_result: dict[str, Any]) -> dict[str, Any]:
    candidate_keys = [
        (row["candidate_index"], row["baseline_rank"], row["selected_waypoint"])
        for row in extraction_result["candidate_rows"]
    ]
    sorted_candidate_keys = sorted(candidate_keys, key=lambda item: (item[0], item[1], item[2]))
    edge_keys = [row["candidate_pair_key"] for row in extraction_result["edge_rows"]]
    reasons = []
    if candidate_keys != sorted_candidate_keys:
        reasons.append("candidate_order_not_stable")
    if edge_keys != sorted(edge_keys):
        reasons.append("edge_order_not_stable")
    return {
        "schema_version": DETERMINISM_AUDIT_SCHEMA_VERSION,
        "status": "passed" if not reasons else "failed",
        "reason_codes": reasons,
        "candidate_order_stable": candidate_keys == sorted_candidate_keys,
        "edge_order_stable": edge_keys == sorted(edge_keys),
    }


def _boundary_audit(stage3_summary: dict[str, Any] | None) -> dict[str, Any]:
    violations = []
    if isinstance(stage3_summary, dict):
        for field in BOUNDARY_FIELDS:
            if stage3_summary.get(field) is True:
                violations.append({"source": "topology_observation_contract", "field": field, "value": True})
    return {
        "schema_version": BOUNDARY_AUDIT_SCHEMA_VERSION,
        "status": "passed" if not violations else "failed",
        "reason_codes": [] if not violations else ["topology_feature_extraction_boundary_violation"],
        "violations": violations,
        **global_99_boundary_defaults(),
        "real_world_release_approved": False,
        "real_world_performance_claimed": False,
        "default_policy_replacement_approved": False,
        "real_executor_connection_approved": False,
    }


def _decision(
    *,
    stage3_summary: dict[str, Any] | None,
    stage3_contract: dict[str, Any] | None,
    boundary_audit: dict[str, Any],
    field_audit: dict[str, Any],
    determinism_audit: dict[str, Any],
    extraction_error_reasons: list[str],
) -> dict[str, Any]:
    reasons: list[str] = []
    stage3_invalid = False
    if not isinstance(stage3_summary, dict):
        stage3_invalid = True
        reasons.append("missing_topology_observation_contract_summary")
    else:
        if stage3_summary.get("status") != "passed":
            stage3_invalid = True
            reasons.append("topology_observation_contract_not_passed")
        if stage3_summary.get("next_required_change") != "topology_feature_extraction_audit":
            stage3_invalid = True
            reasons.append("topology_observation_contract_next_required_change_invalid")
    if not isinstance(stage3_contract, dict):
        stage3_invalid = True
        reasons.append("missing_topology_observation_contract")
    reasons.extend(extraction_error_reasons)
    reasons.extend(field_audit.get("reason_codes", []))
    reasons.extend(determinism_audit.get("reason_codes", []))
    reasons.extend(boundary_audit.get("reason_codes", []))
    reasons = unique_sorted(reasons)
    status = "passed" if not reasons else "failed"
    if status == "passed":
        next_required_change = PASS_NEXT_REQUIRED_CHANGE
    elif stage3_invalid:
        next_required_change = FIX_STAGE3_NEXT_REQUIRED_CHANGE
    else:
        next_required_change = FAIL_NEXT_REQUIRED_CHANGE
    return {"status": status, "reason_codes": reasons, "next_required_change": next_required_change}


def _summary(
    *,
    generated_at: str,
    config_path: Path,
    output_root: Path,
    paths: dict[str, Path],
    stage3_summary_path: Path,
    stage3_contract_path: Path,
    extraction_result: dict[str, Any],
    field_audit: dict[str, Any],
    determinism_audit: dict[str, Any],
    boundary_audit: dict[str, Any],
    decision: dict[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": decision["status"],
        "reason_codes": decision["reason_codes"],
        "config": str(config_path),
        "output_root": str(output_root),
        "summary": str(paths["summary"]),
        "manifest": str(paths["manifest"]),
        "source_topology_observation_contract_summary": str(stage3_summary_path),
        "source_topology_observation_contract": str(stage3_contract_path),
        "feature_extraction_audit_passed": decision["status"] == "passed",
        "field_audit_passed": field_audit["status"] == "passed",
        "determinism_audit_passed": determinism_audit["status"] == "passed",
        "boundary_audit_passed": boundary_audit["status"] == "passed",
        "scenario_id": extraction_result["scenario_id"],
        "candidate_count": len(extraction_result["candidate_rows"]),
        "edge_count": len(extraction_result["edge_rows"]),
        "coverage_memory_cell_count": extraction_result["coverage_memory_cell_count"],
        "reachable_safe_cell_count": extraction_result["reachable_safe_cell_count"],
        "frontier_cell_count": extraction_result["frontier_cell_count"],
        "candidate_topology_field_count": field_audit["candidate_topology_field_count"],
        "candidate_edge_field_count": field_audit["candidate_edge_field_count"],
        "coverage_memory_field_count": field_audit["coverage_memory_field_count"],
        "candidate_missing_field_count": field_audit["candidate_missing_field_count"],
        "edge_missing_field_count": field_audit["edge_missing_field_count"],
        "memory_missing_field_count": field_audit["memory_missing_field_count"],
        "finite_numeric_violation_count": field_audit["finite_numeric_violation_count"],
        "next_required_change": decision["next_required_change"],
        **global_99_boundary_defaults(),
        "real_world_release_approved": False,
        "real_world_performance_claimed": False,
        "default_policy_replacement_approved": False,
        "real_executor_connection_approved": False,
        "uses_path_planner": False,
        "uses_ppo_policy": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "git_provenance": {"current": git_snapshot(repo_root)},
    }


def _manifest(generated_at: str, config_path: Path, output_root: Path, paths: dict[str, Path], summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "generated_at": generated_at,
        "config_path": str(config_path),
        "output_root": str(output_root),
        "status": summary["status"],
        "next_required_change": summary["next_required_change"],
        "artifacts": {key: str(value) for key, value in paths.items()},
    }


def _rejection_report(
    decision: dict[str, Any],
    field_audit: dict[str, Any],
    determinism_audit: dict[str, Any],
    boundary_audit: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": REJECTION_REPORT_SCHEMA_VERSION,
        "status": "passed" if decision["status"] == "passed" else "failed",
        "reason_codes": decision["reason_codes"],
        "next_required_change": decision["next_required_change"],
        "field_rejections": field_audit.get("reason_codes", []),
        "determinism_rejections": determinism_audit.get("reason_codes", []),
        "boundary_rejections": boundary_audit.get("violations", []),
    }


def _render_report(summary: dict[str, Any], field_audit: dict[str, Any], rejection_report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Xunce Topology Feature Extraction Audit v1",
            "",
            f"- status: `{summary['status']}`",
            f"- reason_codes: `{summary['reason_codes']}`",
            f"- next_required_change: `{summary['next_required_change']}`",
            f"- candidate_count: `{summary['candidate_count']}`",
            f"- edge_count: `{summary['edge_count']}`",
            f"- coverage_memory_cell_count: `{summary['coverage_memory_cell_count']}`",
            "",
            "## Field Audit",
            "",
            json.dumps(field_audit, ensure_ascii=False, indent=2),
            "",
            "## Rejections",
            "",
            json.dumps(rejection_report, ensure_ascii=False, indent=2),
            "",
        ]
    )


def _missing_field_count(rows: list[dict[str, Any]], fields: list[dict[str, Any]]) -> int:
    if not rows and fields:
        return len(fields)
    missing = 0
    for row in rows:
        features = row.get("features", {})
        indicators = row.get("missing_indicators", {})
        for field in fields:
            name = str(field.get("name"))
            indicator = str(field.get("missing_indicator"))
            if name not in features or indicator not in indicators or indicators[indicator] is True:
                missing += 1
    return missing


def _finite_numeric_violations(rows: list[dict[str, Any]], fields: list[dict[str, Any]]) -> int:
    numeric_fields = {str(field.get("name")) for field in fields if field.get("type") == "numeric"}
    violations = 0
    for row in rows:
        features = row.get("features", {})
        for name in numeric_fields:
            if name in features and not isfinite(float(features[name])):
                violations += 1
    return violations


def _cells_from_config(values: Any, label: str) -> set[Cell]:
    if values is None:
        return set()
    if not isinstance(values, list):
        raise ConfigError(f"{label} must be a list")
    return {cell_from_config(value, f"{label}[]") for value in values}


def _roi_group_id(cell: Cell, width: int, height: int) -> int:
    x, y = cell
    return (1 if x >= width / 2 else 0) + (2 if y >= height / 2 else 0)


def _roi_groups(cells: set[Cell], width: int, height: int) -> dict[int, set[Cell]]:
    groups: dict[int, set[Cell]] = {}
    for cell in cells:
        groups.setdefault(_roi_group_id(cell, width, height), set()).add(cell)
    return groups


def _path_bottleneck_score(path: list[Cell], navigation_cells: set[Cell]) -> float:
    if not path:
        return 0.0
    scores = []
    for cell in path:
        degree = sum(1 for neighbor in neighbors4(cell) if neighbor in navigation_cells)
        scores.append((4 - degree) / 4.0)
    return max(scores) if scores else 0.0


def _bfs_distance(start: Cell, target: Cell, navigation_cells: set[Cell]) -> int:
    distance, _ = bfs_tree(start, navigation_cells)
    return int(distance.get(target, 10**9))


def _safe_div(numerator: float | int, denominator: float | int) -> float:
    denominator_value = float(denominator)
    if denominator_value == 0.0:
        return 0.0
    return float(numerator) / denominator_value


def _int_value(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{label} must be an integer")
    return value


def _float_value(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ConfigError(f"{label} must be numeric")
    return float(value)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
