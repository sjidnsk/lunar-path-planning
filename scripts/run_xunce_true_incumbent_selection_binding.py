from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:
    from git_provenance import git_snapshot
    from global_99_coverage_contract import ConfigError, resolve_path, unique_sorted, utc_now, write_json, write_jsonl
    from global_99_governance_common import global_99_boundary_defaults
except ModuleNotFoundError:  # pragma: no cover
    from scripts.git_provenance import git_snapshot
    from scripts.global_99_coverage_contract import ConfigError, resolve_path, unique_sorted, utc_now, write_json, write_jsonl
    from scripts.global_99_governance_common import global_99_boundary_defaults


CONFIG_SCHEMA_VERSION = "xunce-true-incumbent-selection-binding-config/v1"
SUMMARY_SCHEMA_VERSION = "xunce-true-incumbent-selection-binding-summary/v1"
DEFAULT_CONFIG = "configs/xunce_true_incumbent_selection_binding_v1.json"
DEFAULT_OUTPUT_ROOT = "outputs/path_feedback_batch_xunce_true_incumbent_selection_binding_v1"

MATERIALIZATION_SUMMARY_FILE = "xunce-candidate-level-coverage-opportunity-summary.json"
CANDIDATE_SOURCE_SUMMARY_FILE = "xunce-true-frontier-nbv-candidate-source-summary.json"
EXPANSION_SUMMARY_FILE = "xunce-high-fidelity-real-map-roi-expansion-summary.json"
EXPANSION_SLICES_FILE = "xunce-high-fidelity-real-map-slices.jsonl"
PATH_FEEDBACK_AUDIT_FILE = "xunce-high-fidelity-path-feedback-audit.json"
MODEL_INFERENCE_RESULTS_FILE = "xunce-high-fidelity-model-inference-results.jsonl"
MODEL_INFERENCE_AUDIT_FILE = "xunce-high-fidelity-model-inference-audit.json"
COMPARISON_SUMMARY_FILE = "xunce-high-fidelity-real-map-comparison-summary.json"

SUMMARY_FILE = "xunce-true-incumbent-selection-binding-summary.json"
AUDIT_FILE = "xunce-true-incumbent-selection-binding-audit.json"
REPORT_FILE = "xunce-true-incumbent-selection-binding-report.md"

PASS_NEXT_REQUIRED_CHANGE = "run_safe_efficient_opportunity_root_cause_audit"
FIX_MODEL_INFERENCE_NEXT_REQUIRED_CHANGE = "run_xunce_high_fidelity_real_map_comparison"
FIX_ALIGNMENT_NEXT_REQUIRED_CHANGE = "repair_stage18e_candidate_alignment"
FIX_INCUMBENT_NEXT_REQUIRED_CHANGE = "fix_incumbent_policy_checkpoint"

BOUNDARY_FIELDS = tuple(global_99_boundary_defaults()) + (
    "publishes_checkpoint",
    "replaces_default_policy",
    "connects_real_executor",
    "starts_online_canary",
    "runs_new_ppo_update",
    "real_world_release_approved",
    "real_world_performance_claimed",
    "default_policy_replacement_approved",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Bind true incumbent checkpoint selections onto Stage 18E coverage candidates.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--repo-root")
    parser.add_argument("--source-materialized-coverage-root")
    parser.add_argument("--source-model-inference-root")
    args = parser.parse_args(argv)
    repo_root = Path(args.repo_root).resolve() if args.repo_root else Path(__file__).resolve().parents[1]
    overrides = {
        key: value
        for key, value in {
            "source_materialized_coverage_root": args.source_materialized_coverage_root,
            "source_model_inference_root": args.source_model_inference_root,
        }.items()
        if value is not None
    }
    try:
        summary = run_xunce_true_incumbent_selection_binding(
            config_path=resolve_path(Path(args.config), repo_root),
            output_root=resolve_path(Path(args.output_root), repo_root),
            repo_root=repo_root,
            config_overrides=overrides,
        )
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": summary["status"],
                "reason_codes": summary["reason_codes"],
                "true_incumbent_selection_bound": summary["true_incumbent_selection_bound"],
                "next_required_change": summary["next_required_change"],
                "summary": summary["summary"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if summary["status"] == "passed" else 1


def run_xunce_true_incumbent_selection_binding(
    *,
    config_path: Path,
    output_root: Path,
    repo_root: Path,
    config_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    repo_root = Path(repo_root)
    config = _load_config(config_path, repo_root, config_overrides=config_overrides)
    output_root.mkdir(parents=True, exist_ok=True)
    source = _load_source(config)
    bound_path_feedback, binding_rows, metrics = _bind(config, source)
    decision = _decision(config, source, metrics)
    paths = _paths(output_root)
    generated_at = utc_now()
    summary = _summary(
        generated_at=generated_at,
        config=config,
        config_path=config_path,
        output_root=output_root,
        repo_root=repo_root,
        source=source,
        metrics=metrics,
        decision=decision,
        paths=paths,
    )
    write_json(paths["summary"], summary)
    write_json(paths["audit"], _audit(source, metrics, decision))
    paths["report"].write_text(_render_report(summary), encoding="utf-8")
    write_json(paths["path_feedback"], bound_path_feedback)
    write_json(paths["expansion_summary"], _bound_expansion_summary(source["expansion_summary"], summary))
    write_jsonl(paths["slices"], source["slices"])
    write_json(paths["materialization_summary"], _bound_materialization_summary(source["materialization_summary"], summary))
    write_jsonl(paths["binding_rows"], binding_rows)
    return summary


def _load_config(path: Path, repo_root: Path, *, config_overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"config file does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ConfigError("config root must be an object")
    if payload.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ConfigError(f"schema_version must be {CONFIG_SCHEMA_VERSION!r}")
    if config_overrides:
        payload = {**payload, **config_overrides}
    normalized = dict(payload)
    for key in ("source_materialized_coverage_root", "source_model_inference_root"):
        value = payload.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ConfigError(f"{key} must be a non-empty string")
        normalized[key] = str(resolve_path(Path(value), repo_root))
    normalized["required_scenario_count"] = _positive_int(payload.get("required_scenario_count", 24), "required_scenario_count")
    for key in ("require_true_model_inference", "require_candidate_cell_match", "allow_fallback_action_index"):
        normalized[key] = _bool(payload.get(key, key != "allow_fallback_action_index"), key)
    normalized["canary_traffic_fraction"] = _nonnegative_float(payload.get("canary_traffic_fraction", 0.0), "canary_traffic_fraction")
    return normalized


def _load_source(config: dict[str, Any]) -> dict[str, Any]:
    materialized_root = Path(config["source_materialized_coverage_root"])
    inference_root = Path(config["source_model_inference_root"])
    reasons: list[str] = []
    materialization_summary = _read_materialization_summary(materialized_root, reasons)
    expansion_summary = _read_json(materialized_root / EXPANSION_SUMMARY_FILE, reasons, "missing_stage18e_materialization")
    slices = _read_jsonl(materialized_root / EXPANSION_SLICES_FILE, reasons, "missing_stage18e_materialization_slices")
    path_feedback = _read_json(materialized_root / PATH_FEEDBACK_AUDIT_FILE, reasons, "missing_stage18e_materialization")
    inference_rows = _read_jsonl(inference_root / MODEL_INFERENCE_RESULTS_FILE, reasons, "missing_model_inference_artifact")
    inference_audit = _read_json(inference_root / MODEL_INFERENCE_AUDIT_FILE, reasons, "missing_model_inference_artifact")
    comparison_summary = _read_json(inference_root / COMPARISON_SUMMARY_FILE, [], "missing_model_inference_artifact")
    scenarios = path_feedback.get("scenarios") if isinstance(path_feedback.get("scenarios"), list) else []
    return {
        "materialized_root": materialized_root,
        "inference_root": inference_root,
        "materialization_summary": materialization_summary,
        "expansion_summary": expansion_summary,
        "slices": [row for row in slices if isinstance(row, dict)],
        "path_feedback": path_feedback,
        "scenarios": [row for row in scenarios if isinstance(row, dict)],
        "inference_rows": [row for row in inference_rows if isinstance(row, dict)],
        "inference_audit": inference_audit,
        "comparison_summary": comparison_summary,
        "reason_codes": unique_sorted(reasons),
    }


def _read_materialization_summary(root: Path, reasons: list[str]) -> dict[str, Any]:
    for filename in (MATERIALIZATION_SUMMARY_FILE, CANDIDATE_SOURCE_SUMMARY_FILE):
        path = root / filename
        if path.is_file():
            return _read_json(path, reasons, "missing_stage18e_materialization")
    reasons.append("missing_stage18e_materialization")
    return {}


def _paths(output_root: Path) -> dict[str, Path]:
    return {
        "summary": output_root / SUMMARY_FILE,
        "audit": output_root / AUDIT_FILE,
        "report": output_root / REPORT_FILE,
        "path_feedback": output_root / PATH_FEEDBACK_AUDIT_FILE,
        "expansion_summary": output_root / EXPANSION_SUMMARY_FILE,
        "slices": output_root / EXPANSION_SLICES_FILE,
        "materialization_summary": output_root / MATERIALIZATION_SUMMARY_FILE,
        "binding_rows": output_root / "xunce-true-incumbent-selection-binding-rows.jsonl",
    }


def _bind(config: dict[str, Any], source: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    inference_by_scenario = {str(row.get("scenario_id")): row for row in source["inference_rows"]}
    bound = dict(source["path_feedback"])
    bound_scenarios: list[dict[str, Any]] = []
    binding_rows: list[dict[str, Any]] = []
    candidate_cell_mismatch_count = 0
    missing_inference_count = 0
    non_true_inference_count = 0
    proxy_selection_count = 0
    incumbent_missing_count = 0
    xunce_missing_count = 0
    fallback_count = 0
    finite_issue_count = 0
    reason_codes = list(source["reason_codes"])

    audit = source["inference_audit"]
    summary = source["comparison_summary"]
    true_model_inference_executed = bool(audit.get("true_model_inference_executed") is True and summary.get("true_model_inference_executed", True) is True)
    proxy_selection_used = bool(audit.get("proxy_selection_used") is True or summary.get("proxy_selection_used") is True)
    xunce_checkpoint_loaded = bool(audit.get("xunce_checkpoint_loaded") is True and summary.get("xunce_checkpoint_loaded", True) is True)
    incumbent_checkpoint_loaded = bool(audit.get("incumbent_checkpoint_loaded") is True and summary.get("incumbent_checkpoint_loaded", True) is True)

    for scenario_index, scenario in enumerate(source["scenarios"][: config["required_scenario_count"]]):
        scenario_copy = dict(scenario)
        scenario_id = str(scenario_copy.get("scenario_id") or f"scenario-{scenario_index:04d}")
        candidates = _candidate_rows(scenario_copy)
        inference = inference_by_scenario.get(scenario_id)
        if inference is None:
            missing_inference_count += 1
            bound_scenarios.append(scenario_copy)
            continue
        if inference.get("true_model_inference_executed") is not True:
            non_true_inference_count += 1
        if inference.get("proxy_selection_used") is True:
            proxy_selection_count += 1
        expected_cells = [_cell(candidate.get("cell")) for candidate in candidates]
        observed_cells = [_cell(cell) for cell in inference.get("candidate_cells", [])]
        cells_match = expected_cells == observed_cells
        if not cells_match:
            candidate_cell_mismatch_count += 1
        incumbent = inference.get("incumbent") if isinstance(inference.get("incumbent"), dict) else {}
        xunce = inference.get("xunce") if isinstance(inference.get("xunce"), dict) else {}
        incumbent_index = _selected_index(incumbent)
        xunce_index = _selected_index(xunce)
        if incumbent_index is None:
            incumbent_missing_count += 1
        if xunce_index is None:
            xunce_missing_count += 1
        if incumbent.get("finite_outputs") is not True or xunce.get("finite_outputs") is not True:
            finite_issue_count += 1
        if incumbent_index == 0 and not incumbent:
            fallback_count += 1
        if incumbent_index is not None and cells_match:
            scenario_copy.update(
                {
                    "incumbent_selected_action_index": incumbent_index,
                    "incumbent_selected_probability": incumbent.get("selected_probability"),
                    "incumbent_selected_rank": incumbent.get("selected_rank"),
                    "incumbent_value": incumbent.get("value"),
                    "incumbent_latency_ms": incumbent.get("latency_ms"),
                    "incumbent_selection_source": "true_checkpoint_inference",
                }
            )
        if xunce_index is not None and cells_match:
            scenario_copy.update(
                {
                    "xunce_selected_action_index": xunce_index,
                    "xunce_selected_probability": xunce.get("selected_probability"),
                    "xunce_selected_rank": xunce.get("selected_rank"),
                    "xunce_value": xunce.get("value"),
                    "xunce_latency_ms": xunce.get("latency_ms"),
                    "xunce_selection_source": "true_checkpoint_inference",
                }
            )
        binding_rows.append(
            {
                "schema_version": "xunce-true-incumbent-selection-binding-row/v1",
                "scenario_id": scenario_id,
                "candidate_cells_match": cells_match,
                "expected_candidate_cells": expected_cells,
                "inference_candidate_cells": observed_cells,
                "incumbent_selected_action_index": incumbent_index,
                "xunce_selected_action_index": xunce_index,
                "incumbent_selection_source": scenario_copy.get("incumbent_selection_source"),
                "xunce_selection_source": scenario_copy.get("xunce_selection_source"),
                "true_model_inference_executed": inference.get("true_model_inference_executed") is True,
            }
        )
        bound_scenarios.append(scenario_copy)

    if not true_model_inference_executed:
        reason_codes.append("true_inference_not_executed")
    if proxy_selection_used or proxy_selection_count:
        reason_codes.append("proxy_selection_used")
    if not xunce_checkpoint_loaded:
        reason_codes.append("missing_xunce_checkpoint")
    if not incumbent_checkpoint_loaded:
        reason_codes.append("missing_true_incumbent_selection")
    if missing_inference_count:
        reason_codes.append("missing_model_inference_artifact")
    if non_true_inference_count:
        reason_codes.append("true_inference_not_executed")
    if candidate_cell_mismatch_count:
        reason_codes.append("candidate_cell_mismatch")
    if incumbent_missing_count:
        reason_codes.append("missing_true_incumbent_selection")
    if finite_issue_count:
        reason_codes.append("non_finite_model_outputs")
    bound["scenarios"] = bound_scenarios
    bound["scenario_count"] = len(bound_scenarios)
    metrics = {
        "reason_codes": unique_sorted(reason_codes),
        "scenario_count": len(bound_scenarios),
        "true_model_inference_executed": true_model_inference_executed and non_true_inference_count == 0,
        "proxy_selection_used": proxy_selection_used or proxy_selection_count > 0,
        "xunce_checkpoint_loaded": xunce_checkpoint_loaded,
        "incumbent_checkpoint_loaded": incumbent_checkpoint_loaded,
        "incumbent_selected_action_index_missing_count": incumbent_missing_count,
        "xunce_selected_action_index_missing_count": xunce_missing_count,
        "fallback_action_index_0_count": fallback_count,
        "candidate_cell_mismatch_count": candidate_cell_mismatch_count,
        "missing_model_inference_row_count": missing_inference_count,
        "non_true_model_inference_row_count": non_true_inference_count,
        "finite_output_issue_count": finite_issue_count,
    }
    return bound, binding_rows, metrics


def _decision(config: dict[str, Any], source: dict[str, Any], metrics: dict[str, Any]) -> dict[str, Any]:
    reasons = list(metrics["reason_codes"])
    if config["require_true_model_inference"] and not metrics["true_model_inference_executed"]:
        reasons.append("true_inference_not_executed")
    if config["require_candidate_cell_match"] and metrics["candidate_cell_mismatch_count"]:
        reasons.append("candidate_cell_mismatch")
    if not config["allow_fallback_action_index"] and metrics["fallback_action_index_0_count"]:
        reasons.append("fallback_action_index_0_not_allowed")
    reasons = unique_sorted(reasons)
    if not reasons:
        return {"status": "passed", "reason_codes": [], "next_required_change": PASS_NEXT_REQUIRED_CHANGE}
    if "missing_model_inference_artifact" in reasons or "true_inference_not_executed" in reasons or "proxy_selection_used" in reasons:
        next_change = FIX_MODEL_INFERENCE_NEXT_REQUIRED_CHANGE
    elif "candidate_cell_mismatch" in reasons:
        next_change = FIX_ALIGNMENT_NEXT_REQUIRED_CHANGE
    else:
        next_change = FIX_INCUMBENT_NEXT_REQUIRED_CHANGE
    return {"status": "failed", "reason_codes": reasons, "next_required_change": next_change}


def _summary(
    *,
    generated_at: str,
    config: dict[str, Any],
    config_path: Path,
    output_root: Path,
    repo_root: Path,
    source: dict[str, Any],
    metrics: dict[str, Any],
    decision: dict[str, Any],
    paths: dict[str, Path],
) -> dict[str, Any]:
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": decision["status"],
        "reason_codes": decision["reason_codes"],
        "next_required_change": decision["next_required_change"],
        "true_incumbent_selection_bound": decision["status"] == "passed",
        **{key: value for key, value in metrics.items() if key != "reason_codes"},
        "source_materialized_coverage_root": config["source_materialized_coverage_root"],
        "source_model_inference_root": config["source_model_inference_root"],
        "source_materialization_status": source["materialization_summary"].get("status"),
        "canary_traffic_fraction": config["canary_traffic_fraction"],
        "config": str(config_path),
        "output_root": str(output_root),
        "summary": str(paths["summary"]),
        "bound_path_feedback": str(paths["path_feedback"]),
        "git_provenance": git_snapshot(repo_root),
        **_boundary_fields(),
    }


def _audit(source: dict[str, Any], metrics: dict[str, Any], decision: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "xunce-true-incumbent-selection-binding-audit/v1",
        "status": decision["status"],
        "reason_codes": decision["reason_codes"],
        "next_required_change": decision["next_required_change"],
        "source_inference_audit": source["inference_audit"],
        **metrics,
        **_boundary_fields(),
    }


def _bound_expansion_summary(source_summary: dict[str, Any], summary: dict[str, Any]) -> dict[str, Any]:
    payload = dict(source_summary)
    payload["status"] = summary["status"]
    payload["reason_codes"] = list(summary["reason_codes"])
    payload["next_required_change"] = summary["next_required_change"]
    payload["true_incumbent_selection_bound"] = summary["true_incumbent_selection_bound"]
    payload["fallback_action_index_0_count"] = summary["fallback_action_index_0_count"]
    payload.update(_boundary_fields())
    return payload


def _bound_materialization_summary(source_summary: dict[str, Any], summary: dict[str, Any]) -> dict[str, Any]:
    payload = dict(source_summary)
    payload["status"] = summary["status"]
    payload["reason_codes"] = list(summary["reason_codes"])
    payload["next_required_change"] = summary["next_required_change"]
    payload["true_incumbent_selection_bound"] = summary["true_incumbent_selection_bound"]
    payload["fallback_action_index_0_count"] = summary["fallback_action_index_0_count"]
    payload.update(_boundary_fields())
    return payload


def _render_report(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Xunce True Incumbent Selection Binding",
            "",
            f"- status: `{summary['status']}`",
            f"- true_incumbent_selection_bound: `{summary['true_incumbent_selection_bound']}`",
            f"- next_required_change: `{summary['next_required_change']}`",
            f"- incumbent_selected_action_index_missing_count: `{summary['incumbent_selected_action_index_missing_count']}`",
            f"- fallback_action_index_0_count: `{summary['fallback_action_index_0_count']}`",
            f"- candidate_cell_mismatch_count: `{summary['candidate_cell_mismatch_count']}`",
        ]
    )


def _candidate_rows(scenario: dict[str, Any]) -> list[dict[str, Any]]:
    feedback = scenario.get("path_feedback")
    candidates = feedback.get("candidates") if isinstance(feedback, dict) else None
    return [row for row in candidates if isinstance(row, dict)] if isinstance(candidates, list) else []


def _selected_index(detail: dict[str, Any]) -> int | None:
    value = detail.get("selected_action_index")
    return int(value) if isinstance(value, int) else None


def _cell(value: Any) -> list[int] | None:
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        try:
            return [int(value[0]), int(value[1])]
        except (TypeError, ValueError):
            return None
    return None


def _read_json(path: Path, reasons: list[str], reason_code: str) -> dict[str, Any]:
    if not path.is_file():
        reasons.append(reason_code)
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        reasons.append(reason_code)
        return {}
    return payload if isinstance(payload, dict) else {}


def _read_jsonl(path: Path, reasons: list[str], reason_code: str) -> list[dict[str, Any]]:
    if not path.is_file():
        reasons.append(reason_code)
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            reasons.append(reason_code)
            continue
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _boundary_fields() -> dict[str, bool]:
    return {field: False for field in BOUNDARY_FIELDS}


def _positive_int(value: Any, field: str) -> int:
    if not isinstance(value, int) or value <= 0:
        raise ConfigError(f"{field} must be a positive integer")
    return value


def _nonnegative_float(value: Any, field: str) -> float:
    if not isinstance(value, (int, float)) or float(value) < 0.0:
        raise ConfigError(f"{field} must be a non-negative number")
    return float(value)


def _bool(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"{field} must be a boolean")
    return value


if __name__ == "__main__":
    raise SystemExit(main())
