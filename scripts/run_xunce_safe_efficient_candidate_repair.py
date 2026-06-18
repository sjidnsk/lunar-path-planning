from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
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


CONFIG_SCHEMA_VERSION = "xunce-safe-efficient-candidate-repair-config/v1"
SUMMARY_SCHEMA_VERSION = "xunce-safe-efficient-candidate-repair-summary/v1"
DEFAULT_CONFIG = "configs/xunce_safe_efficient_candidate_repair_v1.json"
DEFAULT_OUTPUT_ROOT = "outputs/path_feedback_batch_xunce_safe_efficient_candidate_repair_v1"

BINDING_SUMMARY_FILE = "xunce-true-incumbent-selection-binding-summary.json"
ROOT_CAUSE_SUMMARY_FILE = "xunce-safe-efficient-opportunity-root-cause-audit-summary.json"
MATERIALIZATION_SUMMARY_FILE = "xunce-candidate-level-coverage-opportunity-summary.json"
EXPANSION_SUMMARY_FILE = "xunce-high-fidelity-real-map-roi-expansion-summary.json"
EXPANSION_SLICES_FILE = "xunce-high-fidelity-real-map-slices.jsonl"
PATH_FEEDBACK_AUDIT_FILE = "xunce-high-fidelity-path-feedback-audit.json"

SUMMARY_FILE = "xunce-safe-efficient-candidate-repair-summary.json"
OVERLAY_FILE = "xunce-safe-efficient-candidate-repair-overlay.jsonl"
AUDIT_FILE = "xunce-safe-efficient-candidate-repair-audit.json"
REPORT_FILE = "xunce-safe-efficient-candidate-repair-report.md"

PASS_NEXT_REQUIRED_CHANGE = "rerun_stage18f1_with_repaired_root"
FIX_BINDING_NEXT_REQUIRED_CHANGE = "run_true_incumbent_selection_binding"
FIX_VALIDATION_NEXT_REQUIRED_CHANGE = "repair_path_feedback_candidate_validation"
EXPAND_COMPLEXITY_NEXT_REQUIRED_CHANGE = "expand_roi_or_map_complexity"
EXPAND_ROI_NEXT_REQUIRED_CHANGE = "expand_roi_or_refine_roi_weighting"

TOLERANCE = 1.0e-12
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
    parser = argparse.ArgumentParser(description="Repair Xunce candidates into validated safe-efficient coverage opportunities.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--repo-root")
    parser.add_argument("--source-bound-coverage-root")
    parser.add_argument("--source-root-cause-audit-root")
    args = parser.parse_args(argv)
    repo_root = Path(args.repo_root).resolve() if args.repo_root else Path(__file__).resolve().parents[1]
    overrides = {
        key: value
        for key, value in {
            "source_bound_coverage_root": args.source_bound_coverage_root,
            "source_root_cause_audit_root": args.source_root_cause_audit_root,
        }.items()
        if value is not None
    }
    try:
        summary = run_xunce_safe_efficient_candidate_repair(
            config_path=resolve_path(Path(args.config), repo_root),
            output_root=resolve_path(Path(args.output_root), repo_root),
            repo_root=repo_root,
            config_overrides=overrides,
        )
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"status": summary["status"], "reason_codes": summary["reason_codes"], "safe_efficient_opportunity_count": summary["safe_efficient_opportunity_count"], "next_required_change": summary["next_required_change"], "summary": summary["summary"]}, ensure_ascii=False, sort_keys=True))
    return 0 if summary["status"] == "passed" else 1


def run_xunce_safe_efficient_candidate_repair(
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
    repaired, overlay_rows, metrics = _repair(config, source)
    decision = _decision(config, source, metrics)
    paths = _paths(output_root)
    generated_at = utc_now()
    summary = _summary(generated_at, config, config_path, output_root, repo_root, source, metrics, decision, paths)
    write_json(paths["summary"], summary)
    write_jsonl(paths["overlay"], overlay_rows)
    write_json(paths["audit"], {"schema_version": "xunce-safe-efficient-candidate-repair-audit/v1", **decision, **metrics, **_boundary_fields()})
    paths["report"].write_text(_render_report(summary), encoding="utf-8")
    write_json(paths["path_feedback"], repaired)
    write_json(paths["expansion_summary"], _repaired_expansion_summary(source["expansion_summary"], summary))
    write_jsonl(paths["slices"], source["slices"])
    write_json(paths["materialization_summary"], _repaired_materialization_summary(source["materialization_summary"], summary))
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
    for key in ("source_bound_coverage_root", "source_root_cause_audit_root"):
        value = payload.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ConfigError(f"{key} must be a non-empty string")
        normalized[key] = str(resolve_path(Path(value), repo_root))
    normalized["required_scenario_count"] = _positive_int(payload.get("required_scenario_count", 24), "required_scenario_count")
    normalized["candidate_generation_mode"] = _string(payload.get("candidate_generation_mode", "cost_guarded_frontier_interpolation"), "candidate_generation_mode")
    normalized["interpolation_fractions"] = _fractions(payload.get("interpolation_fractions", [0.25, 0.5, 0.75]))
    normalized["max_repaired_candidates_per_scenario"] = _positive_int(payload.get("max_repaired_candidates_per_scenario", 6), "max_repaired_candidates_per_scenario")
    normalized["min_safe_efficient_opportunity_count"] = _positive_int(payload.get("min_safe_efficient_opportunity_count", 1), "min_safe_efficient_opportunity_count")
    normalized["min_roi_group_with_safe_efficient_opportunity"] = _positive_int(payload.get("min_roi_group_with_safe_efficient_opportunity", 3), "min_roi_group_with_safe_efficient_opportunity")
    normalized["path_budget_m"] = _positive_float(payload.get("path_budget_m", 5000.0), "path_budget_m")
    normalized["canary_traffic_fraction"] = _nonnegative_float(payload.get("canary_traffic_fraction", 0.0), "canary_traffic_fraction")
    return normalized


def _load_source(config: dict[str, Any]) -> dict[str, Any]:
    bound_root = Path(config["source_bound_coverage_root"])
    audit_root = Path(config["source_root_cause_audit_root"])
    reasons: list[str] = []
    binding_summary = _read_json(bound_root / BINDING_SUMMARY_FILE, reasons, "true_incumbent_binding_missing")
    materialization_summary = _read_json(bound_root / MATERIALIZATION_SUMMARY_FILE, reasons, "true_incumbent_binding_missing")
    expansion_summary = _read_json(bound_root / EXPANSION_SUMMARY_FILE, reasons, "true_incumbent_binding_missing")
    slices = _read_jsonl(bound_root / EXPANSION_SLICES_FILE, reasons, "true_incumbent_binding_missing")
    path_feedback = _read_json(bound_root / PATH_FEEDBACK_AUDIT_FILE, reasons, "true_incumbent_binding_missing")
    root_cause_summary = _read_json(audit_root / ROOT_CAUSE_SUMMARY_FILE, [], "missing_root_cause_audit")
    scenarios = path_feedback.get("scenarios") if isinstance(path_feedback.get("scenarios"), list) else []
    return {
        "bound_root": bound_root,
        "audit_root": audit_root,
        "binding_summary": binding_summary,
        "root_cause_summary": root_cause_summary,
        "materialization_summary": materialization_summary,
        "expansion_summary": expansion_summary,
        "slices": [row for row in slices if isinstance(row, dict)],
        "path_feedback": path_feedback,
        "scenarios": [row for row in scenarios if isinstance(row, dict)],
        "reason_codes": unique_sorted(reasons),
    }


def _paths(output_root: Path) -> dict[str, Path]:
    return {
        "summary": output_root / SUMMARY_FILE,
        "overlay": output_root / OVERLAY_FILE,
        "audit": output_root / AUDIT_FILE,
        "report": output_root / REPORT_FILE,
        "path_feedback": output_root / PATH_FEEDBACK_AUDIT_FILE,
        "expansion_summary": output_root / EXPANSION_SUMMARY_FILE,
        "slices": output_root / EXPANSION_SLICES_FILE,
        "materialization_summary": output_root / MATERIALIZATION_SUMMARY_FILE,
    }


def _repair(config: dict[str, Any], source: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    repaired = dict(source["path_feedback"])
    repaired_scenarios: list[dict[str, Any]] = []
    overlay_rows: list[dict[str, Any]] = []
    roi_safe_counts: dict[str, int] = defaultdict(int)
    safe_count = 0
    safe_scenario_count = 0
    unvalidated_positive = 0
    no_validated_repaired = 0
    fallback_count = 0
    open_grid_count = 0
    frontier_counts: list[int] = []
    spread_values: list[float] = []
    candidate_count = 0
    repair_candidate_count = 0

    for scenario_index, scenario in enumerate(source["scenarios"][: config["required_scenario_count"]]):
        scenario_copy = dict(scenario)
        scenario_id = str(scenario.get("scenario_id") or f"scenario-{scenario_index:04d}")
        roi_group = str(scenario.get("roi_group") or scenario.get("scenario_group") or "unknown")
        candidates = [dict(row) for row in _candidate_rows(scenario)]
        incumbent_index = _selected_index(scenario.get("incumbent_selected_action_index"))
        incumbent = _candidate_at(candidates, incumbent_index)
        if scenario.get("incumbent_selection_source") == "fallback_action_index_0":
            fallback_count += 1
        existing_cells = {_cell_key(candidate.get("cell")) for candidate in candidates}
        repaired_candidates: list[dict[str, Any]] = []
        scenario_safe = 0
        scenario_unvalidated_positive = 0

        for action_index, candidate in enumerate(candidates):
            enriched = dict(candidate)
            safe = _safe_efficient(enriched, incumbent) and _candidate_valid(enriched)
            if safe:
                enriched.update(
                    {
                        "candidate_repair_source": "cost_guarded_frontier_interpolation/v1",
                        "proposal_validated_by_path_feedback": True,
                        "proposal_only": False,
                        "safe_efficient_opportunity": True,
                    }
                )
                safe_count += 1
                scenario_safe += 1
                roi_safe_counts[roi_group] += 1
                repair_candidate_count += 1
            elif enriched.get("open_grid_fallback_used") is True:
                open_grid_count += 1
            repaired_candidates.append(enriched)

        proposals = _proposal_rows(config, scenario_id, roi_group, candidates, incumbent, existing_cells)
        for proposal in proposals:
            if proposal.get("safe_efficient_opportunity") is True and proposal.get("proposal_validated_by_path_feedback") is not True:
                scenario_unvalidated_positive += 1
                unvalidated_positive += 1
            repaired_candidates.append(proposal)

        if scenario_safe == 0:
            no_validated_repaired += 1
        else:
            safe_scenario_count += 1
        for candidate in repaired_candidates:
            if candidate.get("proposal_only") is True:
                continue
            candidate_count += 1
        frontier_count = _pareto_frontier_count([candidate for candidate in repaired_candidates if candidate.get("proposal_only") is not True])
        frontier_counts.append(frontier_count)
        spread_values.append(_spread([_cost_efficiency(candidate) for candidate in repaired_candidates if _candidate_valid(candidate) and candidate.get("proposal_only") is not True]))
        scenario_copy["path_feedback"] = dict(scenario_copy.get("path_feedback") or {})
        scenario_copy["path_feedback"]["candidates"] = repaired_candidates
        repaired_scenarios.append(scenario_copy)
        overlay_rows.append(
            {
                "schema_version": "xunce-safe-efficient-candidate-repair-overlay-row/v1",
                "scenario_id": scenario_id,
                "roi_group": roi_group,
                "safe_efficient_opportunity_count": scenario_safe,
                "proposal_unvalidated_positive_count": scenario_unvalidated_positive,
                "candidate_pareto_frontier_count": frontier_count,
                "candidate_count": len(repaired_candidates),
            }
        )

    repaired["scenarios"] = repaired_scenarios
    repaired["scenario_count"] = len(repaired_scenarios)
    repaired["candidate_count"] = candidate_count
    repaired["candidate_repair_source"] = "cost_guarded_frontier_interpolation/v1"
    metrics = {
        "reason_codes": unique_sorted(source["reason_codes"]),
        "scenario_count": len(repaired_scenarios),
        "candidate_count": candidate_count,
        "repair_candidate_count": repair_candidate_count,
        "safe_efficient_opportunity_count": safe_count,
        "safe_efficient_opportunity_scenario_count": safe_scenario_count,
        "roi_group_with_safe_efficient_opportunity_count": sum(1 for value in roi_safe_counts.values() if value > 0),
        "cost_efficient_coverage_spread_range": _mean(spread_values),
        "candidate_pareto_frontier_count": _mean(frontier_counts),
        "fallback_action_index_0_count": fallback_count,
        "proposal_unvalidated_positive_count": unvalidated_positive,
        "open_grid_fallback_count": open_grid_count,
        "no_validated_repaired_candidate_scenario_count": no_validated_repaired,
    }
    return repaired, overlay_rows, metrics


def _proposal_rows(config: dict[str, Any], scenario_id: str, roi_group: str, candidates: list[dict[str, Any]], incumbent: dict[str, Any] | None, existing_cells: set[tuple[int, int] | None]) -> list[dict[str, Any]]:
    if not incumbent:
        return []
    high_coverage = [
        candidate
        for candidate in candidates
        if _candidate_valid(candidate) and _coverage(candidate) > _coverage(incumbent) + TOLERANCE and not _safe_efficient(candidate, incumbent)
    ]
    high_coverage = sorted(high_coverage, key=lambda item: _coverage(item), reverse=True)[: int(config["max_repaired_candidates_per_scenario"])]
    proposals: list[dict[str, Any]] = []
    for source in high_coverage:
        start = _cell_tuple(incumbent.get("cell"))
        end = _cell_tuple(source.get("cell"))
        if start is None or end is None:
            continue
        for fraction in config["interpolation_fractions"]:
            cell = (round(start[0] + (end[0] - start[0]) * fraction), round(start[1] + (end[1] - start[1]) * fraction))
            if cell in existing_cells:
                continue
            proposals.append(
                {
                    "schema_version": "xunce-safe-efficient-candidate-proposal/v1",
                    "scenario_id": scenario_id,
                    "roi_group": roi_group,
                    "cell": [int(cell[0]), int(cell[1])],
                    "reachable": False,
                    "path_cost": None,
                    "risk": None,
                    "roi_weighted_coverage_delta": _coverage(source),
                    "path_line_coverage_delta": _coverage(source),
                    "revisit_penalty": _revisit(source),
                    "candidate_repair_source": "cost_guarded_frontier_interpolation/v1",
                    "proposal_validated_by_path_feedback": False,
                    "proposal_only": True,
                    "safe_efficient_opportunity": False,
                    "source_incumbent_cell": incumbent.get("cell"),
                    "source_high_coverage_cell": source.get("cell"),
                    "interpolation_fraction": fraction,
                }
            )
    return proposals[: int(config["max_repaired_candidates_per_scenario"])]


def _decision(config: dict[str, Any], source: dict[str, Any], metrics: dict[str, Any]) -> dict[str, Any]:
    reasons = list(metrics["reason_codes"])
    if source["binding_summary"].get("true_incumbent_selection_bound") is not True or metrics["fallback_action_index_0_count"] > 0:
        reasons.append("true_incumbent_binding_missing")
    if metrics["no_validated_repaired_candidate_scenario_count"] >= metrics["scenario_count"]:
        reasons.append("no_validated_repaired_candidates")
    if metrics["safe_efficient_opportunity_count"] < int(config["min_safe_efficient_opportunity_count"]):
        reasons.append("safe_efficient_opportunity_still_zero")
    if metrics["roi_group_with_safe_efficient_opportunity_count"] < int(config["min_roi_group_with_safe_efficient_opportunity"]):
        reasons.append("roi_safe_efficient_spread_insufficient")
    if metrics["cost_efficient_coverage_spread_range"] <= 0.005:
        reasons.append("cost_efficient_coverage_spread_insufficient")
    if metrics["candidate_pareto_frontier_count"] <= 1.0:
        reasons.append("candidate_pareto_frontier_insufficient")
    if metrics["proposal_unvalidated_positive_count"] > 0:
        reasons.append("proposal_unvalidated_positive")
    if metrics["open_grid_fallback_count"] > 0:
        reasons.append("open_grid_fallback")
    reasons = unique_sorted(reasons)
    if not reasons:
        return {"status": "passed", "reason_codes": [], "next_required_change": PASS_NEXT_REQUIRED_CHANGE}
    if "true_incumbent_binding_missing" in reasons:
        next_change = FIX_BINDING_NEXT_REQUIRED_CHANGE
    elif "no_validated_repaired_candidates" in reasons or "proposal_unvalidated_positive" in reasons:
        next_change = FIX_VALIDATION_NEXT_REQUIRED_CHANGE
    elif "safe_efficient_opportunity_still_zero" in reasons:
        next_change = EXPAND_COMPLEXITY_NEXT_REQUIRED_CHANGE
    else:
        next_change = EXPAND_ROI_NEXT_REQUIRED_CHANGE
    return {"status": "failed", "reason_codes": reasons, "next_required_change": next_change}


def _summary(generated_at: str, config: dict[str, Any], config_path: Path, output_root: Path, repo_root: Path, source: dict[str, Any], metrics: dict[str, Any], decision: dict[str, Any], paths: dict[str, Path]) -> dict[str, Any]:
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": decision["status"],
        "reason_codes": decision["reason_codes"],
        "next_required_change": decision["next_required_change"],
        "candidate_generation_mode": config["candidate_generation_mode"],
        **{key: value for key, value in metrics.items() if key != "reason_codes"},
        "source_bound_coverage_root": config["source_bound_coverage_root"],
        "source_root_cause_audit_root": config["source_root_cause_audit_root"],
        "canary_traffic_fraction": config["canary_traffic_fraction"],
        "config": str(config_path),
        "output_root": str(output_root),
        "summary": str(paths["summary"]),
        "repaired_path_feedback": str(paths["path_feedback"]),
        "git_provenance": git_snapshot(repo_root),
        **_boundary_fields(),
    }


def _repaired_expansion_summary(source_summary: dict[str, Any], summary: dict[str, Any]) -> dict[str, Any]:
    payload = dict(source_summary)
    payload["status"] = summary["status"]
    payload["reason_codes"] = list(summary["reason_codes"])
    payload["next_required_change"] = summary["next_required_change"]
    payload["safe_efficient_opportunity_count"] = summary["safe_efficient_opportunity_count"]
    payload.update(_boundary_fields())
    return payload


def _repaired_materialization_summary(source_summary: dict[str, Any], summary: dict[str, Any]) -> dict[str, Any]:
    payload = dict(source_summary)
    payload["status"] = summary["status"]
    payload["reason_codes"] = list(summary["reason_codes"])
    payload["next_required_change"] = summary["next_required_change"]
    payload["safe_efficient_opportunity_count"] = summary["safe_efficient_opportunity_count"]
    payload["roi_group_with_safe_efficient_opportunity_count"] = summary["roi_group_with_safe_efficient_opportunity_count"]
    payload["candidate_coverage_spread_range"] = summary["cost_efficient_coverage_spread_range"]
    payload.update(_boundary_fields())
    return payload


def _candidate_rows(scenario: dict[str, Any]) -> list[dict[str, Any]]:
    feedback = scenario.get("path_feedback")
    candidates = feedback.get("candidates") if isinstance(feedback, dict) else None
    return [row for row in candidates if isinstance(row, dict)] if isinstance(candidates, list) else []


def _candidate_valid(candidate: dict[str, Any] | None) -> bool:
    return bool(
        candidate
        and candidate.get("reachable") is True
        and candidate.get("open_grid_fallback_used") is not True
        and math.isfinite(_path_cost(candidate))
        and math.isfinite(_risk(candidate))
    )


def _selected_index(value: Any) -> int | None:
    return int(value) if isinstance(value, int) else None


def _candidate_at(candidates: list[dict[str, Any]], index: int | None) -> dict[str, Any] | None:
    if index is None or index < 0 or index >= len(candidates):
        return None
    return candidates[index]


def _coverage(candidate: dict[str, Any] | None) -> float:
    if not candidate:
        return 0.0
    for field in ("roi_weighted_coverage_delta", "expected_coverage_rate_delta", "path_line_coverage_delta"):
        if field in candidate:
            return _float(candidate.get(field))
    return 0.0


def _path_cost(candidate: dict[str, Any] | None) -> float:
    return _float(candidate.get("path_cost")) if candidate else 0.0


def _risk(candidate: dict[str, Any] | None) -> float:
    return _float(candidate.get("risk")) if candidate else 0.0


def _revisit(candidate: dict[str, Any] | None) -> float:
    return _float(candidate.get("revisit_penalty")) if candidate else 0.0


def _cost_efficiency(candidate: dict[str, Any] | None) -> float:
    return _coverage(candidate) / max(_path_cost(candidate) * (1.0 + _risk(candidate)), TOLERANCE)


def _safe_efficient(candidate: dict[str, Any] | None, incumbent: dict[str, Any] | None) -> bool:
    return bool(
        _candidate_valid(candidate)
        and _coverage(candidate) > _coverage(incumbent) + TOLERANCE
        and _path_cost(candidate) <= _path_cost(incumbent) + TOLERANCE
        and _risk(candidate) <= _risk(incumbent) + TOLERANCE
        and _cost_efficiency(candidate) > _cost_efficiency(incumbent) + TOLERANCE
    )


def _pareto_frontier_count(candidates: list[dict[str, Any]]) -> int:
    frontier = 0
    for candidate in candidates:
        if not _candidate_valid(candidate):
            continue
        dominated = False
        for other in candidates:
            if other is candidate or not _candidate_valid(other):
                continue
            weak = _coverage(other) >= _coverage(candidate) - TOLERANCE and _path_cost(other) <= _path_cost(candidate) + TOLERANCE and _risk(other) <= _risk(candidate) + TOLERANCE
            strict = _coverage(other) > _coverage(candidate) + TOLERANCE or _path_cost(other) < _path_cost(candidate) - TOLERANCE or _risk(other) < _risk(candidate) - TOLERANCE
            if weak and strict:
                dominated = True
                break
        if not dominated:
            frontier += 1
    return frontier


def _cell_tuple(value: Any) -> tuple[int, int] | None:
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        try:
            return int(value[0]), int(value[1])
        except (TypeError, ValueError):
            return None
    return None


def _cell_key(value: Any) -> tuple[int, int] | None:
    return _cell_tuple(value)


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


def _render_report(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Xunce Safe-Efficient Candidate Repair",
            "",
            f"- status: `{summary['status']}`",
            f"- next_required_change: `{summary['next_required_change']}`",
            f"- safe_efficient_opportunity_count: `{summary['safe_efficient_opportunity_count']}`",
            f"- roi_group_with_safe_efficient_opportunity_count: `{summary['roi_group_with_safe_efficient_opportunity_count']}`",
            f"- proposal_unvalidated_positive_count: `{summary['proposal_unvalidated_positive_count']}`",
        ]
    )


def _boundary_fields() -> dict[str, bool]:
    return {field: False for field in BOUNDARY_FIELDS}


def _positive_int(value: Any, field: str) -> int:
    if not isinstance(value, int) or value <= 0:
        raise ConfigError(f"{field} must be a positive integer")
    return value


def _positive_float(value: Any, field: str) -> float:
    if not isinstance(value, (int, float)) or float(value) <= 0.0:
        raise ConfigError(f"{field} must be a positive number")
    return float(value)


def _nonnegative_float(value: Any, field: str) -> float:
    if not isinstance(value, (int, float)) or float(value) < 0.0:
        raise ConfigError(f"{field} must be a non-negative number")
    return float(value)


def _string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ConfigError(f"{field} must be a non-empty string")
    return value


def _fractions(value: Any) -> list[float]:
    if not isinstance(value, list):
        raise ConfigError("interpolation_fractions must be a list")
    fractions = []
    for item in value:
        if not isinstance(item, (int, float)) or float(item) <= 0.0 or float(item) >= 1.0:
            raise ConfigError("interpolation_fractions must contain values between 0 and 1")
        fractions.append(float(item))
    return fractions


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def _spread(values: list[float]) -> float:
    return max(values) - min(values) if values else 0.0


if __name__ == "__main__":
    raise SystemExit(main())
