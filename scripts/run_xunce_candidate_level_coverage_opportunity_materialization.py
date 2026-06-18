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


CONFIG_SCHEMA_VERSION = "xunce-candidate-level-coverage-opportunity-materialization-config/v1"
SUMMARY_SCHEMA_VERSION = "xunce-candidate-level-coverage-opportunity-summary/v1"
DEFAULT_CONFIG = "configs/xunce_candidate_level_coverage_opportunity_materialization_v1.json"
DEFAULT_OUTPUT_ROOT = "outputs/path_feedback_batch_xunce_candidate_level_coverage_opportunity_materialization_v1"

EXPANSION_SUMMARY_FILE = "xunce-high-fidelity-real-map-roi-expansion-summary.json"
EXPANSION_SLICES_FILE = "xunce-high-fidelity-real-map-slices.jsonl"
PATH_FEEDBACK_AUDIT_FILE = "xunce-high-fidelity-path-feedback-audit.json"

SUMMARY_FILE = "xunce-candidate-level-coverage-opportunity-summary.json"
OVERLAY_FILE = "xunce-candidate-coverage-overlay.jsonl"
SPREAD_FILE = "xunce-candidate-coverage-spread-by-scenario.jsonl"
ROI_SUMMARY_FILE = "xunce-candidate-coverage-roi-summary.json"
PATH_FEEDBACK_AUDIT_OUT_FILE = "xunce-candidate-level-path-feedback-audit.json"
MANIFEST_FILE = "xunce-candidate-level-coverage-opportunity-manifest.json"
REPORT_FILE = "xunce-candidate-level-coverage-opportunity-report.md"

PASS_NEXT_REQUIRED_CHANGE = "run_oracle_separability_benchmark"
FIX_STAGE18A_NEXT_REQUIRED_CHANGE = "run_xunce_high_fidelity_real_map_roi_expansion"
FIX_INPUT_NEXT_REQUIRED_CHANGE = "repair_candidate_materialization_inputs"
FIX_MATERIALIZATION_NEXT_REQUIRED_CHANGE = "repair_candidate_level_coverage_materialization"
FIX_ROI_NEXT_REQUIRED_CHANGE = "expand_roi_or_refine_roi_weighting"

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
    parser = argparse.ArgumentParser(description="Materialize candidate-level Xunce coverage opportunities.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--repo-root")
    parser.add_argument("--source-roi-expansion-root")
    parser.add_argument("--source-coverage-comparison-root")
    args = parser.parse_args(argv)
    repo_root = Path(args.repo_root).resolve() if args.repo_root else Path(__file__).resolve().parents[1]
    overrides = {
        key: value
        for key, value in {
            "source_roi_expansion_root": args.source_roi_expansion_root,
            "source_coverage_comparison_root": args.source_coverage_comparison_root,
        }.items()
        if value is not None
    }
    try:
        summary = run_xunce_candidate_level_coverage_opportunity_materialization(
            config_path=resolve_path(Path(args.config), repo_root),
            output_root=resolve_path(Path(args.output_root), repo_root),
            repo_root=repo_root,
            config_overrides=overrides,
        )
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"status": summary["status"], "reason_codes": summary["reason_codes"], "next_required_change": summary["next_required_change"], "summary": summary["summary"]}, ensure_ascii=False, sort_keys=True))
    return 0 if summary["status"] == "passed" else 1


def run_xunce_candidate_level_coverage_opportunity_materialization(
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
    enriched, overlay_rows, spread_rows, roi_summary, metrics = _materialize(config, source)
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
    manifest = {
        "schema_version": "xunce-candidate-level-coverage-opportunity-manifest/v1",
        "generated_at": generated_at,
        "config": str(config_path),
        "output_root": str(output_root),
        "source_roi_expansion_root": config["source_roi_expansion_root"],
        "source_coverage_comparison_root": config["source_coverage_comparison_root"],
        "artifacts": {key: str(value) for key, value in paths.items()},
        **_boundary_fields(),
    }

    write_json(paths["summary"], summary)
    write_jsonl(paths["overlay"], overlay_rows)
    write_jsonl(paths["spread"], spread_rows)
    write_json(paths["roi_summary"], roi_summary)
    write_json(paths["path_feedback_audit"], _candidate_level_audit(source, metrics, decision))
    write_json(paths["manifest"], manifest)
    write_json(paths["expansion_summary"], _enriched_expansion_summary(source["expansion_summary"], summary))
    write_jsonl(paths["slices"], source["slices"])
    write_json(paths["enriched_path_feedback"], enriched)
    paths["report"].write_text(_render_report(summary), encoding="utf-8")
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
    for key in ("source_roi_expansion_root", "source_coverage_comparison_root"):
        value = payload.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ConfigError(f"{key} must be a non-empty string")
        normalized[key] = str(resolve_path(Path(value), repo_root))
    normalized["required_scenario_count"] = _positive_int(payload.get("required_scenario_count", 24), "required_scenario_count")
    normalized["rollout_steps"] = _positive_int(payload.get("rollout_steps", 10), "rollout_steps")
    normalized["coverage_radius_cells"] = _nonnegative_int(payload.get("coverage_radius_cells", 1), "coverage_radius_cells")
    normalized["coverage_denominator_cells"] = _positive_int(payload.get("coverage_denominator_cells", 1000), "coverage_denominator_cells")
    normalized["min_candidate_coverage_spread"] = _nonnegative_float(payload.get("min_candidate_coverage_spread", 0.005), "min_candidate_coverage_spread")
    normalized["min_roi_group_with_nonzero_spread_count"] = _positive_int(payload.get("min_roi_group_with_nonzero_spread_count", 3), "min_roi_group_with_nonzero_spread_count")
    normalized["canary_traffic_fraction"] = _nonnegative_float(payload.get("canary_traffic_fraction", 0.0), "canary_traffic_fraction")
    return normalized


def _load_source(config: dict[str, Any]) -> dict[str, Any]:
    root = Path(config["source_roi_expansion_root"])
    coverage_root = Path(config["source_coverage_comparison_root"])
    reasons: list[str] = []
    expansion_summary = _read_json(root / EXPANSION_SUMMARY_FILE, reasons, "missing_stage18a_roi_expansion")
    slices = _read_jsonl(root / EXPANSION_SLICES_FILE, reasons, "missing_stage18a_roi_expansion_slices")
    path_feedback = _read_json(root / PATH_FEEDBACK_AUDIT_FILE, reasons, "missing_stage18a_path_feedback")
    coverage_summary = _read_json(coverage_root / "xunce-exploration-coverage-comparison-summary.json", [], "missing_stage18c_coverage_summary")
    scenarios = path_feedback.get("scenarios") if isinstance(path_feedback.get("scenarios"), list) else []
    return {
        "root": root,
        "coverage_root": coverage_root,
        "expansion_summary": expansion_summary,
        "slices": [row for row in slices if isinstance(row, dict)],
        "path_feedback": path_feedback,
        "coverage_summary": coverage_summary,
        "scenarios": [row for row in scenarios if isinstance(row, dict)],
        "reason_codes": unique_sorted(reasons),
    }


def _paths(output_root: Path) -> dict[str, Path]:
    return {
        "summary": output_root / SUMMARY_FILE,
        "overlay": output_root / OVERLAY_FILE,
        "spread": output_root / SPREAD_FILE,
        "roi_summary": output_root / ROI_SUMMARY_FILE,
        "path_feedback_audit": output_root / PATH_FEEDBACK_AUDIT_OUT_FILE,
        "enriched_path_feedback": output_root / PATH_FEEDBACK_AUDIT_FILE,
        "expansion_summary": output_root / EXPANSION_SUMMARY_FILE,
        "slices": output_root / EXPANSION_SLICES_FILE,
        "manifest": output_root / MANIFEST_FILE,
        "report": output_root / REPORT_FILE,
    }


def _materialize(config: dict[str, Any], source: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    enriched = dict(source["path_feedback"])
    enriched_scenarios: list[dict[str, Any]] = []
    overlay_rows: list[dict[str, Any]] = []
    spread_rows: list[dict[str, Any]] = []
    roi_spreads: dict[str, list[float]] = defaultdict(list)
    reason_codes: list[str] = list(source["reason_codes"])
    nonzero_count = 0
    useful_opportunities = 0
    pareto_counts: list[int] = []
    reachable_count = 0
    candidate_count = 0
    geometry_missing_count = 0

    for scenario_index, scenario in enumerate(source["scenarios"][: config["required_scenario_count"]]):
        scenario_copy = dict(scenario)
        scenario_id = str(scenario_copy.get("scenario_id") or f"scenario-{scenario_index:04d}")
        roi_group = str(scenario_copy.get("roi_group") or scenario_copy.get("scenario_group") or "unknown")
        start_cell = _cell_tuple(scenario_copy.get("start_cell")) or _cell_tuple(scenario_copy.get("selected_cell_before_path_feedback")) or (0, 0)
        candidates = _candidate_rows(scenario_copy)
        enriched_candidates: list[dict[str, Any]] = []
        coverage_values: list[float] = []
        valid_candidates: list[dict[str, Any]] = []
        covered = set(_footprint(start_cell, radius=int(config["coverage_radius_cells"])))
        for action_index, candidate in enumerate(candidates):
            candidate_count += 1
            enriched_candidate = dict(candidate)
            valid, missing = _candidate_validity(enriched_candidate)
            if missing:
                geometry_missing_count += 1
            if valid:
                reachable_count += 1
                endpoint_cells = set(_footprint(_cell_tuple(enriched_candidate["cell"]) or start_cell, radius=int(config["coverage_radius_cells"])))
                path_cells = _coverage_cells(start=start_cell, end=_cell_tuple(enriched_candidate["cell"]) or start_cell, radius=int(config["coverage_radius_cells"]))
                endpoint_new = endpoint_cells - covered
                path_new = path_cells - covered
                overlap = path_cells & covered
                endpoint_delta = len(endpoint_new) / float(config["coverage_denominator_cells"])
                path_delta = len(path_new) / float(config["coverage_denominator_cells"])
                roi_weight = _roi_weight(roi_group)
                roi_weighted = path_delta * roi_weight
                revisit_penalty = len(overlap) / float(config["coverage_denominator_cells"])
                enriched_candidate.update(
                    {
                        "endpoint_coverage_delta": endpoint_delta,
                        "path_line_coverage_delta": path_delta,
                        "expected_coverage_rate_delta": path_delta,
                        "expected_new_coverage_cell_count": len(path_new),
                        "expected_new_coverage_area": float(len(path_new)),
                        "roi_weighted_coverage_delta": roi_weighted,
                        "revisit_penalty": revisit_penalty,
                        "coverage_gain_per_path_cost": _safe_ratio(path_delta, _float(enriched_candidate.get("path_cost"))),
                        "coverage_gain_per_risk": _safe_ratio(path_delta, _float(enriched_candidate.get("risk"))),
                        "coverage_opportunity_source": "geometric_counterfactual_from_stage18a_candidate/v1",
                    }
                )
                coverage_values.append(roi_weighted)
                valid_candidates.append(enriched_candidate)
                if roi_weighted > TOLERANCE:
                    nonzero_count += 1
            else:
                enriched_candidate.update(
                    {
                        "endpoint_coverage_delta": 0.0,
                        "path_line_coverage_delta": 0.0,
                        "expected_coverage_rate_delta": 0.0,
                        "expected_new_coverage_cell_count": 0,
                        "roi_weighted_coverage_delta": 0.0,
                        "revisit_penalty": 0.0,
                        "coverage_gain_per_path_cost": 0.0,
                        "coverage_gain_per_risk": 0.0,
                        "coverage_opportunity_source": "geometric_counterfactual_from_stage18a_candidate/v1",
                    }
                )
            enriched_candidates.append(enriched_candidate)
        _rank_candidates(enriched_candidates)
        spread = _spread_range(coverage_values)
        pareto_count = _pareto_frontier_count(valid_candidates)
        pareto_counts.append(pareto_count)
        roi_spreads[roi_group].append(spread)
        if spread > float(config["min_candidate_coverage_spread"]) and pareto_count > 1:
            useful_opportunities += 1
        spread_rows.append(
            {
                "schema_version": "xunce-candidate-coverage-spread-by-scenario/v1",
                "scenario_id": scenario_id,
                "roi_group": roi_group,
                "candidate_count": len(candidates),
                "reachable_candidate_count": len(valid_candidates),
                "nonzero_coverage_candidate_count": sum(1 for value in coverage_values if value > TOLERANCE),
                "candidate_coverage_spread_range": spread,
                "candidate_coverage_spread_std": _std(coverage_values),
                "candidate_pareto_frontier_count": pareto_count,
            }
        )
        for candidate in enriched_candidates:
            overlay_rows.append(
                {
                    "schema_version": "xunce-candidate-coverage-overlay-row/v1",
                    "scenario_id": scenario_id,
                    "roi_group": roi_group,
                    "action_index": candidate.get("action_index"),
                    "cell": candidate.get("cell"),
                    "reachable": candidate.get("reachable"),
                    "expected_coverage_rate_delta": candidate["expected_coverage_rate_delta"],
                    "path_line_coverage_delta": candidate["path_line_coverage_delta"],
                    "roi_weighted_coverage_delta": candidate["roi_weighted_coverage_delta"],
                    "revisit_penalty": candidate["revisit_penalty"],
                    "coverage_opportunity_rank": candidate.get("coverage_opportunity_rank"),
                    "coverage_opportunity_source": candidate["coverage_opportunity_source"],
                }
            )
        scenario_copy["path_feedback"] = dict(scenario_copy.get("path_feedback") or {})
        scenario_copy["path_feedback"]["candidates"] = enriched_candidates
        enriched_scenarios.append(scenario_copy)

    if geometry_missing_count:
        reason_codes.append("candidate_geometry_insufficient")
    roi_rows = [
        {
            "roi_group": roi,
            "scenario_count": len(values),
            "max_candidate_coverage_spread": max(values) if values else 0.0,
            "mean_candidate_coverage_spread": _mean(values),
            "nonzero_spread": any(value > TOLERANCE for value in values),
        }
        for roi, values in sorted(roi_spreads.items())
    ]
    roi_summary = {
        "schema_version": "xunce-candidate-coverage-roi-summary/v1",
        "roi_group_count": len(roi_rows),
        "roi_group_with_nonzero_spread_count": sum(1 for row in roi_rows if row["nonzero_spread"]),
        "families": roi_rows,
    }
    metrics = {
        "reason_codes": unique_sorted(reason_codes),
        "scenario_count": len(enriched_scenarios),
        "candidate_count": candidate_count,
        "reachable_candidate_count": reachable_count,
        "nonzero_coverage_candidate_count": nonzero_count,
        "candidate_coverage_spread_range": _mean([row["candidate_coverage_spread_range"] for row in spread_rows]),
        "candidate_coverage_spread_std": _mean([row["candidate_coverage_spread_std"] for row in spread_rows]),
        "candidate_pareto_frontier_count": _mean(pareto_counts),
        "useful_disagreement_opportunity_count": useful_opportunities,
        "roi_group_with_nonzero_spread_count": roi_summary["roi_group_with_nonzero_spread_count"],
        "candidate_geometry_missing_count": geometry_missing_count,
    }
    enriched["scenarios"] = enriched_scenarios
    enriched["scenario_count"] = len(enriched_scenarios)
    enriched["candidate_count"] = candidate_count
    enriched["reachable_count"] = reachable_count
    enriched["coverage_opportunity_source"] = "geometric_counterfactual_from_stage18a_candidate/v1"
    return enriched, overlay_rows, spread_rows, roi_summary, metrics


def _decision(config: dict[str, Any], source: dict[str, Any], metrics: dict[str, Any]) -> dict[str, Any]:
    reasons = list(metrics["reason_codes"])
    if source["expansion_summary"].get("status") != "passed":
        reasons.append("missing_stage18a_roi_expansion")
    if metrics["candidate_geometry_missing_count"]:
        reasons.append("candidate_geometry_insufficient")
    if metrics["candidate_coverage_spread_range"] <= float(config["min_candidate_coverage_spread"]):
        reasons.append("candidate_coverage_spread_insufficient")
    if metrics["roi_group_with_nonzero_spread_count"] < int(config["min_roi_group_with_nonzero_spread_count"]):
        reasons.append("roi_spread_insufficient")
    if metrics["useful_disagreement_opportunity_count"] <= 0 or metrics["nonzero_coverage_candidate_count"] <= 0 or metrics["candidate_pareto_frontier_count"] <= 1.0:
        reasons.append("candidate_coverage_spread_insufficient")
    reasons = unique_sorted(reasons)
    if not reasons:
        return {"status": "passed", "reason_codes": [], "next_required_change": PASS_NEXT_REQUIRED_CHANGE}
    if "missing_stage18a_roi_expansion" in reasons or any(reason.startswith("missing_stage18a") for reason in reasons):
        next_change = FIX_STAGE18A_NEXT_REQUIRED_CHANGE
    elif "candidate_geometry_insufficient" in reasons:
        next_change = FIX_INPUT_NEXT_REQUIRED_CHANGE
    else:
        if "candidate_coverage_spread_insufficient" in reasons:
            next_change = FIX_MATERIALIZATION_NEXT_REQUIRED_CHANGE
        else:
            next_change = FIX_ROI_NEXT_REQUIRED_CHANGE
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
        **{key: value for key, value in metrics.items() if key != "reason_codes"},
        "coverage_opportunity_source": "geometric_counterfactual_from_stage18a_candidate/v1",
        "source_roi_expansion_root": config["source_roi_expansion_root"],
        "source_coverage_comparison_root": config["source_coverage_comparison_root"],
        "source_roi_expansion_status": source["expansion_summary"].get("status"),
        "config": str(config_path),
        "output_root": str(output_root),
        "summary": str(paths["summary"]),
        "enriched_path_feedback": str(paths["enriched_path_feedback"]),
        "canary_traffic_fraction": config["canary_traffic_fraction"],
        "git_provenance": git_snapshot(repo_root),
        **_boundary_fields(),
    }


def _candidate_level_audit(source: dict[str, Any], metrics: dict[str, Any], decision: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "xunce-candidate-level-path-feedback-audit/v1",
        "status": decision["status"],
        "reason_codes": decision["reason_codes"],
        "source_scenario_count": len(source["scenarios"]),
        "candidate_count": metrics["candidate_count"],
        "reachable_candidate_count": metrics["reachable_candidate_count"],
        "nonzero_coverage_candidate_count": metrics["nonzero_coverage_candidate_count"],
        "coverage_opportunity_source": "geometric_counterfactual_from_stage18a_candidate/v1",
        **_boundary_fields(),
    }


def _enriched_expansion_summary(source_summary: dict[str, Any], summary: dict[str, Any]) -> dict[str, Any]:
    payload = dict(source_summary)
    payload["status"] = summary["status"]
    payload["reason_codes"] = list(summary["reason_codes"])
    payload["next_required_change"] = summary["next_required_change"]
    payload["candidate_coverage_spread_range"] = summary["candidate_coverage_spread_range"]
    payload["roi_group_with_nonzero_spread_count"] = summary["roi_group_with_nonzero_spread_count"]
    payload.update(_boundary_fields())
    return payload


def _candidate_rows(scenario: dict[str, Any]) -> list[dict[str, Any]]:
    feedback = scenario.get("path_feedback")
    candidates = feedback.get("candidates") if isinstance(feedback, dict) else None
    if isinstance(candidates, list):
        return [row for row in candidates if isinstance(row, dict)]
    return []


def _candidate_validity(candidate: dict[str, Any]) -> tuple[bool, bool]:
    required = ("cell", "path_cost", "risk", "reachable")
    missing = any(key not in candidate for key in required)
    valid = (
        not missing
        and candidate.get("reachable") is True
        and candidate.get("open_grid_fallback_used") is not True
        and _cell_tuple(candidate.get("cell")) is not None
        and math.isfinite(_float(candidate.get("path_cost")))
        and math.isfinite(_float(candidate.get("risk")))
    )
    return valid, missing


def _rank_candidates(candidates: list[dict[str, Any]]) -> None:
    ranked = sorted(
        enumerate(candidates),
        key=lambda item: (
            float(item[1].get("roi_weighted_coverage_delta", 0.0)),
            float(item[1].get("path_line_coverage_delta", 0.0)),
            -float(item[1].get("path_cost", 0.0) or 0.0),
        ),
        reverse=True,
    )
    for rank, (index, _candidate) in enumerate(ranked, start=1):
        candidates[index]["coverage_opportunity_rank"] = rank


def _pareto_frontier_count(candidates: list[dict[str, Any]]) -> int:
    frontier = 0
    for candidate in candidates:
        coverage = float(candidate.get("roi_weighted_coverage_delta", 0.0))
        cost = _float(candidate.get("path_cost"))
        risk = _float(candidate.get("risk"))
        dominated = False
        for other in candidates:
            if other is candidate:
                continue
            other_coverage = float(other.get("roi_weighted_coverage_delta", 0.0))
            other_cost = _float(other.get("path_cost"))
            other_risk = _float(other.get("risk"))
            if other_coverage >= coverage - TOLERANCE and other_cost <= cost + TOLERANCE and other_risk <= risk + TOLERANCE:
                if other_coverage > coverage + TOLERANCE or other_cost < cost - TOLERANCE or other_risk < risk - TOLERANCE:
                    dominated = True
                    break
        if not dominated:
            frontier += 1
    return frontier


def _coverage_cells(*, start: tuple[int, int], end: tuple[int, int], radius: int) -> set[tuple[int, int]]:
    cells = set(_footprint(end, radius=radius))
    for cell in _line_cells(start, end):
        cells.update(_footprint(cell, radius=radius))
    return cells


def _footprint(cell: tuple[int, int], *, radius: int) -> set[tuple[int, int]]:
    return {(cell[0] + dx, cell[1] + dy) for dx in range(-radius, radius + 1) for dy in range(-radius, radius + 1)}


def _line_cells(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    x0, y0 = start
    x1, y1 = end
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    cells: list[tuple[int, int]] = []
    while True:
        cells.append((x0, y0))
        if x0 == x1 and y0 == y1:
            break
        error2 = 2 * err
        if error2 > -dy:
            err -= dy
            x0 += sx
        if error2 < dx:
            err += dx
            y0 += sy
    return cells


def _roi_weight(roi_group: str) -> float:
    return 1.0 + (sum(ord(ch) for ch in roi_group) % 5) * 0.05


def _cell_tuple(value: Any) -> tuple[int, int] | None:
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        try:
            return int(value[0]), int(value[1])
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


def _render_report(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Xunce Candidate-Level Coverage Opportunity Materialization",
            "",
            f"- status: `{summary['status']}`",
            f"- next_required_change: `{summary['next_required_change']}`",
            f"- candidate_coverage_spread_range: `{summary['candidate_coverage_spread_range']}`",
            f"- roi_group_with_nonzero_spread_count: `{summary['roi_group_with_nonzero_spread_count']}`",
            f"- useful_disagreement_opportunity_count: `{summary['useful_disagreement_opportunity_count']}`",
        ]
    )


def _boundary_fields() -> dict[str, bool]:
    return {field: False for field in BOUNDARY_FIELDS}


def _positive_int(value: Any, field: str) -> int:
    if not isinstance(value, int) or value <= 0:
        raise ConfigError(f"{field} must be a positive integer")
    return value


def _nonnegative_int(value: Any, field: str) -> int:
    if not isinstance(value, int) or value < 0:
        raise ConfigError(f"{field} must be a non-negative integer")
    return value


def _nonnegative_float(value: Any, field: str) -> float:
    if not isinstance(value, (int, float)) or float(value) < 0.0:
        raise ConfigError(f"{field} must be a non-negative number")
    return float(value)


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def _std(values: list[float]) -> float:
    return statistics.pstdev(values) if len(values) > 1 else 0.0


def _spread_range(values: list[float]) -> float:
    return max(values) - min(values) if values else 0.0


def _safe_ratio(numerator: Any, denominator: Any) -> float:
    den = _float(denominator)
    if abs(den) <= TOLERANCE:
        return 0.0
    return _float(numerator) / den


if __name__ == "__main__":
    raise SystemExit(main())
