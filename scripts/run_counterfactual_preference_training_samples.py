from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CONFIG_SCHEMA_VERSION = "counterfactual-preference-training-samples-config/v1"
SUMMARY_SCHEMA_VERSION = "counterfactual-preference-training-summary/v1"
EXCLUSION_SCHEMA_VERSION = "counterfactual-preference-exclusion-report/v1"
MINING_SCHEMA_VERSION = "planner-validated-trainable-target-mining-summary/v1"
CANDIDATE_SCHEMA_VERSION = "anchor-projection-candidate-generation-summary/v1"
NOT_SELECTED_DECISION = "rejected_not_source_selected"
PREFERENCE_DECISIONS = {
    "selected_over_alternative_negative",
    "tradeoff_preference_pair",
}


class ConfigError(ValueError):
    pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Mine counterfactual preference samples from not-source-selected anchor-projection contexts."
    )
    parser.add_argument("--batch-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--planner-validated-trainable-target-mining-summary")
    parser.add_argument("--anchor-projection-candidate-generation-summary")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    batch_root = _resolve_path(args.batch_root, repo_root)
    config_path = _resolve_path(args.config, repo_root)
    try:
        config = _load_config(config_path)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    input_files = config["input_files"]
    mining_path = (
        _resolve_path(args.planner_validated_trainable_target_mining_summary, repo_root)
        if args.planner_validated_trainable_target_mining_summary
        else batch_root / input_files["planner_validated_trainable_target_mining_summary"]
    )
    candidate_path = (
        _resolve_path(args.anchor_projection_candidate_generation_summary, repo_root)
        if args.anchor_projection_candidate_generation_summary
        else batch_root / input_files["anchor_projection_candidate_generation_summary"]
    )
    result = mine_counterfactual_preference_samples(
        batch_root=batch_root,
        mining_path=mining_path,
        candidate_path=candidate_path,
        config=config,
        repo_root=repo_root,
    )
    print(
        json.dumps(
            {
                "status": "config validated" if result["summary"]["status"] == "passed" else "validation failed",
                "batch_root": _display_path(batch_root, repo_root),
                "reason_codes": result["summary"]["reason_codes"],
                "source_selection_not_selected_count": result["summary"][
                    "source_selection_not_selected_count"
                ],
                "preference_pair_count": result["summary"]["preference_pair_count"],
                "hard_positive_added_count": result["summary"]["hard_positive_added_count"],
                "summary": _display_path(result["summary_path"], repo_root),
                "samples": _display_path(result["samples_path"], repo_root),
                "exclusion_report": _display_path(result["exclusion_path"], repo_root),
            },
            ensure_ascii=False,
        )
    )
    if args.validate_only or args.dry_run:
        return 1 if result["summary"]["status"] == "failed" else 0
    _write_outputs(result)
    return 1 if result["summary"]["status"] == "failed" else 0


def mine_counterfactual_preference_samples(
    *,
    batch_root: Path,
    mining_path: Path,
    candidate_path: Path,
    config: dict[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    reason_codes: list[str] = []
    mining = _load_source(
        mining_path,
        expected_schema=MINING_SCHEMA_VERSION,
        label="planner_validated_mining_summary",
        reason_codes=reason_codes,
    )
    candidate = _load_source(
        candidate_path,
        expected_schema=CANDIDATE_SCHEMA_VERSION,
        label="anchor_projection_candidate_generation_summary",
        reason_codes=reason_codes,
    )
    validation = config["validation"]
    if validation.get("fail_on_input_failure", True):
        if mining.get("status") != "passed":
            _append_reason(reason_codes, "planner_validated_mining_summary_failed")
        if candidate.get("status") not in (None, "passed"):
            _append_reason(reason_codes, "anchor_projection_candidate_generation_summary_failed")
    if validation.get("fail_on_provenance_mismatch", True):
        if _int_value(mining.get("current_git_provenance_mismatch_count")) > 0 or _int_value(
            mining.get("git_provenance_mismatch_count")
        ) > 0:
            _append_reason(reason_codes, "current_git_provenance_mismatch")
    if validation.get("fail_on_fallback_or_open_grid", True):
        if _int_value(mining.get("fallback_or_open_grid_count")) > 0:
            _append_reason(reason_codes, "fallback_or_open_grid_blocks_counterfactual_preference")
    if validation.get("fail_on_safety_regression", True):
        if _int_value(mining.get("safety_regression_count")) > 0:
            _append_reason(reason_codes, "safety_regression_blocks_counterfactual_preference")
    if _int_value(candidate.get("candidate_contract_alignment_gap_count")) > 0:
        _append_reason(reason_codes, "candidate_contract_alignment_gap_count_nonzero")

    records = _merged_records(mining, candidate)
    target_records = [record for record in records if _decision(record) == NOT_SELECTED_DECISION]
    decisions = [_classify(record) for record in target_records]
    samples = [
        _sample_record(record, decision=decision, sample_index=index, config=config)
        for index, (record, decision) in enumerate(zip(target_records, decisions))
        if decision in PREFERENCE_DECISIONS
    ]
    exclusions = [
        _exclusion_record(record, decision=decision)
        for record, decision in zip(target_records, decisions)
        if decision not in PREFERENCE_DECISIONS
    ]
    counts = Counter(decisions)
    hard_positive_added_count = 0
    expected = config.get("expected_counts", {})
    _check_expected_count(
        reason_codes,
        expected,
        "source_selection_not_selected_count",
        len(target_records),
    )
    _check_expected_count(reason_codes, expected, "preference_pair_count", len(samples))
    _check_expected_count(
        reason_codes,
        expected,
        "selected_over_alternative_negative_count",
        counts["selected_over_alternative_negative"],
    )
    _check_expected_count(
        reason_codes,
        expected,
        "tradeoff_preference_pair_count",
        counts["tradeoff_preference_pair"],
    )
    _check_expected_count(
        reason_codes,
        expected,
        "rejected_binding_or_distance_required_count",
        counts["rejected_binding_or_distance_required"],
    )
    _check_expected_count(
        reason_codes,
        expected,
        "hard_positive_added_count",
        hard_positive_added_count,
    )

    status = "failed" if reason_codes else "passed"
    output_files = config["output_files"]
    samples_path = batch_root / output_files["samples"]
    summary_path = batch_root / output_files["summary"]
    exclusion_path = batch_root / output_files["exclusion_report"]
    generated_at = datetime.now(timezone.utc).isoformat()
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": status,
        "reason_codes": reason_codes,
        "failure_reason_code_counts": dict(sorted(Counter(reason_codes).items())),
        "batch_root": _display_path(batch_root, repo_root),
        "source_summaries": {
            "planner_validated_trainable_target_mining_summary": _source_descriptor(
                mining_path,
                mining,
                repo_root,
            ),
            "anchor_projection_candidate_generation_summary": _source_descriptor(
                candidate_path,
                candidate,
                repo_root,
            ),
        },
        "source_selection_not_selected_count": len(target_records),
        "preference_pair_count": len(samples),
        "selected_over_alternative_negative_count": counts[
            "selected_over_alternative_negative"
        ],
        "tradeoff_preference_pair_count": counts["tradeoff_preference_pair"],
        "rejected_binding_or_distance_required_count": counts[
            "rejected_binding_or_distance_required"
        ],
        "rejected_quality_regression_count": counts["rejected_quality_regression"],
        "hard_positive_added_count": hard_positive_added_count,
        "samples": _display_path(samples_path, repo_root),
        "exclusion_report": _display_path(exclusion_path, repo_root),
        "runs_training": False,
        "dry_run_only": False,
        "publishes_checkpoint": False,
        "performance_claimed": False,
        "does_not_modify_default_astar": True,
        "does_not_modify_ppo": True,
        "does_not_modify_network": True,
        "does_not_modify_action_space": True,
        "does_not_relax_default_distance_contract": True,
        "no_ackermann_feasible_trajectory_claim": True,
        "non_goals": list(config.get("non_goals", [])),
    }
    exclusion_report = {
        "schema_version": EXCLUSION_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": status,
        "excluded_count": len(exclusions),
        "excluded_decision_counts": dict(sorted(Counter(item["preference_decision"] for item in exclusions).items())),
        "excluded_records": exclusions,
    }
    return {
        "summary": summary,
        "samples": samples,
        "exclusion_report": exclusion_report,
        "samples_path": samples_path,
        "summary_path": summary_path,
        "exclusion_path": exclusion_path,
    }


def _classify(record: dict[str, Any]) -> str:
    if record.get("source_selection_quality_regression") is True:
        return "rejected_quality_regression"
    if record.get("contract_safe") is not True:
        return "rejected_binding_or_distance_required"
    path_margin = _float_value(record.get("source_selection_path_cost_margin_vs_best_alternative"))
    risk_margin = _float_value(record.get("source_selection_risk_margin_vs_best_alternative"))
    if path_margin is None or risk_margin is None:
        return "rejected_binding_or_distance_required"
    if path_margin > 0.0 and risk_margin < 0.0:
        return "tradeoff_preference_pair"
    return "selected_over_alternative_negative"


def _sample_record(
    record: dict[str, Any],
    *,
    decision: str,
    sample_index: int,
    config: dict[str, Any],
) -> dict[str, Any]:
    weights = config["preference"]
    weight = (
        float(weights.get("tradeoff_weight", 0.35))
        if decision == "tradeoff_preference_pair"
        else float(weights.get("selected_over_alternative_weight", 1.0))
    )
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "sample_index": sample_index,
        "context_id": record.get("context_id"),
        "context_id_schema_version": record.get("context_id_schema_version"),
        "context_id_source": record.get("context_id_source"),
        "legacy_identity_fallback_used": record.get("legacy_identity_fallback_used"),
        "run_id": record.get("run_id"),
        "scenario_id": record.get("scenario_id"),
        "scenario_group": record.get("scenario_group"),
        "scenario_seed": record.get("scenario_seed"),
        "scenario_variant_id": record.get("scenario_variant_id"),
        "diagnostic_profile": record.get("diagnostic_profile"),
        "planning_backend": record.get("planning_backend"),
        "top_k": record.get("top_k"),
        "preference_decision": decision,
        "hard_positive": False,
        "sample_weight": weight,
        "selected": {
            "action_index": record.get("selected_action_index"),
            "cell": _list_cell(record.get("selected_cell")),
            "path_cost": _float_value(record.get("selected_candidate_path_cost")),
            "risk": _float_value(record.get("selected_candidate_risk")),
            "utility": _float_value(record.get("selected_candidate_utility")),
            "candidate_features": _candidate_features(
                cell=_list_cell(record.get("selected_cell")),
                utility=_float_value(record.get("selected_candidate_utility")),
                path_cost=_float_value(record.get("selected_candidate_path_cost")),
                risk=_float_value(record.get("selected_candidate_risk")),
            ),
        },
        "alternative": {
            "source_action_index": record.get("source_action_index"),
            "policy_target_cell": _list_cell(record.get("policy_target_cell")),
            "execution_goal_cell": _list_cell(record.get("execution_goal_cell")),
            "target_binding_mode": record.get("target_binding_mode"),
            "ppo_consumable_action": bool(record.get("ppo_consumable_action")),
            "contract_safe": bool(record.get("contract_safe")),
            "planner_validated_distance_exception": bool(
                record.get("planner_validated_distance_exception")
            ),
            "path_cost": _float_value(record.get("projected_candidate_path_cost")),
            "risk": _float_value(record.get("projected_candidate_risk")),
            "utility": _float_value(record.get("projected_candidate_utility")),
            "candidate_features": _candidate_features(
                cell=_list_cell(record.get("policy_target_cell")),
                utility=_float_value(record.get("projected_candidate_utility")),
                path_cost=_float_value(record.get("projected_candidate_path_cost")),
                risk=_float_value(record.get("projected_candidate_risk")),
            ),
        },
        "margins": {
            "path_cost_margin_vs_best_alternative": _float_value(
                record.get("source_selection_path_cost_margin_vs_best_alternative")
            ),
            "risk_margin_vs_best_alternative": _float_value(
                record.get("source_selection_risk_margin_vs_best_alternative")
            ),
            "projection_distance_cells": _float_value(record.get("projection_distance_cells")),
            "projection_distance_m": _float_value(record.get("projection_distance_m")),
        },
        "global_features": _global_features(),
        "candidate_missing_indicators": _missing_indicators(),
        "selection_contract": {
            "source_selection_status": record.get("source_selection_status"),
            "final_training_decision": _decision(record),
            "not_source_selected_is_not_hard_positive": True,
        },
    }


def _exclusion_record(record: dict[str, Any], *, decision: str) -> dict[str, Any]:
    return {
        "run_id": record.get("run_id"),
        "scenario_id": record.get("scenario_id"),
        "source_action_index": record.get("source_action_index"),
        "policy_target_cell": _list_cell(record.get("policy_target_cell")),
        "execution_goal_cell": _list_cell(record.get("execution_goal_cell")),
        "preference_decision": decision,
        "contract_safe": bool(record.get("contract_safe")),
        "ppo_consumable_action": bool(record.get("ppo_consumable_action")),
        "target_binding_mode": record.get("target_binding_mode"),
        "projection_distance_cells": record.get("projection_distance_cells"),
        "projection_distance_m": record.get("projection_distance_m"),
    }


def _merged_records(mining: dict[str, Any], candidate: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = {
        _record_key(record): record
        for record in candidate.get("context_records", [])
        if isinstance(record, dict)
    }
    records = []
    for record in mining.get("final_decision_records", []):
        if not isinstance(record, dict):
            continue
        merged = dict(candidates.get(_record_key(record), {}))
        merged.update(record)
        records.append(merged)
    return records


def _record_key(record: dict[str, Any]) -> tuple[Any, ...]:
    return (
        record.get("run_id"),
        record.get("scenario_id"),
        _int_value(record.get("source_action_index")),
        tuple(_list_cell(record.get("policy_target_cell")) or []),
        tuple(_list_cell(record.get("execution_goal_cell")) or []),
    )


def _candidate_features(
    *,
    cell: list[int] | None,
    utility: float | None,
    path_cost: float | None,
    risk: float | None,
) -> list[float]:
    if cell is None:
        return [0.0 for _ in range(15)]
    x, y = cell
    return [
        _clip01(x / 100.0),
        _clip01(y / 100.0),
        _clip_signed(x / 100.0),
        _clip_signed(y / 100.0),
        _clip01(((x * x + y * y) ** 0.5) / 141.5),
        _clip01(utility or 0.0),
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        _clip01(utility or 0.0),
        _clip01(risk or 0.0),
        _clip01((path_cost or 0.0) / 100.0),
        _clip01((path_cost or 0.0) / 100.0),
    ]


def _global_features() -> list[float]:
    return [0.0 for _ in range(8)]


def _missing_indicators() -> list[list[float]]:
    return [[0.0 for _ in range(8)], [0.0 for _ in range(8)]]


def _write_outputs(result: dict[str, Any]) -> None:
    result["summary_path"].parent.mkdir(parents=True, exist_ok=True)
    result["samples_path"].write_text(
        "".join(json.dumps(sample, ensure_ascii=False) + "\n" for sample in result["samples"]),
        encoding="utf-8",
    )
    result["summary_path"].write_text(
        json.dumps(result["summary"], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    result["exclusion_path"].write_text(
        json.dumps(result["exclusion_report"], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _load_config(path: Path) -> dict[str, Any]:
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
    for section in ("input_files", "output_files", "validation", "expected_counts", "preference"):
        if not isinstance(payload.get(section), dict):
            raise ConfigError(f"{section} must be an object")
    return payload


def _load_source(
    path: Path,
    *,
    expected_schema: str,
    label: str,
    reason_codes: list[str],
) -> dict[str, Any]:
    if not path.is_file():
        _append_reason(reason_codes, f"{label}_missing")
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        _append_reason(reason_codes, f"{label}_invalid_json")
        return {}
    if not isinstance(payload, dict):
        _append_reason(reason_codes, f"{label}_invalid_json_root")
        return {}
    if payload.get("schema_version") != expected_schema:
        _append_reason(reason_codes, f"{label}_schema_version_mismatch")
    return payload


def _source_descriptor(path: Path, payload: dict[str, Any], repo_root: Path) -> dict[str, Any]:
    return {
        "path": _display_path(path, repo_root),
        "exists": path.is_file(),
        "schema_version": payload.get("schema_version"),
        "status": payload.get("status"),
    }


def _check_expected_count(
    reason_codes: list[str],
    expected: dict[str, Any],
    field: str,
    actual: int,
) -> None:
    if field in expected and actual != _int_value(expected.get(field)):
        _append_reason(reason_codes, f"{field}_mismatch")


def _resolve_path(value: str | Path, repo_root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def _display_path(path: Path, repo_root: Path) -> str:
    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        return str(path)


def _decision(record: dict[str, Any]) -> str:
    return str(record.get("final_training_decision") or "")


def _append_reason(reasons: list[str], reason: str) -> None:
    if reason not in reasons:
        reasons.append(reason)


def _int_value(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _float_value(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _list_cell(value: Any) -> list[int] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    return [int(value[0]), int(value[1])]


def _clip01(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


def _clip_signed(value: float) -> float:
    return min(1.0, max(-1.0, float(value)))


if __name__ == "__main__":
    raise SystemExit(main())
