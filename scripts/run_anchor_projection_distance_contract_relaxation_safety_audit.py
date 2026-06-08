from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from math import isfinite
from pathlib import Path
from typing import Any

from git_provenance import git_snapshot as _git_snapshot
from git_provenance import inspect_source_git_provenance as _inspect_source_git_provenance
from git_provenance import public_git as _public_git


CONFIG_SCHEMA_VERSION = "anchor-projection-distance-contract-relaxation-safety-audit-config/v1"
SUMMARY_SCHEMA_VERSION = "anchor-projection-distance-contract-relaxation-safety-audit-summary/v1"
ANCHOR_CANDIDATE_SCHEMA_VERSION = "anchor-projection-candidate-generation-summary/v1"
ANCHOR_CONTRACT_SCHEMA_VERSION = "anchor-projection-evidence-contract-summary/v1"
READINESS_SCHEMA_VERSION = "policy-training-readiness-review-summary/v1"
DISTANCE_REJECT_REASONS = {
    "projection_distance_cells_exceeds_contract",
    "projection_distance_m_exceeds_contract",
}


class ConfigError(ValueError):
    pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit whether anchor projection distance contract relaxation is safe."
    )
    parser.add_argument("--batch-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--anchor-projection-candidate-generation-summary",
        help="Defaults to <batch-root>/anchor-projection-candidate-generation-summary.json.",
    )
    parser.add_argument(
        "--anchor-projection-evidence-contract-summary",
        help="Defaults to <batch-root>/anchor-projection-evidence-contract-summary.json.",
    )
    parser.add_argument(
        "--policy-training-readiness-review-summary",
        help="Defaults to <batch-root>/policy-training-readiness-review-summary.json.",
    )
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

    candidate_path = (
        _resolve_path(args.anchor_projection_candidate_generation_summary, repo_root)
        if args.anchor_projection_candidate_generation_summary
        else batch_root / "anchor-projection-candidate-generation-summary.json"
    )
    contract_path = (
        _resolve_path(args.anchor_projection_evidence_contract_summary, repo_root)
        if args.anchor_projection_evidence_contract_summary
        else batch_root / "anchor-projection-evidence-contract-summary.json"
    )
    readiness_path = (
        _resolve_path(args.policy_training_readiness_review_summary, repo_root)
        if args.policy_training_readiness_review_summary
        else batch_root / "policy-training-readiness-review-summary.json"
    )

    summary = analyze_distance_contract_relaxation_safety(
        batch_root=batch_root,
        candidate_path=candidate_path,
        contract_path=contract_path,
        readiness_path=readiness_path,
        config=config,
        repo_root=repo_root,
    )
    output_file = batch_root / config["output_files"]["distance_contract_relaxation_safety_audit_summary"]
    print(
        json.dumps(
            {
                "status": "config validated" if summary["status"] == "passed" else "validation failed",
                "batch_root": _display_path(batch_root, repo_root),
                "reason_codes": summary["reason_codes"],
                "recommendation": summary["recommendation"],
                "distance_contract_relaxation_safety_audit_summary": _display_path(
                    output_file,
                    repo_root,
                ),
            },
            ensure_ascii=False,
        )
    )
    if args.validate_only or args.dry_run:
        return 1 if summary["status"] == "failed" else 0
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return 1 if summary["status"] == "failed" else 0


def analyze_distance_contract_relaxation_safety(
    *,
    batch_root: Path,
    candidate_path: Path,
    contract_path: Path,
    readiness_path: Path,
    config: dict[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    reason_codes: list[str] = []
    source_summaries: dict[str, Any] = {}
    candidate = _load_source(
        candidate_path,
        label="anchor_projection_candidate_generation_summary",
        expected_schema=ANCHOR_CANDIDATE_SCHEMA_VERSION,
        repo_root=repo_root,
        reason_codes=reason_codes,
        source_summaries=source_summaries,
    )
    contract = _load_optional_source(
        contract_path,
        label="anchor_projection_evidence_contract_summary",
        expected_schema=ANCHOR_CONTRACT_SCHEMA_VERSION,
        repo_root=repo_root,
        reason_codes=reason_codes,
        source_summaries=source_summaries,
    )
    readiness = _load_optional_source(
        readiness_path,
        label="policy_training_readiness_review_summary",
        expected_schema=READINESS_SCHEMA_VERSION,
        repo_root=repo_root,
        reason_codes=reason_codes,
        source_summaries=source_summaries,
    )
    if config["validation"]["fail_on_input_failure"]:
        for label, payload in (
            ("anchor_projection_candidate_generation_summary", candidate),
            ("anchor_projection_evidence_contract_summary", contract),
            ("policy_training_readiness_review_summary", readiness),
        ):
            if payload.get("status") == "failed":
                _append_reason(reason_codes, f"{label}_failed")

    current_git = _git_snapshot(repo_root)
    source_git_matches = [
        _inspect_source_git_provenance(
            payload,
            label=label,
            current_git=current_git,
            require_current_git_match=config["validation"]["require_current_git_match"],
            reason_codes=reason_codes,
        )
        for label, payload in (
            ("anchor_projection_candidate_generation_summary", candidate),
            ("anchor_projection_evidence_contract_summary", contract),
            ("policy_training_readiness_review_summary", readiness),
        )
        if payload
    ]

    fallback_or_open_grid_count = max(
        _fallback_or_open_grid_count(candidate),
        _fallback_or_open_grid_count(contract),
        _fallback_or_open_grid_count(readiness),
    )
    if (
        config["validation"]["fail_on_fallback_or_open_grid"]
        and fallback_or_open_grid_count > 0
    ):
        _append_reason(reason_codes, "fallback_or_open_grid_blocks_distance_contract_relaxation_audit")
    safety_regression_count = max(
        _int_value(candidate.get("safety_regression_count")),
        _int_value(contract.get("safety_regression_count")),
        _int_value(readiness.get("safety_regression_count")),
    )
    if config["validation"]["fail_on_safety_regression"] and safety_regression_count > 0:
        _append_reason(reason_codes, "safety_regression_blocks_distance_contract_relaxation_audit")

    contract_alignment_gap_count = max(
        _int_value(contract.get("candidate_contract_alignment_gap_count")),
        _int_value(readiness.get("anchor_projection_candidate_contract_alignment_gap_count")),
    )
    if contract_alignment_gap_count > 0:
        _append_reason(reason_codes, "candidate_contract_alignment_gap_blocks_distance_contract_relaxation_audit")

    platform_goal_contract_mismatch_count = _int_value(candidate.get("platform_goal_contract_mismatch_count"))
    trainable_count = _int_value(candidate.get("trainable_anchor_projection_count"))
    nontrainable_count = _int_value(candidate.get("nontrainable_blocked_target_count"))
    if trainable_count + nontrainable_count != platform_goal_contract_mismatch_count:
        _append_reason(reason_codes, "anchor_projection_context_accounting_mismatch")

    source_selection_quality_regression_count = _int_value(
        candidate.get("source_selection_quality_regression_count")
    )
    if (
        source_selection_quality_regression_count
        > config["relaxation_thresholds"]["max_source_selection_quality_regression_count"]
    ):
        _append_reason(reason_codes, "source_selection_quality_regression_blocks_distance_contract_relaxation")

    audit_proxy_positive_count = max(
        _int_value(candidate.get("audit_proxy_positive_count")),
        _int_value(candidate.get("positive_training_evidence_contains_audit_proxy_anchor_count")),
        _int_value(contract.get("audit_proxy_positive_count")),
        _int_value(contract.get("positive_training_evidence_contains_audit_proxy_anchor_count")),
    )
    if audit_proxy_positive_count > config["relaxation_thresholds"]["max_audit_proxy_positive_count"]:
        _append_reason(reason_codes, "audit_proxy_positive_blocks_distance_contract_relaxation")

    context_records = candidate.get("context_records") if isinstance(candidate.get("context_records"), list) else []
    distance_rejected_contexts = [
        context for context in context_records if isinstance(context, dict) and _has_distance_reject(context)
    ]
    expected_distance_count = _int_value(
        _nested_dict(candidate, "distance_contract_rejected_by_distance_bin").get("count")
    )
    if expected_distance_count and expected_distance_count != len(distance_rejected_contexts):
        _append_reason(reason_codes, "distance_contract_context_count_mismatch")

    source_selected_distance_rejected = [
        context for context in distance_rejected_contexts if context.get("projected_candidate_source_selected") is True
    ]
    not_source_selected_distance_rejected = [
        context for context in distance_rejected_contexts if context.get("projected_candidate_source_selected") is not True
    ]
    eligible_contexts, ineligible_contexts = _split_relaxation_eligibility(
        source_selected_distance_rejected,
        config=config,
    )
    recommendation = _recommendation(
        reason_codes=reason_codes,
        distance_rejected_count=len(distance_rejected_contexts),
        source_selected_count=len(source_selected_distance_rejected),
        not_source_selected_count=len(not_source_selected_distance_rejected),
        eligible_count=len(eligible_contexts),
    )

    status = "failed" if reason_codes else "passed"
    ready_for_opt_in = recommendation == "opt_in_distance_contract_relaxation_profile_ready"
    max_cells = _max_float(context.get("projection_distance_cells") for context in eligible_contexts)
    max_m = _max_float(context.get("projection_distance_m") for context in eligible_contexts)
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "reason_codes": list(reason_codes),
        "failure_reason_code_counts": dict(sorted(Counter(reason_codes).items())),
        "batch_root": _display_path(batch_root, repo_root),
        "source_summaries": source_summaries,
        "git_provenance": {
            "current": current_git,
            "anchor_projection_candidate_generation": _public_git(candidate),
            "anchor_projection_evidence_contract": _public_git(contract),
            "policy_training_readiness_review": _public_git(readiness),
            "current_matches_sources": all(source_git_matches),
        },
        "current_git_provenance_mismatch_count": int("current_git_provenance_mismatch" in reason_codes),
        "git_provenance_mismatch_count": int("git_provenance_mismatch" in reason_codes),
        "recommendation": recommendation,
        "platform_goal_contract_mismatch_count": platform_goal_contract_mismatch_count,
        "trainable_anchor_projection_count": trainable_count,
        "nontrainable_blocked_target_count": nontrainable_count,
        "distance_contract_rejected_count": len(distance_rejected_contexts),
        "source_selected_distance_rejected_count": len(source_selected_distance_rejected),
        "not_source_selected_distance_rejected_count": len(not_source_selected_distance_rejected),
        "distance_contract_rejected_by_distance_bin": candidate.get(
            "distance_contract_rejected_by_distance_bin",
            {},
        ),
        "source_candidate_not_selected_by_best_alternative_reason": candidate.get(
            "source_candidate_not_selected_by_best_alternative_reason",
            {},
        ),
        "fallback_or_open_grid_count": fallback_or_open_grid_count,
        "safety_regression_count": safety_regression_count,
        "source_selection_quality_regression_count": source_selection_quality_regression_count,
        "audit_proxy_positive_count": audit_proxy_positive_count,
        "candidate_contract_alignment_gap_count": contract_alignment_gap_count,
        "relaxation_safety": {
            "eligible_source_selected_distance_rejected_count": len(eligible_contexts),
            "ineligible_source_selected_distance_rejected_count": len(ineligible_contexts),
            "eligible_contexts": _public_contexts(eligible_contexts),
            "ineligible_contexts": _public_contexts(ineligible_contexts),
            "not_source_selected_distance_rejected_contexts": _public_contexts(
                not_source_selected_distance_rejected
            ),
        },
        "relaxation_profile": {
            "ready_for_opt_in_relaxation": ready_for_opt_in,
            "profile_id": "anchor_projection_distance_contract_relaxation_opt_in_v1"
            if ready_for_opt_in
            else None,
            "default_contract_unchanged": True,
            "opt_in_only": True,
            "max_projection_distance_cells": int(max_cells) if max_cells is not None and max_cells.is_integer() else max_cells,
            "max_projection_distance_m": max_m,
            "max_path_cost_margin_vs_selected": config["relaxation_thresholds"][
                "max_path_cost_margin_vs_selected"
            ],
            "max_risk_margin_vs_selected": config["relaxation_thresholds"][
                "max_risk_margin_vs_selected"
            ],
        },
        "readiness_impact": {
            "input_training_readiness_status": readiness.get("training_readiness_status"),
            "input_training_blockers": readiness.get("training_blockers", []),
            "recommended_readiness_status": (
                "needs_training_contract_refinement"
                if not ready_for_opt_in
                else "needs_training_contract_refinement"
            ),
            "summary_passed_is_not_ppo_readiness": True,
        },
        "runs_training": False,
        "audit_only": True,
        "no_ppo_training": True,
        "does_not_modify_default_astar": True,
        "does_not_modify_ppo": True,
        "does_not_modify_network": True,
        "does_not_modify_action_space": True,
        "no_ackermann_feasible_trajectory_claim": True,
        "non_goals": list(config.get("non_goals", [])),
    }


def _split_relaxation_eligibility(
    contexts: list[dict[str, Any]],
    *,
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    eligible: list[dict[str, Any]] = []
    ineligible: list[dict[str, Any]] = []
    for context in contexts:
        reasons = _context_ineligible_reasons(context, config=config)
        annotated = {**context, "distance_contract_relaxation_ineligible_reasons": reasons}
        if reasons:
            ineligible.append(annotated)
        else:
            eligible.append(annotated)
    return eligible, ineligible


def _context_ineligible_reasons(context: dict[str, Any], *, config: dict[str, Any]) -> list[str]:
    thresholds = config["relaxation_thresholds"]
    reasons: list[str] = []
    if context.get("source_selection_quality_regression") is True:
        reasons.append("source_selection_quality_regression")
    if context.get("positive_audit_proxy") is True:
        reasons.append("audit_proxy_positive")
    if context.get("training_use") != "trainable_anchor_projection_contrast":
        reasons.append("source_training_use_not_trainable")
    if context.get("comparison_scope") != "projected_target_anchor_contrast":
        reasons.append("comparison_scope_not_projected_target_anchor_contrast")
    distance_cells = _float_optional(context.get("projection_distance_cells"))
    if distance_cells is None:
        reasons.append("projection_distance_cells_missing")
    elif distance_cells > thresholds["max_opt_in_projection_distance_cells"]:
        reasons.append("projection_distance_cells_exceeds_relaxation_audit_limit")
    distance_m = _float_optional(context.get("projection_distance_m"))
    if distance_m is None:
        reasons.append("projection_distance_m_missing")
    elif distance_m > thresholds["max_opt_in_projection_distance_m"]:
        reasons.append("projection_distance_m_exceeds_relaxation_audit_limit")
    path_margin = _float_optional(context.get("path_cost_margin_vs_selected"))
    if path_margin is None:
        reasons.append("path_cost_margin_missing")
    elif path_margin > thresholds["max_path_cost_margin_vs_selected"]:
        reasons.append("path_cost_margin_exceeds_relaxation_limit")
    risk_margin = _float_optional(context.get("risk_margin_vs_selected"))
    if risk_margin is None:
        reasons.append("risk_margin_missing")
    elif risk_margin > thresholds["max_risk_margin_vs_selected"]:
        reasons.append("risk_margin_exceeds_relaxation_limit")
    return reasons


def _recommendation(
    *,
    reason_codes: list[str],
    distance_rejected_count: int,
    source_selected_count: int,
    not_source_selected_count: int,
    eligible_count: int,
) -> str:
    if reason_codes:
        return "fix_validation_failures_before_distance_contract_relaxation"
    if distance_rejected_count == 0:
        return "no_distance_contract_relaxation_needed"
    if source_selected_count > 0 and eligible_count == source_selected_count and not_source_selected_count == 0:
        return "opt_in_distance_contract_relaxation_profile_ready"
    return "keep_current_training_distance_contract"


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
    output_files = payload.get("output_files")
    if not isinstance(output_files, dict) or not _nonempty_string(
        output_files.get("distance_contract_relaxation_safety_audit_summary")
    ):
        raise ConfigError(
            "output_files.distance_contract_relaxation_safety_audit_summary must be a non-empty string"
        )
    validation = payload.get("validation") if isinstance(payload.get("validation"), dict) else {}
    thresholds = payload.get("relaxation_thresholds")
    if not isinstance(thresholds, dict):
        raise ConfigError("relaxation_thresholds must be an object")
    normalized_thresholds = {
        "max_opt_in_projection_distance_cells": _nonnegative_float(
            thresholds.get("max_opt_in_projection_distance_cells", 3),
            "relaxation_thresholds.max_opt_in_projection_distance_cells",
        ),
        "max_opt_in_projection_distance_m": _nonnegative_float(
            thresholds.get("max_opt_in_projection_distance_m", 1.5),
            "relaxation_thresholds.max_opt_in_projection_distance_m",
        ),
        "max_path_cost_margin_vs_selected": _nonnegative_float(
            thresholds.get("max_path_cost_margin_vs_selected", 0.0),
            "relaxation_thresholds.max_path_cost_margin_vs_selected",
        ),
        "max_risk_margin_vs_selected": _nonnegative_float(
            thresholds.get("max_risk_margin_vs_selected", 0.0),
            "relaxation_thresholds.max_risk_margin_vs_selected",
        ),
        "max_source_selection_quality_regression_count": int(
            thresholds.get("max_source_selection_quality_regression_count", 0) or 0
        ),
        "max_audit_proxy_positive_count": int(
            thresholds.get("max_audit_proxy_positive_count", 0) or 0
        ),
    }
    for key in ("max_source_selection_quality_regression_count", "max_audit_proxy_positive_count"):
        if normalized_thresholds[key] < 0:
            raise ConfigError(f"relaxation_thresholds.{key} must be >= 0")
    config = dict(payload)
    config["validation"] = {
        "require_current_git_match": bool(validation.get("require_current_git_match", True)),
        "fail_on_input_failure": bool(validation.get("fail_on_input_failure", True)),
        "fail_on_fallback_or_open_grid": bool(validation.get("fail_on_fallback_or_open_grid", True)),
        "fail_on_safety_regression": bool(validation.get("fail_on_safety_regression", True)),
    }
    config["relaxation_thresholds"] = normalized_thresholds
    return config


def _load_source(
    path: Path,
    *,
    label: str,
    expected_schema: str,
    repo_root: Path,
    reason_codes: list[str],
    source_summaries: dict[str, Any],
) -> dict[str, Any]:
    record: dict[str, Any] = {"path": _display_path(path, repo_root), "exists": path.is_file()}
    if not path.is_file():
        _append_reason(reason_codes, f"{label}_missing")
        source_summaries[label] = record
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        _append_reason(reason_codes, f"{label}_invalid_json")
        source_summaries[label] = record
        return {}
    if not isinstance(payload, dict):
        _append_reason(reason_codes, f"{label}_not_object")
        source_summaries[label] = record
        return {}
    record.update(
        {
            "schema_version": payload.get("schema_version"),
            "status": payload.get("status"),
            "reason_codes": _string_list(payload.get("reason_codes")),
        }
    )
    if payload.get("schema_version") != expected_schema:
        _append_reason(reason_codes, f"{label}_schema_mismatch")
    source_summaries[label] = record
    return payload


def _load_optional_source(
    path: Path,
    *,
    label: str,
    expected_schema: str,
    repo_root: Path,
    reason_codes: list[str],
    source_summaries: dict[str, Any],
) -> dict[str, Any]:
    if path.is_file():
        return _load_source(
            path,
            label=label,
            expected_schema=expected_schema,
            repo_root=repo_root,
            reason_codes=reason_codes,
            source_summaries=source_summaries,
        )
    source_summaries[label] = {
        "path": _display_path(path, repo_root),
        "exists": False,
        "optional": True,
    }
    return {}


def _public_contexts(contexts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fields = (
        "run_id",
        "scenario_id",
        "projection_distance_cells",
        "projection_distance_m",
        "path_cost_margin_vs_selected",
        "risk_margin_vs_selected",
        "projected_candidate_source_selected",
        "training_use",
        "comparison_scope",
        "source_selection_quality_regression",
        "positive_audit_proxy",
        "distance_contract_relaxation_ineligible_reasons",
    )
    return [{field: context.get(field) for field in fields if field in context} for context in contexts]


def _has_distance_reject(context: dict[str, Any]) -> bool:
    reasons = context.get("reject_reasons")
    return isinstance(reasons, list) and bool(DISTANCE_REJECT_REASONS.intersection(reasons))


def _fallback_or_open_grid_count(payload: dict[str, Any]) -> int:
    return max(
        _int_value(payload.get("open_grid_fallback_used_count")),
        _int_value(payload.get("open_grid_fallback_count")),
        _int_value(payload.get("fallback_or_open_grid_count")),
        _int_value(payload.get("fallback_used_count")),
    )


def _nested_dict(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    return value if isinstance(value, dict) else {}


def _int_value(value: Any) -> int:
    if isinstance(value, bool) or value is None:
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _float_optional(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if isfinite(number) else None


def _max_float(values: Any) -> float | None:
    clean = [_float_optional(value) for value in values]
    clean = [value for value in clean if value is not None]
    return max(clean) if clean else None


def _nonnegative_float(value: Any, label: str) -> float:
    number = _float_optional(value)
    if number is None or number < 0.0:
        raise ConfigError(f"{label} must be a non-negative number")
    return number


def _nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value)


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def _resolve_path(value: str | Path, repo_root: Path) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else repo_root / path


def _display_path(path: Path, repo_root: Path) -> str:
    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        return str(path)


def _append_reason(reason_codes: list[str], reason: str) -> None:
    if reason not in reason_codes:
        reason_codes.append(reason)


if __name__ == "__main__":
    raise SystemExit(main())
