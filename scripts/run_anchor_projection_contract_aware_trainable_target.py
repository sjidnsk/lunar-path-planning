from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from git_provenance import git_snapshot as _git_snapshot
from git_provenance import inspect_source_git_provenance as _inspect_source_git_provenance
from git_provenance import public_git as _public_git


CONFIG_SCHEMA_VERSION = "anchor-projection-contract-aware-trainable-target-config/v1"
SUMMARY_SCHEMA_VERSION = "anchor-projection-contract-aware-trainable-target-summary/v1"
CANDIDATE_SCHEMA_VERSION = "anchor-projection-candidate-generation-summary/v1"


class ConfigError(ValueError):
    pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Summarize opt-in contract-aware trainable anchor-projection targets."
    )
    parser.add_argument("--batch-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--anchor-projection-candidate-generation-summary",
        help="Defaults to <batch-root>/anchor-projection-candidate-generation-summary.json.",
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
    summary = analyze_contract_aware_trainable_target(
        batch_root=batch_root,
        candidate_path=candidate_path,
        config=config,
        repo_root=repo_root,
    )
    output_file = batch_root / config["output_files"]["contract_aware_trainable_target_summary"]
    print(
        json.dumps(
            {
                "status": "config validated" if summary["status"] == "passed" else "validation failed",
                "batch_root": _display_path(batch_root, repo_root),
                "reason_codes": summary["reason_codes"],
                "ppo_consumable_trainable_target_count": summary[
                    "ppo_consumable_trainable_target_count"
                ],
                "next_required_change": summary["next_required_change"],
                "contract_aware_trainable_target_summary": _display_path(output_file, repo_root),
            },
            ensure_ascii=False,
        )
    )
    if args.validate_only or args.dry_run:
        return 1 if summary["status"] == "failed" else 0
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return 1 if summary["status"] == "failed" else 0


def analyze_contract_aware_trainable_target(
    *,
    batch_root: Path,
    candidate_path: Path,
    config: dict[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    reason_codes: list[str] = []
    candidate = _load_source(
        candidate_path,
        expected_schema=CANDIDATE_SCHEMA_VERSION,
        label="anchor_projection_candidate_generation_summary",
        reason_codes=reason_codes,
    )
    if config["validation"]["fail_on_input_failure"] and candidate.get("status") == "failed":
        _append_reason(reason_codes, "anchor_projection_candidate_generation_summary_failed")

    current_git = _git_snapshot(repo_root)
    current_matches_sources = _inspect_source_git_provenance(
        candidate,
        label="anchor_projection_candidate_generation_summary",
        current_git=current_git,
        require_current_git_match=config["validation"]["require_current_git_match"],
        reason_codes=reason_codes,
        allow_dirty_current_git_match=config["validation"].get(
            "allow_dirty_current_git_match",
            False,
        ),
    )

    fallback_or_open_grid_count = _int_value(
        candidate.get("fallback_or_open_grid_count", candidate.get("open_grid_fallback_used_count"))
    )
    if config["validation"]["fail_on_fallback_or_open_grid"] and fallback_or_open_grid_count > 0:
        _append_reason(reason_codes, "fallback_or_open_grid_blocks_contract_aware_trainable_target")
    safety_regression_count = _int_value(candidate.get("safety_regression_count"))
    if config["validation"]["fail_on_safety_regression"] and safety_regression_count > 0:
        _append_reason(reason_codes, "safety_regression_blocks_contract_aware_trainable_target")

    contexts = [
        context for context in candidate.get("context_records", []) if isinstance(context, dict)
    ] if isinstance(candidate.get("context_records"), list) else []
    trainable_contexts = [context for context in contexts if context.get("trainable") is True]
    ppo_consumable_contexts = [
        context for context in trainable_contexts if _ppo_consumable_action(context)
    ]
    nontrainable_count = _int_value(candidate.get("nontrainable_blocked_target_count"))
    distance_rejected_count = _distance_rejected_count(candidate, contexts)
    source_not_selected_count = _source_candidate_not_selected_count(candidate, contexts)
    no_contract_safe_substitute_count = sum(
        1
        for context in contexts
        if context.get("trainable") is not True
        and not _contract_safe(context)
    )
    baselines = config["baseline_counts"]
    nontrainable_delta = nontrainable_count - _int_value(
        baselines.get("nontrainable_blocked_target_count")
    )
    distance_delta = distance_rejected_count - _int_value(
        baselines.get("distance_contract_rejected_count")
    )
    source_not_selected_delta = source_not_selected_count - _int_value(
        baselines.get("source_candidate_not_selected_count")
    )
    ppo_count = len(ppo_consumable_contexts)
    alignment_gap_count = _int_value(candidate.get("candidate_contract_alignment_gap_count"))
    main_success_gate_failures: list[str] = []
    if ppo_count <= 0:
        main_success_gate_failures.append("ppo_consumable_trainable_target_count_below_threshold")
    if nontrainable_delta >= 0:
        main_success_gate_failures.append("nontrainable_blocked_target_count_not_reduced")
    if alignment_gap_count > 0:
        main_success_gate_failures.append("candidate_contract_alignment_gap_count_nonzero")
    next_required_change = (
        "action_or_target_contract_change_required" if main_success_gate_failures else None
    )
    recommended_blockers = (
        []
        if not main_success_gate_failures
        else ["anchor_projection_nontrainable_contexts_remain"]
    )
    status = "failed" if reason_codes else "passed"
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "reason_codes": list(reason_codes),
        "failure_reason_code_counts": dict(sorted(Counter(reason_codes).items())),
        "batch_root": _display_path(batch_root, repo_root),
        "source_summaries": {
            "anchor_projection_candidate_generation_summary": {
                "path": _display_path(candidate_path, repo_root),
                "exists": candidate_path.is_file(),
                "schema_version": candidate.get("schema_version"),
                "status": candidate.get("status"),
            }
        },
        "git_provenance": {
            "current": current_git,
            "anchor_projection_candidate_generation": _public_git(candidate),
            "current_matches_sources": current_matches_sources,
        },
        "current_git_provenance_mismatch_count": int("current_git_provenance_mismatch" in reason_codes),
        "git_provenance_mismatch_count": int("git_provenance_mismatch" in reason_codes),
        "contract_trainable_contrast_count": len(trainable_contexts),
        "ppo_consumable_trainable_target_count": ppo_count,
        "nontrainable_blocked_target_count": nontrainable_count,
        "nontrainable_blocked_target_count_delta": nontrainable_delta,
        "distance_contract_rejected_count": distance_rejected_count,
        "distance_contract_rejected_count_delta": distance_delta,
        "source_candidate_not_selected_count": source_not_selected_count,
        "source_candidate_not_selected_count_delta": source_not_selected_delta,
        "no_contract_safe_reachable_substitute_count": no_contract_safe_substitute_count,
        "main_success_gate_failures": main_success_gate_failures,
        "next_required_change": next_required_change,
        "readiness_impact": {
            "recommended_training_blockers": recommended_blockers,
            "recommended_training_readiness_status": (
                "needs_training_contract_refinement"
                if recommended_blockers
                else "contract_aware_trainable_targets_available_for_limited_dry_run_review"
            ),
            "summary_passed_is_not_ppo_readiness": True,
        },
        "fallback_or_open_grid_count": fallback_or_open_grid_count,
        "safety_regression_count": safety_regression_count,
        "candidate_contract_alignment_gap_count": alignment_gap_count,
        "runs_training": False,
        "audit_only": True,
        "no_ppo_training": True,
        "does_not_modify_default_astar": True,
        "does_not_modify_ppo": True,
        "does_not_modify_network": True,
        "does_not_modify_action_space": True,
        "does_not_relax_default_distance_contract": True,
        "no_ackermann_feasible_trajectory_claim": True,
        "non_goals": list(config.get("non_goals", [])),
    }


def _ppo_consumable_action(context: dict[str, Any]) -> bool:
    if context.get("ppo_consumable_action") is True:
        return True
    generation = context.get("candidate_generation")
    if isinstance(generation, dict) and generation.get("ppo_consumable_action") is True:
        return True
    gate = context.get("trainability_gate")
    return isinstance(gate, dict) and gate.get("ppo_consumable_action") is True


def _contract_safe(context: dict[str, Any]) -> bool:
    if context.get("contract_safe") is True:
        return True
    generation = context.get("candidate_generation")
    if isinstance(generation, dict) and generation.get("contract_safe") is True:
        return True
    gate = context.get("trainability_gate")
    return isinstance(gate, dict) and gate.get("contract_safe") is True


def _distance_rejected_count(candidate: dict[str, Any], contexts: list[dict[str, Any]]) -> int:
    explicit = candidate.get("distance_contract_rejected_count")
    if explicit is not None:
        return _int_value(explicit)
    return sum(1 for context in contexts if _has_distance_reject(context))


def _source_candidate_not_selected_count(candidate: dict[str, Any], contexts: list[dict[str, Any]]) -> int:
    reasons = candidate.get("source_candidate_not_selected_by_best_alternative_reason")
    if isinstance(reasons, dict):
        return sum(_int_value(value) for value in reasons.values())
    return sum(
        1
        for context in contexts
        if context.get("projected_candidate_source_selected") is not True
    )


def _has_distance_reject(context: dict[str, Any]) -> bool:
    return any(
        reason in {
            "projection_distance_cells_exceeds_contract",
            "projection_distance_m_exceeds_contract",
        }
        for reason in _string_list(context.get("reject_reasons"))
    )


def _load_config(path: Path) -> dict[str, Any]:
    payload = _load_json(path)
    if payload.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ConfigError(f"{path}: expected schema_version={CONFIG_SCHEMA_VERSION!r}")
    if not isinstance(payload.get("output_files"), dict):
        raise ConfigError(f"{path}: output_files must be an object")
    if not payload["output_files"].get("contract_aware_trainable_target_summary"):
        raise ConfigError(f"{path}: output_files.contract_aware_trainable_target_summary is required")
    if not isinstance(payload.get("validation"), dict):
        raise ConfigError(f"{path}: validation must be an object")
    if not isinstance(payload.get("baseline_counts"), dict):
        raise ConfigError(f"{path}: baseline_counts must be an object")
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
    payload = _load_json(path)
    if payload.get("schema_version") != expected_schema:
        _append_reason(reason_codes, f"{label}_schema_mismatch")
    return payload


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ConfigError(f"{path}: file not found")
    except json.JSONDecodeError as exc:
        raise ConfigError(f"{path}: invalid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ConfigError(f"{path}: JSON root must be an object")
    return payload


def _resolve_path(value: str | Path, repo_root: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return repo_root / path


def _display_path(path: Path, repo_root: Path) -> str:
    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        return str(path)


def _int_value(value: Any) -> int:
    if isinstance(value, bool) or value is None:
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if item is not None]


def _append_reason(reason_codes: list[str], reason: str) -> None:
    if reason not in reason_codes:
        reason_codes.append(reason)


if __name__ == "__main__":
    raise SystemExit(main())
