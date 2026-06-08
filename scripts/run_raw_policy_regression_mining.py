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
from git_provenance import git_snapshots_match as _git_snapshots_match

from run_fresh_holdout_policy_candidate_evaluation import (
    _candidate_features,
    _global_features,
    _missing_indicators,
)
from run_scenario_disjoint_policy_rollout_evaluation import (
    _candidate_action_mask_valid,
    _collect_holdout_scenarios,
)


CONFIG_SCHEMA_VERSION = "raw-policy-regression-mining-config/v1"
SUMMARY_SCHEMA_VERSION = "raw-policy-regression-mining-summary/v1"
SAMPLE_SCHEMA_VERSION = "raw-policy-regression-preference-sample/v1"
EXCLUSION_SCHEMA_VERSION = "raw-policy-regression-exclusion-report/v1"
SAMPLE_TYPE = "raw_policy_regression_preference_pair"
NEXT_REQUIRED_CHANGE = "policy_objective_or_feature_refinement_required"


class ConfigError(ValueError):
    pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Mine raw policy regressions into pairwise preference samples."
    )
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--holdout-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    source_root = _resolve_path(args.source_root, repo_root)
    holdout_root = _resolve_path(args.holdout_root, repo_root)
    config_path = _resolve_path(args.config, repo_root)
    try:
        config = _load_config(config_path)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2

    output_paths = _output_paths(holdout_root, config)
    summary, samples, diagnostics, exclusion_report = run_raw_policy_regression_mining(
        source_root=source_root,
        holdout_root=holdout_root,
        config=config,
        repo_root=repo_root,
        output_paths=output_paths,
    )
    print(
        json.dumps(
            {
                "status": "config validated" if summary["status"] == "passed" else "validation failed",
                "source_root": _display_path(source_root, repo_root),
                "holdout_root": _display_path(holdout_root, repo_root),
                "reason_codes": summary["reason_codes"],
                "raw_policy_regression_input_count": summary["raw_policy_regression_input_count"],
                "raw_policy_regression_preference_pair_count": summary[
                    "raw_policy_regression_preference_pair_count"
                ],
                "summary": _display_path(output_paths["summary"], repo_root),
            },
            ensure_ascii=False,
        )
    )
    if not args.validate_only:
        holdout_root.mkdir(parents=True, exist_ok=True)
        _write_json(output_paths["summary"], summary)
        _write_json(output_paths["exclusion_report"], exclusion_report)
        output_paths["samples"].write_text(
            "".join(json.dumps(sample, ensure_ascii=False) + "\n" for sample in samples),
            encoding="utf-8",
        )
        output_paths["diagnostics"].write_text(
            "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in diagnostics),
            encoding="utf-8",
        )
    return 1 if summary["status"] == "failed" else 0


def run_raw_policy_regression_mining(
    *,
    source_root: Path,
    holdout_root: Path,
    config: dict[str, Any],
    repo_root: Path,
    output_paths: dict[str, Path],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    reason_codes: list[str] = []
    paths = _input_paths(source_root, holdout_root, config)
    source_batch = _load_json(paths["source_batch_summary"], "source_batch_summary", reason_codes)
    rollout_summary = _load_json(paths["rollout_summary"], "rollout_summary", reason_codes)
    decisions = _load_jsonl(paths["rollout_decisions"], "rollout_decisions", reason_codes)

    validation = config["validation"]
    if validation.get("require_source_batch_passed", True):
        if _int_value(source_batch.get("failed_count")) or _string_list(source_batch.get("reason_codes")):
            _append_reason(reason_codes, "source_batch_failed")
    if validation.get("require_rollout_summary_passed", True):
        if rollout_summary.get("status") != "passed" or _string_list(rollout_summary.get("reason_codes")):
            _append_reason(reason_codes, "rollout_summary_failed")

    current_git = _git_snapshot(repo_root)
    allow_dirty_match = bool(validation.get("allow_dirty_current_git_match"))
    source_git_current_matches = _source_batch_git_current_matches(
        source_batch,
        current_git,
        allow_dirty_match=allow_dirty_match,
    )
    rollout_git_current_matches = _summary_git_matches_current(
        rollout_summary,
        current_git,
        allow_dirty_match=allow_dirty_match,
    )
    if validation.get("require_current_git_match", True):
        if not source_git_current_matches:
            _append_reason(reason_codes, "source_batch_git_current_mismatch")
        if not rollout_git_current_matches:
            _append_reason(reason_codes, "rollout_summary_git_current_mismatch")

    scenario_groups = _collect_holdout_scenarios(holdout_root, repo_root)
    group_index = _index_groups(scenario_groups)
    raw_regression_decisions = [
        decision
        for decision in decisions
        if decision.get("raw_policy_decision_class") == "regression"
        or _string_list(decision.get("raw_policy_regression_reason_codes"))
    ]
    samples: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    seen_decision_keys: set[str] = set()
    sample_mode = str(config.get("sample", {}).get("mode") or "training")

    for decision in raw_regression_decisions:
        key = _decision_key(decision)
        if key in seen_decision_keys:
            exclusions.append(_exclusion(decision, "duplicate_raw_policy_regression_decision"))
            continue
        seen_decision_keys.add(key)
        group = _match_group(decision, group_index)
        if group is None:
            exclusions.append(_exclusion(decision, "holdout_scenario_group_missing"))
            continue
        preferred = _find_candidate(group, decision.get("source_selected_context_id"))
        alternative = _find_candidate(group, decision.get("raw_policy_selected_context_id"))
        if preferred is None:
            exclusions.append(_exclusion(decision, "source_selected_candidate_missing"))
            continue
        if alternative is None:
            exclusions.append(_exclusion(decision, "raw_policy_candidate_missing"))
            continue
        exclusion_reason = _sample_exclusion_reason(preferred, alternative)
        if exclusion_reason:
            exclusions.append(_exclusion(decision, exclusion_reason))
            continue
        if sample_mode == "diagnostic_only":
            diagnostics.append(_diagnostic(decision, group, preferred, alternative))
        else:
            samples.append(_sample(decision, group, preferred, alternative, config=config))

    hard_positive_added_count = 0
    if hard_positive_added_count > _int_value(validation.get("max_hard_positive_added_count")):
        _append_reason(reason_codes, "hard_positive_added_count_nonzero")
    expected = validation.get("expected_raw_policy_regression_input_count")
    if expected is not None and len(raw_regression_decisions) != _int_value(expected):
        _append_reason(reason_codes, "raw_policy_regression_input_count_mismatch")
    if sample_mode != "diagnostic_only" and len(samples) < _int_value(
        validation.get("min_raw_policy_regression_preference_pair_count")
    ):
        _append_reason(reason_codes, "raw_policy_regression_preference_pair_count_below_threshold")

    status = "failed" if reason_codes else "passed"
    exclusion_report = {
        "schema_version": EXCLUSION_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "status": status,
        "reason_codes": reason_codes,
        "exclusion_count": len(exclusions),
        "exclusion_reason_counts": dict(sorted(Counter(item["reason_code"] for item in exclusions).items())),
        "exclusions": exclusions,
        "hard_positive_added_count": hard_positive_added_count,
        "non_goals": list(config.get("non_goals", [])),
    }
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "status": status,
        "reason_codes": reason_codes,
        "failure_reason_code_counts": dict(sorted(Counter(reason_codes).items())),
        "source_root": _display_path(source_root, repo_root),
        "holdout_root": _display_path(holdout_root, repo_root),
        "rollout_summary_path": _display_path(paths["rollout_summary"], repo_root),
        "rollout_decisions_path": _display_path(paths["rollout_decisions"], repo_root),
        "samples_path": _display_path(output_paths["samples"], repo_root),
        "diagnostics_path": _display_path(output_paths["diagnostics"], repo_root),
        "exclusion_report_path": _display_path(output_paths["exclusion_report"], repo_root),
        "sample_mode": sample_mode,
        "raw_policy_regression_input_count": len(raw_regression_decisions),
        "raw_policy_regression_preference_pair_count": len(samples),
        "raw_policy_regression_diagnostic_count": len(diagnostics),
        "raw_policy_regression_excluded_count": len(exclusions),
        "unique_raw_policy_regression_decision_count": len(seen_decision_keys),
        "hard_positive_added_count": hard_positive_added_count,
        "source_batch_git_current_matches_sources": source_git_current_matches,
        "rollout_summary_git_current_matches_sources": rollout_git_current_matches,
        "next_required_change": NEXT_REQUIRED_CHANGE if status == "failed" else None,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "performance_claimed": False,
        "formal_training_ready_claimed": False,
        "git_provenance": {"current": current_git, "current_matches_sources": True},
        "non_goals": list(config.get("non_goals", [])),
    }
    return summary, samples, diagnostics, exclusion_report


def _sample(
    decision: dict[str, Any],
    group: dict[str, Any],
    preferred: dict[str, Any],
    alternative: dict[str, Any],
    *,
    config: dict[str, Any],
) -> dict[str, Any]:
    path_delta = _delta(alternative.get("path_cost"), preferred.get("path_cost"))
    risk_delta = _delta(alternative.get("risk"), preferred.get("risk"))
    return {
        "schema_version": SAMPLE_SCHEMA_VERSION,
        "sample_type": SAMPLE_TYPE,
        "context_id": preferred.get("context_id"),
        "alternative_context_id": alternative.get("context_id"),
        "scenario_id": group.get("scenario_id"),
        "scenario_group": group.get("scenario_group"),
        "scenario_seed": group.get("scenario_seed"),
        "scenario_variant_id": group.get("scenario_variant_id"),
        "diagnostic_profile": group.get("diagnostic_profile"),
        "planning_backend": group.get("planning_backend"),
        "source_path": group.get("source_path"),
        "source_selected_action_index": decision.get("source_selected_action_index"),
        "raw_policy_selected_action_index": decision.get("raw_policy_selected_action_index"),
        "policy_selected_action_index": decision.get("policy_selected_action_index"),
        "raw_policy_logit_margin_vs_source": _float_or_none(
            decision.get("raw_policy_logit_margin_vs_source")
        ),
        "source_selected_policy_logit": _float_or_none(
            decision.get("source_selected_policy_logit")
        ),
        "raw_policy_selected_policy_logit": _float_or_none(
            decision.get("raw_policy_selected_policy_logit")
        ),
        "path_cost_delta": path_delta,
        "risk_delta": risk_delta,
        "raw_policy_regression_reason_codes": _string_list(
            decision.get("raw_policy_regression_reason_codes")
        ),
        "preferred": _sample_side(preferred),
        "alternative": _sample_side(alternative),
        "selected": _sample_side(preferred),
        "global_features": _global_features(group.get("candidates", [])),
        "candidate_missing_indicators": [
            _missing_indicators(preferred),
            _missing_indicators(alternative),
        ],
        "sample_weight": float(config.get("sample", {}).get("sample_weight", 1.0)),
        "hard_positive_added": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "performance_claimed": False,
    }


def _sample_side(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "context_id": candidate.get("context_id"),
        "action_index": candidate.get("source_action_index"),
        "source_action_index": candidate.get("source_action_index"),
        "policy_target_cell": candidate.get("policy_target_cell"),
        "execution_goal_cell": candidate.get("execution_goal_cell"),
        "target_binding_mode": candidate.get("target_binding_mode"),
        "path_cost": candidate.get("path_cost"),
        "risk": candidate.get("risk"),
        "utility": candidate.get("utility"),
        "contract_safe": candidate.get("contract_safe"),
        "reachable": candidate.get("reachable"),
        "replan_required": candidate.get("replan_required"),
        "source_selection_status": candidate.get("source_selection_status"),
        "candidate_features": _candidate_features(candidate),
    }


def _diagnostic(
    decision: dict[str, Any],
    group: dict[str, Any],
    preferred: dict[str, Any],
    alternative: dict[str, Any],
) -> dict[str, Any]:
    path_delta = _delta(alternative.get("path_cost"), preferred.get("path_cost"))
    risk_delta = _delta(alternative.get("risk"), preferred.get("risk"))
    return {
        "schema_version": "raw-policy-regression-diagnostic/v1",
        "sample_type": "raw_policy_regression_diagnostic",
        "split_role": "eval_diagnostic",
        "context_id": preferred.get("context_id"),
        "alternative_context_id": alternative.get("context_id"),
        "scenario_id": group.get("scenario_id"),
        "scenario_group": group.get("scenario_group"),
        "scenario_seed": group.get("scenario_seed"),
        "scenario_variant_id": group.get("scenario_variant_id"),
        "source_selected_action_index": decision.get("source_selected_action_index"),
        "raw_policy_selected_action_index": decision.get("raw_policy_selected_action_index"),
        "path_cost_delta": path_delta,
        "risk_delta": risk_delta,
        "raw_policy_regression_reason_codes": _string_list(
            decision.get("raw_policy_regression_reason_codes")
        ),
        "hard_positive_added": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "performance_claimed": False,
    }


def _sample_exclusion_reason(preferred: dict[str, Any], alternative: dict[str, Any]) -> str | None:
    if not preferred.get("context_id") or not alternative.get("context_id"):
        return "context_id_missing"
    if not _candidate_action_mask_valid(preferred) or not _candidate_action_mask_valid(alternative):
        return "action_mask_invalid"
    for label, candidate in (("preferred", preferred), ("alternative", alternative)):
        if _float_or_none(candidate.get("path_cost")) is None:
            return f"{label}_path_cost_missing"
        if _float_or_none(candidate.get("risk")) is None:
            return f"{label}_risk_missing"
    return None


def _index_groups(groups: list[dict[str, Any]]) -> dict[tuple[str | None, str | None, str | None], dict[str, Any]]:
    index: dict[tuple[str | None, str | None, str | None], dict[str, Any]] = {}
    for group in groups:
        keys = [
            (group.get("source_path"), group.get("scenario_id"), group.get("run_id")),
            (None, group.get("scenario_id"), group.get("run_id")),
            (None, group.get("scenario_id"), None),
        ]
        for key in keys:
            index.setdefault(key, group)
    return index


def _match_group(
    decision: dict[str, Any],
    index: dict[tuple[str | None, str | None, str | None], dict[str, Any]],
) -> dict[str, Any] | None:
    keys = [
        (decision.get("source_path"), decision.get("scenario_id"), decision.get("run_id")),
        (None, decision.get("scenario_id"), decision.get("run_id")),
        (None, decision.get("scenario_id"), None),
    ]
    for key in keys:
        group = index.get(key)
        if group is not None:
            return group
    return None


def _find_candidate(group: dict[str, Any], context_id: Any) -> dict[str, Any] | None:
    if not context_id:
        return None
    for candidate in group.get("candidates", []):
        if candidate.get("context_id") == context_id:
            return candidate
    return None


def _decision_key(decision: dict[str, Any]) -> str:
    return "|".join(
        str(decision.get(field) or "")
        for field in (
            "source_path",
            "scenario_id",
            "source_selected_context_id",
            "raw_policy_selected_context_id",
        )
    )


def _exclusion(decision: dict[str, Any], reason_code: str) -> dict[str, Any]:
    return {
        "reason_code": reason_code,
        "scenario_id": decision.get("scenario_id"),
        "source_path": decision.get("source_path"),
        "source_selected_context_id": decision.get("source_selected_context_id"),
        "raw_policy_selected_context_id": decision.get("raw_policy_selected_context_id"),
        "raw_policy_regression_reason_codes": _string_list(
            decision.get("raw_policy_regression_reason_codes")
        ),
    }


def _input_paths(source_root: Path, holdout_root: Path, config: dict[str, Any]) -> dict[str, Path]:
    inputs = config["input_files"]
    return {
        "source_batch_summary": source_root / inputs["source_batch_summary"],
        "rollout_summary": holdout_root / inputs["rollout_summary"],
        "rollout_decisions": holdout_root / inputs["rollout_decisions"],
    }


def _output_paths(holdout_root: Path, config: dict[str, Any]) -> dict[str, Path]:
    outputs = config["output_files"]
    return {
        "summary": holdout_root / outputs["summary"],
        "samples": holdout_root / outputs["samples"],
        "diagnostics": holdout_root
        / outputs.get("diagnostics", "raw-policy-regression-diagnostics.jsonl"),
        "exclusion_report": holdout_root / outputs["exclusion_report"],
    }


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
    for section in ("input_files", "output_files", "sample", "validation"):
        if not isinstance(payload.get(section), dict):
            raise ConfigError(f"{section} must be an object")
    return payload


def _load_json(path: Path, label: str, reason_codes: list[str]) -> dict[str, Any]:
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
    return payload


def _load_jsonl(path: Path, label: str, reason_codes: list[str]) -> list[dict[str, Any]]:
    if not path.is_file():
        _append_reason(reason_codes, f"{label}_missing")
        return []
    rows: list[dict[str, Any]] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                rows.append(payload)
    except json.JSONDecodeError:
        _append_reason(reason_codes, f"{label}_invalid_json")
    return rows


def _summary_git_matches_current(
    payload: dict[str, Any],
    current_git: dict[str, Any],
    *,
    allow_dirty_match: bool = False,
) -> bool:
    git = payload.get("git_provenance") if isinstance(payload.get("git_provenance"), dict) else {}
    source_current = git.get("current") if isinstance(git.get("current"), dict) else {}
    if not source_current or git.get("current_matches_sources") is False:
        return False
    return _git_snapshots_match(source_current, current_git, allow_dirty_match=allow_dirty_match)


def _source_batch_git_current_matches(
    payload: dict[str, Any],
    current_git: dict[str, Any],
    *,
    allow_dirty_match: bool = False,
) -> bool:
    git = payload.get("git_provenance") if isinstance(payload.get("git_provenance"), dict) else {}
    if git:
        return _summary_git_matches_current(
            payload,
            current_git,
            allow_dirty_match=allow_dirty_match,
        )
    return _int_value(payload.get("current_git_provenance_mismatch_count")) == 0


def _resolve_path(value: str | Path, repo_root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def _display_path(path: Path, repo_root: Path) -> str:
    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        return str(path)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _append_reason(reasons: list[str], reason: str) -> None:
    if reason not in reasons:
        reasons.append(reason)


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def _int_value(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _float_or_none(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if isfinite(numeric) else None


def _delta(value: Any, reference: Any) -> float | None:
    numeric = _float_or_none(value)
    baseline = _float_or_none(reference)
    if numeric is None or baseline is None:
        return None
    return numeric - baseline


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
