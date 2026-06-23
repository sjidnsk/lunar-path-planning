from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_EXPLORER_SRC = SCRIPT_DIR.parent / "model-explorer" / "src"
if str(MODEL_EXPLORER_SRC) not in sys.path:
    sys.path.insert(0, str(MODEL_EXPLORER_SRC))

from model_explorer.policy.canonical_reward import compute_canonical_reward_components, load_canonical_reward_profile


CONFIG_SCHEMA_VERSION = "xunce-stage20-1-same-candidate-oracle-imitation-evidence-config/v1"
SUMMARY_SCHEMA_VERSION = "xunce-stage20-1-same-candidate-oracle-imitation-summary/v1"
LABEL_SCHEMA_VERSION = "xunce-stage20-1-on-policy-teacher-label/v1"
EXCLUSION_SCHEMA_VERSION = "xunce-stage20-1-teacher-label-exclusion/v1"
DATASET_STATS_SCHEMA_VERSION = "xunce-stage20-1-dataset-stats/v1"
ROUTING_SCHEMA_VERSION = "xunce-stage20-1-next-stage-routing/v1"
MANIFEST_SCHEMA_VERSION = "xunce-stage20-1-manifest/v1"

DEFAULT_CONFIG = "configs/xunce_stage20_1_same_candidate_oracle_imitation_evidence_v1.json"
DEFAULT_STAGE19_ROOT = (
    "D:/CodexDownloads/lunar-path-planning/stage19_evaluator_critic_preflight/"
    "outputs/path_feedback_batch_xunce_stage19_evaluator_critic_preflight_v1"
)
DEFAULT_STAGE20_ROOT = (
    "D:/CodexDownloads/lunar-path-planning/stage20_reward_rerank_oracle_imitation_dataset/"
    "outputs/path_feedback_batch_xunce_stage20_reward_rerank_oracle_imitation_dataset_v1"
)
DEFAULT_OUTPUT_ROOT = (
    "D:/CodexDownloads/lunar-path-planning/stage20_1_same_candidate_oracle_imitation_evidence/"
    "outputs/path_feedback_batch_xunce_stage20_1_same_candidate_oracle_imitation_evidence_v1"
)

STAGE19_SUMMARY_FILE = "xunce-stage19-evaluator-critic-preflight-summary.json"
STAGE19_TARGET_FILE = "xunce-stage19-practical-target-selection.json"
STAGE20_SUMMARY_FILE = "xunce-stage20-oracle-imitation-summary.json"
COVERAGE_SUMMARY_FILE = "xunce-exploration-coverage-comparison-summary.json"
TEACHER_LABEL_AUDIT_FILE = "xunce-exploration-coverage-on-policy-oracle-teacher-labels.jsonl"
COVERAGE_STEPS_FILE = "xunce-exploration-coverage-steps.jsonl"
CANDIDATE_METRIC_AUDIT_FILE = "xunce-exploration-coverage-candidate-metric-audit.jsonl"

SUMMARY_FILE = "xunce-stage20-1-same-candidate-oracle-imitation-summary.json"
LABELS_FILE = "xunce-stage20-1-on-policy-teacher-labels.jsonl"
EXCLUSION_FILE = "xunce-stage20-1-teacher-label-exclusion-report.jsonl"
DATASET_STATS_FILE = "xunce-stage20-1-dataset-stats.json"
ROUTING_FILE = "xunce-stage20-1-next-stage-routing.json"
REPORT_FILE = "xunce-stage20-1-report.md"
MANIFEST_FILE = "xunce-stage20-1-manifest.json"

ROUTE_INPUTS = "rerun_stage20_1_required_inputs"
ROUTE_LABEL_ROLLOUT = "run_stage20_1_on_policy_teacher_label_rollout"
ROUTE_RERUN_STAGE20 = "rerun_stage20_oracle_imitation_dataset_with_expanded_labels"
ROUTE_PREFLIGHT = "stage20_1_supervised_oracle_imitation_checkpoint_preflight"

BOUNDARY_FIELDS = (
    "stage20_authorized",
    "training_or_release_authorized",
    "runs_new_ppo_update",
    "publishes_checkpoint",
    "replaces_default_policy",
    "connects_real_executor",
    "starts_online_canary",
    "real_world_release_approved",
    "real_world_performance_claimed",
)


class ConfigError(ValueError):
    pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Collect Stage 20.1 same-candidate oracle imitation evidence.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--stage19-evaluator-critic-preflight-root", default=DEFAULT_STAGE19_ROOT)
    parser.add_argument("--stage20-oracle-imitation-dataset-root", default=DEFAULT_STAGE20_ROOT)
    parser.add_argument("--coverage-comparison-root")
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args(argv)

    repo_root = Path(args.repo_root).resolve()
    try:
        summary = run_xunce_stage20_1_same_candidate_oracle_imitation_evidence(
            config_path=Path(args.config),
            stage19_evaluator_critic_preflight_root=Path(args.stage19_evaluator_critic_preflight_root),
            stage20_oracle_imitation_dataset_root=Path(args.stage20_oracle_imitation_dataset_root),
            coverage_comparison_root=Path(args.coverage_comparison_root) if args.coverage_comparison_root else None,
            output_root=Path(args.output_root),
            repo_root=repo_root,
        )
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": summary["status"],
                "next_required_change": summary["next_required_change"],
                "trainable_label_count": summary["trainable_label_count"],
                "stage20_authorized": summary["stage20_authorized"],
                "summary": summary["summary"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if summary["status"] != "failed" else 1


def run_xunce_stage20_1_same_candidate_oracle_imitation_evidence(
    *,
    config_path: Path,
    stage19_evaluator_critic_preflight_root: Path,
    stage20_oracle_imitation_dataset_root: Path,
    coverage_comparison_root: Path | None,
    output_root: Path,
    repo_root: Path,
) -> dict[str, Any]:
    repo_root = Path(repo_root).resolve()
    config = _load_config(_resolve_path(config_path, repo_root))
    stage19_root = _resolve_path(stage19_evaluator_critic_preflight_root, repo_root)
    stage20_root = _resolve_path(stage20_oracle_imitation_dataset_root, repo_root)
    output_root = _resolve_path(output_root, repo_root)
    output_root.mkdir(parents=True, exist_ok=True)

    stage19_summary = _read_json(stage19_root / STAGE19_SUMMARY_FILE)
    target = _read_json(stage19_root / STAGE19_TARGET_FILE)
    stage20_summary = _read_json(stage20_root / STAGE20_SUMMARY_FILE)
    selected_root = _selected_root(target)
    coverage_root = _resolve_path(coverage_comparison_root, repo_root) if coverage_comparison_root is not None else selected_root

    blocking = _boundary_rejections(config)
    blocking.extend(_input_rejections(stage19_summary, target, stage20_summary, coverage_root))

    coverage_summary = _read_json(coverage_root / COVERAGE_SUMMARY_FILE) if coverage_root is not None else {}
    label_rows = _read_jsonl(coverage_root / TEACHER_LABEL_AUDIT_FILE) if coverage_root is not None else []
    expected_profile = _expected_teacher_profile(target, coverage_summary)
    if not label_rows and coverage_root is not None:
        label_rows = _synthesize_on_policy_labels_from_candidate_audit(
            coverage_root=coverage_root,
            coverage_summary=coverage_summary,
            expected_profile=expected_profile,
        )
    blocking.extend(_lineage_rejections(expected_profile, coverage_summary, label_rows))

    trainable: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    if not blocking:
        trainable, exclusions = _build_trainable_labels(label_rows, expected_profile=expected_profile)

    route = _route(config=config, blocking=blocking, label_count=len(trainable), raw_label_count=len(label_rows))
    status = "failed" if route == ROUTE_INPUTS else "passed"
    generated_at = datetime.now(timezone.utc).isoformat()
    stats = _dataset_stats(trainable=trainable, exclusions=exclusions, raw_rows=label_rows)
    routing = _routing_payload(route)
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": status,
        "stage19_evaluator_critic_preflight_root": str(stage19_root.resolve()),
        "stage20_oracle_imitation_dataset_root": str(stage20_root.resolve()),
        "coverage_comparison_root": str(coverage_root.resolve()) if coverage_root is not None else None,
        "teacher_profile_id": expected_profile.get("teacher_profile_id"),
        "teacher_profile_hash": expected_profile.get("teacher_profile_hash"),
        "trainable_label_count": len(trainable),
        "excluded_label_count": len(exclusions),
        "raw_teacher_label_count": len(label_rows),
        "dataset_stats": stats,
        "next_stage_routing": routing,
        "next_required_change": route,
        "blocking_reason_codes": blocking,
        "reason_codes": list(blocking),
        "stage20_authorized": False,
        "training_or_release_authorized": False,
        "runs_new_ppo_update": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
        "summary": str((output_root / SUMMARY_FILE).resolve()),
    }
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "generated_at": generated_at,
        "summary": str((output_root / SUMMARY_FILE).resolve()),
        "teacher_labels": str((output_root / LABELS_FILE).resolve()),
        "exclusion_report": str((output_root / EXCLUSION_FILE).resolve()),
        "dataset_stats": str((output_root / DATASET_STATS_FILE).resolve()),
        "routing": str((output_root / ROUTING_FILE).resolve()),
        "report": str((output_root / REPORT_FILE).resolve()),
    }

    _write_json(output_root / SUMMARY_FILE, summary)
    _write_jsonl(output_root / LABELS_FILE, trainable)
    _write_jsonl(output_root / EXCLUSION_FILE, exclusions)
    _write_json(output_root / DATASET_STATS_FILE, stats)
    _write_json(output_root / ROUTING_FILE, routing)
    _write_json(output_root / MANIFEST_FILE, manifest)
    (output_root / REPORT_FILE).write_text(_render_report(summary), encoding="utf-8")
    return summary


def _load_config(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    if payload.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ConfigError(f"schema_version must be {CONFIG_SCHEMA_VERSION}")
    config = dict(payload)
    config["min_trainable_label_count_for_checkpoint_preflight"] = _int_required(
        config, "min_trainable_label_count_for_checkpoint_preflight"
    )
    for field in BOUNDARY_FIELDS:
        config[field] = bool(config.get(field, False))
    config["canary_traffic_fraction"] = float(config.get("canary_traffic_fraction", 0.0) or 0.0)
    return config


def _input_rejections(
    stage19_summary: dict[str, Any],
    target: dict[str, Any],
    stage20_summary: dict[str, Any],
    coverage_root: Path | None,
) -> list[str]:
    reasons: list[str] = []
    if stage19_summary.get("schema_version") != "xunce-stage19-evaluator-critic-preflight-summary/v1":
        reasons.append("invalid_stage19_summary_schema")
    if target.get("schema_version") != "xunce-stage19-practical-target-selection/v1":
        reasons.append("invalid_stage19_practical_target_schema")
    if stage20_summary.get("schema_version") != "xunce-stage20-oracle-imitation-summary/v1":
        reasons.append("invalid_stage20_summary_schema")
    if coverage_root is None:
        reasons.append("missing_selected_coverage_root")
    elif not coverage_root.exists():
        reasons.append("missing_selected_coverage_root")
    if _boundary_violation(stage19_summary) or _boundary_violation(stage20_summary):
        reasons.append("boundary_violation")
    return reasons


def _lineage_rejections(expected: dict[str, Any], coverage_summary: dict[str, Any], rows: list[dict[str, Any]]) -> list[str]:
    reasons: list[str] = []
    expected_id = expected.get("teacher_profile_id")
    expected_hash = expected.get("teacher_profile_hash")
    if coverage_summary.get("schema_version") != "xunce-exploration-coverage-comparison-summary/v1":
        reasons.append("invalid_coverage_summary_schema")
    summary_hash = coverage_summary.get("on_policy_oracle_teacher_profile_hash") or coverage_summary.get(
        "canonical_reward_rerank_profile_hash"
    )
    summary_id = coverage_summary.get("on_policy_oracle_teacher_profile_id") or coverage_summary.get("canonical_reward_rerank_profile_id")
    if expected_id and summary_id and expected_id != summary_id:
        reasons.append("teacher_profile_id_mismatch")
    if expected_hash and summary_hash and expected_hash != summary_hash:
        reasons.append("teacher_profile_hash_mismatch")
    if not rows:
        reasons.append("missing_on_policy_teacher_label_audit")
    for row in rows:
        if expected_id and row.get("teacher_profile_id") != expected_id:
            reasons.append("teacher_profile_id_mismatch")
            break
    for row in rows:
        if expected_hash and row.get("teacher_profile_hash") != expected_hash:
            reasons.append("teacher_profile_hash_mismatch")
            break
    return sorted(set(reasons))


def _expected_teacher_profile(target: dict[str, Any], coverage_summary: dict[str, Any]) -> dict[str, Any]:
    primary = target.get("primary") if isinstance(target.get("primary"), dict) else {}
    return {
        "teacher_profile_id": primary.get("canonical_reward_rerank_profile_id")
        or target.get("canonical_reward_rerank_profile_id")
        or coverage_summary.get("on_policy_oracle_teacher_profile_id")
        or coverage_summary.get("canonical_reward_rerank_profile_id"),
        "teacher_profile_hash": primary.get("canonical_reward_rerank_profile_hash")
        or target.get("canonical_reward_rerank_profile_hash")
        or coverage_summary.get("on_policy_oracle_teacher_profile_hash")
        or coverage_summary.get("canonical_reward_rerank_profile_hash"),
    }


def _selected_root(target: dict[str, Any]) -> Path | None:
    primary = target.get("primary") if isinstance(target.get("primary"), dict) else {}
    value = target.get("selected_root") or primary.get("root") or primary.get("selected_root")
    return Path(str(value)) if value else None


def _synthesize_on_policy_labels_from_candidate_audit(
    *,
    coverage_root: Path,
    coverage_summary: dict[str, Any],
    expected_profile: dict[str, Any],
) -> list[dict[str, Any]]:
    steps_path = coverage_root / COVERAGE_STEPS_FILE
    candidate_path = coverage_root / CANDIDATE_METRIC_AUDIT_FILE
    if not steps_path.is_file() or not candidate_path.is_file():
        return []
    profile_path = coverage_summary.get("on_policy_oracle_teacher_profile") or coverage_summary.get("canonical_reward_rerank_profile")
    if not profile_path:
        return []
    profile = load_canonical_reward_profile(Path(str(profile_path)))
    if expected_profile.get("teacher_profile_hash") and profile.profile_hash != expected_profile.get("teacher_profile_hash"):
        return []
    normalized = coverage_summary.get("normalized_config") if isinstance(coverage_summary.get("normalized_config"), dict) else {}
    denominator = _float_value(normalized.get("coverage_denominator_cells")) or 1000.0
    xunce_steps: dict[tuple[str, int, str, str], dict[str, Any]] = {}
    for row in _iter_jsonl(steps_path):
        if row.get("policy") != "xunce":
            continue
        if row.get("true_model_inference_executed") is not True or row.get("finite_outputs") is not True:
            continue
        key = (
            str(row.get("scenario_id")),
            int(row.get("step_index", 0) or 0),
            str(row.get("candidate_set_hash") or ""),
            str(row.get("covered_cells_hash") or ""),
        )
        xunce_steps[key] = row
    if not xunce_steps:
        return []

    grouped: dict[tuple[str, int, str, str], list[dict[str, Any]]] = {}
    for row in _iter_jsonl(candidate_path):
        if row.get("policy") != "xunce" and row.get("executing_policy") != "xunce":
            continue
        key = (
            str(row.get("scenario_id")),
            int(row.get("step_index", 0) or 0),
            str(row.get("candidate_set_hash") or ""),
            str(row.get("covered_cells_hash") or ""),
        )
        if key not in xunce_steps:
            continue
        grouped.setdefault(key, []).append(row)

    labels: list[dict[str, Any]] = []
    for key, rows in sorted(grouped.items()):
        step = xunce_steps[key]
        xunce_index = _int_value(step.get("selected_action_index"))
        by_index = {_int_value(row.get("candidate_index")): row for row in rows}
        xunce_row = by_index.get(xunce_index)
        teacher_row, teacher_result = _select_teacher_candidate(rows, profile=profile, denominator=denominator)
        if teacher_row is None or teacher_result is None:
            continue
        teacher_index = _int_value(teacher_row.get("candidate_index"))
        xunce_metrics = _metrics_from_candidate_audit_row(xunce_row)
        teacher_metrics = _metrics_from_candidate_audit_row(teacher_row)
        xunce_coverage = _float_value(xunce_metrics.get("new_covered_cell_count")) or 0.0
        teacher_coverage = _float_value(teacher_metrics.get("new_covered_cell_count")) or 0.0
        xunce_cost = _float_value(xunce_metrics.get("path_cost"))
        teacher_cost = _float_value(teacher_metrics.get("path_cost"))
        teacher_higher_coverage = teacher_coverage > xunce_coverage
        teacher_lower_cost = teacher_cost is not None and xunce_cost is not None and teacher_cost < xunce_cost
        teacher_acceptable_cost = (
            teacher_cost is not None
            and xunce_cost is not None
            and teacher_cost <= max(xunce_cost, 1.0e-9) * 1.25
        )
        if teacher_higher_coverage and teacher_acceptable_cost:
            sample_weight = 1.0
        elif teacher_coverage == xunce_coverage and teacher_lower_cost:
            sample_weight = 0.5
        else:
            sample_weight = 0.0
        hard_risk_clean_pair = bool(
            xunce_row is not None
            and not _candidate_audit_hard_risk_violation(xunce_row)
            and not _candidate_audit_hard_risk_violation(teacher_row)
        )
        labels.append(
            {
                "schema_version": "xunce-on-policy-oracle-teacher-label/v1",
                "scenario_id": step.get("scenario_id"),
                "roi_group": step.get("roi_group"),
                "split": step.get("split"),
                "step_index": step.get("step_index"),
                "baseline_policy": "xunce",
                "teacher_policy": "canonical_reward_rerank_oracle",
                "same_candidate_set": True,
                "candidate_set_id": teacher_row.get("candidate_set_id"),
                "candidate_set_hash": step.get("candidate_set_hash"),
                "current_cell": step.get("current_cell_before") or teacher_row.get("current_cell"),
                "covered_cells_hash": step.get("covered_cells_hash"),
                "teacher_action_index": teacher_index,
                "xunce_action_index": xunce_index,
                "teacher_profile_id": profile.profile_id,
                "teacher_profile_hash": profile.profile_hash,
                "teacher_reward_components": dict(teacher_result.components),
                "teacher_reward": float(teacher_result.reward),
                "xunce_candidate_metrics": xunce_metrics,
                "teacher_candidate_metrics": teacher_metrics,
                "oracle_new_covered_cell_count": teacher_metrics.get("new_covered_cell_count"),
                "xunce_new_covered_cell_count": xunce_metrics.get("new_covered_cell_count"),
                "oracle_path_cost": teacher_metrics.get("path_cost"),
                "xunce_path_cost": xunce_metrics.get("path_cost"),
                "oracle_soft_risk_exposure": teacher_metrics.get("soft_risk_exposure"),
                "xunce_soft_risk_exposure": xunce_metrics.get("soft_risk_exposure"),
                "hard_risk_clean_pair": hard_risk_clean_pair,
                "teacher_selected_higher_coverage": bool(teacher_higher_coverage),
                "teacher_selected_lower_or_acceptable_cost": bool(teacher_lower_cost or teacher_acceptable_cost),
                "sample_weight": sample_weight,
                "training_signal_type": "teacher_imitation_label",
                "counterfactual_type": "xunce_on_policy_same_candidate_set_candidate_audit_replay",
            }
        )
    return labels


def _select_teacher_candidate(rows: list[dict[str, Any]], *, profile: Any, denominator: float) -> tuple[dict[str, Any] | None, Any | None]:
    best: tuple[float, float, float, int, dict[str, Any], Any] | None = None
    for row in rows:
        if row.get("action_mask_valid") is not True:
            continue
        if _candidate_audit_hard_risk_violation(row):
            continue
        index = _int_value(row.get("candidate_index"))
        if index is None:
            continue
        coverage = _float_value(row.get("expected_new_coverage_cell_count")) or 0.0
        path_cost = _float_value(row.get("path_cost")) or 0.0
        soft_risk = _float_value(row.get("soft_risk_exposure"))
        if soft_risk is None:
            soft_risk = _float_value(row.get("path_risk_exposure"))
        if soft_risk is None:
            soft_risk = _float_value(row.get("risk_cost_weighted")) or 0.0
        result = compute_canonical_reward_components(
            {
                "coverage_gain_rate": coverage / max(float(denominator), 1.0e-9),
                "roi_coverage": _float_value(row.get("roi_weighted_coverage_delta")) or coverage,
                "information_gain": 0.0,
                "path_cost_m": path_cost,
                "soft_risk_exposure": soft_risk,
                "hard_risk_violation": False,
                "fallback_used": False,
                "failure": False,
            },
            profile,
        )
        score = (float(result.reward), coverage, -(_float_value(row.get("risk")) or 0.0), -index, row, result)
        if best is None or score[:4] > best[:4]:
            best = score
    if best is None:
        return None, None
    return best[4], best[5]


def _metrics_from_candidate_audit_row(row: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(row, dict):
        return {}
    soft_risk = _float_value(row.get("soft_risk_exposure"))
    if soft_risk is None:
        soft_risk = _float_value(row.get("path_risk_exposure"))
    if soft_risk is None:
        soft_risk = _float_value(row.get("risk_cost_weighted"))
    return {
        "candidate_index": _int_value(row.get("candidate_index")),
        "candidate_cell": row.get("candidate_cell"),
        "new_covered_cell_count": _float_value(row.get("expected_new_coverage_cell_count")),
        "expected_new_coverage_cell_count": _float_value(row.get("expected_new_coverage_cell_count")),
        "roi_weighted_coverage_delta": _float_value(row.get("roi_weighted_coverage_delta")),
        "path_cost": _float_value(row.get("path_cost")),
        "risk": _float_value(row.get("risk")),
        "risk_cost_weighted": _float_value(row.get("risk_cost_weighted")),
        "soft_risk_exposure": soft_risk,
        "path_allowed_by_risk": row.get("path_allowed_by_risk"),
        "hard_risk_flags": row.get("hard_risk_flags") if isinstance(row.get("hard_risk_flags"), list) else [],
        "hard_risk_violation": _candidate_audit_hard_risk_violation(row),
    }


def _candidate_audit_hard_risk_violation(row: dict[str, Any]) -> bool:
    return row.get("path_allowed_by_risk") is False or bool(row.get("hard_risk_flags") or [])


def _build_trainable_labels(
    rows: list[dict[str, Any]], *, expected_profile: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    labels: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        reason = _exclusion_reason(row)
        if reason:
            exclusions.append(_exclusion(row, reason_code=reason, source_index=index))
            continue
        labels.append(_sample_from_label(row, source_index=index, expected_profile=expected_profile))
    return labels, exclusions


def _exclusion_reason(row: dict[str, Any]) -> str | None:
    if row.get("baseline_policy") != "xunce":
        return "baseline_policy_not_xunce"
    if row.get("same_candidate_set") is not True:
        return "different_candidate_set"
    if row.get("hard_risk_clean_pair") is not True:
        return "hard_risk_not_clean"
    teacher_index = _int_value(row.get("teacher_action_index"))
    xunce_index = _int_value(row.get("xunce_action_index"))
    if teacher_index is None or xunce_index is None:
        return "missing_selected_action_index"
    if teacher_index == xunce_index:
        return "teacher_and_xunce_selected_same_action"
    if _float_value(row.get("sample_weight")) is None or float(row.get("sample_weight") or 0.0) <= 0.0:
        return "invalid_sample_weight"
    if not isinstance(row.get("candidate_set_hash"), str) or not row.get("candidate_set_hash"):
        return "missing_candidate_set_hash"
    if not isinstance(row.get("covered_cells_hash"), str) or not row.get("covered_cells_hash"):
        return "missing_covered_cells_hash"
    return None


def _sample_from_label(row: dict[str, Any], *, source_index: int, expected_profile: dict[str, Any]) -> dict[str, Any]:
    teacher_metrics = row.get("teacher_candidate_metrics") if isinstance(row.get("teacher_candidate_metrics"), dict) else {}
    xunce_metrics = row.get("xunce_candidate_metrics") if isinstance(row.get("xunce_candidate_metrics"), dict) else {}
    return {
        "schema_version": LABEL_SCHEMA_VERSION,
        "source_index": int(source_index),
        "source_type": "stage20_1_on_policy_teacher_label",
        "scenario_id": str(row.get("scenario_id")),
        "step_index": _int_value(row.get("step_index")),
        "baseline_policy": "xunce",
        "teacher_policy": "canonical_reward_rerank_oracle",
        "same_candidate_set": True,
        "candidate_set_hash": row.get("candidate_set_hash"),
        "current_cell": row.get("current_cell"),
        "covered_cells_hash": row.get("covered_cells_hash"),
        "teacher_action_index": _int_value(row.get("teacher_action_index")),
        "xunce_action_index": _int_value(row.get("xunce_action_index")),
        "teacher_profile_id": expected_profile.get("teacher_profile_id") or row.get("teacher_profile_id"),
        "teacher_profile_hash": expected_profile.get("teacher_profile_hash") or row.get("teacher_profile_hash"),
        "teacher_reward_components": row.get("teacher_reward_components"),
        "xunce_candidate_metrics": xunce_metrics,
        "teacher_candidate_metrics": teacher_metrics,
        "oracle_new_covered_cell_count": _float_value(teacher_metrics.get("new_covered_cell_count")),
        "xunce_new_covered_cell_count": _float_value(xunce_metrics.get("new_covered_cell_count")),
        "oracle_path_cost": _float_value(teacher_metrics.get("path_cost")),
        "xunce_path_cost": _float_value(xunce_metrics.get("path_cost")),
        "oracle_soft_risk_exposure": _float_value(teacher_metrics.get("soft_risk_exposure")),
        "xunce_soft_risk_exposure": _float_value(xunce_metrics.get("soft_risk_exposure")),
        "hard_risk_clean_pair": True,
        "teacher_selected_higher_coverage": row.get("teacher_selected_higher_coverage"),
        "teacher_selected_lower_or_acceptable_cost": row.get("teacher_selected_lower_or_acceptable_cost"),
        "sample_weight": float(row.get("sample_weight") or 0.0),
        "training_signal_type": "teacher_imitation_label",
    }


def _exclusion(row: dict[str, Any], *, reason_code: str, source_index: int) -> dict[str, Any]:
    return {
        "schema_version": EXCLUSION_SCHEMA_VERSION,
        "source_index": int(source_index),
        "scenario_id": row.get("scenario_id"),
        "step_index": row.get("step_index"),
        "baseline_policy": row.get("baseline_policy"),
        "candidate_set_hash": row.get("candidate_set_hash"),
        "reason_code": reason_code,
        "same_candidate_set": row.get("same_candidate_set"),
        "hard_risk_clean_pair": row.get("hard_risk_clean_pair"),
    }


def _dataset_stats(*, trainable: list[dict[str, Any]], exclusions: list[dict[str, Any]], raw_rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": DATASET_STATS_SCHEMA_VERSION,
        "raw_teacher_label_count": len(raw_rows),
        "trainable_label_count": len(trainable),
        "excluded_label_count": len(exclusions),
        "scenario_count": len({row.get("scenario_id") for row in trainable}),
        "exclusion_reason_counts": dict(Counter(row["reason_code"] for row in exclusions)),
    }


def _route(*, config: dict[str, Any], blocking: list[str], label_count: int, raw_label_count: int) -> str:
    if blocking:
        return ROUTE_INPUTS
    if raw_label_count <= 0:
        return ROUTE_LABEL_ROLLOUT
    if label_count >= config["min_trainable_label_count_for_checkpoint_preflight"]:
        return ROUTE_PREFLIGHT
    return ROUTE_RERUN_STAGE20


def _routing_payload(route: str) -> dict[str, Any]:
    return {
        "schema_version": ROUTING_SCHEMA_VERSION,
        "primary_route": route,
        "stage20_authorized": False,
        "training_or_release_authorized": False,
        "runs_new_ppo_update": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
    }


def _render_report(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Xunce Stage 20.1 Same-Candidate Oracle Imitation Evidence",
            "",
            f"- status: `{summary['status']}`",
            f"- next_required_change: `{summary['next_required_change']}`",
            f"- trainable_label_count: `{summary['trainable_label_count']}`",
            f"- excluded_label_count: `{summary['excluded_label_count']}`",
            f"- stage20_authorized: `{summary['stage20_authorized']}`",
            "",
            "Stage 20.1 collects oracle teacher labels on Xunce on-policy candidate sets. It does not change Xunce actions, run PPO, publish checkpoints, replace the default policy, connect an executor, or start canary traffic.",
            "",
        ]
    )


def _boundary_rejections(config: dict[str, Any]) -> list[str]:
    reasons = [field for field in BOUNDARY_FIELDS if config.get(field) is True]
    if float(config.get("canary_traffic_fraction", 0.0) or 0.0) != 0.0:
        reasons.append("canary_traffic_fraction")
    return reasons


def _boundary_violation(payload: dict[str, Any]) -> bool:
    if not isinstance(payload, dict):
        return False
    if float(payload.get("canary_traffic_fraction", 0.0) or 0.0) != 0.0:
        return True
    return any(payload.get(field) is True for field in BOUNDARY_FIELDS)


def _resolve_path(path: Path | None, repo_root: Path) -> Path:
    if path is None:
        raise ConfigError("path must not be None")
    return path if path.is_absolute() else (repo_root / path)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _iter_jsonl(path: Path):
    if not path.is_file():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _int_required(payload: dict[str, Any], key: str) -> int:
    if key not in payload or isinstance(payload[key], bool):
        raise ConfigError(f"{key} must be an integer")
    value = int(payload[key])
    if value < 0:
        raise ConfigError(f"{key} must be non-negative")
    return value


def _int_value(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _float_value(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
