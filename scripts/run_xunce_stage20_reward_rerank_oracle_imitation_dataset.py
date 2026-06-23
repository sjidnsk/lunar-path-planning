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


CONFIG_SCHEMA_VERSION = "xunce-stage20-oracle-imitation-config/v1"
SUMMARY_SCHEMA_VERSION = "xunce-stage20-oracle-imitation-summary/v1"
SAMPLE_SCHEMA_VERSION = "xunce-stage20-oracle-imitation-teacher-sample/v1"
EXCLUSION_SCHEMA_VERSION = "xunce-stage20-oracle-imitation-exclusion/v1"
DATASET_STATS_SCHEMA_VERSION = "xunce-stage20-oracle-imitation-dataset-stats/v1"
DRY_RUN_SCHEMA_VERSION = "xunce-stage20-oracle-imitation-dry-run/v1"
ROUTING_SCHEMA_VERSION = "xunce-stage20-next-stage-routing/v1"
MANIFEST_SCHEMA_VERSION = "xunce-stage20-oracle-imitation-manifest/v1"

DEFAULT_CONFIG = "configs/xunce_stage20_reward_rerank_oracle_imitation_dataset_v1.json"
DEFAULT_STAGE19_ROOT = (
    "D:/CodexDownloads/lunar-path-planning/stage19_evaluator_critic_preflight/"
    "outputs/path_feedback_batch_xunce_stage19_evaluator_critic_preflight_v1"
)
DEFAULT_OUTPUT_ROOT = (
    "D:/CodexDownloads/lunar-path-planning/stage20_reward_rerank_oracle_imitation_dataset/"
    "outputs/path_feedback_batch_xunce_stage20_reward_rerank_oracle_imitation_dataset_v1"
)

STAGE19_SUMMARY_FILE = "xunce-stage19-evaluator-critic-preflight-summary.json"
STAGE19_PRACTICAL_TARGET_FILE = "xunce-stage19-practical-target-selection.json"
STAGE19_CRITIC_FILE = "xunce-stage19-critic-target-readiness.json"
STAGE19_PREFERENCE_FILE = "xunce-stage19-preference-pair-audit.jsonl"
STAGE20_1_SUMMARY_FILE = "xunce-stage20-1-same-candidate-oracle-imitation-summary.json"
STAGE20_1_LABELS_FILE = "xunce-stage20-1-on-policy-teacher-labels.jsonl"

SUMMARY_FILE = "xunce-stage20-oracle-imitation-summary.json"
SAMPLES_FILE = "xunce-stage20-oracle-imitation-teacher-samples.jsonl"
EXCLUSION_FILE = "xunce-stage20-oracle-imitation-exclusion-report.jsonl"
DATASET_STATS_FILE = "xunce-stage20-oracle-imitation-dataset-stats.json"
DRY_RUN_FILE = "xunce-stage20-oracle-imitation-dry-run-summary.json"
ROUTING_FILE = "xunce-stage20-next-stage-routing.json"
REPORT_FILE = "xunce-stage20-report.md"
MANIFEST_FILE = "xunce-stage20-manifest.json"

ROUTE_BOUNDARY = "resolve_stage20_oracle_imitation_boundary_rejections"
ROUTE_INPUTS = "rerun_stage19_evaluator_critic_preflight"
ROUTE_COLLECT = "collect_more_reward_rerank_same_candidate_preference_evidence"
ROUTE_PREFLIGHT = "stage20_1_supervised_oracle_imitation_checkpoint_preflight"

BOUNDARY_FIELDS = (
    "stage20_authorized",
    "runs_new_ppo_update",
    "publishes_checkpoint",
    "replaces_default_policy",
    "connects_real_executor",
    "starts_online_canary",
    "training_or_release_authorized",
    "real_world_release_approved",
    "real_world_performance_claimed",
)


class ConfigError(ValueError):
    pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build Stage 20 reward-rerank oracle imitation dataset.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--stage19-evaluator-critic-preflight-root", default=DEFAULT_STAGE19_ROOT)
    parser.add_argument("--stage20-1-same-candidate-oracle-imitation-root")
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args(argv)

    try:
        summary = run_xunce_stage20_reward_rerank_oracle_imitation_dataset(
            config_path=Path(args.config),
            stage19_evaluator_critic_preflight_root=Path(args.stage19_evaluator_critic_preflight_root),
            stage20_1_same_candidate_oracle_imitation_root=(
                Path(args.stage20_1_same_candidate_oracle_imitation_root)
                if args.stage20_1_same_candidate_oracle_imitation_root
                else None
            ),
            output_root=Path(args.output_root),
            repo_root=Path(args.repo_root),
        )
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": summary["status"],
                "next_required_change": summary["next_required_change"],
                "trainable_pair_count": summary["trainable_pair_count"],
                "dry_run_executed": summary["dry_run_summary"]["dry_run_executed"],
                "stage20_authorized": summary["stage20_authorized"],
                "summary": str(Path(summary["summary"]).resolve()),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if summary["status"] != "failed" else 1


def run_xunce_stage20_reward_rerank_oracle_imitation_dataset(
    *,
    config_path: Path,
    stage19_evaluator_critic_preflight_root: Path,
    stage20_1_same_candidate_oracle_imitation_root: Path | None = None,
    output_root: Path,
    repo_root: Path,
) -> dict[str, Any]:
    repo_root = Path(repo_root).resolve()
    config = _load_config(_resolve_path(config_path, repo_root))
    stage19_root = _resolve_path(stage19_evaluator_critic_preflight_root, repo_root)
    output_root = _resolve_path(output_root, repo_root)
    output_root.mkdir(parents=True, exist_ok=True)

    blocking = _boundary_rejections(config)
    stage19_summary = _read_json(stage19_root / STAGE19_SUMMARY_FILE)
    practical_target = _read_json(stage19_root / STAGE19_PRACTICAL_TARGET_FILE)
    critic = _read_json(stage19_root / STAGE19_CRITIC_FILE)
    preference_rows = _read_jsonl(stage19_root / STAGE19_PREFERENCE_FILE)
    blocking.extend(_stage19_input_rejections(stage19_summary, practical_target, critic, preference_rows))
    stage20_1_root = (
        _resolve_path(stage20_1_same_candidate_oracle_imitation_root, repo_root)
        if stage20_1_same_candidate_oracle_imitation_root is not None
        else None
    )
    stage20_1_summary: dict[str, Any] = {}
    stage20_1_rows: list[dict[str, Any]] = []
    if stage20_1_root is not None:
        stage20_1_summary = _read_json(stage20_1_root / STAGE20_1_SUMMARY_FILE)
        stage20_1_rows = _read_jsonl(stage20_1_root / STAGE20_1_LABELS_FILE)
        blocking.extend(
            _stage20_1_input_rejections(
                summary=stage20_1_summary,
                labels=stage20_1_rows,
                practical_target=practical_target,
            )
        )

    samples: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    stage19_sample_count = 0
    stage20_1_sample_count = 0
    if not blocking:
        samples, exclusions = _build_samples(
            preference_rows,
            config=config,
            stage19_summary=stage19_summary,
            practical_target=practical_target,
        )
        stage19_sample_count = len(samples)
        if stage20_1_rows:
            stage20_1_samples, stage20_1_exclusions = _build_stage20_1_samples(
                stage20_1_rows,
                stage19_summary=stage19_summary,
                practical_target=practical_target,
            )
            samples = _dedupe_samples(samples + stage20_1_samples)
            stage20_1_sample_count = len([row for row in samples if row.get("source_type") == "stage20_1_on_policy_teacher_label"])
            exclusions.extend(stage20_1_exclusions)
    dry_run = _dry_run(samples, config=config) if not blocking else _dry_run_skipped("blocking_input_rejections", 0)
    dataset_stats = _dataset_stats(
        samples=samples,
        exclusions=exclusions,
        preference_rows=preference_rows,
        stage19_trainable_count=stage19_sample_count,
        stage20_1_trainable_count=stage20_1_sample_count,
        stage20_1_raw_label_count=len(stage20_1_rows),
    )
    route = _route(config=config, blocking=blocking, sample_count=len(samples), dry_run=dry_run)
    status = "failed" if route in {ROUTE_BOUNDARY, ROUTE_INPUTS} else "passed"

    routing = {
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
    generated_at = datetime.now(timezone.utc).isoformat()
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": status,
        "stage19_evaluator_critic_preflight_root": str(stage19_root.resolve()),
        "stage20_1_same_candidate_oracle_imitation_root": str(stage20_1_root.resolve()) if stage20_1_root else None,
        "stage19_summary_schema_version": stage19_summary.get("schema_version"),
        "stage20_1_summary_schema_version": stage20_1_summary.get("schema_version"),
        "profile_id": stage19_summary.get("profile_id"),
        "profile_version": stage19_summary.get("profile_version"),
        "profile_hash": stage19_summary.get("profile_hash"),
        "teacher_policy": "canonical_reward_rerank_oracle",
        "teacher_profile_id": _teacher_profile_id(practical_target),
        "teacher_profile_hash": _teacher_profile_hash(practical_target),
        "trainable_pair_count": len(samples),
        "excluded_pair_count": len(exclusions),
        "dataset_stats": dataset_stats,
        "dry_run_summary": dry_run,
        "next_stage_routing": routing,
        "next_required_change": route,
        "blocking_reason_codes": blocking,
        "reason_codes": list(blocking) + list(dry_run.get("reason_codes", [])),
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
        "teacher_samples": str((output_root / SAMPLES_FILE).resolve()),
        "exclusion_report": str((output_root / EXCLUSION_FILE).resolve()),
        "dataset_stats": str((output_root / DATASET_STATS_FILE).resolve()),
        "dry_run_summary": str((output_root / DRY_RUN_FILE).resolve()),
        "routing": str((output_root / ROUTING_FILE).resolve()),
        "report": str((output_root / REPORT_FILE).resolve()),
    }

    _write_json(output_root / SUMMARY_FILE, summary)
    _write_jsonl(output_root / SAMPLES_FILE, samples)
    _write_jsonl(output_root / EXCLUSION_FILE, exclusions)
    _write_json(output_root / DATASET_STATS_FILE, dataset_stats)
    _write_json(output_root / DRY_RUN_FILE, dry_run)
    _write_json(output_root / ROUTING_FILE, routing)
    _write_json(output_root / MANIFEST_FILE, manifest)
    (output_root / REPORT_FILE).write_text(_render_report(summary), encoding="utf-8")
    return summary


def _load_config(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    if payload.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ConfigError(f"schema_version must be {CONFIG_SCHEMA_VERSION}")
    config = dict(payload)
    config["min_trainable_pair_count_for_dry_run"] = _int_required(config, "min_trainable_pair_count_for_dry_run")
    config["min_trainable_pair_count_for_checkpoint_training"] = _int_required(
        config, "min_trainable_pair_count_for_checkpoint_training"
    )
    for key in ("require_same_candidate_set", "require_hard_risk_clean_pair"):
        config[key] = _bool_required(config, key)
    for key in BOUNDARY_FIELDS:
        config[key] = bool(config.get(key, False))
    config["canary_traffic_fraction"] = float(config.get("canary_traffic_fraction", 0.0) or 0.0)
    return config


def _build_samples(
    rows: list[dict[str, Any]],
    *,
    config: dict[str, Any],
    stage19_summary: dict[str, Any],
    practical_target: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    samples: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        reason = _exclusion_reason(row, config=config)
        weight = _sample_weight(row)
        if reason is None and weight is None:
            reason = "oracle_not_preferred_by_coverage_cost_rule"
        if reason is not None:
            exclusions.append(_exclusion(row, reason_code=reason, source_index=index))
            continue
        samples.append(
            {
                "schema_version": SAMPLE_SCHEMA_VERSION,
                "source_index": index,
                "source_type": "stage19_preference_pair_audit",
                "scenario_id": str(row.get("scenario_id")),
                "step_index": _int_value(row.get("step_index")),
                "candidate_set_hash": row.get("oracle_candidate_set_hash"),
                "teacher_action_index": _int_value(row.get("oracle_selected_action_index")),
                "xunce_action_index": _int_value(row.get("baseline_selected_action_index")),
                "teacher_policy": "canonical_reward_rerank_oracle",
                "teacher_profile_id": _teacher_profile_id(practical_target),
                "teacher_profile_hash": _teacher_profile_hash(practical_target),
                "profile_id": stage19_summary.get("profile_id"),
                "profile_version": stage19_summary.get("profile_version"),
                "profile_hash": stage19_summary.get("profile_hash"),
                "oracle_new_covered_cell_count": _float_value(row.get("oracle_new_covered_cell_count")),
                "xunce_new_covered_cell_count": _float_value(row.get("baseline_new_covered_cell_count")),
                "oracle_path_cost": _float_value(row.get("oracle_path_cost")),
                "xunce_path_cost": _float_value(row.get("baseline_path_cost")),
                "oracle_soft_risk_exposure": _float_value(row.get("oracle_soft_risk_exposure")),
                "xunce_soft_risk_exposure": _float_value(row.get("baseline_soft_risk_exposure")),
                "hard_risk_clean_pair": True,
                "sample_weight": float(weight),
                "training_signal_type": "teacher_imitation_label",
            }
        )
    return samples, exclusions


def _build_stage20_1_samples(
    rows: list[dict[str, Any]],
    *,
    stage19_summary: dict[str, Any],
    practical_target: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    samples: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        reason = _stage20_1_sample_exclusion_reason(row, practical_target=practical_target)
        if reason is not None:
            exclusions.append(_exclusion(row, reason_code=reason, source_index=index))
            continue
        samples.append(
            {
                "schema_version": SAMPLE_SCHEMA_VERSION,
                "source_index": int(row.get("source_index", index) or index),
                "source_type": "stage20_1_on_policy_teacher_label",
                "scenario_id": str(row.get("scenario_id")),
                "step_index": _int_value(row.get("step_index")),
                "candidate_set_hash": row.get("candidate_set_hash"),
                "teacher_action_index": _int_value(row.get("teacher_action_index")),
                "xunce_action_index": _int_value(row.get("xunce_action_index")),
                "teacher_policy": "canonical_reward_rerank_oracle",
                "teacher_profile_id": _teacher_profile_id(practical_target),
                "teacher_profile_hash": _teacher_profile_hash(practical_target),
                "profile_id": stage19_summary.get("profile_id"),
                "profile_version": stage19_summary.get("profile_version"),
                "profile_hash": stage19_summary.get("profile_hash"),
                "oracle_new_covered_cell_count": _float_value(row.get("oracle_new_covered_cell_count")),
                "xunce_new_covered_cell_count": _float_value(row.get("xunce_new_covered_cell_count")),
                "oracle_path_cost": _float_value(row.get("oracle_path_cost")),
                "xunce_path_cost": _float_value(row.get("xunce_path_cost")),
                "oracle_soft_risk_exposure": _float_value(row.get("oracle_soft_risk_exposure")),
                "xunce_soft_risk_exposure": _float_value(row.get("xunce_soft_risk_exposure")),
                "hard_risk_clean_pair": True,
                "sample_weight": float(row.get("sample_weight") or 0.0),
                "training_signal_type": "teacher_imitation_label",
            }
        )
    return samples, exclusions


def _stage20_1_sample_exclusion_reason(row: dict[str, Any], *, practical_target: dict[str, Any]) -> str | None:
    if row.get("baseline_policy") != "xunce":
        return "baseline_policy_not_xunce"
    if row.get("same_candidate_set") is not True:
        return "different_candidate_set"
    if row.get("hard_risk_clean_pair") is not True:
        return "hard_risk_not_clean"
    if row.get("teacher_policy") != "canonical_reward_rerank_oracle":
        return "teacher_policy_not_oracle"
    if row.get("teacher_profile_hash") != _teacher_profile_hash(practical_target):
        return "stage20_1_teacher_profile_hash_mismatch"
    teacher_index = _int_value(row.get("teacher_action_index"))
    xunce_index = _int_value(row.get("xunce_action_index"))
    if teacher_index is None or xunce_index is None:
        return "missing_selected_action_index"
    if teacher_index == xunce_index:
        return "oracle_and_xunce_selected_same_action"
    weight = _float_value(row.get("sample_weight"))
    if weight is None or weight <= 0.0:
        return "invalid_sample_weight"
    if not row.get("candidate_set_hash"):
        return "missing_candidate_set_hash"
    return None


def _dedupe_samples(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for row in rows:
        key = (
            row.get("scenario_id"),
            row.get("step_index"),
            row.get("candidate_set_hash"),
            row.get("teacher_action_index"),
            row.get("xunce_action_index"),
        )
        if key in seen:
            continue
        seen.add(key)
        output.append(row)
    return output


def _exclusion_reason(row: dict[str, Any], *, config: dict[str, Any]) -> str | None:
    if row.get("baseline_policy") != "xunce":
        return "baseline_policy_not_xunce"
    if config["require_same_candidate_set"] and row.get("same_candidate_set") is not True:
        return "different_candidate_set"
    if row.get("oracle_candidate_set_hash") != row.get("baseline_candidate_set_hash"):
        return "different_candidate_set"
    if config["require_hard_risk_clean_pair"] and row.get("hard_risk_clean_pair") is not True:
        return "hard_risk_not_clean"
    if _int_value(row.get("oracle_selected_action_index")) == _int_value(row.get("baseline_selected_action_index")):
        return "oracle_and_xunce_selected_same_action"
    if _int_value(row.get("oracle_selected_action_index")) is None or _int_value(row.get("baseline_selected_action_index")) is None:
        return "missing_selected_action_index"
    return None


def _sample_weight(row: dict[str, Any]) -> float | None:
    oracle_cov = _float_value(row.get("oracle_new_covered_cell_count"))
    xunce_cov = _float_value(row.get("baseline_new_covered_cell_count"))
    oracle_cost = _float_value(row.get("oracle_path_cost"))
    xunce_cost = _float_value(row.get("baseline_path_cost"))
    if oracle_cov is None or xunce_cov is None or oracle_cost is None or xunce_cost is None:
        return None
    if oracle_cov > xunce_cov and (oracle_cost <= xunce_cost * 1.25 or row.get("oracle_selected_lower_or_acceptable_cost") is True):
        return 1.0
    if oracle_cov == xunce_cov and oracle_cost < xunce_cost:
        return 0.5
    return None


def _dry_run(samples: list[dict[str, Any]], *, config: dict[str, Any]) -> dict[str, Any]:
    if len(samples) < config["min_trainable_pair_count_for_dry_run"]:
        return _dry_run_skipped("trainable_pair_count_below_dry_run_minimum", len(samples))
    try:
        from model_explorer.policy.features import PolicyObservation
        from model_explorer.policy.rollout import EpisodeMetrics, RolloutEpisode, RolloutInfo, RolloutTransition
        from model_explorer.policy.training import train_policy_on_episodes

        transitions = []
        for sample in samples:
            observation = PolicyObservation(
                candidate_feature_names=("coverage", "path_cost", "soft_risk_exposure"),
                candidate_features=(
                    (
                        float(sample["oracle_new_covered_cell_count"] or 0.0),
                        float(sample["oracle_path_cost"] or 0.0),
                        float(sample["oracle_soft_risk_exposure"] or 0.0),
                    ),
                    (
                        float(sample["xunce_new_covered_cell_count"] or 0.0),
                        float(sample["xunce_path_cost"] or 0.0),
                        float(sample["xunce_soft_risk_exposure"] or 0.0),
                    ),
                ),
                global_feature_names=("step_index", "sample_weight"),
                global_features=(float(sample["step_index"] or 0), float(sample["sample_weight"])),
                action_mask=(True, True),
                candidate_cells=(None, None),
                candidate_missing_feature_names=((), ()),
                candidate_missing_indicator_names=(),
                candidate_missing_indicators=((), ()),
            )
            margin = max(
                0.0,
                float(sample["oracle_new_covered_cell_count"] or 0.0) - float(sample["xunce_new_covered_cell_count"] or 0.0),
            ) / 100.0
            transitions.append(
                RolloutTransition(
                    observation=observation,
                    action_index=0,
                    log_prob=0.0,
                    value=0.0,
                    reward=1.0,
                    next_observation=None,
                    done=True,
                    info=RolloutInfo(
                        coverage_rate_delta=0.0,
                        path_cost=float(sample["oracle_path_cost"] or 0.0),
                        risk=0.0,
                        extra={
                            "teacher_action_index": 0,
                            "teacher_score_margin": margin,
                            "stage20_source_sample_index": sample["source_index"],
                        },
                    ),
                )
            )
        episode = RolloutEpisode(
            transitions=tuple(transitions),
            metrics=EpisodeMetrics(final_coverage_rate=0.0, cumulative_coverage_rate_delta=0.0),
        )
        result = train_policy_on_episodes(
            (episode,),
            seed=0,
            hidden_size=16,
            learning_rate=1.0e-3,
            epochs=2,
            teacher_imitation_weight=1.0,
            return_mode="reward_as_return",
        )
    except Exception as exc:  # noqa: BLE001
        return {
            "schema_version": DRY_RUN_SCHEMA_VERSION,
            "dry_run_executed": True,
            "dry_run_passed": False,
            "trainable_sample_count": len(samples),
            "ignored_sample_count": 0,
            "reason_codes": ["teacher_imitation_dry_run_failed"],
            "training_error": str(exc),
            "runs_new_ppo_update": False,
            "publishes_checkpoint": False,
        }
    teacher = result.get("teacher_imitation", {})
    return {
        "schema_version": DRY_RUN_SCHEMA_VERSION,
        "dry_run_executed": True,
        "dry_run_passed": True,
        "trainable_sample_count": len(samples),
        "ignored_sample_count": int(teacher.get("ignored_teacher_label_count", 0) or 0),
        "teacher_imitation_loss": float(teacher.get("teacher_imitation_loss", 0.0) or 0.0),
        "teacher_action_accuracy": float(teacher.get("teacher_action_accuracy", 0.0) or 0.0),
        "training_result": {
            "sample_count": result.get("sample_count"),
            "epochs": result.get("epochs"),
            "teacher_imitation": teacher,
            "warnings": result.get("warnings", []),
        },
        "reason_codes": [],
        "runs_new_ppo_update": False,
        "publishes_checkpoint": False,
    }


def _dry_run_skipped(reason: str, sample_count: int) -> dict[str, Any]:
    return {
        "schema_version": DRY_RUN_SCHEMA_VERSION,
        "dry_run_executed": False,
        "dry_run_passed": False,
        "trainable_sample_count": int(sample_count),
        "ignored_sample_count": 0,
        "teacher_imitation_loss": None,
        "teacher_action_accuracy": None,
        "reason_codes": [reason],
        "runs_new_ppo_update": False,
        "publishes_checkpoint": False,
    }


def _route(*, config: dict[str, Any], blocking: list[str], sample_count: int, dry_run: dict[str, Any]) -> str:
    if blocking:
        if any(reason in BOUNDARY_FIELDS or "boundary" in reason for reason in blocking):
            return ROUTE_BOUNDARY
        return ROUTE_INPUTS
    if sample_count < config["min_trainable_pair_count_for_checkpoint_training"]:
        return ROUTE_COLLECT
    if dry_run.get("dry_run_passed") is not True:
        return ROUTE_COLLECT
    return ROUTE_PREFLIGHT


def _dataset_stats(
    *,
    samples: list[dict[str, Any]],
    exclusions: list[dict[str, Any]],
    preference_rows: list[dict[str, Any]],
    stage19_trainable_count: int = 0,
    stage20_1_trainable_count: int = 0,
    stage20_1_raw_label_count: int = 0,
) -> dict[str, Any]:
    return {
        "schema_version": DATASET_STATS_SCHEMA_VERSION,
        "preference_audit_row_count": len(preference_rows),
        "stage19_trainable_pair_count": int(stage19_trainable_count),
        "stage20_1_raw_label_count": int(stage20_1_raw_label_count),
        "stage20_1_trainable_label_count": int(stage20_1_trainable_count),
        "trainable_pair_count": len(samples),
        "excluded_pair_count": len(exclusions),
        "same_candidate_xunce_pair_count": sum(
            1
            for row in preference_rows
            if row.get("baseline_policy") == "xunce" and row.get("same_candidate_set") is True
        ),
        "exclusion_reason_counts": dict(Counter(row["reason_code"] for row in exclusions)),
        "scenario_count": len({row["scenario_id"] for row in samples}),
    }


def _stage19_input_rejections(
    summary: dict[str, Any],
    practical_target: dict[str, Any],
    critic: dict[str, Any],
    preference_rows: list[dict[str, Any]],
) -> list[str]:
    reasons: list[str] = []
    if summary.get("schema_version") != "xunce-stage19-evaluator-critic-preflight-summary/v1":
        reasons.append("invalid_stage19_summary_schema")
    if summary.get("next_required_change") != "stage20_reward_rerank_oracle_preference_dataset_preparation":
        reasons.append("stage19_not_routed_to_stage20_preference_dataset_preparation")
    if summary.get("stage20_authorized") is not False or summary.get("training_or_release_authorized") is not False:
        reasons.append("stage19_authorization_boundary_not_false")
    if practical_target.get("schema_version") != "xunce-stage19-practical-target-selection/v1":
        reasons.append("invalid_stage19_practical_target_schema")
    if critic.get("schema_version") != "xunce-stage19-critic-target-readiness/v1":
        reasons.append("invalid_stage19_critic_schema")
    if critic.get("critic_target_ready") is not True:
        reasons.append("stage19_critic_target_not_ready")
    if not preference_rows:
        reasons.append("missing_stage19_preference_pair_audit")
    return reasons


def _stage20_1_input_rejections(
    *,
    summary: dict[str, Any],
    labels: list[dict[str, Any]],
    practical_target: dict[str, Any],
) -> list[str]:
    reasons: list[str] = []
    if summary.get("schema_version") != "xunce-stage20-1-same-candidate-oracle-imitation-summary/v1":
        reasons.append("invalid_stage20_1_summary_schema")
    if summary.get("teacher_profile_hash") != _teacher_profile_hash(practical_target):
        reasons.append("stage20_1_teacher_profile_hash_mismatch")
    if summary.get("teacher_profile_id") != _teacher_profile_id(practical_target):
        reasons.append("stage20_1_teacher_profile_id_mismatch")
    if not labels:
        reasons.append("missing_stage20_1_on_policy_teacher_labels")
    for key in BOUNDARY_FIELDS:
        if summary.get(key) is True:
            reasons.append(f"stage20_1_{key}")
    if float(summary.get("canary_traffic_fraction", 0.0) or 0.0) != 0.0:
        reasons.append("stage20_1_canary_traffic_fraction")
    return reasons


def _boundary_rejections(config: dict[str, Any]) -> list[str]:
    reasons = [key for key in BOUNDARY_FIELDS if config.get(key) is True]
    if float(config.get("canary_traffic_fraction", 0.0) or 0.0) != 0.0:
        reasons.append("canary_traffic_fraction")
    return reasons


def _exclusion(row: dict[str, Any], *, reason_code: str, source_index: int) -> dict[str, Any]:
    return {
        "schema_version": EXCLUSION_SCHEMA_VERSION,
        "source_index": source_index,
        "scenario_id": row.get("scenario_id"),
        "step_index": row.get("step_index"),
        "baseline_policy": row.get("baseline_policy"),
        "reason_code": reason_code,
        "same_candidate_set": row.get("same_candidate_set"),
        "hard_risk_clean_pair": row.get("hard_risk_clean_pair"),
    }


def _teacher_profile_id(practical_target: dict[str, Any]) -> str | None:
    primary = practical_target.get("primary")
    if isinstance(primary, dict):
        value = primary.get("canonical_reward_rerank_profile_id")
        if value is not None:
            return str(value)
    return None


def _teacher_profile_hash(practical_target: dict[str, Any]) -> str | None:
    primary = practical_target.get("primary")
    if isinstance(primary, dict):
        value = primary.get("canonical_reward_rerank_profile_hash")
        if value is not None:
            return str(value)
    return None


def _render_report(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Xunce Stage 20 Reward-Rerank Oracle Imitation Dataset",
            "",
            f"- status: `{summary['status']}`",
            f"- next_required_change: `{summary['next_required_change']}`",
            f"- trainable_pair_count: `{summary['trainable_pair_count']}`",
            f"- excluded_pair_count: `{summary['excluded_pair_count']}`",
            f"- dry_run_executed: `{summary['dry_run_summary']['dry_run_executed']}`",
            f"- dry_run_passed: `{summary['dry_run_summary']['dry_run_passed']}`",
            f"- stage20_authorized: `{summary['stage20_authorized']}`",
            "",
            "Stage 20 converts strict same-candidate reward-rerank oracle choices into teacher labels and runs a local imitation dry-run. It does not run PPO, publish checkpoints, replace the default policy, connect an executor, or start canary traffic.",
            "",
        ]
    )


def _resolve_path(path: Path, repo_root: Path) -> Path:
    return path if path.is_absolute() else (repo_root / path)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


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


def _bool_required(payload: dict[str, Any], key: str) -> bool:
    if not isinstance(payload.get(key), bool):
        raise ConfigError(f"{key} must be boolean")
    return bool(payload[key])


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
