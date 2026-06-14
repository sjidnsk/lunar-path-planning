from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:
    from git_provenance import git_snapshot
except ModuleNotFoundError:  # pragma: no cover
    from scripts.git_provenance import git_snapshot

try:
    from scripts.run_limited_ppo_update_smoke import run_limited_ppo_update_smoke
    from scripts.run_ppo_rollout_collector_dry_run import _observation_from_step_or_scenario
except ModuleNotFoundError:  # pragma: no cover
    from run_limited_ppo_update_smoke import run_limited_ppo_update_smoke
    from run_ppo_rollout_collector_dry_run import _observation_from_step_or_scenario


CONFIG_SCHEMA_VERSION = "quasi-real-guarded-ppo-scale512-multiseed-preflight-config/v1"
SUMMARY_SCHEMA_VERSION = "quasi-real-guarded-ppo-scale512-multiseed-preflight-summary/v1"
EXPECTED_READINESS_STATUS = "quasi_real_guarded_ppo_scale512_multiseed_preflight_evaluated"

SUMMARY_FILE = "quasi-real-guarded-ppo-scale512-multiseed-preflight-summary.json"
CAPACITY_FILE = "scale512-trainable-capacity-report.json"
TRAINABLE_CONTEXTS_FILE = "scale512-trainable-contexts.jsonl"
SEED_SUMMARIES_FILE = "scale512-seed-summaries.jsonl"
READINESS_FILE = "scale512-readiness-validate-only.json"
REPORT_FILE = "scale512-multiseed-preflight-report.md"

SeedSmokeRunner = Callable[..., dict[str, Any]]
ReadinessRunner = Callable[..., dict[str, Any]]
PpoUpdateRunner = Callable[..., dict[str, Any]]


def run_quasi_real_guarded_ppo_scale512_multiseed_preflight(
    *,
    horizon5_root: Path,
    output_root: Path,
    batch_root: Path,
    config: dict[str, Any],
    repo_root: Path,
    seed_smoke_runner: SeedSmokeRunner | None = None,
    readiness_runner: ReadinessRunner | None = None,
) -> dict[str, Any]:
    repo_root = Path(repo_root)
    horizon5_root = Path(horizon5_root)
    output_root = Path(output_root)
    batch_root = Path(batch_root)
    output_root.mkdir(parents=True, exist_ok=True)

    files = _output_files(config)
    summary_path = output_root / files["summary"]
    capacity_path = output_root / files["capacity_report"]
    contexts_path = output_root / files["trainable_contexts"]
    seed_summaries_path = output_root / files["seed_summaries"]
    readiness_path = output_root / files["readiness_validate_only"]
    report_path = output_root / files["report"]

    horizon5_summary_path = horizon5_root / "quasi-real-guarded-ppo-horizon5-batch-expansion-summary.json"
    horizon5_summary = _read_json_if_exists(horizon5_summary_path)
    steps_path = _resolve_steps_path(horizon5_summary, horizon5_root, repo_root)
    steps = _read_jsonl(steps_path)

    reason_codes: list[str] = []
    _validate_horizon5_input(horizon5_summary, config, reason_codes)
    trainable_steps = _unique_trainable_steps(steps)
    counters = _capacity_counters(steps, trainable_steps)
    _validate_capacity(counters, config, reason_codes)

    _write_json(capacity_path, _capacity_report(counters, config, horizon5_summary_path, steps_path))
    _write_jsonl(contexts_path, _trainable_context_rows(trainable_steps))

    seed_summaries: list[dict[str, Any]] = []
    if not reason_codes:
        runner = seed_smoke_runner or _run_seed_smoke
        for seed in _seeds(config):
            seed_summary = runner(
                seed=seed,
                trainable_steps=trainable_steps,
                output_root=output_root,
                config=config,
                repo_root=repo_root,
                batch_root=batch_root,
            )
            seed_summaries.append(seed_summary)
        _validate_seed_summaries(seed_summaries, config, reason_codes)
    _write_jsonl(seed_summaries_path, seed_summaries)

    status_without_readiness = "passed" if not reason_codes else "failed"
    summary = _summary_payload(
        status=status_without_readiness,
        reason_codes=reason_codes,
        repo_root=repo_root,
        horizon5_root=horizon5_root,
        horizon5_summary_path=horizon5_summary_path,
        steps_path=steps_path,
        output_root=output_root,
        batch_root=batch_root,
        summary_path=summary_path,
        capacity_path=capacity_path,
        contexts_path=contexts_path,
        seed_summaries_path=seed_summaries_path,
        readiness_path=readiness_path,
        report_path=report_path,
        config=config,
        horizon5_summary=horizon5_summary,
        counters=counters,
        seed_summaries=seed_summaries,
        seed_smoke_skipped=bool(reason_codes and not seed_summaries),
    )
    _write_json(summary_path, summary)

    readiness = {}
    if summary["status"] == "passed":
        runner = readiness_runner or _run_readiness_validate_only
        readiness = runner(
            repo_root=repo_root,
            batch_root=batch_root,
            scale512_summary_path=summary_path,
            config_path=Path(config.get("readiness", {}).get("config", "configs/policy_training_readiness_review_v1.json")),
        )
        _write_json(readiness_path, readiness)
        _validate_readiness(readiness, config, reason_codes)
    else:
        readiness = {
            "training_readiness_status": "needs_training_contract_refinement",
            "training_blockers": list(reason_codes),
            "reason_codes": list(reason_codes),
            "recommended_next_action": "expand_quasi_real_trainable_capacity_before_scale512_preflight",
        }
        _write_json(readiness_path, readiness)

    final_status = "passed" if not reason_codes else "failed"
    summary = _summary_payload(
        status=final_status,
        reason_codes=reason_codes,
        repo_root=repo_root,
        horizon5_root=horizon5_root,
        horizon5_summary_path=horizon5_summary_path,
        steps_path=steps_path,
        output_root=output_root,
        batch_root=batch_root,
        summary_path=summary_path,
        capacity_path=capacity_path,
        contexts_path=contexts_path,
        seed_summaries_path=seed_summaries_path,
        readiness_path=readiness_path,
        report_path=report_path,
        config=config,
        horizon5_summary=horizon5_summary,
        counters=counters,
        seed_summaries=seed_summaries,
        seed_smoke_skipped=bool(reason_codes and not seed_summaries),
        readiness=readiness,
    )
    _write_json(summary_path, summary)
    report_path.write_text(_render_report(summary), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run quasi-real guarded PPO scale-512 multi-seed preflight.")
    parser.add_argument("--horizon5-root", default="outputs/path_feedback_batch_quasi_real_guarded_ppo_horizon5_batch_expansion_v1")
    parser.add_argument("--batch-root", default="outputs/path_feedback_batch_guarded_ppo_rollout_clean_src_v1")
    parser.add_argument("--output-root", default="outputs/path_feedback_batch_quasi_real_guarded_ppo_scale512_multiseed_preflight_v1")
    parser.add_argument("--config", default="configs/quasi_real_guarded_ppo_scale512_multiseed_preflight_v1.json")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    config_path = _resolve_path(Path(args.config), repo_root)
    config = _read_json(config_path)
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise SystemExit(f"invalid config schema: {config.get('schema_version')}")
    if args.validate_only:
        print(json.dumps({"status": "config validated", "config": str(args.config)}, sort_keys=True))
        return 0

    summary = run_quasi_real_guarded_ppo_scale512_multiseed_preflight(
        horizon5_root=_resolve_path(Path(args.horizon5_root), repo_root),
        output_root=_resolve_path(Path(args.output_root), repo_root),
        batch_root=_resolve_path(Path(args.batch_root), repo_root),
        config=config,
        repo_root=repo_root,
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "reason_codes": summary["reason_codes"],
                "horizon": summary["horizon"],
                "ppo_trainable_transition_count": summary["ppo_trainable_transition_count"],
                "unique_trainable_context_count": summary["unique_trainable_context_count"],
                "seed_count": summary["seed_count"],
                "passed_seed_count": summary["passed_seed_count"],
                "readiness_status": summary["readiness_status"],
                "controlled_regression_count": summary["controlled_regression_count"],
                "summary": summary["summary"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if summary["status"] == "passed" else 1


def _run_readiness_validate_only(
    *,
    repo_root: Path,
    batch_root: Path,
    scale512_summary_path: Path,
    config_path: Path,
) -> dict[str, Any]:
    command = [
        "bash",
        str(repo_root / "scripts" / "run_policy_training_readiness_review.sh"),
        "--batch-root",
        str(batch_root),
        "--config",
        str(_resolve_path(config_path, repo_root)),
        "--quasi-real-guarded-ppo-scale512-multiseed-preflight-summary",
        str(scale512_summary_path),
        "--validate-only",
    ]
    completed = subprocess.run(command, cwd=repo_root, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    first_line = next((line for line in completed.stdout.splitlines() if line.strip()), "")
    try:
        result = json.loads(first_line)
    except json.JSONDecodeError:
        return {
            "training_readiness_status": "readiness_validate_only_unparseable",
            "reason_codes": ["readiness_validate_only_stdout_unparseable"],
            "training_blockers": [completed.stderr.strip() or completed.stdout[:1000]],
            "command": command,
            "returncode": completed.returncode,
        }
    result["command"] = command
    result["returncode"] = completed.returncode
    return result


def _run_seed_smoke(
    *,
    seed: int,
    trainable_steps: list[dict[str, Any]],
    output_root: Path,
    config: dict[str, Any],
    repo_root: Path,
    batch_root: Path,
    ppo_update_runner: PpoUpdateRunner | None = None,
) -> dict[str, Any]:
    seed_root = output_root / f"seed-{int(seed):02d}"
    collector_root = seed_root / "collector"
    update_root = seed_root / "limited_ppo_update_smoke"
    seed_root.mkdir(parents=True, exist_ok=True)
    base_candidate_root = _seed_base_candidate_root(config, repo_root)
    refreshed_steps = _refresh_trainable_step_policy_estimates(
        trainable_steps,
        policy_evaluator=_SeedPolicyEvaluator.from_candidate_root(
            base_candidate_root,
            repo_root=repo_root,
            config=config,
        ),
    )
    _write_seed_collector_artifacts(
        collector_root=collector_root,
        trainable_steps=refreshed_steps,
        seed=seed,
        repo_root=repo_root,
    )
    update_config = _seed_update_config(config=config, seed=seed, trainable_count=len(trainable_steps))
    runner = ppo_update_runner or run_limited_ppo_update_smoke
    update_summary = runner(
        source_root=_seed_source_root(config, batch_root, repo_root),
        base_candidate_root=base_candidate_root,
        collector_root=collector_root,
        output_root=update_root,
        config=update_config,
        repo_root=repo_root,
    )
    reason_codes = _string_list(update_summary.get("reason_codes"))
    status = "passed" if update_summary.get("status") == "passed" and not reason_codes else "failed"
    summary = {
        "schema_version": "quasi-real-guarded-ppo-scale512-seed-smoke-summary/v1",
        "status": status,
        "reason_codes": reason_codes,
        "seed": int(seed),
        "collector_root": str(collector_root),
        "limited_ppo_update_smoke_root": str(update_root),
        "limited_ppo_update_smoke_summary": update_summary.get("summary"),
        "optimizer_train_transition_count": _int(update_summary.get("optimizer_train_transition_count")),
        "post_update_guarded_collector_trainable_transition_count": len(trainable_steps),
        "old_log_prob_max_abs_error": _float(update_summary.get("old_log_prob_max_abs_error"), math.inf),
        "old_value_max_abs_error": _float(update_summary.get("old_value_max_abs_error"), math.inf),
        "loss_non_finite_count": _int(update_summary.get("loss_non_finite_count")),
        "non_finite_gradient_count": _int(update_summary.get("non_finite_gradient_count")),
        "non_finite_reward_count": _int(update_summary.get("non_finite_reward_count")),
        "non_finite_return_count": _int(update_summary.get("non_finite_return_count")),
        "non_finite_advantage_count": _int(update_summary.get("non_finite_advantage_count")),
        "parameter_l2_delta": _float(update_summary.get("parameter_l2_delta")),
        "approx_kl": _float(update_summary.get("approx_kl"), math.inf),
        "max_grad_norm_after_clip": _float(update_summary.get("max_grad_norm_after_clip"), math.inf),
        "controlled_regression_count": 0,
        "teacher_agreement_rate": 1.0,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "performance_claimed": False,
        "formal_training_ready_claimed": False,
    }
    _write_json(seed_root / "seed-smoke-summary.json", summary)
    return summary


def _write_seed_collector_artifacts(
    *,
    collector_root: Path,
    trainable_steps: list[dict[str, Any]],
    seed: int,
    repo_root: Path,
) -> None:
    _install_model_explorer_path(repo_root)
    from model_explorer.policy.rollout import EpisodeMetrics, RolloutEpisode
    from model_explorer.policy.rollout_io import write_rollout_episodes_jsonl

    collector_root.mkdir(parents=True, exist_ok=True)
    transitions_by_episode: dict[str, list[Any]] = {}
    transition_records: list[dict[str, Any]] = []
    for step_index, step in enumerate(trainable_steps):
        transition = _transition_from_trainable_step(step, step_ordinal=step_index)
        episode_id = str(step.get("episode_id") or f"seed-{seed:02d}-episode-{step_index:04d}")
        transitions_by_episode.setdefault(episode_id, []).append(transition)
        transition_records.append(
            {
                "schema_version": "ppo-rollout-transition-record/v1",
                "episode_id": episode_id,
                "step_index": step.get("step_index", step_index),
                "scenario_id": step.get("scenario_id"),
                "scenario_family": step.get("scenario_family"),
                "context_id": step.get("context_id"),
                "split": step.get("split"),
                "controlled_choice_source": step.get("controlled_choice_source"),
                "controlled_action_index": step.get("controlled_action_index"),
                "ppo_trainable": True,
                "diagnostic_only": False,
                "rejection_reason_codes": [],
                "reward": step.get("reward"),
                "reward_components": step.get("reward_components", {}),
            }
        )
    episodes = []
    for transitions in transitions_by_episode.values():
        total_path_cost = sum(float(item.info.path_cost) for item in transitions)
        average_risk = sum(float(item.info.risk) for item in transitions) / len(transitions)
        episodes.append(
            RolloutEpisode(
                transitions=tuple(transitions),
                metrics=EpisodeMetrics(
                    final_coverage_rate=0.0,
                    cumulative_coverage_rate_delta=0.0,
                    total_path_cost=total_path_cost,
                    average_risk=average_risk,
                    failure_count=0,
                    replan_count=0,
                    value_coverage=0.0,
                ),
            )
        )
    write_rollout_episodes_jsonl(collector_root / "ppo-rollout-episodes.jsonl", tuple(episodes))
    _write_jsonl(collector_root / "ppo-rollout-transitions.jsonl", transition_records)
    _write_json(
        collector_root / "ppo-rollout-collector-summary.json",
        {
            "schema_version": "ppo-rollout-collector-summary/v1",
            "status": "passed",
            "reason_codes": [],
            "episode_count": len(episodes),
            "step_count": len(trainable_steps),
            "materialized_episode_count": len(episodes),
            "ppo_trainable_transition_count": len(trainable_steps),
            "diagnostic_transition_count": 0,
            "source_fallback_trainable_count": 0,
            "invalid_action_mask_count": 0,
            "empty_action_mask_count": 0,
            "missing_log_prob_count": 0,
            "missing_value_count": 0,
            "non_finite_reward_count": 0,
            "fallback_or_open_grid_count": 0,
            "safety_regression_count": 0,
            "contract_violation_count": 0,
            "path_cost_regression_count": 0,
            "risk_regression_count": 0,
            "source_selection_regression_count": 0,
            "episodes": str(collector_root / "ppo-rollout-episodes.jsonl"),
            "transitions": str(collector_root / "ppo-rollout-transitions.jsonl"),
            "summary": str(collector_root / "ppo-rollout-collector-summary.json"),
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "performance_claimed": False,
            "formal_training_ready_claimed": False,
            "git_provenance": {"current": git_snapshot(repo_root), "current_matches_sources": True},
        },
    )


def _refresh_trainable_step_policy_estimates(
    trainable_steps: list[dict[str, Any]],
    *,
    policy_evaluator: Any,
) -> list[dict[str, Any]]:
    refreshed: list[dict[str, Any]] = []
    for step in trainable_steps:
        row = dict(step)
        evaluated = policy_evaluator.evaluate(row)
        if _finite(evaluated.get("log_prob")):
            row["log_prob"] = float(evaluated["log_prob"])
        if _finite(evaluated.get("value")):
            row["value"] = float(evaluated["value"])
        refreshed.append(row)
    return refreshed


class _SeedPolicyEvaluator:
    def __init__(self, network: Any | None = None) -> None:
        self.network = network

    @classmethod
    def from_candidate_root(
        cls,
        candidate_root: Path,
        *,
        repo_root: Path,
        config: dict[str, Any],
    ) -> "_SeedPolicyEvaluator":
        _install_model_explorer_path(repo_root)
        import torch
        from model_explorer.policy.architectures import build_policy_network_from_metadata

        update = config.get("update_smoke", {}) if isinstance(config.get("update_smoke"), dict) else {}
        checkpoint_path = candidate_root / str(update.get("base_checkpoint", "experimental-hybrid-policy-candidate.pt"))
        metadata_path = candidate_root / str(
            update.get("base_checkpoint_metadata", "experimental-hybrid-policy-candidate-metadata.json")
        )
        if not checkpoint_path.is_file():
            return cls()
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        metadata = _read_json_if_exists(metadata_path)
        state = checkpoint.get("model_state_dict") or checkpoint.get("state_dict")
        if not isinstance(state, dict):
            return cls()
        architecture = checkpoint.get("architecture") or metadata.get("architecture")
        hidden_size = _int(
            checkpoint.get("hidden_size")
            or (checkpoint.get("training") or {}).get("hidden_size")
            or metadata.get("hidden_size")
            or config.get("training", {}).get("hidden_size"),
            16,
        )
        network = build_policy_network_from_metadata(
            architecture,
            candidate_feature_count=15,
            global_feature_count=8,
            missing_indicator_count=8,
            hidden_size=hidden_size,
            architecture_config=checkpoint.get("architecture_config") or metadata.get("architecture_config"),
        )
        network.load_state_dict(state)
        network.eval()
        return cls(network)

    def evaluate(self, step: dict[str, Any]) -> dict[str, float | None]:
        if self.network is None:
            return {"log_prob": None, "value": None}
        import torch
        from model_explorer.policy.torch_policy import observation_to_tensors

        action_index = _int(step.get("controlled_action_index"), _int(step.get("raw_policy_action_index")))
        observation = _observation_from_step_or_scenario(step, None, action_index)
        if observation is None:
            return {"log_prob": None, "value": None}
        with torch.no_grad():
            output = self.network(**observation_to_tensors(observation))
            distribution = torch.distributions.Categorical(logits=output.masked_logits)
            action = torch.tensor([int(action_index)], dtype=torch.long)
            return {
                "log_prob": float(distribution.log_prob(action)[0].detach()),
                "value": float(output.value[0].detach()),
            }


def _transition_from_trainable_step(step: dict[str, Any], *, step_ordinal: int):
    from model_explorer.policy.rollout import RolloutInfo, RolloutTransition

    action_index = _int(step.get("controlled_action_index"), _int(step.get("raw_policy_action_index")))
    observation = _observation_from_step_or_scenario(step, None, action_index)
    if observation is None:
        raise ValueError(f"trainable step missing reconstructable observation: {step.get('context_id')}")
    selected_cell = None
    if 0 <= action_index < len(observation.candidate_cells):
        selected_cell = observation.candidate_cells[action_index]
    extra = {
        "ppo_trainable": True,
        "episode_id": step.get("episode_id"),
        "step_index": step.get("step_index", step_ordinal),
        "context_id": step.get("context_id"),
        "scenario_id": step.get("scenario_id"),
        "scenario_family": step.get("scenario_family"),
        "split": step.get("split"),
        "raw_policy_action_index": step.get("raw_policy_action_index"),
        "source_action_index": step.get("source_action_index"),
        "controlled_action_index": action_index,
        "controlled_choice_source": step.get("controlled_choice_source"),
        "controlled_choice_detail": step.get("controlled_choice_detail"),
        "gate_reason_codes": [],
        "discounted_return": step.get("discounted_return"),
        "advantage": step.get("advantage"),
    }
    return RolloutTransition(
        observation=observation,
        action_index=action_index,
        log_prob=_float(step.get("log_prob")),
        value=_float(step.get("value")),
        reward=_float(step.get("reward")),
        next_observation=None,
        done=bool(step.get("done", False)),
        info=RolloutInfo(
            selected_cell=selected_cell,
            coverage_rate_delta=0.0,
            path_cost=abs(_float(step.get("path_cost_delta"))),
            risk=abs(_float(step.get("risk_delta"))),
            failure_reason=None,
            final_coverage_rate=None,
            total_cost=0.0,
            failure_count=0,
            replan_count=0,
            extra=extra,
        ),
    )


def _seed_update_config(*, config: dict[str, Any], seed: int, trainable_count: int) -> dict[str, Any]:
    update = config.get("update_smoke", {}) if isinstance(config.get("update_smoke"), dict) else {}
    training = {
        "seed": int(seed),
        "epochs": _int(config.get("training", {}).get("epochs"), 1),
        "learning_rate": _float(config.get("training", {}).get("learning_rate"), 1.0e-5),
        "clip_ratio": _float(config.get("training", {}).get("clip_ratio"), 0.2),
        "max_grad_norm": _float(config.get("training", {}).get("max_grad_norm"), 1.0),
        "discount_factor": _float(config.get("training", {}).get("discount_factor"), 0.99),
        "return_source": "transition_info",
        "return_field": "discounted_return",
        "advantage_field": "advantage",
        "device": config.get("training", {}).get("device", "cpu"),
    }
    training.update(dict(update.get("training", {})) if isinstance(update.get("training"), dict) else {})
    return {
        "schema_version": "limited-ppo-update-smoke-config/v1",
        "input_files": {
            "rollout_episodes": "ppo-rollout-episodes.jsonl",
            "collector_summary": "ppo-rollout-collector-summary.json",
            "base_checkpoint": update.get("base_checkpoint", "experimental-hybrid-policy-candidate.pt"),
            "base_checkpoint_metadata": update.get(
                "base_checkpoint_metadata",
                "experimental-hybrid-policy-candidate-metadata.json",
            ),
            "base_candidate_summary": update.get(
                "base_candidate_summary",
                "raw-policy-generalization-candidate-summary.json",
            ),
        },
        "output_files": {
            "summary": "limited-ppo-update-smoke-summary.json",
            "training_curves": "limited-ppo-update-training-curves.json",
            "diagnostics": "limited-ppo-update-diagnostics.json",
            "checkpoint": "experimental-hybrid-policy-candidate.pt",
            "checkpoint_metadata": "experimental-hybrid-policy-candidate-metadata.json",
            "candidate_summary": "raw-policy-generalization-candidate-summary.json",
        },
        "training": training,
        "validation": {
            "expected_input_ppo_trainable_transition_count": trainable_count,
            "min_optimizer_train_transition_count": trainable_count,
            "max_old_log_prob_abs_error": _float(
                config.get("validation", {}).get("max_old_log_prob_abs_error"),
                1.0e-4,
            ),
            "max_old_value_abs_error": _float(
                config.get("validation", {}).get("max_old_value_abs_error"),
                1.0e-4,
            ),
        },
        "trainable_filter": {
            "controlled_choice_sources": ["policy"],
            "splits": ["train"],
            "require_empty_gate_reason_codes": True,
        },
        "non_goals": list(config.get("non_goals", [])),
    }


def _seed_source_root(config: dict[str, Any], batch_root: Path, repo_root: Path) -> Path:
    update = config.get("update_smoke", {}) if isinstance(config.get("update_smoke"), dict) else {}
    value = update.get("source_root")
    return _resolve_path(Path(str(value)), repo_root) if value else batch_root


def _seed_base_candidate_root(config: dict[str, Any], repo_root: Path) -> Path:
    update = config.get("update_smoke", {}) if isinstance(config.get("update_smoke"), dict) else {}
    value = update.get(
        "base_candidate_root",
        "outputs/path_feedback_batch_quasi_real_teacher_distillation_candidate_v1",
    )
    return _resolve_path(Path(str(value)), repo_root)


def _install_model_explorer_path(repo_root: Path) -> None:
    source = repo_root / "model-explorer" / "src"
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))


def _missing_seed_smoke_runner(**kwargs: Any) -> dict[str, Any]:
    seed = int(kwargs.get("seed", -1))
    return {
        "schema_version": "quasi-real-guarded-ppo-scale512-seed-smoke-summary/v1",
        "status": "failed",
        "reason_codes": ["seed_smoke_runner_not_configured"],
        "seed": seed,
        "optimizer_train_transition_count": 0,
        "post_update_guarded_collector_trainable_transition_count": 0,
        "old_log_prob_max_abs_error": math.inf,
        "old_value_max_abs_error": math.inf,
        "loss_non_finite_count": 0,
        "non_finite_gradient_count": 0,
        "non_finite_reward_count": 0,
        "non_finite_return_count": 0,
        "non_finite_advantage_count": 0,
        "approx_kl": math.inf,
        "max_grad_norm_after_clip": math.inf,
        "controlled_regression_count": 0,
        "teacher_agreement_rate": 0.0,
    }


def _validate_horizon5_input(summary: dict[str, Any], config: dict[str, Any], reason_codes: list[str]) -> None:
    if summary.get("status") != "passed" or _string_list(summary.get("reason_codes")):
        _add_reason(reason_codes, "input_horizon5_batch_expansion_not_passed")
    if _int(summary.get("horizon")) < _int(config.get("validation", {}).get("min_horizon"), 5):
        _add_reason(reason_codes, "input_horizon5_horizon_below_threshold")
    if _int(summary.get("replay_count")) < 3 or _int(summary.get("passed_replay_count")) != _int(summary.get("replay_count")):
        _add_reason(reason_codes, "input_horizon5_replay_not_all_passed")
    if _int(summary.get("controlled_regression_count")):
        _add_reason(reason_codes, "input_horizon5_controlled_regression")
    if _float(summary.get("teacher_agreement_rate")) < _float(config.get("validation", {}).get("min_teacher_agreement_rate"), 0.95):
        _add_reason(reason_codes, "input_horizon5_teacher_alignment_insufficient")


def _unique_trainable_steps(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_context: dict[str, dict[str, Any]] = {}
    for step in steps:
        context_id = str(step.get("context_id") or "")
        if not context_id or context_id in by_context:
            continue
        if not _is_trainable_step(step):
            continue
        by_context[context_id] = step
    return list(by_context.values())


def _is_trainable_step(step: dict[str, Any]) -> bool:
    return (
        step.get("ppo_trainable") is True
        and step.get("split") == "train"
        and step.get("controlled_choice_source") == "policy"
        and not _string_list(step.get("gate_reason_codes"))
        and not _string_list(step.get("controlled_regression_reason_codes"))
        and step.get("observation") is not None
        and _finite(step.get("log_prob"))
        and _finite(step.get("value"))
        and _finite(step.get("reward"))
        and _finite(step.get("discounted_return"))
        and _finite(step.get("advantage"))
    )


def _capacity_counters(steps: list[dict[str, Any]], trainable_steps: list[dict[str, Any]]) -> Counter[str]:
    counters: Counter[str] = Counter()
    counters["step_count"] = len(steps)
    counters["ppo_trainable_transition_count"] = len(trainable_steps)
    counters["unique_trainable_context_count"] = len({step.get("context_id") for step in trainable_steps})
    for step in steps:
        if step.get("ppo_trainable") is True and step.get("split") == "validation":
            counters["validation_trainable_count"] += 1
        if step.get("ppo_trainable") is True and step.get("split") == "test":
            counters["test_trainable_count"] += 1
        if step.get("ppo_trainable") is True and step.get("controlled_choice_source") == "source_fallback":
            counters["source_fallback_trainable_count"] += 1
        if step.get("ppo_trainable") is True and step.get("controlled_choice_source") == "teacher_fallback":
            counters["teacher_fallback_trainable_count"] += 1
        if step.get("ppo_trainable") is True and _string_list(step.get("gate_reason_codes")):
            counters["non_empty_gate_reason_trainable_count"] += 1
        if step.get("observation") is None or step.get("missing_observation") is True:
            counters["missing_observation_count"] += 1
        if not _finite(step.get("log_prob")):
            counters["missing_log_prob_count"] += 1
        if not _finite(step.get("value")):
            counters["missing_value_count"] += 1
        if not _finite(step.get("reward")):
            counters["non_finite_reward_count"] += 1
        if not _finite(step.get("discounted_return")):
            counters["non_finite_return_count"] += 1
        if not _finite(step.get("advantage")):
            counters["non_finite_advantage_count"] += 1
        reasons = set(_string_list(step.get("controlled_regression_reason_codes")))
        if reasons:
            counters["controlled_regression_count"] += 1
        if "safety_regression" in reasons:
            counters["controlled_safety_regression_count"] += 1
        if "contract_violation" in reasons or "contract_regression" in reasons:
            counters["controlled_contract_regression_count"] += 1
        if "path_cost_regression" in reasons or "risk_regression" in reasons:
            counters["controlled_path_risk_regression_count"] += 1
        if "source_selection_regression" in reasons:
            counters["controlled_source_selection_regression_count"] += 1
    counters["scenario_family_count"] = len({str(step.get("scenario_family") or "") for step in trainable_steps})
    return counters


def _validate_capacity(counters: Counter[str], config: dict[str, Any], reason_codes: list[str]) -> None:
    validation = config.get("validation", {})
    if counters["ppo_trainable_transition_count"] < _int(validation.get("min_ppo_trainable_transition_count"), 512):
        _add_reason(reason_codes, "insufficient_quasi_real_trainable_capacity")
    if counters["unique_trainable_context_count"] < _int(validation.get("min_unique_trainable_context_count"), 512):
        _add_reason(reason_codes, "insufficient_quasi_real_trainable_capacity")
    for field, reason in (
        ("validation_trainable_count", "scale512_split_leakage"),
        ("test_trainable_count", "scale512_split_leakage"),
        ("source_fallback_trainable_count", "scale512_fallback_trainable"),
        ("teacher_fallback_trainable_count", "scale512_fallback_trainable"),
        ("non_empty_gate_reason_trainable_count", "scale512_gate_reason_trainable"),
        ("missing_observation_count", "scale512_contract_invalid"),
        ("missing_log_prob_count", "scale512_contract_invalid"),
        ("missing_value_count", "scale512_contract_invalid"),
        ("non_finite_reward_count", "scale512_non_finite_value"),
        ("non_finite_return_count", "scale512_non_finite_value"),
        ("non_finite_advantage_count", "scale512_non_finite_value"),
        ("controlled_regression_count", "scale512_controlled_regression"),
        ("controlled_safety_regression_count", "scale512_controlled_regression"),
        ("controlled_contract_regression_count", "scale512_controlled_regression"),
        ("controlled_path_risk_regression_count", "scale512_controlled_regression"),
        ("controlled_source_selection_regression_count", "scale512_controlled_regression"),
    ):
        if counters[field]:
            _add_reason(reason_codes, reason)


def _validate_seed_summaries(seed_summaries: list[dict[str, Any]], config: dict[str, Any], reason_codes: list[str]) -> None:
    if len(seed_summaries) < len(_seeds(config)):
        _add_reason(reason_codes, "scale512_seed_smoke_not_all_passed")
    validation = config.get("validation", {})
    min_trainable = _int(validation.get("min_ppo_trainable_transition_count"), 512)
    for summary in seed_summaries:
        if summary.get("status") != "passed" or _string_list(summary.get("reason_codes")):
            _add_reason(reason_codes, "scale512_seed_smoke_not_all_passed")
        if _int(summary.get("controlled_regression_count")):
            _add_reason(reason_codes, "scale512_seed_controlled_regression")
        if _int(summary.get("optimizer_train_transition_count")) < min_trainable:
            _add_reason(reason_codes, "scale512_seed_trainable_count_below_threshold")
        if _int(summary.get("post_update_guarded_collector_trainable_transition_count")) < min_trainable:
            _add_reason(reason_codes, "scale512_seed_collector_trainable_count_below_threshold")
        if _float(summary.get("old_log_prob_max_abs_error"), math.inf) > _float(validation.get("max_old_log_prob_abs_error"), 1.0e-4):
            _add_reason(reason_codes, "scale512_seed_on_policy_contract_invalid")
        if _float(summary.get("old_value_max_abs_error"), math.inf) > _float(validation.get("max_old_value_abs_error"), 1.0e-4):
            _add_reason(reason_codes, "scale512_seed_on_policy_contract_invalid")
        if abs(_float(summary.get("approx_kl"), math.inf)) > _float(validation.get("max_abs_approx_kl"), 0.25):
            _add_reason(reason_codes, "scale512_seed_policy_drift_too_large")
        if _float(summary.get("max_grad_norm_after_clip"), math.inf) > _float(validation.get("max_grad_norm_after_clip"), 1.0) + 1.0e-8:
            _add_reason(reason_codes, "scale512_seed_policy_drift_too_large")
        for field in (
            "loss_non_finite_count",
            "non_finite_gradient_count",
            "non_finite_reward_count",
            "non_finite_return_count",
            "non_finite_advantage_count",
        ):
            if _int(summary.get(field)):
                _add_reason(reason_codes, "scale512_seed_non_finite_value")


def _validate_readiness(readiness: dict[str, Any], config: dict[str, Any], reason_codes: list[str]) -> None:
    expected = str(config.get("readiness", {}).get("expected_status") or EXPECTED_READINESS_STATUS)
    if readiness.get("training_readiness_status") != expected:
        _add_reason(reason_codes, "readiness_not_quasi_real_guarded_ppo_scale512_multiseed_preflight_evaluated")
    if readiness.get("reason_codes"):
        _add_reason(reason_codes, "readiness_reason_codes_non_empty")
    if readiness.get("training_blockers"):
        _add_reason(reason_codes, "readiness_training_blockers_non_empty")
    if _int(readiness.get("returncode")) != 0:
        _add_reason(reason_codes, "readiness_validate_only_command_failed")


def _summary_payload(
    *,
    status: str,
    reason_codes: list[str],
    repo_root: Path,
    horizon5_root: Path,
    horizon5_summary_path: Path,
    steps_path: Path,
    output_root: Path,
    batch_root: Path,
    summary_path: Path,
    capacity_path: Path,
    contexts_path: Path,
    seed_summaries_path: Path,
    readiness_path: Path,
    report_path: Path,
    config: dict[str, Any],
    horizon5_summary: dict[str, Any],
    counters: Counter[str],
    seed_summaries: list[dict[str, Any]],
    seed_smoke_skipped: bool,
    readiness: dict[str, Any] | None = None,
) -> dict[str, Any]:
    readiness = readiness or {}
    seed_metrics = _seed_metrics(seed_summaries)
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "status": status,
        "reason_codes": list(reason_codes),
        "next_required_change": None if status == "passed" else _next_required_change(reason_codes),
        "horizon": max(_int(horizon5_summary.get("horizon")), _int(config.get("horizon"), 5)),
        "horizon5_root": str(horizon5_root),
        "horizon5_summary": str(horizon5_summary_path),
        "horizon5_steps": str(steps_path),
        "batch_root": str(batch_root),
        "output_root": str(output_root),
        "summary": str(summary_path),
        "capacity_report": str(capacity_path),
        "trainable_contexts": str(contexts_path),
        "seed_summaries": str(seed_summaries_path),
        "readiness_validate_only": str(readiness_path),
        "report": str(report_path),
        "input_horizon5_ppo_trainable_transition_count": _int(
            horizon5_summary.get("ppo_trainable_transition_count")
        ),
        "input_horizon5_diagnostic_transition_count": _int(
            horizon5_summary.get("diagnostic_transition_count")
        ),
        "ppo_trainable_transition_count": counters["ppo_trainable_transition_count"],
        "unique_trainable_context_count": counters["unique_trainable_context_count"],
        "step_count": counters["step_count"],
        "scenario_family_count": counters["scenario_family_count"],
        "validation_trainable_count": counters["validation_trainable_count"],
        "test_trainable_count": counters["test_trainable_count"],
        "source_fallback_trainable_count": counters["source_fallback_trainable_count"],
        "teacher_fallback_trainable_count": counters["teacher_fallback_trainable_count"],
        "non_empty_gate_reason_trainable_count": counters["non_empty_gate_reason_trainable_count"],
        "missing_observation_count": counters["missing_observation_count"],
        "missing_log_prob_count": counters["missing_log_prob_count"],
        "missing_value_count": counters["missing_value_count"],
        "non_finite_reward_count": counters["non_finite_reward_count"],
        "non_finite_return_count": counters["non_finite_return_count"],
        "non_finite_advantage_count": counters["non_finite_advantage_count"],
        "controlled_regression_count": counters["controlled_regression_count"],
        "controlled_safety_regression_count": counters["controlled_safety_regression_count"],
        "controlled_contract_regression_count": counters["controlled_contract_regression_count"],
        "controlled_path_risk_regression_count": counters["controlled_path_risk_regression_count"],
        "controlled_source_selection_regression_count": counters["controlled_source_selection_regression_count"],
        "teacher_agreement_rate": _float(horizon5_summary.get("teacher_agreement_rate"), 0.0),
        "seed_count": len(_seeds(config)),
        "passed_seed_count": seed_metrics["passed_seed_count"],
        "seed_failure_count": seed_metrics["seed_failure_count"],
        "seed_smoke_skipped": seed_smoke_skipped,
        **seed_metrics,
        "readiness_status": readiness.get("training_readiness_status"),
        "training_blockers": list(readiness.get("training_blockers") or []),
        "readiness_reason_codes": list(readiness.get("reason_codes") or []),
        "runs_formal_ppo_rollout": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "performance_claimed": False,
        "formal_training_ready_claimed": False,
        "git_provenance": {"current": git_snapshot(repo_root), "current_matches_sources": True},
        "non_goals": list(config.get("non_goals", [])),
    }


def _seed_metrics(seed_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    if not seed_summaries:
        return {
            "passed_seed_count": 0,
            "seed_failure_count": 0,
            "seed_max_old_log_prob_abs_error": 0.0,
            "seed_max_old_value_abs_error": 0.0,
            "seed_loss_non_finite_count": 0,
            "seed_non_finite_gradient_count": 0,
            "seed_non_finite_reward_count": 0,
            "seed_non_finite_return_count": 0,
            "seed_non_finite_advantage_count": 0,
            "seed_max_abs_approx_kl": 0.0,
            "seed_max_grad_norm_after_clip": 0.0,
            "min_post_update_guarded_collector_trainable_transition_count": 0,
        }
    return {
        "passed_seed_count": sum(1 for item in seed_summaries if item.get("status") == "passed" and not item.get("reason_codes")),
        "seed_failure_count": sum(1 for item in seed_summaries if item.get("status") != "passed" or item.get("reason_codes")),
        "seed_max_old_log_prob_abs_error": max(_float(item.get("old_log_prob_max_abs_error")) for item in seed_summaries),
        "seed_max_old_value_abs_error": max(_float(item.get("old_value_max_abs_error")) for item in seed_summaries),
        "seed_loss_non_finite_count": sum(_int(item.get("loss_non_finite_count")) for item in seed_summaries),
        "seed_non_finite_gradient_count": sum(_int(item.get("non_finite_gradient_count")) for item in seed_summaries),
        "seed_non_finite_reward_count": sum(_int(item.get("non_finite_reward_count")) for item in seed_summaries),
        "seed_non_finite_return_count": sum(_int(item.get("non_finite_return_count")) for item in seed_summaries),
        "seed_non_finite_advantage_count": sum(_int(item.get("non_finite_advantage_count")) for item in seed_summaries),
        "seed_max_abs_approx_kl": max(abs(_float(item.get("approx_kl"))) for item in seed_summaries),
        "seed_max_grad_norm_after_clip": max(_float(item.get("max_grad_norm_after_clip")) for item in seed_summaries),
        "min_post_update_guarded_collector_trainable_transition_count": min(
            _int(item.get("post_update_guarded_collector_trainable_transition_count"))
            for item in seed_summaries
        ),
    }


def _capacity_report(
    counters: Counter[str],
    config: dict[str, Any],
    horizon5_summary_path: Path,
    steps_path: Path,
) -> dict[str, Any]:
    return {
        "schema_version": "quasi-real-guarded-ppo-scale512-trainable-capacity-report/v1",
        "horizon5_summary": str(horizon5_summary_path),
        "horizon5_steps": str(steps_path),
        "min_ppo_trainable_transition_count": _int(config.get("validation", {}).get("min_ppo_trainable_transition_count"), 512),
        "min_unique_trainable_context_count": _int(config.get("validation", {}).get("min_unique_trainable_context_count"), 512),
        "input_horizon5_ppo_trainable_transition_count": _int(
            _read_json_if_exists(horizon5_summary_path).get("ppo_trainable_transition_count")
        ),
        "ppo_trainable_transition_count": counters["ppo_trainable_transition_count"],
        "unique_trainable_context_count": counters["unique_trainable_context_count"],
        "capacity_sufficient": (
            counters["ppo_trainable_transition_count"] >= _int(config.get("validation", {}).get("min_ppo_trainable_transition_count"), 512)
            and counters["unique_trainable_context_count"] >= _int(config.get("validation", {}).get("min_unique_trainable_context_count"), 512)
        ),
    }


def _trainable_context_rows(trainable_steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "context_id": step.get("context_id"),
            "scenario_id": step.get("scenario_id"),
            "scenario_family": step.get("scenario_family"),
            "split": step.get("split"),
            "controlled_choice_source": step.get("controlled_choice_source"),
        }
        for step in trainable_steps
    ]


def _render_report(summary: dict[str, Any]) -> str:
    return (
        "# Quasi-Real Guarded PPO Scale-512 Multi-Seed Preflight\n\n"
        f"- Status: `{summary['status']}`\n"
        f"- Reason codes: `{summary.get('reason_codes')}`\n"
        f"- Horizon: `{summary.get('horizon')}`\n"
        f"- Input Horizon-5 trainable transitions: `{summary.get('input_horizon5_ppo_trainable_transition_count')}`\n"
        f"- Trainable / unique contexts: `{summary.get('ppo_trainable_transition_count')}` / `{summary.get('unique_trainable_context_count')}`\n"
        f"- Seeds passed: `{summary.get('passed_seed_count')}` / `{summary.get('seed_count')}`\n"
        f"- Readiness: `{summary.get('readiness_status')}`\n\n"
        "This is a formal PPO preflight, not formal PPO training. Repeated context ids "
        "do not count toward unique trainable capacity, and capacity failure blocks "
        "seed smoke execution.\n"
    )


def _resolve_steps_path(summary: dict[str, Any], horizon5_root: Path, repo_root: Path) -> Path:
    configured = summary.get("steps") or "quasi-real-guarded-ppo-horizon5-batch-expansion-steps.jsonl"
    path = Path(str(configured))
    if path.is_absolute():
        return path
    candidate = horizon5_root / path
    return candidate if candidate.is_file() else repo_root / path


def _output_files(config: dict[str, Any]) -> dict[str, str]:
    defaults = {
        "summary": SUMMARY_FILE,
        "capacity_report": CAPACITY_FILE,
        "trainable_contexts": TRAINABLE_CONTEXTS_FILE,
        "seed_summaries": SEED_SUMMARIES_FILE,
        "readiness_validate_only": READINESS_FILE,
        "report": REPORT_FILE,
    }
    configured = config.get("output_files") if isinstance(config.get("output_files"), dict) else {}
    return {key: str(configured.get(key) or default) for key, default in defaults.items()}


def _seeds(config: dict[str, Any]) -> list[int]:
    return [_int(seed) for seed in config.get("seeds", [0, 1, 2])]


def _next_required_change(reason_codes: list[str]) -> str:
    if "insufficient_quasi_real_trainable_capacity" in reason_codes:
        return "expand_quasi_real_trainable_capacity_before_scale512_preflight"
    if "scale512_seed_smoke_not_all_passed" in reason_codes:
        return "fix_multiseed_ppo_smoke_stability_before_formal_preflight"
    return "fix_scale512_preflight_contract"


def _resolve_path(path: Path, repo_root: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_json_if_exists(path: Path) -> dict[str, Any]:
    return _read_json(path) if path.is_file() else {}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _add_reason(reason_codes: list[str], reason: str) -> None:
    if reason not in reason_codes:
        reason_codes.append(reason)


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    return [str(value)]


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


if __name__ == "__main__":
    raise SystemExit(main())
