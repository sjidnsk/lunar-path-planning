"""Stage 1 deterministic Smoke v1 workflow and machine artifacts."""

from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path
from typing import Callable

from pydantic import BaseModel, ConfigDict

from lunar_exploration_ppo.configs.stage1 import load_stage1_config
from lunar_exploration_ppo.env.env import (
    EnvAction,
    LunarExplorationEnv,
    StepResult,
    select_conservative_frontier_candidate_index,
    select_conservative_rule_action,
)
from lunar_exploration_ppo.policy.observation import PolicyObservation
from lunar_exploration_ppo.utils.artifact_io import ArtifactStore
from lunar_exploration_ppo.workflows.stage1_artifacts import (
    STAGE1_MACHINE_ARTIFACTS,
    Stage1WorkflowError,
    finite_tree,
    prepare_unique_run_root,
    validate_run_id,
)
from lunar_exploration_ppo.workflows.stage1_metrics import reset_metric, step_metric
from lunar_exploration_ppo.workflows.stage1_review import (
    Stage1ReviewResult,
    record_stage1_independent_review,
)
from lunar_exploration_ppo.workflows.stage1_source import compute_stage1_source_identity


__all__ = [
    "EpisodeResult",
    "Policy",
    "STAGE1_MACHINE_ARTIFACTS",
    "Stage1ReviewResult",
    "Stage1WorkflowError",
    "Stage1WorkflowResult",
    "record_stage1_independent_review",
    "run_episode",
    "run_stage1_smoke_workflow",
    "select_conservative_frontier_candidate_index",
    "select_conservative_rule_action",
    "select_rule_action",
]


class EpisodeResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    episode_index: int
    transition_count: int
    trainable_transition_count: int
    total_reward: float
    final_coverage_rate: float
    done_reason: str
    reset_empty: bool


class Stage1WorkflowResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str
    stage_root: Path
    summary: dict[str, object]
    manifest: dict[str, object]


Policy = Callable[[PolicyObservation], EnvAction]
MetricWriter = Callable[[dict[str, object]], None]


def select_rule_action(observation: PolicyObservation) -> EnvAction:
    return select_conservative_rule_action(observation)


def run_episode(
    env: LunarExplorationEnv,
    *,
    episode_index: int,
    policy: Policy,
    metric_writer: MetricWriter | None = None,
) -> EpisodeResult:
    observation = env.reset()
    if env.last_reset_diagnostics is None:
        raise RuntimeError("reset diagnostics were not created")
    if metric_writer is not None:
        metric_writer(reset_metric(env, episode_index))
    if env.is_done:
        return EpisodeResult(
            episode_index=episode_index,
            transition_count=0,
            trainable_transition_count=0,
            total_reward=0.0,
            final_coverage_rate=env.last_reset_diagnostics.coverage_rate,
            done_reason=env.terminal_reason,
            reset_empty=True,
        )

    transitions = 0
    trainable_transitions = 0
    total_reward = 0.0
    final: StepResult | None = None
    while not env.is_done:
        if not env.needs_policy:
            raise RuntimeError("nonterminal environment has no policy action")
        action = policy(observation)
        final = env.step(action)
        transitions += 1
        trainable_transitions += int(final.trainable)
        total_reward += final.reward
        observation = final.observation
        if metric_writer is not None:
            metric_writer(step_metric(env, episode_index, final))
    if final is None:
        raise RuntimeError("terminal episode did not produce a closing transition")
    return EpisodeResult(
        episode_index=episode_index,
        transition_count=transitions,
        trainable_transition_count=trainable_transitions,
        total_reward=total_reward,
        final_coverage_rate=final.coverage_rate,
        done_reason=final.reason,
        reset_empty=False,
    )


def run_stage1_smoke_workflow(
    *,
    config_path: str | Path,
    run_id: str,
    base_output_root: str | Path | None = None,
) -> Stage1WorkflowResult:
    validate_run_id(run_id)
    config = load_stage1_config(config_path)
    repo_root = Path(__file__).resolve().parents[3]
    execution_source_identity = compute_stage1_source_identity(
        repo_root,
        base_commit=config.foundation_base_commit,
        workflow_import_path=Path(__file__),
    )
    base_root = Path(base_output_root) if base_output_root is not None else Path(config.output_root)
    run_root = (base_root / run_id).expanduser().resolve()
    prepare_unique_run_root(run_root)
    stage_root = run_root / "s1"
    stage_root.mkdir(exist_ok=False)
    store = ArtifactStore(stage_root)
    config_payload = config.model_dump(mode="json")
    config_payload["run_id"] = run_id
    config_payload["execution_source_identity"] = execution_source_identity
    config_bytes = ArtifactStore.canonical_json_bytes(config_payload)
    config_hash = hashlib.sha256(config_bytes).hexdigest()
    store.write_bytes("config.json", config_bytes)

    env = LunarExplorationEnv(config)
    episodes: list[EpisodeResult] = []

    def append_metric(metric: dict[str, object]) -> None:
        if not finite_tree(metric):
            raise Stage1WorkflowError("metrics contain NaN or infinity")
        store.append_jsonl("metrics.jsonl", metric)

    for episode_index in range(config.deterministic_episodes):
        episodes.append(
            run_episode(
                env,
                episode_index=episode_index,
                policy=select_rule_action,
                metric_writer=append_metric,
            )
        )
    final_source_identity = compute_stage1_source_identity(
        repo_root,
        base_commit=config.foundation_base_commit,
        workflow_import_path=Path(__file__),
    )
    if final_source_identity != execution_source_identity:
        raise Stage1WorkflowError("Stage 1 execution source identity drifted during machine run")
    termination_counts = dict(sorted(Counter(episode.done_reason for episode in episodes).items()))
    summary: dict[str, object] = {
        "schema_version": "ppo_highres_frontier_stage1_summary/v1",
        "goal_id": config.goal_id,
        "stage_id": config.stage_id,
        "run_id": run_id,
        "state": "machine_passed",
        "scale_profile": config.scale_profile,
        "episodes_requested": config.deterministic_episodes,
        "episodes_completed": len(episodes),
        "trainable_transition_count": sum(item.trainable_transition_count for item in episodes),
        "termination_counts": termination_counts,
        "mean_final_coverage_rate": sum(item.final_coverage_rate for item in episodes) / len(episodes),
        "mean_total_reward": sum(item.total_reward for item in episodes) / len(episodes),
        "scenario_id": env.scenario_id,
        "scenario_hash": env.scenario_hash,
        "proxy_morphology": env.proxy_morphology_metadata,
        "coverage_mask": env.coverage_metadata,
        "config_hash": config_hash,
        "config_hash_schema": "sha256_file_bytes/v1",
        "checkpoint_state": "stage1_no_checkpoint/v1",
        "execution_source_identity": execution_source_identity,
    }
    if not finite_tree(summary):
        raise Stage1WorkflowError("summary contains NaN or infinity")
    store.write_json("summary.json", summary)
    store.write_json(
        "routing.json",
        {
            "schema_version": "ppo_highres_frontier_stage1_routing/v1",
            "run_id": run_id,
            "route": "awaiting_independent_review",
            "human_approval_required": True,
            "next_stage_entered": False,
        },
    )
    store.write_bytes(
        "report.md",
        (
            "# Stage 1 Smoke v1 machine report\n\n"
            f"Run ID: `{run_id}`\n\n"
            f"- Episodes: {len(episodes)}\n"
            f"- Trainable transitions: {summary['trainable_transition_count']}\n"
            f"- Terminations: `{termination_counts}`\n"
            f"- Mean final coverage: {summary['mean_final_coverage_rate']:.8f}\n"
            f"- Prospective Git tree: `{execution_source_identity['prospective_git_tree']}`\n"
            f"- Reviewed source-set SHA-256: `{execution_source_identity['reviewed_source_set_sha256']}`\n"
            f"- Reviewed source paths: {execution_source_identity['reviewed_path_count']}\n"
            f"- Package import: `{execution_source_identity['package_import_path']}`\n"
            f"- Workflow import: `{execution_source_identity['workflow_import_path']}`\n"
            "- State: `machine_passed`\n"
            "- Route: `awaiting_independent_review`\n"
            "- No checkpoint, approval, gate, executor, or canary artifact was produced.\n"
        ).encode("utf-8"),
    )
    store.append_jsonl(
        "phase-state.jsonl",
        {"state": "machine_passed", "route": "awaiting_independent_review", "run_id": run_id},
    )
    manifest = store.build_manifest(STAGE1_MACHINE_ARTIFACTS)
    store.write_json("manifest.json", manifest)
    return Stage1WorkflowResult(run_id=run_id, stage_root=stage_root, summary=summary, manifest=manifest)
