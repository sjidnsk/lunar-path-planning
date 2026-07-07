from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from xunce_artifact_io import path_is_file, read_json, read_jsonl, write_json, write_jsonl


@dataclass(frozen=True)
class ArtifactName:
    key: str
    canonical: str
    legacy: tuple[str, ...] = ()

    def candidates(self) -> tuple[str, ...]:
        return (self.canonical, *self.legacy)


STAGE21_1_SUMMARY = ArtifactName(
    "stage21_1_summary",
    "summary.json",
    ("xunce-stage21-1-on-policy-ppo-rollout-collector-summary.json",),
)
STAGE21_1_TRAINABLE = ArtifactName("stage21_1_trainable", "trainable.jsonl", ("xunce-stage21-1-ppo-trainable-batch.jsonl",))
STAGE21_1_TRANSITIONS = ArtifactName("stage21_1_transitions", "transitions.jsonl", ("xunce-stage21-1-ppo-rollout-transitions.jsonl",))
STAGE21_1_EPISODES = ArtifactName("stage21_1_episodes", "episodes.jsonl", ("xunce-stage21-1-ppo-rollout-episodes.jsonl",))
STAGE21_1_REJECTIONS = ArtifactName("stage21_1_rejections", "rejections.jsonl", ("xunce-stage21-1-rejection-report.jsonl",))

STAGE21_2_SUMMARY = ArtifactName("stage21_2_summary", "summary.json", ("xunce-stage21-2-coverage-first-reward-summary.json",))
STAGE21_2_REWARDS = ArtifactName("stage21_2_rewards", "rewards.jsonl", ("xunce-stage21-2-reward-contract-evaluation.jsonl",))

STAGE21_3_SUMMARY = ArtifactName("stage21_3_summary", "summary.json", ("xunce-stage21-3-ppo-batch-validation-summary.json",))
STAGE21_3_BATCH = ArtifactName("stage21_3_batch", "batch.jsonl", ("xunce-stage21-3-ppo-trainable-batch.jsonl",))
STAGE21_3_SPLITS = ArtifactName("stage21_3_splits", "splits.json", ("xunce-stage21-3-ppo-batch-splits.json",))

STAGE21_4_SUMMARY = ArtifactName(
    "stage21_4_summary",
    "summary.json",
    ("xunce-stage21-4-tiny-ppo-update-smoke-summary.json",),
)
STAGE21_4_LOSS = ArtifactName("stage21_4_loss", "loss.jsonl", ("xunce-stage21-4-ppo-loss-audit.jsonl",))
STAGE21_4_GRADIENT = ArtifactName("stage21_4_gradient", "gradient.json", ("xunce-stage21-4-gradient-audit.json",))
STAGE21_4_CHECKPOINT = ArtifactName("stage21_4_checkpoint", "checkpoint.json", ("xunce-stage21-4-checkpoint-audit.json",))
STAGE21_4_ROUTING = ArtifactName("stage21_4_routing", "routing.json", ("xunce-stage21-4-next-stage-routing.json",))
STAGE21_4_MANIFEST = ArtifactName("stage21_4_manifest", "manifest.json", ("xunce-stage21-4-manifest.json",))

STAGE21_5_SUMMARY = ArtifactName(
    "stage21_5_summary",
    "summary.json",
    ("xunce-stage21-5-post-update-evaluation-summary.json",),
)
STAGE21_5_RESULTS = ArtifactName("stage21_5_results", "results.json", ("xunce-stage21-5-policy-evaluation-results.json",))
STAGE21_5_DELTA = ArtifactName("stage21_5_delta", "delta.json", ("xunce-stage21-5-trajectory-delta.json",))
STAGE21_5_SCENARIO_DELTA = ArtifactName(
    "stage21_5_scenario_delta",
    "scenario_delta.jsonl",
    ("xunce-stage21-5-scenario-trajectory-delta.jsonl",),
)
STAGE21_5_SELECTED_POSE_EVIDENCE_AUDIT = ArtifactName(
    "stage21_5_selected_pose_evidence_audit",
    "selected_pose_audit.json",
    ("xunce-stage21-5-selected-pose-evidence-audit.json",),
)
STAGE21_5_ROUTING = ArtifactName("stage21_5_routing", "routing.json", ("xunce-stage21-5-next-stage-routing.json",))
STAGE21_5_MANIFEST = ArtifactName("stage21_5_manifest", "manifest.json", ("xunce-stage21-5-manifest.json",))

STAGE26_2_SUMMARY = ArtifactName("stage26_2_summary", "summary.json", ("xunce-stage26-2-summary.json",))
STAGE26_2_STAGE21_4_CONFIG = ArtifactName(
    "stage26_2_stage21_4_config",
    "stage21_4_config.json",
    ("xunce-stage26-2-stage21-4-config.json",),
)
STAGE26_2_STAGE21_4_SUMMARY = ArtifactName(
    "stage26_2_stage21_4_summary",
    "stage21_4_summary.json",
    ("xunce-stage26-2-stage21-4-summary.json",),
)
STAGE26_2_BATCH_AUDIT = ArtifactName(
    "stage26_2_batch_audit",
    "batch_audit.json",
    ("xunce-stage26-2-synthetic-ppo-batch-update-audit.json",),
)
STAGE26_2_LOSS_GRADIENT_AUDIT = ArtifactName(
    "stage26_2_loss_gradient_audit",
    "loss_gradient_audit.json",
    ("xunce-stage26-2-loss-gradient-audit.json",),
)
STAGE26_2_CHECKPOINT_BOUNDARY_AUDIT = ArtifactName(
    "stage26_2_checkpoint_boundary_audit",
    "checkpoint_boundary_audit.json",
    ("xunce-stage26-2-checkpoint-boundary-audit.json",),
)
STAGE26_2_ROUTING = ArtifactName("stage26_2_routing", "routing.json", ("xunce-stage26-2-next-stage-routing.json",))
STAGE26_2_MANIFEST = ArtifactName("stage26_2_manifest", "manifest.json", ("xunce-stage26-2-manifest.json",))

STAGE26_3_SUMMARY = ArtifactName("stage26_3_summary", "summary.json", ("xunce-stage26-3-summary.json",))
STAGE26_3_STAGE21_5_CONFIG = ArtifactName(
    "stage26_3_stage21_5_config",
    "stage21_5_config.json",
    ("xunce-stage26-3-stage21-5-config.json",),
)
STAGE26_3_HIGH_FIDELITY_CONFIG = ArtifactName(
    "stage26_3_high_fidelity_config",
    "high_fidelity_config.json",
    ("xunce-stage26-3-high-fidelity-config.json",),
)
STAGE26_3_STAGE21_5_SUMMARY = ArtifactName(
    "stage26_3_stage21_5_summary",
    "stage21_5_summary.json",
    ("xunce-stage26-3-stage21-5-summary.json",),
)
STAGE26_3_ACTION_AUDIT = ArtifactName(
    "stage26_3_action_audit",
    "action_audit.json",
    ("xunce-stage26-3-synthetic-action-change-audit.json",),
)
STAGE26_3_DELTA_AUDIT = ArtifactName(
    "stage26_3_delta_audit",
    "delta_audit.json",
    ("xunce-stage26-3-trajectory-delta-audit.json",),
)
STAGE26_3_ROUTING = ArtifactName("stage26_3_routing", "routing.json", ("xunce-stage26-3-next-stage-routing.json",))
STAGE26_3_MANIFEST = ArtifactName("stage26_3_manifest", "manifest.json", ("xunce-stage26-3-manifest.json",))


def resolve_artifact(root: str | Path, artifact: ArtifactName) -> tuple[Path | None, str | None]:
    base = Path(root)
    for index, name in enumerate(artifact.candidates()):
        path = base / name
        if path_is_file(path):
            return path, "canonical" if index == 0 else "legacy"
    return None, None


def artifact_path(root: str | Path, artifact: ArtifactName, *, canonical: bool = True) -> Path:
    return Path(root) / (artifact.canonical if canonical else artifact.legacy[0])


def missing_reason(artifact: ArtifactName) -> str:
    return f"missing_{artifact.legacy[0] if artifact.legacy else artifact.canonical}"


def read_json_artifact(root: str | Path, artifact: ArtifactName) -> tuple[dict[str, Any], str]:
    path, source = resolve_artifact(root, artifact)
    if path is None or source is None:
        raise FileNotFoundError(missing_reason(artifact))
    payload = read_json(path)
    payload.setdefault("artifact_source", source)
    return payload, source


def read_jsonl_artifact(root: str | Path, artifact: ArtifactName) -> tuple[list[dict[str, Any]], str]:
    path, source = resolve_artifact(root, artifact)
    if path is None or source is None:
        raise FileNotFoundError(missing_reason(artifact))
    return read_jsonl(path), source


def write_json_artifact(root: str | Path, artifact: ArtifactName, payload: dict[str, Any], *, dual_write: bool = True) -> None:
    canonical_payload = dict(payload)
    canonical_payload.setdefault("artifact_source", "canonical")
    write_json(artifact_path(root, artifact), canonical_payload)
    if dual_write:
        for legacy in artifact.legacy:
            legacy_payload = dict(payload)
            legacy_payload.setdefault("artifact_source", "legacy")
            write_json(Path(root) / legacy, legacy_payload)


def write_jsonl_artifact(root: str | Path, artifact: ArtifactName, rows: list[dict[str, Any]], *, dual_write: bool = True) -> None:
    write_jsonl(artifact_path(root, artifact), rows)
    if dual_write:
        for legacy in artifact.legacy:
            write_jsonl(Path(root) / legacy, rows)
