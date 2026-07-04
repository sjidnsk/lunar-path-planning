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
