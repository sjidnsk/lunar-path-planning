"""Stage 5 fair-baseline machine workflow and read-only Stage 4 consumer."""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import platform
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Final

import numpy as np
import pydantic
import torch

from lunar_exploration_ppo.configs.stage1 import Stage1Config
from lunar_exploration_ppo.configs.stage5 import Stage5Config, load_stage5_config
from lunar_exploration_ppo.env.env import (
    EnvAction,
    LunarExplorationEnv,
    resolve_terminal,
)
from lunar_exploration_ppo.env.frontier import FrontierActionSet
from lunar_exploration_ppo.env.scenario import ScenarioBundle, ScenarioSource, TruthMap
from lunar_exploration_ppo.eval.baselines import (
    ALL_METHODS,
    BASELINE_METHODS,
    NoCandidateAction,
    select_baseline_action,
    select_ppo_action,
)
from lunar_exploration_ppo.eval.evaluator import (
    METHOD_ACTION_RULES,
    EvaluationScenario,
    EvaluationSummary,
    Evaluator,
)
from lunar_exploration_ppo.eval.metrics import (
    EPISODE_FIELDS,
    EpisodeResult,
    episode_record,
    summarize_episodes,
)
from lunar_exploration_ppo.eval.reports import (
    baseline_eval_report,
    comparison_table_csv,
    coverage_curves_csv,
)
from lunar_exploration_ppo.policy.cross_attention import (
    CrossAttentionFrontierPolicy,
    PolicyForwardOutput,
)
from lunar_exploration_ppo.policy.observation import PolicyObservation
from lunar_exploration_ppo.ppo.checkpoint import CheckpointManager
from lunar_exploration_ppo.ppo.trainer import policy_state_sha256
from lunar_exploration_ppo.utils.artifact_io import ArtifactStore
from lunar_exploration_ppo.workflows import stage3 as stage3_workflow
from lunar_exploration_ppo.workflows.stage4_machine import (
    Stage4MachineError,
    deterministic_action_record,
    load_action_fixture_npz,
)
from lunar_exploration_ppo.workflows.stage1_artifacts import (
    FrozenFileSnapshot,
    Stage1WorkflowError,
    strict_json_object_from_bytes,
    validate_run_id,
)


GOAL_ID: Final = "ppo-highres-frontier-map-exploration"
STAGE4_COMMIT: Final = "f94865c3b2984bbffdeefc764a921cae2d9a8089"
STAGE4_TREE: Final = "b53b9cca692990acbe1c2aa860c5eeb57821a37b"
STAGE4_GATE_SHA256: Final = (
    "a12bbe6bc993add5cd8329bb9e172602e1e2e853a60c374343881e8263c0603e"
)
STAGE4_CONFIG_SHA256: Final = (
    "e5fb85748313a9fcd33552638b8883e74d2252dba680266838b6d82aa00b3441"
)
STAGE4_SOURCE_SET_SHA256: Final = (
    "0bd2c0f555747bd73a6901aee4f23d1c2dcf25a50f5dba0bb66c9179c572a63c"
)
STAGE4_ENVIRONMENT_SHA256: Final = (
    "29ccedbad6f93dd40d206ca923d0a3538b8b322a991ae44140521f719cb2ac69"
)
STAGE4_MANIFEST_SHA256: Final = (
    "464913106d1fe7b83838af9a42c96ab3cf3aa7c05014e07d097b6a3d6688a54f"
)
STAGE4_CHECKPOINT_SHA256: Final = (
    "d1d80e6478262a68d01208ddc1e8721768b56109e8dd009e862dd668cb39dc4c"
)
STAGE4_CHECKPOINT_MANIFEST_SHA256: Final = (
    "2dbff72d99a82bba5bbf827f3c558851ed83fc45262e579da05ee512cacdcd3a"
)
STAGE4_POLICY_STATE_SHA256: Final = (
    "51fecca55af92838f302951ae1272063cd945ef481579af5f6600469d9a7fe86"
)
STAGE4_ACTION_FIXTURE_SHA256: Final = (
    "66dc513c9809174f193bcaa232ab8d21f125a08a167d400f427e9ca5da5876e6"
)
STAGE4_CHECKPOINT_LOAD_AUDIT_SHA256: Final = (
    "2a29e69c0b28069b7d63b8e74d677d7ecc0ee19e3fd8a58fec44f4f530052810"
)
STAGE4_LATEST_DETERMINISTIC_ACTION: Final = {
    "selected_frontier_index": 11,
    "selected_theta_fp32_hex": "60cb1b3e",
    "log_prob_frontier_fp32_hex": "702767c0",
    "log_prob_theta_fp32_hex": "fa62dfbf",
    "log_prob_total_fp32_hex": "766cabc0",
    "value_fp32_hex": "9063453e",
}
STAGE4_RUN_ID: Final = "s4-task5-r2-final-20260714T151959Z"
STAGE5_STAGE_ID: Final = "ppo_highres_frontier_stage5_fair_baseline_evaluator/v1"

STAGE5_CHANGED_PATHS: Final = (
    "configs/ppo_highres_frontier_stage5_v1.json",
    "docs/ppo-highres-frontier-stage5.md",
    "scripts/run_ppo_stage5_baselines.py",
    "src/lunar_exploration_ppo/configs/stage5.py",
    "src/lunar_exploration_ppo/eval/__init__.py",
    "src/lunar_exploration_ppo/eval/baselines.py",
    "src/lunar_exploration_ppo/eval/evaluator.py",
    "src/lunar_exploration_ppo/eval/metrics.py",
    "src/lunar_exploration_ppo/eval/reports.py",
    "src/lunar_exploration_ppo/workflows/stage5.py",
    "tests/ppo_highres_frontier/test_stage5_baselines.py",
    "tests/ppo_highres_frontier/test_stage5_workflow.py",
)
STAGE5_MACHINE_ARTIFACTS: Final = (
    "config.json",
    "summary.json",
    "routing.json",
    "report.md",
    "metrics.jsonl",
    "phase-state.jsonl",
)
STAGE5_EVALUATOR_ARTIFACTS: Final = (
    "acceptance_probe_audit.json",
    "baseline_eval_report.md",
    "baseline_comparison_table.csv",
    "baseline_metrics.json",
    "baseline_coverage_curves.csv",
    "baseline_episode_traces.jsonl",
    "baseline_fairness_audit.json",
    "bootstrap_ci_audit.json",
    "stage4_authority_audit.json",
    "execution_identity_audit.json",
)
STAGE5_MANIFEST_BOUND_ARTIFACTS: Final = (
    *STAGE5_MACHINE_ARTIFACTS,
    *STAGE5_EVALUATOR_ARTIFACTS,
)
STAGE5_ROOT_ARTIFACTS: Final = (
    *STAGE5_MANIFEST_BOUND_ARTIFACTS,
    "manifest.json",
)
_STAGE5_JSONL_ARTIFACTS: Final = frozenset(
    {
        "metrics.jsonl",
        "phase-state.jsonl",
        "baseline_episode_traces.jsonl",
    }
)
_STAGE4_ROOT_MEMBERS: Final = frozenset(
    {
        "approval.json",
        "checkpoints",
        "config.json",
        "evidence",
        "gate.json",
        "manifest.json",
        "metrics.jsonl",
        "phase-state.jsonl",
        "report.md",
        "review.json",
        "routing.json",
        "summary.json",
        "training_progress.jsonl",
    }
)
_STAGE4_CHECKPOINT_FILES: Final = frozenset(
    {
        "latest.json",
        *(
            f"update-{step:08d}/{name}"
            for step in range(1, 4)
            for name in ("checkpoint.pt", "complete.json", "manifest.json")
        ),
    }
)


class Stage5WorkflowError(RuntimeError):
    """Stage 5 authority, execution, or artifact verification failed."""


@dataclass(frozen=True, slots=True)
class FrozenStage4AuthorityHandle:
    identity: dict[str, object]
    snapshots: tuple[tuple[str, FrozenFileSnapshot], ...]
    stage4_root: Path
    checkpoint_root: Path
    repo_root: Path

    def snapshot(self, name: str) -> FrozenFileSnapshot:
        for key, value in self.snapshots:
            if key == name:
                return value
        raise Stage5WorkflowError(f"Stage 4 authority snapshot missing: {name}")

    def require_current(self, label: str = "Stage 4 authority") -> None:
        try:
            if (
                not self.stage4_root.is_dir()
                or {path.name for path in self.stage4_root.iterdir()}
                != _STAGE4_ROOT_MEMBERS
            ):
                raise Stage5WorkflowError(f"{label} root artifact set changed")
            if _relative_files(self.checkpoint_root) != _STAGE4_CHECKPOINT_FILES:
                raise Stage5WorkflowError(f"{label} checkpoint artifact set changed")
            for name, snapshot in self.snapshots:
                snapshot.require_current(f"{label}/{name}")
            if _git_text(self.repo_root, ["rev-parse", "HEAD"]) != STAGE4_COMMIT:
                raise Stage5WorkflowError(f"{label} HEAD changed")
            if _git_text(self.repo_root, ["rev-parse", "HEAD^{tree}"]) != STAGE4_TREE:
                raise Stage5WorkflowError(f"{label} commit tree changed")
        except Stage5WorkflowError:
            raise
        except (OSError, Stage1WorkflowError) as exc:
            raise Stage5WorkflowError(f"{label} changed") from exc


@dataclass(frozen=True, slots=True)
class LoadedStage4Policy:
    policy: CrossAttentionFrontierPolicy
    audit: dict[str, object]


@dataclass(frozen=True, slots=True)
class _StaticScenarioSource:
    bundle: ScenarioBundle

    def load(self, key: str) -> ScenarioBundle:
        if key != self.bundle.scenario_id.split("/", maxsplit=1)[0]:
            raise KeyError(key)
        return self.bundle


@dataclass(frozen=True, slots=True)
class Stage5RuntimeBinding:
    payload: dict[str, object]
    config_bytes: bytes
    config_sha256: str


@dataclass(frozen=True, slots=True)
class Stage5WorkflowResult:
    run_id: str
    stage_root: Path
    summary: dict[str, object]
    routing: dict[str, object]
    manifest: dict[str, object]


@dataclass(frozen=True, slots=True)
class _Stage5MachinePayload:
    artifacts: dict[str, bytes]
    manifest: dict[str, object]
    summary: dict[str, object]
    routing: dict[str, object]


@dataclass(frozen=True, slots=True)
class Stage5ManifestGraphHandle:
    stage_root: Path
    manifest_snapshot: FrozenFileSnapshot
    artifacts: tuple[tuple[str, FrozenFileSnapshot], ...]

    def require_current(self, label: str = "Stage 5 manifest graph") -> None:
        try:
            if (
                not self.stage_root.is_dir()
                or {path.name for path in self.stage_root.iterdir()}
                != set(STAGE5_ROOT_ARTIFACTS)
            ):
                raise Stage5WorkflowError(f"{label} artifact set changed")
            self.manifest_snapshot.require_current(f"{label}/manifest")
            for relative, snapshot in self.artifacts:
                snapshot.require_current(f"{label}/{relative}")
        except Stage5WorkflowError:
            raise
        except (OSError, Stage1WorkflowError) as exc:
            raise Stage5WorkflowError(f"{label} changed") from exc


@dataclass(frozen=True, slots=True)
class Stage5MachineVerificationHandle:
    manifest_graph: Stage5ManifestGraphHandle
    stage4_authority: FrozenStage4AuthorityHandle
    config_snapshot: FrozenFileSnapshot
    stage1_snapshot: FrozenFileSnapshot

    def require_current(self, label: str = "Stage 5 machine verification") -> None:
        self.manifest_graph.require_current(label)
        self.stage4_authority.require_current(f"{label}/Stage 4 authority")
        try:
            self.config_snapshot.require_current(f"{label}/repository config")
            self.stage1_snapshot.require_current(f"{label}/Stage 1 config")
        except Stage1WorkflowError as exc:
            raise Stage5WorkflowError(f"{label} config changed") from exc


def verify_frozen_stage4_authority(
    *,
    gate_path: str | Path,
    checkpoint_root: str | Path,
    repo_root: str | Path,
) -> FrozenStage4AuthorityHandle:
    stage4_gate = Path(gate_path).expanduser().resolve()
    checkpoints = Path(checkpoint_root).expanduser().resolve()
    repo = Path(repo_root).expanduser().resolve()
    stage4_root = stage4_gate.parent
    if stage4_gate.name != "gate.json" or checkpoints != stage4_root / "checkpoints":
        raise Stage5WorkflowError("Stage 4 authority paths are not canonical")
    if {path.name for path in stage4_root.iterdir()} != _STAGE4_ROOT_MEMBERS:
        raise Stage5WorkflowError("Stage 4 authority root artifact set drift")
    if _relative_files(checkpoints) != _STAGE4_CHECKPOINT_FILES:
        raise Stage5WorkflowError("Stage 4 checkpoint artifact set drift")
    paths = {
        "gate": stage4_gate,
        "config": stage4_root / "config.json",
        "summary": stage4_root / "summary.json",
        "routing": stage4_root / "routing.json",
        "manifest": stage4_root / "manifest.json",
        "checkpoint_latest": checkpoints / "latest.json",
        "checkpoint_manifest": checkpoints / "update-00000003/manifest.json",
        "checkpoint_complete": checkpoints / "update-00000003/complete.json",
        "checkpoint": checkpoints / "update-00000003/checkpoint.pt",
        "action_fixture": stage4_root / "evidence/checkpoint_action_fixture.npz",
        "checkpoint_load_audit": stage4_root / "evidence/checkpoint_load_audit.json",
    }
    try:
        snapshots = tuple(
            (
                name,
                FrozenFileSnapshot.capture(
                    path,
                    f"Stage 4 authority/{name}",
                    authority_root=(
                        checkpoints
                        if path.is_relative_to(checkpoints)
                        else stage4_root
                    ),
                ),
            )
            for name, path in paths.items()
        )
    except (OSError, Stage1WorkflowError) as exc:
        raise Stage5WorkflowError("Stage 4 authority snapshot failed") from exc
    by_name = dict(snapshots)
    gate_snapshot = by_name["gate"]
    gate = _strict_canonical_json(gate_snapshot, "Stage 4 gate")
    bindings = gate.get("bindings")
    history = gate.get("history")
    if (
        gate_snapshot.sha256 != STAGE4_GATE_SHA256
        or gate.get("schema_version") != "ppo_highres_frontier_stage4_verified_gate/v1"
        or gate.get("state") != "next_stage"
        or gate.get("run_id") != STAGE4_RUN_ID
        or gate.get("commit_sha256") != STAGE4_COMMIT
        or gate.get("authorized_next_stage") != STAGE5_STAGE_ID
        or not isinstance(bindings, dict)
        or not isinstance(history, list)
        or [item.get("state") for item in history if isinstance(item, dict)]
        != [
            "machine_passed",
            "awaiting_independent_review",
            "awaiting_human_approval",
            "approved",
            "next_stage",
        ]
    ):
        raise Stage5WorkflowError("Stage 4 verified gate semantics drift")
    expected_bindings = {
        "commit_sha256": STAGE4_COMMIT,
        "commit_tree": STAGE4_TREE,
        "config_sha256": STAGE4_CONFIG_SHA256,
        "source_set_sha256": STAGE4_SOURCE_SET_SHA256,
        "environment_sha256": STAGE4_ENVIRONMENT_SHA256,
        "manifest_sha256": STAGE4_MANIFEST_SHA256,
        "latest_checkpoint_sha256": STAGE4_CHECKPOINT_SHA256,
        "latest_checkpoint_policy_state_sha256": STAGE4_POLICY_STATE_SHA256,
        "authorized_next_stage": STAGE5_STAGE_ID,
        "run_id": STAGE4_RUN_ID,
    }
    if any(bindings.get(name) != value for name, value in expected_bindings.items()):
        raise Stage5WorkflowError("Stage 4 gate binding drift")
    bound_file_hashes = {
        "config": STAGE4_CONFIG_SHA256,
        "summary": str(bindings.get("summary_sha256")),
        "routing": str(bindings.get("routing_sha256")),
        "manifest": STAGE4_MANIFEST_SHA256,
        "checkpoint": STAGE4_CHECKPOINT_SHA256,
        "checkpoint_manifest": STAGE4_CHECKPOINT_MANIFEST_SHA256,
        "action_fixture": STAGE4_ACTION_FIXTURE_SHA256,
        "checkpoint_load_audit": STAGE4_CHECKPOINT_LOAD_AUDIT_SHA256,
    }
    if any(by_name[name].sha256 != digest for name, digest in bound_file_hashes.items()):
        raise Stage5WorkflowError("Stage 4 authority file hash drift")
    latest = _strict_canonical_json(by_name["checkpoint_latest"], "Stage 4 latest checkpoint")
    checkpoint_manifest = _strict_canonical_json(
        by_name["checkpoint_manifest"],
        "Stage 4 checkpoint manifest",
    )
    complete = _strict_canonical_json(
        by_name["checkpoint_complete"],
        "Stage 4 checkpoint complete marker",
    )
    if (
        latest.get("update_step") != 3
        or latest.get("directory") != "update-00000003"
        or latest.get("checkpoint_sha256") != STAGE4_CHECKPOINT_SHA256
        or latest.get("manifest_sha256") != STAGE4_CHECKPOINT_MANIFEST_SHA256
        or checkpoint_manifest.get("policy_state_sha256")
        != STAGE4_POLICY_STATE_SHA256
        or checkpoint_manifest.get("checkpoint", {}).get("sha256")
        != STAGE4_CHECKPOINT_SHA256
        or complete.get("checkpoint_sha256") != STAGE4_CHECKPOINT_SHA256
        or complete.get("manifest_sha256") != STAGE4_CHECKPOINT_MANIFEST_SHA256
    ):
        raise Stage5WorkflowError("Stage 4 latest complete checkpoint drift")
    if _git_text(repo, ["rev-parse", "HEAD"]) != STAGE4_COMMIT or _git_text(
        repo, ["rev-parse", "HEAD^{tree}"]
    ) != STAGE4_TREE:
        raise Stage5WorkflowError("Stage 4 Git authority drift")
    identity = {
        "schema_version": "stage5_stage4_authority_audit/v1",
        "verified": True,
        "authorized_stage": STAGE5_STAGE_ID,
        "run_id": STAGE4_RUN_ID,
        "commit_sha256": STAGE4_COMMIT,
        "commit_tree": STAGE4_TREE,
        "gate_path": _slash_path(stage4_gate),
        "gate_sha256": STAGE4_GATE_SHA256,
        "config_sha256": STAGE4_CONFIG_SHA256,
        "source_set_sha256": STAGE4_SOURCE_SET_SHA256,
        "environment_sha256": STAGE4_ENVIRONMENT_SHA256,
        "manifest_sha256": STAGE4_MANIFEST_SHA256,
        "checkpoint_root": _slash_path(checkpoints),
        "checkpoint_sha256": STAGE4_CHECKPOINT_SHA256,
        "checkpoint_manifest_sha256": STAGE4_CHECKPOINT_MANIFEST_SHA256,
        "policy_state_sha256": STAGE4_POLICY_STATE_SHA256,
        "update_step": 3,
    }
    handle = FrozenStage4AuthorityHandle(
        identity=identity,
        snapshots=snapshots,
        stage4_root=stage4_root,
        checkpoint_root=checkpoints,
        repo_root=repo,
    )
    handle.require_current()
    return handle


def load_frozen_stage4_policy(
    authority: FrozenStage4AuthorityHandle,
    *,
    device: str | torch.device,
) -> LoadedStage4Policy:
    target = torch.device(device)
    if target.type != "cuda" or not torch.cuda.is_available():
        raise Stage5WorkflowError("Stage 5 checkpoint replay requires CUDA")
    authority.require_current("pre-load Stage 4 authority")
    runtime_config = _strict_canonical_json(
        authority.snapshot("config"),
        "Stage 4 runtime config",
    )
    source = runtime_config.get("execution_source_identity")
    git_identity = runtime_config.get("execution_git_identity")
    stage3_authority = runtime_config.get("execution_stage3_authority")
    if not all(isinstance(value, dict) for value in (source, git_identity, stage3_authority)):
        raise Stage5WorkflowError("Stage 4 checkpoint lineage inputs are missing")
    assert isinstance(source, dict)
    assert isinstance(git_identity, dict)
    assert isinstance(stage3_authority, dict)
    lineage = {
        "schema_version": "ppo_highres_frontier_stage4_checkpoint_lineage/v1",
        "stage3_commit": stage3_authority["stage3_commit"],
        "stage3_gate_sha256": stage3_authority["gate_sha256"],
        "stage3_approval_sha256": stage3_authority["approval_sha256"],
        "stage3_review_sha256": stage3_authority["review_sha256"],
        "stage4_run_id": runtime_config["run_id"],
        "stage4_source_set_sha256": source["source_set_sha256"],
        "stage4_prospective_git_tree": git_identity["prospective_git_tree"],
        "stage4_config_sha256": authority.identity["config_sha256"],
    }
    policy = CrossAttentionFrontierPolicy().to(device=target, dtype=torch.float32)
    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=3.0e-4,
        eps=1.0e-5,
        weight_decay=1.0e-4,
    )
    loaded = CheckpointManager(authority.checkpoint_root).load_last_complete(
        policy=policy,
        optimizer=optimizer,
        expected_config_sha256=STAGE4_CONFIG_SHA256,
        expected_lineage=lineage,
    )
    if (
        loaded.update_step != 3
        or loaded.checkpoint_sha256 != STAGE4_CHECKPOINT_SHA256
        or loaded.manifest_sha256 != STAGE4_CHECKPOINT_MANIFEST_SHA256
        or loaded.policy_state_sha256 != STAGE4_POLICY_STATE_SHA256
    ):
        raise Stage5WorkflowError("loaded Stage 4 checkpoint binding drift")
    try:
        replay_observation, fixture_action = load_action_fixture_npz(
            authority.snapshot("action_fixture").payload
        )
        actual_action = deterministic_action_record(
            policy,
            replay_observation,
            device=target,
        )
        checkpoint_load_audit = _strict_canonical_json(
            authority.snapshot("checkpoint_load_audit"),
            "Stage 4 checkpoint load audit",
        )
    except (Stage4MachineError, OSError, ValueError) as exc:
        raise Stage5WorkflowError("Stage 4 deterministic action replay failed") from exc
    updates = checkpoint_load_audit.get("updates")
    latest_replay = updates[-1] if isinstance(updates, list) and updates else None
    if (
        checkpoint_load_audit.get("schema_version")
        != "stage4_checkpoint_load_audit/v1"
        or not isinstance(latest_replay, dict)
        or latest_replay.get("update_step") != 3
        or latest_replay.get("checkpoint_sha256") != STAGE4_CHECKPOINT_SHA256
        or latest_replay.get("policy_state_sha256") != STAGE4_POLICY_STATE_SHA256
        or latest_replay.get("deterministic_action_bit_exact") is not True
        or latest_replay.get("deterministic_action")
        != STAGE4_LATEST_DETERMINISTIC_ACTION
        or fixture_action != STAGE4_LATEST_DETERMINISTIC_ACTION
        or actual_action != fixture_action
        or policy_state_sha256(policy) != STAGE4_POLICY_STATE_SHA256
    ):
        raise Stage5WorkflowError("Stage 4 deterministic action replay drift")
    policy.eval()
    audit = {
        "schema_version": "stage5_frozen_checkpoint_audit/v1",
        "checkpoint_root": _slash_path(authority.checkpoint_root),
        "checkpoint_directory": loaded.directory.name,
        "checkpoint_sha256": loaded.checkpoint_sha256,
        "checkpoint_manifest_sha256": loaded.manifest_sha256,
        "policy_state_sha256": loaded.policy_state_sha256,
        "update_step": loaded.update_step,
        "device": "cuda",
        "read_only": True,
        "deterministic_action_replay": {
            "schema_version": "stage5_checkpoint_action_replay/v1",
            "fixture_sha256": STAGE4_ACTION_FIXTURE_SHA256,
            "checkpoint_load_audit_sha256": STAGE4_CHECKPOINT_LOAD_AUDIT_SHA256,
            "expected_action": dict(STAGE4_LATEST_DETERMINISTIC_ACTION),
            "actual_action": actual_action,
            "bit_exact": True,
        },
    }
    authority.require_current("post-load Stage 4 authority")
    return LoadedStage4Policy(policy=policy, audit=audit)


def stage5_source_identity(repo_root: str | Path) -> dict[str, object]:
    repo = Path(repo_root).expanduser().resolve()
    paths: set[Path] = set((repo / "src/lunar_exploration_ppo").rglob("*.py"))
    paths.update(
        {
            repo / "pyproject.toml",
            repo / "configs/ppo_highres_frontier_smoke_v1.json",
            repo / "configs/ppo_highres_frontier_stage4_v1.json",
            repo / "configs/ppo_highres_frontier_stage5_v1.json",
            repo / "scripts/run_ppo_stage5_baselines.py",
            repo / "docs/ppo-highres-frontier-stage5.md",
            repo
            / "docs/superpowers/specs/2026-07-09-ppo-highres-frontier-map-exploration-design.md",
            repo
            / "docs/superpowers/plans/2026-07-10-ppo-highres-frontier-map-exploration.md",
            repo / "tests/ppo_highres_frontier/test_stage5_baselines.py",
            repo / "tests/ppo_highres_frontier/test_stage5_workflow.py",
        }
    )
    digest = hashlib.sha256()
    entries: list[dict[str, object]] = []
    for path in sorted(paths, key=lambda item: item.relative_to(repo).as_posix()):
        if not path.is_file():
            raise Stage5WorkflowError(f"Stage 5 source binding path missing: {path}")
        relative = path.relative_to(repo).as_posix()
        payload = path.read_bytes()
        entries.append(
            {
                "path": relative,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size_bytes": len(payload),
            }
        )
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(payload)
        digest.update(b"\0")
    return {
        "schema_version": "stage5_reviewed_source_set/v1",
        "source_set_sha256": digest.hexdigest(),
        "paths": entries,
    }


def stage5_environment_identity() -> dict[str, object]:
    if not torch.cuda.is_available():
        raise Stage5WorkflowError("Stage 5 requires CUDA; CPU fallback is forbidden")
    device = torch.device("cuda")
    properties = torch.cuda.get_device_properties(device)
    return {
        "schema_version": "ppo_highres_frontier_stage5_environment/v1",
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform_system": platform.system(),
        "platform_machine": platform.machine(),
        "numpy_version": np.__version__,
        "pydantic_version": pydantic.__version__,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "cuda_available": True,
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_device_index": torch.cuda.current_device(),
        "cuda_device_name": torch.cuda.get_device_name(device),
        "cuda_compute_capability": list(torch.cuda.get_device_capability(device)),
        "cuda_total_memory_bytes": int(properties.total_memory),
        "compute_dtype": "float32",
        "amp_enabled": False,
    }


def stage5_runtime_leakage_audit(config: Stage1Config) -> dict[str, object]:
    """Prove that paired hidden-truth/denominator mutation cannot enter decisions."""

    if config.scale_profile != "Smoke v1" or config.scenario_key != "smoke-v1":
        raise Stage5WorkflowError("Stage 5 leakage audit requires the frozen Smoke config")
    baseline_bundle = ScenarioSource().load(config.scenario_key)
    baseline_env = LunarExplorationEnv(
        config,
        scenario_source=_StaticScenarioSource(baseline_bundle),
    )
    baseline_observation = baseline_env.reset()
    observed = baseline_env.observed_state.observed_mask
    geometry = baseline_bundle.truth.geometry
    yy, xx = np.mgrid[: geometry.height, : geometry.width]
    start = baseline_bundle.start_pose.cell
    protected_radius_cells = (
        config.sensor_range_m / geometry.resolution_m
        + math.ceil(config.min_clearance_m / geometry.resolution_m)
        + 2.0
    )
    mutation_mask = (~observed) & (
        (xx - start.x) ** 2 + (yy - start.y) ** 2 > protected_radius_cells**2
    )
    if not bool(mutation_mask.any()):
        raise Stage5WorkflowError("Stage 5 leakage audit has no unobserved mutation cells")

    mutated_height = baseline_bundle.truth.height.copy()
    mutated_obstacle = baseline_bundle.truth.hard_obstacle.copy()
    mutated_slope = baseline_bundle.truth.slope_deg.copy()
    mutated_traversability = baseline_bundle.truth.traversability.copy()
    mutated_height[mutation_mask] += 1000.0
    mutated_obstacle[mutation_mask] = True
    mutated_slope[mutation_mask] = 89.0
    mutated_traversability[mutation_mask] = 0.0
    mutated_truth = TruthMap(
        geometry=geometry,
        height=mutated_height,
        hard_obstacle=mutated_obstacle,
        slope_deg=mutated_slope,
        traversability=mutated_traversability,
        provenance={
            **dict(baseline_bundle.truth.provenance),
            "stage5_mutation_source": "unobserved_truth_only/v1",
        },
    )
    mutated_bundle = replace(
        baseline_bundle,
        scenario_hash=_truth_map_sha256(mutated_truth),
        truth=mutated_truth,
    )
    mutated_env = LunarExplorationEnv(
        config,
        scenario_source=_StaticScenarioSource(mutated_bundle),
    )
    mutated_observation = mutated_env.reset()

    truth_fields = ("height", "hard_obstacle", "slope_deg", "traversability")
    observed_truth_equal = all(
        np.array_equal(
            getattr(baseline_bundle.truth, name)[observed],
            getattr(mutated_truth, name)[observed],
        )
        for name in truth_fields
    )
    unobserved_truth_mutated = all(
        not np.array_equal(
            getattr(baseline_bundle.truth, name)[mutation_mask],
            getattr(mutated_truth, name)[mutation_mask],
        )
        for name in truth_fields
    )
    observation_equal = all(
        np.array_equal(left, right)
        for left, right in zip(
            baseline_observation.array_fields(),
            mutated_observation.array_fields(),
            strict=True,
        )
    )
    baseline_candidate_sha256 = _frontier_action_set_sha256(
        baseline_env.current_action_set
    )
    mutated_candidate_sha256 = _frontier_action_set_sha256(
        mutated_env.current_action_set
    )
    baseline_coverable_sha256 = baseline_env.coverage_metadata.get("sha256")
    mutated_coverable_sha256 = mutated_env.coverage_metadata.get("sha256")

    action_replay: dict[str, dict[str, object]] = {}
    for method in BASELINE_METHODS:
        baseline_action = select_baseline_action(
            method,
            baseline_observation,
            np.random.Generator(np.random.PCG64(20260715)),
        )
        mutated_action = select_baseline_action(
            method,
            mutated_observation,
            np.random.Generator(np.random.PCG64(20260715)),
        )
        baseline_record = _env_action_record(baseline_action)
        mutated_record = _env_action_record(mutated_action)
        action_replay[method] = {
            "baseline": baseline_record,
            "mutated": mutated_record,
            "byte_equal": (
                ArtifactStore.canonical_json_bytes(baseline_record)
                == ArtifactStore.canonical_json_bytes(mutated_record)
            ),
        }

    baseline_parameters = list(inspect.signature(select_baseline_action).parameters)
    evaluator_parameters = list(inspect.signature(Evaluator._select_action).parameters)
    selection_source = (
        inspect.getsource(select_baseline_action)
        + inspect.getsource(Evaluator._select_action)
    ).lower()
    audit = {
        "schema_version": "stage5_runtime_leakage_audit/v1",
        "mutation_source": "paired_unobserved_truth_and_coverable_mask/v1",
        "mutation_cell_count": int(mutation_mask.sum()),
        "observed_truth_cells_byte_equal": observed_truth_equal,
        "unobserved_truth_mutated": unobserved_truth_mutated,
        "coverable_mask_mutated": (
            isinstance(baseline_coverable_sha256, str)
            and isinstance(mutated_coverable_sha256, str)
            and baseline_coverable_sha256 != mutated_coverable_sha256
        ),
        "all_observation_arrays_byte_equal": observation_equal,
        "baseline_observation_sha256": _policy_observation_sha256(
            baseline_observation
        ),
        "mutated_observation_sha256": _policy_observation_sha256(
            mutated_observation
        ),
        "candidate_set_byte_equal": (
            baseline_candidate_sha256 == mutated_candidate_sha256
        ),
        "baseline_candidate_set_sha256": baseline_candidate_sha256,
        "mutated_candidate_set_sha256": mutated_candidate_sha256,
        "baseline_coverable_mask_sha256": baseline_coverable_sha256,
        "mutated_coverable_mask_sha256": mutated_coverable_sha256,
        "baseline_action_replay": action_replay,
        "all_baseline_decisions_invariant": all(
            row["byte_equal"] is True for row in action_replay.values()
        ),
        "baseline_selection_parameters": baseline_parameters,
        "evaluator_selection_parameters": evaluator_parameters,
        "truth_or_coverable_parameter_present": any(
            token in parameter.lower()
            for parameter in (*baseline_parameters, *evaluator_parameters)
            for token in ("truth", "coverable")
        ),
        "selection_source_forbidden_token_present": any(
            token in selection_source for token in ("truth", "coverable")
        ),
        "python_internal_inaccessibility_claimed": False,
    }
    return _validated_leakage_audit(audit)


def stage5_git_identity(repo_root: str | Path) -> dict[str, object]:
    try:
        identity = stage3_workflow.stage3_git_identity(
            repo_root,
            base_commit=STAGE4_COMMIT,
        )
    except Exception as exc:
        raise Stage5WorkflowError("Stage 5 prospective Git identity failed") from exc
    identity["schema_version"] = "ppo_highres_frontier_stage5_prospective_git_tree/v1"
    if tuple(identity.get("changed_paths", ())) != STAGE5_CHANGED_PATHS:
        raise Stage5WorkflowError("Stage 5 prospective changed path set drift")
    return identity


def _build_stage5_runtime_binding(
    *,
    config: Stage5Config,
    run_id: str,
    source_identity: dict[str, object],
    environment_identity: dict[str, object],
    git_identity: dict[str, object],
    stage4_authority: dict[str, object],
    stage1_config_binding: dict[str, object],
    checkpoint_audit: dict[str, object],
) -> Stage5RuntimeBinding:
    validate_run_id(run_id)
    if config.run_id is not None:
        raise Stage5WorkflowError("repository Stage 5 config must not bind a run ID")
    if (
        source_identity.get("schema_version") != "stage5_reviewed_source_set/v1"
        or not isinstance(source_identity.get("paths"), list)
        or not source_identity["paths"]
        or environment_identity.get("schema_version")
        != "ppo_highres_frontier_stage5_environment/v1"
        or environment_identity.get("cuda_available") is not True
        or environment_identity.get("compute_dtype") != "float32"
        or environment_identity.get("amp_enabled") is not False
        or git_identity.get("schema_version")
        != "ppo_highres_frontier_stage5_prospective_git_tree/v1"
        or git_identity.get("head_commit") != STAGE4_COMMIT
        or git_identity.get("base_commit") != STAGE4_COMMIT
        or tuple(git_identity.get("changed_paths", ())) != STAGE5_CHANGED_PATHS
        or git_identity.get("real_index_empty") is not True
    ):
        raise Stage5WorkflowError("Stage 5 execution identity binding is incomplete")
    expected_authority = {
        "schema_version": "stage5_stage4_authority_audit/v1",
        "verified": True,
        "authorized_stage": config.stage_id,
        **config.stage4_authority.model_dump(mode="json"),
        "update_step": 3,
    }
    if stage4_authority != expected_authority:
        raise Stage5WorkflowError("Stage 4 authority does not bind Stage 5 config")
    if stage1_config_binding != {
        "path": stage1_config_binding.get("path"),
        "sha256": config.stage1_config_sha256,
    } or not isinstance(stage1_config_binding.get("path"), str):
        raise Stage5WorkflowError("Stage 1 config binding drift")
    expected_checkpoint = {
        "schema_version": "stage5_frozen_checkpoint_audit/v1",
        "checkpoint_root": config.stage4_authority.checkpoint_root,
        "checkpoint_directory": "update-00000003",
        "checkpoint_sha256": config.stage4_authority.checkpoint_sha256,
        "checkpoint_manifest_sha256": (
            config.stage4_authority.checkpoint_manifest_sha256
        ),
        "policy_state_sha256": config.stage4_authority.policy_state_sha256,
        "update_step": 3,
        "device": "cuda",
        "read_only": True,
        "deterministic_action_replay": {
            "schema_version": "stage5_checkpoint_action_replay/v1",
            "fixture_sha256": STAGE4_ACTION_FIXTURE_SHA256,
            "checkpoint_load_audit_sha256": STAGE4_CHECKPOINT_LOAD_AUDIT_SHA256,
            "expected_action": dict(STAGE4_LATEST_DETERMINISTIC_ACTION),
            "actual_action": dict(STAGE4_LATEST_DETERMINISTIC_ACTION),
            "bit_exact": True,
        },
    }
    if checkpoint_audit != expected_checkpoint:
        raise Stage5WorkflowError("Stage 4 checkpoint audit drift")
    payload = config.model_dump(mode="json")
    payload["run_id"] = run_id
    payload["execution_source_identity"] = source_identity
    payload["execution_environment_identity"] = environment_identity
    payload["execution_git_identity"] = git_identity
    payload["execution_stage4_authority"] = stage4_authority
    payload["stage1_config_binding"] = stage1_config_binding
    payload["checkpoint_binding"] = checkpoint_audit
    try:
        config_bytes = ArtifactStore.canonical_json_bytes(payload)
    except (TypeError, ValueError) as exc:
        raise Stage5WorkflowError("Stage 5 runtime config is not canonical") from exc
    return Stage5RuntimeBinding(
        payload=payload,
        config_bytes=config_bytes,
        config_sha256=hashlib.sha256(config_bytes).hexdigest(),
    )


def _build_stage5_machine_payload(
    *,
    config: Stage5Config,
    runtime: Stage5RuntimeBinding,
    summaries: Sequence[EvaluationSummary],
    leakage_audit: Mapping[str, object],
) -> _Stage5MachinePayload:
    ordered = _validated_summaries(config, summaries)
    leakage = _validated_leakage_audit(leakage_audit)
    all_episodes = tuple(
        episode for summary in ordered for episode in summary.episodes
    )
    summary_rows = tuple(summary.metrics for summary in ordered)
    bootstrap_rows = tuple(summary.bootstrap_audit for summary in ordered)
    fairness_rows = tuple(summary.fairness_audit for summary in ordered)
    shared_hashes = {
        str(row["shared_contract_sha256"]) for row in fairness_rows
    }
    if len(shared_hashes) != 1:
        raise Stage5WorkflowError("methods do not share one fairness contract")
    shared_hash = next(iter(shared_hashes))
    comparison = comparison_table_csv(summary_rows, bootstrap_rows)
    curves = coverage_curves_csv(all_episodes)
    evaluator_report = baseline_eval_report(
        summary_rows,
        claim_boundary=config.claim_boundary,
    )
    traces = _canonical_jsonl_bytes(
        [episode_record(episode) for episode in all_episodes]
    )
    metrics_payload = {
        "schema_version": "stage5_baseline_metrics/v1",
        "primary_metric": config.primary_metric,
        "scale_profile": config.scale_profile,
        "method_episode_count": len(all_episodes),
        "zero_distance_policy": config.zero_distance_policy,
        "auc_policy": config.auc_policy,
        "terminal_curve_policy": config.terminal_curve_policy,
        "methods": list(summary_rows),
    }
    fairness_payload = {
        "schema_version": "stage5_combined_fairness_audit/v2",
        "baseline_comparison_contract": config.baseline_comparison_contract,
        "candidate_set_contract": config.candidate_set_contract,
        "information_contract": config.information_contract,
        "shared_contract_sha256": shared_hash,
        "all_methods_share_contract": True,
        "truth_mutation_boundary": "unobserved_truth_cannot_change_decision_input/v1",
        "leakage_audit": leakage,
        "methods": list(fairness_rows),
    }
    bootstrap_payload = {
        "schema_version": "stage5_bootstrap_ci_audit/v1",
        "bootstrap_resamples": config.bootstrap_resamples,
        "bootstrap_seed": config.bootstrap_seed,
        "sampling_unit": config.bootstrap_unit,
        "methods": list(bootstrap_rows),
    }
    acceptance_probe = _build_stage5_acceptance_probe_audit(
        summaries=ordered,
        comparison_csv=comparison,
        coverage_curves_csv=curves,
    )
    acceptance_probe = verify_stage5_acceptance_probe_audit(
        acceptance_probe,
        summaries=ordered,
        comparison_csv=comparison,
        coverage_curves_csv=curves,
    )
    acceptance_probe_bytes = ArtifactStore.canonical_json_bytes(acceptance_probe)
    authority = runtime.payload["execution_stage4_authority"]
    checkpoint = runtime.payload["checkpoint_binding"]
    source = runtime.payload["execution_source_identity"]
    environment = runtime.payload["execution_environment_identity"]
    git_identity = runtime.payload["execution_git_identity"]
    assert isinstance(authority, dict)
    assert isinstance(checkpoint, dict)
    assert isinstance(source, dict)
    assert isinstance(environment, dict)
    assert isinstance(git_identity, dict)
    execution_identity = {
        "schema_version": "stage5_execution_identity_audit/v1",
        "run_id": runtime.payload["run_id"],
        "runtime_config_sha256": runtime.config_sha256,
        "repository_source_set_sha256": source["source_set_sha256"],
        "environment_sha256": _hash_json(environment),
        "prospective_git_tree": git_identity["prospective_git_tree"],
        "changed_path_set_sha256": git_identity["changed_path_set_sha256"],
        "stage4_gate_sha256": authority["gate_sha256"],
        "checkpoint_sha256": checkpoint["checkpoint_sha256"],
        "policy_state_sha256": checkpoint["policy_state_sha256"],
        "scenario_schedule": [
            row.model_dump(mode="json") for row in config.scenario_schedule
        ],
    }
    acceptance = _acceptance_items(config, ordered, leakage, acceptance_probe)
    workflow_summary = {
        "schema_version": "ppo_highres_frontier_stage5_summary/v1",
        "goal_id": config.goal_id,
        "stage_id": config.stage_id,
        "run_id": runtime.payload["run_id"],
        "state": "machine_passed",
        "claim_boundary": config.claim_boundary,
        "performance_advantage_established": False,
        "primary_metric": config.primary_metric,
        "method_episode_count": len(all_episodes),
        "episodes_per_method": config.eval_episode_count,
        "runtime_config_sha256": runtime.config_sha256,
        "source_set_sha256": source["source_set_sha256"],
        "environment_sha256": _hash_json(environment),
        "prospective_git_tree": git_identity["prospective_git_tree"],
        "stage4_gate_sha256": authority["gate_sha256"],
        "checkpoint_sha256": checkpoint["checkpoint_sha256"],
        "policy_state_sha256": checkpoint["policy_state_sha256"],
        "acceptance": {
            "policy": "baseline_evaluator_acceptance/v1",
            "passed": all(acceptance.values()),
            "items": acceptance,
            "probe_artifact": "acceptance_probe_audit.json",
            "probe_sha256": hashlib.sha256(acceptance_probe_bytes).hexdigest(),
        },
        "methods": list(summary_rows),
    }
    if workflow_summary["acceptance"]["passed"] is not True:
        raise Stage5WorkflowError("Stage 5 machine acceptance failed")
    routing = {
        "schema_version": "ppo_highres_frontier_stage5_routing/v1",
        "state": "machine_passed",
        "route": "awaiting_independent_review",
        "authorized_next_stage": config.authorized_next_stage,
        "next_stage_entered": False,
        "claim_boundary": config.claim_boundary,
    }
    phases = (
        {
            "schema_version": "ppo_highres_frontier_stage5_phase_state/v1",
            "sequence": 1,
            "state": "machine_passed",
        },
        {
            "schema_version": "ppo_highres_frontier_stage5_phase_state/v1",
            "sequence": 2,
            "state": "awaiting_independent_review",
        },
    )
    progress_records = _progress_records(ordered, config.max_steps)
    report = _workflow_report(workflow_summary)
    artifacts = {
        "acceptance_probe_audit.json": acceptance_probe_bytes,
        "config.json": runtime.config_bytes,
        "summary.json": ArtifactStore.canonical_json_bytes(workflow_summary),
        "routing.json": ArtifactStore.canonical_json_bytes(routing),
        "report.md": report,
        "metrics.jsonl": _canonical_jsonl_bytes(progress_records),
        "phase-state.jsonl": _canonical_jsonl_bytes(phases),
        "baseline_eval_report.md": evaluator_report,
        "baseline_comparison_table.csv": comparison,
        "baseline_metrics.json": ArtifactStore.canonical_json_bytes(metrics_payload),
        "baseline_coverage_curves.csv": curves,
        "baseline_episode_traces.jsonl": traces,
        "baseline_fairness_audit.json": ArtifactStore.canonical_json_bytes(
            fairness_payload
        ),
        "bootstrap_ci_audit.json": ArtifactStore.canonical_json_bytes(
            bootstrap_payload
        ),
        "stage4_authority_audit.json": ArtifactStore.canonical_json_bytes(authority),
        "execution_identity_audit.json": ArtifactStore.canonical_json_bytes(
            execution_identity
        ),
    }
    if set(artifacts) != set(STAGE5_MANIFEST_BOUND_ARTIFACTS):
        raise Stage5WorkflowError("Stage 5 artifact path set drift")
    manifest = _manifest_for_artifacts(artifacts)
    return _Stage5MachinePayload(
        artifacts=artifacts,
        manifest=manifest,
        summary=workflow_summary,
        routing=routing,
    )


def verify_stage5_manifest_graph(
    stage_root: str | Path,
) -> Stage5ManifestGraphHandle:
    stage = Path(stage_root).expanduser().resolve()
    if stage.name != "s5" or not stage.is_dir():
        raise Stage5WorkflowError("Stage 5 root must be canonical <run>/s5")
    if {path.name for path in stage.iterdir()} != set(STAGE5_ROOT_ARTIFACTS):
        raise Stage5WorkflowError("Stage 5 root artifact set drift")
    if any((stage / name).exists() for name in ("review.json", "approval.json", "gate.json")):
        raise Stage5WorkflowError("Stage 5 root artifact set contains authority output")
    try:
        manifest_snapshot = FrozenFileSnapshot.capture(
            stage / "manifest.json",
            "Stage 5 manifest",
            authority_root=stage,
        )
        manifest = _strict_canonical_json(manifest_snapshot, "Stage 5 manifest")
        entries = manifest.get("artifacts")
        if (
            manifest.get("schema_version") != "sha256_manifest/v1"
            or not isinstance(entries, list)
            or [entry.get("path") for entry in entries if isinstance(entry, dict)]
            != sorted(STAGE5_MANIFEST_BOUND_ARTIFACTS)
        ):
            raise Stage5WorkflowError("Stage 5 manifest paths are not exact")
        snapshots: list[tuple[str, FrozenFileSnapshot]] = []
        for entry in entries:
            if not isinstance(entry, dict) or set(entry) != {
                "path",
                "sha256",
                "size_bytes",
            }:
                raise Stage5WorkflowError("Stage 5 manifest entry schema drift")
            relative = entry["path"]
            if not isinstance(relative, str):
                raise Stage5WorkflowError("Stage 5 manifest path is invalid")
            snapshot = FrozenFileSnapshot.capture(
                stage / relative,
                f"Stage 5 artifact/{relative}",
                authority_root=stage,
            )
            if (
                snapshot.sha256 != entry["sha256"]
                or snapshot.size_bytes != entry["size_bytes"]
            ):
                raise Stage5WorkflowError("Stage 5 manifest artifact hash drift")
            snapshots.append((relative, snapshot))
    except Stage5WorkflowError:
        raise
    except (OSError, Stage1WorkflowError) as exc:
        raise Stage5WorkflowError("Stage 5 manifest graph drift") from exc
    handle = Stage5ManifestGraphHandle(
        stage_root=stage,
        manifest_snapshot=manifest_snapshot,
        artifacts=tuple(snapshots),
    )
    handle.require_current()
    return handle


def run_stage5_workflow(
    *,
    config_path: str | Path,
    run_id: str,
    stage4_gate_path: str | Path,
    checkpoint_root: str | Path,
    base_output_root: str | Path | None = None,
) -> Stage5WorkflowResult:
    """Run exactly 80 Smoke method-episodes and stop at independent review."""

    validate_run_id(run_id)
    repo = Path(__file__).resolve().parents[3]
    config_path_resolved = Path(config_path).expanduser().resolve()
    try:
        config_snapshot = FrozenFileSnapshot.capture(
            config_path_resolved,
            "Stage 5 repository config",
            authority_root=repo,
        )
        config = Stage5Config.model_validate_json(config_snapshot.payload)
        stage1_snapshot = FrozenFileSnapshot.capture(
            repo / config.stage1_config_path,
            "Stage 1 Smoke config",
            authority_root=repo,
        )
        stage1_config = Stage1Config.model_validate_json(stage1_snapshot.payload)
    except (Stage1WorkflowError, ValueError) as exc:
        raise Stage5WorkflowError("Stage 5 repository config snapshot is invalid") from exc
    if stage1_snapshot.sha256 != config.stage1_config_sha256:
        raise Stage5WorkflowError("Stage 1 Smoke config SHA-256 drift")
    expected_gate = Path(config.stage4_authority.gate_path).resolve()
    expected_checkpoint_root = Path(config.stage4_authority.checkpoint_root).resolve()
    if (
        Path(stage4_gate_path).expanduser().resolve() != expected_gate
        or Path(checkpoint_root).expanduser().resolve() != expected_checkpoint_root
    ):
        raise Stage5WorkflowError("Stage 4 authority CLI path drift")
    base = Path(base_output_root) if base_output_root is not None else Path(config.output_root)
    run_root = (base / run_id).expanduser().resolve()
    if run_root.drive.upper() != "D:":
        raise Stage5WorkflowError("Stage 5 runtime output must be on D drive")
    if run_root.exists():
        raise Stage5WorkflowError("Stage 5 run root already exists")

    source_identity = stage5_source_identity(repo)
    environment_identity = stage5_environment_identity()
    git_identity = stage5_git_identity(repo)
    authority = verify_frozen_stage4_authority(
        gate_path=stage4_gate_path,
        checkpoint_root=checkpoint_root,
        repo_root=repo,
    )
    loaded_policy = load_frozen_stage4_policy(authority, device=config.device)
    stage1_binding = {
        "path": str(stage1_snapshot.path),
        "sha256": stage1_snapshot.sha256,
    }
    runtime = _build_stage5_runtime_binding(
        config=config,
        run_id=run_id,
        source_identity=source_identity,
        environment_identity=environment_identity,
        git_identity=git_identity,
        stage4_authority=authority.identity,
        stage1_config_binding=stage1_binding,
        checkpoint_audit=loaded_policy.audit,
    )
    scenarios = tuple(
        EvaluationScenario(**row.model_dump()) for row in config.scenario_schedule
    )
    evaluator = Evaluator(
        env_factory=lambda _scenario: LunarExplorationEnv(stage1_config),
        scale_profile=config.scale_profile,
        max_steps=config.max_steps,
        success_threshold=config.success_threshold,
        zero_distance_policy=config.zero_distance_policy,
        bootstrap_resamples=config.bootstrap_resamples,
        bootstrap_seed=config.bootstrap_seed,
        policy=loaded_policy.policy,
        policy_device=config.device,
    )
    summaries = tuple(
        evaluator.evaluate(method, scenarios) for method in config.methods
    )
    if policy_state_sha256(loaded_policy.policy) != config.stage4_authority.policy_state_sha256:
        raise Stage5WorkflowError("Stage 5 evaluation mutated the frozen policy")
    if (
        stage5_source_identity(repo) != source_identity
        or stage5_environment_identity() != environment_identity
        or stage5_git_identity(repo) != git_identity
    ):
        raise Stage5WorkflowError("Stage 5 source, environment, or Git drifted")
    authority.require_current("Stage 5 post-evaluation Stage 4 authority")
    try:
        config_snapshot.require_current("Stage 5 repository config")
        stage1_snapshot.require_current("Stage 5 Stage 1 config")
    except Stage1WorkflowError as exc:
        raise Stage5WorkflowError("Stage 5 config changed during execution") from exc
    payload = _build_stage5_machine_payload(
        config=config,
        runtime=runtime,
        summaries=summaries,
        leakage_audit=stage5_runtime_leakage_audit(stage1_config),
    )
    run_root.mkdir(parents=True, exist_ok=False)
    stage = run_root / "s5"
    stage.mkdir(exist_ok=False)
    store = ArtifactStore(stage)
    _write_stage5_machine_payload(store, payload)
    verify_stage5_machine_run(
        stage_root=stage,
        repo_root=repo,
        stage4_gate_path=stage4_gate_path,
        checkpoint_root=checkpoint_root,
    ).require_current("Stage 5 controller final verification")
    return Stage5WorkflowResult(
        run_id=run_id,
        stage_root=stage,
        summary=payload.summary,
        routing=payload.routing,
        manifest=payload.manifest,
    )


def _write_stage5_machine_payload(
    store: ArtifactStore,
    payload: _Stage5MachinePayload,
) -> None:
    if set(payload.artifacts) != set(STAGE5_MANIFEST_BOUND_ARTIFACTS):
        raise Stage5WorkflowError("Stage 5 writer artifact set drift")
    if any(store.resolve(relative).exists() for relative in STAGE5_ROOT_ARTIFACTS):
        raise Stage5WorkflowError("Stage 5 writer requires an empty stage root")
    for relative in STAGE5_MANIFEST_BOUND_ARTIFACTS:
        body = payload.artifacts[relative]
        if relative in _STAGE5_JSONL_ARTIFACTS:
            for line_number, line in enumerate(body.splitlines(), start=1):
                try:
                    value = strict_json_object_from_bytes(
                        line,
                        f"Stage 5 {relative} line {line_number}",
                    )
                except Stage1WorkflowError as exc:
                    raise Stage5WorkflowError(
                        f"Stage 5 writer JSONL payload is invalid: {relative}"
                    ) from exc
                store.append_jsonl(relative, value)
        else:
            store.write_bytes(relative, body)
        if store.resolve(relative).read_bytes() != body:
            raise Stage5WorkflowError(f"Stage 5 writer byte replay drift: {relative}")
    store.write_json("manifest.json", payload.manifest)
    if store.resolve("manifest.json").read_bytes() != ArtifactStore.canonical_json_bytes(
        payload.manifest
    ):
        raise Stage5WorkflowError("Stage 5 writer manifest byte replay drift")


def verify_stage5_machine_run(
    *,
    stage_root: str | Path,
    repo_root: str | Path,
    stage4_gate_path: str | Path,
    checkpoint_root: str | Path,
) -> Stage5MachineVerificationHandle:
    stage = Path(stage_root).expanduser().resolve()
    repo = Path(repo_root).expanduser().resolve()
    graph = verify_stage5_manifest_graph(stage)
    snapshots = dict(graph.artifacts)
    runtime_payload = _strict_canonical_json(snapshots["config.json"], "Stage 5 config")
    source = runtime_payload.pop("execution_source_identity", None)
    environment = runtime_payload.pop("execution_environment_identity", None)
    git_identity = runtime_payload.pop("execution_git_identity", None)
    recorded_authority = runtime_payload.pop("execution_stage4_authority", None)
    stage1_binding = runtime_payload.pop("stage1_config_binding", None)
    checkpoint_binding = runtime_payload.pop("checkpoint_binding", None)
    try:
        machine_config = Stage5Config.model_validate(runtime_payload)
        config_snapshot = FrozenFileSnapshot.capture(
            repo / "configs/ppo_highres_frontier_stage5_v1.json",
            "Stage 5 repository config",
            authority_root=repo,
        )
        repository_config = Stage5Config.model_validate_json(config_snapshot.payload)
        stage1_snapshot = FrozenFileSnapshot.capture(
            repo / repository_config.stage1_config_path,
            "Stage 1 Smoke config",
            authority_root=repo,
        )
        stage1_config = Stage1Config.model_validate_json(stage1_snapshot.payload)
    except (Stage1WorkflowError, ValueError) as exc:
        raise Stage5WorkflowError("Stage 5 verifier config binding drift") from exc
    if (
        machine_config.model_dump(exclude={"run_id"})
        != repository_config.model_dump(exclude={"run_id"})
        or machine_config.run_id != stage.parent.name
        or not all(
            isinstance(value, dict)
            for value in (
                source,
                environment,
                git_identity,
                recorded_authority,
                stage1_binding,
                checkpoint_binding,
            )
        )
    ):
        raise Stage5WorkflowError("Stage 5 runtime config semantics drift")
    assert isinstance(source, dict)
    assert isinstance(environment, dict)
    assert isinstance(git_identity, dict)
    assert isinstance(recorded_authority, dict)
    assert isinstance(stage1_binding, dict)
    assert isinstance(checkpoint_binding, dict)
    current_source = stage5_source_identity(repo)
    current_environment = stage5_environment_identity()
    current_git = stage5_git_identity(repo)
    authority = verify_frozen_stage4_authority(
        gate_path=stage4_gate_path,
        checkpoint_root=checkpoint_root,
        repo_root=repo,
    )
    loaded_policy = load_frozen_stage4_policy(authority, device=repository_config.device)
    current_stage1_binding = {
        "path": str(stage1_snapshot.path),
        "sha256": stage1_snapshot.sha256,
    }
    if (
        source != current_source
        or environment != current_environment
        or git_identity != current_git
        or recorded_authority != authority.identity
        or stage1_binding != current_stage1_binding
        or checkpoint_binding != loaded_policy.audit
    ):
        raise Stage5WorkflowError("Stage 5 execution identity drift")
    runtime = _build_stage5_runtime_binding(
        config=repository_config,
        run_id=stage.parent.name,
        source_identity=current_source,
        environment_identity=current_environment,
        git_identity=current_git,
        stage4_authority=authority.identity,
        stage1_config_binding=current_stage1_binding,
        checkpoint_audit=loaded_policy.audit,
    )
    if snapshots["config.json"].payload != runtime.config_bytes:
        raise Stage5WorkflowError("Stage 5 canonical runtime config drift")
    traces = _read_episode_traces(snapshots["baseline_episode_traces.jsonl"].payload)
    fairness = _strict_canonical_json(
        snapshots["baseline_fairness_audit.json"],
        "Stage 5 fairness audit",
    )
    fairness_methods = fairness.get("methods")
    if not isinstance(fairness_methods, list):
        raise Stage5WorkflowError("Stage 5 fairness audit method schema drift")
    fairness_by_method = {
        row.get("method"): row for row in fairness_methods if isinstance(row, dict)
    }
    summaries: list[EvaluationSummary] = []
    for method in repository_config.methods:
        episodes = tuple(row for row in traces if row.method == method)
        metrics, bootstrap = summarize_episodes(
            episodes,
            bootstrap_resamples=repository_config.bootstrap_resamples,
            bootstrap_seed=repository_config.bootstrap_seed,
        )
        if method not in fairness_by_method:
            raise Stage5WorkflowError("Stage 5 fairness audit method missing")
        summaries.append(
            EvaluationSummary(
                method=method,
                scale_profile=repository_config.scale_profile,
                episodes=episodes,
                metrics=metrics,
                bootstrap_audit=bootstrap,
                fairness_audit=fairness_by_method[method],
            )
        )
    acceptance_probe = _strict_canonical_json(
        snapshots["acceptance_probe_audit.json"],
        "Stage 5 acceptance probe audit",
    )
    verify_stage5_acceptance_probe_audit(
        acceptance_probe,
        summaries=summaries,
        comparison_csv=snapshots["baseline_comparison_table.csv"].payload,
        coverage_curves_csv=snapshots["baseline_coverage_curves.csv"].payload,
    )
    expected = _build_stage5_machine_payload(
        config=repository_config,
        runtime=runtime,
        summaries=summaries,
        leakage_audit=stage5_runtime_leakage_audit(stage1_config),
    )
    for relative in STAGE5_MANIFEST_BOUND_ARTIFACTS:
        if snapshots[relative].payload != expected.artifacts[relative]:
            raise Stage5WorkflowError(
                f"Stage 5 deterministic artifact drift: {relative}"
            )
    if graph.manifest_snapshot.payload != ArtifactStore.canonical_json_bytes(
        expected.manifest
    ):
        raise Stage5WorkflowError("Stage 5 manifest semantic drift")
    handle = Stage5MachineVerificationHandle(
        manifest_graph=graph,
        stage4_authority=authority,
        config_snapshot=config_snapshot,
        stage1_snapshot=stage1_snapshot,
    )
    handle.require_current()
    return handle


def _validated_summaries(
    config: Stage5Config,
    summaries: Sequence[EvaluationSummary],
) -> tuple[EvaluationSummary, ...]:
    rows = tuple(summaries)
    by_method = {row.method: row for row in rows if isinstance(row, EvaluationSummary)}
    if len(rows) != len(config.methods) or set(by_method) != set(config.methods):
        raise Stage5WorkflowError("Stage 5 summary method set drift")
    ordered = tuple(by_method[method] for method in config.methods)
    expected_schedule = [row.model_dump(mode="json") for row in config.scenario_schedule]
    episode_schemas: set[tuple[str, ...]] = set()
    for summary in ordered:
        expected_action_rule = METHOD_ACTION_RULES.get(summary.method)
        if (
            summary.scale_profile != config.scale_profile
            or len(summary.episodes) != config.eval_episode_count
            or summary.fairness_audit.get("schema_version")
            != "stage5_method_fairness_audit/v2"
            or summary.fairness_audit.get("method") != summary.method
            or summary.fairness_audit.get(
                "shared_environment_contract_identical"
            )
            is not True
            or summary.fairness_audit.get(
                "method_specific_action_rule_only_difference"
            )
            is not True
            or not isinstance(expected_action_rule, Mapping)
            or summary.fairness_audit.get("candidate_index_rule")
            != expected_action_rule["candidate_index_rule"]
            or summary.fairness_audit.get("theta_rule")
            != expected_action_rule["theta_rule"]
            or "candidate_selection_only_difference" in summary.fairness_audit
            or summary.fairness_audit.get(
                "all_selected_actions_reachable_observed_safe"
            )
            is not True
            or summary.fairness_audit.get("invalid_selected_action_count") != 0
            or (
                summary.method in BASELINE_METHODS
                and summary.fairness_audit.get("all_baseline_thetas_recommended")
                is not True
            )
            or (
                summary.method == "ppo_policy"
                and summary.fairness_audit.get("all_baseline_thetas_recommended")
                is not None
            )
        ):
            raise Stage5WorkflowError("Stage 5 method evaluation did not pass fairness")
        _validate_shared_environment_contract(
            summary.fairness_audit,
            expected_schedule=expected_schedule,
            config=config,
        )
        actual_schedule = [
            {
                "scenario_key": episode.scenario_key,
                "scenario_seed": episode.scenario_seed,
                "terrain_seed": episode.terrain_seed,
                "start_pose_seed": episode.start_pose_seed,
                "evaluation_seed": episode.evaluation_seed,
            }
            for episode in summary.episodes
        ]
        if actual_schedule != expected_schedule:
            raise Stage5WorkflowError("Stage 5 method schedule drift")
        for episode in summary.episodes:
            if episode.method != summary.method or len(episode.coverage_curve) != 65:
                raise Stage5WorkflowError("Stage 5 episode schema or curve drift")
            episode_schemas.add(tuple(episode_record(episode)))
        expected_metrics, expected_bootstrap = summarize_episodes(
            summary.episodes,
            bootstrap_resamples=config.bootstrap_resamples,
            bootstrap_seed=config.bootstrap_seed,
        )
        if summary.metrics != expected_metrics or summary.bootstrap_audit != expected_bootstrap:
            raise Stage5WorkflowError("Stage 5 metrics or bootstrap drift")
    if episode_schemas != {EPISODE_FIELDS}:
        raise Stage5WorkflowError("Stage 5 per-method episode schema differs")
    return ordered


def _validate_shared_environment_contract(
    fairness_audit: Mapping[str, object],
    *,
    expected_schedule: Sequence[Mapping[str, object]],
    config: Stage5Config,
) -> None:
    contract = fairness_audit.get("shared_contract")
    if not isinstance(contract, Mapping):
        raise Stage5WorkflowError("Stage 5 shared environment contract is missing")
    episode_contracts = contract.get("episodes")
    if (
        contract.get("schema_version") != "stage5_shared_environment_contract/v1"
        or contract.get("scale_profile") != config.scale_profile
        or contract.get("max_steps") != config.max_steps
        or contract.get("success_threshold") != config.success_threshold
        or contract.get("scenario_schedule") != list(expected_schedule)
        or not isinstance(episode_contracts, list)
        or len(episode_contracts) != config.eval_episode_count
        or fairness_audit.get("shared_contract_sha256") != _hash_json(contract)
    ):
        raise Stage5WorkflowError("Stage 5 shared environment contract drift")
    for expected_scenario, episode_contract in zip(
        expected_schedule,
        episode_contracts,
        strict=True,
    ):
        if not isinstance(episode_contract, Mapping):
            raise Stage5WorkflowError("Stage 5 shared environment contract row is invalid")
        required_hashes = (
            episode_contract.get("scenario_hash"),
            episode_contract.get("environment_config_sha256"),
            episode_contract.get("coverable_mask_hash"),
        )
        if (
            episode_contract.get("scenario") != expected_scenario
            or episode_contract.get("environment_class")
            != "lunar_exploration_ppo.env.env.LunarExplorationEnv"
            or not all(isinstance(value, str) and len(value) == 64 for value in required_hashes)
            or episode_contract.get("sensor_model_id")
            != "path-tangent-plus-endpoint-theta-fov-90-range-20m-los/v1"
            or episode_contract.get("initial_scan") != "free_reset_scan_20m_90deg/v1"
            or not isinstance(episode_contract.get("frontier_generator_class"), str)
            or episode_contract.get("candidate_feature_schema")
            != "frontier_features_22/v1"
            or episode_contract.get("reachability_prefilter")
            != "observed_safe_connected_component/v1"
            or not isinstance(episode_contract.get("planner_class"), str)
            or episode_contract.get("planner_validation_source")
            != "stage1_path_planner_adapter/v1"
            or episode_contract.get("path_execution_sensor_updates")
            != "shared_environment_step/v1"
            or episode_contract.get("coverage_denominator_source")
            != "coverable_mask_exact"
            or episode_contract.get("coverable_mask_exact") is not True
            or episode_contract.get("coverable_mask_algorithm_id")
            != "exact_reachable_safe_pose_range_los/v1"
            or episode_contract.get("coverable_mask_precompute_scope")
            != "scenario_reset/v1"
            or type(episode_contract.get("coverable_cell_count")) is not int
            or episode_contract["coverable_cell_count"] <= 0
            or episode_contract.get("max_steps") != config.max_steps
            or type(episode_contract.get("stagnation_no_gain_steps")) is not int
            or episode_contract["stagnation_no_gain_steps"] <= 0
            or episode_contract.get("success_threshold") != config.success_threshold
            or not isinstance(episode_contract.get("safety_constants"), Mapping)
        ):
            raise Stage5WorkflowError("Stage 5 shared environment contract is incomplete")


def _acceptance_items(
    config: Stage5Config,
    summaries: Sequence[EvaluationSummary],
    leakage_audit: Mapping[str, object],
    acceptance_probe_audit: Mapping[str, object],
) -> dict[str, bool]:
    by_method = {summary.method: summary for summary in summaries}
    episodes = tuple(
        episode for summary in summaries for episode in summary.episodes
    )
    shared_hashes = {
        summary.fairness_audit["shared_contract_sha256"] for summary in summaries
    }
    probes = acceptance_probe_audit.get("probes")
    if (
        acceptance_probe_audit.get("schema_version")
        != "stage5_acceptance_probe_audit/v1"
        or acceptance_probe_audit.get("passed") is not True
        or not isinstance(probes, Mapping)
    ):
        raise Stage5WorkflowError("Stage 5 acceptance probe audit is invalid")
    return {
        "01_random_valid_frontier_complete": len(by_method["random_valid_frontier"].episodes) == 16,
        "02_nearest_frontier_complete": len(by_method["nearest_frontier"].episodes) == 16,
        "03_max_potential_gain_frontier_complete": len(by_method["max_potential_gain_frontier"].episodes) == 16,
        "04_gain_over_cost_frontier_complete": len(by_method["gain_over_cost_frontier"].episodes) == 16,
        "05_shared_environment_seed_candidate_planner_sensor_budget": len(shared_hashes) == 1,
        "06_hidden_truth_not_decision_input": (
            leakage_audit["unobserved_truth_mutated"] is True
            and leakage_audit["all_observation_arrays_byte_equal"] is True
            and leakage_audit["candidate_set_byte_equal"] is True
            and leakage_audit["all_baseline_decisions_invariant"] is True
        ),
        "07_dense_coverable_mask_not_decision_input": (
            leakage_audit["coverable_mask_mutated"] is True
            and leakage_audit["all_observation_arrays_byte_equal"] is True
            and leakage_audit["all_baseline_decisions_invariant"] is True
        ),
        "08_baseline_actions_reachable_observed_safe": all(
            summary.fairness_audit["all_selected_actions_reachable_observed_safe"]
            for summary in summaries
        ),
        "09_empty_candidate_uses_no_candidate_done": probes["empty_candidate"][
            "passed"
        ]
        is True,
        "10_deterministic_baselines_replayable": probes[
            "deterministic_baselines"
        ]["passed"]
        is True,
        "11_random_baseline_fixed_seed_replayable": probes["random_baseline"][
            "passed"
        ]
        is True,
        "12_ppo_argmax_selected_theta_mu": config.ppo_eval_policy_mode == "deterministic_argmax_frontier_mean_theta/v1",
        "13_ties_lowest_index": probes["lowest_index_ties"]["passed"] is True,
        "14_same_episode_metrics_schema": all(tuple(episode_record(row)) == EPISODE_FIELDS for row in episodes),
        "15_success_rate_threshold_ge_099": all(
            row.success == (row.final_coverage >= config.success_threshold) for row in episodes
        ),
        "16_path_length_success_only": all(
            (row.path_length_to_99_success_only is not None) == row.success for row in episodes
        ),
        "17_coverage_per_meter_finite": all(math.isfinite(row.coverage_per_meter) for row in episodes),
        "18_invalid_and_planner_failure_means": all(
            "invalid_action_count_mean" in summary.metrics
            and "planner_failure_count_mean" in summary.metrics
            for summary in summaries
        ),
        "19_fixed_seed_episode_bootstrap_replayable": all(
            summary.bootstrap_audit["resample_count"] == config.bootstrap_resamples
            and summary.bootstrap_audit["bootstrap_seed"] == config.bootstrap_seed
            and summary.bootstrap_audit["sampling_unit"] == "episode"
            for summary in summaries
        ),
        "20_comparison_csv_written_deterministically": probes["comparison_csv"][
            "passed"
        ]
        is True,
        "21_coverage_curves_csv_written_deterministically": probes[
            "coverage_curves_csv"
        ]["passed"]
        is True,
        "22_eval_report_states_claim_boundary": True,
    }


def _build_stage5_acceptance_probe_audit(
    *,
    summaries: Sequence[EvaluationSummary],
    comparison_csv: bytes,
    coverage_curves_csv: bytes,
) -> dict[str, object]:
    ordered = tuple(summaries)
    if tuple(summary.method for summary in ordered) != ALL_METHODS:
        raise Stage5WorkflowError("Stage 5 acceptance probe summary order drift")
    summary_rows = tuple(summary.metrics for summary in ordered)
    bootstrap_rows = tuple(summary.bootstrap_audit for summary in ordered)
    episodes = tuple(
        episode for summary in ordered for episode in summary.episodes
    )
    replay_input = _acceptance_selector_input("replay")
    tie_input = _acceptance_selector_input("tie")
    empty_input = _acceptance_selector_input("empty")
    canonical_inputs = {
        "schema_version": "stage5_acceptance_probe_inputs/v1",
        "selector_replay": replay_input,
        "lowest_index_tie": tie_input,
        "empty_candidate": empty_input,
        "deterministic_seed": 20260715,
        "random_baseline": {
            "selector_input": "selector_replay",
            "rng_algorithm": "numpy.random.PCG64",
            "seed": 20260715,
            "sequence_length": 64,
        },
        "actual_summary_replay": {
            "comparison_input_artifacts": [
                "baseline_metrics.json",
                "bootstrap_ci_audit.json",
            ],
            "summary_rows_sha256": _hash_json(list(summary_rows)),
            "bootstrap_rows_sha256": _hash_json(list(bootstrap_rows)),
            "coverage_input_artifacts": ["baseline_episode_traces.jsonl"],
            "episode_records_sha256": hashlib.sha256(
                _canonical_jsonl_bytes(
                    [episode_record(episode) for episode in episodes]
                )
            ).hexdigest(),
        },
    }
    probes = {
        "empty_candidate": _empty_candidate_acceptance_probe(empty_input),
        "deterministic_baselines": _deterministic_baseline_acceptance_probe(
            replay_input,
            seed=int(canonical_inputs["deterministic_seed"]),
        ),
        "random_baseline": _random_baseline_acceptance_probe(
            replay_input,
            seed=int(canonical_inputs["random_baseline"]["seed"]),
            sequence_length=int(
                canonical_inputs["random_baseline"]["sequence_length"]
            ),
        ),
        "lowest_index_ties": _lowest_index_tie_acceptance_probe(tie_input),
        "comparison_csv": _comparison_csv_acceptance_probe(
            summary_rows,
            bootstrap_rows,
            published=comparison_csv,
            canonical_input=canonical_inputs["actual_summary_replay"],
        ),
        "coverage_curves_csv": _coverage_curves_acceptance_probe(
            episodes,
            published=coverage_curves_csv,
            canonical_input=canonical_inputs["actual_summary_replay"],
        ),
    }
    bindings = {
        "09_empty_candidate_uses_no_candidate_done": "empty_candidate",
        "10_deterministic_baselines_replayable": "deterministic_baselines",
        "11_random_baseline_fixed_seed_replayable": "random_baseline",
        "13_ties_lowest_index": "lowest_index_ties",
        "20_comparison_csv_written_deterministically": "comparison_csv",
        "21_coverage_curves_csv_written_deterministically": "coverage_curves_csv",
    }
    return {
        "schema_version": "stage5_acceptance_probe_audit/v1",
        "probe_policy": "manifest_bound_current_code_recomputation/v1",
        "canonical_inputs": canonical_inputs,
        "canonical_inputs_sha256": _hash_json(canonical_inputs),
        "probes": probes,
        "acceptance_item_bindings": bindings,
        "passed": all(probes[name]["passed"] is True for name in bindings.values()),
    }


def verify_stage5_acceptance_probe_audit(
    value: Mapping[str, object],
    *,
    summaries: Sequence[EvaluationSummary],
    comparison_csv: bytes,
    coverage_curves_csv: bytes,
) -> dict[str, object]:
    """Recompute every manifest-bound Stage 5 acceptance probe with current code."""

    if not isinstance(value, Mapping):
        raise Stage5WorkflowError("Stage 5 acceptance probe audit is invalid")
    try:
        expected = _build_stage5_acceptance_probe_audit(
            summaries=summaries,
            comparison_csv=comparison_csv,
            coverage_curves_csv=coverage_curves_csv,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise Stage5WorkflowError(
            "Stage 5 acceptance probe recomputation failed"
        ) from exc
    if dict(value) != expected or expected["passed"] is not True:
        raise Stage5WorkflowError("Stage 5 acceptance probe audit drift")
    return expected


def _acceptance_selector_input(kind: str) -> dict[str, object]:
    common = {
        "schema_version": "stage5_selector_probe_input/v1",
        "distance_from_robot_norm": [0.4, 0.1, 0.0, 0.2],
        "potential_coverage_gain_norm": [0.1, 0.3, 1.0, 0.8],
        "reachable_prefilter_cost_norm": [0.0, 0.2, 0.0, 2.0],
        "recommended_theta_sin": [0.0, 1.0, 0.0, -1.0],
        "recommended_theta_cos": [1.0, 0.0, -1.0, 0.0],
        "candidate_valid_mask": [True, True, False, True],
        "ppo_frontier_logits": [1.0, 5.0, 100.0, 4.0],
        "ppo_theta_mu": [-0.4, 0.25, 1.7, -2.0],
    }
    if kind == "replay":
        return common
    if kind == "empty":
        return {**common, "candidate_valid_mask": [False, False, False, False]}
    if kind == "tie":
        return {
            **common,
            "distance_from_robot_norm": [9.0, 0.2, 0.2, 0.2],
            "potential_coverage_gain_norm": [9.0, 0.4, 0.4, 0.4],
            "reachable_prefilter_cost_norm": [0.0, 0.0, 0.0, 0.0],
            "candidate_valid_mask": [False, True, True, True],
            "ppo_frontier_logits": [9.0, 4.0, 4.0, 4.0],
        }
    raise Stage5WorkflowError("Stage 5 acceptance selector input kind is invalid")


def _probe_observation(value: Mapping[str, object]) -> PolicyObservation:
    expected_fields = {
        "schema_version",
        "distance_from_robot_norm",
        "potential_coverage_gain_norm",
        "reachable_prefilter_cost_norm",
        "recommended_theta_sin",
        "recommended_theta_cos",
        "candidate_valid_mask",
        "ppo_frontier_logits",
        "ppo_theta_mu",
    }
    if value.get("schema_version") != "stage5_selector_probe_input/v1" or set(
        value
    ) != expected_fields:
        raise Stage5WorkflowError("Stage 5 selector probe input schema drift")
    count = 4
    numeric_fields = (
        "distance_from_robot_norm",
        "potential_coverage_gain_norm",
        "reachable_prefilter_cost_norm",
        "recommended_theta_sin",
        "recommended_theta_cos",
        "ppo_frontier_logits",
        "ppo_theta_mu",
    )
    if any(
        not isinstance(value[name], list)
        or len(value[name]) != count
        or not all(
            isinstance(item, (int, float)) and math.isfinite(float(item))
            for item in value[name]
        )
        for name in numeric_fields
    ):
        raise Stage5WorkflowError("Stage 5 selector probe numeric input drift")
    mask = value["candidate_valid_mask"]
    if (
        not isinstance(mask, list)
        or len(mask) != count
        or not all(type(item) is bool for item in mask)
    ):
        raise Stage5WorkflowError("Stage 5 selector probe mask input drift")
    features = np.zeros((count, 22), dtype=np.float32)
    features[:, 2] = np.asarray(value["distance_from_robot_norm"], dtype=np.float32)
    features[:, 5] = np.asarray(
        value["potential_coverage_gain_norm"], dtype=np.float32
    )
    features[:, 14] = np.asarray(value["recommended_theta_sin"], dtype=np.float32)
    features[:, 15] = np.asarray(value["recommended_theta_cos"], dtype=np.float32)
    features[:, 18] = np.asarray(
        value["reachable_prefilter_cost_norm"], dtype=np.float32
    )
    return PolicyObservation(
        prior_channels=np.zeros((7, 2, 2), dtype=np.float32),
        coverage_summary=np.zeros((8, 2, 2), dtype=np.float32),
        local_crop=np.zeros((8, 2, 2), dtype=np.float32),
        frontier_features=features,
        pose_features=np.zeros((6,), dtype=np.float32),
        candidate_mask=np.asarray(mask, dtype=bool),
    )


def _probe_policy_output(value: Mapping[str, object]) -> PolicyForwardOutput:
    logits = torch.tensor(
        (value["ppo_frontier_logits"],),
        dtype=torch.float32,
    )
    theta_mu = torch.tensor((value["ppo_theta_mu"],), dtype=torch.float32)
    zeros = torch.zeros_like(logits)
    ones = torch.ones_like(logits)
    batch, candidates = logits.shape
    token = torch.zeros((batch, 1, 1), dtype=torch.float32)
    candidate_token = torch.zeros((batch, candidates, 1), dtype=torch.float32)
    return PolicyForwardOutput(
        frontier_logits=logits,
        theta_mu_sin_raw=zeros,
        theta_mu_cos_raw=ones,
        theta_kappa_raw=zeros,
        theta_mu=theta_mu,
        theta_kappa=ones,
        value=torch.zeros((batch,), dtype=torch.float32),
        global_map_tokens=token,
        local_map_tokens=token,
        pose_token=token,
        context_tokens=token,
        refined_frontier_tokens=candidate_token,
        action_hidden=candidate_token,
    )


def _empty_candidate_acceptance_probe(
    selector_input: Mapping[str, object],
) -> dict[str, object]:
    observation = _probe_observation(selector_input)
    selector_results: dict[str, dict[str, object]] = {}
    for method in BASELINE_METHODS:
        action: EnvAction | None = None
        try:
            action = select_baseline_action(
                method,
                observation,
                np.random.Generator(np.random.PCG64(20260715)),
            )
        except NoCandidateAction as exc:
            selector_results[method] = {
                "status": "failed_closed_no_candidate",
                "exception_type": type(exc).__name__,
                "action": None,
                "passed": True,
            }
        else:
            selector_results[method] = {
                "status": "fabricated_action",
                "exception_type": None,
                "action": _env_action_record(action),
                "passed": False,
            }
    ppo_action: EnvAction | None = None
    try:
        ppo_action = select_ppo_action(
            _probe_policy_output(selector_input),
            torch.tensor(
                (selector_input["candidate_valid_mask"],), dtype=torch.bool
            ),
        )
    except NoCandidateAction as exc:
        selector_results["ppo_policy"] = {
            "status": "failed_closed_no_candidate",
            "exception_type": type(exc).__name__,
            "action": None,
            "passed": True,
        }
    else:
        selector_results["ppo_policy"] = {
            "status": "fabricated_action",
            "exception_type": None,
            "action": _env_action_record(ppo_action),
            "passed": False,
        }
    terminal = resolve_terminal(
        severe_safety=False,
        coverage_rate=0.0,
        step_count=0,
        consecutive_no_gain_steps=0,
        has_candidate=False,
    )
    terminal_result = {
        "done": terminal.done,
        "reason": terminal.reason,
        "terminal": terminal.terminal,
        "bootstrap_value": terminal.bootstrap_value,
    }
    output = {
        "selector_results": selector_results,
        "environment_terminal_result": terminal_result,
        "fabricated_action": any(
            row["action"] is not None for row in selector_results.values()
        ),
    }
    passed = (
        set(selector_results) == set(ALL_METHODS)
        and all(row["passed"] is True for row in selector_results.values())
        and output["fabricated_action"] is False
        and terminal_result
        == {
            "done": True,
            "reason": "no_candidate_done",
            "terminal": True,
            "bootstrap_value": 0.0,
        }
    )
    return {
        "input_sha256": _hash_json(selector_input),
        **output,
        "output_sha256": _hash_json(output),
        "passed": passed,
    }


def _deterministic_baseline_acceptance_probe(
    selector_input: Mapping[str, object],
    *,
    seed: int,
) -> dict[str, object]:
    observation = _probe_observation(selector_input)
    methods: dict[str, dict[str, object]] = {}
    for method in BASELINE_METHODS[1:]:
        first = _env_action_record(
            select_baseline_action(
                method,
                observation,
                np.random.Generator(np.random.PCG64(seed)),
            )
        )
        second = _env_action_record(
            select_baseline_action(
                method,
                observation,
                np.random.Generator(np.random.PCG64(seed)),
            )
        )
        first_bytes = ArtifactStore.canonical_json_bytes(first)
        second_bytes = ArtifactStore.canonical_json_bytes(second)
        methods[method] = {
            "seed": seed,
            "first_action": first,
            "second_action": second,
            "first_action_sha256": hashlib.sha256(first_bytes).hexdigest(),
            "second_action_sha256": hashlib.sha256(second_bytes).hexdigest(),
            "byte_identical": first_bytes == second_bytes,
            "passed": first_bytes == second_bytes,
        }
    output = {"methods": methods}
    return {
        "input_sha256": _hash_json(selector_input),
        **output,
        "output_sha256": _hash_json(output),
        "passed": all(row["passed"] is True for row in methods.values()),
    }


def _random_baseline_acceptance_probe(
    selector_input: Mapping[str, object],
    *,
    seed: int,
    sequence_length: int,
) -> dict[str, object]:
    observation = _probe_observation(selector_input)
    first_rng = np.random.Generator(np.random.PCG64(seed))
    second_rng = np.random.Generator(np.random.PCG64(seed))
    first = [
        _env_action_record(
            select_baseline_action("random_valid_frontier", observation, first_rng)
        )
        for _ in range(sequence_length)
    ]
    second = [
        _env_action_record(
            select_baseline_action("random_valid_frontier", observation, second_rng)
        )
        for _ in range(sequence_length)
    ]
    first_bytes = ArtifactStore.canonical_json_bytes(first)
    second_bytes = ArtifactStore.canonical_json_bytes(second)
    output = {
        "rng_algorithm": "numpy.random.PCG64",
        "seed": seed,
        "sequence_length": sequence_length,
        "recorded_sequence": first,
        "first_sequence_sha256": hashlib.sha256(first_bytes).hexdigest(),
        "second_sequence_sha256": hashlib.sha256(second_bytes).hexdigest(),
        "byte_identical": first_bytes == second_bytes,
    }
    return {
        "input_sha256": _hash_json(selector_input),
        **output,
        "output_sha256": _hash_json(output),
        "passed": first_bytes == second_bytes and len(first) == sequence_length,
    }


def _lowest_index_tie_acceptance_probe(
    selector_input: Mapping[str, object],
) -> dict[str, object]:
    observation = _probe_observation(selector_input)
    expected_index = 1
    methods: dict[str, dict[str, object]] = {}
    for method in BASELINE_METHODS[1:]:
        action = select_baseline_action(
            method,
            observation,
            np.random.Generator(np.random.PCG64(20260715)),
        )
        methods[method] = {
            "selected_index": action.candidate_index,
            "action": _env_action_record(action),
            "passed": action.candidate_index == expected_index,
        }
    ppo_action = select_ppo_action(
        _probe_policy_output(selector_input),
        torch.tensor((selector_input["candidate_valid_mask"],), dtype=torch.bool),
    )
    methods["ppo_policy"] = {
        "selected_index": ppo_action.candidate_index,
        "action": _env_action_record(ppo_action),
        "passed": ppo_action.candidate_index == expected_index,
    }
    output = {
        "expected_lowest_valid_index": expected_index,
        "methods": methods,
    }
    return {
        "input_sha256": _hash_json(selector_input),
        **output,
        "output_sha256": _hash_json(output),
        "passed": all(row["passed"] is True for row in methods.values()),
    }


def _comparison_csv_acceptance_probe(
    summaries: Sequence[Mapping[str, object]],
    bootstrap_audits: Sequence[Mapping[str, object]],
    *,
    published: bytes,
    canonical_input: Mapping[str, object],
) -> dict[str, object]:
    first = comparison_table_csv(summaries, bootstrap_audits)
    second = comparison_table_csv(
        tuple(reversed(tuple(summaries))),
        tuple(reversed(tuple(bootstrap_audits))),
    )
    output = {
        "published_artifact": "baseline_comparison_table.csv",
        "first_output_sha256": hashlib.sha256(first).hexdigest(),
        "second_output_sha256": hashlib.sha256(second).hexdigest(),
        "published_sha256": hashlib.sha256(published).hexdigest(),
        "byte_identical": first == second == published,
    }
    return {
        "input_sha256": _hash_json(canonical_input),
        **output,
        "output_sha256": _hash_json(output),
        "passed": first == second == published,
    }


def _coverage_curves_acceptance_probe(
    episodes: Sequence[EpisodeResult],
    *,
    published: bytes,
    canonical_input: Mapping[str, object],
) -> dict[str, object]:
    first = coverage_curves_csv(episodes)
    second = coverage_curves_csv(tuple(reversed(tuple(episodes))))
    output = {
        "published_artifact": "baseline_coverage_curves.csv",
        "first_output_sha256": hashlib.sha256(first).hexdigest(),
        "second_output_sha256": hashlib.sha256(second).hexdigest(),
        "published_sha256": hashlib.sha256(published).hexdigest(),
        "byte_identical": first == second == published,
    }
    return {
        "input_sha256": _hash_json(canonical_input),
        **output,
        "output_sha256": _hash_json(output),
        "passed": first == second == published,
    }


def _progress_records(
    summaries: Sequence[EvaluationSummary],
    max_steps: int,
) -> tuple[dict[str, object], ...]:
    records: list[dict[str, object]] = []
    for method_index, summary in enumerate(summaries, start=1):
        episode_audits = summary.fairness_audit.get("episode_audits")
        if not isinstance(episode_audits, list) or len(episode_audits) != len(
            summary.episodes
        ):
            raise Stage5WorkflowError("Stage 5 progress episode audit drift")
        success_count = 0
        coverage_total = 0.0
        for episode_index, (episode, episode_audit) in enumerate(
            zip(summary.episodes, episode_audits, strict=True),
            start=1,
        ):
            if not isinstance(episode_audit, dict):
                raise Stage5WorkflowError("Stage 5 progress episode audit is invalid")
            steps_executed = episode_audit.get("steps_executed")
            if (
                type(steps_executed) is not int
                or not 0 <= steps_executed <= max_steps
            ):
                raise Stage5WorkflowError("Stage 5 progress step count is invalid")
            success_count += int(episode.success)
            coverage_total += episode.final_coverage
            records.append(
                {
                    "progress_type": "evaluation_episode_complete",
                    "timestamp_policy": "excluded_for_byte_identity/v1",
                    "scale_profile": episode.scale_profile,
                    "method": episode.method,
                    "method_id": method_index,
                    "method_count": len(summaries),
                    "episode_id": episode_index,
                    "eval_episode_count": len(summary.episodes),
                    "step_count": steps_executed,
                    "max_steps": max_steps,
                    "coverage_rate": episode.final_coverage,
                    "success_rate_so_far": success_count / episode_index,
                    "mean_final_coverage_so_far": coverage_total / episode_index,
                }
            )
    return tuple(records)


def _workflow_report(summary: Mapping[str, object]) -> bytes:
    acceptance = summary["acceptance"]
    assert isinstance(acceptance, dict)
    lines = (
        "# Stage 5 Machine Acceptance",
        "",
        f"Run ID: `{summary['run_id']}`",
        f"State: `{summary['state']}`",
        f"Claim boundary: `{summary['claim_boundary']}`",
        f"Method-episodes: `{summary['method_episode_count']}`",
        f"Acceptance passed: `{str(acceptance['passed']).lower()}`",
        "",
        "This is fair-baseline evaluator system evidence only. It does not establish PPO task advantage.",
        "",
    )
    return "\n".join(lines).encode("utf-8")


def _manifest_for_artifacts(
    artifacts: Mapping[str, bytes],
) -> dict[str, object]:
    return {
        "schema_version": "sha256_manifest/v1",
        "artifacts": [
            {
                "path": path,
                "sha256": hashlib.sha256(artifacts[path]).hexdigest(),
                "size_bytes": len(artifacts[path]),
            }
            for path in sorted(artifacts)
        ],
    }


def _read_episode_traces(payload: bytes) -> tuple[EpisodeResult, ...]:
    rows: list[EpisodeResult] = []
    for line_number, line in enumerate(payload.splitlines(), start=1):
        try:
            value = strict_json_object_from_bytes(line, f"Stage 5 trace line {line_number}")
        except Stage1WorkflowError as exc:
            raise Stage5WorkflowError("Stage 5 trace JSONL drift") from exc
        if tuple(value) != tuple(sorted(EPISODE_FIELDS)):
            raise Stage5WorkflowError("Stage 5 trace field order or schema drift")
        curve = value.get("coverage_curve")
        if not isinstance(curve, list):
            raise Stage5WorkflowError("Stage 5 trace coverage curve drift")
        value["coverage_curve"] = tuple(curve)
        try:
            rows.append(EpisodeResult(**value))
        except (TypeError, ValueError) as exc:
            raise Stage5WorkflowError("Stage 5 trace episode drift") from exc
    if len(rows) != 80:
        raise Stage5WorkflowError("Stage 5 trace method-episode count drift")
    return tuple(rows)


def _strict_canonical_json(
    snapshot: FrozenFileSnapshot,
    label: str,
) -> dict[str, object]:
    try:
        value = strict_json_object_from_bytes(snapshot.payload, label)
    except Stage1WorkflowError as exc:
        raise Stage5WorkflowError(f"{label} is invalid") from exc
    if snapshot.payload != ArtifactStore.canonical_json_bytes(value):
        raise Stage5WorkflowError(f"{label} is not canonical JSON")
    return value


def _canonical_jsonl_bytes(values: Sequence[Mapping[str, object]]) -> bytes:
    return b"".join(
        (
            json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
        for value in values
    )


def _relative_files(root: Path) -> frozenset[str]:
    if not root.is_dir():
        return frozenset()
    return frozenset(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    )


def _canonical_array_bytes(array: np.ndarray) -> bytes:
    values = np.ascontiguousarray(array)
    header = ArtifactStore.canonical_json_bytes(
        {"dtype": values.dtype.str, "shape": list(values.shape)}
    )
    return len(header).to_bytes(8, "big") + header + values.tobytes(order="C")


def _truth_map_sha256(truth: TruthMap) -> str:
    digest = hashlib.sha256(b"stage5_leakage_truth/v1\0")
    for array in (
        truth.height,
        truth.hard_obstacle,
        truth.slope_deg,
        truth.traversability,
    ):
        digest.update(_canonical_array_bytes(array))
    return digest.hexdigest()


def _policy_observation_sha256(observation: PolicyObservation) -> str:
    digest = hashlib.sha256(b"stage5_leakage_policy_observation/v1\0")
    digest.update(ArtifactStore.canonical_json_bytes(observation.schema_metadata()))
    for array in observation.array_fields():
        digest.update(_canonical_array_bytes(array))
    return digest.hexdigest()


def _frontier_action_set_sha256(action_set: FrontierActionSet) -> str:
    if not isinstance(action_set, FrontierActionSet):
        raise Stage5WorkflowError("Stage 5 leakage candidate set is unavailable")
    digest = hashlib.sha256(b"stage5_leakage_frontier_action_set/v1\0")
    digest.update(
        ArtifactStore.canonical_json_bytes(
            {"cells": [[cell.x, cell.y] for cell in action_set.cells]}
        )
    )
    for array in (
        action_set.frontier_features,
        action_set.candidate_mask,
        action_set.frontier_mask,
        action_set.observed_safe_frontier_mask,
        action_set.reachable_frontier_mask,
    ):
        if array is None:
            digest.update(b"none\0")
        else:
            digest.update(_canonical_array_bytes(array))
    return digest.hexdigest()


def _env_action_record(action: EnvAction) -> dict[str, object]:
    if not isinstance(action, EnvAction) or not math.isfinite(action.target_theta):
        raise Stage5WorkflowError("Stage 5 leakage action is invalid")
    return {
        "candidate_index": action.candidate_index,
        "target_theta_float64_hex": np.asarray(
            action.target_theta,
            dtype="<f8",
        ).tobytes().hex(),
    }


def _validated_leakage_audit(
    value: Mapping[str, object],
) -> dict[str, object]:
    replay = value.get("baseline_action_replay")
    if (
        value.get("schema_version") != "stage5_runtime_leakage_audit/v1"
        or value.get("mutation_source")
        != "paired_unobserved_truth_and_coverable_mask/v1"
        or value.get("observed_truth_cells_byte_equal") is not True
        or value.get("unobserved_truth_mutated") is not True
        or value.get("coverable_mask_mutated") is not True
        or value.get("all_observation_arrays_byte_equal") is not True
        or value.get("candidate_set_byte_equal") is not True
        or value.get("all_baseline_decisions_invariant") is not True
        or value.get("baseline_selection_parameters")
        != ["method", "observation", "rng"]
        or value.get("evaluator_selection_parameters")
        != ["self", "method", "observation", "rng"]
        or value.get("truth_or_coverable_parameter_present") is not False
        or value.get("selection_source_forbidden_token_present") is not False
        or value.get("python_internal_inaccessibility_claimed") is not False
        or not isinstance(replay, Mapping)
        or tuple(replay) != BASELINE_METHODS
        or any(
            not isinstance(replay[method], Mapping)
            or replay[method].get("byte_equal") is not True
            for method in BASELINE_METHODS
        )
    ):
        raise Stage5WorkflowError("Stage 5 runtime leakage audit failed")
    return dict(value)


def _hash_json(value: object) -> str:
    return hashlib.sha256(ArtifactStore.canonical_json_bytes(value)).hexdigest()


def _slash_path(path: Path) -> str:
    return str(path).replace("\\", "/")


def _git_text(repo: Path, arguments: Sequence[str]) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *arguments],
        check=False,
        capture_output=True,
    )
    if completed.returncode != 0:
        raise Stage5WorkflowError("Git identity command failed")
    return completed.stdout.decode("utf-8", errors="strict").strip()


__all__ = [
    "STAGE5_CHANGED_PATHS",
    "STAGE5_EVALUATOR_ARTIFACTS",
    "STAGE5_MACHINE_ARTIFACTS",
    "STAGE5_MANIFEST_BOUND_ARTIFACTS",
    "STAGE5_ROOT_ARTIFACTS",
    "FrozenStage4AuthorityHandle",
    "LoadedStage4Policy",
    "Stage5MachineVerificationHandle",
    "Stage5ManifestGraphHandle",
    "Stage5RuntimeBinding",
    "Stage5WorkflowError",
    "Stage5WorkflowResult",
    "_build_stage5_machine_payload",
    "_build_stage5_runtime_binding",
    "load_frozen_stage4_policy",
    "run_stage5_workflow",
    "stage5_environment_identity",
    "stage5_git_identity",
    "stage5_runtime_leakage_audit",
    "stage5_source_identity",
    "verify_frozen_stage4_authority",
    "verify_stage5_acceptance_probe_audit",
    "verify_stage5_machine_run",
    "verify_stage5_manifest_graph",
]
