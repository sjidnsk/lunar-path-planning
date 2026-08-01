"""Stage 6 Standard v1 workflow primitives。"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import stat
import subprocess
import threading
import uuid
from collections.abc import Mapping, Sequence
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field, fields
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal

from lunar_exploration_ppo.configs.schema import DEM_PATH, SLOPE_PATH
from lunar_exploration_ppo.utils.artifact_io import ArtifactStore
from lunar_exploration_ppo.utils.durable_jsonl import (
    DurableJsonl,
    DurableJsonlError,
    RunLease,
    RunLeaseError,
)
from lunar_exploration_ppo.utils.path_security import (
    PathSecurityError,
    is_link_or_reparse,
    lexical_absolute,
    path_identity,
    require_plain_path,
    secure_read_bytes,
)

if TYPE_CHECKING:
    from lunar_exploration_ppo.utils.stage6_input_pinning import Stage6InputPin
    from lunar_exploration_ppo.workflows.stage6_review_authorization import (
        Stage6ReviewAuthorizationHandle,
    )
    from lunar_exploration_ppo.workflows.stage6_terminal_recovery import (
        _Stage6EvidenceHandle,
    )


STAGE5_COMMIT: Final = "b635740ee021258ef31811ec87c60add839fc5f9"
STAGE5_TREE: Final = "edec4c4dbed7f91efa9856a9df899c594bbcf04c"
STAGE5_GATE_SHA256: Final = (
    "5fc93a7fb0d1d23c1c2ee99db14e15f6880c422d1ce85a83946542ebbe238bdf"
)
STAGE5_REVIEW_SHA256: Final = (
    "3c69d565f27bdabf64cf1ca4728be5da420ffa5057444f29af9b717bd2d9d9db"
)
STAGE5_MANIFEST_SHA256: Final = (
    "edee94cf2a0f39f89cf040d850af154e060b07c6a66d0bfc47386d5b05ee5672"
)
STAGE4_CHECKPOINT_SHA256: Final = (
    "d1d80e6478262a68d01208ddc1e8721768b56109e8dd009e862dd668cb39dc4c"
)
STAGE4_POLICY_STATE_SHA256: Final = (
    "51fecca55af92838f302951ae1272063cd945ef481579af5f6600469d9a7fe86"
)
STAGE6_STAGE_ID: Final = "ppo_highres_frontier_stage6_standard_training_eval/v1"
STAGE6_VERIFIED_REVIEW_AUTHORIZATION_FIELDS: Final = frozenset(
    {
        "schema_version",
        "stage_id",
        "formal_run_id",
        "single_seed_scope",
        "authorized",
        "base_commit",
        "authorization_file_sha256",
        "authorization_file_size_bytes",
        "review_identity_sha256",
        "reviewed_prospective_git_tree",
        "prospective_tree_sha256",
        "changed_path_set_sha256",
        "source_set_sha256",
        "config_sha256",
        "data_sha256",
        "environment_sha256",
        "frozen_diff_sha256",
        "frozen_diff_size_bytes",
        "spec_review_sha256",
        "quality_review_sha256",
    }
)
STAGE5_RUN_ID: Final = "s5-task6-fix-r1-20260714T223702Z"
ABSENT_AUTHORITY_SHA256: Final = (
    "3057252298292ae9a47b0879812581277456d1e9d9c0995178967da7c929236d"
)
CANONICAL_STAGE6_OUTPUT_ROOT: Final = Path("D:/xunce/out/ppo_frontier")
HISTORICAL_STAGE6_RUN_IDS: Final = frozenset(
    {
        "s6-standard-r1-20260716T035116Z",
        "s6-standard-r2-20260716T133254Z",
        "s6-standard-single-r1-20260718T062833Z",
    }
)
_FORMAL_STAGE6_RUN_ID_PATTERN: Final = re.compile(
    r"s6-standard-single-r1-(?P<timestamp>\d{8}T\d{6}Z)\Z"
)
_STANDALONE_PREFLIGHT_RUN_ID_PATTERN: Final = re.compile(
    r"s6-standard-single-preflight-r1-(?P<timestamp>\d{8}T\d{6}Z)\Z"
)
CANONICAL_STAGE5_GATE: Final = Path(
    "D:/xunce/out/ppo_frontier/"
    "s5-task6-fix-r1-20260714T223702Z/s5/gate.json"
)
CANONICAL_STAGE4_CHECKPOINT: Final = Path(
    "D:/xunce/out/ppo_frontier/"
    "s4-task5-r2-final-20260714T151959Z/s4/checkpoints/"
    "update-00000003/checkpoint.pt"
)
STAGE6_DATA_INPUT_PATHS: Final = (
    ("dem", Path(DEM_PATH)),
    ("slope_provenance", Path(SLOPE_PATH)),
)
_PREFLIGHT_TIMING_NAMES: Final = frozenset(
    {
        "catalog_build",
        "catalog_cache_reuse",
        "scenario_build",
        "coverable_env_init",
        "reset",
        "step",
        "cuda_policy_load",
        "cuda_forward",
        "cuda_backward",
        "spawn_startup",
        "spawn_reset",
        "spawn_inference",
        "spawn_step",
        "spawn_close",
    }
)

FINAL_EVALUATION_METHODS: Final = (
    "ppo_policy",
    "random_valid_frontier",
    "nearest_frontier",
    "max_potential_gain_frontier",
    "gain_over_cost_frontier",
)
STAGE6_MANIFEST_BOUND_ARTIFACTS: Final = (
    "config.json",
    "preterminal_acceptance.json",
    "summary.json",
    "routing.json",
    "report.md",
    "metrics.jsonl",
    "phase-state.jsonl",
    "job-state.jsonl",
    "training_metrics.jsonl",
    "validation_metrics.jsonl",
    "resource_audit.jsonl",
    "math_audit.jsonl",
    "checkpoint_audit.jsonl",
    "lineage_audit.json",
    "scenario_split_audit.json",
    "standard_training_report.md",
    "standard_eval_report.md",
    "standard_baseline_comparison.csv",
    "standard_coverage_curves.csv",
    "standard_checkpoint_manifest.json",
    "global-best.json",
    "fairness_audit.json",
    "leakage_audit.json",
    "failure_audit.json",
)
STAGE6_OPTIONAL_MANIFEST_BOUND_ARTIFACTS: Final = (
    "source-repair-amendment.json",
    "source-repair-continuation.json",
    "source-repair-supplement.json",
    "source-repair-closure.json",
    "source-repair-frontier-recovery.json",
    "source-repair-sensor-acceleration.json",
)
STAGE6_PLANNING_CHILD_SOURCE_REPAIR_MANIFEST_ARTIFACT: Final = (
    "planning-child-source-repair.json"
)
STAGE6_PLANNING_CHILD_SOURCE_REPAIR_CONTINUATION_MANIFEST_ARTIFACT: Final = (
    "planning-child-source-repair-continuation.json"
)
STAGE6_MANIFEST_PROFILE_OPTIONAL_ARTIFACTS: Final = (
    *STAGE6_OPTIONAL_MANIFEST_BOUND_ARTIFACTS,
    STAGE6_PLANNING_CHILD_SOURCE_REPAIR_MANIFEST_ARTIFACT,
    STAGE6_PLANNING_CHILD_SOURCE_REPAIR_CONTINUATION_MANIFEST_ARTIFACT,
)
STAGE6_COVERAGE_CACHE_MANIFEST_PATH: Final = Path(
    "D:/xunce/review/s6-coverable-cache-r1-20260722T144932/"
    "coverage-cache-manifest.json"
)
STAGE6_COVERAGE_CACHE_MANIFEST_SHA256: Final = (
    "675b923b64cee4cde90d5728550a7f939c41cfbc56e9c237268078b59641081c"
)
STAGE6_COVERAGE_CACHE_MANIFEST_SIZE_BYTES: Final = 519284
STAGE6_COVERAGE_CACHE_ROOT: Final = Path(
    "D:/xunce/cache/ppo_frontier/"
    "s6-standard-single-r1-20260718T220434Z/coverage-v1"
)
STAGE6_COVERAGE_CACHE_ENTRY_SET_SHA256: Final = (
    "c543a277154b41d14bd1a194d6c6e4f2b43075d5b93096c50d1952ee44c43558"
)
STAGE6_COVERAGE_CACHE_RUNTIME_MODE: Final = (
    "persistent_exact_manifest_read_only/v1"
)
STAGE6_COVERAGE_CACHE_FORMAL_AUDIT_PATH: Final = Path(
    "D:/xunce/review/s6-coverable-cache-r1-20260722T144932/formal-audit.json"
)
STAGE6_COVERAGE_CACHE_FORMAL_AUDIT_SHA256: Final = (
    "d7c12737da4c9a43353e02347b9c9d6abd527a78f7a3c6675e9164b6ae5b0068"
)
STAGE6_COVERAGE_CACHE_FORMAL_AUDIT_SIZE_BYTES: Final = 4416
STAGE6_PRODUCTION_SOURCE_PATHS: Final = (
    "configs/ppo_highres_frontier_stage6_v1.json",
    "docs/ppo-highres-frontier-stage6.md",
    "docs/superpowers/plans/2026-07-10-ppo-highres-frontier-map-exploration.md",
    "docs/superpowers/plans/2026-07-16-ppo-stage6-single-seed-pipeline.md",
    "docs/superpowers/plans/2026-07-22-ppo-stage6-coverable-mask-cache.md",
    "docs/superpowers/plans/2026-07-22-ppo-stage6-sensor-hotpath-acceleration.md",
    "docs/superpowers/plans/2026-07-23-ppo-stage6-planning-unknown-buffer-warm-start.md",
    "docs/superpowers/plans/2026-07-25-ppo-stage6-planning-child-source-repair.md",
    "docs/superpowers/plans/2026-07-26-ppo-stage6-planning-child-recovery-capability-consolidation.md",
    "docs/superpowers/specs/2026-07-09-ppo-highres-frontier-map-exploration-design.md",
    "docs/superpowers/specs/2026-07-16-ppo-stage6-single-seed-pipeline-design-addendum.md",
    "docs/superpowers/specs/2026-07-22-ppo-stage6-coverable-mask-cache-design-addendum.md",
    "docs/superpowers/specs/2026-07-22-ppo-stage6-sensor-hotpath-acceleration-design-addendum.md",
    "docs/superpowers/specs/2026-07-23-ppo-stage6-planning-unknown-buffer-design-addendum.md",
    "docs/superpowers/specs/2026-07-25-ppo-stage6-planning-child-source-repair-design.md",
    "docs/superpowers/specs/2026-07-26-ppo-stage6-planning-child-recovery-capability-consolidation-design.md",
    "scripts/benchmark_ppo_stage6_sensor_hotpath.py",
    "scripts/prewarm_ppo_stage6_coverage_cache.py",
    "scripts/run_ppo_stage6_standard.py",
    "scripts/create_ppo_stage6_source_repair_amendment.py",
    "scripts/create_ppo_stage6_planning_warm_start.py",
    "scripts/create_ppo_stage6_planning_child_source_repair.py",
    "scripts/create_ppo_stage6_planning_child_source_repair_continuation.py",
    "src/lunar_exploration_ppo/configs/stage1.py",
    "src/lunar_exploration_ppo/configs/stage6.py",
    "src/lunar_exploration_ppo/env/action_execution.py",
    "src/lunar_exploration_ppo/env/coverage_cache.py",
    "src/lunar_exploration_ppo/env/env.py",
    "src/lunar_exploration_ppo/env/frontier.py",
    "src/lunar_exploration_ppo/env/frontier_oracle.py",
    "src/lunar_exploration_ppo/env/map_state.py",
    "src/lunar_exploration_ppo/env/scenario_catalog.py",
    "src/lunar_exploration_ppo/env/sensor_model.py",
    "src/lunar_exploration_ppo/env/standard_training.py",
    "src/lunar_exploration_ppo/eval/standard.py",
    "src/lunar_exploration_ppo/integrations/path_planner_adapter.py",
    "src/lunar_exploration_ppo/ppo/checkpoint.py",
    "src/lunar_exploration_ppo/ppo/collector.py",
    "src/lunar_exploration_ppo/ppo/checkpoint_retention.py",
    "src/lunar_exploration_ppo/ppo/standard_training.py",
    "src/lunar_exploration_ppo/ppo/trainer.py",
    "src/lunar_exploration_ppo/utils/artifact_io.py",
    "src/lunar_exploration_ppo/utils/stage6_input_pinning.py",
    "src/lunar_exploration_ppo/utils/resource_lifecycle.py",
    "src/lunar_exploration_ppo/utils/resources.py",
    "src/lunar_exploration_ppo/utils/durable_jsonl.py",
    "src/lunar_exploration_ppo/utils/path_security.py",
    "src/lunar_exploration_ppo/workflows/stage6_review_authorization.py",
    "src/lunar_exploration_ppo/workflows/stage6_source_repair.py",
    "src/lunar_exploration_ppo/workflows/stage6_terminal_recovery.py",
    "src/lunar_exploration_ppo/workflows/stage6.py",
    "src/lunar_exploration_ppo/workflows/stage6_planning_warm_start.py",
    "src/lunar_exploration_ppo/workflows/stage6_planning_child_source_repair.py",
    "src/lunar_exploration_ppo/workflows/stage6_planning_child_source_repair_continuation.py",
    "src/lunar_exploration_ppo/workflows/stage6_planning_child_recovery.py",
    "src/lunar_exploration_ppo/workflows/stage6_coverage_cache.py",
)
STAGE6_TEST_SOURCE_PATHS: Final = (
    "tests/ppo_highres_frontier/test_foundation.py",
    "tests/ppo_highres_frontier/test_stage1_behavior_regression.py",
    "tests/ppo_highres_frontier/test_stage2_catalog.py",
    "tests/ppo_highres_frontier/test_stage2_frontier.py",
    "tests/ppo_highres_frontier/test_stage4_checkpoint.py",
    "tests/ppo_highres_frontier/test_stage4_collector.py",
    "tests/ppo_highres_frontier/test_stage4_trainer.py",
    "tests/ppo_highres_frontier/test_stage6_execution.py",
    "tests/ppo_highres_frontier/test_stage6_coverage_cache.py",
    "tests/ppo_highres_frontier/test_stage6_coverage_cache_workflow.py",
    "tests/ppo_highres_frontier/test_stage6_durable_jsonl.py",
    "tests/ppo_highres_frontier/test_stage6_input_pinning.py",
    "tests/ppo_highres_frontier/test_stage6_machine_preflight.py",
    "tests/ppo_highres_frontier/test_stage6_planning_unknown_buffer.py",
    "tests/ppo_highres_frontier/test_stage6_planning_warm_start.py",
    "tests/ppo_highres_frontier/test_stage6_planning_child_source_repair.py",
    "tests/ppo_highres_frontier/test_stage6_planning_child_source_repair_continuation.py",
    "tests/ppo_highres_frontier/test_stage6_planning_child_source_repair_cli.py",
    "tests/ppo_highres_frontier/test_stage6_reset_local_safety_scan.py",
    "tests/ppo_highres_frontier/test_stage6_resource_lifecycle.py",
    "tests/ppo_highres_frontier/test_stage6_resources.py",
    "tests/ppo_highres_frontier/test_stage6_review_authorization.py",
    "tests/ppo_highres_frontier/test_stage6_sensor_acceleration.py",
    "tests/ppo_highres_frontier/test_stage6_source_repair.py",
    "tests/ppo_highres_frontier/test_stage6_standard_config.py",
    "tests/ppo_highres_frontier/test_stage6_standard_env.py",
    "tests/ppo_highres_frontier/test_stage6_standard_eval.py",
    "tests/ppo_highres_frontier/test_stage6_terminal_recovery.py",
    "tests/ppo_highres_frontier/test_stage6_training.py",
    "tests/ppo_highres_frontier/test_stage6_workflow.py",
    "tests/ppo_highres_frontier/test_stage6_planning_child_recovery.py",
)
STAGE6_SOURCE_PATHS: Final = (
    *STAGE6_PRODUCTION_SOURCE_PATHS,
    *STAGE6_TEST_SOURCE_PATHS,
)

_GATE_KEYS: Final = frozenset(
    {
        "authorized_next_stage",
        "bindings",
        "commit_sha256",
        "history",
        "run_id",
        "schema_version",
        "state",
    }
)
_BINDING_KEYS: Final = frozenset(
    {
        "acceptance_probe_sha256",
        "approval_challenge",
        "approval_controller",
        "approval_sha256",
        "authorized_next_stage",
        "claim_boundary",
        "commit_parent",
        "commit_sha256",
        "commit_subject",
        "commit_tree",
        "environment_sha256",
        "fairness_audit_sha256",
        "goal_id",
        "latest_checkpoint_policy_state_sha256",
        "latest_checkpoint_sha256",
        "manifest_sha256",
        "performance_advantage_established",
        "repository_config_sha256",
        "review_recorded_at_utc",
        "review_recorder",
        "review_report",
        "review_sha256",
        "reviewed_base_commit",
        "reviewed_path_set_sha256",
        "reviewed_paths",
        "reviewed_prospective_git_tree",
        "routing_sha256",
        "run_id",
        "runtime_config_sha256",
        "schema_version",
        "source_set_sha256",
        "stage4_gate_sha256",
        "stage_id",
        "summary_sha256",
        "thread_id",
    }
)
_STAGE5_ROOT_MEMBERS: Final = frozenset(
    {
        "acceptance_probe_audit.json",
        "approval.json",
        "baseline_comparison_table.csv",
        "baseline_coverage_curves.csv",
        "baseline_episode_traces.jsonl",
        "baseline_eval_report.md",
        "baseline_fairness_audit.json",
        "baseline_metrics.json",
        "bootstrap_ci_audit.json",
        "config.json",
        "execution_identity_audit.json",
        "gate.json",
        "manifest.json",
        "metrics.jsonl",
        "phase-state.jsonl",
        "report.md",
        "review.json",
        "routing.json",
        "stage4_authority_audit.json",
        "summary.json",
    }
)
_HISTORY_STATES: Final = (
    "machine_passed",
    "awaiting_independent_review",
    "awaiting_human_approval",
    "approved",
    "next_stage",
)


class Stage6WorkflowError(RuntimeError):
    """Stage 6 authority、训练、评估或 artifact 验证失败。"""


_STAGE6_EXECUTION_CAPABILITY_ISSUER: Final = object()
_STAGE6_EXECUTION_CAPABILITY_LOCK: Final = threading.RLock()
_STAGE6_EXECUTION_CAPABILITY_CONDITION: Final = threading.Condition(
    _STAGE6_EXECUTION_CAPABILITY_LOCK
)
_STAGE6_STAGE5_AUTHORITY_PROJECTION_KEYS: Final = (
    "commit_sha256",
    "commit_tree",
    "gate_path",
    "gate_sha256",
    "review_sha256",
    "manifest_sha256",
    "checkpoint_sha256",
    "policy_state_sha256",
    "authorized_stage",
    "performance_advantage_established",
)
_STAGE6_STAGE5_AUTHORITY_AUDIT_KEYS: Final = frozenset(
    {
        *_STAGE6_STAGE5_AUTHORITY_PROJECTION_KEYS,
        "schema_version",
        "verified",
        "run_id",
        "history_states",
    }
)


class _Stage6ExecutionCapability:
    """Process-local authority for one protected Stage 6 execution scope."""

    __slots__ = ("_issuer", "_identity")

    def __new__(
        cls,
        issuer: object,
        identity: object,
    ) -> _Stage6ExecutionCapability:
        if issuer is not _STAGE6_EXECUTION_CAPABILITY_ISSUER:
            raise TypeError("Stage 6 execution capability issuer is invalid")
        instance = super().__new__(cls)
        instance._issuer = issuer
        instance._identity = identity
        return instance

    def __copy__(self) -> object:
        raise TypeError("Stage 6 execution capability cannot be copied")

    def __deepcopy__(self, memo: object) -> object:
        del memo
        raise TypeError("Stage 6 execution capability cannot be copied")

    def __reduce__(self) -> object:
        raise TypeError("Stage 6 execution capability cannot be serialized")

    def __reduce_ex__(self, protocol: int) -> object:
        del protocol
        raise TypeError("Stage 6 execution capability cannot be serialized")


@dataclass(slots=True)
class _Stage6ExecutionCapabilityState:
    identity: object
    formal_run_id: str
    run_root: Path
    stage_root: Path
    repo_root: Path
    config_path: Path
    base_config_payload: bytes
    config_payload: bytes
    execution_identity_payload: bytes
    review_authorization_payload: bytes
    stage5_authority_payload: bytes
    expected_input_requests: tuple[tuple[str, Path], ...]
    pinned_record_snapshot: tuple[tuple[object, ...], ...]
    review_authorization_handle: object
    input_pin: object
    stage5_authority: object
    run_lease: RunLease
    active: bool = True
    revoking: bool = False
    active_operations: int = 0
    operation_threads: dict[int, int] = field(default_factory=dict)


_STAGE6_EXECUTION_CAPABILITY_REGISTRY: dict[
    _Stage6ExecutionCapability,
    _Stage6ExecutionCapabilityState,
] = {}


def _capability_json_payload(value: object, *, label: str) -> bytes:
    if not isinstance(value, Mapping):
        raise Stage6WorkflowError(f"Stage 6 execution capability {label} is invalid")
    try:
        payload = ArtifactStore.canonical_json_bytes(dict(value))
        decoded = json.loads(payload.decode("utf-8"))
    except (TypeError, ValueError, UnicodeError, json.JSONDecodeError) as exc:
        raise Stage6WorkflowError(
            f"Stage 6 execution capability {label} is not canonical"
        ) from exc
    if type(decoded) is not dict or decoded != dict(value):
        raise Stage6WorkflowError(
            f"Stage 6 execution capability {label} is not a plain JSON object"
        )
    return payload


def _capability_config_payload(config: object) -> bytes:
    from lunar_exploration_ppo.configs.stage6 import Stage6Config

    if type(config) is not Stage6Config:
        raise Stage6WorkflowError(
            "Stage 6 execution capability requires the exact frozen config type"
        )
    try:
        return ArtifactStore.canonical_json_bytes(config.model_dump(mode="json"))
    except (AttributeError, TypeError, ValueError) as exc:
        raise Stage6WorkflowError(
            "Stage 6 execution capability config is not canonical"
        ) from exc


def _normalized_capability_input_requests(
    requests: Sequence[tuple[str, str | Path]],
) -> tuple[tuple[tuple[str, ...], Path], ...]:
    if not isinstance(requests, Sequence) or isinstance(requests, (str, bytes)):
        raise Stage6WorkflowError(
            "Stage 6 execution capability input request set is invalid"
        )
    grouped: list[tuple[list[str], Path]] = []
    indexes: dict[str, int] = {}
    labels: set[str] = set()
    for item in requests:
        if not isinstance(item, tuple) or len(item) != 2:
            raise Stage6WorkflowError(
                "Stage 6 execution capability input request row is invalid"
            )
        label, path_value = item
        if (
            not isinstance(label, str)
            or not label
            or label != label.strip()
            or label in labels
        ):
            raise Stage6WorkflowError(
                "Stage 6 execution capability input request label drifted"
            )
        labels.add(label)
        path = lexical_absolute(path_value)
        key = os.path.normcase(os.fspath(path))
        index = indexes.get(key)
        if index is None:
            indexes[key] = len(grouped)
            grouped.append(([label], path))
            continue
        existing_labels, existing_path = grouped[index]
        if existing_path != path:
            raise Stage6WorkflowError(
                "Stage 6 execution capability input request alias drifted"
            )
        existing_labels.append(label)
    if not grouped:
        raise Stage6WorkflowError(
            "Stage 6 execution capability input request set is empty"
        )
    return tuple((tuple(item_labels), path) for item_labels, path in grouped)


def _stage6_pin_record_snapshot(
    input_pin: object,
    *,
    expected_requests: Sequence[tuple[str, str | Path]],
) -> tuple[tuple[object, ...], ...]:
    from lunar_exploration_ppo.utils.stage6_input_pinning import Stage6InputPin

    if type(input_pin) is not Stage6InputPin:
        raise Stage6WorkflowError(
            "Stage 6 execution capability requires an exact Stage6InputPin"
        )
    expected = _normalized_capability_input_requests(expected_requests)
    try:
        input_pin.require_active_issuance(
            "Stage 6 execution capability input pin issuance"
        )
        records = input_pin.records
        paths = input_pin.paths
    except (AttributeError, OSError, TypeError, ValueError) as exc:
        raise Stage6WorkflowError(
            "Stage 6 execution capability input pin records are unavailable"
        ) from exc
    if type(records) is not tuple or type(paths) is not tuple or not records:
        raise Stage6WorkflowError(
            "Stage 6 execution capability input pin record set is empty or invalid"
        )
    snapshot: list[tuple[object, ...]] = []
    actual_groups: list[tuple[tuple[str, ...], Path]] = []
    path_keys: set[str] = set()
    for record in records:
        try:
            record_labels = record.labels
            record_path = record.path
            digest = record.sha256
            size_bytes = record.size_bytes
            identity = tuple(record.identity)
        except (AttributeError, TypeError, ValueError) as exc:
            raise Stage6WorkflowError(
                "Stage 6 execution capability input pin record drifted"
            ) from exc
        if (
            type(record_labels) is not tuple
            or not record_labels
            or any(type(label) is not str or not label for label in record_labels)
            or not isinstance(record_path, Path)
            or lexical_absolute(record_path) != record_path
            or not isinstance(digest, str)
            or len(digest) != 64
            or digest != digest.lower()
            or any(character not in "0123456789abcdef" for character in digest)
            or type(size_bytes) is not int
            or size_bytes < 0
            or not identity
            or any(type(value) is not int for value in identity)
        ):
            raise Stage6WorkflowError(
                "Stage 6 execution capability input pin record drifted"
            )
        key = os.path.normcase(os.fspath(record_path))
        if key in path_keys:
            raise Stage6WorkflowError(
                "Stage 6 execution capability input pin path is duplicated"
            )
        path_keys.add(key)
        actual_groups.append((record_labels, record_path))
        snapshot.append((record_labels, record_path, digest, size_bytes, identity))
    if tuple(paths) != tuple(path for _labels, path in actual_groups):
        raise Stage6WorkflowError(
            "Stage 6 execution capability input pin paths drifted"
        )
    if tuple(actual_groups) != expected:
        raise Stage6WorkflowError(
            "Stage 6 execution capability input pin exact path set drifted"
        )
    return tuple(snapshot)


def _validate_stage6_stage5_authority_binding(
    *,
    config: object,
    stage5_identity: object,
) -> None:
    if not isinstance(stage5_identity, Mapping) or set(stage5_identity) != set(
        _STAGE6_STAGE5_AUTHORITY_AUDIT_KEYS
    ):
        raise Stage6WorkflowError(
            "Stage 6 execution capability Stage 5 authority binding schema drifted"
        )
    try:
        expected_projection = config.stage5_authority.model_dump(mode="json")
    except (AttributeError, TypeError, ValueError) as exc:
        raise Stage6WorkflowError(
            "Stage 6 execution capability Stage 5 config binding is invalid"
        ) from exc
    actual_projection = {
        key: stage5_identity.get(key)
        for key in _STAGE6_STAGE5_AUTHORITY_PROJECTION_KEYS
    }
    if actual_projection != expected_projection:
        raise Stage6WorkflowError(
            "Stage 6 execution capability Stage 5 authority binding drifted"
        )
    if (
        stage5_identity.get("schema_version")
        != "stage6_stage5_authority_audit/v1"
        or stage5_identity.get("verified") is not True
        or stage5_identity.get("run_id") != STAGE5_RUN_ID
        or type(stage5_identity.get("history_states")) is not list
        or stage5_identity.get("history_states") != list(_HISTORY_STATES)
    ):
        raise Stage6WorkflowError(
            "Stage 6 execution capability Stage 5 authority audit binding drifted"
        )


def _validate_stage6_config_pin_binding(
    *,
    config_path: Path,
    base_config_payload: bytes,
    config_identity_payload: bytes,
    execution_identity: Mapping[str, object],
    pin_snapshot: Sequence[tuple[object, ...]],
) -> None:
    base_config_digest = hashlib.sha256(base_config_payload).hexdigest()
    config_identity_digest = hashlib.sha256(
        config_identity_payload
    ).hexdigest()
    matching = [
        row
        for row in pin_snapshot
        if isinstance(row[0], tuple) and "config:canonical" in row[0]
    ]
    if len(matching) != 1:
        raise Stage6WorkflowError(
            "Stage 6 execution capability input pin config binding is missing"
        )
    _labels, pinned_path, pinned_digest, pinned_size, _identity = matching[0]
    if (
        pinned_path != config_path
        or pinned_digest != base_config_digest
        or pinned_size != len(base_config_payload)
        or execution_identity.get("config_sha256")
        != config_identity_digest
    ):
        raise Stage6WorkflowError(
            "Stage 6 execution capability input pin config binding drifted"
        )


def _validate_stage6_input_pin_bindings(
    *,
    repo_root: Path,
    config_path: Path,
    base_config_payload: bytes,
    config_identity_payload: bytes,
    execution_identity: Mapping[str, object],
    stage5_authority: object,
    review_authorization_handle: object,
    verified_review_authorization: Mapping[str, object],
    pin_snapshot: Sequence[tuple[object, ...]],
) -> None:
    _validate_stage6_config_pin_binding(
        config_path=config_path,
        base_config_payload=base_config_payload,
        config_identity_payload=config_identity_payload,
        execution_identity=execution_identity,
        pin_snapshot=pin_snapshot,
    )

    by_label: dict[str, tuple[object, ...]] = {}
    for row in pin_snapshot:
        labels = row[0]
        if type(labels) is not tuple:
            raise Stage6WorkflowError(
                "Stage 6 execution capability input pin binding labels drifted"
            )
        for label in labels:
            if label in by_label:
                raise Stage6WorkflowError(
                    "Stage 6 execution capability input pin binding label duplicated"
                )
            by_label[label] = row

    def require_binding(
        label: str,
        *,
        path: Path,
        sha256: object,
        size_bytes: object,
    ) -> None:
        row = by_label.get(label)
        if (
            row is None
            or row[1] != path
            or row[2] != sha256
            or row[3] != size_bytes
        ):
            raise Stage6WorkflowError(
                f"Stage 6 execution capability input pin binding drifted: {label}"
            )

    def valid_digest(value: object) -> bool:
        return (
            type(value) is str
            and len(value) == 64
            and value == value.lower()
            and all(character in "0123456789abcdef" for character in value)
        )

    source_identity = execution_identity.get("source_identity")
    if (
        not isinstance(source_identity, Mapping)
        or set(source_identity) != {"schema_version", "source_set_sha256", "paths"}
        or source_identity.get("schema_version") != "stage6_current_source_set/v1"
        or source_identity.get("source_set_sha256")
        != execution_identity.get("source_set_sha256")
        or not valid_digest(source_identity.get("source_set_sha256"))
        or type(source_identity.get("paths")) is not list
    ):
        raise Stage6WorkflowError(
            "Stage 6 execution capability input pin source binding schema drifted"
        )
    source_rows = source_identity["paths"]
    if len(source_rows) != len(STAGE6_SOURCE_PATHS):
        raise Stage6WorkflowError(
            "Stage 6 execution capability input pin source binding set drifted"
        )
    for expected_relative, row in zip(
        STAGE6_SOURCE_PATHS,
        source_rows,
        strict=True,
    ):
        if (
            type(row) is not dict
            or set(row) != {"path", "sha256", "size_bytes"}
            or row.get("path") != expected_relative
            or not valid_digest(row.get("sha256"))
            or type(row.get("size_bytes")) is not int
            or row["size_bytes"] < 0
        ):
            raise Stage6WorkflowError(
                "Stage 6 execution capability input pin source binding row drifted"
            )
        source_path = lexical_absolute(repo_root / expected_relative)
        try:
            source_path.relative_to(repo_root)
        except ValueError as exc:
            raise Stage6WorkflowError(
                "Stage 6 execution capability input pin source binding escaped repo"
            ) from exc
        require_binding(
            f"source:{expected_relative}",
            path=source_path,
            sha256=row["sha256"],
            size_bytes=row["size_bytes"],
        )

    require_binding(
        "coverage-cache:manifest",
        path=lexical_absolute(STAGE6_COVERAGE_CACHE_MANIFEST_PATH),
        sha256=STAGE6_COVERAGE_CACHE_MANIFEST_SHA256,
        size_bytes=STAGE6_COVERAGE_CACHE_MANIFEST_SIZE_BYTES,
    )

    data_identity = execution_identity.get("data_identity")
    if (
        not isinstance(data_identity, Mapping)
        or set(data_identity) != {"schema_version", "files", "catalog_sha256"}
        or data_identity.get("schema_version")
        != "stage6_data_catalog_identity/v1"
        or not valid_digest(data_identity.get("catalog_sha256"))
        or data_identity.get("catalog_sha256")
        != execution_identity.get("catalog_sha256")
        or type(data_identity.get("files")) is not list
        or _canonical_sha256(data_identity) != execution_identity.get("data_sha256")
    ):
        raise Stage6WorkflowError(
            "Stage 6 execution capability input pin data binding schema drifted"
        )
    data_rows = data_identity["files"]
    if len(data_rows) != len(STAGE6_DATA_INPUT_PATHS):
        raise Stage6WorkflowError(
            "Stage 6 execution capability input pin data binding set drifted"
        )
    for (expected_label, expected_path_value), row in zip(
        STAGE6_DATA_INPUT_PATHS,
        data_rows,
        strict=True,
    ):
        if (
            type(row) is not dict
            or set(row) != {"label", "path", "sha256", "size_bytes"}
            or row.get("label") != expected_label
            or type(row.get("path")) is not str
            or not valid_digest(row.get("sha256"))
            or type(row.get("size_bytes")) is not int
            or row["size_bytes"] < 0
        ):
            raise Stage6WorkflowError(
                "Stage 6 execution capability input pin data binding row drifted"
            )
        expected_path = lexical_absolute(expected_path_value)
        if lexical_absolute(row["path"]) != expected_path:
            raise Stage6WorkflowError(
                "Stage 6 execution capability input pin data binding path drifted"
            )
        require_binding(
            f"data:{expected_label}",
            path=expected_path,
            sha256=row["sha256"],
            size_bytes=row["size_bytes"],
        )

    snapshots = getattr(stage5_authority, "snapshots", None)
    expected_stage5_names = (
        "gate",
        "approval",
        "review",
        "manifest",
        "checkpoint",
        "checkpoint_manifest",
    )
    if (
        type(snapshots) is not tuple
        or tuple(name for name, _snapshot in snapshots) != expected_stage5_names
    ):
        raise Stage6WorkflowError(
            "Stage 6 execution capability input pin Stage 5 binding set drifted"
        )
    stage5_identity = stage5_authority.identity
    stage5_identity_fields = {
        "gate": "gate_sha256",
        "review": "review_sha256",
        "manifest": "manifest_sha256",
        "checkpoint": "checkpoint_sha256",
    }
    for name, snapshot in snapshots:
        try:
            path = snapshot.path
            digest = snapshot.sha256
            size_bytes = snapshot.size_bytes
        except AttributeError as exc:
            raise Stage6WorkflowError(
                "Stage 6 execution capability input pin Stage 5 binding row drifted"
            ) from exc
        if (
            not isinstance(path, Path)
            or lexical_absolute(path) != path
            or not valid_digest(digest)
            or type(size_bytes) is not int
            or size_bytes < 0
        ):
            raise Stage6WorkflowError(
                "Stage 6 execution capability input pin Stage 5 binding row drifted"
            )
        identity_field = stage5_identity_fields.get(name)
        if identity_field is not None and stage5_identity.get(identity_field) != digest:
            raise Stage6WorkflowError(
                "Stage 6 execution capability input pin Stage 5 identity binding drifted"
            )
        if name == "gate" and lexical_absolute(stage5_identity["gate_path"]) != path:
            raise Stage6WorkflowError(
                "Stage 6 execution capability input pin Stage 5 gate path binding drifted"
            )
        require_binding(
            f"stage5-authority:{name}",
            path=path,
            sha256=digest,
            size_bytes=size_bytes,
        )

    evidence_records = getattr(review_authorization_handle, "evidence_records", None)
    evidence_paths = getattr(review_authorization_handle, "evidence_paths", None)
    if (
        type(evidence_records) is not tuple
        or len(evidence_records) != 5
        or type(evidence_paths) is not tuple
        or tuple(record.path for record in evidence_records) != evidence_paths
    ):
        raise Stage6WorkflowError(
            "Stage 6 execution capability input pin review binding set drifted"
        )
    review_expected_hashes = (
        verified_review_authorization.get("authorization_file_sha256"),
        verified_review_authorization.get("review_identity_sha256"),
        verified_review_authorization.get("frozen_diff_sha256"),
        verified_review_authorization.get("spec_review_sha256"),
        verified_review_authorization.get("quality_review_sha256"),
    )
    for index, (record, expected_digest) in enumerate(
        zip(evidence_records, review_expected_hashes, strict=True),
        start=1,
    ):
        try:
            path = record.path
            digest = record.sha256
            size_bytes = record.size_bytes
        except AttributeError as exc:
            raise Stage6WorkflowError(
                "Stage 6 execution capability input pin review binding row drifted"
            ) from exc
        if (
            not isinstance(path, Path)
            or lexical_absolute(path) != path
            or not valid_digest(digest)
            or digest != expected_digest
            or type(size_bytes) is not int
            or size_bytes < 0
        ):
            raise Stage6WorkflowError(
                "Stage 6 execution capability input pin review identity binding drifted"
            )
        if index == 1 and size_bytes != verified_review_authorization.get(
            "authorization_file_size_bytes"
        ):
            raise Stage6WorkflowError(
                "Stage 6 execution capability input pin review size binding drifted"
            )
        if index == 3 and size_bytes != verified_review_authorization.get(
            "frozen_diff_size_bytes"
        ):
            raise Stage6WorkflowError(
                "Stage 6 execution capability input pin review size binding drifted"
            )
        require_binding(
            f"review-authorization:{index}:{path.name}",
            path=path,
            sha256=digest,
            size_bytes=size_bytes,
        )


def _stage6_capability_state(
    execution_capability: object,
) -> _Stage6ExecutionCapabilityState:
    if type(execution_capability) is not _Stage6ExecutionCapability:
        raise Stage6WorkflowError("Stage 6 execution capability is missing or forged")
    if execution_capability._issuer is not _STAGE6_EXECUTION_CAPABILITY_ISSUER:
        raise Stage6WorkflowError("Stage 6 execution capability issuer drifted")
    with _STAGE6_EXECUTION_CAPABILITY_LOCK:
        state = _STAGE6_EXECUTION_CAPABILITY_REGISTRY.get(execution_capability)
        if (
            type(state) is not _Stage6ExecutionCapabilityState
            or state.active is not True
            or state.identity is not execution_capability._identity
        ):
            raise Stage6WorkflowError(
                "Stage 6 execution capability is revoked or inactive"
            )
        return state


def _require_stage6_execution_capability(
    execution_capability: object,
    *,
    label: str,
    rehash_inputs: bool = False,
    formal_run_id: str | None = None,
    run_root: str | Path | None = None,
    stage_root: str | Path | None = None,
    repo_root: str | Path | None = None,
    config_path: str | Path | None = None,
    config: object | None = None,
    execution_identity: Mapping[str, object] | None = None,
    stage5_authority: Mapping[str, object] | None = None,
    verified_review_authorization: Mapping[str, object] | None = None,
    run_lease: RunLease | None = None,
    review_authorization_handle: object | None = None,
    input_pin: object | None = None,
    stage5_authority_handle: object | None = None,
) -> None:
    if not isinstance(label, str) or not label.strip():
        raise Stage6WorkflowError(
            "Stage 6 execution capability validation label is invalid"
        )
    if type(rehash_inputs) is not bool:
        raise Stage6WorkflowError(
            "Stage 6 execution capability input rehash mode is invalid"
        )
    state = _stage6_capability_state(execution_capability)
    try:
        if formal_run_id is not None and formal_run_id != state.formal_run_id:
            raise Stage6WorkflowError(
                "Stage 6 execution capability formal run id drifted"
            )
        for value, expected, path_label in (
            (run_root, state.run_root, "run root"),
            (stage_root, state.stage_root, "stage root"),
            (repo_root, state.repo_root, "repo root"),
            (config_path, state.config_path, "config path"),
        ):
            if value is not None and lexical_absolute(value) != expected:
                raise Stage6WorkflowError(
                    f"Stage 6 execution capability {path_label} drifted"
                )
        if (
            config is not None
            and _capability_config_payload(config)
            != state.base_config_payload
        ):
            raise Stage6WorkflowError(
                "Stage 6 execution capability config identity drifted"
            )
        if (
            execution_identity is not None
            and _capability_json_payload(
                execution_identity,
                label="execution identity",
            )
            != state.execution_identity_payload
        ):
            raise Stage6WorkflowError(
                "Stage 6 execution capability execution identity drifted"
            )
        if (
            stage5_authority is not None
            and _capability_json_payload(
                stage5_authority,
                label="Stage 5 authority",
            )
            != state.stage5_authority_payload
        ):
            raise Stage6WorkflowError(
                "Stage 6 execution capability Stage 5 authority drifted"
            )
        if (
            verified_review_authorization is not None
            and _capability_json_payload(
                verified_review_authorization,
                label="review authorization",
            )
            != state.review_authorization_payload
        ):
            raise Stage6WorkflowError(
                "Stage 6 execution capability review authorization drifted"
            )
        if run_lease is not None and run_lease is not state.run_lease:
            raise Stage6WorkflowError(
                "Stage 6 execution capability RunLease identity drifted"
            )
        if (
            review_authorization_handle is not None
            and review_authorization_handle is not state.review_authorization_handle
        ):
            raise Stage6WorkflowError(
                "Stage 6 execution capability review handle identity drifted"
            )
        if input_pin is not None and input_pin is not state.input_pin:
            raise Stage6WorkflowError(
                "Stage 6 execution capability InputPin identity drifted"
            )
        if (
            stage5_authority_handle is not None
            and stage5_authority_handle is not state.stage5_authority
        ):
            raise Stage6WorkflowError(
                "Stage 6 execution capability Stage 5 handle identity drifted"
            )
        state.run_lease.require_current()
        state.input_pin.require_current(label, rehash=rehash_inputs)
        if (
            _capability_json_payload(
                state.review_authorization_handle.canonical_record(),
                label="live review authorization",
            )
            != state.review_authorization_payload
        ):
            raise Stage6WorkflowError(
                "Stage 6 execution capability live review authorization drifted"
            )
        if (
            _capability_json_payload(
                state.stage5_authority.identity,
                label="live Stage 5 authority",
            )
            != state.stage5_authority_payload
        ):
            raise Stage6WorkflowError(
                "Stage 6 execution capability live Stage 5 authority drifted"
            )
        current_snapshot = _stage6_pin_record_snapshot(
            state.input_pin,
            expected_requests=state.expected_input_requests,
        )
        if current_snapshot != state.pinned_record_snapshot:
            raise Stage6WorkflowError(
                "Stage 6 execution capability input pin identity drifted"
            )
    except Stage6WorkflowError:
        raise
    except BaseException as exc:
        raise Stage6WorkflowError(
            f"Stage 6 execution capability is not current at {label}"
        ) from exc


def _begin_stage6_execution_operation(
    execution_capability: object,
) -> _Stage6ExecutionCapabilityState:
    if type(execution_capability) is not _Stage6ExecutionCapability:
        raise Stage6WorkflowError(
            "Stage 6 execution operation requires an opaque capability"
        )
    thread_id = threading.get_ident()
    with _STAGE6_EXECUTION_CAPABILITY_CONDITION:
        state = _STAGE6_EXECUTION_CAPABILITY_REGISTRY.get(execution_capability)
        if (
            type(state) is not _Stage6ExecutionCapabilityState
            or state.active is not True
            or state.identity is not execution_capability._identity
            or state.revoking is True
        ):
            raise Stage6WorkflowError(
                "Stage 6 execution capability is revoking or inactive"
            )
        state.active_operations += 1
        state.operation_threads[thread_id] = (
            state.operation_threads.get(thread_id, 0) + 1
        )
        return state


def _end_stage6_execution_operation(
    execution_capability: object,
    expected_state: _Stage6ExecutionCapabilityState,
) -> None:
    thread_id = threading.get_ident()
    with _STAGE6_EXECUTION_CAPABILITY_CONDITION:
        state = _STAGE6_EXECUTION_CAPABILITY_REGISTRY.get(execution_capability)
        if state is not expected_state or type(state) is not _Stage6ExecutionCapabilityState:
            raise Stage6WorkflowError(
                "Stage 6 execution operation registry drifted"
            )
        depth = state.operation_threads.get(thread_id, 0)
        if depth <= 0 or state.active_operations <= 0:
            raise Stage6WorkflowError(
                "Stage 6 execution operation accounting drifted"
            )
        if depth == 1:
            del state.operation_threads[thread_id]
        else:
            state.operation_threads[thread_id] = depth - 1
        state.active_operations -= 1
        if state.active_operations == 0:
            _STAGE6_EXECUTION_CAPABILITY_CONDITION.notify_all()


@contextmanager
def _stage6_execution_operation(
    execution_capability: object,
    *,
    label: str,
    formal_run_id: str | None = None,
    run_root: str | Path | None = None,
    stage_root: str | Path | None = None,
    repo_root: str | Path | None = None,
    config_path: str | Path | None = None,
    config: object | None = None,
    execution_identity: Mapping[str, object] | None = None,
    stage5_authority: Mapping[str, object] | None = None,
    verified_review_authorization: Mapping[str, object] | None = None,
    run_lease: RunLease | None = None,
    review_authorization_handle: object | None = None,
    input_pin: object | None = None,
    stage5_authority_handle: object | None = None,
):
    state = _begin_stage6_execution_operation(execution_capability)
    primary_error: BaseException | None = None
    try:
        _require_stage6_execution_capability(
            execution_capability,
            label=f"{label} entry",
            formal_run_id=formal_run_id,
            run_root=run_root,
            stage_root=stage_root,
            repo_root=repo_root,
            config_path=config_path,
            config=config,
            execution_identity=execution_identity,
            stage5_authority=stage5_authority,
            verified_review_authorization=verified_review_authorization,
            run_lease=run_lease,
            review_authorization_handle=review_authorization_handle,
            input_pin=input_pin,
            stage5_authority_handle=stage5_authority_handle,
        )
        try:
            yield
        except BaseException as exc:
            primary_error = exc
            raise
        finally:
            try:
                _require_stage6_execution_capability(
                    execution_capability,
                    label=f"{label} exit",
                    formal_run_id=formal_run_id,
                    run_root=run_root,
                    stage_root=stage_root,
                    repo_root=repo_root,
                    config_path=config_path,
                    config=config,
                    execution_identity=execution_identity,
                    stage5_authority=stage5_authority,
                    verified_review_authorization=verified_review_authorization,
                    run_lease=run_lease,
                    review_authorization_handle=review_authorization_handle,
                    input_pin=input_pin,
                    stage5_authority_handle=stage5_authority_handle,
                )
            except BaseException as capability_error:
                if primary_error is None:
                    raise
                primary_error.add_note(
                    "suppressed Stage 6 execution capability close validation "
                    f"failure: {capability_error}"
                )
    finally:
        _end_stage6_execution_operation(execution_capability, state)


def _revoke_stage6_execution_capability(
    execution_capability: _Stage6ExecutionCapability,
) -> None:
    if type(execution_capability) is not _Stage6ExecutionCapability:
        raise Stage6WorkflowError(
            "Stage 6 execution capability revocation requires an opaque capability"
        )
    thread_id = threading.get_ident()
    with _STAGE6_EXECUTION_CAPABILITY_CONDITION:
        state = _STAGE6_EXECUTION_CAPABILITY_REGISTRY.get(execution_capability)
        if state is None:
            return
        if type(state) is not _Stage6ExecutionCapabilityState:
            raise Stage6WorkflowError(
                "Stage 6 execution capability revocation drifted"
            )
        if state.operation_threads.get(thread_id, 0) > 0:
            raise Stage6WorkflowError(
                "Stage 6 execution capability cannot revoke its active operation"
            )
        state.revoking = True
        while state.active_operations > 0:
            _STAGE6_EXECUTION_CAPABILITY_CONDITION.wait()
            current = _STAGE6_EXECUTION_CAPABILITY_REGISTRY.get(
                execution_capability
            )
            if current is None:
                return
            if current is not state:
                raise Stage6WorkflowError(
                    "Stage 6 execution capability revocation registry drifted"
                )
        current = _STAGE6_EXECUTION_CAPABILITY_REGISTRY.get(execution_capability)
        if current is None:
            return
        if current is not state:
            raise Stage6WorkflowError(
                "Stage 6 execution capability revocation registry drifted"
            )
        del _STAGE6_EXECUTION_CAPABILITY_REGISTRY[execution_capability]
        state.active = False
        _STAGE6_EXECUTION_CAPABILITY_CONDITION.notify_all()


@contextmanager
def _stage6_execution_capability_scope(
    *,
    config: object,
    config_path: str | Path,
    formal_run_id: str,
    run_root: str | Path,
    stage_root: str | Path,
    repo_root: str | Path,
    execution_identity: Mapping[str, object],
    expected_input_requests: Sequence[tuple[str, str | Path]],
    review_authorization_handle: object,
    input_pin: object,
    stage5_authority: object,
    run_lease: RunLease,
    effective_config_bytes: bytes | None = None,
    planning_warm_start_path: str | Path | None = None,
    planning_child_source_repair_path: str | Path | None = None,
    planning_child_source_repair_continuation_path: (
        str | Path | None
    ) = None,
):
    from lunar_exploration_ppo.utils.stage6_input_pinning import Stage6InputPin
    from lunar_exploration_ppo.workflows.stage6_review_authorization import (
        Stage6ReviewAuthorizationHandle,
    )

    if type(review_authorization_handle) is not Stage6ReviewAuthorizationHandle:
        raise Stage6WorkflowError(
            "Stage 6 execution capability requires an exact fresh review handle"
        )
    if type(input_pin) is not Stage6InputPin:
        raise Stage6WorkflowError(
            "Stage 6 execution capability requires an exact Stage6InputPin"
        )
    if type(stage5_authority) is not FrozenStage5AuthorityHandle:
        raise Stage6WorkflowError(
            "Stage 6 execution capability requires an exact Stage 5 authority handle"
        )
    if type(run_lease) is not RunLease:
        raise Stage6WorkflowError(
            "Stage 6 execution capability requires the exact RunLease handle"
        )
    formal = validate_stage6_formal_run_id(formal_run_id)
    run = lexical_absolute(run_root)
    stage = lexical_absolute(stage_root)
    repo = lexical_absolute(repo_root)
    config_file = lexical_absolute(config_path)
    if run.name != formal or stage != run / "s6":
        raise Stage6WorkflowError(
            "Stage 6 execution capability run/stage root binding drifted"
        )
    base_config_payload = _capability_config_payload(config)
    config_payload = base_config_payload
    if effective_config_bytes is not None:
        from lunar_exploration_ppo.workflows.stage6_planning_warm_start import (
            Stage6PlanningWarmStartError,
            parse_planning_effective_config_bytes,
        )

        try:
            effective = parse_planning_effective_config_bytes(
                effective_config_bytes
            )
        except Stage6PlanningWarmStartError as exc:
            raise Stage6WorkflowError(
                "Stage 6 capability effective config is invalid"
            ) from exc
        if (
            effective.base_config_sha256
            != hashlib.sha256(base_config_payload).hexdigest()
            or effective.base_config != config
        ):
            raise Stage6WorkflowError(
                "Stage 6 capability effective/base config drifted"
            )
        config_payload = effective.effective_config_bytes
    identity_payload = _capability_json_payload(
        execution_identity,
        label="execution identity",
    )
    review_record = review_authorization_handle.canonical_record()
    validated_review = validate_stage6_verified_review_authorization(
        review_record,
        execution_identity=execution_identity,
        formal_run_id=formal,
    )
    review_payload = _capability_json_payload(
        validated_review,
        label="review authorization",
    )
    stage5_payload = _capability_json_payload(
        stage5_authority.identity,
        label="Stage 5 authority",
    )
    _validate_stage6_stage5_authority_binding(
        config=config,
        stage5_identity=stage5_authority.identity,
    )
    if (
        hashlib.sha256(config_payload).hexdigest()
        != execution_identity.get("config_sha256")
    ):
        raise Stage6WorkflowError(
            "Stage 6 execution capability config hash binding drifted"
        )
    provided_requests = tuple(
        (label, lexical_absolute(path)) for label, path in expected_input_requests
    )
    authoritative_requests = _stage6_input_pin_requests(
        repo_root=repo,
        config_path=config_file,
        stage5_authority=stage5_authority,
        review_authorization_handle=review_authorization_handle,
        planning_warm_start_path=planning_warm_start_path,
        planning_child_source_repair_path=(
            planning_child_source_repair_path
        ),
        planning_child_source_repair_continuation_path=(
            planning_child_source_repair_continuation_path
        ),
    )
    if _normalized_capability_input_requests(
        provided_requests
    ) != _normalized_capability_input_requests(authoritative_requests):
        raise Stage6WorkflowError(
            "Stage 6 execution capability input pin request set is not authoritative"
        )
    requests = tuple(
        (label, lexical_absolute(path)) for label, path in authoritative_requests
    )
    pin_snapshot = _stage6_pin_record_snapshot(
        input_pin,
        expected_requests=requests,
    )
    _validate_stage6_input_pin_bindings(
        repo_root=repo,
        config_path=config_file,
        base_config_payload=base_config_payload,
        config_identity_payload=config_payload,
        execution_identity=execution_identity,
        stage5_authority=stage5_authority,
        review_authorization_handle=review_authorization_handle,
        verified_review_authorization=validated_review,
        pin_snapshot=pin_snapshot,
    )
    run_lease.require_current()
    review_authorization_handle.require_current(
        "Stage 6 execution capability issuance"
    )
    stage5_authority.require_current("Stage 6 execution capability issuance")
    input_pin.require_current("Stage 6 execution capability issuance")
    identity = object()
    capability = _Stage6ExecutionCapability(
        _STAGE6_EXECUTION_CAPABILITY_ISSUER,
        identity,
    )
    state = _Stage6ExecutionCapabilityState(
        identity=identity,
        formal_run_id=formal,
        run_root=run,
        stage_root=stage,
        repo_root=repo,
        config_path=config_file,
        base_config_payload=base_config_payload,
        config_payload=config_payload,
        execution_identity_payload=identity_payload,
        review_authorization_payload=review_payload,
        stage5_authority_payload=stage5_payload,
        expected_input_requests=requests,
        pinned_record_snapshot=pin_snapshot,
        review_authorization_handle=review_authorization_handle,
        input_pin=input_pin,
        stage5_authority=stage5_authority,
        run_lease=run_lease,
    )
    with _STAGE6_EXECUTION_CAPABILITY_LOCK:
        _STAGE6_EXECUTION_CAPABILITY_REGISTRY[capability] = state
    primary_error: BaseException | None = None
    try:
        _require_stage6_execution_capability(
            capability,
            label="Stage 6 execution capability issued",
            formal_run_id=formal,
            run_root=run,
            stage_root=stage,
            repo_root=repo,
            config_path=config_file,
            config=config,
            execution_identity=execution_identity,
            stage5_authority=stage5_authority.identity,
            verified_review_authorization=validated_review,
        )
        yield capability
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        try:
            _revoke_stage6_execution_capability(capability)
        except BaseException as revoke_error:
            if primary_error is None:
                raise
            primary_error.add_note(
                f"suppressed Stage 6 execution capability revoke failure: {revoke_error}"
            )


def _validate_stage6_run_id(
    run_id: str,
    *,
    pattern: re.Pattern[str],
    label: str,
) -> str:
    if not isinstance(run_id, str):
        raise Stage6WorkflowError(f"Stage 6 {label} run id must be a string")
    if run_id in HISTORICAL_STAGE6_RUN_IDS:
        raise Stage6WorkflowError(
            "Stage 6 historical R1/R2 run id is immutable and cannot be resumed"
        )
    match = pattern.fullmatch(run_id)
    if match is None:
        raise Stage6WorkflowError(
            f"Stage 6 {label} run id is outside its canonical namespace"
        )
    timestamp = match.group("timestamp")
    try:
        parsed = datetime.strptime(timestamp, "%Y%m%dT%H%M%SZ")
    except ValueError as exc:
        raise Stage6WorkflowError(
            f"Stage 6 {label} run id has an invalid UTC timestamp"
        ) from exc
    if parsed.strftime("%Y%m%dT%H%M%SZ") != timestamp:
        raise Stage6WorkflowError(
            f"Stage 6 {label} run id timestamp is not canonical UTC"
        )
    return run_id


def validate_stage6_formal_run_id(run_id: str) -> str:
    """Validate the sole public namespace authorized for formal Stage 6 runs."""

    return _validate_stage6_run_id(
        run_id,
        pattern=_FORMAL_STAGE6_RUN_ID_PATTERN,
        label="formal",
    )


def validate_stage6_verified_review_authorization(
    value: Mapping[str, object],
    *,
    execution_identity: Mapping[str, object],
    formal_run_id: str,
) -> dict[str, object]:
    """Cross-check the external review authority against current execution bytes."""

    try:
        expected_run_id = validate_stage6_formal_run_id(formal_run_id)
    except (Stage6WorkflowError, TypeError) as exc:
        raise Stage6WorkflowError(
            "Stage 6 verified review authorization formal run id drifted"
        ) from exc
    if (
        not isinstance(value, Mapping)
        or set(value) != STAGE6_VERIFIED_REVIEW_AUTHORIZATION_FIELDS
        or not isinstance(execution_identity, Mapping)
    ):
        raise Stage6WorkflowError(
            "Stage 6 verified review authorization field set drifted"
        )
    verified = dict(value)
    scope = verified.get("single_seed_scope")
    expected_scope = {
        "scope_kind": "single_seed_system_closure/v1",
        "seed": 20260716,
        "updates": 100,
    }
    tree = verified.get("reviewed_prospective_git_tree")
    if (
        verified.get("schema_version")
        != "stage6_verified_review_launch_authorization/v1"
        or verified.get("stage_id") != STAGE6_STAGE_ID
        or verified.get("formal_run_id") != expected_run_id
        or verified.get("authorized") is not True
        or scope != expected_scope
        or verified.get("base_commit") != STAGE5_COMMIT
        or execution_identity.get("schema_version")
        != "stage6_execution_identity/v1"
        or execution_identity.get("base_commit") != STAGE5_COMMIT
        or execution_identity.get("head_commit") != STAGE5_COMMIT
        or execution_identity.get("real_index_empty") is not True
        or not isinstance(tree, str)
        or len(tree) != 40
        or any(character not in "0123456789abcdef" for character in tree)
        or execution_identity.get("prospective_git_tree") != tree
        or verified.get("prospective_tree_sha256")
        != hashlib.sha256(tree.encode("ascii")).hexdigest()
        or verified.get("prospective_tree_sha256")
        != execution_identity.get("prospective_tree_sha256")
        or verified.get("review_identity_sha256")
        != _canonical_sha256(dict(execution_identity))
    ):
        raise Stage6WorkflowError(
            "Stage 6 verified review authorization identity drifted"
        )
    for record_key, identity_key in (
        ("changed_path_set_sha256", "changed_path_set_sha256"),
        ("source_set_sha256", "source_set_sha256"),
        ("config_sha256", "config_sha256"),
        ("data_sha256", "data_sha256"),
        ("environment_sha256", "environment_sha256"),
    ):
        if (
            not _is_sha256(verified.get(record_key))
            or verified.get(record_key) != execution_identity.get(identity_key)
        ):
            raise Stage6WorkflowError(
                "Stage 6 verified review authorization identity drifted"
            )
    for name in (
        "authorization_file_sha256",
        "review_identity_sha256",
        "frozen_diff_sha256",
        "spec_review_sha256",
        "quality_review_sha256",
    ):
        if not _is_sha256(verified.get(name)):
            raise Stage6WorkflowError(
                "Stage 6 verified review authorization evidence drifted"
            )
    for name in ("authorization_file_size_bytes", "frozen_diff_size_bytes"):
        if type(verified.get(name)) is not int or int(verified[name]) < 0:
            raise Stage6WorkflowError(
                "Stage 6 verified review authorization evidence drifted"
            )
    return verified


def validate_stage6_standalone_preflight_run_id(run_id: str) -> str:
    """Validate the namespace reserved for standalone machine preflight."""

    return _validate_stage6_run_id(
        run_id,
        pattern=_STANDALONE_PREFLIGHT_RUN_ID_PATTERN,
        label="preflight",
    )


def _is_link_or_reparse(path: Path) -> bool:
    """Return whether *path* itself is a symlink, junction, or reparse point."""

    return is_link_or_reparse(path)


def _resolve_stage6_run_root(
    base_output_root: str | Path,
    run_id: str,
) -> tuple[Path, Path]:
    """Resolve a run root that is exactly one ordinary child of its base."""

    if (
        not isinstance(run_id, str)
        or not run_id
        or run_id in {".", ".."}
        or run_id != run_id.strip()
        or "/" in run_id
        or "\\" in run_id
        or Path(run_id).is_absolute()
        or len(Path(run_id).parts) != 1
    ):
        raise Stage6WorkflowError("Stage 6 run id must be one safe path component")
    lexical_base = lexical_absolute(base_output_root)
    unresolved = lexical_base / run_id
    try:
        base = require_plain_path(
            lexical_base,
            allow_missing=True,
            leaf_kind="directory",
            label="Stage 6 output base",
        )
        resolved = require_plain_path(
            unresolved,
            base=lexical_base,
            allow_missing=True,
            label="Stage 6 run root",
        )
        lexical_relative = unresolved.relative_to(lexical_base)
        resolved_relative = resolved.relative_to(base)
    except (PathSecurityError, ValueError) as exc:
        raise Stage6WorkflowError(
            "Stage 6 run root contains a link/reparse point or escapes its output base"
        ) from exc
    if lexical_relative.parts != (run_id,) or resolved_relative.parts != (run_id,):
        raise Stage6WorkflowError("Stage 6 run root must be exactly one child directory")
    return base, resolved


def _require_canonical_stage6_output_root(value: str | Path) -> Path:
    candidate = lexical_absolute(value)
    canonical = lexical_absolute(CANONICAL_STAGE6_OUTPUT_ROOT)
    if candidate != canonical:
        raise Stage6WorkflowError(
            "Stage 6 output root must equal canonical D:/xunce/out/ppo_frontier"
        )
    try:
        return require_plain_path(
            candidate,
            allow_missing=True,
            leaf_kind="directory",
            label="canonical Stage 6 output root",
        )
    except PathSecurityError as exc:
        raise Stage6WorkflowError(
            "canonical Stage 6 output root contains a link or reparse point"
        ) from exc


def load_stage4_policy_for_standard(
    *,
    checkpoint_path: str | Path,
    checkpoint_sha256: str,
    policy_state_sha256: str,
    device: str,
) -> object:
    """从已绑定 checkpoint 仅加载 Stage 4 policy weights 到 CUDA FP32。"""

    import torch

    from lunar_exploration_ppo.policy.cross_attention import (
        CrossAttentionFrontierPolicy,
    )
    from lunar_exploration_ppo.ppo.checkpoint import safe_load_checkpoint_payload
    from lunar_exploration_ppo.ppo.trainer import policy_state_sha256 as state_hash

    if device != "cuda" or not torch.cuda.is_available():
        raise Stage6WorkflowError("Stage 6 policy initialization requires CUDA")
    if not _is_sha256(checkpoint_sha256) or not _is_sha256(policy_state_sha256):
        raise Stage6WorkflowError("Stage 4 expected SHA-256 binding is invalid")
    try:
        path = require_plain_path(
            lexical_absolute(checkpoint_path),
            leaf_kind="file",
            label="Stage 4 checkpoint",
        )
    except PathSecurityError as exc:
        raise Stage6WorkflowError(
            "Stage 4 checkpoint contains a link or reparse point"
        ) from exc
    checkpoint_identity = path_identity(path)
    payload_bytes = path.read_bytes()
    if path_identity(path) != checkpoint_identity:
        raise Stage6WorkflowError("Stage 4 checkpoint identity changed while read")
    if hashlib.sha256(payload_bytes).hexdigest() != checkpoint_sha256:
        raise Stage6WorkflowError("Stage 4 checkpoint SHA-256 drift")
    try:
        payload = safe_load_checkpoint_payload(payload_bytes)
    except Exception as exc:
        raise Stage6WorkflowError("Stage 4 checkpoint cannot be decoded") from exc
    if (
        not isinstance(payload, dict)
        or not isinstance(payload.get("model_state_dict"), Mapping)
        or payload.get("policy_state_sha256") != policy_state_sha256
    ):
        raise Stage6WorkflowError("Stage 4 checkpoint policy binding drift")
    policy = CrossAttentionFrontierPolicy().to(device="cuda", dtype=torch.float32)
    try:
        policy.load_state_dict(payload["model_state_dict"], strict=True)
    except (RuntimeError, ValueError) as exc:
        raise Stage6WorkflowError("Stage 4 model-state contract drift") from exc
    if state_hash(policy) != policy_state_sha256:
        raise Stage6WorkflowError("Stage 4 restored policy SHA-256 drift")
    return policy


def validate_stage6_machine_preflight_audit(
    value: Mapping[str, object],
) -> dict[str, object]:
    expected = {
        "schema_version",
        "passed",
        "config_sha256",
        "stage5_gate_sha256",
        "environment_identity",
        "environment_sha256",
        "resources",
        "standard",
        "cuda",
        "collector",
        "timings_seconds",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise Stage6WorkflowError("Stage 6 preflight audit field set drifted")
    if (
        value["schema_version"] != "stage6_machine_preflight/v1"
        or value["passed"] is not True
        or not _is_sha256(value["config_sha256"])
        or not _is_sha256(value["stage5_gate_sha256"])
    ):
        raise Stage6WorkflowError("Stage 6 preflight audit identity drifted")
    environment_identity = value["environment_identity"]
    try:
        environment_valid = validate_stage6_environment_identity(
            environment_identity  # type: ignore[arg-type]
        )
        environment_hash_valid = (
            value["environment_sha256"]
            == stage6_environment_sha256(environment_valid)
        )
    except (Stage6WorkflowError, TypeError, ValueError):
        environment_hash_valid = False
    if not environment_hash_valid:
        raise Stage6WorkflowError("Stage 6 preflight environment identity drifted")
    resources = value["resources"]
    standard = value["standard"]
    cuda = value["cuda"]
    collector = value["collector"]
    timings = value["timings_seconds"]
    if not all(
        isinstance(item, Mapping)
        for item in (resources, standard, cuda, collector, timings)
    ):
        raise Stage6WorkflowError("Stage 6 preflight audit section drifted")
    if (
        set(resources)
        != {
            "d_free_bytes",
            "rss_bytes",
            "rss_source",
            "rss_root_pid",
            "rss_sample_count",
            "rss_latest_process_count",
            "rss_peak_process_count",
            "peak_vram_bytes",
            "warnings",
            "hard_stops",
        }
        or type(resources["d_free_bytes"]) is not int
        or resources["d_free_bytes"] < 100 * 1024**3
        or type(resources["rss_bytes"]) is not int
        or resources["rss_bytes"] >= 20 * 1024**3
        or resources["rss_source"]
        != "process_tree_lifecycle_peak_current_sum/v1"
        or type(resources["rss_root_pid"]) is not int
        or resources["rss_root_pid"] <= 0
        or type(resources["rss_sample_count"]) is not int
        or resources["rss_sample_count"] < 2
        or type(resources["rss_latest_process_count"]) is not int
        or resources["rss_latest_process_count"] < 1
        or type(resources["rss_peak_process_count"]) is not int
        or resources["rss_peak_process_count"] < 9
        or type(resources["peak_vram_bytes"]) is not int
        or resources["peak_vram_bytes"] >= int(10.1 * 1024**3)
        or not isinstance(resources["warnings"], list)
        or resources["hard_stops"] != []
    ):
        raise Stage6WorkflowError("Stage 6 preflight resource gate failed")
    if (
        set(standard)
        != {
            "catalog_sha256",
            "scenario_id",
            "prior_shape",
            "truth_shape",
            "local_crop_shape",
            "candidate_slots",
            "coverable_exact",
            "step_trainable",
            "cache_scope",
        }
        or not _is_sha256(standard["catalog_sha256"])
        or not isinstance(standard["scenario_id"], str)
        or standard["prior_shape"] != [7, 32, 32]
        or standard["truth_shape"] != [256, 256]
        or standard["local_crop_shape"] != [8, 96, 96]
        or standard["candidate_slots"] != 1024
        or standard["coverable_exact"] is not True
        or standard["step_trainable"] is not True
    ):
        raise Stage6WorkflowError("Stage 6 preflight Standard contract failed")
    cache_scope = standard["cache_scope"]
    if not isinstance(cache_scope, Mapping) or set(cache_scope) != {
        "catalog_process_cache_hit",
        "runtime_mode",
        "manifest",
        "formal_audit",
        "cold_spawn",
    }:
        raise Stage6WorkflowError("Stage 6 preflight Standard cache contract failed")
    manifest = cache_scope["manifest"]
    formal_audit = cache_scope["formal_audit"]
    cold_spawn = cache_scope["cold_spawn"]
    expected_manifest = {
        "path": STAGE6_COVERAGE_CACHE_MANIFEST_PATH.as_posix(),
        "sha256": STAGE6_COVERAGE_CACHE_MANIFEST_SHA256,
        "size_bytes": STAGE6_COVERAGE_CACHE_MANIFEST_SIZE_BYTES,
        "cache_root": STAGE6_COVERAGE_CACHE_ROOT.as_posix(),
        "entry_set_sha256": STAGE6_COVERAGE_CACHE_ENTRY_SET_SHA256,
        "entry_count": 1064,
        "split_counts": {
            "train": 700,
            "validation": 150,
            "test": 150,
            "unseen": 64,
        },
    }
    expected_formal_audit = {
        "path": STAGE6_COVERAGE_CACHE_FORMAL_AUDIT_PATH.as_posix(),
        "sha256": STAGE6_COVERAGE_CACHE_FORMAL_AUDIT_SHA256,
        "size_bytes": STAGE6_COVERAGE_CACHE_FORMAL_AUDIT_SIZE_BYTES,
    }
    if (
        cache_scope["catalog_process_cache_hit"] is not True
        or cache_scope["runtime_mode"] != STAGE6_COVERAGE_CACHE_RUNTIME_MODE
        or manifest != expected_manifest
        or formal_audit != expected_formal_audit
        or not isinstance(cold_spawn, Mapping)
        or set(cold_spawn)
        != {
            "verification",
            "scenario_id",
            "worker_pid",
            "parent_pid",
            "start_method",
            "cache_hit",
            "compute_fallback_called",
        }
        or cold_spawn["verification"] != "spawn_worker_reset/v1"
        or cold_spawn["scenario_id"] != standard["scenario_id"]
        or type(cold_spawn["worker_pid"]) is not int
        or cold_spawn["worker_pid"] <= 0
        or type(cold_spawn["parent_pid"]) is not int
        or cold_spawn["parent_pid"] <= 0
        or cold_spawn["worker_pid"] == cold_spawn["parent_pid"]
        or cold_spawn["start_method"] != "spawn"
        or cold_spawn["cache_hit"] is not True
        or cold_spawn["compute_fallback_called"] is not False
    ):
        raise Stage6WorkflowError("Stage 6 preflight Standard cache contract failed")
    if (
        set(cuda)
        != {
            "available",
            "device",
            "compute_dtype",
            "amp_enabled",
            "autocast_enabled",
            "checkpoint_sha256",
            "policy_state_sha256",
            "forward_warmup_count",
            "forward_measurement_count",
            "forward_p95_ms",
            "outputs_finite",
            "joint_logprob_finite",
            "backward_completed",
            "gradient_tensor_count",
            "gradients_finite",
            "peak_vram_bytes",
        }
        or cuda["available"] is not True
        or not isinstance(cuda["device"], str)
        or not cuda["device"].startswith("cuda:")
        or cuda["compute_dtype"] != "float32"
        or cuda["amp_enabled"] is not False
        or cuda["autocast_enabled"] is not False
        or not _is_sha256(cuda["checkpoint_sha256"])
        or not _is_sha256(cuda["policy_state_sha256"])
        or cuda["forward_warmup_count"] != 5
        or cuda["forward_measurement_count"] != 20
        or not isinstance(cuda["forward_p95_ms"], (int, float))
        or not math.isfinite(float(cuda["forward_p95_ms"]))
        or float(cuda["forward_p95_ms"]) > 100.0
        or cuda["outputs_finite"] is not True
        or cuda["joint_logprob_finite"] is not True
        or cuda["backward_completed"] is not True
        or type(cuda["gradient_tensor_count"]) is not int
        or cuda["gradient_tensor_count"] <= 0
        or cuda["gradients_finite"] is not True
        or type(cuda["peak_vram_bytes"]) is not int
        or cuda["peak_vram_bytes"] >= int(10.1 * 1024**3)
    ):
        raise Stage6WorkflowError("Stage 6 preflight CUDA forward/backward failed")
    if cuda["peak_vram_bytes"] != resources["peak_vram_bytes"]:
        raise Stage6WorkflowError("Stage 6 preflight VRAM binding drifted")
    if (
        set(collector)
        != {
            "worker_count",
            "worker_pids",
            "worker_start_methods",
            "inference_pids",
            "parent_pid",
            "trainable_transition_count",
            "snapshot_count",
            "policy_state_sha256",
            "all_workers_closed",
            "residual_child_pids",
        }
        or collector["worker_count"] != 8
        or not isinstance(collector["worker_pids"], list)
        or len(collector["worker_pids"]) != 8
        or len(set(collector["worker_pids"])) != 8
        or collector["worker_start_methods"] != ["spawn"] * 8
        or collector["inference_pids"] != [collector["parent_pid"]]
        or collector["trainable_transition_count"] != 8
        or collector["snapshot_count"] != 8
        or collector["policy_state_sha256"] != cuda["policy_state_sha256"]
        or collector["all_workers_closed"] is not True
        or collector["residual_child_pids"] != []
        or resources["rss_root_pid"] != collector["parent_pid"]
        or cold_spawn["worker_pid"] not in collector["worker_pids"]
        or cold_spawn["parent_pid"] != collector["parent_pid"]
    ):
        raise Stage6WorkflowError("Stage 6 preflight collector contract failed")
    if set(timings) != _PREFLIGHT_TIMING_NAMES or any(
        not isinstance(item, (int, float))
        or not math.isfinite(float(item))
        or float(item) < 0.0
        for item in timings.values()
    ):
        raise Stage6WorkflowError("Stage 6 preflight timing contract failed")
    return dict(value)


def _run_stage6_spawn_preflight(
    *,
    specs: Sequence[object],
    policy: object,
    device: str,
    collector_contract: object,
    timeout_seconds: float,
) -> dict[str, object]:
    import multiprocessing as mp
    import time

    from lunar_exploration_ppo.ppo.collector import RolloutCollector, SpawnVectorEnv
    from lunar_exploration_ppo.utils.resources import ProcessTreeRSSMonitor

    monitor = ProcessTreeRSSMonitor()
    vector_env = None
    worker_pids: tuple[int, ...] = ()
    preflight_result = None
    timings: dict[str, float] = {}
    monitor.start()
    try:
        spawn_started = time.perf_counter()
        vector_env = SpawnVectorEnv(specs, timeout_seconds=timeout_seconds)
        timings["spawn_startup"] = time.perf_counter() - spawn_started
        worker_pids = tuple(vector_env.worker_pids)
        try:
            collector = RolloutCollector(
                policy=policy,
                vector_env=vector_env,
                device=device,
                contract=collector_contract,
            )
            preflight_result = collector.preflight_one_step()
            monitor.sample_now()
        finally:
            close_started = time.perf_counter()
            vector_env.close()
            timings["spawn_close"] = time.perf_counter() - close_started
    finally:
        monitor.stop()
    if preflight_result is None or vector_env is None:
        raise Stage6WorkflowError("Stage 6 collector preflight produced no result")
    active_pids = {child.pid for child in mp.active_children() if child.pid is not None}
    residual = sorted(set(worker_pids) & active_pids)
    return {
        "preflight_result": preflight_result,
        "worker_pids": worker_pids,
        "all_workers_closed": vector_env.closed and not any(vector_env.worker_alive),
        "residual_child_pids": residual,
        "timings_seconds": timings,
        "process_tree_monitor": monitor,
    }


def _stage6_preflight_resource_audit(
    *,
    process_tree_monitor: object,
    peak_vram_bytes: int,
):
    from lunar_exploration_ppo.utils.resources import (
        capture_resource_snapshot,
        evaluate_resource_gates,
    )

    snapshot = capture_resource_snapshot(
        peak_vram_bytes=peak_vram_bytes,
        process_tree_monitor=process_tree_monitor,  # type: ignore[arg-type]
    )
    decision = evaluate_resource_gates(snapshot, preflight=True)
    audit = {
        "d_free_bytes": snapshot.d_free_bytes,
        "rss_bytes": snapshot.rss_bytes,
        "rss_source": snapshot.rss_source,
        "rss_root_pid": snapshot.rss_root_pid,
        "rss_sample_count": snapshot.rss_sample_count,
        "rss_latest_process_count": snapshot.rss_latest_process_count,
        "rss_peak_process_count": snapshot.rss_peak_process_count,
        "peak_vram_bytes": snapshot.peak_vram_bytes,
        "warnings": list(decision.warnings),
        "hard_stops": list(decision.hard_stops),
    }
    return audit, decision


def _load_stage6_preflight_coverage_cache(catalog: object):
    """Load the fixed manifest and audit without permitting a compute fallback."""

    from lunar_exploration_ppo.env.coverage_cache import (
        CoverageCacheError,
        Stage6CoverageManifest,
    )
    from lunar_exploration_ppo.env.standard_training import (
        _validate_coverage_cache_manifest_binding,
    )

    try:
        manifest_read = secure_read_bytes(
            STAGE6_COVERAGE_CACHE_MANIFEST_PATH,
            label="Stage 6 production coverage cache manifest",
        )
        formal_audit_read = secure_read_bytes(
            STAGE6_COVERAGE_CACHE_FORMAL_AUDIT_PATH,
            label="Stage 6 production coverage cache formal audit",
        )
        manifest = Stage6CoverageManifest.load(
            STAGE6_COVERAGE_CACHE_MANIFEST_PATH,
            expected_sha256=STAGE6_COVERAGE_CACHE_MANIFEST_SHA256,
        )
        _validate_coverage_cache_manifest_binding(manifest, catalog)  # type: ignore[arg-type]
    except (CoverageCacheError, OSError, PathSecurityError, TypeError, ValueError) as exc:
        raise Stage6WorkflowError(
            "Stage 6 production coverage cache binding failed"
        ) from exc
    if (
        len(manifest_read.payload) != STAGE6_COVERAGE_CACHE_MANIFEST_SIZE_BYTES
        or hashlib.sha256(manifest_read.payload).hexdigest()
        != STAGE6_COVERAGE_CACHE_MANIFEST_SHA256
        or len(formal_audit_read.payload)
        != STAGE6_COVERAGE_CACHE_FORMAL_AUDIT_SIZE_BYTES
        or hashlib.sha256(formal_audit_read.payload).hexdigest()
        != STAGE6_COVERAGE_CACHE_FORMAL_AUDIT_SHA256
        or manifest.cache_root.as_posix() != STAGE6_COVERAGE_CACHE_ROOT.as_posix()
        or manifest.entry_set_sha256 != STAGE6_COVERAGE_CACHE_ENTRY_SET_SHA256
        or len(manifest.entries) != 1064
        or dict(manifest.split_counts)
        != {
            "train": 700,
            "validation": 150,
            "test": 150,
            "unseen": 64,
        }
    ):
        raise Stage6WorkflowError(
            "Stage 6 production coverage cache identity drifted"
        )
    return manifest


def _stage6_cached_preflight_env_specs(
    *,
    catalog: object,
    sampler_seeds: Sequence[int],
    safety_contract: object,
    config_sha256: str,
) -> tuple[object, ...]:
    """Build the fixed eight spawn specs with no manifest bypass surface."""

    from lunar_exploration_ppo.env.standard_training import standard_env_specs
    from lunar_exploration_ppo.ppo.collector import SpawnEnvSpec
    from lunar_exploration_ppo.ppo.standard_training import (
        _build_cached_standard_training_env,
    )

    legacy_specs = standard_env_specs(
        catalog,  # type: ignore[arg-type]
        split="train",
        sampler_seeds=tuple(sampler_seeds),
        safety_contract=safety_contract,  # type: ignore[arg-type]
        config_sha256=config_sha256,
    )
    if len(legacy_specs) != 8 or any(
        not isinstance(spec, SpawnEnvSpec)
        or not isinstance(spec.kwargs, Mapping)
        for spec in legacy_specs
    ):
        raise Stage6WorkflowError("Stage 6 preflight spawn spec contract drifted")
    return tuple(
        SpawnEnvSpec(
            factory=_build_cached_standard_training_env,
            kwargs={
                **dict(spec.kwargs),
                "coverage_cache_manifest_path": (
                    STAGE6_COVERAGE_CACHE_MANIFEST_PATH.as_posix()
                ),
                "coverage_cache_manifest_sha256": (
                    STAGE6_COVERAGE_CACHE_MANIFEST_SHA256
                ),
            },
        )
        for spec in legacy_specs
    )


def _guard_stage6_machine_preflight(function):
    def guarded(**kwargs: object) -> dict[str, object]:
        run_id = kwargs.get("run_id")
        try:
            validate_stage6_standalone_preflight_run_id(run_id)  # type: ignore[arg-type]
        except Stage6WorkflowError:
            formal_run_id = validate_stage6_formal_run_id(run_id)  # type: ignore[arg-type]
        else:
            if any(
                kwargs.get(name) is not None
                for name in (
                    "execution_capability",
                    "review_authorization_handle",
                    "input_pin",
                    "stage5_authority",
                    "run_lease",
                    "execution_identity",
                    "config",
                )
            ):
                raise Stage6WorkflowError(
                    "Stage 6 standalone preflight cannot use formal capability authority"
                )
            return function(**kwargs)

        base, run_root = _resolve_stage6_run_root(
            kwargs.get("base_output_root"),  # type: ignore[arg-type]
            formal_run_id,
        )
        repo_root = Path(__file__).resolve().parents[3]
        execution_capability = kwargs.get("execution_capability")
        config = kwargs.get("config")
        execution_identity = kwargs.get("execution_identity")
        review_authorization_handle = kwargs.get("review_authorization_handle")
        input_pin = kwargs.get("input_pin")
        stage5_authority = kwargs.get("stage5_authority")
        run_lease = kwargs.get("run_lease")
        with _stage6_execution_operation(
            execution_capability,
            label="Stage 6 formal machine preflight",
            formal_run_id=formal_run_id,
            run_root=run_root,
            stage_root=run_root / "s6",
            repo_root=repo_root,
            config_path=kwargs.get("config_path"),  # type: ignore[arg-type]
            config=config,
            execution_identity=execution_identity,  # type: ignore[arg-type]
            review_authorization_handle=review_authorization_handle,
            input_pin=input_pin,
            stage5_authority_handle=stage5_authority,
            run_lease=run_lease,  # type: ignore[arg-type]
        ):
            if any(
                value is None
                for value in (
                    config,
                    execution_identity,
                    review_authorization_handle,
                    input_pin,
                    stage5_authority,
                    run_lease,
                )
            ):
                raise Stage6WorkflowError(
                    "Stage 6 formal preflight capability context is incomplete"
                )
            return function(**kwargs)

    guarded.__name__ = function.__name__
    guarded.__qualname__ = function.__qualname__
    guarded.__doc__ = function.__doc__
    guarded.__annotations__ = dict(function.__annotations__)
    return guarded


def run_stage6_machine_preflight(
    *,
    config_path: str | Path,
    run_id: str,
    stage5_gate_path: str | Path,
) -> dict[str, object]:
    """执行 Stage 6 Standard v1 机器预检并持久化 canonical audit。"""

    run_id = validate_stage6_standalone_preflight_run_id(run_id)
    return _run_stage6_machine_preflight(
        config_path=config_path,
        run_id=run_id,
        stage5_gate_path=stage5_gate_path,
        base_output_root=CANONICAL_STAGE6_OUTPUT_ROOT,
        require_canonical_output=True,
    )


def _run_stage6_machine_preflight_for_test(
    *,
    config_path: str | Path,
    run_id: str,
    stage5_gate_path: str | Path,
    base_output_root: str | Path,
) -> dict[str, object]:
    """Private D-drive output-root seam used only by unit/real-preflight tests."""

    run_id = validate_stage6_standalone_preflight_run_id(run_id)
    return _run_stage6_machine_preflight(
        config_path=config_path,
        run_id=run_id,
        stage5_gate_path=stage5_gate_path,
        base_output_root=base_output_root,
        require_canonical_output=False,
    )


@_guard_stage6_machine_preflight
def _run_stage6_machine_preflight(**_: object) -> dict[str, object]:
    import os
    import time

    import numpy as np
    import torch

    from lunar_exploration_ppo.configs.stage6 import SafetyContract, load_stage6_config
    from lunar_exploration_ppo.utils.resources import (
        capture_resource_snapshot,
        evaluate_resource_gates,
    )

    config_path = Path(_["config_path"]).expanduser().resolve()
    stage5_gate_path = Path(_["stage5_gate_path"]).expanduser().resolve()
    run_id = _["run_id"]
    base_value = _["base_output_root"]
    require_canonical_output = _["require_canonical_output"]
    repo_root = Path(__file__).resolve().parents[3]
    canonical_config = (repo_root / "configs/ppo_highres_frontier_stage6_v1.json").resolve()
    if config_path != canonical_config or not isinstance(run_id, str) or not run_id:
        raise Stage6WorkflowError("Stage 6 preflight config or run id is not canonical")
    base_config_bytes = config_path.read_bytes()
    config = load_stage6_config(config_path)
    config_bytes = base_config_bytes
    effective_config_bytes = _.get("effective_config_bytes")
    if effective_config_bytes is not None:
        from lunar_exploration_ppo.workflows.stage6_planning_warm_start import (
            Stage6PlanningWarmStartError,
            parse_planning_effective_config_bytes,
        )

        try:
            effective = parse_planning_effective_config_bytes(
                effective_config_bytes  # type: ignore[arg-type]
            )
        except Stage6PlanningWarmStartError as exc:
            raise Stage6WorkflowError(
                "Stage 6 preflight effective config is invalid"
            ) from exc
        if (
            effective.base_config_sha256
            != hashlib.sha256(base_config_bytes).hexdigest()
            or effective.base_config != config
        ):
            raise Stage6WorkflowError(
                "Stage 6 preflight effective/base config drifted"
            )
        config_bytes = effective.effective_config_bytes
    config_sha256 = hashlib.sha256(config_bytes).hexdigest()
    if type(require_canonical_output) is not bool:
        raise Stage6WorkflowError("Stage 6 preflight output-root mode drifted")
    if require_canonical_output:
        _require_canonical_stage6_output_root(config.output_root)
        _require_canonical_stage6_output_root(base_value)
    safety_contract = SafetyContract.from_stage6_config(config)
    authority = verify_frozen_stage5_authority(
        gate_path=stage5_gate_path,
        repo_root=repo_root,
    )
    base = Path(base_value)
    base, run_root = _resolve_stage6_run_root(base, run_id)
    if run_root.drive.upper() != "D:":
        raise Stage6WorkflowError("Stage 6 preflight output must be on D drive")
    preflight_root = run_root / "s6/preflight"
    audit_path = preflight_root / "audit.json"
    try:
        require_plain_path(
            preflight_root,
            base=base,
            allow_missing=True,
            label="Stage 6 preflight path",
        )
    except PathSecurityError as exc:
        raise Stage6WorkflowError(
            "Stage 6 preflight path contains a link or reparse point"
        ) from exc
    if preflight_root.exists() or audit_path.exists():
        raise Stage6WorkflowError("Stage 6 preflight root already exists")

    from lunar_exploration_ppo.env.standard_training import (
        StandardTrainingEnv,
        build_standard_catalog,
    )
    from lunar_exploration_ppo.policy.cross_attention import (
        PolicyForwardOutput,
        batch_policy_observations,
        sample_action,
    )
    from lunar_exploration_ppo.ppo.collector import CollectorContract
    from lunar_exploration_ppo.ppo.trainer import (
        PPOTrainingError,
        policy_state_sha256,
        validate_ppo_math_evidence,
    )

    initial_resources = capture_resource_snapshot(peak_vram_bytes=0)
    initial_decision = evaluate_resource_gates(initial_resources, preflight=True)
    if not initial_decision.passed:
        raise Stage6WorkflowError("Stage 6 preflight resource gate failed before work")
    if not torch.cuda.is_available():
        raise Stage6WorkflowError("Stage 6 preflight requires CUDA")
    environment_identity = stage6_environment_identity()
    environment_sha256 = stage6_environment_sha256(environment_identity)
    torch.cuda.reset_peak_memory_stats()

    timings: dict[str, float] = {}
    catalog_started = time.perf_counter()
    catalog = build_standard_catalog(verify_hashes=True)
    timings["catalog_build"] = time.perf_counter() - catalog_started
    cache_started = time.perf_counter()
    cached_catalog = build_standard_catalog(verify_hashes=True)
    timings["catalog_cache_reuse"] = time.perf_counter() - cache_started
    catalog_cache_hit = cached_catalog is catalog
    if not catalog_cache_hit:
        raise Stage6WorkflowError("Standard catalog process cache did not reuse identity")
    coverage_manifest = _load_stage6_preflight_coverage_cache(catalog)

    direct_env = StandardTrainingEnv(
        catalog,
        split="train",
        sampler_seed=config.training.seeds[0],
        safety_contract=safety_contract,
        config_sha256=config_sha256,
        coverage_cache_manifest_path=(
            STAGE6_COVERAGE_CACHE_MANIFEST_PATH.as_posix()
        ),
        coverage_cache_manifest_sha256=STAGE6_COVERAGE_CACHE_MANIFEST_SHA256,
    )
    observation = direct_env.reset()
    direct_timing = direct_env.last_timing_seconds
    timings["scenario_build"] = direct_timing["scenario_build"]
    timings["coverable_env_init"] = direct_timing["coverable_env_init"]
    timings["reset"] = direct_timing["reset"]
    step_started = time.perf_counter()
    step_result = direct_env.step(direct_env.select_rule_action(observation))
    timings["step"] = time.perf_counter() - step_started
    if not step_result.trainable:
        direct_env.close()
        raise Stage6WorkflowError("Standard direct preflight step was not trainable")
    standard_audit = {
        "catalog_sha256": catalog.sha256,
        "scenario_id": direct_env.scenario_record.scenario_id,
        "prior_shape": list(observation.prior_channels.shape),
        "truth_shape": list(direct_env.scenario_bundle.truth.geometry.shape),
        "local_crop_shape": list(observation.local_crop.shape),
        "candidate_slots": int(observation.frontier_features.shape[0]),
        "coverable_exact": direct_env.coverage_metadata.get("exact") is True,
        "step_trainable": step_result.trainable is True,
        "cache_scope": {
            "catalog_process_cache_hit": True,
            "runtime_mode": STAGE6_COVERAGE_CACHE_RUNTIME_MODE,
            "manifest": {
                "path": STAGE6_COVERAGE_CACHE_MANIFEST_PATH.as_posix(),
                "sha256": STAGE6_COVERAGE_CACHE_MANIFEST_SHA256,
                "size_bytes": STAGE6_COVERAGE_CACHE_MANIFEST_SIZE_BYTES,
                "cache_root": coverage_manifest.cache_root.as_posix(),
                "entry_set_sha256": coverage_manifest.entry_set_sha256,
                "entry_count": len(coverage_manifest.entries),
                "split_counts": dict(coverage_manifest.split_counts),
            },
            "formal_audit": {
                "path": STAGE6_COVERAGE_CACHE_FORMAL_AUDIT_PATH.as_posix(),
                "sha256": STAGE6_COVERAGE_CACHE_FORMAL_AUDIT_SHA256,
                "size_bytes": STAGE6_COVERAGE_CACHE_FORMAL_AUDIT_SIZE_BYTES,
            },
        },
    }

    load_started = time.perf_counter()
    policy = load_stage4_policy_for_standard(
        checkpoint_path=CANONICAL_STAGE4_CHECKPOINT,
        checkpoint_sha256=STAGE4_CHECKPOINT_SHA256,
        policy_state_sha256=STAGE4_POLICY_STATE_SHA256,
        device="cuda",
    )
    timings["cuda_policy_load"] = time.perf_counter() - load_started
    loaded_policy_hash = policy_state_sha256(policy)  # type: ignore[arg-type]
    batch = batch_policy_observations((observation,), device="cuda")
    policy.eval()  # type: ignore[union-attr]
    for _warmup in range(5):
        with torch.inference_mode():
            policy(batch)  # type: ignore[operator]
    torch.cuda.synchronize()
    forward_ms: list[float] = []
    forward_started = time.perf_counter()
    measured_output: PolicyForwardOutput | None = None
    for _measurement in range(20):
        event_start = torch.cuda.Event(enable_timing=True)
        event_end = torch.cuda.Event(enable_timing=True)
        event_start.record()
        with torch.inference_mode():
            measured_output = policy(batch)  # type: ignore[operator]
        event_end.record()
        torch.cuda.synchronize()
        forward_ms.append(float(event_start.elapsed_time(event_end)))
    timings["cuda_forward"] = time.perf_counter() - forward_started
    if not isinstance(measured_output, PolicyForwardOutput):
        direct_env.close()
        raise Stage6WorkflowError("Stage 6 CUDA policy output type drifted")
    output_tensors = tuple(
        getattr(measured_output, field.name)
        for field in fields(measured_output)
        if isinstance(getattr(measured_output, field.name), torch.Tensor)
    )
    outputs_finite = bool(output_tensors) and all(
        bool(torch.isfinite(value).all()) for value in output_tensors
    )

    torch.manual_seed(config.training.seeds[0])
    torch.cuda.manual_seed_all(config.training.seeds[0])
    policy.zero_grad(set_to_none=True)  # type: ignore[union-attr]
    backward_started = time.perf_counter()
    backward_output = policy(batch)  # type: ignore[operator]
    sampled = sample_action(
        backward_output,
        batch.candidate_mask,
        deterministic=False,
    )
    joint_error = float(
        (
            sampled.log_prob_total
            - (sampled.log_prob_frontier + sampled.log_prob_theta)
        )
        .abs()
        .max()
        .detach()
        .cpu()
    )
    loss = -sampled.log_prob_total.mean() + 0.5 * backward_output.value.square().mean()
    if not bool(torch.isfinite(loss)):
        direct_env.close()
        raise Stage6WorkflowError("Stage 6 CUDA joint loss is nonfinite")
    loss.backward()
    torch.cuda.synchronize()
    timings["cuda_backward"] = time.perf_counter() - backward_started
    gradients = tuple(
        parameter.grad
        for parameter in policy.parameters()  # type: ignore[union-attr]
        if parameter.grad is not None
    )
    gradients_finite = bool(gradients) and all(
        bool(torch.isfinite(gradient).all()) for gradient in gradients
    )
    if policy_state_sha256(policy) != loaded_policy_hash:  # type: ignore[arg-type]
        direct_env.close()
        raise Stage6WorkflowError("Stage 6 CUDA preflight mutated policy weights")
    direct_env.close()

    parent_pid = os.getpid()
    specs = _stage6_cached_preflight_env_specs(
        catalog=catalog,
        sampler_seeds=tuple(
            config.training.seeds[0] + index for index in range(config.rollout.num_envs)
        ),
        safety_contract=safety_contract,
        config_sha256=config_sha256,
    )
    spawn_audit = _run_stage6_spawn_preflight(
        specs=specs,
        policy=policy,
        device="cuda",
        collector_contract=CollectorContract(
            config_sha256=config_sha256,
            lineage=authority.identity,
            frontier_extractor_version="stage2_observed_frontier_top_m/v1",
            observation_schema_version=config.observation_schema_version,
            action_space_version=config.action_space_version,
            reward_version=config.reward_version,
            planner_version=config.planner_version,
        ),
        timeout_seconds=900.0,
    )
    preflight_result = spawn_audit["preflight_result"]
    worker_pids = spawn_audit["worker_pids"]
    timings.update(spawn_audit["timings_seconds"])  # type: ignore[arg-type]
    timings["spawn_reset"] = float(preflight_result.timings_seconds["reset"])
    timings["spawn_inference"] = float(
        preflight_result.timings_seconds["inference"]
    )
    timings["spawn_step"] = float(preflight_result.timings_seconds["step"])
    standard_audit["cache_scope"]["cold_spawn"] = {  # type: ignore[index]
        "verification": "spawn_worker_reset/v1",
        "scenario_id": standard_audit["scenario_id"],
        "worker_pid": worker_pids[0],
        "parent_pid": parent_pid,
        "start_method": preflight_result.worker_start_methods[0],
        "cache_hit": True,
        "compute_fallback_called": False,
    }
    collector_audit = {
        "worker_count": len(worker_pids),
        "worker_pids": list(worker_pids),
        "worker_start_methods": list(preflight_result.worker_start_methods),
        "inference_pids": list(preflight_result.inference_pids),
        "parent_pid": parent_pid,
        "trainable_transition_count": preflight_result.trainable_transition_count,
        "snapshot_count": len(preflight_result.snapshot_sha256),
        "policy_state_sha256": preflight_result.policy_state_sha256,
        "all_workers_closed": spawn_audit["all_workers_closed"],
        "residual_child_pids": spawn_audit["residual_child_pids"],
    }

    peak_vram = int(torch.cuda.max_memory_allocated())
    resources_audit, final_decision = _stage6_preflight_resource_audit(
        process_tree_monitor=spawn_audit["process_tree_monitor"],
        peak_vram_bytes=peak_vram,
    )
    cuda_audit = {
        "available": True,
        "device": str(next(policy.parameters()).device),  # type: ignore[union-attr]
        "compute_dtype": "float32",
        "amp_enabled": False,
        "autocast_enabled": bool(torch.is_autocast_enabled("cuda")),
        "checkpoint_sha256": STAGE4_CHECKPOINT_SHA256,
        "policy_state_sha256": loaded_policy_hash,
        "forward_warmup_count": 5,
        "forward_measurement_count": 20,
        "forward_p95_ms": float(np.percentile(np.asarray(forward_ms), 95.0)),
        "outputs_finite": outputs_finite,
        "joint_logprob_finite": math.isfinite(joint_error) and joint_error <= 1e-6,
        "backward_completed": True,
        "gradient_tensor_count": len(gradients),
        "gradients_finite": gradients_finite,
        "peak_vram_bytes": peak_vram,
    }
    audit = {
        "schema_version": "stage6_machine_preflight/v1",
        "passed": final_decision.passed,
        "config_sha256": config_sha256,
        "stage5_gate_sha256": str(authority.identity["gate_sha256"]),
        "environment_identity": environment_identity,
        "environment_sha256": environment_sha256,
        "resources": resources_audit,
        "standard": standard_audit,
        "cuda": cuda_audit,
        "collector": collector_audit,
        "timings_seconds": {name: float(timings[name]) for name in sorted(timings)},
    }
    validated = validate_stage6_machine_preflight_audit(audit)
    authority.require_current("Stage 6 machine preflight authority")
    ArtifactStore(preflight_root).write_json_exclusive("audit.json", validated)
    return validated


@dataclass(frozen=True, slots=True)
class _FileIdentity:
    path: Path
    trusted_base: Path | None
    stat_identity: tuple[int, int, int, int, int]
    sha256: str
    size_bytes: int

    @classmethod
    def _capture_components(
        cls,
        path: Path,
        *,
        base: Path | None,
    ) -> tuple[Path, Path | None, tuple[int, int, int, int, int], bytes]:
        lexical = lexical_absolute(path)
        trusted_base = lexical_absolute(base) if base is not None else None
        try:
            result = secure_read_bytes(
                lexical,
                base=trusted_base,
                label="Stage 6 manifest/authority file",
            )
        except (OSError, PathSecurityError) as exc:
            raise Stage6WorkflowError(
                f"Stage 6 manifest/authority file link or reparse rejection: {exc}"
            ) from exc
        return lexical, trusted_base, result.stat_identity, result.payload

    @classmethod
    def capture(
        cls,
        path: Path,
        *,
        base: Path | None = None,
    ) -> "_FileIdentity":
        lexical, trusted_base, identity, payload = cls._capture_components(
            path,
            base=base,
        )
        return cls(
            path=lexical,
            trusted_base=trusted_base,
            stat_identity=identity,
            sha256=hashlib.sha256(payload).hexdigest(),
            size_bytes=len(payload),
        )

    def require_current(self, label: str) -> None:
        current = self.capture(self.path, base=self.trusted_base)
        if (
            current.stat_identity != self.stat_identity
            or current.sha256 != self.sha256
            or current.size_bytes != self.size_bytes
        ):
            raise Stage6WorkflowError(f"{label} changed")

    @property
    def file_id(self) -> tuple[int, int]:
        """Return the File ID/inode used to reject manifest aliases."""

        return self.stat_identity[:2]

    def read_current_bytes(self, label: str) -> bytes:
        lexical, trusted_base, identity, payload = self._capture_components(
            self.path,
            base=self.trusted_base,
        )
        if (
            lexical != self.path
            or trusted_base != self.trusted_base
            or identity != self.stat_identity
            or hashlib.sha256(payload).hexdigest() != self.sha256
            or len(payload) != self.size_bytes
        ):
            raise Stage6WorkflowError(f"{label} changed")
        return payload


@dataclass(frozen=True, slots=True)
class FrozenStage5AuthorityHandle:
    identity: dict[str, object]
    snapshots: tuple[tuple[str, _FileIdentity], ...]
    stage5_root: Path
    repo_root: Path

    @property
    def evidence_paths(self) -> tuple[Path, ...]:
        """Return gate/approval/review/manifest/checkpoint/index paths for pinning."""

        return tuple(snapshot.path for _name, snapshot in self.snapshots)

    def require_current(self, label: str = "Stage 5 authority") -> None:
        if (
            not self.stage5_root.is_dir()
            or {path.name for path in self.stage5_root.iterdir()}
            != _STAGE5_ROOT_MEMBERS
        ):
            raise Stage6WorkflowError(f"{label} root artifact set changed")
        for name, snapshot in self.snapshots:
            snapshot.require_current(f"{label}/{name}")
        _verify_git_identity(self.repo_root)


def verify_stage5_gate_bytes(
    payload: bytes,
    *,
    repo_root: str | Path,
) -> dict[str, object]:
    gate = _strict_canonical_json(payload, "Stage 5 gate")
    if set(gate) != _GATE_KEYS:
        raise Stage6WorkflowError("Stage 5 gate schema drift")
    if gate.get("schema_version") != "ppo_highres_frontier_stage5_verified_gate/v1":
        raise Stage6WorkflowError("Stage 5 gate schema version drift")
    if gate.get("authorized_next_stage") != STAGE6_STAGE_ID:
        raise Stage6WorkflowError("Stage 5 authorized stage drift")
    if (
        gate.get("state") != "next_stage"
        or gate.get("run_id") != STAGE5_RUN_ID
        or gate.get("commit_sha256") != STAGE5_COMMIT
    ):
        raise Stage6WorkflowError("Stage 5 gate identity drift")
    bindings = gate.get("bindings")
    if not isinstance(bindings, dict) or set(bindings) != _BINDING_KEYS:
        raise Stage6WorkflowError("Stage 5 authority binding schema drift")
    if bindings.get("performance_advantage_established") is not False:
        raise Stage6WorkflowError("Stage 5 performance binding drift")
    expected = {
        "goal_id": "ppo-highres-frontier-map-exploration",
        "stage_id": "ppo_highres_frontier_stage5_fair_baseline_evaluator/v1",
        "run_id": STAGE5_RUN_ID,
        "commit_sha256": STAGE5_COMMIT,
        "commit_tree": STAGE5_TREE,
        "review_sha256": STAGE5_REVIEW_SHA256,
        "manifest_sha256": STAGE5_MANIFEST_SHA256,
        "latest_checkpoint_sha256": STAGE4_CHECKPOINT_SHA256,
        "latest_checkpoint_policy_state_sha256": STAGE4_POLICY_STATE_SHA256,
        "authorized_next_stage": STAGE6_STAGE_ID,
    }
    if any(bindings.get(key) != value for key, value in expected.items()):
        raise Stage6WorkflowError("Stage 5 gate binding drift")
    _validate_gate_history(gate.get("history"), bindings)
    if gate["history"][-1]["bindings"] != bindings:
        raise Stage6WorkflowError("Stage 5 final gate binding drift")
    if hashlib.sha256(payload).hexdigest() != STAGE5_GATE_SHA256:
        raise Stage6WorkflowError("Stage 5 gate SHA-256 drift")
    _verify_git_identity(Path(repo_root).expanduser().resolve())
    return gate


def verify_frozen_stage5_authority(
    *,
    gate_path: str | Path,
    repo_root: str | Path,
) -> FrozenStage5AuthorityHandle:
    gate = Path(gate_path).expanduser().resolve()
    canonical_gate = CANONICAL_STAGE5_GATE.expanduser().resolve()
    if gate != canonical_gate:
        raise Stage6WorkflowError("Stage 5 gate path is not canonical")
    stage5_root = gate.parent
    if (
        not stage5_root.is_dir()
        or {path.name for path in stage5_root.iterdir()} != _STAGE5_ROOT_MEMBERS
    ):
        raise Stage6WorkflowError("Stage 5 authority root artifact set drift")
    repo = Path(repo_root).expanduser().resolve()
    gate_snapshot = _FileIdentity.capture(gate)
    value = verify_stage5_gate_bytes(gate.read_bytes(), repo_root=repo)
    bindings = value["bindings"]
    stage4_checkpoint = CANONICAL_STAGE4_CHECKPOINT.expanduser().resolve()
    checkpoint_manifest = stage4_checkpoint.with_name("manifest.json")
    paths = {
        "gate": gate,
        "approval": stage5_root / "approval.json",
        "review": stage5_root / "review.json",
        "manifest": stage5_root / "manifest.json",
        "checkpoint": stage4_checkpoint,
        "checkpoint_manifest": checkpoint_manifest,
    }
    snapshots = tuple((name, _FileIdentity.capture(path)) for name, path in paths.items())
    by_name = dict(snapshots)
    expected_hashes = {
        "gate": STAGE5_GATE_SHA256,
        "approval": str(bindings["approval_sha256"]),
        "review": STAGE5_REVIEW_SHA256,
        "manifest": STAGE5_MANIFEST_SHA256,
        "checkpoint": STAGE4_CHECKPOINT_SHA256,
    }
    if any(by_name[name].sha256 != digest for name, digest in expected_hashes.items()):
        raise Stage6WorkflowError("Stage 5 authority file hash drift")
    approval = _strict_canonical_json(by_name["approval"].path.read_bytes(), "Stage 5 approval")
    review = _strict_canonical_json(by_name["review"].path.read_bytes(), "Stage 5 review")
    manifest = _strict_canonical_json(by_name["manifest"].path.read_bytes(), "Stage 5 manifest")
    checkpoint_index = _strict_canonical_json(
        by_name["checkpoint_manifest"].path.read_bytes(),
        "Stage 4 checkpoint manifest",
    )
    if (
        approval.get("state") != "approved"
        or review.get("state") != "awaiting_human_approval"
        or review.get("review_status") != "PASS"
        or review.get("spec_verdict") != "PASS"
        or review.get("quality_verdict") != "PASS"
        or review.get("issue_counts", {}).get("critical") != 0
        or review.get("issue_counts", {}).get("important") != 0
    ):
        raise Stage6WorkflowError("Stage 5 approval/review state drift")
    if manifest.get("schema_version") != "sha256_manifest/v1":
        raise Stage6WorkflowError("Stage 5 manifest schema drift")
    if (
        checkpoint_index.get("policy_state_sha256") != STAGE4_POLICY_STATE_SHA256
        or checkpoint_index.get("checkpoint", {}).get("sha256")
        != STAGE4_CHECKPOINT_SHA256
    ):
        raise Stage6WorkflowError("Stage 4 checkpoint lineage drift")
    identity = {
        "schema_version": "stage6_stage5_authority_audit/v1",
        "verified": True,
        "authorized_stage": STAGE6_STAGE_ID,
        "run_id": STAGE5_RUN_ID,
        "commit_sha256": STAGE5_COMMIT,
        "commit_tree": STAGE5_TREE,
        "gate_path": gate.as_posix(),
        "gate_sha256": gate_snapshot.sha256,
        "review_sha256": by_name["review"].sha256,
        "manifest_sha256": by_name["manifest"].sha256,
        "checkpoint_sha256": by_name["checkpoint"].sha256,
        "policy_state_sha256": STAGE4_POLICY_STATE_SHA256,
        "performance_advantage_established": False,
        "history_states": list(_HISTORY_STATES),
    }
    handle = FrozenStage5AuthorityHandle(
        identity=identity,
        snapshots=snapshots,
        stage5_root=stage5_root,
        repo_root=repo,
    )
    handle.require_current()
    return handle


@dataclass(frozen=True, slots=True)
class FinalEvaluationJob:
    split: Literal["test", "unseen"]
    method: str
    episode_index: int
    scenario_id: str
    evaluation_seed: int
    theta_source: str


def build_final_evaluation_plan(
    scenarios: Mapping[str, Sequence[str]],
) -> tuple[FinalEvaluationJob, ...]:
    if set(scenarios) != {"test", "unseen"}:
        raise Stage6WorkflowError("final evaluation split set drifted")
    jobs: list[FinalEvaluationJob] = []
    for split_index, split in enumerate(("test", "unseen")):
        scenario_ids = tuple(scenarios[split])
        if (
            len(scenario_ids) != 64
            or len(set(scenario_ids)) != 64
            or any(not value.startswith(f"{split}/") for value in scenario_ids)
        ):
            raise Stage6WorkflowError("final evaluation requires 64 isolated scenarios")
        for method in FINAL_EVALUATION_METHODS:
            theta_source = (
                "policy_theta_mu/v1"
                if method == "ppo_policy"
                else "candidate_recommended_theta/v1"
            )
            for episode_index, scenario_id in enumerate(scenario_ids):
                jobs.append(
                    FinalEvaluationJob(
                        split=split,  # type: ignore[arg-type]
                        method=method,
                        episode_index=episode_index,
                        scenario_id=scenario_id,
                        evaluation_seed=20260716 + split_index * 64 + episode_index,
                        theta_source=theta_source,
                    )
                )
    return tuple(jobs)


def assert_eval_only_hashes_unchanged(
    before: Mapping[str, str],
    after: Mapping[str, str],
) -> dict[str, str]:
    expected = {
        "checkpoint_sha256",
        "policy_state_sha256",
        "optimizer_state_sha256",
        "config_sha256",
    }
    if set(before) != expected or set(after) != expected:
        raise Stage6WorkflowError("eval-only frozen hash field set drifted")
    if dict(before) != dict(after):
        raise Stage6WorkflowError("eval-only execution mutated frozen state")
    if any(not _is_sha256(value) for value in before.values()):
        raise Stage6WorkflowError("eval-only frozen hash is invalid")
    return dict(before)


@dataclass(frozen=True, slots=True)
class PerformanceDecision:
    established: bool
    claim: str
    ppo_ci95_low: float
    gain_over_cost_ci95_high: float


def decide_performance_advantage(
    *,
    ppo_ci95_low: float,
    gain_over_cost_ci95_high: float,
) -> PerformanceDecision:
    if not all(
        math.isfinite(float(value)) and 0.0 <= float(value) <= 1.0
        for value in (ppo_ci95_low, gain_over_cost_ci95_high)
    ):
        raise Stage6WorkflowError("performance CI bounds are invalid")
    established = float(ppo_ci95_low) > float(gain_over_cost_ci95_high)
    return PerformanceDecision(
        established=established,
        claim="已建立性能优势" if established else "未建立性能优势",
        ppo_ci95_low=float(ppo_ci95_low),
        gain_over_cost_ci95_high=float(gain_over_cost_ci95_high),
    )


def build_machine_acceptance(
    *,
    system_gates_passed: bool,
    performance: PerformanceDecision,
) -> dict[str, object]:
    if not isinstance(performance, PerformanceDecision):
        raise Stage6WorkflowError("performance decision is invalid")
    return {
        "schema_version": "stage6_machine_acceptance/v1",
        "machine_passed": system_gates_passed is True,
        "state": (
            "awaiting_independent_review" if system_gates_passed else "failed"
        ),
        "performance_advantage_established": performance.established,
        "performance_claim": performance.claim,
    }


class Stage6StateJournal:
    _BASE_BINDING_KEYS = frozenset(
        {
            "config_sha256",
            "source_set_sha256",
            "prospective_tree_sha256",
            "data_sha256",
            "environment_identity",
            "environment_sha256",
            "stage5_gate_sha256",
            "formal_run_id",
            "changed_path_set_sha256",
            "review_authorization_record_sha256",
            "authorization_file_sha256",
            "review_identity_sha256",
            "reviewed_prospective_git_tree",
            "frozen_diff_sha256",
            "spec_review_sha256",
            "quality_review_sha256",
            "checkpoint_sha256",
        }
    )
    _SUCCESS_STATES = frozenset(
        {"machine_passed", "awaiting_independent_review"}
    )
    _SUCCESS_BINDING_KEYS = frozenset(
        {
            "preterminal_acceptance_sha256",
            "preterminal_evidence_graph_sha256",
        }
    )
    _COVERAGE_CACHE_BINDING_KEYS = frozenset(
        {
            "coverage_cache_manifest_path",
            "coverage_cache_manifest_sha256",
            "coverage_cache_manifest_size_bytes",
            "coverage_cache_root",
            "coverage_cache_entry_set_sha256",
            "coverage_cache_runtime_mode",
            "coverage_cache_formal_audit_sha256",
            "coverage_cache_formal_audit_size_bytes",
        }
    )
    _CURRENT_BINDING_KEYS = _BASE_BINDING_KEYS | _COVERAGE_CACHE_BINDING_KEYS

    def __init__(self, path: str | Path) -> None:
        self.path = lexical_absolute(path)
        self._durable = DurableJsonl(self.path)

    def append(
        self,
        state: str,
        bindings: Mapping[str, object],
    ) -> dict[str, object]:
        if not isinstance(state, str) or not state:
            raise Stage6WorkflowError("state journal state is invalid")
        if not self._bindings_valid(bindings, state=state):
            raise Stage6WorkflowError("state journal binding drift")
        records = self.verify()
        previous = (
            str(records[-1]["record_hash"])
            if records
            else ABSENT_AUTHORITY_SHA256
        )
        event = {
            "state": state,
            "previous_record_hash": previous,
            "bindings": dict(bindings),
        }
        record = {**event, "record_hash": _canonical_sha256(event)}
        try:
            self._durable.append(record)
        except (DurableJsonlError, OSError) as exc:
            raise Stage6WorkflowError("state journal durable append failed") from exc
        return record

    def verify(self) -> tuple[dict[str, object], ...]:
        try:
            payload = self._durable.recover_and_snapshot()
        except (DurableJsonlError, OSError) as exc:
            raise Stage6WorkflowError(
                "state journal hash chain or committed snapshot failed"
            ) from exc
        return self.verify_snapshot_bytes(payload)

    @classmethod
    def verify_snapshot_bytes(
        cls,
        payload: bytes,
    ) -> tuple[dict[str, object], ...]:
        """Verify an already-bound canonical journal snapshot without path I/O."""

        if type(payload) is not bytes:
            raise Stage6WorkflowError("state journal snapshot bytes drifted")
        if not payload:
            return ()
        previous = ABSENT_AUTHORITY_SHA256
        records: list[dict[str, object]] = []
        try:
            lines = payload.splitlines(keepends=True)
            for line in lines:
                value = json.loads(line.decode("utf-8"))
                if not isinstance(value, dict):
                    raise ValueError("journal row is not an object")
                canonical = (
                    json.dumps(
                        value,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n"
                ).encode("utf-8")
                if line != canonical:
                    raise ValueError("journal row is not canonical")
                event = {
                    "state": value.get("state"),
                    "previous_record_hash": value.get("previous_record_hash"),
                    "bindings": value.get("bindings"),
                }
                bindings = event["bindings"]
                if (
                    not isinstance(event["state"], str)
                    or not event["state"]
                    or not isinstance(bindings, Mapping)
                    or not cls._bindings_valid(
                        bindings,
                        state=str(event["state"]),
                    )
                ):
                    raise Stage6WorkflowError(
                        "Stage 6 state journal binding drift"
                    )
                if (
                    set(value) != {*event, "record_hash"}
                    or event["previous_record_hash"] != previous
                    or value["record_hash"] != _canonical_sha256(event)
                ):
                    raise ValueError("journal hash chain drift")
                records.append(value)
                previous = str(value["record_hash"])
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            raise Stage6WorkflowError("Stage 6 state journal hash chain drift") from exc
        return tuple(records)

    @classmethod
    def _bindings_valid(
        cls,
        bindings: Mapping[str, object],
        *,
        state: str,
    ) -> bool:
        success_keys = (
            cls._SUCCESS_BINDING_KEYS
            if state in cls._SUCCESS_STATES
            else frozenset()
        )
        binding_keys = frozenset(set(bindings) - set(success_keys))
        if (
            set(bindings) & set(cls._SUCCESS_BINDING_KEYS) != set(success_keys)
            or binding_keys
            not in {cls._BASE_BINDING_KEYS, cls._CURRENT_BINDING_KEYS}
        ):
            return False
        environment = bindings.get("environment_identity")
        try:
            environment_valid = validate_stage6_environment_identity(
                environment  # type: ignore[arg-type]
            )
        except (Stage6WorkflowError, TypeError, ValueError):
            return False
        tree = bindings.get("reviewed_prospective_git_tree")
        try:
            formal_run_id_valid = (
                validate_stage6_formal_run_id(
                    str(bindings.get("formal_run_id"))
                )
                == bindings.get("formal_run_id")
            )
        except (Stage6WorkflowError, TypeError, ValueError):
            formal_run_id_valid = False
        hash_fields = (binding_keys | success_keys) - {
            "environment_identity",
            "formal_run_id",
            "reviewed_prospective_git_tree",
            "coverage_cache_manifest_path",
            "coverage_cache_manifest_size_bytes",
            "coverage_cache_root",
            "coverage_cache_runtime_mode",
            "coverage_cache_formal_audit_size_bytes",
        }
        base_valid = (
            all(_is_sha256(bindings.get(key)) for key in hash_fields)
            and bindings.get("environment_sha256")
            == stage6_environment_sha256(environment_valid)
            and formal_run_id_valid
            and isinstance(tree, str)
            and len(tree) == 40
            and all(character in "0123456789abcdef" for character in tree)
            and bindings.get("prospective_tree_sha256")
            == hashlib.sha256(tree.encode("ascii")).hexdigest()
        )
        if not base_valid or binding_keys == cls._BASE_BINDING_KEYS:
            return base_valid
        return (
            isinstance(bindings.get("coverage_cache_manifest_path"), str)
            and bool(bindings.get("coverage_cache_manifest_path"))
            and type(bindings.get("coverage_cache_manifest_size_bytes")) is int
            and int(bindings["coverage_cache_manifest_size_bytes"]) > 0
            and isinstance(bindings.get("coverage_cache_root"), str)
            and bool(bindings.get("coverage_cache_root"))
            and bindings.get("coverage_cache_runtime_mode")
            == "persistent_exact_manifest_read_only/v1"
            and type(bindings.get("coverage_cache_formal_audit_size_bytes")) is int
            and int(bindings["coverage_cache_formal_audit_size_bytes"]) > 0
        )


@dataclass(frozen=True, slots=True)
class _ManifestDirectorySnapshot:
    relative: str
    stat_identity: tuple[int, int, int, int]
    members: tuple[tuple[str, str], ...]


@dataclass(frozen=True, slots=True)
class _Stage6ManifestGraph:
    relative_paths: tuple[str, ...]
    directories: tuple[_ManifestDirectorySnapshot, ...]


@dataclass(frozen=True, slots=True)
class Stage6ManifestHandle:
    stage_root: Path
    repo_root: Path
    source_identity: dict[str, object]
    snapshots: tuple[tuple[str, _FileIdentity], ...]
    manifest_graph: _Stage6ManifestGraph

    def require_current(self, label: str = "Stage 6 manifest graph") -> None:
        graph_before = _capture_stage6_manifest_graph(
            self.stage_root,
            require_manifest=True,
        )
        if graph_before != self.manifest_graph:
            raise Stage6WorkflowError(f"{label} membership changed")
        for relative, snapshot in self.snapshots:
            snapshot.require_current(f"{label}/{relative}")
        if stage6_source_identity(self.repo_root) != self.source_identity:
            raise Stage6WorkflowError(f"{label} current source changed")
        graph_after = _capture_stage6_manifest_graph(
            self.stage_root,
            require_manifest=True,
        )
        if graph_after != self.manifest_graph:
            raise Stage6WorkflowError(f"{label} membership changed")


def stage6_source_identity(repo_root: str | Path) -> dict[str, object]:
    repo = Path(repo_root).expanduser().resolve()
    digest = hashlib.sha256()
    rows: list[dict[str, object]] = []
    for relative in STAGE6_SOURCE_PATHS:
        path = (repo / relative).resolve()
        try:
            path.relative_to(repo)
        except ValueError as exc:
            raise Stage6WorkflowError("Stage 6 source path escaped repository") from exc
        if not path.is_file():
            raise Stage6WorkflowError(f"Stage 6 source path missing: {relative}")
        payload = path.read_bytes()
        sha256 = hashlib.sha256(payload).hexdigest()
        rows.append({"path": relative, "sha256": sha256, "size_bytes": len(payload)})
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(payload)
        digest.update(b"\0")
    return {
        "schema_version": "stage6_current_source_set/v1",
        "source_set_sha256": digest.hexdigest(),
        "paths": rows,
    }


def _stage6_input_pin_requests(
    *,
    repo_root: str | Path,
    config_path: str | Path,
    stage5_authority: FrozenStage5AuthorityHandle,
    review_authorization_handle: Stage6ReviewAuthorizationHandle,
    planning_warm_start_path: str | Path | None = None,
    planning_child_source_repair_path: str | Path | None = None,
    planning_child_source_repair_continuation_path: (
        str | Path | None
    ) = None,
) -> tuple[tuple[str, Path], ...]:
    """Enumerate every reviewed path whose bytes may participate in Stage 6."""

    repo = lexical_absolute(repo_root)
    requests: list[tuple[str, Path]] = [
        ("config:canonical", lexical_absolute(config_path)),
        (
            "coverage-cache:manifest",
            lexical_absolute(STAGE6_COVERAGE_CACHE_MANIFEST_PATH),
        ),
    ]
    requests.extend(
        (f"source:{relative}", lexical_absolute(repo / relative))
        for relative in STAGE6_SOURCE_PATHS
    )
    requests.extend(
        (f"data:{label}", lexical_absolute(path))
        for label, path in STAGE6_DATA_INPUT_PATHS
    )
    requests.extend(
        (f"stage5-authority:{name}", lexical_absolute(snapshot.path))
        for name, snapshot in stage5_authority.snapshots
    )
    requests.extend(
        (
            f"review-authorization:{index}:{path.name}",
            lexical_absolute(path),
        )
        for index, path in enumerate(
            review_authorization_handle.evidence_paths,
            start=1,
        )
    )
    if planning_warm_start_path is not None:
        requests.append(
            (
                "planning-warm-start:artifact",
                lexical_absolute(planning_warm_start_path),
            )
        )
        from lunar_exploration_ppo.workflows.stage6_planning_warm_start import (
            PARENT_STAGE_ROOT,
            parent_u74_input_pin_requests,
        )

        requests.extend(
            (label, lexical_absolute(path))
            for label, path in parent_u74_input_pin_requests(
                PARENT_STAGE_ROOT
            )
        )
    if (
        planning_child_source_repair_continuation_path is not None
        and planning_child_source_repair_path is None
    ):
        raise Stage6WorkflowError(
            "Stage 6 planning child continuation requires its parent"
        )
    if planning_child_source_repair_path is not None:
        if planning_child_source_repair_continuation_path is not None:
            from lunar_exploration_ppo.workflows.stage6_planning_child_source_repair_continuation import (
                Stage6PlanningChildSourceRepairContinuationError,
                planning_child_source_repair_chain_input_pin_requests,
            )

            try:
                requests.extend(
                    planning_child_source_repair_chain_input_pin_requests(
                        planning_child_source_repair_path,
                        planning_child_source_repair_continuation_path,
                    )
                )
            except (
                Stage6PlanningChildSourceRepairContinuationError
            ) as exc:
                raise Stage6WorkflowError(
                    "Stage 6 planning child continuation input pin failed"
                ) from exc
            return tuple(requests)
        from lunar_exploration_ppo.workflows.stage6_planning_child_source_repair import (
            Stage6PlanningChildSourceRepairError,
            planning_child_source_repair_input_pin_requests,
        )

        try:
            requests.extend(
                planning_child_source_repair_input_pin_requests(
                    planning_child_source_repair_path
                )
            )
        except Stage6PlanningChildSourceRepairError as exc:
            raise Stage6WorkflowError(
                "Stage 6 planning child source-repair input pin failed"
            ) from exc
    return tuple(requests)


def validate_stage6_environment_identity(
    value: Mapping[str, object],
) -> dict[str, object]:
    expected = {
        "schema_version",
        "python_version",
        "python_implementation",
        "os_name",
        "platform_system",
        "platform_machine",
        "numpy_version",
        "torch_version",
        "torch_cuda_version",
        "cudnn_version",
        "cuda_available",
        "cuda_device_count",
        "cuda_current_device",
        "cuda_device_name",
        "cuda_compute_capability",
        "cuda_total_vram_bytes",
        "compute_dtype",
        "amp_enabled",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise Stage6WorkflowError("Stage 6 environment identity field set drifted")
    string_fields = (
        "python_version",
        "python_implementation",
        "os_name",
        "platform_system",
        "platform_machine",
        "numpy_version",
        "torch_version",
        "torch_cuda_version",
        "cuda_device_name",
    )
    capability = value["cuda_compute_capability"]
    if (
        value["schema_version"] != "stage6_environment_identity/v1"
        or any(not isinstance(value[field], str) or not value[field] for field in string_fields)
        or type(value["cudnn_version"]) is not int
        or int(value["cudnn_version"]) <= 0
        or value["cuda_available"] is not True
        or type(value["cuda_device_count"]) is not int
        or int(value["cuda_device_count"]) <= 0
        or type(value["cuda_current_device"]) is not int
        or not 0
        <= int(value["cuda_current_device"])
        < int(value["cuda_device_count"])
        or not isinstance(capability, list)
        or len(capability) != 2
        or any(type(part) is not int or part < 0 for part in capability)
        or type(value["cuda_total_vram_bytes"]) is not int
        or int(value["cuda_total_vram_bytes"]) <= 0
        or value["compute_dtype"] != "float32"
        or value["amp_enabled"] is not False
    ):
        raise Stage6WorkflowError("Stage 6 environment identity drifted")
    return dict(value)


def stage6_environment_identity() -> dict[str, object]:
    """Capture the fail-closed runtime identity used by every Stage 6 artifact."""

    import platform

    import numpy as np
    import torch

    if not torch.cuda.is_available():
        raise Stage6WorkflowError("Stage 6 environment identity requires CUDA")
    try:
        device_count = int(torch.cuda.device_count())
        current_device = int(torch.cuda.current_device())
        capability = torch.cuda.get_device_capability(current_device)
        total_vram_bytes = int(
            torch.cuda.get_device_properties(current_device).total_memory
        )
        cudnn_version = torch.backends.cudnn.version()
        cuda_version = torch.version.cuda
        amp_enabled = bool(torch.is_autocast_enabled("cuda"))
        identity = {
            "schema_version": "stage6_environment_identity/v1",
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "os_name": os.name,
            "platform_system": platform.system(),
            "platform_machine": platform.machine(),
            "numpy_version": str(np.__version__),
            "torch_version": str(torch.__version__),
            "torch_cuda_version": str(cuda_version) if cuda_version is not None else "",
            "cudnn_version": cudnn_version,
            "cuda_available": True,
            "cuda_device_count": device_count,
            "cuda_current_device": current_device,
            "cuda_device_name": str(torch.cuda.get_device_name(current_device)),
            "cuda_compute_capability": [int(capability[0]), int(capability[1])],
            "cuda_total_vram_bytes": total_vram_bytes,
            "compute_dtype": "float32",
            "amp_enabled": amp_enabled,
        }
    except (AssertionError, RuntimeError, TypeError, ValueError) as exc:
        raise Stage6WorkflowError("Stage 6 CUDA environment identity failed") from exc
    return validate_stage6_environment_identity(identity)


def stage6_environment_sha256(value: Mapping[str, object]) -> str:
    return _canonical_sha256(validate_stage6_environment_identity(value))


def _stage6_reviewed_source_git_identity(
    repo_root: str | Path,
    *,
    base_commit: str,
    reviewed_paths: Sequence[str],
) -> dict[str, object]:
    """Build a tree from the historical base plus only reviewed source bytes."""

    repo = Path(repo_root).expanduser().resolve()
    normalized_paths: list[str] = []
    for value in reviewed_paths:
        if type(value) is not str:
            raise Stage6WorkflowError("Stage 6 reviewed source path is invalid")
        relative = Path(value)
        normalized = relative.as_posix()
        if (
            not normalized
            or normalized != value.replace("\\", "/")
            or relative.is_absolute()
            or ".." in relative.parts
        ):
            raise Stage6WorkflowError("Stage 6 reviewed source path is invalid")
        try:
            (repo / relative).resolve().relative_to(repo)
        except ValueError as exc:
            raise Stage6WorkflowError(
                "Stage 6 reviewed source path escaped repository"
            ) from exc
        normalized_paths.append(normalized)
    normalized_paths = sorted(normalized_paths)
    if (
        not normalized_paths
        or len(normalized_paths) != len(set(normalized_paths))
    ):
        raise Stage6WorkflowError(
            "Stage 6 reviewed source path set is empty or duplicated"
        )

    def run_git(
        arguments: Sequence[str],
        *,
        label: str,
        environment: Mapping[str, str] | None = None,
        binary: bool = False,
    ) -> bytes | str:
        completed = subprocess.run(
            ["git", "-C", str(repo), *arguments],
            capture_output=True,
            check=False,
            text=not binary,
            encoding=None if binary else "utf-8",
            env=None if environment is None else dict(environment),
        )
        if completed.returncode != 0:
            raise Stage6WorkflowError(f"Stage 6 {label} failed")
        return completed.stdout

    def require_empty_real_index() -> None:
        completed = subprocess.run(
            ["git", "-C", str(repo), "diff", "--cached", "--quiet", "--exit-code"],
            capture_output=True,
            check=False,
            text=False,
        )
        if completed.returncode != 0:
            raise Stage6WorkflowError("Stage 6 real Git index must be empty")

    require_empty_real_index()
    diagnostic_head = str(
        run_git(
            ["rev-parse", "--verify", "HEAD^{commit}"],
            label="HEAD diagnostic",
        )
    ).strip()
    run_git(
        ["cat-file", "-e", f"{base_commit}^{{commit}}"],
        label="historical base commit verification",
    )

    index_root = Path("D:/xunce/tmp/ppo_frontier/git-index").resolve()
    if index_root.drive.upper() != "D:":
        raise Stage6WorkflowError(
            "Stage 6 temporary Git index must be on D drive"
        )
    index_root.mkdir(parents=True, exist_ok=True)
    index_path = (
        index_root / f"stage6-reviewed-source-{uuid.uuid4().hex}.index"
    ).resolve()
    try:
        index_path.relative_to(index_root)
    except ValueError as exc:
        raise Stage6WorkflowError(
            "Stage 6 temporary Git index escaped its root"
        ) from exc
    environment = os.environ.copy()
    environment["GIT_INDEX_FILE"] = str(index_path)
    try:
        run_git(
            ["read-tree", base_commit],
            label="reviewed source read-tree",
            environment=environment,
        )
        run_git(
            ["add", "-A", "--", *normalized_paths],
            label="reviewed source add",
            environment=environment,
        )
        tree = str(
            run_git(
                ["write-tree"],
                label="reviewed source write-tree",
                environment=environment,
            )
        ).strip()
        output = bytes(
            run_git(
                [
                    "diff-tree",
                    "--no-commit-id",
                    "--name-only",
                    "-r",
                    "-z",
                    "--no-renames",
                    base_commit,
                    tree,
                ],
                label="reviewed source path set",
                environment=environment,
                binary=True,
            )
        )
    finally:
        lock_path = Path(f"{index_path}.lock")
        if lock_path.is_file():
            lock_path.unlink()
        if index_path.is_file():
            index_path.unlink()
    require_empty_real_index()
    try:
        changed_paths = sorted(
            value.decode("utf-8", errors="strict").replace("\\", "/")
            for value in output.split(b"\0")
            if value
        )
    except UnicodeDecodeError as exc:
        raise Stage6WorkflowError(
            "Stage 6 reviewed source path set is not UTF-8"
        ) from exc
    unexpected = set(changed_paths) - set(normalized_paths)
    if unexpected:
        raise Stage6WorkflowError(
            "Stage 6 reviewed source tree contains an unreviewed path"
        )
    return {
        "schema_version": "stage6_reviewed_source_git_tree/v1",
        "base_commit": base_commit,
        "diagnostic_head_commit": diagnostic_head,
        "prospective_git_tree": tree,
        "changed_paths": changed_paths,
        "changed_path_set_sha256": _canonical_sha256(changed_paths),
        "real_index_empty": True,
    }


def stage6_execution_identity(
    *,
    repo_root: str | Path,
    config_path: str | Path,
    effective_config_bytes: bytes | None = None,
) -> dict[str, object]:
    """Bind current bytes and the reviewed Stage 6 source tree."""

    from lunar_exploration_ppo.configs.stage6 import load_stage6_config
    from lunar_exploration_ppo.env.standard_training import build_standard_catalog

    repo = Path(repo_root).expanduser().resolve()
    config_file = Path(config_path).expanduser().resolve()
    canonical_config = (
        repo / "configs/ppo_highres_frontier_stage6_v1.json"
    ).resolve()
    if config_file != canonical_config:
        raise Stage6WorkflowError("Stage 6 execution config path is not canonical")
    config_bytes = config_file.read_bytes()
    config = load_stage6_config(config_file)
    config_identity_bytes = config_bytes
    if effective_config_bytes is not None:
        from lunar_exploration_ppo.workflows.stage6_planning_warm_start import (
            Stage6PlanningWarmStartError,
            parse_planning_effective_config_bytes,
        )

        try:
            effective = parse_planning_effective_config_bytes(
                effective_config_bytes
            )
        except Stage6PlanningWarmStartError as exc:
            raise Stage6WorkflowError(
                "Stage 6 effective config identity is invalid"
            ) from exc
        if (
            effective.base_config_sha256
            != hashlib.sha256(config_bytes).hexdigest()
            or effective.base_config != config
        ):
            raise Stage6WorkflowError(
                "Stage 6 effective/base config identity drifted"
            )
        config_identity_bytes = effective.effective_config_bytes
    source = stage6_source_identity(repo)
    environment_identity = stage6_environment_identity()
    environment_sha256 = stage6_environment_sha256(environment_identity)
    try:
        git_identity = _stage6_reviewed_source_git_identity(
            repo,
            base_commit=STAGE5_COMMIT,
            reviewed_paths=STAGE6_SOURCE_PATHS,
        )
    except Exception as exc:
        raise Stage6WorkflowError("Stage 6 prospective Git identity failed") from exc
    changed_paths = sorted(str(path) for path in git_identity["changed_paths"])
    if changed_paths != sorted(STAGE6_SOURCE_PATHS):
        raise Stage6WorkflowError("Stage 6 prospective changed path set drift")

    data_files: list[dict[str, object]] = []
    for label, path_value, expected_size, expected_sha256 in (
        (
            "dem",
            config.data.dem_path,
            config.data.dem_bytes,
            config.data.dem_sha256,
        ),
        (
            "slope_provenance",
            config.data.slope_provenance_path,
            config.data.slope_bytes,
            config.data.slope_sha256,
        ),
    ):
        path = Path(path_value).expanduser().resolve()
        digest = hashlib.sha256()
        size = 0
        try:
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
                    size += len(chunk)
        except OSError as exc:
            raise Stage6WorkflowError(f"Stage 6 {label} data is unreadable") from exc
        if size != expected_size or digest.hexdigest() != expected_sha256:
            raise Stage6WorkflowError(f"Stage 6 {label} data identity drift")
        data_files.append(
            {
                "label": label,
                "path": path.as_posix(),
                "size_bytes": size,
                "sha256": digest.hexdigest(),
            }
        )
    catalog = build_standard_catalog(verify_hashes=True)
    data_identity = {
        "schema_version": "stage6_data_catalog_identity/v1",
        "files": data_files,
        "catalog_sha256": catalog.sha256,
    }
    prospective_tree = str(git_identity["prospective_git_tree"])
    return {
        "schema_version": "stage6_execution_identity/v1",
        "base_commit": STAGE5_COMMIT,
        "head_commit": STAGE5_COMMIT,
        "prospective_git_tree": prospective_tree,
        "prospective_tree_sha256": hashlib.sha256(
            prospective_tree.encode("ascii")
        ).hexdigest(),
        "changed_paths": changed_paths,
        "changed_path_set_sha256": str(
            git_identity["changed_path_set_sha256"]
        ),
        "real_index_empty": git_identity["real_index_empty"],
        "config_sha256": hashlib.sha256(config_identity_bytes).hexdigest(),
        "source_set_sha256": source["source_set_sha256"],
        "data_sha256": _canonical_sha256(data_identity),
        "catalog_sha256": catalog.sha256,
        "environment_identity": environment_identity,
        "environment_sha256": environment_sha256,
        "source_identity": source,
        "data_identity": data_identity,
    }


def _load_stage6_source_repair_for_current_identity(
    *,
    stage: Path,
    current_execution_identity: Mapping[str, object],
    amendment_payload: bytes | None = None,
    continuation_payload: bytes | None = None,
    supplement_payload: bytes | None = None,
    closure_payload: bytes | None = None,
    frontier_recovery_payload: bytes | None = None,
    sensor_acceleration_payload: bytes | None = None,
) -> object | None:
    from lunar_exploration_ppo.workflows.stage6_source_repair import (
        SOURCE_REPAIR_AMENDMENT_NAME,
        SOURCE_REPAIR_CONTINUATION_NAME,
        SOURCE_REPAIR_CLOSURE_NAME,
        SOURCE_REPAIR_FRONTIER_RECOVERY_NAME,
        SOURCE_REPAIR_SENSOR_ACCELERATION_NAME,
        SOURCE_REPAIR_SUPPLEMENT_NAME,
        Stage6SourceRepairError,
        load_stage6_source_repair_context,
    )

    amendment_path = stage / SOURCE_REPAIR_AMENDMENT_NAME
    continuation_path = stage / SOURCE_REPAIR_CONTINUATION_NAME
    supplement_path = stage / SOURCE_REPAIR_SUPPLEMENT_NAME
    closure_path = stage / SOURCE_REPAIR_CLOSURE_NAME
    frontier_recovery_path = stage / SOURCE_REPAIR_FRONTIER_RECOVERY_NAME
    sensor_acceleration_path = stage / SOURCE_REPAIR_SENSOR_ACCELERATION_NAME
    if amendment_payload is None:
        if not os.path.lexists(amendment_path):
            if (
                continuation_payload is not None
                or supplement_payload is not None
                or closure_payload is not None
                or frontier_recovery_payload is not None
                or sensor_acceleration_payload is not None
                or os.path.lexists(continuation_path)
                or os.path.lexists(supplement_path)
                or os.path.lexists(closure_path)
                or os.path.lexists(frontier_recovery_path)
                or os.path.lexists(sensor_acceleration_path)
            ):
                raise Stage6WorkflowError(
                    "Stage 6 source-repair successor has no primary amendment"
                )
            return None
        try:
            amendment_payload = amendment_path.read_bytes()
        except OSError as exc:
            raise Stage6WorkflowError(
                "Stage 6 source-repair amendment is unreadable"
            ) from exc
    if continuation_payload is None and os.path.lexists(continuation_path):
        try:
            continuation_payload = continuation_path.read_bytes()
        except OSError as exc:
            raise Stage6WorkflowError(
                "Stage 6 source-repair continuation is unreadable"
            ) from exc
    if continuation_payload is None and (
        supplement_payload is not None
        or closure_payload is not None
        or frontier_recovery_payload is not None
        or sensor_acceleration_payload is not None
        or os.path.lexists(supplement_path)
        or os.path.lexists(closure_path)
        or os.path.lexists(frontier_recovery_path)
        or os.path.lexists(sensor_acceleration_path)
    ):
        raise Stage6WorkflowError(
            "Stage 6 source-repair supplement has no continuation"
        )
    if supplement_payload is None and os.path.lexists(supplement_path):
        try:
            supplement_payload = supplement_path.read_bytes()
        except OSError as exc:
            raise Stage6WorkflowError(
                "Stage 6 source-repair supplement is unreadable"
            ) from exc
    if supplement_payload is None and (
        closure_payload is not None
        or frontier_recovery_payload is not None
        or sensor_acceleration_payload is not None
        or os.path.lexists(closure_path)
        or os.path.lexists(frontier_recovery_path)
        or os.path.lexists(sensor_acceleration_path)
    ):
        raise Stage6WorkflowError(
            "Stage 6 source-repair closure has no supplement"
        )
    if closure_payload is None and os.path.lexists(closure_path):
        try:
            closure_payload = closure_path.read_bytes()
        except OSError as exc:
            raise Stage6WorkflowError(
                "Stage 6 source-repair closure is unreadable"
            ) from exc
    if closure_payload is None and (
        frontier_recovery_payload is not None
        or sensor_acceleration_payload is not None
        or os.path.lexists(frontier_recovery_path)
        or os.path.lexists(sensor_acceleration_path)
    ):
        raise Stage6WorkflowError(
            "Stage 6 source-repair frontier recovery has no closure"
        )
    if frontier_recovery_payload is None and os.path.lexists(
        frontier_recovery_path
    ):
        try:
            frontier_recovery_payload = frontier_recovery_path.read_bytes()
        except OSError as exc:
            raise Stage6WorkflowError(
                "Stage 6 source-repair frontier recovery is unreadable"
            ) from exc
    if frontier_recovery_payload is None and (
        sensor_acceleration_payload is not None
        or os.path.lexists(sensor_acceleration_path)
    ):
        raise Stage6WorkflowError(
            "Stage 6 source-repair sensor acceleration has no frontier recovery"
        )
    if sensor_acceleration_payload is None and os.path.lexists(
        sensor_acceleration_path
    ):
        try:
            sensor_acceleration_payload = sensor_acceleration_path.read_bytes()
        except OSError as exc:
            raise Stage6WorkflowError(
                "Stage 6 source-repair sensor acceleration is unreadable"
            ) from exc
    source_payload = (
        sensor_acceleration_payload
        if sensor_acceleration_payload is not None
        else (
            frontier_recovery_payload
            if frontier_recovery_payload is not None
            else (
                closure_payload
                if closure_payload is not None
                else (
                    supplement_payload
                    if supplement_payload is not None
                    else (
                        continuation_payload
                        if continuation_payload is not None
                        else amendment_payload
                    )
                )
            )
        )
    )
    source_label = (
        "Stage 6 source-repair sensor acceleration"
        if sensor_acceleration_payload is not None
        else (
            "Stage 6 source-repair frontier recovery"
            if frontier_recovery_payload is not None
            else (
                "Stage 6 source-repair closure"
                if closure_payload is not None
                else (
                    "Stage 6 source-repair supplement"
                    if supplement_payload is not None
                    else (
                        "Stage 6 source-repair continuation"
                        if continuation_payload is not None
                        else "Stage 6 source-repair amendment"
                    )
                )
            )
        )
    )
    source_value = _strict_canonical_json(source_payload, source_label)
    current = source_value.get("current")
    if not isinstance(current, Mapping):
        raise Stage6WorkflowError(
            f"{source_label} current identity is missing"
        )
    current_review = current.get("verified_review_authorization")
    current_immutable = current.get("immutable_bindings")
    if not isinstance(current_review, Mapping) or not isinstance(
        current_immutable, Mapping
    ):
        raise Stage6WorkflowError(
            f"{source_label} current authority is missing"
        )
    try:
        context = load_stage6_source_repair_context(
            stage_root=stage,
            current_execution_identity=current_execution_identity,
            current_verified_review_authorization=current_review,
            current_immutable_bindings=current_immutable,
        )
    except Stage6SourceRepairError as exc:
        raise Stage6WorkflowError(
            "Stage 6 source-repair amendment drifted"
        ) from exc
    if context is None:
        raise Stage6WorkflowError("Stage 6 source-repair amendment disappeared")
    if (
        context.amendment_sha256 != hashlib.sha256(amendment_payload).hexdigest()
        or context.amendment_size_bytes != len(amendment_payload)
        or (
            context.continuation_sha256,
            context.continuation_size_bytes,
            context.supplement_sha256,
            context.supplement_size_bytes,
            context.closure_sha256,
            context.closure_size_bytes,
            context.frontier_recovery_sha256,
            context.frontier_recovery_size_bytes,
            context.sensor_acceleration_sha256,
            context.sensor_acceleration_size_bytes,
        )
        != (
            (
                hashlib.sha256(continuation_payload).hexdigest()
                if continuation_payload is not None
                else None
            ),
            len(continuation_payload) if continuation_payload is not None else None,
            (
                hashlib.sha256(supplement_payload).hexdigest()
                if supplement_payload is not None
                else None
            ),
            len(supplement_payload) if supplement_payload is not None else None,
            (
                hashlib.sha256(closure_payload).hexdigest()
                if closure_payload is not None
                else None
            ),
            len(closure_payload) if closure_payload is not None else None,
            (
                hashlib.sha256(frontier_recovery_payload).hexdigest()
                if frontier_recovery_payload is not None
                else None
            ),
            (
                len(frontier_recovery_payload)
                if frontier_recovery_payload is not None
                else None
            ),
            (
                hashlib.sha256(sensor_acceleration_payload).hexdigest()
                if sensor_acceleration_payload is not None
                else None
            ),
            (
                len(sensor_acceleration_payload)
                if sensor_acceleration_payload is not None
                else None
            ),
        )
    ):
        raise Stage6WorkflowError("Stage 6 source-repair artifact binding drifted")
    return context


def _stage6_lineage_review_authorization(
    stage: Path,
    *,
    payload: bytes | None = None,
    repo_root: str | Path | None = None,
    source_repair_amendment_payload: bytes | None = None,
    source_repair_continuation_payload: bytes | None = None,
    source_repair_supplement_payload: bytes | None = None,
    source_repair_closure_payload: bytes | None = None,
    source_repair_frontier_recovery_payload: bytes | None = None,
    source_repair_sensor_acceleration_payload: bytes | None = None,
) -> dict[str, object]:
    if payload is None:
        try:
            payload = (stage / "lineage_audit.json").read_bytes()
        except OSError as exc:
            raise Stage6WorkflowError(
                "Stage 6 review authorization lineage is unreadable"
            ) from exc
    lineage = _strict_canonical_json(payload, "Stage 6 lineage audit")
    lineage_fields = {
        "schema_version",
        "execution_identity",
        "immutable_bindings",
        "verified_review_authorization",
    }
    lineage_schema = lineage.get("schema_version")
    if lineage_schema == "stage6_lineage_audit/v3":
        lineage_fields.add("planning_warm_start")
    if (
        set(lineage) != lineage_fields
        or lineage_schema
        not in {
            "stage6_lineage_audit/v2",
            "stage6_lineage_audit/v3",
        }
        or not isinstance(lineage.get("execution_identity"), Mapping)
        or not isinstance(lineage.get("immutable_bindings"), Mapping)
        or not isinstance(lineage.get("verified_review_authorization"), Mapping)
        or (
            lineage_schema == "stage6_lineage_audit/v3"
            and not isinstance(
                lineage.get("planning_warm_start"),
                Mapping,
            )
        )
    ):
        raise Stage6WorkflowError(
            "Stage 6 review authorization lineage schema drifted"
        )
    verified = validate_stage6_verified_review_authorization(
        lineage["verified_review_authorization"],  # type: ignore[arg-type]
        execution_identity=lineage["execution_identity"],  # type: ignore[arg-type]
        formal_run_id=str(
            lineage["verified_review_authorization"].get("formal_run_id")  # type: ignore[union-attr]
        ),
    )
    immutable = lineage["immutable_bindings"]
    assert isinstance(immutable, Mapping)
    expected = {
        "formal_run_id": verified["formal_run_id"],
        "changed_path_set_sha256": verified["changed_path_set_sha256"],
        "review_authorization_record_sha256": _canonical_sha256(verified),
        "authorization_file_sha256": verified["authorization_file_sha256"],
        "review_identity_sha256": verified["review_identity_sha256"],
        "reviewed_prospective_git_tree": verified[
            "reviewed_prospective_git_tree"
        ],
        "frozen_diff_sha256": verified["frozen_diff_sha256"],
        "spec_review_sha256": verified["spec_review_sha256"],
        "quality_review_sha256": verified["quality_review_sha256"],
        "prospective_tree_sha256": verified["prospective_tree_sha256"],
        "source_set_sha256": verified["source_set_sha256"],
        "config_sha256": verified["config_sha256"],
        "data_sha256": verified["data_sha256"],
        "environment_sha256": verified["environment_sha256"],
    }
    if any(immutable.get(key) != expected_value for key, expected_value in expected.items()):
        raise Stage6WorkflowError(
            "Stage 6 review authorization immutable lineage drifted"
        )
    from lunar_exploration_ppo.workflows.stage6_source_repair import (
        SOURCE_REPAIR_AMENDMENT_NAME,
        SOURCE_REPAIR_CONTINUATION_NAME,
        SOURCE_REPAIR_CLOSURE_NAME,
        SOURCE_REPAIR_FRONTIER_RECOVERY_NAME,
        SOURCE_REPAIR_SENSOR_ACCELERATION_NAME,
        SOURCE_REPAIR_SUPPLEMENT_NAME,
    )

    amendment_path = stage / SOURCE_REPAIR_AMENDMENT_NAME
    continuation_path = stage / SOURCE_REPAIR_CONTINUATION_NAME
    supplement_path = stage / SOURCE_REPAIR_SUPPLEMENT_NAME
    closure_path = stage / SOURCE_REPAIR_CLOSURE_NAME
    frontier_recovery_path = stage / SOURCE_REPAIR_FRONTIER_RECOVERY_NAME
    sensor_acceleration_path = stage / SOURCE_REPAIR_SENSOR_ACCELERATION_NAME
    if not os.path.lexists(amendment_path):
        if (
            source_repair_amendment_payload is not None
            or source_repair_continuation_payload is not None
            or source_repair_supplement_payload is not None
            or source_repair_closure_payload is not None
            or source_repair_frontier_recovery_payload is not None
            or source_repair_sensor_acceleration_payload is not None
            or os.path.lexists(continuation_path)
            or os.path.lexists(supplement_path)
            or os.path.lexists(closure_path)
            or os.path.lexists(frontier_recovery_path)
            or os.path.lexists(sensor_acceleration_path)
        ):
            raise Stage6WorkflowError(
                "Stage 6 source-repair continuation has no primary amendment"
            )
        return verified
    if repo_root is None:
        raise Stage6WorkflowError(
            "Stage 6 repaired review authorization requires repo identity"
        )
    repo = Path(repo_root).expanduser().resolve()
    current_identity = stage6_execution_identity(
        repo_root=repo,
        config_path=repo / "configs/ppo_highres_frontier_stage6_v1.json",
    )
    context = _load_stage6_source_repair_for_current_identity(
        stage=stage,
        current_execution_identity=current_identity,
        amendment_payload=source_repair_amendment_payload,
        continuation_payload=source_repair_continuation_payload,
        supplement_payload=source_repair_supplement_payload,
        closure_payload=source_repair_closure_payload,
        frontier_recovery_payload=source_repair_frontier_recovery_payload,
        sensor_acceleration_payload=source_repair_sensor_acceleration_payload,
    )
    if context is None or any(
        left != right
        for left, right in (
            (dict(context.origin_execution_identity), lineage["execution_identity"]),
            (dict(context.origin_immutable_bindings), immutable),
            (dict(context.origin_verified_review_authorization), verified),
        )
    ):
        raise Stage6WorkflowError(
            "Stage 6 source-repair origin lineage drifted"
        )
    return dict(context.current_verified_review_authorization)


def _stage6_manifest_review_authorization(
    stage: Path,
    *,
    repo_root: str | Path,
    planning_child_recovery_capability: object | None = None,
    lineage_payload: bytes | None = None,
    source_repair_amendment_payload: bytes | None = None,
    source_repair_continuation_payload: bytes | None = None,
    source_repair_supplement_payload: bytes | None = None,
    source_repair_closure_payload: bytes | None = None,
    source_repair_frontier_recovery_payload: bytes | None = None,
    source_repair_sensor_acceleration_payload: bytes | None = None,
) -> dict[str, object]:
    """Resolve manifest review authority from the already-issued recovery graph."""

    lineage_review = _stage6_lineage_review_authorization(
        stage,
        payload=lineage_payload,
        repo_root=repo_root,
        source_repair_amendment_payload=source_repair_amendment_payload,
        source_repair_continuation_payload=source_repair_continuation_payload,
        source_repair_supplement_payload=source_repair_supplement_payload,
        source_repair_closure_payload=source_repair_closure_payload,
        source_repair_frontier_recovery_payload=(
            source_repair_frontier_recovery_payload
        ),
        source_repair_sensor_acceleration_payload=(
            source_repair_sensor_acceleration_payload
        ),
    )
    if planning_child_recovery_capability is None:
        return lineage_review

    from lunar_exploration_ppo.workflows.stage6_planning_child_recovery import (
        PlanningChildRecoveryCapability,
        _plain_json,
    )

    capability = planning_child_recovery_capability
    if type(capability) is not PlanningChildRecoveryCapability:
        raise Stage6WorkflowError(
            "Stage 6 manifest planning child recovery capability drifted"
        )
    _require_planning_child_recovery_capability(
        capability,
        "manifest review authorization",
    )
    if (
        capability.stage_root != stage
        or capability.formal_run_id != stage.parent.name
    ):
        raise Stage6WorkflowError(
            "Stage 6 manifest planning child recovery root drifted"
        )
    acceptance = capability.acceptance_binding
    current_value = acceptance.get(
        "current_verified_review_authorization"
    )
    origin_value = acceptance.get(
        "origin_verified_review_authorization"
    )
    if not isinstance(current_value, Mapping) or not isinstance(
        origin_value,
        Mapping,
    ):
        raise Stage6WorkflowError(
            "Stage 6 manifest planning child review binding is missing"
        )
    try:
        current_review = _plain_json(current_value)
        origin_review = _plain_json(origin_value)
    except (TypeError, ValueError) as exc:
        raise Stage6WorkflowError(
            "Stage 6 manifest planning child review binding drifted"
        ) from exc
    if (
        type(current_review) is not dict
        or type(origin_review) is not dict
        or origin_review != lineage_review
    ):
        raise Stage6WorkflowError(
            "Stage 6 manifest planning child origin review drifted"
        )
    return current_review


def _stage6_receipt_evidence_binding(payload: bytes) -> dict[str, object]:
    """Extract only a complete v2 receipt's canonical semantic graph binding."""

    from lunar_exploration_ppo.workflows.stage6_terminal_recovery import (
        TerminalRecoveryError,
        _validate_receipt_payload,
    )

    try:
        receipt, _, _ = _validate_receipt_payload(payload)
    except TerminalRecoveryError as exc:
        raise Stage6WorkflowError(
            "Stage 6 preterminal receipt is incomplete or drifted"
        ) from exc
    binding = receipt["preterminal_evidence_binding"]
    assert isinstance(binding, dict)
    return binding


def _resolve_stage6_manifest_publication_authority(
    *,
    stage_root: Path,
    repo_root: Path,
    execution_capability: object | None,
    run_lease: RunLease | None,
) -> tuple[_Stage6ExecutionCapability, RunLease]:
    capability = execution_capability
    lease = run_lease
    if capability is None:
        thread_id = threading.get_ident()
        with _STAGE6_EXECUTION_CAPABILITY_LOCK:
            matches = [
                (candidate, state)
                for candidate, state in _STAGE6_EXECUTION_CAPABILITY_REGISTRY.items()
                if state.active is True
                and state.operation_threads.get(thread_id, 0) > 0
                and state.stage_root == stage_root
                and state.repo_root == repo_root
            ]
        if len(matches) != 1:
            raise Stage6WorkflowError(
                "Stage 6 manifest publication requires an execution capability"
            )
        capability, state = matches[0]
        lease = state.run_lease
    state = _stage6_capability_state(capability)
    if type(lease) is not RunLease or lease is not state.run_lease:
        raise Stage6WorkflowError(
            "Stage 6 manifest publication requires the capability RunLease"
        )
    return capability, lease


@contextmanager
def _stage6_manifest_publication_operation(
    *,
    stage_root: str | Path,
    repo_root: str | Path,
    execution_capability: object | None,
    run_lease: RunLease | None,
    label: str,
):
    stage = lexical_absolute(stage_root)
    repo = lexical_absolute(repo_root)
    capability, lease = _resolve_stage6_manifest_publication_authority(
        stage_root=stage,
        repo_root=repo,
        execution_capability=execution_capability,
        run_lease=run_lease,
    )
    state = _stage6_capability_state(capability)
    with _stage6_execution_operation(
        capability,
        label=label,
        formal_run_id=stage.parent.name,
        run_root=stage.parent,
        stage_root=stage,
        repo_root=repo,
        config_path=state.config_path,
        stage5_authority=state.stage5_authority.identity,
        verified_review_authorization=json.loads(
            state.review_authorization_payload.decode("utf-8")
        ),
        run_lease=lease,
    ):
        yield stage, repo, capability, lease


def write_stage6_manifest(
    *,
    stage_root: str | Path,
    repo_root: str | Path,
    execution_capability: object | None = None,
    run_lease: RunLease | None = None,
    planning_child_recovery_capability: object | None = None,
) -> Path:
    with _stage6_manifest_publication_operation(
        stage_root=stage_root,
        repo_root=repo_root,
        execution_capability=execution_capability,
        run_lease=run_lease,
        label="Stage 6 manifest publication",
    ) as (stage_value, repo, capability, lease):
        stage = _require_plain_stage6_root(
            stage_value,
            label="Stage 6 manifest root",
        )
        store = ArtifactStore(stage)
        relative_paths = _stage6_manifest_relative_paths(stage)
        verified_review_authorization = _stage6_manifest_review_authorization(
            stage,
            repo_root=repo,
            planning_child_recovery_capability=(
                planning_child_recovery_capability
            ),
        )
        _require_stage6_execution_capability(
            capability,
            label="Stage 6 manifest lineage publication",
            formal_run_id=stage.parent.name,
            run_root=stage.parent,
            stage_root=stage,
            repo_root=repo,
            verified_review_authorization=verified_review_authorization,
            run_lease=lease,
        )
        receipt_snapshot = _FileIdentity.capture(
            stage / "preterminal_acceptance.json",
            base=stage,
        )
        receipt_binding = _stage6_receipt_evidence_binding(
            receipt_snapshot.read_current_bytes(
                "Stage 6 manifest preterminal receipt"
            )
        )
        manifest = {
            "schema_version": "stage6_sha256_manifest/v3",
            "source_identity": stage6_source_identity(repo),
            "verified_review_authorization": verified_review_authorization,
            "preterminal_evidence_binding": receipt_binding,
            "artifacts": _build_stage6_manifest_entries(stage, relative_paths),
        }
        receipt_snapshot.require_current(
            "Stage 6 manifest receipt before publication"
        )
        path = store.write_json_exclusive("manifest.json", manifest)
        receipt_snapshot.require_current(
            "Stage 6 manifest receipt after publication"
        )
        return path


def write_or_verify_stage6_manifest(
    stage_root: str | Path,
    *,
    repo_root: str | Path,
    execution_capability: object | None = None,
    run_lease: RunLease | None = None,
    planning_child_recovery_capability: object | None = None,
) -> dict[str, object]:
    """Write the terminal manifest once, or verify existing exact bytes."""

    with _stage6_manifest_publication_operation(
        stage_root=stage_root,
        repo_root=repo_root,
        execution_capability=execution_capability,
        run_lease=run_lease,
        label="Stage 6 manifest write-or-verify publication",
    ) as (stage_value, repo, capability, lease):
        stage = _require_plain_stage6_root(
            stage_value,
            label="Stage 6 manifest write-or-verify root",
        )
        manifest_path = stage / "manifest.json"
        if not manifest_path.exists():
            try:
                write_stage6_manifest(
                    stage_root=stage,
                    repo_root=repo,
                    execution_capability=capability,
                    run_lease=lease,
                    planning_child_recovery_capability=(
                        planning_child_recovery_capability
                    ),
                )
            except FileExistsError:
                pass
        verify_stage6_manifest(
            stage_root=stage,
            repo_root=repo,
            planning_child_recovery_capability=(
                planning_child_recovery_capability
            ),
        ).require_current()
        payload = _FileIdentity.capture(
            manifest_path,
            base=stage,
        ).read_current_bytes("Stage 6 manifest write-or-verify")
        return {
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size_bytes": len(payload),
        }


def _verify_existing_stage6_manifest_identity(
    stage: Path,
    *,
    repo_root: Path,
    planning_child_recovery_capability: object | None = None,
) -> dict[str, object]:
    """Verify an existing terminal manifest and return its exact identity."""

    manifest_path = stage / "manifest.json"
    if not manifest_path.is_file():
        raise Stage6WorkflowError("Stage 6 terminal manifest is missing")
    verify_stage6_manifest(
        stage_root=stage,
        repo_root=repo_root,
        planning_child_recovery_capability=(
            planning_child_recovery_capability
        ),
    ).require_current()
    try:
        payload = manifest_path.read_bytes()
    except OSError as exc:
        raise Stage6WorkflowError("Stage 6 terminal manifest is unreadable") from exc
    return {
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }


def _verify_stage6_terminal_manifest_evidence(
    stage_root: str | Path,
    *,
    source_identity: Mapping[str, object],
    repo_root: str | Path,
    evidence_handle: _Stage6EvidenceHandle,
    planning_child_recovery_capability: object | None = None,
) -> dict[str, object]:
    """Verify one terminal manifest entirely through its captured handle."""

    from lunar_exploration_ppo.workflows.stage6_terminal_recovery import (
        _Stage6EvidenceHandle,
    )

    stage = _require_plain_stage6_root(
        stage_root,
        label="Stage 6 terminal manifest evidence root",
    )
    if (
        type(evidence_handle) is not _Stage6EvidenceHandle
        or evidence_handle.stage != stage
    ):
        raise Stage6WorkflowError(
            "Stage 6 terminal manifest evidence handle is missing or drifted"
        )
    payload = evidence_handle.read_bytes(
        "manifest.json",
        label="Stage 6 terminal manifest identity",
    )
    manifest = _strict_canonical_json(
        payload,
        "Stage 6 terminal manifest snapshot",
    )
    if (
        set(manifest)
        != {
            "schema_version",
            "source_identity",
            "verified_review_authorization",
            "preterminal_evidence_binding",
            "artifacts",
        }
        or manifest.get("schema_version") != "stage6_sha256_manifest/v3"
        or manifest.get("source_identity") != source_identity
    ):
        raise Stage6WorkflowError(
            "Stage 6 terminal manifest snapshot schema or source drifted"
        )
    lineage_payload = evidence_handle.read_bytes(
        "lineage_audit.json",
        label="Stage 6 terminal manifest lineage",
    )
    evidence_paths = set(evidence_handle.file_paths)
    amendment_payload = (
        evidence_handle.read_bytes(
            "source-repair-amendment.json",
            label="Stage 6 terminal source-repair amendment",
        )
        if "source-repair-amendment.json" in evidence_paths
        else None
    )
    continuation_payload = (
        evidence_handle.read_bytes(
            "source-repair-continuation.json",
            label="Stage 6 terminal source-repair continuation",
        )
        if "source-repair-continuation.json" in evidence_paths
        else None
    )
    supplement_payload = (
        evidence_handle.read_bytes(
            "source-repair-supplement.json",
            label="Stage 6 terminal source-repair supplement",
        )
        if "source-repair-supplement.json" in evidence_paths
        else None
    )
    closure_payload = (
        evidence_handle.read_bytes(
            "source-repair-closure.json",
            label="Stage 6 terminal source-repair closure",
        )
        if "source-repair-closure.json" in evidence_paths
        else None
    )
    frontier_recovery_payload = (
        evidence_handle.read_bytes(
            "source-repair-frontier-recovery.json",
            label="Stage 6 terminal source-repair frontier recovery",
        )
        if "source-repair-frontier-recovery.json" in evidence_paths
        else None
    )
    sensor_acceleration_payload = (
        evidence_handle.read_bytes(
            "source-repair-sensor-acceleration.json",
            label="Stage 6 terminal source-repair sensor acceleration",
        )
        if "source-repair-sensor-acceleration.json" in evidence_paths
        else None
    )
    if (
        frontier_recovery_payload is not None
        and closure_payload is None
    ):
        raise Stage6WorkflowError(
            "Stage 6 terminal source-repair frontier recovery has no closure"
        )
    if (
        sensor_acceleration_payload is not None
        and frontier_recovery_payload is None
    ):
        raise Stage6WorkflowError(
            "Stage 6 terminal source-repair sensor acceleration has no "
            "frontier recovery"
        )
    if manifest.get(
        "verified_review_authorization"
    ) != _stage6_manifest_review_authorization(
        stage,
        lineage_payload=lineage_payload,
        repo_root=repo_root,
        planning_child_recovery_capability=(
            planning_child_recovery_capability
        ),
        source_repair_amendment_payload=amendment_payload,
        source_repair_continuation_payload=continuation_payload,
        source_repair_supplement_payload=supplement_payload,
        source_repair_closure_payload=closure_payload,
        source_repair_frontier_recovery_payload=frontier_recovery_payload,
        source_repair_sensor_acceleration_payload=sensor_acceleration_payload,
    ):
        raise Stage6WorkflowError(
            "Stage 6 terminal manifest snapshot lineage drifted"
        )
    receipt_payload = evidence_handle.read_bytes(
        "preterminal_acceptance.json",
        label="Stage 6 terminal manifest receipt",
    )
    if manifest.get(
        "preterminal_evidence_binding"
    ) != _stage6_receipt_evidence_binding(receipt_payload):
        raise Stage6WorkflowError(
            "Stage 6 terminal manifest snapshot receipt drifted"
        )
    entries = manifest.get("artifacts")
    expected_paths = sorted(
        set(evidence_handle.file_paths) - {"manifest.json"}
    )
    if (
        not isinstance(entries, list)
        or [
            row.get("path")
            for row in entries
            if isinstance(row, Mapping)
        ]
        != expected_paths
    ):
        raise Stage6WorkflowError(
            "Stage 6 terminal manifest snapshot membership drifted"
        )
    for row in entries:
        if not isinstance(row, Mapping) or set(row) != {
            "path",
            "sha256",
            "size_bytes",
        }:
            raise Stage6WorkflowError(
                "Stage 6 terminal manifest snapshot entry drifted"
            )
        relative = str(row["path"])
        artifact_payload = evidence_handle.read_bytes(
            relative,
            label=f"Stage 6 terminal manifest artifact {relative}",
        )
        if {
            "sha256": hashlib.sha256(artifact_payload).hexdigest(),
            "size_bytes": len(artifact_payload),
        } != {
            "sha256": row["sha256"],
            "size_bytes": row["size_bytes"],
        }:
            raise Stage6WorkflowError(
                "Stage 6 terminal manifest snapshot artifact drifted"
            )
    return {
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }


def rebind_stage6_manifest_for_terminal_resource(
    *,
    stage_root: str | Path,
    repo_root: str | Path,
) -> Path:
    """Compatibility verifier for the receipt-bound v3 terminal manifest."""

    stage = _require_plain_stage6_root(
        stage_root,
        label="Stage 6 terminal manifest root",
    )
    repo = Path(repo_root).expanduser().resolve()
    manifest_path = stage / "manifest.json"
    from lunar_exploration_ppo.workflows.stage6_terminal_recovery import (
        COMPLETE_STATES,
        detect_stage6_terminal_recovery,
    )

    manifest_verifier = lambda candidate: _verify_existing_stage6_manifest_identity(
        candidate,
        repo_root=repo,
    )
    detection = detect_stage6_terminal_recovery(
        stage_root=stage,
        manifest_verifier=manifest_verifier,
    )
    if (
        detection.get("status") != "valid_terminal_recovery"
        or tuple(detection.get("phase_states", ())) != COMPLETE_STATES
    ):
        raise Stage6WorkflowError(
            "Stage 6 receipt-bound terminal manifest replay failed: "
            f"{detection.get('reason', '')}"
        )
    return manifest_path


def _require_plain_stage6_root(
    stage_root: str | Path,
    *,
    label: str,
) -> Path:
    try:
        return require_plain_path(
            lexical_absolute(stage_root),
            leaf_kind="directory",
            label=label,
        )
    except PathSecurityError as exc:
        raise Stage6WorkflowError(
            f"{label} contains a link or reparse point: {exc}"
        ) from exc


def _manifest_directory_identity(path: Path) -> tuple[int, int, int, int]:
    if _is_link_or_reparse(path):
        raise Stage6WorkflowError(
            f"Stage 6 manifest contains link or reparse point: {path.name}"
        )
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise Stage6WorkflowError(
            f"Stage 6 manifest directory identity is unavailable: {path.name}"
        ) from exc
    if not stat.S_ISDIR(metadata.st_mode):
        raise Stage6WorkflowError(
            f"Stage 6 manifest directory identity changed: {path.name}"
        )
    return (
        int(metadata.st_dev),
        int(metadata.st_ino),
        int(metadata.st_mode),
        int(getattr(metadata, "st_file_attributes", 0)),
    )


def _validate_stage6_manifest_repair_profile(
    stage: Path,
    *,
    top_files: set[str],
) -> None:
    classic = set(STAGE6_OPTIONAL_MANIFEST_BOUND_ARTIFACTS).intersection(
        top_files
    )
    child_name = STAGE6_PLANNING_CHILD_SOURCE_REPAIR_MANIFEST_ARTIFACT
    continuation_name = (
        STAGE6_PLANNING_CHILD_SOURCE_REPAIR_CONTINUATION_MANIFEST_ARTIFACT
    )
    child_present = child_name in top_files
    continuation_present = continuation_name in top_files
    if continuation_present and not child_present:
        raise Stage6WorkflowError(
            "Stage 6 planning child continuation has no parent amendment"
        )
    if child_present and classic:
        raise Stage6WorkflowError(
            "Stage 6 planning child and classic source-repair manifest "
            "profiles are mutually exclusive"
        )
    if not child_present:
        return
    try:
        lineage_payload = _FileIdentity.capture(
            stage / "lineage_audit.json",
            base=stage,
        ).read_current_bytes("Stage 6 planning child manifest lineage")
    except (OSError, Stage6WorkflowError) as exc:
        raise Stage6WorkflowError(
            "Stage 6 planning child manifest requires planning warm-start "
            "lineage"
        ) from exc
    lineage = _strict_canonical_json(
        lineage_payload,
        "Stage 6 planning child manifest lineage",
    )
    if (
        lineage.get("schema_version") != "stage6_lineage_audit/v3"
        or not isinstance(lineage.get("planning_warm_start"), Mapping)
    ):
        raise Stage6WorkflowError(
            "Stage 6 planning child manifest requires planning warm-start "
            "lineage"
        )


def _capture_stage6_manifest_graph(
    stage: Path,
    *,
    require_manifest: bool,
) -> _Stage6ManifestGraph:
    stage = _require_plain_stage6_root(
        stage,
        label="Stage 6 manifest traversal root",
    )
    top_files: set[str] = set()
    directories: set[str] = set()
    manifest_present = False
    relative_paths: list[str] = []
    directory_snapshots: list[_ManifestDirectorySnapshot] = []
    pending_directories = [stage]
    while pending_directories:
        parent = pending_directories.pop()
        relative_parent = "." if parent == stage else parent.relative_to(stage).as_posix()
        try:
            require_plain_path(
                parent,
                base=stage,
                leaf_kind="directory",
                label="Stage 6 manifest directory",
            )
            identity_before = _manifest_directory_identity(parent)
            children = tuple(sorted(parent.iterdir(), key=lambda item: item.name))
        except OSError as exc:
            raise Stage6WorkflowError(
                f"Stage 6 manifest directory is unreadable: {relative_parent}"
            ) from exc
        except PathSecurityError as exc:
            raise Stage6WorkflowError(
                f"Stage 6 manifest directory contains link or reparse point: {relative_parent}"
            ) from exc

        member_rows: list[tuple[str, str]] = []
        for path in children:
            relative = path.relative_to(stage).as_posix()
            if _is_link_or_reparse(path):
                raise Stage6WorkflowError(
                    f"Stage 6 manifest contains link or reparse point: {relative}"
                )
            try:
                metadata = path.lstat()
            except OSError as exc:
                raise Stage6WorkflowError(
                    f"Stage 6 manifest child metadata is unavailable: {relative}"
                ) from exc
            if stat.S_ISDIR(metadata.st_mode):
                member_rows.append((path.name, "directory"))
                if parent == stage:
                    directories.add(path.name)
                    if path.name in {"checkpoints", "episode-traces", "preflight"}:
                        pending_directories.append(path)
                elif not _is_allowed_stage6_directory(relative):
                    raise Stage6WorkflowError(
                        f"Stage 6 manifest contains unknown child: {relative}"
                    )
                else:
                    pending_directories.append(path)
                continue
            if not stat.S_ISREG(metadata.st_mode):
                raise Stage6WorkflowError(
                    f"Stage 6 manifest contains link, reparse, or special child: {relative}"
                )
            member_rows.append((path.name, "file"))
            if parent == stage:
                if path.name == "manifest.json":
                    manifest_present = True
                else:
                    top_files.add(path.name)
                    relative_paths.append(path.name)
            elif not _is_allowed_stage6_child(relative):
                raise Stage6WorkflowError(
                    f"Stage 6 manifest contains unknown child: {relative}"
                )
            else:
                relative_paths.append(relative)

        try:
            member_names_after = tuple(sorted(path.name for path in parent.iterdir()))
            identity_after = _manifest_directory_identity(parent)
        except OSError as exc:
            raise Stage6WorkflowError(
                f"Stage 6 manifest directory changed during traversal: {relative_parent}"
            ) from exc
        if (
            member_names_after != tuple(name for name, _kind in member_rows)
            or identity_after != identity_before
        ):
            raise Stage6WorkflowError(
                "Stage 6 manifest directory membership changed during traversal: "
                f"{relative_parent}"
            )
        directory_snapshots.append(
            _ManifestDirectorySnapshot(
                relative=relative_parent,
                stat_identity=identity_after,
                members=tuple(member_rows),
            )
        )

    required_top_files = set(STAGE6_MANIFEST_BOUND_ARTIFACTS)
    optional_top_files = set(STAGE6_MANIFEST_PROFILE_OPTIONAL_ARTIFACTS)
    if not (
        required_top_files.issubset(top_files)
        and top_files.issubset(required_top_files | optional_top_files)
    ):
        raise Stage6WorkflowError("Stage 6 top-level manifest artifact set drift")
    present_source_repair = tuple(
        name
        for name in STAGE6_OPTIONAL_MANIFEST_BOUND_ARTIFACTS
        if name in top_files
    )
    if present_source_repair != STAGE6_OPTIONAL_MANIFEST_BOUND_ARTIFACTS[
        : len(present_source_repair)
    ]:
        raise Stage6WorkflowError("Stage 6 source-repair manifest chain drift")
    _validate_stage6_manifest_repair_profile(
        stage,
        top_files=top_files,
    )
    if directories != {"checkpoints", "episode-traces", "preflight"}:
        raise Stage6WorkflowError("Stage 6 known artifact directory set drift")
    if require_manifest and not manifest_present:
        raise Stage6WorkflowError("Stage 6 root artifact set drift")
    if "preflight/audit.json" not in relative_paths:
        raise Stage6WorkflowError("Stage 6 preflight audit is missing")
    return _Stage6ManifestGraph(
        relative_paths=tuple(sorted(relative_paths)),
        directories=tuple(
            sorted(directory_snapshots, key=lambda snapshot: snapshot.relative)
        ),
    )


def _stage6_manifest_relative_paths(stage: Path) -> tuple[str, ...]:
    return _capture_stage6_manifest_graph(
        stage,
        require_manifest=False,
    ).relative_paths


def _is_allowed_stage6_directory(relative: str) -> bool:
    return (
        re.fullmatch(r"checkpoints/seed-20260716", relative) is not None
        or re.fullmatch(
            r"checkpoints/seed-20260716/update-[0-9]{8}",
            relative,
        )
        is not None
    )


def _build_stage6_manifest_entries(
    stage: Path,
    relative_paths: Sequence[str],
) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    paths_by_file_id: dict[tuple[int, int], str] = {}
    for relative in sorted(relative_paths):
        snapshot = _FileIdentity.capture(stage / relative, base=stage)
        prior = paths_by_file_id.get(snapshot.file_id)
        if prior is not None:
            raise Stage6WorkflowError(
                "Stage 6 manifest duplicate File ID alias: "
                f"{prior} and {relative}"
            )
        paths_by_file_id[snapshot.file_id] = relative
        entries.append(
            {
                "path": relative,
                "sha256": snapshot.sha256,
                "size_bytes": snapshot.size_bytes,
            }
        )
    return entries


def _is_allowed_stage6_child(relative: str) -> bool:
    if relative == "preflight/audit.json" or relative == "checkpoints/index.jsonl":
        return True
    if re.fullmatch(
        r"checkpoints/seed-20260716/latest\.json",
        relative,
    ):
        return True
    if re.fullmatch(
        r"checkpoints/seed-20260716/"
        r"update-[0-9]{8}/(checkpoint\.pt|manifest\.json|complete\.json)",
        relative,
    ):
        return True
    if re.fullmatch(
        r"episode-traces/validation-seed-20260716-"
        r"update-(010|020|030|040|050|060|070|080|090|100)\.jsonl",
        relative,
    ):
        return True
    methods = "(?:ppo_policy|random_valid_frontier|nearest_frontier|max_potential_gain_frontier|gain_over_cost_frontier)"
    return (
        re.fullmatch(
            rf"episode-traces/final-(test|unseen)-{methods}\.jsonl",
            relative,
        )
        is not None
        or re.fullmatch(
            rf"episode-traces/final-(test|unseen)-{methods}\.summary\.json",
            relative,
        )
        is not None
        or re.fullmatch(
            rf"episode-traces/final-(test|unseen)-{methods}\.commit\.json",
            relative,
        )
        is not None
    )


def verify_stage6_manifest(
    *,
    stage_root: str | Path,
    repo_root: str | Path,
    planning_child_recovery_capability: object | None = None,
) -> Stage6ManifestHandle:
    stage = _require_plain_stage6_root(
        stage_root,
        label="Stage 6 manifest verification root",
    )
    repo = Path(repo_root).expanduser().resolve()
    if stage.name != "s6" or not stage.is_dir():
        raise Stage6WorkflowError("Stage 6 root must be canonical <run>/s6")
    if any((stage / name).exists() for name in ("review.json", "approval.json", "gate.json")):
        raise Stage6WorkflowError("Stage 6 machine root contains authority artifact")
    expected_root = {
        *STAGE6_MANIFEST_BOUND_ARTIFACTS,
        "manifest.json",
        "checkpoints",
        "episode-traces",
        "preflight",
    }
    expected_root.update(
        name
        for name in STAGE6_MANIFEST_PROFILE_OPTIONAL_ARTIFACTS
        if os.path.lexists(stage / name)
    )
    actual_root = {path.name for path in stage.iterdir()}
    if actual_root != expected_root:
        raise Stage6WorkflowError("Stage 6 root artifact set drift")
    manifest_graph = _capture_stage6_manifest_graph(
        stage,
        require_manifest=True,
    )
    expected_paths = manifest_graph.relative_paths
    manifest_snapshot = _FileIdentity.capture(
        stage / "manifest.json",
        base=stage,
    )
    manifest = _strict_canonical_json(
        manifest_snapshot.read_current_bytes("Stage 6 manifest"),
        "Stage 6 manifest",
    )
    if set(manifest) != {
        "schema_version",
        "source_identity",
        "verified_review_authorization",
        "preterminal_evidence_binding",
        "artifacts",
    } or manifest.get("schema_version") != "stage6_sha256_manifest/v3":
        raise Stage6WorkflowError("Stage 6 manifest schema drift")
    manifest_receipt_binding = manifest.get("preterminal_evidence_binding")
    if not isinstance(manifest_receipt_binding, Mapping):
        raise Stage6WorkflowError(
            "Stage 6 manifest preterminal evidence binding drift"
        )
    current_source = stage6_source_identity(repo)
    if manifest.get("source_identity") != current_source:
        raise Stage6WorkflowError("Stage 6 manifest current-code binding drift")
    if manifest.get(
        "verified_review_authorization"
    ) != _stage6_manifest_review_authorization(
        stage,
        repo_root=repo,
        planning_child_recovery_capability=(
            planning_child_recovery_capability
        ),
    ):
        raise Stage6WorkflowError(
            "Stage 6 manifest review authorization binding drift"
        )
    entries = manifest.get("artifacts")
    if not isinstance(entries, list) or [row.get("path") for row in entries if isinstance(row, dict)] != list(
        expected_paths
    ):
        raise Stage6WorkflowError("Stage 6 manifest path set drift")
    snapshots: list[tuple[str, _FileIdentity]] = [("manifest.json", manifest_snapshot)]
    paths_by_file_id = {manifest_snapshot.file_id: "manifest.json"}
    receipt_snapshot: _FileIdentity | None = None
    for row in entries:
        if not isinstance(row, dict) or set(row) != {"path", "sha256", "size_bytes"}:
            raise Stage6WorkflowError("Stage 6 manifest entry drift")
        relative = str(row["path"])
        snapshot = _FileIdentity.capture(stage / relative, base=stage)
        if snapshot.sha256 != row["sha256"] or snapshot.size_bytes != row["size_bytes"]:
            raise Stage6WorkflowError("Stage 6 manifest artifact hash drift")
        prior = paths_by_file_id.get(snapshot.file_id)
        if prior is not None:
            raise Stage6WorkflowError(
                "Stage 6 manifest duplicate File ID alias: "
                f"{prior} and {relative}"
            )
        paths_by_file_id[snapshot.file_id] = relative
        snapshots.append((relative, snapshot))
        if relative == "preterminal_acceptance.json":
            receipt_snapshot = snapshot
    if receipt_snapshot is None:
        raise Stage6WorkflowError(
            "Stage 6 manifest preterminal receipt snapshot is missing"
        )
    receipt_binding = _stage6_receipt_evidence_binding(
        receipt_snapshot.read_current_bytes(
            "Stage 6 manifest-bound preterminal receipt"
        )
    )
    if receipt_binding != manifest_receipt_binding:
        raise Stage6WorkflowError(
            "Stage 6 manifest preterminal evidence binding drift"
        )
    handle = Stage6ManifestHandle(
        stage_root=stage,
        repo_root=repo,
        source_identity=current_source,
        snapshots=tuple(snapshots),
        manifest_graph=manifest_graph,
    )
    handle.require_current()
    return handle


class _PlanningChildMachineVerificationResult(dict[str, object]):
    __slots__ = ("_planning_child_recovery_capability",)

    def __init__(
        self,
        value: Mapping[str, object],
        capability: object,
    ) -> None:
        super().__init__(value)
        self._planning_child_recovery_capability = capability


def _verify_stage6_machine_acceptance(
    *,
    stage_root: str | Path,
    repo_root: str | Path,
    gate_path: str | Path | None = None,
    require_terminal: bool,
    summary_override: Mapping[str, object] | None = None,
    routing_override: Mapping[str, object] | None = None,
    reports_override: Mapping[str, bytes] | None = None,
    planning_child_recovery_capability: object | None = None,
) -> dict[str, object]:
    """Verify pending semantics under one exact, strongly pinned evidence graph."""

    from lunar_exploration_ppo.workflows.stage6_terminal_recovery import (
        MANIFEST_NAME,
        RECEIPT_NAME,
        TERMINAL_ARTIFACT_NAMES,
        TerminalRecoveryError,
        _bind_planning_child_semantic_result,
        _capture_preterminal_evidence_handle,
    )

    stage = _require_plain_stage6_root(
        stage_root,
        label="Stage 6 pending evidence verification root",
    )
    if require_terminal is not False:
        raise Stage6WorkflowError(
            "Stage 6 full semantic verifier is preterminal-only"
        )
    try:
        with _capture_preterminal_evidence_handle(stage) as evidence_handle:
            result = _verify_stage6_machine_acceptance_from_bound_graph(
                stage_root=stage,
                repo_root=repo_root,
                gate_path=gate_path,
                require_terminal=False,
                summary_override=summary_override,
                routing_override=routing_override,
                reports_override=reports_override,
                evidence_handle=evidence_handle,
                planning_child_recovery_capability=(
                    planning_child_recovery_capability
                ),
            )
            evidence_handle.require_current(
                "Stage 6 pending semantic verifier final evidence"
            )
            binding = evidence_handle.binding
            semantic_result = {
                **result,
                "schema_version": "stage6_machine_acceptance_verification/v2",
                "evidence_binding": binding,
            }
            if type(result) is _PlanningChildMachineVerificationResult:
                return _bind_planning_child_semantic_result(
                    semantic_result,
                    capability=(
                        result._planning_child_recovery_capability
                    ),
                )  # type: ignore[return-value]
            return semantic_result
    except TerminalRecoveryError as exc:
        raise Stage6WorkflowError(
            "Stage 6 pending evidence graph could not remain continuously bound"
        ) from exc


def _validate_planning_child_machine_context(
    *,
    capability: object,
    lineage_audit: Mapping[str, object],
    current_execution_identity: Mapping[str, object],
    current_verified_review_authorization: Mapping[str, object],
    current_immutable_bindings: Mapping[str, object],
) -> dict[str, object]:
    from lunar_exploration_ppo.workflows.stage6_planning_child_recovery import (
        PlanningChildRecoveryCapability,
        _plain_json,
    )

    try:
        if type(capability) is not PlanningChildRecoveryCapability:
            raise TypeError("planning child recovery capability is missing")
        acceptance_value = _plain_json(capability.acceptance_binding)
        if not isinstance(acceptance_value, dict):
            raise TypeError("planning child recovery acceptance is invalid")
        acceptance = acceptance_value
        origin_identity = lineage_audit.get("execution_identity")
        origin_immutable_value = lineage_audit.get(
            "immutable_bindings"
        )
        origin_review = lineage_audit.get(
            "verified_review_authorization"
        )
        if (
            not isinstance(acceptance, Mapping)
            or not isinstance(origin_identity, Mapping)
            or not isinstance(origin_immutable_value, Mapping)
            or not isinstance(origin_review, Mapping)
        ):
            raise TypeError("planning child recovery acceptance is invalid")
        origin_immutable = dict(origin_immutable_value)
        current_immutable = dict(current_immutable_bindings)
        current_identity_sha256 = hashlib.sha256(
            ArtifactStore.canonical_json_bytes(
                current_execution_identity
            )
        ).hexdigest()
        origin_identity_sha256 = hashlib.sha256(
            ArtifactStore.canonical_json_bytes(origin_identity)
        ).hexdigest()
        current_immutable_sha256 = hashlib.sha256(
            ArtifactStore.canonical_json_bytes(current_immutable)
        ).hexdigest()
        raw_epochs = acceptance.get("journal_binding_epochs")
        if not isinstance(raw_epochs, (list, tuple)) or not raw_epochs:
            raise TypeError("planning child journal epochs are missing")
        journal_binding_epochs: list[
            tuple[str, Mapping[str, object]]
        ] = []
        for value in raw_epochs:
            if (
                not isinstance(value, (list, tuple))
                or len(value) != 2
                or not isinstance(value[0], str)
                or not isinstance(value[1], Mapping)
            ):
                raise TypeError(
                    "planning child journal epoch drifted"
                )
            journal_binding_epochs.append(
                (value[0], dict(value[1]))
            )
        lineage_epochs = capability.lineage_epochs
        if not lineage_epochs:
            raise TypeError("planning child lineage epochs are missing")
        expected_lineage_epochs = [
            {
                "first_update": epoch.first_update,
                "last_update": epoch.last_update,
                "artifact_sha256": epoch.artifact_sha256,
                "execution_identity_sha256": (
                    epoch.execution_identity_sha256
                ),
            }
            for epoch in lineage_epochs
        ]
        if (
            len(lineage_epochs) != len(journal_binding_epochs)
            or acceptance.get("lineage_epochs")
            != expected_lineage_epochs
        ):
            raise TypeError("planning child recovery epoch count drifted")
        for index, epoch in enumerate(lineage_epochs):
            if (
                type(epoch.first_update) is not int
                or (
                    epoch.last_update is not None
                    and (
                        type(epoch.last_update) is not int
                        or epoch.last_update < epoch.first_update
                    )
                )
                or (
                    index < len(lineage_epochs) - 1
                    and (
                        epoch.last_update
                        != lineage_epochs[index + 1].first_update - 1
                    )
                )
                or (
                    index == len(lineage_epochs) - 1
                    and epoch.last_update is not None
                )
                or not journal_binding_epochs[index][0].endswith(
                    f":{capability.seed}:update:{epoch.first_update:03d}"
                )
            ):
                raise TypeError(
                    "planning child recovery epoch order drifted"
                )
    except (
        TypeError,
        ValueError,
    ) as exc:
        raise Stage6WorkflowError(
            "Stage 6 planning child recovery capability drifted"
        ) from exc
    if (
        lineage_audit.get("schema_version")
        != "stage6_lineage_audit/v3"
        or not isinstance(
            lineage_audit.get("planning_warm_start"),
            Mapping,
        )
        or acceptance.get("schema_version")
        != "stage6_planning_child_recovery_capability_acceptance/v1"
        or acceptance.get("formal_run_id")
        != capability.formal_run_id
        or acceptance.get("seed") != capability.seed
        or acceptance.get("parent_artifact_sha256")
        != capability.parent_artifact_sha256
        or acceptance.get("continuation_artifact_sha256")
        != capability.continuation_artifact_sha256
        or acceptance.get("input_snapshot_sha256")
        != capability.input_snapshot_sha256
        or acceptance.get("current_execution_identity_sha256")
        != current_identity_sha256
        or acceptance.get("current_immutable_bindings_sha256")
        != current_immutable_sha256
        or acceptance.get("current_verified_review_authorization")
        != current_verified_review_authorization
        or acceptance.get("origin_verified_review_authorization")
        != origin_review
        or origin_identity_sha256
        != lineage_epochs[0].execution_identity_sha256
        or current_identity_sha256
        != lineage_epochs[-1].execution_identity_sha256
        or origin_immutable
        != dict(journal_binding_epochs[0][1])
        or dict(journal_binding_epochs[-1][1])
        != current_immutable
    ):
        raise Stage6WorkflowError(
            "Stage 6 planning child recovery acceptance binding drifted"
        )
    return {
        "historical_immutable_bindings": origin_immutable,
        "current_immutable_bindings": current_immutable,
        "journal_binding_epochs": tuple(journal_binding_epochs),
    }


def _stage6_checkpoint_lineage_base(
    *,
    config: object,
    bindings: Mapping[str, object],
    safety_contract: object,
) -> dict[str, object]:
    from lunar_exploration_ppo.configs.stage6 import (
        SafetyContract,
        Stage6Config,
    )

    if (
        type(config) is not Stage6Config
        or not isinstance(bindings, Mapping)
        or type(safety_contract) is not SafetyContract
    ):
        raise Stage6WorkflowError(
            "Stage 6 checkpoint lineage base input drifted"
        )
    try:
        return {
            "stage5_commit": config.stage5_authority.commit_sha256,
            "stage5_gate_sha256": config.stage5_authority.gate_sha256,
            "stage5_manifest_sha256": (
                config.stage5_authority.manifest_sha256
            ),
            "stage4_checkpoint_sha256": (
                config.stage5_authority.checkpoint_sha256
            ),
            "stage4_policy_state_sha256": (
                config.stage5_authority.policy_state_sha256
            ),
            "source_set_sha256": bindings["source_set_sha256"],
            "prospective_tree_sha256": bindings[
                "prospective_tree_sha256"
            ],
            "data_sha256": bindings["data_sha256"],
            "environment_identity": bindings["environment_identity"],
            "environment_sha256": bindings["environment_sha256"],
            "formal_run_id": bindings["formal_run_id"],
            "changed_path_set_sha256": bindings[
                "changed_path_set_sha256"
            ],
            "review_authorization_record_sha256": bindings[
                "review_authorization_record_sha256"
            ],
            "authorization_file_sha256": bindings[
                "authorization_file_sha256"
            ],
            "review_identity_sha256": bindings[
                "review_identity_sha256"
            ],
            "reviewed_prospective_git_tree": bindings[
                "reviewed_prospective_git_tree"
            ],
            "frozen_diff_sha256": bindings["frozen_diff_sha256"],
            "spec_review_sha256": bindings["spec_review_sha256"],
            "quality_review_sha256": bindings["quality_review_sha256"],
            "safety_contract_sha256": safety_contract.sha256,
        }
    except (AttributeError, KeyError, TypeError) as exc:
        raise Stage6WorkflowError(
            "Stage 6 checkpoint lineage base binding drifted"
        ) from exc


def _stage6_planning_checkpoint_lineage_for_update(
    *,
    update: int,
    current_checkpoint_lineage: Mapping[str, object],
    historical_checkpoint_lineage: Mapping[str, object],
    planning_warm_start_context: object,
    planning_warm_start_sha256: str,
    planning_child_source_repair: object | None,
    parent_current_checkpoint_lineage: (
        Mapping[str, object] | None
    ) = None,
) -> dict[str, object]:
    if type(update) is not int:
        raise Stage6WorkflowError(
            "Stage 6 planning checkpoint update drifted"
        )
    if planning_child_source_repair is not None:
        from lunar_exploration_ppo.workflows.stage6_planning_child_recovery import (
            PlanningChildRecoveryCapability,
            Stage6PlanningChildRecoveryError,
            _plain_json,
        )

        if type(planning_child_source_repair) is not PlanningChildRecoveryCapability:
            raise Stage6WorkflowError(
                "Stage 6 planning child recovery capability drifted"
            )
        try:
            lineage = (
                planning_child_source_repair
                .checkpoint_lineage_for_update(update)
            )
        except Stage6PlanningChildRecoveryError as exc:
            raise Stage6WorkflowError(
                "Stage 6 planning child checkpoint lineage drifted"
            ) from exc
        plain_lineage = _plain_json(lineage)
        if not isinstance(plain_lineage, dict):
            raise Stage6WorkflowError(
                "Stage 6 planning child checkpoint lineage drifted"
            )
        return plain_lineage
    try:
        warm_lineage = getattr(
            planning_warm_start_context,
            "checkpoint_lineage_for_update",
        )(
            update,
            warm_start_artifact_sha256=planning_warm_start_sha256,
        )
    except Exception as exc:
        raise Stage6WorkflowError(
            "Stage 6 planning warm-start checkpoint lineage drifted"
        ) from exc
    if not isinstance(warm_lineage, Mapping):
        raise Stage6WorkflowError(
            "Stage 6 planning warm-start checkpoint lineage drifted"
        )
    del historical_checkpoint_lineage
    del parent_current_checkpoint_lineage
    return {
        **dict(current_checkpoint_lineage),
        **dict(warm_lineage),
    }


def _validate_planning_child_resource_boundary(
    *,
    capability: object,
    resource_rows: Sequence[Mapping[str, object]],
    resource_payload: bytes,
) -> dict[str, object]:
    """Validate the exact startup bytes already sealed by the issuer."""

    from lunar_exploration_ppo.workflows.stage6_planning_child_recovery import (
        PlanningChildRecoveryCapability,
    )

    if (
        type(capability) is not PlanningChildRecoveryCapability
        or type(resource_payload) is not bytes
        or not resource_payload
        or any(not isinstance(row, Mapping) for row in resource_rows)
    ):
        raise Stage6WorkflowError(
            "Stage 6 planning child resource capability drifted"
        )
    prefixes = capability.acceptance_binding.get("journal_prefixes")
    binding = (
        prefixes.get("resource_audit.jsonl")
        if isinstance(prefixes, Mapping)
        else None
    )
    if not isinstance(binding, Mapping):
        raise Stage6WorkflowError(
            "Stage 6 planning child resource startup binding is missing"
        )
    size_bytes = binding.get("size_bytes")
    line_count = binding.get("line_count")
    expected_sha256 = binding.get("sha256")
    if (
        type(size_bytes) is not int
        or size_bytes <= 0
        or type(line_count) is not int
        or line_count <= 0
        or not _is_sha256(expected_sha256)
        or len(resource_payload) < size_bytes
        or hashlib.sha256(resource_payload[:size_bytes]).hexdigest()
        != expected_sha256
        or resource_payload[:size_bytes].count(b"\n") != line_count
        or len(resource_rows) < line_count
    ):
        raise Stage6WorkflowError(
            "Stage 6 planning child resource startup bytes drifted"
        )
    return {
        "startup_prefix": {
            "size_bytes": size_bytes,
            "sha256": expected_sha256,
            "line_count": line_count,
        },
        "current_payload": {
            "size_bytes": len(resource_payload),
            "sha256": hashlib.sha256(resource_payload).hexdigest(),
            "line_count": resource_payload.count(b"\n"),
        },
    }


def _stage6_acceptance_repair_bindings(
    *,
    source_repair: object | None,
    planning_child_source_repair: object | None,
) -> dict[str, object]:
    if (
        source_repair is not None
        and planning_child_source_repair is not None
    ):
        raise Stage6WorkflowError(
            "Stage 6 classic and planning child acceptance bindings are "
            "mutually exclusive"
        )
    return {
        "source_repair_binding": (
            getattr(source_repair, "summary_binding")
            if source_repair is not None
            else None
        ),
        "planning_child_source_repair_binding": (
            getattr(
                planning_child_source_repair,
                "evidence_binding",
            )()
            if planning_child_source_repair is not None
            else None
        ),
    }


def _verify_stage6_planning_warm_start_lineage(
    *,
    lineage_audit: Mapping[str, object],
    repo_root: Path,
    stage: Path,
    config_bytes: bytes,
    config_sha256: str,
    origin_review_authorization: Mapping[str, object],
) -> tuple[object, dict[str, object]]:
    from lunar_exploration_ppo.workflows.stage6_planning_warm_start import (
        Stage6PlanningWarmStartError,
        build_planning_child_acceptance_profile,
        planning_warm_start_expected_bindings_from_record,
        validate_planning_warm_start_artifact,
        verify_parent_u74_bundle,
    )

    warm_evidence = lineage_audit.get("planning_warm_start")
    if (
        not isinstance(warm_evidence, Mapping)
        or set(warm_evidence)
        != {
            "artifact",
            "artifact_sha256",
            "artifact_size_bytes",
        }
        or not isinstance(warm_evidence.get("artifact"), Mapping)
    ):
        raise Stage6WorkflowError(
            "Stage 6 planning warm-start lineage evidence drifted"
        )
    artifact_payload = (
        json.dumps(
            warm_evidence["artifact"],
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    if (
        warm_evidence.get("artifact_sha256")
        != hashlib.sha256(artifact_payload).hexdigest()
        or warm_evidence.get("artifact_size_bytes")
        != len(artifact_payload)
    ):
        raise Stage6WorkflowError(
            "Stage 6 planning warm-start artifact identity drifted"
        )
    try:
        expected_warm_bindings = (
            planning_warm_start_expected_bindings_from_record(
                repo_root=repo_root,
                review_authorization_record=origin_review_authorization,
            )
        )
        context = validate_planning_warm_start_artifact(
            warm_evidence["artifact"],  # type: ignore[arg-type]
            expected_child_run_id=stage.parent.name,
            expected_child_config_bytes=config_bytes,
            expected_bindings=expected_warm_bindings,
        )
        acceptance_profile = build_planning_child_acceptance_profile()
        verified_parent = verify_parent_u74_bundle(
            Path(
                "D:/xunce/out/ppo_frontier/"
                "s6-standard-single-r1-20260718T220434Z/s6"
            )
        )
    except Stage6PlanningWarmStartError as exc:
        raise Stage6WorkflowError(
            "Stage 6 planning warm-start lineage replay failed"
        ) from exc
    if (
        verified_parent.resource_accepted is not True
        or verified_parent.u75_attempt1_discarded is not True
        or context.child_effective_config_sha256 != config_sha256
    ):
        raise Stage6WorkflowError(
            "Stage 6 planning warm-start parent/config binding drifted"
        )
    return context, acceptance_profile


def _verify_stage6_machine_acceptance_from_bound_graph(
    *,
    stage_root: str | Path,
    repo_root: str | Path,
    gate_path: str | Path | None = None,
    require_terminal: bool,
    summary_override: Mapping[str, object] | None = None,
    routing_override: Mapping[str, object] | None = None,
    reports_override: Mapping[str, bytes] | None = None,
    evidence_handle: object,
    planning_child_recovery_capability: object | None = None,
) -> dict[str, object]:
    """Replay semantic machine acceptance on top of the recursive manifest."""

    from lunar_exploration_ppo.configs.stage6 import (
        SafetyContract,
        parse_stage6_config_bytes,
    )
    from lunar_exploration_ppo.ppo.checkpoint import (
        CheckpointError,
        inspect_complete_checkpoint_snapshot,
        load_complete_checkpoint_snapshot,
    )
    from lunar_exploration_ppo.ppo.checkpoint_retention import (
        CheckpointRetentionError,
        verify_retained_checkpoint_snapshot,
    )
    from lunar_exploration_ppo.ppo.standard_training import (
        CheckpointReceiptIndex,
        StandardTrainingError,
        ValidationRecord,
        _immutable_bindings_valid,
        _review_authorization_immutable_bindings,
        _runtime_value_sha256,
        build_stage6_acceptance_artifacts,
        build_standard_final_aggregate_artifacts,
        build_standard_training_transactions,
        select_global_best,
        validate_checkpoint_replay_audit,
        validate_math_audit,
        validate_standard_collection_audit,
        validate_standard_training_validation_rows,
        validate_stage6_resource_audit,
        verify_standard_final_evaluation_artifacts_from_bytes,
        verify_journal_checkpoint_bindings,
    )
    from lunar_exploration_ppo.ppo.trainer import (
        PPOTrainingError,
        policy_state_sha256,
        validate_ppo_math_evidence,
    )

    stage = _require_plain_stage6_root(
        stage_root,
        label="Stage 6 machine verification root",
    )
    bound_read = getattr(evidence_handle, "read_bytes", None)
    if not callable(bound_read):
        raise Stage6WorkflowError("Stage 6 evidence handle is invalid")

    def read_evidence(relative: str, label: str) -> bytes:
        try:
            payload = bound_read(relative, label=label)
        except Exception as exc:
            raise Stage6WorkflowError(f"{label} bound read failed") from exc
        if type(payload) is not bytes:
            raise Stage6WorkflowError(f"{label} bound read drifted")
        return payload

    snapshot_directories = getattr(evidence_handle, "directory_snapshot", None)
    if not callable(snapshot_directories):
        raise Stage6WorkflowError("Stage 6 evidence directory handle is invalid")
    try:
        directory_snapshot = snapshot_directories()
    except Exception as exc:
        raise Stage6WorkflowError("Stage 6 evidence directory snapshot failed") from exc
    if not isinstance(directory_snapshot, list):
        raise Stage6WorkflowError("Stage 6 evidence directory snapshot drifted")

    def directory_member_names(relative: str, label: str) -> tuple[str, ...]:
        matches = [
            row
            for row in directory_snapshot
            if isinstance(row, Mapping) and row.get("path") == relative
        ]
        if len(matches) != 1 or set(matches[0]) != {"path", "identity", "members"}:
            raise Stage6WorkflowError(f"{label} directory snapshot drifted")
        members = matches[0].get("members")
        if not isinstance(members, list):
            raise Stage6WorkflowError(f"{label} member snapshot drifted")
        names: list[str] = []
        for member in members:
            if (
                not isinstance(member, Mapping)
                or set(member) != {"name", "kind"}
                or member.get("kind") != "file"
                or not isinstance(member.get("name"), str)
            ):
                raise Stage6WorkflowError(f"{label} member snapshot drifted")
            names.append(str(member["name"]))
        if names != sorted(names) or len(names) != len(set(names)):
            raise Stage6WorkflowError(f"{label} member snapshot drifted")
        return tuple(names)

    def read_checkpoint_snapshot(relative: str, *, label: str) -> bytes:
        return read_evidence(relative, label)

    repo = Path(repo_root).expanduser().resolve()
    if (
        type(require_terminal) is not bool
        or ((summary_override is None) != (routing_override is None))
        or ((summary_override is None) != (reports_override is None))
        or (require_terminal and summary_override is not None)
    ):
        raise Stage6WorkflowError("Stage 6 machine verification mode drifted")
    authority = verify_frozen_stage5_authority(
        gate_path=(CANONICAL_STAGE5_GATE if gate_path is None else gate_path),
        repo_root=repo,
    )
    handle = (
        verify_stage6_manifest(
            stage_root=stage,
            repo_root=repo,
            planning_child_recovery_capability=(
                planning_child_recovery_capability
            ),
        )
        if require_terminal
        else None
    )
    if summary_override is None:
        summary = _strict_canonical_json(
            read_evidence("summary.json", "Stage 6 summary"), "Stage 6 summary"
        )
        routing = _strict_canonical_json(
            read_evidence("routing.json", "Stage 6 routing"), "Stage 6 routing"
        )
    else:
        if not isinstance(summary_override, Mapping) or not isinstance(
            routing_override, Mapping
        ):
            raise Stage6WorkflowError("Stage 6 pending acceptance payload drifted")
        summary = dict(summary_override)
        routing = dict(routing_override)
    global_best = _strict_canonical_json(
        read_evidence("global-best.json", "Stage 6 global best"),
        "Stage 6 global best",
    )
    checkpoint_manifest = _strict_canonical_json(
        read_evidence(
            "standard_checkpoint_manifest.json",
            "Stage 6 checkpoint manifest",
        ),
        "Stage 6 checkpoint manifest",
    )
    config_bytes = read_evidence("config.json", "Stage 6 config")
    planning_warm_start = False
    planning_warm_start_context = None
    acceptance_profile = None
    try:
        config = parse_stage6_config_bytes(config_bytes)
    except ValueError as base_config_error:
        from lunar_exploration_ppo.workflows.stage6_planning_warm_start import (
            Stage6PlanningWarmStartError,
            parse_planning_effective_config_bytes,
        )

        try:
            effective_config = parse_planning_effective_config_bytes(
                config_bytes
            )
        except Stage6PlanningWarmStartError as exc:
            raise Stage6WorkflowError(
                "Stage 6 persisted config is invalid"
            ) from base_config_error
        config = effective_config.base_config
        planning_warm_start = True
    config_sha256 = hashlib.sha256(config_bytes).hexdigest()
    safety_contract = SafetyContract.from_stage6_config(config)
    authority_identity = authority.identity
    if any(
        authority_identity.get(key) != expected
        for key, expected in {
            "gate_sha256": config.stage5_authority.gate_sha256,
            "commit_sha256": config.stage5_authority.commit_sha256,
            "commit_tree": config.stage5_authority.commit_tree,
            "review_sha256": config.stage5_authority.review_sha256,
            "manifest_sha256": config.stage5_authority.manifest_sha256,
            "checkpoint_sha256": config.stage5_authority.checkpoint_sha256,
            "policy_state_sha256": config.stage5_authority.policy_state_sha256,
            "performance_advantage_established": False,
        }.items()
    ):
        raise Stage6WorkflowError("Stage 6 frozen authority identity drifted")
    lineage_audit = _strict_canonical_json(
        read_evidence("lineage_audit.json", "Stage 6 lineage audit"),
        "Stage 6 lineage audit",
    )
    expected_lineage_fields = {
        "schema_version",
        "execution_identity",
        "immutable_bindings",
        "verified_review_authorization",
    }
    expected_lineage_schema = "stage6_lineage_audit/v2"
    if planning_warm_start:
        expected_lineage_fields.add("planning_warm_start")
        expected_lineage_schema = "stage6_lineage_audit/v3"
    if (
        set(lineage_audit) != expected_lineage_fields
        or lineage_audit.get("schema_version")
        != expected_lineage_schema
    ):
        raise Stage6WorkflowError("Stage 6 lineage audit schema drifted")
    recomputed_identity = stage6_execution_identity(
        repo_root=repo,
        config_path=repo / "configs/ppo_highres_frontier_stage6_v1.json",
        effective_config_bytes=(config_bytes if planning_warm_start else None),
    )
    current_expected_immutable = {
        key: recomputed_identity.get(key)
        for key in (
            "config_sha256",
            "source_set_sha256",
            "prospective_tree_sha256",
            "data_sha256",
            "environment_identity",
            "environment_sha256",
        )
    }
    historical_expected_immutable: dict[str, object] | None = None
    ordinal6_current_binding = False
    source_repair = None
    planning_child_source_repair = None
    planning_child_current_binding = False
    origin_review_authorization: Mapping[str, object] | None = None
    evidence_paths = getattr(evidence_handle, "file_paths", ())
    if not isinstance(evidence_paths, tuple):
        raise Stage6WorkflowError("Stage 6 evidence path identity drifted")
    classic_repair_paths = set(
        STAGE6_OPTIONAL_MANIFEST_BOUND_ARTIFACTS
    ).intersection(evidence_paths)
    planning_child_present = (
        STAGE6_PLANNING_CHILD_SOURCE_REPAIR_MANIFEST_ARTIFACT
        in evidence_paths
    )
    planning_child_continuation_present = (
        STAGE6_PLANNING_CHILD_SOURCE_REPAIR_CONTINUATION_MANIFEST_ARTIFACT
        in evidence_paths
    )
    if (
        planning_child_recovery_capability is not None
        and not planning_child_present
    ):
        raise Stage6WorkflowError(
            "Stage 6 recovery capability has no planning child graph"
        )
    if planning_child_continuation_present and not planning_child_present:
        raise Stage6WorkflowError(
            "Stage 6 planning child continuation has no parent amendment"
        )
    if planning_child_present and classic_repair_paths:
        raise Stage6WorkflowError(
            "Stage 6 planning child and classic source-repair evidence "
            "profiles are mutually exclusive"
        )
    if planning_child_present and not planning_warm_start:
        raise Stage6WorkflowError(
            "Stage 6 planning child source-repair requires planning "
            "warm-start lineage"
        )
    if (
        "source-repair-continuation.json" in evidence_paths
        and "source-repair-amendment.json" not in evidence_paths
    ):
        raise Stage6WorkflowError(
            "Stage 6 source-repair continuation has no primary amendment"
        )
    if (
        "source-repair-supplement.json" in evidence_paths
        and "source-repair-continuation.json" not in evidence_paths
    ):
        raise Stage6WorkflowError(
            "Stage 6 source-repair supplement has no continuation"
        )
    if (
        "source-repair-closure.json" in evidence_paths
        and "source-repair-supplement.json" not in evidence_paths
    ):
        raise Stage6WorkflowError(
            "Stage 6 source-repair closure has no supplement"
        )
    if (
        "source-repair-frontier-recovery.json" in evidence_paths
        and "source-repair-closure.json" not in evidence_paths
    ):
        raise Stage6WorkflowError(
            "Stage 6 source-repair frontier recovery has no closure"
        )
    if (
        "source-repair-sensor-acceleration.json" in evidence_paths
        and "source-repair-frontier-recovery.json" not in evidence_paths
    ):
        raise Stage6WorkflowError(
            "Stage 6 source-repair sensor acceleration has no frontier recovery"
        )
    if planning_warm_start and any(
        path.startswith("source-repair-") for path in evidence_paths
    ):
        raise Stage6WorkflowError(
            "Stage 6 planning warm-start cannot use classic source-repair "
            "lineage"
        )
    if "source-repair-amendment.json" in evidence_paths:
        source_repair = _load_stage6_source_repair_for_current_identity(
            stage=stage,
            current_execution_identity=recomputed_identity,
            amendment_payload=read_evidence(
                "source-repair-amendment.json",
                "Stage 6 source-repair amendment",
            ),
            continuation_payload=(
                read_evidence(
                    "source-repair-continuation.json",
                    "Stage 6 source-repair continuation",
                )
                if "source-repair-continuation.json" in evidence_paths
                else None
            ),
            supplement_payload=(
                read_evidence(
                    "source-repair-supplement.json",
                    "Stage 6 source-repair supplement",
                )
                if "source-repair-supplement.json" in evidence_paths
                else None
            ),
            closure_payload=(
                read_evidence(
                    "source-repair-closure.json",
                    "Stage 6 source-repair closure",
                )
                if "source-repair-closure.json" in evidence_paths
                else None
            ),
            frontier_recovery_payload=(
                read_evidence(
                    "source-repair-frontier-recovery.json",
                    "Stage 6 source-repair frontier recovery",
                )
                if "source-repair-frontier-recovery.json" in evidence_paths
                else None
            ),
            sensor_acceleration_payload=(
                read_evidence(
                    "source-repair-sensor-acceleration.json",
                    "Stage 6 source-repair sensor acceleration",
                )
                if "source-repair-sensor-acceleration.json" in evidence_paths
                else None
            ),
        )
        if (
            dict(source_repair.origin_execution_identity)
            != lineage_audit.get("execution_identity")
            or dict(source_repair.origin_immutable_bindings)
            != lineage_audit.get("immutable_bindings")
            or dict(source_repair.origin_verified_review_authorization)
            != lineage_audit.get("verified_review_authorization")
        ):
            raise Stage6WorkflowError("Stage 6 source-repair origin drifted")
        historical_expected_immutable = dict(
            source_repair.origin_immutable_bindings
        )
        review_authorization = dict(
            source_repair.current_verified_review_authorization
        )
        current_expected_immutable["stage5_gate_sha256"] = authority_identity[
            "gate_sha256"
        ]
        current_expected_immutable.update(
            _review_authorization_immutable_bindings(review_authorization)
        )
        if source_repair.sensor_acceleration_sha256 is not None:
            current_expected_immutable.update(
                {
                    "coverage_cache_manifest_path": str(
                        STAGE6_COVERAGE_CACHE_MANIFEST_PATH
                    ).replace("\\", "/"),
                    "coverage_cache_manifest_sha256": (
                        STAGE6_COVERAGE_CACHE_MANIFEST_SHA256
                    ),
                    "coverage_cache_manifest_size_bytes": (
                        STAGE6_COVERAGE_CACHE_MANIFEST_SIZE_BYTES
                    ),
                    "coverage_cache_root": str(
                        STAGE6_COVERAGE_CACHE_ROOT
                    ).replace("\\", "/"),
                    "coverage_cache_entry_set_sha256": (
                        STAGE6_COVERAGE_CACHE_ENTRY_SET_SHA256
                    ),
                    "coverage_cache_runtime_mode": (
                        STAGE6_COVERAGE_CACHE_RUNTIME_MODE
                    ),
                    "coverage_cache_formal_audit_sha256": (
                        STAGE6_COVERAGE_CACHE_FORMAL_AUDIT_SHA256
                    ),
                    "coverage_cache_formal_audit_size_bytes": (
                        STAGE6_COVERAGE_CACHE_FORMAL_AUDIT_SIZE_BYTES
                    ),
                }
            )
        if dict(source_repair.current_immutable_bindings) != current_expected_immutable:
            raise Stage6WorkflowError("Stage 6 source-repair current lineage drifted")
        ordinal6_current_binding = source_repair.sensor_acceleration_sha256 is not None
        expected_immutable = (
            dict(source_repair.current_immutable_bindings)
            if ordinal6_current_binding
            else dict(source_repair.origin_immutable_bindings)
        )
    elif planning_child_present:
        from lunar_exploration_ppo.workflows.stage6_planning_child_recovery import (
            PlanningChildRecoveryCapability,
        )

        if (
            not planning_child_continuation_present
            or type(planning_child_recovery_capability)
            is not PlanningChildRecoveryCapability
        ):
            raise Stage6WorkflowError(
                "Stage 6 planning child machine verifier requires one "
                "issued recovery capability"
            )
        planning_child_source_repair = (
            planning_child_recovery_capability
        )
        _require_planning_child_recovery_capability(
            planning_child_source_repair,
            "machine verifier startup",
        )
        if (
            planning_child_source_repair.stage_root != stage
            or planning_child_source_repair.formal_run_id
            != stage.parent.name
        ):
            raise Stage6WorkflowError(
                "Stage 6 planning child machine capability root drifted"
            )
        child_payload = read_evidence(
            STAGE6_PLANNING_CHILD_SOURCE_REPAIR_MANIFEST_ARTIFACT,
            "Stage 6 planning child parent graph member",
        )
        continuation_payload = read_evidence(
            STAGE6_PLANNING_CHILD_SOURCE_REPAIR_CONTINUATION_MANIFEST_ARTIFACT,
            "Stage 6 planning child continuation graph member",
        )
        if (
            hashlib.sha256(child_payload).hexdigest()
            != planning_child_source_repair.parent_artifact_sha256
            or hashlib.sha256(continuation_payload).hexdigest()
            != planning_child_source_repair.continuation_artifact_sha256
        ):
            raise Stage6WorkflowError(
                "Stage 6 planning child machine graph identity drifted"
            )
        review_record = (
            planning_child_source_repair.acceptance_binding.get(
                "current_verified_review_authorization"
            )
        )
        if not isinstance(review_record, Mapping):
            raise Stage6WorkflowError(
                "Stage 6 planning child current review is missing"
            )
        review_authorization = (
            validate_stage6_verified_review_authorization(
                review_record,
                execution_identity=recomputed_identity,
                formal_run_id=stage.parent.name,
            )
        )
        current_expected_immutable = (
            _stage6_current_immutable_bindings(
                execution_identity=recomputed_identity,
                verified_review_authorization=review_authorization,
                stage5_authority=authority,
            )
        )
        validated_child = _validate_planning_child_machine_context(
            capability=planning_child_source_repair,
            lineage_audit=lineage_audit,
            current_execution_identity=recomputed_identity,
            current_verified_review_authorization=(
                review_authorization
            ),
            current_immutable_bindings=current_expected_immutable,
        )
        historical_expected_immutable = dict(
            validated_child["historical_immutable_bindings"]
        )
        expected_immutable = dict(
            validated_child["current_immutable_bindings"]
        )
        origin_identity = lineage_audit.get("execution_identity")
        origin_review = lineage_audit.get(
            "verified_review_authorization"
        )
        if not isinstance(origin_identity, Mapping) or not isinstance(
            origin_review,
            Mapping,
        ):
            raise Stage6WorkflowError(
                "Stage 6 planning child origin review lineage drifted"
            )
        origin_review_authorization = (
            validate_stage6_verified_review_authorization(
                origin_review,
                execution_identity=origin_identity,
                formal_run_id=stage.parent.name,
            )
        )
        planning_child_current_binding = True
    else:
        if lineage_audit.get("execution_identity") != recomputed_identity:
            raise Stage6WorkflowError("Stage 6 execution identity drifted")
        review_authorization = validate_stage6_verified_review_authorization(
            lineage_audit.get("verified_review_authorization"),  # type: ignore[arg-type]
            execution_identity=recomputed_identity,
            formal_run_id=stage.parent.name,
        )
        origin_review_authorization = review_authorization
        current_expected_immutable["stage5_gate_sha256"] = authority_identity[
            "gate_sha256"
        ]
        current_expected_immutable.update(
            _review_authorization_immutable_bindings(review_authorization)
        )
        expected_immutable = current_expected_immutable
        historical_expected_immutable = dict(expected_immutable)
    if planning_warm_start:
        if origin_review_authorization is None:
            raise Stage6WorkflowError(
                "Stage 6 planning warm-start origin review is missing"
            )
        (
            planning_warm_start_context,
            acceptance_profile,
        ) = _verify_stage6_planning_warm_start_lineage(
            lineage_audit=lineage_audit,
            repo_root=repo,
            stage=stage,
            config_bytes=config_bytes,
            config_sha256=config_sha256,
            origin_review_authorization=origin_review_authorization,
        )
        current_expected_immutable.update(
            {
                "coverage_cache_manifest_path": str(
                    STAGE6_COVERAGE_CACHE_MANIFEST_PATH
                ).replace("\\", "/"),
                "coverage_cache_manifest_sha256": (
                    STAGE6_COVERAGE_CACHE_MANIFEST_SHA256
                ),
                "coverage_cache_manifest_size_bytes": (
                    STAGE6_COVERAGE_CACHE_MANIFEST_SIZE_BYTES
                ),
                "coverage_cache_root": str(
                    STAGE6_COVERAGE_CACHE_ROOT
                ).replace("\\", "/"),
                "coverage_cache_entry_set_sha256": (
                    STAGE6_COVERAGE_CACHE_ENTRY_SET_SHA256
                ),
                "coverage_cache_runtime_mode": (
                    STAGE6_COVERAGE_CACHE_RUNTIME_MODE
                ),
                "coverage_cache_formal_audit_sha256": (
                    STAGE6_COVERAGE_CACHE_FORMAL_AUDIT_SHA256
                ),
                "coverage_cache_formal_audit_size_bytes": (
                    STAGE6_COVERAGE_CACHE_FORMAL_AUDIT_SIZE_BYTES
                ),
            }
        )
        expected_immutable = dict(current_expected_immutable)
        if planning_child_source_repair is None:
            historical_expected_immutable = dict(expected_immutable)
    if (
        historical_expected_immutable is None
        or not _immutable_bindings_valid(historical_expected_immutable)
        or not _immutable_bindings_valid(expected_immutable)
        or lineage_audit.get("immutable_bindings")
        != historical_expected_immutable
        or any(
            key not in summary or summary[key] != value
            for key, value in expected_immutable.items()
        )
    ):
        raise Stage6WorkflowError("Stage 6 immutable lineage binding drifted")
    scenario_audit = _strict_canonical_json(
        read_evidence(
            "scenario_split_audit.json",
            "Stage 6 scenario split audit",
        ),
        "Stage 6 scenario split audit",
    )
    if (
        scenario_audit.get("schema_version")
        != "stage6_scenario_split_audit/v1"
        or scenario_audit.get("catalog_sha256")
        != recomputed_identity.get("catalog_sha256")
        or not isinstance(scenario_audit.get("spatial_audit"), Mapping)
        or scenario_audit["spatial_audit"].get("parent_cross_split_count") != 0  # type: ignore[index]
    ):
        raise Stage6WorkflowError("Stage 6 catalog/scenario lineage drifted")
    if (
        summary.get("state") != "awaiting_independent_review"
        or summary.get("machine_passed") is not True
        or summary.get("acceptance_scope")
        != (
            "single_seed_planning_u74_warm_start_closure/v1"
            if planning_warm_start
            else "single_seed_system_closure/v1"
        )
        or summary.get("optional_seed_extension_blocks_next_stage") is not False
        or summary.get("optional_seed_extension_trigger")
        != "explicit_user_request_only/v1"
        or summary.get("final_evaluation_count") != 10
        or summary.get("final_episode_count") != 640
        or routing.get("state") != summary["state"]
        or routing.get("route") != "awaiting_independent_review"
        or routing.get("machine_passed") is not True
        or routing.get("next_stage_after_human_approval")
        != "ppo_highres_frontier_stage7_kilometer_stress/v1"
        or routing.get("performance_advantage_established")
        != summary.get("performance_advantage_established")
        or (
            planning_warm_start
            and (
                summary.get("acceptance_profile")
                != acceptance_profile
                or routing.get("acceptance_profile")
                != acceptance_profile
            )
        )
    ):
        raise Stage6WorkflowError("Stage 6 final evaluation acceptance drifted")

    phase_rows = Stage6StateJournal.verify_snapshot_bytes(
        read_evidence("phase-state.jsonl", "Stage 6 phase journal")
    )
    terminal_phase_states = (
        "preflight",
        "global_best_frozen",
        "final_test_running",
        "final_unseen_running",
        "baselines_running",
        "machine_passed",
        "awaiting_independent_review",
    )
    expected_phase_states = (
        terminal_phase_states if require_terminal else terminal_phase_states[:5]
    )
    if tuple(row["state"] for row in phase_rows) != expected_phase_states:
        raise Stage6WorkflowError("Stage 6 terminal phase chain drifted")

    metrics = _strict_canonical_jsonl(
        read_evidence("metrics.jsonl", "Stage 6 final metrics"),
        "Stage 6 final metrics",
    )
    expected_order = tuple(
        f"final:{split}:{method}"
        for split in ("test", "unseen")
        for method in FINAL_EVALUATION_METHODS
    )
    expected_keys = set(expected_order)
    if (
        len(metrics) != 10
        or {row.get("transaction_key") for row in metrics} != expected_keys
        or tuple(row.get("transaction_key") for row in metrics) != expected_order
        or any(
            not isinstance(row.get("result"), Mapping)
            or row["result"].get("episode_count") != 64  # type: ignore[union-attr]
            for row in metrics
        )
    ):
        raise Stage6WorkflowError("Stage 6 final evaluation metrics drifted")
    metric_by_key = {str(row["transaction_key"]): row for row in metrics}
    verified_evaluations: dict[str, dict[str, object]] = {}
    try:
        for split in ("test", "unseen"):
            for method in FINAL_EVALUATION_METHODS:
                key = f"{split}:{method}"
                row = metric_by_key[f"final:{key}"]
                if set(row) != {
                    "transaction_key",
                    "kind",
                    "split",
                    "method",
                    "result",
                } or (
                    row["kind"],
                    row["split"],
                    row["method"],
                ) != ("final_evaluation", split, method):
                    raise StandardTrainingError("final metric row schema drifted")
                trace_name = f"final-{split}-{method}.jsonl"
                summary_name = f"final-{split}-{method}.summary.json"
                commit_name = f"final-{split}-{method}.commit.json"
                replay = verify_standard_final_evaluation_artifacts_from_bytes(
                    trace_bytes=read_evidence(
                        f"episode-traces/{trace_name}",
                        f"Stage 6 final evaluation trace {split}:{method}",
                    ),
                    trace_name=trace_name,
                    summary_bytes=read_evidence(
                        f"episode-traces/{summary_name}",
                        f"Stage 6 final evaluation summary {split}:{method}",
                    ),
                    summary_name=summary_name,
                    commit_bytes=read_evidence(
                        f"episode-traces/{commit_name}",
                        f"Stage 6 final evaluation commit {split}:{method}",
                    ),
                    commit_name=commit_name,
                    split=split,
                    method=method,
                    config_sha256=config_sha256,
                    safety_contract=safety_contract,
                    checkpoint_sha256=str(global_best["checkpoint_sha256"]),
                    policy_state_sha256=str(global_best["policy_state_sha256"]),
                    bootstrap_resamples=config.evaluation.bootstrap_resamples,
                    bootstrap_seed=config.evaluation.bootstrap_seed,
                )
                if replay["result"] != row["result"]:
                    raise StandardTrainingError("final metric row did not match trace")
                verified_evaluations[key] = replay
        fairness_file = _strict_canonical_json(
            read_evidence("fairness_audit.json", "Stage 6 fairness audit"),
            "Stage 6 fairness audit",
        )
        isolation = fairness_file.get("eval_only_isolation")
        if not isinstance(isolation, Mapping):
            raise StandardTrainingError("final isolation audit is missing")
        aggregate = build_standard_final_aggregate_artifacts(
            verified_evaluations,
            isolation,  # type: ignore[arg-type]
            safety_contract=safety_contract,
            config_sha256=config_sha256,
        )
    except StandardTrainingError as exc:
        raise Stage6WorkflowError(
            "Stage 6 final evaluation trace replay failed"
        ) from exc
    expected_aggregate_files = {
        "fairness_audit.json": aggregate["fairness_audit"],
        "leakage_audit.json": aggregate["leakage_audit"],
        "failure_audit.json": aggregate["failure_audit"],
    }
    for name, expected in expected_aggregate_files.items():
        actual = _strict_canonical_json(
            read_evidence(name, f"Stage 6 {name}"), f"Stage 6 {name}"
        )
        if actual != expected:
            raise Stage6WorkflowError(f"Stage 6 {name} did not match trace replay")
    if planning_warm_start:
        from lunar_exploration_ppo.ppo.collector import (
            PLANNER_FAILURE_REASONS,
        )

        failure_audit = aggregate["failure_audit"]
        planner_counts = failure_audit.get("planner_failure_counts")
        reset_scan_audit = failure_audit.get("reset_scan_audit")
        if (
            failure_audit.get("schema_version")
            != "stage6_failure_audit/v3"
            or not isinstance(planner_counts, Mapping)
            or set(planner_counts) != set(PLANNER_FAILURE_REASONS)
            or any(
                type(planner_counts.get(reason)) is not int
                or int(planner_counts[reason]) < 0
                for reason in PLANNER_FAILURE_REASONS
            )
            or not isinstance(reset_scan_audit, Mapping)
            or reset_scan_audit.get("scan_order")
            != ["reset_local_safety", "reset_exploration"]
            or reset_scan_audit.get("dual_scan_episode_count") != 640
            or reset_scan_audit.get("passed") is not True
        ):
            raise Stage6WorkflowError(
                "Stage 6 planning diagnostics final audit drifted"
            )
    if (
        read_evidence(
            "standard_baseline_comparison.csv",
            "Stage 6 baseline comparison CSV",
        )
        != aggregate["comparison_csv"]
        or read_evidence(
            "standard_coverage_curves.csv",
            "Stage 6 coverage curves CSV",
        )
        != aggregate["coverage_curves_csv"]
    ):
        raise Stage6WorkflowError("Stage 6 final evaluation CSV replay drifted")

    def replayed_primary_ci(key: str, bound: str) -> float:
        try:
            value = verified_evaluations[key]["result"]
            if not isinstance(value, Mapping):
                raise TypeError
            bootstrap = value["bootstrap_audit"]
            if not isinstance(bootstrap, Mapping):
                raise TypeError
            intervals = bootstrap["metrics"]
            if not isinstance(intervals, Mapping):
                raise TypeError
            interval = intervals["success_rate_under_fixed_step_budget"]
            if not isinstance(interval, Mapping):
                raise TypeError
            candidate = interval[bound]
            if not isinstance(candidate, (int, float)) or not math.isfinite(
                float(candidate)
            ):
                raise TypeError
            return float(candidate)
        except (KeyError, TypeError, ValueError) as exc:
            raise Stage6WorkflowError(
                "Stage 6 performance CI replay drifted"
            ) from exc

    replayed_performance = decide_performance_advantage(
        ppo_ci95_low=min(
            replayed_primary_ci(f"{split}:ppo_policy", "ci95_low")
            for split in ("test", "unseen")
        ),
        gain_over_cost_ci95_high=max(
            replayed_primary_ci(
                f"{split}:gain_over_cost_frontier",
                "ci95_high",
            )
            for split in ("test", "unseen")
        ),
    )
    if (
        summary.get("cross_seed_performance_conclusion") is not False
        or any(
            summary.get(key) != expected
            for key, expected in {
                "performance_advantage_established": replayed_performance.established,
                "performance_claim": replayed_performance.claim,
                "ppo_ci95_low": replayed_performance.ppo_ci95_low,
                "gain_over_cost_ci95_high": (
                    replayed_performance.gain_over_cost_ci95_high
                ),
            }.items()
        )
        or routing.get("performance_advantage_established") is not (
            replayed_performance.established
        )
    ):
        raise Stage6WorkflowError(
            "Stage 6 single-seed performance decision replay drifted"
        )

    receipt_bytes = read_evidence(
        "checkpoints/index.jsonl",
        "Stage 6 checkpoint receipt index",
    )
    receipts = CheckpointReceiptIndex.verify_snapshot_bytes(receipt_bytes)
    receipt_by_key = {
        str(receipt["transaction_key"]): receipt for receipt in receipts
    }
    checkpoint_manifest_fields = {
        "schema_version",
        "receipt_count",
        "receipt_index",
        "global_best",
        "receipts",
    }
    if planning_warm_start:
        checkpoint_manifest_fields.add("acceptance_profile")
    if (
        set(checkpoint_manifest) != checkpoint_manifest_fields
        or checkpoint_manifest["schema_version"]
        != (
            "stage6_checkpoint_manifest/v2"
            if planning_warm_start
            else "stage6_checkpoint_manifest/v1"
        )
        or (
            planning_warm_start
            and checkpoint_manifest.get("acceptance_profile")
            != acceptance_profile
        )
        or checkpoint_manifest["receipt_count"] != len(receipts)
        or checkpoint_manifest["receipts"] != list(receipts)
        or checkpoint_manifest["global_best"] != global_best
        or checkpoint_manifest["receipt_index"]
        != {
            "path": "checkpoints/index.jsonl",
            "sha256": hashlib.sha256(receipt_bytes).hexdigest(),
            "size_bytes": len(receipt_bytes),
        }
    ):
        raise Stage6WorkflowError("Stage 6 checkpoint manifest receipt drifted")
    matching = tuple(
        row
        for row in receipts
        if row["transaction_key"] == global_best.get("transaction_key")
        and row["checkpoint_sha256"] == global_best.get("checkpoint_sha256")
        and row["policy_state_sha256"] == global_best.get("policy_state_sha256")
    )
    if len(matching) != 1:
        raise Stage6WorkflowError("Stage 6 global best receipt drifted")

    if planning_warm_start:
        from lunar_exploration_ppo.workflows.stage6_planning_warm_start import (
            build_planning_child_transactions,
        )

        transactions = build_planning_child_transactions(config)
    else:
        transactions = build_standard_training_transactions(config)
    immutable = dict(expected_immutable)
    for index, row in enumerate(phase_rows):
        bindings = row.get("bindings")
        phase_immutable = (
            historical_expected_immutable
            if (
                ordinal6_current_binding
                or planning_child_current_binding
            )
            and index == 0
            else immutable
        )
        assert phase_immutable is not None
        expected_checkpoint = (
            config.stage5_authority.checkpoint_sha256
            if index == 0
            else global_best["checkpoint_sha256"]
        )
        if (
            not isinstance(bindings, Mapping)
            or set(bindings) != {*phase_immutable, "checkpoint_sha256"}
            or any(
                bindings.get(key) != value
                for key, value in phase_immutable.items()
            )
            or bindings.get("checkpoint_sha256") != expected_checkpoint
        ):
            raise Stage6WorkflowError("Stage 6 phase binding drifted")
    job_rows = Stage6StateJournal.verify_snapshot_bytes(
        read_evidence("job-state.jsonl", "Stage 6 job journal")
    )
    completed: tuple[str, ...] | None = None
    if planning_child_source_repair is not None:
        try:
            binding_epochs = validated_child[
                "journal_binding_epochs"
            ]
            if not isinstance(binding_epochs, tuple):
                raise StandardTrainingError(
                    "planning child journal binding epochs drifted"
                )
            completed = verify_journal_checkpoint_bindings(
                job_rows,
                transactions,
                receipts,
                immutable,
                immutable_binding_epochs=binding_epochs,
            )
        except StandardTrainingError as exc:
            raise Stage6WorkflowError(
                "Stage 6 planning child transaction journal/receipt "
                "verification failed"
            ) from exc
    elif not planning_warm_start:
        try:
            if ordinal6_current_binding:
                from lunar_exploration_ppo.workflows.stage6_source_repair import (
                    SENSOR_ACCELERATION_NEXT_TRANSACTION_KEY,
                )

            completed = verify_journal_checkpoint_bindings(
                job_rows,
                transactions,
                receipts,
                immutable,  # type: ignore[arg-type]
                historical_immutable_bindings=(
                    historical_expected_immutable
                    if ordinal6_current_binding
                    else None
                ),
                current_binding_first_transaction_key=(
                    SENSOR_ACCELERATION_NEXT_TRANSACTION_KEY
                    if ordinal6_current_binding
                    else None
                ),
            )
        except StandardTrainingError as exc:
            raise Stage6WorkflowError(
                "Stage 6 transaction journal/receipt verification failed"
            ) from exc
    if len(receipts) != len(transactions) or (
        completed is not None
        and completed
        != tuple(transaction.key for transaction in transactions)
    ):
        raise Stage6WorkflowError("Stage 6 transaction schedule is incomplete")
    if (
        summary.get("checkpoint_receipt_count") != len(transactions)
        or summary.get("config_sha256")
        != hashlib.sha256(config_bytes).hexdigest()
    ):
        raise Stage6WorkflowError("Stage 6 transaction identity drifted")

    training_rows = _strict_canonical_jsonl(
        read_evidence("training_metrics.jsonl", "Stage 6 training metrics"),
        "Stage 6 training metrics",
    )
    validation_rows = _strict_canonical_jsonl(
        read_evidence("validation_metrics.jsonl", "Stage 6 validation metrics"),
        "Stage 6 validation metrics",
    )
    resource_payload = read_evidence(
        "resource_audit.jsonl",
        "Stage 6 resource audit",
    )
    resource_rows = _strict_canonical_jsonl(
        resource_payload,
        "Stage 6 resource audit",
    )
    math_rows = _strict_canonical_jsonl(
        read_evidence("math_audit.jsonl", "Stage 6 math audit"),
        "Stage 6 math audit",
    )
    checkpoint_rows = _strict_canonical_jsonl(
        read_evidence("checkpoint_audit.jsonl", "Stage 6 checkpoint audit"),
        "Stage 6 checkpoint audit",
    )
    transaction_keys = {transaction.key for transaction in transactions}
    validation_keys = {
        transaction.key
        for transaction in transactions
        if transaction.validation_episodes == 16
    }
    audit_keys = {
        transaction.key
        for transaction in transactions
        if transaction.update in {1, 10, 50, 100}
    }
    if (
        planning_warm_start
        and planning_child_source_repair is None
    ):
        from lunar_exploration_ppo.workflows.stage6_planning_warm_start import (
            Stage6PlanningWarmStartError,
            verify_planning_child_terminal_replay,
        )

        try:
            verify_planning_child_terminal_replay(
                config=config,
                receipt_rows=receipts,
                journal_rows=job_rows,
                immutable_bindings=immutable,
                validation_transaction_keys=tuple(
                    row.get("transaction_key")
                    for row in validation_rows
                ),
                audit_transaction_keys=tuple(
                    row.get("transaction_key")
                    for row in checkpoint_rows
                ),
                acceptance_profile=acceptance_profile,  # type: ignore[arg-type]
            )
        except Stage6PlanningWarmStartError as exc:
            raise Stage6WorkflowError(
                "Stage 6 planning child terminal replay failed"
            ) from exc
    try:
        resource_validation = validate_stage6_resource_audit(
            resource_rows,
            transactions,
            require_terminal=require_terminal,
        )
    except StandardTrainingError as exc:
        raise Stage6WorkflowError("Stage 6 resource audit drifted") from exc
    if planning_child_source_repair is not None:
        _validate_planning_child_resource_boundary(
            capability=planning_child_source_repair,
            resource_rows=resource_rows,
            resource_payload=resource_payload,
        )
    if (
        len(validation_rows) != len(validation_keys)
        or {row.get("transaction_key") for row in validation_rows}
        != validation_keys
        or resource_validation.get("update_accepted_count") != len(transactions)
        or resource_validation.get("final_accepted_count") != 10
        or len(checkpoint_rows) != len(audit_keys)
        or {row.get("transaction_key") for row in checkpoint_rows} != audit_keys
        or any(row.get("passed") is not True for row in checkpoint_rows)
    ):
        raise Stage6WorkflowError("Stage 6 transaction metric/audit counts drifted")
    try:
        training_by_key, seed_best_records = (
            validate_standard_training_validation_rows(
                config=config,
                transactions=transactions,
                receipt_rows=receipts,
                training_rows=training_rows,
                validation_rows=validation_rows,
                planning_warm_start=planning_warm_start,
            )
        )
        completed_seed_bests: list[ValidationRecord] = []
        for seed in config.training.seeds:
            value = seed_best_records.get(seed)
            if value is None:
                raise StandardTrainingError("validation seed best is missing")
            completed_seed_bests.append(
                ValidationRecord(
                    seed=int(value["seed"]),
                    update=int(value["update"]),
                    success_rate_under_fixed_step_budget=float(
                        value["success_rate_under_fixed_step_budget"]
                    ),
                    mean_final_coverage=float(value["mean_final_coverage"]),
                    checkpoint_ref=str(value["checkpoint_ref"]),
                )
            )
        selected_global = select_global_best(
            completed_seed_bests,
            configured_seeds=config.training.seeds,
        )
        selected_global_record = selected_global.to_dict()
        selected_global_key = next(
            item.key
            for item in transactions
            if item.seed == selected_global.seed
            and item.update == selected_global.update
        )
        if (
            global_best.get("record") != selected_global_record
            or global_best.get("transaction_key") != selected_global_key
            or summary.get("global_best") != selected_global_record
        ):
            raise StandardTrainingError("validation global best drifted")
        repair_acceptance_bindings = (
            _stage6_acceptance_repair_bindings(
                source_repair=source_repair,
                planning_child_source_repair=(
                    planning_child_source_repair
                ),
            )
        )
        expected_acceptance = build_stage6_acceptance_artifacts(
            global_best=selected_global_record,
            performance_advantage_established=replayed_performance.established,
            performance_claim=replayed_performance.claim,
            ppo_ci95_low=replayed_performance.ppo_ci95_low,
            gain_over_cost_ci95_high=replayed_performance.gain_over_cost_ci95_high,
            final_evaluation_count=len(metrics),
            final_episode_count=sum(
                int(row["result"]["episode_count"])  # type: ignore[index]
                for row in metrics
            ),
            checkpoint_receipt_count=len(receipts),
            immutable_bindings=expected_immutable,
            source_repair_binding=repair_acceptance_bindings[
                "source_repair_binding"
            ],
            planning_child_source_repair_binding=(
                repair_acceptance_bindings[
                    "planning_child_source_repair_binding"
                ]
            ),
            acceptance_profile=acceptance_profile,
        )
        expected_reports = expected_acceptance.get("reports")
        if (
            summary != expected_acceptance.get("summary")
            or routing != expected_acceptance.get("routing")
            or not isinstance(expected_reports, Mapping)
        ):
            raise StandardTrainingError("Stage 6 acceptance semantic replay drifted")
        if reports_override is not None and set(reports_override) != set(
            expected_reports
        ):
            raise StandardTrainingError("Stage 6 pending report set drifted")
        for report_name, expected_bytes in expected_reports.items():
            actual_bytes = (
                reports_override.get(report_name)
                if reports_override is not None
                else read_evidence(report_name, f"Stage 6 report {report_name}")
            )
            if (
                not isinstance(report_name, str)
                or not isinstance(expected_bytes, bytes)
                or not isinstance(actual_bytes, bytes)
                or actual_bytes != expected_bytes
            ):
                raise StandardTrainingError("Stage 6 report semantic replay drifted")

        safety_contract_dict = safety_contract.to_dict()
        current_checkpoint_lineage = _stage6_checkpoint_lineage_base(
            config=config,
            bindings=expected_immutable,
            safety_contract=safety_contract,
        )
        historical_checkpoint_lineage = _stage6_checkpoint_lineage_base(
            config=config,
            bindings=historical_expected_immutable,
            safety_contract=safety_contract,
        )
        def checkpoint_lineage_for_update(update: int) -> dict[str, object]:
            if source_repair is not None:
                return source_repair.checkpoint_lineage_for_update(update)
            if planning_warm_start:
                if planning_warm_start_context is None:
                    raise StandardTrainingError(
                        "planning warm-start context is missing"
                    )
                warm_evidence_value = lineage_audit.get(
                    "planning_warm_start"
                )
                if not isinstance(warm_evidence_value, Mapping):
                    raise StandardTrainingError(
                        "planning warm-start evidence is missing"
                    )
                return _stage6_planning_checkpoint_lineage_for_update(
                    update=update,
                    current_checkpoint_lineage=(
                        current_checkpoint_lineage
                    ),
                    historical_checkpoint_lineage=(
                        historical_checkpoint_lineage
                    ),
                    planning_warm_start_context=(
                        planning_warm_start_context
                    ),
                    planning_warm_start_sha256=str(
                        warm_evidence_value["artifact_sha256"]
                    ),
                    planning_child_source_repair=(
                        planning_child_source_repair
                    ),
                )
            return dict(current_checkpoint_lineage)

        selected_global_snapshot: dict[str, object] | None = None
        for seed in config.training.seeds:
            seed_best = seed_best_records[seed]
            if planning_warm_start:
                expected_updates = tuple(
                    sorted(
                        {
                            int(seed_best["update"]),
                            100,
                            *(
                                set(
                                    planning_child_source_repair
                                    .protected_checkpoint_updates
                                )
                                if planning_child_source_repair
                                is not None
                                else set()
                            ),
                        }
                    )
                )
            else:
                expected_updates = tuple(
                    sorted(
                        {
                            int(seed_best["update"]),
                            50,
                            100,
                            *(
                                source_repair.protected_checkpoint_updates
                                if source_repair is not None
                                else ()
                            ),
                        }
                    )
                )
            checkpoint_root_relative = f"checkpoints/seed-{seed}"
            verify_retained_checkpoint_snapshot(
                root_relative=checkpoint_root_relative,
                schema_version=config.checkpoint.schema_version,
                latest_update=100,
                expected_updates=expected_updates,
                directory_snapshot=directory_snapshot,
                read_bytes=read_checkpoint_snapshot,
            )
            for update in expected_updates:
                checkpoint_directory_relative = (
                    f"{checkpoint_root_relative}/update-{update:08d}"
                )
                snapshot = {
                    "directory": stage / checkpoint_directory_relative,
                    "member_names": directory_member_names(
                        checkpoint_directory_relative,
                        "Stage 6 retained checkpoint",
                    ),
                    "checkpoint_bytes": read_evidence(
                        f"{checkpoint_directory_relative}/checkpoint.pt",
                        "Stage 6 retained checkpoint payload",
                    ),
                    "manifest_bytes": read_evidence(
                        f"{checkpoint_directory_relative}/manifest.json",
                        "Stage 6 retained checkpoint manifest",
                    ),
                    "complete_bytes": read_evidence(
                        f"{checkpoint_directory_relative}/complete.json",
                        "Stage 6 retained checkpoint complete marker",
                    ),
                }
                inspected = inspect_complete_checkpoint_snapshot(
                    directory=snapshot["directory"],  # type: ignore[arg-type]
                    update_step=update,
                    schema_version=config.checkpoint.schema_version,
                    member_names=snapshot["member_names"],  # type: ignore[arg-type]
                    checkpoint_bytes=snapshot["checkpoint_bytes"],  # type: ignore[arg-type]
                    manifest_bytes=snapshot["manifest_bytes"],  # type: ignore[arg-type]
                    complete_bytes=snapshot["complete_bytes"],  # type: ignore[arg-type]
                    expected_config_sha256=str(expected_immutable["config_sha256"]),
                    expected_lineage=checkpoint_lineage_for_update(update),
                    expected_safety_contract=safety_contract,
                )
                transaction_key = next(
                    item.key
                    for item in transactions
                    if item.seed == seed and item.update == update
                )
                receipt = receipt_by_key[transaction_key]
                if (
                    inspected.checkpoint_sha256 != receipt["checkpoint_sha256"]
                    or inspected.policy_state_sha256
                    != receipt["policy_state_sha256"]
                    or hashlib.sha256(
                        snapshot["complete_bytes"]  # type: ignore[arg-type]
                    ).hexdigest()
                    != receipt["complete_marker_sha256"]
                ):
                    raise StandardTrainingError(
                        "retained checkpoint receipt binding drifted"
                    )
                if seed == selected_global.seed and update == selected_global.update:
                    selected_global_snapshot = snapshot

        import torch

        global_policy = load_stage4_policy_for_standard(
            checkpoint_path=CANONICAL_STAGE4_CHECKPOINT,
            checkpoint_sha256=config.stage5_authority.checkpoint_sha256,
            policy_state_sha256=config.stage5_authority.policy_state_sha256,
            device=config.device,
        )
        if not isinstance(global_policy, torch.nn.Module):
            raise StandardTrainingError("global best policy loader drifted")
        global_optimizer = torch.optim.AdamW(
            global_policy.parameters(),
            lr=config.ppo.learning_rate,
            eps=config.ppo.adam_eps,
            weight_decay=config.ppo.weight_decay,
        )
        if selected_global_snapshot is None:
            raise StandardTrainingError("global best checkpoint snapshot is missing")
        loaded_global = load_complete_checkpoint_snapshot(
            directory=selected_global_snapshot["directory"],  # type: ignore[arg-type]
            update_step=selected_global.update,
            schema_version=config.checkpoint.schema_version,
            member_names=selected_global_snapshot["member_names"],  # type: ignore[arg-type]
            checkpoint_bytes=selected_global_snapshot["checkpoint_bytes"],  # type: ignore[arg-type]
            manifest_bytes=selected_global_snapshot["manifest_bytes"],  # type: ignore[arg-type]
            complete_bytes=selected_global_snapshot["complete_bytes"],  # type: ignore[arg-type]
            policy=global_policy,
            optimizer=global_optimizer,
            expected_config_sha256=str(expected_immutable["config_sha256"]),
            expected_lineage=checkpoint_lineage_for_update(
                selected_global.update
            ),
            expected_safety_contract=safety_contract,
        )
        if (
            loaded_global.checkpoint_sha256 != global_best.get("checkpoint_sha256")
            or loaded_global.policy_state_sha256
            != global_best.get("policy_state_sha256")
            or policy_state_sha256(global_policy)
            != global_best.get("policy_state_sha256")
            or loaded_global.best_record != selected_global_record
            or loaded_global.config_sha256 != expected_immutable["config_sha256"]
            or loaded_global.lineage
            != checkpoint_lineage_for_update(selected_global.update)
            or loaded_global.safety_contract != safety_contract_dict
            or len(global_optimizer.param_groups) != 1
            or global_optimizer.param_groups[0].get("lr")
            != config.ppo.learning_rate
            or global_optimizer.param_groups[0].get("eps") != config.ppo.adam_eps
            or global_optimizer.param_groups[0].get("weight_decay")
            != config.ppo.weight_decay
        ):
            raise StandardTrainingError("global best complete checkpoint drifted")

        expected_checkpoint_transactions = tuple(
            item for item in transactions if item.update in {1, 10, 50, 100}
        )
        if len(checkpoint_rows) != len(expected_checkpoint_transactions):
            raise StandardTrainingError("checkpoint replay audit count drifted")
        for transaction, row in zip(
            expected_checkpoint_transactions,
            checkpoint_rows,
            strict=True,
        ):
            receipt = receipt_by_key[transaction.key]
            validate_checkpoint_replay_audit(
                row,
                transaction_key=transaction.key,
                seed=transaction.seed,
                update=transaction.update,
                checkpoint_sha256=str(receipt["checkpoint_sha256"]),
                complete_marker_sha256=str(
                    receipt["complete_marker_sha256"]
                ),
                policy_state_sha256=str(receipt["policy_state_sha256"]),
                lineage_sha256=_runtime_value_sha256(
                    checkpoint_lineage_for_update(transaction.update)
                ),
            )

        math_payload_fields = {
            "observation_finite",
            "action_finite",
            "logprob_finite",
            "value_finite",
            "advantage_finite",
            "return_finite",
            "ratio_finite",
            "loss_finite",
            "kl_finite",
            "grad_finite",
            "mask_violation_count",
            "snapshot_mismatch_count",
            "stale_policy_transition_count",
            "initial_ratio_max_abs_error",
            "device",
            "allowed_initial_ratio_tolerance",
            "compute_dtype",
            "joint_logprob",
            "joint_logprob_factorization_max_abs_error",
            "grad_post_clip_norm_max",
            "evidence_source",
            "evidence_sha256",
            "sample_count",
            "snapshot_list_sha256",
            "initial_forward_sample_count",
            "forward_sample_count",
            "loss_sample_count",
            "gradient_step_count",
        }
        expected_math_fields = {
            "transaction_key",
            "seed",
            "update",
            "schema_version",
            "passed",
            "collection_audit",
            "policy_state_sha256_before",
            "policy_state_sha256_after",
            "ppo_math_evidence",
            *math_payload_fields,
        }
        if (
            len(math_rows) != len(audit_keys)
            or {row.get("transaction_key") for row in math_rows} != audit_keys
        ):
            raise StandardTrainingError("math audit schedule drifted")
        for row in math_rows:
            if not isinstance(row, Mapping) or set(row) != expected_math_fields:
                raise StandardTrainingError("math audit row schema drifted")
            key = str(row["transaction_key"])
            transaction = next(item for item in transactions if item.key == key)
            training = training_by_key[key]
            if (
                row["seed"] != transaction.seed
                or row["update"] != transaction.update
                or row["schema_version"] != "stage6_math_audit/v1"
                or row["passed"] is not True
                or row["collection_audit"] != training["collection_audit"]
                or row["policy_state_sha256_before"]
                != training["update_metrics"]["policy_state_sha256_before"]  # type: ignore[index]
                or row["policy_state_sha256_after"]
                != training["update_metrics"]["policy_state_sha256_after"]  # type: ignore[index]
            ):
                raise StandardTrainingError("math audit policy binding drifted")
            validate_math_audit(
                {name: row[name] for name in math_payload_fields}
            )
            validate_standard_collection_audit(
                row["collection_audit"],
                expected_device=config.device,
                expected_policy_sha256=str(row["policy_state_sha256_before"]),
                require_dual_scan=planning_warm_start,
            )
            training_metrics = training["update_metrics"]
            if not isinstance(training_metrics, Mapping):
                raise StandardTrainingError("training math evidence is missing")
            optimizer_steps = training_metrics.get("optimizer_steps")
            if type(optimizer_steps) is not int or optimizer_steps <= 0:
                raise StandardTrainingError("training optimizer evidence drifted")
            collection_snapshot_hashes = row["collection_audit"].get(
                "snapshot_sha256"
            )
            if not isinstance(collection_snapshot_hashes, list):
                raise StandardTrainingError("math snapshot evidence is missing")
            validated_evidence = validate_ppo_math_evidence(
                row["ppo_math_evidence"],
                expected_policy_state_sha256=str(
                    row["policy_state_sha256_before"]
                ),
                expected_snapshot_hashes=collection_snapshot_hashes,
                expected_sample_count=int(
                    row["collection_audit"]["trainable_transition_count"]
                ),
                expected_optimizer_steps=optimizer_steps,
            )
            if (
                training_metrics.get("math_evidence") != validated_evidence
                or row["evidence_sha256"]
                != validated_evidence["evidence_sha256"]
                or row["snapshot_list_sha256"]
                != validated_evidence["snapshot_list_sha256"]
            ):
                raise StandardTrainingError("math evidence lineage drifted")
    except (
        KeyError,
        TypeError,
        ValueError,
        PPOTrainingError,
        CheckpointError,
        CheckpointRetentionError,
        StandardTrainingError,
        StopIteration,
    ) as exc:
        raise Stage6WorkflowError(
            "Stage 6 acceptance semantic and "
            "training/validation/checkpoint/on-policy/math audit replay failed"
        ) from exc
    for name in ("fairness_audit.json", "leakage_audit.json", "failure_audit.json"):
        audit = _strict_canonical_json(
            read_evidence(name, f"Stage 6 {name}"), f"Stage 6 {name}"
        )
        if audit.get("passed") is not True:
            raise Stage6WorkflowError(f"Stage 6 failed machine audit: {name}")
    if planning_child_source_repair is not None:
        _require_planning_child_recovery_capability(
            planning_child_source_repair,
            "machine verifier final binding",
        )
    authority.require_current("Stage 6 machine verifier final authority")
    if handle is not None:
        handle.require_current()
    verification_result = {
        "passed": True,
        "state": "awaiting_independent_review",
        "final_evaluation_count": len(metrics),
        "final_episode_count": sum(
            int(row["result"]["episode_count"])  # type: ignore[index]
            for row in metrics
        ),
        "checkpoint_receipt_count": len(receipts),
        "global_best_policy_state_sha256": global_best["policy_state_sha256"],
    }
    if planning_child_source_repair is not None:
        return _PlanningChildMachineVerificationResult(
            verification_result,
            planning_child_source_repair,
        )
    return verification_result


def verify_stage6_machine_acceptance(
    *,
    stage_root: str | Path,
    repo_root: str | Path,
    gate_path: str | Path | None = None,
    planning_child_recovery_capability: object | None = None,
) -> dict[str, object]:
    """Replay only the receipt-bound terminal closure.

    The full semantic verifier runs before ``preterminal_acceptance.json`` is
    committed.  Terminal verification therefore treats the receipt's passed
    semantic result and exact terminal artifact bytes as the sole semantic
    authority; it never loads a model, optimizer, evaluator, or checkpoint.
    """

    from lunar_exploration_ppo.workflows.stage6_terminal_recovery import (
        COMPLETE_STATES,
        TERMINAL_ARTIFACT_NAMES,
        TerminalRecoveryError,
        _capture_terminal_evidence_handle,
        _require_planning_child_recovery_binding,
        detect_stage6_terminal_recovery,
    )

    stage = _require_plain_stage6_root(
        stage_root,
        label="Stage 6 terminal machine verification root",
    )
    repo = Path(repo_root).expanduser().resolve()
    if not (stage / "manifest.json").is_file():
        raise Stage6WorkflowError("Stage 6 terminal manifest is missing")

    try:
        with _capture_terminal_evidence_handle(stage) as evidence:
            evidence.read_bytes(
                "manifest.json",
                label="Stage 6 terminal manifest binding",
            )
            manifest_sources: list[dict[str, object]] = []

            def manifest_verifier(candidate: Path) -> dict[str, object]:
                if candidate != stage:
                    raise Stage6WorkflowError(
                        "Stage 6 terminal manifest verifier root drifted"
                    )
                current_source = stage6_source_identity(repo)
                identity = _verify_stage6_terminal_manifest_evidence(
                    candidate,
                    source_identity=current_source,
                    repo_root=repo,
                    evidence_handle=evidence,
                    planning_child_recovery_capability=(
                        planning_child_recovery_capability
                    ),
                )
                manifest_sources.append(current_source)
                return identity
            child_name = (
                STAGE6_PLANNING_CHILD_SOURCE_REPAIR_MANIFEST_ARTIFACT
            )
            continuation_name = (
                STAGE6_PLANNING_CHILD_SOURCE_REPAIR_CONTINUATION_MANIFEST_ARTIFACT
            )
            child_present_at_capture = child_name in set(
                evidence.file_paths
            )
            continuation_present_at_capture = continuation_name in set(
                evidence.file_paths
            )
            if (
                continuation_present_at_capture
                and not child_present_at_capture
            ):
                raise Stage6WorkflowError(
                    "Stage 6 terminal planning child continuation has no "
                    "parent amendment"
                )
            if not child_present_at_capture:
                detection = detect_stage6_terminal_recovery(
                    stage_root=stage,
                    manifest_verifier=manifest_verifier,
                    evidence_handle=evidence,
                )
                if (
                    detection.get("status")
                    != "valid_terminal_recovery"
                    or tuple(detection.get("phase_states", ()))
                    != COMPLETE_STATES
                ):
                    raise Stage6WorkflowError(
                        "Stage 6 receipt/terminal/resource/phase/manifest "
                        "replay failed: "
                        f"{detection.get('reason', '')}"
                    )
                evidence.require_current(
                    "Stage 6 terminal evidence after recovery replay"
                )

            receipt = _strict_canonical_json(
                evidence.read_bytes(
                    "preterminal_acceptance.json",
                    label="Stage 6 preterminal acceptance receipt",
                ),
                "Stage 6 preterminal acceptance receipt",
            )
            lineage_payload = evidence.read_bytes(
                "lineage_audit.json",
                label="Stage 6 lineage audit",
            )
            lineage = _strict_canonical_json(
                lineage_payload,
                "Stage 6 lineage audit",
            )
            immutable = receipt.get("immutable_bindings")
            evidence_paths = set(evidence.file_paths)
            authority = verify_frozen_stage5_authority(
                gate_path=(
                    CANONICAL_STAGE5_GATE
                    if gate_path is None
                    else gate_path
                ),
                repo_root=repo,
            )
            planning_child_source_repair = None
            child_present = child_name in evidence_paths
            classic_present = set(
                STAGE6_OPTIONAL_MANIFEST_BOUND_ARTIFACTS
            ).intersection(evidence_paths)
            if (
                planning_child_recovery_capability is not None
                and not child_present
            ):
                raise Stage6WorkflowError(
                    "Stage 6 terminal recovery capability has no child graph"
                )
            if child_present and classic_present:
                raise Stage6WorkflowError(
                    "Stage 6 terminal planning child and classic "
                    "source-repair evidence are mutually exclusive"
                )
            historical_immutable_bindings = lineage.get(
                "immutable_bindings"
            )
            current_immutable_bindings = historical_immutable_bindings
            if child_present:
                from lunar_exploration_ppo.workflows.stage6_planning_child_recovery import (
                    PlanningChildRecoveryCapability,
                )

                if (
                    lineage.get("schema_version")
                    != "stage6_lineage_audit/v3"
                    or not isinstance(
                        lineage.get("planning_warm_start"),
                        Mapping,
                    )
                    or not continuation_present_at_capture
                    or type(planning_child_recovery_capability)
                    is not PlanningChildRecoveryCapability
                ):
                    raise Stage6WorkflowError(
                        "Stage 6 terminal planning child requires one "
                        "issued recovery capability"
                    )
                planning_child_source_repair = (
                    planning_child_recovery_capability
                )
                _require_planning_child_recovery_capability(
                    planning_child_source_repair,
                    "terminal machine verifier startup",
                )
                if (
                    planning_child_source_repair.stage_root != stage
                    or planning_child_source_repair.formal_run_id
                    != stage.parent.name
                ):
                    raise Stage6WorkflowError(
                        "Stage 6 terminal planning child capability root "
                        "drifted"
                    )
                config_bytes = evidence.read_bytes(
                    "config.json",
                    label="Stage 6 terminal planning child config",
                )
                current_execution_identity = stage6_execution_identity(
                    repo_root=repo,
                    config_path=(
                        repo
                        / "configs/ppo_highres_frontier_stage6_v1.json"
                    ),
                    effective_config_bytes=config_bytes,
                )
                child_payload = evidence.read_bytes(
                    child_name,
                    label="Stage 6 terminal planning child parent",
                )
                continuation_payload = evidence.read_bytes(
                    continuation_name,
                    label="Stage 6 terminal planning child continuation",
                )
                if (
                    hashlib.sha256(child_payload).hexdigest()
                    != planning_child_source_repair.parent_artifact_sha256
                    or hashlib.sha256(continuation_payload).hexdigest()
                    != planning_child_source_repair
                    .continuation_artifact_sha256
                ):
                    raise Stage6WorkflowError(
                        "Stage 6 terminal planning child graph identity "
                        "drifted"
                    )
                review_record = (
                    planning_child_source_repair.acceptance_binding.get(
                        "current_verified_review_authorization"
                    )
                )
                if not isinstance(review_record, Mapping):
                    raise Stage6WorkflowError(
                        "Stage 6 terminal planning child review is missing"
                    )
                verified_review = (
                    validate_stage6_verified_review_authorization(
                        review_record,
                        execution_identity=current_execution_identity,
                        formal_run_id=stage.parent.name,
                    )
                )
                current_immutable_bindings = (
                    _stage6_current_immutable_bindings(
                        execution_identity=current_execution_identity,
                        verified_review_authorization=verified_review,
                        stage5_authority=authority,
                    )
                )
                validated_child = (
                    _validate_planning_child_machine_context(
                        capability=planning_child_source_repair,
                        lineage_audit=lineage,
                        current_execution_identity=(
                            current_execution_identity
                        ),
                        current_verified_review_authorization=(
                            verified_review
                        ),
                        current_immutable_bindings=(
                            current_immutable_bindings
                        ),
                    )
                )
                historical_immutable_bindings = validated_child[
                    "historical_immutable_bindings"
                ]
                current_immutable_bindings = validated_child[
                    "current_immutable_bindings"
                ]
                detection = detect_stage6_terminal_recovery(
                    stage_root=stage,
                    manifest_verifier=manifest_verifier,
                    evidence_handle=evidence,
                    planning_child_recovery_capability=(
                        planning_child_source_repair
                    ),
                )
                if (
                    detection.get("status")
                    != "valid_terminal_recovery"
                    or tuple(detection.get("phase_states", ()))
                    != COMPLETE_STATES
                ):
                    raise Stage6WorkflowError(
                        "Stage 6 receipt/terminal/resource/phase/manifest "
                        "replay failed: "
                        f"{detection.get('reason', '')}"
                    )
                evidence.require_current(
                    "Stage 6 terminal evidence after planning child "
                    "recovery replay"
                )
            else:
                verified_review = _stage6_lineage_review_authorization(
                    stage,
                    payload=lineage_payload,
                    repo_root=repo,
                    source_repair_amendment_payload=(
                        evidence.read_bytes(
                            "source-repair-amendment.json",
                            label=(
                                "Stage 6 terminal receipt source-repair "
                                "amendment"
                            ),
                        )
                        if "source-repair-amendment.json" in evidence_paths
                        else None
                    ),
                    source_repair_continuation_payload=(
                        evidence.read_bytes(
                            "source-repair-continuation.json",
                            label=(
                                "Stage 6 terminal receipt source-repair "
                                "continuation"
                            ),
                        )
                        if "source-repair-continuation.json" in evidence_paths
                        else None
                    ),
                    source_repair_supplement_payload=(
                        evidence.read_bytes(
                            "source-repair-supplement.json",
                            label=(
                                "Stage 6 terminal receipt source-repair "
                                "supplement"
                            ),
                        )
                        if "source-repair-supplement.json" in evidence_paths
                        else None
                    ),
                    source_repair_closure_payload=(
                        evidence.read_bytes(
                            "source-repair-closure.json",
                            label=(
                                "Stage 6 terminal receipt source-repair "
                                "closure"
                            ),
                        )
                        if "source-repair-closure.json" in evidence_paths
                        else None
                    ),
                    source_repair_frontier_recovery_payload=(
                        evidence.read_bytes(
                            "source-repair-frontier-recovery.json",
                            label=(
                                "Stage 6 terminal receipt source-repair "
                                "frontier recovery"
                            ),
                        )
                        if (
                            "source-repair-frontier-recovery.json"
                            in evidence_paths
                        )
                        else None
                    ),
                    source_repair_sensor_acceleration_payload=(
                        evidence.read_bytes(
                            "source-repair-sensor-acceleration.json",
                            label=(
                                "Stage 6 terminal receipt source-repair "
                                "sensor acceleration"
                            ),
                        )
                        if (
                            "source-repair-sensor-acceleration.json"
                            in evidence_paths
                        )
                        else None
                    ),
                )
            if (
                not isinstance(immutable, Mapping)
                or not isinstance(
                    historical_immutable_bindings,
                    Mapping,
                )
                or not isinstance(current_immutable_bindings, Mapping)
                or lineage.get("immutable_bindings")
                != historical_immutable_bindings
                or immutable != current_immutable_bindings
                or immutable.get("formal_run_id") != stage.parent.name
            ):
                raise Stage6WorkflowError(
                    "Stage 6 receipt authorization/lineage binding drifted"
                )
            if immutable.get("stage5_gate_sha256") != authority.identity.get(
                "gate_sha256"
            ):
                raise Stage6WorkflowError(
                    "Stage 6 receipt frozen authority binding drifted"
                )

            artifact_rows = receipt.get("terminal_artifacts")
            if not isinstance(artifact_rows, list):
                raise Stage6WorkflowError(
                    "Stage 6 receipt terminal artifacts drifted"
                )
            artifacts = {
                row.get("path"): row
                for row in artifact_rows
                if isinstance(row, Mapping) and isinstance(row.get("path"), str)
            }
            if set(artifacts) != set(TERMINAL_ARTIFACT_NAMES):
                raise Stage6WorkflowError(
                    "Stage 6 receipt terminal membership drifted"
                )
            summary = _strict_canonical_json(
                evidence.read_bytes(
                    "summary.json",
                    label="Stage 6 receipt-fixed summary",
                ),
                "Stage 6 receipt-fixed summary",
            )
            routing = _strict_canonical_json(
                evidence.read_bytes(
                    "routing.json",
                    label="Stage 6 receipt-fixed routing",
                ),
                "Stage 6 receipt-fixed routing",
            )
            semantic = receipt.get("semantic_verification")
            global_checkpoint = receipt.get("global_checkpoint_identity")
            if planning_child_source_repair is not None:
                if not isinstance(semantic, Mapping):
                    raise Stage6WorkflowError(
                        "Stage 6 planning child terminal semantic binding "
                        "is missing"
                    )
                _require_planning_child_recovery_binding(
                    semantic,
                    planning_child_source_repair,
                )
                recovery_binding = json.loads(
                    ArtifactStore.canonical_json_bytes(
                        planning_child_source_repair.evidence_binding()
                    ).decode("utf-8")
                )
                if (
                    summary.get("planning_child_recovery")
                    != recovery_binding
                    or routing.get("planning_child_recovery")
                    != recovery_binding
                ):
                    raise Stage6WorkflowError(
                        "Stage 6 planning child terminal acceptance binding "
                        "drifted"
                    )
            if (
                not isinstance(semantic, Mapping)
                or not isinstance(global_checkpoint, Mapping)
                or semantic.get("passed") is not True
                or semantic.get("state") != "awaiting_independent_review"
                or summary.get("state") != semantic.get("state")
                or summary.get("machine_passed") is not True
                or summary.get("final_evaluation_count")
                != semantic.get("final_evaluation_count")
                or summary.get("final_episode_count")
                != semantic.get("final_episode_count")
                or summary.get("checkpoint_receipt_count")
                != semantic.get("checkpoint_receipt_count")
                or summary.get("global_best") != global_checkpoint.get("record")
                or routing.get("state") != semantic.get("state")
                or routing.get("route") != "awaiting_independent_review"
                or routing.get("machine_passed") is not True
            ):
                raise Stage6WorkflowError(
                    "Stage 6 receipt-fixed summary/routing semantic binding drifted"
                )
            if planning_child_source_repair is not None:
                _require_planning_child_recovery_capability(
                    planning_child_source_repair,
                    "terminal machine verifier final binding",
                )
            authority.require_current(
                "Stage 6 terminal receipt verifier authority"
            )
            if not manifest_sources:
                raise Stage6WorkflowError(
                    "Stage 6 terminal manifest was not continuously verified"
                )
            if any(
                stage6_source_identity(repo) != source
                for source in manifest_sources
            ):
                raise Stage6WorkflowError(
                    "Stage 6 terminal verifier source evidence changed"
                )
            evidence.require_current(
                "Stage 6 terminal verifier final evidence"
            )
            result = dict(semantic)
    except (TerminalRecoveryError, OSError) as exc:
        raise Stage6WorkflowError(
            "Stage 6 terminal evidence could not remain continuously bound"
        ) from exc
    return result


def verify_stage6_preterminal_acceptance(
    *,
    stage_root: str | Path,
    repo_root: str | Path,
    gate_path: str | Path | None = None,
    planning_child_recovery_capability: object | None = None,
) -> dict[str, object]:
    return _verify_stage6_machine_acceptance(
        stage_root=stage_root,
        repo_root=repo_root,
        gate_path=gate_path,
        require_terminal=False,
        planning_child_recovery_capability=(
            planning_child_recovery_capability
        ),
    )


def verify_stage6_pending_acceptance(
    *,
    stage_root: str | Path,
    repo_root: str | Path,
    summary: Mapping[str, object],
    routing: Mapping[str, object],
    reports: Mapping[str, bytes],
    gate_path: str | Path | None = None,
    planning_child_recovery_capability: object | None = None,
) -> dict[str, object]:
    """Replay all preterminal evidence without publishing success artifacts."""

    return _verify_stage6_machine_acceptance(
        stage_root=stage_root,
        repo_root=repo_root,
        gate_path=gate_path,
        require_terminal=False,
        summary_override=summary,
        routing_override=routing,
        reports_override=reports,
        planning_child_recovery_capability=(
            planning_child_recovery_capability
        ),
    )


@dataclass(frozen=True, slots=True)
class Stage6WorkflowResult:
    run_id: str
    stage_root: Path
    summary: dict[str, object]
    routing: dict[str, object]


def verify_stage6_machine_preflight_for_resume(
    *,
    config_path: str | Path,
    run_id: str,
    stage5_gate_path: str | Path,
) -> dict[str, object]:
    """Validate an existing canonical preflight without executing it again."""

    run_id = validate_stage6_formal_run_id(run_id)
    from lunar_exploration_ppo.configs.stage6 import load_stage6_config

    config = load_stage6_config(config_path)
    _require_canonical_stage6_output_root(config.output_root)
    return _verify_stage6_machine_preflight_for_resume_at_root(
        config_path=config_path,
        run_id=run_id,
        stage5_gate_path=stage5_gate_path,
        base_output_root=CANONICAL_STAGE6_OUTPUT_ROOT,
    )


def _verify_stage6_machine_preflight_for_resume_for_test(
    *,
    config_path: str | Path,
    run_id: str,
    stage5_gate_path: str | Path,
    base_output_root: str | Path,
) -> dict[str, object]:
    """Private D-drive resume-verification seam for tests."""

    return _verify_stage6_machine_preflight_for_resume_at_root(
        config_path=config_path,
        run_id=run_id,
        stage5_gate_path=stage5_gate_path,
        base_output_root=base_output_root,
    )


def _verify_stage6_machine_preflight_for_resume_at_root(
    *,
    config_path: str | Path,
    run_id: str,
    stage5_gate_path: str | Path,
    base_output_root: str | Path,
    effective_config_bytes: bytes | None = None,
) -> dict[str, object]:
    """Shared verifier after the caller has selected an authorized root."""

    from lunar_exploration_ppo.configs.stage6 import load_stage6_config

    config_file = Path(config_path).expanduser().resolve()
    config = load_stage6_config(config_file)
    base = Path(base_output_root)
    base, run_root = _resolve_stage6_run_root(base, run_id)
    authority = verify_frozen_stage5_authority(
        gate_path=stage5_gate_path,
        repo_root=Path(__file__).resolve().parents[3],
    )
    try:
        require_plain_path(
            run_root,
            base=base,
            leaf_kind="directory",
            label="Stage 6 run root",
        )
        require_plain_path(
            run_root / "s6",
            base=base,
            leaf_kind="directory",
            label="Stage 6 stage root",
        )
        preflight_root = require_plain_path(
            run_root / "s6/preflight",
            base=base,
            leaf_kind="directory",
            label="Stage 6 preflight root",
        )
        audit_path = require_plain_path(
            preflight_root / "audit.json",
            base=base,
            leaf_kind="file",
            label="Stage 6 preflight audit",
        )
    except PathSecurityError as exc:
        raise Stage6WorkflowError(
            "Stage 6 resume path contains a link or reparse point"
        ) from exc
    if (
        run_root.drive.upper() != "D:"
        or {path.name for path in preflight_root.iterdir()} != {"audit.json"}
    ):
        raise Stage6WorkflowError("Stage 6 resume preflight artifact drifted")
    audit_identity = path_identity(audit_path)
    audit_bytes = audit_path.read_bytes()
    if path_identity(audit_path) != audit_identity:
        raise Stage6WorkflowError("Stage 6 preflight audit identity changed while read")
    audit = validate_stage6_machine_preflight_audit(
        _strict_canonical_json(audit_bytes, "Stage 6 preflight audit")
    )
    base_config_bytes = config_file.read_bytes()
    config_identity_bytes = base_config_bytes
    if effective_config_bytes is not None:
        from lunar_exploration_ppo.workflows.stage6_planning_warm_start import (
            Stage6PlanningWarmStartError,
            parse_planning_effective_config_bytes,
        )

        try:
            effective = parse_planning_effective_config_bytes(
                effective_config_bytes
            )
        except Stage6PlanningWarmStartError as exc:
            raise Stage6WorkflowError(
                "Stage 6 resume effective config is invalid"
            ) from exc
        if (
            effective.base_config_sha256
            != hashlib.sha256(base_config_bytes).hexdigest()
            or effective.base_config != config
        ):
            raise Stage6WorkflowError(
                "Stage 6 resume effective/base config drifted"
            )
        config_identity_bytes = effective.effective_config_bytes
    if (
        audit["config_sha256"]
        != hashlib.sha256(config_identity_bytes).hexdigest()
        or audit["stage5_gate_sha256"] != authority.identity["gate_sha256"]
    ):
        raise Stage6WorkflowError("Stage 6 resume preflight binding drifted")
    current_environment = stage6_environment_identity()
    current_environment_sha256 = stage6_environment_sha256(current_environment)
    if (
        audit["environment_identity"] != current_environment
        or audit["environment_sha256"] != current_environment_sha256
    ):
        raise Stage6WorkflowError(
            "Stage 6 resume environment identity drifted"
        )
    authority.require_current("Stage 6 resume preflight authority")
    return audit


def _planning_warm_start_expected_bindings_for_workflow(
    *,
    repo_root: str | Path,
    review_authorization_handle: Stage6ReviewAuthorizationHandle,
    planning_child_recovery_capability: object | None,
) -> dict[str, str]:
    from lunar_exploration_ppo.workflows.stage6_planning_warm_start import (
        Stage6PlanningWarmStartError,
        planning_warm_start_expected_bindings,
        planning_warm_start_expected_bindings_from_record,
    )

    try:
        if planning_child_recovery_capability is None:
            return planning_warm_start_expected_bindings(
                repo_root=repo_root,
                review_authorization_handle=review_authorization_handle,
            )
        acceptance_binding = getattr(
            planning_child_recovery_capability,
            "acceptance_binding",
            None,
        )
        origin_review = (
            acceptance_binding.get(
                "origin_verified_review_authorization"
            )
            if isinstance(acceptance_binding, Mapping)
            else None
        )
        if not isinstance(origin_review, Mapping):
            raise Stage6WorkflowError(
                "Stage 6 planning child origin review authorization is invalid"
            )
        return planning_warm_start_expected_bindings_from_record(
            repo_root=repo_root,
            review_authorization_record=origin_review,
        )
    except Stage6PlanningWarmStartError as exc:
        raise Stage6WorkflowError(
            "Stage 6 planning warm-start evidence binding failed"
        ) from exc


@contextmanager
def _stage6_planning_child_launch_run_lease(
    *,
    base_output_root: str | Path,
    run_id: str,
    enabled: bool,
):
    if not enabled:
        yield None
        return
    _, run_root = _resolve_stage6_run_root(base_output_root, run_id)
    try:
        with RunLease(run_root / ".stage6.lease") as run_lease:
            yield run_lease
    except RunLeaseError as exc:
        raise Stage6WorkflowError(
            "Stage 6 run is already active or its single-writer lease is unavailable"
        ) from exc


def run_stage6_workflow(
    *,
    config_path: str | Path,
    run_id: str,
    stage5_gate_path: str | Path,
    review_authorization_path: str | Path,
    planning_warm_start_path: str | Path | None = None,
    planning_child_source_repair_path: str | Path | None = None,
    planning_child_source_repair_continuation_path: (
        str | Path | None
    ) = None,
) -> Stage6WorkflowResult:
    """执行正式 Stage 6；不提供 force、skip、fixture 或 fake 成功入口。"""

    run_id = validate_stage6_formal_run_id(run_id)
    if (
        planning_child_source_repair_path is not None
        and planning_warm_start_path is None
    ):
        raise Stage6WorkflowError(
            "Stage 6 planning child source-repair requires planning warm-start"
        )
    if (
        planning_child_source_repair_continuation_path is not None
        and planning_child_source_repair_path is None
    ):
        raise Stage6WorkflowError(
            "Stage 6 planning child continuation requires its parent"
        )
    repo = Path(__file__).resolve().parents[3]
    effective_config_bytes = None
    if planning_warm_start_path is not None:
        from lunar_exploration_ppo.workflows.stage6_planning_warm_start import (
            planning_effective_config_bytes,
        )

        effective_config_bytes = planning_effective_config_bytes(
            Path(config_path).read_bytes()
        )
    review_authorization_handle = _verify_review_authorization_before_run(
        authorization_path=review_authorization_path,
        repo_root=repo,
        config_path=config_path,
        expected_run_id=run_id,
        effective_config_bytes=effective_config_bytes,
    )
    _require_review_authorization_current(
        review_authorization_handle,
        "Stage 6 review authorization before input pin acquisition",
    )
    stage5_authority = verify_frozen_stage5_authority(
        gate_path=stage5_gate_path,
        repo_root=repo,
    )
    input_pin = _acquire_stage6_workflow_input_pin(
        repo_root=repo,
        config_path=config_path,
        stage5_authority=stage5_authority,
        review_authorization_handle=review_authorization_handle,
        planning_warm_start_path=planning_warm_start_path,
        planning_child_source_repair_path=(
            planning_child_source_repair_path
        ),
        planning_child_source_repair_continuation_path=(
            planning_child_source_repair_continuation_path
        ),
    )
    from lunar_exploration_ppo.configs.stage6 import load_stage6_config

    try:
        with input_pin:
            execution_identity = _validate_stage6_pinned_execution_identity(
                input_pin=input_pin,
                repo_root=repo,
                config_path=config_path,
                run_id=run_id,
                stage5_authority=stage5_authority,
                review_authorization_handle=review_authorization_handle,
                effective_config_bytes=effective_config_bytes,
            )
            config = load_stage6_config(config_path)
            base = _require_canonical_stage6_output_root(config.output_root)
            with _stage6_planning_child_launch_run_lease(
                base_output_root=base,
                run_id=run_id,
                enabled=planning_child_source_repair_path is not None,
            ) as planning_child_run_lease:
                planning_warm_start_context = None
                planning_warm_start_sha256 = None
                planning_child_recovery_capability = None
                verified_parent_u74 = None
                if planning_child_source_repair_path is not None:
                    planning_child_recovery_capability = (
                        _load_planning_child_recovery_capability_for_workflow(
                            path=planning_child_source_repair_path,
                            continuation_path=(
                                planning_child_source_repair_continuation_path
                            ),
                            run_id=run_id,
                            execution_identity=execution_identity,
                            review_authorization_handle=(
                                review_authorization_handle
                            ),
                            stage5_authority=stage5_authority,
                        )
                    )
                if planning_warm_start_path is not None:
                    from lunar_exploration_ppo.workflows.stage6_planning_warm_start import (
                        PARENT_STAGE_ROOT,
                        load_planning_warm_start_artifact,
                        verify_parent_u74_bundle,
                    )

                    expected_bindings = (
                        _planning_warm_start_expected_bindings_for_workflow(
                            repo_root=repo,
                            review_authorization_handle=(
                                review_authorization_handle
                            ),
                            planning_child_recovery_capability=(
                                planning_child_recovery_capability
                            ),
                        )
                    )
                    (
                        planning_warm_start_context,
                        planning_warm_start_sha256,
                    ) = load_planning_warm_start_artifact(
                        planning_warm_start_path,
                        expected_child_run_id=run_id,
                        expected_child_config_bytes=effective_config_bytes,
                        expected_bindings=expected_bindings,
                    )
                    verified_parent_u74 = verify_parent_u74_bundle(
                        PARENT_STAGE_ROOT,
                        input_pin=input_pin,
                    )
                result = _run_stage6_workflow_at_root(
                    config=config,
                    config_path=config_path,
                    run_id=run_id,
                    stage5_gate_path=stage5_gate_path,
                    base_output_root=base,
                    review_authorization_handle=review_authorization_handle,
                    stage5_authority=stage5_authority,
                    input_pin=input_pin,
                    run_lease=planning_child_run_lease,
                    planning_warm_start_context=planning_warm_start_context,
                    planning_warm_start_sha256=planning_warm_start_sha256,
                    planning_child_recovery_capability=(
                        planning_child_recovery_capability
                    ),
                    verified_parent_u74=verified_parent_u74,
                )
            _require_input_pin_current(
                input_pin,
                "Stage 6 input pin before final workflow return",
                rehash=True,
            )
            return result
    except Stage6WorkflowError:
        raise
    except (OSError, PathSecurityError) as exc:
        raise Stage6WorkflowError("Stage 6 input pin lifecycle failed") from exc


def _run_stage6_workflow_for_test(
    *,
    config_path: str | Path,
    run_id: str,
    stage5_gate_path: str | Path,
    base_output_root: str | Path,
    review_authorization_path: str | Path,
    planning_warm_start_path: str | Path | None = None,
    planning_child_source_repair_path: str | Path | None = None,
    planning_child_source_repair_continuation_path: (
        str | Path | None
    ) = None,
) -> Stage6WorkflowResult:
    """Private output-root seam；只允许 monkeypatch 正式 verifier。"""

    repo = Path(__file__).resolve().parents[3]
    if (
        planning_child_source_repair_path is not None
        and planning_warm_start_path is None
    ):
        raise Stage6WorkflowError(
            "Stage 6 planning child source-repair requires planning warm-start"
        )
    if (
        planning_child_source_repair_continuation_path is not None
        and planning_child_source_repair_path is None
    ):
        raise Stage6WorkflowError(
            "Stage 6 planning child continuation requires its parent"
        )
    effective_config_bytes = None
    if planning_warm_start_path is not None:
        from lunar_exploration_ppo.workflows.stage6_planning_warm_start import (
            planning_effective_config_bytes,
        )

        effective_config_bytes = planning_effective_config_bytes(
            Path(config_path).read_bytes()
        )
    review_authorization_handle = _verify_review_authorization_before_run(
        authorization_path=review_authorization_path,
        repo_root=repo,
        config_path=config_path,
        expected_run_id=run_id,
        effective_config_bytes=effective_config_bytes,
    )
    _require_review_authorization_current(
        review_authorization_handle,
        "Stage 6 review authorization before input pin acquisition",
    )
    stage5_authority = verify_frozen_stage5_authority(
        gate_path=stage5_gate_path,
        repo_root=repo,
    )
    input_pin = _acquire_stage6_workflow_input_pin(
        repo_root=repo,
        config_path=config_path,
        stage5_authority=stage5_authority,
        review_authorization_handle=review_authorization_handle,
        planning_warm_start_path=planning_warm_start_path,
        planning_child_source_repair_path=(
            planning_child_source_repair_path
        ),
        planning_child_source_repair_continuation_path=(
            planning_child_source_repair_continuation_path
        ),
    )

    from lunar_exploration_ppo.configs.stage6 import load_stage6_config

    try:
        with input_pin:
            execution_identity = _validate_stage6_pinned_execution_identity(
                input_pin=input_pin,
                repo_root=repo,
                config_path=config_path,
                run_id=run_id,
                stage5_authority=stage5_authority,
                review_authorization_handle=review_authorization_handle,
                effective_config_bytes=effective_config_bytes,
            )
            config = load_stage6_config(config_path)
            with _stage6_planning_child_launch_run_lease(
                base_output_root=base_output_root,
                run_id=run_id,
                enabled=planning_child_source_repair_path is not None,
            ) as planning_child_run_lease:
                planning_warm_start_context = None
                planning_warm_start_sha256 = None
                planning_child_recovery_capability = None
                verified_parent_u74 = None
                if planning_child_source_repair_path is not None:
                    planning_child_recovery_capability = (
                        _load_planning_child_recovery_capability_for_workflow(
                            path=planning_child_source_repair_path,
                            continuation_path=(
                                planning_child_source_repair_continuation_path
                            ),
                            run_id=run_id,
                            execution_identity=execution_identity,
                            review_authorization_handle=(
                                review_authorization_handle
                            ),
                            stage5_authority=stage5_authority,
                        )
                    )
                if planning_warm_start_path is not None:
                    from lunar_exploration_ppo.workflows.stage6_planning_warm_start import (
                        PARENT_STAGE_ROOT,
                        load_planning_warm_start_artifact,
                        verify_parent_u74_bundle,
                    )

                    expected_bindings = (
                        _planning_warm_start_expected_bindings_for_workflow(
                            repo_root=repo,
                            review_authorization_handle=(
                                review_authorization_handle
                            ),
                            planning_child_recovery_capability=(
                                planning_child_recovery_capability
                            ),
                        )
                    )
                    (
                        planning_warm_start_context,
                        planning_warm_start_sha256,
                    ) = load_planning_warm_start_artifact(
                        planning_warm_start_path,
                        expected_child_run_id=run_id,
                        expected_child_config_bytes=effective_config_bytes,
                        expected_bindings=expected_bindings,
                    )
                    verified_parent_u74 = verify_parent_u74_bundle(
                        PARENT_STAGE_ROOT,
                        input_pin=input_pin,
                    )
                result = _run_stage6_workflow_at_root(
                    config=config,
                    config_path=config_path,
                    run_id=run_id,
                    stage5_gate_path=stage5_gate_path,
                    base_output_root=base_output_root,
                    review_authorization_handle=review_authorization_handle,
                    stage5_authority=stage5_authority,
                    input_pin=input_pin,
                    run_lease=planning_child_run_lease,
                    planning_warm_start_context=planning_warm_start_context,
                    planning_warm_start_sha256=planning_warm_start_sha256,
                    planning_child_recovery_capability=(
                        planning_child_recovery_capability
                    ),
                    verified_parent_u74=verified_parent_u74,
                )
            _require_input_pin_current(
                input_pin,
                "Stage 6 input pin before final workflow return",
                rehash=True,
            )
            return result
    except Stage6WorkflowError:
        raise
    except (OSError, PathSecurityError) as exc:
        raise Stage6WorkflowError("Stage 6 input pin lifecycle failed") from exc


def _acquire_stage6_workflow_input_pin(
    *,
    repo_root: str | Path,
    config_path: str | Path,
    stage5_authority: FrozenStage5AuthorityHandle,
    review_authorization_handle: Stage6ReviewAuthorizationHandle,
    planning_warm_start_path: str | Path | None = None,
    planning_child_source_repair_path: str | Path | None = None,
    planning_child_source_repair_continuation_path: (
        str | Path | None
    ) = None,
) -> Stage6InputPin:
    from lunar_exploration_ppo.utils.stage6_input_pinning import (
        Stage6InputPinError,
        acquire_stage6_input_pin,
    )

    requests = _stage6_input_pin_requests(
        repo_root=repo_root,
        config_path=config_path,
        stage5_authority=stage5_authority,
        review_authorization_handle=review_authorization_handle,
        planning_warm_start_path=planning_warm_start_path,
        planning_child_source_repair_path=(
            planning_child_source_repair_path
        ),
        planning_child_source_repair_continuation_path=(
            planning_child_source_repair_continuation_path
        ),
    )
    try:
        return acquire_stage6_input_pin(requests)
    except Stage6InputPinError as exc:
        raise Stage6WorkflowError("Stage 6 formal input pin acquisition failed") from exc


def _require_input_pin_current(
    input_pin: Stage6InputPin,
    label: str,
    *,
    rehash: bool = False,
) -> None:
    from lunar_exploration_ppo.utils.stage6_input_pinning import Stage6InputPinError

    try:
        input_pin.require_current(label, rehash=rehash)
    except Stage6InputPinError as exc:
        raise Stage6WorkflowError(f"Stage 6 input pin changed at {label}") from exc


def _validate_stage6_pinned_execution_identity(
    *,
    input_pin: Stage6InputPin,
    repo_root: str | Path,
    config_path: str | Path,
    run_id: str,
    stage5_authority: FrozenStage5AuthorityHandle,
    review_authorization_handle: Stage6ReviewAuthorizationHandle,
    effective_config_bytes: bytes | None = None,
) -> dict[str, object]:
    _require_input_pin_current(
        input_pin,
        "Stage 6 input pin acquisition complete",
    )
    stage5_authority.require_current(
        "Stage 6 Stage 5 authority after input pin acquisition"
    )
    _require_review_authorization_current(
        review_authorization_handle,
        "Stage 6 review authorization after input pin acquisition",
    )
    current = stage6_execution_identity(
        repo_root=repo_root,
        config_path=config_path,
        effective_config_bytes=effective_config_bytes,
    )
    validate_stage6_verified_review_authorization(
        _review_authorization_record(review_authorization_handle),
        execution_identity=current,
        formal_run_id=run_id,
    )
    _require_input_pin_current(
        input_pin,
        "Stage 6 input pin after execution identity recomputation",
    )
    stage5_authority.require_current(
        "Stage 6 Stage 5 authority after execution identity recomputation"
    )
    _require_review_authorization_current(
        review_authorization_handle,
        "Stage 6 review authorization after execution identity recomputation",
    )
    return current


def _verify_review_authorization_before_run(
    *,
    authorization_path: str | Path,
    repo_root: str | Path,
    config_path: str | Path,
    expected_run_id: str,
    effective_config_bytes: bytes | None = None,
) -> Stage6ReviewAuthorizationHandle:
    from lunar_exploration_ppo.workflows.stage6_review_authorization import (
        Stage6ReviewAuthorizationError,
        verify_stage6_review_launch_authorization,
    )

    try:
        return verify_stage6_review_launch_authorization(
            authorization_path=authorization_path,
            repo_root=repo_root,
            config_path=config_path,
            expected_run_id=expected_run_id,
            effective_config_bytes=effective_config_bytes,
        )
    except Stage6ReviewAuthorizationError as exc:
        raise Stage6WorkflowError(
            "Stage 6 review launch authorization failed"
        ) from exc


def _require_review_authorization_current(
    handle: Stage6ReviewAuthorizationHandle,
    label: str,
) -> None:
    from lunar_exploration_ppo.workflows.stage6_review_authorization import (
        Stage6ReviewAuthorizationError,
    )

    try:
        handle.require_current(label)
    except Stage6ReviewAuthorizationError as exc:
        raise Stage6WorkflowError(
            f"Stage 6 review authorization changed at {label}"
        ) from exc


def _require_planning_warm_start_current(
    context: object,
    label: str,
) -> None:
    from lunar_exploration_ppo.workflows.stage6_planning_warm_start import (
        PlanningWarmStartContext,
        Stage6PlanningWarmStartError,
    )

    if not isinstance(context, PlanningWarmStartContext):
        raise Stage6WorkflowError(
            "Stage 6 planning warm-start context is invalid"
        )
    try:
        context.require_current(label)
    except Stage6PlanningWarmStartError as exc:
        raise Stage6WorkflowError(
            f"Stage 6 planning warm-start changed at {label}"
        ) from exc


def _validate_stage6_repair_context_combination(
    *,
    planning_warm_start_context: object | None,
    planning_child_recovery_capability: object | None,
    classic_source_repair_context: object | None,
) -> None:
    if (
        planning_child_recovery_capability is not None
        and planning_warm_start_context is None
    ):
        raise Stage6WorkflowError(
            "Stage 6 planning child source-repair requires planning warm-start"
        )
    if classic_source_repair_context is not None and (
        planning_warm_start_context is not None
        or planning_child_recovery_capability is not None
    ):
        raise Stage6WorkflowError(
            "Stage 6 classic source-repair cannot be combined with planning child"
        )


def _stage6_current_immutable_bindings(
    *,
    execution_identity: Mapping[str, object],
    verified_review_authorization: Mapping[str, object],
    stage5_authority: FrozenStage5AuthorityHandle,
) -> dict[str, object]:
    from lunar_exploration_ppo.ppo.standard_training import (
        _review_authorization_immutable_bindings,
    )

    required_identity_fields = (
        "config_sha256",
        "source_set_sha256",
        "prospective_tree_sha256",
        "data_sha256",
        "environment_identity",
        "environment_sha256",
    )
    try:
        result = {
            key: execution_identity[key]
            for key in required_identity_fields
        }
        result["stage5_gate_sha256"] = stage5_authority.identity[
            "gate_sha256"
        ]
        result.update(
            _review_authorization_immutable_bindings(
                verified_review_authorization
            )
        )
    except (KeyError, TypeError) as exc:
        raise Stage6WorkflowError(
            "Stage 6 current immutable identity binding is incomplete"
        ) from exc
    result.update(
        {
            "coverage_cache_manifest_path": (
                STAGE6_COVERAGE_CACHE_MANIFEST_PATH.as_posix()
            ),
            "coverage_cache_manifest_sha256": (
                STAGE6_COVERAGE_CACHE_MANIFEST_SHA256
            ),
            "coverage_cache_manifest_size_bytes": (
                STAGE6_COVERAGE_CACHE_MANIFEST_SIZE_BYTES
            ),
            "coverage_cache_root": STAGE6_COVERAGE_CACHE_ROOT.as_posix(),
            "coverage_cache_entry_set_sha256": (
                STAGE6_COVERAGE_CACHE_ENTRY_SET_SHA256
            ),
            "coverage_cache_runtime_mode": (
                STAGE6_COVERAGE_CACHE_RUNTIME_MODE
            ),
            "coverage_cache_formal_audit_sha256": (
                STAGE6_COVERAGE_CACHE_FORMAL_AUDIT_SHA256
            ),
            "coverage_cache_formal_audit_size_bytes": (
                STAGE6_COVERAGE_CACHE_FORMAL_AUDIT_SIZE_BYTES
            ),
        }
    )
    return result


def _load_planning_child_recovery_capability_for_workflow(
    *,
    path: str | Path,
    continuation_path: str | Path | None = None,
    run_id: str,
    execution_identity: Mapping[str, object],
    review_authorization_handle: Stage6ReviewAuthorizationHandle,
    stage5_authority: FrozenStage5AuthorityHandle,
    loader: object | None = None,
) -> object:
    from lunar_exploration_ppo.workflows.stage6_planning_child_recovery import (
        Stage6PlanningChildRecoveryError,
        issue_planning_child_recovery_capability,
    )

    del loader
    review = validate_stage6_verified_review_authorization(
        _review_authorization_record(review_authorization_handle),
        execution_identity=execution_identity,
        formal_run_id=run_id,
    )
    immutable = _stage6_current_immutable_bindings(
        execution_identity=execution_identity,
        verified_review_authorization=review,
        stage5_authority=stage5_authority,
    )
    artifact_path = lexical_absolute(path)
    if (
        artifact_path.name != "planning-child-source-repair.json"
        or artifact_path.parent.name != "s6"
        or artifact_path.parent.parent.name != run_id
    ):
        raise Stage6WorkflowError(
            "Stage 6 planning child source-repair run path drifted"
        )
    if continuation_path is None:
        raise Stage6WorkflowError(
            "Stage 6 planning child continuation is required"
        )
    continuation_artifact_path = lexical_absolute(continuation_path)
    if (
        continuation_artifact_path
        != artifact_path.parent
        / "planning-child-source-repair-continuation.json"
    ):
        raise Stage6WorkflowError(
            "Stage 6 planning child continuation run path drifted"
        )
    try:
        return issue_planning_child_recovery_capability(
            stage_root=artifact_path.parent,
            parent_artifact_path=artifact_path,
            continuation_artifact_path=continuation_artifact_path,
            current_execution_identity=execution_identity,
            current_verified_review_authorization=review,
            current_immutable_bindings=immutable,
        )
    except Stage6PlanningChildRecoveryError as exc:
        raise Stage6WorkflowError(
            "Stage 6 planning child recovery capability validation failed"
        ) from exc


def _require_planning_child_recovery_capability(
    capability: object,
    label: str,
) -> None:
    from lunar_exploration_ppo.workflows.stage6_planning_child_recovery import (
        PlanningChildRecoveryCapability,
    )

    if (
        not isinstance(capability, PlanningChildRecoveryCapability)
        or not _is_sha256(capability.parent_artifact_sha256)
        or not _is_sha256(capability.continuation_artifact_sha256)
        or not _is_sha256(capability.input_snapshot_sha256)
        or not _is_sha256(capability.capability_sha256)
        or capability.acceptance_binding.get("input_snapshot_sha256")
        != capability.input_snapshot_sha256
    ):
        raise Stage6WorkflowError(
            f"Stage 6 planning child recovery capability drifted at {label}"
        )


def _review_authorization_record(
    handle: Stage6ReviewAuthorizationHandle,
) -> dict[str, object]:
    from lunar_exploration_ppo.workflows.stage6_review_authorization import (
        Stage6ReviewAuthorizationError,
    )

    try:
        record = handle.canonical_record()
        payload = ArtifactStore.canonical_json_bytes(record)
        plain = json.loads(payload.decode("utf-8"))
    except (Stage6ReviewAuthorizationError, TypeError, ValueError) as exc:
        raise Stage6WorkflowError(
            "Stage 6 review authorization record is not canonical"
        ) from exc
    if type(record) is not dict or type(plain) is not dict or plain != record:
        raise Stage6WorkflowError(
            "Stage 6 review authorization record is not a plain canonical object"
        )
    return plain


def _run_stage6_workflow_at_root(
    *,
    config: object,
    config_path: str | Path,
    run_id: str,
    stage5_gate_path: str | Path,
    base_output_root: str | Path,
    review_authorization_handle: Stage6ReviewAuthorizationHandle,
    stage5_authority: FrozenStage5AuthorityHandle,
    input_pin: Stage6InputPin,
    planning_warm_start_context: object | None = None,
    planning_warm_start_sha256: str | None = None,
    planning_child_recovery_capability: object | None = None,
    verified_parent_u74: object | None = None,
    run_lease: RunLease | None = None,
) -> Stage6WorkflowResult:
    repo = Path(__file__).resolve().parents[3]
    base, run_root = _resolve_stage6_run_root(base_output_root, run_id)
    lease_path = run_root / ".stage6.lease"
    if run_lease is not None:
        if type(run_lease) is not RunLease or run_lease.path != lease_path:
            raise Stage6WorkflowError(
                "Stage 6 planning child launch RunLease binding drifted"
            )
        try:
            run_lease.require_current()
        except RunLeaseError as exc:
            raise Stage6WorkflowError(
                "Stage 6 planning child launch RunLease is not current"
            ) from exc
    if (
        planning_child_recovery_capability is not None
        and run_lease is None
    ):
        raise Stage6WorkflowError(
            "Stage 6 planning child launch requires the pre-acquired RunLease"
        )
    warm_bindings = (
        planning_warm_start_context,
        planning_warm_start_sha256,
        verified_parent_u74,
    )
    if any(value is not None for value in warm_bindings):
        if any(value is None for value in warm_bindings):
            raise Stage6WorkflowError(
                "Stage 6 planning warm-start binding is incomplete"
            )
        _require_planning_warm_start_current(
            planning_warm_start_context,
            "before run output side effects",
        )
    if planning_child_recovery_capability is not None:
        _require_planning_child_recovery_capability(
            planning_child_recovery_capability,
            "before run output side effects",
        )
    _validate_stage6_repair_context_combination(
        planning_warm_start_context=planning_warm_start_context,
        planning_child_recovery_capability=(
            planning_child_recovery_capability
        ),
        classic_source_repair_context=None,
    )
    effective_config_bytes = getattr(
        planning_warm_start_context,
        "child_effective_config_bytes",
        None,
    )
    planning_warm_start_path = getattr(
        planning_warm_start_context,
        "artifact_path",
        None,
    )
    planning_child_source_repair_path = (
        planning_child_recovery_capability.stage_root
        / "planning-child-source-repair.json"
        if planning_child_recovery_capability is not None
        else None
    )
    planning_child_source_repair_continuation_path = (
        planning_child_recovery_capability.stage_root
        / "planning-child-source-repair-continuation.json"
        if planning_child_recovery_capability is not None
        else None
    )
    if planning_warm_start_context is not None and (
        type(effective_config_bytes) is not bytes
        or not effective_config_bytes
        or not isinstance(planning_warm_start_path, Path)
    ):
        raise Stage6WorkflowError(
            "Stage 6 planning warm-start effective config/input pin is missing"
        )
    if planning_child_recovery_capability is not None and (
        not isinstance(planning_child_source_repair_path, Path)
        or planning_child_source_repair_path
        != run_root / "s6" / "planning-child-source-repair.json"
    ):
        raise Stage6WorkflowError(
            "Stage 6 planning child source-repair stage root is missing"
        )
    if planning_child_recovery_capability is not None and (
        not isinstance(
            planning_child_source_repair_continuation_path,
            Path,
        )
        or planning_child_source_repair_continuation_path
        != run_root
        / "s6"
        / "planning-child-source-repair-continuation.json"
        or planning_child_recovery_capability
        .continuation_artifact_sha256
        is None
    ):
        raise Stage6WorkflowError(
            "Stage 6 planning child continuation stage root is missing"
        )
    if run_root.drive.upper() != "D:":
        raise Stage6WorkflowError("Stage 6 runtime output must be on D drive")
    _require_review_authorization_current(
        review_authorization_handle,
        "Stage 6 review authorization before run output side effects",
    )
    _require_input_pin_current(
        input_pin,
        "Stage 6 input pin before run output side effects",
    )
    try:
        run_preexisted = False
        if run_root.exists():
            require_plain_path(
                run_root,
                base=base,
                leaf_kind="directory",
                label="Stage 6 run root",
            )
            run_preexisted = any(
                child.name != lease_path.name for child in run_root.iterdir()
            )
        run_root.mkdir(parents=True, exist_ok=True)
        require_plain_path(
            run_root,
            base=base,
            leaf_kind="directory",
            label="Stage 6 run root",
        )
        owns_run_lease = run_lease is None
        if run_lease is None:
            run_lease = RunLease(lease_path)
        lease_scope = run_lease if owns_run_lease else nullcontext(run_lease)
        with lease_scope:
            run_lease.require_current()
            execution_identity = _validate_stage6_pinned_execution_identity(
                input_pin=input_pin,
                repo_root=repo,
                config_path=config_path,
                run_id=run_id,
                stage5_authority=stage5_authority,
                review_authorization_handle=review_authorization_handle,
                effective_config_bytes=effective_config_bytes,
            )
            expected_input_requests = _stage6_input_pin_requests(
                repo_root=repo,
                config_path=config_path,
                stage5_authority=stage5_authority,
                review_authorization_handle=review_authorization_handle,
                planning_warm_start_path=planning_warm_start_path,
                planning_child_source_repair_path=(
                    planning_child_source_repair_path
                ),
                planning_child_source_repair_continuation_path=(
                    planning_child_source_repair_continuation_path
                ),
            )
            with _stage6_execution_capability_scope(
                config=config,
                config_path=config_path,
                formal_run_id=run_id,
                run_root=run_root,
                stage_root=run_root / "s6",
                repo_root=repo,
                execution_identity=execution_identity,
                expected_input_requests=expected_input_requests,
                review_authorization_handle=review_authorization_handle,
                input_pin=input_pin,
                stage5_authority=stage5_authority,
                run_lease=run_lease,
                effective_config_bytes=effective_config_bytes,
                planning_warm_start_path=planning_warm_start_path,
                planning_child_source_repair_path=(
                    planning_child_source_repair_path
                ),
                planning_child_source_repair_continuation_path=(
                    planning_child_source_repair_continuation_path
                ),
            ) as execution_capability:
                return _run_stage6_workflow_locked(
                    config=config,
                    config_path=config_path,
                    run_id=run_id,
                    stage5_gate_path=stage5_gate_path,
                    base=base,
                    run_root=run_root,
                    repo=repo,
                    run_preexisted=run_preexisted,
                    review_authorization_handle=review_authorization_handle,
                    stage5_authority=stage5_authority,
                    input_pin=input_pin,
                    execution_identity=execution_identity,
                    execution_capability=execution_capability,
                    run_lease=run_lease,
                    planning_warm_start_context=planning_warm_start_context,
                    planning_warm_start_sha256=planning_warm_start_sha256,
                    planning_child_recovery_capability=(
                        planning_child_recovery_capability
                    ),
                    verified_parent_u74=verified_parent_u74,
                )
    except (OSError, PathSecurityError, RunLeaseError) as exc:
        raise Stage6WorkflowError(
            "Stage 6 run is already active or its single-writer lease is unavailable"
        ) from exc


def _recovery_resource_gate_from_monitor(monitor: object) -> dict[str, object]:
    """Build a gate record from already-sampled monitor properties only."""

    from lunar_exploration_ppo.utils.resources import (
        ResourceSnapshot,
        evaluate_resource_gates,
    )

    try:
        snapshot = ResourceSnapshot(
            d_free_bytes=int(shutil.disk_usage(Path("D:/")).free),
            rss_bytes=int(monitor.peak_aggregate_rss_bytes),
            peak_vram_bytes=0,
            rss_source="process_tree_lifecycle_peak_current_sum/v1",
            rss_root_pid=int(monitor.root_pid),
            rss_sample_count=int(monitor.sample_count),
            rss_latest_process_count=int(monitor.latest_process_count),
            rss_peak_process_count=int(monitor.peak_process_count),
        )
        decision = evaluate_resource_gates(snapshot, preflight=False)
    except (AttributeError, OSError, TypeError, ValueError) as exc:
        raise Stage6WorkflowError(
            "Stage 6 recovery resource snapshot failed"
        ) from exc
    record = {
        "d_free_bytes": snapshot.d_free_bytes,
        "rss_bytes": snapshot.rss_bytes,
        "peak_vram_bytes": snapshot.peak_vram_bytes,
        "rss_source": snapshot.rss_source,
        "rss_root_pid": snapshot.rss_root_pid,
        "rss_sample_count": snapshot.rss_sample_count,
        "rss_latest_process_count": snapshot.rss_latest_process_count,
        "rss_peak_process_count": snapshot.rss_peak_process_count,
        "warnings": list(decision.warnings),
        "hard_stops": list(decision.hard_stops),
        "passed": decision.passed,
    }
    if not decision.passed:
        raise Stage6WorkflowError(
            "Stage 6 recovery resource hard gate failed: "
            + "; ".join(decision.hard_stops)
        )
    return record


def _recover_stage6_preterminal_receipt(
    *,
    stage_root: str | Path,
    repo_root: str | Path,
    review_authorization_handle: Stage6ReviewAuthorizationHandle,
    input_pin: Stage6InputPin,
    execution_capability: object,
    planning_child_recovery_capability: object | None = None,
) -> dict[str, object]:
    """Append one recovery segment and terminal, then pure-commit receipt bytes."""

    _require_review_authorization_current(
        review_authorization_handle,
        "Stage 6 review authorization at preterminal recovery entry",
    )
    from lunar_exploration_ppo.utils.resources import ProcessTreeRSSMonitor
    from lunar_exploration_ppo.workflows.stage6_terminal_recovery import (
        TerminalRecoveryError,
        append_stage6_recovery_resource_segment,
        append_stage6_recovery_resource_terminal,
        recover_stage6_terminal_commit,
    )

    stage = _require_plain_stage6_root(
        stage_root,
        label="Stage 6 preterminal recovery root",
    )
    monitor = ProcessTreeRSSMonitor(interval_seconds=3600.0)
    try:
        monitor.start()
        if not monitor.running or monitor.sample_count != 1:
            raise Stage6WorkflowError(
                "Stage 6 recovery monitor first sample raced"
            )
        first_sample = _recovery_resource_gate_from_monitor(monitor)
        if first_sample["rss_sample_count"] != 1:
            raise Stage6WorkflowError(
                "Stage 6 recovery segment first sample did not reset to one"
            )
        _require_review_authorization_current(
            review_authorization_handle,
            "Stage 6 review authorization before recovery resource segment append",
        )
        append_stage6_recovery_resource_segment(
            stage_root=stage,
            first_sample=first_sample,
            execution_capability=execution_capability,
            planning_child_recovery_capability=(
                planning_child_recovery_capability
            ),
        )
        monitor.sample_now()
        if monitor.sample_count != 2:
            raise Stage6WorkflowError(
                "Stage 6 recovery terminal sample raced"
            )
    except TerminalRecoveryError as exc:
        raise Stage6WorkflowError(
            "Stage 6 preterminal recovery resource append failed"
        ) from exc
    finally:
        monitor.stop()

    terminal_resource = _recovery_resource_gate_from_monitor(monitor)
    if terminal_resource["rss_sample_count"] != 2:
        raise Stage6WorkflowError(
            "Stage 6 recovery terminal sample count drifted"
        )
    try:
        _require_review_authorization_current(
            review_authorization_handle,
            "Stage 6 review authorization before recovery resource terminal append",
        )
        append_stage6_recovery_resource_terminal(
            stage_root=stage,
            terminal_resource=terminal_resource,
            execution_capability=execution_capability,
            planning_child_recovery_capability=(
                planning_child_recovery_capability
            ),
        )
        _require_input_pin_current(
            input_pin,
            "Stage 6 input pin before preterminal recovery commit",
        )
        _require_review_authorization_current(
            review_authorization_handle,
            "Stage 6 review authorization before preterminal recovery commit",
        )
        result = recover_stage6_terminal_commit(
            stage_root=stage,
            manifest_committer=lambda candidate: write_or_verify_stage6_manifest(
                candidate,
                repo_root=repo_root,
                planning_child_recovery_capability=(
                    planning_child_recovery_capability
                ),
            ),
            execution_capability=execution_capability,
            planning_child_recovery_capability=(
                planning_child_recovery_capability
            ),
        )
        _require_review_authorization_current(
            review_authorization_handle,
            "Stage 6 review authorization after preterminal recovery commit",
        )
        _require_input_pin_current(
            input_pin,
            "Stage 6 input pin after preterminal recovery commit",
        )
        return result
    except TerminalRecoveryError as exc:
        raise Stage6WorkflowError(
            "Stage 6 preterminal recovery terminal commit failed"
        ) from exc


def _verify_stage6_recovery_authorization_bindings(
    *,
    stage_root: str | Path,
    repo_root: str | Path,
    config_path: str | Path,
    run_id: str,
    stage5_gate_path: str | Path,
    review_authorization_handle: Stage6ReviewAuthorizationHandle,
    evidence_handle: _Stage6EvidenceHandle,
    planning_child_recovery_capability: object | None = None,
) -> None:
    """Cross-check current external authorization against receipt and lineage."""

    from lunar_exploration_ppo.workflows.stage6_terminal_recovery import (
        TerminalRecoveryError,
        _Stage6EvidenceHandle,
        _validate_receipt_payload,
    )

    stage = _require_plain_stage6_root(
        stage_root,
        label="Stage 6 recovery authorization root",
    )
    if (
        type(evidence_handle) is not _Stage6EvidenceHandle
        or evidence_handle.stage != stage
    ):
        raise Stage6WorkflowError(
            "Stage 6 recovery authorization evidence handle is missing or drifted"
        )
    try:
        receipt_payload = evidence_handle.read_bytes(
            "preterminal_acceptance.json",
            label="Stage 6 recovery receipt",
        )
        receipt, _, _ = _validate_receipt_payload(receipt_payload)
        lineage_payload = evidence_handle.read_bytes(
            "lineage_audit.json",
            label="Stage 6 recovery lineage",
        )
        lineage = _strict_canonical_json(
            lineage_payload,
            "Stage 6 recovery lineage",
        )
        base_config_payload = (
            Path(config_path).expanduser().resolve().read_bytes()
        )
        persisted_config_payload = evidence_handle.read_bytes(
            "config.json",
            label="Stage 6 recovery persisted config",
        )
    except (OSError, TerminalRecoveryError) as exc:
        raise Stage6WorkflowError(
            "Stage 6 recovery authorization evidence is unreadable"
        ) from exc
    origin_execution_identity = lineage.get("execution_identity")
    if not isinstance(origin_execution_identity, Mapping):
        raise Stage6WorkflowError(
            "Stage 6 recovery execution lineage is missing"
        )
    _require_review_authorization_current(
        review_authorization_handle,
        "Stage 6 review authorization before recovery binding verification",
    )
    effective_config_bytes = None
    if lineage.get("schema_version") == "stage6_lineage_audit/v3":
        from lunar_exploration_ppo.workflows.stage6_planning_warm_start import (
            Stage6PlanningWarmStartError,
            parse_planning_effective_config_bytes,
        )

        try:
            effective_config = parse_planning_effective_config_bytes(
                persisted_config_payload
            )
        except Stage6PlanningWarmStartError as exc:
            raise Stage6WorkflowError(
                "Stage 6 recovery effective config is invalid"
            ) from exc
        if (
            effective_config.base_config_sha256
            != hashlib.sha256(base_config_payload).hexdigest()
        ):
            raise Stage6WorkflowError(
                "Stage 6 recovery effective/base config drifted"
            )
        effective_config_bytes = effective_config.effective_config_bytes
        config_payload = effective_config_bytes
    else:
        if persisted_config_payload != base_config_payload:
            raise Stage6WorkflowError(
                "Stage 6 recovery persisted config drifted"
            )
        config_payload = base_config_payload
    current_execution_identity = stage6_execution_identity(
        repo_root=repo_root,
        config_path=config_path,
        effective_config_bytes=effective_config_bytes,
    )
    current = validate_stage6_verified_review_authorization(
        _review_authorization_record(review_authorization_handle),
        execution_identity=current_execution_identity,
        formal_run_id=run_id,
    )
    evidence_paths = set(evidence_handle.file_paths)
    planning_child_repair = None
    planning_child_repair_name = "planning-child-source-repair.json"
    planning_child_continuation_name = (
        "planning-child-source-repair-continuation.json"
    )
    classic_repair_names = {
        "source-repair-amendment.json",
        "source-repair-continuation.json",
        "source-repair-supplement.json",
        "source-repair-closure.json",
        "source-repair-frontier-recovery.json",
        "source-repair-sensor-acceleration.json",
    }
    if (
        planning_child_repair_name in evidence_paths
        and classic_repair_names.intersection(evidence_paths)
    ):
        raise Stage6WorkflowError(
            "Stage 6 planning child source-repair cannot be combined with "
            "classic source-repair evidence"
        )
    if (
        planning_child_continuation_name in evidence_paths
        and planning_child_repair_name not in evidence_paths
    ):
        raise Stage6WorkflowError(
            "Stage 6 planning child continuation has no parent amendment"
        )
    authority = verify_frozen_stage5_authority(
        gate_path=stage5_gate_path,
        repo_root=repo_root,
    )
    validated_recovery: dict[str, object] | None = None
    if planning_child_repair_name in evidence_paths:
        from lunar_exploration_ppo.workflows.stage6_planning_child_recovery import (
            PlanningChildRecoveryCapability,
        )

        if (
            planning_child_continuation_name not in evidence_paths
            or type(planning_child_recovery_capability)
            is not PlanningChildRecoveryCapability
        ):
            raise Stage6WorkflowError(
                "Stage 6 recovery planning child requires one issued "
                "recovery capability"
            )
        planning_child_repair = (
            planning_child_recovery_capability
        )
        _require_planning_child_recovery_capability(
            planning_child_repair,
            "recovery authorization boundary",
        )
        current_immutable = _stage6_current_immutable_bindings(
            execution_identity=current_execution_identity,
            verified_review_authorization=current,
            stage5_authority=authority,
        )
        if (
            planning_child_repair.stage_root != stage
            or planning_child_repair.formal_run_id != run_id
        ):
            raise Stage6WorkflowError(
                "Stage 6 recovery planning child capability root drifted"
            )
        validated_recovery = _validate_planning_child_machine_context(
            capability=planning_child_repair,
            lineage_audit=lineage,
            current_execution_identity=current_execution_identity,
            current_verified_review_authorization=current,
            current_immutable_bindings=current_immutable,
        )
    elif planning_child_recovery_capability is not None:
        raise Stage6WorkflowError(
            "Stage 6 recovery capability has no planning child graph"
        )
    embedded = _stage6_lineage_review_authorization(
        stage,
        payload=lineage_payload,
        repo_root=repo_root,
        source_repair_amendment_payload=(
            evidence_handle.read_bytes(
                "source-repair-amendment.json",
                label="Stage 6 recovery embedded source-repair amendment",
            )
            if "source-repair-amendment.json" in evidence_paths
            else None
        ),
        source_repair_continuation_payload=(
            evidence_handle.read_bytes(
                "source-repair-continuation.json",
                label="Stage 6 recovery embedded source-repair continuation",
            )
            if "source-repair-continuation.json" in evidence_paths
            else None
        ),
        source_repair_supplement_payload=(
            evidence_handle.read_bytes(
                "source-repair-supplement.json",
                label="Stage 6 recovery embedded source-repair supplement",
            )
            if "source-repair-supplement.json" in evidence_paths
            else None
        ),
        source_repair_closure_payload=(
            evidence_handle.read_bytes(
                "source-repair-closure.json",
                label="Stage 6 recovery embedded source-repair closure",
            )
            if "source-repair-closure.json" in evidence_paths
            else None
        ),
        source_repair_frontier_recovery_payload=(
            evidence_handle.read_bytes(
                "source-repair-frontier-recovery.json",
                label=(
                    "Stage 6 recovery embedded source-repair frontier recovery"
                ),
            )
            if "source-repair-frontier-recovery.json" in evidence_paths
            else None
        ),
        source_repair_sensor_acceleration_payload=(
            evidence_handle.read_bytes(
                "source-repair-sensor-acceleration.json",
                label=(
                    "Stage 6 recovery embedded source-repair sensor acceleration"
                ),
            )
            if "source-repair-sensor-acceleration.json" in evidence_paths
            else None
        ),
    )
    immutable = receipt.get("immutable_bindings")
    lineage_immutable = lineage.get("immutable_bindings")
    config_sha256 = hashlib.sha256(config_payload).hexdigest()
    if planning_child_repair is None:
        common_drift = (
            current != embedded
            or not isinstance(immutable, Mapping)
            or not isinstance(lineage_immutable, Mapping)
            or immutable != lineage_immutable
            or immutable.get("formal_run_id") != run_id
            or stage.parent.name != run_id
            or current.get("formal_run_id") != run_id
            or current.get("config_sha256") != config_sha256
            or immutable.get("config_sha256") != config_sha256
        )
    else:
        if not isinstance(validated_recovery, Mapping):
            raise Stage6WorkflowError(
                "Stage 6 recovery planning child capability is incomplete"
            )
        common_drift = (
            not isinstance(immutable, Mapping)
            or not isinstance(lineage_immutable, Mapping)
            or embedded
            != planning_child_repair.acceptance_binding.get(
                "origin_verified_review_authorization"
            )
            or immutable
            != validated_recovery.get("current_immutable_bindings")
            or lineage_immutable
            != validated_recovery.get(
                "historical_immutable_bindings"
            )
            or immutable.get("formal_run_id") != run_id
            or stage.parent.name != run_id
            or current.get("formal_run_id") != run_id
            or current.get("config_sha256") != config_sha256
            or immutable.get("config_sha256") != config_sha256
        )
    source_repair = None
    if (
        "source-repair-continuation.json" in evidence_paths
        and "source-repair-amendment.json" not in evidence_paths
    ):
        raise Stage6WorkflowError(
            "Stage 6 recovery source-repair continuation has no primary"
        )
    if (
        "source-repair-supplement.json" in evidence_paths
        and "source-repair-continuation.json" not in evidence_paths
    ):
        raise Stage6WorkflowError(
            "Stage 6 recovery source-repair supplement has no continuation"
        )
    if (
        "source-repair-closure.json" in evidence_paths
        and "source-repair-supplement.json" not in evidence_paths
    ):
        raise Stage6WorkflowError(
            "Stage 6 recovery source-repair closure has no supplement"
        )
    if (
        "source-repair-frontier-recovery.json" in evidence_paths
        and "source-repair-closure.json" not in evidence_paths
    ):
        raise Stage6WorkflowError(
            "Stage 6 recovery source-repair frontier recovery has no closure"
        )
    if (
        "source-repair-sensor-acceleration.json" in evidence_paths
        and "source-repair-frontier-recovery.json" not in evidence_paths
    ):
        raise Stage6WorkflowError(
            "Stage 6 recovery source-repair sensor acceleration has no "
            "frontier recovery"
        )
    if "source-repair-amendment.json" in evidence_paths:
        source_repair = _load_stage6_source_repair_for_current_identity(
            stage=stage,
            current_execution_identity=current_execution_identity,
            amendment_payload=evidence_handle.read_bytes(
                "source-repair-amendment.json",
                label="Stage 6 recovery source-repair amendment",
            ),
            continuation_payload=(
                evidence_handle.read_bytes(
                    "source-repair-continuation.json",
                    label="Stage 6 recovery source-repair continuation",
                )
                if "source-repair-continuation.json" in evidence_paths
                else None
            ),
            supplement_payload=(
                evidence_handle.read_bytes(
                    "source-repair-supplement.json",
                    label="Stage 6 recovery source-repair supplement",
                )
                if "source-repair-supplement.json" in evidence_paths
                else None
            ),
            closure_payload=(
                evidence_handle.read_bytes(
                    "source-repair-closure.json",
                    label="Stage 6 recovery source-repair closure",
                )
                if "source-repair-closure.json" in evidence_paths
                else None
            ),
            frontier_recovery_payload=(
                evidence_handle.read_bytes(
                    "source-repair-frontier-recovery.json",
                    label="Stage 6 recovery source-repair frontier recovery",
                )
                if "source-repair-frontier-recovery.json" in evidence_paths
                else None
            ),
            sensor_acceleration_payload=(
                evidence_handle.read_bytes(
                    "source-repair-sensor-acceleration.json",
                    label="Stage 6 recovery source-repair sensor acceleration",
                )
                if "source-repair-sensor-acceleration.json" in evidence_paths
                else None
            ),
        )
    if planning_child_repair is not None:
        if not isinstance(validated_recovery, Mapping):
            raise Stage6WorkflowError(
                "Stage 6 recovery planning child capability is incomplete"
            )
        lineage_epochs = planning_child_repair.lineage_epochs
        origin_identity_sha256 = hashlib.sha256(
            ArtifactStore.canonical_json_bytes(origin_execution_identity)
        ).hexdigest()
        current_identity_sha256 = hashlib.sha256(
            ArtifactStore.canonical_json_bytes(current_execution_identity)
        ).hexdigest()
        identity_drift = (
            len(lineage_epochs) != 3
            or lineage_epochs[0].execution_identity_sha256
            != origin_identity_sha256
            or lineage_epochs[-1].execution_identity_sha256
            != current_identity_sha256
            or planning_child_repair.acceptance_binding.get(
                "origin_verified_review_authorization"
            )
            != lineage.get("verified_review_authorization")
            or planning_child_repair.acceptance_binding.get(
                "current_verified_review_authorization"
            )
            != current
            or validated_recovery.get(
                "historical_immutable_bindings"
            )
            != lineage_immutable
            or validated_recovery.get("current_immutable_bindings")
            != immutable
        )
    elif source_repair is None:
        identity_drift = (
            origin_execution_identity != current_execution_identity
            or lineage.get("verified_review_authorization") != current
            or immutable.get("review_authorization_record_sha256")
            != _canonical_sha256(current)
            or any(
                immutable.get(name) != current.get(name)
                for name in (
                    "source_set_sha256",
                    "prospective_tree_sha256",
                    "data_sha256",
                    "environment_sha256",
                    "changed_path_set_sha256",
                    "authorization_file_sha256",
                    "review_identity_sha256",
                    "reviewed_prospective_git_tree",
                    "frozen_diff_sha256",
                    "spec_review_sha256",
                    "quality_review_sha256",
                )
            )
        )
    else:
        identity_drift = (
            dict(source_repair.origin_execution_identity)
            != origin_execution_identity
            or dict(source_repair.origin_immutable_bindings) != lineage_immutable
            or dict(source_repair.origin_verified_review_authorization)
            != lineage.get("verified_review_authorization")
            or dict(source_repair.current_execution_identity)
            != current_execution_identity
            or dict(source_repair.current_verified_review_authorization) != current
            or any(
                source_repair.current_immutable_bindings.get(name)
                != current.get(name)
                for name in (
                    "source_set_sha256",
                    "prospective_tree_sha256",
                    "data_sha256",
                    "environment_sha256",
                    "changed_path_set_sha256",
                    "authorization_file_sha256",
                    "review_identity_sha256",
                    "reviewed_prospective_git_tree",
                    "frozen_diff_sha256",
                    "spec_review_sha256",
                    "quality_review_sha256",
                )
            )
        )
    if common_drift or identity_drift:
        raise Stage6WorkflowError(
            "Stage 6 current authorization/receipt/lineage binding drifted"
        )
    if immutable.get("stage5_gate_sha256") != authority.identity.get("gate_sha256"):
        raise Stage6WorkflowError(
            "Stage 6 recovery Stage 5 authority binding drifted"
        )
    authority.require_current("Stage 6 recovery external authority")
    _require_review_authorization_current(
        review_authorization_handle,
        "Stage 6 review authorization after recovery binding verification",
    )
    try:
        evidence_handle.require_current(
            "Stage 6 recovery authorization evidence"
        )
    except TerminalRecoveryError as exc:
        raise Stage6WorkflowError(
            "Stage 6 recovery authorization evidence changed"
        ) from exc


def _stage6_recovery_workflow_result(
    *,
    stage_root: str | Path,
    run_id: str,
    evidence_handle: _Stage6EvidenceHandle,
) -> Stage6WorkflowResult:
    """Read summary and routing only from receipt-fixed exact terminal bytes."""

    from lunar_exploration_ppo.workflows.stage6_terminal_recovery import (
        TerminalRecoveryError,
        _Stage6EvidenceHandle,
        _validate_receipt_payload,
    )

    stage = _require_plain_stage6_root(
        stage_root,
        label="Stage 6 recovery result root",
    )
    if (
        type(evidence_handle) is not _Stage6EvidenceHandle
        or evidence_handle.stage != stage
    ):
        raise Stage6WorkflowError(
            "Stage 6 recovery result requires its exact terminal evidence handle"
        )
    try:
        receipt_payload = evidence_handle.read_bytes(
            "preterminal_acceptance.json",
            label="Stage 6 recovery result receipt",
        )
        receipt, _, _ = _validate_receipt_payload(receipt_payload)
    except TerminalRecoveryError as exc:
        raise Stage6WorkflowError(
            "Stage 6 recovery result receipt is incomplete or drifted"
        ) from exc
    rows = receipt.get("terminal_artifacts")
    if not isinstance(rows, list):
        raise Stage6WorkflowError("Stage 6 recovery terminal artifacts drifted")
    by_path = {
        row.get("path"): row
        for row in rows
        if isinstance(row, Mapping) and isinstance(row.get("path"), str)
    }
    parsed: dict[str, dict[str, object]] = {}
    for relative in ("summary.json", "routing.json"):
        row = by_path.get(relative)
        if not isinstance(row, Mapping) or not isinstance(row.get("utf8"), str):
            raise Stage6WorkflowError(
                "Stage 6 recovery receipt-fixed summary/routing is missing"
            )
        expected = str(row["utf8"]).encode("utf-8")
        try:
            actual = evidence_handle.read_bytes(
                relative,
                label=f"Stage 6 recovery {relative}",
            )
        except TerminalRecoveryError as exc:
            raise Stage6WorkflowError(
                "Stage 6 recovery summary/routing is unreadable"
            ) from exc
        if (
            actual != expected
            or hashlib.sha256(actual).hexdigest() != row.get("sha256")
            or len(actual) != row.get("size_bytes")
        ):
            raise Stage6WorkflowError(
                "Stage 6 recovery receipt-fixed summary/routing drifted"
            )
        parsed[relative] = _strict_canonical_json(
            actual,
            f"Stage 6 recovery {relative}",
        )
    summary = parsed["summary.json"]
    routing = parsed["routing.json"]
    semantic = receipt.get("semantic_verification")
    immutable = receipt.get("immutable_bindings")
    global_checkpoint = receipt.get("global_checkpoint_identity")
    if (
        not isinstance(semantic, Mapping)
        or not isinstance(immutable, Mapping)
        or not isinstance(global_checkpoint, Mapping)
        or immutable.get("formal_run_id") != run_id
        or semantic.get("passed") is not True
        or semantic.get("state") != "awaiting_independent_review"
        or summary.get("state") != semantic.get("state")
        or summary.get("machine_passed") is not True
        or summary.get("final_evaluation_count")
        != semantic.get("final_evaluation_count")
        or summary.get("final_episode_count")
        != semantic.get("final_episode_count")
        or summary.get("checkpoint_receipt_count")
        != semantic.get("checkpoint_receipt_count")
        or summary.get("global_best") != global_checkpoint.get("record")
        or routing.get("state") != semantic.get("state")
        or routing.get("route") != "awaiting_independent_review"
        or routing.get("machine_passed") is not True
    ):
        raise Stage6WorkflowError(
            "Stage 6 recovery receipt-fixed result acceptance drifted"
        )
    try:
        evidence_handle.require_current(
            "Stage 6 recovery result evidence"
        )
    except TerminalRecoveryError as exc:
        raise Stage6WorkflowError(
            "Stage 6 recovery result evidence changed"
        ) from exc
    return Stage6WorkflowResult(
        run_id=run_id,
        stage_root=stage,
        summary=summary,
        routing=routing,
    )


def _run_stage6_workflow_locked(
    *,
    config: object,
    config_path: str | Path,
    run_id: str,
    stage5_gate_path: str | Path,
    base: Path,
    run_root: Path,
    repo: Path,
    run_preexisted: bool,
    review_authorization_handle: Stage6ReviewAuthorizationHandle,
    stage5_authority: FrozenStage5AuthorityHandle,
    input_pin: Stage6InputPin,
    execution_identity: Mapping[str, object],
    execution_capability: object,
    run_lease: RunLease | None = None,
    planning_warm_start_context: object | None = None,
    planning_warm_start_sha256: str | None = None,
    planning_child_recovery_capability: object | None = None,
    verified_parent_u74: object | None = None,
) -> Stage6WorkflowResult:
    if run_lease is None:
        run_lease = _stage6_capability_state(execution_capability).run_lease
    _require_stage6_execution_capability(
        execution_capability,
        label="Stage 6 protected workflow entry",
        formal_run_id=run_id,
        run_root=run_root,
        stage_root=run_root / "s6",
        repo_root=repo,
        config_path=config_path,
        config=config,
        execution_identity=execution_identity,
        stage5_authority=stage5_authority.identity,
        verified_review_authorization=_review_authorization_record(
            review_authorization_handle
        ),
    )
    _require_review_authorization_current(
        review_authorization_handle,
        "Stage 6 review authorization after run lease",
    )
    _require_input_pin_current(
        input_pin,
        "Stage 6 input pin after run lease",
    )
    if planning_warm_start_context is not None:
        _require_planning_warm_start_current(
            planning_warm_start_context,
            "after run lease",
        )
    _validate_stage6_repair_context_combination(
        planning_warm_start_context=planning_warm_start_context,
        planning_child_recovery_capability=(
            planning_child_recovery_capability
        ),
        classic_source_repair_context=None,
    )
    if planning_child_recovery_capability is not None:
        _require_planning_child_recovery_capability(
            planning_child_recovery_capability,
            "after run lease",
        )
    effective_config_bytes = getattr(
        planning_warm_start_context,
        "child_effective_config_bytes",
        None,
    )
    if planning_warm_start_context is not None and (
        type(effective_config_bytes) is not bytes
        or not effective_config_bytes
    ):
        raise Stage6WorkflowError(
            "Stage 6 planning warm-start effective config is missing"
        )
    from lunar_exploration_ppo.workflows.stage6_terminal_recovery import (
        COMPLETE_STATES,
        TerminalRecoveryError,
        _capture_terminal_evidence_handle,
        detect_stage6_terminal_recovery,
        recover_stage6_terminal_commit,
    )

    stage_root = run_root / "s6"
    detection = detect_stage6_terminal_recovery(
        stage_root=stage_root,
        manifest_verifier=lambda candidate: _verify_existing_stage6_manifest_identity(
            candidate,
            repo_root=repo,
            planning_child_recovery_capability=(
                planning_child_recovery_capability
            ),
        ),
        planning_child_recovery_capability=(
            planning_child_recovery_capability
        ),
    )
    status = detection.get("status")
    if status == "invalid":
        raise Stage6WorkflowError(
            "Stage 6 terminal recovery detector failed closed: "
            f"{detection.get('reason', '')}"
        )
    if status in {"valid_terminal_recovery", "valid_preterminal_recovery"}:
        try:
            with _capture_terminal_evidence_handle(
                stage_root
            ) as recovery_authorization_evidence:
                _verify_stage6_recovery_authorization_bindings(
                    stage_root=stage_root,
                    repo_root=repo,
                    config_path=config_path,
                    run_id=run_id,
                    stage5_gate_path=stage5_gate_path,
                    review_authorization_handle=review_authorization_handle,
                    evidence_handle=recovery_authorization_evidence,
                    planning_child_recovery_capability=(
                        planning_child_recovery_capability
                    ),
                )
                recovery_authorization_evidence.require_current(
                    "Stage 6 recovery evidence before commit convergence"
                )
        except TerminalRecoveryError as exc:
            raise Stage6WorkflowError(
                "Stage 6 recovery authorization evidence could not remain bound"
            ) from exc
        if status == "valid_preterminal_recovery":
            _recover_stage6_preterminal_receipt(
                stage_root=stage_root,
                repo_root=repo,
                review_authorization_handle=review_authorization_handle,
                input_pin=input_pin,
                execution_capability=execution_capability,
                planning_child_recovery_capability=(
                    planning_child_recovery_capability
                ),
            )
        else:
            try:
                _require_review_authorization_current(
                    review_authorization_handle,
                    "Stage 6 review authorization before terminal recovery commit",
                )
                _require_input_pin_current(
                    input_pin,
                    "Stage 6 input pin before terminal recovery commit",
                )
                recover_stage6_terminal_commit(
                    stage_root=stage_root,
                    manifest_committer=lambda candidate: write_or_verify_stage6_manifest(
                        candidate,
                        repo_root=repo,
                        planning_child_recovery_capability=(
                            planning_child_recovery_capability
                        ),
                    ),
                    execution_capability=execution_capability,
                    planning_child_recovery_capability=(
                        planning_child_recovery_capability
                    ),
                )
                _require_review_authorization_current(
                    review_authorization_handle,
                    "Stage 6 review authorization after terminal recovery commit",
                )
                _require_input_pin_current(
                    input_pin,
                    "Stage 6 input pin after terminal recovery commit",
                )
            except TerminalRecoveryError as exc:
                raise Stage6WorkflowError(
                    "Stage 6 terminal-only recovery commit failed"
                ) from exc
        try:
            with _capture_terminal_evidence_handle(stage_root) as terminal_evidence:

                def manifest_verifier(candidate: Path) -> dict[str, object]:
                    return _verify_stage6_terminal_manifest_evidence(
                        candidate,
                        source_identity=stage6_source_identity(repo),
                        repo_root=repo,
                        evidence_handle=terminal_evidence,
                        planning_child_recovery_capability=(
                            planning_child_recovery_capability
                        ),
                    )

                final_detection = detect_stage6_terminal_recovery(
                    stage_root=stage_root,
                    manifest_verifier=manifest_verifier,
                    evidence_handle=terminal_evidence,
                    planning_child_recovery_capability=(
                        planning_child_recovery_capability
                    ),
                )
                if (
                    final_detection.get("status")
                    != "valid_terminal_recovery"
                    or tuple(final_detection.get("phase_states", ()))
                    != COMPLETE_STATES
                ):
                    raise Stage6WorkflowError(
                        "Stage 6 final terminal recovery replay failed closed: "
                        f"{final_detection.get('reason', '')}"
                    )
                terminal_evidence.require_current(
                    "Stage 6 terminal recovery evidence after final replay"
                )
                _verify_stage6_recovery_authorization_bindings(
                    stage_root=stage_root,
                    repo_root=repo,
                    config_path=config_path,
                    run_id=run_id,
                    stage5_gate_path=stage5_gate_path,
                    review_authorization_handle=review_authorization_handle,
                    evidence_handle=terminal_evidence,
                    planning_child_recovery_capability=(
                        planning_child_recovery_capability
                    ),
                )
                recovery_result = _stage6_recovery_workflow_result(
                    stage_root=stage_root,
                    run_id=run_id,
                    evidence_handle=terminal_evidence,
                )
                _require_review_authorization_current(
                    review_authorization_handle,
                    "Stage 6 review authorization before recovery result return",
                )
                _require_input_pin_current(
                    input_pin,
                    "Stage 6 input pin before recovery result return",
                )
                terminal_evidence.require_current(
                    "Stage 6 terminal recovery evidence immediately before return"
                )
                return recovery_result
        except TerminalRecoveryError as exc:
            raise Stage6WorkflowError(
                "Stage 6 terminal recovery evidence could not remain bound"
            ) from exc
    if status != "no_terminal":
        raise Stage6WorkflowError("Stage 6 terminal detector returned unknown status")

    authority = stage5_authority
    if run_preexisted:
        _verify_stage6_machine_preflight_for_resume_at_root(
            config_path=config_path,
            run_id=run_id,
            stage5_gate_path=stage5_gate_path,
            base_output_root=base,
            effective_config_bytes=effective_config_bytes,
        )
    else:
        _run_stage6_machine_preflight(
            config_path=config_path,
            run_id=run_id,
            stage5_gate_path=stage5_gate_path,
            base_output_root=base,
            require_canonical_output=False,
            config=config,
            execution_identity=execution_identity,
            review_authorization_handle=review_authorization_handle,
            input_pin=input_pin,
            stage5_authority=stage5_authority,
            run_lease=run_lease,
            execution_capability=execution_capability,
            effective_config_bytes=effective_config_bytes,
        )
    _require_review_authorization_current(
        review_authorization_handle,
        "Stage 6 review authorization after machine preflight before backend import",
    )
    _require_input_pin_current(
        input_pin,
        "Stage 6 input pin after machine preflight before backend import",
    )
    if planning_child_recovery_capability is not None:
        _require_planning_child_recovery_capability(
            planning_child_recovery_capability,
            "after machine preflight before backend import",
        )
    try:
        require_plain_path(
            run_root / "s6",
            base=base,
            leaf_kind="directory",
            label="Stage 6 stage root",
        )
    except PathSecurityError as exc:
        raise Stage6WorkflowError(
            "Stage 6 stage root contains a link or reparse point"
        ) from exc
    from lunar_exploration_ppo.ppo.standard_training import execute_standard_training

    _require_review_authorization_current(
        review_authorization_handle,
        "Stage 6 review authorization before training call",
    )
    _require_input_pin_current(
        input_pin,
        "Stage 6 input pin before training call",
    )
    if planning_child_recovery_capability is not None:
        _require_planning_child_recovery_capability(
            planning_child_recovery_capability,
            "before training call",
        )
    result = execute_standard_training(
        config=config,
        run_root=run_root,
        repo_root=repo,
        stage5_authority=authority.identity,
        verified_review_authorization=_review_authorization_record(
            review_authorization_handle
        ),
        execution_capability=execution_capability,
        planning_warm_start_context=planning_warm_start_context,
        planning_warm_start_sha256=planning_warm_start_sha256,
        planning_child_recovery_capability=(
            planning_child_recovery_capability
        ),
        verified_parent_u74=verified_parent_u74,
    )
    if planning_warm_start_context is not None:
        _require_planning_warm_start_current(
            planning_warm_start_context,
            "after training return",
        )
    if planning_child_recovery_capability is not None:
        _require_planning_child_recovery_capability(
            planning_child_recovery_capability,
            "after training return",
        )
    _require_review_authorization_current(
        review_authorization_handle,
        "Stage 6 review authorization after training return",
    )
    _require_input_pin_current(
        input_pin,
        "Stage 6 input pin after training return",
    )
    authority.require_current("Stage 6 final Stage 5 authority")
    return Stage6WorkflowResult(
        run_id=run_id,
        stage_root=Path(result["stage_root"]),
        summary=dict(result["summary"]),
        routing=dict(result["routing"]),
    )


def _validate_gate_history(history: object, bindings: dict[str, object]) -> None:
    if not isinstance(history, list) or len(history) != len(_HISTORY_STATES):
        raise Stage6WorkflowError("Stage 5 gate hash-chain length drift")
    previous = ABSENT_AUTHORITY_SHA256
    for index, state in enumerate(_HISTORY_STATES):
        event_bindings = json.loads(
            json.dumps(bindings, ensure_ascii=False, sort_keys=True)
        )
        if state in {"machine_passed", "awaiting_independent_review"}:
            event_bindings["review_sha256"] = ABSENT_AUTHORITY_SHA256
        if state not in {"approved", "next_stage"}:
            event_bindings["approval_sha256"] = ABSENT_AUTHORITY_SHA256
            event_bindings["commit_sha256"] = ABSENT_AUTHORITY_SHA256
        event = {
            "state": state,
            "previous_record_hash": previous,
            "bindings": event_bindings,
        }
        expected = {**event, "record_hash": _canonical_sha256(event)}
        if history[index] != expected:
            raise Stage6WorkflowError("Stage 5 gate hash-chain drift")
        previous = expected["record_hash"]


def _strict_canonical_json(payload: bytes, label: str) -> dict[str, object]:
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Stage6WorkflowError(f"{label} is not canonical UTF-8 JSON") from exc
    if not isinstance(value, dict) or ArtifactStore.canonical_json_bytes(value) != payload:
        raise Stage6WorkflowError(f"{label} is not canonical JSON")
    return value


def _strict_canonical_jsonl(
    source: Path | bytes,
    label: str,
) -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    try:
        payload = source if isinstance(source, bytes) else source.read_bytes()
        for line in payload.splitlines(keepends=True):
            value = json.loads(line.decode("utf-8"))
            canonical = (
                json.dumps(
                    value,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            ).encode("utf-8")
            if not isinstance(value, dict) or line != canonical:
                raise ValueError("noncanonical JSONL row")
            rows.append(value)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise Stage6WorkflowError(f"{label} is invalid") from exc
    return tuple(rows)


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(ArtifactStore.canonical_json_bytes(value)).hexdigest()


def _is_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return value == value.lower()


def _verify_git_identity(repo_root: Path) -> None:
    if _git_text(repo_root, "HEAD") != STAGE5_COMMIT:
        raise Stage6WorkflowError("Stage 5 commit drift")
    if _git_text(repo_root, "HEAD^{tree}") != STAGE5_TREE:
        raise Stage6WorkflowError("Stage 5 commit tree drift")


def _git_text(repo_root: Path, revision: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", revision],
        capture_output=True,
        check=False,
        text=True,
        encoding="utf-8",
    )
    if completed.returncode != 0:
        raise Stage6WorkflowError(f"git rev-parse failed: {revision}")
    return completed.stdout.strip()


__all__ = [
    "FINAL_EVALUATION_METHODS",
    "FinalEvaluationJob",
    "FrozenStage5AuthorityHandle",
    "PerformanceDecision",
    "STAGE6_MANIFEST_BOUND_ARTIFACTS",
    "STAGE6_PRODUCTION_SOURCE_PATHS",
    "STAGE6_SOURCE_PATHS",
    "STAGE6_TEST_SOURCE_PATHS",
    "Stage6ManifestHandle",
    "Stage6StateJournal",
    "Stage6WorkflowResult",
    "Stage6WorkflowError",
    "assert_eval_only_hashes_unchanged",
    "build_final_evaluation_plan",
    "build_machine_acceptance",
    "decide_performance_advantage",
    "load_stage4_policy_for_standard",
    "rebind_stage6_manifest_for_terminal_resource",
    "run_stage6_machine_preflight",
    "run_stage6_workflow",
    "stage6_execution_identity",
    "stage6_source_identity",
    "validate_stage6_machine_preflight_audit",
    "validate_stage6_formal_run_id",
    "validate_stage6_standalone_preflight_run_id",
    "verify_frozen_stage5_authority",
    "verify_stage6_manifest",
    "verify_stage6_machine_acceptance",
    "verify_stage6_machine_preflight_for_resume",
    "verify_stage6_pending_acceptance",
    "verify_stage6_preterminal_acceptance",
    "verify_stage5_gate_bytes",
    "write_or_verify_stage6_manifest",
    "write_stage6_manifest",
]
