"""Fail-closed U74 -> child U75 planning-safety warm-start bridge."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final, Mapping

from lunar_exploration_ppo.utils.path_security import secure_read_bytes


WARM_START_SCHEMA: Final = "stage6_planning_u74_warm_start/v1"
WARM_START_MODE: Final = "child_lineage_warm_start_not_exact_resume/v1"
EFFECTIVE_CONFIG_SCHEMA: Final = "stage6_planning_effective_config/v2"
PARENT_RUN_ID: Final = "s6-standard-single-r1-20260718T220434Z"
PARENT_SEED: Final = 20260716
PARENT_UPDATE: Final = 74
PARENT_CHECKPOINT_SHA256: Final = (
    "ce9ea8cb047b5acc0ffc4f5d63084f3d54312cf55ecb0b92b8fc20bf43791553"
)
PARENT_MANIFEST_SHA256: Final = (
    "6c0e40d2dd9ea489f4c67a6a43868dc80751f07961f5db8ccf5c804213e340c5"
)
PARENT_COMPLETE_SHA256: Final = (
    "6cd6e6210283fc71a5205a727f82c07ff448458d4ac06ad8080277e2502960ef"
)
PARENT_POLICY_STATE_SHA256: Final = (
    "c1650b7bed6fa22387ac5b5fdb9d80a6aa0baf3a1d425e868d81586070c5db20"
)
PARENT_CONFIG_SHA256: Final = (
    "7b1c37673c105c34e7bd71d04d4f508d3a167cdf77426b0b35c7aff86ba8adae"
)
PARENT_LINEAGE_SHA256: Final = (
    "c7a0a7bf98b5aa4a39094ec01a2a0031d87ad8fc71060323474b20ebb134530c"
)
CHILD_FIRST_UPDATE: Final = 75
CHILD_FINAL_UPDATE: Final = 100
NEW_SEMANTICS_UPDATE_COUNT: Final = 26
PARENT_RECEIPT_KEY: Final = "0073:20260716:update:074"
PARENT_STAGE_ROOT: Final = Path(
    "D:/xunce/out/ppo_frontier/"
    f"{PARENT_RUN_ID}/s6"
)
PARENT_CHECKPOINT_DIRECTORY_RELATIVE: Final = (
    f"checkpoints/seed-{PARENT_SEED}/update-{PARENT_UPDATE:08d}"
)
PARENT_CHECKPOINT_MEMBER_NAMES: Final = tuple(
    sorted(("checkpoint.pt", "manifest.json", "complete.json"))
)
_PARENT_U74_INPUT_SPECS: Final = (
    (
        "planning-warm-start:parent:checkpoint",
        f"{PARENT_CHECKPOINT_DIRECTORY_RELATIVE}/checkpoint.pt",
    ),
    (
        "planning-warm-start:parent:manifest",
        f"{PARENT_CHECKPOINT_DIRECTORY_RELATIVE}/manifest.json",
    ),
    (
        "planning-warm-start:parent:complete",
        f"{PARENT_CHECKPOINT_DIRECTORY_RELATIVE}/complete.json",
    ),
    ("planning-warm-start:parent:config", "config.json"),
    ("planning-warm-start:parent:job-state", "job-state.jsonl"),
    (
        "planning-warm-start:parent:receipt-index",
        "checkpoints/index.jsonl",
    ),
    ("planning-warm-start:parent:lineage-audit", "lineage_audit.json"),
    ("planning-warm-start:parent:resource-audit", "resource_audit.jsonl"),
    (
        "planning-warm-start:parent:source-repair-ordinal1",
        "source-repair-amendment.json",
    ),
    (
        "planning-warm-start:parent:source-repair-ordinal2",
        "source-repair-continuation.json",
    ),
    (
        "planning-warm-start:parent:source-repair-ordinal3",
        "source-repair-supplement.json",
    ),
    (
        "planning-warm-start:parent:source-repair-ordinal4",
        "source-repair-closure.json",
    ),
    (
        "planning-warm-start:parent:source-repair-ordinal5",
        "source-repair-frontier-recovery.json",
    ),
    (
        "planning-warm-start:parent:source-repair-ordinal6",
        "source-repair-sensor-acceleration.json",
    ),
)
DISCARDED_U75_TRANSACTION_KEY: Final = "0074:20260716:update:075"
DISCARDED_U75_ATTEMPT: Final = 1
DISCARDED_U75_SEGMENT_INDEX: Final = 8
CHILD_BEST_VALIDATION_UPDATES: Final = (80, 90, 100)
CHILD_AUDIT_UPDATES: Final = (100,)
CHILD_RETENTION_PERIODIC_UPDATES: Final = (100,)
ACCEPTANCE_PROFILE_SCHEMA: Final = (
    "stage6_planning_warm_start_acceptance/v1"
)
UPDATE_SEMANTICS_STATEMENT: Final = (
    "74 parent-semantics updates + 26 child new-semantics updates"
)
DESIGN_RELATIVE_PATH: Final = (
    "docs/superpowers/specs/"
    "2026-07-23-ppo-stage6-planning-unknown-buffer-design-addendum.md"
)
PLAN_RELATIVE_PATH: Final = (
    "docs/superpowers/plans/"
    "2026-07-23-ppo-stage6-planning-unknown-buffer-warm-start.md"
)
IMPORTED_STATE_NAMES: Final = (
    "model_state_dict",
    "optimizer_state_dict",
    "normalization_stats",
    "rng_state",
    "scenario_sampler_state",
)
DISCARDED_STATE_NAMES: Final = (
    "vector_env_states",
    "best_record",
    "eval_metrics",
    "u75_attempt1_rollout",
    "u75_attempt1_candidate_snapshot",
    "u75_attempt1_old_logprob",
    "u75_attempt1_episode_state",
)


class Stage6PlanningWarmStartError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class PlanningWarmStartContext:
    child_run_id: str
    child_effective_config_sha256: str
    child_effective_config_bytes: bytes | None
    parent_run_id: str
    parent_config_sha256: str
    child_first_update: int
    child_final_update: int
    new_semantics_update_count: int
    imported_state_names: tuple[str, ...]
    discarded_state_names: tuple[str, ...]
    child_best_validation_updates: tuple[int, ...]
    artifact: Mapping[str, object]
    artifact_path: Path | None = None
    artifact_sha256: str | None = None

    def require_current(self, label: str) -> None:
        if self.artifact_path is None or self.artifact_sha256 is None:
            return
        try:
            current = _sha256(self.artifact_path)
        except OSError as exc:
            raise Stage6PlanningWarmStartError(
                f"planning warm-start artifact is unavailable at {label}"
            ) from exc
        if current != self.artifact_sha256:
            raise Stage6PlanningWarmStartError(
                f"planning warm-start artifact changed at {label}"
            )

    def checkpoint_lineage_for_update(
        self,
        update: int,
        *,
        warm_start_artifact_sha256: str,
    ) -> dict[str, object]:
        if (
            type(update) is not int
            or not self.child_first_update <= update <= self.child_final_update
        ):
            raise Stage6PlanningWarmStartError(
                "child checkpoint update is outside U75..U100"
            )
        _require_sha256(
            warm_start_artifact_sha256,
            "warm-start artifact",
        )
        return {
            "warm_start_schema_version": WARM_START_SCHEMA,
            "warm_start_artifact_sha256": warm_start_artifact_sha256,
            "warm_start_mode": WARM_START_MODE,
            "parent_run_id": PARENT_RUN_ID,
            "parent_seed": PARENT_SEED,
            "parent_update": PARENT_UPDATE,
            "parent_checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
            "parent_manifest_sha256": PARENT_MANIFEST_SHA256,
            "parent_complete_sha256": PARENT_COMPLETE_SHA256,
            "parent_policy_state_sha256": PARENT_POLICY_STATE_SHA256,
            "parent_config_sha256": PARENT_CONFIG_SHA256,
            "parent_lineage_sha256": PARENT_LINEAGE_SHA256,
            "child_run_id": self.child_run_id,
            "child_effective_config_sha256": (
                self.child_effective_config_sha256
            ),
            "child_first_update": CHILD_FIRST_UPDATE,
            "child_final_update": CHILD_FINAL_UPDATE,
            "new_semantics_update_count": NEW_SEMANTICS_UPDATE_COUNT,
            "checkpoint_update": update,
        }


@dataclass(frozen=True, slots=True)
class VerifiedParentU74:
    parent_update: int
    checkpoint_sha256: str
    manifest_sha256: str
    complete_sha256: str
    policy_state_sha256: str
    config_sha256: str
    lineage_sha256: str
    resource_accepted: bool
    u75_attempt1_discarded: bool
    checkpoint_payload: Mapping[str, Any]
    checkpoint_bytes: bytes = b""
    manifest_bytes: bytes = b""
    complete_bytes: bytes = b""
    checkpoint_directory: Path = Path(".")
    checkpoint_member_names: tuple[str, ...] = PARENT_CHECKPOINT_MEMBER_NAMES


@dataclass(frozen=True, slots=True)
class PlanningEffectiveConfigSnapshot:
    effective_config_bytes: bytes
    effective_config_sha256: str
    base_config_sha256: str
    base_config: Any
    payload: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class WarmStartRuntimeState:
    update_step: int
    policy_state_sha256: str
    normalization_stats: Mapping[str, object]
    scenario_sampler_state: Mapping[str, object]
    model_state_restored: bool
    optimizer_state_restored: bool
    rng_state_restored: bool
    parent_vector_env_states_discarded: bool
    parent_best_record_discarded: bool
    parent_eval_metrics_discarded: bool


def _require_sha256(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Stage6PlanningWarmStartError(f"{label} SHA-256 is invalid")
    return value


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def planning_effective_config_bytes(base_config_bytes: bytes) -> bytes:
    if not isinstance(base_config_bytes, bytes) or not base_config_bytes:
        raise Stage6PlanningWarmStartError("base config bytes are invalid")
    try:
        base_config_utf8 = base_config_bytes.decode("utf-8")
        base_value = json.loads(base_config_utf8)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Stage6PlanningWarmStartError("base config is invalid") from exc
    if not isinstance(base_value, Mapping):
        raise Stage6PlanningWarmStartError("base config mapping is invalid")
    from lunar_exploration_ppo.configs.stage6 import parse_stage6_config_bytes

    try:
        config = parse_stage6_config_bytes(base_config_bytes)
    except ValueError as exc:
        raise Stage6PlanningWarmStartError("base config is invalid") from exc
    base_config_sha256 = hashlib.sha256(base_config_bytes).hexdigest()
    if base_config_sha256 != PARENT_CONFIG_SHA256:
        raise Stage6PlanningWarmStartError("parent base config SHA drifted")
    if (
        config.planning_unknown_buffer_m != 0.75
        or config.reset_local_safety_scan_range_m != 0.75
        or config.reset_local_safety_scan_fov_deg != 360.0
        or config.reset_local_safety_scan_ray_angle_step_deg != 1.0
    ):
        raise Stage6PlanningWarmStartError(
            "planning safety config semantics drifted"
        )
    payload = {
        "schema_version": EFFECTIVE_CONFIG_SCHEMA,
        "base_config_bytes_utf8": base_config_utf8,
        "base_config_sha256": base_config_sha256,
        "planning_safety": {
            "planning_unknown_buffer_m": 0.75,
            "reset_scan_order": [
                "reset_local_safety",
                "reset_exploration",
            ],
            "reset_local_safety_scan": {
                "range_m": 0.75,
                "fov_deg": 360.0,
                "ray_angle_step_deg": 1.0,
            },
        },
        "warm_start": {
            "schema_version": WARM_START_SCHEMA,
            "mode": WARM_START_MODE,
            "parent_run_id": PARENT_RUN_ID,
            "parent_seed": PARENT_SEED,
            "parent_update": PARENT_UPDATE,
            "parent_config_sha256": PARENT_CONFIG_SHA256,
            "child_first_update": CHILD_FIRST_UPDATE,
            "child_final_update": CHILD_FINAL_UPDATE,
            "child_updates": list(
                range(CHILD_FIRST_UPDATE, CHILD_FINAL_UPDATE + 1)
            ),
            "new_semantics_update_count": NEW_SEMANTICS_UPDATE_COUNT,
            "acceptance_profile": build_planning_child_acceptance_profile(),
        },
    }
    result = _canonical_json_bytes(payload)
    if hashlib.sha256(result).hexdigest() == PARENT_CONFIG_SHA256:
        raise Stage6PlanningWarmStartError(
            "child effective config aliases parent config"
        )
    return result


def planning_effective_config_sha256(base_config_bytes: bytes) -> str:
    return hashlib.sha256(
        planning_effective_config_bytes(base_config_bytes)
    ).hexdigest()


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise Stage6PlanningWarmStartError(f"{label} mapping is invalid")
    return value


def planning_warm_start_expected_bindings(
    *,
    repo_root: str | Path,
    review_authorization_handle: object,
) -> dict[str, str]:
    repo = Path(repo_root).expanduser().resolve()
    canonical_record = getattr(
        review_authorization_handle,
        "canonical_record",
        None,
    )
    if not callable(canonical_record):
        raise Stage6PlanningWarmStartError(
            "review authorization handle is invalid"
        )
    try:
        record = canonical_record()
    except Exception as exc:
        raise Stage6PlanningWarmStartError(
            "review authorization record is unavailable"
        ) from exc
    if not isinstance(record, Mapping):
        raise Stage6PlanningWarmStartError(
            "review authorization record mapping is invalid"
        )
    return planning_warm_start_expected_bindings_from_record(
        repo_root=repo_root,
        review_authorization_record=record,
    )


def planning_warm_start_expected_bindings_from_record(
    *,
    repo_root: str | Path,
    review_authorization_record: Mapping[str, object],
) -> dict[str, str]:
    repo = Path(repo_root).expanduser().resolve()
    if not isinstance(review_authorization_record, Mapping):
        raise Stage6PlanningWarmStartError(
            "review authorization record mapping is invalid"
        )
    paths = {
        "design_sha256": repo / DESIGN_RELATIVE_PATH,
        "plan_sha256": repo / PLAN_RELATIVE_PATH,
    }
    result: dict[str, str] = {}
    for field, path in paths.items():
        try:
            path.resolve().relative_to(repo)
            payload = path.read_bytes()
        except (OSError, ValueError) as exc:
            raise Stage6PlanningWarmStartError(
                f"planning {field} evidence is unavailable"
            ) from exc
        result[field] = hashlib.sha256(payload).hexdigest()
    authorization_fields = {
        "source_set_sha256": "source_set_sha256",
        "spec_review_sha256": "spec_review_sha256",
        "quality_review_sha256": "quality_review_sha256",
        "launch_authorization_sha256": "authorization_file_sha256",
    }
    for field, record_field in authorization_fields.items():
        result[field] = _require_sha256(
            review_authorization_record.get(record_field),
            f"review authorization {record_field}",
        )
    return result


def _exact_mapping(
    value: object,
    fields: set[str],
    label: str,
) -> Mapping[str, object]:
    result = _mapping(value, label)
    if set(result) != fields:
        raise Stage6PlanningWarmStartError(f"{label} field set drifted")
    return result


def parse_planning_effective_config_bytes(
    payload: bytes,
) -> PlanningEffectiveConfigSnapshot:
    if type(payload) is not bytes or not payload:
        raise Stage6PlanningWarmStartError(
            "planning effective config bytes are invalid"
        )
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Stage6PlanningWarmStartError(
            "planning effective config is invalid"
        ) from exc
    root = _exact_mapping(
        value,
        {
            "schema_version",
            "base_config_bytes_utf8",
            "base_config_sha256",
            "planning_safety",
            "warm_start",
        },
        "planning effective config",
    )
    if _canonical_json_bytes(dict(root)) != payload:
        raise Stage6PlanningWarmStartError(
            "planning effective config is not canonical"
        )
    if root["schema_version"] != EFFECTIVE_CONFIG_SCHEMA:
        raise Stage6PlanningWarmStartError(
            "planning effective config schema drifted"
        )
    base_config_utf8 = root["base_config_bytes_utf8"]
    if not isinstance(base_config_utf8, str):
        raise Stage6PlanningWarmStartError(
            "planning effective base config bytes drifted"
        )
    base_config_bytes = base_config_utf8.encode("utf-8")
    base_config_sha256 = hashlib.sha256(base_config_bytes).hexdigest()
    if (
        root["base_config_sha256"] != base_config_sha256
        or base_config_sha256 != PARENT_CONFIG_SHA256
    ):
        raise Stage6PlanningWarmStartError(
            "planning effective base config SHA drifted"
        )
    planning_safety = _exact_mapping(
        root["planning_safety"],
        {
            "planning_unknown_buffer_m",
            "reset_scan_order",
            "reset_local_safety_scan",
        },
        "planning safety",
    )
    reset_scan = _exact_mapping(
        planning_safety["reset_local_safety_scan"],
        {"range_m", "fov_deg", "ray_angle_step_deg"},
        "reset local safety scan",
    )
    if (
        planning_safety["planning_unknown_buffer_m"] != 0.75
        or planning_safety["reset_scan_order"]
        != ["reset_local_safety", "reset_exploration"]
        or dict(reset_scan)
        != {
            "range_m": 0.75,
            "fov_deg": 360.0,
            "ray_angle_step_deg": 1.0,
        }
    ):
        raise Stage6PlanningWarmStartError(
            "planning effective safety semantics drifted"
        )
    warm_start = _mapping(root["warm_start"], "planning warm-start config")
    if dict(warm_start) != {
        "schema_version": WARM_START_SCHEMA,
        "mode": WARM_START_MODE,
        "parent_run_id": PARENT_RUN_ID,
        "parent_seed": PARENT_SEED,
        "parent_update": PARENT_UPDATE,
        "parent_config_sha256": PARENT_CONFIG_SHA256,
        "child_first_update": CHILD_FIRST_UPDATE,
        "child_final_update": CHILD_FINAL_UPDATE,
        "child_updates": list(
            range(CHILD_FIRST_UPDATE, CHILD_FINAL_UPDATE + 1)
        ),
        "new_semantics_update_count": NEW_SEMANTICS_UPDATE_COUNT,
        "acceptance_profile": build_planning_child_acceptance_profile(),
    }:
        raise Stage6PlanningWarmStartError(
            "planning effective warm-start semantics drifted"
        )
    from lunar_exploration_ppo.configs.stage6 import parse_stage6_config_bytes

    try:
        config = parse_stage6_config_bytes(base_config_bytes)
    except ValueError as exc:
        raise Stage6PlanningWarmStartError(
            "planning effective base config is invalid"
        ) from exc
    if (
        config.planning_unknown_buffer_m != 0.75
        or config.reset_local_safety_scan_range_m != 0.75
        or config.reset_local_safety_scan_fov_deg != 360.0
        or config.reset_local_safety_scan_ray_angle_step_deg != 1.0
    ):
        raise Stage6PlanningWarmStartError(
            "planning effective base config semantics drifted"
        )
    return PlanningEffectiveConfigSnapshot(
        effective_config_bytes=payload,
        effective_config_sha256=hashlib.sha256(payload).hexdigest(),
        base_config_sha256=base_config_sha256,
        base_config=config,
        payload=MappingProxyType(dict(root)),
    )


def validate_planning_warm_start_artifact(
    value: Mapping[str, object],
    *,
    expected_child_run_id: str,
    expected_child_config_sha256: str | None = None,
    expected_child_config_bytes: bytes | None = None,
    expected_bindings: Mapping[str, object] | None = None,
) -> PlanningWarmStartContext:
    root = _exact_mapping(
        value,
        {
            "schema_version",
            "mode",
            "parent",
            "discarded_u75_attempt1",
            "child",
            "state_transfer",
            "bindings",
        },
        "warm-start artifact",
    )
    if root["schema_version"] != WARM_START_SCHEMA:
        raise Stage6PlanningWarmStartError("warm-start schema drifted")
    if root["mode"] != WARM_START_MODE:
        raise Stage6PlanningWarmStartError("warm-start mode drifted")
    parent = _exact_mapping(
        root["parent"],
        {
            "run_id",
            "seed",
            "update",
            "checkpoint_sha256",
            "manifest_sha256",
            "complete_sha256",
            "policy_state_sha256",
            "config_sha256",
            "lineage_sha256",
            "resource_accepted",
            "receipt_transaction_key",
            "source_repair_last_ordinal",
        },
        "parent",
    )
    expected_parent = {
        "run_id": PARENT_RUN_ID,
        "seed": PARENT_SEED,
        "update": PARENT_UPDATE,
        "checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
        "manifest_sha256": PARENT_MANIFEST_SHA256,
        "complete_sha256": PARENT_COMPLETE_SHA256,
        "policy_state_sha256": PARENT_POLICY_STATE_SHA256,
        "config_sha256": PARENT_CONFIG_SHA256,
        "lineage_sha256": PARENT_LINEAGE_SHA256,
        "resource_accepted": True,
        "receipt_transaction_key": PARENT_RECEIPT_KEY,
        "source_repair_last_ordinal": 6,
    }
    for field, expected in expected_parent.items():
        if parent.get(field) != expected:
            label = "accepted" if field == "resource_accepted" else field
            raise Stage6PlanningWarmStartError(
                f"parent {label} binding drifted"
            )
    discarded = _exact_mapping(
        root["discarded_u75_attempt1"],
        {
            "transaction_key",
            "attempt",
            "segment_index",
            "only_phase",
            "accepted",
            "discarded",
        },
        "discarded U75 attempt1",
    )
    if (
        discarded["transaction_key"] != DISCARDED_U75_TRANSACTION_KEY
        or discarded["attempt"] != DISCARDED_U75_ATTEMPT
        or discarded["segment_index"] != DISCARDED_U75_SEGMENT_INDEX
        or discarded["only_phase"] != "pre"
        or discarded["accepted"] is not False
        or discarded["discarded"] is not True
    ):
        if discarded["only_phase"] != "pre":
            reason = "U75 attempt1 must contain pre only"
        else:
            reason = "U75 attempt1 discard binding drifted"
        raise Stage6PlanningWarmStartError(reason)
    child = _exact_mapping(
        root["child"],
        {
            "run_id",
            "seed",
            "first_update",
            "final_update",
            "new_semantics_update_count",
            "effective_config_sha256",
        },
        "child",
    )
    if expected_child_config_bytes is not None:
        effective_snapshot = parse_planning_effective_config_bytes(
            expected_child_config_bytes
        )
        expected_config = effective_snapshot.effective_config_sha256
        if (
            expected_child_config_sha256 is not None
            and expected_child_config_sha256 != expected_config
        ):
            raise Stage6PlanningWarmStartError(
                "expected child config bytes/SHA drifted"
            )
    else:
        effective_snapshot = None
        expected_config = _require_sha256(
            expected_child_config_sha256,
            "expected child config",
        )
    if (
        child["run_id"] != expected_child_run_id
        or not isinstance(child["run_id"], str)
        or child["run_id"] == PARENT_RUN_ID
    ):
        raise Stage6PlanningWarmStartError("child run binding drifted")
    if child["seed"] != PARENT_SEED:
        raise Stage6PlanningWarmStartError("child seed binding drifted")
    if child["first_update"] != CHILD_FIRST_UPDATE:
        raise Stage6PlanningWarmStartError("child first update drifted")
    if (
        child["final_update"] != CHILD_FINAL_UPDATE
        or child["new_semantics_update_count"] != NEW_SEMANTICS_UPDATE_COUNT
    ):
        raise Stage6PlanningWarmStartError("child update range drifted")
    if (
        child["effective_config_sha256"] != expected_config
        or expected_config == PARENT_CONFIG_SHA256
    ):
        raise Stage6PlanningWarmStartError("child config binding drifted")
    transfer = _exact_mapping(
        root["state_transfer"],
        {
            "imported",
            "discarded",
            "child_vector_env_reset_count",
            "child_best_validation_updates",
        },
        "state transfer",
    )
    if tuple(transfer["imported"]) != IMPORTED_STATE_NAMES:
        raise Stage6PlanningWarmStartError("imported state set drifted")
    if tuple(transfer["discarded"]) != DISCARDED_STATE_NAMES:
        raise Stage6PlanningWarmStartError("discarded state set drifted")
    if transfer["child_vector_env_reset_count"] != 8:
        raise Stage6PlanningWarmStartError("child vector env reset count drifted")
    if tuple(transfer["child_best_validation_updates"]) != (
        CHILD_BEST_VALIDATION_UPDATES
    ):
        raise Stage6PlanningWarmStartError("child best validation binding drifted")
    bindings = _exact_mapping(
        root["bindings"],
        {
            "design_sha256",
            "plan_sha256",
            "source_set_sha256",
            "spec_review_sha256",
            "quality_review_sha256",
            "launch_authorization_sha256",
        },
        "warm-start bindings",
    )
    for field, candidate in bindings.items():
        _require_sha256(candidate, field)
    if expected_bindings is not None:
        if (
            not isinstance(expected_bindings, Mapping)
            or set(expected_bindings) != set(bindings)
            or any(
                _require_sha256(
                    expected_bindings.get(field),
                    f"expected {field}",
                )
                != bindings[field]
                for field in bindings
            )
        ):
            raise Stage6PlanningWarmStartError(
                "warm-start evidence binding drifted"
            )
    return PlanningWarmStartContext(
        child_run_id=str(child["run_id"]),
        child_effective_config_sha256=expected_config,
        child_effective_config_bytes=(
            effective_snapshot.effective_config_bytes
            if effective_snapshot is not None
            else None
        ),
        parent_run_id=PARENT_RUN_ID,
        parent_config_sha256=PARENT_CONFIG_SHA256,
        child_first_update=CHILD_FIRST_UPDATE,
        child_final_update=CHILD_FINAL_UPDATE,
        new_semantics_update_count=NEW_SEMANTICS_UPDATE_COUNT,
        imported_state_names=IMPORTED_STATE_NAMES,
        discarded_state_names=DISCARDED_STATE_NAMES,
        child_best_validation_updates=CHILD_BEST_VALIDATION_UPDATES,
        artifact=MappingProxyType(dict(root)),
    )


def select_parent_warm_start_state(
    checkpoint_payload: Mapping[str, object],
) -> dict[str, object]:
    if any(name not in checkpoint_payload for name in IMPORTED_STATE_NAMES):
        raise Stage6PlanningWarmStartError(
            "parent checkpoint lacks approved warm-start state"
        )
    for discarded in ("vector_env_states", "best_record", "eval_metrics"):
        if discarded not in checkpoint_payload:
            raise Stage6PlanningWarmStartError(
                f"parent checkpoint lacks explicit discarded {discarded}"
            )
    sampler = checkpoint_payload["scenario_sampler_state"]
    if (
        not isinstance(sampler, Mapping)
        or set(sampler) != {"workers"}
        or not isinstance(sampler["workers"], list)
        or len(sampler["workers"]) != 8
        or any(not isinstance(state, Mapping) for state in sampler["workers"])
    ):
        raise Stage6PlanningWarmStartError(
            "parent scenario sampler state drifted"
        )
    return {name: checkpoint_payload[name] for name in IMPORTED_STATE_NAMES}


def build_planning_child_transactions(config: object) -> tuple[object, ...]:
    from lunar_exploration_ppo.ppo.standard_training import (
        build_standard_training_transactions,
    )

    transactions = build_standard_training_transactions(config)  # type: ignore[arg-type]
    selected = tuple(
        transaction
        for transaction in transactions
        if CHILD_FIRST_UPDATE <= transaction.update <= CHILD_FINAL_UPDATE
    )
    result = tuple(
        replace(transaction, sequence=sequence)
        for sequence, transaction in enumerate(selected)
    )
    if (
        len(result) != NEW_SEMANTICS_UPDATE_COUNT
        or tuple(transaction.sequence for transaction in result)
        != tuple(range(NEW_SEMANTICS_UPDATE_COUNT))
        or result[0].key == DISCARDED_U75_TRANSACTION_KEY
        or result[0].key != "0000:20260716:update:075"
        or result[-1].key != "0025:20260716:update:100"
        or tuple(
            transaction.update
            for transaction in result
            if transaction.validation_episodes
        )
        != CHILD_BEST_VALIDATION_UPDATES
    ):
        raise Stage6PlanningWarmStartError(
            "planning child transaction schedule drifted"
        )
    return result


def build_planning_child_acceptance_profile() -> dict[str, object]:
    return {
        "schema_version": ACCEPTANCE_PROFILE_SCHEMA,
        "parent_semantics_update_count": PARENT_UPDATE,
        "child_new_semantics_update_count": NEW_SEMANTICS_UPDATE_COUNT,
        "child_receipt_count": NEW_SEMANTICS_UPDATE_COUNT,
        "child_first_update": CHILD_FIRST_UPDATE,
        "child_final_update": CHILD_FINAL_UPDATE,
        "child_updates": list(range(CHILD_FIRST_UPDATE, CHILD_FINAL_UPDATE + 1)),
        "child_validation_updates": list(CHILD_BEST_VALIDATION_UPDATES),
        "child_audit_updates": list(CHILD_AUDIT_UPDATES),
        "child_retention_periodic_updates": list(
            CHILD_RETENTION_PERIODIC_UPDATES
        ),
        "update_semantics": UPDATE_SEMANTICS_STATEMENT,
    }


def validate_planning_child_acceptance_profile(
    value: Mapping[str, object],
) -> dict[str, object]:
    expected = build_planning_child_acceptance_profile()
    if not isinstance(value, Mapping) or dict(value) != expected:
        raise Stage6PlanningWarmStartError(
            "planning child acceptance profile drifted"
        )
    return expected


def verify_planning_child_terminal_replay(
    *,
    config: object,
    receipt_rows: object,
    journal_rows: object,
    immutable_bindings: Mapping[str, object],
    validation_transaction_keys: object,
    audit_transaction_keys: object,
    acceptance_profile: Mapping[str, object],
) -> dict[str, object]:
    from lunar_exploration_ppo.ppo.standard_training import (
        CheckpointReceiptIndex,
        StandardTrainingError,
        verify_journal_checkpoint_bindings,
    )

    try:
        profile = validate_planning_child_acceptance_profile(
            acceptance_profile
        )
        transactions = build_planning_child_transactions(config)
        receipts = CheckpointReceiptIndex._validate_rows(
            tuple(receipt_rows)  # type: ignore[arg-type]
        )
        rows = tuple(journal_rows)  # type: ignore[arg-type]
        expected_keys = tuple(transaction.key for transaction in transactions)
        expected_validations = tuple(
            transaction.key
            for transaction in transactions
            if transaction.update in CHILD_BEST_VALIDATION_UPDATES
        )
        expected_audits = tuple(
            transaction.key
            for transaction in transactions
            if transaction.update in CHILD_AUDIT_UPDATES
        )
        if (
            len(receipts) != NEW_SEMANTICS_UPDATE_COUNT
            or tuple(row["transaction_key"] for row in receipts)
            != expected_keys
            or tuple(row["update"] for row in receipts)
            != tuple(range(CHILD_FIRST_UPDATE, CHILD_FINAL_UPDATE + 1))
            or tuple(validation_transaction_keys) != expected_validations  # type: ignore[arg-type]
            or tuple(audit_transaction_keys) != expected_audits  # type: ignore[arg-type]
        ):
            raise Stage6PlanningWarmStartError(
                "planning child terminal schedule drifted"
            )
        completed = verify_journal_checkpoint_bindings(
            rows,
            transactions,
            receipts,
            immutable_bindings,
        )
        if completed != expected_keys:
            raise Stage6PlanningWarmStartError(
                "planning child terminal journal is incomplete"
            )
    except Stage6PlanningWarmStartError:
        raise
    except (KeyError, TypeError, ValueError, StandardTrainingError) as exc:
        raise Stage6PlanningWarmStartError(
            "planning child terminal replay failed"
        ) from exc
    return {
        "schema_version": "stage6_planning_child_terminal_replay/v1",
        "passed": True,
        "parent_semantics_update_count": profile[
            "parent_semantics_update_count"
        ],
        "child_new_semantics_update_count": profile[
            "child_new_semantics_update_count"
        ],
        "first_child_update": CHILD_FIRST_UPDATE,
        "last_child_update": CHILD_FINAL_UPDATE,
        "receipt_count": len(receipts),
        "validation_updates": list(CHILD_BEST_VALIDATION_UPDATES),
        "audit_updates": list(CHILD_AUDIT_UPDATES),
        "update_semantics": UPDATE_SEMANTICS_STATEMENT,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parent_u74_input_pin_requests(
    parent_stage_root: str | Path = PARENT_STAGE_ROOT,
) -> tuple[tuple[str, Path], ...]:
    """Return the exact frozen parent evidence set required by warm start."""

    stage = Path(parent_stage_root).expanduser().resolve()
    return tuple(
        (label, stage / relative)
        for label, relative in _PARENT_U74_INPUT_SPECS
    )


def _capture_parent_u74_inputs(
    stage: Path,
    *,
    input_pin: object | None,
) -> dict[str, bytes]:
    snapshots: dict[str, bytes] = {}
    for label, path in parent_u74_input_pin_requests(stage):
        try:
            if input_pin is None:
                payload = secure_read_bytes(
                    path,
                    base=stage,
                    label=label,
                ).payload
            else:
                read_bytes = getattr(input_pin, "read_bytes", None)
                if not callable(read_bytes):
                    raise TypeError("input pin has no read_bytes capability")
                payload = read_bytes(path, label=label)
        except Exception as exc:
            raise Stage6PlanningWarmStartError(
                f"{label} snapshot capture failed"
            ) from exc
        if type(payload) is not bytes or not payload:
            raise Stage6PlanningWarmStartError(
                f"{label} snapshot is empty"
            )
        snapshots[label] = payload
    return snapshots


def _canonical_json(path: Path, label: str) -> Mapping[str, object]:
    try:
        payload = path.read_bytes()
        value = json.loads(payload.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Stage6PlanningWarmStartError(f"{label} is unreadable") from exc
    canonical = (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    if payload != canonical or not isinstance(value, Mapping):
        raise Stage6PlanningWarmStartError(f"{label} is not canonical")
    return value


def _verify_historical_source_repair_chain(
    checkpoint_lineage: Mapping[str, object],
    source_repair_snapshots: Mapping[str, bytes],
) -> None:
    ordinal6_relative = "source-repair-sensor-acceleration.json"
    ordinal6_bytes = source_repair_snapshots.get(ordinal6_relative)
    expected_ordinal6_sha = checkpoint_lineage.get(
        "source_repair_sensor_acceleration_sha256"
    )
    if (
        checkpoint_lineage.get("source_repair_sensor_acceleration_ordinal") != 6
        or not isinstance(expected_ordinal6_sha, str)
        or type(ordinal6_bytes) is not bytes
        or hashlib.sha256(ordinal6_bytes).hexdigest()
        != expected_ordinal6_sha
    ):
        raise Stage6PlanningWarmStartError(
            "parent source-repair ordinal6 binding drifted"
        )
    value = _canonical_pretty_json_bytes(
        ordinal6_bytes,
        "parent source-repair ordinal6",
    )
    chain = value.get("repair_chain")
    if (
        value.get("repair_ordinal") != 6
        or value.get("formal_run_id") != PARENT_RUN_ID
        or not isinstance(chain, list)
        or len(chain) != 5
    ):
        raise Stage6PlanningWarmStartError(
            "parent source-repair history chain drifted"
        )
    for ordinal, row in enumerate(chain, start=1):
        if not isinstance(row, Mapping) or row.get("repair_ordinal") != ordinal:
            raise Stage6PlanningWarmStartError(
                "parent source-repair history ordinal drifted"
            )
        artifact = _mapping(row.get("artifact"), "source-repair artifact")
        relative = artifact.get("path")
        expected_sha = artifact.get("sha256")
        expected_size = artifact.get("size_bytes")
        if (
            not isinstance(relative, str)
            or not isinstance(expected_sha, str)
            or type(expected_size) is not int
        ):
            raise Stage6PlanningWarmStartError(
                "parent source-repair artifact binding drifted"
            )
        payload = source_repair_snapshots.get(relative)
        if (
            type(payload) is not bytes
            or len(payload) != expected_size
            or hashlib.sha256(payload).hexdigest() != expected_sha
        ):
            raise Stage6PlanningWarmStartError(
                "parent source-repair historical artifact drifted"
            )


def _canonical_pretty_json_bytes(
    payload: bytes,
    label: str,
) -> Mapping[str, object]:
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Stage6PlanningWarmStartError(f"{label} is invalid") from exc
    canonical = (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    if payload != canonical or not isinstance(value, Mapping):
        raise Stage6PlanningWarmStartError(f"{label} is not canonical")
    return value


def _canonical_compact_jsonl_rows(
    payload: bytes,
    label: str,
) -> tuple[dict[str, object], ...]:
    if type(payload) is not bytes or not payload:
        raise Stage6PlanningWarmStartError(f"{label} is empty")
    rows: list[dict[str, object]] = []
    try:
        for line in payload.splitlines(keepends=True):
            value = json.loads(line.decode("utf-8"))
            canonical = _canonical_json_bytes(value)
            if line != canonical or not isinstance(value, dict):
                raise ValueError("noncanonical JSONL row")
            rows.append(value)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise Stage6PlanningWarmStartError(f"{label} is invalid") from exc
    return tuple(rows)


def verify_parent_u74_journal_snapshots(
    *,
    config_bytes: bytes,
    job_state_bytes: bytes,
    receipt_index_bytes: bytes,
    lineage_audit_bytes: bytes,
    source_repair_ordinal6_bytes: bytes,
) -> dict[str, object]:
    from lunar_exploration_ppo.configs.stage6 import parse_stage6_config_bytes
    from lunar_exploration_ppo.ppo.standard_training import (
        CheckpointReceiptIndex,
        StandardTrainingError,
        build_standard_training_transactions,
        verify_journal_checkpoint_bindings,
    )
    from lunar_exploration_ppo.workflows.stage6 import (
        Stage6StateJournal,
        Stage6WorkflowError,
    )

    try:
        if (
            type(config_bytes) is not bytes
            or hashlib.sha256(config_bytes).hexdigest()
            != PARENT_CONFIG_SHA256
        ):
            raise Stage6PlanningWarmStartError(
                "parent U74 journal config drifted"
            )
        config = parse_stage6_config_bytes(config_bytes)
        transactions = build_standard_training_transactions(config)
        journal_rows = Stage6StateJournal.verify_snapshot_bytes(
            job_state_bytes
        )
        receipt_rows = CheckpointReceiptIndex._validate_rows(
            _canonical_compact_jsonl_rows(
                receipt_index_bytes,
                "parent U74 receipt index",
            )
        )
        lineage_audit = _canonical_pretty_json_bytes(
            lineage_audit_bytes,
            "parent U74 lineage audit",
        )
        ordinal6 = _canonical_pretty_json_bytes(
            source_repair_ordinal6_bytes,
            "parent U74 source-repair ordinal6",
        )
        historical = _mapping(
            lineage_audit.get("immutable_bindings"),
            "parent U74 historical immutable bindings",
        )
        current = _mapping(
            _mapping(
                ordinal6.get("current"),
                "parent U74 ordinal6 current",
            ).get("immutable_bindings"),
            "parent U74 current immutable bindings",
        )
        cutover = _mapping(
            ordinal6.get("cutover"),
            "parent U74 ordinal6 cutover",
        )
        if (
            lineage_audit.get("schema_version") != "stage6_lineage_audit/v2"
            or ordinal6.get("repair_ordinal") != 6
            or ordinal6.get("formal_run_id") != PARENT_RUN_ID
            or cutover.get("next_transaction_key")
            != "0049:20260716:update:050"
            or historical.get("config_sha256") != PARENT_CONFIG_SHA256
            or current.get("config_sha256") != PARENT_CONFIG_SHA256
        ):
            raise Stage6PlanningWarmStartError(
                "parent U74 journal immutable lineage drifted"
            )
        completed = verify_journal_checkpoint_bindings(
            journal_rows,
            transactions,
            receipt_rows,
            current,
            historical_immutable_bindings=historical,
            current_binding_first_transaction_key=(
                "0049:20260716:update:050"
            ),
        )
        expected_completed = tuple(
            transaction.key for transaction in transactions[:PARENT_UPDATE]
        )
        if (
            completed != expected_completed
            or len(receipt_rows) != PARENT_UPDATE
            or tuple(
                row["transaction_key"] for row in receipt_rows
            )
            != expected_completed
            or receipt_rows[-1].get("transaction_key")
            != PARENT_RECEIPT_KEY
            or receipt_rows[-1].get("checkpoint_sha256")
            != PARENT_CHECKPOINT_SHA256
            or any(
                row.get("state")
                == f"seed_{PARENT_SEED}_training_update_75"
                for row in journal_rows
            )
        ):
            raise Stage6PlanningWarmStartError(
                "parent U74 journal is not the exact U1-U74 prefix"
            )
    except Stage6PlanningWarmStartError:
        raise
    except (
        KeyError,
        TypeError,
        ValueError,
        StandardTrainingError,
        Stage6WorkflowError,
    ) as exc:
        raise Stage6PlanningWarmStartError(
            "parent U74 journal replay failed"
        ) from exc
    return {
        "schema_version": "stage6_parent_u74_journal_replay/v1",
        "completed_transaction_count": len(completed),
        "receipt_count": len(receipt_rows),
        "last_transaction_key": completed[-1],
        "next_update": PARENT_UPDATE + 1,
    }


def verify_parent_u74_bundle(
    parent_stage_root: str | Path,
    *,
    input_pin: object | None = None,
) -> VerifiedParentU74:
    stage = Path(parent_stage_root).expanduser().resolve()
    if (
        not stage.is_dir()
        or stage.name != "s6"
        or stage.parent.name != PARENT_RUN_ID
    ):
        raise Stage6PlanningWarmStartError("parent run root drifted")
    directory = stage / PARENT_CHECKPOINT_DIRECTORY_RELATIVE
    snapshots = _capture_parent_u74_inputs(stage, input_pin=input_pin)
    by_relative = {
        relative: snapshots[label]
        for label, relative in _PARENT_U74_INPUT_SPECS
    }
    checkpoint_bytes = by_relative[
        f"{PARENT_CHECKPOINT_DIRECTORY_RELATIVE}/checkpoint.pt"
    ]
    manifest_bytes = by_relative[
        f"{PARENT_CHECKPOINT_DIRECTORY_RELATIVE}/manifest.json"
    ]
    complete_bytes = by_relative[
        f"{PARENT_CHECKPOINT_DIRECTORY_RELATIVE}/complete.json"
    ]
    if (
        hashlib.sha256(checkpoint_bytes).hexdigest()
        != PARENT_CHECKPOINT_SHA256
    ):
        raise Stage6PlanningWarmStartError("parent checkpoint SHA drifted")
    if hashlib.sha256(manifest_bytes).hexdigest() != PARENT_MANIFEST_SHA256:
        raise Stage6PlanningWarmStartError("parent manifest SHA drifted")
    if hashlib.sha256(complete_bytes).hexdigest() != PARENT_COMPLETE_SHA256:
        raise Stage6PlanningWarmStartError("parent complete marker SHA drifted")

    from lunar_exploration_ppo.configs.stage6 import SafetyContract
    from lunar_exploration_ppo.ppo.checkpoint import (
        inspect_complete_checkpoint_snapshot,
        safe_load_checkpoint_payload,
    )

    try:
        payload = safe_load_checkpoint_payload(checkpoint_bytes)
    except Exception as exc:
        raise Stage6PlanningWarmStartError(
            "parent checkpoint payload is unreadable"
        ) from exc
    if not isinstance(payload, Mapping):
        raise Stage6PlanningWarmStartError(
            "parent checkpoint payload mapping drifted"
        )
    lineage = _mapping(payload.get("lineage"), "parent lineage")
    lineage_sha = hashlib.sha256(
        (
            json.dumps(
                lineage,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
    ).hexdigest()
    if (
        payload.get("update_step") != PARENT_UPDATE
        or payload.get("config_sha256") != PARENT_CONFIG_SHA256
        or payload.get("policy_state_sha256") != PARENT_POLICY_STATE_SHA256
        or lineage_sha != PARENT_LINEAGE_SHA256
    ):
        raise Stage6PlanningWarmStartError(
            "parent checkpoint payload identity drifted"
        )
    safety = SafetyContract.from_dict(
        _mapping(payload.get("safety_contract"), "parent safety")
    )
    try:
        receipt = inspect_complete_checkpoint_snapshot(
            directory=directory,
            update_step=PARENT_UPDATE,
            schema_version="stage6_complete_checkpoint/v1",
            member_names=PARENT_CHECKPOINT_MEMBER_NAMES,
            checkpoint_bytes=checkpoint_bytes,
            manifest_bytes=manifest_bytes,
            complete_bytes=complete_bytes,
            expected_config_sha256=PARENT_CONFIG_SHA256,
            expected_lineage=lineage,
            expected_safety_contract=safety,
        )
    except Exception as exc:
        raise Stage6PlanningWarmStartError(
            "parent checkpoint snapshot validation failed"
        ) from exc
    if (
        receipt.checkpoint_sha256 != PARENT_CHECKPOINT_SHA256
        or receipt.manifest_sha256 != PARENT_MANIFEST_SHA256
        or receipt.policy_state_sha256 != PARENT_POLICY_STATE_SHA256
    ):
        raise Stage6PlanningWarmStartError(
            "parent checkpoint snapshot receipt drifted"
        )
    select_parent_warm_start_state(payload)
    _verify_historical_source_repair_chain(
        lineage,
        {
            relative: by_relative[relative]
            for _label, relative in _PARENT_U74_INPUT_SPECS
            if relative.startswith("source-repair-")
        },
    )
    verify_parent_u74_journal_snapshots(
        config_bytes=by_relative["config.json"],
        job_state_bytes=by_relative["job-state.jsonl"],
        receipt_index_bytes=by_relative["checkpoints/index.jsonl"],
        lineage_audit_bytes=by_relative["lineage_audit.json"],
        source_repair_ordinal6_bytes=by_relative[
            "source-repair-sensor-acceleration.json"
        ],
    )

    from lunar_exploration_ppo.ppo.standard_training import (
        CheckpointReceiptIndex,
    )

    receipts = CheckpointReceiptIndex._validate_rows(
        _canonical_compact_jsonl_rows(
            by_relative["checkpoints/index.jsonl"],
            "parent U74 receipt index",
        )
    )
    if (
        len(receipts) != PARENT_UPDATE
        or receipts[-1].get("transaction_key") != PARENT_RECEIPT_KEY
        or receipts[-1].get("update") != PARENT_UPDATE
        or receipts[-1].get("checkpoint_sha256")
        != PARENT_CHECKPOINT_SHA256
        or receipts[-1].get("complete_marker_sha256")
        != PARENT_COMPLETE_SHA256
        or receipts[-1].get("policy_state_sha256")
        != PARENT_POLICY_STATE_SHA256
    ):
        raise Stage6PlanningWarmStartError(
            "parent accepted checkpoint receipt drifted"
        )
    resource_rows = _canonical_compact_jsonl_rows(
        by_relative["resource_audit.jsonl"],
        "parent U74 resource audit",
    )
    accepted_u74 = tuple(
        row
        for row in resource_rows
        if row.get("transaction_key") == PARENT_RECEIPT_KEY
        and row.get("phase") == "accepted"
        and row.get("accepted") is True
    )
    u75_rows = tuple(
        row
        for row in resource_rows
        if row.get("transaction_key") == DISCARDED_U75_TRANSACTION_KEY
    )
    if (
        len(accepted_u74) != 1
        or accepted_u74[0].get("checkpoint", {}).get("checkpoint_sha256")
        != PARENT_CHECKPOINT_SHA256
    ):
        raise Stage6PlanningWarmStartError(
            "parent U74 resource acceptance drifted"
        )
    if (
        len(u75_rows) != 1
        or u75_rows[0].get("phase") != "pre"
        or u75_rows[0].get("accepted") is not False
        or u75_rows[0].get("attempt") != DISCARDED_U75_ATTEMPT
        or u75_rows[0].get("segment_index")
        != DISCARDED_U75_SEGMENT_INDEX
    ):
        raise Stage6PlanningWarmStartError(
            "parent U75 attempt1 discard boundary drifted"
        )
    return VerifiedParentU74(
        parent_update=PARENT_UPDATE,
        checkpoint_sha256=PARENT_CHECKPOINT_SHA256,
        manifest_sha256=PARENT_MANIFEST_SHA256,
        complete_sha256=PARENT_COMPLETE_SHA256,
        policy_state_sha256=PARENT_POLICY_STATE_SHA256,
        config_sha256=PARENT_CONFIG_SHA256,
        lineage_sha256=PARENT_LINEAGE_SHA256,
        resource_accepted=True,
        u75_attempt1_discarded=True,
        checkpoint_payload=MappingProxyType(dict(payload)),
        checkpoint_bytes=checkpoint_bytes,
        manifest_bytes=manifest_bytes,
        complete_bytes=complete_bytes,
        checkpoint_directory=directory,
        checkpoint_member_names=PARENT_CHECKPOINT_MEMBER_NAMES,
    )


def restore_parent_u74_runtime(
    *,
    verified_parent: VerifiedParentU74,
    policy: object,
    optimizer: object,
) -> WarmStartRuntimeState:
    from lunar_exploration_ppo.configs.stage6 import SafetyContract
    from lunar_exploration_ppo.ppo.checkpoint import (
        load_complete_checkpoint_snapshot,
    )

    if (
        not isinstance(verified_parent, VerifiedParentU74)
        or verified_parent.parent_update != PARENT_UPDATE
        or verified_parent.checkpoint_sha256 != PARENT_CHECKPOINT_SHA256
        or verified_parent.manifest_sha256 != PARENT_MANIFEST_SHA256
        or verified_parent.complete_sha256 != PARENT_COMPLETE_SHA256
        or verified_parent.policy_state_sha256
        != PARENT_POLICY_STATE_SHA256
        or verified_parent.config_sha256 != PARENT_CONFIG_SHA256
        or verified_parent.lineage_sha256 != PARENT_LINEAGE_SHA256
        or hashlib.sha256(verified_parent.checkpoint_bytes).hexdigest()
        != PARENT_CHECKPOINT_SHA256
        or hashlib.sha256(verified_parent.manifest_bytes).hexdigest()
        != PARENT_MANIFEST_SHA256
        or hashlib.sha256(verified_parent.complete_bytes).hexdigest()
        != PARENT_COMPLETE_SHA256
        or verified_parent.checkpoint_directory.name
        != f"update-{PARENT_UPDATE:08d}"
        or verified_parent.checkpoint_member_names
        != PARENT_CHECKPOINT_MEMBER_NAMES
    ):
        raise Stage6PlanningWarmStartError(
            "verified parent U74 snapshot binding drifted"
        )
    payload = verified_parent.checkpoint_payload
    safety = SafetyContract.from_dict(
        _mapping(payload.get("safety_contract"), "parent safety")
    )
    try:
        loaded = load_complete_checkpoint_snapshot(
            directory=verified_parent.checkpoint_directory,
            update_step=PARENT_UPDATE,
            schema_version="stage6_complete_checkpoint/v1",
            member_names=verified_parent.checkpoint_member_names,
            checkpoint_bytes=verified_parent.checkpoint_bytes,
            manifest_bytes=verified_parent.manifest_bytes,
            complete_bytes=verified_parent.complete_bytes,
            policy=policy,
            optimizer=optimizer,
            expected_config_sha256=PARENT_CONFIG_SHA256,
            expected_lineage=_mapping(payload.get("lineage"), "parent lineage"),
            expected_safety_contract=safety,
        )
    except Exception as exc:
        raise Stage6PlanningWarmStartError(
            "parent U74 runtime restore failed"
        ) from exc
    return WarmStartRuntimeState(
        update_step=PARENT_UPDATE,
        policy_state_sha256=str(loaded.policy_state_sha256),
        normalization_stats=MappingProxyType(
            dict(loaded.normalization_stats)
        ),
        scenario_sampler_state=MappingProxyType(
            dict(loaded.scenario_sampler_state)
        ),
        model_state_restored=True,
        optimizer_state_restored=True,
        rng_state_restored=True,
        parent_vector_env_states_discarded=True,
        parent_best_record_discarded=True,
        parent_eval_metrics_discarded=True,
    )


def load_planning_warm_start_artifact(
    path: str | Path,
    *,
    expected_child_run_id: str,
    expected_child_config_sha256: str | None = None,
    expected_child_config_bytes: bytes | None = None,
    expected_bindings: Mapping[str, object] | None = None,
) -> tuple[PlanningWarmStartContext, str]:
    artifact_path = Path(path).expanduser().resolve()
    value = _canonical_json(artifact_path, "planning warm-start artifact")
    context = validate_planning_warm_start_artifact(
        value,
        expected_child_run_id=expected_child_run_id,
        expected_child_config_sha256=expected_child_config_sha256,
        expected_child_config_bytes=expected_child_config_bytes,
        expected_bindings=expected_bindings,
    )
    artifact_sha256 = _sha256(artifact_path)
    return (
        replace(
            context,
            artifact_path=artifact_path,
            artifact_sha256=artifact_sha256,
        ),
        artifact_sha256,
    )


__all__ = [
    "ACCEPTANCE_PROFILE_SCHEMA",
    "CHILD_FINAL_UPDATE",
    "CHILD_FIRST_UPDATE",
    "EFFECTIVE_CONFIG_SCHEMA",
    "PARENT_CHECKPOINT_SHA256",
    "PARENT_COMPLETE_SHA256",
    "PARENT_CONFIG_SHA256",
    "PARENT_LINEAGE_SHA256",
    "PARENT_MANIFEST_SHA256",
    "PARENT_POLICY_STATE_SHA256",
    "PARENT_RUN_ID",
    "PARENT_SEED",
    "PARENT_STAGE_ROOT",
    "PARENT_UPDATE",
    "PlanningEffectiveConfigSnapshot",
    "PlanningWarmStartContext",
    "Stage6PlanningWarmStartError",
    "VerifiedParentU74",
    "WARM_START_SCHEMA",
    "WarmStartRuntimeState",
    "build_planning_child_acceptance_profile",
    "build_planning_child_transactions",
    "load_planning_warm_start_artifact",
    "parse_planning_effective_config_bytes",
    "planning_effective_config_bytes",
    "planning_effective_config_sha256",
    "parent_u74_input_pin_requests",
    "planning_warm_start_expected_bindings",
    "planning_warm_start_expected_bindings_from_record",
    "restore_parent_u74_runtime",
    "select_parent_warm_start_state",
    "validate_planning_child_acceptance_profile",
    "validate_planning_warm_start_artifact",
    "verify_planning_child_terminal_replay",
    "verify_parent_u74_bundle",
    "verify_parent_u74_journal_snapshots",
]
