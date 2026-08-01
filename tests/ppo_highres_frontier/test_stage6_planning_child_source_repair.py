from __future__ import annotations

import hashlib
import importlib
import io
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import torch

from lunar_exploration_ppo.utils.artifact_io import ArtifactStore


FORMAL_RUN_ID = "s6-standard-single-r1-20260724T000124Z"
SEED = 20260716
ORIGIN_LINEAGE_SCHEMA = "stage6_lineage_audit/v3"
WARM_START_SCHEMA = "stage6_planning_u74_warm_start/v1"
WARM_START_MODE = "child_lineage_warm_start_not_exact_resume/v1"
SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64
SHA_E = "e" * 64
SHA_F = "f" * 64


def _module():
    return importlib.import_module(
        "lunar_exploration_ppo.workflows."
        "stage6_planning_child_source_repair"
    )


def _canonical_bytes(value: object) -> bytes:
    return ArtifactStore.canonical_json_bytes(value)


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _write_json(path: Path, value: object) -> bytes:
    payload = _canonical_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return payload


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> bytes:
    payload = b"".join(
        (
            json.dumps(
                row,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
        for row in rows
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return payload


def _fake_planner_runtime(
    tmp_path: Path,
) -> tuple[Path, dict[str, Path], tuple[str, ...]]:
    package_root = tmp_path / "editable" / "src" / "path_planner"
    sources = {
        "__init__.py": "from .search import AStarPlanner\n",
        "platform.py": "PLATFORM = 'fixture'\n",
        "core/__init__.py": (
            "from .models import Cell, CostGrid, GridSpec, "
            "NeighborPolicy, PlanRequest\n"
        ),
        "core/models.py": (
            "class Cell: pass\n"
            "class CostGrid: pass\n"
            "class GridSpec: pass\n"
            "class NeighborPolicy: pass\n"
            "class PlanRequest: pass\n"
        ),
        "postprocess/__init__.py": "POSTPROCESS = True\n",
        "regions/__init__.py": "REGIONS = True\n",
        "search/__init__.py": "from .astar import AStarPlanner\n",
        "search/astar.py": "class AStarPlanner: pass\n",
        "trajectory/__init__.py": "TRAJECTORY = True\n",
    }
    for relative, text in sources.items():
        path = package_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    excluded = (
        package_root / "v2" / "dirty.py",
        package_root / "search" / "tests" / "test_astar.py",
        package_root / "search" / "__pycache__" / "astar.py",
        package_root / "search" / "notes.txt",
    )
    for path in excluded:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("excluded\n", encoding="utf-8")
    resolved_symbols = {
        "AStarPlanner": package_root / "search" / "astar.py",
        "Cell": package_root / "core" / "models.py",
        "CostGrid": package_root / "core" / "models.py",
        "GridSpec": package_root / "core" / "models.py",
        "NeighborPolicy": package_root / "core" / "models.py",
        "PlanRequest": package_root / "core" / "models.py",
    }
    expected_paths = tuple(sorted(sources))
    return package_root, resolved_symbols, expected_paths


def _build_fake_planner_runtime_identity(
    tmp_path: Path,
) -> tuple[dict[str, object], Path, dict[str, Path], tuple[str, ...]]:
    package_root, resolved_symbols, expected_paths = _fake_planner_runtime(
        tmp_path
    )
    identity = _module().build_legacy_path_planner_runtime_identity(
        package_root=package_root,
        distribution_version="0.1.0",
        resolved_symbol_paths=resolved_symbols,
        loaded_module_names=(),
    )
    return identity, package_root, resolved_symbols, expected_paths


def _resource_snapshot(
    *,
    sample_count: int,
    rss_bytes: int,
    root_pid: int = 4242,
) -> dict[str, object]:
    return {
        "rss_source": "process_tree_lifecycle_peak_current_sum/v1",
        "rss_root_pid": root_pid,
        "rss_sample_count": sample_count,
        "rss_bytes": rss_bytes,
        "passed": True,
    }


@dataclass(frozen=True, slots=True)
class ChildRepairFixture:
    stage_root: Path
    warm_start_path: Path
    current_authorization_path: Path
    spec_review_path: Path
    quality_review_path: Path
    implementation_report_path: Path
    exact_replay_evidence_path: Path
    current_execution_identity: dict[str, object]
    current_verified_review_authorization: dict[str, object]
    current_immutable_bindings: dict[str, object]
    origin_lineage_bytes: bytes
    u84_checkpoint_sha256: str
    u84_manifest_sha256: str
    u84_complete_sha256: str
    u84_policy_state_sha256: str
    u84_lineage: dict[str, object]

    def build_kwargs(self) -> dict[str, object]:
        return {
            "stage_root": self.stage_root,
            "formal_run_id": FORMAL_RUN_ID,
            "seed": SEED,
            "current_execution_identity": self.current_execution_identity,
            "current_verified_review_authorization": (
                self.current_verified_review_authorization
            ),
            "current_immutable_bindings": self.current_immutable_bindings,
            "current_review_authorization_path": (
                self.current_authorization_path
            ),
            "planning_warm_start_path": self.warm_start_path,
            "spec_review_path": self.spec_review_path,
            "quality_review_path": self.quality_review_path,
            "implementation_report_path": self.implementation_report_path,
            "exact_replay_evidence_path": self.exact_replay_evidence_path,
            "created_at_utc": "2026-07-25T00:00:00Z",
        }


def _fixture(tmp_path: Path) -> ChildRepairFixture:
    stage = tmp_path / "out" / FORMAL_RUN_ID / "s6"
    review = tmp_path / "review"
    origin_review = review / "origin-u75"
    current_review = review / "current-u85"
    stage.mkdir(parents=True)
    origin_review.mkdir(parents=True)
    current_review.mkdir(parents=True)

    config_payload = _write_json(
        stage / "config.json",
        {
            "schema_version": "stage6_planning_effective_config/v2",
            "run_id": FORMAL_RUN_ID,
            "seed": SEED,
        },
    )
    config_sha256 = hashlib.sha256(config_payload).hexdigest()

    origin_authorization_payload = _write_json(
        origin_review / "launch-authorization.json",
        {
            "authorized": True,
            "formal_run_id": FORMAL_RUN_ID,
            "execution_identity": {
                "path": "execution-identity.json",
                "sha256": SHA_A,
                "size_bytes": 1,
            },
        },
    )
    origin_authorization_sha256 = hashlib.sha256(
        origin_authorization_payload
    ).hexdigest()
    origin_execution_identity = {
        "schema_version": "stage6_execution_identity/v1",
        "source_set_sha256": SHA_B,
        "config_sha256": config_sha256,
    }
    origin_verified_authorization = {
        "schema_version": "stage6_verified_review_launch_authorization/v1",
        "authorized": True,
        "formal_run_id": FORMAL_RUN_ID,
        "authorization_file_sha256": origin_authorization_sha256,
        "authorization_file_size_bytes": len(origin_authorization_payload),
        "review_identity_sha256": _canonical_sha256(
            origin_execution_identity
        ),
        "spec_review_sha256": SHA_C,
        "quality_review_sha256": SHA_D,
    }
    origin_immutable = {
        "formal_run_id": FORMAL_RUN_ID,
        "source_set_sha256": SHA_B,
        "config_sha256": config_sha256,
        "authorization_file_sha256": origin_authorization_sha256,
        "review_identity_sha256": _canonical_sha256(
            origin_execution_identity
        ),
    }

    warm_artifact = {
        "schema_version": WARM_START_SCHEMA,
        "mode": WARM_START_MODE,
        "parent": {
            "run_id": "s6-standard-single-r1-20260718T220434Z",
            "seed": SEED,
            "update": 74,
            "checkpoint_sha256": SHA_A,
            "manifest_sha256": SHA_B,
            "complete_sha256": SHA_C,
            "policy_state_sha256": SHA_D,
            "config_sha256": SHA_E,
            "lineage_sha256": SHA_F,
        },
        "child": {
            "run_id": FORMAL_RUN_ID,
            "seed": SEED,
            "first_update": 75,
            "final_update": 100,
            "new_semantics_update_count": 26,
            "effective_config_sha256": config_sha256,
        },
        "bindings": {
            "launch_authorization_sha256": origin_authorization_sha256,
            "source_set_sha256": SHA_B,
        },
    }
    warm_payload = _write_json(
        origin_review / "planning-warm-start.json",
        warm_artifact,
    )
    warm_sha256 = hashlib.sha256(warm_payload).hexdigest()
    origin_lineage = {
        "schema_version": ORIGIN_LINEAGE_SCHEMA,
        "execution_identity": origin_execution_identity,
        "immutable_bindings": origin_immutable,
        "verified_review_authorization": origin_verified_authorization,
        "planning_warm_start": {
            "artifact": warm_artifact,
            "artifact_sha256": warm_sha256,
            "artifact_size_bytes": len(warm_payload),
        },
    }
    origin_lineage_bytes = _write_json(
        stage / "lineage_audit.json",
        origin_lineage,
    )

    u84_lineage = {
        "stage5_commit": "1" * 40,
        "formal_run_id": FORMAL_RUN_ID,
        "source_set_sha256": SHA_B,
        "warm_start_schema_version": WARM_START_SCHEMA,
        "warm_start_artifact_sha256": warm_sha256,
        "checkpoint_update": 84,
    }
    checkpoint_buffer = io.BytesIO()
    torch.save({"lineage": u84_lineage}, checkpoint_buffer)
    checkpoint_payload = checkpoint_buffer.getvalue()
    checkpoint_sha256 = hashlib.sha256(checkpoint_payload).hexdigest()
    policy_state_sha256 = SHA_E
    checkpoint_root = (
        stage / "checkpoints" / f"seed-{SEED}" / "update-00000084"
    )
    checkpoint_root.mkdir(parents=True)
    (checkpoint_root / "checkpoint.pt").write_bytes(checkpoint_payload)
    manifest_payload = _write_json(
        checkpoint_root / "manifest.json",
        {
            "schema_version": "stage6_checkpoint_manifest/v1",
            "update_step": 84,
            "checkpoint": {
                "path": "checkpoint.pt",
                "size_bytes": len(checkpoint_payload),
                "sha256": checkpoint_sha256,
            },
            "policy_state_sha256": policy_state_sha256,
            "config_sha256": config_sha256,
            "lineage_sha256": _canonical_sha256(u84_lineage),
        },
    )
    manifest_sha256 = hashlib.sha256(manifest_payload).hexdigest()
    complete_payload = _write_json(
        checkpoint_root / "complete.json",
        {
            "schema_version": "stage6_checkpoint_complete/v1",
            "update_step": 84,
            "checkpoint_sha256": checkpoint_sha256,
            "manifest_sha256": manifest_sha256,
        },
    )
    complete_sha256 = hashlib.sha256(complete_payload).hexdigest()

    receipt_rows: list[dict[str, object]] = []
    training_rows: list[dict[str, object]] = []
    job_rows: list[dict[str, object]] = [
        {
            "state": f"seed_{SEED}_started",
            "bindings": {"formal_run_id": FORMAL_RUN_ID},
            "previous_record_hash": None,
            "record_hash": SHA_A,
        }
    ]
    resource_rows: list[dict[str, object]] = []
    segment_id = "12345678123442348123456789abcdef"
    first_snapshot = _resource_snapshot(sample_count=1, rss_bytes=100)
    resource_rows.append(
        {
            "schema_version": "stage6_resource_lifecycle_segment/v1",
            "kind": "resource_lifecycle_segment",
            "phase": "segment_start",
            "segment_id": segment_id,
            "segment_index": 1,
            "root_pid": 4242,
            "first_sample": first_snapshot,
            "prior_resource_log": {
                "size_bytes": 0,
                "sha256": hashlib.sha256(b"").hexdigest(),
            },
        }
    )
    previous_record_hash = None
    for ordinal, update in enumerate(range(75, 85)):
        transaction_key = f"{ordinal:04d}:{SEED}:update:{update:03d}"
        row_checkpoint_sha256 = (
            checkpoint_sha256 if update == 84 else f"{update:064x}"
        )
        row_complete_sha256 = (
            complete_sha256 if update == 84 else f"{update + 100:064x}"
        )
        row_policy_sha256 = (
            policy_state_sha256 if update == 84 else f"{update + 200:064x}"
        )
        record_hash = f"{update + 300:064x}"
        receipt_rows.append(
            {
                "seed": SEED,
                "update": update,
                "transaction_key": transaction_key,
                "checkpoint_sha256": row_checkpoint_sha256,
                "complete_marker_sha256": row_complete_sha256,
                "policy_state_sha256": row_policy_sha256,
                "previous_record_hash": previous_record_hash,
                "record_hash": record_hash,
            }
        )
        previous_record_hash = record_hash
        training_rows.append(
            {
                "seed": SEED,
                "update": update,
                "transaction_key": transaction_key,
                "checkpoint_sha256": row_checkpoint_sha256,
                "policy_state_sha256": row_policy_sha256,
                "update_metrics": {"loss": float(update)},
                "validation": None,
                "collection_audit": {},
            }
        )
        job_rows.append(
            {
                "state": f"seed_{SEED}_training_update_{update}",
                "bindings": {
                    "formal_run_id": FORMAL_RUN_ID,
                    "checkpoint_sha256": row_checkpoint_sha256,
                },
                "previous_record_hash": job_rows[-1]["record_hash"],
                "record_hash": f"{update + 400:064x}",
            }
        )
        pre = _resource_snapshot(
            sample_count=2 + ordinal * 2,
            rss_bytes=200 + ordinal * 20,
        )
        post = _resource_snapshot(
            sample_count=3 + ordinal * 2,
            rss_bytes=210 + ordinal * 20,
        )
        resource_rows.extend(
            [
                {
                    "schema_version": "stage6_resource_attempt/v1",
                    "kind": "update",
                    "seed": SEED,
                    "update": update,
                    "transaction_key": transaction_key,
                    "attempt": 1,
                    "phase": "pre",
                    "accepted": False,
                    "segment_id": segment_id,
                    "segment_index": 1,
                    "resource": pre,
                },
                {
                    "schema_version": "stage6_resource_attempt/v1",
                    "kind": "update",
                    "seed": SEED,
                    "update": update,
                    "transaction_key": transaction_key,
                    "attempt": 1,
                    "phase": "post",
                    "accepted": False,
                    "segment_id": segment_id,
                    "segment_index": 1,
                    "resource": post,
                },
                {
                    "schema_version": "stage6_resource_acceptance/v2",
                    "kind": "update",
                    "seed": SEED,
                    "update": update,
                    "transaction_key": transaction_key,
                    "attempt": 1,
                    "phase": "accepted",
                    "accepted": True,
                    "segment_id": segment_id,
                    "segment_index": 1,
                    "pre": pre,
                    "post": post,
                    "checkpoint": {
                        "seed": SEED,
                        "update": update,
                        "transaction_key": transaction_key,
                        "checkpoint_sha256": row_checkpoint_sha256,
                        "complete_marker_sha256": row_complete_sha256,
                        "policy_state_sha256": row_policy_sha256,
                    },
                },
            ]
        )
    resource_rows.append(
        {
            "schema_version": "stage6_resource_attempt/v1",
            "kind": "update",
            "seed": SEED,
            "update": 85,
            "transaction_key": f"0010:{SEED}:update:085",
            "attempt": 1,
            "phase": "pre",
            "accepted": False,
            "segment_id": segment_id,
            "segment_index": 1,
            "resource": _resource_snapshot(
                sample_count=22,
                rss_bytes=400,
            ),
        }
    )

    _write_jsonl(stage / "checkpoints" / "index.jsonl", receipt_rows)
    _write_jsonl(stage / "training_metrics.jsonl", training_rows)
    _write_jsonl(stage / "job-state.jsonl", job_rows)
    _write_jsonl(stage / "resource_audit.jsonl", resource_rows)
    _write_jsonl(
        stage / "validation_metrics.jsonl",
        [
            {
                "seed": SEED,
                "update": 80,
                "transaction_key": f"0005:{SEED}:update:080",
                "episode_count": 16,
            }
        ],
    )

    current_execution_identity = {
        "schema_version": "stage6_execution_identity/v1",
        "source_set_sha256": SHA_C,
        "config_sha256": config_sha256,
    }
    current_authorization_payload = _write_json(
        current_review / "launch-authorization.json",
        {
            "authorized": True,
            "formal_run_id": FORMAL_RUN_ID,
            "reviewed_execution_identity": current_execution_identity,
        },
    )
    current_authorization_sha256 = hashlib.sha256(
        current_authorization_payload
    ).hexdigest()
    current_verified_authorization = {
        "schema_version": "stage6_verified_review_launch_authorization/v1",
        "authorized": True,
        "formal_run_id": FORMAL_RUN_ID,
        "authorization_file_sha256": current_authorization_sha256,
        "authorization_file_size_bytes": len(current_authorization_payload),
        "review_identity_sha256": _canonical_sha256(
            current_execution_identity
        ),
        "spec_review_sha256": SHA_D,
        "quality_review_sha256": SHA_E,
    }
    current_immutable = {
        "formal_run_id": FORMAL_RUN_ID,
        "source_set_sha256": SHA_C,
        "config_sha256": config_sha256,
        "authorization_file_sha256": current_authorization_sha256,
        "review_identity_sha256": _canonical_sha256(
            current_execution_identity
        ),
    }
    spec_review_path = current_review / "spec-review.json"
    quality_review_path = current_review / "quality-review.json"
    implementation_report_path = current_review / "implementer-report.md"
    exact_replay_evidence_path = current_review / "exact-replay.json"
    _write_json(
        spec_review_path,
        {
            "verdict": "PASS",
            "finding_counts": {
                "critical": 0,
                "important": 0,
                "minor": 0,
            },
        },
    )
    _write_json(
        quality_review_path,
        {
            "verdict": "PASS",
            "finding_counts": {
                "critical": 0,
                "important": 0,
                "minor": 0,
            },
        },
    )
    implementation_report_path.write_text(
        "# Implementer report\n\nPASS\n",
        encoding="utf-8",
    )
    _write_json(
        exact_replay_evidence_path,
        {
            "schema_version": "stage6_update85_exact_replay/v1",
            "production_candidate_count": 3,
        },
    )
    return ChildRepairFixture(
        stage_root=stage,
        warm_start_path=origin_review / "planning-warm-start.json",
        current_authorization_path=current_review
        / "launch-authorization.json",
        spec_review_path=spec_review_path,
        quality_review_path=quality_review_path,
        implementation_report_path=implementation_report_path,
        exact_replay_evidence_path=exact_replay_evidence_path,
        current_execution_identity=current_execution_identity,
        current_verified_review_authorization=(
            current_verified_authorization
        ),
        current_immutable_bindings=current_immutable,
        origin_lineage_bytes=origin_lineage_bytes,
        u84_checkpoint_sha256=checkpoint_sha256,
        u84_manifest_sha256=manifest_sha256,
        u84_complete_sha256=complete_sha256,
        u84_policy_state_sha256=policy_state_sha256,
        u84_lineage=u84_lineage,
    )


def _build(fixture: ChildRepairFixture) -> dict[str, object]:
    return _module().build_planning_child_source_repair_artifact(
        **fixture.build_kwargs()
    )


def _rewrite_u84_checkpoint_record(
    fixture: ChildRepairFixture,
    record: dict[str, object],
) -> None:
    checkpoint_root = (
        fixture.stage_root
        / "checkpoints"
        / f"seed-{SEED}"
        / "update-00000084"
    )
    checkpoint_buffer = io.BytesIO()
    torch.save(record, checkpoint_buffer)
    checkpoint_payload = checkpoint_buffer.getvalue()
    checkpoint_sha256 = hashlib.sha256(checkpoint_payload).hexdigest()
    (checkpoint_root / "checkpoint.pt").write_bytes(checkpoint_payload)

    manifest = json.loads(
        (checkpoint_root / "manifest.json").read_text(encoding="utf-8")
    )
    manifest["checkpoint"].update(
        {
            "size_bytes": len(checkpoint_payload),
            "sha256": checkpoint_sha256,
        }
    )
    manifest_payload = _write_json(
        checkpoint_root / "manifest.json",
        manifest,
    )
    complete = json.loads(
        (checkpoint_root / "complete.json").read_text(encoding="utf-8")
    )
    complete.update(
        {
            "checkpoint_sha256": checkpoint_sha256,
            "manifest_sha256": hashlib.sha256(manifest_payload).hexdigest(),
        }
    )
    complete_payload = _write_json(
        checkpoint_root / "complete.json",
        complete,
    )
    complete_sha256 = hashlib.sha256(complete_payload).hexdigest()

    receipt_path = fixture.stage_root / "checkpoints" / "index.jsonl"
    receipt_rows = [
        json.loads(line)
        for line in receipt_path.read_text(encoding="utf-8").splitlines()
    ]
    receipt_rows[-1]["checkpoint_sha256"] = checkpoint_sha256
    receipt_rows[-1]["complete_marker_sha256"] = complete_sha256
    _write_jsonl(receipt_path, receipt_rows)

    training_path = fixture.stage_root / "training_metrics.jsonl"
    training_rows = [
        json.loads(line)
        for line in training_path.read_text(encoding="utf-8").splitlines()
    ]
    training_rows[-1]["checkpoint_sha256"] = checkpoint_sha256
    _write_jsonl(training_path, training_rows)

    resource_path = fixture.stage_root / "resource_audit.jsonl"
    resource_rows = [
        json.loads(line)
        for line in resource_path.read_text(encoding="utf-8").splitlines()
    ]
    accepted = next(
        row
        for row in resource_rows
        if row.get("update") == 84 and row.get("phase") == "accepted"
    )
    accepted["checkpoint"]["checkpoint_sha256"] = checkpoint_sha256
    accepted["checkpoint"]["complete_marker_sha256"] = complete_sha256
    _write_jsonl(resource_path, resource_rows)


def _context(
    fixture: ChildRepairFixture,
    artifact: dict[str, object],
    *,
    require_exact_prefix: bool = True,
):
    return _module().validate_planning_child_source_repair_artifact(
        artifact,
        stage_root=fixture.stage_root,
        current_execution_identity=fixture.current_execution_identity,
        current_verified_review_authorization=(
            fixture.current_verified_review_authorization
        ),
        current_immutable_bindings=fixture.current_immutable_bindings,
        require_exact_prefix=require_exact_prefix,
    )


def _rehash_artifact(
    artifact: dict[str, object],
) -> dict[str, object]:
    value = json.loads(json.dumps(artifact))
    value.pop("canonical_sha256")
    value["canonical_sha256"] = _canonical_sha256(value)
    return value


def test_build_child_source_repair_binds_origin_current_and_exact_u85_attempt2(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)

    artifact = _build(fixture)
    context = _context(fixture, artifact)

    assert artifact["schema_version"] == (
        "stage6_planning_child_source_repair/v1"
    )
    assert artifact["mode"] == (
        "same_run_exact_resume_after_reviewed_source_repair/v1"
    )
    assert artifact["accepted_prefix"]["receipt_count"] == 10
    assert artifact["accepted_prefix"]["first_update"] == 75
    assert artifact["accepted_prefix"]["last_update"] == 84
    assert artifact["resume_point"] == {
        "last_complete_update": 84,
        "next_update": 85,
        "abandoned_attempt": 1,
        "next_attempt": 2,
        "abandoned_attempt_has_pre_only": True,
        "last_resource_segment_index": 1,
        "next_resource_segment_index": 2,
        "next_transaction_key": f"0010:{SEED}:update:085",
    }
    assert artifact["origin"]["lineage"]["sha256"] == hashlib.sha256(
        fixture.origin_lineage_bytes
    ).hexdigest()
    assert artifact["origin"]["checkpoint"]["checkpoint_sha256"] == (
        fixture.u84_checkpoint_sha256
    )
    assert artifact["current"][
        "execution_identity_sha256"
    ] == _canonical_sha256(fixture.current_execution_identity)
    assert context.formal_run_id == FORMAL_RUN_ID
    assert context.seed == SEED
    assert context.last_origin_update == 84
    assert context.first_repaired_update == 85
    assert context.next_attempt == 2
    assert context.next_resource_segment_index == 2
    assert context.expected_historical_checkpoint_lineage(84) == (
        fixture.u84_lineage
    )
    with pytest.raises(
        _module().Stage6PlanningChildSourceRepairError,
        match="historical checkpoint update",
    ):
        context.expected_historical_checkpoint_lineage(85)


def test_u84_artifact_build_accepts_production_numpy_rng_checkpoint(
    tmp_path: Path,
) -> None:
    import numpy as np

    fixture = _fixture(tmp_path)
    _rewrite_u84_checkpoint_record(
        fixture,
        {
            "lineage": fixture.u84_lineage,
            "update_step": 84,
            "rng_state": {"numpy": np.random.get_state()},
        },
    )

    artifact = _build(fixture)

    assert artifact["origin"]["checkpoint"]["update"] == 84


def test_u84_artifact_build_rejects_unallowlisted_checkpoint_global(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    _rewrite_u84_checkpoint_record(
        fixture,
        {
            "lineage": fixture.u84_lineage,
            "unsupported_global": Path("not-allowlisted"),
        },
    )

    with pytest.raises(
        _module().Stage6PlanningChildSourceRepairError,
        match="U84 checkpoint payload could not be decoded",
    ):
        _build(fixture)


@pytest.mark.parametrize(
    "binding_name",
    ("immutable_bindings", "verified_review_authorization"),
)
def test_child_source_repair_rejects_origin_run_id_drift(
    tmp_path: Path,
    binding_name: str,
) -> None:
    fixture = _fixture(tmp_path)
    lineage_path = fixture.stage_root / "lineage_audit.json"
    lineage = json.loads(lineage_path.read_text(encoding="utf-8"))
    lineage[binding_name]["formal_run_id"] = "s6-wrong-run-id"
    _write_json(lineage_path, lineage)

    with pytest.raises(
        _module().Stage6PlanningChildSourceRepairError,
        match="origin formal run id drifted",
    ):
        _build(fixture)


@pytest.mark.parametrize(
    ("binding_name", "error"),
    (
        (
            "current_verified_review_authorization",
            "current review authorization formal run id drifted",
        ),
        (
            "current_immutable_bindings",
            "current immutable formal run id drifted",
        ),
    ),
)
def test_child_source_repair_rejects_current_run_id_drift(
    tmp_path: Path,
    binding_name: str,
    error: str,
) -> None:
    fixture = _fixture(tmp_path)
    getattr(fixture, binding_name)["formal_run_id"] = "s6-wrong-run-id"

    with pytest.raises(
        _module().Stage6PlanningChildSourceRepairError,
        match=error,
    ):
        _build(fixture)


def test_child_source_repair_rejects_rewriting_origin_lineage(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    artifact = _build(fixture)
    context = _context(fixture, artifact)
    mutated = fixture.origin_lineage_bytes.replace(
        b'"stage6_lineage_audit/v3"',
        b'"stage6_lineage_audit/v2"',
    )

    with pytest.raises(
        _module().Stage6PlanningChildSourceRepairError,
        match="origin lineage",
    ):
        context.verify_origin_lineage(mutated)


def test_child_source_repair_rejects_non_contiguous_accepted_prefix(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    path = fixture.stage_root / "checkpoints" / "index.jsonl"
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    del rows[4]
    _write_jsonl(path, rows)

    with pytest.raises(
        _module().Stage6PlanningChildSourceRepairError,
        match="checkpoint index.*75\\.\\.84",
    ):
        _build(fixture)


@pytest.mark.parametrize("phase", ["post", "accepted"])
def test_child_source_repair_rejects_u85_attempt1_post_or_accept(
    tmp_path: Path,
    phase: str,
) -> None:
    fixture = _fixture(tmp_path)
    path = fixture.stage_root / "resource_audit.jsonl"
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    pre = dict(rows[-1]["resource"])
    row: dict[str, object] = {
        "schema_version": "stage6_resource_attempt/v1",
        "kind": "update",
        "seed": SEED,
        "update": 85,
        "transaction_key": f"0010:{SEED}:update:085",
        "attempt": 1,
        "phase": phase,
        "accepted": phase == "accepted",
        "segment_id": rows[-1]["segment_id"],
        "segment_index": 1,
    }
    if phase == "post":
        row["accepted"] = False
        row["resource"] = {
            **pre,
            "rss_sample_count": int(pre["rss_sample_count"]) + 1,
            "rss_bytes": int(pre["rss_bytes"]) + 1,
        }
    else:
        post = {
            **pre,
            "rss_sample_count": int(pre["rss_sample_count"]) + 1,
            "rss_bytes": int(pre["rss_bytes"]) + 1,
        }
        rows.append(
            {
                **row,
                "phase": "post",
                "accepted": False,
                "resource": post,
            }
        )
        row["schema_version"] = "stage6_resource_acceptance/v2"
        row["pre"] = pre
        row["post"] = post
        row["checkpoint"] = {}
    rows.append(row)
    _write_jsonl(path, rows)

    with pytest.raises(
        _module().Stage6PlanningChildSourceRepairError,
        match="U85 attempt1.*pre only",
    ):
        _build(fixture)


def test_child_source_repair_derives_next_resource_segment_from_validated_ledger(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    artifact = _build(fixture)

    assert artifact["resume_point"]["last_resource_segment_index"] == 1
    assert artifact["resume_point"]["next_resource_segment_index"] == 2
    assert artifact["accepted_prefix"]["resource_validation"][
        "schema_version"
    ] == "stage6_resource_lifecycle_validation/v2"


@pytest.mark.parametrize("review_name", ["spec", "quality"])
def test_child_source_repair_rejects_spec_or_quality_important_findings(
    tmp_path: Path,
    review_name: str,
) -> None:
    fixture = _fixture(tmp_path)
    review_path = (
        fixture.spec_review_path
        if review_name == "spec"
        else fixture.quality_review_path
    )
    _write_json(
        review_path,
        {
            "verdict": "FAIL",
            "finding_counts": {
                "critical": 0,
                "important": 1,
                "minor": 0,
            },
        },
    )

    with pytest.raises(
        _module().Stage6PlanningChildSourceRepairError,
        match=f"{review_name} review.*C0/I0",
        ):
            _build(fixture)


def test_child_repair_allows_only_tail_append_after_frozen_prefix(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    artifact = _build(fixture)
    context = _context(fixture, artifact)
    resource_path = fixture.stage_root / "resource_audit.jsonl"
    resource_path.write_bytes(
        resource_path.read_bytes()
        + b'{"phase":"runtime_tail","schema_version":"test/v1"}\n'
    )

    context.require_current("after legal journal tail append")
    context.verify_append_only_prefixes("after legal journal tail append")
    with pytest.raises(
        _module().Stage6PlanningChildSourceRepairError,
        match="gained a tail",
    ):
        context.verify_append_only_prefixes(
            "before exact resume",
            require_exact=True,
        )

    assert "resource_audit.jsonl" not in artifact["immutable_inputs"]
    assert "resource_audit.jsonl" in artifact["append_only_prefixes"]


def test_child_repair_rejects_prefix_byte_mutation(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    context = _context(fixture, _build(fixture))
    resource_path = fixture.stage_root / "resource_audit.jsonl"
    payload = bytearray(resource_path.read_bytes())
    payload[0] = ord("[")
    resource_path.write_bytes(payload)

    with pytest.raises(
        _module().Stage6PlanningChildSourceRepairError,
        match="prefix resource_audit\\.jsonl changed",
    ):
        context.verify_append_only_prefixes("runtime")


def test_child_repair_rejects_prefix_truncation(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    context = _context(fixture, _build(fixture))
    resource_path = fixture.stage_root / "resource_audit.jsonl"
    resource_path.write_bytes(resource_path.read_bytes()[:-1])

    with pytest.raises(
        _module().Stage6PlanningChildSourceRepairError,
        match="prefix resource_audit\\.jsonl was truncated",
    ):
        context.verify_append_only_prefixes("runtime")


@pytest.mark.parametrize("line_count_delta", (-1, 1))
def test_child_repair_rejects_rehashed_resource_prefix_line_count_drift(
    tmp_path: Path,
    line_count_delta: int,
) -> None:
    fixture = _fixture(tmp_path)
    artifact = _build(fixture)
    resource_binding = artifact["append_only_prefixes"][
        "resource_audit.jsonl"
    ]
    resource_binding["line_count"] = (
        int(resource_binding["line_count"]) + line_count_delta
    )

    with pytest.raises(
        _module().Stage6PlanningChildSourceRepairError,
        match="resource_audit\\.jsonl.*line count",
    ):
        _context(fixture, _rehash_artifact(artifact))


def test_child_repair_rejects_non_newline_terminated_frozen_prefix(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    artifact = _build(fixture)
    resource_path = fixture.stage_root / "resource_audit.jsonl"
    payload = resource_path.read_bytes()
    assert payload.endswith(b"\n")
    payload = payload[:-1]
    resource_path.write_bytes(payload)
    resource_binding = artifact["append_only_prefixes"][
        "resource_audit.jsonl"
    ]
    resource_binding.update(
        {
            "prefix_size_bytes": len(payload),
            "prefix_sha256": hashlib.sha256(payload).hexdigest(),
            "line_count": payload.count(b"\n"),
        }
    )

    with pytest.raises(
        _module().Stage6PlanningChildSourceRepairError,
        match="resource_audit\\.jsonl.*canonical.*JSONL|newline",
    ):
        _context(fixture, _rehash_artifact(artifact))


def test_child_context_breaks_nested_prefix_input_alias_and_rejects_drift(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    artifact = _build(fixture)
    context = _context(fixture, artifact)
    original_binding = json.loads(
        json.dumps(
            context.artifact["append_only_prefixes"][
                "resource_audit.jsonl"
            ]
        )
    )

    artifact["append_only_prefixes"]["resource_audit.jsonl"][
        "line_count"
    ] = int(original_binding["line_count"]) + 1

    assert (
        context.artifact["append_only_prefixes"]["resource_audit.jsonl"]
        == original_binding
    )

    context.artifact["append_only_prefixes"]["resource_audit.jsonl"][
        "line_count"
    ] = int(original_binding["line_count"]) + 1
    with pytest.raises(
        _module().Stage6PlanningChildSourceRepairError,
        match="in-memory artifact drifted",
    ):
        context.require_current("nested append-only prefix drift")


def test_child_repair_publish_is_atomic_and_exclusive(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    artifact = _build(fixture)
    output = (
        fixture.stage_root
        / _module().PLANNING_CHILD_SOURCE_REPAIR_NAME
    )

    published = _module().publish_planning_child_source_repair_artifact(
        artifact,
        stage_root=fixture.stage_root,
        output_path=output,
    )

    assert published == output
    assert output.read_bytes() == _canonical_bytes(artifact)
    with pytest.raises(FileExistsError):
        _module().publish_planning_child_source_repair_artifact(
            artifact,
            stage_root=fixture.stage_root,
            output_path=output,
        )


def test_child_repair_rejects_path_escape_and_reparse_point(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    artifact = _build(fixture)
    outside = tmp_path / "outside.jsonl"
    outside.write_bytes(b"{}\n")
    escaped = json.loads(json.dumps(artifact))
    escaped["append_only_prefixes"]["resource_audit.jsonl"]["path"] = (
        "../../../outside.jsonl"
    )
    escaped = _rehash_artifact(escaped)

    with pytest.raises(
        _module().Stage6PlanningChildSourceRepairError,
        match="could not be securely read",
    ):
        _context(fixture, escaped)
    with pytest.raises(
        _module().Stage6PlanningChildSourceRepairError,
        match="canonical stage root",
    ):
        _module().publish_planning_child_source_repair_artifact(
            artifact,
            stage_root=fixture.stage_root,
            output_path=tmp_path / "escaped.json",
        )

    alias = fixture.stage_root / "resource-alias.jsonl"
    try:
        os.symlink(outside, alias)
    except OSError:
        return
    reparsed = json.loads(json.dumps(artifact))
    reparsed["append_only_prefixes"]["resource_audit.jsonl"]["path"] = (
        "resource-alias.jsonl"
    )
    reparsed = _rehash_artifact(reparsed)
    with pytest.raises(
        _module().Stage6PlanningChildSourceRepairError,
        match="could not be securely read",
    ):
        _context(fixture, reparsed)


def test_child_repair_runtime_pin_requests_exclude_mutable_journals(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    artifact = _build(fixture)
    output = (
        fixture.stage_root
        / _module().PLANNING_CHILD_SOURCE_REPAIR_NAME
    )
    _module().publish_planning_child_source_repair_artifact(
        artifact,
        stage_root=fixture.stage_root,
        output_path=output,
    )

    requests = _module().planning_child_source_repair_input_pin_requests(
        output
    )

    labels = {label for label, _path in requests}
    paths = {path for _label, path in requests}
    assert "planning-child-source-repair:artifact" in labels
    assert output in paths
    assert (
        fixture.stage_root / "resource_audit.jsonl"
    ) not in paths
    assert (
        fixture.stage_root / "training_metrics.jsonl"
    ) not in paths
    assert fixture.stage_root / "lineage_audit.json" in paths
    assert fixture.stage_root / "checkpoints" / (
        f"seed-{SEED}/update-00000084/checkpoint.pt"
    ) in paths


def test_planner_runtime_identity_binds_actual_editable_legacy_source_set(
    tmp_path: Path,
) -> None:
    identity, package_root, _resolved_symbols, expected_paths = (
        _build_fake_planner_runtime_identity(tmp_path)
    )
    expected_rows = [
        {
            "path": relative,
            "size_bytes": (package_root / relative).stat().st_size,
            "sha256": hashlib.sha256(
                (package_root / relative).read_bytes()
            ).hexdigest(),
        }
        for relative in expected_paths
    ]

    assert identity == {
        "schema_version": (
            "stage6_path_planner_legacy_runtime_source_set/v1"
        ),
        "distribution_name": "path-planner",
        "distribution_version": "0.1.0",
        "package_root": package_root.resolve().as_posix(),
        "source_set_sha256": _canonical_sha256(expected_rows),
        "paths": expected_rows,
        "resolved_symbols": {
            "AStarPlanner": "search/astar.py",
            "Cell": "core/models.py",
            "CostGrid": "core/models.py",
            "GridSpec": "core/models.py",
            "NeighborPolicy": "core/models.py",
            "PlanRequest": "core/models.py",
        },
        "forbidden_loaded_prefix": "path_planner.v2",
    }


def test_planner_runtime_identity_rejects_member_bytes_or_membership_drift(
    tmp_path: Path,
) -> None:
    identity, package_root, resolved_symbols, _expected_paths = (
        _build_fake_planner_runtime_identity(tmp_path)
    )
    astar = package_root / "search" / "astar.py"
    original = astar.read_bytes()
    astar.write_bytes(original + b"# drift\n")

    with pytest.raises(
        _module().Stage6PlanningChildSourceRepairError,
        match="bytes",
    ):
        _module().require_legacy_path_planner_runtime_identity_current(
            identity,
            label="member bytes drift",
            package_root=package_root,
            distribution_version="0.1.0",
            resolved_symbol_paths=resolved_symbols,
            loaded_module_names=(),
        )

    astar.write_bytes(original)
    (package_root / "search" / "new_member.py").write_text(
        "NEW_MEMBER = True\n",
        encoding="utf-8",
    )
    with pytest.raises(
        _module().Stage6PlanningChildSourceRepairError,
        match="membership",
    ):
        _module().require_legacy_path_planner_runtime_identity_current(
            identity,
            label="member set drift",
            package_root=package_root,
            distribution_version="0.1.0",
            resolved_symbol_paths=resolved_symbols,
            loaded_module_names=(),
        )


def test_planner_runtime_identity_rejects_package_root_drift(
    tmp_path: Path,
) -> None:
    identity, _package_root, _resolved_symbols, _expected_paths = (
        _build_fake_planner_runtime_identity(tmp_path / "first")
    )
    other_root, other_symbols, _ = _fake_planner_runtime(tmp_path / "second")

    with pytest.raises(
        _module().Stage6PlanningChildSourceRepairError,
        match="root",
    ):
        _module().require_legacy_path_planner_runtime_identity_current(
            identity,
            label="editable root drift",
            package_root=other_root,
            distribution_version="0.1.0",
            resolved_symbol_paths=other_symbols,
            loaded_module_names=(),
        )


def test_planner_runtime_identity_rejects_resolved_symbol_outside_package_root(
    tmp_path: Path,
) -> None:
    package_root, resolved_symbols, _expected_paths = _fake_planner_runtime(
        tmp_path
    )
    outside = tmp_path / "outside.py"
    outside.write_text("class AStarPlanner: pass\n", encoding="utf-8")
    resolved_symbols["AStarPlanner"] = outside

    with pytest.raises(
        _module().Stage6PlanningChildSourceRepairError,
        match="resolved symbol",
    ):
        _module().build_legacy_path_planner_runtime_identity(
            package_root=package_root,
            distribution_version="0.1.0",
            resolved_symbol_paths=resolved_symbols,
            loaded_module_names=(),
        )


def test_planner_runtime_identity_rejects_loaded_path_planner_v2(
    tmp_path: Path,
) -> None:
    package_root, resolved_symbols, _expected_paths = _fake_planner_runtime(
        tmp_path
    )

    with pytest.raises(
        _module().Stage6PlanningChildSourceRepairError,
        match="forbidden-v2",
    ):
        _module().build_legacy_path_planner_runtime_identity(
            package_root=package_root,
            distribution_version="0.1.0",
            resolved_symbol_paths=resolved_symbols,
            loaded_module_names=("path_planner.v2",),
        )


def test_unrelated_repo_head_or_path_planner_v2_dirty_does_not_change_identity(
    tmp_path: Path,
) -> None:
    identity, package_root, resolved_symbols, _expected_paths = (
        _build_fake_planner_runtime_identity(tmp_path)
    )
    (package_root.parent.parent / "HEAD-DIAGNOSTIC").write_text(
        "56bbd497d8e0cc93975a7dc09affc6800b5f60ff\n",
        encoding="utf-8",
    )
    (package_root / "v2" / "dirty.py").write_text(
        "DIRTY = 'changed but unused'\n",
        encoding="utf-8",
    )

    after = _module().build_legacy_path_planner_runtime_identity(
        package_root=package_root,
        distribution_version="0.1.0",
        resolved_symbol_paths=resolved_symbols,
        loaded_module_names=(),
    )

    assert after["source_set_sha256"] == identity["source_set_sha256"]
    assert after["paths"] == identity["paths"]


def test_stage6_input_pins_include_every_planner_runtime_source_member(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path / "child")
    runtime, package_root, _resolved_symbols, expected_paths = (
        _build_fake_planner_runtime_identity(tmp_path / "planner")
    )
    artifact = _build(fixture)
    artifact["current"]["planner_runtime_identity"] = runtime
    artifact = _rehash_artifact(artifact)
    output = fixture.stage_root / _module().PLANNING_CHILD_SOURCE_REPAIR_NAME
    _write_json(output, artifact)
    monkeypatch.setattr(
        _module(),
        "resolve_legacy_path_planner_runtime_identity",
        lambda: runtime,
        raising=False,
    )

    requests = _module().planning_child_source_repair_input_pin_requests(
        output
    )

    requested_paths = {path for _label, path in requests}
    assert {
        (package_root / relative).resolve() for relative in expected_paths
    }.issubset(requested_paths)
    assert package_root / "v2" / "dirty.py" not in requested_paths
