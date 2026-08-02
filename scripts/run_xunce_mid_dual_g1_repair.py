"""Run and independently recompute the three-case G1 incremental repair.

This module is deliberately separate from the original G1 runner.  It never
rewrites the parent formal root and it labels the resulting 24-case evidence
as a mixed-code incremental repair.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time
from typing import Any, Mapping, Sequence

import xunce_artifact_io as artifact_io
from xunce_mid_dual_contracts import g1_split_statistics


REPAIR_EPISODE_INDICES = (3, 10, 23)
REPAIR_CASES: dict[int, dict[str, object]] = {
    3: {
        "scenario_id": "unseen/scenario-0036/standard-proxy/v1",
        "record_id": "unseen/scenario-0036",
        "lane_id": "lane-3",
        "evaluation_seed": 2026072703,
    },
    10: {
        "scenario_id": "unseen/scenario-0018/standard-proxy/v1",
        "record_id": "unseen/scenario-0018",
        "lane_id": "lane-2",
        "evaluation_seed": 2026072710,
    },
    23: {
        "scenario_id": "unseen/scenario-0062/standard-proxy/v1",
        "record_id": "unseen/scenario-0062",
        "lane_id": "lane-7",
        "evaluation_seed": 2026072723,
    },
}
PARENT_FAILURE_CONTRACTS: dict[int, dict[str, object]] = {
    3: {
        "parent_line_number": 4,
        "denominator_cell_count": 63_000,
        "denominator_sha256": (
            "c27b39df5994429319d51d50b61be07b94170e4bf32fc9e792d0b8152a77dcfa"
        ),
        "covered_cell_count": 1_173,
        "coverage": 0.01861904761904762,
    },
    10: {
        "parent_line_number": 11,
        "denominator_cell_count": 64_291,
        "denominator_sha256": (
            "702d6646807e356cc22a83a0a75fd3ecc1c8566a354de89b2ad7e41edac43aa7"
        ),
        "covered_cell_count": 1_301,
        "coverage": 0.020236113919522174,
    },
    23: {
        "parent_line_number": 24,
        "denominator_cell_count": 57_857,
        "denominator_sha256": (
            "37dffeac83674f97c93e64a7cf2da6c24c3b99585f7f15cb960334263f0a3c96"
        ),
        "covered_cell_count": 1_005,
        "coverage": 0.017370413260279657,
    },
}

PARENT_COVERAGE_SCHEMA = "xunce-mid-dual-g1-coverage-episode/v1"
MERGED_ROW_SCHEMA = "xunce-mid-dual-g1-mixed-repair-row/v1"
RECOMPUTE_SCHEMA = "xunce-mid-dual-g1-repair-recompute/v1"
JOB_STATE_SCHEMA = "xunce-mid-dual-g1-repair-job-state/v1"
CONFIG_SCHEMA = "xunce-mid-dual-g1-repair-config/v1"
EPISODE_LINEAGE_KEYS = frozenset(
    {
        "checkpoint_sha256",
        "policy_state_sha256",
        "scenario_manifest_sha256",
        "formal_config_sha256",
        "repair_code_sha256",
        "repair_static_config_sha256",
    }
)
_REPO_ROOT = Path(__file__).resolve().parents[1]
REPAIR_SOURCE_RELATIVE_PATHS = (
    "configs/xunce_mid_dual_g1_repair_v1.json",
    "configs/ppo_highres_frontier_stage6_v1.json",
    "scripts/run_xunce_mid_dual_g1_repair.py",
    "scripts/run_xunce_mid_dual_g1_coverage.py",
    "scripts/xunce_mid_dual_contracts.py",
    "scripts/xunce_artifact_io.py",
    "src/lunar_exploration_ppo/eval/midterm_reduced.py",
    "src/lunar_exploration_ppo/eval/standard.py",
    "src/lunar_exploration_ppo/env/standard_training.py",
    "src/lunar_exploration_ppo/env/coverage_cache.py",
    "src/lunar_exploration_ppo/env/env.py",
    "src/lunar_exploration_ppo/env/frontier.py",
    "src/lunar_exploration_ppo/env/frontier_oracle.py",
    "src/lunar_exploration_ppo/policy/cross_attention.py",
    "src/lunar_exploration_ppo/integrations/path_planner_adapter.py",
)


class G1RepairBlocked(ValueError):
    """Fail-closed error with a stable machine-readable reason."""


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _canonical_line(value: Mapping[str, object]) -> bytes:
    return _canonical_bytes(value) + b"\n"


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _exact_int(value: object, reason: str) -> int:
    if type(value) is not int:
        raise G1RepairBlocked(reason)
    return value


def _finite_float(value: object, reason: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise G1RepairBlocked(reason)
    return float(value)


def read_parent_coverage_evidence(
    results_path: str | Path,
    *,
    expected_sha256: str,
) -> dict[str, object]:
    """Read exact parent bytes and bind every selected JSON line to its SHA."""

    if not _is_sha256(expected_sha256):
        raise G1RepairBlocked("parent_results_expected_sha256_invalid")
    if not artifact_io.path_is_file(results_path):
        raise G1RepairBlocked("parent_results_missing")
    payload = artifact_io.read_bytes(results_path)
    actual_sha256 = _sha256(payload)
    if actual_sha256 != expected_sha256:
        raise G1RepairBlocked("parent_results_sha256_mismatch")

    episodes: list[dict[str, object]] = []
    non_coverage_row_count = 0
    for line_number, raw_line in enumerate(
        payload.splitlines(keepends=True),
        start=1,
    ):
        if not raw_line.strip():
            continue
        try:
            decoded = json.loads(raw_line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise G1RepairBlocked("parent_results_json_invalid") from exc
        if not isinstance(decoded, dict):
            raise G1RepairBlocked("parent_results_row_invalid")
        if decoded.get("row_kind") != "coverage_episode":
            non_coverage_row_count += 1
            continue
        if decoded.get("schema_version") != PARENT_COVERAGE_SCHEMA:
            raise G1RepairBlocked("parent_coverage_schema_invalid")
        episodes.append(
            {
                **decoded,
                "parent_line_number": line_number,
                "parent_line_size_bytes": len(raw_line),
                "parent_line_bytes_sha256": _sha256(raw_line),
                "parent_results_sha256": actual_sha256,
            }
        )

    indices = [row.get("episode_index") for row in episodes]
    if len(episodes) != 24 or indices != list(range(24)):
        raise G1RepairBlocked("parent_episode_index_incomplete_or_duplicate")
    if any(row.get("lane_id") != f"lane-{index % 8}" for index, row in enumerate(episodes)):
        raise G1RepairBlocked("parent_lane_identity_mismatch")
    if len({row.get("scenario_id") for row in episodes}) != 24:
        raise G1RepairBlocked("parent_scenario_identity_duplicate")
    for index in REPAIR_EPISODE_INDICES:
        expected = REPAIR_CASES[index]
        failure = PARENT_FAILURE_CONTRACTS[index]
        row = episodes[index]
        if (
            row.get("scenario_id") != expected["scenario_id"]
            or row.get("lane_id") != expected["lane_id"]
        ):
            raise G1RepairBlocked("parent_repair_case_identity_mismatch")
        if (
            row.get("parent_line_number") != failure["parent_line_number"]
            or row.get("denominator_cell_count")
            != failure["denominator_cell_count"]
            or row.get("denominator_sha256") != failure["denominator_sha256"]
            or row.get("initial_covered_cell_count")
            != failure["covered_cell_count"]
            or row.get("final_covered_cell_count")
            != failure["covered_cell_count"]
            or row.get("coverage") != failure["coverage"]
            or row.get("steps_executed") != 0
            or row.get("termination_reason") != "no_candidate_done"
        ):
            raise G1RepairBlocked("parent_repair_failure_contract_mismatch")
    return {
        "schema_version": "xunce-mid-dual-g1-parent-evidence/v1",
        "results_path": str(Path(results_path)).replace("\\", "/"),
        "results_size_bytes": len(payload),
        "results_sha256": actual_sha256,
        "non_coverage_row_count": non_coverage_row_count,
        "episodes": episodes,
    }


def validate_first_step_gate(
    decision: Mapping[str, object],
    step: Mapping[str, object],
    *,
    candidate_count: int,
    start_cell_xy: tuple[int, int],
) -> dict[str, object]:
    """Validate the repair-only zero-distance first observation action."""

    if candidate_count != 1:
        raise G1RepairBlocked("first_step_candidate_count_invalid")
    if (
        not isinstance(start_cell_xy, tuple)
        or len(start_cell_xy) != 2
        or any(type(value) is not int for value in start_cell_xy)
    ):
        raise G1RepairBlocked("first_step_start_cell_invalid")
    selected_index = _exact_int(
        decision.get("selected_candidate_index"),
        "first_step_selected_index_invalid",
    )
    if selected_index != 0:
        raise G1RepairBlocked("first_step_selected_index_invalid")
    selected_cell = decision.get("selected_candidate_cell_xy")
    if selected_cell != list(start_cell_xy):
        raise G1RepairBlocked("first_step_candidate_not_start")
    selected_theta = _finite_float(
        decision.get("selected_theta"),
        "first_step_selected_theta_invalid",
    )
    planned_path = step.get("planned_path_cells")
    if planned_path != [list(start_cell_xy)]:
        raise G1RepairBlocked("first_step_planned_path_not_single_start")
    path_length = _finite_float(
        step.get("path_length_m"),
        "first_step_path_invalid",
    )
    if path_length != 0.0:
        raise G1RepairBlocked("first_step_path_nonzero")
    gain = _exact_int(
        step.get("coverage_gain_cells"),
        "first_step_coverage_gain_invalid",
    )
    if gain <= 0:
        raise G1RepairBlocked("first_step_coverage_gain_nonpositive")
    if step.get("invalid_action") is not False:
        raise G1RepairBlocked("first_step_invalid_action")
    if step.get("safety_violation") is not False:
        raise G1RepairBlocked("first_step_safety_violation")
    return {
        "passed": True,
        "candidate_count": 1,
        "selected_candidate_index": 0,
        "candidate_cell_xy": list(start_cell_xy),
        "start_cell_xy": list(start_cell_xy),
        "selected_theta": selected_theta,
        "planned_path_cells": planned_path,
        "coverage_gain_cells": gain,
        "path_length_m": 0.0,
        "invalid_action": False,
        "safety_violation": False,
    }


def _validate_repair_episode(row: Mapping[str, object], index: int) -> dict[str, object]:
    expected = REPAIR_CASES[index]
    if (
        row.get("row_kind") != "repair_coverage_episode"
        or row.get("schema_version")
        != "xunce-mid-dual-g1-repair-episode/v1"
        or row.get("episode_index") != index
        or row.get("episode_id") != f"unseen24-episode-{index:02d}"
        or row.get("scenario_id") != expected["scenario_id"]
        or row.get("record_id") != expected["record_id"]
        or row.get("lane_id") != expected["lane_id"]
        or row.get("evaluation_seed") != expected["evaluation_seed"]
        or row.get("first_step_gate", {}).get("passed") is not True
    ):
        raise G1RepairBlocked("repair_episode_identity_or_gate_invalid")
    gate = row.get("first_step_gate")
    if not isinstance(gate, Mapping):
        raise G1RepairBlocked("repair_episode_identity_or_gate_invalid")
    start = gate.get("start_cell_xy")
    if (
        not isinstance(start, list)
        or len(start) != 2
        or any(type(value) is not int for value in start)
    ):
        raise G1RepairBlocked("first_step_start_cell_invalid")
    rebuilt_gate = validate_first_step_gate(
        {
            "selected_candidate_index": gate.get(
                "selected_candidate_index"
            ),
            "selected_candidate_cell_xy": gate.get("candidate_cell_xy"),
            "selected_theta": gate.get("selected_theta"),
        },
        {
            "planned_path_cells": gate.get("planned_path_cells"),
            "coverage_gain_cells": gate.get("coverage_gain_cells"),
            "path_length_m": gate.get("path_length_m"),
            "invalid_action": gate.get("invalid_action"),
            "safety_violation": gate.get("safety_violation"),
        },
        candidate_count=_exact_int(
            gate.get("candidate_count"),
            "first_step_candidate_count_invalid",
        ),
        start_cell_xy=(start[0], start[1]),
    )
    if dict(gate) != rebuilt_gate:
        raise G1RepairBlocked("first_step_gate_payload_mismatch")
    denominator = _exact_int(
        row.get("denominator_cell_count"),
        "repair_denominator_invalid",
    )
    initial = _exact_int(
        row.get("initial_covered_cell_count"),
        "repair_initial_count_invalid",
    )
    final = _exact_int(
        row.get("final_covered_cell_count"),
        "repair_final_count_invalid",
    )
    coverage = _finite_float(row.get("coverage"), "repair_coverage_invalid")
    if (
        denominator <= 0
        or not 0 <= initial <= final <= denominator
        or coverage != final / denominator
        or not _is_sha256(row.get("denominator_sha256"))
    ):
        raise G1RepairBlocked("repair_integer_coverage_binding_invalid")
    return dict(row)


def _load_config(path: str | Path) -> tuple[dict[str, Any], str]:
    if not artifact_io.path_is_file(path):
        raise G1RepairBlocked("repair_config_missing")
    payload = artifact_io.read_bytes(path)
    try:
        decoded = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise G1RepairBlocked("repair_config_invalid") from exc
    if not isinstance(decoded, dict) or decoded.get("schema_version") != CONFIG_SCHEMA:
        raise G1RepairBlocked("repair_config_invalid")
    return decoded, _sha256(payload)


def _required_mapping(value: object, reason: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise G1RepairBlocked(reason)
    return value


def _file_sha256(path: str | Path, reason: str) -> str:
    if not artifact_io.path_is_file(path):
        raise G1RepairBlocked(f"{reason}_missing")
    return _sha256(artifact_io.read_bytes(path))


def compute_repair_code_lineage() -> dict[str, object]:
    digest = hashlib.sha256()
    digest.update(b"xunce-mid-dual-g1-repair-code-lineage/v1\0")
    rows: list[dict[str, object]] = []
    for relative in REPAIR_SOURCE_RELATIVE_PATHS:
        path = _REPO_ROOT / relative
        payload = artifact_io.read_bytes(path)
        path_bytes = relative.encode("utf-8")
        digest.update(len(path_bytes).to_bytes(8, "little"))
        digest.update(path_bytes)
        digest.update(len(payload).to_bytes(8, "little"))
        digest.update(payload)
        rows.append(
            {
                "path": relative,
                "size_bytes": len(payload),
                "sha256": _sha256(payload),
            }
        )
    return {
        "schema_version": "xunce-mid-dual-g1-repair-code-lineage/v1",
        "hash_algorithm": "sha256-domain-separated-path-length-bytes/v1",
        "required_sources": rows,
        "code_sha256": digest.hexdigest(),
    }


def validate_contract(config_path: str | Path) -> dict[str, object]:
    """Validate every immutable parent, input, policy and repair binding."""

    config, config_sha256 = _load_config(config_path)
    parent = _required_mapping(config.get("parent"), "repair_parent_config_invalid")
    manifest = _required_mapping(
        config.get("scenario_manifest"),
        "repair_manifest_config_invalid",
    )
    checkpoint = _required_mapping(
        config.get("checkpoint"),
        "repair_checkpoint_config_invalid",
    )
    execution = _required_mapping(
        config.get("execution"),
        "repair_execution_config_invalid",
    )
    if (
        config.get("gate_id") != "g1"
        or config.get("runner_id") != "run_xunce_mid_dual_g1_repair/v1"
        or parent.get("formal_config_sha256")
        != "82e0d9a034bbf609af5b73a70fcfd78cc18c5f89d4c09c8e45097b441fe357a6"
        or execution.get("formal_config_sha256_for_environment")
        != parent.get("formal_config_sha256")
        or execution.get("cohort") != "unseen24"
        or execution.get("evaluation_seed_start") != 2026072700
        or execution.get("max_steps") != 128
        or execution.get("environment_success_threshold") != 0.99
    ):
        raise G1RepairBlocked("repair_config_contract_drift")

    parent_config_path = str(parent.get("config_path"))
    parent_config_sha = _file_sha256(
        parent_config_path,
        "parent_config",
    )
    if parent_config_sha != parent.get("config_file_sha256"):
        raise G1RepairBlocked("parent_config_file_sha256_mismatch")
    parent_config = artifact_io.read_json(parent_config_path)
    if (
        parent_config.get("config_sha256") != parent.get("formal_config_sha256")
        or parent_config.get("scenario_manifest", {}).get("sha256")
        != manifest.get("sha256")
        or parent_config.get("checkpoint", {}).get("sha256")
        != checkpoint.get("sha256")
        or parent_config.get("checkpoint", {}).get("policy_state_sha256")
        != checkpoint.get("policy_state_sha256")
    ):
        raise G1RepairBlocked("parent_effective_config_binding_mismatch")

    parent_evidence = read_parent_coverage_evidence(
        str(parent.get("p04_results_path")),
        expected_sha256=str(parent.get("p04_results_sha256")),
    )
    if _file_sha256(str(manifest.get("path")), "scenario_manifest") != manifest.get(
        "sha256"
    ):
        raise G1RepairBlocked("scenario_manifest_sha256_mismatch")
    if _file_sha256(str(checkpoint.get("path")), "checkpoint") != checkpoint.get(
        "sha256"
    ):
        raise G1RepairBlocked("checkpoint_sha256_mismatch")

    from lunar_exploration_ppo.env.standard_training import build_standard_catalog
    from lunar_exploration_ppo.eval.midterm_reduced import (
        build_midterm_reduced_job_plan,
        load_frozen_scenario_manifest,
    )

    catalog = build_standard_catalog(verify_hashes=True)
    frozen = load_frozen_scenario_manifest(
        bundle_root=Path(str(manifest["path"])).parent,
        expected_manifest_sha256=str(manifest["sha256"]),
    )
    plan = build_midterm_reduced_job_plan(
        catalog=catalog,
        frozen_manifest=frozen,
        cohort="unseen24",
        evaluation_seed_start=2026072700,
    )
    config_cases = config.get("repair_cases")
    if not isinstance(config_cases, list) or len(config_cases) != 3:
        raise G1RepairBlocked("repair_case_config_invalid")
    case_by_index = {
        row.get("episode_index"): row
        for row in config_cases
        if isinstance(row, Mapping)
    }
    if tuple(sorted(case_by_index)) != REPAIR_EPISODE_INDICES:
        raise G1RepairBlocked("repair_case_config_invalid")
    job_rows: list[dict[str, object]] = []
    for index in REPAIR_EPISODE_INDICES:
        expected = REPAIR_CASES[index]
        configured = case_by_index[index]
        job = plan.jobs[index]
        frozen_id = frozen.cohorts["unseen24"][index]
        parent_row = parent_evidence["episodes"][index]
        if (
            dict(configured) != {"episode_index": index, **expected}
            or job.episode_index != index
            or job.scenario_id != expected["record_id"]
            or job.evaluation_seed != expected["evaluation_seed"]
            or frozen_id != expected["scenario_id"]
            or parent_row["scenario_id"] != frozen_id
            or parent_row["lane_id"] != expected["lane_id"]
        ):
            raise G1RepairBlocked("repair_job_plan_binding_mismatch")
        proof = frozen.denominator_proofs[frozen_id]
        if (
            proof.get("coverable_cell_count")
            != PARENT_FAILURE_CONTRACTS[index]["denominator_cell_count"]
            or proof.get("coverable_mask_sha256")
            != PARENT_FAILURE_CONTRACTS[index]["denominator_sha256"]
        ):
            raise G1RepairBlocked("repair_denominator_proof_mismatch")
        job_rows.append(
            {
                "episode_index": index,
                "record_id": job.scenario_id,
                "scenario_id": frozen_id,
                "lane_id": expected["lane_id"],
                "evaluation_seed": job.evaluation_seed,
                "denominator_cell_count": proof["coverable_cell_count"],
                "denominator_sha256": proof["coverable_mask_sha256"],
            }
        )

    repair_lineage = _required_mapping(
        config.get("repair_lineage"),
        "repair_lineage_config_invalid",
    )
    required_artifacts = repair_lineage.get("required_artifacts")
    runtime_sources = repair_lineage.get("runtime_sources")
    if (
        not isinstance(required_artifacts, list)
        or not isinstance(runtime_sources, list)
    ):
        raise G1RepairBlocked("repair_lineage_config_invalid")
    artifact_rows: list[dict[str, object]] = []
    for row in required_artifacts:
        if not isinstance(row, Mapping) or not _is_sha256(row.get("sha256")):
            raise G1RepairBlocked("repair_lineage_config_invalid")
        actual = _file_sha256(str(row.get("path")), "repair_lineage_artifact")
        if actual != row["sha256"]:
            raise G1RepairBlocked("repair_lineage_artifact_sha256_mismatch")
        artifact_rows.append(dict(row))
    if (
        not any(
            row.get("name") == "repair_only_patch"
            and row.get("sha256")
            == "57fe0d4655f2d9dead34f1c437eb4905cfce1ac0de5a84aaa9dbbc19132ce819"
            for row in artifact_rows
        )
        or not any(
            row.get("name") == "pre_stage6_dirty_patch"
            and row.get("sha256")
            == "4bf00523075e30ea28bd92219d35f7c38ce32040322e26f48638516064298134"
            for row in artifact_rows
        )
    ):
        raise G1RepairBlocked("repair_lineage_v2_required_artifact_missing")
    runtime_rows: list[dict[str, object]] = []
    for row in runtime_sources:
        if not isinstance(row, Mapping) or not _is_sha256(row.get("sha256")):
            raise G1RepairBlocked("repair_runtime_source_config_invalid")
        actual = _file_sha256(
            _REPO_ROOT / str(row.get("path")),
            "repair_runtime_source",
        )
        if actual != row["sha256"]:
            raise G1RepairBlocked("repair_runtime_source_sha256_mismatch")
        runtime_rows.append({**dict(row), "actual_sha256": actual})
    expected_runtime = {
        "src/lunar_exploration_ppo/env/frontier.py": (
            "2fff448478730c5059b558fbfc771c6dae775a4c2dd7a2346293dc8c12eceeaf"
        ),
        "src/lunar_exploration_ppo/env/frontier_oracle.py": (
            "ed59309b75251152e3ac736f8f0db9c686d5b4ab09a22b01d20ef921c0fcf5e3"
        ),
    }
    if {str(row["path"]): str(row["actual_sha256"]) for row in runtime_rows} != expected_runtime:
        raise G1RepairBlocked("repair_runtime_source_set_mismatch")

    code_lineage = compute_repair_code_lineage()
    return {
        "schema_version": "xunce-mid-dual-g1-repair-contract-audit/v1",
        "status": "verified",
        "config_path": str(Path(config_path)).replace("\\", "/"),
        "config_sha256": config_sha256,
        "parent": {
            "config_file_sha256": parent_config_sha,
            "formal_config_sha256": parent["formal_config_sha256"],
            "results_sha256": parent_evidence["results_sha256"],
            "coverage_episode_count": len(parent_evidence["episodes"]),
            "non_coverage_row_count": parent_evidence[
                "non_coverage_row_count"
            ],
        },
        "scenario_manifest_sha256": manifest["sha256"],
        "checkpoint_sha256": checkpoint["sha256"],
        "policy_state_sha256": checkpoint["policy_state_sha256"],
        "repair_cases": list(REPAIR_EPISODE_INDICES),
        "jobs": job_rows,
        "repair_lineage": {
            "repair_only_patch_sha256": (
                "57fe0d4655f2d9dead34f1c437eb4905cfce1ac0de5a84aaa9dbbc19132ce819"
            ),
            "pre_stage6_dirty_patch_sha256": (
                "4bf00523075e30ea28bd92219d35f7c38ce32040322e26f48638516064298134"
            ),
            "required_artifacts": artifact_rows,
            "runtime_sources": runtime_rows,
            "runtime_sources_match_post_snapshots": True,
        },
        "code_lineage": code_lineage,
    }


def build_mixed_repair_rows(
    *,
    parent_evidence: Mapping[str, object],
    repair_episodes: Sequence[Mapping[str, object]],
    repair_results_sha256: str,
    repair_code_sha256: str,
) -> list[dict[str, object]]:
    """Replace exactly three parent failures without hiding mixed code lineage."""

    if (
        parent_evidence.get("schema_version")
        != "xunce-mid-dual-g1-parent-evidence/v1"
        or not _is_sha256(parent_evidence.get("results_sha256"))
        or not _is_sha256(repair_results_sha256)
        or not _is_sha256(repair_code_sha256)
    ):
        raise G1RepairBlocked("mixed_repair_lineage_invalid")
    parent_rows = parent_evidence.get("episodes")
    if not isinstance(parent_rows, list) or len(parent_rows) != 24:
        raise G1RepairBlocked("parent_episode_index_incomplete_or_duplicate")
    repair_by_index: dict[int, dict[str, object]] = {}
    for value in repair_episodes:
        if not isinstance(value, Mapping):
            raise G1RepairBlocked("repair_episode_schema_invalid")
        index = _exact_int(
            value.get("episode_index"),
            "repair_episode_index_invalid",
        )
        if index not in REPAIR_EPISODE_INDICES or index in repair_by_index:
            raise G1RepairBlocked("repair_episode_index_incomplete_or_duplicate")
        repair_by_index[index] = _validate_repair_episode(value, index)
    if tuple(sorted(repair_by_index)) != REPAIR_EPISODE_INDICES:
        raise G1RepairBlocked("repair_episode_index_incomplete_or_duplicate")

    merged: list[dict[str, object]] = []
    for index in range(24):
        parent = parent_rows[index]
        if not isinstance(parent, Mapping) or parent.get("episode_index") != index:
            raise G1RepairBlocked("parent_episode_index_incomplete_or_duplicate")
        if index in repair_by_index:
            episode = repair_by_index[index]
            origin = "repair_rerun"
            source_line_sha256 = _sha256(_canonical_line(episode))
            lineage = {
                "repair_results_sha256": repair_results_sha256,
                "repair_result_line_sha256": source_line_sha256,
                "repair_code_sha256": repair_code_sha256,
                "parent_replaced_line_number": parent["parent_line_number"],
                "parent_replaced_line_bytes_sha256": parent[
                    "parent_line_bytes_sha256"
                ],
            }
        else:
            episode = {
                key: value
                for key, value in parent.items()
                if not key.startswith("parent_")
            }
            origin = "parent_read_only"
            source_line_sha256 = str(parent["parent_line_bytes_sha256"])
            lineage = {
                "parent_results_sha256": parent_evidence["results_sha256"],
                "parent_line_number": parent["parent_line_number"],
                "parent_line_size_bytes": parent["parent_line_size_bytes"],
                "parent_line_bytes_sha256": source_line_sha256,
                "parent_code_sha256": parent.get("code_sha256"),
            }
        merged.append(
            {
                "row_kind": "mixed_repair_episode",
                "schema_version": MERGED_ROW_SCHEMA,
                "mixed_code_incremental_repair": True,
                "homogeneous_code_execution": False,
                "episode_index": index,
                "episode_id": f"unseen24-episode-{index:02d}",
                "scenario_id": episode["scenario_id"],
                "lane_id": episode["lane_id"],
                "result_origin": origin,
                "source_line_sha256": source_line_sha256,
                "result_sha256": _sha256(_canonical_bytes(episode)),
                "lineage": lineage,
                "episode": episode,
            }
        )
    return merged


def independently_recompute_merged(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Recompute the 24-case gate only from integer numerator/denominator pairs."""

    materialized = list(rows)
    indices = [
        row.get("episode_index") if isinstance(row, Mapping) else None
        for row in materialized
    ]
    if len(materialized) != 24 or sorted(indices) != list(range(24)):
        raise G1RepairBlocked("merged_episode_index_incomplete_or_duplicate")
    ordered = sorted(materialized, key=lambda row: int(row["episode_index"]))
    origins = [row.get("result_origin") for row in ordered]
    if origins.count("parent_read_only") != 21 or origins.count("repair_rerun") != 3:
        raise G1RepairBlocked("merged_lineage_cardinality_invalid")

    job_ids: list[str] = []
    lane_ids: list[str] = []
    coverage_values: list[float] = []
    integer_rows: list[dict[str, object]] = []
    for index, wrapper in enumerate(ordered):
        if (
            wrapper.get("schema_version") != MERGED_ROW_SCHEMA
            or wrapper.get("mixed_code_incremental_repair") is not True
            or wrapper.get("homogeneous_code_execution") is not False
            or wrapper.get("episode_index") != index
            or not isinstance(wrapper.get("episode"), Mapping)
        ):
            raise G1RepairBlocked("merged_row_schema_invalid")
        episode = wrapper["episode"]
        denominator = _exact_int(
            episode.get("denominator_cell_count"),
            "merged_denominator_invalid",
        )
        final = _exact_int(
            episode.get("final_covered_cell_count"),
            "merged_final_count_invalid",
        )
        if denominator <= 0 or not 0 <= final <= denominator:
            raise G1RepairBlocked("merged_integer_coverage_invalid")
        recomputed = final / denominator
        recorded = _finite_float(
            episode.get("coverage"),
            "merged_recorded_coverage_invalid",
        )
        if recorded != recomputed:
            raise G1RepairBlocked("merged_integer_coverage_mismatch")
        if index in REPAIR_EPISODE_INDICES:
            _validate_repair_episode(episode, index)
        job_ids.append(str(wrapper["episode_id"]))
        lane_ids.append(str(wrapper["lane_id"]))
        coverage_values.append(recomputed)
        integer_rows.append(
            {
                "episode_index": index,
                "episode_id": wrapper["episode_id"],
                "scenario_id": wrapper["scenario_id"],
                "lane_id": wrapper["lane_id"],
                "result_origin": wrapper["result_origin"],
                "final_covered_cell_count": final,
                "denominator_cell_count": denominator,
                "coverage": recomputed,
            }
        )
    statistics = g1_split_statistics(
        job_ids,
        coverage_values,
        lane_ids=lane_ids,
    )
    if statistics.get("status") == "blocked":
        raise G1RepairBlocked(
            f"g1_split_statistics_blocked:{statistics.get('blocking_reason')}"
        )
    return {
        "schema_version": RECOMPUTE_SCHEMA,
        "mixed_code_incremental_repair": True,
        "homogeneous_code_execution": False,
        "integer_recompute_passed": True,
        "coverage_values_sha256": _sha256(_canonical_bytes(integer_rows)),
        "episodes": integer_rows,
        **statistics,
    }


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    """Write a new or replaced artifact with file fsync before atomic rename."""

    artifact_io.make_dirs(path.parent)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    if artifact_io.path_exists(temporary):
        raise G1RepairBlocked("atomic_temporary_path_exists")
    artifact_io.write_bytes(temporary, payload)
    descriptor = os.open(artifact_io.windows_safe_path(temporary), os.O_RDWR)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(
        artifact_io.windows_safe_path(temporary),
        artifact_io.windows_safe_path(path),
    )


class RepairArtifactStore:
    """Per-scenario durable store with an append-only accepted-state journal."""

    def __init__(
        self,
        root: str | Path,
        *,
        immutable_config_sha256: str,
        expected_episode_lineage: Mapping[str, object],
    ) -> None:
        if not _is_sha256(immutable_config_sha256):
            raise G1RepairBlocked("immutable_config_sha256_invalid")
        if (
            not isinstance(expected_episode_lineage, Mapping)
            or set(expected_episode_lineage) != EPISODE_LINEAGE_KEYS
            or any(
                not _is_sha256(expected_episode_lineage[key])
                for key in EPISODE_LINEAGE_KEYS
            )
        ):
            raise G1RepairBlocked("expected_episode_lineage_invalid")
        self.root = Path(root)
        self.immutable_config_sha256 = immutable_config_sha256
        self.expected_episode_lineage = dict(expected_episode_lineage)
        self.state_path = self.root / "job-state.jsonl"

    def _validate_episode_lineage(
        self,
        episode: Mapping[str, object],
    ) -> None:
        if {
            key: episode.get(key) for key in EPISODE_LINEAGE_KEYS
        } != self.expected_episode_lineage:
            raise G1RepairBlocked("repair_episode_lineage_mismatch")

    def _state_rows(self) -> list[dict[str, object]]:
        if not artifact_io.path_is_file(self.state_path):
            return []
        try:
            rows = artifact_io.read_jsonl(self.state_path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise G1RepairBlocked("job_state_invalid") from exc
        seen: set[int] = set()
        for row in rows:
            index = row.get("episode_index")
            if (
                row.get("schema_version") != JOB_STATE_SCHEMA
                or row.get("status") != "accepted"
                or row.get("immutable_config_sha256")
                != self.immutable_config_sha256
                or type(index) is not int
                or index not in REPAIR_EPISODE_INDICES
                or index in seen
                or not _is_sha256(row.get("results_sha256"))
                or not _is_sha256(row.get("audit_sha256"))
            ):
                raise G1RepairBlocked("job_state_invalid")
            seen.add(index)
        return rows

    def _episode_paths(self, episode_index: int) -> tuple[Path, Path]:
        base = (
            self.root
            / "phases"
            / "p03"
            / "episodes"
            / f"e{episode_index:02d}"
        )
        return base / "results.jsonl", base / "audit.json"

    def _verify_accepted_row(self, row: Mapping[str, object]) -> None:
        index = int(row["episode_index"])
        results_path, audit_path = self._episode_paths(index)
        if not artifact_io.path_is_file(results_path):
            raise G1RepairBlocked("accepted_episode_results_missing")
        if _sha256(artifact_io.read_bytes(results_path)) != row["results_sha256"]:
            raise G1RepairBlocked("accepted_episode_results_sha256_mismatch")
        try:
            result_rows = artifact_io.read_jsonl(results_path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise G1RepairBlocked("accepted_episode_results_invalid") from exc
        if (
            not result_rows
            or any(result.get("episode_index") != index for result in result_rows)
        ):
            raise G1RepairBlocked("accepted_episode_results_invalid")
        _validate_repair_episode(result_rows[0], index)
        self._validate_episode_lineage(result_rows[0])
        if not artifact_io.path_is_file(audit_path):
            raise G1RepairBlocked("accepted_episode_audit_missing")
        if _sha256(artifact_io.read_bytes(audit_path)) != row["audit_sha256"]:
            raise G1RepairBlocked("accepted_episode_audit_sha256_mismatch")

    def accepted_episode_indices(self) -> tuple[int, ...]:
        rows = self._state_rows()
        for row in rows:
            self._verify_accepted_row(row)
        return tuple(sorted(int(row["episode_index"]) for row in rows))

    def accept_episode(
        self,
        *,
        episode_index: int,
        rows: Sequence[Mapping[str, object]],
    ) -> dict[str, object]:
        if episode_index not in REPAIR_EPISODE_INDICES:
            raise G1RepairBlocked("accepted_episode_index_invalid")
        materialized = [dict(row) for row in rows]
        if (
            not materialized
            or materialized[0].get("episode_index") != episode_index
            or any(row.get("episode_index") != episode_index for row in materialized)
        ):
            raise G1RepairBlocked("accepted_episode_rows_invalid")
        _validate_repair_episode(materialized[0], episode_index)
        self._validate_episode_lineage(materialized[0])
        results_payload = b"".join(_canonical_line(row) for row in materialized)
        results_sha256 = _sha256(results_payload)
        audit = {
            "schema_version": "xunce-mid-dual-g1-repair-episode-audit/v1",
            "episode_index": episode_index,
            "row_count": len(materialized),
            "results_sha256": results_sha256,
            "immutable_config_sha256": self.immutable_config_sha256,
            "status": "complete",
        }
        audit_payload = json.dumps(
            audit,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        ).encode("utf-8") + b"\n"
        audit_sha256 = _sha256(audit_payload)

        states = self._state_rows()
        accepted = {
            int(row["episode_index"]): row
            for row in states
        }
        if episode_index in accepted:
            self._verify_accepted_row(accepted[episode_index])
            if (
                accepted[episode_index]["results_sha256"] != results_sha256
                or accepted[episode_index]["audit_sha256"] != audit_sha256
            ):
                raise G1RepairBlocked("accepted_episode_resume_payload_mismatch")
            return {
                "status": "already_accepted",
                "episode_index": episode_index,
                "results_sha256": results_sha256,
                "audit_sha256": audit_sha256,
            }

        results_path, audit_path = self._episode_paths(episode_index)
        if artifact_io.path_exists(results_path) or artifact_io.path_exists(audit_path):
            if (
                not artifact_io.path_is_file(results_path)
                or not artifact_io.path_is_file(audit_path)
                or artifact_io.read_bytes(results_path) != results_payload
                or artifact_io.read_bytes(audit_path) != audit_payload
            ):
                raise G1RepairBlocked("unaccepted_episode_artifact_conflict")
        else:
            _atomic_write_bytes(results_path, results_payload)
            _atomic_write_bytes(audit_path, audit_payload)

        state_row = {
            "schema_version": JOB_STATE_SCHEMA,
            "status": "accepted",
            "episode_index": episode_index,
            "immutable_config_sha256": self.immutable_config_sha256,
            "results_path": str(results_path.relative_to(self.root)).replace(
                "\\",
                "/",
            ),
            "results_sha256": results_sha256,
            "audit_path": str(audit_path.relative_to(self.root)).replace(
                "\\",
                "/",
            ),
            "audit_sha256": audit_sha256,
        }
        existing_payload = (
            artifact_io.read_bytes(self.state_path)
            if artifact_io.path_is_file(self.state_path)
            else b""
        )
        _atomic_write_bytes(
            self.state_path,
            existing_payload + _canonical_line(state_row),
        )
        return {
            "status": "accepted",
            "episode_index": episode_index,
            "results_sha256": results_sha256,
            "audit_sha256": audit_sha256,
        }


def _jsonable(value: object) -> object:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        if isinstance(value, float) and not math.isfinite(value):
            raise G1RepairBlocked("runtime_diagnostics_nonfinite")
        return value
    if hasattr(value, "item"):
        return _jsonable(value.item())
    raise G1RepairBlocked(
        f"runtime_diagnostics_not_json:{type(value).__name__}"
    )


def _write_once_or_verify(path: Path, payload: bytes, reason: str) -> None:
    if artifact_io.path_exists(path):
        if not artifact_io.path_is_file(path) or artifact_io.read_bytes(path) != payload:
            raise G1RepairBlocked(reason)
        return
    _atomic_write_bytes(path, payload)


def _json_artifact_bytes(value: Mapping[str, object]) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        indent=2,
    ).encode("utf-8") + b"\n"


def _validated_output_root(config: Mapping[str, object], run_id: str) -> Path:
    if not re.fullmatch(r"g1-repair-[A-Za-z0-9._-]{1,80}", run_id):
        raise G1RepairBlocked("repair_run_id_invalid")
    output_base = Path(str(config.get("output_base"))).resolve()
    output_root = (output_base / run_id).resolve()
    if (
        output_base.drive.upper() != "D:"
        or output_root.parent != output_base
        or output_root.name != run_id
    ):
        raise G1RepairBlocked("repair_output_root_invalid")
    return output_root


def prepare_run_root(
    *,
    config_path: str | Path,
    run_id: str,
    contract_audit: Mapping[str, object],
) -> tuple[Path, str]:
    config, static_config_sha256 = _load_config(config_path)
    output_root = _validated_output_root(config, run_id)
    artifact_io.make_dirs(output_root)
    code_lineage = contract_audit.get("code_lineage")
    if not isinstance(code_lineage, Mapping):
        raise G1RepairBlocked("contract_code_lineage_missing")
    contract_payload = _json_artifact_bytes(contract_audit)
    lineage_payload = _json_artifact_bytes(code_lineage)
    effective = {
        "schema_version": "xunce-mid-dual-g1-repair-effective-config/v1",
        "gate_id": "g1",
        "runner_id": "run_xunce_mid_dual_g1_repair/v1",
        "run_id": run_id,
        "output_root": str(output_root).replace("\\", "/"),
        "static_config_path": str(Path(config_path).resolve()).replace("\\", "/"),
        "static_config_sha256": static_config_sha256,
        "formal_config_sha256_for_environment": config["execution"][
            "formal_config_sha256_for_environment"
        ],
        "contract_audit_sha256": _sha256(contract_payload),
        "repair_code_sha256": code_lineage["code_sha256"],
        "parent_results_sha256": config["parent"]["p04_results_sha256"],
        "scenario_manifest_sha256": config["scenario_manifest"]["sha256"],
        "checkpoint_sha256": config["checkpoint"]["sha256"],
        "policy_state_sha256": config["checkpoint"]["policy_state_sha256"],
        "repair_cases": config["repair_cases"],
        "mixed_code_incremental_repair": True,
        "homogeneous_code_execution": False,
        "replay_status": "not_run_pending",
        "figure_status": "pending",
    }
    effective_payload = _json_artifact_bytes(effective)
    effective_sha256 = _sha256(effective_payload)
    _write_once_or_verify(
        output_root / "config.json",
        effective_payload,
        "repair_effective_config_drift",
    )
    _write_once_or_verify(
        output_root / "contract-audit.json",
        contract_payload,
        "repair_contract_audit_drift",
    )
    _write_once_or_verify(
        output_root / "source-lineage.json",
        lineage_payload,
        "repair_source_lineage_drift",
    )

    lineage = _required_mapping(
        config.get("repair_lineage"),
        "repair_lineage_config_invalid",
    )
    artifacts = lineage.get("required_artifacts")
    assert isinstance(artifacts, list)
    for row in artifacts:
        assert isinstance(row, Mapping)
        destination = (
            output_root
            / "lineage"
            / "evidence"
            / Path(str(row["path"])).name
        )
        source_payload = artifact_io.read_bytes(str(row["path"]))
        if _sha256(source_payload) != row["sha256"]:
            raise G1RepairBlocked("repair_lineage_artifact_sha256_mismatch")
        _write_once_or_verify(
            destination,
            source_payload,
            "repair_lineage_copy_drift",
        )
    return output_root, effective_sha256


def _load_runtime_context(config: Mapping[str, object]):
    from lunar_exploration_ppo.configs.stage6 import (
        SafetyContract,
        load_stage6_config,
    )
    from lunar_exploration_ppo.env.standard_training import build_standard_catalog
    from lunar_exploration_ppo.eval.midterm_reduced import (
        build_midterm_reduced_job_plan,
        load_frozen_scenario_manifest,
    )

    catalog = build_standard_catalog(verify_hashes=True)
    manifest = config["scenario_manifest"]
    frozen = load_frozen_scenario_manifest(
        bundle_root=Path(str(manifest["path"])).parent,
        expected_manifest_sha256=str(manifest["sha256"]),
    )
    plan = build_midterm_reduced_job_plan(
        catalog=catalog,
        frozen_manifest=frozen,
        cohort="unseen24",
        evaluation_seed_start=2026072700,
    )
    stage6 = load_stage6_config(
        _REPO_ROOT / "configs/ppo_highres_frontier_stage6_v1.json"
    )
    safety_contract = SafetyContract.from_stage6_config(stage6)
    return catalog, frozen, plan, safety_contract


def _run_one_episode(
    *,
    episode_index: int,
    catalog: object,
    frozen: object,
    plan: object,
    safety_contract: object,
    policy: object,
    formal_config_sha256: str,
    repair_code_sha256: str,
    static_config_sha256: str,
) -> list[dict[str, object]]:
    import torch
    from lunar_exploration_ppo.env.standard_training import StandardEvaluationEnv
    from lunar_exploration_ppo.eval.standard import (
        build_standard_decision_record,
        select_standard_evaluation_actions,
    )
    from lunar_exploration_ppo.policy.cross_attention import (
        batch_policy_observations,
    )

    job = plan.jobs[episode_index]
    expected = REPAIR_CASES[episode_index]
    frozen_id = frozen.cohorts["unseen24"][episode_index]
    proof = frozen.denominator_proofs[frozen_id]
    denominator = _exact_int(
        proof.get("coverable_cell_count"),
        "runtime_denominator_invalid",
    )
    denominator_sha256 = proof.get("coverable_mask_sha256")
    if (
        job.episode_index != episode_index
        or job.scenario_id != expected["record_id"]
        or job.evaluation_seed != expected["evaluation_seed"]
        or frozen_id != expected["scenario_id"]
        or not _is_sha256(denominator_sha256)
    ):
        raise G1RepairBlocked("runtime_job_plan_binding_mismatch")

    environment = StandardEvaluationEnv(
        catalog,
        split=job.split,
        scenario_ids=(job.scenario_id,),
        safety_contract=safety_contract,
        config_sha256=formal_config_sha256,
    )
    started = time.perf_counter()
    try:
        observation = environment.reset()
        reset_diagnostics = environment.last_reset_diagnostics
        if reset_diagnostics is None:
            raise G1RepairBlocked("runtime_reset_diagnostics_missing")
        metadata = environment.coverage_metadata
        if (
            environment.coverable_cell_count != denominator
            or metadata.get("coverable_cell_count") != denominator
            or metadata.get("sha256") != denominator_sha256
            or metadata.get("algorithm_id")
            != "exact_reachable_safe_pose_range_los/v1"
            or metadata.get("exact") is not True
        ):
            raise G1RepairBlocked("runtime_denominator_metadata_mismatch")
        initial_rate = float(reset_diagnostics.coverage_rate)
        initial_count = int(round(initial_rate * denominator))
        if (
            not 0 <= initial_count <= denominator
            or not math.isclose(
                initial_rate,
                initial_count / denominator,
                rel_tol=0.0,
                abs_tol=1.0e-15,
            )
        ):
            raise G1RepairBlocked("runtime_initial_coverage_not_integer_exact")
        if environment.is_done or not environment.needs_policy:
            raise G1RepairBlocked("runtime_repair_reset_not_actionable")

        coverage_curve = [initial_rate]
        cumulative_path_curve = [0.0]
        cumulative_path = 0.0
        covered_count = initial_count
        decision_rows: list[dict[str, object]] = []
        step_rows: list[dict[str, object]] = []
        first_step_gate: dict[str, object] | None = None
        invalid_action_count = 0
        safety_violation_count = 0
        planner_failure_counts: dict[str, int] = {}
        for step_index in range(128):
            if environment.is_done:
                break
            candidate_count = int(environment.current_action_set.candidate_count)
            start_cell = (
                int(environment.pose.cell.x),
                int(environment.pose.cell.y),
            )
            batch = batch_policy_observations((observation,), device="cuda")
            with torch.inference_mode():
                output = policy(batch)
            action = select_standard_evaluation_actions(
                method="ppo_policy",
                observations=(observation,),
                evaluation_seeds=(job.evaluation_seed,),
                policy_output=output,
            )[0]
            decision = build_standard_decision_record(
                method="ppo_policy",
                sequence_index=step_index,
                episode_index=episode_index,
                step_index=step_index,
                worker_index=episode_index % 8,
                observation=observation,
                action=action,
                selector_inputs={
                    "observation_batch": batch,
                    "policy_output": output,
                },
                policy_batch_row_index=0,
            )
            try:
                selected_cell = environment.current_action_set.cells[
                    action.candidate_index
                ]
            except IndexError as exc:
                raise G1RepairBlocked(
                    "runtime_selected_candidate_cell_missing"
                ) from exc
            decision_row = {
                "row_kind": "repair_decision",
                "schema_version": "xunce-mid-dual-g1-repair-decision/v1",
                "episode_index": episode_index,
                "episode_id": f"unseen24-episode-{episode_index:02d}",
                "scenario_id": frozen_id,
                "record_id": job.scenario_id,
                "lane_id": expected["lane_id"],
                "evaluation_seed": job.evaluation_seed,
                "step_index": step_index,
                "worker_index": episode_index % 8,
                "selected_candidate_index": action.candidate_index,
                "selected_candidate_cell_xy": [
                    int(selected_cell.x),
                    int(selected_cell.y),
                ],
                "selected_theta": float(action.target_theta),
                "trace": decision,
            }
            result = environment.step(action)
            planned_path = [
                [int(cell.x), int(cell.y)]
                for cell in result.diagnostics.planned_path_cells
            ]
            executed_path = float(result.diagnostics.path_length_m)
            cumulative_path += executed_path
            gain = int(result.coverage_gain_cells)
            covered_count += gain
            derived_coverage = covered_count / denominator
            if (
                gain < 0
                or covered_count > denominator
                or not math.isclose(
                    float(result.coverage_rate),
                    derived_coverage,
                    rel_tol=0.0,
                    abs_tol=1.0e-15,
                )
            ):
                raise G1RepairBlocked("runtime_step_integer_coverage_mismatch")
            failure_reason = result.diagnostics.planner.get(
                "failure_reason",
                "none",
            )
            if failure_reason != "none":
                planner_failure_counts[str(failure_reason)] = (
                    planner_failure_counts.get(str(failure_reason), 0) + 1
                )
            invalid_action_count += int(result.diagnostics.invalid_action)
            safety_violation_count += int(
                result.diagnostics.safety_violation
            )
            step_row = {
                "row_kind": "repair_planner_call",
                "schema_version": "xunce-mid-dual-g1-repair-planner-call/v1",
                "episode_index": episode_index,
                "episode_id": f"unseen24-episode-{episode_index:02d}",
                "scenario_id": frozen_id,
                "record_id": job.scenario_id,
                "lane_id": expected["lane_id"],
                "evaluation_seed": job.evaluation_seed,
                "step_index": step_index,
                "decision_sha256": decision["decision_sha256"],
                "selected_candidate_index": action.candidate_index,
                "selected_candidate_cell_xy": [
                    int(selected_cell.x),
                    int(selected_cell.y),
                ],
                "selected_theta": float(action.target_theta),
                "planned_path_cells": planned_path,
                "coverage_gain_cells": gain,
                "coverage_rate": derived_coverage,
                "path_length_m": executed_path,
                "cumulative_path_length_m": cumulative_path,
                "invalid_action": bool(result.diagnostics.invalid_action),
                "safety_violation": bool(
                    result.diagnostics.safety_violation
                ),
                "done": bool(result.done),
                "termination_reason": (
                    str(result.reason) if result.done else "none"
                ),
                "planner_diagnostics": _jsonable(
                    result.diagnostics.planner
                ),
                "execution_diagnostics": _jsonable(
                    result.diagnostics.execution
                ),
                "sensor_diagnostics": _jsonable(result.diagnostics.sensor),
                "frontier_diagnostics": _jsonable(
                    result.diagnostics.frontier
                ),
            }
            if step_index == 0:
                first_step_gate = validate_first_step_gate(
                    decision_row,
                    step_row,
                    candidate_count=candidate_count,
                    start_cell_xy=start_cell,
                )
            decision_rows.append(decision_row)
            step_rows.append(step_row)
            coverage_curve.append(derived_coverage)
            cumulative_path_curve.append(cumulative_path)
            observation = result.observation
            if result.done:
                break
        if (
            not environment.is_done
            or first_step_gate is None
            or not step_rows
            or step_rows[-1]["done"] is not True
        ):
            raise G1RepairBlocked("runtime_episode_did_not_terminate")
        final_count = covered_count
        final_coverage = final_count / denominator
        episode_row = {
            "row_kind": "repair_coverage_episode",
            "schema_version": "xunce-mid-dual-g1-repair-episode/v1",
            "gate_id": "g1",
            "runner_id": "run_xunce_mid_dual_g1_repair/v1",
            "mixed_code_incremental_repair": True,
            "episode_index": episode_index,
            "episode_id": f"unseen24-episode-{episode_index:02d}",
            "scenario_id": frozen_id,
            "record_id": job.scenario_id,
            "lane_id": expected["lane_id"],
            "evaluation_seed": job.evaluation_seed,
            "denominator_source": (
                "reachable_observable_free_highres_cells/v1"
            ),
            "denominator_algorithm": (
                "exact_reachable_safe_pose_range_los/v1"
            ),
            "denominator_sha256": denominator_sha256,
            "denominator_cell_count": denominator,
            "initial_covered_cell_count": initial_count,
            "final_covered_cell_count": final_count,
            "coverage": final_coverage,
            "coverage_curve": coverage_curve,
            "cumulative_path_length_curve_m": cumulative_path_curve,
            "steps_executed": len(step_rows),
            "termination_reason": str(environment.terminal_reason),
            "invalid_action_count": invalid_action_count,
            "safety_violation_count": safety_violation_count,
            "masked_action_count": 0,
            "planner_failure_counts": dict(sorted(planner_failure_counts.items())),
            "elapsed_ms": (time.perf_counter() - started) * 1000.0,
            "first_step_gate": first_step_gate,
            "reset_diagnostics": _jsonable(reset_diagnostics),
            "coverage_metadata": _jsonable(metadata),
            "checkpoint_sha256": (
                "35e04c86f9f973af028fb08f0175d42ab45378d09e1f2b96ee6aad6e4c12b5b5"
            ),
            "policy_state_sha256": (
                "3123e6adde99be41e3bd5cc2f3843068e5416892395926d759c2c6a243ffd381"
            ),
            "scenario_manifest_sha256": (
                "37a3e639b4decfaf660ff66408c5c7109808c5f44aa22c18e98cd901050bb54a"
            ),
            "formal_config_sha256": formal_config_sha256,
            "repair_static_config_sha256": static_config_sha256,
            "repair_code_sha256": repair_code_sha256,
        }
        _validate_repair_episode(episode_row, episode_index)
        return [episode_row, *decision_rows, *step_rows]
    finally:
        environment.close()


def _accepted_episode_rows(
    store: RepairArtifactStore,
) -> tuple[list[dict[str, object]], bytes]:
    indices = store.accepted_episode_indices()
    if indices != REPAIR_EPISODE_INDICES:
        raise G1RepairBlocked("repair_primary_incomplete")
    all_rows: list[dict[str, object]] = []
    episodes: list[dict[str, object]] = []
    for index in indices:
        results_path, _ = store._episode_paths(index)
        rows = artifact_io.read_jsonl(results_path)
        _validate_repair_episode(rows[0], index)
        episodes.append(rows[0])
        all_rows.extend(rows)
    payload = b"".join(_canonical_line(row) for row in all_rows)
    return episodes, payload


def finalize_primary(
    *,
    config: Mapping[str, object],
    output_root: Path,
    store: RepairArtifactStore,
    repair_code_sha256: str,
) -> dict[str, object]:
    parent = config["parent"]
    parent_before = read_parent_coverage_evidence(
        parent["p04_results_path"],
        expected_sha256=parent["p04_results_sha256"],
    )
    episodes, raw_payload = _accepted_episode_rows(store)
    raw_path = output_root / "raw" / "repair-results.jsonl"
    _write_once_or_verify(
        raw_path,
        raw_payload,
        "repair_aggregate_raw_drift",
    )
    raw_sha256 = _sha256(raw_payload)
    merged = build_mixed_repair_rows(
        parent_evidence=parent_before,
        repair_episodes=episodes,
        repair_results_sha256=raw_sha256,
        repair_code_sha256=repair_code_sha256,
    )
    merged_payload = b"".join(_canonical_line(row) for row in merged)
    merged_path = output_root / "merged-results.jsonl"
    _write_once_or_verify(
        merged_path,
        merged_payload,
        "repair_merged_results_drift",
    )
    recompute = independently_recompute_merged(merged)
    recompute_payload = _json_artifact_bytes(recompute)
    recompute_path = output_root / "independent-recompute.json"
    _write_once_or_verify(
        recompute_path,
        recompute_payload,
        "repair_independent_recompute_drift",
    )
    parent_after_sha = _file_sha256(
        parent["p04_results_path"],
        "parent_results",
    )
    if parent_after_sha != parent_before["results_sha256"]:
        raise G1RepairBlocked("parent_results_changed_during_repair")
    manifest = {
        "schema_version": "xunce-mid-dual-g1-repair-manifest/v1",
        "status": "primary_complete",
        "mixed_code_incremental_repair": True,
        "homogeneous_code_execution": False,
        "parent_results_sha256": parent_after_sha,
        "repair_results_sha256": raw_sha256,
        "merged_results_sha256": _sha256(merged_payload),
        "independent_recompute_sha256": _sha256(recompute_payload),
        "accepted_episode_indices": list(REPAIR_EPISODE_INDICES),
        "midterm_reduced_passed": recompute["midterm_reduced_passed"],
        "final_threshold_reduced_passed": recompute[
            "final_threshold_reduced_passed"
        ],
        "coverage_80_count": recompute["coverage_80_count"],
        "coverage_99_count": recompute["coverage_99_count"],
        "mean": recompute["mean"],
        "replay": {"status": "not_run_pending", "scenario_count": 3},
        "figures": {
            "status": "pending",
            "renderer": "independent_python_renderer/v1",
        },
    }
    manifest_payload = _json_artifact_bytes(manifest)
    _write_once_or_verify(
        output_root / "manifest.json",
        manifest_payload,
        "repair_manifest_drift",
    )
    return manifest


def run_dry_run(
    *,
    config_path: str | Path,
    run_id: str,
) -> dict[str, object]:
    audit = validate_contract(config_path)
    output_root, effective_sha256 = prepare_run_root(
        config_path=config_path,
        run_id=run_id,
        contract_audit=audit,
    )
    dry_run = {
        "schema_version": "xunce-mid-dual-g1-repair-dry-run/v1",
        "status": "ready",
        "run_id": run_id,
        "output_root": str(output_root).replace("\\", "/"),
        "effective_config_sha256": effective_sha256,
        "repair_cases": audit["jobs"],
        "estimated_wall_minutes": {"lower": 100, "upper": 150},
        "formal_execution_started": False,
    }
    _write_once_or_verify(
        output_root / "dry-run.json",
        _json_artifact_bytes(dry_run),
        "repair_dry_run_drift",
    )
    return dry_run


def run_formal(
    *,
    config_path: str | Path,
    run_id: str,
) -> dict[str, object]:
    audit = validate_contract(config_path)
    config, static_config_sha256 = _load_config(config_path)
    output_root, effective_sha256 = prepare_run_root(
        config_path=config_path,
        run_id=run_id,
        contract_audit=audit,
    )
    store = RepairArtifactStore(
        output_root,
        immutable_config_sha256=effective_sha256,
        expected_episode_lineage={
            "checkpoint_sha256": config["checkpoint"]["sha256"],
            "policy_state_sha256": config["checkpoint"][
                "policy_state_sha256"
            ],
            "scenario_manifest_sha256": config["scenario_manifest"][
                "sha256"
            ],
            "formal_config_sha256": config["parent"][
                "formal_config_sha256"
            ],
            "repair_code_sha256": audit["code_lineage"]["code_sha256"],
            "repair_static_config_sha256": static_config_sha256,
        },
    )
    accepted = set(store.accepted_episode_indices())
    pending = [
        index for index in REPAIR_EPISODE_INDICES if index not in accepted
    ]
    if pending:
        from lunar_exploration_ppo.eval.midterm_reduced import (
            load_midterm_update80_policy,
        )

        catalog, frozen, plan, safety_contract = _load_runtime_context(config)
        policy = load_midterm_update80_policy()
        policy.eval()
        for index in pending:
            rows = _run_one_episode(
                episode_index=index,
                catalog=catalog,
                frozen=frozen,
                plan=plan,
                safety_contract=safety_contract,
                policy=policy,
                formal_config_sha256=config["parent"][
                    "formal_config_sha256"
                ],
                repair_code_sha256=audit["code_lineage"]["code_sha256"],
                static_config_sha256=static_config_sha256,
            )
            store.accept_episode(episode_index=index, rows=rows)
    return finalize_primary(
        config=config,
        output_root=output_root,
        store=store,
        repair_code_sha256=audit["code_lineage"]["code_sha256"],
    )


def _default_run_id() -> str:
    return datetime.now().strftime("g1-repair-%Y%m%d-%H%M%S-cst")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="G1 three-case mixed-code incremental repair runner",
    )
    parser.add_argument(
        "--mode",
        choices=("validate-contract", "dry-run", "formal"),
        required=True,
    )
    parser.add_argument(
        "--config",
        default=str(
            _REPO_ROOT / "configs/xunce_mid_dual_g1_repair_v1.json"
        ),
    )
    parser.add_argument("--run-id")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.mode == "validate-contract":
            result = validate_contract(args.config)
        else:
            run_id = args.run_id or _default_run_id()
            if args.mode == "dry-run":
                result = run_dry_run(
                    config_path=args.config,
                    run_id=run_id,
                )
            else:
                result = run_formal(
                    config_path=args.config,
                    run_id=run_id,
                )
    except G1RepairBlocked as exc:
        print(
            json.dumps(
                {"status": "blocked", "blocking_reason": str(exc)},
                ensure_ascii=False,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


__all__ = [
    "G1RepairBlocked",
    "REPAIR_CASES",
    "REPAIR_EPISODE_INDICES",
    "RepairArtifactStore",
    "build_mixed_repair_rows",
    "independently_recompute_merged",
    "read_parent_coverage_evidence",
    "run_dry_run",
    "run_formal",
    "validate_contract",
    "validate_first_step_gate",
]


if __name__ == "__main__":
    raise SystemExit(main())
