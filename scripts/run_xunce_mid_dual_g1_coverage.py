"""Run the frozen update80 G1 coverage qualification experiment.

The runner owns orchestration and evidence serialization only.  Formal scene
selection and policy inference stay inside the verified Task 3/Task 4
adapters.  Every formal coverage value is reconstructed from integer covered
and denominator cell counts before it can enter gate statistics.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import os
import platform
import re
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path, PureWindowsPath
from typing import Any

import xunce_artifact_io as artifact_io
from xunce_artifact_paths import (
    MID_DUAL_CONFIG,
    MID_DUAL_MANIFEST,
    MID_DUAL_PHASE_STATE,
    artifact_path,
)
from xunce_mid_dual_artifacts import MidDualRunStore
from xunce_mid_dual_contracts import (
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    FINAL_COVERAGE_THRESHOLD,
    G1_EPISODES_PER_FORMAL_SPLIT,
    G1_LANE_SIZES,
    MID_COVERAGE_THRESHOLD,
    SCALE_PROFILE,
    evaluate_g1_split,
)


G1_CONFIG_SCHEMA_VERSION = "xunce-mid-dual-g1-coverage-config/v1"
G1_EFFECTIVE_CONFIG_SCHEMA_VERSION = "xunce-mid-dual-g1-effective-config/v1"
G1_INPUT_AUDIT_SCHEMA_VERSION = "xunce-mid-dual-g1-input-audit/v1"
G1_COVERAGE_EPISODE_SCHEMA_VERSION = (
    "xunce-mid-dual-g1-coverage-episode/v1"
)
G1_DECISION_SCHEMA_VERSION = "xunce-mid-dual-g1-decision/v1"
G1_PLANNER_CALL_SCHEMA_VERSION = "xunce-mid-dual-g1-planner-call/v1"
G1_SUMMARY_SCHEMA_VERSION = "xunce-mid-dual-g1-summary/v1"
G1_SPLIT_SUMMARY_SCHEMA_VERSION = "xunce-mid-dual-g1-split-summary/v1"
G1_ROUTING_SCHEMA_VERSION = "xunce-mid-dual-g1-routing/v1"
G1_REPLAY_SCHEMA_VERSION = "xunce-mid-dual-g1-replay-audit/v1"
G1_REPORT_RENDERER = "xunce-mid-dual-g1-report-renderer/v1"
G1_REPAIR_LINEAGE_SCHEMA_VERSION = "xunce-mid-dual-g1-repair-lineage/v1"
G1_RUNNER_ID = "run_xunce_mid_dual_g1_coverage/v1"
G1_GATE_ID = "g1"
G1_OUTPUT_BASE = "D:/xunce/out/mid_dual/g1"
G1_REQUIRED_PHASE_IDS = (
    "p01",
    "p02",
    "p03",
    "p04",
    "p05",
    "p06",
    "p07",
)
G1_REQUIRED_PHASES = (
    "preflight",
    "validation_dry_run",
    "test_q24",
    "unseen24",
    "replay3",
    "recompute",
    "finalize",
)
G1_PHASE_NAMES = dict(zip(G1_REQUIRED_PHASE_IDS, G1_REQUIRED_PHASES, strict=True))
G1_FORMAL_SPLITS = ("test_q24", "unseen24")
G1_MODES = ("preflight", "dry-run", "formal", "test-c-confirmation")

UPDATE80_CHECKPOINT_PATH = (
    "D:/xunce/out/ppo_frontier/"
    "s6-standard-single-r1-20260724T000124Z/s6/"
    "checkpoints/seed-20260716/update-00000080/checkpoint.pt"
)
UPDATE80_CHECKPOINT_SHA256 = (
    "35e04c86f9f973af028fb08f0175d42ab45378d09e1f2b96ee6aad6e4c12b5b5"
)
UPDATE80_POLICY_STATE_SHA256 = (
    "3123e6adde99be41e3bd5cc2f3843068e5416892395926d759c2c6a243ffd381"
)
DENOMINATOR_SOURCE = "reachable_observable_free_highres_cells/v1"
DENOMINATOR_ALGORITHM = "exact_reachable_safe_pose_range_los/v1"
FROZEN_MANIFEST_SCHEMA = "mid-dual-scenario-freeze/v1"

_REPO_ROOT = Path(__file__).resolve().parents[1]
_CANONICAL_CONFIG_PATH = (
    _REPO_ROOT / "configs/xunce_mid_dual_g1_coverage_v1.json"
)
_STAGE6_CONFIG_PATH = _REPO_ROOT / "configs/ppo_highres_frontier_stage6_v1.json"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")
_WINDOWS_RESERVED = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{index}" for index in range(1, 10)),
    *(f"LPT{index}" for index in range(1, 10)),
}

G1_REQUIRED_SOURCE_RELATIVE_PATHS = (
    "configs/xunce_mid_dual_g1_coverage_v1.json",
    "configs/ppo_highres_frontier_stage6_v1.json",
    "scripts/run_xunce_mid_dual_g1_coverage.py",
    "scripts/xunce_mid_dual_contracts.py",
    "scripts/xunce_mid_dual_artifacts.py",
    "scripts/xunce_artifact_io.py",
    "scripts/xunce_artifact_paths.py",
    "src/lunar_exploration_ppo/eval/midterm_reduced.py",
    "src/lunar_exploration_ppo/eval/standard.py",
    "src/lunar_exploration_ppo/env/coverage_cache.py",
    "src/lunar_exploration_ppo/env/standard_training.py",
    "src/lunar_exploration_ppo/env/env.py",
    "src/lunar_exploration_ppo/integrations/path_planner_adapter.py",
)

_SCHEMAS = {
    "effective_config": G1_EFFECTIVE_CONFIG_SCHEMA_VERSION,
    "input_audit": G1_INPUT_AUDIT_SCHEMA_VERSION,
    "coverage_episode": G1_COVERAGE_EPISODE_SCHEMA_VERSION,
    "decision": G1_DECISION_SCHEMA_VERSION,
    "planner_call": G1_PLANNER_CALL_SCHEMA_VERSION,
    "summary": G1_SUMMARY_SCHEMA_VERSION,
    "routing": G1_ROUTING_SCHEMA_VERSION,
    "replay": G1_REPLAY_SCHEMA_VERSION,
    "report_renderer": G1_REPORT_RENDERER,
}

G1_COVERAGE_EPISODE_KEYS = frozenset(
    {
        "row_kind",
        "schema_version",
        "gate_id",
        "runner_id",
        "phase_id",
        "phase_name",
        "scale_profile",
        "run_id",
        "split",
        "episode_id",
        "episode_index",
        "scenario_id",
        "lane_id",
        "source_sha256",
        "config_sha256",
        "input_sha256",
        "code_sha256",
        "scenario_manifest_sha256",
        "checkpoint_sha256",
        "policy_state_sha256",
        "denominator_source",
        "denominator_algorithm",
        "denominator_sha256",
        "denominator_cell_count",
        "initial_covered_cell_count",
        "final_covered_cell_count",
        "coverage",
        "elapsed_ms",
        "steps_executed",
        "termination_reason",
        "safety_violation_count",
        "masked_action_count",
    }
)
G1_TRACE_ROW_KEYS = frozenset(
    {
        "row_kind",
        "schema_version",
        "gate_id",
        "runner_id",
        "phase_id",
        "phase_name",
        "scale_profile",
        "run_id",
        "split",
        "episode_id",
        "episode_index",
        "scenario_id",
        "lane_id",
        "step_index",
        "source_sha256",
        "config_sha256",
        "input_sha256",
        "code_sha256",
        "scenario_manifest_sha256",
        "checkpoint_sha256",
        "policy_state_sha256",
        "trace",
    }
)
G1_SPLIT_SUMMARY_KEYS = frozenset(
    {
        "schema_version",
        "split",
        "status",
        "sample_count",
        "mean",
        "median",
        "sample_stddev",
        "min",
        "max",
        "coverage_80_count",
        "coverage_99_count",
        "bootstrap_ci",
        "safety_violation_count",
        "masked_action_count",
        "safety_clean",
        "masked_action_clean",
        "g1_coverage_80_passed",
        "g1_coverage_99_passed",
    }
)
G1_SUMMARY_KEYS = frozenset(
    {
        "schema_version",
        "gate_id",
        "runner_id",
        "scale_profile",
        "run_id",
        "mode",
        "config_sha256",
        "input_sha256",
        "code_sha256",
        "scenario_manifest_sha256",
        "checkpoint_sha256",
        "policy_state_sha256",
        "status",
        "formal_evidence_eligible",
        "formal_episode_count",
        "split_order",
        "unseen_execution_status",
        "splits",
        "safety_clean",
        "masked_action_clean",
        "replay",
        "g1_coverage_80_passed",
        "g1_coverage_99_passed",
        "failure_reasons",
        "blockers",
    }
)


class G1Blocked(ValueError):
    """Stable fail-closed reason for missing or inconsistent G1 evidence."""

    def __init__(self, reason: str) -> None:
        if not isinstance(reason, str) or not reason:
            reason = "g1_blocked"
        self.reason = reason
        super().__init__(reason)


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _bytes_sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None


def _require_sha256(value: object, reason: str) -> str:
    if not _is_sha256(value):
        raise G1Blocked(reason)
    return str(value)


def _exact_nonnegative_int(value: object, reason: str) -> int:
    if type(value) is not int or value < 0:
        raise G1Blocked(reason)
    return value


def _finite_nonnegative(value: object, reason: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0.0
    ):
        raise G1Blocked(reason)
    return float(value)


def _validated_run_id(value: object) -> str:
    if (
        not isinstance(value, str)
        or _RUN_ID_RE.fullmatch(value) is None
        or value in {".", ".."}
        or ".." in value
        or value.endswith((".", " "))
        or "/" in value
        or "\\" in value
        or value.split(".", 1)[0].upper() in _WINDOWS_RESERVED
    ):
        raise G1Blocked("run_id_invalid")
    return value


def _deep_copy_json(value: Mapping[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(dict(value), ensure_ascii=False))


def validate_g1_config_payload(payload: object) -> dict[str, Any]:
    """Accept exactly the committed fixed-update80 base config."""

    top_keys = {
        "schema_version",
        "gate_id",
        "runner_id",
        "scale_profile",
        "output_base",
        "required_phase_ids",
        "required_phases",
        "checkpoint",
        "scenario_manifest",
        "denominator",
        "execution",
        "bootstrap",
        "schemas",
    }
    if not isinstance(payload, Mapping) or set(payload) != top_keys:
        raise G1Blocked("g1_config_drift")
    expected_checkpoint = {
        "update": 80,
        "path": UPDATE80_CHECKPOINT_PATH,
        "sha256": UPDATE80_CHECKPOINT_SHA256,
        "policy_state_sha256": UPDATE80_POLICY_STATE_SHA256,
        "device": "cuda",
        "dtype": "float32",
    }
    expected_manifest = {
        "schema_version": FROZEN_MANIFEST_SCHEMA,
        "path_contract": "absolute_d_drive_manifest_argument/v1",
        "sha256_binding": "runtime_exact_manifest_bytes/v1",
    }
    expected_denominator = {
        "source": DENOMINATOR_SOURCE,
        "algorithm": DENOMINATOR_ALGORITHM,
        "integer_count_binding": True,
    }
    expected_execution = {
        "worker_count": 8,
        "lane_sizes": list(G1_LANE_SIZES),
        "episodes_per_formal_split": G1_EPISODES_PER_FORMAL_SPLIT,
        "max_steps": 128,
        "environment_success_threshold": 0.99,
        "test_q24_evaluation_seed_start": 2026072600,
        "unseen24_evaluation_seed_start": 2026072700,
        "test_c24_evaluation_seed_start": 2026072800,
        "validation3_evaluation_seed_start": 2026072500,
        "replay3_evaluation_seed_start": 2026072900,
    }
    expected_bootstrap = {
        "unit": "episode",
        "seed": BOOTSTRAP_SEED,
        "resamples": BOOTSTRAP_RESAMPLES,
        "confidence_level": 0.95,
    }
    if (
        payload.get("schema_version") != G1_CONFIG_SCHEMA_VERSION
        or payload.get("gate_id") != G1_GATE_ID
        or payload.get("runner_id") != G1_RUNNER_ID
        or payload.get("scale_profile") != SCALE_PROFILE
        or payload.get("output_base") != G1_OUTPUT_BASE
        or payload.get("required_phase_ids") != list(G1_REQUIRED_PHASE_IDS)
        or payload.get("required_phases") != list(G1_REQUIRED_PHASES)
        or payload.get("checkpoint") != expected_checkpoint
        or payload.get("scenario_manifest") != expected_manifest
        or payload.get("denominator") != expected_denominator
        or payload.get("execution") != expected_execution
        or payload.get("bootstrap") != expected_bootstrap
        or payload.get("schemas") != _SCHEMAS
    ):
        raise G1Blocked("g1_config_drift")
    return _deep_copy_json(payload)


def load_g1_config(path: str | Path) -> dict[str, Any]:
    try:
        payload = artifact_io.read_json(path)
    except (OSError, ValueError, TypeError) as exc:
        raise G1Blocked(f"g1_config_unreadable:{type(exc).__name__}") from exc
    return validate_g1_config_payload(payload)


def _formal_episode_id(split: str, episode_index: int) -> str:
    return f"{split}-episode-{episode_index:02d}"


def _phase_for_split(split: str) -> tuple[str, str]:
    if split in {"test_q24", "test_c24"}:
        return "p03", split
    if split == "unseen24":
        return "p04", split
    raise G1Blocked("g1_split_invalid")


def make_g1_coverage_episode_row(
    *,
    run_id: str,
    split: str,
    episode_index: int,
    scenario_id: str,
    config_sha256: str,
    input_sha256: str,
    code_sha256: str,
    scenario_manifest_sha256: str,
    denominator_sha256: str,
    denominator_cell_count: int,
    initial_covered_cell_count: int,
    final_covered_cell_count: int,
    elapsed_ms: float,
    steps_executed: int,
    termination_reason: str,
    safety_violation_count: int,
    masked_action_count: int,
) -> dict[str, object]:
    """Construct one exact, integer-bound coverage episode row."""

    run_id = _validated_run_id(run_id)
    phase_id, phase_name = _phase_for_split(split)
    index = _exact_nonnegative_int(episode_index, "g1_episode_index_invalid")
    if index >= G1_EPISODES_PER_FORMAL_SPLIT:
        raise G1Blocked("g1_episode_index_invalid")
    if not isinstance(scenario_id, str) or not scenario_id:
        raise G1Blocked("g1_scenario_id_invalid")
    for value, reason in (
        (config_sha256, "g1_config_sha256_invalid"),
        (input_sha256, "g1_input_sha256_invalid"),
        (code_sha256, "g1_code_sha256_invalid"),
        (scenario_manifest_sha256, "g1_manifest_sha256_invalid"),
        (denominator_sha256, "g1_denominator_sha256_invalid"),
    ):
        _require_sha256(value, reason)
    denominator = _exact_nonnegative_int(
        denominator_cell_count,
        "g1_denominator_count_invalid",
    )
    initial = _exact_nonnegative_int(
        initial_covered_cell_count,
        "g1_initial_covered_count_invalid",
    )
    final = _exact_nonnegative_int(
        final_covered_cell_count,
        "g1_final_covered_count_invalid",
    )
    if denominator <= 0 or not 0 <= initial <= final <= denominator:
        raise G1Blocked("g1_integer_coverage_binding_invalid")
    steps = _exact_nonnegative_int(steps_executed, "g1_steps_invalid")
    safety = _exact_nonnegative_int(
        safety_violation_count,
        "g1_safety_count_invalid",
    )
    masked = _exact_nonnegative_int(
        masked_action_count,
        "g1_masked_action_count_invalid",
    )
    if not isinstance(termination_reason, str) or not termination_reason:
        raise G1Blocked("g1_termination_reason_invalid")
    row = {
        "row_kind": "coverage_episode",
        "schema_version": G1_COVERAGE_EPISODE_SCHEMA_VERSION,
        "gate_id": G1_GATE_ID,
        "runner_id": G1_RUNNER_ID,
        "phase_id": phase_id,
        "phase_name": phase_name,
        "scale_profile": SCALE_PROFILE,
        "run_id": run_id,
        "split": split,
        "episode_id": _formal_episode_id(split, index),
        "episode_index": index,
        "scenario_id": scenario_id,
        "lane_id": f"lane-{index % 8}",
        "source_sha256": code_sha256,
        "config_sha256": config_sha256,
        "input_sha256": input_sha256,
        "code_sha256": code_sha256,
        "scenario_manifest_sha256": scenario_manifest_sha256,
        "checkpoint_sha256": UPDATE80_CHECKPOINT_SHA256,
        "policy_state_sha256": UPDATE80_POLICY_STATE_SHA256,
        "denominator_source": DENOMINATOR_SOURCE,
        "denominator_algorithm": DENOMINATOR_ALGORITHM,
        "denominator_sha256": denominator_sha256,
        "denominator_cell_count": denominator,
        "initial_covered_cell_count": initial,
        "final_covered_cell_count": final,
        "coverage": final / denominator,
        "elapsed_ms": _finite_nonnegative(elapsed_ms, "g1_elapsed_ms_invalid"),
        "steps_executed": steps,
        "termination_reason": termination_reason,
        "safety_violation_count": safety,
        "masked_action_count": masked,
    }
    if set(row) != G1_COVERAGE_EPISODE_KEYS:
        raise AssertionError("internal G1 coverage row schema drifted")
    return row


def _validate_coverage_row(
    row: object,
    *,
    expected_split: str,
) -> dict[str, object]:
    if not isinstance(row, Mapping) or set(row) != G1_COVERAGE_EPISODE_KEYS:
        raise G1Blocked("g1_coverage_row_schema_invalid")
    copied = dict(row)
    index = _exact_nonnegative_int(
        copied["episode_index"],
        "g1_episode_index_invalid",
    )
    denominator = _exact_nonnegative_int(
        copied["denominator_cell_count"],
        "g1_denominator_count_invalid",
    )
    initial = _exact_nonnegative_int(
        copied["initial_covered_cell_count"],
        "g1_initial_covered_count_invalid",
    )
    final = _exact_nonnegative_int(
        copied["final_covered_cell_count"],
        "g1_final_covered_count_invalid",
    )
    coverage = _finite_nonnegative(copied["coverage"], "g1_coverage_invalid")
    _finite_nonnegative(copied["elapsed_ms"], "g1_elapsed_ms_invalid")
    _exact_nonnegative_int(copied["steps_executed"], "g1_steps_invalid")
    _exact_nonnegative_int(
        copied["safety_violation_count"],
        "g1_safety_count_invalid",
    )
    _exact_nonnegative_int(
        copied["masked_action_count"],
        "g1_masked_action_count_invalid",
    )
    if (
        copied["row_kind"] != "coverage_episode"
        or copied["schema_version"] != G1_COVERAGE_EPISODE_SCHEMA_VERSION
        or copied["gate_id"] != G1_GATE_ID
        or copied["runner_id"] != G1_RUNNER_ID
        or copied["scale_profile"] != SCALE_PROFILE
        or copied["split"] != expected_split
        or copied["phase_name"] != expected_split
        or copied["phase_id"] != _phase_for_split(expected_split)[0]
        or copied["episode_id"] != _formal_episode_id(expected_split, index)
        or copied["lane_id"] != f"lane-{index % 8}"
        or copied["checkpoint_sha256"] != UPDATE80_CHECKPOINT_SHA256
        or copied["policy_state_sha256"] != UPDATE80_POLICY_STATE_SHA256
        or copied["denominator_source"] != DENOMINATOR_SOURCE
        or copied["denominator_algorithm"] != DENOMINATOR_ALGORITHM
        or not isinstance(copied["scenario_id"], str)
        or not copied["scenario_id"]
        or not isinstance(copied["termination_reason"], str)
        or not copied["termination_reason"]
        or denominator <= 0
        or not 0 <= initial <= final <= denominator
        or not 0.0 <= coverage <= 1.0
        or coverage != final / denominator
    ):
        raise G1Blocked("g1_coverage_row_binding_invalid")
    for field_name in (
        "source_sha256",
        "config_sha256",
        "input_sha256",
        "code_sha256",
        "scenario_manifest_sha256",
        "checkpoint_sha256",
        "policy_state_sha256",
        "denominator_sha256",
    ):
        _require_sha256(copied[field_name], f"g1_{field_name}_invalid")
    if copied["source_sha256"] != copied["code_sha256"]:
        raise G1Blocked("g1_source_code_sha256_mismatch")
    return copied


def recompute_g1_split_summary(
    rows: Sequence[Mapping[str, object]],
    *,
    expected_split: str,
) -> dict[str, object]:
    """Recompute one exact 24-row split from integer coverage evidence."""

    if expected_split not in {"test_q24", "unseen24", "test_c24"}:
        raise G1Blocked("g1_split_invalid")
    materialized = [
        _validate_coverage_row(row, expected_split=expected_split)
        for row in rows
    ]
    if len(materialized) != G1_EPISODES_PER_FORMAL_SPLIT:
        raise G1Blocked("g1_formal_episode_count_mismatch")
    if {int(row["episode_index"]) for row in materialized} != set(range(24)):
        raise G1Blocked("g1_episode_index_set_mismatch")
    if len({str(row["episode_id"]) for row in materialized}) != 24:
        raise G1Blocked("g1_episode_id_duplicate")
    if len({str(row["scenario_id"]) for row in materialized}) != 24:
        raise G1Blocked("g1_scenario_id_duplicate")
    ordered = sorted(materialized, key=lambda row: int(row["episode_index"]))
    first = ordered[0]
    lineage_fields = (
        "run_id",
        "config_sha256",
        "input_sha256",
        "code_sha256",
        "scenario_manifest_sha256",
    )
    if any(
        row[field] != first[field]
        for row in ordered
        for field in lineage_fields
    ):
        raise G1Blocked("g1_row_lineage_mismatch")
    statistics = evaluate_g1_split(
        [str(row["episode_id"]) for row in ordered],
        [float(row["coverage"]) for row in ordered],
        lane_ids=[str(row["lane_id"]) for row in ordered],
    )
    if statistics.get("status") == "blocked":
        raise G1Blocked(
            f"g1_split_statistics_blocked:{statistics.get('blocking_reason')}"
        )
    safety_count = sum(int(row["safety_violation_count"]) for row in ordered)
    masked_count = sum(int(row["masked_action_count"]) for row in ordered)
    safety_clean = safety_count == 0
    masked_clean = masked_count == 0
    midterm = (
        statistics["midterm_reduced_passed"] is True
        and safety_clean
        and masked_clean
    )
    final = (
        statistics["final_threshold_reduced_passed"] is True
        and safety_clean
        and masked_clean
    )
    summary = {
        "schema_version": G1_SPLIT_SUMMARY_SCHEMA_VERSION,
        "split": expected_split,
        "status": "passed" if midterm else "failed",
        "sample_count": 24,
        "mean": statistics["mean"],
        "median": statistics["median"],
        "sample_stddev": statistics["sample_stddev"],
        "min": statistics["min"],
        "max": statistics["max"],
        "coverage_80_count": statistics["coverage_80_count"],
        "coverage_99_count": statistics["coverage_99_count"],
        "bootstrap_ci": statistics["bootstrap_ci"],
        "safety_violation_count": safety_count,
        "masked_action_count": masked_count,
        "safety_clean": safety_clean,
        "masked_action_clean": masked_clean,
        "g1_coverage_80_passed": midterm,
        "g1_coverage_99_passed": final,
    }
    if set(summary) != G1_SPLIT_SUMMARY_KEYS:
        raise AssertionError("internal G1 split summary schema drifted")
    return summary


def _coverage_rows_for_split(
    rows: Sequence[Mapping[str, object]],
    split: str,
) -> list[dict[str, object]]:
    return [
        dict(row)
        for row in rows
        if isinstance(row, Mapping)
        and row.get("row_kind") == "coverage_episode"
        and row.get("split") == split
    ]


def _summary_lineage(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, str]:
    if not rows:
        return {
            "run_id": "",
            "config_sha256": "",
            "input_sha256": "",
            "code_sha256": "",
            "scenario_manifest_sha256": "",
        }
    first = rows[0]
    return {
        key: str(first[key])
        for key in (
            "run_id",
            "config_sha256",
            "input_sha256",
            "code_sha256",
            "scenario_manifest_sha256",
        )
    }


def _build_summary(
    *,
    test_q24: Mapping[str, object],
    unseen24: Mapping[str, object] | None,
    lineage: Mapping[str, str],
    mode: str,
    replay: Mapping[str, object] | None,
) -> dict[str, object]:
    unseen_ran = unseen24 is not None
    midterm = bool(
        unseen_ran
        and test_q24["g1_coverage_80_passed"] is True
        and unseen24["g1_coverage_80_passed"] is True
    )
    final = bool(
        unseen_ran
        and test_q24["g1_coverage_99_passed"] is True
        and unseen24["g1_coverage_99_passed"] is True
    )
    safety_clean = bool(
        test_q24["safety_clean"] is True
        and (unseen24 is None or unseen24["safety_clean"] is True)
    )
    masked_clean = bool(
        test_q24["masked_action_clean"] is True
        and (unseen24 is None or unseen24["masked_action_clean"] is True)
    )
    reasons: list[str] = []
    if test_q24["g1_coverage_80_passed"] is not True:
        reasons.append("test_q24_coverage_80_failed")
    elif unseen24 is not None and unseen24["g1_coverage_80_passed"] is not True:
        reasons.append("unseen24_coverage_80_failed")
    if not safety_clean:
        reasons.append("g1_safety_not_clean")
    if not masked_clean:
        reasons.append("g1_masked_action_not_clean")
    summary = {
        "schema_version": G1_SUMMARY_SCHEMA_VERSION,
        "gate_id": G1_GATE_ID,
        "runner_id": G1_RUNNER_ID,
        "scale_profile": SCALE_PROFILE,
        "run_id": lineage["run_id"],
        "mode": mode,
        "config_sha256": lineage["config_sha256"],
        "input_sha256": lineage["input_sha256"],
        "code_sha256": lineage["code_sha256"],
        "scenario_manifest_sha256": lineage["scenario_manifest_sha256"],
        "checkpoint_sha256": UPDATE80_CHECKPOINT_SHA256,
        "policy_state_sha256": UPDATE80_POLICY_STATE_SHA256,
        "status": "passed" if midterm else "failed",
        "formal_evidence_eligible": True,
        "formal_episode_count": 48 if unseen_ran else 24,
        "split_order": list(G1_FORMAL_SPLITS),
        "unseen_execution_status": (
            "complete"
            if unseen_ran
            else "skipped_due_to_test_q24_failure"
        ),
        "splits": {
            "test_q24": dict(test_q24),
            "unseen24": dict(unseen24) if unseen24 is not None else None,
        },
        "safety_clean": safety_clean,
        "masked_action_clean": masked_clean,
        "replay": dict(replay) if replay is not None else None,
        "g1_coverage_80_passed": midterm,
        "g1_coverage_99_passed": final,
        "failure_reasons": reasons,
        "blockers": [],
    }
    if set(summary) != G1_SUMMARY_KEYS:
        raise AssertionError("internal G1 summary schema drifted")
    return summary


def recompute_g1_summary(
    *,
    test_q24_rows: Sequence[Mapping[str, object]],
    unseen24_rows: Sequence[Mapping[str, object]],
    mode: str = "formal",
    replay: Mapping[str, object] | None = None,
) -> dict[str, object]:
    q_summary = recompute_g1_split_summary(
        test_q24_rows,
        expected_split="test_q24",
    )
    unseen_summary = recompute_g1_split_summary(
        unseen24_rows,
        expected_split="unseen24",
    )
    return _build_summary(
        test_q24=q_summary,
        unseen24=unseen_summary,
        lineage=_summary_lineage(test_q24_rows),
        mode=mode,
        replay=replay,
    )


def execute_formal_gate(
    evaluate_split: Callable[[str], Sequence[Mapping[str, object]]],
) -> dict[str, object]:
    """Enforce Test-Q24 qualification before any Unseen-24 invocation."""

    q_rows = [dict(row) for row in evaluate_split("test_q24")]
    q_coverage = _coverage_rows_for_split(q_rows, "test_q24")
    q_summary = recompute_g1_split_summary(
        q_coverage,
        expected_split="test_q24",
    )
    if q_summary["g1_coverage_80_passed"] is not True:
        return {
            "rows": q_rows,
            "evaluated_splits": ["test_q24"],
            "summary": _build_summary(
                test_q24=q_summary,
                unseen24=None,
                lineage=_summary_lineage(q_coverage),
                mode="formal",
                replay=None,
            ),
        }
    unseen_rows = [dict(row) for row in evaluate_split("unseen24")]
    unseen_coverage = _coverage_rows_for_split(unseen_rows, "unseen24")
    unseen_summary = recompute_g1_split_summary(
        unseen_coverage,
        expected_split="unseen24",
    )
    return {
        "rows": [*q_rows, *unseen_rows],
        "evaluated_splits": ["test_q24", "unseen24"],
        "summary": _build_summary(
            test_q24=q_summary,
            unseen24=unseen_summary,
            lineage=_summary_lineage(q_coverage),
            mode="formal",
            replay=None,
        ),
    }


def run_validation_dry_run(
    scenario_ids: Sequence[str],
    executor: Callable[[Sequence[str]], Sequence[Mapping[str, object]]],
) -> dict[str, object]:
    """Execute exactly the frozen Validation-3 cohort, once and in order."""

    frozen = tuple(scenario_ids)
    if (
        len(frozen) != 3
        or len(set(frozen)) != 3
        or any(
            not isinstance(value, str)
            or not value.startswith("validation/")
            for value in frozen
        )
    ):
        raise G1Blocked("validation3_frozen_cohort_invalid")
    results = tuple(executor(frozen))
    if (
        len(results) != 3
        or any(not isinstance(row, Mapping) for row in results)
        or [row.get("scenario_id") for row in results] != list(frozen)
    ):
        raise G1Blocked("validation3_execution_mismatch")
    return {
        "schema_version": "xunce-mid-dual-g1-validation-dry-run-audit/v1",
        "status": "passed",
        "scenario_count": 3,
        "scenario_ids": list(frozen),
        "trace_sha256": _canonical_sha256(results),
        "results": [dict(row) for row in results],
    }


def validate_test_c_request(
    *,
    mode: str,
    repair_lineage: str | Path | None,
    current_code_sha256: str,
    current_config_sha256: str,
) -> dict[str, object]:
    if mode != "test-c-confirmation":
        raise G1Blocked("test_c_explicit_mode_required")
    if repair_lineage is None:
        raise G1Blocked("repair_lineage_required")
    path = Path(repair_lineage)
    if not path.is_absolute() or not artifact_io.path_is_file(path):
        raise G1Blocked("repair_lineage_required")
    try:
        payload = artifact_io.read_json(path)
    except (OSError, ValueError, TypeError) as exc:
        raise G1Blocked("repair_lineage_invalid") from exc
    required = {
        "schema_version",
        "repair_id",
        "parent_run_id",
        "code_sha256",
        "config_sha256",
    }
    if (
        set(payload) != required
        or payload.get("schema_version") != G1_REPAIR_LINEAGE_SCHEMA_VERSION
        or not isinstance(payload.get("repair_id"), str)
        or not payload["repair_id"]
        or not isinstance(payload.get("parent_run_id"), str)
        or not payload["parent_run_id"]
        or not _is_sha256(payload.get("code_sha256"))
        or not _is_sha256(payload.get("config_sha256"))
    ):
        raise G1Blocked("repair_lineage_invalid")
    if (
        payload["code_sha256"] == current_code_sha256
        and payload["config_sha256"] == current_config_sha256
    ):
        raise G1Blocked("repair_lineage_not_new")
    return {
        **payload,
        "repair_lineage_sha256": _bytes_sha256(artifact_io.read_bytes(path)),
    }


def _validate_replay_trace(trace: object) -> tuple[dict[str, object], ...]:
    if not isinstance(trace, Sequence) or isinstance(trace, (str, bytes)):
        raise G1Blocked("replay_schema_invalid")
    rows: list[dict[str, object]] = []
    for row in trace:
        if (
            not isinstance(row, Mapping)
            or set(row)
            != {
                "scenario_id",
                "actions",
                "coverage_curve",
                "termination_reason",
            }
            or not isinstance(row["scenario_id"], str)
            or not row["scenario_id"]
            or not isinstance(row["actions"], list)
            or not isinstance(row["coverage_curve"], list)
            or not row["coverage_curve"]
            or not isinstance(row["termination_reason"], str)
            or not row["termination_reason"]
        ):
            raise G1Blocked("replay_schema_invalid")
        for action in row["actions"]:
            if (
                not isinstance(action, Mapping)
                or set(action)
                != {"selected_candidate_index", "selected_theta"}
                or type(action["selected_candidate_index"]) is not int
                or action["selected_candidate_index"] < 0
                or isinstance(action["selected_theta"], bool)
                or not isinstance(action["selected_theta"], (int, float))
                or not math.isfinite(float(action["selected_theta"]))
            ):
                raise G1Blocked("replay_action_invalid")
        for value in row["coverage_curve"]:
            coverage = _finite_nonnegative(value, "replay_coverage_invalid")
            if coverage > 1.0:
                raise G1Blocked("replay_coverage_invalid")
        rows.append(dict(row))
    if (
        len(rows) != 3
        or len({str(row["scenario_id"]) for row in rows}) != 3
    ):
        raise G1Blocked("replay3_cohort_invalid")
    return tuple(rows)


def validate_g1_replay(
    reference: Sequence[Mapping[str, object]],
    replay: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    first = _validate_replay_trace(reference)
    second = _validate_replay_trace(replay)
    if _canonical_bytes(first) != _canonical_bytes(second):
        raise G1Blocked("replay_mismatch")
    return {
        "schema_version": G1_REPLAY_SCHEMA_VERSION,
        "status": "passed",
        "passed": True,
        "scenario_count": 3,
        "scenario_ids": [str(row["scenario_id"]) for row in first],
        "reference_sha256": _canonical_sha256(first),
        "replay_sha256": _canonical_sha256(second),
        "actions_match": True,
        "coverage_curves_match": True,
        "termination_match": True,
    }


def preflight_scenario_manifest(
    manifest_path: str | Path,
    *,
    loader: Callable[[str | Path], object],
) -> dict[str, object]:
    """Short-circuit missing input before a CUDA/checkpoint loader can run."""

    if not artifact_io.path_is_file(manifest_path):
        return {
            "status": "blocked",
            "blocking_reason": "scenario_manifest_missing",
        }
    try:
        loaded = loader(manifest_path)
    except Exception as exc:
        return {
            "status": "blocked",
            "blocking_reason": (
                f"scenario_manifest_invalid:{type(exc).__name__}"
            ),
        }
    return {"status": "verified", "manifest": loaded}


def run_midterm_reduced_evaluation(**kwargs: object) -> object:
    """Late-bound Task 4 adapter; keeps missing-manifest preflight CUDA-free."""

    from lunar_exploration_ppo.eval.midterm_reduced import (
        run_midterm_reduced_evaluation as implementation,
    )

    return implementation(**kwargs)


def _evaluate_formal_cohort(
    *,
    catalog: object,
    frozen_bundle_root: str | Path,
    frozen_manifest_sha256: str,
    cohort: str,
    evaluation_seed_start: int,
    safety_contract: object,
    config_sha256: str,
    resource_guard: Callable[[str], None] | None = None,
) -> object:
    """Delegate formal identity, update80 loading and inference to Task 4."""

    return run_midterm_reduced_evaluation(
        catalog=catalog,
        frozen_bundle_root=frozen_bundle_root,
        frozen_manifest_sha256=frozen_manifest_sha256,
        cohort=cohort,
        evaluation_seed_start=evaluation_seed_start,
        bootstrap_resamples=BOOTSTRAP_RESAMPLES,
        bootstrap_seed=BOOTSTRAP_SEED,
        safety_contract=safety_contract,
        config_sha256=config_sha256,
        resource_guard=resource_guard,
    )


def _required_source_paths() -> tuple[Path, ...]:
    paths = tuple(_REPO_ROOT / relative for relative in G1_REQUIRED_SOURCE_RELATIVE_PATHS)
    missing = [
        relative
        for relative, path in zip(
            G1_REQUIRED_SOURCE_RELATIVE_PATHS,
            paths,
            strict=True,
        )
        if not artifact_io.path_is_file(path)
    ]
    if missing:
        raise G1Blocked(f"g1_required_source_missing:{missing[0]}")
    return paths


def compute_g1_code_lineage() -> dict[str, object]:
    """Hash the exact manifest-bound source bytes with path domain separation."""

    digest = hashlib.sha256()
    digest.update(b"xunce-mid-dual-g1-code-lineage/v1\0")
    rows: list[dict[str, object]] = []
    for relative, path in zip(
        G1_REQUIRED_SOURCE_RELATIVE_PATHS,
        _required_source_paths(),
        strict=True,
    ):
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
                "sha256": _bytes_sha256(payload),
            }
        )
    return {
        "schema_version": "xunce-mid-dual-g1-code-lineage/v1",
        "hash_algorithm": "sha256-domain-separated-path-length-bytes/v1",
        "required_sources": rows,
        "code_sha256": digest.hexdigest(),
    }


def _validate_d_manifest_path(path: str | Path) -> Path:
    raw = str(path)
    windows = PureWindowsPath(raw)
    if (
        not windows.is_absolute()
        or windows.drive.upper() != "D:"
        or windows.name != "manifest.json"
        or any(part in {"", ".", ".."} for part in windows.parts[1:])
    ):
        raise G1Blocked("scenario_manifest_path_invalid")
    return Path(raw)


def _load_frozen_manifest(
    manifest_path: Path,
) -> tuple[object, dict[str, Any], str]:
    from lunar_exploration_ppo.eval.midterm_reduced import (
        load_frozen_scenario_manifest,
    )

    before = artifact_io.read_bytes(manifest_path)
    manifest_sha256 = _bytes_sha256(before)
    try:
        raw = json.loads(
            before.decode("utf-8"),
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"nonfinite:{value}")
            ),
        )
    except (UnicodeDecodeError, ValueError, TypeError) as exc:
        raise G1Blocked("scenario_manifest_invalid") from exc
    if not isinstance(raw, dict):
        raise G1Blocked("scenario_manifest_invalid")
    try:
        frozen = load_frozen_scenario_manifest(
            bundle_root=manifest_path.parent,
            expected_manifest_sha256=manifest_sha256,
        )
    except Exception as exc:
        raise G1Blocked(
            f"scenario_manifest_invalid:{type(exc).__name__}"
        ) from exc
    after = artifact_io.read_bytes(manifest_path)
    if before != after:
        raise G1Blocked("scenario_manifest_bytes_drift")
    return frozen, raw, manifest_sha256


def _build_input_audit(
    *,
    manifest_path: Path,
    frozen: object,
    raw_manifest: Mapping[str, Any],
    manifest_sha256: str,
) -> dict[str, object]:
    cohorts = getattr(frozen, "cohorts", None)
    proofs = getattr(frozen, "denominator_proofs", None)
    if not isinstance(cohorts, Mapping) or not isinstance(proofs, Mapping):
        raise G1Blocked("scenario_manifest_adapter_invalid")
    formal_jobs: list[dict[str, object]] = []
    for split in G1_FORMAL_SPLITS:
        scenario_ids = cohorts.get(split)
        if (
            not isinstance(scenario_ids, Sequence)
            or isinstance(scenario_ids, (str, bytes))
            or len(scenario_ids) != 24
        ):
            raise G1Blocked(f"{split}_frozen_cohort_invalid")
        for index, scenario_id in enumerate(scenario_ids):
            proof = proofs.get(scenario_id)
            if (
                not isinstance(scenario_id, str)
                or not isinstance(proof, Mapping)
                or proof.get("coverage_denominator_source")
                != DENOMINATOR_SOURCE
                or proof.get("coverage_denominator_algorithm")
                != DENOMINATOR_ALGORITHM
                or proof.get("exact") is not True
                or not _is_sha256(proof.get("coverable_mask_sha256"))
                or type(proof.get("coverable_cell_count")) is not int
                or proof["coverable_cell_count"] <= 0
            ):
                raise G1Blocked("g1_denominator_proof_invalid")
            formal_jobs.append(
                {
                    "split": split,
                    "scenario_id": scenario_id,
                    "episode_id": _formal_episode_id(split, index),
                    "episode_index": index,
                    "lane_id": f"lane-{index % 8}",
                    "denominator_sha256": proof["coverable_mask_sha256"],
                    "denominator_cell_count": proof["coverable_cell_count"],
                    "denominator_source": DENOMINATOR_SOURCE,
                    "denominator_algorithm": DENOMINATOR_ALGORITHM,
                }
            )
    expected_auxiliary = {
        "test_c24": 24,
        "validation3": 3,
        "replay3": 3,
    }
    auxiliary: dict[str, list[str]] = {}
    for name, expected_count in expected_auxiliary.items():
        values = cohorts.get(name)
        if (
            not isinstance(values, Sequence)
            or isinstance(values, (str, bytes))
            or len(values) != expected_count
            or len(set(values)) != expected_count
            or any(not isinstance(value, str) or not value for value in values)
        ):
            raise G1Blocked(f"{name}_frozen_cohort_invalid")
        auxiliary[name] = list(values)
    audit = {
        "schema_version": G1_INPUT_AUDIT_SCHEMA_VERSION,
        "gate_id": G1_GATE_ID,
        "runner_id": G1_RUNNER_ID,
        "scale_profile": SCALE_PROFILE,
        "status": "verified",
        "scenario_manifest_path": str(manifest_path).replace("\\", "/"),
        "scenario_manifest_schema_version": FROZEN_MANIFEST_SCHEMA,
        "scenario_manifest_sha256": manifest_sha256,
        "scenario_manifest_canonical_sha256": _canonical_sha256(raw_manifest),
        "policy_blind_attestation_sha256": _canonical_sha256(
            raw_manifest.get("policy_blind_attestation")
        ),
        "checkpoint_sha256": UPDATE80_CHECKPOINT_SHA256,
        "policy_state_sha256": UPDATE80_POLICY_STATE_SHA256,
        "denominator_source": DENOMINATOR_SOURCE,
        "denominator_algorithm": DENOMINATOR_ALGORITHM,
        "formal_split_order": list(G1_FORMAL_SPLITS),
        "formal_jobs": formal_jobs,
        "test_c24_scenario_ids": auxiliary["test_c24"],
        "validation3_scenario_ids": auxiliary["validation3"],
        "replay3_scenario_ids": auxiliary["replay3"],
    }
    if len(formal_jobs) != 48:
        raise G1Blocked("g1_formal_input_count_mismatch")
    return audit


def _missing_input_audit(manifest_path: Path, reason: str) -> dict[str, object]:
    return {
        "schema_version": G1_INPUT_AUDIT_SCHEMA_VERSION,
        "gate_id": G1_GATE_ID,
        "runner_id": G1_RUNNER_ID,
        "scale_profile": SCALE_PROFILE,
        "status": "blocked",
        "blocking_reason": reason,
        "scenario_manifest_path": str(manifest_path).replace("\\", "/"),
        "scenario_manifest_schema_version": FROZEN_MANIFEST_SCHEMA,
        "scenario_manifest_sha256": None,
        "checkpoint_sha256": UPDATE80_CHECKPOINT_SHA256,
        "policy_state_sha256": UPDATE80_POLICY_STATE_SHA256,
        "denominator_source": DENOMINATOR_SOURCE,
        "denominator_algorithm": DENOMINATOR_ALGORITHM,
        "formal_split_order": list(G1_FORMAL_SPLITS),
        "formal_jobs": [],
        "test_c24_scenario_ids": [],
        "validation3_scenario_ids": [],
        "replay3_scenario_ids": [],
    }


def _effective_config(
    *,
    base_config: Mapping[str, Any],
    base_config_sha256: str,
    run_id: str,
    mode: str,
    manifest_path: Path,
    manifest_sha256: str | None,
    input_audit: Mapping[str, object],
    code_lineage: Mapping[str, object],
    repair_lineage: Mapping[str, object] | None,
) -> dict[str, object]:
    output_root = str(Path(G1_OUTPUT_BASE) / run_id).replace("\\", "/")
    return {
        "schema_version": G1_EFFECTIVE_CONFIG_SCHEMA_VERSION,
        "gate_id": G1_GATE_ID,
        "runner_id": G1_RUNNER_ID,
        "scale_profile": SCALE_PROFILE,
        "run_id": run_id,
        "mode": mode,
        "output_root": output_root,
        "required_phase_ids": list(G1_REQUIRED_PHASE_IDS),
        "required_phases": list(G1_REQUIRED_PHASES),
        "checkpoint": _deep_copy_json(base_config["checkpoint"]),
        "scenario_manifest": {
            "path": str(manifest_path).replace("\\", "/"),
            "schema_version": FROZEN_MANIFEST_SCHEMA,
            "sha256": manifest_sha256,
        },
        "denominator": _deep_copy_json(base_config["denominator"]),
        "execution": _deep_copy_json(base_config["execution"]),
        "bootstrap": _deep_copy_json(base_config["bootstrap"]),
        "schemas": _deep_copy_json(base_config["schemas"]),
        "base_config_sha256": base_config_sha256,
        "input_audit": {
            "path": "g1_input_audit.json",
            "schema_version": G1_INPUT_AUDIT_SCHEMA_VERSION,
            "sha256": _canonical_sha256(input_audit),
        },
        "input_sha256": _canonical_sha256(input_audit),
        "source_lineage": {
            "schema_version": code_lineage["schema_version"],
            "hash_algorithm": code_lineage["hash_algorithm"],
            "required_sources": list(code_lineage["required_sources"]),
        },
        "code_sha256": code_lineage["code_sha256"],
        "repair_lineage": (
            dict(repair_lineage) if repair_lineage is not None else None
        ),
    }


def _git_output(*args: str, cwd: Path = _REPO_ROOT) -> str:
    result = subprocess.run(
        ("git", *args),
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def _environment_probe() -> dict[str, object]:
    import numpy as np
    import torch

    if not torch.cuda.is_available():
        raise G1Blocked("cuda_unavailable")

    class MemoryStatus(ctypes.Structure):
        _fields_ = (
            ("length", ctypes.c_ulong),
            ("memory_load", ctypes.c_ulong),
            ("total_physical", ctypes.c_ulonglong),
            ("available_physical", ctypes.c_ulonglong),
            ("total_page_file", ctypes.c_ulonglong),
            ("available_page_file", ctypes.c_ulonglong),
            ("total_virtual", ctypes.c_ulonglong),
            ("available_virtual", ctypes.c_ulonglong),
            ("available_extended_virtual", ctypes.c_ulonglong),
        )

    memory = MemoryStatus()
    memory.length = ctypes.sizeof(MemoryStatus)
    if os.name == "nt":
        ok = ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory))
        if not ok:
            raise G1Blocked("memory_probe_failed")
        memory_bytes = int(memory.total_physical)
    else:
        memory_bytes = 1
    driver_result = subprocess.run(
        (
            "nvidia-smi",
            "--query-gpu=driver_version",
            "--format=csv,noheader",
        ),
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    driver = (
        driver_result.stdout.splitlines()[0].strip()
        if driver_result.returncode == 0 and driver_result.stdout.strip()
        else "unavailable"
    )
    thread_names = (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    )
    return {
        "windows_version": platform.platform(),
        "cpu_model": platform.processor() or "unknown-cpu",
        "cpu_logical_count": os.cpu_count() or 1,
        "memory_bytes": memory_bytes,
        "gpu": {
            "model": torch.cuda.get_device_name(0),
            "driver": driver,
            "cuda": str(torch.version.cuda or "unknown"),
        },
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "frozen_dependencies": [
            f"numpy=={np.__version__}",
            f"torch=={torch.__version__}",
        ],
        "python_hash_seed": os.environ.get("PYTHONHASHSEED", "not-set"),
        "thread_variables": {
            name: os.environ.get(name, "not-set") for name in thread_names
        },
        "worker_start_method": "spawn",
        "power_mode": "formal-exclusive-requested/v1",
    }


def _verify_update80_checkpoint() -> None:
    if not artifact_io.path_is_file(UPDATE80_CHECKPOINT_PATH):
        raise G1Blocked("update80_checkpoint_missing")
    if _bytes_sha256(
        artifact_io.read_bytes(UPDATE80_CHECKPOINT_PATH)
    ) != UPDATE80_CHECKPOINT_SHA256:
        raise G1Blocked("update80_checkpoint_sha256_mismatch")


def _load_production_context() -> tuple[object, object]:
    from lunar_exploration_ppo.configs.stage6 import (
        SafetyContract,
        parse_stage6_config_bytes,
    )
    from lunar_exploration_ppo.env.standard_training import (
        build_standard_catalog,
    )

    config = parse_stage6_config_bytes(
        artifact_io.read_bytes(_STAGE6_CONFIG_PATH)
    )
    return (
        build_standard_catalog(verify_hashes=True),
        SafetyContract.from_stage6_config(config),
    )


def _trace_envelope(
    *,
    row_kind: str,
    split: str,
    episode_index: int,
    scenario_id: str,
    step_index: int,
    trace: Mapping[str, object],
    run_id: str,
    config_sha256: str,
    input_sha256: str,
    code_sha256: str,
    manifest_sha256: str,
) -> dict[str, object]:
    phase_id, phase_name = _phase_for_split(split)
    if row_kind == "decision":
        schema_version = G1_DECISION_SCHEMA_VERSION
    elif row_kind == "planner_call":
        schema_version = G1_PLANNER_CALL_SCHEMA_VERSION
    else:
        raise G1Blocked("g1_trace_row_kind_invalid")
    index = _exact_nonnegative_int(episode_index, "g1_episode_index_invalid")
    step = _exact_nonnegative_int(step_index, "g1_step_index_invalid")
    row = {
        "row_kind": row_kind,
        "schema_version": schema_version,
        "gate_id": G1_GATE_ID,
        "runner_id": G1_RUNNER_ID,
        "phase_id": phase_id,
        "phase_name": phase_name,
        "scale_profile": SCALE_PROFILE,
        "run_id": run_id,
        "split": split,
        "episode_id": _formal_episode_id(split, index),
        "episode_index": index,
        "scenario_id": scenario_id,
        "lane_id": f"lane-{index % 8}",
        "step_index": step,
        "source_sha256": code_sha256,
        "config_sha256": config_sha256,
        "input_sha256": input_sha256,
        "code_sha256": code_sha256,
        "scenario_manifest_sha256": manifest_sha256,
        "checkpoint_sha256": UPDATE80_CHECKPOINT_SHA256,
        "policy_state_sha256": UPDATE80_POLICY_STATE_SHA256,
        "trace": dict(trace),
    }
    if set(row) != G1_TRACE_ROW_KEYS:
        raise AssertionError("internal G1 trace row schema drifted")
    try:
        _canonical_bytes(row)
    except (TypeError, ValueError) as exc:
        raise G1Blocked("g1_trace_not_json_serializable") from exc
    return row


def _normalize_formal_evaluation(
    evaluation: object,
    *,
    cohort: str,
    run_id: str,
    config_sha256: str,
    input_sha256: str,
    code_sha256: str,
    manifest_sha256: str,
) -> list[dict[str, object]]:
    """Project the verified Task 4 trace into the gate-specific exact schemas."""

    frozen = getattr(evaluation, "frozen_manifest", None)
    execution = getattr(evaluation, "execution", None)
    trace_summary = getattr(evaluation, "trace_summary", None)
    if (
        frozen is None
        or getattr(frozen, "manifest_sha256", None) != manifest_sha256
        or execution is None
        or not isinstance(trace_summary, Mapping)
        or trace_summary.get("cohort") != cohort
        or trace_summary.get("episode_count") != 24
    ):
        raise G1Blocked("g1_task4_evaluation_binding_invalid")
    cohorts = getattr(frozen, "cohorts", None)
    proofs = getattr(frozen, "denominator_proofs", None)
    if not isinstance(cohorts, Mapping) or not isinstance(proofs, Mapping):
        raise G1Blocked("g1_task4_frozen_binding_invalid")
    scenario_ids = tuple(cohorts.get(cohort, ()))
    summary_episodes = trace_summary.get("episodes")
    episode_rows = getattr(execution, "episode_rows", None)
    decision_rows = getattr(execution, "decision_rows", None)
    step_rows = getattr(execution, "step_rows", None)
    if (
        len(scenario_ids) != 24
        or not isinstance(summary_episodes, list)
        or len(summary_episodes) != 24
        or not isinstance(episode_rows, tuple)
        or len(episode_rows) != 24
        or not isinstance(decision_rows, tuple)
        or not isinstance(step_rows, tuple)
    ):
        raise G1Blocked("g1_task4_trace_incomplete")
    episodes_by_index: dict[int, Mapping[str, object]] = {}
    for row in episode_rows:
        if not isinstance(row, Mapping):
            raise G1Blocked("g1_task4_episode_schema_invalid")
        index = _exact_nonnegative_int(
            row.get("episode_index"),
            "g1_task4_episode_index_invalid",
        )
        if index in episodes_by_index:
            raise G1Blocked("g1_task4_episode_index_duplicate")
        episodes_by_index[index] = row
    steps_by_index: dict[int, list[Mapping[str, object]]] = {
        index: [] for index in range(24)
    }
    for row in step_rows:
        if not isinstance(row, Mapping):
            raise G1Blocked("g1_task4_step_schema_invalid")
        index = _exact_nonnegative_int(
            row.get("episode_index"),
            "g1_task4_step_episode_index_invalid",
        )
        if index not in steps_by_index:
            raise G1Blocked("g1_task4_step_episode_index_invalid")
        steps_by_index[index].append(row)

    coverage_rows: list[dict[str, object]] = []
    for index, (scenario_id, trace_episode) in enumerate(
        zip(scenario_ids, summary_episodes, strict=True)
    ):
        raw_episode = episodes_by_index.get(index)
        proof = proofs.get(scenario_id)
        if (
            not isinstance(trace_episode, Mapping)
            or not isinstance(raw_episode, Mapping)
            or not isinstance(proof, Mapping)
            or trace_episode.get("episode_index") != index
            or trace_episode.get("frozen_scenario_id") != scenario_id
            or trace_episode.get("lane_id") != f"lane-{index % 8}"
            or trace_episode.get("denominator_sha256")
            != proof.get("coverable_mask_sha256")
            or trace_episode.get("denominator_cell_count")
            != proof.get("coverable_cell_count")
        ):
            raise G1Blocked("g1_task4_episode_denominator_binding_invalid")
        initial = _exact_nonnegative_int(
            raw_episode.get("initial_covered_cell_count"),
            "g1_task4_initial_count_invalid",
        )
        ordered_steps = sorted(
            steps_by_index[index],
            key=lambda row: _exact_nonnegative_int(
                row.get("step_index"),
                "g1_task4_step_index_invalid",
            ),
        )
        if [
            row.get("step_index") for row in ordered_steps
        ] != list(range(len(ordered_steps))):
            raise G1Blocked("g1_task4_step_sequence_invalid")
        final = initial + sum(
            _exact_nonnegative_int(
                row.get("coverage_gain_cells"),
                "g1_task4_coverage_gain_invalid",
            )
            for row in ordered_steps
        )
        denominator = _exact_nonnegative_int(
            proof.get("coverable_cell_count"),
            "g1_denominator_count_invalid",
        )
        recorded_final = raw_episode.get("final_coverage")
        if (
            isinstance(recorded_final, bool)
            or not isinstance(recorded_final, (int, float))
            or not math.isfinite(float(recorded_final))
            or float(recorded_final) != final / denominator
            or float(trace_episode.get("final_coverage", math.nan))
            != final / denominator
        ):
            raise G1Blocked("g1_task4_final_count_binding_invalid")
        planner_elapsed_ms = 0.0
        for step in ordered_steps:
            diagnostics = step.get("planner_diagnostics")
            if not isinstance(diagnostics, Mapping):
                raise G1Blocked("g1_task4_planner_diagnostics_invalid")
            if "elapsed_ms" in diagnostics:
                planner_elapsed_ms += _finite_nonnegative(
                    diagnostics["elapsed_ms"],
                    "g1_task4_planner_elapsed_invalid",
                )
            elif "elapsed_ns" in diagnostics:
                planner_elapsed_ms += (
                    _exact_nonnegative_int(
                        diagnostics["elapsed_ns"],
                        "g1_task4_planner_elapsed_invalid",
                    )
                    / 1_000_000.0
                )
        coverage_rows.append(
            make_g1_coverage_episode_row(
                run_id=run_id,
                split=cohort,
                episode_index=index,
                scenario_id=scenario_id,
                config_sha256=config_sha256,
                input_sha256=input_sha256,
                code_sha256=code_sha256,
                scenario_manifest_sha256=manifest_sha256,
                denominator_sha256=str(proof["coverable_mask_sha256"]),
                denominator_cell_count=denominator,
                initial_covered_cell_count=initial,
                final_covered_cell_count=final,
                elapsed_ms=planner_elapsed_ms,
                steps_executed=_exact_nonnegative_int(
                    raw_episode.get("steps_executed"),
                    "g1_task4_steps_invalid",
                ),
                termination_reason=str(raw_episode.get("termination_reason")),
                safety_violation_count=_exact_nonnegative_int(
                    raw_episode.get("safety_violation_count"),
                    "g1_task4_safety_count_invalid",
                ),
                masked_action_count=_exact_nonnegative_int(
                    raw_episode.get("invalid_action_count"),
                    "g1_task4_masked_count_invalid",
                ),
            )
        )

    trace_rows: list[dict[str, object]] = []
    for kind, rows in (
        ("decision", decision_rows),
        ("planner_call", step_rows),
    ):
        for trace in rows:
            if not isinstance(trace, Mapping):
                raise G1Blocked("g1_task4_trace_schema_invalid")
            index = _exact_nonnegative_int(
                trace.get("episode_index"),
                "g1_task4_trace_episode_index_invalid",
            )
            if index >= 24:
                raise G1Blocked("g1_task4_trace_episode_index_invalid")
            step_index = _exact_nonnegative_int(
                trace.get("step_index"),
                "g1_task4_trace_step_index_invalid",
            )
            trace_rows.append(
                _trace_envelope(
                    row_kind=kind,
                    split=cohort,
                    episode_index=index,
                    scenario_id=scenario_ids[index],
                    step_index=step_index,
                    trace=trace,
                    run_id=run_id,
                    config_sha256=config_sha256,
                    input_sha256=input_sha256,
                    code_sha256=code_sha256,
                    manifest_sha256=manifest_sha256,
                )
            )
    return [*coverage_rows, *trace_rows]


def _run_auxiliary_once(
    *,
    catalog: object,
    safety_contract: object,
    frozen: object,
    scenario_ids: Sequence[str],
    evaluation_seed_start: int,
    config_sha256: str,
    policy: object,
) -> tuple[dict[str, object], ...]:
    import torch
    from lunar_exploration_ppo.env.standard_training import (
        StandardEvaluationEnv,
    )
    from lunar_exploration_ppo.eval.standard import (
        select_standard_evaluation_actions,
    )
    from lunar_exploration_ppo.policy.cross_attention import (
        batch_policy_observations,
    )

    records = {
        f"{record.scenario_id}/standard-proxy/v1": record
        for record in getattr(catalog, "records", ())
    }
    results: list[dict[str, object]] = []
    for index, frozen_id in enumerate(scenario_ids):
        record = records.get(frozen_id)
        if record is None:
            raise G1Blocked("auxiliary_scenario_not_in_catalog")
        environment = StandardEvaluationEnv(
            catalog,
            split=record.split,
            scenario_ids=(record.scenario_id,),
            safety_contract=safety_contract,
            config_sha256=config_sha256,
        )
        observation = environment.reset()
        actions: list[dict[str, object]] = []
        coverage_curve: list[float] = []
        for step_index in range(128):
            if environment.is_done:
                break
            if not environment.needs_policy:
                raise G1Blocked("auxiliary_policy_state_invalid")
            batch = batch_policy_observations((observation,), device="cuda")
            with torch.inference_mode():
                output = policy(batch)
            action = select_standard_evaluation_actions(
                method="ppo_policy",
                observations=(observation,),
                evaluation_seeds=(
                    evaluation_seed_start + index * 128 + step_index,
                ),
                policy_output=output,
            )[0]
            actions.append(
                {
                    "selected_candidate_index": action.candidate_index,
                    "selected_theta": float(action.target_theta),
                }
            )
            step = environment.step(action)
            coverage_curve.append(float(step.coverage_rate))
            observation = step.observation
            if step.done:
                break
        if not environment.is_done:
            raise G1Blocked("auxiliary_episode_did_not_terminate")
        results.append(
            {
                "scenario_id": frozen_id,
                "actions": actions,
                "coverage_curve": coverage_curve or [0.0],
                "termination_reason": str(environment.terminal_reason),
            }
        )
    return tuple(results)


def _load_update80_policy() -> object:
    from lunar_exploration_ppo.eval.midterm_reduced import (
        load_midterm_update80_policy,
    )

    return load_midterm_update80_policy()


def _run_validation_production(
    *,
    catalog: object,
    safety_contract: object,
    frozen: object,
    config_sha256: str,
    evaluation_seed_start: int,
) -> dict[str, object]:
    cohorts = getattr(frozen, "cohorts", {})
    scenario_ids = tuple(cohorts.get("validation3", ()))
    policy = _load_update80_policy()
    return run_validation_dry_run(
        scenario_ids,
        lambda selected: _run_auxiliary_once(
            catalog=catalog,
            safety_contract=safety_contract,
            frozen=frozen,
            scenario_ids=selected,
            evaluation_seed_start=evaluation_seed_start,
            config_sha256=config_sha256,
            policy=policy,
        ),
    )


def _run_replay_production(
    *,
    catalog: object,
    safety_contract: object,
    frozen: object,
    config_sha256: str,
    evaluation_seed_start: int,
) -> dict[str, object]:
    cohorts = getattr(frozen, "cohorts", {})
    scenario_ids = tuple(cohorts.get("replay3", ()))
    policy = _load_update80_policy()
    reference = _run_auxiliary_once(
        catalog=catalog,
        safety_contract=safety_contract,
        frozen=frozen,
        scenario_ids=scenario_ids,
        evaluation_seed_start=evaluation_seed_start,
        config_sha256=config_sha256,
        policy=policy,
    )
    replay = _run_auxiliary_once(
        catalog=catalog,
        safety_contract=safety_contract,
        frozen=frozen,
        scenario_ids=scenario_ids,
        evaluation_seed_start=evaluation_seed_start,
        config_sha256=config_sha256,
        policy=policy,
    )
    return validate_g1_replay(reference, replay)


def validate_canonical_g1_summary(summary: object) -> dict[str, object]:
    if not isinstance(summary, Mapping) or set(summary) != G1_SUMMARY_KEYS:
        raise G1Blocked("g1_summary_schema_invalid")
    copied = dict(summary)
    if (
        copied.get("schema_version") != G1_SUMMARY_SCHEMA_VERSION
        or copied.get("gate_id") != G1_GATE_ID
        or copied.get("runner_id") != G1_RUNNER_ID
        or copied.get("scale_profile") != SCALE_PROFILE
        or copied.get("status") not in {"passed", "failed", "blocked"}
        or type(copied.get("formal_evidence_eligible")) is not bool
        or type(copied.get("g1_coverage_80_passed")) is not bool
        or type(copied.get("g1_coverage_99_passed")) is not bool
        or not isinstance(copied.get("splits"), Mapping)
        or not isinstance(copied.get("failure_reasons"), list)
        or not isinstance(copied.get("blockers"), list)
    ):
        raise G1Blocked("g1_summary_schema_invalid")
    for name, value in copied["splits"].items():
        if value is not None and (
            not isinstance(value, Mapping)
            or set(value) != G1_SPLIT_SUMMARY_KEYS
            or value.get("split") != name
        ):
            raise G1Blocked("g1_split_summary_schema_invalid")
    return copied


def render_g1_report(summary: Mapping[str, object]) -> str:
    """Canonical deterministic renderer; the report is never a metric input."""

    value = validate_canonical_g1_summary(summary)
    split_rows: list[str] = []
    for split in value["split_order"]:
        split_summary = value["splits"].get(split)
        if split_summary is None:
            split_rows.append(f"| {split} | 未执行 | 0 | - | 0 | 0 |")
            continue
        split_rows.append(
            "| "
            f"{split} | {split_summary['status']} | "
            f"{split_summary['sample_count']} | "
            f"{float(split_summary['mean']):.12f} | "
            f"{split_summary['coverage_80_count']} | "
            f"{split_summary['coverage_99_count']} |"
        )
    replay = value.get("replay")
    replay_status = (
        str(replay.get("status"))
        if isinstance(replay, Mapping)
        else "not_run"
    )
    lines = [
        "# G1 覆盖率资格试验报告",
        "",
        f"- schema: `{G1_REPORT_RENDERER}`",
        f"- run_id: `{value['run_id']}`",
        f"- status: `{value['status']}`",
        f"- scale_profile: `{SCALE_PROFILE}`",
        f"- checkpoint_sha256: `{UPDATE80_CHECKPOINT_SHA256}`",
        f"- policy_state_sha256: `{UPDATE80_POLICY_STATE_SHA256}`",
        f"- formal_episode_count: `{value['formal_episode_count']}`",
        f"- replay3: `{replay_status}`",
        f"- g1_coverage_80_passed: `{str(value['g1_coverage_80_passed']).lower()}`",
        f"- g1_coverage_99_passed: `{str(value['g1_coverage_99_passed']).lower()}`",
        "",
        "| split | status | n | macro mean | >=0.80 | >=0.99 |",
        "|---|---:|---:|---:|---:|---:|",
        *split_rows,
        "",
        "本结果只代表固定 update80 与缩减规模 8×3 场景样本；"
        "逐行整数覆盖结果是指标输入，本文本报告不是指标输入。",
        "",
    ]
    return "\n".join(lines)


def _routing(summary: Mapping[str, object]) -> dict[str, object]:
    status = str(summary["status"])
    return {
        "schema_version": G1_ROUTING_SCHEMA_VERSION,
        "gate_id": G1_GATE_ID,
        "runner_id": G1_RUNNER_ID,
        "scale_profile": SCALE_PROFILE,
        "run_id": summary["run_id"],
        "status": status,
        "formal_evidence_eligible": status != "blocked",
        "g1_coverage_80_passed": summary["g1_coverage_80_passed"],
        "g1_coverage_99_passed": summary["g1_coverage_99_passed"],
        "blocking_reason": (
            summary["blockers"][0] if summary["blockers"] else None
        ),
        "failure_reasons": list(summary["failure_reasons"]),
        "next_route": (
            "aggregate"
            if status in {"passed", "failed"}
            else "repair_input_or_environment"
        ),
    }


def _blocked_summary(
    *,
    run_id: str,
    mode: str,
    config_sha256: str,
    input_sha256: str,
    code_sha256: str,
    manifest_sha256: str | None,
    reason: str,
) -> dict[str, object]:
    summary = {
        "schema_version": G1_SUMMARY_SCHEMA_VERSION,
        "gate_id": G1_GATE_ID,
        "runner_id": G1_RUNNER_ID,
        "scale_profile": SCALE_PROFILE,
        "run_id": run_id,
        "mode": mode,
        "config_sha256": config_sha256,
        "input_sha256": input_sha256,
        "code_sha256": code_sha256,
        "scenario_manifest_sha256": manifest_sha256 or "",
        "checkpoint_sha256": UPDATE80_CHECKPOINT_SHA256,
        "policy_state_sha256": UPDATE80_POLICY_STATE_SHA256,
        "status": "blocked",
        "formal_evidence_eligible": False,
        "formal_episode_count": 0,
        "split_order": list(G1_FORMAL_SPLITS),
        "unseen_execution_status": "not_started",
        "splits": {"test_q24": None, "unseen24": None},
        "safety_clean": False,
        "masked_action_clean": False,
        "replay": None,
        "g1_coverage_80_passed": False,
        "g1_coverage_99_passed": False,
        "failure_reasons": [],
        "blockers": [reason],
    }
    if set(summary) != G1_SUMMARY_KEYS:
        raise AssertionError("internal blocked G1 summary schema drifted")
    return summary


def _test_c_summary(
    *,
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    split = recompute_g1_split_summary(rows, expected_split="test_c24")
    lineage = _summary_lineage(rows)
    passed = split["g1_coverage_80_passed"] is True
    summary = {
        "schema_version": G1_SUMMARY_SCHEMA_VERSION,
        "gate_id": G1_GATE_ID,
        "runner_id": G1_RUNNER_ID,
        "scale_profile": SCALE_PROFILE,
        "run_id": lineage["run_id"],
        "mode": "test-c-confirmation",
        "config_sha256": lineage["config_sha256"],
        "input_sha256": lineage["input_sha256"],
        "code_sha256": lineage["code_sha256"],
        "scenario_manifest_sha256": lineage["scenario_manifest_sha256"],
        "checkpoint_sha256": UPDATE80_CHECKPOINT_SHA256,
        "policy_state_sha256": UPDATE80_POLICY_STATE_SHA256,
        "status": "passed" if passed else "failed",
        "formal_evidence_eligible": True,
        "formal_episode_count": 0,
        "split_order": ["test_c24"],
        "unseen_execution_status": "not_applicable_test_c_confirmation",
        "splits": {"test_c24": split},
        "safety_clean": split["safety_clean"],
        "masked_action_clean": split["masked_action_clean"],
        "replay": None,
        "g1_coverage_80_passed": False,
        "g1_coverage_99_passed": False,
        "failure_reasons": (
            [] if passed else ["test_c24_coverage_80_failed"]
        ),
        "blockers": [],
    }
    if set(summary) != G1_SUMMARY_KEYS:
        raise AssertionError("internal Test-C summary schema drifted")
    return summary


def _phase_state_rows(store: MidDualRunStore) -> list[dict[str, Any]]:
    path = artifact_path(store.run_root, MID_DUAL_PHASE_STATE)
    return artifact_io.read_jsonl(path)


def _accepted_phase_rows(
    store: MidDualRunStore,
    phase_id: str,
) -> list[dict[str, Any]]:
    state = next(
        (
            row
            for row in _phase_state_rows(store)
            if row.get("phase_id") == phase_id
        ),
        None,
    )
    if state is None:
        raise G1Blocked(f"{phase_id}_not_accepted")
    rows_path = state.get("rows_path")
    if not isinstance(rows_path, str):
        raise G1Blocked(f"{phase_id}_state_invalid")
    return artifact_io.read_jsonl(store.run_root / rows_path)


def _accepted_phase_audit(
    store: MidDualRunStore,
    phase_id: str,
) -> dict[str, Any]:
    attempts_path = store.run_root / "phase-attempts.jsonl"
    attempts = artifact_io.read_jsonl(attempts_path)
    accepted = next(
        (
            row
            for row in attempts
            if row.get("phase_id") == phase_id
            and row.get("status") == "accepted"
        ),
        None,
    )
    if accepted is None or not isinstance(accepted.get("audit_path"), str):
        raise G1Blocked(f"{phase_id}_audit_missing")
    return artifact_io.read_json(store.run_root / accepted["audit_path"])


def _accept_phase(
    store: MidDualRunStore,
    *,
    phase_id: str,
    rows: Sequence[Mapping[str, object]],
    audit: Mapping[str, object],
) -> None:
    if phase_id in store.accepted_phase_ids:
        return
    expected = f"p{len(store.accepted_phase_ids) + 1:02d}"
    if phase_id != expected:
        raise G1Blocked("g1_resume_phase_prefix_invalid")
    phase_audit = {
        "schema_version": "xunce-mid-dual-g1-phase-audit/v1",
        "gate_id": G1_GATE_ID,
        "runner_id": G1_RUNNER_ID,
        "phase_id": phase_id,
        "phase_name": G1_PHASE_NAMES[phase_id],
        **dict(audit),
    }
    attempt_id = store.write_phase_attempt(phase_id, rows, phase_audit)
    row_sha256 = store.phase_attempt_row_sha256(phase_id, attempt_id)
    store.accept_phase(phase_id, attempt_id, row_sha256)


def _open_store(
    run_root: Path,
    effective_config: Mapping[str, object],
) -> tuple[MidDualRunStore, bool]:
    expected_sha256 = _canonical_sha256(effective_config)
    if artifact_io.path_exists(run_root):
        if artifact_io.path_is_file(artifact_path(run_root, MID_DUAL_MANIFEST)):
            MidDualRunStore.verify_manifest(run_root)
            raise G1Blocked("run_already_finalized")
        return (
            MidDualRunStore.load_for_resume(run_root, expected_sha256),
            False,
        )
    return MidDualRunStore.create_new(run_root, effective_config), True


def _root_and_submodule_commits() -> tuple[str, str]:
    root_commit = _git_output("rev-parse", "HEAD") or "unavailable-root-commit"
    submodule_path = _REPO_ROOT / "path-planner"
    submodule_commit = (
        _git_output("rev-parse", "HEAD", cwd=submodule_path)
        if artifact_io.path_exists(submodule_path)
        else ""
    )
    return root_commit, submodule_commit or "not-present"


def _capture_valid_preflight(
    *,
    store: MidDualRunStore,
    newly_created: bool,
    manifest_sha256: str,
    input_sha256: str,
    code_sha256: str,
) -> None:
    if "p01" in store.accepted_phase_ids:
        return
    if newly_created:
        root_commit, submodule_commit = _root_and_submodule_commits()
        lineage = store.capture_lineage(
            _required_source_paths(),
            root_commit,
            submodule_commit,
        )
        if lineage.get("formal_evidence_eligible") is not True:
            raise G1Blocked("g1_lineage_capture_blocked")
        _verify_update80_checkpoint()
        environment = store.capture_environment(_environment_probe)
        if environment.get("formal_evidence_eligible") is not True:
            raise G1Blocked(
                str(
                    environment.get(
                        "blocking_reason",
                        "g1_environment_capture_blocked",
                    )
                )
            )
    _accept_phase(
        store,
        phase_id="p01",
        rows=(),
        audit={
            "status": "passed",
            "scenario_manifest_sha256": manifest_sha256,
            "input_sha256": input_sha256,
            "code_sha256": code_sha256,
            "checkpoint_sha256": UPDATE80_CHECKPOINT_SHA256,
            "policy_state_sha256": UPDATE80_POLICY_STATE_SHA256,
        },
    )


def _capture_missing_preflight(
    *,
    store: MidDualRunStore,
    newly_created: bool,
    reason: str,
) -> None:
    if newly_created:
        root_commit, submodule_commit = _root_and_submodule_commits()
        store.capture_lineage(
            _required_source_paths(),
            root_commit,
            submodule_commit,
        )
    if "p01" not in store.accepted_phase_ids:
        _accept_phase(
            store,
            phase_id="p01",
            rows=(),
            audit={
                "status": "blocked",
                "blocking_reason": reason,
                "cuda_loader_called": False,
            },
        )


def _finalize_blocked(
    *,
    store: MidDualRunStore,
    run_id: str,
    mode: str,
    input_audit: Mapping[str, object],
    code_sha256: str,
    manifest_sha256: str | None,
    reason: str,
) -> dict[str, object]:
    summary = _blocked_summary(
        run_id=run_id,
        mode=mode,
        config_sha256=store.config_sha256,
        input_sha256=_canonical_sha256(input_audit),
        code_sha256=code_sha256,
        manifest_sha256=manifest_sha256,
        reason=reason,
    )
    report = render_g1_report(summary)
    store.finalize(
        summary,
        _routing(summary),
        report,
        {"g1_input": input_audit},
    )
    return {
        "execution_status": "complete",
        "gate_status": "blocked",
        "run_root": str(store.run_root).replace("\\", "/"),
        "blocking_reason": reason,
        "summary": summary,
    }


def run_g1(
    *,
    config_path: str | Path,
    scenario_manifest: str | Path,
    run_id: str,
    mode: str,
    repair_lineage: str | Path | None = None,
) -> dict[str, object]:
    """Create or resume one gate-specific G1 run without parameter overrides."""

    run_id = _validated_run_id(run_id)
    if mode not in G1_MODES:
        raise G1Blocked("g1_mode_invalid")
    if mode != "test-c-confirmation" and repair_lineage is not None:
        raise G1Blocked("repair_lineage_only_allowed_for_test_c")
    supplied_config = Path(config_path).resolve()
    if supplied_config != _CANONICAL_CONFIG_PATH.resolve():
        raise G1Blocked("g1_config_path_not_canonical")
    base_config = load_g1_config(supplied_config)
    base_config_sha256 = _bytes_sha256(
        artifact_io.read_bytes(supplied_config)
    )
    manifest_path = _validate_d_manifest_path(scenario_manifest)
    code_lineage = compute_g1_code_lineage()
    code_sha256 = str(code_lineage["code_sha256"])
    repair_audit: dict[str, object] | None = None
    if mode == "test-c-confirmation":
        repair_audit = validate_test_c_request(
            mode=mode,
            repair_lineage=repair_lineage,
            current_code_sha256=code_sha256,
            current_config_sha256=base_config_sha256,
        )

    run_root = Path(G1_OUTPUT_BASE) / run_id
    if not artifact_io.path_is_file(manifest_path):
        reason = "scenario_manifest_missing"
        input_audit = _missing_input_audit(manifest_path, reason)
        effective = _effective_config(
            base_config=base_config,
            base_config_sha256=base_config_sha256,
            run_id=run_id,
            mode=mode,
            manifest_path=manifest_path,
            manifest_sha256=None,
            input_audit=input_audit,
            code_lineage=code_lineage,
            repair_lineage=repair_audit,
        )
        store, created = _open_store(run_root, effective)
        _capture_missing_preflight(
            store=store,
            newly_created=created,
            reason=reason,
        )
        return _finalize_blocked(
            store=store,
            run_id=run_id,
            mode=mode,
            input_audit=input_audit,
            code_sha256=code_sha256,
            manifest_sha256=None,
            reason=reason,
        )

    frozen: object
    raw_manifest: dict[str, Any]
    manifest_sha256: str
    try:
        frozen, raw_manifest, manifest_sha256 = _load_frozen_manifest(
            manifest_path
        )
        input_audit = _build_input_audit(
            manifest_path=manifest_path,
            frozen=frozen,
            raw_manifest=raw_manifest,
            manifest_sha256=manifest_sha256,
        )
    except G1Blocked as exc:
        input_audit = _missing_input_audit(manifest_path, exc.reason)
        effective = _effective_config(
            base_config=base_config,
            base_config_sha256=base_config_sha256,
            run_id=run_id,
            mode=mode,
            manifest_path=manifest_path,
            manifest_sha256=None,
            input_audit=input_audit,
            code_lineage=code_lineage,
            repair_lineage=repair_audit,
        )
        store, created = _open_store(run_root, effective)
        _capture_missing_preflight(
            store=store,
            newly_created=created,
            reason=exc.reason,
        )
        return _finalize_blocked(
            store=store,
            run_id=run_id,
            mode=mode,
            input_audit=input_audit,
            code_sha256=code_sha256,
            manifest_sha256=None,
            reason=exc.reason,
        )

    effective = _effective_config(
        base_config=base_config,
        base_config_sha256=base_config_sha256,
        run_id=run_id,
        mode=mode,
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha256,
        input_audit=input_audit,
        code_lineage=code_lineage,
        repair_lineage=repair_audit,
    )
    store: MidDualRunStore | None = None
    try:
        store, created = _open_store(run_root, effective)
        _capture_valid_preflight(
            store=store,
            newly_created=created,
            manifest_sha256=manifest_sha256,
            input_sha256=_canonical_sha256(input_audit),
            code_sha256=code_sha256,
        )
        if mode == "preflight":
            return {
                "execution_status": "incomplete",
                "gate_status": "preflight_complete",
                "run_root": str(run_root).replace("\\", "/"),
                "accepted_phase_ids": list(store.accepted_phase_ids),
            }

        catalog, safety_contract = _load_production_context()
        execution_config = base_config["execution"]
        if "p02" not in store.accepted_phase_ids:
            dry_run = _run_validation_production(
                catalog=catalog,
                safety_contract=safety_contract,
                frozen=frozen,
                config_sha256=store.config_sha256,
                evaluation_seed_start=execution_config[
                    "validation3_evaluation_seed_start"
                ],
            )
            _accept_phase(
                store,
                phase_id="p02",
                rows=(),
                audit={"status": "passed", "dry_run": dry_run},
            )
        if mode == "dry-run":
            return {
                "execution_status": "incomplete",
                "gate_status": "dry_run_complete",
                "run_root": str(run_root).replace("\\", "/"),
                "accepted_phase_ids": list(store.accepted_phase_ids),
            }

        primary_cohort = (
            "test_c24" if mode == "test-c-confirmation" else "test_q24"
        )
        if "p03" not in store.accepted_phase_ids:
            evaluation = _evaluate_formal_cohort(
                catalog=catalog,
                frozen_bundle_root=manifest_path.parent,
                frozen_manifest_sha256=manifest_sha256,
                cohort=primary_cohort,
                evaluation_seed_start=execution_config[
                    f"{primary_cohort}_evaluation_seed_start"
                ],
                safety_contract=safety_contract,
                config_sha256=store.config_sha256,
            )
            primary_rows = _normalize_formal_evaluation(
                evaluation,
                cohort=primary_cohort,
                run_id=run_id,
                config_sha256=store.config_sha256,
                input_sha256=_canonical_sha256(input_audit),
                code_sha256=code_sha256,
                manifest_sha256=manifest_sha256,
            )
            _accept_phase(
                store,
                phase_id="p03",
                rows=primary_rows,
                audit={
                    "status": "complete",
                    "cohort": primary_cohort,
                    "coverage_episode_count": len(
                        _coverage_rows_for_split(
                            primary_rows,
                            primary_cohort,
                        )
                    ),
                    "decision_count": sum(
                        row.get("row_kind") == "decision"
                        for row in primary_rows
                    ),
                    "planner_call_count": sum(
                        row.get("row_kind") == "planner_call"
                        for row in primary_rows
                    ),
                },
            )
        else:
            primary_rows = _accepted_phase_rows(store, "p03")
        primary_coverage = _coverage_rows_for_split(
            primary_rows,
            primary_cohort,
        )
        primary_summary = recompute_g1_split_summary(
            primary_coverage,
            expected_split=primary_cohort,
        )

        replay_audit: dict[str, object] | None = None
        if mode == "test-c-confirmation":
            for phase_id in ("p04", "p05"):
                _accept_phase(
                    store,
                    phase_id=phase_id,
                    rows=(),
                    audit={
                        "status": "skipped",
                        "reason": "not_applicable_test_c_confirmation",
                    },
                )
            summary = _test_c_summary(rows=primary_coverage)
        else:
            test_q_passed = (
                primary_summary["g1_coverage_80_passed"] is True
            )
            if not test_q_passed:
                _accept_phase(
                    store,
                    phase_id="p04",
                    rows=(),
                    audit={
                        "status": "skipped",
                        "reason": "test_q24_coverage_80_failed",
                    },
                )
                _accept_phase(
                    store,
                    phase_id="p05",
                    rows=(),
                    audit={
                        "status": "skipped",
                        "reason": "test_q24_coverage_80_failed",
                    },
                )
                unseen_summary = None
            else:
                if "p04" not in store.accepted_phase_ids:
                    unseen_evaluation = _evaluate_formal_cohort(
                        catalog=catalog,
                        frozen_bundle_root=manifest_path.parent,
                        frozen_manifest_sha256=manifest_sha256,
                        cohort="unseen24",
                        evaluation_seed_start=execution_config[
                            "unseen24_evaluation_seed_start"
                        ],
                        safety_contract=safety_contract,
                        config_sha256=store.config_sha256,
                    )
                    unseen_rows = _normalize_formal_evaluation(
                        unseen_evaluation,
                        cohort="unseen24",
                        run_id=run_id,
                        config_sha256=store.config_sha256,
                        input_sha256=_canonical_sha256(input_audit),
                        code_sha256=code_sha256,
                        manifest_sha256=manifest_sha256,
                    )
                    _accept_phase(
                        store,
                        phase_id="p04",
                        rows=unseen_rows,
                        audit={
                            "status": "complete",
                            "cohort": "unseen24",
                            "coverage_episode_count": len(
                                _coverage_rows_for_split(
                                    unseen_rows,
                                    "unseen24",
                                )
                            ),
                            "decision_count": sum(
                                row.get("row_kind") == "decision"
                                for row in unseen_rows
                            ),
                            "planner_call_count": sum(
                                row.get("row_kind") == "planner_call"
                                for row in unseen_rows
                            ),
                        },
                    )
                else:
                    unseen_rows = _accepted_phase_rows(store, "p04")
                unseen_coverage = _coverage_rows_for_split(
                    unseen_rows,
                    "unseen24",
                )
                unseen_summary = recompute_g1_split_summary(
                    unseen_coverage,
                    expected_split="unseen24",
                )
                if "p05" not in store.accepted_phase_ids:
                    replay_audit = _run_replay_production(
                        catalog=catalog,
                        safety_contract=safety_contract,
                        frozen=frozen,
                        config_sha256=store.config_sha256,
                        evaluation_seed_start=execution_config[
                            "replay3_evaluation_seed_start"
                        ],
                    )
                    _accept_phase(
                        store,
                        phase_id="p05",
                        rows=(),
                        audit={
                            "status": "passed",
                            "replay": replay_audit,
                        },
                    )
                else:
                    recorded_replay = _accepted_phase_audit(
                        store,
                        "p05",
                    ).get("replay")
                    if not isinstance(recorded_replay, Mapping):
                        raise G1Blocked("replay_audit_missing")
                    replay_audit = dict(recorded_replay)
            summary = _build_summary(
                test_q24=primary_summary,
                unseen24=unseen_summary,
                lineage=_summary_lineage(primary_coverage),
                mode=mode,
                replay=replay_audit,
            )

        _accept_phase(
            store,
            phase_id="p06",
            rows=(),
            audit={
                "status": "complete",
                "canonical_summary_sha256": _canonical_sha256(summary),
                "canonical_summary": summary,
            },
        )
        _accept_phase(
            store,
            phase_id="p07",
            rows=(),
            audit={
                "status": "ready",
                "accepted_phase_ids": list(G1_REQUIRED_PHASE_IDS),
            },
        )
        report = render_g1_report(summary)
        store.finalize(
            summary,
            _routing(summary),
            report,
            {"g1_input": input_audit},
        )
        MidDualRunStore.verify_manifest(store.run_root)
        return {
            "execution_status": "complete",
            "gate_status": summary["status"],
            "run_root": str(run_root).replace("\\", "/"),
            "summary": summary,
        }
    except G1Blocked as exc:
        if store is None:
            raise
        return _finalize_blocked(
            store=store,
            run_id=run_id,
            mode=mode,
            input_audit=input_audit,
            code_sha256=code_sha256,
            manifest_sha256=manifest_sha256,
            reason=exc.reason,
        )
    except Exception as exc:
        if store is None:
            raise G1Blocked(
                f"g1_runner_exception:{type(exc).__name__}"
            ) from exc
        return _finalize_blocked(
            store=store,
            run_id=run_id,
            mode=mode,
            input_audit=input_audit,
            code_sha256=code_sha256,
            manifest_sha256=manifest_sha256,
            reason=f"g1_runner_exception:{type(exc).__name__}",
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the fixed-update80 reduced G1 coverage gate."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--scenario-manifest", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--mode", required=True, choices=G1_MODES)
    parser.add_argument("--repair-lineage")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = run_g1(
            config_path=args.config,
            scenario_manifest=args.scenario_manifest,
            run_id=args.run_id,
            mode=args.mode,
            repair_lineage=args.repair_lineage,
        )
    except G1Blocked as exc:
        result = {
            "execution_status": "not_started",
            "gate_status": "blocked",
            "blocking_reason": exc.reason,
        }
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DENOMINATOR_ALGORITHM",
    "DENOMINATOR_SOURCE",
    "G1Blocked",
    "G1_CONFIG_SCHEMA_VERSION",
    "G1_COVERAGE_EPISODE_KEYS",
    "G1_COVERAGE_EPISODE_SCHEMA_VERSION",
    "G1_EFFECTIVE_CONFIG_SCHEMA_VERSION",
    "G1_FORMAL_SPLITS",
    "G1_GATE_ID",
    "G1_INPUT_AUDIT_SCHEMA_VERSION",
    "G1_MODES",
    "G1_OUTPUT_BASE",
    "G1_REQUIRED_PHASE_IDS",
    "G1_REQUIRED_PHASES",
    "G1_REQUIRED_SOURCE_RELATIVE_PATHS",
    "G1_RUNNER_ID",
    "G1_SPLIT_SUMMARY_KEYS",
    "G1_SUMMARY_KEYS",
    "G1_SUMMARY_SCHEMA_VERSION",
    "UPDATE80_CHECKPOINT_PATH",
    "UPDATE80_CHECKPOINT_SHA256",
    "UPDATE80_POLICY_STATE_SHA256",
    "compute_g1_code_lineage",
    "execute_formal_gate",
    "load_g1_config",
    "make_g1_coverage_episode_row",
    "preflight_scenario_manifest",
    "recompute_g1_split_summary",
    "recompute_g1_summary",
    "render_g1_report",
    "run_g1",
    "run_validation_dry_run",
    "validate_canonical_g1_summary",
    "validate_g1_config_payload",
    "validate_g1_replay",
    "validate_test_c_request",
]
