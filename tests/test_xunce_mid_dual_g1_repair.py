from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "scripts/run_xunce_mid_dual_g1_repair.py"
SCRIPTS = REPO_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

TARGETS = {
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
PARENT_FAILURES = {
    3: {
        "denominator_cell_count": 63_000,
        "denominator_sha256": (
            "c27b39df5994429319d51d50b61be07b94170e4bf32fc9e792d0b8152a77dcfa"
        ),
        "covered_cell_count": 1_173,
        "coverage": 0.01861904761904762,
    },
    10: {
        "denominator_cell_count": 64_291,
        "denominator_sha256": (
            "702d6646807e356cc22a83a0a75fd3ecc1c8566a354de89b2ad7e41edac43aa7"
        ),
        "covered_cell_count": 1_301,
        "coverage": 0.020236113919522174,
    },
    23: {
        "denominator_cell_count": 57_857,
        "denominator_sha256": (
            "37dffeac83674f97c93e64a7cf2da6c24c3b99585f7f15cb960334263f0a3c96"
        ),
        "covered_cell_count": 1_005,
        "coverage": 0.017370413260279657,
    },
}


def _load_module():
    assert MODULE_PATH.is_file(), "G1 repair runner has not been implemented"
    spec = importlib.util.spec_from_file_location(
        "run_xunce_mid_dual_g1_repair",
        MODULE_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _canonical_line(row: dict[str, object]) -> bytes:
    return (
        json.dumps(
            row,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _parent_episode(index: int) -> dict[str, object]:
    target = TARGETS.get(index)
    failure = PARENT_FAILURES.get(index)
    denominator = (
        failure["denominator_cell_count"]
        if failure is not None
        else 10_000 + index
    )
    final_count = (
        failure["covered_cell_count"]
        if failure is not None
        else 9_000 + index
    )
    return {
        "row_kind": "coverage_episode",
        "schema_version": "xunce-mid-dual-g1-coverage-episode/v1",
        "gate_id": "g1",
        "runner_id": "run_xunce_mid_dual_g1_coverage/v1",
        "phase_id": "p04",
        "phase_name": "unseen24",
        "scale_profile": "midterm_reduced_w8x3_update80/v1",
        "run_id": "parent-run",
        "split": "unseen24",
        "episode_id": f"unseen24-episode-{index:02d}",
        "episode_index": index,
        "scenario_id": (
            target["scenario_id"]
            if target is not None
            else f"unseen/fixture-{index:04d}/standard-proxy/v1"
        ),
        "lane_id": f"lane-{index % 8}",
        "source_sha256": "1" * 64,
        "config_sha256": "2" * 64,
        "input_sha256": "3" * 64,
        "code_sha256": "1" * 64,
        "scenario_manifest_sha256": "4" * 64,
        "checkpoint_sha256": "5" * 64,
        "policy_state_sha256": "6" * 64,
        "denominator_source": (
            "reachable_observable_free_highres_cells/v1"
        ),
        "denominator_algorithm": (
            "exact_reachable_safe_pose_range_los/v1"
        ),
        "denominator_sha256": (
            failure["denominator_sha256"]
            if failure is not None
            else f"{index + 16:064x}"
        ),
        "denominator_cell_count": denominator,
        "initial_covered_cell_count": (
            failure["covered_cell_count"] if failure is not None else 1_000
        ),
        "final_covered_cell_count": final_count,
        "coverage": (
            failure["coverage"] if failure is not None else final_count / denominator
        ),
        "elapsed_ms": 0.0,
        "steps_executed": 0 if failure is not None else 5,
        "termination_reason": (
            "no_candidate_done" if failure is not None else "success_done"
        ),
        "safety_violation_count": 0,
        "masked_action_count": 0,
    }


def _write_parent(path: Path) -> tuple[str, list[dict[str, object]]]:
    rows = [_parent_episode(index) for index in range(24)]
    trace = {
        "row_kind": "decision",
        "schema_version": "xunce-mid-dual-g1-decision/v1",
        "episode_index": 0,
        "step_index": 0,
    }
    payload = b"".join(_canonical_line(row) for row in [*rows, trace])
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest(), rows


def _repair_episode(index: int, *, final_count: int = 8_500) -> dict[str, object]:
    target = TARGETS[index]
    denominator = 10_000
    return {
        "row_kind": "repair_coverage_episode",
        "schema_version": "xunce-mid-dual-g1-repair-episode/v1",
        "episode_index": index,
        "episode_id": f"unseen24-episode-{index:02d}",
        "scenario_id": target["scenario_id"],
        "record_id": target["record_id"],
        "lane_id": target["lane_id"],
        "evaluation_seed": target["evaluation_seed"],
        "denominator_sha256": f"{index + 32:064x}",
        "denominator_cell_count": denominator,
        "initial_covered_cell_count": 1_000,
        "final_covered_cell_count": final_count,
        "coverage": final_count / denominator,
        "steps_executed": 4,
        "termination_reason": "success_done",
        "safety_violation_count": 0,
        "masked_action_count": 0,
        "checkpoint_sha256": "5" * 64,
        "policy_state_sha256": "6" * 64,
        "scenario_manifest_sha256": "4" * 64,
        "formal_config_sha256": "2" * 64,
        "repair_code_sha256": "7" * 64,
        "repair_static_config_sha256": "a" * 64,
        "first_step_gate": {
            "passed": True,
            "candidate_count": 1,
            "selected_candidate_index": 0,
            "candidate_cell_xy": [12, 9],
            "start_cell_xy": [12, 9],
            "selected_theta": 0.123,
            "planned_path_cells": [[12, 9]],
            "coverage_gain_cells": 50,
            "path_length_m": 0.0,
            "invalid_action": False,
            "safety_violation": False,
        },
    }


def test_parent_reader_rejects_any_full_file_hash_drift(tmp_path: Path) -> None:
    repair = _load_module()
    parent_path = tmp_path / "results.jsonl"
    expected_sha, _ = _write_parent(parent_path)

    evidence = repair.read_parent_coverage_evidence(
        parent_path,
        expected_sha256=expected_sha,
    )
    assert evidence["results_sha256"] == expected_sha
    assert len(evidence["episodes"]) == 24
    assert evidence["non_coverage_row_count"] == 1
    assert [row["episode_index"] for row in evidence["episodes"]] == list(
        range(24)
    )
    assert all(
        len(row["parent_line_bytes_sha256"]) == 64
        for row in evidence["episodes"]
    )

    parent_path.write_bytes(parent_path.read_bytes() + b" ")
    with pytest.raises(repair.G1RepairBlocked, match="parent_results_sha256_mismatch"):
        repair.read_parent_coverage_evidence(
            parent_path,
            expected_sha256=expected_sha,
        )


def test_parent_reader_rejects_replacing_a_non_candidate_parent_row(
    tmp_path: Path,
) -> None:
    repair = _load_module()
    parent_path = tmp_path / "results.jsonl"
    _, rows = _write_parent(parent_path)
    rows[3]["steps_executed"] = 1
    trace = {
        "row_kind": "decision",
        "schema_version": "xunce-mid-dual-g1-decision/v1",
        "episode_index": 0,
        "step_index": 0,
    }
    payload = b"".join(_canonical_line(row) for row in [*rows, trace])
    parent_path.write_bytes(payload)
    with pytest.raises(
        repair.G1RepairBlocked,
        match="parent_repair_failure_contract_mismatch",
    ):
        repair.read_parent_coverage_evidence(
            parent_path,
            expected_sha256=hashlib.sha256(payload).hexdigest(),
        )


def test_first_step_gate_uses_actual_theta_and_rejects_nonzero_path() -> None:
    repair = _load_module()
    gate = repair.validate_first_step_gate(
        {
            "selected_candidate_index": 0,
            "selected_candidate_cell_xy": [4, 7],
            "selected_theta": 0.08425975,
        },
        {
            "planned_path_cells": [[4, 7]],
            "coverage_gain_cells": 1_301,
            "path_length_m": 0.0,
            "invalid_action": False,
            "safety_violation": False,
        },
        candidate_count=1,
        start_cell_xy=(4, 7),
    )
    assert gate["passed"] is True
    assert gate["selected_theta"] == pytest.approx(0.08425975)

    with pytest.raises(repair.G1RepairBlocked, match="first_step_path_nonzero"):
        repair.validate_first_step_gate(
            {
                "selected_candidate_index": 0,
                "selected_candidate_cell_xy": [4, 7],
                "selected_theta": -2.4,
            },
            {
                "planned_path_cells": [[4, 7]],
                "coverage_gain_cells": 1,
                "path_length_m": 0.5,
                "invalid_action": False,
                "safety_violation": False,
            },
            candidate_count=1,
            start_cell_xy=(4, 7),
        )


def test_merge_is_explicitly_mixed_code_and_recomputes_integer_coverage(
    tmp_path: Path,
) -> None:
    repair = _load_module()
    parent_path = tmp_path / "results.jsonl"
    parent_sha, _ = _write_parent(parent_path)
    parent = repair.read_parent_coverage_evidence(
        parent_path,
        expected_sha256=parent_sha,
    )
    repair_rows = [
        _repair_episode(index, final_count=8_500) for index in TARGETS
    ]

    merged = repair.build_mixed_repair_rows(
        parent_evidence=parent,
        repair_episodes=repair_rows,
        repair_results_sha256="8" * 64,
        repair_code_sha256="7" * 64,
    )
    assert len(merged) == 24
    assert sum(row["result_origin"] == "parent_read_only" for row in merged) == 21
    assert sum(row["result_origin"] == "repair_rerun" for row in merged) == 3
    assert all(row["mixed_code_incremental_repair"] is True for row in merged)
    assert {
        row["episode_index"]
        for row in merged
        if row["result_origin"] == "repair_rerun"
    } == set(TARGETS)

    summary = repair.independently_recompute_merged(merged)
    expected_coverages = [
        (
            8_500 / 10_000
            if index in TARGETS
            else (9_000 + index) / (10_000 + index)
        )
        for index in range(24)
    ]
    assert summary["sample_count"] == 24
    assert summary["mean"] == pytest.approx(
        sum(expected_coverages) / len(expected_coverages)
    )
    assert summary["coverage_80_count"] == 24
    assert summary["coverage_99_count"] == 0
    assert summary["bootstrap_ci"]["resamples"] == 2_000
    assert summary["bootstrap_ci"]["seed"] == 20260726
    assert summary["midterm_reduced_passed"] is True
    assert summary["final_threshold_reduced_passed"] is False
    assert summary["mixed_code_incremental_repair"] is True

    duplicated = [*merged[:-1], merged[0]]
    with pytest.raises(
        repair.G1RepairBlocked,
        match="merged_episode_index_incomplete_or_duplicate",
    ):
        repair.independently_recompute_merged(duplicated)

    forged = _repair_episode(3)
    forged["first_step_gate"]["candidate_cell_xy"] = [99, 99]
    with pytest.raises(
        repair.G1RepairBlocked,
        match="first_step_candidate_not_start",
    ):
        repair.build_mixed_repair_rows(
            parent_evidence=parent,
            repair_episodes=[
                forged,
                _repair_episode(10),
                _repair_episode(23),
            ],
            repair_results_sha256="8" * 64,
            repair_code_sha256="7" * 64,
        )


def test_durable_episode_accept_is_idempotent_and_resume_validates_bytes(
    tmp_path: Path,
) -> None:
    repair = _load_module()
    store = repair.RepairArtifactStore(
        tmp_path / "g1-repair-run",
        immutable_config_sha256="9" * 64,
        expected_episode_lineage={
            "checkpoint_sha256": "5" * 64,
            "policy_state_sha256": "6" * 64,
            "scenario_manifest_sha256": "4" * 64,
            "formal_config_sha256": "2" * 64,
            "repair_code_sha256": "7" * 64,
            "repair_static_config_sha256": "a" * 64,
        },
    )
    rows = [
        _repair_episode(3),
        {
            "row_kind": "decision",
            "schema_version": "xunce-mid-dual-g1-repair-decision/v1",
            "episode_index": 3,
            "step_index": 0,
        },
    ]
    accepted = store.accept_episode(episode_index=3, rows=rows)
    assert accepted["status"] == "accepted"
    assert store.accepted_episode_indices() == (3,)
    state_before = (tmp_path / "g1-repair-run/job-state.jsonl").read_bytes()

    resumed = store.accept_episode(episode_index=3, rows=rows)
    assert resumed["status"] == "already_accepted"
    assert (tmp_path / "g1-repair-run/job-state.jsonl").read_bytes() == state_before

    results_path = (
        tmp_path
        / "g1-repair-run"
        / "phases"
        / "p03"
        / "episodes"
        / "e03"
        / "results.jsonl"
    )
    results_path.write_bytes(results_path.read_bytes() + b" ")
    with pytest.raises(
        repair.G1RepairBlocked,
        match="accepted_episode_results_sha256_mismatch",
    ):
        store.accepted_episode_indices()


def test_durable_accept_rejects_a_forged_passed_first_step_gate(
    tmp_path: Path,
) -> None:
    repair = _load_module()
    store = repair.RepairArtifactStore(
        tmp_path / "g1-repair-run",
        immutable_config_sha256="9" * 64,
        expected_episode_lineage={
            "checkpoint_sha256": "5" * 64,
            "policy_state_sha256": "6" * 64,
            "scenario_manifest_sha256": "4" * 64,
            "formal_config_sha256": "2" * 64,
            "repair_code_sha256": "7" * 64,
            "repair_static_config_sha256": "a" * 64,
        },
    )
    forged = _repair_episode(10)
    forged["first_step_gate"]["coverage_gain_cells"] = 0
    with pytest.raises(
        repair.G1RepairBlocked,
        match="first_step_coverage_gain_nonpositive",
    ):
        store.accept_episode(episode_index=10, rows=[forged])


def test_durable_accept_rejects_self_consistent_wrong_episode_lineage(
    tmp_path: Path,
) -> None:
    repair = _load_module()
    store = repair.RepairArtifactStore(
        tmp_path / "g1-repair-run",
        immutable_config_sha256="9" * 64,
        expected_episode_lineage={
            "checkpoint_sha256": "5" * 64,
            "policy_state_sha256": "6" * 64,
            "scenario_manifest_sha256": "4" * 64,
            "formal_config_sha256": "2" * 64,
            "repair_code_sha256": "7" * 64,
            "repair_static_config_sha256": "a" * 64,
        },
    )
    forged = _repair_episode(23)
    forged["repair_code_sha256"] = "0" * 64
    with pytest.raises(
        repair.G1RepairBlocked,
        match="repair_episode_lineage_mismatch",
    ):
        store.accept_episode(episode_index=23, rows=[forged])
