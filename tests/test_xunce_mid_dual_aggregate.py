"""Task 10 独立 aggregate：只从 manifest 与逐行结果复算双门槛。"""

from __future__ import annotations

import importlib
from pathlib import Path
import sys
from typing import Callable

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from xunce_mid_dual_artifacts import MidDualRunStore  # noqa: E402
from xunce_mid_dual_contracts import (  # noqa: E402
    G2_PLATFORMS,
    G2_REPEATS,
    SCALE_PROFILE,
)


_HASH_A = "a" * 64
_HASH_B = "b" * 64
_HASH_C = "c" * 64


def _aggregate_module():
    try:
        return importlib.import_module("run_xunce_mid_dual_aggregate")
    except ModuleNotFoundError:
        pytest.fail("Task 10 aggregate runner is not implemented")


def _environment_probe() -> dict[str, object]:
    return {
        "windows_version": "Windows test",
        "cpu_model": "test cpu",
        "cpu_logical_count": 8,
        "memory_bytes": 1024,
        "gpu": {"model": "test gpu", "driver": "test driver", "cuda": "test cuda"},
        "python_executable": "D:/conda_envs/lunar-explorer/python.exe",
        "python_version": "3.12.13",
        "frozen_dependencies": ["pytest==8.0"],
        "python_hash_seed": "0",
        "thread_variables": {"OMP_NUM_THREADS": "1"},
        "worker_start_method": "spawn",
        "power_mode": "best-performance",
    }


def _source_config(gate_id: str) -> dict[str, object]:
    return {
        "schema_version": "mid-dual-effective-config/v1",
        "gate_id": gate_id,
        "run_id": f"{gate_id}-run",
        "scale_profile": SCALE_PROFILE,
        "input_sha256": _HASH_A,
        "code_sha256": _HASH_B,
        "required_phase_ids": ["p01"],
    }


def _create_source_root(
    tmp_path: Path,
    gate_id: str,
    row_factory: Callable[[str], list[dict[str, object]]],
    summary: dict[str, object],
) -> Path:
    root = tmp_path / gate_id
    store = MidDualRunStore.create_new(root, _source_config(gate_id))
    rows = row_factory(store.config_sha256)
    attempt = store.write_phase_attempt("p01", rows, {"gate_id": gate_id})
    store.accept_phase("p01", attempt, store.phase_attempt_row_sha256("p01", attempt))
    store.capture_lineage([Path(__file__)], _HASH_A[:40], _HASH_B[:40])
    assert store.capture_environment(_environment_probe)["status"] == "captured"
    stored = {
        "schema_version": f"mid-dual-{gate_id}-summary/v1",
        "scale_profile": SCALE_PROFILE,
        "gate_id": gate_id,
        **summary,
    }
    store.finalize(stored, {"status": stored["status"]}, f"{gate_id} source report", {})
    assert store.verify_manifest(root) is True
    return root


def _g1_rows(
    config_sha256: str,
    *,
    coverage: float = 0.99,
    mutation: Callable[[list[dict[str, object]]], None] | None = None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for split in ("test_q24", "unseen24"):
        for index in range(24):
            rows.append(
                {
                    "row_kind": "g1_coverage_episode",
                    "schema_version": "coverage-episode-row/v1",
                    "scale_profile": SCALE_PROFILE,
                    "run_id": "g1-run",
                    "episode_id": f"{split}-episode-{index:02d}",
                    "scenario_id": f"{split}-scenario-{index:02d}",
                    "split": split,
                    "lane_id": f"lane-{index % 8}",
                    "source_sha256": _HASH_B,
                    "config_sha256": config_sha256,
                    "input_sha256": _HASH_A,
                    "code_sha256": _HASH_B,
                    "checkpoint_sha256": _HASH_C,
                    "denominator_sha256": _HASH_A,
                    "coverage": coverage,
                    "elapsed_ms": 10.0,
                    "safety_violation_count": 0,
                    "masked_action_count": 0,
                }
            )
    if mutation is not None:
        mutation(rows)
    return rows


def _g2_rows(
    config_sha256: str,
    *,
    elapsed_ms: float = 100.0,
    mutation: Callable[[list[dict[str, object]]], None] | None = None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for platform in G2_PLATFORMS:
        request_index = 0
        for scale, reachable_count, request_count in (
            ("standard", 30, 33),
            ("kilometer", 8, 10),
        ):
            for scale_index in range(request_count):
                reachable = scale_index < reachable_count
                request_id = f"{platform}-request-{request_index:02d}"
                for repeat in range(G2_REPEATS):
                    rows.append(
                        {
                            "row_kind": "g2_planning_call",
                            "schema_version": "planning-call-row/v1",
                            "scale_profile": SCALE_PROFILE,
                            "run_id": "g2-run",
                            "episode_id": request_id,
                            "request_id": request_id,
                            "call_id": f"{platform}-call-{request_index:02d}-{repeat}",
                            "platform": platform,
                            "scale": scale,
                            "outcome_kind": "reachable" if reachable else "unreachable",
                            "source_sha256": _HASH_B,
                            "config_sha256": config_sha256,
                            "input_sha256": _HASH_A,
                            "code_sha256": _HASH_B,
                            "request_sha256": _HASH_C,
                            "provider_sha256": _HASH_B,
                            "oracle_sha256": _HASH_A,
                            "elapsed_ms": elapsed_ms,
                            "input_validation_ms": 10.0,
                            "platform_instantiation_ms": 10.0,
                            "search_ms": elapsed_ms - 40.0,
                            "complete_route_validation_ms": 10.0,
                            "result_assembly_ms": 10.0,
                            "provider_success": reachable,
                            "route_l2_valid": reachable,
                            "semantic_digest": f"semantic-{platform}-{request_index:02d}",
                            "formal_sample": True,
                            "repeat_index": repeat,
                        }
                    )
                request_index += 1
    if mutation is not None:
        mutation(rows)
    return rows


def _g3_rows(
    config_sha256: str,
    *,
    coverage: float = 0.99,
    elapsed_ms: float = 100.0,
    mutation: Callable[[list[dict[str, object]]], None] | None = None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index in range(10):
        split = "test_q24" if index < 5 else "unseen24"
        episode_id = f"wheel-episode-{index:02d}"
        request_id = f"wheel-request-{index:02d}"
        rows.append(
            {
                "row_kind": "g3_wheel_step",
                "schema_version": "closed-loop-step-row/v1",
                "scale_profile": SCALE_PROFILE,
                "run_id": "g3-run",
                "episode_id": episode_id,
                "scenario_id": f"{split}-scenario-{index % 5:02d}",
                "split": split,
                "step_id": f"{episode_id}-step-00",
                "candidate_id": f"candidate-{index:02d}",
                "request_id": request_id,
                "route_request_id": request_id,
                "feedback_request_id": request_id,
                "source_sha256": _HASH_B,
                "config_sha256": config_sha256,
                "input_sha256": _HASH_A,
                "code_sha256": _HASH_B,
                "planner_sha256": _HASH_C,
                "coverage": coverage,
                "paired_g1_coverage": coverage,
                "planner_elapsed_ms": elapsed_ms,
                "is_terminal": True,
                "join_valid": True,
                "safety_violation_count": 0,
                "masked_action_count": 0,
                "candidate_route_mismatch_count": 0,
            }
        )
    for platform in ("legged", "hopper"):
        for index in range(3):
            request_id = f"{platform}-request-{index:02d}"
            rows.append(
                {
                    "row_kind": "g3_interface_replay",
                    "schema_version": "interface-replay-row/v1",
                    "scale_profile": SCALE_PROFILE,
                    "run_id": "g3-run",
                    "episode_id": f"{platform}-episode-{index:02d}",
                    "replay_id": f"{platform}-replay-{index:02d}",
                    "request_id": request_id,
                    "platform": platform,
                    "source_sha256": _HASH_B,
                    "config_sha256": config_sha256,
                    "input_sha256": _HASH_A,
                    "code_sha256": _HASH_B,
                    "request_sha256": _HASH_C,
                    "result_sha256": _HASH_C,
                    "elapsed_ms": elapsed_ms,
                    "platform_match": True,
                    "request_match": True,
                    "result_match": True,
                    "timing_contract_valid": True,
                    "formal_input_eligible": True,
                }
            )
    if mutation is not None:
        mutation(rows)
    return rows


def _valid_roots(tmp_path: Path) -> tuple[Path, Path, Path]:
    g1 = _create_source_root(
        tmp_path,
        "g1",
        _g1_rows,
        {
            "status": "passed",
            "sample_count": 48,
            "g1_coverage_80_passed": True,
            "g1_coverage_99_passed": True,
        },
    )
    g2 = _create_source_root(
        tmp_path,
        "g2",
        _g2_rows,
        {
            "status": "passed",
            "formal_call_count": 645,
            "g2_all_platforms_2s_passed": True,
            "g2_all_platforms_1s_passed": True,
        },
    )
    g3 = _create_source_root(
        tmp_path,
        "g3",
        _g3_rows,
        {
            "status": "passed",
            "wheel_episode_count": 10,
            "interface_replay_count": 6,
            "g3_midterm_crosscheck_passed": True,
            "g3_final_crosscheck_passed": True,
        },
    )
    return g1, g2, g3


def test_aggregate_reads_results_and_manifest_not_gate_summary_values(tmp_path: Path) -> None:
    """Catch a forged passing gate summary overriding failed raw G1 rows."""
    g1 = _create_source_root(
        tmp_path,
        "g1",
        lambda config_sha256: _g1_rows(config_sha256, coverage=0.79),
        {
            "status": "passed",
            "sample_count": 48,
            "g1_coverage_80_passed": True,
            "g1_coverage_99_passed": True,
        },
    )
    g2 = _create_source_root(
        tmp_path,
        "g2",
        _g2_rows,
        {
            "status": "passed",
            "formal_call_count": 645,
            "g2_all_platforms_2s_passed": True,
            "g2_all_platforms_1s_passed": True,
        },
    )
    g3 = _create_source_root(
        tmp_path,
        "g3",
        _g3_rows,
        {
            "status": "passed",
            "wheel_episode_count": 10,
            "interface_replay_count": 6,
            "g3_midterm_crosscheck_passed": True,
            "g3_final_crosscheck_passed": True,
        },
    )

    result = _aggregate_module().aggregate_completed_roots(g1, g2, g3)

    assert result["status"] == "blocked"
    assert result["gates"]["g1"]["g1_coverage_80_passed"] is False
    assert "g1_stored_summary_mismatch" in result["blockers"]


def test_aggregate_recomputes_g1_g2_and_g3_from_raw_rows(tmp_path: Path) -> None:
    """Catch an aggregate that forwards gate booleans instead of recalculating rows."""
    result = _aggregate_module().aggregate_completed_roots(*_valid_roots(tmp_path))

    assert result["status"] == "passed"
    assert result["midterm_reduced_gate_passed"] is True
    assert result["final_threshold_reduced_gate_passed"] is True
    assert result["sample_counts"] == {
        "g1_total_episodes": 48,
        "g1_test_q24": 24,
        "g1_unseen24": 24,
        "g2_formal_calls": 645,
        "g3_wheel_episodes": 10,
        "g3_interface_replays": 6,
    }
    assert len(result["gates"]["g2"]["timing_by_platform_scale"]) == 6


@pytest.mark.parametrize("mutation_kind", ("scale_profile", "input_sha256", "code_sha256"))
def test_aggregate_rejects_scale_profile_input_and_code_hash_drift(
    tmp_path: Path,
    mutation_kind: str,
) -> None:
    """Catch row provenance that no longer matches the manifest-bound source config."""

    def mutate(rows: list[dict[str, object]]) -> None:
        rows[0][mutation_kind] = "wrong-profile/v1" if mutation_kind == "scale_profile" else _HASH_C

    g1 = _create_source_root(
        tmp_path,
        "g1",
        lambda config_sha256: _g1_rows(config_sha256, mutation=mutate),
        {
            "status": "passed",
            "sample_count": 48,
            "g1_coverage_80_passed": True,
            "g1_coverage_99_passed": True,
        },
    )
    g2, g3 = _valid_roots(tmp_path / "other")[1:]
    result = _aggregate_module().aggregate_completed_roots(g1, g2, g3)

    assert result["status"] == "blocked"
    assert any(mutation_kind in reason for reason in result["blockers"])


@pytest.mark.parametrize(
    "mutation_kind",
    ("missing", "duplicate", "cross_platform", "duplicate_repeat", "invalid_outcome"),
)
def test_aggregate_rejects_missing_duplicate_and_cross_platform_rows(
    tmp_path: Path,
    mutation_kind: str,
) -> None:
    """Catch an incomplete, duplicate, or platform-mixed formal G2 matrix."""

    def mutate(rows: list[dict[str, object]]) -> None:
        if mutation_kind == "missing":
            rows.pop()
        elif mutation_kind == "duplicate":
            rows[-1]["call_id"] = rows[0]["call_id"]
        elif mutation_kind == "cross_platform":
            rows[215]["request_id"] = rows[0]["request_id"]
        elif mutation_kind == "duplicate_repeat":
            rows[1]["repeat_index"] = rows[0]["repeat_index"]
        else:
            rows[-1]["outcome_kind"] = "unknown"

    g1 = _create_source_root(
        tmp_path,
        "g1",
        _g1_rows,
        {
            "status": "passed",
            "sample_count": 48,
            "g1_coverage_80_passed": True,
            "g1_coverage_99_passed": True,
        },
    )
    g2 = _create_source_root(
        tmp_path,
        "g2",
        lambda config_sha256: _g2_rows(config_sha256, mutation=mutate),
        {
            "status": "passed",
            "formal_call_count": 645,
            "g2_all_platforms_2s_passed": True,
            "g2_all_platforms_1s_passed": True,
        },
    )
    g3 = _create_source_root(
        tmp_path,
        "g3",
        _g3_rows,
        {
            "status": "passed",
            "wheel_episode_count": 10,
            "interface_replay_count": 6,
            "g3_midterm_crosscheck_passed": True,
            "g3_final_crosscheck_passed": True,
        },
    )
    result = _aggregate_module().aggregate_completed_roots(g1, g2, g3)

    assert result["status"] == "blocked"
    assert any("g2_" in reason for reason in result["blockers"])


@pytest.mark.parametrize(
    "mutation_kind",
    ("missing_episode", "duplicate_replay", "split_leakage"),
)
def test_aggregate_rejects_g3_missing_duplicate_and_split_leakage(
    tmp_path: Path,
    mutation_kind: str,
) -> None:
    """Catch a G3 matrix with a missing episode, duplicate replay, or split leak."""

    def mutate(rows: list[dict[str, object]]) -> None:
        if mutation_kind == "missing_episode":
            rows.pop(0)
        elif mutation_kind == "duplicate_replay":
            rows[-1]["replay_id"] = rows[-2]["replay_id"]
        else:
            leaked = dict(rows[0])
            leaked["step_id"] = "wheel-episode-00-step-01"
            leaked["split"] = "unseen24"
            leaked["is_terminal"] = False
            rows.append(leaked)

    g1, g2, _ = _valid_roots(tmp_path / "valid")
    g3 = _create_source_root(
        tmp_path,
        "g3",
        lambda config_sha256: _g3_rows(config_sha256, mutation=mutate),
        {
            "status": "passed",
            "wheel_episode_count": 10,
            "interface_replay_count": 6,
            "g3_midterm_crosscheck_passed": True,
            "g3_final_crosscheck_passed": True,
        },
    )

    result = _aggregate_module().aggregate_completed_roots(g1, g2, g3)

    assert result["status"] == "blocked"
    assert any("g3_" in reason for reason in result["blockers"])


def test_aggregate_blocks_when_recomputed_and_stored_summaries_differ(tmp_path: Path) -> None:
    """Catch a complete root whose stored G2 gate disagrees with the raw rows."""
    g1, _, g3 = _valid_roots(tmp_path / "valid")
    g2 = _create_source_root(
        tmp_path,
        "g2",
        _g2_rows,
        {
            "status": "failed",
            "formal_call_count": 645,
            "g2_all_platforms_2s_passed": False,
            "g2_all_platforms_1s_passed": False,
        },
    )

    result = _aggregate_module().aggregate_completed_roots(g1, g2, g3)

    assert result["status"] == "blocked"
    assert result["gates"]["g2"]["g2_all_platforms_2s_passed"] is True
    assert "g2_stored_summary_mismatch" in result["blockers"]


def test_midterm_reduced_truth_table_requires_all_three_gates() -> None:
    """Catch OR/majority routing at the reduced midterm boundary."""
    truth = _aggregate_module().midterm_reduced_truth_table
    assert truth(True, True, True) is True
    assert truth(False, True, True) is False
    assert truth(True, False, True) is False
    assert truth(True, True, False) is False


def test_final_threshold_reduced_truth_table_requires_all_three_gates() -> None:
    """Catch final-threshold routing that omits any G1/G2/G3 qualification."""
    truth = _aggregate_module().final_threshold_reduced_truth_table
    assert truth(True, True, True) is True
    assert truth(False, True, True) is False
    assert truth(True, False, True) is False
    assert truth(True, True, False) is False


def test_blocked_is_not_rendered_as_failed_or_passed() -> None:
    """Catch evidence-not-ready being described as a measured metric failure."""
    module = _aggregate_module()
    summary = module.blocked_summary(["missing_g2_manifest"])
    report = module.render_report(summary)

    assert summary["status"] == "blocked"
    assert summary["midterm_reduced_gate_passed"] is False
    assert summary["final_threshold_reduced_gate_passed"] is False
    assert "证据未就绪" in report
    assert "指标未通过" not in report


def test_report_contains_reduced_scale_qualifier_and_exact_sample_counts(tmp_path: Path) -> None:
    """Catch a report that drops the reduced qualifier or hides exact denominators."""
    module = _aggregate_module()
    summary = module.aggregate_completed_roots(*_valid_roots(tmp_path))
    report = module.render_report(summary)

    assert "缩减规模中期实验（G1 24 场景/split）" in report
    assert "G1 Test-Q24 | 24" in report
    assert "G1 Unseen-24 | 24" in report
    assert "G2 正式调用 | 645" in report
    assert "G3 轮式闭环 | 10" in report
    assert "G3 接口回放 | 6" in report
    for group in (
        "wheel/standard",
        "wheel/kilometer",
        "legged/standard",
        "legged/kilometer",
        "hopper/standard",
        "hopper/kilometer",
    ):
        assert group in report
    assert "G1 Test-Q24" in report and "G1 Unseen-24" in report
    assert "G3 轮式覆盖均值" in report


def test_aggregate_config_rejects_frozen_actual_sample_count_drift(tmp_path: Path) -> None:
    """Catch a CLI config that silently changes the Task 10 exact scale."""
    from xunce_artifact_io import read_json, write_json

    module = _aggregate_module()
    source = Path(__file__).resolve().parents[1] / "configs" / "xunce_mid_dual_aggregate_v1.json"
    config = read_json(source)
    config["actual_sample_counts"]["g2_formal_calls"] = 644
    path = tmp_path / "aggregate-config.json"
    write_json(path, config)

    with pytest.raises(ValueError, match="actual_sample_counts"):
        module._load_aggregate_config(path)
