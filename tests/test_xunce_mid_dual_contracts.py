"""Behavioral contracts for the reduced midterm dual-gate evidence chain."""

from __future__ import annotations

import math
import sys
import hashlib
import importlib
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from xunce_mid_dual_contracts import (  # noqa: E402
    BOOTSTRAP_RESAMPLES,
    CoverageEpisodeRow,
    FINAL_COVERAGE_THRESHOLD,
    FINAL_TIME_MS,
    G1_EPISODES_PER_FORMAL_SPLIT,
    G1_LANE_SIZES,
    G2_FORMAL_CALLS,
    G2_KILOMETER_REQUESTS,
    G2_PLATFORMS,
    G2_REPEATS,
    G2_STANDARD_REQUESTS,
    MID_COVERAGE_THRESHOLD,
    MID_TIME_MS,
    PlanningCallRow,
    SCALE_PROFILE,
    evaluate_formal_g2,
    evaluate_g1_split,
    evaluate_g2_platform,
    episode_bootstrap_ci,
    nearest_rank,
    reachable_request_success_rate,
    route_gate,
)


_HASH = "a" * 64


def _formal_g2_rows() -> list[PlanningCallRow]:
    rows: list[PlanningCallRow] = []
    for platform in G2_PLATFORMS:
        request_index = 0
        for scale, count in (("standard", 33), ("kilometer", 10)):
            for _ in range(count):
                outcome_kind = "reachable" if request_index < 38 else "unreachable"
                request_id = f"{platform}-request-{request_index:02d}"
                for repeat in range(G2_REPEATS):
                    rows.append(
                        PlanningCallRow(
                            schema_version="planning-call-row/v1",
                            scale_profile=SCALE_PROFILE,
                            run_id="run-1",
                            episode_id="episode-1",
                            request_id=request_id,
                            call_id=f"{platform}-call-{request_index:02d}-{repeat}",
                            platform=platform,
                            scale=scale,
                            outcome_kind=outcome_kind,
                            source_sha256=_HASH,
                            config_sha256=_HASH,
                            request_sha256=_HASH,
                            provider_sha256=_HASH,
                            oracle_sha256=_HASH,
                            elapsed_ms=10.0,
                            input_validation_ms=1.0,
                            platform_instantiation_ms=1.0,
                            search_ms=6.0,
                            complete_route_validation_ms=1.0,
                            result_assembly_ms=1.0,
                            provider_success=True,
                            route_l2_valid=True,
                            semantic_digest=f"semantic-{platform}-{request_index:02d}",
                        )
                    )
                request_index += 1
    return rows


def _coverage_row() -> CoverageEpisodeRow:
    return CoverageEpisodeRow(
        schema_version="coverage-episode-row/v1",
        scale_profile=SCALE_PROFILE,
        run_id="run-1",
        episode_id="episode-1",
        scenario_id="scenario-1",
        source_sha256=_HASH,
        config_sha256=_HASH,
        checkpoint_sha256=_HASH,
        denominator_sha256=_HASH,
        coverage=0.80,
        elapsed_ms=10.0,
        safety_violation_count=0,
        masked_action_count=0,
    )


def test_scale_profile_freezes_w8x3_update80_contract() -> None:
    """Catch accidental widening, reshaping, or checkpoint-update drift."""
    assert SCALE_PROFILE == "midterm_reduced_w8x3_update80/v1"
    assert G1_EPISODES_PER_FORMAL_SPLIT == 24
    assert G1_LANE_SIZES == (3, 3, 3, 3, 3, 3, 3, 3)
    assert sum(G1_LANE_SIZES) == G1_EPISODES_PER_FORMAL_SPLIT


def test_g1_requires_exactly_24_unique_jobs_and_eight_lanes_of_three() -> None:
    """Catch a G1 denominator with duplicate/missing jobs or unequal lanes."""
    jobs = tuple(f"scene-{index:02d}" for index in range(24))
    balanced_lanes = tuple(f"lane-{index // 3}" for index in range(24))
    assert evaluate_g1_split(jobs, [0.80] * 24, lane_ids=balanced_lanes)["status"] == "passed"

    duplicate_jobs = (*jobs[:-1], jobs[-2])
    assert evaluate_g1_split(duplicate_jobs, [0.80] * 24, lane_ids=balanced_lanes)["status"] == "blocked"
    assert evaluate_g1_split(jobs[:-1], [0.80] * 23, lane_ids=balanced_lanes[:-1])["status"] == "blocked"

    unbalanced_lanes = ("lane-0",) * 24
    assert evaluate_g1_split(jobs, [0.80] * 24, lane_ids=unbalanced_lanes)["status"] == "blocked"


def test_g2_requires_exactly_645_formal_calls() -> None:
    """Catch changes that silently reduce the formal G2 sample matrix."""
    assert G2_PLATFORMS == ("wheel", "legged", "hopper")
    assert G2_STANDARD_REQUESTS == 33
    assert G2_KILOMETER_REQUESTS == 10
    assert G2_REPEATS == 5
    assert G2_FORMAL_CALLS == 645
    assert len(G2_PLATFORMS) * (G2_STANDARD_REQUESTS + G2_KILOMETER_REQUESTS) * G2_REPEATS == G2_FORMAL_CALLS


def test_threshold_boundaries_are_inclusive_at_080_099_1000_2000() -> None:
    """Catch strict comparisons that reject values exactly at a gate boundary."""
    lanes = tuple(f"lane-{index // 3}" for index in range(24))
    coverage = evaluate_g1_split(tuple(f"scene-{index}" for index in range(24)), [0.80] * 24, lane_ids=lanes)
    assert coverage["midterm_reduced_passed"] is True
    assert coverage["final_threshold_reduced_passed"] is False

    final_coverage = evaluate_g1_split(tuple(f"final-{index}" for index in range(24)), [0.99] * 24, lane_ids=lanes)
    assert final_coverage["final_threshold_reduced_passed"] is True

    mid_timing = evaluate_g2_platform(
        platform="wheel",
        scale="standard",
        outcome_kind="success",
        elapsed_ms=[2000.0] * 5,
    )
    assert mid_timing["midterm_reduced_passed"] is True
    final_timing = evaluate_g2_platform(
        platform="wheel",
        scale="standard",
        outcome_kind="success",
        elapsed_ms=[1000.0] * 5,
    )
    assert final_timing["final_threshold_reduced_passed"] is True
    assert MID_COVERAGE_THRESHOLD == 0.80
    assert FINAL_COVERAGE_THRESHOLD == 0.99
    assert MID_TIME_MS == 2000.0
    assert FINAL_TIME_MS == 1000.0


def test_nearest_rank_uses_ceil_qn_one_based_index() -> None:
    """Catch interpolation or zero-based percentile indexing in timing reports."""
    assert nearest_rank([10.0, 20.0, 30.0, 40.0], 0.95) == 40.0
    assert nearest_rank([10.0, 20.0, 30.0, 40.0], 0.50) == 20.0
    assert nearest_rank([30.0, 10.0, 20.0], 0.01) == 10.0


def test_episode_bootstrap_is_fixed_seed_and_uses_2000_resamples() -> None:
    """Catch nondeterministic or wrongly sized episode-level bootstrap output."""
    values = [0.10, 0.20, 0.30, 0.40]
    first = episode_bootstrap_ci(values)
    second = episode_bootstrap_ci(values)
    assert BOOTSTRAP_RESAMPLES == 2000
    assert first == second
    assert first["resamples"] == 2000
    assert first["unit"] == "episode"
    assert first["lower"] <= 0.25 <= first["upper"]


def test_nonfinite_values_block_instead_of_being_filtered() -> None:
    """Catch NaN/inf filtering that would understate failed formal samples."""
    jobs = tuple(f"scene-{index}" for index in range(24))
    lanes = tuple(f"lane-{index // 3}" for index in range(24))
    assert evaluate_g1_split(jobs, [0.80] * 23 + [math.nan], lane_ids=lanes)["status"] == "blocked"
    assert evaluate_g2_platform(
        platform="wheel",
        scale="standard",
        outcome_kind="success",
        elapsed_ms=[1.0, math.inf],
    )["status"] == "blocked"


def test_formal_g2_blocks_missing_extra_duplicate_and_incomplete_platform_matrix() -> None:
    """Catch any 645-call matrix that is not exactly three 43-request platforms."""
    rows = _formal_g2_rows()
    assert evaluate_formal_g2(rows)["status"] == "passed"
    assert evaluate_formal_g2(rows[:-1])["status"] == "blocked"

    extra = [*rows, replace(rows[0], call_id="wheel-extra-call")]
    assert evaluate_formal_g2(extra)["status"] == "blocked"

    duplicate_call_id = [*rows]
    duplicate_call_id[-1] = replace(duplicate_call_id[-1], call_id=duplicate_call_id[0].call_id)
    assert evaluate_formal_g2(duplicate_call_id)["status"] == "blocked"

    incomplete_platform = [*rows]
    incomplete_platform[-1] = replace(
        incomplete_platform[-1],
        platform="wheel",
        request_id="wheel-request-extra",
        call_id="wheel-call-extra",
        semantic_digest="semantic-wheel-extra",
    )
    assert evaluate_formal_g2(incomplete_platform)["status"] == "blocked"


def test_formal_g2_gates_each_platform_scale_outcome_partition() -> None:
    """Catch a slow outcome partition hidden by a fast platform-scale aggregate p95."""
    rows = _formal_g2_rows()
    slow_indices = [
        index
        for index, row in enumerate(rows)
        if row.platform == "wheel" and row.scale == "kilometer" and row.outcome_kind == "unreachable"
    ][:2]
    for index in slow_indices:
        rows[index] = replace(rows[index], elapsed_ms=2001.0, search_ms=1997.0)

    routed = evaluate_formal_g2(rows)
    assert routed["midterm_reduced_passed"] is False
    assert routed["final_threshold_reduced_passed"] is False
    partition = routed["timing_by_platform_scale_outcome"][("wheel", "kilometer", "unreachable")]
    assert partition["p95_ms"] == 2001.0


@pytest.mark.parametrize("hash_field", ["provider_sha256", "oracle_sha256"])
def test_formal_g2_blocks_repeat_provider_or_oracle_provenance_drift(hash_field: str) -> None:
    """Catch a 645-call matrix whose repeated request changes executor provenance."""
    rows = _formal_g2_rows()
    rows[1] = replace(rows[1], **{hash_field: "b" * 64})
    routed = evaluate_formal_g2(rows)
    assert routed["status"] == "blocked"
    assert routed["blocking_reason"] == "g2_request_repeat_or_provenance_mismatch"


def test_reachable_consensus_requires_per_platform_38_unique_repeated_requests() -> None:
    """Catch undersized, duplicate, cross-platform, or 37/38 reachable evidence."""
    rows = _formal_g2_rows()
    assert reachable_request_success_rate(rows)["status"] == "passed"
    assert reachable_request_success_rate(rows[:5])["status"] == "blocked"

    duplicate_call_id = [*rows]
    duplicate_call_id[1] = replace(duplicate_call_id[1], call_id=duplicate_call_id[0].call_id)
    assert reachable_request_success_rate(duplicate_call_id)["status"] == "blocked"

    cross_platform_request_id = [*rows]
    cross_platform_request_id[215] = replace(cross_platform_request_id[215], request_id=rows[0].request_id)
    assert reachable_request_success_rate(cross_platform_request_id)["status"] == "blocked"

    only_37_reachable = [*rows]
    for index, row in enumerate(only_37_reachable):
        if row.platform == "hopper" and row.request_id == "hopper-request-37":
            only_37_reachable[index] = replace(row, outcome_kind="unreachable")
    assert reachable_request_success_rate(only_37_reachable)["status"] == "blocked"


@pytest.mark.parametrize("malformed", [True, "1.0", None])
def test_statistics_boundaries_block_malformed_values_without_coercion(malformed: object) -> None:
    """Catch bool/string/None coercion or leaked conversion errors at a gate boundary."""
    lanes = tuple(f"lane-{index // 3}" for index in range(24))
    assert evaluate_g1_split(tuple(f"scene-{index}" for index in range(24)), [0.80] * 23 + [malformed], lane_ids=lanes)["status"] == "blocked"
    assert evaluate_g2_platform(
        platform="wheel",
        scale="standard",
        outcome_kind="success",
        elapsed_ms=[10.0, malformed],
    )["status"] == "blocked"


@pytest.mark.parametrize("malformed", [0, "", []])
def test_g1_blocks_malformed_job_ids_before_set_operations(malformed: object) -> None:
    """Catch non-string/empty/unhashable formal G1 identifiers leaking or raising."""
    jobs: list[object] = [f"scene-{index}" for index in range(24)]
    jobs[0] = malformed
    lanes = tuple(f"lane-{index // 3}" for index in range(24))
    assert evaluate_g1_split(jobs, [0.80] * 24, lane_ids=lanes)["status"] == "blocked"


def test_episode_bootstrap_rejects_seed_override() -> None:
    """Catch callers changing the frozen bootstrap random seed."""
    with pytest.raises(TypeError):
        episode_bootstrap_ci([0.10, 0.20], seed=1)


@pytest.mark.parametrize("field_name", ["provider_success", "route_l2_valid"])
@pytest.mark.parametrize("malformed", ["false", 1, None])
def test_planning_call_row_rejects_non_bool_success_and_validity_evidence(field_name: str, malformed: object) -> None:
    """Catch truthy strings/integers leaking through G2 success and validity evidence."""
    with pytest.raises(ValueError):
        replace(_formal_g2_rows()[0], **{field_name: malformed})


@pytest.mark.parametrize("field_name", ["safety_violation_count", "masked_action_count"])
@pytest.mark.parametrize("malformed", [True, 1.5, None, "0"])
def test_coverage_episode_row_rejects_non_integer_count_evidence(field_name: str, malformed: object) -> None:
    """Catch bool, fractional, null, and string evidence in coverage count fields."""
    with pytest.raises(ValueError):
        replace(_coverage_row(), **{field_name: malformed})


def test_reduced_pass_fields_have_no_unqualified_pass_alias() -> None:
    """Catch a bare pass field that loses the reduced-scale qualification."""
    routing = route_gate(midterm=True, final=False)
    assert routing == {
        "status": "passed",
        "midterm_reduced_passed": True,
        "final_threshold_reduced_passed": False,
    }
    assert "passed" not in routing


def _artifact_store_type() -> type[object]:
    return importlib.import_module("xunce_mid_dual_artifacts").MidDualRunStore


def _artifact_config(required_phase_ids: tuple[str, ...] = ("p01",)) -> dict[str, object]:
    return {
        "schema_version": "mid-dual-effective-config/v1",
        "run_id": "test-run",
        "seed": 7,
        "required_phase_ids": list(required_phase_ids),
    }


def _artifact_rows(phase_id: str) -> list[dict[str, object]]:
    return [{"phase_id": phase_id, "row_id": f"{phase_id}-row-1", "value": 1}]


def _environment_probe() -> dict[str, object]:
    return {
        "windows_version": "Windows test",
        "cpu_model": "test cpu",
        "cpu_logical_count": 8,
        "memory_bytes": 1024,
        "gpu": {"model": "test gpu", "driver": "test driver", "cuda": "test cuda"},
        "python_executable": "D:/conda_envs/lunar-explorer/python.exe",
        "python_version": "3.12.0",
        "frozen_dependencies": ["pytest==8.0"],
        "python_hash_seed": "0",
        "thread_variables": {"OMP_NUM_THREADS": "1"},
        "worker_start_method": "spawn",
        "power_mode": "best-performance",
    }


def _capture_valid_preflight(store: object) -> None:
    store.capture_lineage([Path(__file__)], "a" * 40, "b" * 40)
    assert store.capture_environment(_environment_probe)["status"] == "captured"


def test_run_store_refuses_an_existing_run_root(tmp_path: Path) -> None:
    """Catch accidental overwrite of an existing experiment evidence root."""
    run_root = tmp_path / "existing"
    run_root.mkdir()
    with pytest.raises(FileExistsError):
        _artifact_store_type().create_new(run_root, _artifact_config())


def test_resume_accepts_only_a_contiguous_hash_valid_phase_prefix(tmp_path: Path) -> None:
    """Catch resume after a missing phase or a changed accepted phase payload."""
    store_type = _artifact_store_type()
    store = store_type.create_new(tmp_path / "resume", _artifact_config(("p01", "p02")))
    first_attempt = store.write_phase_attempt("p01", _artifact_rows("p01"), {"kind": "first"})
    first_hash = store.accept_phase("p01", first_attempt, store.phase_attempt_row_sha256("p01", first_attempt))
    assert len(first_hash) == 64
    second_attempt = store.write_phase_attempt("p02", _artifact_rows("p02"), {"kind": "second"})
    store.accept_phase("p02", second_attempt, store.phase_attempt_row_sha256("p02", second_attempt))
    config_sha256 = store.config_sha256
    resumed = store_type.load_for_resume(tmp_path / "resume", config_sha256)
    assert resumed.accepted_phase_ids == ("p01", "p02")
    from xunce_artifact_io import write_jsonl

    write_jsonl(tmp_path / "resume" / "phases" / "p02" / second_attempt / "results.jsonl", [{"changed": True}])
    with pytest.raises(ValueError, match="hash"):
        store_type.load_for_resume(tmp_path / "resume", config_sha256)


def test_incomplete_phase_is_not_merged_into_final_results(tmp_path: Path) -> None:
    """Catch merging a written-but-unaccepted phase into formal final evidence."""
    store = _artifact_store_type().create_new(tmp_path / "incomplete", _artifact_config())
    accepted_attempt = store.write_phase_attempt("p01", _artifact_rows("p01"), {"kind": "accepted"})
    store.accept_phase("p01", accepted_attempt, store.phase_attempt_row_sha256("p01", accepted_attempt))
    store.write_phase_attempt("p02", _artifact_rows("p02"), {"kind": "incomplete"})
    _capture_valid_preflight(store)
    store.finalize({"status": "complete"}, {"status": "passed"}, "report", {})
    from xunce_artifact_io import read_jsonl

    assert read_jsonl(tmp_path / "incomplete" / "results.jsonl") == _artifact_rows("p01")


def test_finalize_writes_the_seven_required_canonical_artifacts_once(tmp_path: Path) -> None:
    """Catch incomplete finalization or overwriting a canonical evidence artifact."""
    store = _artifact_store_type().create_new(tmp_path / "final", _artifact_config())
    attempt = store.write_phase_attempt("p01", _artifact_rows("p01"), {})
    store.accept_phase("p01", attempt, store.phase_attempt_row_sha256("p01", attempt))
    _capture_valid_preflight(store)
    store.finalize({"status": "complete"}, {"status": "passed"}, "report", {})
    from xunce_artifact_io import path_is_file

    assert all(
        path_is_file(tmp_path / "final" / name)
        for name in ("config.json", "results.jsonl", "summary.json", "routing.json", "manifest.json", "phase-state.jsonl", "report.md")
    )
    with pytest.raises(FileExistsError):
        store.finalize({"status": "complete"}, {"status": "passed"}, "report", {})


def test_manifest_hashes_every_artifact_except_itself(tmp_path: Path) -> None:
    """Catch a manifest that omits a canonical artifact or tries to hash itself."""
    store = _artifact_store_type().create_new(tmp_path / "manifest", _artifact_config())
    attempt = store.write_phase_attempt("p01", _artifact_rows("p01"), {})
    store.accept_phase("p01", attempt, store.phase_attempt_row_sha256("p01", attempt))
    _capture_valid_preflight(store)
    store.finalize({"status": "complete"}, {"status": "passed"}, "report", {})
    from xunce_artifact_io import read_bytes, read_json

    manifest = read_json(tmp_path / "manifest" / "manifest.json")
    hashed = {entry["path"]: entry["sha256"] for entry in manifest["artifacts"]}
    assert "manifest.json" not in hashed
    assert set(("config.json", "results.jsonl", "summary.json", "routing.json", "phase-state.jsonl", "report.md")).issubset(hashed)
    assert all(hashlib.sha256(read_bytes(tmp_path / "manifest" / path)).hexdigest() == digest for path, digest in hashed.items())
    assert store.verify_manifest(tmp_path / "manifest") is True


def test_blocked_preflight_is_terminal_evidence_but_not_pass_evidence(tmp_path: Path) -> None:
    """Catch a blocked machine preflight being reported as a formal gate pass."""
    store = _artifact_store_type().create_new(tmp_path / "blocked", _artifact_config())
    store.finalize({"status": "blocked"}, {"status": "blocked"}, "blocked before execution", {})
    from xunce_artifact_io import read_json, read_jsonl

    assert read_jsonl(tmp_path / "blocked" / "results.jsonl") == []
    summary = read_json(tmp_path / "blocked" / "summary.json")
    routing = read_json(tmp_path / "blocked" / "routing.json")
    assert summary["formal_evidence_eligible"] is False
    assert routing["formal_evidence_eligible"] is False
    assert routing["status"] == "blocked"
    assert routing.get("midterm_reduced_passed") is not True


def test_artifact_module_uses_xunce_io_and_paths_only() -> None:
    """Catch bypassing the repository artifact I/O and canonical-path contracts."""
    module = importlib.import_module("xunce_mid_dual_artifacts")
    source = Path(module.__file__).read_text(encoding="utf-8")
    assert "xunce_artifact_io" in source
    assert "xunce_artifact_paths" in source
    assert not any(token in source for token in ("Path.read_text(", "Path.write_text(", "Path.exists(", "Path.is_file(", "Path.open(", "Path.mkdir("))


def test_lineage_audit_snapshots_dirty_required_sources_by_sha256(tmp_path: Path) -> None:
    """Catch missing byte-for-byte snapshots for required untracked source evidence."""
    source = tmp_path / "required-source.py"
    source.write_bytes(b"required source bytes\n")
    store = _artifact_store_type().create_new(tmp_path / "lineage-run", _artifact_config())
    audit = store.capture_lineage([source], "a" * 40, "b" * 40)
    row = audit["required_sources"][0]
    assert row["status"] == "untracked"
    assert row["snapshot_path"] == "lineage/s0001.bin"
    from xunce_artifact_io import read_bytes

    assert read_bytes(tmp_path / "lineage-run" / row["snapshot_path"]) == b"required source bytes\n"
    assert row["sha256"] == hashlib.sha256(b"required source bytes\n").hexdigest()


def test_environment_audit_binds_python_cpu_gpu_threads_and_power_mode(tmp_path: Path) -> None:
    """Catch guessed or incomplete environment evidence entering a formal run."""
    store = _artifact_store_type().create_new(tmp_path / "environment", _artifact_config())
    audit = store.capture_environment(_environment_probe)
    assert audit["status"] == "captured"
    assert audit["probe"]["python_executable"] == "D:/conda_envs/lunar-explorer/python.exe"
    assert audit["probe"]["cpu_logical_count"] == 8
    assert audit["probe"]["gpu"]["driver"] == "test driver"
    assert audit["probe"]["thread_variables"] == {"OMP_NUM_THREADS": "1"}
    assert audit["probe"]["power_mode"] == "best-performance"


def _finalized_artifact_store(tmp_path: Path, name: str) -> object:
    store = _artifact_store_type().create_new(tmp_path / name, _artifact_config())
    attempt = store.write_phase_attempt("p01", _artifact_rows("p01"), {"kind": "accepted"})
    store.accept_phase("p01", attempt, store.phase_attempt_row_sha256("p01", attempt))
    _capture_valid_preflight(store)
    store.finalize({"status": "complete"}, {"status": "passed"}, "report", {})
    return store


def test_manifest_rejects_snapshot_byte_drift_and_missing_or_empty_entries(tmp_path: Path) -> None:
    """Catch manifest verification that trusts an incomplete self-declared file list."""
    from xunce_artifact_io import read_json, write_json, write_text

    source = tmp_path / "dirty-source.py"
    source.write_bytes(b"original lineage bytes\n")
    store = _artifact_store_type().create_new(tmp_path / "snapshot", _artifact_config())
    store.capture_lineage([source], "a" * 40, "b" * 40)
    attempt = store.write_phase_attempt("p01", _artifact_rows("p01"), {})
    store.accept_phase("p01", attempt, store.phase_attempt_row_sha256("p01", attempt))
    store.finalize({"status": "complete"}, {"status": "passed"}, "report", {})
    run_root = tmp_path / "snapshot"
    write_text(run_root / "lineage" / "s0001.bin", "tampered lineage bytes\n")
    with pytest.raises(ValueError, match="manifest"):
        store.verify_manifest(run_root)

    for name, mutate in (
        ("missing", lambda entries: entries[:-1]),
        ("empty", lambda entries: []),
    ):
        other = _finalized_artifact_store(tmp_path, name)
        manifest_path = tmp_path / name / "manifest.json"
        manifest = read_json(manifest_path)
        manifest["artifacts"] = mutate(manifest["artifacts"])
        write_json(manifest_path, manifest)
        with pytest.raises(ValueError, match="manifest"):
            other.verify_manifest(tmp_path / name)


@pytest.mark.parametrize(
    "mutation",
    (
        lambda entries: [*entries, dict(entries[0])],
        lambda entries: [{**entries[0], "path": "../outside.json"}, *entries[1:]],
        lambda entries: [{**entries[0], "path": "D:/outside.json"}, *entries[1:]],
        lambda entries: [{**entries[0], "sha256": "not-a-sha256"}, *entries[1:]],
    ),
)
def test_manifest_rejects_duplicate_unsafe_or_malformed_entries(tmp_path: Path, mutation: object) -> None:
    """Catch duplicate, traversal, absolute, or non-digest manifest declarations."""
    from xunce_artifact_io import read_json, write_json

    store = _finalized_artifact_store(tmp_path, "bad-manifest")
    manifest_path = tmp_path / "bad-manifest" / "manifest.json"
    manifest = read_json(manifest_path)
    manifest["artifacts"] = mutation(manifest["artifacts"])
    write_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="manifest"):
        store.verify_manifest(tmp_path / "bad-manifest")


def test_blocked_preflight_clears_every_pass_signal_from_both_terminal_payloads(tmp_path: Path) -> None:
    """Catch caller-provided pass fields surviving an environment-preflight block."""
    from xunce_artifact_io import read_json

    store = _artifact_store_type().create_new(tmp_path / "blocked-pass", _artifact_config())
    assert store.capture_environment(lambda: {})["status"] == "blocked"
    store.finalize(
        {"status": "passed", "pass": True, "midterm_reduced_passed": True},
        {"status": "passed", "pass": True, "midterm_reduced_passed": True, "final_threshold_reduced_passed": True},
        "blocked preflight",
        {},
    )
    for payload in (read_json(tmp_path / "blocked-pass" / "summary.json"), read_json(tmp_path / "blocked-pass" / "routing.json")):
        assert payload["status"] == "blocked"
        assert payload["formal_evidence_eligible"] is False
        assert payload["midterm_reduced_passed"] is False
        assert payload["final_threshold_reduced_passed"] is False
        assert all(value is not True for key, value in payload.items() if "pass" in key.lower())


def test_nonblocked_finalize_requires_the_exact_nonempty_required_phase_sequence(tmp_path: Path) -> None:
    """Catch zero, gapped, or truncated phase evidence being finalized as eligible."""
    store_type = _artifact_store_type()
    zero = store_type.create_new(tmp_path / "zero", _artifact_config(("p01",)))
    _capture_valid_preflight(zero)
    with pytest.raises(ValueError, match="required phase"):
        zero.finalize({"status": "complete"}, {"status": "passed"}, "report", {})

    tail = store_type.create_new(tmp_path / "tail", _artifact_config(("p01", "p02")))
    attempt = tail.write_phase_attempt("p01", _artifact_rows("p01"), {})
    tail.accept_phase("p01", attempt, tail.phase_attempt_row_sha256("p01", attempt))
    _capture_valid_preflight(tail)
    with pytest.raises(ValueError, match="required phase"):
        tail.finalize({"status": "complete"}, {"status": "passed"}, "report", {})

    with pytest.raises(ValueError, match="contiguous"):
        store_type.create_new(tmp_path / "gap", _artifact_config(("p01", "p03")))


def test_lineage_audit_snapshots_a_tracked_rename_at_its_new_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Catch rename porcelain parsing that mislabels the required new path as clean."""
    repository = tmp_path / "rename-repository"
    repository.mkdir()
    subprocess.run(("git", "init"), cwd=repository, check=True, capture_output=True)
    subprocess.run(("git", "config", "user.email", "test@example.invalid"), cwd=repository, check=True)
    subprocess.run(("git", "config", "user.name", "Task 2 Test"), cwd=repository, check=True)
    original = repository / "original.py"
    original.write_bytes(b"tracked rename bytes\n")
    subprocess.run(("git", "add", "original.py"), cwd=repository, check=True)
    subprocess.run(("git", "commit", "-m", "tracked fixture"), cwd=repository, check=True, capture_output=True)
    renamed = repository / "renamed.py"
    subprocess.run(("git", "mv", "original.py", "renamed.py"), cwd=repository, check=True)
    monkeypatch.chdir(repository)
    store = _artifact_store_type().create_new(repository / "out", _artifact_config())
    row = store.capture_lineage([renamed], "a" * 40, "b" * 40)["required_sources"][0]
    from xunce_artifact_io import read_bytes

    assert row["status"].startswith("R")
    assert read_bytes(repository / "out" / row["snapshot_path"]) == b"tracked rename bytes\n"


@pytest.mark.parametrize(
    "gpu",
    (
        {},
        {"model": "", "driver": "driver", "cuda": "cuda"},
        {"model": "model", "driver": None, "cuda": "cuda"},
        {"model": "model", "driver": "driver", "cuda": []},
        "not-a-mapping",
    ),
)
def test_environment_audit_blocks_missing_or_malformed_gpu_subfields(tmp_path: Path, gpu: object) -> None:
    """Catch captured environment evidence without auditable GPU model, driver, and CUDA fields."""
    probe = _environment_probe()
    probe["gpu"] = gpu
    store = _artifact_store_type().create_new(tmp_path / "bad-gpu", _artifact_config())
    audit = store.capture_environment(lambda: probe)
    assert audit["status"] == "blocked"
    assert audit["formal_evidence_eligible"] is False


def test_eligible_finalize_without_captured_environment_forces_blocked_terminal_evidence(tmp_path: Path) -> None:
    """Catch a pass-capable finalization that never captured machine preflight evidence."""
    from xunce_artifact_io import read_json

    store = _artifact_store_type().create_new(tmp_path / "missing-environment", _artifact_config())
    attempt = store.write_phase_attempt("p01", _artifact_rows("p01"), {})
    store.accept_phase("p01", attempt, store.phase_attempt_row_sha256("p01", attempt))
    store.finalize({"status": "complete"}, {"status": "passed"}, "report", {})
    assert read_json(tmp_path / "missing-environment" / "summary.json")["status"] == "blocked"
    assert read_json(tmp_path / "missing-environment" / "routing.json")["formal_evidence_eligible"] is False


def test_blocked_environment_preflight_persists_across_resume_and_cannot_become_eligible(tmp_path: Path) -> None:
    """Catch restart clearing a persisted environment block before terminal evidence is written."""
    from xunce_artifact_io import read_json

    store_type = _artifact_store_type()
    store = store_type.create_new(tmp_path / "resumed-block", _artifact_config())
    source = tmp_path / "lineage-source.py"
    source.write_bytes(b"lineage source\n")
    store.capture_lineage([source], "a" * 40, "b" * 40)
    assert store.capture_environment(lambda: {})["status"] == "blocked"
    resumed = store_type.load_for_resume(tmp_path / "resumed-block", store.config_sha256)
    resumed.finalize({"status": "passed"}, {"status": "passed"}, "report", {})
    assert read_json(tmp_path / "resumed-block" / "summary.json")["status"] == "blocked"
    assert read_json(tmp_path / "resumed-block" / "manifest.json")["formal_evidence_eligible"] is False


@pytest.mark.parametrize(
    "field_name,value",
    (
        ("schema_version", "mid-dual-manifest/v0"),
        ("config_sha256", "b" * 64),
        ("formal_evidence_eligible", False),
    ),
)
def test_manifest_rejects_header_only_schema_config_or_eligibility_drift(tmp_path: Path, field_name: str, value: object) -> None:
    """Catch an otherwise intact manifest whose semantic header no longer matches the run."""
    from xunce_artifact_io import read_json, write_json

    store = _finalized_artifact_store(tmp_path, f"header-{field_name}")
    manifest_path = tmp_path / f"header-{field_name}" / "manifest.json"
    manifest = read_json(manifest_path)
    manifest[field_name] = value
    write_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="manifest"):
        store.verify_manifest(tmp_path / f"header-{field_name}")


@pytest.mark.parametrize("rewritten_rows_path", ("phases/p01/a02/results.jsonl", "../phases/p01/a02/results.jsonl"))
def test_resume_rejects_phase_state_rows_path_not_bound_to_accepted_attempt(tmp_path: Path, rewritten_rows_path: str) -> None:
    """Catch phase state pointing at an identical but unaccepted attempt or an unsafe path."""
    from xunce_artifact_io import read_jsonl, write_jsonl

    store_type = _artifact_store_type()
    store = store_type.create_new(tmp_path / "phase-state-path", _artifact_config())
    accepted_attempt = store.write_phase_attempt("p01", _artifact_rows("p01"), {})
    store.accept_phase("p01", accepted_attempt, store.phase_attempt_row_sha256("p01", accepted_attempt))
    store.write_phase_attempt("p01", _artifact_rows("p01"), {})
    state_path = tmp_path / "phase-state-path" / "phase-state.jsonl"
    state = read_jsonl(state_path)
    state[0]["rows_path"] = rewritten_rows_path
    write_jsonl(state_path, state)
    with pytest.raises(ValueError, match="phase-state"):
        store_type.load_for_resume(tmp_path / "phase-state-path", store.config_sha256)


@pytest.mark.parametrize(
    "name",
    ("environment", "lineage", "store_index", "config", "results", "summary", "routing", "manifest", "phase_state", "report"),
)
def test_finalize_rejects_reserved_extra_audit_names_before_writing_final_artifacts(tmp_path: Path, name: str) -> None:
    """Catch extra-audit names that could overwrite canonical or authority artifacts."""
    from xunce_artifact_io import path_is_file

    store = _artifact_store_type().create_new(tmp_path / "reserved-extra", _artifact_config())
    attempt = store.write_phase_attempt("p01", _artifact_rows("p01"), {})
    store.accept_phase("p01", attempt, store.phase_attempt_row_sha256("p01", attempt))
    _capture_valid_preflight(store)
    with pytest.raises(ValueError, match="reserved"):
        store.finalize({"status": "complete"}, {"status": "passed"}, "report", {name: {"kind": "test"}})
    assert not any(path_is_file(tmp_path / "reserved-extra" / filename) for filename in ("results.jsonl", "summary.json", "routing.json", "report.md", "manifest.json"))


@pytest.mark.parametrize("mutation", ("remove_row", "snapshot_path", "sha256", "size_bytes", "snapshot_bytes", "delete_snapshot"))
def test_resume_and_finalize_block_tampered_required_lineage_evidence(tmp_path: Path, mutation: str) -> None:
    """Catch a required dirty source losing its exact durable snapshot binding."""
    from xunce_artifact_io import path_is_file, read_json, write_json, write_text

    source = tmp_path / "required-dirty.py"
    source.write_bytes(b"durable source evidence\n")
    run_root = tmp_path / f"lineage-{mutation}"
    store = _artifact_store_type().create_new(run_root, _artifact_config())
    attempt = store.write_phase_attempt("p01", _artifact_rows("p01"), {})
    store.accept_phase("p01", attempt, store.phase_attempt_row_sha256("p01", attempt))
    row = store.capture_lineage([source], "a" * 40, "b" * 40)["required_sources"][0]
    assert store.capture_environment(_environment_probe)["status"] == "captured"
    audit_path = run_root / "lineage_audit.json"
    audit = read_json(audit_path)
    if mutation == "remove_row":
        audit["required_sources"] = []
        write_json(audit_path, audit)
    elif mutation == "snapshot_path":
        audit["required_sources"][0]["snapshot_path"] = "lineage/s9999.bin"
        write_json(audit_path, audit)
    elif mutation == "sha256":
        audit["required_sources"][0]["sha256"] = "c" * 64
        write_json(audit_path, audit)
    elif mutation == "size_bytes":
        audit["required_sources"][0]["size_bytes"] = 1
        write_json(audit_path, audit)
    elif mutation == "snapshot_bytes":
        write_text(run_root / row["snapshot_path"], "tampered\n")
    else:
        (run_root / row["snapshot_path"]).unlink()
    resumed = _artifact_store_type().load_for_resume(run_root, store.config_sha256)
    resumed.finalize({"status": "complete"}, {"status": "passed"}, "report", {})
    assert read_json(run_root / "summary.json")["status"] == "blocked"
    assert path_is_file(run_root / "manifest.json")
    assert read_json(run_root / "manifest.json")["formal_evidence_eligible"] is False


def test_lineage_snapshots_a_repository_local_ignored_required_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Catch ignored-but-untracked required sources being mistaken for clean tracked files."""
    from xunce_artifact_io import read_bytes

    repository = tmp_path / "ignored-repository"
    repository.mkdir()
    subprocess.run(("git", "init"), cwd=repository, check=True, capture_output=True)
    (repository / ".gitignore").write_text("required.ignored\n", encoding="utf-8")
    source = repository / "required.ignored"
    source.write_bytes(b"ignored source evidence\n")
    monkeypatch.chdir(repository)
    store = _artifact_store_type().create_new(repository / "out", _artifact_config())
    row = store.capture_lineage([source], "a" * 40, "b" * 40)["required_sources"][0]
    assert row["status"] == "untracked"
    assert row["snapshot_path"] == "lineage/s0001.bin"
    assert read_bytes(repository / "out" / row["snapshot_path"]) == b"ignored source evidence\n"


@pytest.mark.parametrize(
    "field_name,value",
    (
        ("windows_version", []),
        ("cpu_model", {}),
        ("cpu_logical_count", 0),
        ("cpu_logical_count", True),
        ("memory_bytes", 0),
        ("memory_bytes", False),
        ("gpu", "not-a-mapping"),
        ("python_executable", []),
        ("python_version", {}),
        ("frozen_dependencies", "not-a-list"),
        ("frozen_dependencies", []),
        ("frozen_dependencies", [""]),
        ("python_hash_seed", 0),
        ("thread_variables", []),
        ("thread_variables", {"OMP_NUM_THREADS": 1}),
        ("worker_start_method", ""),
        ("power_mode", None),
    ),
)
def test_environment_schema_invalid_probe_persists_blocked_across_resume(tmp_path: Path, field_name: str, value: object) -> None:
    """Catch malformed environment schema fields becoming eligible after a restart."""
    from xunce_artifact_io import read_json

    probe = _environment_probe()
    probe[field_name] = value
    run_root = tmp_path / f"environment-{field_name}-{type(value).__name__}"
    store = _artifact_store_type().create_new(run_root, _artifact_config())
    attempt = store.write_phase_attempt("p01", _artifact_rows("p01"), {})
    store.accept_phase("p01", attempt, store.phase_attempt_row_sha256("p01", attempt))
    store.capture_lineage([Path(__file__)], "a" * 40, "b" * 40)
    assert store.capture_environment(lambda: probe)["status"] == "blocked"
    resumed = _artifact_store_type().load_for_resume(run_root, store.config_sha256)
    resumed.finalize({"status": "complete"}, {"status": "passed"}, "report", {})
    assert read_json(run_root / "routing.json")["formal_evidence_eligible"] is False
