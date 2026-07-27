from __future__ import annotations

from contextlib import redirect_stdout
import copy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import struct
import sys
import threading
import zipfile

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "run_xunce_mid_dual_g2_planning_time.py"
INPUTS_PATH = REPO_ROOT / "scripts" / "xunce_mid_dual_g2_inputs.py"
CONFIG_PATH = REPO_ROOT / "configs" / "xunce_mid_dual_g2_planning_time_v1.json"
SCALE_PROFILE = "midterm_reduced_w8x3_update80/v1"
PLATFORMS = ("wheel", "legged", "hopper")
SHA_A = hashlib.sha256(b"a").hexdigest()
SHA_B = hashlib.sha256(b"b").hexdigest()
SHA_C = hashlib.sha256(b"c").hexdigest()
HIGH_PERFORMANCE_GUID = "8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c"


def _load(path: Path, name: str):
    assert path.is_file(), f"implementation is missing: {path.name}"
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _module():
    return _load(SCRIPT_PATH, "run_xunce_mid_dual_g2_planning_time_tested")


def _inputs_module():
    return _load(INPUTS_PATH, "xunce_mid_dual_g2_inputs_task8_tested")


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _domain_hash(domain: str, *parts: bytes) -> str:
    payload = bytearray()
    for part in (domain.encode("utf-8"), *parts):
        payload.extend(struct.pack(">Q", len(part)))
        payload.extend(part)
    return hashlib.sha256(payload).hexdigest()


def _formal_environment_policy(lease_path: Path) -> dict[str, object]:
    return {
        "schema_version": (
            "xunce-mid-dual-g2-formal-environment-policy/v1"
        ),
        "lease_path": lease_path.as_posix(),
        "allowed_power_scheme_guids": [HIGH_PERFORMANCE_GUID],
        "required_thread_variables": {
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        },
        "sample_window_seconds": 2.0,
        "limits": {
            "cpu_percent_max": 20.0,
            "memory_percent_max": 85.0,
            "memory_available_bytes_min": 4_294_967_296,
            "disk_busy_percent_max": 20.0,
            "disk_free_bytes_min": 10_737_418_240,
        },
        "competing_process_patterns": [
            "run_xunce_mid_dual_g1_coverage.py",
            "run_xunce_mid_dual_g2_planning_time.py",
            "run_xunce_mid_dual_g3_closed_loop.py",
            "run_ppo_stage6_standard.py",
            "pytest",
            "training",
            "formal",
        ],
        "exclude_current_process_tree": True,
    }


def _formal_environment_snapshot(
    *,
    phase: str,
    pid: int,
) -> dict[str, object]:
    return {
        "schema_version": (
            "xunce-mid-dual-g2-formal-environment-observation/v1"
        ),
        "phase": phase,
        "captured_utc": "2026-07-27T08:00:00.000000Z",
        "host": "fixture-host",
        "pid": pid,
        "power_scheme_guid": HIGH_PERFORMANCE_GUID,
        "power_probe_sha256": SHA_A,
        "thread_variables": {
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        },
        "sample_window_seconds": 2.0,
        "cpu_percent": 10.0,
        "memory_percent": 50.0,
        "memory_available_bytes": 8_589_934_592,
        "disk_busy_percent": 10.0,
        "disk_free_bytes": 21_474_836_480,
        "disk_root": "D:/",
        "competing_processes": [],
        "excluded_process_ids": [pid],
        "process_inventory_sha256": SHA_B,
    }


def _request_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for platform in PLATFORMS:
        request_index = 0
        for scale, normal, hard, unreachable in (
            ("standard", 23, 7, 3),
            ("kilometer", 6, 2, 2),
        ):
            classes = (
                ["normal_reachable"] * normal
                + ["hard_reachable"] * hard
                + ["unreachable"] * unreachable
            )
            for request_class in classes:
                provider_sha = hashlib.sha256(
                    f"provider:{platform}:{request_index}".encode()
                ).hexdigest()
                truth_sha = hashlib.sha256(
                    f"truth:{platform}:{request_index}".encode()
                ).hexdigest()
                reachable = request_class != "unreachable"
                rows.append(
                    {
                        "platform": platform,
                        "scale": scale,
                        "request_index": request_index,
                        "request_id": (
                            f"g2i-req-{platform}-{scale}-{provider_sha[:20]}"
                        ),
                        "request_sha256": provider_sha,
                        "provider_request_sha256": provider_sha,
                        "truth_sha256": truth_sha,
                        "truth_request_sha256": truth_sha,
                        "truth_certificate_sha256": hashlib.sha256(
                            f"certificate:{platform}:{request_index}".encode()
                        ).hexdigest(),
                        "terrain_sha256": hashlib.sha256(
                            f"terrain:{platform}:{request_index}".encode()
                        ).hexdigest(),
                        "request_class": request_class,
                        "outcome_kind": (
                            "reachable" if reachable else "unreachable"
                        ),
                        "expected_success": reachable,
                        "expected_route_l2_valid": reachable,
                    }
                )
                request_index += 1
    return rows


def _formal_rows(
    module,
    *,
    elapsed_ns: int = 100_000_000,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    component = elapsed_ns // 5
    for request in _request_rows():
        semantic = module.expected_semantic_digest(
            request_sha256=request["request_sha256"],
            provider_success=request["expected_success"],
            route_l2_valid=request["expected_route_l2_valid"],
        )
        for repeat in range(5):
            rows.append(
                {
                    "row_kind": "g2_planning_call",
                    "schema_version": (
                        "xunce-mid-dual-g2-planning-call-row/v1"
                    ),
                    "scale_profile": SCALE_PROFILE,
                    "run_id": "g2-test-run",
                    "episode_id": request["request_id"],
                    "request_id": request["request_id"],
                    "call_id": (
                        f"g2-call-{request['request_sha256'][:24]}-r{repeat}"
                    ),
                    "platform": request["platform"],
                    "scale": request["scale"],
                    "request_class": request["request_class"],
                    "outcome_kind": request["outcome_kind"],
                    "source_sha256": SHA_A,
                    "config_sha256": SHA_B,
                    "input_sha256": SHA_C,
                    "code_sha256": SHA_A,
                    "request_sha256": request["request_sha256"],
                    "truth_sha256": request["truth_sha256"],
                    "provider_sha256": SHA_A,
                    "provider_source_bytes_sha256": SHA_B,
                    "oracle_sha256": SHA_C,
                    "oracle_source_bytes_sha256": hashlib.sha256(
                        b"oracle-source"
                    ).hexdigest(),
                    "provider_result_sha256": semantic,
                    "oracle_result_sha256": request[
                        "truth_certificate_sha256"
                    ],
                    "total_ns": elapsed_ns,
                    "input_validation_ns": component,
                    "platform_instantiation_ns": component,
                    "search_ns": component,
                    "complete_route_validation_ns": component,
                    "result_assembly_ns": elapsed_ns - component * 4,
                    "elapsed_ms": elapsed_ns / 1_000_000.0,
                    "input_validation_ms": component / 1_000_000.0,
                    "platform_instantiation_ms": component / 1_000_000.0,
                    "search_ms": component / 1_000_000.0,
                    "complete_route_validation_ms": (
                        component / 1_000_000.0
                    ),
                    "result_assembly_ms": (
                        (elapsed_ns - component * 4) / 1_000_000.0
                    ),
                    "provider_success": request["expected_success"],
                    "route_l2_valid": request[
                        "expected_route_l2_valid"
                    ],
                    "semantic_digest": semantic,
                    "timing_contract_id": "five-phase-sequential-ns/v1",
                    "formal_sample": True,
                    "repeat_index": repeat,
                }
            )
    return rows


def _npy(array: np.ndarray) -> bytes:
    stream = io.BytesIO()
    np.lib.format.write_array(
        stream,
        np.ascontiguousarray(array),
        version=(2, 0),
        allow_pickle=False,
    )
    return stream.getvalue()


def _terrain_blob() -> bytes:
    arrays = {
        "height_mm": np.zeros((5, 5), dtype="<i4"),
        "cell_class": np.zeros((5, 5), dtype="u1"),
        "known": np.ones((5, 5), dtype="u1"),
        "confidence_ppm": np.full(
            (5, 5),
            1_000_000,
            dtype="<u4",
        ),
    }
    metadata = {
        "geometry_schema_version": "g2-neutral-terrain/v2",
        "macro_cell_size_mm": 500,
        "provenance": {
            "physical_obstacle_cells_written": False,
            "source_kind": "procedural_simulation_proxy/v1",
        },
        "sub_20m_layer_source_kind": None,
    }
    output = io.BytesIO()
    with zipfile.ZipFile(
        output,
        "w",
        compression=zipfile.ZIP_STORED,
        allowZip64=False,
    ) as archive:
        for name in (
            "height_mm",
            "cell_class",
            "known",
            "confidence_ppm",
        ):
            info = zipfile.ZipInfo(
                f"{name}.npy",
                date_time=(1980, 1, 1, 0, 0, 0),
            )
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(info, _npy(arrays[name]))
        info = zipfile.ZipInfo(
            "metadata.json",
            date_time=(1980, 1, 1, 0, 0, 0),
        )
        info.compress_type = zipfile.ZIP_STORED
        info.create_system = 3
        info.external_attr = 0o100644 << 16
        archive.writestr(info, _canonical(metadata))
    return output.getvalue()


def _provider_envelope(platform: str, terrain: bytes) -> dict[str, object]:
    profile_id = {
        "wheel": "scout-mini-wheel-kinematic-sqp/v1",
        "legged": "legged-static-crawl-simulation-proxy-midterm/v1",
        "hopper": (
            "hopper-generic-internal-computational-simulation-proxy-"
            "midterm/v1"
        ),
    }[platform]
    inputs = _inputs_module()
    stack = inputs.PLATFORM_STACKS[platform]
    terrain_geometry_sha256 = inputs._decode_truth_terrain(terrain)[2]
    envelope = {
        "schema_version": "xunce-mid-dual-g2-provider-request/v1",
        "request_id": f"provider-smoke-{platform}",
        "platform": platform,
        "scale": "standard",
        "platform_stack": stack,
        "canonical_request_metadata": {
            "accelerator_policy": "disabled",
            "determinism_seed": 7,
            "goal_state": {
                "heading_rad": 0.0,
                "x_m": 2.25,
                "y_m": 1.25,
            },
            "objective_profile": {
                "distance_weight": 1.0,
                "energy_weight": 0.0,
                "risk_weight": 0.0,
                "time_weight": 0.0,
            },
            "platform_profile_id": profile_id,
            "request_id": f"provider-smoke-{platform}",
            "resource_budget": {
                "max_expanded_states": 8192,
                "max_memory_bytes": (
                    536_870_912 if platform == "hopper" else 33_554_432
                ),
                "max_route_states": 129,
            },
            "start_state": {
                "heading_rad": 0.0,
                "x_m": 0.25,
                "y_m": 1.25,
            },
            "timeout_s": 2.0,
        },
        "terrain_binding": {
            "geometry_schema_version": "g2-neutral-terrain/v2",
            "height": 5,
            "macro_cell_size_mm": 500,
            "physical_obstacle_cells_written": False,
            "sub_20m_layer_source_kind": None,
            "terrain_geometry_sha256": terrain_geometry_sha256,
            "terrain_sha256": hashlib.sha256(terrain).hexdigest(),
            "width": 5,
        },
    }
    return {
        **envelope,
        "provider_request_sha256": _domain_hash(
            "xunce-mid-dual-g2-provider-request/v1",
            _canonical(envelope),
            terrain,
        ),
    }


def test_formal_schedule_contains_exactly_645_unique_calls() -> None:
    module = _module()
    schedule = module.build_formal_schedule(
        "g2t2-candidate-" + "a" * 24,
        _request_rows(),
    )
    assert len(schedule["calls"]) == 645
    assert len({row["call_id"] for row in schedule["calls"]}) == 645
    assert {
        row["repeat_index"] for row in schedule["calls"]
    } == set(range(5))
    assert all(row["formal_sample"] is True for row in schedule["calls"])
    assert module.build_formal_schedule(
        "g2t2-candidate-" + "a" * 24,
        _request_rows(),
    )["schedule_sha256"] == schedule["schedule_sha256"]


def test_each_platform_scale_group_has_165_or_50_samples() -> None:
    module = _module()
    calls = module.build_formal_schedule(
        "g2t2-candidate-" + "a" * 24,
        _request_rows(),
    )["calls"]
    assert {
        f"{platform}/{scale}": sum(
            row["platform"] == platform and row["scale"] == scale
            for row in calls
        )
        for platform in PLATFORMS
        for scale in ("standard", "kilometer")
    } == {
        "wheel/standard": 165,
        "wheel/kilometer": 50,
        "legged/standard": 165,
        "legged/kilometer": 50,
        "hopper/standard": 165,
        "hopper/kilometer": 50,
    }
    for platform in PLATFORMS:
        for repeat in range(5):
            assert sum(
                row["platform"] == platform
                and row["repeat_index"] == repeat
                for row in calls
            ) == 43


def test_warmup_cold_start_and_worker_one_are_nonformal() -> None:
    module = _module()
    schedules = module.build_nonformal_schedules(_request_rows())
    assert len(schedules["cold_start"]) == 3
    assert len(schedules["warmup"]) == 30
    assert len(schedules["worker_one"]) == 129
    assert len(schedules["worker_four"]) == 129
    assert all(
        row["formal_sample"] is False
        for rows in schedules.values()
        for row in rows
    )
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    assert config["execution"]["formal_worker_count"] == 4
    assert config["execution"]["diagnostic_worker_count"] == 1
    assert config["execution"]["warmup_requests_per_platform"] == 10
    assert config["execution"]["cold_start_calls_per_platform"] == 1


def test_preload_occurs_before_worker_and_timer_starts_inside_worker() -> None:
    module = _module()
    main_thread = threading.get_ident()
    prepare_threads: list[int] = []
    execute_threads: list[int] = []

    def prepare(task):
        prepare_threads.append(threading.get_ident())
        return task

    def execute(task):
        execute_threads.append(threading.get_ident())
        return {"call_id": task["call_id"]}

    rows = module.execute_preloaded_batch(
        [{"call_id": f"call-{index}"} for index in range(8)],
        prepare_task=prepare,
        execute_task=execute,
        max_workers=4,
    )
    assert len(rows) == 8
    assert prepare_threads == [main_thread] * 8
    assert execute_threads
    assert all(thread_id != main_thread for thread_id in execute_threads)


def test_five_phase_ns_contract_and_total_are_exact() -> None:
    module = _module()
    timing = {
        "input_validation_ns": 2,
        "platform_instantiation_ns": 3,
        "search_ns": 5,
        "complete_route_validation_ns": 7,
        "result_assembly_ns": 11,
        "total_ns": 28,
    }
    assert module.validate_timing_payload(timing) == timing
    for field, value in (
        ("input_validation_ns", -1),
        ("search_ns", float("nan")),
        ("total_ns", 29),
    ):
        mutated = dict(timing)
        mutated[field] = value
        with pytest.raises(module.G2Blocked, match="g2_timing_contract"):
            module.validate_timing_payload(mutated)


def test_thresholds_are_recomputed_per_platform_scale_and_subgroup() -> None:
    module = _module()
    summary = module.recompute_g2_summary(_formal_rows(module))
    assert summary["formal_call_count"] == 645
    assert summary["g2_all_platforms_2s_passed"] is True
    assert summary["g2_all_platforms_1s_passed"] is True
    assert summary["timing_by_platform_scale"]["wheel/standard"][
        "sample_count"
    ] == 165
    assert summary["timing_by_platform_scale"]["wheel/standard"][
        "proportion_at_or_below_1000ms"
    ] == 1.0

    rows = _formal_rows(module)
    rows[0]["total_ns"] = 2_000_000_001
    rows[0]["search_ns"] = 1_920_000_001
    rows[0]["elapsed_ms"] = 2000.000001
    rows[0]["search_ms"] = 1920.000001
    failed = module.recompute_g2_summary(rows)
    assert failed["status"] == "failed"
    assert failed["timing_by_platform_scale"]["wheel/standard"][
        "over_2000_count"
    ] == 1
    assert failed["g2_all_platforms_2s_passed"] is False


def test_valid_evidence_with_wrong_provider_outcome_is_failed_not_blocked() -> None:
    module = _module()
    unreachable = _formal_rows(module)
    selected = next(
        row for row in unreachable if row["outcome_kind"] == "unreachable"
    )
    selected["provider_success"] = True
    selected["route_l2_valid"] = True
    selected["semantic_digest"] = module.expected_semantic_digest(
        request_sha256=selected["request_sha256"],
        provider_success=True,
        route_l2_valid=True,
    )
    selected["provider_result_sha256"] = selected["semantic_digest"]
    result = module.recompute_g2_summary(unreachable)
    assert result["status"] == "failed"
    assert result["correctness_passed"] is False

    reachable = _formal_rows(module)
    selected = next(
        row for row in reachable if row["outcome_kind"] == "reachable"
    )
    selected["provider_success"] = False
    selected["route_l2_valid"] = False
    selected["semantic_digest"] = module.expected_semantic_digest(
        request_sha256=selected["request_sha256"],
        provider_success=False,
        route_l2_valid=False,
    )
    selected["provider_result_sha256"] = selected["semantic_digest"]
    result = module.recompute_g2_summary(reachable)
    assert result["status"] == "failed"
    assert result["reachable_correctness"][
        "reachable_unique_success_count_by_platform"
    ]["wheel"] == 37


def test_duplicate_repeat_or_malformed_row_blocks_recomputation() -> None:
    module = _module()
    rows = _formal_rows(module)
    rows[1]["repeat_index"] = rows[0]["repeat_index"]
    with pytest.raises(module.G2Blocked, match="g2_repeat_matrix"):
        module.recompute_g2_summary(rows)

    rows = _formal_rows(module)
    rows[0]["unexpected"] = True
    with pytest.raises(module.G2Blocked, match="g2_row_schema"):
        module.recompute_g2_summary(rows)


def test_worker_one_semantics_must_match_all_five_formal_repeats() -> None:
    module = _module()
    formal = _formal_rows(module)
    diagnostic = []
    for request in _request_rows():
        selected = next(
            row
            for row in formal
            if row["request_id"] == request["request_id"]
        )
        diagnostic.append(
            {
                "request_id": selected["request_id"],
                "request_sha256": selected["request_sha256"],
                "semantic_digest": selected["semantic_digest"],
                "provider_success": selected["provider_success"],
                "route_l2_valid": selected["route_l2_valid"],
                "formal_sample": False,
            }
        )
    assert module.compare_worker_semantics(formal, diagnostic)["matched"] is True
    diagnostic[0]["semantic_digest"] = SHA_A
    with pytest.raises(module.G2Blocked, match="g2_worker_semantic_drift"):
        module.compare_worker_semantics(formal, diagnostic)


def test_preformal_worker_one_and_four_semantics_are_exactly_equivalent() -> None:
    module = _module()
    formal = _formal_rows(module)
    worker_one = module._diagnostic_projection(  # noqa: SLF001
        [row for row in formal if row["repeat_index"] == 0]
    )
    worker_four = [dict(row) for row in worker_one]

    audit = module.compare_diagnostic_worker_semantics(
        worker_one,
        worker_four,
    )

    assert audit == {
        "schema_version": (
            "xunce-mid-dual-g2-preformal-worker-equivalence/v1"
        ),
        "matched": True,
        "request_count": 129,
        "worker_one_count": 1,
        "worker_four_count": 4,
        "worker_one_results_sha256": module._canonical_sha256(  # noqa: SLF001
            worker_one
        ),
        "worker_four_results_sha256": module._canonical_sha256(  # noqa: SLF001
            worker_four
        ),
    }
    worker_four[0]["semantic_digest"] = SHA_A
    with pytest.raises(
        module.G2Blocked,
        match="g2_preformal_worker_semantic_drift",
    ):
        module.compare_diagnostic_worker_semantics(
            worker_one,
            worker_four,
        )


def test_preformal_static_cache_audit_executes_read_only_terrain_validation() -> None:
    module = _module()
    terrain = _terrain_blob()
    terrain_sha256 = hashlib.sha256(terrain).hexdigest()

    audit = module.execute_read_only_static_cache_audit(
        {terrain_sha256: terrain}
    )

    assert audit["status"] == "passed"
    assert audit["read_only"] is True
    assert audit["terrain_count"] == 1
    assert audit["entries"][0]["terrain_sha256"] == terrain_sha256
    assert audit["entries"][0]["payload_sha256_before"] == terrain_sha256
    assert audit["entries"][0]["payload_sha256_after"] == terrain_sha256
    with pytest.raises(
        module.G2Blocked,
        match="g2_static_cache_payload_hash",
    ):
        module.execute_read_only_static_cache_audit(
            {SHA_A: terrain}
        )


def test_preformal_diagnostic_gate_requires_all_four_hashed_artifacts() -> None:
    module = _module()
    formal = _formal_rows(module)
    worker_one = module._diagnostic_projection(  # noqa: SLF001
        [row for row in formal if row["repeat_index"] == 0]
    )
    worker_four = [dict(row) for row in worker_one]
    equivalence = module.compare_diagnostic_worker_semantics(
        worker_one,
        worker_four,
    )
    terrain = _terrain_blob()
    terrain_sha256 = hashlib.sha256(terrain).hexdigest()
    cache_audit = module.execute_read_only_static_cache_audit(
        {terrain_sha256: terrain}
    )
    phase_audit = {
        "status": "complete",
        "formal_sample": False,
        "worker_one_count": 1,
        "worker_four_count": 4,
        "request_count": 129,
        "worker_one_results": worker_one,
        "worker_one_results_sha256": module._canonical_sha256(  # noqa: SLF001
            worker_one
        ),
        "worker_four_results": worker_four,
        "worker_four_results_sha256": module._canonical_sha256(  # noqa: SLF001
            worker_four
        ),
        "worker_equivalence": equivalence,
        "worker_equivalence_sha256": module._canonical_sha256(  # noqa: SLF001
            equivalence
        ),
        "static_cache_audit": cache_audit,
        "static_cache_audit_sha256": module._canonical_sha256(  # noqa: SLF001
            cache_audit
        ),
    }

    validated = module.validate_preformal_diagnostic_evidence(
        phase_audit,
        {terrain_sha256: terrain},
    )

    assert validated["worker_equivalence"]["matched"] is True
    assert validated["static_cache_audit"]["status"] == "passed"
    for missing_field in (
        "worker_one_results",
        "worker_four_results",
        "worker_equivalence",
        "static_cache_audit",
    ):
        malformed = dict(phase_audit)
        malformed.pop(missing_field)
        with pytest.raises(
            module.G2Blocked,
            match="g2_preformal_diagnostic_evidence",
        ):
            module.validate_preformal_diagnostic_evidence(
                malformed,
                {terrain_sha256: terrain},
            )
    hash_drift = dict(phase_audit)
    hash_drift["worker_four_results_sha256"] = SHA_A
    with pytest.raises(
        module.G2Blocked,
        match="g2_preformal_diagnostic_evidence",
    ):
        module.validate_preformal_diagnostic_evidence(
            hash_drift,
            {terrain_sha256: terrain},
        )


def test_static_cache_rejects_route_or_goal_specific_answers() -> None:
    module = _module()
    accepted = module.validate_static_cache_payload(
        {
            "cache_contract": (
                "immutable-terrain-static-validation-only/v1"
            ),
            "terrain_sha256": SHA_A,
            "static_validation_sha256": SHA_B,
        }
    )
    assert accepted["terrain_sha256"] == SHA_A
    for forbidden in ("route", "request_id", "goal_state", "outcome"):
        payload = dict(accepted)
        payload[forbidden] = "forbidden-answer"
        with pytest.raises(module.G2Blocked, match="g2_route_answer_cache"):
            module.validate_static_cache_payload(payload)


def test_interrupted_batch_resumes_without_repeating_completed_calls(
    tmp_path: Path,
) -> None:
    module = _module()
    calls = module.build_formal_schedule(
        "g2t2-candidate-" + "a" * 24,
        _request_rows(),
    )["calls"][:7]
    state_path = tmp_path / "job-state.jsonl"
    attempts: dict[str, int] = {}

    def interrupting_worker(task):
        call_id = task["call_id"]
        attempts[call_id] = attempts.get(call_id, 0) + 1
        if sum(attempts.values()) == 4:
            raise KeyboardInterrupt
        return {"call_id": call_id, "value": call_id}

    with pytest.raises(module.G2Interrupted):
        module.execute_recoverable_batch(
            calls,
            execute_task=interrupting_worker,
            state_path=state_path,
            schedule_sha256=SHA_A,
            max_workers=1,
        )
    completed_before = {
        row["call_id"]
        for row in module.artifact_io.read_jsonl(state_path)
    }
    assert 0 < len(completed_before) < len(calls)

    def completing_worker(task):
        call_id = task["call_id"]
        attempts[call_id] = attempts.get(call_id, 0) + 1
        return {"call_id": call_id, "value": call_id}

    rows = module.execute_recoverable_batch(
        calls,
        execute_task=completing_worker,
        state_path=state_path,
        schedule_sha256=SHA_A,
        max_workers=1,
    )
    assert len(rows) == len(calls)
    assert all(attempts[call_id] == 1 for call_id in completed_before)


def test_approval_snapshot_hash_drift_blocks_before_execution() -> None:
    module = _module()
    approval = _canonical({"approved": True}) + b"\n"
    binding = {
        "approval_artifact_sha256": hashlib.sha256(approval).hexdigest(),
        "formal_evidence_eligible": True,
    }
    assert module.validate_approval_snapshot(binding, approval)[
        "formal_evidence_eligible"
    ] is True
    with pytest.raises(module.G2Blocked, match="g2_approval_drift"):
        module.validate_approval_snapshot(binding, approval + b" ")


def test_code_identity_and_effective_config_bind_runtime_source_and_30deg() -> None:
    module = _module()
    inputs = _inputs_module()
    closure = inputs.capture_path_planner_runtime_source_closure()

    code_sha256 = module._code_lineage_sha256(closure)  # noqa: SLF001
    effective = module._effective_config(  # noqa: SLF001
        run_id="g2-runtime-binding-test",
        input_sha256=SHA_A,
        code_sha256=code_sha256,
        runtime_source_closure_sha256=closure[
            "path_planner_runtime_source_closure_sha256"
        ],
        formal_environment_gate=_formal_environment_policy(
            Path(
                "D:/xunce/out/mid_dual/g2/"
                ".formal-exclusive-lease.json"
            )
        ),
    )

    assert effective["active_platforms"] == ["wheel", "legged", "hopper"]
    assert effective["platform_invariants"] == {
        platform: {"max_traversable_slope_deg": 30.0}
        for platform in PLATFORMS
    }
    assert effective["path_planner_runtime_source_closure_sha256"] == (
        closure["path_planner_runtime_source_closure_sha256"]
    )
    assert effective["evidence_binding"][
        "runtime_source_closure_audit_path"
    ] == "g2_runtime_source_closure_audit.json"
    assert effective["evidence_binding"][
        "platform_invariants_audit_path"
    ] == "g2_platform_invariants_audit.json"
    assert effective["formal_environment_gate"][
        "allowed_power_scheme_guids"
    ] == [HIGH_PERFORMANCE_GUID]
    assert effective["evidence_binding"][
        "formal_environment_audit_path"
    ] == "g2_formal_environment_audit.json"
    changed_sources = [
        dict(row) for row in closure["required_sources"]
    ]
    changed_sources[0] = {
        **changed_sources[0],
        "size_bytes": int(changed_sources[0]["size_bytes"]) + 1,
        "sha256": SHA_A,
    }
    dirty_closure = inputs.build_path_planner_runtime_source_closure(
        submodule_commit=closure["submodule_commit"],
        dirty_inventory=[
            {
                "path": changed_sources[0]["logical_path"],
                "status": " M",
            }
        ],
        required_sources=changed_sources,
    )
    assert module._code_lineage_sha256(  # noqa: SLF001
        dirty_closure
    ) != code_sha256
    with pytest.raises(module.G2Blocked, match="g2_platform_invariant"):
        module.validate_platform_invariants(
            ["wheel", "legged", "hopper"],
            {
                **effective["platform_invariants"],
                "hopper": {"max_traversable_slope_deg": 30.1},
            },
        )


def test_task7_adapter_builds_exact_three_provider_stacks() -> None:
    inputs = _inputs_module()
    terrain = _terrain_blob()
    for platform, provider_name in (
        ("wheel", "WheelKinematicSQPProviderV2"),
        ("legged", "LeggedPrimitiveProviderV2"),
        ("hopper", "HopperPrimitiveProviderV2"),
    ):
        request = inputs.decode_provider_execution_request(
            _provider_envelope(platform, terrain),
            terrain,
        )
        stack = inputs.build_approved_platform_execution_stack(
            platform,
            request,
        )
        assert type(stack["provider"]).__name__ == provider_name
        assert stack["anchor"].snapshot is request.terrain_snapshot
        assert stack["profile"].profile_id == request.platform_profile_id


def test_missing_input_cli_is_blocked_with_zero_rows_and_never_calls_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    monkeypatch.setattr(module, "G2_OUTPUT_BASE", str(tmp_path / "g2"))

    def forbidden(*_args, **_kwargs):
        raise AssertionError("provider must not run after blocked preflight")

    monkeypatch.setattr(module, "execute_provider_timed_call", forbidden)
    output = io.StringIO()
    with redirect_stdout(output):
        code = module.main(
            [
                "--config",
                str(CONFIG_PATH),
                "--input-bundle",
                "D:/xunce/inputs/mid_dual/g2/missing",
                "--run-id",
                "task8-missing-input",
                "--mode",
                "preflight",
            ]
        )
    payload = json.loads(output.getvalue())
    assert code == 0
    assert payload["execution_status"] == "complete"
    assert payload["gate_status"] == "blocked"
    assert payload["formal_row_count"] == 0
    assert payload["formal_environment_gate_status"] == "not_run"

    source = SCRIPT_PATH.read_text(encoding="utf-8")
    for forbidden_text in (
        ".read_text(",
        ".write_text(",
        ".read_bytes(",
        ".write_bytes(",
        ".open(",
        ".mkdir(",
        ".exists(",
        ".is_file(",
        "time.sleep",
    ):
        assert forbidden_text not in source


def test_formal_environment_policy_is_frozen_and_effective() -> None:
    module = _module()
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    policy = module.validate_formal_environment_policy(
        config["execution"]["formal_environment_gate"]
    )

    assert policy == _formal_environment_policy(
        Path(
            "D:/xunce/out/mid_dual/g2/"
            ".formal-exclusive-lease.json"
        )
    )


@pytest.mark.parametrize("mutation", ("power", "thread_missing"))
def test_load_config_rejects_formal_environment_policy_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    module = _module()
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    policy = config["execution"]["formal_environment_gate"]
    if mutation == "power":
        policy["allowed_power_scheme_guids"] = [
            "381b4222-f694-41f0-9685-ff5bb260df2e"
        ]
    else:
        del policy["required_thread_variables"]["OMP_NUM_THREADS"]
    drifted_path = tmp_path / "g2-config-drift.json"
    module.artifact_io.write_json(drifted_path, config)
    monkeypatch.setattr(module, "CANONICAL_CONFIG_PATH", drifted_path)

    with pytest.raises(module.G2Blocked, match="g2_config_invalid"):
        module._load_config(drifted_path)  # noqa: SLF001


def test_windows_formal_environment_capture_uses_mocked_system_probes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    policy = _formal_environment_policy(
        tmp_path / "formal-exclusive-lease.json"
    )
    calls: list[object] = []
    for name, value in policy["required_thread_variables"].items():
        monkeypatch.setenv(name, value)

    def power_probe():
        calls.append("power")
        return HIGH_PERFORMANCE_GUID, SHA_A

    def process_probe(patterns, current_pid):
        calls.append(("process", list(patterns), current_pid))
        return [], [current_pid], SHA_B

    def load_probe(window):
        calls.append(("load", window))
        return 12.5, 3.5

    def memory_probe():
        calls.append("memory")
        return 40.0, 8_589_934_592

    def disk_probe():
        calls.append("disk")
        return "D:/", 21_474_836_480

    monkeypatch.setattr(module, "_probe_active_power_scheme", power_probe)
    monkeypatch.setattr(module, "_probe_process_inventory", process_probe)
    monkeypatch.setattr(module, "_sample_cpu_and_disk_busy", load_probe)
    monkeypatch.setattr(module, "_probe_memory_status", memory_probe)
    monkeypatch.setattr(module, "_probe_disk_free", disk_probe)

    observed = module.capture_windows_formal_environment(
        "start",
        policy,
        platform_name="nt",
    )

    assert module.validate_g2_formal_environment(observed, policy) == observed
    assert observed["power_scheme_guid"] == HIGH_PERFORMANCE_GUID
    assert observed["cpu_percent"] == 12.5
    assert observed["disk_busy_percent"] == 3.5
    assert observed["memory_percent"] == 40.0
    assert observed["memory_available_bytes"] == 8_589_934_592
    assert observed["disk_free_bytes"] == 21_474_836_480
    assert calls == [
        "power",
        ("process", policy["competing_process_patterns"], os.getpid()),
        ("load", 2.0),
        "memory",
        "disk",
    ]


def test_non_windows_formal_environment_capture_is_blocked(
    tmp_path: Path,
) -> None:
    module = _module()
    policy = _formal_environment_policy(
        tmp_path / "formal-exclusive-lease.json"
    )

    with pytest.raises(
        module.G2Blocked,
        match="g2_formal_windows_required",
    ):
        module.capture_windows_formal_environment(
            "start",
            policy,
            platform_name="posix",
        )


def test_power_scheme_parser_requires_one_actual_guid() -> None:
    module = _module()
    output = (
        b"Power Scheme GUID: "
        + HIGH_PERFORMANCE_GUID.encode("ascii")
        + b"  (High performance)\r\n"
    )

    assert module._parse_active_power_scheme(output) == (  # noqa: SLF001
        HIGH_PERFORMANCE_GUID
    )
    with pytest.raises(
        module.G2Blocked,
        match="g2_formal_power_probe_failed",
    ):
        module._parse_active_power_scheme(b"not-recorded")  # noqa: SLF001


def test_process_inventory_excludes_only_current_lineage_and_descendants() -> None:
    module = _module()
    records = [
        {
            "pid": 0,
            "parent_pid": 0,
            "name": "System Idle Process",
            "command_line": "",
        },
        {
            "pid": 10,
            "parent_pid": 0,
            "name": "powershell.exe",
            "command_line": "launcher --mode formal",
        },
        {
            "pid": 20,
            "parent_pid": 10,
            "name": "python.exe",
            "command_line": "run_xunce_mid_dual_g2_planning_time.py",
        },
        {
            "pid": 30,
            "parent_pid": 20,
            "name": "powershell.exe",
            "command_line": "read-only process probe formal",
        },
        {
            "pid": 40,
            "parent_pid": 10,
            "name": "python.exe",
            "command_line": "python -m pytest tests",
        },
        {
            "pid": 50,
            "parent_pid": 1,
            "name": "python.exe",
            "command_line": "training worker",
        },
    ]
    patterns = ["formal", "pytest", "training"]

    competing, excluded = module._classify_process_inventory(  # noqa: SLF001
        records,
        patterns=patterns,
        current_pid=20,
    )

    assert excluded == [10, 20, 30]
    assert [row["pid"] for row in competing] == [40, 50]
    assert [row["matched_pattern"] for row in competing] == [
        "pytest",
        "training",
    ]
    assert all(len(row["command_sha256"]) == 64 for row in competing)


def test_formal_lease_is_atomic_and_never_deletes_foreign_owner(
    tmp_path: Path,
) -> None:
    module = _module()
    lease_path = tmp_path / "formal-exclusive-lease.json"
    first = module.G2FormalLease(
        lease_path=lease_path,
        run_id="g2-lease-first",
        run_root=tmp_path / "run-first",
    )
    second = module.G2FormalLease(
        lease_path=lease_path,
        run_id="g2-lease-second",
        run_root=tmp_path / "run-second",
    )

    first_payload = first.acquire()
    with pytest.raises(module.G2Blocked, match="g2_formal_lease_contended"):
        second.acquire()
    assert module.artifact_io.read_json(lease_path) == first_payload

    foreign = {
        **first_payload,
        "run_id": "foreign-owner",
        "nonce": "f" * 32,
        "lease_sha256": SHA_A,
    }
    module.artifact_io.write_json(lease_path, foreign)
    assert first.release() is False
    assert module.artifact_io.read_json(lease_path) == foreign
    lease_path.unlink()


def test_stale_formal_lease_is_not_implicitly_recovered(
    tmp_path: Path,
) -> None:
    module = _module()
    lease_path = tmp_path / "formal-exclusive-lease.json"
    stale = {
        "schema_version": "xunce-mid-dual-g2-formal-lease/v1",
        "host": "old-host",
        "pid": 999_999,
        "process_start_utc": "2026-01-01T00:00:00.000000Z",
        "run_id": "old-run",
        "run_root": "D:/xunce/out/mid_dual/g2/old-run",
        "nonce": "a" * 32,
        "acquired_utc": "2026-01-01T00:00:01.000000Z",
        "lease_sha256": SHA_A,
    }
    module.artifact_io.write_json(lease_path, stale)
    before = module.artifact_io.read_bytes(lease_path)
    lease = module.G2FormalLease(
        lease_path=lease_path,
        run_id="new-run",
        run_root=tmp_path / "new-run",
    )

    with pytest.raises(module.G2Blocked, match="g2_formal_lease_contended"):
        lease.acquire()

    assert module.artifact_io.read_bytes(lease_path) == before


@pytest.mark.parametrize(
    ("mutation", "reason"),
    (
        ("not_recorded", "g2_formal_environment_not_recorded"),
        ("power", "g2_formal_power_scheme_not_allowed"),
        ("thread", "g2_formal_thread_settings_invalid"),
        ("process", "g2_formal_competing_process"),
        ("cpu", "g2_formal_cpu_load_exceeded"),
        ("memory_percent", "g2_formal_memory_load_exceeded"),
        ("memory_available", "g2_formal_memory_available_below_min"),
        ("disk_busy", "g2_formal_disk_busy_exceeded"),
        ("disk_free", "g2_formal_disk_free_below_min"),
    ),
)
def test_formal_environment_validation_fails_closed(
    tmp_path: Path,
    mutation: str,
    reason: str,
) -> None:
    module = _module()
    policy = _formal_environment_policy(
        tmp_path / "formal-exclusive-lease.json"
    )
    snapshot = _formal_environment_snapshot(
        phase="start",
        pid=os.getpid(),
    )
    if mutation == "not_recorded":
        snapshot["power_scheme_guid"] = "not-recorded"
    elif mutation == "power":
        snapshot["power_scheme_guid"] = (
            "381b4222-f694-41f0-9685-ff5bb260df2e"
        )
    elif mutation == "thread":
        snapshot["thread_variables"]["OMP_NUM_THREADS"] = "2"
    elif mutation == "process":
        snapshot["competing_processes"] = [
            {
                "pid": 4321,
                "parent_pid": 1,
                "name": "python.exe",
                "matched_pattern": "pytest",
                "command_sha256": SHA_C,
            }
        ]
    elif mutation == "cpu":
        snapshot["cpu_percent"] = 20.1
    elif mutation == "memory_percent":
        snapshot["memory_percent"] = 85.1
    elif mutation == "memory_available":
        snapshot["memory_available_bytes"] = 4_294_967_295
    elif mutation == "disk_busy":
        snapshot["disk_busy_percent"] = 20.1
    else:
        snapshot["disk_free_bytes"] = 10_737_418_239

    with pytest.raises(module.G2Blocked, match=reason):
        module.validate_g2_formal_environment(snapshot, policy)


def test_formal_guard_blocks_before_provider_and_releases_lease(
    tmp_path: Path,
) -> None:
    module = _module()
    lease_path = tmp_path / "formal-exclusive-lease.json"
    policy = _formal_environment_policy(lease_path)
    lease = module.G2FormalLease(
        lease_path=lease_path,
        run_id="g2-start-blocked",
        run_root=tmp_path / "g2-start-blocked",
    )
    phases: list[str] = []
    provider_calls: list[str] = []

    def capture(phase: str, _policy: Mapping[str, object]):
        phases.append(phase)
        snapshot = _formal_environment_snapshot(
            phase=phase,
            pid=os.getpid(),
        )
        snapshot["host"] = lease.payload["host"]
        if phase == "start":
            snapshot["power_scheme_guid"] = "not-recorded"
        return snapshot

    guard = module.G2FormalEnvironmentGuard(
        run_id="g2-start-blocked",
        policy=policy,
        lease=lease,
        capture_environment=capture,
    )
    with pytest.raises(
        module.G2Blocked,
        match="g2_formal_environment_not_recorded",
    ):
        with guard:
            provider_calls.append("provider-called")

    assert provider_calls == []
    assert phases == ["start", "end"]
    assert not module.artifact_io.path_exists(lease_path)
    assert guard.audit["status"] == "blocked"
    assert guard.audit["formal_evidence_eligible"] is False
    assert guard.audit["lease_released"] is True
    assert guard.audit["start_observation"]["power_scheme_guid"] == (
        "not-recorded"
    )


def test_formal_guard_end_failure_and_body_exception_release(
    tmp_path: Path,
) -> None:
    module = _module()
    policy = _formal_environment_policy(
        tmp_path / "formal-exclusive-lease.json"
    )

    def run(*, fail_end: bool, fail_body: bool):
        lease = module.G2FormalLease(
            lease_path=Path(policy["lease_path"]),
            run_id=f"g2-end-{fail_end}-{fail_body}",
            run_root=tmp_path / f"run-{fail_end}-{fail_body}",
        )

        def capture(phase: str, _policy: Mapping[str, object]):
            snapshot = _formal_environment_snapshot(
                phase=phase,
                pid=os.getpid(),
            )
            snapshot["host"] = lease.payload["host"]
            if phase == "end" and fail_end:
                snapshot["disk_busy_percent"] = 20.1
            return snapshot

        guard = module.G2FormalEnvironmentGuard(
            run_id=f"g2-end-{fail_end}-{fail_body}",
            policy=policy,
            lease=lease,
            capture_environment=capture,
        )
        if fail_body:
            with pytest.raises(RuntimeError, match="body failed"):
                with guard:
                    raise RuntimeError("body failed")
        else:
            with pytest.raises(
                module.G2Blocked,
                match="g2_formal_disk_busy_exceeded",
            ):
                with guard:
                    guard.set_formal_row_count(645)
        assert not module.artifact_io.path_exists(policy["lease_path"])
        assert guard.audit["lease_released"] is True
        assert guard.audit["end_observation"]["phase"] == "end"
        return guard.audit

    end_failed = run(fail_end=True, fail_body=False)
    body_failed = run(fail_end=False, fail_body=True)
    assert end_failed["formal_evidence_eligible"] is False
    assert body_failed["formal_evidence_eligible"] is False


def test_formal_guard_requires_exactly_645_rows_for_pass(
    tmp_path: Path,
) -> None:
    module = _module()
    policy = _formal_environment_policy(
        tmp_path / "formal-exclusive-lease.json"
    )
    lease = module.G2FormalLease(
        lease_path=Path(policy["lease_path"]),
        run_id="g2-row-count",
        run_root=tmp_path / "g2-row-count",
    )

    def capture(phase: str, _policy):
        snapshot = _formal_environment_snapshot(
            phase=phase,
            pid=os.getpid(),
        )
        snapshot["host"] = lease.payload["host"]
        return snapshot

    guard = module.G2FormalEnvironmentGuard(
        run_id="g2-row-count",
        policy=policy,
        lease=lease,
        capture_environment=capture,
    )
    with pytest.raises(
        module.G2Blocked,
        match="g2_formal_row_count_invalid",
    ):
        with guard:
            guard.set_formal_row_count(644)

    assert not module.artifact_io.path_exists(policy["lease_path"])
    assert guard.audit["status"] == "blocked"
    assert guard.audit["formal_row_count"] == 0
    assert guard.audit["formal_evidence_eligible"] is False


def test_formal_guard_supports_serial_resume_and_binds_audit(
    tmp_path: Path,
) -> None:
    module = _module()
    lease_path = tmp_path / "formal-exclusive-lease.json"
    policy = _formal_environment_policy(lease_path)
    audits: list[dict[str, object]] = []
    for resume_index in range(2):
        lease = module.G2FormalLease(
            lease_path=lease_path,
            run_id="g2-resume",
            run_root=tmp_path / "g2-resume",
        )

        def capture(phase: str, _policy: Mapping[str, object]):
            snapshot = _formal_environment_snapshot(
                phase=phase,
                pid=os.getpid(),
            )
            snapshot["host"] = lease.payload["host"]
            snapshot["captured_utc"] = (
                f"2026-07-27T08:00:0{resume_index}.000000Z"
            )
            return snapshot

        guard = module.G2FormalEnvironmentGuard(
            run_id="g2-resume",
            policy=policy,
            lease=lease,
            capture_environment=capture,
        )
        with guard:
            guard.set_formal_row_count(645)
        audits.append(guard.audit)
        assert not module.artifact_io.path_exists(lease_path)

    assert audits[0]["lease_sha256"] != audits[1]["lease_sha256"]
    for audit in audits:
        binding = module.formal_environment_audit_binding(audit)
        assert binding == {
            "formal_environment_audit_schema_version": (
                "xunce-mid-dual-g2-formal-environment-audit/v1"
            ),
            "formal_environment_audit_sha256": hashlib.sha256(
                _canonical(audit)
            ).hexdigest(),
            "formal_lease_sha256": audit["lease_sha256"],
        }


def test_p04_crash_window_preserves_embedded_environment_audit_on_resume(
    tmp_path: Path,
) -> None:
    module = _module()
    lease_path = tmp_path / "formal-exclusive-lease.json"
    policy = _formal_environment_policy(lease_path)

    def run_guard(captured_second: int):
        lease = module.G2FormalLease(
            lease_path=lease_path,
            run_id="g2-crash-window",
            run_root=tmp_path / "g2-crash-window",
        )

        def capture(phase: str, _policy):
            snapshot = _formal_environment_snapshot(
                phase=phase,
                pid=os.getpid(),
            )
            snapshot["host"] = lease.payload["host"]
            snapshot["captured_utc"] = (
                f"2026-07-27T08:00:{captured_second:02d}.000000Z"
            )
            return snapshot

        guard = module.G2FormalEnvironmentGuard(
            run_id="g2-crash-window",
            policy=policy,
            lease=lease,
            capture_environment=capture,
        )
        with guard:
            guard.set_formal_row_count(645)
        return guard.audit

    original_audit = run_guard(1)
    p04_audit = {
        "schema_version": "xunce-mid-dual-g2-phase-audit/v1",
        "phase_id": "p04",
        **module.formal_environment_phase_audit_fields(original_audit),
    }
    frozen_p04_bytes = _canonical(p04_audit)

    resumed_audit = run_guard(2)
    verified_original = module.validate_formal_environment_phase_audit(
        p04_audit
    )

    assert verified_original == original_audit
    assert _canonical(p04_audit) == frozen_p04_bytes
    assert original_audit["lease_sha256"] != resumed_audit["lease_sha256"]
    assert p04_audit["formal_environment_audit_sha256"] != (
        module.formal_environment_audit_binding(resumed_audit)[
            "formal_environment_audit_sha256"
        ]
    )
    tampered = copy.deepcopy(p04_audit)
    tampered["formal_environment_audit"][
        "start_observation"
    ]["cpu_percent"] = 10.5
    with pytest.raises(
        module.G2Blocked,
        match="g2_formal_environment_audit_invalid",
    ):
        module.validate_formal_environment_phase_audit(tampered)


def test_formal_report_records_actual_power_and_required_threads(
    tmp_path: Path,
) -> None:
    module = _module()
    lease_path = tmp_path / "formal-exclusive-lease.json"
    policy = _formal_environment_policy(lease_path)
    lease = module.G2FormalLease(
        lease_path=lease_path,
        run_id="g2-report-environment",
        run_root=tmp_path / "g2-report-environment",
    )

    def capture(phase: str, _policy):
        snapshot = _formal_environment_snapshot(
            phase=phase,
            pid=os.getpid(),
        )
        snapshot["host"] = lease.payload["host"]
        return snapshot

    guard = module.G2FormalEnvironmentGuard(
        run_id="g2-report-environment",
        policy=policy,
        lease=lease,
        capture_environment=capture,
    )
    with guard:
        guard.set_formal_row_count(645)

    report = module._render_report(  # noqa: SLF001
        {
            "run_id": "g2-report-environment",
            "status": "passed",
            "recomputed": {
                "formal_call_count": 645,
                "g2_all_platforms_2s_passed": True,
                "g2_all_platforms_1s_passed": True,
                "correctness_passed": True,
            },
        },
        formal_environment_audit=guard.audit,
    )

    assert HIGH_PERFORMANCE_GUID in report
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        assert f"{name}=1" in report
    assert "起始 CPU：12.500%" not in report
    assert "起始 CPU：10.000%" in report
    assert "结束磁盘忙碌率：10.000%" in report


@pytest.mark.parametrize("resume_from_p04", (False, True))
def test_run_g2_formal_path_guards_all_provider_work_and_final_audit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    resume_from_p04: bool,
) -> None:
    module = _module()
    inputs = __import__("xunce_mid_dual_g2_inputs")
    run_id = "g2-r7-resume" if resume_from_p04 else "g2-r7-new"
    run_root = tmp_path / "g2" / run_id
    lease_path = tmp_path / "g2" / "formal-exclusive-lease.json"
    config = copy.deepcopy(
        json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    )
    config["execution"]["formal_environment_gate"] = (
        _formal_environment_policy(lease_path)
    )
    events: list[str] = []
    phase_rows: dict[str, list[dict[str, object]]] = {}
    diagnostic_row = {
        "request_id": "diagnostic-request",
        "request_sha256": SHA_A,
        "semantic_digest": SHA_B,
        "provider_success": True,
        "route_l2_valid": True,
    }
    formal_rows = [{"formal_row": index} for index in range(645)]
    seeded_audits: dict[str, dict[str, object]] = {
        "p03": {
            "worker_one_results": [diagnostic_row],
        },
        "p04": {
            "phase_id": "p04",
            "status": "complete",
            "formal_sample": True,
        },
    }
    prior_p04_environment_audit: dict[str, object] | None = None
    if resume_from_p04:
        prior_lease = module.G2FormalLease(
            lease_path=lease_path,
            run_id=run_id,
            run_root=run_root,
        )

        def capture_prior(phase: str, _policy):
            snapshot = _formal_environment_snapshot(
                phase=phase,
                pid=os.getpid(),
            )
            snapshot["host"] = prior_lease.payload["host"]
            snapshot["captured_utc"] = (
                "2026-07-27T07:59:59.000000Z"
            )
            return snapshot

        prior_guard = module.G2FormalEnvironmentGuard(
            run_id=run_id,
            policy=config["execution"]["formal_environment_gate"],
            lease=prior_lease,
            capture_environment=capture_prior,
        )
        with prior_guard:
            prior_guard.set_formal_row_count(645)
        prior_p04_environment_audit = prior_guard.audit
        seeded_audits["p04"].update(
            module.formal_environment_phase_audit_fields(
                prior_p04_environment_audit
            )
        )

    class FakeStore:
        def __init__(self) -> None:
            self.run_root = run_root
            self.config_sha256 = SHA_C
            self.accepted_phase_ids = (
                ["p01", "p02", "p03", "p04"]
                if resume_from_p04
                else []
            )
            self.finalized: dict[str, object] | None = None

        def finalize(
            self,
            summary,
            routing,
            report,
            extra_audits,
        ) -> None:
            events.append("finalize")
            self.finalized = {
                "summary": summary,
                "routing": routing,
                "report": report,
                "extra_audits": extra_audits,
            }

    store = FakeStore()

    monkeypatch.setattr(
        module,
        "_load_config",
        lambda _path: (config, SHA_A),
    )
    monkeypatch.setattr(module, "G2_OUTPUT_BASE", str(tmp_path / "g2"))
    monkeypatch.setattr(
        inputs,
        "capture_path_planner_runtime_source_closure",
        lambda: {
            "path_planner_runtime_source_closure_sha256": SHA_A,
        },
    )
    monkeypatch.setattr(module, "_code_lineage_sha256", lambda _value: SHA_B)
    monkeypatch.setattr(
        module,
        "_read_execution_bundle",
        lambda *_args, **_kwargs: {
            "input_audit": {"status": "ready"},
            "input_sha256": SHA_A,
            "manifest_sha256": SHA_B,
            "manifest": {"input_set_id": "g2-r7-input"},
            "requests": [{"request_id": "fixture"}],
            "terrain_payloads": {},
        },
    )
    monkeypatch.setattr(
        module,
        "_open_store",
        lambda _root, _effective: (store, not resume_from_p04),
    )

    def capture_preflight(fake_store, **_kwargs):
        if "p01" not in fake_store.accepted_phase_ids:
            fake_store.accepted_phase_ids.append("p01")

    monkeypatch.setattr(module, "_capture_preflight", capture_preflight)
    monkeypatch.setattr(
        module,
        "build_nonformal_schedules",
        lambda _requests: {
            "cold_start": [{"kind": "cold"}],
            "warmup": [{"kind": "warmup"}],
            "worker_one": [{"kind": "worker-one"}],
            "worker_four": [{"kind": "worker-four"}],
        },
    )
    monkeypatch.setattr(
        module,
        "_hydrate_calls",
        lambda calls, **_kwargs: list(calls),
    )

    def execute_preloaded(calls, **_kwargs):
        assert module.artifact_io.path_is_file(lease_path)
        events.append(f"provider:{calls[0]['kind']}")
        return [diagnostic_row]

    monkeypatch.setattr(
        module,
        "execute_preloaded_batch",
        execute_preloaded,
    )
    monkeypatch.setattr(
        module,
        "compare_diagnostic_worker_semantics",
        lambda *_args: {"status": "passed"},
    )
    monkeypatch.setattr(
        module,
        "execute_read_only_static_cache_audit",
        lambda _payloads: {"status": "passed"},
    )

    def accept_phase(fake_store, *, phase_id, rows, audit):
        events.append(f"accept:{phase_id}")
        phase_rows[phase_id] = [dict(row) for row in rows]
        seeded_audits[phase_id] = {
            "phase_id": phase_id,
            **dict(audit),
        }
        fake_store.accepted_phase_ids.append(phase_id)

    monkeypatch.setattr(module, "_accept_phase", accept_phase)
    monkeypatch.setattr(
        module,
        "_accepted_phase_audit",
        lambda _store, phase_id: seeded_audits[phase_id],
    )
    monkeypatch.setattr(
        module,
        "validate_preformal_diagnostic_evidence",
        lambda audit, _payloads: audit,
    )
    monkeypatch.setattr(
        module,
        "build_formal_schedule",
        lambda *_args: {
            "schema_version": "g2-formal-schedule/v1",
            "input_set_id": "g2-r7-input",
            "repeat_count": 5,
            "formal_worker_count": 4,
            "calls": [{"kind": "formal"}] * 645,
            "schedule_sha256": SHA_C,
        },
    )

    def execute_formal(*_args, **_kwargs):
        assert module.artifact_io.path_is_file(lease_path)
        events.append("provider:formal")
        return formal_rows

    monkeypatch.setattr(
        module,
        "execute_recoverable_batch",
        execute_formal,
    )
    monkeypatch.setattr(
        module,
        "_accepted_phase_rows",
        lambda _store, phase_id: (
            formal_rows if phase_id == "p04" else phase_rows[phase_id]
        ),
    )
    monkeypatch.setattr(
        module,
        "recompute_g2_summary",
        lambda rows: {
            "status": "passed",
            "formal_call_count": len(rows),
        },
    )
    monkeypatch.setattr(
        module,
        "compare_worker_semantics",
        lambda *_args: {"status": "passed"},
    )
    monkeypatch.setattr(
        module,
        "_render_report",
        lambda _summary, **_kwargs: "report",
    )
    monkeypatch.setattr(
        module,
        "_source_report_audit",
        lambda **_kwargs: {"status": "passed"},
    )
    monkeypatch.setattr(
        module,
        "_routing",
        lambda _summary: {"status": "passed"},
    )
    monkeypatch.setattr(
        module.MidDualRunStore,
        "verify_manifest",
        staticmethod(lambda _root: True),
    )

    def capture_environment(phase, _policy):
        events.append(f"environment:{phase}")
        snapshot = _formal_environment_snapshot(
            phase=phase,
            pid=os.getpid(),
        )
        snapshot["host"] = module.host_platform.node() or "unknown-host"
        return snapshot

    monkeypatch.setattr(
        module,
        "capture_windows_formal_environment",
        capture_environment,
    )

    result = module.run_g2(
        config_path=CONFIG_PATH,
        input_bundle=tmp_path / "input",
        run_id=run_id,
        mode="formal",
    )

    assert result["formal_row_count"] == 645
    assert result["formal_environment_gate_status"] == "passed"
    assert not module.artifact_io.path_exists(lease_path)
    assert store.finalized is not None
    environment_audit = store.finalized["extra_audits"][
        "g2_formal_environment"
    ]
    assert environment_audit["status"] == "passed"
    assert environment_audit["formal_row_count"] == 645
    assert environment_audit["lease_released"] is True
    assert events.count("environment:start") == 1
    assert events.count("environment:end") == 1
    assert events.index("environment:start") < min(
        index
        for index, event in enumerate(events)
        if event.startswith("provider:")
    ) if not resume_from_p04 else events.index(
        "environment:start"
    ) < events.index(
        "environment:end"
    )
    assert events.index("environment:end") < events.index("finalize")
    if resume_from_p04:
        assert not any(event.startswith("provider:") for event in events)
        assert prior_p04_environment_audit is not None
        assert module.validate_formal_environment_phase_audit(
            seeded_audits["p04"]
        ) == prior_p04_environment_audit
        assert environment_audit["lease_sha256"] != (
            prior_p04_environment_audit["lease_sha256"]
        )
    else:
        assert events.index("environment:end") < events.index("accept:p04")
        assert module.validate_formal_environment_phase_audit(
            seeded_audits["p04"]
        ) == environment_audit
