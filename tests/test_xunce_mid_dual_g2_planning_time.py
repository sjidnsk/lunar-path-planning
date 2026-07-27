from __future__ import annotations

from contextlib import redirect_stdout
import hashlib
import importlib.util
import io
import json
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
