"""Task 10 独立 aggregate：只从 manifest 与逐行结果复算双门槛。"""

from __future__ import annotations

import importlib
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import struct
import sys
from typing import Any, Callable, Mapping, Sequence

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import xunce_artifact_io as artifact_io  # noqa: E402
from xunce_mid_dual_artifacts import MidDualRunStore  # noqa: E402
from xunce_mid_dual_contracts import (  # noqa: E402
    G2_PLATFORMS,
    G2_REPEATS,
    SCALE_PROFILE,
)


_HASH_A = "a" * 64
_HASH_B = "b" * 64
_HASH_C = "c" * 64
_UPDATE80_CHECKPOINT_SHA256 = (
    "35e04c86f9f973af028fb08f0175d42ab45378d09e1f2b96ee6aad6e4c12b5b5"
)
_UPDATE80_POLICY_STATE_SHA256 = (
    "3123e6adde99be41e3bd5cc2f3843068e5416892395926d759c2c6a243ffd381"
)
_DENOMINATOR_SOURCE = "reachable_observable_free_highres_cells/v1"
_DENOMINATOR_ALGORITHM = "exact_reachable_safe_pose_range_los/v1"
_TEST_SOURCE = "tests/test_xunce_mid_dual_aggregate.py"


_FIXTURE_SOURCE_CONTRACTS: dict[str, dict[str, object]] = {
    "g1": {
        "schema_version": "xunce-mid-dual-source-contract/v1",
        "gate_id": "g1",
        "config_schema_version": "xunce-mid-dual-g1-effective-config/v1",
        "runner_id": "run_xunce_mid_dual_g1_coverage/v1",
        "required_phase_ids": [
            "p01",
            "p02",
            "p03",
            "p04",
            "p05",
            "p06",
            "p07",
        ],
        "output_base": "D:/xunce/out/mid_dual/g1",
        "input_audit_schema_version": "xunce-mid-dual-g1-input-audit/v1",
        "report_audit_schema_version": "xunce-mid-dual-source-report-audit/v1",
        "report_renderer_id": "xunce-mid-dual-g1-report-renderer/v1",
        "required_lineage_sources": [_TEST_SOURCE],
    },
    "g2": {
        "schema_version": "xunce-mid-dual-source-contract/v1",
        "gate_id": "g2",
        "config_schema_version": "xunce-mid-dual-g2-planning-time-config/v1",
        "runner_id": "xunce-mid-dual-g2-planning-time-runner/v1",
        "required_phase_ids": ["p01", "p02", "p03", "p04"],
        "output_base": "D:/xunce/out/mid_dual/g2",
        "input_audit_schema_version": "xunce-mid-dual-g2-input-audit/v1",
        "report_audit_schema_version": "xunce-mid-dual-source-report-audit/v1",
        "report_renderer_id": "xunce-mid-dual-g2-canonical-report/v1",
        "required_lineage_sources": [_TEST_SOURCE],
    },
    "g3": {
        "schema_version": "xunce-mid-dual-source-contract/v1",
        "gate_id": "g3",
        "config_schema_version": "mid-dual-g3-config/v1",
        "runner_id": "run_xunce_mid_dual_g3_closed_loop/v1",
        "required_phase_ids": ["p01", "p02"],
        "output_base": "D:/xunce/out/mid_dual/g3",
        "input_audit_schema_version": "xunce-mid-dual-g3-input-audit/v1",
        "report_audit_schema_version": "xunce-mid-dual-source-report-audit/v1",
        "report_renderer_id": "xunce-mid-dual-g3-canonical-report/v1",
        "required_lineage_sources": [_TEST_SOURCE],
    },
}


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


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_text(payload: str) -> str:
    return _sha256_bytes(payload.encode("utf-8"))


def _json_text(payload: Mapping[str, object]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def _json_sha256(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return _sha256_bytes(encoded)


def _g2_semantic_digest(
    request_sha256: str,
    provider_success: bool,
) -> str:
    payload = json.dumps(
        {
            "request_sha256": request_sha256,
            "provider_success": provider_success,
            "route_l2_valid": provider_success,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    parts = (b"xunce-mid-dual-g2-provider-semantics/v1", payload)
    framed = b"".join(struct.pack(">Q", len(part)) + part for part in parts)
    return _sha256_bytes(framed)


def _artifact_text_bytes(text: str) -> bytes:
    return text.replace("\n", os.linesep).encode("utf-8")


def _nearest_rank(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(quantile * len(ordered)) - 1)]


def _timing_expected(values: Sequence[float]) -> dict[str, object]:
    materialized = [float(value) for value in values]
    mean_ms = statistics.mean(materialized)
    p95_ms = _nearest_rank(materialized, 0.95)
    maximum = max(materialized)
    return {
        "sample_count": len(materialized),
        "mean_ms": mean_ms,
        "p50_ms": _nearest_rank(materialized, 0.50),
        "p95_ms": p95_ms,
        "p99_ms": _nearest_rank(materialized, 0.99),
        "sample_stddev_ms": (
            statistics.stdev(materialized) if len(materialized) > 1 else 0.0
        ),
        "min_ms": min(materialized),
        "max_ms": maximum,
        "at_or_below_1000_count": sum(value <= 1000.0 for value in materialized),
        "over_2000_count": sum(value > 2000.0 for value in materialized),
        "midterm_reduced_passed": (
            mean_ms <= 2000.0 and p95_ms <= 2000.0 and maximum <= 2000.0
        ),
        "final_threshold_reduced_passed": (
            mean_ms <= 1000.0
            and p95_ms <= 1000.0
            and sum(value <= 1000.0 for value in materialized)
            / len(materialized)
            >= 0.95
            and maximum <= 2000.0
        ),
    }


def _coverage_expected(values: Sequence[float]) -> dict[str, object]:
    materialized = [float(value) for value in values]
    mean_value = statistics.mean(materialized)
    count80 = sum(value >= 0.80 for value in materialized)
    count99 = sum(value >= 0.99 for value in materialized)
    midterm = mean_value >= 0.80 and count80 >= 23
    final = mean_value >= 0.99 and count99 >= 23
    return {
        "status": "passed" if midterm else "failed",
        "sample_count": len(materialized),
        "mean": mean_value,
        "p50": _nearest_rank(materialized, 0.50),
        "p95": _nearest_rank(materialized, 0.95),
        "p99": _nearest_rank(materialized, 0.99),
        "sample_stddev": (
            statistics.stdev(materialized) if len(materialized) > 1 else 0.0
        ),
        "min": min(materialized),
        "max": max(materialized),
        "coverage_80_count": count80,
        "coverage_99_count": count99,
        "midterm_reduced_passed": midterm,
        "final_threshold_reduced_passed": final,
    }


def _lineage_code_sha256() -> str:
    source = Path(__file__).resolve()
    payload = [
        {
            "logical_path": _TEST_SOURCE,
            "sha256": _sha256_bytes(artifact_io.read_bytes(source)),
            "size_bytes": artifact_io.file_size(source),
        }
    ]
    domain = b"xunce-mid-dual-code/v1\0"
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return _sha256_bytes(domain + canonical)


def _g1_input_audit() -> dict[str, object]:
    formal_jobs: list[dict[str, object]] = []
    for split in ("test_q24", "unseen24"):
        for index in range(24):
            scenario_id = f"{split}-scenario-{index:02d}"
            formal_jobs.append(
                {
                    "split": split,
                    "scenario_id": scenario_id,
                    "episode_id": f"{split}-episode-{index:02d}",
                    "episode_index": index,
                    "lane_id": f"lane-{index % 8}",
                    "denominator_mask_sha256": _sha256_text(
                        f"denominator:{scenario_id}"
                    ),
                    "denominator_cell_count": 100,
                    "denominator_source": _DENOMINATOR_SOURCE,
                    "denominator_algorithm": _DENOMINATOR_ALGORITHM,
                }
            )
    return {
        "schema_version": "xunce-mid-dual-g1-input-audit/v1",
        "gate_id": "g1",
        "runner_id": "run_xunce_mid_dual_g1_coverage/v1",
        "scale_profile": SCALE_PROFILE,
        "status": "verified",
        "scenario_manifest_path": "D:/xunce/inputs/mid_dual/g1/manifest.json",
        "scenario_manifest_schema_version": "mid-dual-scenario-freeze/v1",
        "scenario_manifest_sha256": _sha256_text("fixture-g1-freeze"),
        "scenario_manifest_canonical_sha256": _sha256_text(
            "fixture-g1-freeze-canonical"
        ),
        "policy_blind_attestation_sha256": _sha256_text(
            "fixture-policy-blind"
        ),
        "checkpoint_sha256": _UPDATE80_CHECKPOINT_SHA256,
        "policy_state_sha256": _UPDATE80_POLICY_STATE_SHA256,
        "denominator_source": _DENOMINATOR_SOURCE,
        "denominator_algorithm": _DENOMINATOR_ALGORITHM,
        "formal_split_order": ["test_q24", "unseen24"],
        "formal_jobs": formal_jobs,
        "test_c24_scenario_ids": [
            f"test_c24-scenario-{index:02d}" for index in range(24)
        ],
        "validation3_scenario_ids": [
            f"validation-scenario-{index:02d}" for index in range(3)
        ],
        "replay3_scenario_ids": [
            f"replay-scenario-{index:02d}" for index in range(3)
        ],
    }


def _g2_input_audit() -> dict[str, object]:
    requests: list[dict[str, object]] = []
    for platform in G2_PLATFORMS:
        platform_index = 0
        for scale, normal_count, hard_count, unreachable_count in (
            ("standard", 23, 7, 3),
            ("kilometer", 6, 2, 2),
        ):
            classes = (
                ["normal_reachable"] * normal_count
                + ["hard_reachable"] * hard_count
                + ["unreachable"] * unreachable_count
            )
            for scale_index, request_class in enumerate(classes):
                request_id = f"{platform}-request-{platform_index:02d}"
                request_sha256 = _sha256_text(
                    f"request:{platform}:{scale}:{scale_index}"
                )
                expected_success = request_class != "unreachable"
                requests.append(
                    {
                        "schema_version": "xunce-mid-dual-g2-truth-provider-crosswalk/v1",
                        "platform": platform,
                        "scale": scale,
                        "request_index": platform_index,
                        "request_id": request_id,
                        "provider_request_sha256": request_sha256,
                        "request_class": request_class,
                        "outcome_kind": (
                            "reachable" if expected_success else "unreachable"
                        ),
                        "truth_request_sha256": _sha256_text(f"truth:{request_sha256}"),
                        "truth_certificate_sha256": _sha256_text(
                            f"oracle-result:{request_sha256}"
                        ),
                        "terrain_sha256": _sha256_text(f"terrain:{request_sha256}"),
                        "expected_success": expected_success,
                        "expected_route_l2_valid": expected_success,
                    }
                )
                platform_index += 1
    return {
        "schema_version": "xunce-mid-dual-g2-input-audit/v1",
        "gate_id": "g2",
        "scale_profile": SCALE_PROFILE,
        "status": "ready",
        "formal_evidence_eligible": True,
        "blockers": [],
        "truth_bundle_root": "D:/xunce/inputs/mid_dual/g2/truth",
        "truth_manifest_sha256": _sha256_text("fixture-g2-truth-manifest"),
        "truth_freeze_sha256": _sha256_text("fixture-g2-truth-freeze"),
        "truth_payload_root_sha256": _sha256_text("fixture-g2-payload-root"),
        "truth_source_attestations_sha256": _sha256_text("fixture-g2-attestations"),
        "input_set_id": "fixture-g2-input-set",
        "manifest_core_sha256": _sha256_text("fixture-g2-manifest-core"),
        "authorization_sha256": _sha256_text("fixture-g2-authorization"),
        "candidate_boundary": {
            "technical_independence": "T2_candidate",
            "organizational_independence": "project_internal",
            "source_attestation_status": "pending_o2_signature",
            "candidate_formal_evidence_eligible": False,
        },
        "provider_source": {
            "identity": "path-planner-v2-provider/v1",
            "source_bytes_sha256": _sha256_text("provider-source-bytes"),
            "implementation_sha256": _sha256_text("provider-implementation"),
        },
        "oracle_source": {
            "identity": "independent-finite-oracle/v1",
            "source_bytes_sha256": _sha256_text("oracle-source-bytes"),
            "implementation_sha256": _sha256_text("oracle-implementation"),
        },
        "expected_approval_artifact_path": "D:/xunce/inputs/mid_dual/g2-approvals/fixture/artifact-bound-o2-approval.json",
        "approval": {
            "schema_version": "xunce-mid-dual-g2-artifact-bound-approval/v1",
            "approval_id": "fixture-approval",
            "formal_evidence_eligible": True,
            "provider_source": {
                "identity": "path-planner-v2-provider/v1",
                "source_bytes_sha256": _sha256_text("provider-source-bytes"),
                "implementation_sha256": _sha256_text("provider-implementation"),
            },
            "oracle_source": {
                "identity": "independent-finite-oracle/v1",
                "source_bytes_sha256": _sha256_text("oracle-source-bytes"),
                "implementation_sha256": _sha256_text("oracle-implementation"),
            },
            "g2_scope_authorized": True,
            "g3_scope_authorized": True,
            "physical_capability_claimed": False,
            "hardware_certification_claimed": False,
            "approval_record_sha256": _sha256_text("fixture-approval-record"),
            "approval_artifact_sha256": _sha256_text("fixture-approval-artifact"),
            "approval_artifact_path": "D:/xunce/inputs/mid_dual/g2-approvals/fixture/artifact-bound-o2-approval.json",
        },
        "hopper_resolution": {"formal_evidence_eligible": True, "blockers": []},
        "primitive_label_audit": {"status": "verified"},
        "small_map_optimum_audit": {"status": "verified"},
        "request_matrix_audit": {"status": "verified"},
        "ppo_target_audit": {"status": "verified"},
        "g3_replay_cohort": {"cohort_sha256": _sha256_text("fixture-g3-cohort")},
        "g3_replay_cohort_sha256": _sha256_text("fixture-g3-cohort"),
        "requests": requests,
        "provider_requests": [],
        "terrain_sha256": sorted({row["terrain_sha256"] for row in requests}),
        "formal_row_count": 0,
    }


def _manifest_sha256(root: Path) -> str:
    return _sha256_bytes(artifact_io.read_bytes(root / "manifest.json"))


def _g3_input_audit(g1_root: Path, g2_root: Path) -> dict[str, object]:
    g1_rows = artifact_io.read_jsonl(g1_root / "results.jsonl")
    g2_rows = artifact_io.read_jsonl(g2_root / "results.jsonl")
    wheel: list[dict[str, object]] = []
    for split in ("test_q24", "unseen24"):
        selected = [row for row in g1_rows if row.get("split") == split][:5]
        for row in selected:
            wheel.append(
                {
                    "split": split,
                    "scenario_id": row["scenario_id"],
                    "g1_episode_id": row["episode_id"],
                    "g1_episode_index": row["episode_index"],
                    "lane_id": row["lane_id"],
                    "wheel_episode_id": f"wheel-{row['episode_id']}",
                }
            )
    interface: list[dict[str, object]] = []
    for platform in ("legged", "hopper"):
        grouped_by_request: dict[str, list[dict[str, object]]] = {}
        for row in g2_rows:
            if row.get("platform") != platform:
                continue
            request_id = str(row["request_id"])
            grouped_by_request.setdefault(request_id, []).append(row)
        complete_groups = [
            selected_rows
            for selected_rows in grouped_by_request.values()
            if sorted(
                int(row["repeat_index"]) for row in selected_rows
            )
            == list(range(G2_REPEATS))
        ][:3]
        assert len(complete_groups) == 3
        for index, selected_rows in enumerate(complete_groups):
            ordered = sorted(
                selected_rows,
                key=lambda row: int(row["repeat_index"]),
            )
            assert [row["repeat_index"] for row in ordered] == list(
                range(G2_REPEATS)
            )
            row = ordered[0]
            interface.append(
                {
                    "platform": platform,
                    "replay_id": f"{platform}-replay-{index:02d}",
                    "g2_reference_call_ids": [
                        selected["call_id"] for selected in ordered
                    ],
                    "request_id": row["request_id"],
                    "request_sha256": row["request_sha256"],
                    "g2_semantic_digest": row["semantic_digest"],
                    "timing_contract_id": "five-phase-sequential-ns/v1",
                }
            )
    return {
        "schema_version": "xunce-mid-dual-g3-input-audit/v1",
        "gate_id": "g3",
        "run_id": "g3-run",
        "scale_profile": SCALE_PROFILE,
        "formal_evidence_eligible": True,
        "g1_source_manifest_sha256": _manifest_sha256(g1_root),
        "g2_source_manifest_sha256": _manifest_sha256(g2_root),
        "wheel_selections": wheel,
        "interface_selections": interface,
    }


def _source_config(
    gate_id: str,
    input_audit: Mapping[str, object],
    *,
    upstream_binding: Mapping[str, object] | None = None,
) -> dict[str, object]:
    contract = _FIXTURE_SOURCE_CONTRACTS[gate_id]
    payload: dict[str, object] = {
        "schema_version": contract["config_schema_version"],
        "gate_id": gate_id,
        "runner_id": contract["runner_id"],
        "run_id": f"{gate_id}-run",
        "output_root": contract["output_base"],
        "scale_profile": SCALE_PROFILE,
        "input_sha256": _sha256_bytes(_json_text(input_audit).encode("utf-8")),
        "code_sha256": _lineage_code_sha256(),
        "required_phase_ids": list(contract["required_phase_ids"]),
        "source_contract_sha256": _json_sha256(contract),
        "evidence_binding": {
            "schema_version": f"xunce-mid-dual-{gate_id}-evidence-binding/v1",
            "input_audit_path": "source_input_audit.json",
            "lineage_audit_path": "lineage_audit.json",
            "report_audit_path": "source_report_audit.json",
        },
    }
    if gate_id == "g3":
        payload["upstream_binding"] = dict(upstream_binding or {})
    return payload


def _expected_g1(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    splits = {
        split: _coverage_expected(
            [float(row["coverage"]) for row in rows if row.get("split") == split]
        )
        for split in ("test_q24", "unseen24")
    }
    safety_clean = all(row.get("safety_violation_count") == 0 for row in rows)
    masked_clean = all(row.get("masked_action_count") == 0 for row in rows)
    midterm = (
        safety_clean
        and masked_clean
        and all(item["midterm_reduced_passed"] is True for item in splits.values())
    )
    final = (
        safety_clean
        and masked_clean
        and all(
            item["final_threshold_reduced_passed"] is True
            for item in splits.values()
        )
    )
    return {
        "status": "passed" if midterm else "failed",
        "sample_count": len(rows),
        "g1_coverage_80_passed": midterm,
        "g1_coverage_99_passed": final,
        "safety_clean": safety_clean,
        "masked_action_clean": masked_clean,
        "checkpoint_sha256": _UPDATE80_CHECKPOINT_SHA256,
        "policy_state_sha256": _UPDATE80_POLICY_STATE_SHA256,
        "denominator_integrity_passed": True,
        "splits": splits,
    }


def _expected_g2(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    timing_by_platform_scale: dict[str, dict[str, object]] = {}
    timing_by_platform_scale_outcome: dict[str, dict[str, object]] = {}
    timing_by_platform_scale_class: dict[str, dict[str, object]] = {}
    class_counts: dict[str, dict[str, int]] = {}
    for platform in G2_PLATFORMS:
        for scale in ("standard", "kilometer"):
            selected = [
                row
                for row in rows
                if row.get("platform") == platform and row.get("scale") == scale
            ]
            timing_by_platform_scale[f"{platform}/{scale}"] = _timing_expected(
                [float(row["elapsed_ms"]) for row in selected]
            )
            class_counts[f"{platform}/{scale}"] = {
                request_class: len(
                    {
                        str(row["request_id"])
                        for row in selected
                        if row.get("request_class") == request_class
                    }
                )
                for request_class in (
                    "normal_reachable",
                    "hard_reachable",
                    "unreachable",
                )
            }
            for outcome in ("reachable", "unreachable"):
                subset = [row for row in selected if row.get("outcome_kind") == outcome]
                timing_by_platform_scale_outcome[
                    f"{platform}/{scale}/{outcome}"
                ] = _timing_expected(
                    [float(row["elapsed_ms"]) for row in subset]
                )
            for request_class in (
                "normal_reachable",
                "hard_reachable",
                "unreachable",
            ):
                subset = [
                    row
                    for row in selected
                    if row.get("request_class") == request_class
                ]
                timing_by_platform_scale_class[
                    f"{platform}/{scale}/{request_class}"
                ] = _timing_expected(
                    [float(row["elapsed_ms"]) for row in subset]
                )
    reachable_successes = {
        platform: len(
            {
                str(row["request_id"])
                for row in rows
                if row.get("platform") == platform
                and row.get("outcome_kind") == "reachable"
                and row.get("provider_success") is True
                and row.get("route_l2_valid") is True
            }
        )
        for platform in G2_PLATFORMS
    }
    unreachable_correct = all(
        row.get("provider_success") is False and row.get("route_l2_valid") is False
        for row in rows
        if row.get("outcome_kind") == "unreachable"
    )
    correctness = (
        reachable_successes == {platform: 38 for platform in G2_PLATFORMS}
        and unreachable_correct
    )
    partitions = [
        *timing_by_platform_scale.values(),
        *timing_by_platform_scale_outcome.values(),
        *timing_by_platform_scale_class.values(),
    ]
    midterm = correctness and all(
        item["midterm_reduced_passed"] is True for item in partitions
    )
    final = correctness and all(
        item["final_threshold_reduced_passed"] is True for item in partitions
    )
    return {
        "status": "passed" if midterm else "failed",
        "formal_call_count": len(rows),
        "unique_request_count": len({str(row["request_id"]) for row in rows}),
        "g2_all_platforms_2s_passed": midterm,
        "g2_all_platforms_1s_passed": final,
        "correctness_passed": correctness,
        "request_class_counts_by_platform_scale": class_counts,
        "timing_by_platform_scale": timing_by_platform_scale,
        "timing_by_platform_scale_outcome": timing_by_platform_scale_outcome,
        "timing_by_platform_scale_class": timing_by_platform_scale_class,
        "reachable_correctness": {
            "reachable_unique_request_count_by_platform": {
                platform: 38 for platform in G2_PLATFORMS
            },
            "reachable_unique_success_count_by_platform": reachable_successes,
            "unreachable_correct": unreachable_correct,
            "semantic_consensus": True,
            "truth_provider_crosswalk_valid": True,
        },
    }


def _expected_g3(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    wheel = [row for row in rows if row.get("row_kind") == "g3_wheel_step"]
    interface = [
        row for row in rows if row.get("row_kind") == "g3_interface_replay"
    ]
    terminal = [row for row in wheel if row.get("is_terminal") is True]
    final_coverages = [float(row["coverage"]) for row in terminal]
    paired = [float(row["paired_g1_coverage"]) for row in terminal]
    coverage_mean = statistics.mean(final_coverages)
    paired_deltas = [
        actual - reference
        for actual, reference in zip(final_coverages, paired, strict=True)
    ]
    wheel_timing = _timing_expected(
        [float(row["total_ns"]) / 1_000_000.0 for row in wheel]
    )
    interface_timing = _timing_expected(
        [float(row["total_ns"]) / 1_000_000.0 for row in interface]
    )
    integrity = all(
        row.get("planner_success") is True
        and row.get("planner_failure_reason") == "none"
        and row.get("safety_violation_count") == 0
        and row.get("masked_action_count") == 0
        and row.get("candidate_route_mismatch_count") == 0
        for row in wheel
    )
    interface_correct = all(
        row.get("g2_semantic_digest") == row.get("replay_semantic_digest")
        and row.get("timing_contract_id") == "five-phase-sequential-ns/v1"
        and row.get("formal_input_eligible") is True
        for row in interface
    )
    coverage_mid = coverage_mean >= 0.80 and all(
        value >= 0.80 for value in final_coverages
    )
    coverage_final = coverage_mean >= 0.99 and all(
        value >= 0.99 for value in final_coverages
    )
    shared = (
        integrity
        and interface_correct
        and statistics.mean(paired_deltas) >= -0.01
    )
    midterm = (
        shared
        and coverage_mid
        and wheel_timing["midterm_reduced_passed"] is True
        and interface_timing["midterm_reduced_passed"] is True
    )
    final = (
        shared
        and coverage_final
        and wheel_timing["final_threshold_reduced_passed"] is True
        and interface_timing["final_threshold_reduced_passed"] is True
    )
    return {
        "status": "passed" if midterm else "failed",
        "wheel_episode_count": len({str(row["episode_id"]) for row in wheel}),
        "wheel_step_count": len(wheel),
        "interface_replay_count": len(interface),
        "split_episode_counts": {
            split: len(
                {
                    str(row["episode_id"])
                    for row in wheel
                    if row.get("split") == split
                }
            )
            for split in ("test_q24", "unseen24")
        },
        "g3_midterm_crosscheck_passed": midterm,
        "g3_final_crosscheck_passed": final,
        "wheel_coverage_mean": coverage_mean,
        "wheel_coverage_80_count": sum(value >= 0.80 for value in final_coverages),
        "wheel_coverage_99_count": sum(value >= 0.99 for value in final_coverages),
        "paired_g1_coverage_delta_mean": statistics.mean(paired_deltas),
        "paired_g1_coverage_delta_sample_stddev": (
            statistics.stdev(paired_deltas) if len(paired_deltas) > 1 else 0.0
        ),
        "wheel_timing": wheel_timing,
        "interface_timing": interface_timing,
        "wheel_integrity_passed": integrity,
        "interface_correctness_passed": interface_correct,
    }


def _expected_gate_summary(
    gate_id: str,
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    return {
        "g1": _expected_g1,
        "g2": _expected_g2,
        "g3": _expected_g3,
    }[gate_id](rows)


def _create_source_root(
    tmp_path: Path,
    gate_id: str,
    row_factory: Callable[[str], list[dict[str, object]]],
    summary: dict[str, object],
    *,
    input_audit: dict[str, object] | None = None,
    input_mutation: Callable[[dict[str, object]], None] | None = None,
    upstream_binding: Mapping[str, object] | None = None,
) -> Path:
    root = tmp_path / gate_id
    if input_audit is None:
        if gate_id == "g1":
            input_audit = _g1_input_audit()
        elif gate_id == "g2":
            input_audit = _g2_input_audit()
        else:
            raise AssertionError("G3 fixtures require an explicit cross-root input audit")
    if input_mutation is not None:
        input_mutation(input_audit)
    effective_config = _source_config(
        gate_id,
        input_audit,
        upstream_binding=upstream_binding,
    )
    store = MidDualRunStore.create_new(root, effective_config)
    rows = row_factory(store.config_sha256)
    for row in rows:
        row["run_id"] = f"{gate_id}-run"
        if row.get("scale_profile") == SCALE_PROFILE:
            row["scale_profile"] = SCALE_PROFILE
        if row.get("input_sha256") == _HASH_A:
            row["input_sha256"] = effective_config["input_sha256"]
        if row.get("code_sha256") == _HASH_B:
            row["code_sha256"] = effective_config["code_sha256"]
        if row.get("source_sha256") == _HASH_B:
            row["source_sha256"] = effective_config["code_sha256"]
    for phase_id in _FIXTURE_SOURCE_CONTRACTS[gate_id]["required_phase_ids"]:
        phase_rows = rows if phase_id == _FIXTURE_SOURCE_CONTRACTS[gate_id]["required_phase_ids"][-1] else []
        attempt = store.write_phase_attempt(
            str(phase_id),
            phase_rows,
            {"gate_id": gate_id, "phase_id": phase_id},
        )
        store.accept_phase(
            str(phase_id),
            attempt,
            store.phase_attempt_row_sha256(str(phase_id), attempt),
        )
    store.capture_lineage([Path(__file__)], _HASH_A[:40], _HASH_B[:40])
    assert store.capture_environment(_environment_probe)["status"] == "captured"
    recomputed = _expected_gate_summary(gate_id, rows)
    recomputed.update(summary)
    stored = {
        "schema_version": f"xunce-mid-dual-{gate_id}-canonical-summary/v1",
        "scale_profile": SCALE_PROFILE,
        "gate_id": gate_id,
        "run_id": f"{gate_id}-run",
        "status": recomputed["status"],
        "formal_evidence_eligible": True,
        "recomputed": recomputed,
    }
    report = f"# {gate_id.upper()} canonical source report\n"
    report_audit = {
        "schema_version": "xunce-mid-dual-source-report-audit/v1",
        "gate_id": gate_id,
        "run_id": f"{gate_id}-run",
        "renderer_id": _FIXTURE_SOURCE_CONTRACTS[gate_id]["report_renderer_id"],
        "summary_projection_sha256": _json_sha256(recomputed),
        "stored_summary_bytes_sha256": _sha256_bytes(
            _artifact_text_bytes(_json_text(stored))
        ),
        "report_bytes_sha256": _sha256_bytes(_artifact_text_bytes(report)),
        "formal_evidence_eligible": True,
    }
    store.finalize(
        stored,
        {
            "schema_version": f"xunce-mid-dual-{gate_id}-routing/v1",
            "gate_id": gate_id,
            "run_id": f"{gate_id}-run",
            "status": stored["status"],
            "formal_evidence_eligible": True,
        },
        report,
        {"source_input": input_audit, "source_report": report_audit},
    )
    assert store.verify_manifest(root) is True
    return root


def _g1_rows(
    config_sha256: str,
    *,
    coverage: float = 0.99,
    mutation: Callable[[list[dict[str, object]]], None] | None = None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    audit = _g1_input_audit()
    for frozen in audit["formal_jobs"]:
        denominator_count = int(frozen["denominator_cell_count"])
        split = str(frozen["split"])
        phase_id = "p03" if split == "test_q24" else "p04"
        rows.append(
            {
                "row_kind": "coverage_episode",
                "schema_version": "xunce-mid-dual-g1-coverage-episode/v1",
                "gate_id": "g1",
                "runner_id": "run_xunce_mid_dual_g1_coverage/v1",
                "phase_id": phase_id,
                "phase_name": split,
                "scale_profile": SCALE_PROFILE,
                "run_id": "g1-run",
                "episode_id": frozen["episode_id"],
                "scenario_id": frozen["scenario_id"],
                "split": frozen["split"],
                "episode_index": frozen["episode_index"],
                "lane_id": frozen["lane_id"],
                "source_sha256": _HASH_B,
                "config_sha256": config_sha256,
                "input_sha256": _HASH_A,
                "code_sha256": _HASH_B,
                "scenario_manifest_sha256": audit[
                    "scenario_manifest_sha256"
                ],
                "checkpoint_sha256": _UPDATE80_CHECKPOINT_SHA256,
                "policy_state_sha256": _UPDATE80_POLICY_STATE_SHA256,
                "denominator_sha256": frozen["denominator_mask_sha256"],
                "denominator_cell_count": denominator_count,
                "denominator_source": _DENOMINATOR_SOURCE,
                "denominator_algorithm": _DENOMINATOR_ALGORITHM,
                "initial_covered_cell_count": 0,
                "final_covered_cell_count": int(
                    round(coverage * denominator_count)
                ),
                "coverage": coverage,
                "elapsed_ms": 10.0,
                "steps_executed": 10,
                "termination_reason": "environment_done",
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
    input_audit: Mapping[str, object] | None = None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    audit = dict(input_audit or _g2_input_audit())
    provider = audit["provider_source"]
    oracle = audit["oracle_source"]
    for request in audit["requests"]:
        for repeat in range(G2_REPEATS):
            rows.append(
                {
                    "row_kind": "g2_planning_call",
                    "schema_version": "xunce-mid-dual-g2-planning-call-row/v1",
                    "scale_profile": SCALE_PROFILE,
                    "run_id": "g2-run",
                    "episode_id": request["request_id"],
                    "request_id": request["request_id"],
                    "call_id": (
                        f"{request['platform']}-call-"
                        f"{int(request['request_index']):02d}-{repeat}"
                    ),
                    "platform": request["platform"],
                    "scale": request["scale"],
                    "request_class": request["request_class"],
                    "outcome_kind": request["outcome_kind"],
                    "source_sha256": _HASH_B,
                    "config_sha256": config_sha256,
                    "input_sha256": _HASH_A,
                    "code_sha256": _HASH_B,
                    "request_sha256": request["provider_request_sha256"],
                    "truth_sha256": request["truth_request_sha256"],
                    "provider_sha256": provider["implementation_sha256"],
                    "provider_source_bytes_sha256": provider[
                        "source_bytes_sha256"
                    ],
                    "oracle_sha256": oracle["implementation_sha256"],
                    "oracle_source_bytes_sha256": oracle[
                        "source_bytes_sha256"
                    ],
                    "provider_result_sha256": _sha256_text(
                        f"provider-result:{request['provider_request_sha256']}"
                    ),
                    "oracle_result_sha256": request["truth_certificate_sha256"],
                    "input_validation_ns": 10_000_000,
                    "platform_instantiation_ns": 10_000_000,
                    "search_ns": int((elapsed_ms - 40.0) * 1_000_000),
                    "complete_route_validation_ns": 10_000_000,
                    "result_assembly_ns": 10_000_000,
                    "total_ns": int(elapsed_ms * 1_000_000),
                    "elapsed_ms": elapsed_ms,
                    "input_validation_ms": 10.0,
                    "platform_instantiation_ms": 10.0,
                    "search_ms": elapsed_ms - 40.0,
                    "complete_route_validation_ms": 10.0,
                    "result_assembly_ms": 10.0,
                    "provider_success": request["expected_success"],
                    "route_l2_valid": request["expected_route_l2_valid"],
                    "semantic_digest": _g2_semantic_digest(
                        request["provider_request_sha256"],
                        request["expected_success"],
                    ),
                    "timing_contract_id": "five-phase-sequential-ns/v1",
                    "formal_sample": True,
                    "repeat_index": repeat,
                }
            )
    if mutation is not None:
        mutation(rows)
    return rows


def _g3_rows(
    config_sha256: str,
    *,
    coverage: float = 0.99,
    elapsed_ms: float = 100.0,
    mutation: Callable[[list[dict[str, object]]], None] | None = None,
    input_audit: Mapping[str, object],
) -> list[dict[str, object]]:
    del config_sha256
    g3 = importlib.import_module("run_xunce_mid_dual_g3_closed_loop")
    rows: list[dict[str, object]] = []
    for selection in input_audit["wheel_selections"]:
        pre_snapshot_sha256 = _sha256_text(
            f"pre:{selection['scenario_id']}:0"
        )
        for step_index in range(2):
            decision_sha256 = _sha256_text(
                f"decision:{selection['scenario_id']}:{step_index}"
            )
            candidate_id = (
                f"g3-candidate:{selection['scenario_id']}:{step_index:03d}"
            )
            selected_cell = [step_index + 1, step_index + 2]
            selected_theta = 0.25 + step_index * 0.10
            candidate_sha256 = g3.deterministic_candidate_sha256(
                scenario_id=str(selection["scenario_id"]),
                step_index=step_index,
                pre_snapshot_sha256=pre_snapshot_sha256,
                decision_sha256=decision_sha256,
                candidate_id=candidate_id,
                selected_candidate_cell_xy=selected_cell,
                selected_theta=selected_theta,
            )
            request_id = g3.deterministic_request_id(
                scenario_id=str(selection["scenario_id"]),
                step_index=step_index,
                pre_snapshot_sha256=pre_snapshot_sha256,
            )
            request_sha256 = g3.deterministic_request_sha256(
                request_id=request_id,
                candidate_sha256=candidate_sha256,
                pre_snapshot_sha256=pre_snapshot_sha256,
            )
            planned_path_cells = [[0, 0], selected_cell]
            planner_path_length_m = 1.0 + step_index
            route_result_sha256 = g3.deterministic_route_result_sha256(
                request_sha256=request_sha256,
                planner_success=True,
                planner_failure_reason="none",
                planned_path_cells=planned_path_cells,
                planner_path_length_m=planner_path_length_m,
                route_endpoint_cell_xy=selected_cell,
                route_endpoint_theta=selected_theta,
            )
            feedback_sha256 = g3.deterministic_feedback_sha256(
                route_result_sha256=route_result_sha256,
                feedback_pose_cell_xy=selected_cell,
                feedback_theta=selected_theta,
                coverage=coverage,
                safety_violation_count=0,
                masked_action_count=0,
            )
            post_snapshot_sha256 = _sha256_text(
                f"next-state:{feedback_sha256}"
            )
            total_ns = int(round(elapsed_ms * 1_000_000.0))
            components = [total_ns // 5] * 4
            components.append(total_ns - sum(components))
            rows.append(
                {
                    "row_kind": "g3_wheel_step",
                    "schema_version": "mid-dual-g3-wheel-step/v1",
                    "scale_profile": SCALE_PROFILE,
                    "run_id": "g3-run",
                    "split": selection["split"],
                    "episode_index": selection["g1_episode_index"],
                    "episode_id": selection["wheel_episode_id"],
                    "scenario_id": selection["scenario_id"],
                    "lane_id": selection["lane_id"],
                    "step_id": (
                        f"{selection['wheel_episode_id']}-step-{step_index:02d}"
                    ),
                    "step_index": step_index,
                    "is_terminal": step_index == 1,
                    "termination_reason": (
                        "coverage_complete" if step_index == 1 else "none"
                    ),
                    "decision_sha256": decision_sha256,
                    "candidate_id": candidate_id,
                    "candidate_sha256": candidate_sha256,
                    "request_candidate_sha256": candidate_sha256,
                    "selected_candidate_cell_xy": selected_cell,
                    "planned_path_cells": planned_path_cells,
                    "planner_path_length_m": planner_path_length_m,
                    "route_endpoint_cell_xy": selected_cell,
                    "feedback_pose_cell_xy": selected_cell,
                    "selected_theta": selected_theta,
                    "route_endpoint_theta": selected_theta,
                    "feedback_theta": selected_theta,
                    "pre_snapshot_sha256": pre_snapshot_sha256,
                    "request_id": request_id,
                    "request_sha256": request_sha256,
                    "route_request_id": request_id,
                    "route_request_sha256": request_sha256,
                    "route_result_sha256": route_result_sha256,
                    "feedback_request_id": request_id,
                    "feedback_route_result_sha256": route_result_sha256,
                    "feedback_sha256": feedback_sha256,
                    "post_snapshot_parent_sha256": feedback_sha256,
                    "post_snapshot_sha256": post_snapshot_sha256,
                    "coverage": coverage,
                    "paired_g1_coverage": 0.99,
                    "safety_violation_count": 0,
                    "masked_action_count": 0,
                    "candidate_route_mismatch_count": 0,
                    "planner_success": True,
                    "planner_failure_reason": "none",
                    "input_validation_ns": components[0],
                    "platform_instantiation_ns": components[1],
                    "search_ns": components[2],
                    "complete_route_validation_ns": components[3],
                    "result_assembly_ns": components[4],
                    "total_ns": total_ns,
                }
            )
            pre_snapshot_sha256 = post_snapshot_sha256
    for selection in input_audit["interface_selections"]:
        total_ns = int(round(elapsed_ms * 1_000_000.0))
        components = [total_ns // 5] * 4
        components.append(total_ns - sum(components))
        rows.append(
            {
                "row_kind": "g3_interface_replay",
                "schema_version": "mid-dual-g3-interface-replay/v1",
                "scale_profile": SCALE_PROFILE,
                "run_id": "g3-run",
                "replay_id": selection["replay_id"],
                "platform": selection["platform"],
                "request_id": selection["request_id"],
                "request_sha256": selection["request_sha256"],
                "g2_reference_call_ids": selection[
                    "g2_reference_call_ids"
                ],
                "g2_semantic_digest": selection["g2_semantic_digest"],
                "replay_semantic_digest": selection["g2_semantic_digest"],
                "timing_contract_id": selection["timing_contract_id"],
                "formal_input_eligible": True,
                "input_validation_ns": components[0],
                "platform_instantiation_ns": components[1],
                "search_ns": components[2],
                "complete_route_validation_ns": components[3],
                "result_assembly_ns": components[4],
                "total_ns": total_ns,
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
    g3 = _create_g3_source_root(tmp_path, g1, g2)
    return g1, g2, g3


def _create_g3_source_root(
    tmp_path: Path,
    g1_root: Path,
    g2_root: Path,
    *,
    mutation: Callable[[list[dict[str, object]]], None] | None = None,
    summary: Mapping[str, object] | None = None,
    input_mutation: Callable[[dict[str, object]], None] | None = None,
) -> Path:
    input_audit = _g3_input_audit(g1_root, g2_root)
    if input_mutation is not None:
        input_mutation(input_audit)
    upstream_binding = {
        "g1_source_manifest_sha256": input_audit[
            "g1_source_manifest_sha256"
        ],
        "g2_source_manifest_sha256": input_audit[
            "g2_source_manifest_sha256"
        ],
    }
    return _create_source_root(
        tmp_path,
        "g3",
        lambda config_sha256: _g3_rows(
            config_sha256,
            input_audit=input_audit,
            mutation=mutation,
        ),
        dict(
            summary
            or {
                "status": "passed",
                "wheel_episode_count": 10,
                "interface_replay_count": 6,
                "g3_midterm_crosscheck_passed": True,
                "g3_final_crosscheck_passed": True,
            }
        ),
        input_audit=input_audit,
        upstream_binding=upstream_binding,
    )


def _aggregate(
    roots: tuple[Path, Path, Path],
) -> dict[str, object]:
    return _aggregate_module().aggregate_completed_roots(
        *roots,
        source_contracts=_FIXTURE_SOURCE_CONTRACTS,
    )


def _create_legacy_source_root(
    root: Path,
    gate_id: str,
    row_factory: Callable[[str], list[dict[str, object]]],
) -> Path:
    config = {
        "schema_version": "mid-dual-effective-config/v1",
        "gate_id": gate_id,
        "run_id": f"{gate_id}-run",
        "scale_profile": SCALE_PROFILE,
        "input_sha256": _HASH_A,
        "code_sha256": _HASH_B,
        "required_phase_ids": ["p01"],
    }
    store = MidDualRunStore.create_new(root, config)
    rows = row_factory(store.config_sha256)
    for row in rows:
        row["run_id"] = f"{gate_id}-run"
        row["scale_profile"] = SCALE_PROFILE
        row["config_sha256"] = store.config_sha256
        row["input_sha256"] = _HASH_A
        row["code_sha256"] = _HASH_B
        row["source_sha256"] = _HASH_B
    attempt = store.write_phase_attempt("p01", rows, {"gate_id": gate_id})
    store.accept_phase("p01", attempt, store.phase_attempt_row_sha256("p01", attempt))
    store.capture_lineage([Path(__file__)], _HASH_A[:40], _HASH_B[:40])
    store.capture_environment(_environment_probe)
    summary = {
        "schema_version": f"mid-dual-{gate_id}-summary/v1",
        "scale_profile": SCALE_PROFILE,
        "gate_id": gate_id,
        **_expected_gate_summary(gate_id, rows),
    }
    store.finalize(summary, {"status": summary["status"]}, f"{gate_id} report", {})
    assert store.verify_manifest(root) is True
    return root


def _legacy_roots(tmp_path: Path) -> tuple[Path, Path, Path]:
    g1 = _create_legacy_source_root(tmp_path / "g1", "g1", _g1_rows)
    g2 = _create_legacy_source_root(tmp_path / "g2", "g2", _g2_rows)
    g3_input = _g3_input_audit(g1, g2)
    g3 = _create_legacy_source_root(
        tmp_path / "g3",
        "g3",
        lambda config_sha256: _g3_rows(
            config_sha256,
            input_audit=g3_input,
        ),
    )
    return g1, g2, g3


def _refresh_manifest_entry(root: Path, relative_path: str) -> None:
    manifest = artifact_io.read_json(root / "manifest.json")
    for entry in manifest["artifacts"]:
        if entry["path"] == relative_path:
            entry["sha256"] = _sha256_bytes(
                artifact_io.read_bytes(root / relative_path)
            )
            break
    else:
        raise AssertionError(f"missing manifest entry: {relative_path}")
    artifact_io.write_json(root / "manifest.json", manifest)


def _rewrite_results(
    root: Path,
    mutation: Callable[[list[dict[str, object]]], None],
) -> None:
    rows = artifact_io.read_jsonl(root / "results.jsonl")
    mutation(rows)
    artifact_io.write_jsonl(root / "results.jsonl", rows)
    _refresh_manifest_entry(root, "results.jsonl")


def _rewrite_json(
    root: Path,
    relative_path: str,
    mutation: Callable[[dict[str, object]], None],
) -> None:
    payload = artifact_io.read_json(root / relative_path)
    mutation(payload)
    artifact_io.write_json(root / relative_path, payload)
    _refresh_manifest_entry(root, relative_path)


def test_generic_manifest_valid_roots_are_not_formal_sources(tmp_path: Path) -> None:
    """Catch generic MidDualRunStore self-labels being accepted as gate evidence."""
    result = _aggregate(_legacy_roots(tmp_path))

    assert result["status"] == "blocked"
    assert result["formal_evidence_eligible"] is False
    assert any("config_schema_version" in reason for reason in result["blockers"])


@pytest.mark.parametrize(
    ("mutation_kind", "blocking_fragment"),
    (
        ("coverage_two", "coverage_out_of_range"),
        ("checkpoint", "checkpoint_binding"),
        ("denominator", "denominator_binding"),
        ("scenario", "frozen_episode_binding"),
        ("lane", "frozen_episode_binding"),
    ),
)
def test_g1_exact_freeze_update80_and_integer_denominator_fail_closed(
    tmp_path: Path,
    mutation_kind: str,
    blocking_fragment: str,
) -> None:
    """Catch G1 evidence that escapes the frozen update80 integer-count join."""
    g1, g2, g3 = _valid_roots(tmp_path)

    def mutate(rows: list[dict[str, object]]) -> None:
        row = rows[0]
        if mutation_kind == "coverage_two":
            row["final_covered_cell_count"] = 200
            row["coverage"] = 2.0
        elif mutation_kind == "checkpoint":
            row["checkpoint_sha256"] = _HASH_C
        elif mutation_kind == "denominator":
            row["denominator_sha256"] = _HASH_C
        elif mutation_kind == "scenario":
            row["scenario_id"] = "invented-scenario"
        else:
            row["lane_id"] = "lane-7"

    _rewrite_results(g1, mutate)
    result = _aggregate((g1, g2, g3))

    assert result["status"] == "blocked"
    assert any(blocking_fragment in reason for reason in result["blockers"])


@pytest.mark.parametrize(
    ("mutation_kind", "blocking_fragment"),
    (
        ("cross_scale", "request_matrix"),
        ("unknown", "outcome_taxonomy"),
        ("provider_oracle_same", "provider_oracle_identity"),
        ("single_request_hash", "request_hash_unique"),
    ),
)
def test_g2_exact_input_matrix_and_approval_chain_fail_closed(
    tmp_path: Path,
    mutation_kind: str,
    blocking_fragment: str,
) -> None:
    """Catch a self-consistent but unapproved or wrongly partitioned G2 input set."""
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
    input_audit = _g2_input_audit()
    requests = input_audit["requests"]
    if mutation_kind == "cross_scale":
        standard_hard = next(
            row
            for row in requests
            if row["scale"] == "standard"
            and row["request_class"] == "hard_reachable"
        )
        kilometer_unreachable = next(
            row
            for row in requests
            if row["scale"] == "kilometer"
            and row["request_class"] == "unreachable"
        )
        standard_hard["request_class"] = "unreachable"
        standard_hard["outcome_kind"] = "unreachable"
        standard_hard["expected_success"] = False
        standard_hard["expected_route_l2_valid"] = False
        kilometer_unreachable["request_class"] = "hard_reachable"
        kilometer_unreachable["outcome_kind"] = "reachable"
        kilometer_unreachable["expected_success"] = True
        kilometer_unreachable["expected_route_l2_valid"] = True
    elif mutation_kind == "unknown":
        target_id = requests[0]["request_id"]
        for row in requests:
            if row["request_id"] == target_id:
                row["outcome_kind"] = "unknown"
    elif mutation_kind == "provider_oracle_same":
        input_audit["provider_source"] = dict(input_audit["oracle_source"])
    else:
        for row in requests:
            row["provider_request_sha256"] = _HASH_C
    g2 = _create_source_root(
        tmp_path,
        "g2",
        lambda config_sha256: _g2_rows(
            config_sha256,
            input_audit=input_audit,
        ),
        {
            "status": "passed",
            "formal_call_count": 645,
            "g2_all_platforms_2s_passed": True,
            "g2_all_platforms_1s_passed": True,
        },
        input_audit=input_audit,
    )
    g3 = _create_g3_source_root(tmp_path, g1, g2)

    result = _aggregate((g1, g2, g3))

    assert result["status"] == "blocked"
    assert any(blocking_fragment in reason for reason in result["blockers"])


@pytest.mark.parametrize(
    ("mutation_kind", "blocking_fragment"),
    (
        ("raw_ns", "timing_component_sum"),
        ("derived_ms", "timing_ms_projection"),
        ("provider_result", "provider_result_consensus"),
    ),
)
def test_g2_task8_raw_ns_and_provider_result_are_fail_closed(
    tmp_path: Path,
    mutation_kind: str,
    blocking_fragment: str,
) -> None:
    """Task8 formal rows retain raw ns; ms and provider results are derived evidence."""
    _g1, g2, _g3 = _valid_roots(tmp_path)
    rows = artifact_io.read_jsonl(g2 / "results.jsonl")

    def mutate(rows: list[dict[str, object]]) -> None:
        if mutation_kind == "raw_ns":
            rows[0]["total_ns"] = int(rows[0]["total_ns"]) + 1
        elif mutation_kind == "derived_ms":
            rows[0]["elapsed_ms"] = 101.0
        else:
            rows[1]["provider_result_sha256"] = _HASH_C

    mutate(rows)
    module = _aggregate_module()
    with pytest.raises(module.AggregateBlocked, match=blocking_fragment):
        module.recompute_g2(
            rows,
            artifact_io.read_json(g2 / "config.json"),
            artifact_io.read_json(g2 / "source_input_audit.json"),
        )



def test_g2_results_must_be_the_canonical_p04_formal_payload(
    tmp_path: Path,
) -> None:
    """Task8 diagnostics remain in phase audits; results.jsonl is p04-only."""
    g1, g2, g3 = _valid_roots(tmp_path)
    _rewrite_results(g2, lambda rows: rows.pop())

    result = _aggregate((g1, g2, g3))

    assert result["status"] == "blocked"
    assert "g2_results_not_canonical_p04" in result["blockers"]


@pytest.mark.parametrize(
    ("mutation_kind", "blocking_fragment"),
    (
        ("rogue_scenario", "g1_cross_root_join"),
        ("rogue_request", "g2_cross_root_join"),
        ("step_gap", "step_sequence"),
        ("terminal_then_step", "terminal_position"),
        ("duplicate_interface_request", "interface_request_unique"),
    ),
)
def test_g3_cross_root_join_and_sequence_fail_closed(
    tmp_path: Path,
    mutation_kind: str,
    blocking_fragment: str,
) -> None:
    """Catch G3 self-reported joins that do not exist in G1/G2 raw evidence."""
    g1, g2, g3 = _valid_roots(tmp_path)

    def mutate(rows: list[dict[str, object]]) -> None:
        wheel = [row for row in rows if row["row_kind"] == "g3_wheel_step"]
        interface = [
            row for row in rows if row["row_kind"] == "g3_interface_replay"
        ]
        if mutation_kind == "rogue_scenario":
            wheel[0]["scenario_id"] = "rogue-scenario"
        elif mutation_kind == "rogue_request":
            interface[0]["request_id"] = "rogue-request"
        elif mutation_kind == "step_gap":
            wheel[1]["step_index"] = 2
        elif mutation_kind == "terminal_then_step":
            wheel[0]["is_terminal"] = True
            wheel[1]["is_terminal"] = False
        else:
            interface[-1]["request_id"] = interface[-2]["request_id"]
            interface[-1]["request_sha256"] = interface[-2]["request_sha256"]

    _rewrite_results(g3, mutate)
    result = _aggregate((g1, g2, g3))

    assert result["status"] == "blocked"
    assert any(blocking_fragment in reason for reason in result["blockers"])


@pytest.mark.parametrize(
    ("artifact_name", "blocking_fragment"),
    (
        ("summary.json", "stored_summary_mismatch"),
        ("report.md", "report_audit_mismatch"),
    ),
)
def test_complete_stored_summary_and_report_are_fail_closed_consistency_checks(
    tmp_path: Path,
    artifact_name: str,
    blocking_fragment: str,
) -> None:
    """Catch one nested summary leaf or report byte drifting behind a valid manifest."""
    g1, g2, g3 = _valid_roots(tmp_path)
    if artifact_name == "summary.json":
        _rewrite_json(
            g2,
            artifact_name,
            lambda payload: payload["recomputed"]["timing_by_platform_scale"][
                "wheel/standard"
            ].__setitem__("p95_ms", 101.0),
        )
    else:
        artifact_io.write_text(g2 / artifact_name, "# drifted report\n")
        _refresh_manifest_entry(g2, artifact_name)

    result = _aggregate((g1, g2, g3))

    assert result["status"] == "blocked"
    assert any(blocking_fragment in reason for reason in result["blockers"])


def test_results_replacement_after_manifest_verification_is_blocked(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catch the old verify-then-reread TOCTOU window for canonical results."""
    module = _aggregate_module()
    roots = _valid_roots(tmp_path)
    g1 = roots[0].resolve()
    original_verify = module.MidDualRunStore.verify_manifest
    replaced = False

    def verify_then_replace(root: str | Path) -> bool:
        nonlocal replaced
        result = original_verify(root)
        if Path(root).resolve() == g1 and not replaced:
            rows = artifact_io.read_jsonl(g1 / "results.jsonl")
            compact = "\n" + "".join(
                json.dumps(row, ensure_ascii=False, sort_keys=False) + "\n"
                for row in rows
            )
            artifact_io.write_text(g1 / "results.jsonl", compact)
            replaced = True
        return result

    monkeypatch.setattr(
        module.MidDualRunStore,
        "verify_manifest",
        staticmethod(verify_then_replace),
    )
    result = module.aggregate_completed_roots(
        *roots,
        source_contracts=_FIXTURE_SOURCE_CONTRACTS,
    )

    assert replaced is True
    assert result["status"] == "blocked"
    assert any("immutable_bytes_drift" in reason for reason in result["blockers"])


def test_unhashable_or_malformed_rows_become_stable_blocked_results(
    tmp_path: Path,
) -> None:
    """Catch malformed JSON values escaping as TypeError before grouping."""
    g1, g2, g3 = _valid_roots(tmp_path)

    def mutate(rows: list[dict[str, object]]) -> None:
        rows[0]["request_id"] = ["not", "hashable"]

    _rewrite_results(g2, mutate)
    result = _aggregate((g1, g2, g3))

    assert result["status"] == "blocked"
    assert result["formal_evidence_eligible"] is False
    assert any(reason.startswith("g2_") for reason in result["blockers"])


def test_formal_path_run_id_resume_and_cli_status_contracts(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catch C-drive output, path-like IDs, invalid resume, and one-state CLI output."""
    module = _aggregate_module()
    source_config = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "xunce_mid_dual_aggregate_v1.json"
    )
    config = artifact_io.read_json(source_config)
    config["output_root"] = "C:/xunce/out/mid_dual/aggregate"
    bad_config = tmp_path / "bad-config.json"
    artifact_io.write_json(bad_config, config)
    with pytest.raises(ValueError, match="D:/xunce/out/mid_dual/aggregate"):
        module._load_aggregate_config(bad_config)
    for run_id in (".", "..", "a/../b", "CON", "run."):
        with pytest.raises(ValueError, match="run_id"):
            module._validate_run_id(run_id)

    effective = {
        "schema_version": "xunce-mid-dual-aggregate-config/v1",
        "scale_profile": SCALE_PROFILE,
        "required_phase_ids": ["p01"],
        "run_id": "resume-run",
    }
    resume_root = tmp_path / "resume-run"
    store = MidDualRunStore.create_new(resume_root, effective)
    with pytest.raises(ValueError, match="resume_config_mismatch"):
        module._open_aggregate_store(
            resume_root,
            {**effective, "run_id": "different"},
            mode="resume",
        )
    resumed = module._open_aggregate_store(
        resume_root,
        effective,
        mode="resume",
    )
    assert resumed.accepted_phase_ids == ()

    summary_root = tmp_path / "cli-result"
    artifact_io.make_dirs(summary_root)
    artifact_io.write_json(
        summary_root / "summary.json",
        {"status": "failed"},
    )
    monkeypatch.setattr(module, "run_aggregate", lambda **_: summary_root)
    assert module.main(
        [
            "--config",
            str(source_config),
            "--g1-root",
            "D:/xunce/out/mid_dual/g1/r1",
            "--g2-root",
            "D:/xunce/out/mid_dual/g2/r2",
            "--g3-root",
            "D:/xunce/out/mid_dual/g3/r3",
            "--run-id",
            "aggregate-r1",
            "--mode",
            "create",
        ]
    ) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["execution_status"] == "complete"
    assert payload["gate_status"] == "failed"
    assert "status" not in payload


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
    g3 = _create_g3_source_root(tmp_path, g1, g2)

    result = _aggregate((g1, g2, g3))

    assert result["status"] == "blocked"
    assert result["gates"]["g1"]["g1_coverage_80_passed"] is False
    assert "g1_stored_summary_mismatch" in result["blockers"]


def test_aggregate_recomputes_g1_g2_and_g3_from_raw_rows(tmp_path: Path) -> None:
    """Catch an aggregate that forwards gate booleans instead of recalculating rows."""
    result = _aggregate(_valid_roots(tmp_path))

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
    g2 = _create_source_root(
        tmp_path / "other",
        "g2",
        _g2_rows,
        {
            "status": "passed",
            "formal_call_count": 645,
            "g2_all_platforms_2s_passed": True,
            "g2_all_platforms_1s_passed": True,
        },
    )
    g3 = _create_g3_source_root(tmp_path / "other", g1, g2)
    result = _aggregate((g1, g2, g3))

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
    g3 = _create_g3_source_root(tmp_path, g1, g2)
    result = _aggregate((g1, g2, g3))

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
    g3 = _create_g3_source_root(tmp_path, g1, g2, mutation=mutate)

    result = _aggregate((g1, g2, g3))

    assert result["status"] == "blocked"
    assert any("g3_" in reason for reason in result["blockers"])


def test_aggregate_blocks_when_recomputed_and_stored_summaries_differ(tmp_path: Path) -> None:
    """Catch a complete root whose stored G2 gate disagrees with the raw rows."""
    g1 = _create_source_root(
        tmp_path / "valid",
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
            "status": "failed",
            "formal_call_count": 645,
            "g2_all_platforms_2s_passed": False,
            "g2_all_platforms_1s_passed": False,
        },
    )
    g3 = _create_g3_source_root(tmp_path / "valid", g1, g2)

    result = _aggregate((g1, g2, g3))

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
    summary = _aggregate(_valid_roots(tmp_path))
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
