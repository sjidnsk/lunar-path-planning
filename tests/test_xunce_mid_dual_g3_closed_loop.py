"""G3 闭环计时与跨门强联接合同。"""

from __future__ import annotations

import copy
from contextlib import redirect_stdout
import hashlib
import importlib.util
import inspect
import io
import json
from pathlib import Path
import struct
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_xunce_mid_dual_g3_closed_loop.py"
CONFIG = ROOT / "configs/xunce_mid_dual_g3_closed_loop_v1.json"
SCALE_PROFILE = "midterm_reduced_w8x3_update80/v1"
ACTIVE_PLATFORMS = ["wheel", "legged", "hopper"]
PLATFORM_INVARIANTS = {
    platform: {"max_traversable_slope_deg": 30.0}
    for platform in ACTIVE_PLATFORMS
}
AUTHORIZED_G1_ROOT = (
    "D:/xunce/out/mid_dual/g1/g1-formal-20260727-0650-cst"
)
AUTHORIZED_G1_ENVELOPE = (
    "D:/xunce/out/mid_dual/g1-assessment-use/"
    "g1-existing-run-assessment-v1-"
    "1e8a98015839b9190c86efc531134b7cbde1be393250cf40479b19a98c5b0fe8"
)
UPDATE80_CHECKPOINT_SHA256 = (
    "35e04c86f9f973af028fb08f0175d42ab45378d09e1f2b96ee6aad6e4c12b5b5"
)
UPDATE80_POLICY_STATE_SHA256 = (
    "3123e6adde99be41e3bd5cc2f3843068e5416892395926d759c2c6a243ffd381"
)


def _module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("mid_dual_g3", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    scripts_path = str(ROOT / "scripts")
    sys.path.insert(0, scripts_path)
    try:
        spec.loader.exec_module(module)
    finally:
        assert sys.path[0] == scripts_path
        sys.path.pop(0)
    return module


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _jsonl_bytes(rows: list[dict[str, object]]) -> bytes:
    return b"".join(
        json.dumps(
            row,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
        for row in rows
    )


def _framed_domain_sha256(domain: str, *parts: bytes) -> str:
    payload = bytearray()
    for part in (domain.encode("utf-8"), *parts):
        payload.extend(struct.pack(">Q", len(part)))
        payload.extend(part)
    return hashlib.sha256(bytes(payload)).hexdigest()


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _native_g1_evidence_binding(manifest_sha256: str) -> dict[str, object]:
    return {
        "g1_evidence_kind": "native_completed_run_v1",
        "g1_source_manifest_sha256": manifest_sha256,
    }


def _g2_r6_runtime_fixture() -> dict[str, object]:
    closure_schema = "xunce-mid-dual-path-planner-runtime-source-closure/v1"
    source_paths = sorted(
        (
            "src/path_planner/v2/api.py",
            "src/path_planner/v2/formal_request_codec.py",
            "src/path_planner/v2/hopper_api.py",
            "src/path_planner/v2/hopper_authority.py",
            "src/path_planner/v2/hopper_route_validation.py",
            "src/path_planner/v2/profiles.py",
            "src/path_planner/v2/providers/fixture.py",
            "src/path_planner/v2/terrain.py",
            "src/path_planner/v2/validation.py",
        )
    )
    closure_core = {
        "schema_version": closure_schema,
        "submodule_commit": "1" * 40,
        "dirty_inventory": [],
        "required_sources": [
            {
                "logical_path": path,
                "size_bytes": index + 1,
                "sha256": _sha(f"path-planner:{path}"),
            }
            for index, path in enumerate(source_paths)
        ],
        "active_platforms": list(ACTIVE_PLATFORMS),
        "platform_invariants": copy.deepcopy(PLATFORM_INVARIANTS),
    }
    closure = {
        **closure_core,
        "path_planner_runtime_source_closure_sha256": (
            _framed_domain_sha256(
                closure_schema,
                _canonical_bytes(closure_core),
            )
        ),
    }
    closure_sha256 = str(
        closure["path_planner_runtime_source_closure_sha256"]
    )
    local_source_sha256 = _sha("g2-local-source")
    code_sha256 = _framed_domain_sha256(
        "xunce-mid-dual-g2-code-with-runtime-source/v1",
        _canonical_bytes(
            {
                "local_source_sha256": local_source_sha256,
                "path_planner_runtime_source_closure_sha256": closure_sha256,
            }
        ),
    )
    config_sha256 = _sha("g2-config")
    input_sha256 = _sha("g2-input")
    config = {
        "run_id": "g2-formal",
        "config_sha256": config_sha256,
        "input_sha256": input_sha256,
        "code_sha256": code_sha256,
        "active_platforms": list(ACTIVE_PLATFORMS),
        "platform_invariants": copy.deepcopy(PLATFORM_INVARIANTS),
        "path_planner_runtime_source_closure_sha256": closure_sha256,
        "evidence_binding": {
            "schema_version": "xunce-mid-dual-g2-evidence-binding/v1",
            "input_audit_path": "g2_input_audit.json",
            "lineage_audit_path": "lineage_audit.json",
            "report_audit_path": "g2_report_audit.json",
            "runtime_source_closure_audit_path": (
                "g2_runtime_source_closure_audit.json"
            ),
            "platform_invariants_audit_path": (
                "g2_platform_invariants_audit.json"
            ),
        },
    }
    input_audit = {
        "active_platforms": list(ACTIVE_PLATFORMS),
        "platform_invariants": copy.deepcopy(PLATFORM_INVARIANTS),
        "path_planner_runtime_source_closure": closure,
        "path_planner_runtime_source_closure_sha256": closure_sha256,
    }
    approval = {
        "path_planner_runtime_source_closure": closure,
        "path_planner_runtime_source_closure_sha256": closure_sha256,
    }
    hopper_resolution = {
        "path_planner_runtime_source_closure_sha256": closure_sha256,
    }
    phase_audits = {
        phase_id: {
            "active_platforms": list(ACTIVE_PLATFORMS),
            "platform_invariants": copy.deepcopy(PLATFORM_INVARIANTS),
            "path_planner_runtime_source_closure_sha256": closure_sha256,
        }
        for phase_id in ("p01", "p02", "p03", "p04")
    }
    summary = {
        "active_platforms": list(ACTIVE_PLATFORMS),
        "platform_invariants": copy.deepcopy(PLATFORM_INVARIANTS),
        "path_planner_runtime_source_closure_sha256": closure_sha256,
    }
    platform_audit = {
        "schema_version": (
            "xunce-mid-dual-g2-platform-invariants-audit/v1"
        ),
        "gate_id": "g2",
        "scale_profile": SCALE_PROFILE,
        "run_id": "g2-formal",
        "status": "passed",
        "formal_evidence_eligible": True,
        "active_platforms": list(ACTIVE_PLATFORMS),
        "platform_invariants": copy.deepcopy(PLATFORM_INVARIANTS),
        "path_planner_runtime_source_closure_sha256": closure_sha256,
        "config_sha256": config_sha256,
        "input_sha256": input_sha256,
        "code_sha256": code_sha256,
    }
    return {
        "closure": closure,
        "closure_sha256": closure_sha256,
        "local_source_sha256": local_source_sha256,
        "config": config,
        "input_audit": input_audit,
        "approval": approval,
        "hopper_resolution": hopper_resolution,
        "phase_audits": phase_audits,
        "summary": summary,
        "platform_audit": platform_audit,
    }


def _scenario_id(split: str, index: int) -> str:
    return f"{split}/scenario-{index:02d}/standard-proxy/v1"


def _frozen_manifest() -> dict[str, object]:
    test_q24 = [_scenario_id("test", index) for index in range(24)]
    unseen24 = [_scenario_id("unseen", index) for index in range(24)]
    return {
        "schema_version": "mid-dual-scenario-freeze/v1",
        "completion_status": "complete",
        "cohorts": {
            "test_q24": test_q24,
            "unseen24": unseen24,
            "g3_test_q5": test_q24[:5],
            "g3_unseen5": unseen24[:5],
        },
    }


def _g1_rows(*, coverage: float = 0.99) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for split, source_split in (("test_q24", "test"), ("unseen24", "unseen")):
        for index in range(24):
            denominator = 10_000
            covered = int(round(coverage * denominator))
            code_sha = _sha("g1-code")
            rows.append(
                {
                    "row_kind": "coverage_episode",
                    "schema_version": (
                        "xunce-mid-dual-g1-coverage-episode/v1"
                    ),
                    "gate_id": "g1",
                    "runner_id": "run_xunce_mid_dual_g1_coverage/v1",
                    "phase_id": "p03" if split == "test_q24" else "p04",
                    "phase_name": split,
                    "scale_profile": SCALE_PROFILE,
                    "run_id": "g1-formal-run",
                    "split": split,
                    "episode_index": index,
                    "episode_id": f"{split}-episode-{index:02d}",
                    "scenario_id": _scenario_id(source_split, index),
                    "lane_id": f"lane-{index % 8}",
                    "source_sha256": code_sha,
                    "config_sha256": _sha("g1-config"),
                    "input_sha256": _sha("g1-input"),
                    "code_sha256": code_sha,
                    "scenario_manifest_sha256": _sha(
                        "frozen-scenario-manifest"
                    ),
                    "checkpoint_sha256": UPDATE80_CHECKPOINT_SHA256,
                    "policy_state_sha256": UPDATE80_POLICY_STATE_SHA256,
                    "denominator_source": (
                        "reachable_observable_free_highres_cells/v1"
                    ),
                    "denominator_algorithm": (
                        "exact_reachable_safe_pose_range_los/v1"
                    ),
                    "denominator_sha256": _sha(
                        f"mask:{source_split}:{index}"
                    ),
                    "denominator_cell_count": denominator,
                    "initial_covered_cell_count": 0,
                    "final_covered_cell_count": covered,
                    "coverage": covered / denominator,
                    "elapsed_ms": 100.0,
                    "steps_executed": 1,
                    "termination_reason": "success_done",
                    "safety_violation_count": 0,
                    "masked_action_count": 0,
                }
            )
    return rows


def _existing_run_assessment_fixture(
    *,
    numerical_result_status: str = "failed",
) -> tuple[SimpleNamespace, dict[str, object], list[dict[str, object]]]:
    manifest_sha256 = _sha("frozen-scenario-manifest")
    rows = _g1_rows(coverage=0.99)
    if numerical_result_status == "failed":
        for row in rows:
            if (
                row["split"] == "unseen24"
                and int(row["episode_index"]) in {3, 10, 23}
            ):
                row["final_covered_cell_count"] = 200
                row["coverage"] = 0.02
                row["termination_reason"] = "no_candidate_done"
    for row in rows:
        row["run_id"] = "g1-formal-20260727-0650-cst"
        row["scenario_manifest_sha256"] = manifest_sha256

    config = {
        "schema_version": "xunce-mid-dual-g1-effective-config/v1",
        "gate_id": "g1",
        "runner_id": "run_xunce_mid_dual_g1_coverage/v1",
        "scale_profile": SCALE_PROFILE,
        "run_id": "g1-formal-20260727-0650-cst",
        "mode": "formal",
        "output_root": AUTHORIZED_G1_ROOT,
        "required_phase_ids": [
            "p01",
            "p02",
            "p03",
            "p04",
            "p05",
            "p06",
            "p07",
        ],
        "checkpoint": {
            "update": 80,
            "sha256": UPDATE80_CHECKPOINT_SHA256,
            "policy_state_sha256": UPDATE80_POLICY_STATE_SHA256,
        },
        "scenario_manifest": {
            "sha256": manifest_sha256,
        },
        "config_sha256": _sha("g1-config"),
        "input_sha256": _sha("g1-input"),
        "code_sha256": _sha("g1-code"),
    }
    split_metrics: dict[str, dict[str, object]] = {}
    for split in ("test_q24", "unseen24"):
        values = [
            float(row["coverage"]) for row in rows if row["split"] == split
        ]
        split_metrics[split] = {
            "sample_count": 24,
            "mean": sum(values) / 24,
            "coverage_80_count": sum(value >= 0.80 for value in values),
            "coverage_99_count": sum(value >= 0.99 for value in values),
            "safety_violation_count": 0,
            "masked_action_count": 0,
            "integer_denominator_check": "passed",
            "phase_audit_equality": True,
        }
    review_metrics = {
        "schema_version": (
            "xunce-mid-dual-g1-existing-run-independent-recompute/v1"
        ),
        "formal_episode_count": 48,
        "split_order": ["test_q24", "unseen24"],
        "splits": split_metrics,
        "phase_state_sha256": _sha("phase-state"),
        "input_audit_canonical_sha256": _sha("g1-input"),
        "scenario_manifest_sha256": manifest_sha256,
        "max_traversable_slope_deg": 30.0,
        "safety_clean": True,
        "masked_action_clean": True,
        "review_result_status": numerical_result_status,
        "replay_required": False,
        "replay_status": "not_run_by_ruling",
    }
    snapshot = {
        "config.json": (
            json.dumps(config, ensure_ascii=False, sort_keys=True) + "\n"
        ).encode("utf-8"),
        "phase-state.jsonl": b"phase-state-fixture\n",
        "phase-attempts.jsonl": b"phase-attempts-fixture\n",
        "phases/p01/a01/audit.json": b"{}\n",
        "phases/p01/a01/results.jsonl": b"",
        "phases/p02/a01/audit.json": b"{}\n",
        "phases/p02/a01/results.jsonl": b"",
        "phases/p03/a01/audit.json": b"{}\n",
        "phases/p03/a01/results.jsonl": _jsonl_bytes(
            [row for row in rows if row["split"] == "test_q24"]
        ),
        "phases/p04/a01/audit.json": b"{}\n",
        "phases/p04/a01/results.jsonl": _jsonl_bytes(
            [row for row in rows if row["split"] == "unseen24"]
        ),
        "external/scenario-manifest.json": b"{}\n",
    }
    verified = SimpleNamespace(
        source_root=Path(AUTHORIZED_G1_ROOT),
        review_root=Path("D:/xunce/out/mid_dual/g1-review/review"),
        envelope_root=Path(AUTHORIZED_G1_ENVELOPE),
        phase_snapshot_sha256=_sha("phase-snapshot"),
        stop_evidence_sha256=_sha("stop-evidence"),
        review_manifest_sha256=_sha("review-manifest"),
        review_result_sha256=_sha("review-result"),
        envelope_sha256=_sha("assessment-envelope"),
        envelope_manifest_sha256=_sha("assessment-manifest"),
        verifier_source_sha256=_sha("assessment-verifier"),
        assessment_use_status="authorized_existing_run",
        strict_prestart_lineage_status="not_satisfied",
        lineage_limitation_acknowledged=True,
        numerical_gate_status="recomputed_from_raw_rows",
        numerical_result_status=numerical_result_status,
        review_metrics=review_metrics,
        snapshot=snapshot,
    )
    return verified, {
        "manifest_sha256": manifest_sha256,
    }, rows


def _timing_ns(total_ns: int = 100_000_000) -> dict[str, int]:
    component = total_ns // 5
    return {
        "input_validation_ns": component,
        "platform_instantiation_ns": component,
        "search_ns": component,
        "complete_route_validation_ns": component,
        "result_assembly_ns": total_ns - component * 4,
        "total_ns": total_ns,
    }


def _wheel_rows(
    module: ModuleType,
    *,
    coverage: float = 0.99,
    total_ns: int = 100_000_000,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for split, source_split in (("test_q24", "test"), ("unseen24", "unseen")):
        for index in range(5):
            scenario_id = _scenario_id(source_split, index)
            episode_id = f"g3:{split}:{index:02d}"
            pre = _sha(f"pre:{scenario_id}:0")
            request_id = module.deterministic_request_id(
                scenario_id=scenario_id,
                step_index=0,
                pre_snapshot_sha256=pre,
            )
            candidate_id = f"candidate-{split}-{index:02d}"
            decision_sha = _sha(f"decision:{scenario_id}:0")
            candidate_cell = [index + 1, index + 2]
            candidate_sha = module.deterministic_candidate_sha256(
                scenario_id=scenario_id,
                step_index=0,
                pre_snapshot_sha256=pre,
                decision_sha256=decision_sha,
                candidate_id=candidate_id,
                selected_candidate_cell_xy=candidate_cell,
                selected_theta=0.0,
            )
            request_sha = module.deterministic_request_sha256(
                request_id=request_id,
                candidate_sha256=candidate_sha,
                pre_snapshot_sha256=pre,
            )
            planned_path = [[0, 0], candidate_cell]
            route_sha = module.deterministic_route_result_sha256(
                request_sha256=request_sha,
                planner_success=True,
                planner_failure_reason="none",
                planned_path_cells=planned_path,
                planner_path_length_m=1.0,
                route_endpoint_cell_xy=candidate_cell,
                route_endpoint_theta=0.0,
            )
            feedback_sha = module.deterministic_feedback_sha256(
                route_result_sha256=route_sha,
                feedback_pose_cell_xy=candidate_cell,
                feedback_theta=0.0,
                coverage=coverage,
                safety_violation_count=0,
                masked_action_count=0,
            )
            post = _sha(f"post:{feedback_sha}")
            rows.append(
                {
                    "row_kind": "g3_wheel_step",
                    "schema_version": "mid-dual-g3-wheel-step/v1",
                    "scale_profile": SCALE_PROFILE,
                    "run_id": "g3-formal-run",
                    "split": split,
                    "episode_index": index,
                    "episode_id": episode_id,
                    "scenario_id": scenario_id,
                    "lane_id": f"lane-{index % 8}",
                    "step_id": f"{episode_id}:step:000",
                    "step_index": 0,
                    "is_terminal": True,
                    "termination_reason": "success_done",
                    "decision_sha256": decision_sha,
                    "candidate_id": candidate_id,
                    "candidate_sha256": candidate_sha,
                    "request_candidate_sha256": candidate_sha,
                    "selected_candidate_cell_xy": candidate_cell,
                    "planned_path_cells": planned_path,
                    "planner_path_length_m": 1.0,
                    "route_endpoint_cell_xy": candidate_cell,
                    "feedback_pose_cell_xy": candidate_cell,
                    "selected_theta": 0.0,
                    "route_endpoint_theta": 0.0,
                    "feedback_theta": 0.0,
                    "pre_snapshot_sha256": pre,
                    "request_id": request_id,
                    "request_sha256": request_sha,
                    "route_request_id": request_id,
                    "route_request_sha256": request_sha,
                    "route_result_sha256": route_sha,
                    "feedback_request_id": request_id,
                    "feedback_route_result_sha256": route_sha,
                    "feedback_sha256": feedback_sha,
                    "post_snapshot_parent_sha256": feedback_sha,
                    "post_snapshot_sha256": post,
                    "coverage": coverage,
                    "paired_g1_coverage": coverage,
                    "safety_violation_count": 0,
                    "masked_action_count": 0,
                    "candidate_route_mismatch_count": 0,
                    "planner_success": True,
                    "planner_failure_reason": "none",
                    **_timing_ns(total_ns),
                }
            )
    return rows


def _g2_reference_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for platform in ("legged", "hopper"):
        for index in range(3):
            request_id = f"g2-{platform}-{index:02d}"
            request_sha = _sha(f"g2-request:{platform}:{index}")
            semantic = _sha(f"g2-semantic:{platform}:{index}")
            for repeat in range(5):
                rows.append(
                    {
                        "row_kind": "g2_planning_call",
                        "platform": platform,
                        "request_id": request_id,
                        "request_sha256": request_sha,
                        "call_id": f"{request_id}:repeat:{repeat}",
                        "repeat_index": repeat,
                        "semantic_digest": semantic,
                        "provider_success": True,
                        "route_l2_valid": True,
                        "timing_contract_id": "five-phase-sequential-ns/v1",
                    }
                )
    return rows


def _rebind_route_and_feedback(
    module: ModuleType,
    row: dict[str, object],
) -> None:
    route_sha = module.deterministic_route_result_sha256(
        request_sha256=row["request_sha256"],
        planner_success=row["planner_success"],
        planner_failure_reason=row["planner_failure_reason"],
        planned_path_cells=row["planned_path_cells"],
        planner_path_length_m=row["planner_path_length_m"],
        route_endpoint_cell_xy=row["route_endpoint_cell_xy"],
        route_endpoint_theta=row["route_endpoint_theta"],
    )
    row["route_result_sha256"] = route_sha
    row["feedback_route_result_sha256"] = route_sha
    feedback_sha = module.deterministic_feedback_sha256(
        route_result_sha256=route_sha,
        feedback_pose_cell_xy=row["feedback_pose_cell_xy"],
        feedback_theta=row["feedback_theta"],
        coverage=row["coverage"],
        safety_violation_count=row["safety_violation_count"],
        masked_action_count=row["masked_action_count"],
    )
    row["feedback_sha256"] = feedback_sha
    row["post_snapshot_parent_sha256"] = feedback_sha


def _interface_rows(
    *,
    formal_input_eligible: bool = True,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for platform in ("legged", "hopper"):
        for index in range(3):
            request_id = f"g2-{platform}-{index:02d}"
            request_sha = _sha(f"g2-request:{platform}:{index}")
            semantic = _sha(f"g2-semantic:{platform}:{index}")
            rows.append(
                {
                    "row_kind": "g3_interface_replay",
                    "schema_version": "mid-dual-g3-interface-replay/v1",
                    "scale_profile": SCALE_PROFILE,
                    "run_id": "g3-formal-run",
                    "replay_id": f"g3-replay-{platform}-{index:02d}",
                    "platform": platform,
                    "request_id": request_id,
                    "request_sha256": request_sha,
                    "g2_reference_call_ids": [
                        f"{request_id}:repeat:{repeat}" for repeat in range(5)
                    ],
                    "g2_semantic_digest": semantic,
                    "replay_semantic_digest": semantic,
                    "timing_contract_id": "five-phase-sequential-ns/v1",
                    "formal_input_eligible": formal_input_eligible,
                    **_timing_ns(),
                }
            )
    return rows


def _evaluate(
    module: ModuleType,
    *,
    wheel_rows: list[dict[str, object]] | None = None,
    g1_rows: list[dict[str, object]] | None = None,
    interface_rows: list[dict[str, object]] | None = None,
    g2_rows: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    return module.evaluate_g3_evidence(
        frozen_manifest=_frozen_manifest(),
        wheel_rows=_wheel_rows(module) if wheel_rows is None else wheel_rows,
        g1_rows=_g1_rows() if g1_rows is None else g1_rows,
        interface_rows=(
            _interface_rows() if interface_rows is None else interface_rows
        ),
        g2_rows=_g2_reference_rows() if g2_rows is None else g2_rows,
    )


def test_g3_manifest_is_exactly_five_q24_plus_five_unseen24() -> None:
    module = _module()
    selected = module.validate_g3_manifest(_frozen_manifest())
    assert selected == {
        "test_q24": tuple(_scenario_id("test", index) for index in range(5)),
        "unseen24": tuple(_scenario_id("unseen", index) for index in range(5)),
    }

    duplicated = _frozen_manifest()
    duplicated["cohorts"]["g3_test_q5"][4] = duplicated["cohorts"][
        "g3_test_q5"
    ][0]
    result = module.evaluate_g3_evidence(
        frozen_manifest=duplicated,
        wheel_rows=[],
        g1_rows=[],
        interface_rows=[],
        g2_rows=[],
    )
    assert result["status"] == "blocked"
    assert "g3_frozen_manifest" in result["blockers"]


def test_wheel_step_joins_candidate_request_route_feedback_one_to_one() -> None:
    module = _module()
    assert _evaluate(module)["status"] == "passed"

    rows = _wheel_rows(module)
    rows[0]["feedback_route_result_sha256"] = _sha("wrong-route")
    result = _evaluate(module, wheel_rows=rows)
    assert result["status"] == "blocked"
    assert "g3_wheel_hash_chain" in result["blockers"]

    rows = _wheel_rows(module)
    rows[0]["request_id"] = "self-reported-request"
    result = _evaluate(module, wheel_rows=rows)
    assert result["status"] == "blocked"
    assert "g3_request_id" in result["blockers"]


def test_wheel_gate_requires_ten_of_ten_threshold_successes() -> None:
    module = _module()
    rows = _wheel_rows(module, coverage=0.80)
    rows[0]["coverage"] = 0.7999
    _rebind_route_and_feedback(module, rows[0])
    result = _evaluate(
        module,
        wheel_rows=rows,
        g1_rows=_g1_rows(coverage=0.80),
    )
    assert result["status"] == "failed"
    assert result["wheel"]["coverage_80_count"] == 9
    assert result["wheel"]["coverage_all_episodes_passed"] is False
    assert result["wheel"]["coverage_mean_passed"] is False
    assert result["g3_midterm_crosscheck_passed"] is False


def test_wheel_g1_paired_coverage_delta_mean_is_at_least_minus_001() -> None:
    module = _module()
    rows = _wheel_rows(module, coverage=0.98)
    for row in rows:
        row["paired_g1_coverage"] = 0.99
    result = _evaluate(module, wheel_rows=rows)
    assert result["wheel"]["paired_g1_coverage_delta_mean"] == -0.01
    assert result["wheel"]["paired_g1_delta_passed"] is True
    assert result["status"] == "passed"

    rows[0]["coverage"] = 0.9799
    _rebind_route_and_feedback(module, rows[0])
    result = _evaluate(module, wheel_rows=rows)
    assert result["status"] == "failed"
    assert result["wheel"]["paired_g1_coverage_delta_mean"] < -0.01
    assert result["wheel"]["paired_g1_delta_passed"] is False


def test_g3_timing_formulas_cover_midterm_final_and_95_percent_boundary() -> None:
    module = _module()

    nineteen_of_twenty = module._timing_statistics(  # noqa: SLF001
        [900.0] * 19 + [1900.0]
    )
    assert nineteen_of_twenty["mean_ms"] <= 1000.0
    assert nineteen_of_twenty["p95_ms"] <= 1000.0
    assert nineteen_of_twenty["max_ms"] <= 2000.0
    assert nineteen_of_twenty["count_le_1000ms"] == 19
    assert nineteen_of_twenty["at_or_below_1000_fraction"] == 0.95
    assert nineteen_of_twenty["midterm_reduced_passed"] is True
    assert nineteen_of_twenty["final_threshold_reduced_passed"] is True

    eighteen_of_twenty = module._timing_statistics(  # noqa: SLF001
        [800.0] * 18 + [1100.0, 1200.0]
    )
    assert eighteen_of_twenty["mean_ms"] <= 1000.0
    assert eighteen_of_twenty["max_ms"] <= 2000.0
    assert eighteen_of_twenty["at_or_below_1000_fraction"] == 0.90
    assert eighteen_of_twenty["final_threshold_reduced_passed"] is False

    over_absolute_max = module._timing_statistics(  # noqa: SLF001
        [100.0] * 20 + [2000.0001]
    )
    assert over_absolute_max["max_ms"] > 2000.0
    assert over_absolute_max["midterm_reduced_passed"] is False
    assert over_absolute_max["final_threshold_reduced_passed"] is False


def test_every_planner_call_preserves_the_g2_timing_field_contract() -> None:
    module = _module()
    rows = _wheel_rows(module)
    rows[0]["total_ns"] += 1
    result = _evaluate(module, wheel_rows=rows)
    assert result["status"] == "blocked"
    assert "g3_timing_contract" in result["blockers"]

    interface = _interface_rows()
    interface[0]["search_ns"] = -1
    result = _evaluate(module, interface_rows=interface)
    assert result["status"] == "blocked"
    assert "g3_timing_contract" in result["blockers"]


def test_legged_and_hopper_each_have_three_platform_correct_replays() -> None:
    module = _module()
    result = _evaluate(module)
    assert result["interface"]["platform_counts"] == {
        "legged": 3,
        "hopper": 3,
    }
    assert result["interface"]["cross_root_join_passed"] is True
    assert result["interface"]["interface_correctness_passed"] is True
    assert result["interface"]["required_replay_identities_passed"] is True

    interface = _interface_rows()
    interface[0]["request_id"] = interface[1]["request_id"]
    interface[0]["request_sha256"] = interface[1]["request_sha256"]
    interface[0]["g2_reference_call_ids"] = copy.deepcopy(
        interface[1]["g2_reference_call_ids"]
    )
    interface[0]["g2_semantic_digest"] = interface[1][
        "g2_semantic_digest"
    ]
    interface[0]["replay_semantic_digest"] = interface[1][
        "replay_semantic_digest"
    ]
    result = _evaluate(module, interface_rows=interface)
    assert result["status"] == "blocked"
    assert "g3_interface_request_matrix" in result["blockers"]


def test_no_candidate_unreachable_timeout_and_stale_snapshot_fail_closed() -> None:
    module = _module()
    for field, value, blocker in (
        ("planner_success", False, "g3_planner_failure"),
        ("planner_failure_reason", "timeout", "g3_planner_failure"),
        (
            "post_snapshot_parent_sha256",
            _sha("stale"),
            "g3_wheel_hash_chain",
        ),
    ):
        rows = _wheel_rows(module)
        rows[0][field] = value
        if field in {"planner_success", "planner_failure_reason"}:
            _rebind_route_and_feedback(module, rows[0])
        result = _evaluate(module, wheel_rows=rows)
        assert result["status"] == "blocked"
        assert blocker in result["blockers"]


def test_hopper_diagnostic_fixture_cannot_make_g3_formal_pass() -> None:
    module = _module()
    interface = _interface_rows()
    for row in interface:
        if row["platform"] == "hopper":
            row["formal_input_eligible"] = False
    result = _evaluate(module, interface_rows=interface)
    assert result["status"] == "blocked"
    assert "g3_hopper_formal_input_ineligible" in result["blockers"]
    assert result["g3_midterm_crosscheck_passed"] is False
    assert result["g3_final_crosscheck_passed"] is False


def test_g3_config_freezes_exact_scale_counts_thresholds_and_two_phases() -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert _module().validate_g3_config_payload(payload) == payload
    assert payload["schema_version"] == "mid-dual-g3-config/v1"
    assert payload["runner_id"] == "run_xunce_mid_dual_g3_closed_loop/v1"
    assert payload["scale_profile"] == SCALE_PROFILE
    assert payload["output_base"] == "D:/xunce/out/mid_dual/g3"
    assert payload["required_phase_ids"] == ["p01", "p02"]
    assert payload["required_phases"] == [
        "wheel_closed_loop",
        "interface_replay",
    ]
    assert payload["formal_counts"] == {
        "wheel_episodes": 10,
        "test_q24": 5,
        "unseen24": 5,
        "legged_replays": 3,
        "hopper_replays": 3,
    }
    assert payload["checkpoint_sha256"] == UPDATE80_CHECKPOINT_SHA256
    assert payload["policy_state_sha256"] == UPDATE80_POLICY_STATE_SHA256
    assert payload["active_platforms"] == ACTIVE_PLATFORMS
    assert payload["platform_invariants"] == PLATFORM_INVARIANTS
    assert payload["phase_contract"] == {
        "p01": {
            "name": "wheel_closed_loop",
            "row_kind": "g3_wheel_step",
            "episodes": 10,
            "formal_only": True,
        },
        "p02": {
            "name": "interface_replay",
            "row_kind": "g3_interface_replay",
            "platform_counts": {"legged": 3, "hopper": 3},
            "formal_only": True,
            "requires": "p01",
        },
    }
    assert payload["wheel_identity"] == {
        "platform": "wheel",
        "profile": "ppo-standard-wheel-grid/v1",
        "capability_revision": "ppo-path-planner-adapter/v1",
    }


def test_g3_slope_invariant_is_exact_for_every_active_platform() -> None:
    module = _module()
    assert module.validate_platform_invariants(
        ACTIVE_PLATFORMS,
        PLATFORM_INVARIANTS,
    ) == {
        "active_platforms": ACTIVE_PLATFORMS,
        "platform_invariants": PLATFORM_INVARIANTS,
    }

    for platform in ACTIVE_PLATFORMS:
        drifted = copy.deepcopy(PLATFORM_INVARIANTS)
        drifted[platform]["max_traversable_slope_deg"] = 29.999
        with pytest.raises(
            module.G3Blocked,
            match="g3_platform_slope_invariant",
        ):
            module.validate_platform_invariants(ACTIVE_PLATFORMS, drifted)

    missing = copy.deepcopy(PLATFORM_INVARIANTS)
    del missing["hopper"]
    with pytest.raises(
        module.G3Blocked,
        match="g3_platform_slope_invariant",
    ):
        module.validate_platform_invariants(ACTIVE_PLATFORMS, missing)


@pytest.mark.parametrize("status", ("passed", "failed"))
def test_g3_report_is_explicitly_reduced_scale_for_passed_and_failed(
    status: str,
) -> None:
    module = _module()
    recomputed = {
        "status": status,
        "scale_profile": SCALE_PROFILE,
        "active_platforms": ACTIVE_PLATFORMS,
        "platform_invariants": PLATFORM_INVARIANTS,
        "full_scale_acceptance": False,
        "wheel_episode_count": 10,
        "wheel_step_count": 10,
        "interface_replay_count": 6,
        "wheel_coverage_mean": 0.8,
        "wheel_coverage_80_count": 10,
        "paired_g1_coverage_delta_mean": -0.01,
        "wheel_timing": {
            "mean_ms": 900.0,
            "p95_ms": 950.0,
            "max_ms": 1000.0,
            "at_or_below_1000_fraction": 1.0,
        },
        "interface_timing": {
            "mean_ms": 900.0,
            "p95_ms": 950.0,
            "max_ms": 1000.0,
            "at_or_below_1000_fraction": 1.0,
        },
        "wheel_coverage_all_episodes_passed": status == "passed",
        "wheel_coverage_mean_passed": status == "passed",
        "paired_g1_delta_passed": status == "passed",
        "wheel_timing_midterm_passed": status == "passed",
        "wheel_timing_final_passed": status == "passed",
        "interface_timing_midterm_passed": status == "passed",
        "interface_timing_final_passed": status == "passed",
        "interface_correctness_passed": True,
        "required_replay_identities_passed": True,
        "g3_midterm_crosscheck_passed": status == "passed",
        "g3_final_crosscheck_passed": status == "passed",
    }
    summary = module._g3_summary(  # noqa: SLF001
        run_id=f"g3-{status}",
        recomputed=recomputed,
    )
    report = module._render_g3_report(summary)  # noqa: SLF001
    audit = module._g3_report_audit(  # noqa: SLF001
        summary=summary,
        report=report,
    )
    routing = module._g3_routing(summary)  # noqa: SLF001

    assert report.startswith("# G3 缩减规模闭环覆盖与接口回放正式实验报告")
    assert f"- 缩减规模判定状态：`{status}`" in report
    assert f"`{SCALE_PROFILE}`" in report
    assert "10/10 wheel episode" in report
    assert "mean / P95 / max ≤ 2000 ms" in report
    assert "mean / P95 ≤ 1000 ms" in report
    assert "至少 95% ≤ 1000 ms" in report
    assert "3 条 legged + 3 条 Hopper" in report
    assert "不构成全尺度验收" in report
    assert "- 缩减规模结论：" in report
    assert "G1 独立复算数值证据" in report
    for forbidden in (
        "G1 原生正式根",
        "strict_prestart_lineage",
        "Python",
        "Pydantic",
        "Rasterio",
        "DLL",
        "CUDA",
        "worker 启动",
        "pre-start audit passed",
    ):
        assert forbidden not in report
    assert summary["scale_profile"] == SCALE_PROFILE
    assert summary["full_scale_acceptance"] is False
    assert summary["active_platforms"] == ACTIVE_PLATFORMS
    assert summary["platform_invariants"] == PLATFORM_INVARIANTS
    assert audit["scale_profile"] == SCALE_PROFILE
    assert audit["full_scale_acceptance"] is False
    assert audit["active_platforms"] == ACTIVE_PLATFORMS
    assert audit["platform_invariants"] == PLATFORM_INVARIANTS
    assert routing["scale_profile"] == SCALE_PROFILE
    assert routing["full_scale_acceptance"] is False
    assert routing["active_platforms"] == ACTIVE_PLATFORMS
    assert routing["platform_invariants"] == PLATFORM_INVARIANTS

    drifted = copy.deepcopy(recomputed)
    drifted["platform_invariants"]["hopper"][
        "max_traversable_slope_deg"
    ] = 30.1
    with pytest.raises(
        module.G3Blocked,
        match="g3_platform_slope_invariant",
    ):
        module._g3_summary(  # noqa: SLF001
            run_id=f"g3-{status}-drifted",
            recomputed=drifted,
        )


def test_g3_production_uses_exact_ten_without_standard_schedule_padding() -> None:
    module = _module()
    source = inspect.getsource(module.execute_update80_wheel_rows)
    assert "run_standard_evaluation_jobs" not in source
    assert "build_standard_evaluation_schedule" not in source
    assert 'for split in ("test_q24", "unseen24")' in source
    assert "selected = validate_g3_manifest" in source
    assert "g1 = _g1_index" in source


def test_g3_runner_uses_artifact_io_and_missing_preflight_never_completes() -> None:
    module = _module()
    source = SCRIPT.read_text(encoding="utf-8")
    for forbidden in (
        ".read_text(",
        ".write_text(",
        ".read_bytes(",
        ".write_bytes(",
        ".open(",
        ".mkdir(",
        ".exists(",
        ".is_file(",
    ):
        assert forbidden not in source

    stream = io.StringIO()
    with redirect_stdout(stream):
        return_code = module._main(
            [
                "--config",
                str(CONFIG),
                "--run-id",
                "g3-implementation-preflight",
                "--mode",
                "preflight",
            ]
        )
    result = json.loads(stream.getvalue())
    assert return_code == 1
    assert result["execution_status"] == "not_started"
    assert result["gate_status"] == "blocked"
    assert result["formal_evidence_eligible"] is False
    assert set(result["blockers"]) == {
        "g3_frozen_bundle_root_missing",
        "g3_g1_root_missing",
        "g3_g2_root_missing",
    }


def test_authorized_stopped_g1_root_uses_only_verified_assessment_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    verified, frozen, expected_rows = _existing_run_assessment_fixture(
        numerical_result_status="failed"
    )
    calls: list[tuple[Path, Path]] = []

    class _AssessmentBlocked(RuntimeError):
        def __init__(self, reason: str) -> None:
            self.reason = reason
            super().__init__(reason)

    def _verify(*, envelope_root: Path, expected_source_root: Path):
        calls.append((envelope_root, expected_source_root))
        return verified

    assessment_module = SimpleNamespace(
        AUTHORIZED_SOURCE_ROOT=AUTHORIZED_G1_ROOT,
        ExistingRunAssessmentBlocked=_AssessmentBlocked,
        verify_existing_run_assessment=_verify,
    )
    monkeypatch.setattr(
        module,
        "_import_local_script",
        lambda name: (
            assessment_module
            if name == "xunce_mid_dual_g1_existing_run_assessment"
            else pytest.fail(f"unexpected import: {name}")
        ),
    )
    monkeypatch.setattr(
        module,
        "load_verified_g1_root",
        lambda *args, **kwargs: pytest.fail("native fallback is forbidden"),
    )

    source = module.load_verified_g1_source(
        root=Path(AUTHORIZED_G1_ROOT),
        frozen_bundle=frozen,
        g1_assessment_envelope=Path(AUTHORIZED_G1_ENVELOPE),
    )

    assert calls == [
        (Path(AUTHORIZED_G1_ENVELOPE), Path(AUTHORIZED_G1_ROOT))
    ]
    assert source["g1_evidence_kind"] == "existing_run_assessment_v1"
    assert source["manifest_sha256"] == verified.envelope_manifest_sha256
    assert source["coverage_rows"] == expected_rows
    assert source["assessment_binding"] == {
        "g1_evidence_kind": "existing_run_assessment_v1",
        "g1_source_manifest_sha256": verified.envelope_manifest_sha256,
        "assessment_envelope_sha256": verified.envelope_sha256,
        "assessment_manifest_sha256": verified.envelope_manifest_sha256,
        "phase_snapshot_sha256": verified.phase_snapshot_sha256,
        "stop_evidence_sha256": verified.stop_evidence_sha256,
        "review_manifest_sha256": verified.review_manifest_sha256,
        "review_result_sha256": verified.review_result_sha256,
        "assessment_use_status": "authorized_existing_run",
        "strict_prestart_lineage_status": "not_satisfied",
        "lineage_limitation_acknowledged": True,
        "numerical_gate_status": "recomputed_from_raw_rows",
        "numerical_result_status": "failed",
    }
    assert "native_summary" not in source


def test_g1_assessment_scope_dispatch_never_falls_back_to_native(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    verified, frozen, _ = _existing_run_assessment_fixture()
    verifier_calls = 0
    native_calls = 0

    def _verify(**kwargs):
        nonlocal verifier_calls
        verifier_calls += 1
        return verified

    assessment_module = SimpleNamespace(
        AUTHORIZED_SOURCE_ROOT=AUTHORIZED_G1_ROOT,
        verify_existing_run_assessment=_verify,
    )
    monkeypatch.setattr(
        module,
        "_import_local_script",
        lambda name: assessment_module,
    )

    def _native(*args, **kwargs):
        nonlocal native_calls
        native_calls += 1
        return {"route": "native"}

    monkeypatch.setattr(module, "load_verified_g1_root", _native)

    with pytest.raises(
        module.G3Blocked,
        match="g3_g1_existing_run_assessment_required",
    ):
        module.load_verified_g1_source(
            root=Path(AUTHORIZED_G1_ROOT),
            frozen_bundle=frozen,
            g1_assessment_envelope=None,
        )
    assert verifier_calls == 0
    assert native_calls == 0

    with pytest.raises(
        module.G3Blocked,
        match="g3_g1_existing_run_assessment_scope_mismatch",
    ):
        module.load_verified_g1_source(
            root=Path("D:/xunce/out/mid_dual/g1/future-formal-run"),
            frozen_bundle=frozen,
            g1_assessment_envelope=Path(AUTHORIZED_G1_ENVELOPE),
        )
    assert verifier_calls == 0
    assert native_calls == 0

    assert module.load_verified_g1_source(
        root=Path("D:/xunce/out/mid_dual/g1/future-formal-run"),
        frozen_bundle=frozen,
        g1_assessment_envelope=None,
    ) == {"route": "native"}
    assert native_calls == 1


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("assessment_use_status", "unauthorized"),
        ("strict_prestart_lineage_status", "satisfied"),
        ("lineage_limitation_acknowledged", False),
        ("numerical_gate_status", "trusted_stored_summary"),
        ("phase_snapshot_sha256", "not-a-sha"),
        ("review_manifest_sha256", "not-a-sha"),
    ),
)
def test_g1_assessment_status_or_hash_drift_blocks(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
) -> None:
    module = _module()
    verified, frozen, _ = _existing_run_assessment_fixture()
    setattr(verified, field, value)
    assessment_module = SimpleNamespace(
        AUTHORIZED_SOURCE_ROOT=AUTHORIZED_G1_ROOT,
        verify_existing_run_assessment=lambda **kwargs: verified,
    )
    monkeypatch.setattr(
        module,
        "_import_local_script",
        lambda name: assessment_module,
    )

    with pytest.raises(
        module.G3Blocked,
        match="g3_g1_existing_run_assessment_invalid",
    ):
        module.load_verified_g1_source(
            root=Path(AUTHORIZED_G1_ROOT),
            frozen_bundle=frozen,
            g1_assessment_envelope=Path(AUTHORIZED_G1_ENVELOPE),
        )


def test_g3_cli_passes_explicit_g1_assessment_envelope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    captured: dict[str, object] = {}

    def _run_g3(**kwargs):
        captured.update(kwargs)
        return {
            "execution_status": "not_started",
            "gate_status": "preflight_passed",
            "formal_evidence_eligible": False,
            "blockers": [],
        }

    monkeypatch.setattr(module, "run_g3", _run_g3)
    stream = io.StringIO()
    with redirect_stdout(stream):
        return_code = module._main(
            [
                "--config",
                str(CONFIG),
                "--frozen-bundle-root",
                "D:/xunce/inputs/mid_dual/scenarios/freeze",
                "--g1-root",
                AUTHORIZED_G1_ROOT,
                "--g1-assessment-envelope",
                AUTHORIZED_G1_ENVELOPE,
                "--g2-root",
                "D:/xunce/out/mid_dual/g2/g2-formal",
                "--run-id",
                "g3-assessment-preflight",
                "--mode",
                "preflight",
            ]
        )
    assert return_code == 0
    assert captured["g1_assessment_envelope"] == AUTHORIZED_G1_ENVELOPE


def test_g2_r6_runtime_source_closure_and_platform_audits_are_exact() -> None:
    module = _module()
    fixture = _g2_r6_runtime_fixture()

    validated = module.validate_g2_r6_runtime_evidence(
        config=fixture["config"],
        input_audit=fixture["input_audit"],
        approval=fixture["approval"],
        hopper_resolution=fixture["hopper_resolution"],
        phase_audits=fixture["phase_audits"],
        summary=fixture["summary"],
        runtime_closure_audit=fixture["closure"],
        platform_invariants_audit=fixture["platform_audit"],
        local_source_sha256=fixture["local_source_sha256"],
    )

    assert validated["runtime_source_closure"] == fixture["closure"]
    assert (
        validated["path_planner_runtime_source_closure_sha256"]
        == fixture["closure_sha256"]
    )
    assert validated["platform_invariants"] == PLATFORM_INVARIANTS


@pytest.mark.parametrize(
    ("target", "mutation", "reason"),
    (
        (
            "config",
            lambda payload: payload["evidence_binding"].__setitem__(
                "runtime_source_closure_audit_path",
                "wrong.json",
            ),
            "g3_g2_config_invalid",
        ),
        (
            "input_audit",
            lambda payload: payload.__setitem__(
                "path_planner_runtime_source_closure_sha256",
                _sha("drifted-closure"),
            ),
            "g3_g2_runtime_source_closure_invalid",
        ),
        (
            "approval",
            lambda payload: payload.__setitem__(
                "path_planner_runtime_source_closure",
                {},
            ),
            "g3_g2_runtime_source_closure_invalid",
        ),
        (
            "hopper_resolution",
            lambda payload: payload.__setitem__(
                "path_planner_runtime_source_closure_sha256",
                _sha("drifted-hopper-closure"),
            ),
            "g3_g2_runtime_source_closure_invalid",
        ),
        (
            "platform_audit",
            lambda payload: payload["platform_invariants"]["hopper"].__setitem__(
                "max_traversable_slope_deg",
                30.1,
            ),
            "g3_platform_slope_invariant",
        ),
    ),
)
def test_g2_r6_runtime_source_evidence_drift_blocks(
    target: str,
    mutation,
    reason: str,
) -> None:
    module = _module()
    fixture = _g2_r6_runtime_fixture()
    mutation(fixture[target])

    with pytest.raises(module.G3Blocked, match=reason):
        module.validate_g2_r6_runtime_evidence(
            config=fixture["config"],
            input_audit=fixture["input_audit"],
            approval=fixture["approval"],
            hopper_resolution=fixture["hopper_resolution"],
            phase_audits=fixture["phase_audits"],
            summary=fixture["summary"],
            runtime_closure_audit=fixture["closure"],
            platform_invariants_audit=fixture["platform_audit"],
            local_source_sha256=fixture["local_source_sha256"],
        )


def test_timing_diagnostics_have_five_nonoverlapping_ns_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lunar_exploration_ppo.integrations import (
        path_planner_adapter as adapter_module,
    )
    from lunar_exploration_ppo.utils.geometry import CellXY, GridGeometry

    ticks = iter((0, 2, 5, 10, 17, 28))
    monkeypatch.setattr(adapter_module, "perf_counter_ns", lambda: next(ticks))

    result = adapter_module.PathPlannerAdapter(
        GridGeometry(3, 3, 0.5)
    ).validate(
        np.ones((3, 3), dtype=bool),
        CellXY(0, 0),
        CellXY(2, 2),
        0.5,
    )

    timing_names = (
        "input_validation_ns",
        "platform_instantiation_ns",
        "search_ns",
        "complete_route_validation_ns",
        "result_assembly_ns",
    )
    assert result.valid is True
    assert {name: result.diagnostics[name] for name in timing_names} == {
        "input_validation_ns": 2,
        "platform_instantiation_ns": 3,
        "search_ns": 5,
        "complete_route_validation_ns": 7,
        "result_assembly_ns": 11,
    }
    assert result.diagnostics["run_start_ns"] == 0
    assert result.diagnostics["final_end_ns"] == 28
    assert (
        result.diagnostics["total_ns"]
        == result.diagnostics["final_end_ns"]
        - result.diagnostics["run_start_ns"]
        == 28
    )


def test_timing_diagnostics_sum_to_total_ns() -> None:
    from lunar_exploration_ppo.integrations.path_planner_adapter import (
        PathPlannerAdapter,
    )
    from lunar_exploration_ppo.utils.geometry import CellXY, GridGeometry

    result = PathPlannerAdapter(GridGeometry(3, 3, 0.5)).validate(
        np.ones((3, 3), dtype=bool),
        CellXY(0, 0),
        CellXY(2, 2),
        0.5,
    )
    values = tuple(
        result.diagnostics[name]
        for name in (
            "input_validation_ns",
            "platform_instantiation_ns",
            "search_ns",
            "complete_route_validation_ns",
            "result_assembly_ns",
        )
    )
    assert all(type(value) is int and value >= 0 for value in values)
    assert result.diagnostics["total_ns"] == sum(values)
    assert result.diagnostics["total_ns"] == (
        result.diagnostics["final_end_ns"]
        - result.diagnostics["run_start_ns"]
    )


def test_timing_does_not_change_route_cells_length_theta_or_failure_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lunar_exploration_ppo.integrations import (
        path_planner_adapter as adapter_module,
    )
    from lunar_exploration_ppo.utils.geometry import CellXY, GridGeometry

    adapter = adapter_module.PathPlannerAdapter(GridGeometry(3, 3, 0.5))
    mask = np.ones((3, 3), dtype=bool)
    baseline = adapter.validate(mask, CellXY(0, 0), CellXY(2, 2), 0.5)
    ticks = iter((100, 103, 108, 115, 126, 139))
    monkeypatch.setattr(adapter_module, "perf_counter_ns", lambda: next(ticks))

    timed = adapter.validate(mask, CellXY(0, 0), CellXY(2, 2), 0.5)
    assert (
        timed.valid,
        timed.failure_reason,
        timed.failure_classification,
        timed.path_cells,
        timed.path_world,
        timed.target_theta,
        timed.path_length_m,
    ) == (
        baseline.valid,
        baseline.failure_reason,
        baseline.failure_classification,
        baseline.path_cells,
        baseline.path_world,
        baseline.target_theta,
        baseline.path_length_m,
    )
    assert timed.diagnostics["run_start_ns"] == 100
    assert timed.diagnostics["final_end_ns"] == 139
    assert (
        timed.diagnostics["total_ns"]
        == timed.diagnostics["final_end_ns"]
        - timed.diagnostics["run_start_ns"]
        == 39
    )


def test_early_failure_records_zero_for_unentered_phases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lunar_exploration_ppo.integrations import (
        path_planner_adapter as adapter_module,
    )
    from lunar_exploration_ppo.utils.geometry import CellXY, GridGeometry

    ticks = iter((0, 5, 12))
    monkeypatch.setattr(adapter_module, "perf_counter_ns", lambda: next(ticks))

    result = adapter_module.PathPlannerAdapter(
        GridGeometry(3, 3, 0.5)
    ).validate(
        np.ones((3, 3), dtype=bool),
        CellXY(0, 0),
        CellXY(9, 9),
        0.5,
    )
    assert result.valid is False
    assert result.failure_reason == "target_out_of_bounds"
    assert result.diagnostics["input_validation_ns"] == 5
    assert result.diagnostics["platform_instantiation_ns"] == 0
    assert result.diagnostics["search_ns"] == 0
    assert result.diagnostics["complete_route_validation_ns"] == 0
    assert result.diagnostics["result_assembly_ns"] == 7
    assert result.diagnostics["run_start_ns"] == 0
    assert result.diagnostics["final_end_ns"] == 12
    assert (
        result.diagnostics["total_ns"]
        == result.diagnostics["final_end_ns"]
        - result.diagnostics["run_start_ns"]
        == 12
    )


@pytest.mark.parametrize(
    ("case", "expected_reason"),
    (
        ("invalid_theta", "invalid_theta"),
        ("invalid_observed_safe_mask", "invalid_safe_mask"),
        ("invalid_planning_safe_mask", "invalid_safe_mask"),
        ("start_out_of_bounds", "start_out_of_bounds"),
        ("target_out_of_bounds", "target_out_of_bounds"),
        ("start_physical_unsafe", "start_physical_unsafe"),
        ("start_unknown_buffer_unsafe", "start_unknown_buffer_unsafe"),
        ("endpoint_physical_unsafe", "endpoint_physical_unsafe"),
        ("endpoint_unknown_buffer_unsafe", "endpoint_unknown_buffer_unsafe"),
        ("planner_no_path", "planner_no_path"),
    ),
)
def test_every_input_failure_uses_one_continuous_interval_and_preserves_semantics(
    monkeypatch: pytest.MonkeyPatch,
    case: str,
    expected_reason: str,
) -> None:
    from lunar_exploration_ppo.integrations import (
        path_planner_adapter as adapter_module,
    )
    from lunar_exploration_ppo.utils.geometry import CellXY, GridGeometry

    geometry = GridGeometry(3, 3, 0.5)
    observed_safe = np.ones(geometry.shape, dtype=bool)
    planning_safe = observed_safe.copy()
    start = CellXY(0, 1)
    target = CellXY(2, 1)
    theta = 0.5
    if case == "invalid_theta":
        theta = float("nan")
    elif case == "invalid_observed_safe_mask":
        observed_safe = np.ones((2, 2), dtype=bool)
    elif case == "invalid_planning_safe_mask":
        planning_safe = np.ones((2, 2), dtype=bool)
    elif case == "start_out_of_bounds":
        start = CellXY(9, 1)
    elif case == "target_out_of_bounds":
        target = CellXY(9, 1)
    elif case == "start_physical_unsafe":
        observed_safe[start.y, start.x] = False
    elif case == "start_unknown_buffer_unsafe":
        planning_safe[start.y, start.x] = False
    elif case == "endpoint_physical_unsafe":
        observed_safe[target.y, target.x] = False
    elif case == "endpoint_unknown_buffer_unsafe":
        planning_safe[target.y, target.x] = False
    elif case == "planner_no_path":
        planning_safe[:, 1] = False
    else:  # pragma: no cover - the parameter table is exhaustive
        raise AssertionError(case)

    adapter = adapter_module.PathPlannerAdapter(geometry)
    baseline = adapter.validate(
        observed_safe,
        planning_safe,
        start,
        target,
        theta,
    )
    ticks = iter((0, 5, 12))
    monkeypatch.setattr(adapter_module, "perf_counter_ns", lambda: next(ticks))
    timed = adapter.validate(
        observed_safe,
        planning_safe,
        start,
        target,
        theta,
    )

    assert timed.failure_reason == expected_reason
    assert (
        timed.valid,
        timed.failure_reason,
        timed.failure_classification,
        timed.path_cells,
        timed.path_world,
        timed.target_theta,
        timed.path_length_m,
    ) == (
        baseline.valid,
        baseline.failure_reason,
        baseline.failure_classification,
        baseline.path_cells,
        baseline.path_world,
        baseline.target_theta,
        baseline.path_length_m,
    )
    assert timed.diagnostics["input_validation_ns"] == 5
    assert timed.diagnostics["platform_instantiation_ns"] == 0
    assert timed.diagnostics["search_ns"] == 0
    assert timed.diagnostics["complete_route_validation_ns"] == 0
    assert timed.diagnostics["result_assembly_ns"] == 7
    assert timed.diagnostics["run_start_ns"] == 0
    assert timed.diagnostics["final_end_ns"] == 12
    assert (
        timed.diagnostics["input_validation_ns"]
        + timed.diagnostics["result_assembly_ns"]
        == timed.diagnostics["final_end_ns"]
        - timed.diagnostics["run_start_ns"]
        == timed.diagnostics["total_ns"]
        == 12
    )


def test_planner_no_path_failure_keeps_route_zero_and_continuous_assembly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lunar_exploration_ppo.integrations import (
        path_planner_adapter as adapter_module,
    )
    from lunar_exploration_ppo.utils.geometry import CellXY, GridGeometry

    class _NoPathResult:
        success = False
        failure_reason = None
        expanded_count = 4
        path_cells = ()

    class _NoPathPlanner:
        def plan(self, grid, request):
            del grid, request
            return _NoPathResult()

    monkeypatch.setattr(adapter_module, "AStarPlanner", _NoPathPlanner)
    geometry = GridGeometry(3, 3, 0.5)
    observed_safe = np.ones(geometry.shape, dtype=bool)
    planning_safe = observed_safe.copy()
    adapter = adapter_module.PathPlannerAdapter(geometry)
    baseline = adapter.validate(
        observed_safe,
        planning_safe,
        CellXY(0, 1),
        CellXY(2, 1),
        0.5,
    )
    ticks = iter((0, 2, 5, 10, 17))
    monkeypatch.setattr(adapter_module, "perf_counter_ns", lambda: next(ticks))
    timed = adapter.validate(
        observed_safe,
        planning_safe,
        CellXY(0, 1),
        CellXY(2, 1),
        0.5,
    )

    assert timed.failure_reason == baseline.failure_reason == "planner_no_path"
    assert timed.diagnostics["complete_route_validation_ns"] == 0
    assert timed.diagnostics["result_assembly_ns"] == 7
    assert timed.diagnostics["run_start_ns"] == 0
    assert timed.diagnostics["final_end_ns"] == 17
    assert timed.diagnostics["total_ns"] == 17


@pytest.mark.parametrize(
    ("case", "expected_reason"),
    (
        ("physical", "path_physical_unsafe"),
        ("unknown_buffer", "path_unknown_buffer_unsafe"),
        ("out_of_bounds", "path_out_of_bounds"),
    ),
)
def test_route_failure_keeps_reason_and_assembly_adjacent_to_route_end(
    monkeypatch: pytest.MonkeyPatch,
    case: str,
    expected_reason: str,
) -> None:
    from lunar_exploration_ppo.integrations import (
        path_planner_adapter as adapter_module,
    )
    from lunar_exploration_ppo.utils.geometry import CellXY, GridGeometry

    middle = adapter_module.Cell(1, 1)
    if case == "out_of_bounds":
        middle = adapter_module.Cell(9, 9)

    class _UnsafePathResult:
        success = True
        failure_reason = None
        expanded_count = 3
        path_cells = (
            adapter_module.Cell(0, 1),
            middle,
            adapter_module.Cell(2, 1),
        )

    class _UnsafePathPlanner:
        def plan(self, grid, request):
            del grid, request
            return _UnsafePathResult()

    monkeypatch.setattr(adapter_module, "AStarPlanner", _UnsafePathPlanner)
    geometry = GridGeometry(3, 3, 0.5)
    observed_safe = np.ones(geometry.shape, dtype=bool)
    planning_safe = observed_safe.copy()
    if case == "physical":
        observed_safe[1, 1] = False
    elif case == "unknown_buffer":
        planning_safe[1, 1] = False
    adapter = adapter_module.PathPlannerAdapter(geometry)
    baseline = adapter.validate(
        observed_safe,
        planning_safe,
        CellXY(0, 1),
        CellXY(2, 1),
        0.5,
    )
    ticks = iter((0, 2, 5, 10, 17, 28))
    monkeypatch.setattr(adapter_module, "perf_counter_ns", lambda: next(ticks))
    timed = adapter.validate(
        observed_safe,
        planning_safe,
        CellXY(0, 1),
        CellXY(2, 1),
        0.5,
    )

    assert timed.failure_reason == baseline.failure_reason == expected_reason
    assert timed.diagnostics["complete_route_validation_ns"] == 7
    assert timed.diagnostics["result_assembly_ns"] == 11
    assert timed.diagnostics["run_start_ns"] == 0
    assert timed.diagnostics["final_end_ns"] == 28
    assert timed.diagnostics["total_ns"] == 28


@pytest.mark.parametrize(
    "run_id",
    (
        "",
        ".",
        "..",
        "../escape",
        r"..\escape",
        "CON",
        "g3-run.",
        "g3-run ",
        "nested/run",
    ),
)
def test_g3_rejects_unsafe_run_id_before_creating_output(
    run_id: str,
) -> None:
    module = _module()
    with pytest.raises(module.G3Blocked, match="g3_run_id_invalid"):
        module.validate_g3_run_id(run_id)


def test_g3_strict_root_contract_requires_single_child_absolute_d_roots() -> None:
    module = _module()
    validated = module.validate_g3_root_contract(
        frozen_bundle_root=(
            "D:/xunce/inputs/mid_dual/scenarios/freeze-0123456789abcdef"
        ),
        g1_root="D:/xunce/out/mid_dual/g1/g1-formal",
        g2_root="D:/xunce/out/mid_dual/g2/g2-formal",
        run_id="g3-formal",
        require_existing=False,
    )
    assert validated["output_root"].as_posix().casefold() == (
        "d:/xunce/out/mid_dual/g3/g3-formal"
    )

    invalid = (
        {
            "frozen_bundle_root": "relative/freeze",
            "g1_root": "D:/xunce/out/mid_dual/g1/nested/g1",
            "g2_root": "D:/xunce/out/mid_dual/g2/g2-formal",
        },
        {
            "frozen_bundle_root": (
                "D:/xunce/inputs/mid_dual/scenarios/freeze"
            ),
            "g1_root": "C:/unsafe/g1",
            "g2_root": "D:/xunce/out/mid_dual/g2/g2-formal",
        },
        {
            "frozen_bundle_root": (
                "D:/xunce/inputs/mid_dual/scenarios/freeze"
            ),
            "g1_root": "D:/xunce/out/mid_dual/g1/g1-formal",
            "g2_root": "D:/xunce/out/mid_dual/g1/g1-formal",
        },
    )
    for roots in invalid:
        with pytest.raises(module.G3Blocked):
            module.validate_g3_root_contract(
                **roots,
                run_id="g3-formal",
                require_existing=False,
            )


def test_g3_effective_config_has_task10_wrapper_and_strong_upstream_binding() -> None:
    module = _module()
    base = json.loads(CONFIG.read_text(encoding="utf-8"))
    g1_manifest = _sha("g1-manifest")
    g2_manifest = _sha("g2-manifest")
    g1_evidence_binding = _native_g1_evidence_binding(g1_manifest)
    input_audit = {
        "schema_version": "xunce-mid-dual-g3-input-audit/v1",
        "gate_id": "g3",
        "run_id": "g3-formal",
        "scale_profile": SCALE_PROFILE,
        "formal_evidence_eligible": True,
        "active_platforms": ACTIVE_PLATFORMS,
        "platform_invariants": PLATFORM_INVARIANTS,
        "g1_source_manifest_sha256": g1_manifest,
        "g2_source_manifest_sha256": g2_manifest,
        "g1_evidence_binding": g1_evidence_binding,
        "wheel_selections": [],
        "interface_selections": [],
    }
    lineage = {
        "schema_version": "xunce-mid-dual-code/v1",
        "required_sources": [],
        "code_sha256": _sha("g3-code"),
    }
    upstream = {
        "g1_source_manifest_sha256": g1_manifest,
        "g2_source_manifest_sha256": g2_manifest,
        "freeze_manifest_sha256": _sha("freeze-manifest"),
        "g1_config_sha256": _sha("g1-config"),
        "g1_input_sha256": _sha("g1-input"),
        "g1_code_sha256": _sha("g1-code"),
        "g2_config_sha256": _sha("g2-config"),
        "g2_input_sha256": _sha("g2-input"),
        "g2_code_sha256": _sha("g2-code"),
        "g2_input_set_id": "g2-input-set",
        "g2_approval_sha256": _sha("g2-approval"),
        "g2_cohort_sha256": _sha("g2-cohort"),
        "g2_provider_identity_sha256": _sha("provider-identity"),
        "g2_oracle_identity_sha256": _sha("oracle-identity"),
        "g2_hopper_resolution_sha256": _sha("hopper-resolution"),
        "active_platforms": ACTIVE_PLATFORMS,
        "platform_invariants": PLATFORM_INVARIANTS,
        "g1_evidence_binding": g1_evidence_binding,
    }
    effective = module.build_g3_effective_config(
        base_config=base,
        run_id="g3-formal",
        input_audit=input_audit,
        code_lineage=lineage,
        upstream_binding=upstream,
    )
    assert set(effective) == {
        "schema_version",
        "gate_id",
        "runner_id",
        "run_id",
        "output_root",
        "scale_profile",
        "input_sha256",
        "code_sha256",
        "required_phase_ids",
        "source_contract_sha256",
        "evidence_binding",
        "upstream_binding",
        "active_platforms",
        "platform_invariants",
        "g1_evidence_binding",
    }
    assert effective["evidence_binding"] == {
        "schema_version": "xunce-mid-dual-g3-evidence-binding/v1",
        "input_audit_path": "g3_input_audit.json",
        "lineage_audit_path": "lineage_audit.json",
        "report_audit_path": "g3_report_audit.json",
        "platform_invariants_audit_path": (
            "platform_invariants_audit.json"
        ),
    }
    assert effective["upstream_binding"] == upstream
    assert effective["g1_evidence_binding"] == g1_evidence_binding
    assert effective["active_platforms"] == ACTIVE_PLATFORMS
    assert effective["platform_invariants"] == PLATFORM_INVARIANTS

    drifted_input = copy.deepcopy(input_audit)
    drifted_input["platform_invariants"]["wheel"][
        "max_traversable_slope_deg"
    ] = 29.0
    with pytest.raises(
        module.G3Blocked,
        match="g3_platform_slope_invariant",
    ):
        module.build_g3_effective_config(
            base_config=base,
            run_id="g3-formal",
            input_audit=drifted_input,
            code_lineage=lineage,
            upstream_binding=upstream,
        )

    drifted_upstream = copy.deepcopy(upstream)
    drifted_upstream["platform_invariants"]["legged"][
        "max_traversable_slope_deg"
    ] = 31.0
    with pytest.raises(
        module.G3Blocked,
        match="g3_platform_slope_invariant",
    ):
        module.build_g3_effective_config(
            base_config=base,
            run_id="g3-formal",
            input_audit=input_audit,
            code_lineage=lineage,
            upstream_binding=drifted_upstream,
        )


def test_g1_native_effective_config_does_not_require_generic_evidence_wrapper() -> None:
    module = _module()
    native = {
        "schema_version": "xunce-mid-dual-g1-effective-config/v1",
        "gate_id": "g1",
        "runner_id": "run_xunce_mid_dual_g1_coverage/v1",
        "scale_profile": SCALE_PROFILE,
        "run_id": "g1-formal",
        "mode": "formal",
        "output_root": "D:/xunce/out/mid_dual/g1/g1-formal",
        "required_phase_ids": [
            "p01",
            "p02",
            "p03",
            "p04",
            "p05",
            "p06",
            "p07",
        ],
        "required_phases": [
            "preflight",
            "validation_dry_run",
            "test_q24",
            "unseen24",
            "replay3",
            "recompute",
            "finalize",
        ],
        "input_audit": {
            "path": "g1_input_audit.json",
            "schema_version": "xunce-mid-dual-g1-input-audit/v1",
            "sha256": _sha("g1-input"),
        },
        "input_sha256": _sha("g1-input"),
        "code_sha256": _sha("g1-code"),
        "checkpoint": {
            "update": 80,
            "sha256": UPDATE80_CHECKPOINT_SHA256,
            "policy_state_sha256": UPDATE80_POLICY_STATE_SHA256,
        },
        "scenario_manifest": {
            "path": "D:/xunce/inputs/mid_dual/scenarios/freeze/manifest.json",
            "schema_version": "mid-dual-scenario-freeze/v1",
            "sha256": _sha("freeze-manifest"),
        },
        "denominator": {},
        "execution": {},
        "bootstrap": {},
        "schemas": {},
        "base_config_sha256": _sha("g1-base-config"),
        "source_lineage": {},
        "repair_lineage": None,
        "config_sha256": _sha("g1-effective-config"),
    }
    validated = module.validate_g1_native_effective_config_projection(
        native,
        expected_root=Path(
            "D:/xunce/out/mid_dual/g1/g1-formal"
        ),
    )
    assert "evidence_binding" not in validated
    assert validated["required_phase_ids"][-1] == "p07"


def test_g3_phase_semantics_reject_fixture_wrong_kind_and_p02_before_p01() -> None:
    module = _module()
    wheel, _, _ = module.materialize_wheel_authority(
        _wheel_rows(module),
        config_sha256=_sha("g3-config"),
        input_sha256=_sha("g3-input"),
        code_sha256=_sha("g3-code"),
        frozen_manifest_sha256=_sha("freeze-manifest"),
        g1_source_manifest_sha256=_sha("g1-manifest"),
        g2_source_manifest_sha256=_sha("g2-manifest"),
    )
    p01_projection = module.validate_g3_phase_rows(
        phase_id="p01",
        rows=wheel,
        accepted_phase_ids=(),
    )
    assert p01_projection["active_platforms"] == ACTIVE_PLATFORMS
    assert p01_projection["platform_invariants"] == PLATFORM_INVARIANTS
    with pytest.raises(module.G3Blocked, match="g3_phase_requires_p01"):
        module.validate_g3_phase_rows(
            phase_id="p02",
            rows=_interface_rows(),
            accepted_phase_ids=(),
        )

    fixture = copy.deepcopy(wheel)
    fixture[0]["execution_class"] = "failure_fixture"
    with pytest.raises(module.G3Blocked, match="g3_formal_fixture_partition"):
        module.validate_g3_phase_rows(
            phase_id="p01",
            rows=fixture,
            accepted_phase_ids=(),
        )

    wrong_kind = copy.deepcopy(wheel)
    wrong_kind[0]["row_kind"] = "g3_interface_replay"
    with pytest.raises(module.G3Blocked, match="g3_phase_row_kind"):
        module.validate_g3_phase_rows(
            phase_id="p01",
            rows=wrong_kind,
            accepted_phase_ids=(),
        )
