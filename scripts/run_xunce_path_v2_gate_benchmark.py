from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import path_v2_gate_artifacts as gate_artifacts
import run_xunce_path_v2_g0_baseline_and_isolation as gate0
import xunce_artifact_io as artifact_io


SCHEMA_VERSION = "xunce-path-v2-gate1-contract/v1"
STAGE_ID = "xunce-path-v2-gate1-contract"
FORMAL_PYTHON = Path("D:/conda_envs/lunar-explorer/python.exe")
FORMAL_TEMP_ROOT = Path("D:/xunce/tmp/path_v2_g1")
EXPECTED_BRANCH = "codex/multiplatform-path-planner-v2"
ORIGINAL_BASE_COMMIT = "b635740ee021258ef31811ec87c60add839fc5f9"
GATE_INPUT_COMMIT = "b2a36d31f3802eb5a37fcfcf594f74a499aa719b"
EXPECTED_PYTHON_VERSION = "3.12.13"
LEGACY_EXPECTED = {
    "passed": 156,
    "skipped": 17,
    "failures": 0,
    "errors": 0,
}
ALLOWED_SKIP_DEPENDENCY = "pydrake"
PATH_PLANNER_WORKING_DIRECTORY = "path-planner"
PATH_PLANNER_PYTHONPATH = ["path-planner/src"]
PASS_ROUTE = "implement_path_v2_wheel_provider"
FOCUSED_TARGETS = (
    "tests/test_v2_contracts.py",
    "tests/test_v2_serialization.py",
    "tests/test_v2_terrain.py",
    "tests/test_v2_fine_safety_anchor.py",
    "tests/test_v2_profiles.py",
    "tests/test_v2_api.py",
)
CANONICAL_ARTIFACT_NAMES = frozenset(
    {
        "config.json",
        "summary.json",
        "routing.json",
        "results.jsonl",
        "phase-state.jsonl",
        "review.json",
        "report.md",
        "manifest.json",
    }
)

GATE2_SCHEMA_VERSION = "xunce-path-v2-gate2-wheel/v1"
GATE2_STAGE_ID = "xunce-path-v2-gate2-wheel"
GATE2_INPUT_COMMIT = "0cc3eb9728a77473dd436dc2b05e2df009a296c7"
GATE2_FORMAL_TEMP_ROOT = Path("D:/xunce/tmp/path_v2_g2")
GATE2_PASS_ROUTE = "implement_path_v2_lazy_validation_and_cache"
GATE2_EXECUTE_ROUTE = "execute_gate2_wheel_evidence"
GATE2_FOCUSED_TARGETS = (
    "tests/test_v2_benchmark.py",
    "tests/test_v2_wheel_provider.py",
    "tests/test_v2_route_validation.py",
    "tests/test_v2_geometry.py",
    "tests/test_v2_profiles.py",
    "tests/test_v2_wheel_contracts.py",
    "tests/test_v2_contracts.py",
    "tests/test_v2_api.py",
    "tests/test_v2_runtime.py",
    "tests/test_hybrid_astar.py",
    "tests/test_astar.py",
)
GATE2_INPUTS = {
    "primitive_audit": "D:/xunce/inputs/path_v2/g2/independent_wheel_oracle_labels.jsonl",
    "exact_map_quality": "D:/xunce/inputs/path_v2/g2/independent_wheel_exact_map_optima.jsonl",
    "standard_episodes": "D:/xunce/inputs/path_v2/g2/standard_wheel_schedule.jsonl",
}
GATE2_BASELINE_EVIDENCE = {
    "schema_version": "xunce-path-v2-gate0-path-planner-junit/v1",
    "path": "D:/xunce/out/path_v2/g0/path_planner_baseline.junit.xml",
    "sha256": "90a02eb6af78805bfdf2fcce4828283cb3f4d77f30aae55519f533896a06e676",
    "test_count": 173,
}
GATE2_THRESHOLDS = {
    "min_primitive_independent_samples": 10000,
    "max_primitive_false_positives": 0,
    "min_primitive_recall": 0.98,
    "min_primitive_complete_l2_ratio": 1.0,
    "min_exact_map_independent_cases": 1,
    "min_exact_map_success_ratio": 1.0,
    "min_exact_map_resource_cost_ratio": 1.0,
    "max_exact_map_resource_cost_ratio": 1.10,
    "min_exact_map_complete_l2_ratio": 1.0,
    "min_standard_independent_episodes": 100,
    "min_standard_reachable_success_ratio": 0.99,
    "min_standard_complete_l2_ratio": 1.0,
    "max_standard_p95_runtime_ms": 250.0,
    "hard_timeout_ms": 2000.0,
    "max_hard_timeout_violations": 0,
}
GATE2_BLOCKERS = (
    ("primitive_audit", "provide_independent_wheel_oracle_labels"),
    ("exact_map_quality", "provide_independent_wheel_exact_map_optima"),
    ("standard_episodes", "provide_standard_wheel_schedule"),
)
GATE2_ROW_API = {
    "primitive_audit": ("PrimitiveAuditRowV2", "aggregate_primitive_audit_v2"),
    "exact_map_quality": ("ExactMapQualityRowV2", "aggregate_exact_map_quality_v2"),
    "standard_episodes": ("StandardEpisodeRowV2", "aggregate_standard_episodes_v2"),
}

GATE3_SCHEMA_VERSION = "xunce-path-v2-gate3-accelerators/v1"
GATE3_STAGE_ID = "xunce-path-v2-gate3-accelerators"
GATE3_INPUT_COMMIT = "d6b6b93c7e2c148195cd907010f46bd88e97ce2b"
GATE3_FORMAL_OUTPUT_ROOT = Path("D:/xunce/out/path_v2/g3")
GATE3_FORMAL_TEMP_ROOT = Path("D:/xunce/tmp/path_v2_g3")
GATE3_PROBE_SUBPROCESS_TIMEOUT_S = 30.0
GATE3_PROBE_TIMEOUT_RETURNCODE = 124
GATE3_PASS_ROUTE = "implement_path_v2_legged_static_stability_oracle"
GATE3_EXECUTE_ROUTE = "execute_gate3_accelerator_evidence"
GATE3_FOCUSED_TARGETS = (
    "tests/test_v2_search.py",
    "tests/test_v2_hierarchy.py",
    "tests/test_v2_cache.py",
    "tests/test_v2_lazy_validation.py",
    "tests/test_v2_route_validation.py",
    "tests/test_v2_wheel_provider.py",
    "tests/test_v2_wheel_contracts.py",
    "tests/test_v2_api.py",
    "tests/test_v2_runtime.py",
    "tests/test_v2_serialization.py",
    "tests/test_v2_contracts.py",
    "tests/test_hybrid_astar.py",
    "tests/test_astar.py",
    "../tests/test_xunce_path_v2_gate_benchmark.py",
)
GATE3_CASES = (
    "fine_only",
    "multi_heuristic_only",
    "hierarchy_only",
    "lazy_validation_only",
    "lazy_validation_plus_cache",
    "full_v2",
)
GATE3_WORKER_COUNTS = (1, 4)
GATE3_HASH_SEEDS = (11, 29, 47)
GATE3_REPEAT_COUNT = 3
GATE3_DISABLED_ACCELERATORS = (
    {"accelerator_id": "hierarchy", "reason": "not_integrated_into_provider"},
    {"accelerator_id": "lazy_validation", "reason": "not_integrated_into_provider"},
    {"accelerator_id": "multi_heuristic", "reason": "not_integrated_into_provider"},
    {"accelerator_id": "validation_cache", "reason": "not_integrated_into_provider"},
)

GATE4_SCHEMA_VERSION = "xunce-path-v2-gate4-legged/v1"
GATE4_STAGE_ID = "xunce-path-v2-gate4-legged"
GATE4_INPUT_COMMIT = "b7271935d0ad39a597df789e51aa62caa525eec6"
GATE4_FORMAL_OUTPUT_ROOT = Path("D:/xunce/out/path_v2/g4")
GATE4_FORMAL_TEMP_ROOT = Path("D:/xunce/tmp/path_v2_g4")
GATE4_PASS_ROUTE = "implement_path_v2_lunar_ballistics_and_hopper_proxy_profile"
GATE4_EXECUTE_ROUTE = "execute_gate4_legged_evidence"
GATE4_FOCUSED_TARGETS = (
    "tests/test_v2_benchmark.py",
    "tests/test_v2_legged_oracle.py",
    "tests/test_v2_legged_provider.py",
    "tests/test_v2_route_validation.py",
    "tests/test_v2_profiles.py",
    "tests/test_v2_geometry.py",
    "tests/test_v2_search.py",
    "tests/test_v2_api.py",
    "tests/test_v2_runtime.py",
    "tests/test_v2_contracts.py",
    "tests/test_v2_serialization.py",
    "tests/test_v2_fine_safety_anchor.py",
    "tests/test_hybrid_astar.py",
    "tests/test_astar.py",
    "../tests/test_xunce_path_v2_gate_benchmark.py",
)
GATE4_INPUTS = {
    "primitive_audit": "D:/xunce/inputs/path_v2/g4/independent_legged_oracle_labels.jsonl",
    "exact_map_quality": "D:/xunce/inputs/path_v2/g4/independent_legged_exact_map_optima.jsonl",
    "standard_episodes": "D:/xunce/inputs/path_v2/g4/standard_legged_schedule.jsonl",
}
GATE4_THRESHOLDS = {
    "min_primitive_independent_samples": 10000,
    "max_primitive_false_positives": 0,
    "min_primitive_recall": 0.98,
    "min_primitive_complete_l2_ratio": 1.0,
    "min_exact_map_independent_cases": 1,
    "min_exact_map_success_ratio": 1.0,
    "min_exact_map_resource_cost_ratio": 1.0,
    "max_exact_map_resource_cost_ratio": 1.10,
    "min_exact_map_complete_l2_ratio": 1.0,
    "min_standard_independent_episodes": 100,
    "min_standard_reachable_success_ratio": 0.99,
    "max_standard_unreachable_successes": 0,
    "min_standard_complete_l2_ratio": 1.0,
    "max_standard_p95_runtime_ms": 250.0,
    "hard_timeout_ms": 2000.0,
    "max_hard_timeout_violations": 0,
}
GATE4_CAPABILITY_DISCLOSURE = {
    "platform_kind": "legged",
    "platform_type": "legged_static_crawl",
    "capability": "simulation_proxy",
    "capability_revision": "simulation_proxy_static_crawl/v1",
    "primitive_capability": "simulation_proxy",
    "resource_proxy_id": "legged_static_crawl_relative_resource/v1",
    "route_validator_id": "path-planner-v2-legged-route-l2/v1",
    "dynamic_gait_claimed": False,
    "real_robot_stability_claimed": False,
}
GATE4_DATASET_CONTRACT = {
    "schema_version": "xunce-path-v2-gate4b-blocked-intake/v1",
    "accepts_formal_inputs": False,
    "future_stage_required": "xunce-path-v2-gate4c-legged-evidence",
    "formal_sources": {
        "primitive_audit": "independent-legged-oracle/v1",
        "exact_map_quality": "independent-legged-exact-solver/v1",
        "standard_episodes": "independent-standard-legged/v1",
    },
}
GATE4_TRUSTED_INPUT = {
    "approved_input_sha256": None,
    "approved_case_envelope_sha256": None,
    "approved_producer_id": None,
    "approved_producer_revision": None,
    "approval_id": None,
    "approved_provider_source_commit": "7d5c4dfc8e2a374855e6d8b962b69466c3353265",
    "approved_provider_build_id": None,
    "approved_collector_id": None,
    "approved_collector_revision": None,
    "approved_profile_id": "legged-static-crawl/v1",
    "approved_capability_revision": "simulation_proxy_static_crawl/v1",
    "approved_step_validator_id": "path-planner-v2-legged-static-stability/v1",
    "approved_route_validator_id": "path-planner-v2-legged-route-l2/v1",
}
GATE4_BLOCKERS = (
    ("primitive_audit", "provide_independent_legged_oracle_labels"),
    ("exact_map_quality", "provide_independent_legged_exact_map_optima"),
    ("standard_episodes", "provide_standard_legged_schedule"),
)
GATE4_BENCHMARK_ROW_CLASSES = (
    "PrimitiveAuditRowV2",
    "ExactMapQualityRowV2",
    "StandardEpisodeRowV2",
)
GATE4_BENCHMARK_AGGREGATES = (
    "aggregate_primitive_audit_v2",
    "aggregate_exact_map_quality_v2",
    "aggregate_standard_episodes_v2",
)
GATE4_PHASES = (
    "preflight",
    "focused",
    "full",
    "primitive-audit",
    "exact-map-quality",
    "standard-episodes",
    "boundary-review",
)


class _Gate2LoaderContractError(RuntimeError):
    pass


BYTE_PROBE_CODE = r"""
import hashlib
import json
import path_planner
import numpy as np

from path_planner.core import Cell, CostGrid, GridSpec, PlanRequest
from path_planner.search import AStarPlanner
from path_planner.v2.api import plan_v2
from path_planner.v2.contracts import (
    AcceleratorPolicyV2,
    ObjectiveProfileV2,
    PlanningRequestV2,
    PoseStateV2,
    ResourceBudgetV2,
)
from path_planner.v2.profiles import PlatformProfileRegistryV2
from path_planner.v2.serialization import canonical_json_bytes

request = PlanningRequestV2(
    request_id="gate1-byte-probe",
    platform_profile_id="unknown-profile/v1",
    start_state=PoseStateV2(0.25, 0.25, 0.0),
    goal_state=PoseStateV2(0.75, 0.75, 0.0),
    terrain_snapshot=object(),
    objective_profile=ObjectiveProfileV2(),
    resource_budget=ResourceBudgetV2(),
    timeout_s=1.0,
    accelerator_policy=AcceleratorPolicyV2.DISABLED,
    determinism_seed=1,
)
outcome = plan_v2(
    request,
    registry=PlatformProfileRegistryV2(()),
    providers={},
)
canonical = canonical_json_bytes(outcome)
spec = GridSpec(width=2, height=2, resolution=0.5)
grid = CostGrid(
    spec=spec,
    cost=np.ones((2, 2)),
    passable_mask=np.ones((2, 2), dtype=bool),
)
legacy_schema = AStarPlanner().plan(
    grid,
    PlanRequest(start=Cell(0, 0), goal=Cell(1, 1)),
).to_route_dict(spec)["schema_version"]
print(json.dumps({
    "canonical_hex": canonical.hex(),
    "digest": hashlib.sha256(canonical).hexdigest(),
    "root_has_plan_v2": "plan_v2" in path_planner.__dict__,
    "v1_route_schema": legacy_schema,
}, sort_keys=True, separators=(",", ":")))
"""


GATE3_PROBE_CODE = r"""
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
import os

import numpy as np
import path_planner.v2.validation as validation_module

from path_planner.core import Cell
from path_planner.v2.api import plan_v2
from path_planner.v2.cache import ValidationCacheV2
from path_planner.v2.contracts import (
    AcceleratorPolicyV2,
    ObjectiveProfileV2,
    PlanningRequestV2,
    PlanningSuccessV2,
    PlatformKindV2,
    PoseStateV2,
    ResourceBudgetV2,
    ValidationLevelV2,
)
from path_planner.v2.hierarchy import (
    ConservativeHierarchyV2,
    HierarchyHintStatusV2,
)
from path_planner.v2.profiles import (
    PlatformProfileRegistryV2,
    PlatformProfileV2,
    WheelProfileV2,
)
from path_planner.v2.providers.wheel import WheelPrimitiveProviderV2
from path_planner.v2.runtime import PlanningDeadlineV2
from path_planner.v2.search import SearchQueueEntryV2, StableSearchQueueV2
from path_planner.v2.serialization import canonical_json_bytes
from path_planner.v2.terrain import (
    FineGridGeometryV2,
    FineSafetyAnchorV2,
    TerrainProvenanceV2,
    TerrainSnapshotV2,
)
from path_planner.v2.validation import (
    RouteValidationResultV2,
    WheelValidationResultV2,
    validate_route,
    validate_route_l2,
)

CASES = tuple(json.loads(os.environ["PATH_V2_GATE3_CASES"]))
WORKER_COUNT = int(os.environ["PATH_V2_GATE3_WORKER_COUNT"])


def snapshot(*, mixed=False):
    shape = (8, 8)
    traversable = np.ones(shape, dtype=bool)
    hard = np.zeros(shape, dtype=bool)
    observed = np.ones(shape, dtype=bool)
    if mixed:
        observed[0, 0] = False
        hard[0, 1] = True
        traversable[0, 1] = False
    return TerrainSnapshotV2(
        geometry=FineGridGeometryV2(width=8, height=8, frame_id="moon"),
        elevation_m=np.zeros(shape, dtype=np.float64),
        slope_deg=np.zeros(shape, dtype=np.float64),
        traversable_mask=traversable,
        hard_obstacle_mask=hard,
        observed_mask=observed,
        confidence=np.ones(shape, dtype=np.float64),
        provenance=TerrainProvenanceV2(
            source_kind="synthetic_terrain_obstacle_proxy/v1",
            source_id="gate3-component-probe",
            source_hash="gate3-component-probe-hash",
            physical_obstacle_cells_written=False,
        ),
    )


generic_profile = PlatformProfileV2(
    profile_id="wheel-gate3/v1",
    platform_kind=PlatformKindV2.WHEEL,
    capability_revision="wheel-gate3-capability/v1",
    simulation_proxy=False,
    max_traversable_slope_deg=30.0,
    goal_position_tolerance_m=0.0,
    goal_heading_tolerance_rad=0.0,
)
wheel_profile = WheelProfileV2(profile=generic_profile)
terrain = snapshot()
state = PoseStateV2(2.25, 2.25, 0.0)
request = PlanningRequestV2(
    request_id="gate3-component-probe",
    platform_profile_id=generic_profile.profile_id,
    start_state=state,
    goal_state=state,
    terrain_snapshot=terrain,
    objective_profile=ObjectiveProfileV2(),
    resource_budget=ResourceBudgetV2(),
    timeout_s=1.0,
    accelerator_policy=AcceleratorPolicyV2.OPTIONAL,
    determinism_seed=17,
)
provider = WheelPrimitiveProviderV2(wheel_profile)
outcome = plan_v2(
    request,
    registry=PlatformProfileRegistryV2((generic_profile,)),
    providers={generic_profile.profile_id: provider},
    monotonic_clock=lambda: 0.0,
)
fatal_reason = None
if type(outcome) is not PlanningSuccessV2:
    telemetry = getattr(outcome, "search_telemetry", None)
    fatal_reason = (
        "planning_deadline_expired"
        if getattr(telemetry, "timed_out", False) is True
        else "fine_anchor_failed"
    )

fine_result = None
if fatal_reason is None:
    anchor = FineSafetyAnchorV2(terrain)
    deadline = PlanningDeadlineV2(0.0, 100.0, lambda: 0.0)
    try:
        fine_result = validate_route_l2(
            outcome.route,
            request,
            anchor,
            wheel_profile,
            deadline,
        )
    except TimeoutError:
        fatal_reason = "planning_deadline_expired"
    except Exception:
        fatal_reason = "l2_authority_malformed"
    else:
        if type(fine_result) is not WheelValidationResultV2:
            fatal_reason = "l2_authority_malformed"
        elif (
            fine_result.timed_out is True
            or fine_result.reason_code == "planning_deadline_expired"
        ):
            fatal_reason = "planning_deadline_expired"
        elif not fine_result.evidence.passed:
            fatal_reason = "l2_rejected"
        elif fine_result.validated_route_hash is None:
            fatal_reason = "l2_authority_malformed"


def recheck_fine_l2_authority():
    try:
        rechecked = validate_route_l2(
            outcome.route,
            request,
            anchor,
            wheel_profile,
            PlanningDeadlineV2(0.0, 100.0, lambda: 0.0),
        )
    except TimeoutError:
        return "planning_deadline_expired"
    except Exception:
        return "l2_authority_malformed"
    if type(rechecked) is not WheelValidationResultV2:
        return "l2_authority_malformed"
    if (
        rechecked.timed_out is True
        or rechecked.reason_code == "planning_deadline_expired"
    ):
        return "planning_deadline_expired"
    if not rechecked.evidence.passed:
        return "l2_rejected"
    if (
        rechecked.reason_code != fine_result.reason_code
        or rechecked.validated_route_hash != fine_result.validated_route_hash
        or rechecked.evidence.validator_id != fine_result.evidence.validator_id
    ):
        return "l2_authority_malformed"
    return None


def route_validation_fatal_reason(result):
    if type(result) is not RouteValidationResultV2:
        return "l2_authority_malformed"
    if result.reason_code == "planning_deadline_expired":
        return "planning_deadline_expired"
    if result.reason_code == "route_l2_authority_unavailable":
        return "l2_authority_malformed"
    if result.l2_result is not None and type(result.l2_result) is not WheelValidationResultV2:
        return "l2_authority_malformed"
    if result.l2_result is not None and (
        result.l2_result.timed_out is True
        or result.l2_result.reason_code == "planning_deadline_expired"
    ):
        return "planning_deadline_expired"
    if result.l2_result is not None and not result.l2_result.evidence.passed:
        return "l2_rejected"
    return None


def validate_route_with_l2_tracking(*args, **kwargs):
    original_l2 = validation_module.validate_route_l2
    scope = {
        "component_exception": None,
        "l2_fault_reason": None,
        "restored": False,
    }

    def tracked_l2(*l2_args, **l2_kwargs):
        try:
            result = original_l2(*l2_args, **l2_kwargs)
        except TimeoutError:
            scope["l2_fault_reason"] = "planning_deadline_expired"
            raise
        except Exception:
            scope["l2_fault_reason"] = "l2_authority_malformed"
            raise
        if type(result) is not WheelValidationResultV2:
            scope["l2_fault_reason"] = "l2_authority_malformed"
        elif (
            result.timed_out is True
            or result.reason_code == "planning_deadline_expired"
        ):
            scope["l2_fault_reason"] = "planning_deadline_expired"
        elif not result.evidence.passed:
            scope["l2_fault_reason"] = "l2_rejected"
        return result

    validation_module.validate_route_l2 = tracked_l2
    try:
        result = validate_route(*args, **kwargs)
    except TimeoutError:
        scope["component_exception"] = "timeout"
        result = None
    except Exception:
        scope["component_exception"] = "exception"
        result = None
    finally:
        validation_module.validate_route_l2 = original_l2
        scope["restored"] = validation_module.validate_route_l2 is original_l2
    return result, scope


def validation_scope_fatal_reason(scope):
    if scope["restored"] is not True:
        return "l2_authority_malformed"
    if scope["l2_fault_reason"] is not None:
        return scope["l2_fault_reason"]
    if scope["component_exception"] == "timeout":
        return "planning_deadline_expired"
    if scope["component_exception"] == "exception":
        return recheck_fine_l2_authority()
    return None


entries = ()
expected_order = ()
search_ok = True
if fatal_reason is None:
    try:
        entries = (
            SearchQueueEntryV2(
                "anchor-first",
                (0,),
                "a",
                0.0,
                0.0,
                (("resource", 9.0),),
            ),
            SearchQueueEntryV2(
                "aux-first",
                (1,),
                "a",
                1.0,
                0.0,
                (("resource", 0.0),),
            ),
            SearchQueueEntryV2(
                "later",
                (2,),
                "a",
                0.5,
                1.0,
                (("resource", 2.0),),
            ),
        )
        expected_order = tuple(
            entry.candidate_id
            for entry in sorted(
                entries,
                key=lambda entry: (
                    entry.path_cost + entry.anchor_heuristic,
                    entry.path_cost,
                    entry.state_key,
                    entry.primitive_key,
                    entry.candidate_id,
                ),
            )
        )
        queue = StableSearchQueueV2()
        batches = ((entries[0], entries[2]), (entries[1],))
        with ThreadPoolExecutor(max_workers=WORKER_COUNT) as executor:
            futures = tuple(
                executor.submit(lambda batch: batch, batch) for batch in batches
            )
            completion_order = (0, 1) if WORKER_COUNT == 1 else (1, 0)
            for index in completion_order:
                queue.extend(futures[index].result())
        suggestion = queue.suggest("resource")
        popped = []
        while len(queue):
            popped.append(queue.pop_anchor().candidate_id)
        search_ok = (
            suggestion is not None
            and suggestion.candidate_id == "aux-first"
            and tuple(popped) == expected_order
            and expected_order[0] == "anchor-first"
        )
    except TimeoutError:
        fatal_reason = "planning_deadline_expired"
        search_ok = False
    except Exception:
        search_ok = False

hierarchy_ok = True
if fatal_reason is None:
    try:
        mixed_anchor = FineSafetyAnchorV2(snapshot(mixed=True))
        hierarchy = ConservativeHierarchyV2.build(
            mixed_anchor,
            max_slope_deg=30.0,
            deadline=PlanningDeadlineV2(0.0, 100.0, lambda: 0.0),
        )
        mixed_hint = hierarchy.hint(2, Cell(0, 0))
        hierarchy_ok = (
            mixed_hint.status is HierarchyHintStatusV2.UNKNOWN
            and mixed_anchor.query(Cell(0, 0), 30.0).reason_code
            == "terrain_unknown"
            and mixed_anchor.query(Cell(1, 0), 30.0).reason_code
            == "terrain_hard_obstacle"
        )
    except TimeoutError:
        fatal_reason = "planning_deadline_expired"
        hierarchy_ok = False
    except Exception:
        hierarchy_ok = False

lazy_ok = False
cache_ok = False
if fatal_reason is None:
    lazy_result, lazy_scope = validate_route_with_l2_tracking(
        outcome.route,
        anchor,
        wheel_profile,
        ValidationLevelV2.L2,
        request=request,
        deadline=PlanningDeadlineV2(0.0, 100.0, lambda: 0.0),
    )
    lazy_scope_reason = validation_scope_fatal_reason(lazy_scope)
    if lazy_scope_reason is not None:
        fatal_reason = lazy_scope_reason
    elif lazy_scope["component_exception"] is None:
        lazy_ok = (
            type(lazy_result) is RouteValidationResultV2
            and lazy_result.success
            and lazy_result.l2_result is not None
            and lazy_result.l2_result.evidence.passed
            and lazy_result.l2_result.reason_code == fine_result.reason_code
            and lazy_result.l2_result.validated_route_hash
            == fine_result.validated_route_hash
        )
        fatal_reason = route_validation_fatal_reason(lazy_result)

if fatal_reason is None:
    try:
        cache = ValidationCacheV2()
    except TimeoutError:
        fatal_reason = "planning_deadline_expired"
        cache = None
    except Exception:
        cache = None

    if fatal_reason is None and cache is not None:
        cached_first, cached_first_scope = validate_route_with_l2_tracking(
            outcome.route,
            anchor,
            wheel_profile,
            ValidationLevelV2.L2,
            request=request,
            deadline=PlanningDeadlineV2(0.0, 100.0, lambda: 0.0),
            cache=cache,
        )
        cached_first_scope_reason = validation_scope_fatal_reason(cached_first_scope)
        if cached_first_scope_reason is not None:
            fatal_reason = cached_first_scope_reason
        elif cached_first_scope["component_exception"] is None:
            fatal_reason = route_validation_fatal_reason(cached_first)
            if fatal_reason is None:
                cached_second, cached_second_scope = validate_route_with_l2_tracking(
                    outcome.route,
                    anchor,
                    wheel_profile,
                    ValidationLevelV2.L2,
                    request=request,
                    deadline=PlanningDeadlineV2(0.0, 100.0, lambda: 0.0),
                    cache=cache,
                )
                cached_second_scope_reason = validation_scope_fatal_reason(
                    cached_second_scope
                )
                if cached_second_scope_reason is not None:
                    fatal_reason = cached_second_scope_reason
                elif cached_second_scope["component_exception"] is None:
                    fatal_reason = route_validation_fatal_reason(cached_second)
                    cache_ok = (
                        fatal_reason is None
                        and type(cached_first) is RouteValidationResultV2
                        and type(cached_second) is RouteValidationResultV2
                        and cached_first.success
                        and cached_second.success
                        and cached_second.cache_hits == 1
                        and cached_first.l2_result is not None
                        and cached_second.l2_result is not None
                        and cached_first.l2_result.reason_code
                        == fine_result.reason_code
                        and cached_second.l2_result.reason_code
                        == fine_result.reason_code
                        and cached_first.l2_result.validated_route_hash
                        == fine_result.validated_route_hash
                        and cached_second.l2_result.validated_route_hash
                        == fine_result.validated_route_hash
                    )

accelerator_used = bool(
    getattr(getattr(outcome, "search_telemetry", None), "accelerator_used", True)
)
projection = {
    "authoritative_order": list(expected_order),
    "outcome_category": type(outcome).__name__,
    "provider_termination": getattr(
        getattr(outcome, "search_telemetry", None),
        "termination_reason",
        "missing",
    ),
    "route_reason": getattr(fine_result, "reason_code", fatal_reason),
    "validated_route_hash": getattr(fine_result, "validated_route_hash", None),
    "l2_validator": getattr(
        getattr(fine_result, "evidence", None),
        "validator_id",
        None,
    ),
    "accelerator_used": accelerator_used,
}
fine_only_digest = sha256(canonical_json_bytes(projection)).hexdigest()

enabled_by_case = {
    "fine_only": (),
    "multi_heuristic_only": ("multi_heuristic",),
    "hierarchy_only": ("hierarchy",),
    "lazy_validation_only": ("lazy_validation",),
    "lazy_validation_plus_cache": ("lazy_validation", "validation_cache"),
    "full_v2": (
        "hierarchy",
        "lazy_validation",
        "multi_heuristic",
        "validation_cache",
    ),
}
component_ok = {
    "hierarchy": hierarchy_ok,
    "lazy_validation": lazy_ok,
    "multi_heuristic": search_ok,
    "validation_cache": cache_ok,
}
rows = []
for case_id in CASES:
    enabled = enabled_by_case[case_id]
    runtime_disabled = (
        []
        if fatal_reason is not None
        else [
            {"accelerator_id": accelerator_id, "reason": "component_probe_failed"}
            for accelerator_id in enabled
            if not component_ok[accelerator_id]
        ]
    )
    fallback_isolated = all(
        item["accelerator_id"] in enabled for item in runtime_disabled
    )
    rows.append(
        {
            "case_id": case_id,
            "status": (
                "passed" if fatal_reason is None and fallback_isolated else "failed"
            ),
            "decision_digest": fine_only_digest,
            "fine_only_digest": fine_only_digest,
            "safety_equivalent": fatal_reason is None,
            "authoritative_order_preserved": (
                "multi_heuristic" not in enabled
                or search_ok
                or "multi_heuristic" in {
                    item["accelerator_id"] for item in runtime_disabled
                }
            ),
            "suggestion_non_authoritative": (
                "multi_heuristic" not in enabled
                or search_ok
                or "multi_heuristic" in {
                    item["accelerator_id"] for item in runtime_disabled
                }
            ),
            "hierarchy_conservative": (
                "hierarchy" not in enabled
                or hierarchy_ok
                or "hierarchy" in {
                    item["accelerator_id"] for item in runtime_disabled
                }
            ),
            "cache_l2_equivalent": (
                "validation_cache" not in enabled
                or cache_ok
                or "validation_cache" in {
                    item["accelerator_id"] for item in runtime_disabled
                }
            ),
            "l2_authority_preserved": fatal_reason is None,
            "fallback_isolated": fallback_isolated,
            "fatal_reason": fatal_reason,
            "accelerator_used": accelerator_used,
            "runtime_disabled_accelerators": runtime_disabled,
        }
    )

print(json.dumps({"rows": rows}, sort_keys=True, separators=(",", ":")))
"""


def _case_outcome(testcase) -> str:
    if testcase.find("failure") is not None:
        return "failures"
    if testcase.find("error") is not None:
        return "errors"
    if testcase.find("skipped") is not None:
        return "skipped"
    return "passed"


def _empty_counts() -> dict[str, int]:
    return {"passed": 0, "skipped": 0, "failures": 0, "errors": 0}


def _is_v2_nodeid(nodeid: str) -> bool:
    module_path = nodeid.split("::", 1)[0].replace("\\", "/")
    return any(part.startswith("test_v2_") for part in module_path.split("/"))


def audit_full_junit(
    path: Path,
    *,
    expected_legacy: dict[str, Any],
    allowed_skip_dependency: str,
) -> dict[str, Any]:
    total = _empty_counts()
    legacy = _empty_counts()
    v2 = _empty_counts()
    legacy_skip_messages: list[str] = []
    for testcase in gate0._junit_testcases(Path(path)):
        nodeid = gate0._testcase_nodeid(testcase)
        outcome = _case_outcome(testcase)
        total[outcome] += 1
        bucket = v2 if _is_v2_nodeid(nodeid) else legacy
        bucket[outcome] += 1
        if bucket is legacy and outcome == "skipped":
            skipped = testcase.find("skipped")
            legacy_skip_messages.append(
                " ".join(
                    filter(
                        None,
                        (skipped.attrib.get("message", ""), skipped.text or ""),
                    )
                )
            )

    expected = {
        key: int(expected_legacy[key])
        for key in ("passed", "skipped", "failures", "errors")
    }
    dependency = str(allowed_skip_dependency).lower()
    skip_contract = (
        len(legacy_skip_messages) == legacy["skipped"]
        and all(dependency in message.lower() for message in legacy_skip_messages)
    )
    v2_green = (
        v2["failures"] == 0
        and v2["errors"] == 0
        and v2["skipped"] == 0
    )
    status = "passed" if legacy == expected and skip_contract and v2_green else "failed"
    return {
        "schema_version": "xunce-path-v2-gate1-full-junit-audit/v1",
        "status": status,
        "total": total,
        "legacy": legacy,
        "v2": v2,
        "legacy_skip_messages": legacy_skip_messages,
        "legacy_skip_contract": skip_contract,
    }


def _audit_focused_junit(path: Path) -> dict[str, Any]:
    summary = gate0.parse_junit(Path(path))
    status = (
        "passed"
        if summary.tests > 0
        and summary.failures == 0
        and summary.errors == 0
        and summary.skipped == 0
        else "failed"
    )
    return {
        "schema_version": "xunce-path-v2-gate1-focused-junit-audit/v1",
        "status": status,
        "tests": summary.tests,
        "passed": summary.passed,
        "skipped": summary.skipped,
        "failures": summary.failures,
        "errors": summary.errors,
    }


def audit_repeat_digests(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    normalized = list(rows)
    digests = [row.get("digest") for row in normalized]
    seeds = [str(row.get("python_hash_seed")) for row in normalized]
    repeats = [row.get("repeat") for row in normalized]
    valid_digests = all(
        isinstance(digest, str)
        and len(digest) == 64
        and all(character in "0123456789abcdef" for character in digest)
        for digest in digests
    )
    passed = (
        len(normalized) >= 3
        and repeats == list(range(1, len(normalized) + 1))
        and len(seeds) == len(set(seeds))
        and valid_digests
        and len(set(digests)) == 1
    )
    return {
        "schema_version": "xunce-path-v2-gate1-byte-repeat-audit/v1",
        "status": "passed" if passed else "failed",
        "repeat_count": len(normalized),
        "python_hash_seeds": seeds,
        "digests": digests,
        "stable_digest": digests[0] if passed else None,
    }


def boundaries_match(configured: Any) -> bool:
    return (
        isinstance(configured, dict)
        and set(configured) == set(gate_artifacts.BOUNDARY_FIELDS)
        and all(configured[field] is False for field in gate_artifacts.BOUNDARY_FIELDS)
    )


def validate_output_root(repo_root: Path, output_root: Path) -> Path:
    repo_root = Path(repo_root).resolve()
    output_root = Path(output_root).resolve()
    if output_root == repo_root or output_root.is_relative_to(repo_root):
        raise ValueError("output_root must be outside repo")
    return output_root


def _assert_no_stale_artifacts(output_root: Path) -> None:
    root = Path(output_root).resolve()
    safe_root = artifact_io.windows_safe_path(root)
    if not os.path.exists(safe_root):
        return
    if not os.path.isdir(safe_root):
        raise RuntimeError("output_root exists but is not a directory")
    stale = sorted(
        entry.name
        for entry in os.scandir(safe_root)
        if entry.name not in CANONICAL_ARTIFACT_NAMES
        or not entry.is_file(follow_symlinks=False)
    )
    if stale:
        raise RuntimeError(f"stale noncanonical Gate 1 artifacts: {stale}")


def _assert_gate3_output_root_absent(output_root: Path) -> None:
    root = Path(output_root).resolve()
    if os.path.lexists(artifact_io.windows_safe_path(root)):
        raise RuntimeError("Gate 3 output_root already exists")


def _windows_safe_lexical_absolute_path(path: str | Path) -> str:
    absolute = os.path.abspath(os.fspath(path))
    if os.name != "nt" or absolute.startswith("\\\\?\\"):
        return absolute
    if absolute.startswith("\\\\"):
        return "\\\\?\\UNC\\" + absolute[2:]
    return "\\\\?\\" + absolute


def _assert_gate4_output_root_absent(output_root: Path) -> None:
    if os.path.lexists(_windows_safe_lexical_absolute_path(output_root)):
        raise RuntimeError("Gate 4 output_root already exists")


def _git(repo_root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return completed.stdout.strip()


def _audit_nested_git(repo_root: Path, expected_branch: str) -> dict[str, Any]:
    nested_root = repo_root / "path-planner"
    try:
        gitlink = _git(repo_root, "rev-parse", "HEAD:path-planner")
        head = _git(nested_root, "rev-parse", "HEAD")
        branch = _git(nested_root, "rev-parse", "--abbrev-ref", "HEAD")
        porcelain = _git(nested_root, "status", "--porcelain", "--untracked-files=all")
    except subprocess.CalledProcessError as exc:
        return {
            "schema_version": "xunce-path-v2-gate1-nested-git-audit/v1",
            "status": "failed",
            "error": (exc.stderr or type(exc).__name__).strip(),
        }
    checks = {
        "head_matches_gitlink": head == gitlink,
        "branch_matches": branch == expected_branch,
        "clean_tree": not porcelain,
    }
    return {
        "schema_version": "xunce-path-v2-gate1-nested-git-audit/v1",
        "status": "passed" if all(checks.values()) else "failed",
        "gitlink": gitlink,
        "head": head,
        "branch": branch,
        "expected_branch": expected_branch,
        "dirty_paths": porcelain.splitlines(),
        **checks,
    }


def _strict_equal(actual: Any, expected: Any) -> bool:
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return set(actual) == set(expected) and all(
            _strict_equal(actual[key], value) for key, value in expected.items()
        )
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _strict_equal(left, right) for left, right in zip(actual, expected, strict=True)
        )
    return actual == expected


def _require_frozen(configured: Any, expected: Any, field: str) -> None:
    if not _strict_equal(configured, expected):
        raise ValueError(f"frozen {field} must equal {expected!r}")


def _validate_config(config: dict[str, Any]) -> None:
    if config.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION!r}")
    if config.get("stage_id") != STAGE_ID:
        raise ValueError(f"stage_id must be {STAGE_ID!r}")
    expected_git = config.get("expected_git", {})
    _require_frozen(expected_git.get("branch"), EXPECTED_BRANCH, "expected_git.branch")
    _require_frozen(
        expected_git.get("nested_branch"),
        EXPECTED_BRANCH,
        "expected_git.nested_branch",
    )
    _require_frozen(
        expected_git.get("base_commit"),
        ORIGINAL_BASE_COMMIT,
        "expected_git.base_commit",
    )
    _require_frozen(
        expected_git.get("gate_input_commit"),
        GATE_INPUT_COMMIT,
        "expected_git.gate_input_commit",
    )
    _require_frozen(config.get("python"), FORMAL_PYTHON.as_posix(), "python")
    _require_frozen(
        config.get("expected_python_version"),
        EXPECTED_PYTHON_VERSION,
        "expected_python_version",
    )
    _require_frozen(config.get("temp_root"), FORMAL_TEMP_ROOT.as_posix(), "temp_root")
    focused = config.get("focused", {})
    full = config.get("full", {})
    _require_frozen(
        focused.get("working_directory"),
        PATH_PLANNER_WORKING_DIRECTORY,
        "focused.working_directory",
    )
    _require_frozen(
        focused.get("pythonpath"),
        PATH_PLANNER_PYTHONPATH,
        "focused.pythonpath",
    )
    _require_frozen(
        focused.get("pytest_targets"),
        list(FOCUSED_TARGETS),
        "focused.pytest_targets",
    )
    _require_frozen(
        full.get("working_directory"),
        PATH_PLANNER_WORKING_DIRECTORY,
        "full.working_directory",
    )
    _require_frozen(
        full.get("pythonpath"),
        PATH_PLANNER_PYTHONPATH,
        "full.pythonpath",
    )
    _require_frozen(full.get("pytest_targets"), ["tests"], "full.pytest_targets")
    _require_frozen(full.get("legacy_expected"), LEGACY_EXPECTED, "full.legacy_expected")
    _require_frozen(
        full.get("allowed_skip_dependency"),
        ALLOWED_SKIP_DEPENDENCY,
        "full.allowed_skip_dependency",
    )
    byte_repeat = config.get("byte_repeat", {})
    repeat_count = byte_repeat.get("repeat_count")
    seeds = byte_repeat.get("python_hash_seeds")
    if (
        isinstance(repeat_count, bool)
        or not isinstance(repeat_count, int)
        or repeat_count < 3
        or not isinstance(seeds, list)
        or len(seeds) != repeat_count
        or len({str(seed) for seed in seeds}) != repeat_count
    ):
        raise ValueError("byte_repeat requires at least three distinct hash seeds")
    if config.get("pass_route") != PASS_ROUTE:
        raise ValueError(f"pass_route must be {PASS_ROUTE!r}")


def _validate_gate2_config(config: dict[str, Any]) -> None:
    expected_keys = {
        "schema_version",
        "stage_id",
        "python",
        "expected_python_version",
        "expected_git",
        "temp_root",
        "focused",
        "full",
        "inputs",
        "baseline_evidence",
        "thresholds",
        "boundaries",
        "pass_route",
    }
    _require_frozen(set(config), expected_keys, "top-level keys")
    _require_frozen(config.get("schema_version"), GATE2_SCHEMA_VERSION, "schema_version")
    _require_frozen(config.get("stage_id"), GATE2_STAGE_ID, "stage_id")
    _require_frozen(config.get("python"), FORMAL_PYTHON.as_posix(), "python")
    _require_frozen(
        config.get("expected_python_version"),
        EXPECTED_PYTHON_VERSION,
        "expected_python_version",
    )
    _require_frozen(
        config.get("expected_git"),
        {
            "branch": EXPECTED_BRANCH,
            "base_commit": ORIGINAL_BASE_COMMIT,
            "gate_input_commit": GATE2_INPUT_COMMIT,
            "nested_branch": EXPECTED_BRANCH,
        },
        "expected_git",
    )
    _require_frozen(
        config.get("temp_root"),
        GATE2_FORMAL_TEMP_ROOT.as_posix(),
        "temp_root",
    )
    _require_frozen(
        config.get("focused"),
        {
            "working_directory": PATH_PLANNER_WORKING_DIRECTORY,
            "pythonpath": PATH_PLANNER_PYTHONPATH,
            "pytest_targets": list(GATE2_FOCUSED_TARGETS),
        },
        "focused",
    )
    _require_frozen(
        config.get("full"),
        {
            "working_directory": PATH_PLANNER_WORKING_DIRECTORY,
            "pythonpath": PATH_PLANNER_PYTHONPATH,
            "pytest_targets": ["tests"],
            "legacy_expected": LEGACY_EXPECTED,
            "allowed_skip_dependency": ALLOWED_SKIP_DEPENDENCY,
        },
        "full",
    )
    _require_frozen(config.get("inputs"), GATE2_INPUTS, "inputs")
    _require_frozen(
        config.get("baseline_evidence"),
        GATE2_BASELINE_EVIDENCE,
        "baseline_evidence",
    )
    _require_frozen(config.get("thresholds"), GATE2_THRESHOLDS, "thresholds")
    _require_frozen(
        config.get("boundaries"),
        gate_artifacts.BOUNDARY_FIELDS,
        "boundaries",
    )
    _require_frozen(config.get("pass_route"), GATE2_PASS_ROUTE, "pass_route")


def _validate_gate3_config(config: dict[str, Any]) -> None:
    expected_keys = {
        "schema_version",
        "stage_id",
        "python",
        "expected_python_version",
        "expected_git",
        "formal_output_root",
        "temp_root",
        "probe_subprocess_timeout_s",
        "focused",
        "full",
        "baseline_evidence",
        "ablation",
        "expected_disabled_accelerators",
        "boundaries",
        "pass_route",
    }
    _require_frozen(set(config), expected_keys, "top-level keys")
    _require_frozen(config.get("schema_version"), GATE3_SCHEMA_VERSION, "schema_version")
    _require_frozen(config.get("stage_id"), GATE3_STAGE_ID, "stage_id")
    _require_frozen(config.get("python"), FORMAL_PYTHON.as_posix(), "python")
    _require_frozen(
        config.get("expected_python_version"),
        EXPECTED_PYTHON_VERSION,
        "expected_python_version",
    )
    _require_frozen(
        config.get("expected_git"),
        {
            "branch": EXPECTED_BRANCH,
            "base_commit": ORIGINAL_BASE_COMMIT,
            "gate_input_commit": GATE3_INPUT_COMMIT,
            "nested_branch": EXPECTED_BRANCH,
        },
        "expected_git",
    )
    _require_frozen(
        config.get("formal_output_root"),
        GATE3_FORMAL_OUTPUT_ROOT.as_posix(),
        "formal_output_root",
    )
    _require_frozen(
        config.get("temp_root"),
        GATE3_FORMAL_TEMP_ROOT.as_posix(),
        "temp_root",
    )
    _require_frozen(
        config.get("probe_subprocess_timeout_s"),
        GATE3_PROBE_SUBPROCESS_TIMEOUT_S,
        "probe_subprocess_timeout_s",
    )
    _require_frozen(
        config.get("focused"),
        {
            "working_directory": PATH_PLANNER_WORKING_DIRECTORY,
            "pythonpath": PATH_PLANNER_PYTHONPATH,
            "pytest_targets": list(GATE3_FOCUSED_TARGETS),
        },
        "focused",
    )
    _require_frozen(
        config.get("full"),
        {
            "working_directory": PATH_PLANNER_WORKING_DIRECTORY,
            "pythonpath": PATH_PLANNER_PYTHONPATH,
            "pytest_targets": ["tests"],
            "legacy_expected": LEGACY_EXPECTED,
            "allowed_skip_dependency": ALLOWED_SKIP_DEPENDENCY,
        },
        "full",
    )
    _require_frozen(
        config.get("baseline_evidence"),
        GATE2_BASELINE_EVIDENCE,
        "baseline_evidence",
    )
    _require_frozen(
        config.get("ablation"),
        {
            "cases": list(GATE3_CASES),
            "worker_counts": list(GATE3_WORKER_COUNTS),
            "python_hash_seeds": list(GATE3_HASH_SEEDS),
            "repeat_count": GATE3_REPEAT_COUNT,
        },
        "ablation",
    )
    _require_frozen(
        config.get("expected_disabled_accelerators"),
        list(GATE3_DISABLED_ACCELERATORS),
        "expected_disabled_accelerators",
    )
    _require_frozen(
        config.get("boundaries"),
        gate_artifacts.BOUNDARY_FIELDS,
        "boundaries",
    )
    _require_frozen(config.get("pass_route"), GATE3_PASS_ROUTE, "pass_route")


def _validate_gate4_config(config: dict[str, Any]) -> None:
    expected_keys = {
        "schema_version",
        "stage_id",
        "python",
        "expected_python_version",
        "expected_git",
        "formal_output_root",
        "temp_root",
        "focused",
        "full",
        "inputs",
        "baseline_evidence",
        "thresholds",
        "capability_disclosure",
        "dataset_contract",
        "trusted_inputs",
        "boundaries",
        "pass_route",
    }
    _require_frozen(set(config), expected_keys, "top-level keys")
    _require_frozen(config.get("schema_version"), GATE4_SCHEMA_VERSION, "schema_version")
    _require_frozen(config.get("stage_id"), GATE4_STAGE_ID, "stage_id")
    _require_frozen(config.get("python"), FORMAL_PYTHON.as_posix(), "python")
    _require_frozen(
        config.get("expected_python_version"),
        EXPECTED_PYTHON_VERSION,
        "expected_python_version",
    )
    _require_frozen(
        config.get("expected_git"),
        {
            "branch": EXPECTED_BRANCH,
            "base_commit": ORIGINAL_BASE_COMMIT,
            "gate_input_commit": GATE4_INPUT_COMMIT,
            "nested_branch": EXPECTED_BRANCH,
        },
        "expected_git",
    )
    _require_frozen(
        config.get("formal_output_root"),
        GATE4_FORMAL_OUTPUT_ROOT.as_posix(),
        "formal_output_root",
    )
    _require_frozen(
        config.get("temp_root"),
        GATE4_FORMAL_TEMP_ROOT.as_posix(),
        "temp_root",
    )
    _require_frozen(
        config.get("focused"),
        {
            "working_directory": PATH_PLANNER_WORKING_DIRECTORY,
            "pythonpath": PATH_PLANNER_PYTHONPATH,
            "pytest_targets": list(GATE4_FOCUSED_TARGETS),
        },
        "focused",
    )
    _require_frozen(
        config.get("full"),
        {
            "working_directory": PATH_PLANNER_WORKING_DIRECTORY,
            "pythonpath": PATH_PLANNER_PYTHONPATH,
            "pytest_targets": ["tests"],
            "legacy_expected": LEGACY_EXPECTED,
            "allowed_skip_dependency": ALLOWED_SKIP_DEPENDENCY,
        },
        "full",
    )
    _require_frozen(config.get("inputs"), GATE4_INPUTS, "inputs")
    _require_frozen(
        config.get("baseline_evidence"),
        GATE2_BASELINE_EVIDENCE,
        "baseline_evidence",
    )
    _require_frozen(config.get("thresholds"), GATE4_THRESHOLDS, "thresholds")
    _require_frozen(
        config.get("capability_disclosure"),
        GATE4_CAPABILITY_DISCLOSURE,
        "capability_disclosure",
    )
    _require_frozen(
        config.get("dataset_contract"),
        GATE4_DATASET_CONTRACT,
        "dataset_contract",
    )
    _require_frozen(
        config.get("trusted_inputs"),
        {name: dict(GATE4_TRUSTED_INPUT) for name in GATE4_INPUTS},
        "trusted_inputs",
    )
    _require_frozen(
        config.get("boundaries"),
        gate_artifacts.BOUNDARY_FIELDS,
        "boundaries",
    )
    _require_frozen(config.get("pass_route"), GATE4_PASS_ROUTE, "pass_route")


def _validate_gate4_output_mode(output_root: Path, *, execute_tests: bool) -> None:
    resolved_output = Path(output_root).resolve()
    resolved_formal = GATE4_FORMAL_OUTPUT_ROOT.resolve()
    if execute_tests and resolved_output != resolved_formal:
        raise ValueError("Gate 4 formal execution requires exact formal_output_root")
    if not execute_tests and resolved_output == resolved_formal:
        raise ValueError("Gate 4 dry run must not use formal_output_root")


def _junit_records(
    content: bytes,
) -> tuple[list[ET.Element], dict[str, dict[str, str]], list[str]]:
    testcases = list(ET.fromstring(content).iter("testcase"))
    records: dict[str, dict[str, str]] = {}
    duplicates: list[str] = []
    for testcase in testcases:
        nodeid = gate0._testcase_nodeid(testcase)
        if nodeid in records:
            duplicates.append(nodeid)
            continue
        skipped = testcase.find("skipped")
        skip_message = ""
        if skipped is not None:
            skip_message = " ".join(
                filter(
                    None,
                    (skipped.attrib.get("message", ""), skipped.text or ""),
                )
            )
        records[nodeid] = {
            "outcome": _case_outcome(testcase),
            "skip_message": skip_message,
        }
    return testcases, records, sorted(set(duplicates))


def _counts_for_records(records: Sequence[dict[str, str]]) -> dict[str, int]:
    counts = _empty_counts()
    for record in records:
        counts[record["outcome"]] += 1
    return counts


def _audit_gate2_full_junit(
    path: Path,
    *,
    expected_legacy: dict[str, Any],
    allowed_skip_dependency: str,
    baseline_evidence: dict[str, Any],
) -> dict[str, Any]:
    baseline = {
        key: int(expected_legacy[key])
        for key in ("passed", "skipped", "failures", "errors")
    }
    baseline_path = Path(str(baseline_evidence["path"]))
    expected_hash = str(baseline_evidence["sha256"])
    expected_test_count = int(baseline_evidence["test_count"])
    evidence_review = {
        "schema_version": baseline_evidence.get("schema_version"),
        "path": baseline_path.as_posix(),
        "expected_sha256": expected_hash,
        "actual_sha256": None,
        "expected_test_count": expected_test_count,
        "actual_test_count": None,
        "status": "failed",
    }
    try:
        baseline_bytes = artifact_io.read_bytes(baseline_path)
        evidence_review["actual_sha256"] = hashlib.sha256(baseline_bytes).hexdigest()
        baseline_cases, baseline_records, baseline_duplicates = _junit_records(
            baseline_bytes
        )
        evidence_review["actual_test_count"] = len(baseline_cases)
        current_bytes = artifact_io.read_bytes(Path(path))
        current_cases, current_records, current_duplicates = _junit_records(
            current_bytes
        )
    except (ET.ParseError, KeyError, OSError, TypeError, ValueError) as exc:
        return {
            "schema_version": "xunce-path-v2-gate2-full-junit-audit/v1",
            "status": "failed",
            "total": _empty_counts(),
            "baseline": baseline,
            "legacy": _empty_counts(),
            "v2": _empty_counts(),
            "baseline_evidence": evidence_review,
            "baseline_not_reduced": False,
            "skip_contract": False,
            "v2_green": False,
            "missing_baseline_nodeids": [],
            "drifted_baseline_nodeids": [],
            "duplicate_baseline_nodeids": [],
            "duplicate_current_nodeids": [],
            "error_type": type(exc).__name__,
        }

    baseline_source_counts = _counts_for_records(tuple(baseline_records.values()))
    baseline_evidence_valid = (
        evidence_review["actual_sha256"] == expected_hash
        and len(baseline_cases) == expected_test_count
        and not baseline_duplicates
        and baseline_source_counts == baseline
    )
    evidence_review["status"] = "passed" if baseline_evidence_valid else "failed"
    baseline_nodeids = set(baseline_records)
    current_nodeids = set(current_records)
    missing = sorted(baseline_nodeids - current_nodeids)
    drifted = sorted(
        nodeid
        for nodeid in baseline_nodeids & current_nodeids
        if baseline_records[nodeid]["outcome"] != current_records[nodeid]["outcome"]
    )
    legacy_records = [
        current_records[nodeid]
        for nodeid in sorted(baseline_nodeids & current_nodeids)
    ]
    added_records = [
        current_records[nodeid]
        for nodeid in sorted(current_nodeids - baseline_nodeids)
    ]
    legacy = _counts_for_records(legacy_records)
    v2 = _counts_for_records(added_records)
    total = _counts_for_records(
        tuple({"outcome": _case_outcome(testcase)} for testcase in current_cases)
    )
    skip_messages = [
        current_records[nodeid]["skip_message"]
        for nodeid in sorted(baseline_nodeids & current_nodeids)
        if current_records[nodeid]["outcome"] == "skipped"
    ]
    dependency = str(allowed_skip_dependency).lower()
    skip_contract = (
        len(skip_messages) == baseline["skipped"]
        and all(dependency in message.lower() for message in skip_messages)
    )
    baseline_not_reduced = (
        baseline_evidence_valid
        and not missing
        and not drifted
        and not current_duplicates
        and legacy == baseline
    )
    v2_green = (
        not current_duplicates
        and v2["failures"] == 0
        and v2["errors"] == 0
        and v2["skipped"] == 0
    )
    passed = baseline_not_reduced and skip_contract and v2_green
    return {
        "schema_version": "xunce-path-v2-gate2-full-junit-audit/v1",
        "status": "passed" if passed else "failed",
        "total": total,
        "baseline": baseline,
        "legacy": legacy,
        "v2": v2,
        "baseline_evidence": evidence_review,
        "baseline_not_reduced": baseline_not_reduced,
        "skip_contract": skip_contract,
        "v2_green": v2_green,
        "skip_messages": skip_messages,
        "missing_baseline_nodeids": missing,
        "drifted_baseline_nodeids": drifted,
        "duplicate_baseline_nodeids": baseline_duplicates,
        "duplicate_current_nodeids": current_duplicates,
    }


def _gate2_runtime_audit(repo_root: Path) -> dict[str, Any]:
    superproject = gate0.audit_git_identity(
        repo_root,
        EXPECTED_BRANCH,
        ORIGINAL_BASE_COMMIT,
    )
    gate_input = gate0.audit_git_identity(
        repo_root,
        EXPECTED_BRANCH,
        GATE2_INPUT_COMMIT,
    )
    nested = _audit_nested_git(repo_root, EXPECTED_BRANCH)
    imports = gate0.audit_import_origins(FORMAL_PYTHON, repo_root)
    python_version_matches = imports.get("python_version") == EXPECTED_PYTHON_VERSION
    original_base_is_ancestor = (
        superproject.get("status") == "passed"
        and superproject.get("base_is_ancestor") is True
    )
    gate_input_is_ancestor = (
        gate_input.get("status") == "passed"
        and gate_input.get("base_is_ancestor") is True
    )
    passed = (
        original_base_is_ancestor
        and gate_input_is_ancestor
        and nested.get("status") == "passed"
        and imports.get("status") == "passed"
        and python_version_matches
    )
    return {
        "schema_version": "xunce-path-v2-gate2-runtime-audit/v1",
        "status": "passed" if passed else "failed",
        "superproject_git": superproject,
        "gate_input_git": gate_input,
        "nested_git": nested,
        "import_origins": imports,
        "python_version_matches": python_version_matches,
        "original_base_is_ancestor": original_base_is_ancestor,
        "gate_input_is_ancestor": gate_input_is_ancestor,
    }


def _audit_gate3_full_junit(
    path: Path,
    *,
    expected_legacy: dict[str, Any],
    allowed_skip_dependency: str,
    baseline_evidence: dict[str, Any],
) -> dict[str, Any]:
    audit = _audit_gate2_full_junit(
        path,
        expected_legacy=expected_legacy,
        allowed_skip_dependency=allowed_skip_dependency,
        baseline_evidence=baseline_evidence,
    )
    return {
        **audit,
        "schema_version": "xunce-path-v2-gate3-full-junit-audit/v1",
    }


def _audit_gate4_full_junit(
    path: Path,
    *,
    expected_legacy: dict[str, Any],
    allowed_skip_dependency: str,
    baseline_evidence: dict[str, Any],
) -> dict[str, Any]:
    audit = _audit_gate2_full_junit(
        path,
        expected_legacy=expected_legacy,
        allowed_skip_dependency=allowed_skip_dependency,
        baseline_evidence=baseline_evidence,
    )
    return {
        **audit,
        "schema_version": "xunce-path-v2-gate4-full-junit-audit/v1",
    }


def _gate3_runtime_audit(repo_root: Path) -> dict[str, Any]:
    superproject = gate0.audit_git_identity(
        repo_root,
        EXPECTED_BRANCH,
        ORIGINAL_BASE_COMMIT,
    )
    gate_input = gate0.audit_git_identity(
        repo_root,
        EXPECTED_BRANCH,
        GATE3_INPUT_COMMIT,
    )
    nested = _audit_nested_git(repo_root, EXPECTED_BRANCH)
    imports = gate0.audit_import_origins(FORMAL_PYTHON, repo_root)
    python_version_matches = imports.get("python_version") == EXPECTED_PYTHON_VERSION
    original_base_is_ancestor = (
        superproject.get("status") == "passed"
        and superproject.get("base_is_ancestor") is True
    )
    gate_input_is_ancestor = (
        gate_input.get("status") == "passed"
        and gate_input.get("base_is_ancestor") is True
    )
    passed = (
        original_base_is_ancestor
        and gate_input_is_ancestor
        and nested.get("status") == "passed"
        and imports.get("status") == "passed"
        and python_version_matches
    )
    return {
        "schema_version": "xunce-path-v2-gate3-runtime-audit/v1",
        "status": "passed" if passed else "failed",
        "superproject_git": superproject,
        "gate_input_git": gate_input,
        "nested_git": nested,
        "import_origins": imports,
        "python_version_matches": python_version_matches,
        "original_base_is_ancestor": original_base_is_ancestor,
        "gate_input_is_ancestor": gate_input_is_ancestor,
    }


def _gate4_runtime_audit(repo_root: Path) -> dict[str, Any]:
    superproject = gate0.audit_git_identity(
        repo_root,
        EXPECTED_BRANCH,
        ORIGINAL_BASE_COMMIT,
    )
    gate_input = gate0.audit_git_identity(
        repo_root,
        EXPECTED_BRANCH,
        GATE4_INPUT_COMMIT,
    )
    nested = _audit_nested_git(repo_root, EXPECTED_BRANCH)
    imports = gate0.audit_import_origins(FORMAL_PYTHON, repo_root)
    python_version_matches = imports.get("python_version") == EXPECTED_PYTHON_VERSION
    original_base_is_ancestor = (
        superproject.get("status") == "passed"
        and superproject.get("base_is_ancestor") is True
    )
    gate_input_is_ancestor = (
        gate_input.get("status") == "passed"
        and gate_input.get("base_is_ancestor") is True
    )
    passed = (
        original_base_is_ancestor
        and gate_input_is_ancestor
        and nested.get("status") == "passed"
        and imports.get("status") == "passed"
        and python_version_matches
    )
    return {
        "schema_version": "xunce-path-v2-gate4-runtime-audit/v1",
        "status": "passed" if passed else "failed",
        "superproject_git": superproject,
        "gate_input_git": gate_input,
        "nested_git": nested,
        "import_origins": imports,
        "python_version_matches": python_version_matches,
        "original_base_is_ancestor": original_base_is_ancestor,
        "gate_input_is_ancestor": gate_input_is_ancestor,
    }


def _gate4_preflight(config: dict[str, Any], repo_root: Path) -> dict[str, Any]:
    return _gate4_runtime_audit(repo_root)


def _gate4_postflight(config: dict[str, Any], repo_root: Path) -> dict[str, Any]:
    return _gate4_runtime_audit(repo_root)


def _gate3_preflight(config: dict[str, Any], repo_root: Path) -> dict[str, Any]:
    return _gate3_runtime_audit(repo_root)


def _gate3_postflight(config: dict[str, Any], repo_root: Path) -> dict[str, Any]:
    return _gate3_runtime_audit(repo_root)


def _gate2_preflight(config: dict[str, Any], repo_root: Path) -> dict[str, Any]:
    return _gate2_runtime_audit(repo_root)


def _gate2_postflight(config: dict[str, Any], repo_root: Path) -> dict[str, Any]:
    return _gate2_runtime_audit(repo_root)


def _runtime_audit(repo_root: Path) -> dict[str, Any]:
    superproject = gate0.audit_git_identity(
        repo_root,
        EXPECTED_BRANCH,
        ORIGINAL_BASE_COMMIT,
    )
    gate_input = gate0.audit_git_identity(
        repo_root,
        EXPECTED_BRANCH,
        GATE_INPUT_COMMIT,
    )
    nested = _audit_nested_git(repo_root, EXPECTED_BRANCH)
    imports = gate0.audit_import_origins(FORMAL_PYTHON, repo_root)
    python_version_matches = imports.get("python_version") == EXPECTED_PYTHON_VERSION
    original_base_is_ancestor = (
        superproject.get("status") == "passed"
        and superproject.get("base_is_ancestor") is True
    )
    gate_input_is_ancestor = (
        gate_input.get("status") == "passed"
        and gate_input.get("base_is_ancestor") is True
    )
    passed = (
        original_base_is_ancestor
        and gate_input_is_ancestor
        and nested.get("status") == "passed"
        and imports.get("status") == "passed"
        and python_version_matches
    )
    return {
        "schema_version": "xunce-path-v2-gate1-preflight/v1",
        "status": "passed" if passed else "failed",
        "superproject_git": superproject,
        "gate_input_git": gate_input,
        "nested_git": nested,
        "import_origins": imports,
        "python_version_matches": python_version_matches,
        "original_base_is_ancestor": original_base_is_ancestor,
        "gate_input_is_ancestor": gate_input_is_ancestor,
    }


def _preflight(config: dict[str, Any], repo_root: Path) -> dict[str, Any]:
    return _runtime_audit(repo_root)


def _postflight(config: dict[str, Any], repo_root: Path) -> dict[str, Any]:
    return _runtime_audit(repo_root)


def _selected_identity(audit: dict[str, Any]) -> dict[str, Any]:
    superproject = audit.get("superproject_git", {})
    gate_input = audit.get("gate_input_git", {})
    nested = audit.get("nested_git", {})
    imports = audit.get("import_origins", {})
    return {
        "superproject": {
            key: superproject.get(key)
            for key in ("head", "branch", "git_dir", "git_common_dir")
        },
        "gate_input": {
            key: gate_input.get(key)
            for key in ("head", "branch", "git_dir", "git_common_dir")
        },
        "nested": {
            key: nested.get(key)
            for key in ("head", "branch", "gitlink")
        },
        "imports": {
            "python": imports.get("python"),
            "python_version": imports.get("python_version"),
            "python_no_user_site": imports.get("python_no_user_site"),
            "pythonpath": imports.get("pythonpath"),
            "path_planner_origin": imports.get("path_planner", {}).get("origin"),
            "ppo_origin": imports.get("lunar_exploration_ppo", {}).get("origin"),
            "path_planner_from_worktree": imports.get("path_planner_from_worktree"),
            "ppo_from_worktree": imports.get("ppo_from_worktree"),
        },
    }


def _postflight_matches(
    preflight: dict[str, Any],
    postflight: dict[str, Any],
) -> bool:
    superproject = postflight.get("superproject_git", {})
    gate_input = postflight.get("gate_input_git", {})
    nested = postflight.get("nested_git", {})
    imports = postflight.get("import_origins", {})
    return (
        postflight.get("status") == "passed"
        and superproject.get("status") == "passed"
        and superproject.get("clean_tree") is True
        and gate_input.get("status") == "passed"
        and gate_input.get("clean_tree") is True
        and nested.get("status") == "passed"
        and nested.get("clean_tree") is True
        and nested.get("head_matches_gitlink") is True
        and imports.get("status") == "passed"
        and imports.get("python_no_user_site") is True
        and postflight.get("python_version_matches") is True
        and postflight.get("original_base_is_ancestor") is True
        and postflight.get("gate_input_is_ancestor") is True
        and _selected_identity(preflight) == _selected_identity(postflight)
    )


def _common_env(repo_root: Path, attempt_root: Path) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
            "PYTHONPATH": str((repo_root / "path-planner" / "src").resolve()),
            "TEMP": str(attempt_root),
            "TMP": str(attempt_root),
            "MPLCONFIGDIR": str(attempt_root / "mpl"),
        }
    )
    return env


def _gate4_benchmark_contract_not_run() -> dict[str, Any]:
    return {
        "schema_version": "xunce-path-v2-gate4-benchmark-contract-audit/v1",
        "status": "not_run",
        "hard_timeout_ms": GATE4_THRESHOLDS["hard_timeout_ms"],
        "typed_row_api": False,
    }


def _audit_gate4_benchmark_contract(repo_root: Path) -> dict[str, Any]:
    base = {
        "schema_version": "xunce-path-v2-gate4-benchmark-contract-audit/v1",
        "status": "failed",
        "hard_timeout_ms": GATE4_THRESHOLDS["hard_timeout_ms"],
        "typed_row_api": False,
        "repair_route": "repair_gate4_benchmark_contract",
    }
    try:
        nested_src = str((Path(repo_root) / "path-planner" / "src").resolve())
        if not sys.path or sys.path[0] != nested_src:
            sys.path.insert(0, nested_src)
        benchmark = importlib.import_module("path_planner.v2.benchmark")
        expected_origin = (
            Path(repo_root)
            / "path-planner"
            / "src"
            / "path_planner"
            / "v2"
            / "benchmark.py"
        ).resolve()
        actual_origin = Path(str(getattr(benchmark, "__file__", ""))).resolve()
        if actual_origin != expected_origin:
            raise ImportError("Gate 4 benchmark module origin is outside the worktree")
        hard_timeout_ms = getattr(benchmark, "HARD_TIMEOUT_MS_V2")
        if type(hard_timeout_ms) is not float or hard_timeout_ms != GATE4_THRESHOLDS[
            "hard_timeout_ms"
        ]:
            raise TypeError("Gate 4 benchmark hard timeout contract drifted")
        for name in GATE4_BENCHMARK_ROW_CLASSES:
            row_type = getattr(benchmark, name)
            if (
                type(row_type) is not type
                or row_type.__name__ != name
                or row_type.__module__ != "path_planner.v2.benchmark"
            ):
                raise TypeError(f"Gate 4 benchmark row contract drifted: {name}")
        for name in GATE4_BENCHMARK_AGGREGATES:
            if not callable(getattr(benchmark, name)):
                raise TypeError(f"Gate 4 benchmark aggregate contract drifted: {name}")
    except Exception as exc:
        return {**base, "error_type": type(exc).__name__}
    return {
        "schema_version": "xunce-path-v2-gate4-benchmark-contract-audit/v1",
        "status": "passed",
        "hard_timeout_ms": GATE4_THRESHOLDS["hard_timeout_ms"],
        "typed_row_api": True,
    }


def _gate4_dataset_status(
    *,
    dataset_name: str,
    path: Path,
    trusted_input: dict[str, Any],
    formal_source: str,
) -> dict[str, Any]:
    input_path = Path(path)
    return {
        "dataset_name": dataset_name,
        "path": input_path.as_posix(),
        "status": "untrusted" if artifact_io.path_is_file(input_path) else "missing",
        "content_read": False,
        "formal_source": formal_source,
        "trusted_input": dict(trusted_input),
    }


def _run_pytest(
    *,
    python: Path,
    repo_root: Path,
    targets: Sequence[str],
    junit_path: Path,
    basetemp: Path,
    env: dict[str, str],
) -> dict[str, Any]:
    artifact_io.write_text(
        junit_path,
        '<?xml version="1.0" encoding="utf-8"?><testsuite tests="0" />',
    )
    command = [str(python), "-m", "pytest", "-p", "no:cacheprovider", "-q"]
    command.extend(str(target) for target in targets)
    command.extend(("--basetemp", str(basetemp), "--junitxml", str(junit_path)))
    completed = subprocess.run(
        command,
        cwd=repo_root / "path-planner",
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return {
        "command": command,
        "returncode": int(completed.returncode),
        "stdout_tail": completed.stdout[-8000:],
        "stderr_tail": completed.stderr[-8000:],
    }


def _run_byte_repeats(
    *,
    python: Path,
    repo_root: Path,
    seeds: Sequence[Any],
    common_env: dict[str, str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for repeat, raw_seed in enumerate(seeds, start=1):
        seed = str(raw_seed)
        env = common_env.copy()
        env["PYTHONHASHSEED"] = seed
        completed = subprocess.run(
            [str(python), "-c", BYTE_PROBE_CODE],
            cwd=repo_root / "path-planner",
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        payload: dict[str, Any] = {}
        if completed.returncode == 0:
            try:
                payload = json.loads(completed.stdout.strip().splitlines()[-1])
            except (IndexError, json.JSONDecodeError, TypeError, ValueError):
                payload = {}
        rows.append(
            {
                "suite": "byte-repeat",
                "check": "canonical_unknown_profile",
                "repeat": repeat,
                "python_hash_seed": seed,
                "returncode": int(completed.returncode),
                "digest": payload.get("digest"),
                "canonical_hex": payload.get("canonical_hex"),
                "root_has_plan_v2": payload.get("root_has_plan_v2"),
                "v1_route_schema": payload.get("v1_route_schema"),
                "stderr_tail": completed.stderr[-2000:],
            }
        )
    return rows


def _run_gate3_probe_process(
    *,
    python: Path,
    repo_root: Path,
    worker_count: int,
    hash_seed: int,
    repeat: int,
    common_env: dict[str, str],
    timeout_s: float = GATE3_PROBE_SUBPROCESS_TIMEOUT_S,
) -> dict[str, Any]:
    env = common_env.copy()
    env.update(
        {
            "PYTHONHASHSEED": str(hash_seed),
            "PATH_V2_GATE3_CASES": json.dumps(
                list(GATE3_CASES),
                separators=(",", ":"),
            ),
            "PATH_V2_GATE3_WORKER_COUNT": str(worker_count),
            "PATH_V2_GATE3_REPEAT": str(repeat),
        }
    )
    command = [str(python), "-c", GATE3_PROBE_CODE]
    rows: list[dict[str, Any]] = []
    stable_failure_reason: str | None = None
    try:
        completed = subprocess.run(
            command,
            cwd=repo_root / "path-planner",
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        returncode = GATE3_PROBE_TIMEOUT_RETURNCODE
        stable_failure_reason = "probe_subprocess_timeout"
    else:
        returncode = int(completed.returncode)
        if completed.returncode != 0:
            stable_failure_reason = "probe_subprocess_failed"
        else:
            try:
                payload = json.loads(completed.stdout.strip().splitlines()[-1])
                raw_rows = payload["rows"]
                if (
                    type(payload) is not dict
                    or type(raw_rows) is not list
                    or len(raw_rows) != len(GATE3_CASES)
                    or any(type(row) is not dict for row in raw_rows)
                    or [row.get("case_id") for row in raw_rows]
                    != list(GATE3_CASES)
                ):
                    raise ValueError("invalid Gate 3 probe rows")
                for raw in raw_rows:
                    row = dict(raw)
                    row.update(
                        worker_count=worker_count,
                        python_hash_seed=hash_seed,
                        repeat=repeat,
                    )
                    rows.append(row)
            except (
                IndexError,
                KeyError,
                TypeError,
                ValueError,
                json.JSONDecodeError,
            ):
                stable_failure_reason = "probe_output_invalid"
                rows = []
    return {
        "command": command,
        "environment": {
            "PYTHONHASHSEED": str(hash_seed),
            "PATH_V2_GATE3_WORKER_COUNT": str(worker_count),
            "PATH_V2_GATE3_REPEAT": str(repeat),
            "PYTHONNOUSERSITE": env.get("PYTHONNOUSERSITE"),
            "PYTHONDONTWRITEBYTECODE": env.get("PYTHONDONTWRITEBYTECODE"),
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": env.get(
                "PYTEST_DISABLE_PLUGIN_AUTOLOAD"
            ),
            "PYTHONPATH": env.get("PYTHONPATH"),
        },
        "worker_count": worker_count,
        "python_hash_seed": hash_seed,
        "repeat": repeat,
        "returncode": returncode,
        "stable_failure_reason": stable_failure_reason,
        "rows": rows,
    }


def _failed_gate3_probe_rows(
    *,
    worker_count: int,
    hash_seed: int,
    repeat: int,
    reason: str,
) -> list[dict[str, Any]]:
    return [
        {
            "case_id": case_id,
            "worker_count": worker_count,
            "python_hash_seed": hash_seed,
            "repeat": repeat,
            "status": "failed",
            "decision_digest": None,
            "fine_only_digest": None,
            "safety_equivalent": False,
            "authoritative_order_preserved": False,
            "suggestion_non_authoritative": False,
            "hierarchy_conservative": False,
            "cache_l2_equivalent": False,
            "l2_authority_preserved": False,
            "fallback_isolated": False,
            "fatal_reason": reason,
            "accelerator_used": None,
            "runtime_disabled_accelerators": [],
        }
        for case_id in GATE3_CASES
    ]


def _run_gate3_probes(
    *,
    python: Path,
    repo_root: Path,
    common_env: dict[str, str],
    timeout_s: float = GATE3_PROBE_SUBPROCESS_TIMEOUT_S,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    commands: list[dict[str, Any]] = []
    for worker_count in GATE3_WORKER_COUNTS:
        for hash_seed in GATE3_HASH_SEEDS:
            for repeat in range(1, GATE3_REPEAT_COUNT + 1):
                result = _run_gate3_probe_process(
                    python=python,
                    repo_root=repo_root,
                    worker_count=worker_count,
                    hash_seed=hash_seed,
                    repeat=repeat,
                    common_env=common_env,
                    timeout_s=timeout_s,
                )
                commands.append(
                    {
                        key: result[key]
                        for key in (
                            "command",
                            "environment",
                            "worker_count",
                            "python_hash_seed",
                            "repeat",
                            "returncode",
                            "stable_failure_reason",
                        )
                    }
                )
                if result["stable_failure_reason"] is None:
                    rows.extend(result["rows"])
                else:
                    rows.extend(
                        _failed_gate3_probe_rows(
                            worker_count=worker_count,
                            hash_seed=hash_seed,
                            repeat=repeat,
                            reason=result["stable_failure_reason"],
                        )
                    )
    case_order = {case_id: index for index, case_id in enumerate(GATE3_CASES)}
    rows.sort(
        key=lambda row: (
            case_order.get(row.get("case_id"), len(case_order)),
            row.get("worker_count", -1),
            row.get("python_hash_seed", -1),
            row.get("repeat", -1),
        )
    )
    audit = _audit_gate3_probe_rows(rows)
    return {**audit, "rows": rows, "commands": commands}


def _gate3_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _audit_gate3_probe_rows(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    expected_keys = [
        (case_id, worker_count, hash_seed, repeat)
        for case_id in GATE3_CASES
        for worker_count in GATE3_WORKER_COUNTS
        for hash_seed in GATE3_HASH_SEEDS
        for repeat in range(1, GATE3_REPEAT_COUNT + 1)
    ]
    actual_keys: list[tuple[object, object, object, object]] = []
    row_shapes_valid = type(rows) in (list, tuple)
    matrix_keys_valid = row_shapes_valid
    normalized_rows: list[dict[str, Any]] = []
    if row_shapes_valid:
        for row in rows:
            if type(row) is not dict:
                row_shapes_valid = False
                matrix_keys_valid = False
                break
            normalized_rows.append(row)
            key = (
                row.get("case_id"),
                row.get("worker_count"),
                row.get("python_hash_seed"),
                row.get("repeat"),
            )
            actual_keys.append(key)
            if not (
                type(key[0]) is str
                and type(key[1]) is int
                and type(key[2]) is int
                and type(key[3]) is int
            ):
                matrix_keys_valid = False
    matrix_complete = (
        row_shapes_valid
        and matrix_keys_valid
        and len(actual_keys) == len(expected_keys)
        and set(actual_keys) == set(expected_keys)
    )
    stable_row_order = (
        row_shapes_valid and matrix_keys_valid and actual_keys == expected_keys
    )

    enabled_by_case = {
        "fine_only": set(),
        "multi_heuristic_only": {"multi_heuristic"},
        "hierarchy_only": {"hierarchy"},
        "lazy_validation_only": {"lazy_validation"},
        "lazy_validation_plus_cache": {"lazy_validation", "validation_cache"},
        "full_v2": {
            "hierarchy",
            "lazy_validation",
            "multi_heuristic",
            "validation_cache",
        },
    }
    fallback_isolation = row_shapes_valid
    runtime_fallback_count = 0
    fatal_reasons: set[str] = set()
    for row in normalized_rows:
        runtime_disabled = row.get("runtime_disabled_accelerators")
        if type(runtime_disabled) is not list:
            fallback_isolation = False
            continue
        ids: list[str] = []
        for item in runtime_disabled:
            valid_item = (
                type(item) is dict
                and set(item) == {"accelerator_id", "reason"}
                and type(item.get("accelerator_id")) is str
                and item.get("reason") == "component_probe_failed"
            )
            if not valid_item:
                fallback_isolation = False
                continue
            ids.append(item["accelerator_id"])
        enabled = enabled_by_case.get(row.get("case_id"), set())
        if ids != sorted(set(ids)) or not set(ids) <= enabled:
            fallback_isolation = False
        if row.get("fallback_isolated") is not True:
            fallback_isolation = False
        runtime_fallback_count += len(ids)
        fatal_reason = row.get("fatal_reason")
        if type(fatal_reason) is str:
            fatal_reasons.add(fatal_reason)

    digests = [row.get("decision_digest") for row in normalized_rows]
    one_decision_digest = (
        bool(digests)
        and all(_gate3_sha256(digest) for digest in digests)
        and len(set(digests)) == 1
        and all(row.get("fine_only_digest") == digests[0] for row in normalized_rows)
    )
    safety_equivalence = bool(normalized_rows) and all(
        row.get("status") == "passed"
        and row.get("safety_equivalent") is True
        and row.get("authoritative_order_preserved") is True
        and row.get("l2_authority_preserved") is True
        for row in normalized_rows
    )
    suggestion_non_authority = bool(normalized_rows) and all(
        row.get("suggestion_non_authoritative") is True for row in normalized_rows
    )
    hierarchy_conservatism = bool(normalized_rows) and all(
        row.get("hierarchy_conservative") is True for row in normalized_rows
    )
    cache_l2_equivalence = bool(normalized_rows) and all(
        row.get("cache_l2_equivalent") is True for row in normalized_rows
    )
    fatal_authority_clean = bool(normalized_rows) and all(
        row.get("fatal_reason") is None for row in normalized_rows
    )
    provider_accelerator_unused = bool(normalized_rows) and all(
        row.get("accelerator_used") is False for row in normalized_rows
    )
    checks = {
        "matrix_complete": matrix_complete,
        "stable_row_order": stable_row_order,
        "one_decision_digest": one_decision_digest,
        "safety_equivalence": safety_equivalence,
        "suggestion_non_authority": suggestion_non_authority,
        "hierarchy_conservatism": hierarchy_conservatism,
        "cache_l2_equivalence": cache_l2_equivalence,
        "fallback_isolation": fallback_isolation,
        "fatal_authority_clean": fatal_authority_clean,
        "provider_accelerator_unused": provider_accelerator_unused,
    }
    passed = all(checks.values())
    return {
        "schema_version": "xunce-path-v2-gate3-probe-audit/v1",
        "status": "passed" if passed else "failed",
        "row_count": len(normalized_rows),
        "decision_digest": digests[0] if one_decision_digest else None,
        "runtime_fallback_count": runtime_fallback_count,
        "fatal_reasons": sorted(fatal_reasons),
        **checks,
    }


def _report(summary: dict[str, Any]) -> str:
    full = summary["full"]
    legacy = full.get("legacy", _empty_counts())
    return (
        "# Path Planner v2 Gate 1 合同报告\n\n"
        f"- Gate 状态：`{summary['status']}`\n"
        f"- focused：`{summary['focused'].get('status', 'not_run')}`\n"
        f"- legacy 分账：{legacy.get('passed', 0)} passed / "
        f"{legacy.get('skipped', 0)} skipped / {legacy.get('failures', 0)} failed / "
        f"{legacy.get('errors', 0)} errors\n"
        f"- byte repeat：`{summary['byte_repeat'].get('status', 'not_run')}`\n"
        f"- 下一步：`{summary['next_required_change']}`\n\n"
        "本 Gate 只冻结 profile、provider、API 与安全边界合同；不发布 checkpoint、"
        "不替换默认策略、不连接真实 executor、不启动 online canary。\n"
    )


def _dry_run_payloads(config: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    boundary_ok = boundaries_match(config.get("boundaries"))
    status = "dry_run" if boundary_ok else "failed"
    next_change = (
        "execute_gate1_contract" if boundary_ok else "restore_gate1_safety_boundaries"
    )
    preflight = {"status": "not_run"}
    postflight = {"status": "not_run"}
    focused = {"status": "not_run", "tests": 0, "passed": 0, "skipped": 0, "failures": 0, "errors": 0}
    full = {
        "status": "not_run",
        "total": _empty_counts(),
        "legacy": _empty_counts(),
        "v2": _empty_counts(),
    }
    byte_repeat = {"status": "not_run", "repeat_count": 0, "digests": []}
    summary = {
        "schema_version": SCHEMA_VERSION,
        "stage_id": STAGE_ID,
        "status": status,
        "next_required_change": next_change,
        "preflight": preflight,
        "postflight": postflight,
        "checks": {
            "preflight": False,
            "postflight": False,
            "focused": False,
            "full": False,
            "byte_repeat": False,
            "boundaries_strict_false": boundary_ok,
        },
        "focused": focused,
        "full": full,
        "byte_repeat": byte_repeat,
        **gate_artifacts.BOUNDARY_FIELDS,
    }
    rows = [
        {"suite": "boundary-review", "check": field, "status": "passed" if boundary_ok else "failed", "value": False}
        for field in sorted(gate_artifacts.BOUNDARY_FIELDS)
    ]
    phases = [
        {"phase": phase, "status": "not_run" if phase != "boundary-review" else ("completed" if boundary_ok else "failed")}
        for phase in ("preflight", "focused", "full", "byte-repeat", "boundary-review")
    ]
    review = {
        "schema_version": "xunce-path-v2-gate1-review/v1",
        "status": "dry_run" if boundary_ok else "failed",
        "checks": {"boundaries_strict_false": boundary_ok},
        "preflight": preflight,
        "postflight": postflight,
    }
    routing = {
        "schema_version": "xunce-path-v2-gate1-routing/v1",
        "stage_id": STAGE_ID,
        "status": status,
        "route": next_change,
        **gate_artifacts.BOUNDARY_FIELDS,
    }
    return summary, routing, rows, phases, review


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in pairs:
        if key in payload:
            raise ValueError(f"duplicate JSON object key: {key}")
        payload[key] = value
    return payload


def _load_gate2_dataset(
    *,
    dataset_name: str,
    path: Path,
    repo_root: Path,
) -> dict[str, Any]:
    input_path = Path(path)
    base = {
        "dataset_name": dataset_name,
        "input_path": input_path.as_posix(),
        "rows": [],
        "summary": None,
        "input_sha256": None,
        "rows_sha256": None,
    }
    if not artifact_io.path_is_file(input_path):
        return {**base, "status": "missing"}

    try:
        nested_src = str((Path(repo_root) / "path-planner" / "src").resolve())
        if not sys.path or sys.path[0] != nested_src:
            sys.path.insert(0, nested_src)
        benchmark = importlib.import_module("path_planner.v2.benchmark")
        expected_origin = (
            Path(repo_root) / "path-planner" / "src" / "path_planner" / "v2" / "benchmark.py"
        ).resolve()
        actual_origin = Path(str(getattr(benchmark, "__file__"))).resolve()
        if actual_origin != expected_origin:
            raise ImportError("Gate 2 benchmark module origin is outside the worktree")
        hard_timeout_ms = getattr(benchmark, "HARD_TIMEOUT_MS_V2")
        if (
            isinstance(hard_timeout_ms, bool)
            or not isinstance(hard_timeout_ms, (int, float))
            or float(hard_timeout_ms) != GATE2_THRESHOLDS["hard_timeout_ms"]
        ):
            raise _Gate2LoaderContractError(
                "Gate 2 benchmark hard timeout contract drifted"
            )
        row_type_name, aggregate_name = GATE2_ROW_API[dataset_name]
        row_type = getattr(benchmark, row_type_name)
        aggregate = getattr(benchmark, aggregate_name)
        input_bytes = artifact_io.read_bytes(input_path)
        base["input_sha256"] = hashlib.sha256(input_bytes).hexdigest()
        payloads: list[dict[str, Any]] = []
        for line_number, line in enumerate(
            input_bytes.decode("utf-8").splitlines(),
            start=1,
        ):
            if not line.strip():
                raise ValueError(
                    f"Gate 2 JSONL row {line_number} must not be blank"
                )
            payload = json.loads(line, object_pairs_hook=_reject_duplicate_json_keys)
            if not isinstance(payload, dict):
                raise ValueError(
                    f"Gate 2 JSONL row {line_number} must be an object"
                )
            payloads.append(payload)
        parsed = tuple(row_type.from_dict(payload) for payload in payloads)
        aggregate_summary = aggregate(parsed)
        rows = [row.to_dict() for row in aggregate_summary.rows]
        summary = aggregate_summary.to_dict()
        canonical_rows = json.dumps(
            rows,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return {
            **base,
            "status": "loaded",
            "rows": rows,
            "summary": summary,
            "input_sha256": hashlib.sha256(input_bytes).hexdigest(),
            "rows_sha256": hashlib.sha256(canonical_rows).hexdigest(),
        }
    except (AttributeError, ImportError, KeyError, _Gate2LoaderContractError) as exc:
        return {
            **base,
            "status": "internal_error",
            "error_type": type(exc).__name__,
        }
    except (
        TypeError,
        ValueError,
        OverflowError,
        OSError,
        UnicodeError,
        json.JSONDecodeError,
        RecursionError,
    ) as exc:
        return {
            **base,
            "status": "invalid",
            "error_type": type(exc).__name__,
        }


def _gate2_formal_count(dataset_name: str, dataset: dict[str, Any]) -> int:
    summary = dataset.get("summary")
    if not isinstance(summary, dict):
        return 0
    field = (
        "formal_episode_count"
        if dataset_name == "standard_episodes"
        else "formal_row_count"
    )
    value = summary.get(field)
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0


def _gate2_number(summary: Any, field: str) -> float | int | None:
    if not isinstance(summary, dict):
        return None
    value = summary.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value


def _gate2_exact_cost_ratios(summary: Any) -> tuple[float, ...] | None:
    if not isinstance(summary, dict):
        return None
    raw_ratios = summary.get("resource_cost_ratios")
    if not isinstance(raw_ratios, list):
        return None
    ratios: list[float] = []
    for item in raw_ratios:
        if not isinstance(item, dict) or set(item) != {"row_id", "ratio"}:
            return None
        row_id = item.get("row_id")
        ratio = item.get("ratio")
        if not isinstance(row_id, str) or not row_id:
            return None
        if isinstance(ratio, bool) or not isinstance(ratio, (int, float)):
            return None
        try:
            normalized_ratio = float(ratio)
        except OverflowError:
            return None
        if not math.isfinite(normalized_ratio):
            return None
        ratios.append(normalized_ratio)
    return tuple(ratios)


def _evaluate_gate2(
    *,
    preflight: dict[str, Any],
    postflight_ok: bool,
    focused: dict[str, Any],
    full: dict[str, Any],
    boundary_ok: bool,
    datasets: dict[str, dict[str, Any]],
) -> tuple[str, str, list[str], dict[str, bool]]:
    primitive = datasets["primitive_audit"]
    exact = datasets["exact_map_quality"]
    standard = datasets["standard_episodes"]
    primitive_summary = primitive.get("summary")
    exact_summary = exact.get("summary")
    standard_summary = standard.get("summary")
    exact_cost_ratios = _gate2_exact_cost_ratios(exact_summary)

    primitive_ready = (
        primitive.get("status") == "loaded"
        and _gate2_formal_count("primitive_audit", primitive)
        >= GATE2_THRESHOLDS["min_primitive_independent_samples"]
    )
    exact_ready = (
        exact.get("status") == "loaded"
        and _gate2_formal_count("exact_map_quality", exact)
        >= GATE2_THRESHOLDS["min_exact_map_independent_cases"]
    )
    standard_ready = (
        standard.get("status") == "loaded"
        and _gate2_formal_count("standard_episodes", standard)
        >= GATE2_THRESHOLDS["min_standard_independent_episodes"]
    )

    checks = {
        "preflight": preflight.get("status") == "passed",
        "postflight": postflight_ok,
        "focused": focused.get("status") == "passed",
        "full": full.get("status") == "passed",
        "benchmark_loader": all(
            dataset.get("status") != "internal_error"
            for dataset in datasets.values()
        ),
        "boundaries_strict_false": boundary_ok,
        "primitive_formal_denominator": primitive_ready,
        "exact_map_formal_denominator": exact_ready,
        "standard_formal_denominator": standard_ready,
        "primitive_false_positives": (
            primitive_ready
            and _gate2_number(primitive_summary, "false_positive_count") is not None
            and _gate2_number(primitive_summary, "false_positive_count")
            <= GATE2_THRESHOLDS["max_primitive_false_positives"]
        ),
        "primitive_recall": (
            primitive_ready
            and _gate2_number(primitive_summary, "primitive_recall") is not None
            and _gate2_number(primitive_summary, "primitive_recall")
            >= GATE2_THRESHOLDS["min_primitive_recall"]
        ),
        "primitive_complete_l2": (
            primitive_ready
            and _gate2_number(primitive_summary, "complete_l2_ratio")
            == GATE2_THRESHOLDS["min_primitive_complete_l2_ratio"]
        ),
        "exact_map_success": (
            exact_ready
            and _gate2_number(exact_summary, "provider_success_ratio") is not None
            and _gate2_number(exact_summary, "provider_success_ratio")
            >= GATE2_THRESHOLDS["min_exact_map_success_ratio"]
        ),
        "exact_map_complete_l2": (
            exact_ready
            and _gate2_number(exact_summary, "complete_l2_ratio")
            == GATE2_THRESHOLDS["min_exact_map_complete_l2_ratio"]
        ),
        "exact_map_resource_cost": (
            exact_ready
            and exact_cost_ratios is not None
            and len(exact_cost_ratios)
            == _gate2_number(exact_summary, "provider_success_count")
            and bool(exact_cost_ratios)
            and all(
                GATE2_THRESHOLDS["min_exact_map_resource_cost_ratio"]
                <= ratio
                <= GATE2_THRESHOLDS["max_exact_map_resource_cost_ratio"]
                for ratio in exact_cost_ratios
            )
            and _gate2_number(exact_summary, "max_resource_cost_ratio") is not None
            and max(exact_cost_ratios)
            == _gate2_number(exact_summary, "max_resource_cost_ratio")
            and _gate2_number(exact_summary, "max_resource_cost_ratio")
            <= GATE2_THRESHOLDS["max_exact_map_resource_cost_ratio"]
        ),
        "standard_reachable_success": (
            standard_ready
            and _gate2_number(standard_summary, "reachable_query_success_ratio")
            is not None
            and _gate2_number(standard_summary, "reachable_query_success_ratio")
            >= GATE2_THRESHOLDS["min_standard_reachable_success_ratio"]
        ),
        "standard_complete_l2": (
            standard_ready
            and _gate2_number(standard_summary, "complete_l2_ratio")
            == GATE2_THRESHOLDS["min_standard_complete_l2_ratio"]
        ),
        "standard_p95_runtime": (
            standard_ready
            and _gate2_number(standard_summary, "runtime_p95_ms") is not None
            and _gate2_number(standard_summary, "runtime_p95_ms")
            <= GATE2_THRESHOLDS["max_standard_p95_runtime_ms"]
        ),
        "hard_timeout_bound": all(
            ready
            and _gate2_number(dataset.get("summary"), "hard_timeout_violation_count")
            is not None
            and _gate2_number(dataset.get("summary"), "hard_timeout_violation_count")
            <= GATE2_THRESHOLDS["max_hard_timeout_violations"]
            for dataset, ready in (
                (primitive, primitive_ready),
                (exact, exact_ready),
                (standard, standard_ready),
            )
        ),
    }

    code_check_order = (
        ("preflight", "restore_gate2_runtime_isolation"),
        ("postflight", "restore_gate2_runtime_isolation"),
        ("boundaries_strict_false", "restore_gate2_safety_boundaries"),
        ("focused", "restore_gate2_focused_contracts"),
        ("full", "restore_path_planner_v1_regression"),
        ("benchmark_loader", "repair_gate2_benchmark_loader"),
    )
    for check, route in code_check_order:
        if not checks[check]:
            return "failed", route, [], checks

    blockers = [
        blocker
        for (dataset_name, blocker), ready in zip(
            GATE2_BLOCKERS,
            (primitive_ready, exact_ready, standard_ready),
            strict=True,
        )
        if not ready
    ]
    if blockers:
        return "blocked", blockers[0], blockers, checks

    metric_check_order = (
        ("primitive_false_positives", "repair_wheel_oracle_false_positives"),
        ("primitive_complete_l2", "repair_wheel_complete_l2_validation"),
        ("primitive_recall", "repair_wheel_primitive_recall"),
        ("hard_timeout_bound", "repair_wheel_hard_timeout_bound"),
        ("exact_map_success", "repair_wheel_exact_map_reachability"),
        ("exact_map_complete_l2", "repair_wheel_exact_map_l2_validation"),
        ("exact_map_resource_cost", "repair_wheel_exact_map_resource_cost"),
        ("standard_reachable_success", "repair_wheel_standard_reachability"),
        ("standard_complete_l2", "repair_wheel_standard_l2_validation"),
        ("standard_p95_runtime", "repair_wheel_standard_runtime"),
    )
    for check, route in metric_check_order:
        if not checks[check]:
            return "failed", route, [], checks
    return "passed", GATE2_PASS_ROUTE, [], checks


def _gate2_report(summary: dict[str, Any]) -> str:
    primitive = summary["datasets"]["primitive_audit"].get("summary") or {}
    exact = summary["datasets"]["exact_map_quality"].get("summary") or {}
    standard = summary["datasets"]["standard_episodes"].get("summary") or {}
    blockers = summary.get("blocking_reasons", [])
    blocker_text = "、".join(f"`{item}`" for item in blockers) if blockers else "无"
    return (
        "# Path Planner v2 Gate 2 轮式证据报告\n\n"
        f"- Gate 状态：`{summary['status']}`\n"
        f"- 下一步：`{summary['next_required_change']}`\n"
        f"- 正式阻塞项：{blocker_text}\n"
        f"- 独立 primitive 分母：{primitive.get('formal_row_count', 0)}\n"
        f"- oracle false positive：{primitive.get('false_positive_count', 'N/A')}\n"
        f"- primitive recall：{primitive.get('primitive_recall', 'N/A')}\n"
        f"- exact-map 最大成本比：{exact.get('max_resource_cost_ratio', 'N/A')}\n"
        f"- Standard 独立 episodes：{standard.get('formal_episode_count', 0)}\n"
        f"- Standard p95：{standard.get('runtime_p95_ms', 'N/A')} ms\n\n"
        "自标注行不进入正式分母；缺失的独立 oracle、exact optimum 或 Standard schedule "
        "只会产生可审计 blocker，不会伪造通过。v1 仍为默认；本 Gate 不发布 checkpoint、"
        "不替换 default policy、不连接 executor、不启动 canary。\n"
    )


def _gate2_dry_run_payloads(config: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    boundary_ok = boundaries_match(config.get("boundaries"))
    status = "dry_run" if boundary_ok else "failed"
    route = GATE2_EXECUTE_ROUTE if boundary_ok else "restore_gate2_safety_boundaries"
    datasets = {
        name: {
            "dataset_name": name,
            "input_path": config["inputs"][name],
            "status": "not_run",
            "rows": [],
            "summary": None,
        }
        for name in GATE2_INPUTS
    }
    checks = {
        "preflight": False,
        "postflight": False,
        "focused": False,
        "full": False,
        "benchmark_loader": False,
        "boundaries_strict_false": boundary_ok,
        "primitive_formal_denominator": False,
        "exact_map_formal_denominator": False,
        "standard_formal_denominator": False,
    }
    summary = {
        "schema_version": GATE2_SCHEMA_VERSION,
        "stage_id": GATE2_STAGE_ID,
        "status": status,
        "next_required_change": route,
        "blocking_reasons": [],
        "preflight": {"status": "not_run"},
        "postflight": {"status": "not_run"},
        "checks": checks,
        "focused": {"status": "not_run"},
        "full": {"status": "not_run"},
        "datasets": datasets,
        **gate_artifacts.BOUNDARY_FIELDS,
    }
    routing = {
        "schema_version": "xunce-path-v2-gate2-routing/v1",
        "stage_id": GATE2_STAGE_ID,
        "status": status,
        "route": route,
        "blocking_reasons": [],
        **gate_artifacts.BOUNDARY_FIELDS,
    }
    rows = [
        {
            "suite": "boundary-review",
            "check": field,
            "status": "passed" if boundary_ok else "failed",
            "value": False,
        }
        for field in sorted(gate_artifacts.BOUNDARY_FIELDS)
    ]
    phases = [
        {
            "phase": phase,
            "status": "completed" if phase == "boundary-review" and boundary_ok else "not_run",
        }
        for phase in (
            "preflight",
            "focused",
            "full",
            "primitive-audit",
            "exact-map-quality",
            "standard-episodes",
            "boundary-review",
        )
    ]
    review = {
        "schema_version": "xunce-path-v2-gate2-review/v1",
        "status": status,
        "checks": checks,
        "preflight": summary["preflight"],
        "postflight": summary["postflight"],
        "datasets": datasets,
    }
    return summary, routing, rows, phases, review


def _run_gate2_benchmark(
    *,
    config: dict[str, Any],
    output_root: Path,
    repo_root: Path,
    execute_tests: bool,
) -> dict[str, Any]:
    _validate_gate2_config(config)
    _assert_no_stale_artifacts(output_root)
    if not execute_tests:
        summary, routing, rows, phases, review = _gate2_dry_run_payloads(config)
    else:
        preflight = _gate2_preflight(config, repo_root)
        attempt_id = datetime.now(timezone.utc).strftime("attempt-%Y%m%dT%H%M%SZ") + f"-{os.getpid()}"
        attempt_root = Path(str(config["temp_root"])) / attempt_id
        artifact_io.make_dirs(attempt_root / "mpl")
        env = _common_env(repo_root, attempt_root)
        python = Path(str(config["python"])).resolve()
        focused_run: dict[str, Any] = {"status": "not_run", "returncode": None}
        full_run: dict[str, Any] = {"status": "not_run", "returncode": None}
        focused: dict[str, Any] = {"status": "not_run"}
        full: dict[str, Any] = {"status": "not_run"}
        datasets: dict[str, dict[str, Any]] = {
            name: {
                "dataset_name": name,
                "input_path": config["inputs"][name],
                "status": "not_run",
                "rows": [],
                "summary": None,
            }
            for name in GATE2_INPUTS
        }

        if preflight.get("status") == "passed":
            focused_junit = attempt_root / "focused.junit.xml"
            full_junit = attempt_root / "full.junit.xml"
            focused_run = _run_pytest(
                python=python,
                repo_root=repo_root,
                targets=GATE2_FOCUSED_TARGETS,
                junit_path=focused_junit,
                basetemp=attempt_root / "focused-basetemp",
                env=env,
            )
            focused = _audit_focused_junit(focused_junit)
            focused["returncode"] = focused_run["returncode"]
            focused["status"] = (
                "passed"
                if focused.get("status") == "passed" and focused_run["returncode"] == 0
                else "failed"
            )
            full_run = _run_pytest(
                python=python,
                repo_root=repo_root,
                targets=("tests",),
                junit_path=full_junit,
                basetemp=attempt_root / "full-basetemp",
                env=env,
            )
            full = _audit_gate2_full_junit(
                full_junit,
                expected_legacy=LEGACY_EXPECTED,
                allowed_skip_dependency=ALLOWED_SKIP_DEPENDENCY,
                baseline_evidence=config["baseline_evidence"],
            )
            full["returncode"] = full_run["returncode"]
            full["status"] = (
                "passed"
                if full.get("status") == "passed" and full_run["returncode"] == 0
                else "failed"
            )
            if focused["status"] == "passed" and full["status"] == "passed":
                datasets = {
                    name: _load_gate2_dataset(
                        dataset_name=name,
                        path=Path(config["inputs"][name]),
                        repo_root=repo_root,
                    )
                    for name in GATE2_INPUTS
                }

        postflight = _gate2_postflight(config, repo_root)
        postflight_ok = _postflight_matches(preflight, postflight)
        boundary_ok = boundaries_match(config.get("boundaries"))
        status, route, blockers, checks = _evaluate_gate2(
            preflight=preflight,
            postflight_ok=postflight_ok,
            focused=focused,
            full=full,
            boundary_ok=boundary_ok,
            datasets=datasets,
        )
        public_datasets = {
            name: {key: value for key, value in dataset.items() if key != "rows"}
            for name, dataset in datasets.items()
        }
        summary = {
            "schema_version": GATE2_SCHEMA_VERSION,
            "stage_id": GATE2_STAGE_ID,
            "status": status,
            "next_required_change": route,
            "blocking_reasons": blockers,
            "preflight": preflight,
            "postflight": postflight,
            "checks": checks,
            "focused": focused,
            "full": full,
            "datasets": public_datasets,
            **gate_artifacts.BOUNDARY_FIELDS,
        }
        routing = {
            "schema_version": "xunce-path-v2-gate2-routing/v1",
            "stage_id": GATE2_STAGE_ID,
            "status": status,
            "route": route,
            "blocking_reasons": blockers,
            **gate_artifacts.BOUNDARY_FIELDS,
        }
        rows = [
            {
                "suite": "preflight",
                "check": "git_import_identity",
                "status": preflight.get("status", "failed"),
            },
            {
                "suite": "focused",
                "check": "pytest",
                "status": focused.get("status", "not_run"),
                "passed": focused.get("passed", 0),
                "skipped": focused.get("skipped", 0),
            },
            {
                "suite": "full",
                "check": "pytest",
                "status": full.get("status", "not_run"),
                **full.get("total", _empty_counts()),
            },
            {
                "suite": "boundary-review",
                "check": "runtime_postflight",
                "status": "passed" if postflight_ok else "failed",
            },
        ]
        for name, blocker in GATE2_BLOCKERS:
            dataset = datasets[name]
            dataset_suite = name.replace("_", "-")
            if dataset.get("rows"):
                rows.extend(
                    {"suite": dataset_suite, **row}
                    for row in dataset["rows"]
                )
            else:
                rows.append(
                    {
                        "suite": dataset_suite,
                        "check": "formal_input",
                        "status": dataset.get("status", "not_run"),
                        "reason": (
                            "repair_gate2_benchmark_loader"
                            if dataset.get("status") == "internal_error"
                            else blocker
                        ),
                    }
                )
        rows.extend(
            {
                "suite": "boundary-review",
                "check": field,
                "status": "passed" if boundary_ok else "failed",
                "value": False,
            }
            for field in sorted(gate_artifacts.BOUNDARY_FIELDS)
        )
        rows.sort(
            key=lambda row: (
                str(row.get("suite", "")),
                int(row.get("seed", -1)),
                str(
                    row.get(
                        "row_id",
                        row.get("episode_id", row.get("check", "")),
                    )
                ),
            )
        )
        phases = [
            {
                "phase": "preflight",
                "status": "completed" if preflight.get("status") == "passed" else "failed",
            },
            {
                "phase": "focused",
                "status": "completed" if focused.get("status") == "passed" else focused.get("status", "not_run"),
            },
            {
                "phase": "full",
                "status": "completed" if full.get("status") == "passed" else full.get("status", "not_run"),
            },
            *[
                {
                    "phase": name.replace("_", "-"),
                    "status": (
                        "failed"
                        if datasets[name].get("status") == "internal_error"
                        else (
                            "not_run"
                            if datasets[name].get("status") == "not_run"
                            else (
                                "completed"
                                if datasets[name].get("status") == "loaded"
                                and checks[
                                    {
                                        "primitive_audit": "primitive_formal_denominator",
                                        "exact_map_quality": "exact_map_formal_denominator",
                                        "standard_episodes": "standard_formal_denominator",
                                    }[name]
                                ]
                                else "blocked"
                            )
                        )
                    ),
                }
                for name in GATE2_INPUTS
            ],
            {
                "phase": "boundary-review",
                "status": "completed" if boundary_ok and postflight_ok else "failed",
            },
        ]
        isolation_env = {
            key: env[key]
            for key in (
                "PYTHONNOUSERSITE",
                "PYTHONDONTWRITEBYTECODE",
                "PYTEST_DISABLE_PLUGIN_AUTOLOAD",
                "PYTHONPATH",
                "TEMP",
                "TMP",
                "MPLCONFIGDIR",
            )
        }
        review = {
            "schema_version": "xunce-path-v2-gate2-review/v1",
            "status": status,
            "checks": checks,
            "preflight": preflight,
            "postflight": postflight,
            "datasets": public_datasets,
            "execution": {
                "attempt_root": str(attempt_root),
                "isolation_env": isolation_env,
                "focused_command_result": focused_run,
                "full_command_result": full_run,
            },
        }

    gate_artifacts.write_gate_artifacts(
        output_root=output_root,
        config=config,
        summary=summary,
        routing=routing,
        rows=rows,
        phases=phases,
        review=review,
        report=_gate2_report(summary),
    )
    return summary


def _evaluate_gate3(
    *,
    preflight: dict[str, Any],
    postflight_ok: bool,
    focused: dict[str, Any],
    full: dict[str, Any],
    probe_audit: dict[str, Any],
    boundary_ok: bool,
) -> tuple[str, str, dict[str, bool]]:
    checks = {
        "preflight": preflight.get("status") == "passed",
        "postflight": postflight_ok,
        "focused": focused.get("status") == "passed",
        "full": full.get("status") == "passed",
        "ablation_matrix": probe_audit.get("matrix_complete") is True
        and probe_audit.get("stable_row_order") is True,
        "one_decision_digest": probe_audit.get("one_decision_digest") is True,
        "safety_equivalence": probe_audit.get("safety_equivalence") is True,
        "suggestion_non_authority": probe_audit.get("suggestion_non_authority") is True,
        "hierarchy_conservatism": probe_audit.get("hierarchy_conservatism") is True,
        "cache_l2_equivalence": probe_audit.get("cache_l2_equivalence") is True,
        "fallback_isolation": probe_audit.get("fallback_isolation") is True,
        "fatal_authority_clean": probe_audit.get("fatal_authority_clean") is True,
        "disabled_accelerator_disclosure": True,
        "provider_accelerator_unused": probe_audit.get(
            "provider_accelerator_unused"
        )
        is True,
        "boundaries_strict_false": boundary_ok,
    }
    if not checks["preflight"] or not checks["postflight"]:
        route = "restore_gate3_runtime_isolation"
    elif not checks["boundaries_strict_false"]:
        route = "restore_gate3_safety_boundaries"
    elif not checks["focused"]:
        route = "restore_gate3_focused_contracts"
    elif not checks["full"]:
        route = "restore_path_planner_v1_regression"
    elif any(
        reason in probe_audit.get("fatal_reasons", ())
        for reason in (
            "probe_subprocess_timeout",
            "probe_subprocess_failed",
            "probe_output_invalid",
        )
    ):
        route = "repair_gate3_deterministic_component_probes"
    elif not checks["fatal_authority_clean"]:
        route = "restore_gate3_fine_l2_authority"
    elif not checks["ablation_matrix"] or not checks["one_decision_digest"]:
        route = "repair_gate3_deterministic_component_probes"
    elif not (
        checks["safety_equivalence"]
        and checks["suggestion_non_authority"]
        and checks["hierarchy_conservatism"]
        and checks["cache_l2_equivalence"]
    ):
        route = "repair_gate3_accelerator_safety_equivalence"
    elif not checks["fallback_isolation"]:
        route = "repair_gate3_accelerator_fallback_isolation"
    elif not checks["provider_accelerator_unused"]:
        route = "remove_unintegrated_provider_accelerator_claim"
    else:
        route = GATE3_PASS_ROUTE
    status = "passed" if all(checks.values()) else "failed"
    return status, route, checks


def _gate3_report(summary: dict[str, Any]) -> str:
    disabled = "、".join(
        f"{item['accelerator_id']} ({item['reason']})"
        for item in summary["disabled_accelerators"]
    ) or "无"
    return (
        "# Path Planner v2 Gate 3 加速器证据\n\n"
        f"- 状态：`{summary['status']}`\n"
        f"- 下一路由：`{summary['next_required_change']}`\n"
        f"- 禁用加速器：`{disabled}`\n\n"
        "v1 仍为默认，v2 仍为 opt-in。当前组件均处于 disabled，原因按 "
        "`not_integrated_into_provider` 明确披露，尚未接入 wheel provider 的端到端"
        "加速路径。本 Gate 仅为 simulation proxy 组件证据，不声明 runtime speedup "
        "或成功率提升。synthetic terrain 仅是 proxy，不是 physical obstacle 数据，"
        "不得标作 `physical_obstacle_cells`。本 Gate 不发布 checkpoint、不替换 "
        "default policy、不连接 executor、不启动 canary。\n"
    )


def _gate3_dry_run_payloads(config: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    boundary_ok = boundaries_match(config.get("boundaries"))
    status = "dry_run" if boundary_ok else "failed"
    route = GATE3_EXECUTE_ROUTE if boundary_ok else "restore_gate3_safety_boundaries"
    disabled = [dict(item) for item in GATE3_DISABLED_ACCELERATORS]
    checks = {
        "preflight": False,
        "postflight": False,
        "focused": False,
        "full": False,
        "ablation_matrix": False,
        "one_decision_digest": False,
        "safety_equivalence": False,
        "fallback_isolation": False,
        "disabled_accelerator_disclosure": True,
        "provider_accelerator_unused": True,
        "boundaries_strict_false": boundary_ok,
    }
    summary = {
        "schema_version": GATE3_SCHEMA_VERSION,
        "stage_id": GATE3_STAGE_ID,
        "status": status,
        "next_required_change": route,
        "blocking_reasons": [],
        "checks": checks,
        "preflight": {"status": "not_run"},
        "postflight": {"status": "not_run"},
        "focused": {"status": "not_run"},
        "full": {"status": "not_run"},
        "ablation": {"status": "not_run", "row_count": 0, "decision_digest": None},
        "disabled_accelerators": disabled,
        **gate_artifacts.BOUNDARY_FIELDS,
    }
    routing = {
        "schema_version": "xunce-path-v2-gate3-routing/v1",
        "stage_id": GATE3_STAGE_ID,
        "status": status,
        "route": route,
        "blocking_reasons": [],
        "disabled_accelerators": disabled,
        **gate_artifacts.BOUNDARY_FIELDS,
    }
    rows = [
        {"suite": "test", "check": suite, "status": "not_run"}
        for suite in ("focused", "full")
    ]
    rows.extend(
        {
            "suite": "ablation",
            "case_id": case_id,
            "worker_count": worker_count,
            "python_hash_seed": hash_seed,
            "repeat": repeat,
            "status": "not_run",
        }
        for case_id in GATE3_CASES
        for worker_count in GATE3_WORKER_COUNTS
        for hash_seed in GATE3_HASH_SEEDS
        for repeat in range(1, GATE3_REPEAT_COUNT + 1)
    )
    rows.extend(
        {
            "suite": "fallback",
            "accelerator_id": item["accelerator_id"],
            "status": "not_run",
        }
        for item in disabled
    )
    rows.extend(
        {
            "suite": "disclosure",
            "accelerator_id": item["accelerator_id"],
            "reason": item["reason"],
            "status": "passed",
        }
        for item in disabled
    )
    rows.extend(
        {
            "suite": "boundary",
            "check": field,
            "status": "passed" if boundary_ok else "failed",
            "value": False,
        }
        for field in sorted(gate_artifacts.BOUNDARY_FIELDS)
    )
    phases = [
        {
            "phase": phase,
            "status": (
                "completed"
                if phase in {"disclosure", "boundary-review"} and boundary_ok
                else "not_run"
            ),
        }
        for phase in (
            "preflight",
            "focused",
            "full",
            "ablation",
            "fallback",
            "disclosure",
            "boundary-review",
            "postflight",
        )
    ]
    review = {
        "schema_version": "xunce-path-v2-gate3-review/v1",
        "status": status,
        "checks": checks,
        "preflight": summary["preflight"],
        "postflight": summary["postflight"],
        "execution": {"status": "not_run", "commands": [], "environment": {}},
        "probe_audit": summary["ablation"],
    }
    return summary, routing, rows, phases, review


def _run_gate3_benchmark(
    *,
    config: dict[str, Any],
    output_root: Path,
    repo_root: Path,
    execute_tests: bool,
) -> dict[str, Any]:
    _validate_gate3_config(config)
    _assert_gate3_output_root_absent(output_root)
    if not execute_tests:
        summary, routing, rows, phases, review = _gate3_dry_run_payloads(config)
    else:
        preflight = _gate3_preflight(config, repo_root)
        attempt_id = (
            datetime.now(timezone.utc).strftime("attempt-%Y%m%dT%H%M%SZ")
            + f"-{os.getpid()}"
        )
        attempt_root = Path(str(config["temp_root"])) / attempt_id
        artifact_io.make_dirs(attempt_root / "mpl")
        env = _common_env(repo_root, attempt_root)
        python = Path(str(config["python"])).resolve()
        focused_run: dict[str, Any] = {"status": "not_run", "returncode": None}
        full_run: dict[str, Any] = {"status": "not_run", "returncode": None}
        focused: dict[str, Any] = {"status": "not_run"}
        full: dict[str, Any] = {"status": "not_run"}
        probe_result: dict[str, Any] = {
            **_audit_gate3_probe_rows([]),
            "rows": [],
            "commands": [],
        }
        if preflight.get("status") == "passed":
            focused_junit = attempt_root / "focused.junit.xml"
            full_junit = attempt_root / "full.junit.xml"
            focused_run = _run_pytest(
                python=python,
                repo_root=repo_root,
                targets=GATE3_FOCUSED_TARGETS,
                junit_path=focused_junit,
                basetemp=attempt_root / "focused-basetemp",
                env=env,
            )
            focused = _audit_focused_junit(focused_junit)
            focused["returncode"] = focused_run["returncode"]
            focused["status"] = (
                "passed"
                if focused.get("status") == "passed"
                and focused_run["returncode"] == 0
                else "failed"
            )
            full_run = _run_pytest(
                python=python,
                repo_root=repo_root,
                targets=("tests",),
                junit_path=full_junit,
                basetemp=attempt_root / "full-basetemp",
                env=env,
            )
            full = _audit_gate3_full_junit(
                full_junit,
                expected_legacy=LEGACY_EXPECTED,
                allowed_skip_dependency=ALLOWED_SKIP_DEPENDENCY,
                baseline_evidence=config["baseline_evidence"],
            )
            full["returncode"] = full_run["returncode"]
            full["status"] = (
                "passed"
                if full.get("status") == "passed" and full_run["returncode"] == 0
                else "failed"
            )
            if focused["status"] == "passed" and full["status"] == "passed":
                probe_result = _run_gate3_probes(
                    python=python,
                    repo_root=repo_root,
                    common_env=env,
                    timeout_s=config["probe_subprocess_timeout_s"],
                )

        postflight = _gate3_postflight(config, repo_root)
        postflight_ok = _postflight_matches(preflight, postflight)
        boundary_ok = boundaries_match(config.get("boundaries"))
        status, route, checks = _evaluate_gate3(
            preflight=preflight,
            postflight_ok=postflight_ok,
            focused=focused,
            full=full,
            probe_audit=probe_result,
            boundary_ok=boundary_ok,
        )
        disabled = [dict(item) for item in GATE3_DISABLED_ACCELERATORS]
        public_probe_audit = {
            key: value
            for key, value in probe_result.items()
            if key not in {"rows", "commands"}
        }
        summary = {
            "schema_version": GATE3_SCHEMA_VERSION,
            "stage_id": GATE3_STAGE_ID,
            "status": status,
            "next_required_change": route,
            "blocking_reasons": [],
            "checks": checks,
            "preflight": preflight,
            "postflight": postflight,
            "focused": focused,
            "full": full,
            "ablation": public_probe_audit,
            "disabled_accelerators": disabled,
            **gate_artifacts.BOUNDARY_FIELDS,
        }
        routing = {
            "schema_version": "xunce-path-v2-gate3-routing/v1",
            "stage_id": GATE3_STAGE_ID,
            "status": status,
            "route": route,
            "blocking_reasons": [],
            "disabled_accelerators": disabled,
            **gate_artifacts.BOUNDARY_FIELDS,
        }
        rows = [
            {
                "suite": "test",
                "check": "focused",
                "status": focused.get("status", "not_run"),
                "passed": focused.get("passed", 0),
                "skipped": focused.get("skipped", 0),
            },
            {
                "suite": "test",
                "check": "full",
                "status": full.get("status", "not_run"),
                **full.get("total", _empty_counts()),
            },
        ]
        rows.extend(
            {"suite": "ablation", **row} for row in probe_result["rows"]
        )
        rows.extend(
            {
                "suite": "fallback",
                "accelerator_id": item["accelerator_id"],
                "status": "passed" if checks["fallback_isolation"] else "failed",
                "runtime_disable_count": sum(
                    1
                    for row in probe_result["rows"]
                    for runtime_item in row.get("runtime_disabled_accelerators", [])
                    if runtime_item.get("accelerator_id") == item["accelerator_id"]
                ),
            }
            for item in disabled
        )
        rows.extend(
            {
                "suite": "disclosure",
                "accelerator_id": item["accelerator_id"],
                "reason": item["reason"],
                "status": "passed",
            }
            for item in disabled
        )
        rows.extend(
            {
                "suite": "boundary",
                "check": field,
                "status": "passed" if boundary_ok else "failed",
                "value": False,
            }
            for field in sorted(gate_artifacts.BOUNDARY_FIELDS)
        )
        phases = [
            {
                "phase": "preflight",
                "status": (
                    "completed" if preflight.get("status") == "passed" else "failed"
                ),
            },
            {
                "phase": "focused",
                "status": (
                    "completed"
                    if focused.get("status") == "passed"
                    else focused.get("status", "not_run")
                ),
            },
            {
                "phase": "full",
                "status": (
                    "completed"
                    if full.get("status") == "passed"
                    else full.get("status", "not_run")
                ),
            },
            {
                "phase": "ablation",
                "status": (
                    "completed"
                    if public_probe_audit.get("status") == "passed"
                    else public_probe_audit.get("status", "not_run")
                ),
            },
            {
                "phase": "fallback",
                "status": "completed" if checks["fallback_isolation"] else "failed",
            },
            {"phase": "disclosure", "status": "completed"},
            {
                "phase": "boundary-review",
                "status": "completed" if boundary_ok else "failed",
            },
            {
                "phase": "postflight",
                "status": "completed" if postflight_ok else "failed",
            },
        ]
        isolation_env = {
            key: env[key]
            for key in (
                "PYTHONNOUSERSITE",
                "PYTHONDONTWRITEBYTECODE",
                "PYTEST_DISABLE_PLUGIN_AUTOLOAD",
                "PYTHONPATH",
                "TEMP",
                "TMP",
                "MPLCONFIGDIR",
            )
        }
        review = {
            "schema_version": "xunce-path-v2-gate3-review/v1",
            "status": status,
            "checks": checks,
            "preflight": preflight,
            "postflight": postflight,
            "execution": {
                "attempt_root": str(attempt_root),
                "isolation_env": isolation_env,
                "focused_command_result": focused_run,
                "full_command_result": full_run,
                "probe_commands": probe_result["commands"],
            },
            "focused_junit": focused,
            "full_junit": full,
            "probe_audit": public_probe_audit,
        }
    gate_artifacts.write_gate_artifacts_atomically(
        output_root=output_root,
        config=config,
        summary=summary,
        routing=routing,
        rows=rows,
        phases=phases,
        review=review,
        report=_gate3_report(summary),
    )
    return summary


def _gate4_dataset_not_run(config: dict[str, Any], name: str) -> dict[str, Any]:
    return {
        "dataset_name": name,
        "path": config["inputs"][name],
        "status": "not_run",
        "content_read": False,
        "formal_source": config["dataset_contract"]["formal_sources"][name],
        "trusted_input": dict(config["trusted_inputs"][name]),
    }


def _gate4_report(summary: dict[str, Any]) -> str:
    blockers = summary.get("blocking_reasons", [])
    blocker_text = "、".join(f"`{item}`" for item in blockers) if blockers else "无"
    dataset_lines = "".join(
        f"- {name}: path=`{item['path']}`, status=`{item['status']}`, "
        f"content_read={str(item['content_read']).lower()}\n"
        for name, item in summary["datasets"].items()
    )
    threshold_lines = "".join(
        f"- {name}={value}\n" for name, value in summary["thresholds"].items()
    )
    capability_lines = "".join(
        f"- {name}={str(value).lower() if isinstance(value, bool) else value}\n"
        for name, value in summary["capability_disclosure"].items()
    )
    boundary_lines = "".join(
        f"- {name}=false\n" for name in gate_artifacts.BOUNDARY_FIELDS
    )
    return (
        "# Path Planner v2 Gate 4B 足式正式证据快照\n\n"
        f"- Gate 状态：`{summary['status']}`\n"
        f"- 下一路由：`{summary['next_required_change']}`\n"
        f"- 完整阻塞项：{blocker_text}\n"
        "- formal_metrics_status=not_evaluated\n\n"
        "## 正式输入\n\n"
        f"{dataset_lines}\n"
        "正式独立证据未获批准，本次未读取输入内容。未取得 Gate 4 formal pass；"
        "未读取或聚合正式输入；不构成动态步态或实机稳定性证据。\n\n"
        "## 冻结目标门槛\n\n"
        f"{threshold_lines}\n"
        "## 能力披露\n\n"
        f"{capability_lines}\n"
        "## 安全边界\n\n"
        f"{boundary_lines}\n"
        "v1 仍为默认，v2 仍为 opt-in；不发布 checkpoint、不替换 default policy、"
        "不连接 executor、不启动 canary。\n"
    )


def _gate4_dry_run_payloads(config: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    datasets = {
        name: _gate4_dataset_not_run(config, name) for name in GATE4_INPUTS
    }
    benchmark_contract = _gate4_benchmark_contract_not_run()
    checks = {
        "preflight": False,
        "postflight": False,
        "focused": False,
        "full": False,
        "benchmark_contract": False,
        "boundaries_strict_false": True,
    }
    summary = {
        "schema_version": GATE4_SCHEMA_VERSION,
        "stage_id": GATE4_STAGE_ID,
        "status": "dry_run",
        "next_required_change": GATE4_EXECUTE_ROUTE,
        "blocking_reasons": [],
        "formal_metrics_status": "not_evaluated",
        "thresholds": dict(GATE4_THRESHOLDS),
        "capability_disclosure": dict(GATE4_CAPABILITY_DISCLOSURE),
        "dataset_contract": config["dataset_contract"],
        "benchmark_contract": benchmark_contract,
        "preflight": {"status": "not_run"},
        "postflight": {"status": "not_run"},
        "checks": checks,
        "focused": {"status": "not_run"},
        "full": {"status": "not_run"},
        "datasets": datasets,
        **gate_artifacts.BOUNDARY_FIELDS,
    }
    routing = {
        "schema_version": "xunce-path-v2-gate4-routing/v1",
        "stage_id": GATE4_STAGE_ID,
        "status": "dry_run",
        "route": GATE4_EXECUTE_ROUTE,
        "blocking_reasons": [],
        "formal_metrics_status": "not_evaluated",
        "capability_disclosure": dict(GATE4_CAPABILITY_DISCLOSURE),
        **gate_artifacts.BOUNDARY_FIELDS,
    }
    rows = [
        {
            "suite": "formal-input",
            "dataset": name,
            "path": config["inputs"][name],
            "status": "not_run",
            "content_read": False,
        }
        for name in GATE4_INPUTS
    ]
    rows.extend(
        {
            "suite": "boundary-review",
            "check": field,
            "status": "passed",
            "value": False,
        }
        for field in sorted(gate_artifacts.BOUNDARY_FIELDS)
    )
    phases = [
        {
            "phase": phase,
            "status": "not_run",
        }
        for phase in GATE4_PHASES
    ]
    review = {
        "schema_version": "xunce-path-v2-gate4-review/v1",
        "status": "dry_run",
        "formal_metrics_status": "not_evaluated",
        "thresholds": dict(GATE4_THRESHOLDS),
        "capability_disclosure": dict(GATE4_CAPABILITY_DISCLOSURE),
        "dataset_contract": config["dataset_contract"],
        "benchmark_contract": benchmark_contract,
        "checks": checks,
        "preflight": summary["preflight"],
        "postflight": summary["postflight"],
        "datasets": datasets,
        "execution": {"status": "not_run", "commands": [], "environment": {}},
    }
    return summary, routing, rows, phases, review


def _evaluate_gate4(
    *,
    preflight: dict[str, Any],
    postflight_ok: bool,
    focused: dict[str, Any],
    full: dict[str, Any],
    benchmark_contract: dict[str, Any],
) -> tuple[str, str, list[str], dict[str, bool]]:
    checks = {
        "preflight": preflight.get("status") == "passed",
        "postflight": postflight_ok,
        "focused": focused.get("status") == "passed",
        "full": full.get("status") == "passed",
        "benchmark_contract": benchmark_contract.get("status") == "passed",
        "boundaries_strict_false": True,
    }
    if not checks["preflight"] or not checks["postflight"]:
        return "failed", "restore_gate4_runtime_isolation", [], checks
    if not checks["focused"]:
        return "failed", "restore_gate4_focused_contracts", [], checks
    if not checks["full"]:
        return "failed", "restore_path_planner_v1_regression", [], checks
    if not checks["benchmark_contract"]:
        return "failed", "repair_gate4_benchmark_contract", [], checks
    blockers = [blocker for _, blocker in GATE4_BLOCKERS]
    return "blocked", blockers[0], blockers, checks


def _gate4_phase_status(
    phase: str,
    *,
    preflight: dict[str, Any],
    focused: dict[str, Any],
    full: dict[str, Any],
    datasets: dict[str, dict[str, Any]],
    postflight_ok: bool,
) -> str:
    if phase == "preflight":
        return "completed" if preflight.get("status") == "passed" else "failed"
    if phase == "focused":
        status = focused.get("status", "not_run")
        return "completed" if status == "passed" else status
    if phase == "full":
        status = full.get("status", "not_run")
        return "completed" if status == "passed" else status
    if phase == "boundary-review":
        return "completed" if postflight_ok else "failed"
    dataset_name = phase.replace("-", "_")
    status = datasets[dataset_name]["status"]
    return "blocked" if status in {"missing", "untrusted"} else "not_run"


def _run_gate4_benchmark(
    *,
    config: dict[str, Any],
    output_root: Path,
    repo_root: Path,
    execute_tests: bool,
) -> dict[str, Any]:
    output_root = Path(output_root)
    _validate_gate4_config(config)
    _validate_gate4_output_mode(output_root, execute_tests=execute_tests)
    _assert_gate4_output_root_absent(output_root)
    output_root = validate_output_root(repo_root, output_root)
    if not execute_tests:
        summary, routing, rows, phases, review = _gate4_dry_run_payloads(config)
    else:
        datasets = {
            name: _gate4_dataset_not_run(config, name) for name in GATE4_INPUTS
        }
        benchmark_contract = _gate4_benchmark_contract_not_run()
        focused_run: dict[str, Any] = {"status": "not_run", "returncode": None}
        full_run: dict[str, Any] = {"status": "not_run", "returncode": None}
        focused: dict[str, Any] = {"status": "not_run"}
        full: dict[str, Any] = {"status": "not_run"}
        attempt_root: Path | None = None
        env: dict[str, str] = {}

        preflight = _gate4_preflight(config, repo_root)
        if preflight.get("status") == "passed":
            attempt_id = (
                datetime.now(timezone.utc).strftime("attempt-%Y%m%dT%H%M%SZ")
                + f"-{os.getpid()}"
            )
            attempt_root = Path(str(config["temp_root"])) / attempt_id
            artifact_io.make_dirs(attempt_root / "mpl")
            env = _common_env(repo_root, attempt_root)
            python = Path(str(config["python"])).resolve()
            focused_junit = attempt_root / "focused.junit.xml"
            full_junit = attempt_root / "full.junit.xml"
            focused_run = _run_pytest(
                python=python,
                repo_root=repo_root,
                targets=GATE4_FOCUSED_TARGETS,
                junit_path=focused_junit,
                basetemp=attempt_root / "focused-basetemp",
                env=env,
            )
            focused = _audit_focused_junit(focused_junit)
            focused["returncode"] = focused_run["returncode"]
            focused["status"] = (
                "passed"
                if focused.get("status") == "passed"
                and focused_run["returncode"] == 0
                else "failed"
            )
            full_run = _run_pytest(
                python=python,
                repo_root=repo_root,
                targets=("tests",),
                junit_path=full_junit,
                basetemp=attempt_root / "full-basetemp",
                env=env,
            )
            full = _audit_gate4_full_junit(
                full_junit,
                expected_legacy=LEGACY_EXPECTED,
                allowed_skip_dependency=ALLOWED_SKIP_DEPENDENCY,
                baseline_evidence=config["baseline_evidence"],
            )
            full["returncode"] = full_run["returncode"]
            full["status"] = (
                "passed"
                if full.get("status") == "passed" and full_run["returncode"] == 0
                else "failed"
            )
            if focused["status"] == "passed" and full["status"] == "passed":
                benchmark_contract = _audit_gate4_benchmark_contract(repo_root)
                if benchmark_contract.get("status") == "passed":
                    datasets = {
                        name: _gate4_dataset_status(
                            dataset_name=name,
                            path=Path(config["inputs"][name]),
                            trusted_input=config["trusted_inputs"][name],
                            formal_source=config["dataset_contract"]["formal_sources"][name],
                        )
                        for name in GATE4_INPUTS
                    }

        postflight = _gate4_postflight(config, repo_root)
        postflight_ok = _postflight_matches(preflight, postflight)
        status, route, blockers, checks = _evaluate_gate4(
            preflight=preflight,
            postflight_ok=postflight_ok,
            focused=focused,
            full=full,
            benchmark_contract=benchmark_contract,
        )
        summary = {
            "schema_version": GATE4_SCHEMA_VERSION,
            "stage_id": GATE4_STAGE_ID,
            "status": status,
            "next_required_change": route,
            "blocking_reasons": blockers,
            "formal_metrics_status": "not_evaluated",
            "thresholds": dict(GATE4_THRESHOLDS),
            "capability_disclosure": dict(GATE4_CAPABILITY_DISCLOSURE),
            "dataset_contract": config["dataset_contract"],
            "benchmark_contract": benchmark_contract,
            "preflight": preflight,
            "postflight": postflight,
            "checks": checks,
            "focused": focused,
            "full": full,
            "datasets": datasets,
            **gate_artifacts.BOUNDARY_FIELDS,
        }
        routing = {
            "schema_version": "xunce-path-v2-gate4-routing/v1",
            "stage_id": GATE4_STAGE_ID,
            "status": status,
            "route": route,
            "blocking_reasons": blockers,
            "formal_metrics_status": "not_evaluated",
            "capability_disclosure": dict(GATE4_CAPABILITY_DISCLOSURE),
            **gate_artifacts.BOUNDARY_FIELDS,
        }
        rows = [
            {
                "suite": "preflight",
                "check": "git_import_identity",
                "status": preflight.get("status", "failed"),
            },
            {
                "suite": "focused",
                "check": "pytest",
                "status": focused.get("status", "not_run"),
                "passed": focused.get("passed", 0),
                "skipped": focused.get("skipped", 0),
            },
            {
                "suite": "full",
                "check": "pytest",
                "status": full.get("status", "not_run"),
                **full.get("total", _empty_counts()),
            },
            {
                "suite": "benchmark-contract",
                "check": "typed-evidence-api",
                "status": benchmark_contract.get("status", "not_run"),
            },
        ]
        rows.extend(
            {
                "suite": "formal-input",
                "dataset": name,
                "path": config["inputs"][name],
                "status": datasets[name]["status"],
                "content_read": False,
                "reason": blocker,
            }
            for name, blocker in GATE4_BLOCKERS
        )
        rows.append(
            {
                "suite": "boundary-review",
                "check": "runtime_postflight",
                "status": "passed" if postflight_ok else "failed",
            }
        )
        rows.extend(
            {
                "suite": "boundary-review",
                "check": field,
                "status": "passed",
                "value": False,
            }
            for field in sorted(gate_artifacts.BOUNDARY_FIELDS)
        )
        phases = [
            {
                "phase": phase,
                "status": _gate4_phase_status(
                    phase,
                    preflight=preflight,
                    focused=focused,
                    full=full,
                    datasets=datasets,
                    postflight_ok=postflight_ok,
                ),
            }
            for phase in GATE4_PHASES
        ]
        isolation_env = {
            key: env[key]
            for key in (
                "PYTHONNOUSERSITE",
                "PYTHONDONTWRITEBYTECODE",
                "PYTEST_DISABLE_PLUGIN_AUTOLOAD",
                "PYTHONPATH",
                "TEMP",
                "TMP",
                "MPLCONFIGDIR",
            )
            if key in env
        }
        review = {
            "schema_version": "xunce-path-v2-gate4-review/v1",
            "status": status,
            "formal_metrics_status": "not_evaluated",
            "thresholds": dict(GATE4_THRESHOLDS),
            "capability_disclosure": dict(GATE4_CAPABILITY_DISCLOSURE),
            "dataset_contract": config["dataset_contract"],
            "benchmark_contract": benchmark_contract,
            "checks": checks,
            "preflight": preflight,
            "postflight": postflight,
            "datasets": datasets,
            "execution": {
                "attempt_root": str(attempt_root) if attempt_root is not None else None,
                "isolation_env": isolation_env,
                "focused_command_result": focused_run,
                "full_command_result": full_run,
            },
        }
    gate_artifacts.write_gate_artifacts_atomically(
        output_root=output_root,
        config=config,
        summary=summary,
        routing=routing,
        rows=rows,
        phases=phases,
        review=review,
        report=_gate4_report(summary),
    )
    return summary


def run_gate_benchmark(
    config_path: Path,
    output_root: Path,
    repo_root: Path,
    execute_tests: bool = True,
) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    repo_root = Path(repo_root).resolve()
    config = artifact_io.read_json(config_path)
    if (
        config.get("schema_version") == GATE4_SCHEMA_VERSION
        or config.get("stage_id") == GATE4_STAGE_ID
    ):
        return _run_gate4_benchmark(
            config=config,
            output_root=Path(output_root),
            repo_root=repo_root,
            execute_tests=execute_tests,
        )
    output_root = validate_output_root(repo_root, output_root)
    if config.get("schema_version") == GATE3_SCHEMA_VERSION:
        return _run_gate3_benchmark(
            config=config,
            output_root=output_root,
            repo_root=repo_root,
            execute_tests=execute_tests,
        )
    if config.get("schema_version") == GATE2_SCHEMA_VERSION:
        return _run_gate2_benchmark(
            config=config,
            output_root=output_root,
            repo_root=repo_root,
            execute_tests=execute_tests,
        )
    _validate_config(config)

    if execute_tests:
        preflight = _preflight(config, repo_root)
        if preflight.get("status") != "passed":
            raise RuntimeError("Gate 1 git/import preflight failed before output creation")
    else:
        preflight = {"status": "not_run"}

    _assert_no_stale_artifacts(output_root)

    if not execute_tests:
        summary, routing, rows, phases, review = _dry_run_payloads(config)
    else:
        attempt_id = datetime.now(timezone.utc).strftime("attempt-%Y%m%dT%H%M%SZ") + f"-{os.getpid()}"
        attempt_root = Path(str(config["temp_root"])) / attempt_id
        artifact_io.make_dirs(attempt_root / "mpl")
        env = _common_env(repo_root, attempt_root)
        python = Path(str(config["python"])).resolve()
        focused_junit = attempt_root / "focused.junit.xml"
        full_junit = attempt_root / "full.junit.xml"
        focused_run = _run_pytest(
            python=python,
            repo_root=repo_root,
            targets=FOCUSED_TARGETS,
            junit_path=focused_junit,
            basetemp=attempt_root / "focused-basetemp",
            env=env,
        )
        focused = _audit_focused_junit(focused_junit)
        focused["returncode"] = focused_run["returncode"]
        focused["status"] = (
            "passed"
            if focused["status"] == "passed" and focused_run["returncode"] == 0
            else "failed"
        )

        full_run = _run_pytest(
            python=python,
            repo_root=repo_root,
            targets=("tests",),
            junit_path=full_junit,
            basetemp=attempt_root / "full-basetemp",
            env=env,
        )
        full = audit_full_junit(
            full_junit,
            expected_legacy=LEGACY_EXPECTED,
            allowed_skip_dependency=ALLOWED_SKIP_DEPENDENCY,
        )
        full["returncode"] = full_run["returncode"]
        full["status"] = (
            "passed"
            if full["status"] == "passed" and full_run["returncode"] == 0
            else "failed"
        )

        repeat_rows = _run_byte_repeats(
            python=python,
            repo_root=repo_root,
            seeds=config["byte_repeat"]["python_hash_seeds"],
            common_env=env,
        )
        byte_repeat = audit_repeat_digests(repeat_rows)
        probe_boundary_ok = all(
            row["returncode"] == 0
            and row["root_has_plan_v2"] is False
            and row["v1_route_schema"] == "path-planner-route/v1"
            for row in repeat_rows
        )
        postflight = _postflight(config, repo_root)
        postflight_ok = _postflight_matches(preflight, postflight)
        boundary_ok = boundaries_match(config.get("boundaries")) and probe_boundary_ok
        checks = {
            "preflight": preflight["status"] == "passed",
            "postflight": postflight_ok,
            "focused": focused["status"] == "passed",
            "full": full["status"] == "passed",
            "byte_repeat": byte_repeat["status"] == "passed",
            "boundaries_strict_false": boundary_ok,
        }
        passed = all(checks.values())
        status = "passed" if passed else "failed"
        if not postflight_ok:
            next_change = "restore_gate1_runtime_isolation"
        elif not boundaries_match(config.get("boundaries")):
            next_change = "restore_gate1_safety_boundaries"
        elif not focused["status"] == "passed":
            next_change = "restore_gate1_focused_contracts"
        elif not full["status"] == "passed":
            next_change = "restore_path_planner_v1_regression"
        elif byte_repeat["status"] != "passed" or not probe_boundary_ok:
            next_change = "restore_gate1_byte_stability_and_v1_isolation"
        else:
            next_change = PASS_ROUTE

        summary = {
            "schema_version": SCHEMA_VERSION,
            "stage_id": STAGE_ID,
            "status": status,
            "next_required_change": next_change,
            "preflight": preflight,
            "postflight": postflight,
            "checks": checks,
            "focused": focused,
            "full": full,
            "byte_repeat": byte_repeat,
            **gate_artifacts.BOUNDARY_FIELDS,
        }
        routing = {
            "schema_version": "xunce-path-v2-gate1-routing/v1",
            "stage_id": STAGE_ID,
            "status": status,
            "route": next_change,
            **gate_artifacts.BOUNDARY_FIELDS,
        }
        rows = [
            {"suite": "preflight", "check": "git_import_identity", "status": preflight["status"]},
            {"suite": "boundary-review", "check": "runtime_postflight", "status": "passed" if postflight_ok else "failed"},
            {"suite": "focused", "check": "pytest", "status": focused["status"], "passed": focused["passed"], "skipped": focused["skipped"]},
            {"suite": "full", "check": "pytest", "status": full["status"], **full["total"]},
            {"suite": "full", "check": "legacy_counts", "status": "passed" if full["legacy"] == LEGACY_EXPECTED else "failed", **full["legacy"]},
            {"suite": "full", "check": "legacy_skips", "status": "passed" if full["legacy_skip_contract"] else "failed", "skipped": full["legacy"]["skipped"]},
            *repeat_rows,
            *[
                {"suite": "boundary-review", "check": field, "status": "passed" if config["boundaries"].get(field) is False else "failed", "value": False}
                for field in sorted(gate_artifacts.BOUNDARY_FIELDS)
            ],
            {"suite": "boundary-review", "check": "root_api_isolation", "status": "passed" if probe_boundary_ok else "failed"},
        ]
        rows.sort(key=lambda row: (str(row.get("suite", "")), str(row.get("check", "")), int(row.get("repeat", 0))))
        phases = [
            {"phase": "preflight", "status": "completed"},
            {"phase": "focused", "status": "completed" if focused["status"] == "passed" else "failed"},
            {"phase": "full", "status": "completed" if full["status"] == "passed" else "failed"},
            {"phase": "byte-repeat", "status": "completed" if byte_repeat["status"] == "passed" else "failed"},
            {"phase": "boundary-review", "status": "completed" if boundary_ok and postflight_ok else "failed"},
        ]
        isolation_env = {
            key: env[key]
            for key in (
                "PYTHONNOUSERSITE",
                "PYTHONDONTWRITEBYTECODE",
                "PYTEST_DISABLE_PLUGIN_AUTOLOAD",
                "PYTHONPATH",
                "TEMP",
                "TMP",
                "MPLCONFIGDIR",
            )
        }
        review = {
            "schema_version": "xunce-path-v2-gate1-review/v1",
            "status": status,
            "checks": checks,
            "preflight": preflight,
            "postflight": postflight,
            "execution": {
                "attempt_root": str(attempt_root),
                "isolation_env": isolation_env,
                "focused_command_result": focused_run,
                "full_command_result": full_run,
            },
            "focused_junit": focused,
            "full_junit": full,
            "repeat_rows": repeat_rows,
            "byte_repeat": byte_repeat,
            "probe_boundary_ok": probe_boundary_ok,
        }

    gate_artifacts.write_gate_artifacts(
        output_root=output_root,
        config=config,
        summary=summary,
        routing=routing,
        rows=rows,
        phases=phases,
        review=review,
        report=_report(summary),
    )
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Path Planner v2 gate evidence")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/xunce_path_v2_gate1_contract_v1.json"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("D:/xunce/out/path_v2/g1"),
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--dry-run", action="store_true")
    return parser


def status_exit_code(status: str) -> int:
    return 0 if status in {"passed", "blocked", "dry_run"} else 1


def main() -> int:
    args = _parser().parse_args()
    summary = run_gate_benchmark(
        config_path=args.config,
        output_root=args.output_root,
        repo_root=args.repo_root,
        execute_tests=not args.dry_run,
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return status_exit_code(summary["status"])


if __name__ == "__main__":
    raise SystemExit(main())
