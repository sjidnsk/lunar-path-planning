"""Contracts for the policy-blind midterm scenario freeze."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


def _descriptor_rows(count: int = 80) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index in range(count):
        rows.append(
            {
                "scenario_id": f"catalog/scenario-{index:03d}",
                "scenario_hash": hashlib.sha256(f"scenario-{index}".encode()).hexdigest(),
                "slope_p90_deg": float(5 + index % 17),
                "hard_obstacle_fraction": float((index % 11) / 100),
                "start_pose_bin": [index % 2, (index // 2) % 2, index % 4],
                "initial_observed_coverable_fraction": float((index % 7) / 10),
                "initial_valid_frontier_count": 10 + index % 13,
                "coverable_cell_count": 100 + index,
                "parent_roi": f"roi-{index % 10}",
                "density_profile": ("low", "medium", "high")[index % 3],
                "start_to_farthest_candidate_distance_bin": index % 3,
            }
        )
    return rows


def _freeze_module():
    import freeze_xunce_mid_dual_scenarios as freeze

    return freeze


def test_descriptor_extraction_contains_only_policy_independent_fields() -> None:
    """Catch a descriptor serializer that admits policy or outcome evidence."""
    freeze = _freeze_module()
    descriptor = freeze.extract_policy_blind_descriptor(
        {
            **_descriptor_rows(1)[0],
            "irrelevant_static_note": "ignored",
        }
    )

    assert tuple(descriptor) == freeze.DESCRIPTOR_FIELDS
    assert all("policy" not in key and "checkpoint" not in key for key in descriptor)


def test_quantile_bins_are_stable_under_input_order_changes() -> None:
    """Catch rank binning that depends on catalog iteration order."""
    freeze = _freeze_module()
    rows = _descriptor_rows(18)

    forward = freeze.assign_stable_rank_tertiles(rows)
    reverse = freeze.assign_stable_rank_tertiles(list(reversed(rows)))

    assert forward == reverse
    assert {bin_value for bins in forward.values() for bin_value in bins.values()} <= {0, 1, 2}


def test_freeze_selects_disjoint_test_q24_and_test_c24() -> None:
    """Catch C24 selection reusing a Q24 scenario."""
    freeze = _freeze_module()
    result = freeze.freeze_selection(_descriptor_rows(), selection_seed="unit-seed")

    assert len(result["test_q24"]) == 24
    assert len(result["test_c24"]) == 24
    assert not set(result["test_q24"]).intersection(result["test_c24"])


def test_freeze_selects_exact_unseen24_and_g3_five_plus_five() -> None:
    """Catch a freeze with the wrong unseen or closed-loop cohort sizes."""
    freeze = _freeze_module()
    result = freeze.freeze_selection(_descriptor_rows(), selection_seed="unit-seed")

    assert len(result["unseen24"]) == 24
    assert len(result["g3_test_q5"]) == 5
    assert len(result["g3_unseen5"]) == 5
    assert set(result["g3_test_q5"]).issubset(result["test_q24"])
    assert set(result["g3_unseen5"]).issubset(result["unseen24"])
    assert len(result["validation3"]) == 3
    assert len(result["replay3"]) == 3


def test_freeze_rejects_coverage_or_planner_result_fields() -> None:
    """Catch outcome fields leaking into a policy-blind selection input."""
    freeze = _freeze_module()

    with pytest.raises(ValueError, match="forbidden"):
        freeze.extract_policy_blind_descriptor({**_descriptor_rows(1)[0], "coverage_result": 0.99})
    with pytest.raises(ValueError, match="forbidden"):
        freeze.extract_policy_blind_descriptor({**_descriptor_rows(1)[0], "planner_success": True})


def test_same_catalog_and_config_produce_byte_identical_manifest() -> None:
    """Catch nondeterministic selection or JSON serialization drift."""
    freeze = _freeze_module()
    config = {"schema_version": "mid-dual-scenario-freeze/v1", "selection_seed": "unit-seed"}

    first = freeze.build_frozen_manifest(_descriptor_rows(), config=config)
    second = freeze.build_frozen_manifest(list(reversed(_descriptor_rows())), config=config)

    assert first == second


def _coverage_manifest(tmp_path: Path):
    from lunar_exploration_ppo.env.coverage import CoverageMasks
    from lunar_exploration_ppo.env.coverage_cache import CoverageCacheKey, Stage6CoverageManifest, serialize_coverage_entry
    from lunar_exploration_ppo.env.scenario import LowResolutionPrior, ScenarioBundle, TruthMap
    from lunar_exploration_ppo.utils.artifact_io import ArtifactStore
    from lunar_exploration_ppo.utils.geometry import CellXY, GridGeometry, PoseXYTheta, WorldXY

    geometry = GridGeometry(4, 4, 1.0, origin=WorldXY(0.0, 0.0))
    truth = TruthMap(
        geometry=geometry,
        height=np.zeros(geometry.shape, dtype=np.float64),
        hard_obstacle=np.zeros(geometry.shape, dtype=bool),
        slope_deg=np.zeros(geometry.shape, dtype=np.float64),
        traversability=np.ones(geometry.shape, dtype=np.float64),
        provenance={"synthetic_source_kind": "synthetic_terrain_obstacle_proxy/v1"},
    )
    scenario = ScenarioBundle(
        scenario_id="audit/scenario-0",
        scenario_hash="a" * 64,
        truth=truth,
        prior=LowResolutionPrior(channels=np.zeros((7, 2, 2), dtype=np.float32), resolution_m=2.0, value_prior_source="constant_neutral/v1"),
        start_pose=PoseXYTheta(CellXY(1, 1), 0.0),
        proxy_catalog=None,
        proxy_layer_hashes={},
    )
    key = CoverageCacheKey.build(scenario, sensor_range_m=3.0, min_clearance_m=0.0, max_slope_deg=30.0, traversability_threshold=0.5)
    mask = np.ones(geometry.shape, dtype=bool)
    masks = CoverageMasks(mask, mask, mask, {
        "algorithm_id": "exact_reachable_safe_pose_range_los/v1",
        "sha256": hashlib.sha256(np.ascontiguousarray(mask).tobytes()).hexdigest(),
        "exact": True,
        "precompute_scope": "scenario_reset/v1",
        "coverable_cell_count": int(mask.size),
    })
    payload = serialize_coverage_entry(key, masks)
    root = tmp_path / "cache"
    path = root / "entries" / key.sha256[:2] / f"{key.sha256}.npz"
    path.parent.mkdir(parents=True)
    path.write_bytes(payload)
    entries = [{
        "scenario_id": scenario.scenario_id, "scenario_hash": scenario.scenario_hash,
        "split": "test", "key_sha256": key.sha256,
        "path": path.relative_to(root).as_posix(), "sha256": hashlib.sha256(payload).hexdigest(), "size_bytes": len(payload),
    }]
    manifest = {
        "schema_version": "stage6_exact_coverable_cache_manifest/v1", "cache_root": str(root), "catalog_sha256": "b" * 64,
        "split_counts": {"train": 0, "validation": 0, "test": 1, "unseen": 0},
        "generation_environment": {"producer": "stage6_coverage_cache_prewarm/v1", "python_version": "3.12", "numpy_version": "2", "platform": "test"},
        "integrity": {"complete": True, "entry_count": 1, "valid_entry_count": 1, "missing_entry_count": 0, "duplicate_entry_count": 0, "corrupt_entry_count": 0},
        "entry_set_sha256": hashlib.sha256(ArtifactStore.canonical_json_bytes(entries)).hexdigest(), "entries": entries,
    }
    raw = ArtifactStore.canonical_json_bytes(manifest)
    manifest_path = tmp_path / "cache-manifest.json"
    manifest_path.write_bytes(raw)
    return Stage6CoverageManifest.load(manifest_path, expected_sha256=hashlib.sha256(raw).hexdigest()), scenario, path


def test_denominator_audit_proves_mask_identity_and_nonempty_count(tmp_path: Path) -> None:
    """Catch an audit that does not bind the exact nonempty cached mask bytes."""
    freeze = _freeze_module()
    manifest, scenario, _ = _coverage_manifest(tmp_path)

    audit = manifest.scenario_audit(scenario.scenario_id)
    proof = freeze.prove_denominator_identity(audit, np.ones((4, 4), dtype=bool))

    assert proof["semantic_alias_proven"] is True
    assert proof["coverage_denominator_source"] == "reachable_observable_free_highres_cells/v1"
    assert proof["coverable_cell_count"] == 16


def test_denominator_hash_drift_blocks_the_whole_manifest(tmp_path: Path) -> None:
    """Catch one denominator drift being emitted as a partially valid freeze."""
    freeze = _freeze_module()
    manifest, scenario, _ = _coverage_manifest(tmp_path)
    audit = manifest.scenario_audit(scenario.scenario_id)

    with pytest.raises(ValueError, match="denominator"):
        freeze.build_frozen_manifest(
            _descriptor_rows(), config={"schema_version": "mid-dual-scenario-freeze/v1", "selection_seed": "unit-seed"},
            denominator_audits=[(audit, np.zeros((4, 4), dtype=bool))],
        )
