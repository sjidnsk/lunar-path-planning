"""Contracts for the policy-blind midterm scenario freeze."""

from __future__ import annotations

import hashlib
import io
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
    return _split_rows()[:count]


def _freeze_module():
    import freeze_xunce_mid_dual_scenarios as freeze

    return freeze


def test_descriptor_extraction_contains_only_policy_independent_fields() -> None:
    """Catch a descriptor serializer that admits policy or outcome evidence."""
    freeze = _freeze_module()
    descriptor = freeze.extract_policy_blind_descriptor(_descriptor_rows(1)[0])

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


def test_same_catalog_and_config_produce_byte_identical_manifest(tmp_path: Path) -> None:
    """Catch nondeterministic selection or JSON serialization drift."""
    freeze = _freeze_module()
    package = _formal_package(tmp_path)
    first = freeze.build_freeze_bundle_from_paths(
        config_path=package["config_path"], descriptor_catalog_path=package["descriptor_path"],
        source_manifest_path=package["source_manifest_path"], coverage_manifest_path=package["coverage_path"],
        coverage_manifest_sha256=package["coverage_sha256"], reconstruction_index_path=package["reconstruction_path"],
    )["manifest.json"]
    lines = Path(package["descriptor_path"]).read_text(encoding="utf-8").splitlines()
    Path(package["descriptor_path"]).write_text("\n".join(reversed(lines)) + "\n", encoding="utf-8")
    second = freeze.build_freeze_bundle_from_paths(
        config_path=package["config_path"], descriptor_catalog_path=package["descriptor_path"],
        source_manifest_path=package["source_manifest_path"], coverage_manifest_path=package["coverage_path"],
        coverage_manifest_sha256=package["coverage_sha256"], reconstruction_index_path=package["reconstruction_path"],
    )["manifest.json"]

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
    descriptor = {
        "scenario_id": scenario.scenario_id,
        "scenario_hash": scenario.scenario_hash,
        "source_split": "test",
        "coverable_cell_count": 16,
    }
    reconstruction = {
        "scenario_id": scenario.scenario_id,
        "scenario_hash": scenario.scenario_hash,
        "key_sha256": audit["key_sha256"],
        "mask_path": "reconstructed.npz",
        "mask_file_sha256": "f" * 64,
        "mask_size_bytes": 1,
    }
    proof = freeze.prove_denominator_identity(
        audit,
        descriptor,
        reconstruction,
        np.ones((4, 4), dtype=bool),
    )

    assert proof["semantic_alias_proven"] is True
    assert proof["coverage_denominator_source"] == "reachable_observable_free_highres_cells/v1"
    assert proof["coverable_cell_count"] == 16


def test_denominator_hash_drift_blocks_the_whole_manifest(tmp_path: Path) -> None:
    """Catch one denominator drift being emitted as a partially valid freeze."""
    freeze = _freeze_module()
    manifest, scenario, _ = _coverage_manifest(tmp_path)
    audit = manifest.scenario_audit(scenario.scenario_id)
    descriptor = {
        "scenario_id": scenario.scenario_id,
        "scenario_hash": scenario.scenario_hash,
        "source_split": "test",
        "coverable_cell_count": 16,
    }
    reconstruction = {
        "scenario_id": scenario.scenario_id,
        "scenario_hash": scenario.scenario_hash,
        "key_sha256": audit["key_sha256"],
        "mask_path": "reconstructed.npz",
        "mask_file_sha256": "f" * 64,
        "mask_size_bytes": 1,
    }
    with pytest.raises(ValueError, match="denominator"):
        freeze.prove_denominator_identity(
            audit,
            descriptor,
            reconstruction,
            np.zeros((4, 4), dtype=bool),
        )


@pytest.mark.parametrize(
    "field,value",
    (
        ("scenario_id", "test/not-the-audited-scenario"),
        ("scenario_hash", "f" * 64),
        ("source_split", "unseen"),
        ("coverable_cell_count", 15),
    ),
)
def test_descriptor_to_stage6_audit_identity_drift_blocks_proof(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    freeze = _freeze_module()
    manifest, scenario, _ = _coverage_manifest(tmp_path)
    audit = manifest.scenario_audit(scenario.scenario_id)
    descriptor = {
        "scenario_id": scenario.scenario_id,
        "scenario_hash": scenario.scenario_hash,
        "source_split": "test",
        "coverable_cell_count": 16,
    }
    descriptor[field] = value
    reconstruction = {
        "scenario_id": scenario.scenario_id,
        "scenario_hash": scenario.scenario_hash,
        "key_sha256": audit["key_sha256"],
        "mask_path": "reconstructed.npz",
        "mask_file_sha256": "f" * 64,
        "mask_size_bytes": 1,
    }
    with pytest.raises(ValueError, match="denominator"):
        freeze.prove_denominator_identity(
            audit,
            descriptor,
            reconstruction,
            np.ones((4, 4), dtype=bool),
        )


@pytest.mark.parametrize(
    "mask",
    (
        np.ones((2, 8), dtype=bool),
        np.zeros((4, 4), dtype=bool),
        np.ones((4, 4), dtype=np.uint8),
    ),
)
def test_reconstructed_mask_shape_count_and_dtype_drift_block_proof(
    tmp_path: Path,
    mask: np.ndarray,
) -> None:
    freeze = _freeze_module()
    manifest, scenario, _ = _coverage_manifest(tmp_path)
    audit = manifest.scenario_audit(scenario.scenario_id)
    descriptor = {
        "scenario_id": scenario.scenario_id,
        "scenario_hash": scenario.scenario_hash,
        "source_split": "test",
        "coverable_cell_count": 16,
    }
    reconstruction = {
        "scenario_id": scenario.scenario_id,
        "scenario_hash": scenario.scenario_hash,
        "key_sha256": audit["key_sha256"],
        "mask_path": "reconstructed.npz",
        "mask_file_sha256": "f" * 64,
        "mask_size_bytes": 1,
    }
    with pytest.raises(ValueError, match="denominator"):
        freeze.prove_denominator_identity(audit, descriptor, reconstruction, mask)


_POOL_HASH_FIELDS = (
    "scenario_id",
    "scenario_hash",
    "source_split",
    "slope_p90_deg",
    "hard_obstacle_fraction",
    "start_pose_bin",
    "initial_observed_coverable_fraction",
    "initial_valid_frontier_count",
    "coverable_cell_count",
    "parent_roi",
    "density_profile",
    "start_to_farthest_candidate_distance_bin",
)


def _canonical_jsonl(rows: list[dict[str, object]]) -> bytes:
    return "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for row in sorted(rows, key=lambda item: str(item["scenario_id"]))
    ).encode("utf-8")


def _split_rows() -> list[dict[str, object]]:
    split_counts = {"test": 50, "unseen": 25, "validation": 4}
    rows: list[dict[str, object]] = []
    absolute_index = 0
    for split, count in split_counts.items():
        pool_rows: list[dict[str, object]] = []
        for split_index in range(count):
            scenario_id = f"{split}/scenario-{split_index:03d}"
            row = {
                "scenario_id": scenario_id,
                "scenario_hash": hashlib.sha256(scenario_id.encode()).hexdigest(),
                "source_split": split,
                "slope_p90_deg": float(5 + absolute_index % 17),
                "hard_obstacle_fraction": float((absolute_index % 11) / 100),
                "start_pose_bin": [absolute_index % 2, (absolute_index // 2) % 2, absolute_index % 4],
                "initial_observed_coverable_fraction": float((absolute_index % 7) / 10),
                "initial_valid_frontier_count": 10 + absolute_index % 13,
                "coverable_cell_count": 16,
                "parent_roi": f"{split}-roi-{absolute_index % 10}",
                "density_profile": ("low", "medium", "high")[absolute_index % 3],
                "start_to_farthest_candidate_distance_bin": absolute_index % 3,
            }
            pool_rows.append(row)
            absolute_index += 1
        pool_basis = [{field: row[field] for field in _POOL_HASH_FIELDS} for row in pool_rows]
        pool_hash = hashlib.sha256(_canonical_jsonl(pool_basis)).hexdigest()
        for row in pool_rows:
            row["source_pool_sha256"] = pool_hash
        rows.extend(pool_rows)
    return rows


@pytest.mark.parametrize(
    "pose",
    (
        [-1, 0, 0],
        [2, 0, 0],
        [0, -1, 0],
        [0, 2, 0],
        [0, 0, -1],
        [0, 0, 4],
        [True, 0, 0],
        [0, False, 0],
        [0, 0, True],
    ),
)
def test_start_pose_bin_rejects_negative_out_of_range_and_bool(pose: list[object]) -> None:
    freeze = _freeze_module()
    row = _split_rows()[0]
    row["start_pose_bin"] = pose
    with pytest.raises(ValueError, match="start_pose_bin"):
        freeze.extract_policy_blind_descriptor(row)


def test_source_pools_are_split_bound_and_replay_is_a_formal_subset() -> None:
    freeze = _freeze_module()
    rows = _split_rows()
    result = freeze.freeze_selection(rows, selection_seed="split-seed")
    split_by_id = {str(row["scenario_id"]): row["source_split"] for row in rows}

    assert {split_by_id[item] for item in result["test_q24"] + result["test_c24"]} == {"test"}
    assert {split_by_id[item] for item in result["unseen24"]} == {"unseen"}
    assert {split_by_id[item] for item in result["validation3"]} == {"validation"}
    assert set(result["replay3"]).issubset(result["test_q24"])


def test_cross_split_or_distinct_id_hash_reuse_fails_whole_freeze() -> None:
    freeze = _freeze_module()
    rows = _split_rows()
    rows[-1]["scenario_hash"] = rows[0]["scenario_hash"]
    with pytest.raises(ValueError, match="hash"):
        freeze.freeze_selection(rows, selection_seed="split-seed")


def test_source_pool_hash_drift_fails_whole_freeze() -> None:
    freeze = _freeze_module()
    rows = _split_rows()
    rows[0]["source_pool_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="source pool"):
        freeze.freeze_selection(rows, selection_seed="split-seed")


def test_formal_manifest_builder_rejects_missing_or_empty_denominator_evidence() -> None:
    freeze = _freeze_module()
    with pytest.raises(ValueError, match="denominator|evidence"):
        freeze.build_frozen_manifest(None)
    with pytest.raises(ValueError, match="denominator"):
        freeze.build_frozen_manifest([])


def test_stage6_scenario_audit_preserves_split_and_complete_entry_evidence(tmp_path: Path) -> None:
    manifest, scenario, _ = _coverage_manifest(tmp_path)
    audit = manifest.scenario_audit(scenario.scenario_id)
    assert audit["split"] == "test"
    assert set(audit) == {
        "scenario_id", "scenario_hash", "split", "entry_path", "entry_sha256", "entry_size_bytes",
        "key_sha256", "key", "coverable_mask_sha256", "coverable_cell_count", "geometry", "algorithm_id", "exact",
    }


def test_stage6_scenario_audit_rejects_manifest_hash_not_bound_to_entry_key(tmp_path: Path) -> None:
    from lunar_exploration_ppo.env.coverage_cache import CoverageCacheError, Stage6CoverageManifest
    from lunar_exploration_ppo.utils.artifact_io import ArtifactStore

    manifest, scenario, _ = _coverage_manifest(tmp_path)
    payload = json.loads(manifest.path.read_text(encoding="utf-8"))
    payload["entries"][0]["scenario_hash"] = "c" * 64
    payload["entry_set_sha256"] = hashlib.sha256(
        ArtifactStore.canonical_json_bytes(payload["entries"])
    ).hexdigest()
    raw = ArtifactStore.canonical_json_bytes(payload)
    path = tmp_path / "key-hash-drift-manifest.json"
    path.write_bytes(raw)
    drifted = Stage6CoverageManifest.load(path, expected_sha256=hashlib.sha256(raw).hexdigest())

    with pytest.raises(CoverageCacheError, match="scenario hash|key"):
        drifted.scenario_audit(scenario.scenario_id)


def _write_bytes(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> bytes:
    raw = _canonical_jsonl(rows)
    _write_bytes(path, raw)
    return raw


def _formal_package(tmp_path: Path) -> dict[str, object]:
    from lunar_exploration_ppo.env.coverage import CoverageMasks
    from lunar_exploration_ppo.env.coverage_cache import CoverageCacheKey, serialize_coverage_entry
    from lunar_exploration_ppo.env.scenario import LowResolutionPrior, ScenarioBundle, TruthMap
    from lunar_exploration_ppo.utils.artifact_io import ArtifactStore
    from lunar_exploration_ppo.utils.geometry import CellXY, GridGeometry, PoseXYTheta, WorldXY

    package_root = tmp_path / "inputs"
    rows = _split_rows()
    descriptor_path = package_root / "descriptors.jsonl"
    descriptor_bytes = _write_jsonl(descriptor_path, rows)
    artifacts: dict[str, dict[str, object]] = {}
    for name, payload in (
        ("standard_catalog", b"standard catalog bytes\n"),
        ("standard_source", b"standard source bytes\n"),
        ("static_truth_cache", b"static truth cache bytes\n"),
        ("reset_state", b"reset state bytes\n"),
        ("generator", b"descriptor generator implementation\n"),
        ("approval", b"external approval artifact\n"),
    ):
        path = package_root / f"{name}.bin"
        artifacts[name] = {"artifact_id": f"{name}/v1", "path": str(path), "sha256": _write_bytes(path, payload)}

    catalog_sha256 = str(artifacts["standard_catalog"]["sha256"])
    cache_root = package_root / "coverage-cache"
    geometry = GridGeometry(4, 4, 1.0, origin=WorldXY(0.0, 0.0))
    truth = TruthMap(
        geometry=geometry,
        height=np.zeros(geometry.shape, dtype=np.float64),
        hard_obstacle=np.zeros(geometry.shape, dtype=bool),
        slope_deg=np.zeros(geometry.shape, dtype=np.float64),
        traversability=np.ones(geometry.shape, dtype=np.float64),
        provenance={"synthetic_source_kind": "synthetic_terrain_obstacle_proxy/v1"},
    )
    prior = LowResolutionPrior(channels=np.zeros((7, 2, 2), dtype=np.float32), resolution_m=2.0, value_prior_source="constant_neutral/v1")
    mask = np.ones(geometry.shape, dtype=bool)
    mask_sha256 = hashlib.sha256(np.ascontiguousarray(mask).tobytes()).hexdigest()
    entries: list[dict[str, object]] = []
    reconstruction_rows: list[dict[str, object]] = []
    for row in rows:
        scenario = ScenarioBundle(
            scenario_id=str(row["scenario_id"]), scenario_hash=str(row["scenario_hash"]), truth=truth, prior=prior,
            start_pose=PoseXYTheta(CellXY(1, 1), 0.0), proxy_catalog=None, proxy_layer_hashes={},
        )
        key = CoverageCacheKey.build(scenario, sensor_range_m=3.0, min_clearance_m=0.0, max_slope_deg=30.0, traversability_threshold=0.5)
        masks = CoverageMasks(mask, mask, mask, {
            "algorithm_id": "exact_reachable_safe_pose_range_los/v1", "sha256": mask_sha256,
            "exact": True, "precompute_scope": "scenario_reset/v1", "coverable_cell_count": 16,
        })
        entry_payload = serialize_coverage_entry(key, masks)
        entry_path = cache_root / "entries" / key.sha256[:2] / f"{key.sha256}.npz"
        _write_bytes(entry_path, entry_payload)
        entries.append({
            "scenario_id": scenario.scenario_id, "scenario_hash": scenario.scenario_hash,
            "split": row["source_split"], "key_sha256": key.sha256,
            "path": entry_path.relative_to(cache_root).as_posix(), "sha256": hashlib.sha256(entry_payload).hexdigest(), "size_bytes": len(entry_payload),
        })
        mask_buffer = io.BytesIO()
        np.savez(mask_buffer, coverable_mask=mask)
        mask_payload = mask_buffer.getvalue()
        mask_path = package_root / "reconstructed" / f"{hashlib.sha256(scenario.scenario_id.encode()).hexdigest()}.npz"
        reconstruction_rows.append({
            "scenario_id": scenario.scenario_id, "scenario_hash": scenario.scenario_hash, "key_sha256": key.sha256,
            "mask_path": str(mask_path), "mask_file_sha256": _write_bytes(mask_path, mask_payload), "mask_size_bytes": len(mask_payload),
        })
    entries.sort(key=lambda item: str(item["scenario_id"]))
    split_counts = {split: sum(entry["split"] == split for entry in entries) for split in ("train", "validation", "test", "unseen")}
    coverage_payload = {
        "schema_version": "stage6_exact_coverable_cache_manifest/v1", "cache_root": str(cache_root), "catalog_sha256": catalog_sha256,
        "split_counts": split_counts,
        "generation_environment": {"producer": "stage6_coverage_cache_prewarm/v1", "python_version": "3.12", "numpy_version": "2", "platform": "test"},
        "integrity": {"complete": True, "entry_count": len(entries), "valid_entry_count": len(entries), "missing_entry_count": 0, "duplicate_entry_count": 0, "corrupt_entry_count": 0},
        "entry_set_sha256": hashlib.sha256(ArtifactStore.canonical_json_bytes(entries)).hexdigest(), "entries": entries,
    }
    coverage_bytes = ArtifactStore.canonical_json_bytes(coverage_payload)
    coverage_path = package_root / "coverage-manifest.json"
    coverage_sha256 = _write_bytes(coverage_path, coverage_bytes)
    reconstruction_path = package_root / "reconstruction-index.jsonl"
    reconstruction_bytes = _write_jsonl(reconstruction_path, reconstruction_rows)
    pool_hashes = {split: str(next(row["source_pool_sha256"] for row in rows if row["source_split"] == split)) for split in ("test", "unseen", "validation")}
    source_manifest = {
        "schema_version": "mid-dual-scenario-source-manifest/v1",
        "descriptor_catalog": {"artifact_id": "descriptor-catalog/v1", "path": str(descriptor_path), "sha256": hashlib.sha256(descriptor_bytes).hexdigest()},
        "standard_catalog": artifacts["standard_catalog"], "standard_source": artifacts["standard_source"],
        "static_truth_cache": artifacts["static_truth_cache"], "reset_state": artifacts["reset_state"],
        "descriptor_generator": {"id": "policy-blind-descriptor-generator/v1", "version": "1.0.0", "implementation_path": artifacts["generator"]["path"], "implementation_sha256": artifacts["generator"]["sha256"]},
        "stage6_coverage_manifest": {"path": str(coverage_path), "sha256": coverage_sha256, "catalog_sha256": catalog_sha256},
        "policy_blind_attestation": {
            "attestation_id": "external-policy-blind-attestation/v1", "approval_id": "approval-20260726/v1",
            "approval_artifact_path": artifacts["approval"]["path"], "approval_artifact_sha256": artifacts["approval"]["sha256"],
            "forbidden_inputs_used": {"policy": False, "checkpoint": False, "reward": False, "runtime": False, "result": False},
        },
        "source_pools": {split: {"sha256": digest} for split, digest in pool_hashes.items()},
    }
    source_manifest_bytes = json.dumps(source_manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    source_manifest_path = package_root / "source-manifest.json"
    _write_bytes(source_manifest_path, source_manifest_bytes)
    config = {
        "schema_version": "mid-dual-scenario-freeze/v1", "selection_seed": "unit-seed",
        "coverage_denominator_source": "reachable_observable_free_highres_cells/v1",
        "coverage_denominator_algorithm": "exact_reachable_safe_pose_range_los/v1",
        "source_manifest_schema": "mid-dual-scenario-source-manifest/v1",
        "reconstruction_index_schema": "mid-dual-denominator-reconstruction-index/v1",
        "publication_mode": "data_files_then_completion_manifest/v1",
        "cohort_sizes": {"test_q24": 24, "test_c24": 24, "unseen24": 24, "g3_test_q5": 5, "g3_unseen5": 5, "validation3": 3, "replay3": 3},
    }
    config_bytes = json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    config_path = package_root / "config.json"
    _write_bytes(config_path, config_bytes)
    return {
        "rows": rows, "descriptor_path": descriptor_path, "source_manifest": source_manifest,
        "source_manifest_path": source_manifest_path, "coverage_path": coverage_path, "coverage_sha256": coverage_sha256,
        "reconstruction_rows": reconstruction_rows, "reconstruction_path": reconstruction_path, "config_path": config_path,
    }


def _build_args(package: dict[str, object], output_root: Path, *, execute: bool = True) -> list[str]:
    args = [
        "--config", str(package["config_path"]), "--descriptor-catalog", str(package["descriptor_path"]),
        "--source-manifest", str(package["source_manifest_path"]), "--coverage-manifest", str(package["coverage_path"]),
        "--coverage-manifest-sha256", str(package["coverage_sha256"]), "--reconstruction-index", str(package["reconstruction_path"]),
        "--output-root", str(output_root),
    ]
    if execute:
        args.append("--execute")
    return args


@pytest.mark.parametrize("mutation", ("missing_field", "fake_sha", "bytes_drift", "policy_derived", "missing_approval"))
def test_source_manifest_failures_block_the_whole_freeze(tmp_path: Path, mutation: str) -> None:
    freeze = _freeze_module()
    package = _formal_package(tmp_path)
    source = json.loads(json.dumps(package["source_manifest"]))
    if mutation == "missing_field":
        source.pop("reset_state")
    elif mutation == "fake_sha":
        source["standard_source"]["sha256"] = "f" * 64
    elif mutation == "bytes_drift":
        Path(source["static_truth_cache"]["path"]).write_bytes(b"drift\n")
    elif mutation == "policy_derived":
        source["policy_blind_attestation"]["forbidden_inputs_used"]["policy"] = True
    else:
        source["policy_blind_attestation"].pop("approval_id")
    Path(package["source_manifest_path"]).write_text(json.dumps(source, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    with pytest.raises(ValueError, match="source manifest|attestation|artifact"):
        freeze.build_freeze_bundle_from_paths(
            config_path=package["config_path"], descriptor_catalog_path=package["descriptor_path"],
            source_manifest_path=package["source_manifest_path"], coverage_manifest_path=package["coverage_path"],
            coverage_manifest_sha256=package["coverage_sha256"], reconstruction_index_path=package["reconstruction_path"],
        )


@pytest.mark.parametrize("mutation", ("missing", "empty", "duplicate", "extra", "hash", "key", "mask_hash", "mask_size"))
def test_reconstruction_index_must_be_exact_and_one_to_one(tmp_path: Path, mutation: str) -> None:
    freeze = _freeze_module()
    package = _formal_package(tmp_path)
    rows = json.loads(json.dumps(package["reconstruction_rows"]))
    if mutation == "missing":
        rows.pop()
    elif mutation == "empty":
        rows = []
    elif mutation == "duplicate":
        rows.append(dict(rows[0]))
    elif mutation == "extra":
        extra = dict(rows[0]); extra["scenario_id"] = "test/extra"; rows.append(extra)
    elif mutation == "hash":
        rows[0]["scenario_hash"] = "f" * 64
    elif mutation == "key":
        rows[0]["key_sha256"] = "f" * 64
    elif mutation == "mask_hash":
        rows[0]["mask_file_sha256"] = "f" * 64
    else:
        rows[0]["mask_size_bytes"] += 1
    _write_jsonl(Path(package["reconstruction_path"]), rows)
    with pytest.raises(ValueError, match="reconstruction|denominator|mask"):
        freeze.build_freeze_bundle_from_paths(
            config_path=package["config_path"], descriptor_catalog_path=package["descriptor_path"], source_manifest_path=package["source_manifest_path"],
            coverage_manifest_path=package["coverage_path"], coverage_manifest_sha256=package["coverage_sha256"], reconstruction_index_path=package["reconstruction_path"],
        )


def test_cli_without_execute_reads_nothing_and_writes_nothing(tmp_path: Path) -> None:
    freeze = _freeze_module()
    output_root = tmp_path / "out"
    with pytest.raises(SystemExit, match="--execute"):
        freeze.main([
            "--config", str(tmp_path / "missing-config"), "--descriptor-catalog", str(tmp_path / "missing-descriptors"),
            "--source-manifest", str(tmp_path / "missing-source"), "--coverage-manifest", str(tmp_path / "missing-coverage"),
            "--coverage-manifest-sha256", "f" * 64, "--reconstruction-index", str(tmp_path / "missing-index"),
            "--output-root", str(output_root),
        ])
    assert not output_root.exists()


def test_legacy_fake_audit_cli_cannot_produce_a_complete_root(tmp_path: Path) -> None:
    freeze = _freeze_module()
    output_root = tmp_path / "legacy-out"
    with pytest.raises(SystemExit):
        freeze.main(
            [
                "--config",
                str(tmp_path / "config.json"),
                "--descriptor-catalog",
                str(tmp_path / "descriptors.jsonl"),
                "--denominator-audits",
                str(tmp_path / "fake-audits.jsonl"),
                "--output-root",
                str(output_root),
                "--execute",
            ]
        )
    assert not output_root.exists()


def test_cli_bundle_is_complete_recalculable_order_stable_and_not_overwritten(tmp_path: Path) -> None:
    freeze = _freeze_module()
    package = _formal_package(tmp_path)
    output_root = tmp_path / "out"
    assert freeze.main(_build_args(package, output_root)) == 0
    roots = list(output_root.iterdir())
    assert len(roots) == 1
    freeze_root = roots[0]
    assert freeze.verify_frozen_bundle(freeze_root) is True
    manifest_before = (freeze_root / "manifest.json").read_bytes()
    descriptor_rows = Path(package["descriptor_path"]).read_text(encoding="utf-8").splitlines()
    Path(package["descriptor_path"]).write_text("\n".join(reversed(descriptor_rows)) + "\n", encoding="utf-8")
    reconstruction_rows = Path(package["reconstruction_path"]).read_text(encoding="utf-8").splitlines()
    Path(package["reconstruction_path"]).write_text("\n".join(reversed(reconstruction_rows)) + "\n", encoding="utf-8")
    rebuilt = freeze.build_freeze_bundle_from_paths(
        config_path=package["config_path"], descriptor_catalog_path=package["descriptor_path"], source_manifest_path=package["source_manifest_path"],
        coverage_manifest_path=package["coverage_path"], coverage_manifest_sha256=package["coverage_sha256"], reconstruction_index_path=package["reconstruction_path"],
    )
    assert rebuilt["manifest.json"] == manifest_before
    with pytest.raises(FileExistsError, match="complete"):
        freeze.main(_build_args(package, output_root))

    manifest = json.loads(manifest_before)
    manifest["cohorts"]["replay3"] = ["unseen/scenario-000", *manifest["cohorts"]["replay3"][1:]]
    (freeze_root / "manifest.json").write_bytes(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    with pytest.raises(ValueError, match="semantic manifest"):
        freeze.verify_frozen_bundle(freeze_root)


@pytest.mark.parametrize("failed_name", ("denominator-proofs.jsonl", "manifest.json"))
def test_publish_failure_never_leaves_accepted_completion_and_is_resumable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failed_name: str) -> None:
    freeze = _freeze_module()
    package = _formal_package(tmp_path)
    bundle = freeze.build_freeze_bundle_from_paths(
        config_path=package["config_path"], descriptor_catalog_path=package["descriptor_path"], source_manifest_path=package["source_manifest_path"],
        coverage_manifest_path=package["coverage_path"], coverage_manifest_sha256=package["coverage_sha256"], reconstruction_index_path=package["reconstruction_path"],
    )
    output_root = tmp_path / "fault-out"
    original = freeze.artifact_io.write_text
    failed = False

    def fail_once(path, text, *, encoding="utf-8"):
        nonlocal failed
        if not failed and Path(path).name == failed_name:
            failed = True
            original(path, text[: max(1, len(text) // 2)], encoding=encoding)
            raise OSError("injected publish failure")
        return original(path, text, encoding=encoding)

    monkeypatch.setattr(freeze.artifact_io, "write_text", fail_once)
    with pytest.raises(OSError, match="injected"):
        freeze.publish_frozen_bundle(bundle, output_root=output_root)
    freeze_root = output_root / freeze.freeze_id(bundle["manifest.json"])
    with pytest.raises((FileNotFoundError, ValueError)):
        freeze.verify_frozen_bundle(freeze_root)
    monkeypatch.setattr(freeze.artifact_io, "write_text", original)
    recovered = freeze.publish_frozen_bundle(bundle, output_root=output_root)
    assert recovered == freeze_root
    assert freeze.verify_frozen_bundle(freeze_root) is True


def test_incomplete_root_drift_blocks_recovery(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    freeze = _freeze_module()
    package = _formal_package(tmp_path)
    bundle = freeze.build_freeze_bundle_from_paths(
        config_path=package["config_path"],
        descriptor_catalog_path=package["descriptor_path"],
        source_manifest_path=package["source_manifest_path"],
        coverage_manifest_path=package["coverage_path"],
        coverage_manifest_sha256=package["coverage_sha256"],
        reconstruction_index_path=package["reconstruction_path"],
    )
    output_root = tmp_path / "drift-out"
    original = freeze.artifact_io.write_text

    def fail_on_proofs(path, text, *, encoding="utf-8"):
        if Path(path).name == "denominator-proofs.jsonl":
            raise OSError("injected data failure")
        return original(path, text, encoding=encoding)

    monkeypatch.setattr(freeze.artifact_io, "write_text", fail_on_proofs)
    with pytest.raises(OSError, match="injected"):
        freeze.publish_frozen_bundle(bundle, output_root=output_root)
    freeze_root = output_root / freeze.freeze_id(bundle["manifest.json"])
    (freeze_root / "config.json").write_text("drift", encoding="utf-8")
    monkeypatch.setattr(freeze.artifact_io, "write_text", original)
    with pytest.raises(ValueError, match="drifted"):
        freeze.publish_frozen_bundle(bundle, output_root=output_root)
