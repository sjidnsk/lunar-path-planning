from __future__ import annotations

import hashlib
import importlib.util
import io
import json
from dataclasses import asdict, dataclass
from pathlib import Path
import sys
from types import SimpleNamespace
import zipfile

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "prepare_xunce_mid_dual_scenario_sources.py"
CONFIG_PATH = REPO_ROOT / "configs" / "xunce_mid_dual_scenario_sources_v1.json"
AUTHORIZATION_SHA256 = (
    "720e11ef04ad2b57283421809a077ccf1f0b35167a9f482082241398ad0214d2"
)
SOURCE_SPLIT_COUNTS = {"test": 150, "unseen": 64, "validation": 150}
SAFETY_CONTRACT = {
    "vehicle_radius_m": 0.4215874761,
    "safety_margin_m": 0.10,
    "min_clearance_m": 0.5215874761,
    "traversability_threshold": 0.50,
    "max_traversable_slope_deg": 30.0,
}


def _load_module():
    assert SCRIPT_PATH.is_file(), "scenario source preparer is missing"
    spec = importlib.util.spec_from_file_location("mid_dual_scenario_sources", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    if str(SCRIPT_PATH.parent) not in sys.path:
        sys.path.insert(0, str(SCRIPT_PATH.parent))
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_freeze_module():
    path = REPO_ROOT / "scripts" / "freeze_xunce_mid_dual_scenarios.py"
    spec = importlib.util.spec_from_file_location("mid_dual_scenario_freeze_for_source", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _descriptor(scenario_id: str, split: str, scenario_hash: str) -> dict[str, object]:
    return {
        "scenario_id": scenario_id,
        "scenario_hash": scenario_hash,
        "source_split": split,
        "source_pool_sha256": "0" * 64,
        "slope_p90_deg": 12.5,
        "hard_obstacle_fraction": 0.125,
        "start_pose_bin": [0, 1, 2],
        "initial_observed_coverable_fraction": 0.0625,
        "initial_valid_frontier_count": 8,
        "coverable_cell_count": 1024,
        "parent_roi": "r00001-c00002",
        "density_profile": "medium",
        "start_to_farthest_candidate_distance_bin": 0,
    }


def _complete_payloads(
    module,
    *,
    source_root: Path | None,
) -> tuple[dict[str, bytes], dict[str, bytes]]:
    rows: list[dict[str, object]] = []
    for split, count in SOURCE_SPLIT_COUNTS.items():
        for index in range(count):
            scenario_id = f"{split}/scenario-{index:03d}/standard-proxy/v1"
            scenario_hash = hashlib.sha256(scenario_id.encode("utf-8")).hexdigest()
            rows.append(_descriptor(scenario_id, split, scenario_hash))
    descriptors = module.bind_source_pool_hashes(rows)
    pools = {
        split: next(
            str(row["source_pool_sha256"])
            for row in descriptors
            if row["source_split"] == split
        )
        for split in SOURCE_SPLIT_COUNTS
    }
    mask_payload = module.serialize_reconstruction_mask(
        np.asarray([[True, False], [True, True]], dtype=np.bool_)
    )
    masks = {
        f"masks/{row['scenario_hash']}.npz": mask_payload
        for row in descriptors
    }
    reconstruction = []
    for row in descriptors:
        relative = f"masks/{row['scenario_hash']}.npz"
        mask_path = relative if source_root is None else str((source_root / relative).resolve())
        reconstruction.append(
            {
                "scenario_id": row["scenario_id"],
                "scenario_hash": row["scenario_hash"],
                "key_sha256": hashlib.sha256(
                    f"key:{row['scenario_id']}".encode("utf-8")
                ).hexdigest(),
                "mask_path": mask_path,
                "mask_file_sha256": hashlib.sha256(mask_payload).hexdigest(),
                "mask_size_bytes": len(mask_payload),
            }
        )
    catalog = module._canonical_json_bytes({"catalog": "fixture"})
    standard_source = module._canonical_json_bytes(
        {
            "schema_version": "mid-dual-standard-source-provenance/v1",
            "source_config": {
                "path": "C:/repo/configs/xunce_mid_dual_scenario_sources_v1.json",
                "sha256": "1" * 64,
            },
            "safety_contract": SAFETY_CONTRACT,
            "sensor_range_m": 20.0,
            "forbidden_inputs_used": {
                "policy": False,
                "checkpoint": False,
                "reward": False,
                "runtime": False,
                "result": False,
            },
        }
    )
    static = module._canonical_jsonl_bytes(
        {
            "scenario_id": row["scenario_id"],
            "scenario_hash": row["scenario_hash"],
        }
        for row in descriptors
    )
    reset = module._canonical_jsonl_bytes(
        {
            "scenario_id": row["scenario_id"],
            "scenario_hash": row["scenario_hash"],
        }
        for row in descriptors
    )
    descriptor_bytes = module._canonical_jsonl_bytes(descriptors)
    reconstruction_bytes = module._canonical_jsonl_bytes(reconstruction)
    approval = module._canonical_json_bytes(
        {
            "schema_version": "mid-dual-policy-blind-source-approval/v1",
            "formal_gate_pass_approved": False,
        }
    )
    relative = {
        "descriptor_catalog": "descriptors.jsonl",
        "standard_catalog": "standard-catalog.json",
        "standard_source": "standard-source.json",
        "static_truth_cache": "static-truth-index.jsonl",
        "reset_state": "reset-state-index.jsonl",
    }
    payload_by_name = {
        "descriptors.jsonl": descriptor_bytes,
        "standard-catalog.json": catalog,
        "standard-source.json": standard_source,
        "static-truth-index.jsonl": static,
        "reset-state-index.jsonl": reset,
    }
    refs = {
        name: {
            "artifact_id": f"{name}/v1",
            "path": (
                path
                if source_root is None
                else str((source_root / path).resolve())
            ),
            "sha256": hashlib.sha256(payload_by_name[path]).hexdigest(),
        }
        for name, path in relative.items()
    }
    source_manifest = module.build_source_manifest(
        artifact_refs=refs,
        descriptor_generator_path="C:/repo/scripts/prepare_xunce_mid_dual_scenario_sources.py",
        descriptor_generator_sha256="2" * 64,
        coverage_manifest_path="D:/xunce/out/cache/coverage-cache-manifest.json",
        coverage_manifest_sha256="3" * 64,
        coverage_catalog_sha256=hashlib.sha256(catalog).hexdigest(),
        approval_path=(
            "policy-blind-approval.json"
            if source_root is None
            else str((source_root / "policy-blind-approval.json").resolve())
        ),
        approval_sha256=hashlib.sha256(approval).hexdigest(),
        source_pool_hashes=pools,
    )
    return (
        {
            **payload_by_name,
            "reconstruction-index.jsonl": reconstruction_bytes,
            "policy-blind-approval.json": approval,
            "source-manifest.json": module._canonical_json_bytes(source_manifest),
        },
        masks,
    )


def test_config_pins_policy_blind_full_source_contract() -> None:
    assert CONFIG_PATH.is_file(), "scenario source config is missing"
    value = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    assert value == {
        "schema_version": "mid-dual-scenario-source-materialization/v1",
        "catalog_split_counts": {"test": 150, "unseen": 64, "validation": 150},
        "excluded_splits": ["train"],
        "coverage_denominator_source": "reachable_observable_free_highres_cells/v1",
        "coverage_denominator_algorithm": "exact_reachable_safe_pose_range_los/v1",
        "distance_bin_algorithm": "within_split_stable_rank_tertiles/v1",
        "policy_blind_approval_id": "mid-dual-policy-blind-source-approval-20260727/v1",
        "project_authorization_sha256": AUTHORIZATION_SHA256,
        "publication_mode": "data_files_then_completion_manifest/v1",
        "sensor_range_m": 20.0,
        "safety_contract_source": "mid-dual-policy-blind-safety-contract/v1",
        "safety_contract": SAFETY_CONTRACT,
    }


def test_source_builder_never_reads_or_binds_the_full_stage6_config() -> None:
    source = SCRIPT_PATH.read_text(encoding="utf-8")
    assert "ppo_highres_frontier_stage6_v1.json" not in source
    assert "parse_stage6_config_bytes" not in source
    assert '"stage6_config": {' not in source


def test_pool_hash_binding_is_order_stable_and_split_isolated() -> None:
    module = _load_module()
    test_a = _descriptor("test-a", "test", "1" * 64)
    test_b = _descriptor("test-b", "test", "2" * 64)
    unseen = _descriptor("unseen-a", "unseen", "3" * 64)
    validation = _descriptor("validation-a", "validation", "4" * 64)

    forward = module.bind_source_pool_hashes([test_b, validation, test_a, unseen])
    reverse = module.bind_source_pool_hashes([unseen, test_a, validation, test_b])

    assert forward == reverse
    by_id = {row["scenario_id"]: row for row in forward}
    assert by_id["test-a"]["source_pool_sha256"] == (
        "10274d3332c9f5e441eb0e52594a08864ba5ac453d1d3c715a06ae8c988b0c18"
    )
    assert by_id["test-b"]["source_pool_sha256"] == by_id["test-a"]["source_pool_sha256"]
    assert by_id["unseen-a"]["source_pool_sha256"] == (
        "799f084a79a17b451cff4fdfb35bb8cc77a51fd630f7152021cc3d4053f79d64"
    )
    assert by_id["validation-a"]["source_pool_sha256"] == (
        "978cab39ec0e7c7853d9733019320051b119e04405009152dc2d309384329dc9"
    )


def test_distance_bins_are_stable_rank_tertiles_with_id_ties() -> None:
    module = _load_module()
    rows = [
        {"scenario_id": "b", "source_split": "test", "max_reset_candidate_distance_m": 5.0},
        {"scenario_id": "a", "source_split": "test", "max_reset_candidate_distance_m": 5.0},
        {"scenario_id": "c", "source_split": "test", "max_reset_candidate_distance_m": 9.0},
        {"scenario_id": "u", "source_split": "unseen", "max_reset_candidate_distance_m": 99.0},
    ]
    assert module.assign_distance_bins(rows) == {
        "a": 0,
        "b": 1,
        "c": 2,
        "u": 0,
    }
    assert module.assign_distance_bins(list(reversed(rows))) == {
        "a": 0,
        "b": 1,
        "c": 2,
        "u": 0,
    }


def test_coverable_mask_npz_is_canonical_and_byte_stable() -> None:
    module = _load_module()
    mask = np.asarray([[True, False], [True, True]], dtype=np.bool_)
    first = module.serialize_reconstruction_mask(mask)
    second = module.serialize_reconstruction_mask(mask.copy())
    assert first == second
    assert hashlib.sha256(first).hexdigest() == (
        "a0b7cd10dcf856475ee1e143fd37d986b282499e9173dab7c05ab5f646ec09db"
    )
    with zipfile.ZipFile(io.BytesIO(first), "r") as archive:
        assert archive.namelist() == ["coverable_mask.npy"]
        info = archive.getinfo("coverable_mask.npy")
        assert info.compress_type == zipfile.ZIP_STORED
        assert info.date_time == (1980, 1, 1, 0, 0, 0)
        with archive.open(info, "r") as handle:
            loaded = np.load(handle, allow_pickle=False)
    assert loaded.dtype == np.dtype(bool)
    assert np.array_equal(loaded, mask)


def test_standard_catalog_artifact_uses_the_catalogs_authoritative_encoding() -> None:
    module = _load_module()
    payload = {
        "schema_version": "standard_unseen_scenario_catalog/v1",
        "split_policy": "fixed_seed_disjoint_split/v1",
        "sources": {"dem": {"sha256": "1" * 64}},
        "records": [{"scenario_id": "test/a"}],
    }
    authoritative = (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")

    class Catalog:
        sha256 = hashlib.sha256(authoritative).hexdigest()

        def to_dict(self):
            return {"catalog_sha256": self.sha256, **payload}

    result = module._catalog_artifact_bytes(Catalog())
    assert result == authoritative
    assert hashlib.sha256(result).hexdigest() == Catalog.sha256


def test_source_manifest_binds_artifacts_authorization_and_forbidden_inputs() -> None:
    module = _load_module()
    refs = {
        name: {
            "artifact_id": f"{name}/v1",
            "path": f"D:/xunce/inputs/mid_dual/source/{name}.json",
            "sha256": character * 64,
        }
        for name, character in {
            "descriptor_catalog": "1",
            "standard_catalog": "2",
            "standard_source": "3",
            "static_truth_cache": "4",
            "reset_state": "5",
        }.items()
    }
    value = module.build_source_manifest(
        artifact_refs=refs,
        descriptor_generator_path="C:/repo/scripts/prepare_xunce_mid_dual_scenario_sources.py",
        descriptor_generator_sha256="6" * 64,
        coverage_manifest_path="D:/xunce/cache/run/coverage-cache-manifest.json",
        coverage_manifest_sha256="7" * 64,
        coverage_catalog_sha256="2" * 64,
        approval_path="D:/xunce/inputs/mid_dual/source/policy-blind-approval.json",
        approval_sha256="8" * 64,
        source_pool_hashes={"test": "9" * 64, "unseen": "a" * 64, "validation": "b" * 64},
    )
    assert value["schema_version"] == "mid-dual-scenario-source-manifest/v1"
    assert value["policy_blind_attestation"] == {
        "attestation_id": "mid-dual-policy-blind-source-attestation/v1",
        "approval_id": "mid-dual-policy-blind-source-approval-20260727/v1",
        "approval_artifact_path": "D:/xunce/inputs/mid_dual/source/policy-blind-approval.json",
        "approval_artifact_sha256": "8" * 64,
        "forbidden_inputs_used": {
            "policy": False,
            "checkpoint": False,
            "reward": False,
            "runtime": False,
            "result": False,
        },
    }
    assert value["stage6_coverage_manifest"]["catalog_sha256"] == "2" * 64


def test_project_authorization_must_exist_and_match_exact_bytes(tmp_path: Path) -> None:
    module = _load_module()
    path = tmp_path / "authorization.md"
    path.write_bytes(b"not the approved authorization")
    with pytest.raises(ValueError, match="authorization SHA-256 drifted"):
        module.validate_project_authorization(path)
    with pytest.raises(ValueError, match="authorization is unreadable"):
        module.validate_project_authorization(tmp_path / "missing.md")


def test_completion_requires_exact_8_data_364_mask_closed_set() -> None:
    module = _load_module()
    with pytest.raises(ValueError, match="exact source data file set"):
        module.build_completion_manifest(
            {"standard-catalog.json": b"{}"},
            {"masks/" + "a" * 64 + ".npz": b"mask"},
        )


def test_publish_writes_completion_last_binds_source_id_and_rejects_extras(
    tmp_path: Path,
) -> None:
    module = _load_module()
    preliminary_data, masks = _complete_payloads(module, source_root=None)
    source_id = module.compute_source_id(
        preliminary_data,
        masks,
        source_root=None,
    )
    root = tmp_path / source_id
    data, masks = _complete_payloads(module, source_root=root)
    completion = module.build_completion_manifest(
        data,
        masks,
        source_root=root,
    )
    root = module.publish_source_bundle(
        data_files=data,
        mask_files=masks,
        completion_manifest=completion,
        output_root=tmp_path,
    )
    assert (root / "manifest.json").read_bytes() == completion
    assert root.name == source_id
    (root / "unexpected.bin").write_bytes(b"rogue")
    with pytest.raises(ValueError, match="exact closed file set"):
        module.verify_source_bundle(root)
    (root / "unexpected.bin").unlink()
    with pytest.raises(FileExistsError, match="complete scenario source root"):
        module.publish_source_bundle(
            data_files=data,
            mask_files=masks,
            completion_manifest=completion,
            output_root=tmp_path,
        )
    with pytest.raises(ValueError, match="source ID"):
        module.publish_source_bundle(
            data_files=data,
            mask_files=masks,
            completion_manifest=completion,
            output_root=tmp_path,
            source_id="0" * 16,
        )


def test_publish_recovers_exact_prefix_but_rejects_noncontiguous_root(
    tmp_path: Path,
) -> None:
    module = _load_module()
    preliminary, masks = _complete_payloads(module, source_root=None)
    source_id = module.compute_source_id(preliminary, masks, source_root=None)
    root = tmp_path / source_id
    data, masks = _complete_payloads(module, source_root=root)
    completion = module.build_completion_manifest(data, masks, source_root=root)
    order = module.publication_order(data, masks)
    first = order[0]
    (root / first).parent.mkdir(parents=True, exist_ok=True)
    (root / first).write_bytes(data[first][: max(1, len(data[first]) // 2)])
    published = module.publish_source_bundle(
        data_files=data,
        mask_files=masks,
        completion_manifest=completion,
        output_root=tmp_path,
    )
    assert published == root
    assert module.verify_source_bundle(root) is True

    second_base = tmp_path / "second"
    second_root = second_base / source_id
    second_data, second_masks = _complete_payloads(module, source_root=second_root)
    second_completion = module.build_completion_manifest(
        second_data,
        second_masks,
        source_root=second_root,
    )
    second_root.mkdir(parents=True)
    late = order[-2]
    late_payload = (
        second_masks[late] if late in second_masks else second_data[late]
    )
    (second_root / late).parent.mkdir(parents=True, exist_ok=True)
    (second_root / late).write_bytes(late_payload)
    with pytest.raises(ValueError, match="contiguous publication prefix"):
        module.publish_source_bundle(
            data_files=second_data,
            mask_files=second_masks,
            completion_manifest=second_completion,
            output_root=second_base,
        )


def test_execution_paths_require_absolute_d_drive_boundaries() -> None:
    module = _load_module()
    coverage, output = module.validate_execution_paths(
        "D:/xunce/out/mid_dual/cache/coverage-cache-manifest.json",
        "D:/xunce/inputs/mid_dual/scenario-sources",
    )
    assert coverage.is_absolute()
    assert output.is_absolute()
    for coverage_path, output_root in (
        ("relative/manifest.json", "D:/xunce/inputs/mid_dual/scenario-sources"),
        ("D:/xunce/out/cache/manifest.json", "relative-output"),
        ("C:/cache/manifest.json", "D:/xunce/inputs/mid_dual/scenario-sources"),
        ("D:/xunce/out/cache/manifest.json", "C:/inputs/scenario-sources"),
        (
            "D:/xunce/out/cache/manifest.json",
            "D:/xunce/inputs/mid_dual/../escaped",
        ),
    ):
        with pytest.raises(ValueError, match="absolute D"):
            module.validate_execution_paths(coverage_path, output_root)


def test_full_1064_builder_materializes_exact_364_and_feeds_task3(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module()
    freeze = _load_freeze_module()

    @dataclass(frozen=True)
    class Record:
        scenario_id: str
        split: str
        parent_roi: str
        density_profile: str
        ordinal: int

    records: list[Record] = []
    ordinal = 0
    for split, count in {
        "train": 700,
        "validation": 150,
        "test": 150,
        "unseen": 64,
    }.items():
        for index in range(count):
            records.append(
                Record(
                    scenario_id=f"{split}/record-{index:03d}",
                    split=split,
                    parent_roi=f"roi-{index % 7}",
                    density_profile=("low", "medium", "high")[index % 3],
                    ordinal=ordinal,
                )
            )
            ordinal += 1

    catalog_payload = {
        "schema_version": "fake-standard-catalog/v1",
        "sources": {"fixture": {"sha256": "4" * 64}},
        "records": [asdict(record) for record in records],
    }
    catalog_bytes = (
        json.dumps(
            catalog_payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")

    class Catalog:
        sha256 = hashlib.sha256(catalog_bytes).hexdigest()

        def __init__(self):
            self.records = tuple(records)

        def to_dict(self):
            return {"catalog_sha256": self.sha256, **catalog_payload}

    catalog = Catalog()

    mask = np.asarray([[True, False], [True, True]], dtype=np.bool_)
    mask_bytes_sha = hashlib.sha256(np.ascontiguousarray(mask).tobytes()).hexdigest()

    class Factory:
        def __init__(self, catalog_value):
            assert catalog_value is catalog

        def build(self, record):
            scenario_id = f"{record.scenario_id}/standard-proxy/v1"
            scenario_hash = hashlib.sha256(scenario_id.encode("utf-8")).hexdigest()
            geometry = SimpleNamespace(width=2, height=2, resolution_m=0.5)
            return SimpleNamespace(
                ordinal=record.ordinal,
                scenario_id=scenario_id,
                scenario_hash=scenario_hash,
                start_pose=SimpleNamespace(
                    cell=SimpleNamespace(x=record.ordinal % 2, y=(record.ordinal // 2) % 2),
                    theta=float((record.ordinal % 4) * np.pi / 2.0),
                ),
                truth=SimpleNamespace(
                    geometry=geometry,
                    slope_deg=np.asarray([[1.0, 2.0], [3.0, 4.0]]),
                    hard_obstacle=np.asarray([[False, False], [False, True]]),
                ),
                proxy_catalog=SimpleNamespace(sha256="5" * 64),
                proxy_layer_hashes={"fixture": "6" * 64},
            )

    class Coverage:
        catalog_sha256 = catalog.sha256
        split_counts = {"train": 700, "validation": 150, "test": 150, "unseen": 64}

        @staticmethod
        def scenario_audit(scenario_id):
            scenario_hash = hashlib.sha256(scenario_id.encode("utf-8")).hexdigest()
            split = scenario_id.split("/", 1)[0]
            key_sha = hashlib.sha256(f"key:{scenario_id}".encode("utf-8")).hexdigest()
            return {
                "scenario_id": scenario_id,
                "scenario_hash": scenario_hash,
                "split": split,
                "entry_path": f"D:/xunce/cache/{key_sha}.npz",
                "entry_sha256": "7" * 64,
                "entry_size_bytes": 1,
                "key_sha256": key_sha,
                "key": {"scenario_id": scenario_id},
                "coverable_mask_sha256": mask_bytes_sha,
                "coverable_cell_count": 3,
                "geometry": {"width": 2, "height": 2},
                "algorithm_id": module.DENOMINATOR_ALGORITHM,
                "exact": True,
            }

        @staticmethod
        def load_masks(bundle, **settings):
            assert settings == {
                "sensor_range_m": 20.0,
                "min_clearance_m": 0.5215874761,
                "max_slope_deg": 30.0,
                "traversability_threshold": 0.5,
            }
            return SimpleNamespace(coverable_mask=mask, coverable_cell_count=3)

    from lunar_exploration_ppo.env import coverage_cache, scenario_catalog, standard_training

    monkeypatch.setattr(standard_training, "build_standard_catalog", lambda: catalog)
    monkeypatch.setattr(scenario_catalog, "StandardScenarioFactory", Factory)
    monkeypatch.setattr(
        coverage_cache.Stage6CoverageManifest,
        "load",
        staticmethod(lambda *_args, **_kwargs: Coverage),
    )

    def reset_probe(*, bundle, masks, safety_contract, config_sha256):
        assert masks.coverable_cell_count == 3
        assert safety_contract.max_traversable_slope_deg == 30.0
        assert len(config_sha256) == 64
        return (
            {
                "scenario_id": bundle.scenario_id,
                "scenario_hash": bundle.scenario_hash,
                "start_pose": {"x": 0, "y": 0, "theta": 0.0},
                "start_pose_bin": module._pose_bin(bundle),
                "observed_mask_sha256": "8" * 64,
                "initial_observed_coverable_count": 1,
                "initial_observed_coverable_fraction": 1.0 / 3.0,
                "initial_valid_frontier_count": 2,
                "candidate_endpoints_sha256": "9" * 64,
                "max_reset_candidate_distance_m": float(bundle.ordinal % 17),
                "terminal_reason": "running",
            },
            object(),
        )

    monkeypatch.setattr(module, "_reset_probe", reset_probe)
    coverage_manifest_path = Path(
        "D:/xunce/out/mid_dual/fake-cache/coverage-cache-manifest.json"
    )
    coverage_manifest_sha = "a" * 64
    data, masks, completion, source_id = module.build_source_bundle_from_paths(
        config_path=CONFIG_PATH,
        coverage_manifest_path=coverage_manifest_path,
        coverage_manifest_sha256=coverage_manifest_sha,
        output_root=module.OUTPUT_BASE,
    )
    assert len(masks) == 364
    descriptors = [
        json.loads(line)
        for line in data["descriptors.jsonl"].decode("utf-8").splitlines()
    ]
    assert {
        split: sum(row["source_split"] == split for row in descriptors)
        for split in SOURCE_SPLIT_COUNTS
    } == SOURCE_SPLIT_COUNTS
    assert all(row["source_split"] != "train" for row in descriptors)
    source_root = module.OUTPUT_BASE / source_id
    assert (
        module.build_completion_manifest(data, masks, source_root=source_root)
        == completion
    )

    normalized_descriptors, pool_hashes = freeze._validated_descriptors(descriptors)
    reconstruction_rows, _ = freeze._validated_reconstruction_rows(
        data["reconstruction-index.jsonl"],
        normalized_descriptors,
    )
    source = json.loads(data["source-manifest.json"])
    path_payloads = {
        str(source[name]["path"]): data[relative]
        for name, relative in {
            "descriptor_catalog": "descriptors.jsonl",
            "standard_catalog": "standard-catalog.json",
            "standard_source": "standard-source.json",
            "static_truth_cache": "static-truth-index.jsonl",
            "reset_state": "reset-state-index.jsonl",
        }.items()
    }
    path_payloads[str(source["policy_blind_attestation"]["approval_artifact_path"])] = data[
        "policy-blind-approval.json"
    ]
    path_payloads[str(source["descriptor_generator"]["implementation_path"])] = (
        SCRIPT_PATH.read_bytes()
    )
    monkeypatch.setattr(
        freeze.artifact_io,
        "read_bytes",
        lambda path: path_payloads[str(path)],
    )
    validated_source, _ = freeze._validate_source_manifest(
        source_manifest_bytes=data["source-manifest.json"],
        descriptor_catalog_path=source["descriptor_catalog"]["path"],
        descriptor_catalog_bytes=data["descriptors.jsonl"],
        descriptors=normalized_descriptors,
        source_pool_hashes=pool_hashes,
        coverage_manifest_path=coverage_manifest_path,
        coverage_manifest_sha256=coverage_manifest_sha,
        coverage_catalog_sha256=catalog.sha256,
    )
    assert validated_source["schema_version"] == freeze.SOURCE_MANIFEST_SCHEMA

    first = normalized_descriptors[0]
    reconstruction = {
        row["scenario_id"]: row for row in reconstruction_rows
    }[first["scenario_id"]]
    with np.load(
        io.BytesIO(masks[f"masks/{first['scenario_hash']}.npz"]),
        allow_pickle=False,
    ) as archive:
        reconstructed_mask = np.array(archive["coverable_mask"], copy=True)
    proof = freeze.prove_denominator_identity(
        Coverage.scenario_audit(first["scenario_id"]),
        first,
        reconstruction,
        reconstructed_mask,
    )
    assert proof["semantic_alias_proven"] is True


def test_cli_without_execute_refuses_before_any_artifact_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module()

    def forbidden(*_args, **_kwargs):
        raise AssertionError("no-execute path touched an artifact")

    monkeypatch.setattr(module.artifact_io, "read_bytes", forbidden)
    monkeypatch.setattr(module.artifact_io, "make_dirs", forbidden)
    with pytest.raises(SystemExit, match="without --execute"):
        module.main([])
