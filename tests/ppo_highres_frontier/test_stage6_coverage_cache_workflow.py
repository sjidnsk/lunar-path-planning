"""Task B2 contracts for the resumable Stage 6 exact cache prewarm workflow."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import pickle
import subprocess
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from lunar_exploration_ppo.env.coverage import compute_coverage_masks
from lunar_exploration_ppo.env.coverage_cache import CoverageCacheError, Stage6CoverageManifest
from lunar_exploration_ppo.env.scenario import LowResolutionPrior, ScenarioBundle, TruthMap
from lunar_exploration_ppo.env.scenario_catalog import ScenarioCatalogRecord
from lunar_exploration_ppo.utils.artifact_io import ArtifactStore
from lunar_exploration_ppo.utils.geometry import CellXY, GridGeometry, PoseXYTheta


pytestmark = pytest.mark.filterwarnings("ignore:artifact path length .* exceeds 180 characters:RuntimeWarning")


@dataclass(frozen=True)
class _Catalog:
    records: tuple[ScenarioCatalogRecord, ...]
    sha256: str = "c" * 64

    def to_dict(self) -> dict[str, object]:
        return {"schema_version": "test-catalog/v1", "records": []}


def _record(index: int, split: str = "train") -> ScenarioCatalogRecord:
    return ScenarioCatalogRecord(
        scenario_id=f"record-{index:04d}", split=split, density_profile="low",
        parent_roi="roi", parent_window=SimpleNamespace(), source_window=SimpleNamespace(),
        source_window_transform=(), source_window_bounds=(), scenario_seed_hex="0",
        start_pose_seed_hex="0", terrain_seed_hex="0", proxy_seed_hex="0",
        seed_derivation_version="test", seed_derivation_input="test",
    )


def _catalog(*records: ScenarioCatalogRecord) -> _Catalog:
    return _Catalog(tuple(records))


def _full_catalog() -> _Catalog:
    return _catalog(
        *(_record(index, "train") for index in range(700)),
        *(_record(700 + index, "validation") for index in range(150)),
        *(_record(850 + index, "test") for index in range(150)),
        *(_record(1000 + index, "unseen") for index in range(64)),
    )


def _bundle(record: ScenarioCatalogRecord) -> ScenarioBundle:
    geometry = GridGeometry(8, 8, 1.0)
    shape = geometry.shape
    truth = TruthMap(
        geometry=geometry, height=np.zeros(shape), hard_obstacle=np.zeros(shape, dtype=bool),
        slope_deg=np.zeros(shape), traversability=np.ones(shape), provenance={},
    )
    scenario_id = f"{record.scenario_id}/standard-proxy/v1"
    scenario_hash = hashlib.sha256(scenario_id.encode("utf-8")).hexdigest()
    return ScenarioBundle(
        scenario_id=scenario_id, scenario_hash=scenario_hash, truth=truth,
        prior=LowResolutionPrior(channels=np.zeros((7, 32, 32), dtype=np.float32), resolution_m=2.0, value_prior_source="test/v1"),
        start_pose=PoseXYTheta(CellXY(3, 3), 0.0), proxy_catalog=None, proxy_layer_hashes={},
    )


def _worker_result(module, record: ScenarioCatalogRecord):
    bundle = _bundle(record)
    key = module.CoverageCacheKey.build(
        bundle, sensor_range_m=20.0, min_clearance_m=0.5215874761,
        max_slope_deg=30.0, traversability_threshold=0.5,
    )
    masks = compute_coverage_masks(
        bundle.truth, bundle.start_pose.cell, sensor_range_m=20.0,
        min_clearance_m=0.5215874761, max_slope_deg=30.0, traversability_threshold=0.5,
    )
    payload = module.serialize_coverage_entry(key, masks)
    return {
        "record_id": record.scenario_id, "scenario_id": bundle.scenario_id,
        "scenario_hash": bundle.scenario_hash, "split": record.split,
        "key_sha256": key.sha256, "key_payload": dict(key.payload), "entry_bytes": payload,
    }


def _expected_identity(module, record: ScenarioCatalogRecord):
    bundle = _bundle(record)
    key = module.CoverageCacheKey.build(
        bundle, sensor_range_m=20.0, min_clearance_m=0.5215874761,
        max_slope_deg=30.0, traversability_threshold=0.5,
    )
    return module._ExpectedRecordIdentity(
        record_id=record.scenario_id,
        scenario_id=bundle.scenario_id,
        scenario_hash=bundle.scenario_hash,
        split=record.split,
        key_sha256=key.sha256,
        key_payload=dict(key.payload),
        path=f"entries/{key.sha256[:2]}/{key.sha256}.npz",
    )


def _install_fake_identity(monkeypatch, module) -> None:
    monkeypatch.setattr(
        module,
        "_expected_record_identity",
        lambda catalog, record: _expected_identity(module, record),
    )


def _install_fake_workers(monkeypatch, module, *, mode: str = "normal") -> None:
    _install_fake_identity(monkeypatch, module)
    monkeypatch.setattr(
        module,
        "_validate_catalog",
        lambda catalog: tuple(sorted(catalog.records, key=lambda record: record.scenario_id)),
    )
    def fake(jobs, *, workers, latch=None):
        del workers, latch
        results = [_worker_result(module, job.record) for job in jobs]
        if mode == "duplicate":
            results.append(results[0])
        elif mode == "unknown":
            results[0] = {**results[0], "record_id": "unknown"}
        elif mode == "missing":
            results.pop()
        elif mode == "exception":
            raise RuntimeError("worker exploded")
        return results
    monkeypatch.setattr(module, "_spawn_worker_results", fake)


def _run(module, tmp_path: Path, catalog: _Catalog, **kwargs):
    return module.prewarm_stage6_coverage_cache(
        output_root=tmp_path / "review", cache_root=tmp_path / "cache-base",
        formal_run_id="formal-test", catalog=catalog, workers=1, **kwargs,
    )


def _create_directory_link_or_skip(link: Path, target: Path) -> None:
    try:
        link.symlink_to(target, target_is_directory=True)
        return
    except (NotImplementedError, OSError) as symlink_error:
        if os.name == "nt":
            completed = subprocess.run(
                ["cmd", "/d", "/c", "mklink", "/J", os.fspath(link), os.fspath(target)],
                capture_output=True,
                check=False,
            )
            if completed.returncode == 0:
                return
        pytest.skip(f"directory link creation is unavailable: {symlink_error}")


@pytest.mark.parametrize(
    "formal_run_id",
    [
        pytest.param("..", id="parent-component"),
        pytest.param(".", id="current-component"),
        pytest.param("CON", id="reserved-con"),
        pytest.param("con.txt", id="reserved-con-extension"),
        pytest.param("PRN", id="reserved-prn"),
        pytest.param("AUX.log", id="reserved-aux-extension"),
        pytest.param("NUL", id="reserved-nul"),
        pytest.param("\x00", id="embedded-nul"),
        pytest.param("COM1", id="reserved-com1"),
        pytest.param("COM9.txt", id="reserved-com9-extension"),
        pytest.param("LPT1", id="reserved-lpt1"),
        pytest.param("LPT9.log", id="reserved-lpt9-extension"),
        pytest.param("name.", id="trailing-dot"),
        pytest.param("name ", id="trailing-space"),
        pytest.param(" name", id="leading-space"),
        pytest.param("a/b", id="forward-slash"),
        pytest.param(r"a\b", id="backslash"),
        pytest.param("C:evil", id="drive-relative"),
        pytest.param("name:stream", id="ads"),
        pytest.param(r"C:\absolute", id="drive-absolute"),
        pytest.param(r"\\server\share", id="unc"),
        pytest.param("/absolute", id="rooted"),
        pytest.param("", id="empty"),
        pytest.param("   ", id="whitespace"),
        pytest.param("formal-测试", id="unicode-outside-allowlist"),
        pytest.param("formal+test", id="ascii-outside-allowlist"),
    ],
)
def test_invalid_formal_run_id_fails_before_store_construction_or_writes(
    monkeypatch,
    tmp_path: Path,
    formal_run_id: str,
) -> None:
    module = importlib.import_module("lunar_exploration_ppo.workflows.stage6_coverage_cache")
    constructed_roots: list[Path] = []

    class StoreConstructionForbidden:
        def __init__(self, root) -> None:
            constructed_roots.append(Path(root))
            raise AssertionError("ArtifactStore was constructed before formal run ID rejection")

    monkeypatch.setattr(module, "ArtifactStore", StoreConstructionForbidden)
    monkeypatch.setattr(module, "_validate_catalog", lambda value: tuple(value.records))
    output_root = tmp_path / "review"
    cache_base = tmp_path / "cache-base"

    with pytest.raises(module.CoverageCacheWorkflowError):
        module.prewarm_stage6_coverage_cache(
            output_root=output_root,
            cache_root=cache_base,
            formal_run_id=formal_run_id,
            catalog=_catalog(_record(0)),
            workers=1,
        )

    assert constructed_roots == []
    assert not output_root.exists()
    assert not cache_base.exists()
    assert tuple(tmp_path.iterdir()) == ()


def test_existing_cache_run_link_escaping_base_fails_before_store_construction_or_writes(
    monkeypatch,
    tmp_path: Path,
) -> None:
    module = importlib.import_module("lunar_exploration_ppo.workflows.stage6_coverage_cache")
    cache_base = tmp_path / "cache-base"
    outside = tmp_path / "outside"
    cache_base.mkdir()
    outside.mkdir()
    run_link = cache_base / "formal-test"
    _create_directory_link_or_skip(run_link, outside)
    assert run_link.resolve() == outside.resolve()

    constructed_roots: list[Path] = []

    class StoreConstructionForbidden:
        def __init__(self, root) -> None:
            constructed_roots.append(Path(root))
            raise AssertionError("ArtifactStore was constructed before cache target rejection")

    monkeypatch.setattr(module, "ArtifactStore", StoreConstructionForbidden)
    monkeypatch.setattr(module, "_validate_catalog", lambda value: tuple(value.records))
    output_root = tmp_path / "review"

    with pytest.raises(module.CoverageCacheWorkflowError):
        module.prewarm_stage6_coverage_cache(
            output_root=output_root,
            cache_root=cache_base,
            formal_run_id="formal-test",
            catalog=_catalog(_record(0)),
            workers=1,
        )

    assert constructed_roots == []
    assert not output_root.exists()
    assert tuple(cache_base.iterdir()) == (run_link,)
    assert tuple(outside.iterdir()) == ()


def test_fixed_formal_run_id_uses_exact_two_level_cache_target(monkeypatch, tmp_path: Path) -> None:
    module = importlib.import_module("lunar_exploration_ppo.workflows.stage6_coverage_cache")
    constructed_roots: list[Path] = []

    class ValidationComplete(Exception):
        pass

    class RecordingStore:
        def __init__(self, root) -> None:
            self.root = Path(root).expanduser().resolve()
            constructed_roots.append(self.root)

    monkeypatch.setattr(module, "ArtifactStore", RecordingStore)
    monkeypatch.setattr(module, "_validate_catalog", lambda value: tuple(value.records))
    monkeypatch.setattr(module, "_preflight_resources", lambda: (_ for _ in ()).throw(ValidationComplete))
    formal_run_id = "s6-standard-single-r1-20260718T220434Z"
    cache_base = tmp_path / "cache-base"
    output_root = tmp_path / "review"

    with pytest.raises(ValidationComplete):
        module.prewarm_stage6_coverage_cache(
            output_root=output_root,
            cache_root=cache_base,
            formal_run_id=formal_run_id,
            catalog=_catalog(_record(0)),
            workers=1,
        )

    expected = (cache_base / formal_run_id / "coverage-v1").resolve()
    assert constructed_roots == [output_root.resolve(), expected]
    assert tuple(tmp_path.iterdir()) == ()


def test_full_catalog_requires_exact_split_cardinality() -> None:
    module = importlib.import_module("lunar_exploration_ppo.workflows.stage6_coverage_cache")
    assert module._validate_catalog(_full_catalog()) == tuple(sorted(_full_catalog().records, key=lambda record: record.scenario_id))
    with pytest.raises(module.CoverageCacheWorkflowError, match="split count"):
        module._validate_catalog(_catalog(_record(0)))


def test_worker_payload_is_spawn_safe_and_has_no_parent_writer_state() -> None:
    module = importlib.import_module("lunar_exploration_ppo.workflows.stage6_coverage_cache")
    payload = module._WorkerPayload(catalog_payload={"schema_version": "test-catalog/v1"}, record_id="record-0000")
    assert pickle.loads(pickle.dumps(payload)) == payload
    assert set(payload.__dataclass_fields__) == {"catalog_payload", "record_id"}


def test_worker_rejects_nonfinite_truth_before_exact_mask_compute() -> None:
    module = importlib.import_module("lunar_exploration_ppo.workflows.stage6_coverage_cache")
    bundle = SimpleNamespace(truth=SimpleNamespace(
        height=np.array([[float("nan")]]), slope_deg=np.zeros((1, 1)), traversability=np.ones((1, 1)),
    ))
    with pytest.raises(module.CoverageCacheWorkflowError, match="non-finite"):
        module._assert_finite_truth(bundle)


def test_two_entries_publish_resume_and_manifest_is_deterministic(monkeypatch, tmp_path: Path) -> None:
    module = importlib.import_module("lunar_exploration_ppo.workflows.stage6_coverage_cache")
    catalog = _catalog(_record(1), _record(0))
    _install_fake_workers(monkeypatch, module)

    first = _run(module, tmp_path, catalog)
    first_bytes = (tmp_path / "review" / "coverage-cache-manifest.json").read_bytes()
    second = _run(module, tmp_path, catalog)

    assert first.published_count == 2
    assert second.skipped_count == 2
    assert (tmp_path / "review" / "report.md").is_file()
    assert first.manifest_sha256 == second.manifest_sha256
    assert first_bytes == (tmp_path / "review" / "coverage-cache-manifest.json").read_bytes()
    manifest = json.loads(first_bytes)
    assert [entry["scenario_id"] for entry in manifest["entries"]] == sorted(entry["scenario_id"] for entry in manifest["entries"])


def test_interrupt_then_resume_keeps_parent_owned_append_only_progress(monkeypatch, tmp_path: Path) -> None:
    module = importlib.import_module("lunar_exploration_ppo.workflows.stage6_coverage_cache")
    catalog = _catalog(_record(0), _record(1))
    _install_fake_workers(monkeypatch, module)
    with pytest.raises(KeyboardInterrupt):
        _run(module, tmp_path, catalog, interrupt_after=1)
    progress = tmp_path / "review" / "progress.jsonl"
    before = progress.read_bytes()
    summary = _run(module, tmp_path, catalog)
    assert summary.completed_count == 2
    assert progress.read_bytes().startswith(before)


def test_stream_failure_persists_first_result_and_resume_schedules_only_remaining(monkeypatch, tmp_path: Path) -> None:
    module = importlib.import_module("lunar_exploration_ppo.workflows.stage6_coverage_cache")
    catalog = _catalog(_record(0), _record(1), _record(2))
    _install_fake_identity(monkeypatch, module)

    def first_then_fail(jobs, *, workers, latch=None):
        del workers, latch
        scheduled = tuple(jobs)

        def stream():
            yield _worker_result(module, scheduled[0].record)
            raise RuntimeError("stream exploded after first result")

        return stream()

    monkeypatch.setattr(module, "_validate_catalog", lambda value: tuple(value.records))
    monkeypatch.setattr(module, "_spawn_worker_results", first_then_fail)
    with pytest.raises(module.CoverageCacheWorkflowError, match="worker failure"):
        _run(module, tmp_path, catalog)

    progress_path = tmp_path / "review" / "progress.jsonl"
    rows = [json.loads(line) for line in progress_path.read_text(encoding="utf-8").splitlines()]
    assert [row["record_id"] for row in rows] == ["record-0000"]
    assert (tmp_path / "cache-base" / "formal-test" / "coverage-v1" / rows[0]["descriptor"]["path"]).is_file()

    resumed_jobs: list[str] = []

    def remaining_only(jobs, *, workers, latch=None):
        del workers, latch
        scheduled = tuple(jobs)
        resumed_jobs.extend(job.record.scenario_id for job in scheduled)
        return (_worker_result(module, job.record) for job in scheduled)

    monkeypatch.setattr(module, "_spawn_worker_results", remaining_only)
    summary = _run(module, tmp_path, catalog)
    assert resumed_jobs == ["record-0001", "record-0002"]
    assert summary.completed_count == 3
    assert summary.published_count == 2
    assert summary.skipped_count == 1


def test_future_wait_polls_resource_latch_and_hard_stop_cancels_bounded_pending(monkeypatch) -> None:
    module = importlib.import_module("lunar_exploration_ppo.workflows.stage6_coverage_cache")
    future = SimpleNamespace(cancel_calls=0)

    def result(*, timeout):
        assert timeout > 0
        raise FutureTimeoutError

    def cancel():
        future.cancel_calls += 1
        return True

    future.result = result
    future.cancel = cancel
    executor_state = SimpleNamespace(submitted=0, shutdown_calls=[])

    class FakeExecutor:
        def __init__(self, *, max_workers, mp_context):
            assert max_workers == 1
            assert mp_context.get_start_method() == "spawn"

        def submit(self, function, payload):
            assert function is module._worker_compute
            assert payload.record_id == "record-0000"
            executor_state.submitted += 1
            return future

        def shutdown(self, *, wait, cancel_futures):
            executor_state.shutdown_calls.append((wait, cancel_futures))

    class HardStopLatch:
        def __init__(self):
            self.boundaries = []

        def poll(self, boundary):
            self.boundaries.append(boundary)
            raise module.ResourceHardStopError("D free space below runtime gate")

    monkeypatch.setattr(module, "ProcessPoolExecutor", FakeExecutor)
    latch = HardStopLatch()
    jobs = (
        module._Job(record=_record(0), catalog_payload={}),
        module._Job(record=_record(1), catalog_payload={}),
    )
    with pytest.raises(module.ResourceHardStopError, match="D free space"):
        next(iter(module._spawn_worker_results(jobs, workers=1, latch=latch)))
    assert latch.boundaries == ["worker-wait:record-0000"]
    assert executor_state.submitted == 1
    assert future.cancel_calls == 1
    assert executor_state.shutdown_calls[-1][1] is True


def test_existing_corrupt_entry_fails_closed(monkeypatch, tmp_path: Path) -> None:
    module = importlib.import_module("lunar_exploration_ppo.workflows.stage6_coverage_cache")
    catalog = _catalog(_record(0))
    _install_fake_workers(monkeypatch, module)
    _run(module, tmp_path, catalog)
    entry = next((tmp_path / "cache-base" / "formal-test" / "coverage-v1" / "entries").rglob("*.npz"))
    entry.write_bytes(b"corrupt")
    with pytest.raises(module.CoverageCacheWorkflowError, match="existing cache entry"):
        _run(module, tmp_path, catalog)


def test_progress_descriptor_rejects_noncanonical_entry_path(monkeypatch, tmp_path: Path) -> None:
    module = importlib.import_module("lunar_exploration_ppo.workflows.stage6_coverage_cache")
    _install_fake_workers(monkeypatch, module)
    _run(module, tmp_path, _catalog(_record(0)))
    row = json.loads((tmp_path / "review" / "progress.jsonl").read_text(encoding="utf-8"))
    descriptor = dict(row["descriptor"])
    canonical = tmp_path / "cache-base" / "formal-test" / "coverage-v1" / descriptor["path"]
    alias = canonical.parent / "alias.npz"
    alias.write_bytes(canonical.read_bytes())
    descriptor["path"] = alias.relative_to(tmp_path / "cache-base" / "formal-test" / "coverage-v1").as_posix()
    with pytest.raises(module.CoverageCacheWorkflowError, match="canonical"):
        module._validate_saved_descriptor(
            ArtifactStore(tmp_path / "cache-base" / "formal-test" / "coverage-v1"),
            descriptor,
            expected_identity=_expected_identity(module, _record(0)),
        )


@pytest.mark.parametrize("drift", ["split", "scenario_hash"])
def test_saved_progress_descriptor_is_bound_to_current_catalog_record(monkeypatch, tmp_path: Path, drift: str) -> None:
    module = importlib.import_module("lunar_exploration_ppo.workflows.stage6_coverage_cache")
    catalog = _catalog(_record(0))
    _install_fake_workers(monkeypatch, module)
    _run(module, tmp_path, catalog)
    progress_path = tmp_path / "review" / "progress.jsonl"
    row = json.loads(progress_path.read_text(encoding="utf-8"))
    if drift == "split":
        row["descriptor"]["split"] = "validation"
    else:
        row["descriptor"]["scenario_hash"] = "f" * 64
    progress_path.write_bytes(
        (json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    )
    with pytest.raises(module.CoverageCacheWorkflowError, match="record identity"):
        _run(module, tmp_path, catalog)


@pytest.mark.parametrize("drift", ["scenario_id", "scenario_hash"])
def test_worker_result_is_bound_to_expected_record(monkeypatch, tmp_path: Path, drift: str) -> None:
    module = importlib.import_module("lunar_exploration_ppo.workflows.stage6_coverage_cache")
    catalog = _catalog(_record(0))
    _install_fake_identity(monkeypatch, module)
    monkeypatch.setattr(module, "_validate_catalog", lambda value: tuple(value.records))

    def drifted(jobs, *, workers, latch=None):
        del workers, latch
        result = _worker_result(module, tuple(jobs)[0].record)
        if drift == "scenario_id":
            result["scenario_id"] = "wrong/standard-proxy/v1"
        else:
            result["scenario_hash"] = "f" * 64
        return [result]

    monkeypatch.setattr(module, "_spawn_worker_results", drifted)
    with pytest.raises(module.CoverageCacheWorkflowError, match="record identity"):
        _run(module, tmp_path, catalog)


@pytest.mark.parametrize("mode", ["duplicate", "unknown", "missing", "exception"])
def test_bad_worker_result_fails_closed(monkeypatch, tmp_path: Path, mode: str) -> None:
    module = importlib.import_module("lunar_exploration_ppo.workflows.stage6_coverage_cache")
    _install_fake_workers(monkeypatch, module, mode=mode)
    with pytest.raises(module.CoverageCacheWorkflowError):
        _run(module, tmp_path, _catalog(_record(0), _record(1)))


def test_producer_invokes_strict_b1_manifest_loader(monkeypatch, tmp_path: Path) -> None:
    module = importlib.import_module("lunar_exploration_ppo.workflows.stage6_coverage_cache")
    _install_fake_workers(monkeypatch, module)

    class RejectingManifest:
        @classmethod
        def load(cls, path, *, expected_sha256):
            del cls, path, expected_sha256
            raise CoverageCacheError("strict loader sentinel")

    monkeypatch.setattr(module, "Stage6CoverageManifest", RejectingManifest, raising=False)
    with pytest.raises(module.CoverageCacheWorkflowError, match="strict validation"):
        _run(module, tmp_path, _catalog(_record(0)))


def test_produced_manifest_passes_b1_loader_and_identity_checks(monkeypatch, tmp_path: Path) -> None:
    module = importlib.import_module("lunar_exploration_ppo.workflows.stage6_coverage_cache")
    catalog = _catalog(_record(0), _record(1))
    _install_fake_workers(monkeypatch, module)
    summary = _run(module, tmp_path, catalog)
    manifest_path = tmp_path / "review" / "coverage-cache-manifest.json"
    loaded = Stage6CoverageManifest.load(manifest_path, expected_sha256=summary.manifest_sha256)
    assert loaded.catalog_sha256 == catalog.sha256
    assert dict(loaded.split_counts) == {"train": 2, "validation": 0, "test": 0, "unseen": 0}
    assert len(loaded.entries) == 2
    assert loaded.entry_set_sha256 == summary.entry_set_sha256


def test_existing_manifest_must_be_byte_identical(monkeypatch, tmp_path: Path) -> None:
    module = importlib.import_module("lunar_exploration_ppo.workflows.stage6_coverage_cache")
    catalog = _catalog(_record(0))
    _install_fake_workers(monkeypatch, module)
    _run(module, tmp_path, catalog)
    manifest_path = tmp_path / "review" / "coverage-cache-manifest.json"
    manifest_path.write_bytes(b"different manifest bytes\n")
    with pytest.raises(module.CoverageCacheWorkflowError, match="manifest.*drift"):
        _run(module, tmp_path, catalog)


def test_resource_preflight_and_runtime_latch_are_enforced(monkeypatch, tmp_path: Path) -> None:
    module = importlib.import_module("lunar_exploration_ppo.workflows.stage6_coverage_cache")
    _install_fake_workers(monkeypatch, module)
    monkeypatch.setattr(module, "_preflight_resources", lambda: (_ for _ in ()).throw(module.CoverageCacheWorkflowError("D free space below 100 GiB preflight")))
    with pytest.raises(module.CoverageCacheWorkflowError, match="100 GiB"):
        _run(module, tmp_path, _catalog(_record(0)))


def test_cli_exposes_required_roots_workers_and_dry_run(monkeypatch, tmp_path: Path) -> None:
    module = importlib.import_module("scripts.prewarm_ppo_stage6_coverage_cache")
    captured = {}
    monkeypatch.setattr(module, "prewarm_stage6_coverage_cache", lambda **kwargs: captured.update(kwargs) or SimpleNamespace(to_dict=lambda: {"ok": True}))
    assert module.main([
        "--output-root", str(tmp_path / "review"), "--cache-root", str(tmp_path / "cache"),
        "--formal-run-id", "formal-test", "--workers", "2", "--dry-run-scenarios", "2",
    ]) == 0
    assert captured["workers"] == 2
    assert captured["dry_run_scenarios"] == 2
    assert captured["formal_run_id"] == "formal-test"


def test_worker_rejects_coherent_cross_record_spoof_before_publish(monkeypatch, tmp_path: Path) -> None:
    module = importlib.import_module("lunar_exploration_ppo.workflows.stage6_coverage_cache")
    expected = _record(0)
    source = _record(1)
    _install_fake_identity(monkeypatch, module)
    monkeypatch.setattr(module, "_validate_catalog", lambda value: tuple(value.records))

    def spoofed(jobs, *, workers, latch=None):
        del workers, latch
        result = _worker_result(module, source)
        result.update(
            record_id=expected.scenario_id,
            scenario_id=f"{expected.scenario_id}/standard-proxy/v1",
            split=expected.split,
        )
        return [result]

    monkeypatch.setattr(module, "_spawn_worker_results", spoofed)
    with pytest.raises(module.CoverageCacheWorkflowError, match="record identity"):
        _run(module, tmp_path, _catalog(expected))


def test_resume_rejects_coherent_cross_record_descriptor_spoof(monkeypatch, tmp_path: Path) -> None:
    module = importlib.import_module("lunar_exploration_ppo.workflows.stage6_coverage_cache")
    source = _record(1)
    expected = _record(0)
    _install_fake_workers(monkeypatch, module)
    catalog = _catalog(expected, source)
    _run(module, tmp_path, catalog)

    progress_path = tmp_path / "review" / "progress.jsonl"
    rows = [json.loads(line) for line in progress_path.read_text(encoding="utf-8").splitlines()]
    source_row = next(row for row in rows if row["record_id"] == source.scenario_id)
    spoofed_row = next(row for row in rows if row["record_id"] == expected.scenario_id)
    spoofed_row["descriptor"] = dict(source_row["descriptor"])
    spoofed_row["record_id"] = expected.scenario_id
    descriptor = spoofed_row["descriptor"]
    descriptor.update(
        scenario_id=f"{expected.scenario_id}/standard-proxy/v1",
        split=expected.split,
    )
    progress_path.write_bytes(
        ("\n".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            for row in (spoofed_row,)
        ) + "\n").encode("utf-8")
    )
    monkeypatch.setattr(module, "_publish_or_validate_manifest", lambda *args, **kwargs: None)

    with pytest.raises(module.CoverageCacheWorkflowError, match="record identity"):
        _run(module, tmp_path, catalog)
