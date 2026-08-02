"""Resumable, parent-owned prewarm for Stage 6 exact coverable-mask entries."""

from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import os
import platform
import re
import sys
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from lunar_exploration_ppo.env.coverage import compute_coverage_masks
from lunar_exploration_ppo.env.coverage_cache import (
    CoverageCacheError,
    CoverageCacheKey,
    Stage6CoverageManifest,
    load_coverage_entry,
    serialize_coverage_entry,
)
from lunar_exploration_ppo.env.scenario_catalog import ScenarioCatalogRecord, StandardScenarioCatalog, StandardScenarioFactory
from lunar_exploration_ppo.env.standard_training import SPLIT_COUNTS, _catalog_from_payload, build_standard_catalog
from lunar_exploration_ppo.utils.artifact_io import ArtifactStore
from lunar_exploration_ppo.utils.durable_jsonl import DurableJsonl, DurableJsonlError
from lunar_exploration_ppo.utils.path_security import PathSecurityError, require_plain_path, secure_read_bytes
from lunar_exploration_ppo.utils.resources import (
    ProcessTreeRSSMonitor,
    ResourceHardStopError,
    ResourceHardStopLatch,
    capture_resource_snapshot,
    evaluate_resource_gates,
)


_MANIFEST_SCHEMA = "stage6_exact_coverable_cache_manifest/v1"
_PROGRESS_SCHEMA = "stage6_coverage_cache_progress/v1"
_CONFIG_SCHEMA = "stage6_coverage_cache_prewarm_config/v1"
_SUMMARY_SCHEMA = "stage6_coverage_cache_prewarm_summary/v1"
_WORKER_RESULT_POLL_SECONDS = 0.5
_FORMAL_RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_WINDOWS_RESERVED_BASENAMES = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{index}" for index in range(1, 10)}
    | {f"LPT{index}" for index in range(1, 10)}
)
_SAFETY = {
    "sensor_range_m": 20.0,
    "min_clearance_m": 0.5215874761,
    "max_slope_deg": 30.0,
    "traversability_threshold": 0.5,
}


class CoverageCacheWorkflowError(RuntimeError):
    """The prewarm workflow cannot prove an exact, resumable cache state."""


def _validated_cache_target(cache_root: str | Path, formal_run_id: str) -> Path:
    reserved_basename = formal_run_id.split(".", 1)[0].upper() if isinstance(formal_run_id, str) else ""
    if (
        not isinstance(formal_run_id, str)
        or formal_run_id != formal_run_id.strip()
        or _FORMAL_RUN_ID_PATTERN.fullmatch(formal_run_id) is None
        or formal_run_id.endswith((".", " "))
        or reserved_basename in _WINDOWS_RESERVED_BASENAMES
    ):
        raise CoverageCacheWorkflowError("formal run ID must be a Windows-safe plain path component")

    try:
        cache_base = require_plain_path(
            cache_root,
            allow_missing=True,
            leaf_kind="directory",
            label="coverage cache base",
        )
        cache_target = require_plain_path(
            cache_base / formal_run_id / "coverage-v1",
            base=cache_base,
            allow_missing=True,
            leaf_kind="directory",
            label="coverage cache target",
        )
        relative_target = cache_target.relative_to(cache_base)
    except (OSError, PathSecurityError, ValueError) as exc:
        raise CoverageCacheWorkflowError(
            "coverage cache target must be an exact plain two-level descendant of cache root"
        ) from exc
    if relative_target.parts != (formal_run_id, "coverage-v1"):
        raise CoverageCacheWorkflowError(
            "coverage cache target must be an exact plain two-level descendant of cache root"
        )
    return cache_target


@dataclass(frozen=True, slots=True)
class _WorkerPayload:
    catalog_payload: dict[str, object]
    record_id: str


@dataclass(frozen=True, slots=True)
class _Job:
    record: ScenarioCatalogRecord
    catalog_payload: dict[str, object]


@dataclass(frozen=True, slots=True)
class _ExpectedRecordIdentity:
    record_id: str
    scenario_id: str
    scenario_hash: str
    split: str
    key_sha256: str
    key_payload: dict[str, object]
    path: str


@dataclass(frozen=True, slots=True)
class CoverageCacheSummary:
    output_root: str
    cache_root: str
    formal_run_id: str
    catalog_sha256: str
    scheduled_count: int
    completed_count: int
    published_count: int
    skipped_count: int
    manifest_sha256: str
    manifest_size_bytes: int
    entry_set_sha256: str

    def to_dict(self) -> dict[str, object]:
        return {"schema_version": _SUMMARY_SCHEMA, **asdict(self)}


def prewarm_stage6_coverage_cache(
    *,
    output_root: str | Path,
    cache_root: str | Path,
    formal_run_id: str,
    workers: int = 16,
    dry_run_scenarios: int | None = None,
    catalog: StandardScenarioCatalog | None = None,
    interrupt_after: int | None = None,
) -> CoverageCacheSummary:
    """Prewarm one deterministic catalog; only this parent writes artifacts."""

    cache_target = _validated_cache_target(cache_root, formal_run_id)
    if type(workers) is not int or not 1 <= workers <= 16:
        raise CoverageCacheWorkflowError("workers must be an integer in [1, 16]")
    if dry_run_scenarios is not None and (type(dry_run_scenarios) is not int or dry_run_scenarios < 1):
        raise CoverageCacheWorkflowError("dry-run-scenarios must be a positive integer")
    if interrupt_after is not None and (type(interrupt_after) is not int or interrupt_after < 1):
        raise CoverageCacheWorkflowError("interrupt_after must be a positive integer")

    actual_catalog = build_standard_catalog() if catalog is None else catalog
    records = _validate_catalog(actual_catalog)
    selected = records if dry_run_scenarios is None else records[:dry_run_scenarios]
    if not selected:
        raise CoverageCacheWorkflowError("prewarm selected no scenarios")
    try:
        catalog_payload = actual_catalog.to_dict()
    except Exception as exc:
        raise CoverageCacheWorkflowError("catalog payload is not spawn-safe") from exc

    output_store = ArtifactStore(output_root)
    cache_store = ArtifactStore(cache_target)
    if cache_store.root != cache_target:
        raise CoverageCacheWorkflowError(
            "coverage cache target must be an exact plain two-level descendant of cache root"
        )
    _preflight_resources()
    _write_or_validate_config(output_store, cache_store.root, actual_catalog.sha256, formal_run_id, selected)
    progress = DurableJsonl(output_store.resolve("progress.jsonl"))
    completed = _read_completed_progress(progress, cache_store)
    expected_identity_by_id = {
        record.scenario_id: _expected_record_identity(actual_catalog, record)
        for record in selected
    }
    expected_record_ids = {record.scenario_id for record in selected}
    if not set(completed).issubset(expected_record_ids):
        raise CoverageCacheWorkflowError("progress contains a scenario outside this selected catalog")

    skipped = 0
    published = 0
    for record_id, descriptor in completed.items():
        _validate_saved_descriptor(
            cache_store,
            descriptor,
            expected_identity=expected_identity_by_id[record_id],
        )
        skipped += 1

    pending = tuple(record for record in selected if record.scenario_id not in completed)
    monitor = ProcessTreeRSSMonitor()
    latch: ResourceHardStopLatch | None = None
    result_stream: Iterable[Mapping[str, object]] | None = None
    try:
        monitor.start()
        latch = ResourceHardStopLatch(
            attempt_id=f"coverage-cache:{formal_run_id}",
            snapshot_provider=lambda: capture_resource_snapshot(peak_vram_bytes=0, process_tree_monitor=monitor),
        )
        result_stream = _spawn_worker_results(
            tuple(_Job(record=record, catalog_payload=catalog_payload) for record in pending),
            workers=workers,
            latch=latch,
        )
        for ordinal, (record, result) in enumerate(
            zip(pending, result_stream, strict=True),
            start=1,
        ):
            _poll_resources(latch, f"before-publish:{record.scenario_id}")
            validated_result = _validate_worker_result(
                result,
                expected_identity=expected_identity_by_id[record.scenario_id],
            )
            descriptor, was_published = _publish_or_validate_result(cache_store, validated_result)
            completed[record.scenario_id] = descriptor
            progress.append({"schema_version": _PROGRESS_SCHEMA, "record_id": record.scenario_id, "descriptor": descriptor})
            if was_published:
                published += 1
            else:
                skipped += 1
            if interrupt_after is not None and ordinal >= interrupt_after:
                raise KeyboardInterrupt("intentional prewarm interrupt for resume verification")
            _poll_resources(latch, f"after-publish:{record.scenario_id}")
    except ResourceHardStopError as exc:
        raise CoverageCacheWorkflowError(f"resource hard stop: {exc}") from exc
    except CoverageCacheWorkflowError:
        raise
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        raise CoverageCacheWorkflowError("coverage cache worker failure") from exc
    finally:
        close = getattr(result_stream, "close", None)
        if callable(close):
            close()
        monitor.stop()

    if set(completed) != expected_record_ids:
        raise CoverageCacheWorkflowError("prewarm completed with missing scenario results")
    entries = [completed[record.scenario_id] for record in selected]
    entries.sort(key=lambda item: str(item["scenario_id"]))
    manifest = _manifest_payload(cache_store.root, actual_catalog.sha256, entries)
    manifest_bytes = ArtifactStore.canonical_json_bytes(manifest)
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    _publish_or_validate_manifest(
        output_store,
        manifest_bytes=manifest_bytes,
        manifest_sha256=manifest_sha256,
        expected_catalog_sha256=actual_catalog.sha256,
        expected_split_counts=dict(manifest["split_counts"]),
        expected_entry_count=len(entries),
        expected_entry_set_sha256=str(manifest["entry_set_sha256"]),
        expected_cache_root=cache_store.root,
    )
    summary = CoverageCacheSummary(
        output_root=str(output_store.root), cache_root=str(cache_store.root), formal_run_id=formal_run_id,
        catalog_sha256=actual_catalog.sha256, scheduled_count=len(selected), completed_count=len(completed),
        published_count=published, skipped_count=skipped,
        manifest_sha256=manifest_sha256, manifest_size_bytes=len(manifest_bytes),
        entry_set_sha256=str(manifest["entry_set_sha256"]),
    )
    output_store.write_json("summary.json", summary.to_dict())
    output_store.write_bytes("report.md", _render_review_report(summary).encode("utf-8"))
    return summary


def _validate_catalog(catalog: StandardScenarioCatalog) -> tuple[ScenarioCatalogRecord, ...]:
    records = tuple(catalog.records)
    counts = {split: sum(record.split == split for record in records) for split in SPLIT_COUNTS}
    if counts != SPLIT_COUNTS or len(records) != sum(SPLIT_COUNTS.values()):
        raise CoverageCacheWorkflowError("catalog split count drifted from 700/150/150/64")
    if len({record.scenario_id for record in records}) != len(records):
        raise CoverageCacheWorkflowError("catalog has duplicate record IDs")
    return tuple(sorted(records, key=lambda record: record.scenario_id))


def _worker_compute(payload: _WorkerPayload) -> dict[str, object]:
    catalog = _catalog_from_payload(payload.catalog_payload)
    matches = tuple(record for record in catalog.records if record.scenario_id == payload.record_id)
    if len(matches) != 1:
        raise CoverageCacheWorkflowError("worker payload record is missing or ambiguous")
    record = matches[0]
    bundle = StandardScenarioFactory(catalog).build(record)
    _assert_finite_truth(bundle)
    key = CoverageCacheKey.build(bundle, **_SAFETY)
    masks = compute_coverage_masks(bundle.truth, bundle.start_pose.cell, **_SAFETY)
    return {
        "record_id": record.scenario_id,
        "scenario_id": bundle.scenario_id,
        "scenario_hash": bundle.scenario_hash,
        "split": record.split,
        "key_sha256": key.sha256,
        "key_payload": dict(key.payload),
        "entry_bytes": serialize_coverage_entry(key, masks),
    }


def _expected_record_identity(
    catalog: StandardScenarioCatalog,
    record: ScenarioCatalogRecord,
) -> _ExpectedRecordIdentity:
    bundle = StandardScenarioFactory(catalog).build(record)
    key = CoverageCacheKey.build(bundle, **_SAFETY)
    path = (Path("entries") / key.sha256[:2] / f"{key.sha256}.npz").as_posix()
    return _ExpectedRecordIdentity(
        record_id=record.scenario_id,
        scenario_id=bundle.scenario_id,
        scenario_hash=bundle.scenario_hash,
        split=record.split,
        key_sha256=key.sha256,
        key_payload=dict(key.payload),
        path=path,
    )


def _assert_finite_truth(bundle: object) -> None:
    truth = getattr(bundle, "truth", None)
    if truth is None or any(
        not np.isfinite(getattr(truth, field)).all()
        for field in ("height", "slope_deg", "traversability")
    ):
        raise CoverageCacheWorkflowError("worker scenario truth contains non-finite values")


def _spawn_worker_results(
    jobs: Sequence[_Job],
    *,
    workers: int,
    latch: ResourceHardStopLatch,
) -> Iterable[dict[str, object]]:
    if not jobs:
        return
    context = mp.get_context("spawn")
    executor = ProcessPoolExecutor(max_workers=workers, mp_context=context)
    futures: dict[int, object] = {}
    next_submit = 0
    completed_normally = False
    try:
        while next_submit < min(workers, len(jobs)):
            job = jobs[next_submit]
            payload = _WorkerPayload(catalog_payload=job.catalog_payload, record_id=job.record.scenario_id)
            futures[next_submit] = executor.submit(_worker_compute, payload)
            next_submit += 1

        for index, job in enumerate(jobs):
            future = futures[index]
            while True:
                try:
                    result = future.result(timeout=_WORKER_RESULT_POLL_SECONDS)
                    break
                except FutureTimeoutError:
                    _poll_resources(latch, f"worker-wait:{job.record.scenario_id}")
            del futures[index]
            if not isinstance(result, dict):
                raise CoverageCacheWorkflowError("worker result value contract drifted")
            yield result
            if next_submit < len(jobs):
                next_job = jobs[next_submit]
                payload = _WorkerPayload(
                    catalog_payload=next_job.catalog_payload,
                    record_id=next_job.record.scenario_id,
                )
                futures[next_submit] = executor.submit(_worker_compute, payload)
                next_submit += 1
        completed_normally = True
    finally:
        if completed_normally:
            executor.shutdown(wait=True, cancel_futures=False)
        else:
            for future in futures.values():
                cancel = getattr(future, "cancel", None)
                if callable(cancel):
                    cancel()
            executor.shutdown(wait=True, cancel_futures=True)


def _validate_worker_result(
    result: Mapping[str, object],
    *,
    expected_identity: _ExpectedRecordIdentity,
) -> Mapping[str, object]:
    required = {"record_id", "scenario_id", "scenario_hash", "split", "key_sha256", "key_payload", "entry_bytes"}
    if set(result) != required:
        raise CoverageCacheWorkflowError("worker result schema drifted")
    _validate_record_identity(
        result,
        expected_identity=expected_identity,
        label="worker result",
        require_record_id=True,
    )
    if not all(type(result[field]) is str for field in ("key_sha256",)) or not isinstance(result["entry_bytes"], bytes):
        raise CoverageCacheWorkflowError("worker result value contract drifted")
    key_payload = result["key_payload"]
    assert isinstance(key_payload, dict)
    try:
        key = CoverageCacheKey._from_payload(key_payload)
        if key.sha256 != result["key_sha256"] or key_payload != expected_identity.key_payload:
            raise CoverageCacheWorkflowError("worker result key SHA-256 drifted")
        load_coverage_entry(result["entry_bytes"], expected_key=key)
    except CoverageCacheWorkflowError:
        raise
    except (CoverageCacheError, TypeError, ValueError) as exc:
        raise CoverageCacheWorkflowError("worker result cache payload drifted") from exc
    return result


def _publish_or_validate_result(cache_store: ArtifactStore, result: Mapping[str, object]) -> tuple[dict[str, object], bool]:
    key = CoverageCacheKey._from_payload(result["key_payload"])  # validated at worker boundary
    relative = Path("entries") / key.sha256[:2] / f"{key.sha256}.npz"
    path = cache_store.resolve(relative)
    payload = result["entry_bytes"]
    assert isinstance(payload, bytes)
    published = False
    if os.path.lexists(path):
        try:
            existing = secure_read_bytes(path, base=cache_store.root, label="coverage cache entry").payload
            load_coverage_entry(existing, expected_key=key)
        except (OSError, PathSecurityError, CoverageCacheError) as exc:
            raise CoverageCacheWorkflowError("existing cache entry is corrupt") from exc
        payload = existing
    else:
        try:
            cache_store.write_bytes_exclusive(relative, payload)
            published = True
        except FileExistsError:
            try:
                existing = secure_read_bytes(path, base=cache_store.root, label="coverage cache entry").payload
                load_coverage_entry(existing, expected_key=key)
            except (OSError, PathSecurityError, CoverageCacheError) as exc:
                raise CoverageCacheWorkflowError("existing cache entry is corrupt") from exc
            payload = existing
    return ({
        "scenario_id": result["scenario_id"], "scenario_hash": result["scenario_hash"], "split": result["split"],
        "key_sha256": key.sha256, "path": relative.as_posix(),
        "sha256": hashlib.sha256(payload).hexdigest(), "size_bytes": len(payload),
        "key_payload": result["key_payload"],
    }, published)


def _validate_saved_descriptor(
    cache_store: ArtifactStore,
    descriptor: Mapping[str, object],
    *,
    expected_identity: _ExpectedRecordIdentity,
) -> None:
    required = {"scenario_id", "scenario_hash", "split", "key_sha256", "path", "sha256", "size_bytes", "key_payload"}
    if set(descriptor) != required:
        raise CoverageCacheWorkflowError("progress descriptor schema drifted")
    if not isinstance(descriptor["key_payload"], dict):
        raise CoverageCacheWorkflowError("progress descriptor key payload drifted")
    _validate_record_identity(
        descriptor,
        expected_identity=expected_identity,
        label="progress descriptor",
        require_record_id=False,
    )
    key = CoverageCacheKey._from_payload(descriptor["key_payload"])
    if (
        descriptor["key_sha256"] != key.sha256
        or descriptor["key_payload"] != expected_identity.key_payload
        or descriptor["key_sha256"] != expected_identity.key_sha256
    ):
        raise CoverageCacheWorkflowError("progress descriptor key SHA-256 drifted")
    if descriptor["path"] != expected_identity.path:
        raise CoverageCacheWorkflowError("progress descriptor entry path is not canonical")
    path = cache_store.resolve(str(descriptor["path"]))
    try:
        payload = secure_read_bytes(path, base=cache_store.root, label="coverage cache entry").payload
        load_coverage_entry(payload, expected_key=key)
    except (OSError, PathSecurityError, CoverageCacheError) as exc:
        raise CoverageCacheWorkflowError("existing cache entry is corrupt") from exc
    if len(payload) != descriptor["size_bytes"] or hashlib.sha256(payload).hexdigest() != descriptor["sha256"]:
        raise CoverageCacheWorkflowError("existing cache entry digest drifted")


def _validate_record_identity(
    value: Mapping[str, object],
    *,
    expected_identity: _ExpectedRecordIdentity,
    label: str,
    require_record_id: bool,
) -> None:
    key_payload = value.get("key_payload")
    if (
        (require_record_id and value.get("record_id") != expected_identity.record_id)
        or value.get("scenario_id") != expected_identity.scenario_id
        or value.get("scenario_hash") != expected_identity.scenario_hash
        or value.get("split") != expected_identity.split
        or not isinstance(key_payload, dict)
        or key_payload != expected_identity.key_payload
    ):
        raise CoverageCacheWorkflowError(f"{label} current record identity drifted")


def _read_completed_progress(progress: DurableJsonl, cache_store: ArtifactStore) -> dict[str, dict[str, object]]:
    try:
        snapshot = progress.recover_and_snapshot()
    except DurableJsonlError as exc:
        raise CoverageCacheWorkflowError("progress recovery failed closed") from exc
    completed: dict[str, dict[str, object]] = {}
    for raw in snapshot.splitlines():
        try:
            row = json.loads(raw)
        except ValueError as exc:
            raise CoverageCacheWorkflowError("progress JSON is invalid") from exc
        if not isinstance(row, dict) or set(row) != {"schema_version", "record_id", "descriptor"} or row["schema_version"] != _PROGRESS_SCHEMA or type(row["record_id"]) is not str or not isinstance(row["descriptor"], dict):
            raise CoverageCacheWorkflowError("progress schema drifted")
        record_id = row["record_id"]
        if record_id in completed:
            raise CoverageCacheWorkflowError("progress has duplicate completed scenario")
        descriptor = dict(row["descriptor"])
        if descriptor.get("scenario_id") != f"{record_id}/standard-proxy/v1":
            raise CoverageCacheWorkflowError("progress scenario identity drifted")
        completed[record_id] = descriptor
    return completed


def _manifest_payload(cache_root: Path, catalog_sha256: str, descriptors: Sequence[Mapping[str, object]]) -> dict[str, object]:
    entries = []
    for descriptor in descriptors:
        entries.append({field: descriptor[field] for field in ("scenario_id", "scenario_hash", "split", "key_sha256", "path", "sha256", "size_bytes")})
    split_counts = {split: sum(entry["split"] == split for entry in entries) for split in SPLIT_COUNTS}
    return {
        "schema_version": _MANIFEST_SCHEMA,
        "cache_root": str(cache_root), "catalog_sha256": catalog_sha256, "split_counts": split_counts,
        "generation_environment": {"producer": "stage6_coverage_cache_prewarm/v1", "python_version": platform.python_version(), "numpy_version": np.__version__, "platform": platform.platform()},
        "integrity": {"complete": True, "entry_count": len(entries), "valid_entry_count": len(entries), "missing_entry_count": 0, "duplicate_entry_count": 0, "corrupt_entry_count": 0},
        "entry_set_sha256": hashlib.sha256(ArtifactStore.canonical_json_bytes(entries)).hexdigest(), "entries": entries,
    }


def _write_or_validate_config(output_store: ArtifactStore, cache_root: Path, catalog_sha256: str, formal_run_id: str, selected: Sequence[ScenarioCatalogRecord]) -> None:
    payload = {"schema_version": _CONFIG_SCHEMA, "cache_root": str(cache_root), "catalog_sha256": catalog_sha256, "formal_run_id": formal_run_id, "selected_record_ids": [record.scenario_id for record in selected], "safety": _SAFETY}
    path = output_store.resolve("config.json")
    raw = ArtifactStore.canonical_json_bytes(payload)
    if os.path.lexists(path):
        try:
            existing = secure_read_bytes(path, base=output_store.root, label="prewarm config").payload
        except (OSError, PathSecurityError) as exc:
            raise CoverageCacheWorkflowError("prewarm config secure read failed") from exc
        if existing != raw:
            raise CoverageCacheWorkflowError("prewarm input drifted from existing output root")
        return
    try:
        output_store.write_bytes_exclusive("config.json", raw)
    except FileExistsError:
        try:
            existing = secure_read_bytes(path, base=output_store.root, label="prewarm config").payload
        except (OSError, PathSecurityError) as exc:
            raise CoverageCacheWorkflowError("prewarm config secure read failed") from exc
        if existing != raw:
            raise CoverageCacheWorkflowError("prewarm input drifted from existing output root")


def _publish_or_validate_manifest(
    output_store: ArtifactStore,
    *,
    manifest_bytes: bytes,
    manifest_sha256: str,
    expected_catalog_sha256: str,
    expected_split_counts: dict[str, int],
    expected_entry_count: int,
    expected_entry_set_sha256: str,
    expected_cache_root: Path,
) -> None:
    relative = "coverage-cache-manifest.json"
    path = output_store.resolve(relative)
    if os.path.lexists(path):
        try:
            existing = secure_read_bytes(path, base=output_store.root, label="coverage cache manifest").payload
        except (OSError, PathSecurityError) as exc:
            raise CoverageCacheWorkflowError("existing coverage cache manifest secure read failed") from exc
        if existing != manifest_bytes:
            raise CoverageCacheWorkflowError("existing coverage cache manifest bytes drifted")
    else:
        try:
            output_store.write_bytes_exclusive(relative, manifest_bytes)
        except FileExistsError:
            try:
                existing = secure_read_bytes(path, base=output_store.root, label="coverage cache manifest").payload
            except (OSError, PathSecurityError) as exc:
                raise CoverageCacheWorkflowError("existing coverage cache manifest secure read failed") from exc
            if existing != manifest_bytes:
                raise CoverageCacheWorkflowError("existing coverage cache manifest bytes drifted")
    try:
        loaded = Stage6CoverageManifest.load(path, expected_sha256=manifest_sha256)
    except CoverageCacheError as exc:
        raise CoverageCacheWorkflowError("produced coverage cache manifest failed strict validation") from exc
    if (
        loaded.catalog_sha256 != expected_catalog_sha256
        or dict(loaded.split_counts) != expected_split_counts
        or len(loaded.entries) != expected_entry_count
        or loaded.entry_set_sha256 != expected_entry_set_sha256
        or loaded.cache_root != expected_cache_root
        or loaded.integrity.get("entry_count") != expected_entry_count
        or loaded.integrity.get("valid_entry_count") != expected_entry_count
    ):
        raise CoverageCacheWorkflowError("produced coverage cache manifest identity drifted after strict validation")


def _preflight_resources() -> None:
    decision = evaluate_resource_gates(capture_resource_snapshot(peak_vram_bytes=0), preflight=True)
    if decision.hard_stops:
        raise CoverageCacheWorkflowError("; ".join(decision.hard_stops))


def _poll_resources(latch: ResourceHardStopLatch, boundary: str) -> None:
    latch.poll(boundary)


def _render_review_report(summary: CoverageCacheSummary) -> str:
    return (
        "# Stage 6 exact coverage-cache prewarm\n\n"
        f"- Formal run ID: `{summary.formal_run_id}`\n"
        f"- Scheduled/completed: {summary.scheduled_count}/{summary.completed_count}\n"
        f"- Published/skipped: {summary.published_count}/{summary.skipped_count}\n"
        f"- Manifest SHA-256: `{summary.manifest_sha256}`\n"
        f"- Entry-set SHA-256: `{summary.entry_set_sha256}`\n"
        "- Parent-owned progress and manifest completed without a resource hard stop.\n"
    )


__all__ = ["CoverageCacheSummary", "CoverageCacheWorkflowError", "prewarm_stage6_coverage_cache"]
