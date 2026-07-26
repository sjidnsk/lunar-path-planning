"""Recoverable canonical artifact store for the reduced midterm dual-gate run."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import subprocess
from typing import Any, Callable, Iterable, Mapping

import xunce_artifact_io as artifact_io
from xunce_artifact_paths import (
    MID_DUAL_CONFIG,
    MID_DUAL_MANIFEST,
    MID_DUAL_PHASE_STATE,
    MID_DUAL_REPORT,
    MID_DUAL_RESULTS,
    MID_DUAL_ROUTING,
    MID_DUAL_SUMMARY,
    artifact_path,
)


_ATTEMPT_INDEX = "phase-attempts.jsonl"
_STORE_INDEX = "store-index.json"
_LINEAGE_AUDIT = "lineage_audit.json"
_ENVIRONMENT_AUDIT = "environment_audit.json"
_MANIFEST_SCHEMA_VERSION = "mid-dual-manifest/v1"
_LINEAGE_AUDIT_SCHEMA_VERSION = "mid-dual-lineage-audit/v1"
_ENVIRONMENT_AUDIT_SCHEMA_VERSION = "mid-dual-environment-audit/v1"
_REQUIRED_ENVIRONMENT_FIELDS = (
    "windows_version",
    "cpu_model",
    "cpu_logical_count",
    "memory_bytes",
    "gpu",
    "python_executable",
    "python_version",
    "frozen_dependencies",
    "python_hash_seed",
    "thread_variables",
    "worker_start_method",
    "power_mode",
)


def _json_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _bytes_sha256(path: Path) -> str:
    return hashlib.sha256(artifact_io.read_bytes(path)).hexdigest()


def _phase_number(phase_id: str) -> int:
    if len(phase_id) != 3 or not phase_id.startswith("p") or not phase_id[1:].isdigit() or phase_id == "p00":
        raise ValueError("phase_id must use the pNN format starting at p01")
    return int(phase_id[1:])


def _attempt_number(attempt_id: str) -> int:
    if len(attempt_id) != 3 or not attempt_id.startswith("a") or not attempt_id[1:].isdigit() or attempt_id == "a00":
        raise ValueError("attempt_id must use the aNN format starting at a01")
    return int(attempt_id[1:])


def _required_phase_sequence(value: object) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError("required_phase_ids must be a non-empty list")
    phase_ids = tuple(value)
    if any(not isinstance(phase_id, str) for phase_id in phase_ids):
        raise ValueError("required_phase_ids must contain phase IDs")
    expected = tuple(f"p{index:02d}" for index in range(1, len(phase_ids) + 1))
    if phase_ids != expected:
        raise ValueError("required_phase_ids must be a contiguous p01 sequence")
    return phase_ids


def _manifest_relative_path(path: object) -> str:
    if not isinstance(path, str) or not path or "\\" in path:
        raise ValueError("manifest path must be a relative POSIX path")
    posix_path = PurePosixPath(path)
    windows_path = PureWindowsPath(path)
    if posix_path.is_absolute() or windows_path.is_absolute() or any(part in ("", ".", "..") for part in posix_path.parts):
        raise ValueError("manifest path must not be absolute or traverse")
    return path


def _sha256_digest(value: object) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError("manifest digest must be a lowercase SHA-256 hex digest")
    return value


def _valid_environment_probe(probe: object) -> bool:
    if not isinstance(probe, Mapping):
        return False
    if any(probe.get(field) in (None, "") for field in _REQUIRED_ENVIRONMENT_FIELDS):
        return False
    gpu = probe.get("gpu")
    return isinstance(gpu, Mapping) and all(isinstance(gpu.get(field), str) and gpu[field].strip() for field in ("model", "driver", "cuda"))


def _valid_environment_audit(audit: object) -> bool:
    return (
        isinstance(audit, Mapping)
        and audit.get("schema_version") == _ENVIRONMENT_AUDIT_SCHEMA_VERSION
        and audit.get("status") == "captured"
        and audit.get("formal_evidence_eligible") is True
        and _valid_environment_probe(audit.get("probe"))
    )


def _valid_lineage_audit(audit: object) -> bool:
    if not isinstance(audit, Mapping) or audit.get("schema_version") != _LINEAGE_AUDIT_SCHEMA_VERSION:
        return False
    if not all(isinstance(audit.get(field), str) and audit[field] for field in ("root_commit", "submodule_commit")):
        return False
    return isinstance(audit.get("required_sources"), list) and isinstance(audit.get("status_inventory"), list)


class MidDualRunStore:
    """Persist only accepted, hash-verified phase evidence into final artifacts."""

    def __init__(self, run_root: str | Path, config_sha256: str, required_phase_ids: tuple[str, ...], attempts: list[dict[str, Any]]) -> None:
        self.run_root = Path(run_root)
        self.config_sha256 = config_sha256
        self.required_phase_ids = required_phase_ids
        self._attempts = attempts
        self._preflight_blocked = False
        self._lineage_captured = False
        self._environment_captured = False

    @classmethod
    def create_new(cls, run_root: str | Path, effective_config: Mapping[str, Any]) -> "MidDualRunStore":
        root = Path(run_root)
        if artifact_io.path_exists(root):
            raise FileExistsError(f"run root already exists: {root}")
        payload = dict(effective_config)
        supplied_sha256 = payload.pop("config_sha256", None)
        required_phase_ids = _required_phase_sequence(payload.get("required_phase_ids"))
        payload["required_phase_ids"] = list(required_phase_ids)
        config_sha256 = _json_sha256(payload)
        if supplied_sha256 is not None and supplied_sha256 != config_sha256:
            raise ValueError("effective config_sha256 does not match the effective config")
        payload["config_sha256"] = config_sha256
        artifact_io.make_dirs(root)
        artifact_io.write_json(artifact_path(root, MID_DUAL_CONFIG), payload)
        artifact_io.write_jsonl(artifact_path(root, MID_DUAL_PHASE_STATE), [])
        artifact_io.write_jsonl(root / _ATTEMPT_INDEX, [])
        artifact_io.write_json(root / _STORE_INDEX, {"extra_audit_paths": []})
        return cls(root, config_sha256, required_phase_ids, [])

    @classmethod
    def load_for_resume(cls, run_root: str | Path, expected_config_sha256: str) -> "MidDualRunStore":
        root = Path(run_root)
        config_path = artifact_path(root, MID_DUAL_CONFIG)
        if not artifact_io.path_is_file(config_path):
            raise FileNotFoundError("missing_config.json")
        config = artifact_io.read_json(config_path)
        recorded_sha256 = config.pop("config_sha256", None)
        if not isinstance(recorded_sha256, str) or recorded_sha256 != _json_sha256(config):
            raise ValueError("config drift detected")
        if recorded_sha256 != expected_config_sha256:
            raise ValueError("expected config_sha256 does not match the run")
        required_phase_ids = _required_phase_sequence(config.get("required_phase_ids"))
        attempt_path = root / _ATTEMPT_INDEX
        attempts = artifact_io.read_jsonl(attempt_path) if artifact_io.path_is_file(attempt_path) else []
        store = cls(root, recorded_sha256, required_phase_ids, attempts)
        store._validate_accepted_prefix()
        store._restore_preflight_state()
        return store

    @property
    def accepted_phase_ids(self) -> tuple[str, ...]:
        return tuple(item["phase_id"] for item in self._accepted_attempts())

    def _canonical_path(self, name: str) -> Path:
        artifacts = {
            "config.json": MID_DUAL_CONFIG,
            "results.jsonl": MID_DUAL_RESULTS,
            "summary.json": MID_DUAL_SUMMARY,
            "routing.json": MID_DUAL_ROUTING,
            "manifest.json": MID_DUAL_MANIFEST,
            "phase-state.jsonl": MID_DUAL_PHASE_STATE,
            "report.md": MID_DUAL_REPORT,
        }
        if name in artifacts:
            return artifact_path(self.run_root, artifacts[name])
        return self.run_root / name

    def _attempt_paths(self, phase_id: str, attempt_id: str) -> tuple[Path, Path, str, str]:
        _phase_number(phase_id)
        _attempt_number(attempt_id)
        relative_root = Path("phases") / phase_id / attempt_id
        return (
            self.run_root / relative_root / "results.jsonl",
            self.run_root / relative_root / "audit.json",
            str(relative_root / "results.jsonl").replace("\\", "/"),
            str(relative_root / "audit.json").replace("\\", "/"),
        )

    def _accepted_attempts(self) -> list[dict[str, Any]]:
        return sorted((item for item in self._attempts if item.get("status") == "accepted"), key=lambda item: _phase_number(item["phase_id"]))

    def _write_attempt_index(self) -> None:
        artifact_io.write_jsonl(self.run_root / _ATTEMPT_INDEX, self._attempts)

    def _write_phase_state(self) -> None:
        state = [
            {
                "phase_id": item["phase_id"],
                "attempt_id": item["attempt_id"],
                "row_sha256": item["row_sha256"],
                "rows_path": item["rows_path"],
            }
            for item in self._accepted_attempts()
        ]
        artifact_io.write_jsonl(artifact_path(self.run_root, MID_DUAL_PHASE_STATE), state)

    def _validate_accepted_prefix(self) -> None:
        accepted = self._accepted_attempts()
        phase_ids = [item.get("phase_id") for item in accepted]
        if len(phase_ids) != len(set(phase_ids)):
            raise ValueError("duplicate accepted phase_id")
        expected = [f"p{index:02d}" for index in range(1, len(accepted) + 1)]
        if phase_ids != expected:
            raise ValueError("accepted phases must be a contiguous p01 prefix")
        phase_state = artifact_io.read_jsonl(artifact_path(self.run_root, MID_DUAL_PHASE_STATE))
        if len(phase_state) != len(accepted):
            raise ValueError("phase-state does not match accepted attempts")
        for state, attempt in zip(phase_state, accepted, strict=True):
            if state.get("phase_id") != attempt["phase_id"] or state.get("attempt_id") != attempt["attempt_id"]:
                raise ValueError("phase-state does not match accepted attempts")
            expected_rows_path = self._attempt_paths(attempt["phase_id"], attempt["attempt_id"])[2]
            if attempt.get("rows_path") != expected_rows_path:
                raise ValueError("phase-state accepted attempt has a noncanonical rows_path")
            try:
                state_rows_path = _manifest_relative_path(state.get("rows_path"))
            except ValueError as exc:
                raise ValueError("phase-state rows_path is unsafe") from exc
            if state_rows_path != expected_rows_path:
                raise ValueError("phase-state rows_path does not bind the accepted attempt")
            rows_path = self.run_root / attempt["rows_path"]
            if not artifact_io.path_is_file(rows_path):
                raise ValueError("accepted phase results are missing")
            if _bytes_sha256(rows_path) != attempt["row_sha256"] or state.get("row_sha256") != attempt["row_sha256"]:
                raise ValueError("accepted phase row hash drift")

    def write_phase_attempt(self, phase_id: str, rows: Iterable[Mapping[str, Any]], audit: Mapping[str, Any]) -> str:
        _phase_number(phase_id)
        materialized_rows = [dict(row) for row in rows]
        attempts_for_phase = [item for item in self._attempts if item.get("phase_id") == phase_id]
        attempt_id = f"a{len(attempts_for_phase) + 1:02d}"
        rows_path, audit_path, rows_relative, audit_relative = self._attempt_paths(phase_id, attempt_id)
        if artifact_io.path_exists(rows_path) or artifact_io.path_exists(audit_path):
            raise FileExistsError("phase attempt path already exists")
        artifact_io.write_jsonl(rows_path, materialized_rows)
        artifact_io.write_json(audit_path, dict(audit))
        self._attempts.append(
            {
                "phase_id": phase_id,
                "attempt_id": attempt_id,
                "status": "written",
                "row_sha256": _bytes_sha256(rows_path),
                "rows_path": rows_relative,
                "audit_path": audit_relative,
            }
        )
        self._write_attempt_index()
        return attempt_id

    def phase_attempt_row_sha256(self, phase_id: str, attempt_id: str) -> str:
        for item in self._attempts:
            if item.get("phase_id") == phase_id and item.get("attempt_id") == attempt_id:
                return str(item["row_sha256"])
        raise ValueError("unknown phase attempt")

    def accept_phase(self, phase_id: str, attempt_id: str, row_sha256: str) -> str:
        _phase_number(phase_id)
        _attempt_number(attempt_id)
        next_phase = f"p{len(self._accepted_attempts()) + 1:02d}"
        if phase_id != next_phase:
            raise ValueError("accepted phases must be contiguous")
        selected = next((item for item in self._attempts if item.get("phase_id") == phase_id and item.get("attempt_id") == attempt_id), None)
        if selected is None or selected.get("status") != "written":
            raise ValueError("phase attempt is not available for acceptance")
        rows_path = self.run_root / selected["rows_path"]
        if selected["row_sha256"] != row_sha256 or _bytes_sha256(rows_path) != row_sha256:
            raise ValueError("phase attempt row hash mismatch")
        selected["status"] = "accepted"
        self._write_attempt_index()
        self._write_phase_state()
        return row_sha256

    def _git_output(self, *args: str) -> str:
        result = subprocess.run(("git", *args), cwd=Path.cwd(), check=False, capture_output=True, text=True, encoding="utf-8")
        return result.stdout.strip() if result.returncode == 0 else ""

    def _status_inventory(self) -> tuple[str, dict[str, str]]:
        repository_root = self._git_output("rev-parse", "--show-toplevel")
        status: dict[str, str] = {}
        result = subprocess.run(
            ("git", "status", "--porcelain=v1", "-z"),
            cwd=Path.cwd(),
            check=False,
            capture_output=True,
        )
        if result.returncode != 0:
            return repository_root, status
        records = result.stdout.split(b"\0")
        index = 0
        while index < len(records):
            record = records[index]
            index += 1
            if not record:
                continue
            if len(record) < 4:
                continue
            xy = record[:2].decode("ascii")
            current_path = record[3:].decode("utf-8", errors="surrogateescape").replace("\\", "/")
            status[current_path] = xy
            if "R" in xy or "C" in xy:
                index += 1
        return repository_root, status

    def capture_lineage(self, required_source_paths: Iterable[str | Path], root_commit: str, submodule_commit: str) -> dict[str, Any]:
        repository_root, statuses = self._status_inventory()
        root = Path(repository_root) if repository_root else None
        required_rows: list[dict[str, Any]] = []
        snapshot_index = 0
        for source in required_source_paths:
            path = Path(source)
            if not artifact_io.path_is_file(path):
                raise FileNotFoundError(f"required lineage source is missing: {path}")
            external_source = root is None
            try:
                relative = os.path.relpath(path, root) if root is not None else str(path)
            except ValueError:
                relative = str(path)
                external_source = True
            relative = relative.replace("\\", "/")
            status = statuses.get(relative, "untracked" if external_source or relative.startswith("../") else "clean")
            row = {
                "original_relative_path": relative,
                "status": status,
                "size_bytes": artifact_io.file_size(path),
                "sha256": _bytes_sha256(path),
            }
            if status != "clean":
                snapshot_index += 1
                snapshot = Path("lineage") / f"s{snapshot_index:04d}.bin"
                artifact_io.copy_file(path, self.run_root / snapshot)
                row["snapshot_path"] = str(snapshot).replace("\\", "/")
            required_rows.append(row)
        audit = {
            "schema_version": _LINEAGE_AUDIT_SCHEMA_VERSION,
            "root_commit": root_commit,
            "submodule_commit": submodule_commit,
            "branch": self._git_output("branch", "--show-current"),
            "required_sources": required_rows,
            "status_inventory": [{"path": path, "status": status} for path, status in sorted(statuses.items())],
        }
        artifact_io.write_json(self._canonical_path(_LINEAGE_AUDIT), audit)
        self._lineage_captured = _valid_lineage_audit(audit)
        if not self._lineage_captured:
            self._preflight_blocked = True
        return audit

    def capture_environment(self, environment_probe: Callable[[], Mapping[str, Any]]) -> dict[str, Any]:
        try:
            probe = dict(environment_probe())
            if not _valid_environment_probe(probe):
                raise ValueError("environment probe is incomplete")
            audit = {"schema_version": _ENVIRONMENT_AUDIT_SCHEMA_VERSION, "status": "captured", "formal_evidence_eligible": True, "probe": probe}
            self._environment_captured = True
        except Exception as exc:
            self._preflight_blocked = True
            audit = {
                "schema_version": _ENVIRONMENT_AUDIT_SCHEMA_VERSION,
                "status": "blocked",
                "formal_evidence_eligible": False,
                "blocking_reason": f"environment_probe_failed:{type(exc).__name__}",
            }
        artifact_io.write_json(self._canonical_path(_ENVIRONMENT_AUDIT), audit)
        return audit

    def _restore_preflight_state(self) -> None:
        lineage_path = self._canonical_path(_LINEAGE_AUDIT)
        environment_path = self._canonical_path(_ENVIRONMENT_AUDIT)
        try:
            self._lineage_captured = artifact_io.path_is_file(lineage_path) and _valid_lineage_audit(artifact_io.read_json(lineage_path))
        except (OSError, ValueError):
            self._lineage_captured = False
        try:
            self._environment_captured = artifact_io.path_is_file(environment_path) and _valid_environment_audit(artifact_io.read_json(environment_path))
        except (OSError, ValueError):
            self._environment_captured = False
        if not self._lineage_captured or not self._environment_captured:
            self._preflight_blocked = True

    @staticmethod
    def _expected_artifact_paths(root: Path) -> set[str]:
        paths = {"config.json", _ATTEMPT_INDEX, _STORE_INDEX, "phase-state.jsonl", "results.jsonl", "summary.json", "routing.json", "report.md"}
        attempts_path = root / _ATTEMPT_INDEX
        if not artifact_io.path_is_file(attempts_path):
            raise ValueError("manifest store index is missing phase attempts")
        for item in artifact_io.read_jsonl(attempts_path):
            for field_name in ("rows_path", "audit_path"):
                paths.add(_manifest_relative_path(item.get(field_name)))
        lineage_path = root / _LINEAGE_AUDIT
        if artifact_io.path_is_file(lineage_path):
            paths.add(_LINEAGE_AUDIT)
            lineage_audit = artifact_io.read_json(lineage_path)
            rows = lineage_audit.get("required_sources")
            if not isinstance(rows, list):
                raise ValueError("manifest lineage audit is malformed")
            for row in rows:
                if not isinstance(row, dict):
                    raise ValueError("manifest lineage audit is malformed")
                snapshot_path = row.get("snapshot_path")
                if snapshot_path is not None:
                    paths.add(_manifest_relative_path(snapshot_path))
        if artifact_io.path_is_file(root / _ENVIRONMENT_AUDIT):
            paths.add(_ENVIRONMENT_AUDIT)
        store_index_path = root / _STORE_INDEX
        if not artifact_io.path_is_file(store_index_path):
            raise ValueError("manifest store index is missing")
        extra_paths = artifact_io.read_json(store_index_path).get("extra_audit_paths")
        if not isinstance(extra_paths, list):
            raise ValueError("manifest store index is malformed")
        for extra_path in extra_paths:
            paths.add(_manifest_relative_path(extra_path))
        return paths

    @staticmethod
    def _blocked_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
        blocked = dict(payload)
        blocked["status"] = "blocked"
        blocked["formal_evidence_eligible"] = False
        blocked["midterm_reduced_passed"] = False
        blocked["final_threshold_reduced_passed"] = False
        for key in tuple(blocked):
            if "pass" in key.lower():
                blocked[key] = False
        return blocked

    def finalize(self, summary: Mapping[str, Any], routing: Mapping[str, Any], report: str, extra_audits: Mapping[str, Mapping[str, Any]]) -> None:
        manifest_path = artifact_path(self.run_root, MID_DUAL_MANIFEST)
        if artifact_io.path_is_file(manifest_path):
            self.verify_manifest(self.run_root)
        if any(artifact_io.path_exists(artifact_path(self.run_root, artifact)) for artifact in (MID_DUAL_RESULTS, MID_DUAL_SUMMARY, MID_DUAL_ROUTING, MID_DUAL_MANIFEST, MID_DUAL_REPORT)):
            raise FileExistsError("final canonical artifacts already exist")
        resumed = self.load_for_resume(self.run_root, self.config_sha256)
        self._attempts = resumed._attempts
        self._preflight_blocked = self._preflight_blocked or resumed._preflight_blocked
        self._lineage_captured = resumed._lineage_captured
        self._environment_captured = resumed._environment_captured
        accepted = self._accepted_attempts()
        rows: list[dict[str, Any]] = []
        for attempt in accepted:
            rows.extend(artifact_io.read_jsonl(self.run_root / attempt["rows_path"]))
        blocked = self._preflight_blocked or not self._lineage_captured or not self._environment_captured or summary.get("status") == "blocked" or routing.get("status") == "blocked"
        if not blocked and self.accepted_phase_ids != self.required_phase_ids:
            raise ValueError("accepted phases do not match the required phase sequence")
        final_summary = dict(summary)
        final_routing = dict(routing)
        if blocked:
            rows = []
            final_summary = self._blocked_payload(final_summary)
            final_routing = self._blocked_payload(final_routing)
        else:
            final_summary["formal_evidence_eligible"] = True
            final_routing["formal_evidence_eligible"] = True
        artifact_io.write_jsonl(artifact_path(self.run_root, MID_DUAL_RESULTS), rows)
        artifact_io.write_json(artifact_path(self.run_root, MID_DUAL_SUMMARY), final_summary)
        artifact_io.write_json(artifact_path(self.run_root, MID_DUAL_ROUTING), final_routing)
        artifact_io.write_text(artifact_path(self.run_root, MID_DUAL_REPORT), report)
        extra_paths: list[str] = []
        for name, payload in extra_audits.items():
            if not name.replace("_", "").isalnum():
                raise ValueError("extra audit name must be alphanumeric with underscores")
            extra_path = f"{name}_audit.json"
            extra_paths.append(extra_path)
            artifact_io.write_json(self._canonical_path(extra_path), dict(payload))
        artifact_io.write_json(self._canonical_path(_STORE_INDEX), {"extra_audit_paths": extra_paths})
        artifact_paths = sorted(self._expected_artifact_paths(self.run_root))
        manifest = {
            "schema_version": _MANIFEST_SCHEMA_VERSION,
            "config_sha256": self.config_sha256,
            "formal_evidence_eligible": not blocked,
            "artifacts": [{"path": path, "sha256": _bytes_sha256(self._canonical_path(path))} for path in artifact_paths],
        }
        artifact_io.write_json(artifact_path(self.run_root, MID_DUAL_MANIFEST), manifest)

    @staticmethod
    def verify_manifest(run_root: str | Path) -> bool:
        root = Path(run_root)
        manifest_path = artifact_path(root, MID_DUAL_MANIFEST)
        if not artifact_io.path_is_file(manifest_path):
            raise FileNotFoundError("missing_manifest.json")
        manifest = artifact_io.read_json(manifest_path)
        if manifest.get("schema_version") != _MANIFEST_SCHEMA_VERSION:
            raise ValueError("manifest schema_version is unsupported")
        config = artifact_io.read_json(artifact_path(root, MID_DUAL_CONFIG))
        config_sha256 = config.pop("config_sha256", None)
        if not isinstance(config_sha256, str) or config_sha256 != _json_sha256(config) or manifest.get("config_sha256") != config_sha256:
            raise ValueError("manifest config_sha256 drift")
        resumed = MidDualRunStore.load_for_resume(root, config_sha256)
        summary = artifact_io.read_json(artifact_path(root, MID_DUAL_SUMMARY))
        routing = artifact_io.read_json(artifact_path(root, MID_DUAL_ROUTING))
        expected_eligible = (
            not resumed._preflight_blocked
            and resumed._lineage_captured
            and resumed._environment_captured
            and summary.get("status") != "blocked"
            and routing.get("status") != "blocked"
            and summary.get("formal_evidence_eligible") is True
            and routing.get("formal_evidence_eligible") is True
        )
        if type(manifest.get("formal_evidence_eligible")) is not bool or manifest["formal_evidence_eligible"] is not expected_eligible:
            raise ValueError("manifest formal_evidence_eligible drift")
        entries = manifest.get("artifacts")
        if not isinstance(entries, list):
            raise ValueError("manifest artifacts must be a list")
        expected_paths = MidDualRunStore._expected_artifact_paths(root)
        paths: list[str] = []
        for entry in entries:
            if not isinstance(entry, dict):
                raise ValueError("invalid manifest entry")
            path = _manifest_relative_path(entry.get("path"))
            digest = _sha256_digest(entry.get("sha256"))
            paths.append(path)
            target = root / path
            if not artifact_io.path_is_file(target) or _bytes_sha256(target) != digest:
                raise ValueError("manifest hash drift")
        if "manifest.json" in paths or len(paths) != len(set(paths)) or set(paths) != expected_paths:
            raise ValueError("manifest artifact set is incomplete or non-authoritative")
        return True
