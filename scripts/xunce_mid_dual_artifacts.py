"""Recoverable canonical artifact store for the reduced midterm dual-gate run."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
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
_LINEAGE_AUDIT = "lineage_audit.json"
_ENVIRONMENT_AUDIT = "environment_audit.json"
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


class MidDualRunStore:
    """Persist only accepted, hash-verified phase evidence into final artifacts."""

    def __init__(self, run_root: str | Path, config_sha256: str, attempts: list[dict[str, Any]]) -> None:
        self.run_root = Path(run_root)
        self.config_sha256 = config_sha256
        self._attempts = attempts
        self._preflight_blocked = False

    @classmethod
    def create_new(cls, run_root: str | Path, effective_config: Mapping[str, Any]) -> "MidDualRunStore":
        root = Path(run_root)
        if artifact_io.path_exists(root):
            raise FileExistsError(f"run root already exists: {root}")
        payload = dict(effective_config)
        supplied_sha256 = payload.pop("config_sha256", None)
        config_sha256 = _json_sha256(payload)
        if supplied_sha256 is not None and supplied_sha256 != config_sha256:
            raise ValueError("effective config_sha256 does not match the effective config")
        payload["config_sha256"] = config_sha256
        artifact_io.make_dirs(root)
        artifact_io.write_json(artifact_path(root, MID_DUAL_CONFIG), payload)
        artifact_io.write_jsonl(artifact_path(root, MID_DUAL_PHASE_STATE), [])
        artifact_io.write_jsonl(root / _ATTEMPT_INDEX, [])
        return cls(root, config_sha256, [])

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
        attempt_path = root / _ATTEMPT_INDEX
        attempts = artifact_io.read_jsonl(attempt_path) if artifact_io.path_is_file(attempt_path) else []
        store = cls(root, recorded_sha256, attempts)
        store._validate_accepted_prefix()
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
        raw_status = self._git_output("status", "--porcelain")
        status: dict[str, str] = {}
        for line in raw_status.splitlines():
            if len(line) >= 4:
                status[line[3:].replace("\\", "/")] = line[:2]
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
            "schema_version": "mid-dual-lineage-audit/v1",
            "root_commit": root_commit,
            "submodule_commit": submodule_commit,
            "branch": self._git_output("branch", "--show-current"),
            "required_sources": required_rows,
            "status_inventory": [{"path": path, "status": status} for path, status in sorted(statuses.items())],
        }
        artifact_io.write_json(self._canonical_path(_LINEAGE_AUDIT), audit)
        return audit

    def capture_environment(self, environment_probe: Callable[[], Mapping[str, Any]]) -> dict[str, Any]:
        try:
            probe = dict(environment_probe())
            missing = [field for field in _REQUIRED_ENVIRONMENT_FIELDS if probe.get(field) in (None, "")]
            if missing:
                raise ValueError("missing environment fields: " + ", ".join(missing))
            audit = {"schema_version": "mid-dual-environment-audit/v1", "status": "captured", "formal_evidence_eligible": True, "probe": probe}
        except Exception as exc:
            self._preflight_blocked = True
            audit = {
                "schema_version": "mid-dual-environment-audit/v1",
                "status": "blocked",
                "formal_evidence_eligible": False,
                "blocking_reason": f"environment_probe_failed:{type(exc).__name__}",
            }
        artifact_io.write_json(self._canonical_path(_ENVIRONMENT_AUDIT), audit)
        return audit

    def _final_artifact_paths(self, extra_audits: Mapping[str, Mapping[str, Any]]) -> list[str]:
        paths = ["config.json", _ATTEMPT_INDEX, "phase-state.jsonl"]
        for item in self._attempts:
            paths.extend((item["rows_path"], item["audit_path"]))
        for audit_name in (_LINEAGE_AUDIT, _ENVIRONMENT_AUDIT):
            if artifact_io.path_is_file(self._canonical_path(audit_name)):
                paths.append(audit_name)
        paths.extend(f"{name}_audit.json" for name in extra_audits)
        paths.extend(("results.jsonl", "summary.json", "routing.json", "report.md"))
        return paths

    def finalize(self, summary: Mapping[str, Any], routing: Mapping[str, Any], report: str, extra_audits: Mapping[str, Mapping[str, Any]]) -> None:
        manifest_path = artifact_path(self.run_root, MID_DUAL_MANIFEST)
        if artifact_io.path_is_file(manifest_path):
            self.verify_manifest(self.run_root)
        if any(artifact_io.path_exists(artifact_path(self.run_root, artifact)) for artifact in (MID_DUAL_RESULTS, MID_DUAL_SUMMARY, MID_DUAL_ROUTING, MID_DUAL_MANIFEST, MID_DUAL_REPORT)):
            raise FileExistsError("final canonical artifacts already exist")
        resumed = self.load_for_resume(self.run_root, self.config_sha256)
        self._attempts = resumed._attempts
        accepted = self._accepted_attempts()
        rows: list[dict[str, Any]] = []
        for attempt in accepted:
            rows.extend(artifact_io.read_jsonl(self.run_root / attempt["rows_path"]))
        blocked = self._preflight_blocked or summary.get("status") == "blocked" or routing.get("status") == "blocked"
        final_summary = dict(summary)
        final_routing = dict(routing)
        if blocked:
            rows = []
            final_summary["formal_evidence_eligible"] = False
            final_routing["formal_evidence_eligible"] = False
            final_routing["status"] = "blocked"
            final_routing.pop("midterm_reduced_passed", None)
        else:
            final_summary.setdefault("formal_evidence_eligible", True)
            final_routing.setdefault("formal_evidence_eligible", True)
        artifact_io.write_jsonl(artifact_path(self.run_root, MID_DUAL_RESULTS), rows)
        artifact_io.write_json(artifact_path(self.run_root, MID_DUAL_SUMMARY), final_summary)
        artifact_io.write_json(artifact_path(self.run_root, MID_DUAL_ROUTING), final_routing)
        artifact_io.write_text(artifact_path(self.run_root, MID_DUAL_REPORT), report)
        for name, payload in extra_audits.items():
            if not name.replace("_", "").isalnum():
                raise ValueError("extra audit name must be alphanumeric with underscores")
            artifact_io.write_json(self._canonical_path(f"{name}_audit.json"), dict(payload))
        artifact_paths = self._final_artifact_paths(extra_audits)
        manifest = {
            "schema_version": "mid-dual-manifest/v1",
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
        entries = manifest.get("artifacts")
        if not isinstance(entries, list):
            raise ValueError("manifest artifacts must be a list")
        paths = [entry.get("path") for entry in entries if isinstance(entry, dict)]
        if "manifest.json" in paths or len(paths) != len(set(paths)):
            raise ValueError("manifest must hash every artifact except itself exactly once")
        for entry in entries:
            if not isinstance(entry, dict) or not isinstance(entry.get("path"), str) or not isinstance(entry.get("sha256"), str):
                raise ValueError("invalid manifest entry")
            target = root / entry["path"]
            if not artifact_io.path_is_file(target) or _bytes_sha256(target) != entry["sha256"]:
                raise ValueError("manifest hash drift")
        return True
