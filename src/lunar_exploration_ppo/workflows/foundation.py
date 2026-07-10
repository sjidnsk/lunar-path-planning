from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Final

from pydantic import BaseModel, ConfigDict

from lunar_exploration_ppo.configs.schema import FoundationConfig, load_foundation_config, resolve_device
from lunar_exploration_ppo.utils.artifact_io import ArtifactStore


MACHINE_ARTIFACTS: Final = (
    "config.json",
    "summary.json",
    "routing.json",
    "report.md",
    "metrics.jsonl",
    "phase-state.jsonl",
)
FORBIDDEN_ARTIFACT_NAMES: Final = frozenset({"approval.json", "gate.json"})
_RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_WINDOWS_RESERVED_BASENAMES = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{index}" for index in range(1, 10)}
    | {f"LPT{index}" for index in range(1, 10)}
)


class FoundationPreflightError(RuntimeError):
    """Raised when a Foundation preflight run cannot start safely."""


class FoundationPreflightResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str
    stage_root: Path
    manifest: dict[str, object]


class FoundationReviewResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str
    stage_root: Path
    review_path: Path


def run_foundation_preflight(
    *,
    config_path: str | Path,
    run_id: str,
    base_output_root: str | Path | None = None,
) -> FoundationPreflightResult:
    validate_foundation_run_id(run_id)

    config = load_foundation_config(config_path)
    device = resolve_device(config.device)
    base_root = Path(base_output_root) if base_output_root is not None else Path(config.output_root)
    run_root = (base_root / run_id).expanduser().resolve()
    _prepare_unique_run_root(run_root)
    stage_root = run_root / "s0"
    stage_root.mkdir(exist_ok=False)
    store = ArtifactStore(stage_root)

    config_payload = config.model_dump(mode="json")
    config_payload["run_id"] = run_id
    config_bytes = ArtifactStore.canonical_json_bytes(config_payload)
    config_hash = hashlib.sha256(config_bytes).hexdigest()
    store.write_bytes("config.json", config_bytes)
    store.write_json(
        "summary.json",
        {
            "schema_version": "foundation_preflight_summary/v1",
            "goal_id": config.goal_id,
            "stage_id": config.stage_id,
            "run_id": run_id,
            "state": "machine_passed",
            "device": device,
            "config_hash": config_hash,
            "config_hash_schema": "sha256_file_bytes/v1",
        },
    )
    store.write_json(
        "routing.json",
        {
            "schema_version": "foundation_preflight_routing/v1",
            "run_id": run_id,
            "route": "awaiting_independent_review",
            "next_stage_entered": False,
            "human_approval_required": True,
        },
    )
    store.write_bytes(
        "report.md",
        (
            "# Foundation preflight\n\n"
            f"Run ID: `{run_id}`\n\n"
            "机器检查已通过，当前仅路由到独立审查。\n\n"
            "- 未生成审批文件。\n"
            "- 未进入 Stage 1。\n"
            "- 未连接 executor 或 canary。\n"
        ).encode("utf-8"),
    )
    store.append_jsonl(
        "metrics.jsonl",
        {"event": "foundation_contract_check", "passed": True, "device": device, "run_id": run_id},
    )
    store.append_jsonl(
        "phase-state.jsonl",
        {"state": "machine_passed", "route": "awaiting_independent_review", "run_id": run_id},
    )
    manifest = store.build_manifest(MACHINE_ARTIFACTS)
    store.write_json("manifest.json", manifest)
    return FoundationPreflightResult(run_id=run_id, stage_root=stage_root, manifest=manifest)


def record_foundation_independent_review(
    *,
    stage_root: str | Path,
    environment_manifest: str | Path,
    review_source_hash: str,
    review_package_hash: str,
    spec_verdict: str,
    quality_verdict: str,
    critical_count: int,
    important_count: int,
    minor_count: int,
) -> FoundationReviewResult:
    if spec_verdict != "approved" or quality_verdict != "approved":
        raise FoundationPreflightError("independent review verdicts must both be approved")
    if critical_count != 0 or important_count != 0:
        raise FoundationPreflightError("independent review cannot advance with Critical or Important findings")
    if minor_count < 0:
        raise FoundationPreflightError("independent review issue counts cannot be negative")
    for label, value in (
        ("review_source_hash", review_source_hash),
        ("review_package_hash", review_package_hash),
    ):
        if re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise FoundationPreflightError(f"independent review {label} must be a lowercase SHA-256")

    stage = Path(stage_root).expanduser().resolve()
    run_id, config, config_hash = _verify_machine_stage_for_review(stage)
    environment_path = Path(environment_manifest).expanduser().resolve()
    environment_bytes = _verify_environment_manifest_for_review(environment_path)
    store = ArtifactStore(stage)

    summary = _read_json_object(stage / "summary.json", "summary")
    summary["state"] = "awaiting_human_approval"
    routing = _read_json_object(stage / "routing.json", "routing")
    routing["route"] = "awaiting_human_approval"
    routing["human_approval_required"] = True
    store.write_json("summary.json", summary)
    store.write_json("routing.json", routing)
    review_event = {
        "event": "independent_review_approved",
        "run_id": run_id,
        "spec_verdict": spec_verdict,
        "quality_verdict": quality_verdict,
        "critical_count": critical_count,
        "important_count": important_count,
        "minor_count": minor_count,
    }
    store.append_jsonl("metrics.jsonl", review_event)
    store.append_jsonl(
        "phase-state.jsonl",
        {"state": "awaiting_independent_review", "route": "awaiting_independent_review", "run_id": run_id},
    )
    store.append_jsonl(
        "phase-state.jsonl",
        {"state": "awaiting_human_approval", "route": "awaiting_human_approval", "run_id": run_id},
    )
    report_path = stage / "report.md"
    report_text = report_path.read_text(encoding="utf-8")
    store.write_bytes(
        "report.md",
        (
            report_text.rstrip()
            + "\n\n## Independent review\n\n"
            + f"- Spec verdict: `{spec_verdict}`\n"
            + f"- Quality verdict: `{quality_verdict}`\n"
            + f"- Findings: Critical={critical_count}, Important={important_count}, Minor={minor_count}\n"
            + "- Route: `awaiting_human_approval`\n"
        ).encode("utf-8"),
    )
    manifest = store.build_manifest(MACHINE_ARTIFACTS)
    store.write_json("manifest.json", manifest)
    manifest_hash = hashlib.sha256((stage / "manifest.json").read_bytes()).hexdigest()
    repository_identity_hash = hashlib.sha256(
        ArtifactStore.canonical_json_bytes(config.repository_identity.model_dump(mode="json"))
    ).hexdigest()
    review = {
        "schema_version": "foundation_independent_review/v2",
        "goal_id": config.goal_id,
        "stage_id": config.stage_id,
        "run_id": run_id,
        "spec_verdict": spec_verdict,
        "quality_verdict": quality_verdict,
        "issue_counts": {
            "critical": critical_count,
            "important": important_count,
            "minor": minor_count,
        },
        "review_source_hash": review_source_hash,
        "review_package_hash": review_package_hash,
        "repository_identity_hash": repository_identity_hash,
        "config_hash": config_hash,
        "manifest_hash": manifest_hash,
        "environment_manifest_hash": hashlib.sha256(environment_bytes).hexdigest(),
    }
    review_path = store.write_json("review.json", review)
    return FoundationReviewResult(run_id=run_id, stage_root=stage, review_path=review_path)


def validate_foundation_run_id(run_id: str) -> None:
    basename = run_id.split(".", 1)[0].upper()
    if (
        _RUN_ID_PATTERN.fullmatch(run_id) is None
        or run_id.endswith((".", " "))
        or basename in _WINDOWS_RESERVED_BASENAMES
    ):
        raise FoundationPreflightError(
            "run_id must be a Windows-safe identifier and not a reserved device basename"
        )


def _prepare_unique_run_root(run_root: Path) -> None:
    if run_root.exists():
        entries = tuple(run_root.iterdir())
        forbidden = sorted(path.name for path in entries if path.name in FORBIDDEN_ARTIFACT_NAMES)
        if forbidden:
            raise FoundationPreflightError(f"forbidden approval artifacts already exist: {', '.join(forbidden)}")
        if entries:
            raise FoundationPreflightError(f"Foundation run root is non-empty: {run_root}")
        return
    run_root.mkdir(parents=True, exist_ok=False)


def _verify_machine_stage_for_review(stage: Path) -> tuple[str, FoundationConfig, str]:
    if not stage.is_dir() or stage.name != "s0":
        raise FoundationPreflightError("independent review requires a canonical machine stage root")
    run_id = stage.parent.name
    validate_foundation_run_id(run_id)
    if (stage / "review.json").exists():
        raise FoundationPreflightError("independent review was already recorded")
    if (stage / "approval.json").exists() or (stage / "gate.json").exists():
        raise FoundationPreflightError("independent review found forbidden approval artifacts")
    summary = _read_json_object(stage / "summary.json", "summary")
    routing = _read_json_object(stage / "routing.json", "routing")
    if summary.get("state") != "machine_passed" or routing.get("route") != "awaiting_independent_review":
        raise FoundationPreflightError("independent review requires machine_passed stage state")
    if summary.get("run_id") != run_id or routing.get("run_id") != run_id:
        raise FoundationPreflightError("independent review run_id declarations mismatch")
    config_path = stage / "config.json"
    config_bytes = config_path.read_bytes()
    config_hash = hashlib.sha256(config_bytes).hexdigest()
    if summary.get("config_hash") != config_hash:
        raise FoundationPreflightError("independent review config hash mismatch")
    config = load_foundation_config(config_path)
    if config.run_id != run_id:
        raise FoundationPreflightError("independent review config run_id mismatch")
    manifest = _read_json_object(stage / "manifest.json", "manifest")
    paths = [entry.get("path") if isinstance(entry, dict) else None for entry in manifest.get("artifacts", [])]
    if len(paths) != 6 or len(set(paths)) != 6 or set(paths) != set(MACHINE_ARTIFACTS):
        raise FoundationPreflightError("independent review machine manifest is not exact")
    for entry in manifest["artifacts"]:
        artifact = (stage / entry["path"]).resolve()
        try:
            artifact.relative_to(stage)
        except ValueError as exc:
            raise FoundationPreflightError("independent review artifact escapes stage root") from exc
        if not artifact.is_file():
            raise FoundationPreflightError(f"independent review artifact is missing: {entry['path']}")
        payload = artifact.read_bytes()
        if entry.get("size_bytes") != len(payload) or entry.get("sha256") != hashlib.sha256(payload).hexdigest():
            raise FoundationPreflightError(f"independent review artifact hash mismatch: {entry['path']}")
    phases = [json.loads(line) for line in (stage / "phase-state.jsonl").read_text(encoding="utf-8").splitlines()]
    if [event.get("state") for event in phases] != ["machine_passed"]:
        raise FoundationPreflightError("independent review phase-state is not fresh machine state")
    return run_id, config, config_hash


def _verify_environment_manifest_for_review(path: Path) -> bytes:
    if not path.is_file():
        raise FoundationPreflightError("independent review environment manifest is missing")
    payload = path.read_bytes()
    manifest = _read_json_object(path, "environment manifest")
    if manifest.get("schema_version") != "ppo_foundation_environment_lock_manifest/v1":
        raise FoundationPreflightError("independent review environment manifest schema is invalid")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise FoundationPreflightError("independent review environment manifest files are missing")
    manifest_root = path.parent.resolve()
    for entry in files:
        if not isinstance(entry, dict) or not isinstance(entry.get("file"), str):
            raise FoundationPreflightError("independent review environment manifest entry is invalid")
        relative = Path(entry["file"])
        if relative.is_absolute() or ".." in relative.parts:
            raise FoundationPreflightError("independent review environment manifest path is unsafe")
        source = (manifest_root / relative).resolve()
        try:
            source.relative_to(manifest_root)
        except ValueError as exc:
            raise FoundationPreflightError("independent review source escapes environment manifest root") from exc
        if not source.is_file():
            raise FoundationPreflightError("independent review environment manifest file is missing")
        source_bytes = source.read_bytes()
        if entry.get("size_bytes") != len(source_bytes) or entry.get("sha256") != hashlib.sha256(source_bytes).hexdigest():
            raise FoundationPreflightError("independent review environment manifest hash mismatch")
    return payload


def _read_json_object(path: Path, label: str) -> dict[str, object]:
    if not path.is_file():
        raise FoundationPreflightError(f"independent review {label} is missing")
    try:
        value = json.loads(path.read_bytes().decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FoundationPreflightError(f"independent review {label} is invalid JSON") from exc
    if not isinstance(value, dict):
        raise FoundationPreflightError(f"independent review {label} must be an object")
    return value
