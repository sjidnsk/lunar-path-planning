"""Verify and publish the bounded Stage 6 R3 terminal closure.

This command never trains, evaluates, publishes a checkpoint, writes approval or
gate artifacts, or enters Stage 7.  It binds the completed planning-child
training root to the completed R3 evaluation root and replays the ten final
traces through the production final-evaluation verifier.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

from lunar_exploration_ppo.configs.stage6 import SafetyContract
from lunar_exploration_ppo.ppo.standard_training import (
    CheckpointReceiptIndex,
    ValidationRecord,
    completed_transaction_keys_from_journal,
    select_seed_best,
    validate_checkpoint_replay_audit,
    validate_standard_training_validation_rows,
    verify_standard_final_evaluation_artifacts_from_bytes,
)
from lunar_exploration_ppo.utils.artifact_io import ArtifactStore
from lunar_exploration_ppo.utils.resource_lifecycle import (
    validate_resource_lifecycle_rows,
)
from lunar_exploration_ppo.workflows.stage6 import (
    FINAL_EVALUATION_METHODS,
    Stage6StateJournal,
)
from lunar_exploration_ppo.workflows.stage6_planning_child_source_repair import (
    validate_planning_child_source_repair_bytes,
)
from lunar_exploration_ppo.workflows.stage6_planning_warm_start import (
    build_planning_child_transactions,
    parse_planning_effective_config_bytes,
)


SCHEMA = "stage6_r3_machine_acceptance/v1"
FORMAL_RUN_ID = "s6-standard-single-r1-20260724T000124Z"
SEED = 20260716
EXPECTED_UPDATES = tuple(range(75, 101))
EXPECTED_VALIDATION_UPDATES = (80, 90, 100)
EXPECTED_SCREEN_UPDATES = (80, 100)
EXPECTED_FINAL_SPLITS = ("test", "unseen")
EXPECTED_FINAL_KEYS = tuple(
    f"{split}:{method}"
    for split in EXPECTED_FINAL_SPLITS
    for method in FINAL_EVALUATION_METHODS
)
DEFAULT_STAGE_ROOT = Path(
    "D:/xunce/out/ppo_frontier/"
    "s6-standard-single-r1-20260724T000124Z/s6"
)
DEFAULT_EVAL_ROOT = Path(
    "D:/xunce/out/ppo_frontier/s6-r3-eval-r1-20260731T144106Z"
)
DEFAULT_STDERR_LOG = Path(
    "D:/xunce/out/ppo_frontier-logs/"
    "s6-r3-eval-r1-20260731T144106Z.stderr.log"
)
RUNNER_BASENAMES = {
    "run_ppo_stage6_standard.py",
    "run_ppo_stage6_r3_evaluation_pipeline.py",
}


class Stage6R3ClosureError(RuntimeError):
    """Raised when the bounded Stage 6 R3 closure cannot be proven."""


def require_canonical_closure_paths(
    stage_root: Path,
    eval_root: Path,
    stderr_log: Path,
) -> None:
    """Reject substituted roots for the one formal Stage 6 closure."""

    actual = tuple(
        path.expanduser().resolve()
        for path in (stage_root, eval_root, stderr_log)
    )
    expected = tuple(
        path.expanduser().resolve()
        for path in (DEFAULT_STAGE_ROOT, DEFAULT_EVAL_ROOT, DEFAULT_STDERR_LOG)
    )
    if actual != expected:
        raise Stage6R3ClosureError("canonical input path binding drifted")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise Stage6R3ClosureError(f"{label} is not a mapping")
    return value


def _read_json(path: Path, label: str, *, canonical: bool = True) -> dict[str, Any]:
    try:
        payload = path.read_bytes()
        value = json.loads(payload.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Stage6R3ClosureError(f"{label} is unreadable") from exc
    if not isinstance(value, dict):
        raise Stage6R3ClosureError(f"{label} is not a JSON object")
    if canonical and ArtifactStore.canonical_json_bytes(value) != payload:
        raise Stage6R3ClosureError(f"{label} is not canonical JSON")
    return value


def _read_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise Stage6R3ClosureError(f"{label} is unreadable") from exc
    rows: list[dict[str, Any]] = []
    try:
        for line in payload.splitlines(keepends=True):
            row = json.loads(line.decode("utf-8"))
            canonical = (
                json.dumps(
                    row,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n"
            ).encode("utf-8")
            if not isinstance(row, dict) or canonical != line:
                raise ValueError("non-canonical row")
            rows.append(row)
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise Stage6R3ClosureError(f"{label} is not canonical JSONL") from exc
    return rows


def _file_identity(path: Path) -> dict[str, object]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise Stage6R3ClosureError(f"required file is missing: {resolved}")
    return {
        "path": resolved.as_posix(),
        "sha256": _sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def _capture_manifest(paths: Sequence[Path]) -> list[dict[str, object]]:
    unique = sorted(
        {path.expanduser().resolve() for path in paths},
        key=lambda path: path.as_posix(),
    )
    return [_file_identity(path) for path in unique]


def _root_files(root: Path) -> list[Path]:
    if not root.is_dir():
        raise Stage6R3ClosureError(f"required root is missing: {root}")
    return sorted(
        (path for path in root.rglob("*") if path.is_file()),
        key=lambda path: path.as_posix(),
    )


def capture_closure_input_manifest(
    stage_root: Path,
    eval_root: Path,
    stderr_log: Path,
    external_paths: Sequence[Path] = (),
) -> list[dict[str, object]]:
    """Capture both root membership and bytes for mutation detection."""

    return _capture_manifest(
        [
            *_root_files(stage_root),
            *_root_files(eval_root),
            stderr_log,
            *external_paths,
        ]
    )


def _verify_file_binding(binding: object, label: str) -> Path:
    row = _mapping(binding, label)
    path_value = row.get("path")
    if (
        not isinstance(path_value, str)
        or not _is_sha256(row.get("sha256"))
        or type(row.get("size_bytes")) is not int
        or int(row["size_bytes"]) < 0
    ):
        raise Stage6R3ClosureError(f"{label} binding drifted")
    path = Path(path_value).expanduser().resolve()
    identity = _file_identity(path)
    if (
        identity["sha256"] != row["sha256"]
        or identity["size_bytes"] != row["size_bytes"]
    ):
        raise Stage6R3ClosureError(f"{label} bytes drifted")
    return path


def verify_checkpoint_member_bindings(
    latest_checkpoint: Mapping[str, object],
    stage_root: Path,
) -> tuple[Path, Path, Path]:
    """Verify the frozen U85 bundle members against their exact paths and bytes."""

    members = _mapping(
        latest_checkpoint.get("members"),
        "planning-child checkpoint members",
    )
    names = ("checkpoint.pt", "manifest.json", "complete.json")
    if latest_checkpoint.get("update") != 85 or set(members) != set(names):
        raise Stage6R3ClosureError("planning-child checkpoint members drifted")
    expected_root = (
        stage_root.expanduser().resolve()
        / "checkpoints"
        / f"seed-{SEED}"
        / "update-00000085"
    )
    top_level_fields = {
        "checkpoint.pt": "checkpoint_sha256",
        "manifest.json": "manifest_sha256",
        "complete.json": "complete_marker_sha256",
    }
    verified: list[Path] = []
    for name in names:
        binding = _mapping(
            members.get(name),
            f"planning-child checkpoint member {name}",
        )
        path = _verify_file_binding(
            binding,
            f"planning-child checkpoint member {name}",
        )
        if (
            path != (expected_root / name).resolve()
            or latest_checkpoint.get(top_level_fields[name]) != binding.get("sha256")
        ):
            raise Stage6R3ClosureError(
                f"planning-child checkpoint member {name} binding drifted"
            )
        verified.append(path)
    return verified[0], verified[1], verified[2]


def _resource_snapshot_passed(value: object) -> bool:
    if not isinstance(value, Mapping):
        return False
    decision = value.get("decision")
    if not isinstance(decision, Mapping):
        return False
    hard_stops = decision.get("hard_stops")
    warnings = decision.get("warnings")
    if hard_stops != [] or not isinstance(warnings, list):
        return False
    return "passed" not in decision or decision.get("passed") is True


def _safety_metrics_passed(value: object) -> bool:
    if not isinstance(value, Mapping):
        return False
    for name in (
        "safety_violation_count",
        "invalid_action_count_mean",
        "planner_failure_count_mean",
    ):
        metric = value.get(name)
        if (
            isinstance(metric, bool)
            or not isinstance(metric, (int, float))
            or float(metric) != 0.0
        ):
            return False
    return True


def _runtime_source_identity(repo_root: Path) -> dict[str, object]:
    relative_paths = [Path("configs/ppo_highres_frontier_stage6_v1.json")]
    relative_paths.extend(
        sorted(
            path.relative_to(repo_root)
            for path in (repo_root / "src/lunar_exploration_ppo").rglob("*.py")
        )
    )
    relative_paths.append(Path("scripts/run_ppo_stage6_r3_evaluation_pipeline.py"))
    rows: list[dict[str, object]] = []
    for relative in relative_paths:
        path = repo_root / relative
        payload = path.read_bytes()
        rows.append(
            {
                "path": relative.as_posix(),
                "sha256": _sha256_bytes(payload),
                "size_bytes": len(payload),
            }
        )
    canonical = ArtifactStore.canonical_json_bytes(rows)
    return {
        "schema_version": "stage6_r3_runtime_source_identity/v1",
        "source_set_sha256": _sha256_bytes(canonical),
        "paths": rows,
    }


def _checkpoint_identity(stage_root: Path, update: int) -> dict[str, object]:
    root = stage_root / f"checkpoints/seed-{SEED}/update-{update:08d}"
    if not root.is_dir():
        raise Stage6R3ClosureError(f"checkpoint U{update} is missing")
    members = {path.name for path in root.iterdir() if path.is_file()}
    if members != {"checkpoint.pt", "manifest.json", "complete.json"}:
        raise Stage6R3ClosureError(f"checkpoint U{update} member set drifted")
    checkpoint = root / "checkpoint.pt"
    manifest_path = root / "manifest.json"
    complete_path = root / "complete.json"
    manifest = _read_json(manifest_path, f"checkpoint U{update} manifest")
    checkpoint_sha256 = _sha256_file(checkpoint)
    checkpoint_binding = _mapping(
        manifest.get("checkpoint"), f"checkpoint U{update} payload binding"
    )
    if (
        manifest.get("update_step") != update
        or checkpoint_binding.get("sha256") != checkpoint_sha256
        or not _is_sha256(manifest.get("policy_state_sha256"))
        or not _is_sha256(manifest.get("config_sha256"))
        or not _is_sha256(manifest.get("lineage_sha256"))
    ):
        raise Stage6R3ClosureError(f"checkpoint U{update} manifest drifted")
    return {
        "update": update,
        "root": root.resolve().as_posix(),
        "checkpoint_path": checkpoint.resolve().as_posix(),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_size_bytes": checkpoint.stat().st_size,
        "manifest_sha256": _sha256_file(manifest_path),
        "complete_sha256": _sha256_file(complete_path),
        "policy_state_sha256": manifest["policy_state_sha256"],
        "config_sha256": manifest["config_sha256"],
        "lineage_sha256": manifest["lineage_sha256"],
    }


def _git(repo_root: Path, *args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo_root), *args],
            text=True,
            encoding="utf-8",
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise Stage6R3ClosureError("git identity is unreadable") from exc


def _active_stage6_runners() -> list[dict[str, object]]:
    try:
        import psutil
    except ImportError:
        if os.name != "nt":
            raise Stage6R3ClosureError(
                "psutil is required for non-Windows runner closure"
            ) from None
        try:
            output = subprocess.check_output(
                [
                    "powershell.exe",
                    "-NoProfile",
                    "-Command",
                    "Get-CimInstance Win32_Process | "
                    "Select-Object ProcessId,CommandLine | "
                    "ConvertTo-Json -Compress",
                ],
                text=True,
                encoding="utf-8",
            ).strip()
            raw = json.loads(output) if output else []
        except (
            OSError,
            subprocess.CalledProcessError,
            UnicodeDecodeError,
            json.JSONDecodeError,
        ) as exc:
            raise Stage6R3ClosureError(
                "Windows process inventory is unreadable"
            ) from exc
        process_rows = raw if isinstance(raw, list) else [raw]
        rows: list[dict[str, object]] = []
        for process in process_rows:
            if not isinstance(process, Mapping):
                continue
            command_line = process.get("CommandLine")
            if not isinstance(command_line, str):
                continue
            matched = sorted(
                name for name in RUNNER_BASENAMES if name in command_line
            )
            if matched:
                rows.append(
                    {
                        "pid": int(process["ProcessId"]),
                        "scripts": matched,
                    }
                )
        return sorted(rows, key=lambda row: int(row["pid"]))
    rows: list[dict[str, object]] = []
    for process in psutil.process_iter(["pid", "cmdline"]):
        try:
            command = process.info.get("cmdline") or []
        except (psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess):
            continue
        matched = sorted(
            {
                Path(str(argument).replace("\\", "/")).name
                for argument in command
                if Path(str(argument).replace("\\", "/")).name
                in RUNNER_BASENAMES
            }
        )
        if matched:
            rows.append(
                {
                    "pid": int(process.info["pid"]),
                    "scripts": matched,
                }
            )
    return sorted(rows, key=lambda row: int(row["pid"]))


def build_standard_replay_artifacts(
    *,
    trace_bytes: bytes,
    trace_name: str,
    split: str,
    method: str,
    config_sha256: str,
    checkpoint_sha256: str,
    policy_state_sha256: str,
    result: Mapping[str, object],
) -> dict[str, object]:
    """Adapt one immutable R3 result to the production replay byte contract."""

    if (
        type(trace_bytes) is not bytes
        or not trace_bytes
        or not isinstance(trace_name, str)
        or not trace_name
        or not isinstance(result, Mapping)
    ):
        raise Stage6R3ClosureError("final replay input drifted")
    standard_result = dict(result)
    existing_count = standard_result.get("episode_count")
    if existing_count not in (None, 64):
        raise Stage6R3ClosureError("final replay episode count drifted")
    standard_result["episode_count"] = 64
    trace_binding = {
        "path": trace_name,
        "sha256": _sha256_bytes(trace_bytes),
        "size_bytes": len(trace_bytes),
    }
    summary_name = f"{Path(trace_name).stem}.completion.json"
    commit_name = f"{Path(trace_name).stem}.commit.json"
    completion = {
        "schema_version": "stage6_final_eval_completion/v1",
        "split": split,
        "method": method,
        "episode_count": 64,
        "trace": trace_binding,
        "config_sha256": config_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "policy_state_sha256": policy_state_sha256,
        "result": standard_result,
    }
    summary_bytes = ArtifactStore.canonical_json_bytes(completion)
    commit = {
        "schema_version": "stage6_final_eval_commit/v1",
        "split": split,
        "method": method,
        "attempt": 1,
        "config_sha256": config_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "policy_state_sha256": policy_state_sha256,
        "trace": trace_binding,
        "summary": {
            "path": summary_name,
            "sha256": _sha256_bytes(summary_bytes),
            "size_bytes": len(summary_bytes),
        },
    }
    return {
        "summary_name": summary_name,
        "summary_bytes": summary_bytes,
        "commit_name": commit_name,
        "commit_bytes": ArtifactStore.canonical_json_bytes(commit),
    }


def validate_final_summary_schedule(
    value: Mapping[str, object],
    *,
    expected_source_set_sha256: str,
    expected_config_sha256: str,
    expected_checkpoint_sha256: str,
    expected_policy_state_sha256: str,
    expected_update: int,
) -> tuple[str, ...]:
    """Validate the exact 2 splits x 5 methods R3 final schedule."""

    results = _mapping(value.get("results"), "R3 final results")
    global_best = _mapping(value.get("global_best"), "R3 embedded global best")
    record = _mapping(global_best.get("record"), "R3 embedded best record")
    if (
        value.get("schema_version") != "stage6_r3_final_summary/v1"
        or value.get("evaluation_count") != 10
        or value.get("episode_count") != 640
        or value.get("source_set_sha256") != expected_source_set_sha256
        or value.get("config_sha256") != expected_config_sha256
        or set(results) != set(EXPECTED_FINAL_KEYS)
        or record.get("update") != expected_update
        or global_best.get("checkpoint_sha256") != expected_checkpoint_sha256
        or global_best.get("policy_state_sha256") != expected_policy_state_sha256
        or global_best.get("source_set_sha256") != expected_source_set_sha256
    ):
        raise Stage6R3ClosureError("R3 final evaluation schedule drifted")
    for key in EXPECTED_FINAL_KEYS:
        split, method = key.split(":", maxsplit=1)
        row = _mapping(results[key], f"R3 final result {key}")
        resources = row.get("resources")
        result = row.get("result")
        metrics = result.get("metrics") if isinstance(result, Mapping) else None
        if (
            row.get("schema_version") != "stage6_r3_evaluation_result/v1"
            or row.get("split") != split
            or row.get("method") != method
            or row.get("episode_count") != 64
            or row.get("trace_line_count") != 64
            or row.get("update") != expected_update
            or row.get("checkpoint_sha256") != expected_checkpoint_sha256
            or row.get("policy_state_sha256") != expected_policy_state_sha256
            or row.get("config_sha256") != expected_config_sha256
            or row.get("source_set_sha256") != expected_source_set_sha256
            or not _is_sha256(row.get("trace_sha256"))
            or not isinstance(row.get("result"), Mapping)
        ):
            raise Stage6R3ClosureError("R3 final evaluation schedule drifted")
        if not _safety_metrics_passed(metrics):
            raise Stage6R3ClosureError("R3 final safety metrics drifted")
        if (
            not isinstance(resources, list)
            or not resources
            or any(not _resource_snapshot_passed(resource) for resource in resources)
        ):
            raise Stage6R3ClosureError("R3 final resource acceptance drifted")
    return EXPECTED_FINAL_KEYS


def _verify_planning_child_chain(
    stage_root: Path,
    receipts: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], list[Path]]:
    parent_path = stage_root / "planning-child-source-repair.json"
    continuation_path = (
        stage_root / "planning-child-source-repair-continuation.json"
    )
    parent_payload = parent_path.read_bytes()
    validated_parent = validate_planning_child_source_repair_bytes(
        parent_payload,
        stage_root=stage_root,
        artifact_path=parent_path,
    )
    continuation_payload = continuation_path.read_bytes()
    continuation = _read_json(
        continuation_path,
        "planning-child continuation",
    )
    parent_binding = _mapping(
        continuation.get("parent"), "planning-child continuation parent"
    )
    current = _mapping(
        continuation.get("current"), "planning-child continuation current"
    )
    accepted_anchor = _mapping(
        continuation.get("accepted_anchor"),
        "planning-child continuation accepted anchor",
    )
    latest_checkpoint = _mapping(
        accepted_anchor.get("latest_complete_checkpoint"),
        "planning-child continuation U85 checkpoint",
    )
    lineage_epoch = _mapping(
        continuation.get("lineage_epoch"),
        "planning-child continuation lineage epoch",
    )
    if (
        continuation.get("schema_version")
        != "stage6_planning_child_source_repair_continuation/v2"
        or continuation.get("formal_run_id") != FORMAL_RUN_ID
        or continuation.get("seed") != SEED
        or parent_binding.get("sha256") != _sha256_bytes(parent_payload)
        or parent_binding.get("size_bytes") != len(parent_payload)
        or parent_binding.get("canonical_sha256")
        != validated_parent.canonical_sha256
        or accepted_anchor.get("last_accepted_update") != 85
        or accepted_anchor.get("last_accepted_transaction_key")
        != "0010:20260716:update:085"
        or latest_checkpoint.get("update") != 85
        or latest_checkpoint.get("checkpoint_sha256")
        != receipts[10]["checkpoint_sha256"]
        or latest_checkpoint.get("policy_state_sha256")
        != receipts[10]["policy_state_sha256"]
        or lineage_epoch.get("first_update") != 86
        or lineage_epoch.get("parent_artifact_sha256")
        != _sha256_bytes(parent_payload)
        or lineage_epoch.get("current_execution_identity_sha256")
        != current.get("execution_identity_sha256")
    ):
        raise Stage6R3ClosureError("planning-child continuation chain drifted")
    verify_checkpoint_member_bindings(latest_checkpoint, stage_root)
    external_paths = [
        _verify_file_binding(
            current.get("authorization"),
            "planning-child continuation authorization",
        )
    ]
    review = _mapping(
        continuation.get("review_evidence"),
        "planning-child continuation review evidence",
    )
    for label in sorted(review):
        external_paths.append(
            _verify_file_binding(
                review[label], f"planning-child continuation review {label}"
            )
        )
    prefixes = _mapping(
        continuation.get("journal_prefixes"),
        "planning-child continuation journal prefixes",
    )
    for relative, raw_binding in prefixes.items():
        binding = _mapping(raw_binding, f"planning-child prefix {relative}")
        if not isinstance(relative, str) or binding.get("path") != relative:
            raise Stage6R3ClosureError("planning-child prefix path drifted")
        path = stage_root / relative
        size = binding.get("size_bytes")
        count = binding.get("line_count")
        if type(size) is not int or type(count) is not int:
            raise Stage6R3ClosureError("planning-child prefix binding drifted")
        payload = path.read_bytes()[: int(size)]
        if (
            len(payload) != size
            or _sha256_bytes(payload) != binding.get("sha256")
            or len(payload.splitlines()) != count
        ):
            raise Stage6R3ClosureError("planning-child frozen prefix drifted")
    return (
        {
            "parent_artifact_sha256": _sha256_bytes(parent_payload),
            "continuation_artifact_sha256": _sha256_bytes(continuation_payload),
            "continuation_execution_identity_sha256": current.get(
                "execution_identity_sha256"
            ),
            "continuation_authorization_sha256": _mapping(
                current.get("authorization"),
                "planning-child continuation authorization",
            ).get("sha256"),
            "accepted_anchor_update": 85,
            "continuation_first_update": 86,
            "frozen_prefix_count": len(prefixes),
        },
        external_paths,
    )


def _verify_training_root(stage_root: Path) -> tuple[dict[str, object], list[Path]]:
    root = stage_root.expanduser().resolve()
    if root.name != "s6" or root.parent.name != FORMAL_RUN_ID:
        raise Stage6R3ClosureError("formal Stage 6 root identity drifted")
    config_path = root / "config.json"
    config_bytes = config_path.read_bytes()
    effective = parse_planning_effective_config_bytes(config_bytes)
    config_sha256 = _sha256_bytes(config_bytes)
    if effective.effective_config_sha256 != config_sha256:
        raise Stage6R3ClosureError("effective config SHA drifted")
    config = effective.base_config
    transactions = build_planning_child_transactions(config)
    receipts = CheckpointReceiptIndex.verify_snapshot_bytes(
        (root / "checkpoints/index.jsonl").read_bytes()
    )
    training_rows = _read_jsonl(root / "training_metrics.jsonl", "training metrics")
    validation_rows = _read_jsonl(
        root / "validation_metrics.jsonl", "validation metrics"
    )
    _, seed_best = validate_standard_training_validation_rows(
        config=config,
        transactions=transactions,
        receipt_rows=receipts,
        training_rows=training_rows,
        validation_rows=validation_rows,
        planning_warm_start=True,
    )
    if (
        tuple(transaction.update for transaction in transactions) != EXPECTED_UPDATES
        or tuple(
            transaction.update
            for transaction in transactions
            if transaction.validation_episodes == 16
        )
        != EXPECTED_VALIDATION_UPDATES
        or len(receipts) != 26
        or len(training_rows) != 26
        or len(validation_rows) != 3
    ):
        raise Stage6R3ClosureError("planning-child training schedule drifted")
    job_rows = Stage6StateJournal.verify_snapshot_bytes(
        (root / "job-state.jsonl").read_bytes()
    )
    completed = completed_transaction_keys_from_journal(job_rows, transactions)
    if (
        completed != tuple(transaction.key for transaction in transactions)
        or job_rows[-1].get("state") != f"seed_{SEED}_complete"
    ):
        raise Stage6R3ClosureError("planning-child job journal is incomplete")
    row_index = 0
    for transaction, receipt in zip(transactions, receipts, strict=True):
        for _state in transaction.commit_states:
            bindings = _mapping(
                job_rows[row_index].get("bindings"), "job-state bindings"
            )
            if (
                bindings.get("formal_run_id") != FORMAL_RUN_ID
                or bindings.get("config_sha256") != config_sha256
                or bindings.get("checkpoint_sha256")
                != receipt["checkpoint_sha256"]
            ):
                raise Stage6R3ClosureError("job-state receipt binding drifted")
            row_index += 1
    global_best = _read_json(root / "global-best.json", "training global best")
    best_record = _mapping(global_best.get("record"), "training best record")
    u80_receipt = receipts[5]
    if (
        global_best.get("schema_version") != "stage6_global_best/v1"
        or global_best.get("transaction_key") != "0005:20260716:update:080"
        or best_record.get("update") != 80
        or global_best.get("checkpoint_sha256")
        != u80_receipt["checkpoint_sha256"]
        or global_best.get("policy_state_sha256")
        != u80_receipt["policy_state_sha256"]
        or seed_best[SEED].get("update") != 80
    ):
        raise Stage6R3ClosureError("training global best drifted")
    checkpoints = {
        update: _checkpoint_identity(root, update) for update in (80, 100)
    }
    checkpoint_rows = _read_jsonl(
        root / "checkpoint_audit.jsonl", "checkpoint replay audit"
    )
    if len(checkpoint_rows) != 1:
        raise Stage6R3ClosureError("checkpoint replay audit count drifted")
    audit = checkpoint_rows[0]
    validate_checkpoint_replay_audit(
        audit,
        transaction_key=str(receipts[-1]["transaction_key"]),
        seed=SEED,
        update=100,
        checkpoint_sha256=str(receipts[-1]["checkpoint_sha256"]),
        complete_marker_sha256=str(receipts[-1]["complete_marker_sha256"]),
        policy_state_sha256=str(receipts[-1]["policy_state_sha256"]),
        lineage_sha256=str(audit.get("lineage_sha256")),
    )
    resource_rows = _read_jsonl(root / "resource_audit.jsonl", "resource audit")
    resource_validation = validate_resource_lifecycle_rows(
        resource_rows, require_terminal=False
    )
    accepted_updates = [
        row
        for row in resource_rows
        if row.get("kind") == "update"
        and row.get("phase") == "accepted"
        and row.get("accepted") is True
    ]
    legacy_final_rows = [
        row for row in resource_rows if row.get("kind") == "final_evaluation"
    ]
    if (
        resource_validation.get("passed") is not True
        or len(accepted_updates) != 26
        or [row.get("transaction_key") for row in accepted_updates]
        != [transaction.key for transaction in transactions]
        or len(legacy_final_rows) != 1
        or legacy_final_rows[0].get("phase") != "pre"
        or legacy_final_rows[0].get("accepted") is not False
    ):
        raise Stage6R3ClosureError("training resource lifecycle drifted")
    planning_chain, external_paths = _verify_planning_child_chain(root, receipts)
    return (
        {
            "formal_run_id": FORMAL_RUN_ID,
            "seed": SEED,
            "stage_root": root.as_posix(),
            "config_sha256": config_sha256,
            "accepted_update_count": 26,
            "first_update": 75,
            "last_update": 100,
            "validation_updates": list(EXPECTED_VALIDATION_UPDATES),
            "checkpoint_receipt_count": len(receipts),
            "job_state_count": len(job_rows),
            "resource_segment_count": resource_validation.get("segment_count"),
            "resource_accepted_update_count": len(accepted_updates),
            "legacy_incomplete_final_attempt": {
                "transaction_key": legacy_final_rows[0].get("transaction_key"),
                "phase": "pre",
                "accepted": False,
                "used_as_final_evidence": False,
            },
            "training_global_best": global_best,
            "u80_checkpoint": checkpoints[80],
            "u100_checkpoint": checkpoints[100],
            "u100_checkpoint_replay_passed": True,
            "planning_child_chain": planning_chain,
        },
        external_paths,
    )


def _verify_screen(
    *,
    eval_root: Path,
    source_set_sha256: str,
    config_sha256: str,
    checkpoints: Mapping[int, Mapping[str, object]],
    global_best: Mapping[str, object],
) -> dict[str, object]:
    summaries: dict[str, object] = {}
    records: list[ValidationRecord] = []
    for update in EXPECTED_SCREEN_UPDATES:
        stem = f"screen/u{update:03d}-validation-16"
        summary_path = eval_root / f"{stem}.summary.json"
        trace_path = eval_root / f"{stem}.jsonl"
        summary = _read_json(summary_path, f"R3 U{update} screen summary")
        trace_bytes = trace_path.read_bytes()
        checkpoint = checkpoints[update]
        resources = summary.get("resources")
        if (
            summary.get("schema_version") != "stage6_r3_evaluation_result/v1"
            or summary.get("split") != "validation"
            or summary.get("method") != "ppo_policy"
            or summary.get("episode_count") != 16
            or summary.get("trace_line_count") != 16
            or len(trace_bytes.splitlines()) != 16
            or summary.get("trace_sha256") != _sha256_bytes(trace_bytes)
            or summary.get("source_set_sha256") != source_set_sha256
            or summary.get("config_sha256") != config_sha256
            or summary.get("checkpoint_sha256")
            != checkpoint["checkpoint_sha256"]
            or summary.get("policy_state_sha256")
            != checkpoint["policy_state_sha256"]
            or not isinstance(resources, list)
            or not resources
            or any(not _resource_snapshot_passed(row) for row in resources)
        ):
            raise Stage6R3ClosureError(f"R3 U{update} screen binding drifted")
        result = _mapping(summary.get("result"), f"R3 U{update} screen result")
        metrics = _mapping(result.get("metrics"), f"R3 U{update} screen metrics")
        if not _safety_metrics_passed(metrics):
            raise Stage6R3ClosureError(f"R3 U{update} screen safety drifted")
        records.append(
            ValidationRecord(
                seed=SEED,
                update=update,
                success_rate_under_fixed_step_budget=float(
                    metrics["success_rate_under_fixed_step_budget"]
                ),
                mean_final_coverage=float(metrics["mean_final_coverage"]),
                checkpoint_ref=f"update-{update:08d}",
            )
        )
        summaries[str(update)] = summary
    selected = select_seed_best(records)
    if (
        selected.update != 80
        or _sha256_bytes(ArtifactStore.canonical_json_bytes(summaries))
        != global_best.get("screen_summary_sha256")
    ):
        raise Stage6R3ClosureError("R3 global-best screen selection drifted")
    return {
        "updates": list(EXPECTED_SCREEN_UPDATES),
        "episodes_each": 16,
        "selected_update": selected.update,
        "records": [record.to_dict() for record in records],
    }


def _verify_eval_root(
    *,
    repo_root: Path,
    stage_root: Path,
    eval_root: Path,
    stderr_log: Path,
    training: Mapping[str, object],
) -> dict[str, object]:
    root = eval_root.expanduser().resolve()
    # The completed R3 writer serialized an int-keyed {80: ..., 100: ...}
    # mapping.  JSON parsing turns those keys into strings, so a second
    # sort_keys dump reverses their order even though the original writer was
    # deterministic.  Bind the exact file bytes in the input manifest and
    # validate every semantic field below instead of requiring a lossy re-dump.
    identity = _read_json(
        root / "identity.json", "R3 identity", canonical=False
    )
    recorded_source = _mapping(identity.get("source_identity"), "R3 source identity")
    current_source = _runtime_source_identity(repo_root)
    if (
        identity.get("schema_version") != "stage6_r3_evaluation_pipeline/v1"
        or Path(str(identity.get("stage_root"))).resolve() != stage_root.resolve()
        or Path(str(identity.get("output_root"))).resolve() != root
        or identity.get("git_head") != _git(repo_root, "rev-parse", "HEAD")
        or identity.get("git_branch") != _git(repo_root, "branch", "--show-current")
        or dict(recorded_source) != current_source
        or identity.get("config_sha256") != training["config_sha256"]
    ):
        raise Stage6R3ClosureError("R3 execution identity drifted")
    source_set_sha256 = str(current_source["source_set_sha256"])
    checkpoints = {
        update: _checkpoint_identity(stage_root, update)
        for update in EXPECTED_SCREEN_UPDATES
    }
    recorded_checkpoints = _mapping(identity.get("checkpoints"), "R3 checkpoints")
    if any(
        dict(_mapping(recorded_checkpoints.get(str(update)), f"R3 U{update}"))
        != checkpoints[update]
        for update in EXPECTED_SCREEN_UPDATES
    ):
        raise Stage6R3ClosureError("R3 checkpoint identity drifted")
    global_best = _read_json(root / "global-best.json", "R3 global best")
    selected_record = _mapping(global_best.get("record"), "R3 best record")
    selected_update = selected_record.get("update")
    if (
        global_best.get("schema_version") != "stage6_r3_global_best/v1"
        or selected_update != 80
        or global_best.get("source_set_sha256") != source_set_sha256
        or global_best.get("checkpoint_sha256")
        != checkpoints[80]["checkpoint_sha256"]
        or global_best.get("policy_state_sha256")
        != checkpoints[80]["policy_state_sha256"]
        or global_best.get("checkpoint_sha256")
        != _mapping(training["training_global_best"], "training global best").get(
            "checkpoint_sha256"
        )
    ):
        raise Stage6R3ClosureError("R3 global best drifted")
    screen = _verify_screen(
        eval_root=root,
        source_set_sha256=source_set_sha256,
        config_sha256=str(training["config_sha256"]),
        checkpoints=checkpoints,
        global_best=global_best,
    )
    final_summary = _read_json(root / "final-summary.json", "R3 final summary")
    validate_final_summary_schedule(
        final_summary,
        expected_source_set_sha256=source_set_sha256,
        expected_config_sha256=str(training["config_sha256"]),
        expected_checkpoint_sha256=str(checkpoints[80]["checkpoint_sha256"]),
        expected_policy_state_sha256=str(checkpoints[80]["policy_state_sha256"]),
        expected_update=80,
    )
    effective = parse_planning_effective_config_bytes(
        (stage_root / "config.json").read_bytes()
    )
    safety_contract = SafetyContract.from_stage6_config(effective.base_config)
    results = _mapping(final_summary["results"], "R3 final results")
    replay_rows: list[dict[str, object]] = []
    metric_rows: dict[str, object] = {}
    for key in EXPECTED_FINAL_KEYS:
        split, method = key.split(":", maxsplit=1)
        row = _mapping(results[key], f"R3 final result {key}")
        expected_trace_relative = f"final/{split}-{method}-64.jsonl"
        if row.get("trace_path") != expected_trace_relative:
            raise Stage6R3ClosureError("R3 final trace path drifted")
        trace_path = root / expected_trace_relative
        summary_path = root / f"final/{split}-{method}-64.summary.json"
        persisted_summary = _read_json(summary_path, f"R3 final summary {key}")
        if dict(row) != persisted_summary:
            raise Stage6R3ClosureError("R3 final summary graph drifted")
        trace_bytes = trace_path.read_bytes()
        if (
            len(trace_bytes.splitlines()) != 64
            or _sha256_bytes(trace_bytes) != row.get("trace_sha256")
        ):
            raise Stage6R3ClosureError("R3 final trace binding drifted")
        result = _mapping(row.get("result"), f"R3 final result payload {key}")
        replay_artifacts = build_standard_replay_artifacts(
            trace_bytes=trace_bytes,
            trace_name=trace_path.name,
            split=split,
            method=method,
            config_sha256=str(training["config_sha256"]),
            checkpoint_sha256=str(checkpoints[80]["checkpoint_sha256"]),
            policy_state_sha256=str(checkpoints[80]["policy_state_sha256"]),
            result=result,
        )
        replay = verify_standard_final_evaluation_artifacts_from_bytes(
            trace_bytes=trace_bytes,
            trace_name=trace_path.name,
            summary_bytes=replay_artifacts["summary_bytes"],  # type: ignore[arg-type]
            summary_name=str(replay_artifacts["summary_name"]),
            commit_bytes=replay_artifacts["commit_bytes"],  # type: ignore[arg-type]
            commit_name=str(replay_artifacts["commit_name"]),
            split=split,
            method=method,
            config_sha256=str(training["config_sha256"]),
            safety_contract=safety_contract,
            checkpoint_sha256=str(checkpoints[80]["checkpoint_sha256"]),
            policy_state_sha256=str(checkpoints[80]["policy_state_sha256"]),
            bootstrap_resamples=effective.base_config.evaluation.bootstrap_resamples,
            bootstrap_seed=effective.base_config.evaluation.bootstrap_seed,
        )
        planner_failures = replay.get("planner_failure_counts")
        if (
            replay.get("episode_count") != 64
            or replay.get("safety_violation_count") != 0
            or not isinstance(planner_failures, Mapping)
            or sum(int(value) for value in planner_failures.values()) != 0
        ):
            raise Stage6R3ClosureError("R3 final production replay failed safety")
        replay_rows.append(
            {
                "key": key,
                "episode_count": 64,
                "safety_violation_count": 0,
                "planner_failure_count": (
                    None
                    if planner_failures is None
                    else sum(int(value) for value in planner_failures.values())
                ),
                "trace_sha256": row["trace_sha256"],
                "summary_sha256": _sha256_file(summary_path),
            }
        )
        metric_rows[key] = _mapping(result.get("metrics"), f"R3 metrics {key}")
    state_rows = _read_jsonl(root / "state.jsonl", "R3 state journal")
    if (
        not state_rows
        or state_rows[-1].get("event") != "pipeline_completed"
        or state_rows[-1].get("evaluation_count") != 10
        or state_rows[-1].get("episode_count") != 640
    ):
        raise Stage6R3ClosureError("R3 pipeline terminal state drifted")
    stderr_identity = _file_identity(stderr_log)
    if stderr_identity["size_bytes"] != 0:
        raise Stage6R3ClosureError("R3 stderr is not empty")
    return {
        "eval_root": root.as_posix(),
        "identity_sha256": _sha256_file(root / "identity.json"),
        "source_set_sha256": source_set_sha256,
        "global_best_sha256": _sha256_file(root / "global-best.json"),
        "final_summary_sha256": _sha256_file(root / "final-summary.json"),
        "selected_update": 80,
        "selected_checkpoint_sha256": checkpoints[80]["checkpoint_sha256"],
        "selected_policy_state_sha256": checkpoints[80]["policy_state_sha256"],
        "screen": screen,
        "evaluation_count": 10,
        "episode_count": 640,
        "production_replay_passed": True,
        "production_replay": replay_rows,
        "metrics": metric_rows,
        "stderr": stderr_identity,
    }


def verify_stage6_r3_closure(
    *,
    repo_root: Path,
    stage_root: Path,
    eval_root: Path,
    stderr_log: Path,
) -> dict[str, object]:
    """Run one read-only, currentness-bound Stage 6 terminal verification."""

    repo = repo_root.expanduser().resolve()
    stage = stage_root.expanduser().resolve()
    evaluation = eval_root.expanduser().resolve()
    stderr = stderr_log.expanduser().resolve()
    require_canonical_closure_paths(stage, evaluation, stderr)
    runners_before = _active_stage6_runners()
    if runners_before:
        raise Stage6R3ClosureError("Stage 6 runner is still active")
    base_manifest_before = capture_closure_input_manifest(
        stage,
        evaluation,
        stderr,
    )
    source_before = _runtime_source_identity(repo)
    training, external_paths = _verify_training_root(stage)
    if (
        capture_closure_input_manifest(stage, evaluation, stderr)
        != base_manifest_before
    ):
        raise Stage6R3ClosureError(
            "Stage 6 root inputs changed during training verification"
        )
    manifest_before = capture_closure_input_manifest(
        stage,
        evaluation,
        stderr,
        external_paths,
    )
    final_evaluation = _verify_eval_root(
        repo_root=repo,
        stage_root=stage,
        eval_root=evaluation,
        stderr_log=stderr,
        training=training,
    )
    manifest_after = capture_closure_input_manifest(
        stage,
        evaluation,
        stderr,
        external_paths,
    )
    source_after = _runtime_source_identity(repo)
    runners_after = _active_stage6_runners()
    if (
        manifest_after != manifest_before
        or source_after != source_before
        or runners_after
    ):
        raise Stage6R3ClosureError("Stage 6 closure inputs changed during verification")
    return {
        "schema_version": SCHEMA,
        "status": "passed",
        "closure_state": "machine_passed_awaiting_independent_review",
        "scope": {
            "stage": 6,
            "single_seed": SEED,
            "training_rerun": False,
            "final_evaluation_rerun": False,
            "checkpoint_published": False,
            "default_policy_replaced": False,
            "executor_connected": False,
            "canary_started": False,
            "stage7_entered": False,
        },
        "repo": {
            "root": repo.as_posix(),
            "git_head": _git(repo, "rev-parse", "HEAD"),
            "git_branch": _git(repo, "branch", "--show-current"),
        },
        "runner_processes": [],
        "training": training,
        "final_evaluation": final_evaluation,
        "input_manifest": manifest_before,
    }


def _report(payload: Mapping[str, object]) -> bytes:
    training = _mapping(payload.get("training"), "closure training")
    final = _mapping(payload.get("final_evaluation"), "closure final evaluation")
    metrics = _mapping(final.get("metrics"), "closure metrics")
    ppo_test = _mapping(metrics.get("test:ppo_policy"), "test PPO metrics")
    ppo_unseen = _mapping(metrics.get("unseen:ppo_policy"), "unseen PPO metrics")
    lines = [
        "# Stage 6 R3 terminal closure",
        "",
        "- Machine verification: PASS",
        f"- State: {payload.get('closure_state')}",
        f"- Formal run: {training.get('formal_run_id')}",
        f"- Training: {training.get('accepted_update_count')}/26 accepted updates (U75-U100)",
        f"- Validation updates: {training.get('validation_updates')}",
        f"- Selected global best: U{final.get('selected_update')}",
        f"- Final evaluation: {final.get('evaluation_count')}/10 groups, {final.get('episode_count')}/640 episodes",
        f"- Test PPO success: {ppo_test.get('success_rate_under_fixed_step_budget')}",
        f"- Test PPO final coverage: {ppo_test.get('mean_final_coverage')}",
        f"- Unseen PPO success: {ppo_unseen.get('success_rate_under_fixed_step_budget')}",
        f"- Unseen PPO final coverage: {ppo_unseen.get('mean_final_coverage')}",
        "- Production trace replay: PASS",
        "- Legacy canonical final pre row: recorded as incomplete and not used",
        "- Checkpoint publication/default-policy replacement/executor/canary: not performed",
        "- Boundary: stopped in Stage 6; Stage 7 not entered",
        "",
    ]
    return "\n".join(lines).encode("utf-8")


def publish_closure_artifacts(
    output_root: Path,
    payload: Mapping[str, object],
) -> tuple[Path, Path]:
    """Publish the machine result and human report exactly once."""

    root = output_root.expanduser().resolve()
    if root.exists():
        raise FileExistsError(root)
    report_payload = _report(payload)
    store = ArtifactStore(root)
    machine = store.write_json_exclusive("machine-acceptance.json", dict(payload))
    report = store.write_bytes_exclusive("report.md", report_payload)
    return machine, report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--stage-root", type=Path, default=DEFAULT_STAGE_ROOT)
    parser.add_argument("--eval-root", type=Path, default=DEFAULT_EVAL_ROOT)
    parser.add_argument("--stderr-log", type=Path, default=DEFAULT_STDERR_LOG)
    parser.add_argument("--output-root", type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--publish", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = verify_stage6_r3_closure(
        repo_root=args.repo_root,
        stage_root=args.stage_root,
        eval_root=args.eval_root,
        stderr_log=args.stderr_log,
    )
    if args.dry_run:
        print(
            json.dumps(
                {
                    "dry_run": "passed",
                    "status": payload["status"],
                    "closure_state": payload["closure_state"],
                    "accepted_update_count": payload["training"]["accepted_update_count"],  # type: ignore[index]
                    "selected_update": payload["final_evaluation"]["selected_update"],  # type: ignore[index]
                    "evaluation_count": payload["final_evaluation"]["evaluation_count"],  # type: ignore[index]
                    "episode_count": payload["final_evaluation"]["episode_count"],  # type: ignore[index]
                },
                sort_keys=True,
            )
        )
        return 0
    if args.output_root is None:
        raise Stage6R3ClosureError("--publish requires --output-root")
    output_root = args.output_root.expanduser().resolve()
    if output_root.drive.upper() != "D:":
        raise Stage6R3ClosureError("closure output root must be on D drive")
    machine, report = publish_closure_artifacts(output_root, payload)
    print(
        json.dumps(
            {
                "published": "machine_passed_awaiting_independent_review",
                "machine_acceptance": machine.as_posix(),
                "report": report.as_posix(),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
