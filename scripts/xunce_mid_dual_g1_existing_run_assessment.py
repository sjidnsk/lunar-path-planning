from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import stat
import statistics
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Mapping, Sequence


AUTHORIZED_RUN_ID = "g1-formal-20260727-0650-cst"
AUTHORIZED_SOURCE_ROOT = (
    "D:/xunce/out/mid_dual/g1/g1-formal-20260727-0650-cst"
)
AUTHORIZED_PRESERVATION_ROOT = (
    "D:/xunce/out/mid_dual/g1-lineage-preserve/"
    "g1-formal-20260727-0650-cst-active-snapshot-v1"
)
AUTHORIZED_REVIEW_BASE_ROOT = "D:/xunce/out/mid_dual/g1-review"
AUTHORIZED_ENVELOPE_BASE_ROOT = "D:/xunce/out/mid_dual/g1-assessment-use"
AUTHORIZED_PHASE_STATE_SHA256 = (
    "1e8a98015839b9190c86efc531134b7cbde1be393250cf40479b19a98c5b0fe8"
)
AUTHORIZED_SCENARIO_MANIFEST_SHA256 = (
    "37a3e639b4decfaf660ff66408c5c7109808c5f44aa22c18e98cd901050bb54a"
)
AUTHORIZED_INPUT_SHA256 = (
    "d358d063e5de944cb7d052031433a6913e872787ed0d816a952dfd0bbfa4191d"
)
AUTHORIZED_CONFIG_SEMANTIC_SHA256 = (
    "82e0d9a034bbf609af5b73a70fcfd78cc18c5f89d4c09c8e45097b441fe357a6"
)
AUTHORIZED_PRESERVATION_METADATA_SHA256 = (
    "41979abc3dac685331769ebc7d3c52ae1f14758408649bd67a696ea05530cf5f"
)
AUTHORIZED_PARENT_PID = 17212
AUTHORIZED_WORKER_PIDS = (
    48712,
    44764,
    44180,
    39500,
    40300,
    36024,
    36532,
    45828,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
GENERATOR_SOURCE_RELATIVE_PATH = (
    "scripts/xunce_mid_dual_g1_existing_run_assessment.py"
)

AUTHORITY_ALLOWLIST = {
    (
        ".superpowers/sdd/2026-07-26-midterm-dual-gate-experiment/"
        "g1-existing-run-assessment-use-ruling.md"
    ): "9d02d80aacf0dbff5046d1fb149f5044ddfc6bfd42faaf0b553746947f89331d",
    (
        ".superpowers/sdd/2026-07-26-midterm-dual-gate-experiment/"
        "g1-lineage-salvage-review.md"
    ): "1ff2af3c3eb2fe61e02754aa04c00702f1aee082bfbfd8050f3b5031dd1393c4",
    (
        ".superpowers/sdd/2026-07-26-midterm-dual-gate-experiment/"
        "task-11-consolidated-repair-brief.md"
    ): "cc444b1fd3446f1fbe0a57e7d9115f601f9c7ff8f26e70edd0094100f840edec",
}

AUTHORIZED_SOURCE_FILE_SHA256 = {
    "config.json": (
        "571b741df72ac8e5b87fbb8eeb1188a5ede49a447594a7109f2eaf1d0b8e0a9a"
    ),
    "phase-state.jsonl": AUTHORIZED_PHASE_STATE_SHA256,
    "phase-attempts.jsonl": (
        "84ddda293a8bf4d5bc581ec795a23d7438751ed6e17da83221e3f89c3e395eeb"
    ),
    "phases/p01/a01/audit.json": (
        "dfdf2c81c90598dd99d95ddd1f28e069dbdbe70827416eab367b798fda65e5d2"
    ),
    "phases/p01/a01/results.jsonl": (
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    ),
    "phases/p02/a01/audit.json": (
        "49510920fba930519d1105e78c4bfc285f1f3fbad5bd8f2c4a4a11305c0dec50"
    ),
    "phases/p02/a01/results.jsonl": (
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    ),
    "phases/p03/a01/audit.json": (
        "a1dfaa9d513d62a7b9bd535c5b64e69dd66a96202f4ceaee7f5845ce3e21bfb4"
    ),
    "phases/p03/a01/results.jsonl": (
        "465ffab8e973d7e586d648d54f6bac001d38163da87af0375ecefc37e37b60d1"
    ),
    "phases/p04/a01/audit.json": (
        "9bdc3deb73ff1850ba538d59527000d73e807cd3ef942679de64da5f1e3faf66"
    ),
    "phases/p04/a01/results.jsonl": (
        "425d6393409d78fbeb0fb0619212c8927b713a971cd20e04fef7ba3df1853bab"
    ),
}

AUTHORIZED_PRESERVATION_ARTIFACT_SHA256 = {
    "root-commit.tar": (
        "3349fe3afe50f729f863ee8a0e18047ebaa6c6ef1d39f96851686c571463e69d"
    ),
    "path-planner-commit.tar": (
        "7ef19c7473f671d83ab3562b048e484bfb83f5c62f3327d718fe155507220e11"
    ),
    "runtime-working-tree.tar": (
        "59b447915d7a2c7a2098f72206556a1bb74be6135b9e1f04aca3440def84a11f"
    ),
    "runtime-file-inventory.json": (
        "0453e81e5e482328985ebb51d2b6287924e77289021a40cd70ad6049e49a2826"
    ),
    "runtime-files.txt": (
        "59b667b473b66f7e6eef2959b401dac476f783e504bc92e4190aac5c17e9afef"
    ),
    "root-status.txt": (
        "c4c09f8e221b1d1113cc9d65efb131f5c1f22228194392c3f246fc2dc3b5c0c6"
    ),
    "path-planner-status.txt": (
        "01ba4719c80b6fe911b091a7c05124b64eeece964e09c058ef8f9805daca546b"
    ),
}

SCALE_PROFILE = "midterm_reduced_w8x3_update80/v1"
G1_RUNNER_ID = "run_xunce_mid_dual_g1_coverage/v1"
G1_GATE_ID = "g1"
CHECKPOINT_SHA256 = (
    "35e04c86f9f973af028fb08f0175d42ab45378d09e1f2b96ee6aad6e4c12b5b5"
)
POLICY_STATE_SHA256 = (
    "3123e6adde99be41e3bd5cc2f3843068e5416892395926d759c2c6a243ffd381"
)
DENOMINATOR_SOURCE = "reachable_observable_free_highres_cells/v1"
DENOMINATOR_ALGORITHM = "exact_reachable_safe_pose_range_los/v1"
FROZEN_MANIFEST_SCHEMA = "mid-dual-scenario-freeze/v1"
INPUT_AUDIT_SCHEMA = "xunce-mid-dual-g1-input-audit/v1"
PHASE_AUDIT_SCHEMA = "xunce-mid-dual-g1-phase-audit/v1"
REVIEW_SCHEMA = "xunce-mid-dual-g1-existing-run-review/v1"
STOP_EVIDENCE_SCHEMA = "xunce-mid-dual-g1-existing-run-stop-evidence/v1"
ASSESSMENT_SCHEMA = "xunce-mid-dual-g1-existing-run-assessment-use/v1"
SOURCE_SNAPSHOT_SCHEMA = (
    "xunce-mid-dual-g1-existing-run-source-phase-snapshot/v1"
)
DERIVED_INPUT_SCHEMA = (
    "xunce-mid-dual-g1-existing-run-reconstructed-input-audit/v1"
)
REVIEW_MANIFEST_SCHEMA = (
    "xunce-mid-dual-g1-existing-run-review-manifest/v1"
)
ASSESSMENT_MANIFEST_SCHEMA = (
    "xunce-mid-dual-g1-existing-run-assessment-manifest/v1"
)

REQUIRED_PHASE_IDS = ("p01", "p02", "p03", "p04")
NATIVE_REQUIRED_PHASE_IDS = ("p01", "p02", "p03", "p04", "p05", "p06", "p07")
FORMAL_SPLITS = ("test_q24", "unseen24")
PHASE_SPLITS = {"p03": "test_q24", "p04": "unseen24"}
PHASE_NAMES = {
    "p01": "preflight",
    "p02": "validation_dry_run",
    "p03": "test_q24",
    "p04": "unseen24",
}
BOOTSTRAP_SEED = 20260726
BOOTSTRAP_RESAMPLES = 2000
MIDTERM_THRESHOLD = 0.80
FINAL_THRESHOLD = 0.99
MINIMUM_PASS_COUNT = 23
EXPECTED_DENOMINATOR_PROOF_COUNT = 364
EXPECTED_LANE_COUNTS = {f"lane-{index}": 3 for index in range(8)}
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")

UNEXPECTED_SOURCE_RELATIVE_PATHS = (
    "g1_input_audit.json",
    "results.jsonl",
    "summary.json",
    "routing.json",
    "report.md",
    "manifest.json",
    "manifest.sha256",
    "phases/p05",
    "phases/p06",
    "phases/p07",
)

_COVERAGE_KEYS = frozenset(
    {
        "row_kind",
        "schema_version",
        "gate_id",
        "runner_id",
        "phase_id",
        "phase_name",
        "scale_profile",
        "run_id",
        "split",
        "episode_id",
        "episode_index",
        "scenario_id",
        "lane_id",
        "source_sha256",
        "config_sha256",
        "input_sha256",
        "code_sha256",
        "scenario_manifest_sha256",
        "checkpoint_sha256",
        "policy_state_sha256",
        "denominator_source",
        "denominator_algorithm",
        "denominator_sha256",
        "denominator_cell_count",
        "initial_covered_cell_count",
        "final_covered_cell_count",
        "coverage",
        "elapsed_ms",
        "steps_executed",
        "termination_reason",
        "safety_violation_count",
        "masked_action_count",
    }
)
_TRACE_ROW_KEYS = frozenset(
    {
        "row_kind",
        "schema_version",
        "gate_id",
        "runner_id",
        "phase_id",
        "phase_name",
        "scale_profile",
        "run_id",
        "split",
        "episode_id",
        "episode_index",
        "scenario_id",
        "lane_id",
        "source_sha256",
        "config_sha256",
        "input_sha256",
        "code_sha256",
        "scenario_manifest_sha256",
        "checkpoint_sha256",
        "policy_state_sha256",
        "step_index",
        "trace",
    }
)


class ExistingRunAssessmentBlocked(RuntimeError):
    """A stable fail-closed assessment rejection."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


@dataclass(frozen=True, slots=True)
class VerifiedExistingRunAssessment:
    source_root: Path
    review_root: Path
    envelope_root: Path
    phase_snapshot_sha256: str
    stop_evidence_sha256: str
    review_manifest_sha256: str
    review_result_sha256: str
    envelope_sha256: str
    envelope_manifest_sha256: str
    verifier_source_sha256: str
    assessment_use_status: str
    strict_prestart_lineage_status: str
    lineage_limitation_acknowledged: bool
    numerical_gate_status: str
    numerical_result_status: str
    review_metrics: Mapping[str, object]
    snapshot: Mapping[str, bytes]


def _blocked(reason: str) -> ExistingRunAssessmentBlocked:
    return ExistingRunAssessmentBlocked(reason)


def _bytes_sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise _blocked("existing_run_json_canonicalization_invalid") from exc


def _canonical_sha256(value: object) -> str:
    return _bytes_sha256(_canonical_bytes(value))


def _json_bytes(value: object) -> bytes:
    try:
        return (
            json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise _blocked("existing_run_json_serialization_invalid") from exc


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise _blocked("existing_run_json_duplicate_key")
        result[key] = value
    return result


def _reject_nonfinite_constant(_: str) -> object:
    raise _blocked("existing_run_json_nonfinite")


def _strict_json_object(payload: bytes, reason: str) -> dict[str, object]:
    try:
        text = payload.decode("utf-8-sig")
        value = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite_constant,
        )
    except ExistingRunAssessmentBlocked:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError) as exc:
        raise _blocked(reason) from exc
    if not isinstance(value, dict):
        raise _blocked(reason)
    return value


def _strict_jsonl(payload: bytes, reason: str) -> list[dict[str, object]]:
    try:
        text = payload.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise _blocked(reason) from exc
    rows: list[dict[str, object]] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        rows.append(_strict_json_object(line.encode("utf-8"), reason))
    return rows


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and SHA256_PATTERN.fullmatch(value) is not None


def _exact_int(value: object, reason: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise _blocked(reason)
    return value


def _finite_number(value: object, reason: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _blocked(reason)
    number = float(value)
    if not math.isfinite(number) or number < minimum:
        raise _blocked(reason)
    return number


def _path_text(path: Path) -> str:
    return path.as_posix()


def _is_link_or_reparse(path: Path) -> bool:
    try:
        metadata = os.lstat(path)
    except OSError as exc:
        raise _blocked("existing_run_unsafe_path") from exc
    attributes = int(getattr(metadata, "st_file_attributes", 0))
    reparse_flag = int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0))
    return stat.S_ISLNK(metadata.st_mode) or bool(
        reparse_flag and attributes & reparse_flag
    )


def _assert_existing_components_safe(path: Path) -> None:
    absolute = Path(path)
    if not absolute.is_absolute():
        raise _blocked("existing_run_unsafe_path")
    parts = absolute.parts
    if not parts:
        raise _blocked("existing_run_unsafe_path")
    current = Path(parts[0])
    for part in parts[1:]:
        current = current / part
        if not current.exists():
            raise _blocked("existing_run_unsafe_path")
        if _is_link_or_reparse(current):
            raise _blocked("existing_run_unsafe_path")


def _assert_output_ancestors_safe(path: Path) -> None:
    absolute = Path(path)
    if not absolute.is_absolute():
        raise _blocked("existing_run_unsafe_path")
    existing = absolute
    while not existing.exists():
        parent = existing.parent
        if parent == existing:
            raise _blocked("existing_run_unsafe_path")
        existing = parent
    _assert_existing_components_safe(existing)


def _canonical_existing_root(
    path: Path,
    *,
    expected: str,
    reason: str,
) -> Path:
    raw = Path(path)
    if not raw.is_absolute() or ".." in raw.parts:
        raise _blocked(reason)
    try:
        resolved = raw.resolve(strict=True)
    except OSError as exc:
        raise _blocked(reason) from exc
    if not resolved.is_dir() or _path_text(resolved) != expected:
        raise _blocked(reason)
    try:
        _assert_existing_components_safe(resolved)
    except ExistingRunAssessmentBlocked as exc:
        raise _blocked(reason) from exc
    return resolved


def _canonical_existing_file(path: Path, reason: str) -> Path:
    raw = Path(path)
    if not raw.is_absolute() or ".." in raw.parts:
        raise _blocked(reason)
    try:
        resolved = raw.resolve(strict=True)
    except OSError as exc:
        raise _blocked(reason) from exc
    if not resolved.is_file():
        raise _blocked(reason)
    try:
        _assert_existing_components_safe(resolved)
    except ExistingRunAssessmentBlocked as exc:
        raise _blocked(reason) from exc
    return resolved


def _canonical_new_root(path: Path, *, expected: str) -> Path:
    raw = Path(path)
    if not raw.is_absolute() or ".." in raw.parts:
        raise _blocked("existing_run_output_scope_mismatch")
    try:
        resolved = raw.resolve(strict=False)
    except OSError as exc:
        raise _blocked("existing_run_output_scope_mismatch") from exc
    if _path_text(resolved) != expected:
        raise _blocked("existing_run_output_scope_mismatch")
    _assert_output_ancestors_safe(resolved)
    if resolved.exists():
        raise _blocked("existing_run_output_exists")
    return resolved


def _safe_relative_path(value: object, reason: str) -> str:
    if not isinstance(value, str) or not value:
        raise _blocked(reason)
    if "\\" in value:
        raise _blocked(reason)
    relative = PurePosixPath(value)
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or "." in relative.parts
        or relative.as_posix() != value
    ):
        raise _blocked(reason)
    return value


def _stable_read(path: Path, reason: str) -> bytes:
    resolved = _canonical_existing_file(path, reason)
    try:
        before = os.stat(resolved, follow_symlinks=False)
        payload = resolved.read_bytes()
        after = os.stat(resolved, follow_symlinks=False)
    except OSError as exc:
        raise _blocked(reason) from exc
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    if identity_before != identity_after or len(payload) != before.st_size:
        raise _blocked(reason)
    return payload


def _write_exclusive(path: Path, payload: bytes) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as handle:
            handle.write(payload)
    except FileExistsError as exc:
        raise _blocked("existing_run_output_exists") from exc
    except OSError as exc:
        raise _blocked("existing_run_output_write_failed") from exc


def _directory_file_set(root: Path) -> set[str]:
    files: set[str] = set()
    try:
        for current, directories, names in os.walk(root, followlinks=False):
            current_path = Path(current)
            for name in [*directories, *names]:
                child = current_path / name
                if _is_link_or_reparse(child):
                    raise _blocked("existing_run_unsafe_path")
            for name in names:
                relative = (current_path / name).relative_to(root).as_posix()
                files.add(relative)
    except OSError as exc:
        raise _blocked("existing_run_unsafe_path") from exc
    return files


def _verify_authorities() -> None:
    for relative, expected_sha256 in AUTHORITY_ALLOWLIST.items():
        safe = _safe_relative_path(relative, "existing_run_authority_invalid")
        payload = _stable_read(
            Path(REPOSITORY_ROOT) / PurePosixPath(safe),
            "existing_run_authority_invalid",
        )
        if _bytes_sha256(payload) != expected_sha256:
            raise _blocked("existing_run_authority_invalid")


def _verify_preservation(
    preservation_root: Path,
    *,
    source_root_text: str,
) -> dict[str, object]:
    root = _canonical_existing_root(
        preservation_root,
        expected=AUTHORIZED_PRESERVATION_ROOT,
        reason="existing_run_preservation_invalid",
    )
    expected_files = {
        "snapshot-metadata.json",
        *AUTHORIZED_PRESERVATION_ARTIFACT_SHA256.keys(),
    }
    if _directory_file_set(root) != expected_files:
        raise _blocked("existing_run_preservation_invalid")
    metadata_bytes = _stable_read(
        root / "snapshot-metadata.json",
        "existing_run_preservation_invalid",
    )
    if (
        _bytes_sha256(metadata_bytes)
        != AUTHORIZED_PRESERVATION_METADATA_SHA256
    ):
        raise _blocked("existing_run_preservation_invalid")
    metadata = _strict_json_object(
        metadata_bytes,
        "existing_run_preservation_invalid",
    )
    if (
        metadata.get("schema_version")
        != "xunce-mid-dual-active-runtime-preservation-snapshot/v1"
        or metadata.get("run_id") != AUTHORIZED_RUN_ID
        or metadata.get("formal_root") != source_root_text
        or metadata.get("formal_evidence_eligible") is not False
        or metadata.get("artifacts")
        != AUTHORIZED_PRESERVATION_ARTIFACT_SHA256
    ):
        raise _blocked("existing_run_preservation_invalid")
    for relative, expected_sha256 in (
        AUTHORIZED_PRESERVATION_ARTIFACT_SHA256.items()
    ):
        payload = _stable_read(
            root / relative,
            "existing_run_preservation_invalid",
        )
        if _bytes_sha256(payload) != expected_sha256:
            raise _blocked("existing_run_preservation_invalid")
    return {
        "root": _path_text(root),
        "snapshot_metadata_sha256": (
            AUTHORIZED_PRESERVATION_METADATA_SHA256
        ),
        "artifact_sha256": dict(
            AUTHORIZED_PRESERVATION_ARTIFACT_SHA256
        ),
        "formal_evidence_eligible": False,
        "evidence_role": "preservation_anchor_not_prestart_seal",
    }


def _phase_snapshot_files(source_root: Path) -> dict[str, bytes]:
    for unexpected in UNEXPECTED_SOURCE_RELATIVE_PATHS:
        candidate = source_root / PurePosixPath(unexpected)
        if candidate.exists():
            raise _blocked("existing_run_unexpected_artifact")
    snapshot: dict[str, bytes] = {}
    for relative, expected_sha256 in AUTHORIZED_SOURCE_FILE_SHA256.items():
        safe = _safe_relative_path(
            relative,
            "existing_run_phase_snapshot_invalid",
        )
        payload = _stable_read(
            source_root / PurePosixPath(safe),
            "existing_run_phase_snapshot_invalid",
        )
        if _bytes_sha256(payload) != expected_sha256:
            raise _blocked("existing_run_source_hash_mismatch")
        snapshot[safe] = payload
    return snapshot


def _validate_config(
    config: Mapping[str, object],
    *,
    source_root_text: str,
) -> None:
    checkpoint = config.get("checkpoint")
    execution = config.get("execution")
    denominator = config.get("denominator")
    bootstrap = config.get("bootstrap")
    scenario_manifest = config.get("scenario_manifest")
    input_audit = config.get("input_audit")
    if not all(
        isinstance(value, Mapping)
        for value in (
            checkpoint,
            execution,
            denominator,
            bootstrap,
            scenario_manifest,
            input_audit,
        )
    ):
        raise _blocked("existing_run_contract_invalid")
    assert isinstance(checkpoint, Mapping)
    assert isinstance(execution, Mapping)
    assert isinstance(denominator, Mapping)
    assert isinstance(bootstrap, Mapping)
    assert isinstance(scenario_manifest, Mapping)
    assert isinstance(input_audit, Mapping)
    lane_sizes = execution.get("lane_sizes")
    if (
        config.get("schema_version")
        != "xunce-mid-dual-g1-effective-config/v1"
        or config.get("gate_id") != G1_GATE_ID
        or config.get("runner_id") != G1_RUNNER_ID
        or config.get("scale_profile") != SCALE_PROFILE
        or config.get("run_id") != AUTHORIZED_RUN_ID
        or config.get("mode") != "formal"
        or config.get("output_root") != source_root_text
        or config.get("required_phase_ids") != list(NATIVE_REQUIRED_PHASE_IDS)
        or config.get("config_sha256")
        != AUTHORIZED_CONFIG_SEMANTIC_SHA256
        or config.get("input_sha256") != AUTHORIZED_INPUT_SHA256
        or not _is_sha256(config.get("code_sha256"))
        or checkpoint.get("update") != 80
        or checkpoint.get("sha256") != CHECKPOINT_SHA256
        or checkpoint.get("policy_state_sha256") != POLICY_STATE_SHA256
        or execution.get("worker_count") != 8
        or execution.get("episodes_per_formal_split") != 24
        or lane_sizes != [3] * 8
        or denominator.get("source") != DENOMINATOR_SOURCE
        or denominator.get("algorithm") != DENOMINATOR_ALGORITHM
        or denominator.get("integer_count_binding") is not True
        or bootstrap.get("seed") != BOOTSTRAP_SEED
        or bootstrap.get("resamples") != BOOTSTRAP_RESAMPLES
        or bootstrap.get("confidence_level") != 0.95
        or bootstrap.get("unit") != "episode"
        or scenario_manifest.get("schema_version") != FROZEN_MANIFEST_SCHEMA
        or scenario_manifest.get("sha256")
        != AUTHORIZED_SCENARIO_MANIFEST_SHA256
        or input_audit.get("path") != "g1_input_audit.json"
        or input_audit.get("schema_version") != INPUT_AUDIT_SCHEMA
        or input_audit.get("sha256") != AUTHORIZED_INPUT_SHA256
    ):
        raise _blocked("existing_run_contract_invalid")
    if not isinstance(scenario_manifest.get("path"), str):
        raise _blocked("existing_run_contract_invalid")


def _validate_manifest_and_reconstruct_input(
    *,
    manifest_path: Path,
    manifest_bytes: bytes,
    config: Mapping[str, object],
) -> tuple[dict[str, object], dict[tuple[str, int], dict[str, object]]]:
    if _bytes_sha256(manifest_bytes) != AUTHORIZED_SCENARIO_MANIFEST_SHA256:
        raise _blocked("existing_run_input_manifest_invalid")
    manifest = _strict_json_object(
        manifest_bytes,
        "existing_run_input_manifest_invalid",
    )
    cohorts = manifest.get("cohorts")
    proofs_value = manifest.get("denominator_proofs")
    attestation = manifest.get("policy_blind_attestation")
    if (
        manifest.get("schema_version") != FROZEN_MANIFEST_SCHEMA
        or manifest.get("completion_status") != "complete"
        or not isinstance(cohorts, Mapping)
        or not isinstance(proofs_value, list)
        or len(proofs_value) != EXPECTED_DENOMINATOR_PROOF_COUNT
        or not isinstance(attestation, Mapping)
    ):
        raise _blocked("existing_run_input_manifest_invalid")
    forbidden = attestation.get("forbidden_inputs_used")
    if (
        not isinstance(forbidden, Mapping)
        or set(forbidden)
        != {"checkpoint", "policy", "result", "reward", "runtime"}
        or any(value is not False for value in forbidden.values())
    ):
        raise _blocked("existing_run_input_manifest_invalid")
    proofs: dict[str, dict[str, object]] = {}
    for value in proofs_value:
        if not isinstance(value, dict):
            raise _blocked("existing_run_input_manifest_invalid")
        scenario_id = value.get("scenario_id")
        key = value.get("key")
        if (
            not isinstance(scenario_id, str)
            or not scenario_id
            or scenario_id in proofs
            or value.get("coverage_denominator_source")
            != DENOMINATOR_SOURCE
            or value.get("coverage_denominator_algorithm")
            != DENOMINATOR_ALGORITHM
            or value.get("exact") is not True
            or not _is_sha256(value.get("coverable_mask_sha256"))
            or type(value.get("coverable_cell_count")) is not int
            or int(value["coverable_cell_count"]) <= 0
            or not isinstance(key, Mapping)
            or key.get("max_slope_deg") != 30.0
        ):
            raise _blocked("existing_run_input_manifest_invalid")
        proofs[scenario_id] = value
    expected_cohort_sizes = {
        "test_q24": 24,
        "unseen24": 24,
        "test_c24": 24,
        "validation3": 3,
        "replay3": 3,
    }
    normalized_cohorts: dict[str, list[str]] = {}
    for cohort, expected_size in expected_cohort_sizes.items():
        values = cohorts.get(cohort)
        if (
            not isinstance(values, list)
            or len(values) != expected_size
            or len(set(values)) != expected_size
            or any(not isinstance(value, str) or not value for value in values)
        ):
            raise _blocked("existing_run_input_manifest_invalid")
        normalized_cohorts[cohort] = list(values)
    formal_scenarios = [
        *normalized_cohorts["test_q24"],
        *normalized_cohorts["unseen24"],
    ]
    if (
        len(set(formal_scenarios)) != 48
        or any(scenario_id not in proofs for scenario_id in formal_scenarios)
    ):
        raise _blocked("existing_run_input_manifest_invalid")

    formal_jobs: list[dict[str, object]] = []
    jobs: dict[tuple[str, int], dict[str, object]] = {}
    for split in FORMAL_SPLITS:
        for index, scenario_id in enumerate(normalized_cohorts[split]):
            proof = proofs[scenario_id]
            job = {
                "split": split,
                "scenario_id": scenario_id,
                "episode_id": f"{split}-episode-{index:02d}",
                "episode_index": index,
                "lane_id": f"lane-{index % 8}",
                "denominator_sha256": proof["coverable_mask_sha256"],
                "denominator_cell_count": proof["coverable_cell_count"],
                "denominator_source": DENOMINATOR_SOURCE,
                "denominator_algorithm": DENOMINATOR_ALGORITHM,
            }
            formal_jobs.append(job)
            jobs[(split, index)] = job
    input_audit = {
        "schema_version": INPUT_AUDIT_SCHEMA,
        "gate_id": G1_GATE_ID,
        "runner_id": G1_RUNNER_ID,
        "scale_profile": SCALE_PROFILE,
        "status": "verified",
        "scenario_manifest_path": _path_text(manifest_path),
        "scenario_manifest_schema_version": FROZEN_MANIFEST_SCHEMA,
        "scenario_manifest_sha256": _bytes_sha256(manifest_bytes),
        "scenario_manifest_canonical_sha256": _canonical_sha256(manifest),
        "policy_blind_attestation_sha256": _canonical_sha256(attestation),
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "policy_state_sha256": POLICY_STATE_SHA256,
        "denominator_source": DENOMINATOR_SOURCE,
        "denominator_algorithm": DENOMINATOR_ALGORITHM,
        "formal_split_order": list(FORMAL_SPLITS),
        "formal_jobs": formal_jobs,
        "test_c24_scenario_ids": normalized_cohorts["test_c24"],
        "validation3_scenario_ids": normalized_cohorts["validation3"],
        "replay3_scenario_ids": normalized_cohorts["replay3"],
    }
    config_input = config.get("input_audit")
    if (
        _canonical_sha256(input_audit) != AUTHORIZED_INPUT_SHA256
        or config.get("input_sha256") != _canonical_sha256(input_audit)
        or not isinstance(config_input, Mapping)
        or config_input.get("sha256") != _canonical_sha256(input_audit)
    ):
        raise _blocked("existing_run_input_reconstruction_mismatch")
    return input_audit, jobs


def _validate_phase_prefix(
    snapshot: Mapping[str, bytes],
) -> tuple[
    dict[str, object],
    dict[str, dict[str, object]],
    dict[str, dict[str, object]],
]:
    config = _strict_json_object(
        snapshot["config.json"],
        "existing_run_phase_snapshot_invalid",
    )
    state_rows = _strict_jsonl(
        snapshot["phase-state.jsonl"],
        "existing_run_phase_snapshot_invalid",
    )
    attempt_rows = _strict_jsonl(
        snapshot["phase-attempts.jsonl"],
        "existing_run_phase_snapshot_invalid",
    )
    if (
        len(state_rows) != 4
        or len(attempt_rows) != 4
        or [row.get("phase_id") for row in state_rows]
        != list(REQUIRED_PHASE_IDS)
        or [row.get("phase_id") for row in attempt_rows]
        != list(REQUIRED_PHASE_IDS)
    ):
        raise _blocked("existing_run_phase_snapshot_invalid")
    states: dict[str, dict[str, object]] = {}
    attempts: dict[str, dict[str, object]] = {}
    audits: dict[str, dict[str, object]] = {}
    for state, attempt, phase_id in zip(
        state_rows,
        attempt_rows,
        REQUIRED_PHASE_IDS,
        strict=True,
    ):
        expected_rows = f"phases/{phase_id}/a01/results.jsonl"
        expected_audit = f"phases/{phase_id}/a01/audit.json"
        if (
            set(state)
            != {"attempt_id", "phase_id", "row_sha256", "rows_path"}
            or set(attempt)
            != {
                "attempt_id",
                "audit_path",
                "phase_id",
                "row_sha256",
                "rows_path",
                "status",
            }
            or state.get("attempt_id") != "a01"
            or attempt.get("attempt_id") != "a01"
            or state.get("phase_id") != phase_id
            or attempt.get("phase_id") != phase_id
            or state.get("rows_path") != expected_rows
            or attempt.get("rows_path") != expected_rows
            or attempt.get("audit_path") != expected_audit
            or attempt.get("status") != "accepted"
            or state.get("row_sha256") != attempt.get("row_sha256")
            or not _is_sha256(state.get("row_sha256"))
            or _bytes_sha256(snapshot[expected_rows])
            != state.get("row_sha256")
        ):
            raise _blocked("existing_run_phase_snapshot_invalid")
        audit = _strict_json_object(
            snapshot[expected_audit],
            "existing_run_phase_snapshot_invalid",
        )
        expected_status = "complete" if phase_id in PHASE_SPLITS else "passed"
        if (
            audit.get("schema_version") != PHASE_AUDIT_SCHEMA
            or audit.get("gate_id") != G1_GATE_ID
            or audit.get("runner_id") != G1_RUNNER_ID
            or audit.get("phase_id") != phase_id
            or audit.get("phase_name") != PHASE_NAMES[phase_id]
            or audit.get("status") != expected_status
        ):
            raise _blocked("existing_run_phase_snapshot_invalid")
        if phase_id in {"p01", "p02"} and snapshot[expected_rows] != b"":
            raise _blocked("existing_run_phase_snapshot_invalid")
        if phase_id == "p01" and (
            audit.get("checkpoint_sha256") != CHECKPOINT_SHA256
            or audit.get("policy_state_sha256") != POLICY_STATE_SHA256
            or audit.get("scenario_manifest_sha256")
            != AUTHORIZED_SCENARIO_MANIFEST_SHA256
            or audit.get("input_sha256") != AUTHORIZED_INPUT_SHA256
            or audit.get("code_sha256") != config.get("code_sha256")
        ):
            raise _blocked("existing_run_phase_snapshot_invalid")
        states[phase_id] = state
        attempts[phase_id] = attempt
        audits[phase_id] = audit
    return (
        {"rows": state_rows, "sha256": _bytes_sha256(snapshot["phase-state.jsonl"])},
        attempts,
        audits,
    )


def _validate_common_row(
    row: Mapping[str, object],
    *,
    split: str,
    phase_id: str,
    config: Mapping[str, object],
) -> tuple[int, int]:
    index = _exact_int(
        row.get("episode_index"),
        "existing_run_raw_rows_invalid",
    )
    if index >= 24:
        raise _blocked("existing_run_raw_rows_invalid")
    step_index = (
        _exact_int(row.get("step_index"), "existing_run_raw_rows_invalid")
        if "step_index" in row
        else -1
    )
    if (
        row.get("gate_id") != G1_GATE_ID
        or row.get("runner_id") != G1_RUNNER_ID
        or row.get("phase_id") != phase_id
        or row.get("phase_name") != split
        or row.get("scale_profile") != SCALE_PROFILE
        or row.get("run_id") != AUTHORIZED_RUN_ID
        or row.get("split") != split
        or row.get("episode_id") != f"{split}-episode-{index:02d}"
        or row.get("lane_id") != f"lane-{index % 8}"
        or row.get("source_sha256") != config.get("code_sha256")
        or row.get("code_sha256") != config.get("code_sha256")
        or row.get("config_sha256")
        != AUTHORIZED_CONFIG_SEMANTIC_SHA256
        or row.get("input_sha256") != AUTHORIZED_INPUT_SHA256
        or row.get("scenario_manifest_sha256")
        != AUTHORIZED_SCENARIO_MANIFEST_SHA256
        or row.get("checkpoint_sha256") != CHECKPOINT_SHA256
        or row.get("policy_state_sha256") != POLICY_STATE_SHA256
        or not isinstance(row.get("scenario_id"), str)
        or not row.get("scenario_id")
    ):
        raise _blocked("existing_run_raw_rows_invalid")
    return index, step_index


def _nearest_rank(values: Sequence[float], quantile: float) -> float:
    if not values or not 0.0 <= quantile <= 1.0:
        raise _blocked("existing_run_independent_recompute_mismatch")
    index = max(1, math.ceil(quantile * len(values))) - 1
    return sorted(values)[index]


def _bootstrap_ci(values: Sequence[float]) -> dict[str, object]:
    generator = random.Random(BOOTSTRAP_SEED)
    sampled_means = sorted(
        statistics.mean(generator.choice(values) for _ in range(len(values)))
        for _ in range(BOOTSTRAP_RESAMPLES)
    )
    return {
        "unit": "episode",
        "seed": BOOTSTRAP_SEED,
        "resamples": BOOTSTRAP_RESAMPLES,
        "confidence_level": 0.95,
        "lower": _nearest_rank(sampled_means, 0.025),
        "upper": _nearest_rank(sampled_means, 0.975),
    }


def _recompute_split(
    *,
    split: str,
    phase_id: str,
    raw_rows: Sequence[dict[str, object]],
    audit: Mapping[str, object],
    config: Mapping[str, object],
    jobs: Mapping[tuple[str, int], dict[str, object]],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    coverage_rows = [
        row for row in raw_rows if row.get("row_kind") == "coverage_episode"
    ]
    decision_rows = [
        row for row in raw_rows if row.get("row_kind") == "decision"
    ]
    planner_rows = [
        row for row in raw_rows if row.get("row_kind") == "planner_call"
    ]
    if len(raw_rows) != (
        len(coverage_rows) + len(decision_rows) + len(planner_rows)
    ):
        raise _blocked("existing_run_raw_rows_invalid")
    if (
        len(coverage_rows) != 24
        or len(decision_rows) != len(planner_rows)
        or audit.get("cohort") != split
        or audit.get("coverage_episode_count") != len(coverage_rows)
        or audit.get("decision_count") != len(decision_rows)
        or audit.get("planner_call_count") != len(planner_rows)
    ):
        raise _blocked("existing_run_raw_rows_invalid")

    normalized_coverage: list[dict[str, object]] = []
    episode_steps: dict[int, int] = {}
    for row in coverage_rows:
        if set(row) != _COVERAGE_KEYS:
            raise _blocked("existing_run_raw_rows_invalid")
        index, _ = _validate_common_row(
            row,
            split=split,
            phase_id=phase_id,
            config=config,
        )
        if row.get("schema_version") != "xunce-mid-dual-g1-coverage-episode/v1":
            raise _blocked("existing_run_raw_rows_invalid")
        job = jobs.get((split, index))
        if job is None:
            raise _blocked("existing_run_raw_rows_invalid")
        denominator = _exact_int(
            row.get("denominator_cell_count"),
            "existing_run_raw_rows_invalid",
            minimum=1,
        )
        initial = _exact_int(
            row.get("initial_covered_cell_count"),
            "existing_run_raw_rows_invalid",
        )
        final = _exact_int(
            row.get("final_covered_cell_count"),
            "existing_run_raw_rows_invalid",
        )
        coverage = _finite_number(
            row.get("coverage"),
            "existing_run_raw_rows_invalid",
        )
        steps = _exact_int(
            row.get("steps_executed"),
            "existing_run_raw_rows_invalid",
        )
        _finite_number(
            row.get("elapsed_ms"),
            "existing_run_raw_rows_invalid",
        )
        if (
            row.get("scenario_id") != job["scenario_id"]
            or row.get("episode_id") != job["episode_id"]
            or row.get("lane_id") != job["lane_id"]
            or row.get("denominator_sha256") != job["denominator_sha256"]
            or denominator != job["denominator_cell_count"]
            or row.get("denominator_source") != DENOMINATOR_SOURCE
            or row.get("denominator_algorithm") != DENOMINATOR_ALGORITHM
            or not 0 <= initial <= final <= denominator
            or coverage != final / denominator
            or coverage > 1.0
            or row.get("safety_violation_count") != 0
            or row.get("masked_action_count") != 0
            or not isinstance(row.get("termination_reason"), str)
            or not row.get("termination_reason")
            or index in episode_steps
        ):
            raise _blocked("existing_run_raw_rows_invalid")
        episode_steps[index] = steps
        normalized_coverage.append(dict(row))
    if set(episode_steps) != set(range(24)):
        raise _blocked("existing_run_raw_rows_invalid")

    decisions: dict[tuple[int, int], tuple[str, str]] = {}
    planners: dict[tuple[int, int], tuple[str, str]] = {}
    decision_count_by_episode = {index: 0 for index in range(24)}
    planner_count_by_episode = {index: 0 for index in range(24)}
    for row in decision_rows:
        if set(row) != _TRACE_ROW_KEYS:
            raise _blocked("existing_run_raw_rows_invalid")
        index, step_index = _validate_common_row(
            row,
            split=split,
            phase_id=phase_id,
            config=config,
        )
        trace = row.get("trace")
        if (
            row.get("schema_version") != "xunce-mid-dual-g1-decision/v1"
            or not isinstance(trace, Mapping)
            or not _is_sha256(trace.get("join_key"))
            or not _is_sha256(trace.get("decision_sha256"))
            or (index, step_index) in decisions
            or row.get("scenario_id") != jobs[(split, index)]["scenario_id"]
        ):
            raise _blocked("existing_run_raw_rows_invalid")
        decisions[(index, step_index)] = (
            str(trace["join_key"]),
            str(trace["decision_sha256"]),
        )
        decision_count_by_episode[index] += 1
    for row in planner_rows:
        if set(row) != _TRACE_ROW_KEYS:
            raise _blocked("existing_run_raw_rows_invalid")
        index, step_index = _validate_common_row(
            row,
            split=split,
            phase_id=phase_id,
            config=config,
        )
        trace = row.get("trace")
        if (
            row.get("schema_version") != "xunce-mid-dual-g1-planner-call/v1"
            or not isinstance(trace, Mapping)
            or not _is_sha256(trace.get("join_key"))
            or not _is_sha256(trace.get("decision_sha256"))
            or trace.get("invalid_action") is not False
            or trace.get("safety_violation") is not False
            or (index, step_index) in planners
            or row.get("scenario_id") != jobs[(split, index)]["scenario_id"]
        ):
            raise _blocked("existing_run_raw_rows_invalid")
        planners[(index, step_index)] = (
            str(trace["join_key"]),
            str(trace["decision_sha256"]),
        )
        planner_count_by_episode[index] += 1
    if decisions != planners:
        raise _blocked("existing_run_raw_rows_invalid")
    for index in range(24):
        if (
            episode_steps[index] != decision_count_by_episode[index]
            or episode_steps[index] != planner_count_by_episode[index]
        ):
            raise _blocked("existing_run_raw_rows_invalid")

    ordered = sorted(normalized_coverage, key=lambda row: int(row["episode_index"]))
    lane_counts: dict[str, int] = {}
    for row in ordered:
        lane = str(row["lane_id"])
        lane_counts[lane] = lane_counts.get(lane, 0) + 1
    if lane_counts != EXPECTED_LANE_COUNTS:
        raise _blocked("existing_run_raw_rows_invalid")
    coverages = [float(row["coverage"]) for row in ordered]
    coverage_80_count = sum(value >= MIDTERM_THRESHOLD for value in coverages)
    coverage_99_count = sum(value >= FINAL_THRESHOLD for value in coverages)
    mean_value = statistics.mean(coverages)
    metrics = {
        "sample_count": 24,
        "mean": mean_value,
        "median": statistics.median(coverages),
        "sample_stddev": statistics.stdev(coverages),
        "min": min(coverages),
        "max": max(coverages),
        "coverage_80_count": coverage_80_count,
        "coverage_99_count": coverage_99_count,
        "bootstrap_ci": _bootstrap_ci(coverages),
        "g1_coverage_80_passed": (
            mean_value >= MIDTERM_THRESHOLD
            and coverage_80_count >= MINIMUM_PASS_COUNT
        ),
        "g1_coverage_99_passed": (
            mean_value >= FINAL_THRESHOLD
            and coverage_99_count >= MINIMUM_PASS_COUNT
        ),
        "safety_violation_count": 0,
        "masked_action_count": 0,
        "integer_denominator_check": "passed",
        "lane_counts": lane_counts,
        "row_kind_counts": {
            "coverage_episode": len(coverage_rows),
            "decision": len(decision_rows),
            "planner_call": len(planner_rows),
        },
        "phase_audit_equality": True,
    }
    return metrics, ordered


def recompute_existing_run_numerics_from_snapshot(
    snapshot: Mapping[str, bytes],
) -> dict[str, object]:
    """Independently recompute the fixed p03+p04 numerical result using stdlib."""

    required = {
        "config.json",
        "phase-state.jsonl",
        "phase-attempts.jsonl",
        "external/scenario-manifest.json",
        *[
            f"phases/{phase_id}/a01/{name}"
            for phase_id in REQUIRED_PHASE_IDS
            for name in ("audit.json", "results.jsonl")
        ],
    }
    if set(snapshot) != required or any(
        type(value) is not bytes for value in snapshot.values()
    ):
        raise _blocked("existing_run_phase_snapshot_invalid")
    config = _strict_json_object(
        snapshot["config.json"],
        "existing_run_contract_invalid",
    )
    source_root_text = config.get("output_root")
    if not isinstance(source_root_text, str):
        raise _blocked("existing_run_contract_invalid")
    _validate_config(config, source_root_text=source_root_text)
    manifest_mapping = config.get("scenario_manifest")
    assert isinstance(manifest_mapping, Mapping)
    manifest_path_value = manifest_mapping.get("path")
    if not isinstance(manifest_path_value, str):
        raise _blocked("existing_run_input_manifest_invalid")
    manifest_path = Path(manifest_path_value)
    input_audit, jobs = _validate_manifest_and_reconstruct_input(
        manifest_path=manifest_path,
        manifest_bytes=snapshot["external/scenario-manifest.json"],
        config=config,
    )
    phase_state, _, audits = _validate_phase_prefix(snapshot)
    split_metrics: dict[str, dict[str, object]] = {}
    coverage_projection: list[dict[str, object]] = []
    for phase_id, split in PHASE_SPLITS.items():
        raw_rows = _strict_jsonl(
            snapshot[f"phases/{phase_id}/a01/results.jsonl"],
            "existing_run_raw_rows_invalid",
        )
        metrics, coverage_rows = _recompute_split(
            split=split,
            phase_id=phase_id,
            raw_rows=raw_rows,
            audit=audits[phase_id],
            config=config,
            jobs=jobs,
        )
        split_metrics[split] = metrics
        for row in coverage_rows:
            coverage_projection.append(
                {
                    key: row[key]
                    for key in (
                        "split",
                        "episode_id",
                        "episode_index",
                        "scenario_id",
                        "lane_id",
                        "denominator_sha256",
                        "denominator_cell_count",
                        "initial_covered_cell_count",
                        "final_covered_cell_count",
                        "coverage",
                        "steps_executed",
                        "termination_reason",
                        "safety_violation_count",
                        "masked_action_count",
                    )
                }
            )
    scenario_ids = [
        str(row["scenario_id"]) for row in coverage_projection
    ]
    episode_ids = [str(row["episode_id"]) for row in coverage_projection]
    if len(set(scenario_ids)) != 48 or len(set(episode_ids)) != 48:
        raise _blocked("existing_run_raw_rows_invalid")
    midterm_passed = all(
        split_metrics[split]["g1_coverage_80_passed"] is True
        for split in FORMAL_SPLITS
    )
    final_passed = all(
        split_metrics[split]["g1_coverage_99_passed"] is True
        for split in FORMAL_SPLITS
    )
    return {
        "schema_version": (
            "xunce-mid-dual-g1-existing-run-independent-recompute/v1"
        ),
        "formal_episode_count": 48,
        "split_order": list(FORMAL_SPLITS),
        "splits": split_metrics,
        "coverage_projection_sha256": _canonical_sha256(coverage_projection),
        "phase_state_sha256": phase_state["sha256"],
        "input_audit_canonical_sha256": _canonical_sha256(input_audit),
        "scenario_manifest_sha256": _bytes_sha256(
            snapshot["external/scenario-manifest.json"]
        ),
        "max_traversable_slope_deg": 30.0,
        "safety_clean": True,
        "masked_action_clean": True,
        "midterm_reduced_passed": midterm_passed,
        "final_threshold_reduced_passed": final_passed,
        "review_result_status": "passed" if midterm_passed else "failed",
        "replay_required": False,
        "replay_status": "not_run_by_ruling",
    }


def _collect_source_snapshot(
    source_root: Path,
) -> tuple[Path, dict[str, bytes], dict[str, object]]:
    root = _canonical_existing_root(
        source_root,
        expected=AUTHORIZED_SOURCE_ROOT,
        reason="existing_run_scope_mismatch",
    )
    if root.name != AUTHORIZED_RUN_ID:
        raise _blocked("existing_run_scope_mismatch")
    source_snapshot = _phase_snapshot_files(root)
    config = _strict_json_object(
        source_snapshot["config.json"],
        "existing_run_contract_invalid",
    )
    _validate_config(config, source_root_text=_path_text(root))
    scenario_mapping = config.get("scenario_manifest")
    assert isinstance(scenario_mapping, Mapping)
    scenario_path_text = scenario_mapping.get("path")
    if not isinstance(scenario_path_text, str):
        raise _blocked("existing_run_input_manifest_invalid")
    scenario_path = _canonical_existing_file(
        Path(scenario_path_text),
        "existing_run_input_manifest_invalid",
    )
    if _path_text(scenario_path) != scenario_path_text:
        raise _blocked("existing_run_input_manifest_invalid")
    manifest_bytes = _stable_read(
        scenario_path,
        "existing_run_input_manifest_invalid",
    )
    source_snapshot["external/scenario-manifest.json"] = manifest_bytes
    input_audit, _ = _validate_manifest_and_reconstruct_input(
        manifest_path=scenario_path,
        manifest_bytes=manifest_bytes,
        config=config,
    )
    _validate_phase_prefix(source_snapshot)
    metrics = recompute_existing_run_numerics_from_snapshot(source_snapshot)
    if metrics["phase_state_sha256"] != AUTHORIZED_PHASE_STATE_SHA256:
        raise _blocked("existing_run_source_hash_mismatch")
    return root, source_snapshot, {
        "config": config,
        "input_audit": input_audit,
        "scenario_manifest_path": scenario_path,
        "metrics": metrics,
    }


def _validate_stop_evidence(
    payload: bytes,
    *,
    source_root_text: str,
    source_snapshot: Mapping[str, bytes],
) -> dict[str, object]:
    evidence = _strict_json_object(
        payload,
        "existing_run_stop_evidence_invalid",
    )
    exact_keys = {
        "schema_version",
        "run_id",
        "source_root",
        "phase_state_sha256",
        "p04_results_sha256",
        "p04_audit_sha256",
        "p04_accepted_at",
        "observed_at",
        "parent_pid",
        "observed_worker_pids",
        "process_query_raw_bytes_hex",
        "process_query_raw_sha256",
        "matching_parent_process_count",
        "matching_worker_process_count",
        "p05_phase_present",
    }
    workers = evidence.get("observed_worker_pids")
    raw_hex = evidence.get("process_query_raw_bytes_hex")
    try:
        raw_query = bytes.fromhex(raw_hex) if isinstance(raw_hex, str) else b""
        accepted_at = datetime.fromisoformat(str(evidence.get("p04_accepted_at")))
        observed_at = datetime.fromisoformat(str(evidence.get("observed_at")))
    except (ValueError, TypeError) as exc:
        raise _blocked("existing_run_stop_evidence_invalid") from exc
    if (
        set(evidence) != exact_keys
        or evidence.get("schema_version") != STOP_EVIDENCE_SCHEMA
        or evidence.get("run_id") != AUTHORIZED_RUN_ID
        or evidence.get("source_root") != source_root_text
        or evidence.get("phase_state_sha256")
        != _bytes_sha256(source_snapshot["phase-state.jsonl"])
        or evidence.get("p04_results_sha256")
        != _bytes_sha256(
            source_snapshot["phases/p04/a01/results.jsonl"]
        )
        or evidence.get("p04_audit_sha256")
        != _bytes_sha256(source_snapshot["phases/p04/a01/audit.json"])
        or evidence.get("parent_pid") != AUTHORIZED_PARENT_PID
        or not isinstance(workers, list)
        or tuple(workers) != AUTHORIZED_WORKER_PIDS
        or len(set(workers)) != 8
        or not raw_query
        or evidence.get("process_query_raw_sha256")
        != _bytes_sha256(raw_query)
        or evidence.get("matching_parent_process_count") != 0
        or evidence.get("matching_worker_process_count") != 0
        or evidence.get("p05_phase_present") is not False
        or accepted_at.tzinfo is None
        or observed_at.tzinfo is None
        or observed_at < accepted_at
    ):
        raise _blocked("existing_run_stop_evidence_invalid")
    return evidence


def _derived_input_evidence(
    *,
    input_audit: Mapping[str, object],
    scenario_manifest_path: Path,
    scenario_manifest_bytes: bytes,
) -> dict[str, object]:
    return {
        "schema_version": DERIVED_INPUT_SCHEMA,
        "evidence_kind": (
            "reconstructed_from_frozen_scenario_manifest/v1"
        ),
        "source_root_artifact_status": "absent_expected",
        "source_root_artifact_path": "g1_input_audit.json",
        "scenario_manifest": {
            "source_path": _path_text(scenario_manifest_path),
            "copy_path": "snapshot/scenario-manifest.json",
            "sha256": _bytes_sha256(scenario_manifest_bytes),
        },
        "reconstructed_input_audit_sha256": _canonical_sha256(input_audit),
        "reconstructed_input_audit": dict(input_audit),
    }


def _snapshot_index(
    *,
    source_root: Path,
    source_snapshot: Mapping[str, bytes],
    scenario_manifest_path: Path,
    input_evidence: Mapping[str, object],
) -> dict[str, object]:
    artifacts: list[dict[str, object]] = []
    for relative in sorted(
        key for key in source_snapshot if key != "external/scenario-manifest.json"
    ):
        payload = source_snapshot[relative]
        artifacts.append(
            {
                "source_path": f"{_path_text(source_root)}/{relative}",
                "source_relative_path": relative,
                "copy_path": f"snapshot/{relative}",
                "sha256": _bytes_sha256(payload),
                "size_bytes": len(payload),
            }
        )
    manifest_bytes = source_snapshot["external/scenario-manifest.json"]
    artifacts.append(
        {
            "source_path": _path_text(scenario_manifest_path),
            "source_relative_path": None,
            "copy_path": "snapshot/scenario-manifest.json",
            "sha256": _bytes_sha256(manifest_bytes),
            "size_bytes": len(manifest_bytes),
        }
    )
    return {
        "schema_version": SOURCE_SNAPSHOT_SCHEMA,
        "source_run_id": AUTHORIZED_RUN_ID,
        "source_root": _path_text(source_root),
        "required_phase_ids": list(REQUIRED_PHASE_IDS),
        "phase_state_sha256": _bytes_sha256(
            source_snapshot["phase-state.jsonl"]
        ),
        "phase_attempts_sha256": _bytes_sha256(
            source_snapshot["phase-attempts.jsonl"]
        ),
        "artifacts": artifacts,
        "reconstructed_input_audit": dict(input_evidence),
    }


def _review_payload(
    *,
    source_root: Path,
    source_snapshot: Mapping[str, bytes],
    snapshot_index: Mapping[str, object],
    metrics: Mapping[str, object],
    verifier_source_sha256: str,
) -> dict[str, object]:
    return {
        "schema_version": REVIEW_SCHEMA,
        "source_run": {
            "run_id": AUTHORIZED_RUN_ID,
            "root": _path_text(source_root),
            "config_file_sha256": _bytes_sha256(
                source_snapshot["config.json"]
            ),
            "config_semantic_sha256": AUTHORIZED_CONFIG_SEMANTIC_SHA256,
            "phase_state_sha256": _bytes_sha256(
                source_snapshot["phase-state.jsonl"]
            ),
            "attempt_index_sha256": _bytes_sha256(
                source_snapshot["phase-attempts.jsonl"]
            ),
        },
        "phase_snapshot_sha256": _canonical_sha256(snapshot_index),
        "input_hashes": {
            "input_audit_canonical_sha256": AUTHORIZED_INPUT_SHA256,
            "scenario_manifest_sha256": (
                AUTHORIZED_SCENARIO_MANIFEST_SHA256
            ),
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "policy_state_sha256": POLICY_STATE_SHA256,
        },
        "verifier_source_sha256": verifier_source_sha256,
        "review_result_status": metrics["review_result_status"],
        "coverage_projection_sha256": metrics[
            "coverage_projection_sha256"
        ],
        "metrics": dict(metrics),
        "integer_denominator_check": "passed",
        "slope_invariant_check": {
            "status": "passed",
            "max_traversable_slope_deg": 30.0,
            "proof_count": EXPECTED_DENOMINATOR_PROOF_COUNT,
        },
        "safety_check": {"status": "passed", "violation_count": 0},
        "masked_action_check": {"status": "passed", "masked_action_count": 0},
        "phase_audit_equality": {"p03": True, "p04": True},
    }


def _manifest_entries(files: Mapping[str, bytes]) -> list[dict[str, object]]:
    return [
        {
            "path": relative,
            "sha256": _bytes_sha256(payload),
            "size_bytes": len(payload),
        }
        for relative, payload in sorted(files.items())
    ]


def _assessment_envelope(
    *,
    source_root: Path,
    preservation: Mapping[str, object],
    review_root: Path,
    review_manifest_sha256: str,
    review_result_sha256: str,
    stop_evidence_sha256: str,
    verifier_source_sha256: str,
    source_snapshot: Mapping[str, bytes],
    metrics: Mapping[str, object],
) -> dict[str, object]:
    audit_hashes = {
        phase_id: _bytes_sha256(
            source_snapshot[f"phases/{phase_id}/a01/audit.json"]
        )
        for phase_id in REQUIRED_PHASE_IDS
    }
    results_hashes = {
        phase_id: _bytes_sha256(
            source_snapshot[f"phases/{phase_id}/a01/results.jsonl"]
        )
        for phase_id in REQUIRED_PHASE_IDS
    }
    return {
        "schema_version": ASSESSMENT_SCHEMA,
        "assessment_id": "g1-existing-run-assessment-use/v1",
        "scope": {
            "run_id": AUTHORIZED_RUN_ID,
            "source_root": _path_text(source_root),
        },
        "assessment_use_status": "authorized_existing_run",
        "strict_prestart_lineage_status": "not_satisfied",
        "lineage_limitation_acknowledged": True,
        "numerical_gate_status": "recomputed_from_raw_rows",
        "numerical_result_status": metrics["review_result_status"],
        "authorities": [
            {"path": path, "sha256": sha256}
            for path, sha256 in AUTHORITY_ALLOWLIST.items()
        ],
        "source_run": {
            "run_id": AUTHORIZED_RUN_ID,
            "root": _path_text(source_root),
            "config_sha256": _bytes_sha256(source_snapshot["config.json"]),
            "config_semantic_sha256": AUTHORIZED_CONFIG_SEMANTIC_SHA256,
            "phase_state_sha256": _bytes_sha256(
                source_snapshot["phase-state.jsonl"]
            ),
            "attempt_index_sha256": _bytes_sha256(
                source_snapshot["phase-attempts.jsonl"]
            ),
            "input_audit_sha256": AUTHORIZED_INPUT_SHA256,
            "scenario_manifest_sha256": (
                AUTHORIZED_SCENARIO_MANIFEST_SHA256
            ),
            "required_phase_ids": list(REQUIRED_PHASE_IDS),
            "native_final_manifest_status": "not_expected",
            "p05_replay_status": "not_run_by_ruling",
            "p06_p07_finalization_status": "not_run_by_ruling",
        },
        "preservation": dict(preservation),
        "independent_recompute": {
            "review_root": _path_text(review_root),
            "review_manifest_sha256": review_manifest_sha256,
            "review_result_sha256": review_result_sha256,
            "review_result_status": metrics["review_result_status"],
            "verifier_source_sha256": verifier_source_sha256,
            "stop_evidence_sha256": stop_evidence_sha256,
            "input_sha256": AUTHORIZED_INPUT_SHA256,
            "phase_state_sha256": _bytes_sha256(
                source_snapshot["phase-state.jsonl"]
            ),
            "attempt_index_sha256": _bytes_sha256(
                source_snapshot["phase-attempts.jsonl"]
            ),
            "phase_audit_sha256": audit_hashes,
            "phase_results_sha256": results_hashes,
            "coverage_projection_sha256": metrics[
                "coverage_projection_sha256"
            ],
        },
        "generator_source_sha256": verifier_source_sha256,
    }


def create_existing_run_assessment(
    *,
    source_root: Path,
    preservation_root: Path,
    stop_evidence_path: Path,
    review_root: Path,
    envelope_root: Path,
) -> Path:
    """Create the one authorized p01-p04 review and assessment-use envelope."""

    source, snapshot, collected = _collect_source_snapshot(source_root)
    phase_state_sha256 = _bytes_sha256(snapshot["phase-state.jsonl"])
    expected_review = _path_text(
        Path(AUTHORIZED_REVIEW_BASE_ROOT)
        / f"g1-existing-run-review-v1-{phase_state_sha256}"
    )
    expected_envelope = _path_text(
        Path(AUTHORIZED_ENVELOPE_BASE_ROOT)
        / f"g1-existing-run-assessment-v1-{phase_state_sha256}"
    )
    review = _canonical_new_root(review_root, expected=expected_review)
    envelope = _canonical_new_root(envelope_root, expected=expected_envelope)
    _verify_authorities()
    preservation = _verify_preservation(
        preservation_root,
        source_root_text=_path_text(source),
    )
    stop_path = _canonical_existing_file(
        stop_evidence_path,
        "existing_run_stop_evidence_invalid",
    )
    stop_bytes = _stable_read(
        stop_path,
        "existing_run_stop_evidence_invalid",
    )
    _validate_stop_evidence(
        stop_bytes,
        source_root_text=_path_text(source),
        source_snapshot=snapshot,
    )
    verifier_source_bytes = _stable_read(
        Path(__file__),
        "existing_run_verifier_source_invalid",
    )
    verifier_source_sha256 = _bytes_sha256(verifier_source_bytes)
    input_audit = collected["input_audit"]
    scenario_manifest_path = collected["scenario_manifest_path"]
    metrics = collected["metrics"]
    assert isinstance(input_audit, Mapping)
    assert isinstance(scenario_manifest_path, Path)
    assert isinstance(metrics, Mapping)
    input_evidence = _derived_input_evidence(
        input_audit=input_audit,
        scenario_manifest_path=scenario_manifest_path,
        scenario_manifest_bytes=snapshot["external/scenario-manifest.json"],
    )
    snapshot_index = _snapshot_index(
        source_root=source,
        source_snapshot=snapshot,
        scenario_manifest_path=scenario_manifest_path,
        input_evidence=input_evidence,
    )
    review_payload = _review_payload(
        source_root=source,
        source_snapshot=snapshot,
        snapshot_index=snapshot_index,
        metrics=metrics,
        verifier_source_sha256=verifier_source_sha256,
    )
    derived_input_bytes = _json_bytes(input_evidence)
    review_files: dict[str, bytes] = {
        "review.json": _json_bytes(review_payload),
        "source-phase-snapshot.json": _json_bytes(snapshot_index),
        "stop-evidence.json": stop_bytes,
        "snapshot/reconstructed-g1-input-audit.json": derived_input_bytes,
    }
    for relative, payload in snapshot.items():
        copy_relative = (
            "snapshot/scenario-manifest.json"
            if relative == "external/scenario-manifest.json"
            else f"snapshot/{relative}"
        )
        review_files[copy_relative] = payload
    review_manifest = {
        "schema_version": REVIEW_MANIFEST_SCHEMA,
        "source_run_id": AUTHORIZED_RUN_ID,
        "source_phase_state_sha256": phase_state_sha256,
        "verifier_source": {
            "path": GENERATOR_SOURCE_RELATIVE_PATH,
            "sha256": verifier_source_sha256,
        },
        "files": _manifest_entries(review_files),
    }
    review_manifest_bytes = _json_bytes(review_manifest)
    review_manifest_sha256 = _bytes_sha256(review_manifest_bytes)
    review_result_sha256 = _bytes_sha256(review_files["review.json"])

    envelope_payload = _assessment_envelope(
        source_root=source,
        preservation=preservation,
        review_root=review,
        review_manifest_sha256=review_manifest_sha256,
        review_result_sha256=review_result_sha256,
        stop_evidence_sha256=_bytes_sha256(stop_bytes),
        verifier_source_sha256=verifier_source_sha256,
        source_snapshot=snapshot,
        metrics=metrics,
    )
    envelope_bytes = _json_bytes(envelope_payload)
    assessment_manifest_without_self = {
        "schema_version": ASSESSMENT_MANIFEST_SCHEMA,
        "source_run_id": AUTHORIZED_RUN_ID,
        "files": _manifest_entries(
            {"assessment-use-envelope.json": envelope_bytes}
        ),
        "external_bindings": {
            "review_root": _path_text(review),
            "review_manifest_sha256": review_manifest_sha256,
            "review_result_sha256": review_result_sha256,
            "verifier_source_sha256": verifier_source_sha256,
        },
    }
    assessment_manifest = {
        **assessment_manifest_without_self,
        "canonical_sha256": _canonical_sha256(
            assessment_manifest_without_self
        ),
    }
    assessment_manifest_bytes = _json_bytes(assessment_manifest)

    try:
        review.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise _blocked("existing_run_output_exists") from exc
    except OSError as exc:
        raise _blocked("existing_run_output_write_failed") from exc
    for relative, payload in sorted(review_files.items()):
        _write_exclusive(review / PurePosixPath(relative), payload)
    _write_exclusive(review / "review-manifest.json", review_manifest_bytes)
    _write_exclusive(
        review / "review-manifest.sha256",
        (review_manifest_sha256 + "\n").encode("ascii"),
    )

    try:
        envelope.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise _blocked("existing_run_output_exists") from exc
    except OSError as exc:
        raise _blocked("existing_run_output_write_failed") from exc
    _write_exclusive(
        envelope / "assessment-use-envelope.json",
        envelope_bytes,
    )
    _write_exclusive(
        envelope / "assessment-use-manifest.json",
        assessment_manifest_bytes,
    )
    _write_exclusive(
        envelope / "assessment-use-manifest.sha256",
        (
            _bytes_sha256(assessment_manifest_bytes) + "\n"
        ).encode("ascii"),
    )
    return envelope


def _read_manifest_bound_files(
    *,
    root: Path,
    manifest: Mapping[str, object],
    manifest_name: str,
    sidecar_name: str,
    reason: str,
) -> dict[str, bytes]:
    entries = manifest.get("files")
    if not isinstance(entries, list):
        raise _blocked(reason)
    expected_files = {manifest_name, sidecar_name}
    payloads: dict[str, bytes] = {}
    for entry in entries:
        if (
            not isinstance(entry, Mapping)
            or set(entry) != {"path", "sha256", "size_bytes"}
        ):
            raise _blocked(reason)
        relative = _safe_relative_path(entry.get("path"), reason)
        if relative in payloads:
            raise _blocked(reason)
        payload = _stable_read(root / PurePosixPath(relative), reason)
        if (
            entry.get("sha256") != _bytes_sha256(payload)
            or entry.get("size_bytes") != len(payload)
        ):
            raise _blocked(reason)
        payloads[relative] = payload
        expected_files.add(relative)
    if _directory_file_set(root) != expected_files:
        raise _blocked(reason)
    return payloads


def _verify_review(
    *,
    review_root: Path,
    expected_source_root: Path,
    verifier_source_sha256: str,
) -> tuple[
    dict[str, object],
    dict[str, object],
    dict[str, bytes],
    str,
    str,
    str,
]:
    expected_review_text = _path_text(
        Path(AUTHORIZED_REVIEW_BASE_ROOT)
        / (
            "g1-existing-run-review-v1-"
            f"{AUTHORIZED_PHASE_STATE_SHA256}"
        )
    )
    review = _canonical_existing_root(
        review_root,
        expected=expected_review_text,
        reason="existing_run_review_manifest_invalid",
    )
    manifest_bytes = _stable_read(
        review / "review-manifest.json",
        "existing_run_review_manifest_invalid",
    )
    sidecar = _stable_read(
        review / "review-manifest.sha256",
        "existing_run_review_manifest_invalid",
    )
    try:
        sidecar_value = sidecar.decode("ascii").strip()
    except UnicodeDecodeError as exc:
        raise _blocked("existing_run_review_manifest_invalid") from exc
    if sidecar_value != _bytes_sha256(manifest_bytes):
        raise _blocked("existing_run_review_manifest_invalid")
    manifest = _strict_json_object(
        manifest_bytes,
        "existing_run_review_manifest_invalid",
    )
    if (
        manifest.get("schema_version") != REVIEW_MANIFEST_SCHEMA
        or manifest.get("source_run_id") != AUTHORIZED_RUN_ID
        or manifest.get("source_phase_state_sha256")
        != AUTHORIZED_PHASE_STATE_SHA256
        or manifest.get("verifier_source")
        != {
            "path": GENERATOR_SOURCE_RELATIVE_PATH,
            "sha256": verifier_source_sha256,
        }
    ):
        raise _blocked("existing_run_review_manifest_invalid")
    payloads = _read_manifest_bound_files(
        root=review,
        manifest=manifest,
        manifest_name="review-manifest.json",
        sidecar_name="review-manifest.sha256",
        reason="existing_run_review_manifest_invalid",
    )
    review_payload = _strict_json_object(
        payloads.get("review.json", b""),
        "existing_run_review_manifest_invalid",
    )
    snapshot_index = _strict_json_object(
        payloads.get("source-phase-snapshot.json", b""),
        "existing_run_review_manifest_invalid",
    )
    if (
        snapshot_index.get("schema_version") != SOURCE_SNAPSHOT_SCHEMA
        or snapshot_index.get("source_run_id") != AUTHORIZED_RUN_ID
        or snapshot_index.get("source_root")
        != _path_text(expected_source_root)
        or snapshot_index.get("required_phase_ids")
        != list(REQUIRED_PHASE_IDS)
        or snapshot_index.get("phase_state_sha256")
        != AUTHORIZED_PHASE_STATE_SHA256
    ):
        raise _blocked("existing_run_review_manifest_invalid")
    source_snapshot: dict[str, bytes] = {}
    artifacts = snapshot_index.get("artifacts")
    if not isinstance(artifacts, list):
        raise _blocked("existing_run_review_manifest_invalid")
    for entry in artifacts:
        if not isinstance(entry, Mapping):
            raise _blocked("existing_run_review_manifest_invalid")
        copy_path = _safe_relative_path(
            entry.get("copy_path"),
            "existing_run_review_manifest_invalid",
        )
        payload = payloads.get(copy_path)
        if (
            payload is None
            or entry.get("sha256") != _bytes_sha256(payload)
            or entry.get("size_bytes") != len(payload)
        ):
            raise _blocked("existing_run_review_manifest_invalid")
        source_relative = entry.get("source_relative_path")
        if source_relative is None:
            if copy_path != "snapshot/scenario-manifest.json":
                raise _blocked("existing_run_review_manifest_invalid")
            source_snapshot["external/scenario-manifest.json"] = payload
        else:
            relative = _safe_relative_path(
                source_relative,
                "existing_run_review_manifest_invalid",
            )
            if copy_path != f"snapshot/{relative}":
                raise _blocked("existing_run_review_manifest_invalid")
            source_snapshot[relative] = payload
    derived_input = _strict_json_object(
        payloads.get(
            "snapshot/reconstructed-g1-input-audit.json",
            b"",
        ),
        "existing_run_review_manifest_invalid",
    )
    indexed_derived = snapshot_index.get("reconstructed_input_audit")
    if (
        derived_input != indexed_derived
        or derived_input.get("schema_version") != DERIVED_INPUT_SCHEMA
        or derived_input.get("evidence_kind")
        != "reconstructed_from_frozen_scenario_manifest/v1"
        or derived_input.get("source_root_artifact_status")
        != "absent_expected"
        or derived_input.get("source_root_artifact_path")
        != "g1_input_audit.json"
    ):
        raise _blocked("existing_run_review_manifest_invalid")
    metrics = recompute_existing_run_numerics_from_snapshot(source_snapshot)
    expected_review = _review_payload(
        source_root=expected_source_root,
        source_snapshot=source_snapshot,
        snapshot_index=snapshot_index,
        metrics=metrics,
        verifier_source_sha256=verifier_source_sha256,
    )
    if review_payload != expected_review:
        raise _blocked("existing_run_independent_recompute_mismatch")
    stop_bytes = payloads.get("stop-evidence.json")
    if stop_bytes is None:
        raise _blocked("existing_run_review_manifest_invalid")
    _validate_stop_evidence(
        stop_bytes,
        source_root_text=_path_text(expected_source_root),
        source_snapshot=source_snapshot,
    )
    return (
        review_payload,
        snapshot_index,
        source_snapshot,
        _bytes_sha256(manifest_bytes),
        _bytes_sha256(payloads["review.json"]),
        _bytes_sha256(stop_bytes),
    )


def verify_existing_run_assessment(
    *,
    envelope_root: Path,
    expected_source_root: Path,
) -> VerifiedExistingRunAssessment:
    """Verify the exact one-run envelope and re-run the stdlib recomputation."""

    source, live_snapshot, collected = _collect_source_snapshot(
        expected_source_root
    )
    verifier_source_bytes = _stable_read(
        Path(__file__),
        "existing_run_verifier_source_invalid",
    )
    verifier_source_sha256 = _bytes_sha256(verifier_source_bytes)
    expected_envelope_text = _path_text(
        Path(AUTHORIZED_ENVELOPE_BASE_ROOT)
        / (
            "g1-existing-run-assessment-v1-"
            f"{AUTHORIZED_PHASE_STATE_SHA256}"
        )
    )
    envelope = _canonical_existing_root(
        envelope_root,
        expected=expected_envelope_text,
        reason="existing_run_assessment_manifest_invalid",
    )
    expected_envelope_files = {
        "assessment-use-envelope.json",
        "assessment-use-manifest.json",
        "assessment-use-manifest.sha256",
    }
    if _directory_file_set(envelope) != expected_envelope_files:
        raise _blocked("existing_run_assessment_manifest_invalid")
    envelope_bytes = _stable_read(
        envelope / "assessment-use-envelope.json",
        "existing_run_assessment_manifest_invalid",
    )
    manifest_bytes = _stable_read(
        envelope / "assessment-use-manifest.json",
        "existing_run_assessment_manifest_invalid",
    )
    sidecar = _stable_read(
        envelope / "assessment-use-manifest.sha256",
        "existing_run_assessment_manifest_invalid",
    )
    try:
        sidecar_value = sidecar.decode("ascii").strip()
    except UnicodeDecodeError as exc:
        raise _blocked("existing_run_assessment_manifest_invalid") from exc
    if sidecar_value != _bytes_sha256(manifest_bytes):
        raise _blocked("existing_run_assessment_manifest_invalid")
    manifest = _strict_json_object(
        manifest_bytes,
        "existing_run_assessment_manifest_invalid",
    )
    canonical_sha256 = manifest.get("canonical_sha256")
    manifest_without_self = dict(manifest)
    manifest_without_self.pop("canonical_sha256", None)
    if (
        manifest.get("schema_version") != ASSESSMENT_MANIFEST_SCHEMA
        or manifest.get("source_run_id") != AUTHORIZED_RUN_ID
        or canonical_sha256 != _canonical_sha256(manifest_without_self)
        or manifest.get("files")
        != _manifest_entries(
            {"assessment-use-envelope.json": envelope_bytes}
        )
    ):
        raise _blocked("existing_run_assessment_manifest_invalid")
    envelope_payload = _strict_json_object(
        envelope_bytes,
        "existing_run_assessment_manifest_invalid",
    )
    scope = envelope_payload.get("scope")
    source_run = envelope_payload.get("source_run")
    independent = envelope_payload.get("independent_recompute")
    external = manifest.get("external_bindings")
    if not all(
        isinstance(value, Mapping)
        for value in (scope, source_run, independent, external)
    ):
        raise _blocked("existing_run_assessment_manifest_invalid")
    assert isinstance(scope, Mapping)
    assert isinstance(source_run, Mapping)
    assert isinstance(independent, Mapping)
    assert isinstance(external, Mapping)
    authorities_expected = [
        {"path": path, "sha256": sha256}
        for path, sha256 in AUTHORITY_ALLOWLIST.items()
    ]
    if (
        set(envelope_payload)
        != {
            "schema_version",
            "assessment_id",
            "scope",
            "assessment_use_status",
            "strict_prestart_lineage_status",
            "lineage_limitation_acknowledged",
            "numerical_gate_status",
            "numerical_result_status",
            "authorities",
            "source_run",
            "preservation",
            "independent_recompute",
            "generator_source_sha256",
        }
        or envelope_payload.get("schema_version") != ASSESSMENT_SCHEMA
        or envelope_payload.get("assessment_id")
        != "g1-existing-run-assessment-use/v1"
        or scope.get("run_id") != AUTHORIZED_RUN_ID
        or scope.get("source_root") != _path_text(source)
        or envelope_payload.get("assessment_use_status")
        != "authorized_existing_run"
        or envelope_payload.get("strict_prestart_lineage_status")
        != "not_satisfied"
        or envelope_payload.get("lineage_limitation_acknowledged") is not True
        or envelope_payload.get("numerical_gate_status")
        != "recomputed_from_raw_rows"
        or envelope_payload.get("numerical_result_status")
        not in {"passed", "failed"}
        or envelope_payload.get("authorities") != authorities_expected
        or envelope_payload.get("generator_source_sha256")
        != verifier_source_sha256
        or source_run.get("required_phase_ids") != list(REQUIRED_PHASE_IDS)
        or source_run.get("native_final_manifest_status") != "not_expected"
        or source_run.get("p05_replay_status") != "not_run_by_ruling"
        or source_run.get("p06_p07_finalization_status")
        != "not_run_by_ruling"
    ):
        raise _blocked("existing_run_assessment_manifest_invalid")
    preservation = _verify_preservation(
        Path(AUTHORIZED_PRESERVATION_ROOT),
        source_root_text=_path_text(source),
    )
    if envelope_payload.get("preservation") != preservation:
        raise _blocked("existing_run_preservation_invalid")
    review_root_value = independent.get("review_root")
    if (
        not isinstance(review_root_value, str)
        or external.get("review_root") != review_root_value
    ):
        raise _blocked("existing_run_assessment_manifest_invalid")
    expected_review_text = _path_text(
        Path(AUTHORIZED_REVIEW_BASE_ROOT)
        / (
            "g1-existing-run-review-v1-"
            f"{AUTHORIZED_PHASE_STATE_SHA256}"
        )
    )
    if review_root_value != expected_review_text:
        raise _blocked("existing_run_assessment_manifest_invalid")
    review_root = Path(review_root_value)
    (
        review_payload,
        snapshot_index,
        reviewed_snapshot,
        review_manifest_sha256,
        review_result_sha256,
        stop_evidence_sha256,
    ) = _verify_review(
        review_root=review_root,
        expected_source_root=source,
        verifier_source_sha256=verifier_source_sha256,
    )
    if set(reviewed_snapshot) != set(live_snapshot):
        raise _blocked("existing_run_source_hash_mismatch")
    for relative, payload in reviewed_snapshot.items():
        if live_snapshot.get(relative) != payload:
            raise _blocked("existing_run_source_hash_mismatch")
    live_metrics = collected["metrics"]
    assert isinstance(live_metrics, Mapping)
    if review_payload.get("metrics") != dict(live_metrics):
        raise _blocked("existing_run_independent_recompute_mismatch")
    if (
        independent.get("review_manifest_sha256")
        != review_manifest_sha256
        or independent.get("review_result_sha256")
        != review_result_sha256
        or independent.get("stop_evidence_sha256")
        != stop_evidence_sha256
        or independent.get("verifier_source_sha256")
        != verifier_source_sha256
        or independent.get("review_result_status")
        != review_payload.get("review_result_status")
        or independent.get("coverage_projection_sha256")
        != review_payload.get("coverage_projection_sha256")
        or external.get("review_manifest_sha256")
        != review_manifest_sha256
        or external.get("review_result_sha256") != review_result_sha256
        or external.get("verifier_source_sha256")
        != verifier_source_sha256
        or envelope_payload.get("numerical_result_status")
        != review_payload.get("review_result_status")
    ):
        raise _blocked("existing_run_assessment_manifest_invalid")
    expected_envelope = _assessment_envelope(
        source_root=source,
        preservation=preservation,
        review_root=review_root,
        review_manifest_sha256=review_manifest_sha256,
        review_result_sha256=review_result_sha256,
        stop_evidence_sha256=stop_evidence_sha256,
        verifier_source_sha256=verifier_source_sha256,
        source_snapshot=reviewed_snapshot,
        metrics=live_metrics,
    )
    if envelope_payload != expected_envelope:
        raise _blocked("existing_run_assessment_manifest_invalid")
    return VerifiedExistingRunAssessment(
        source_root=source,
        review_root=review_root,
        envelope_root=envelope,
        phase_snapshot_sha256=_canonical_sha256(snapshot_index),
        stop_evidence_sha256=stop_evidence_sha256,
        review_manifest_sha256=review_manifest_sha256,
        review_result_sha256=review_result_sha256,
        envelope_sha256=_bytes_sha256(envelope_bytes),
        envelope_manifest_sha256=_bytes_sha256(manifest_bytes),
        verifier_source_sha256=verifier_source_sha256,
        assessment_use_status="authorized_existing_run",
        strict_prestart_lineage_status="not_satisfied",
        lineage_limitation_acknowledged=True,
        numerical_gate_status="recomputed_from_raw_rows",
        numerical_result_status=str(
            envelope_payload["numerical_result_status"]
        ),
        review_metrics=MappingProxyType(dict(live_metrics)),
        snapshot=MappingProxyType(dict(reviewed_snapshot)),
    )


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create and verify the fixed G1 p01-p04 existing-run assessment."
        )
    )
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--preservation-root", type=Path, required=True)
    parser.add_argument("--stop-evidence", type=Path, required=True)
    parser.add_argument("--review-root", type=Path, required=True)
    parser.add_argument("--envelope-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _argument_parser().parse_args(argv)
    try:
        envelope_root = create_existing_run_assessment(
            source_root=args.source_root,
            preservation_root=args.preservation_root,
            stop_evidence_path=args.stop_evidence,
            review_root=args.review_root,
            envelope_root=args.envelope_root,
        )
        verified = verify_existing_run_assessment(
            envelope_root=envelope_root,
            expected_source_root=args.source_root,
        )
    except ExistingRunAssessmentBlocked as exc:
        print(
            json.dumps(
                {
                    "execution_status": "blocked",
                    "blocking_reason": exc.reason,
                },
                sort_keys=True,
            )
        )
        return 2
    print(
        json.dumps(
            {
                "execution_status": "complete",
                "numerical_result_status": verified.numerical_result_status,
                "assessment_use_status": verified.assessment_use_status,
                "envelope_root": _path_text(verified.envelope_root),
                "envelope_manifest_sha256": (
                    verified.envelope_manifest_sha256
                ),
            },
            sort_keys=True,
        )
    )
    return 0


__all__ = [
    "ASSESSMENT_SCHEMA",
    "AUTHORIZED_RUN_ID",
    "AUTHORIZED_SOURCE_ROOT",
    "AUTHORITY_ALLOWLIST",
    "ExistingRunAssessmentBlocked",
    "VerifiedExistingRunAssessment",
    "create_existing_run_assessment",
    "main",
    "recompute_existing_run_numerics_from_snapshot",
    "verify_existing_run_assessment",
]


if __name__ == "__main__":
    raise SystemExit(main())
