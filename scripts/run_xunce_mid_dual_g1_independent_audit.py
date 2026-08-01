"""Independently audit and visualize one completed G1 incremental repair.

This offline entrypoint deliberately does not import the formal G1 runner,
frontier/oracle code, or Stage6 workflow code.  It reads completed JSON
artifacts only and publishes a new, non-overwriting audit directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import random
import statistics
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import xunce_artifact_io as artifact_io


MERGED_SCHEMA = "xunce-mid-dual-g1-mixed-repair-row/v1"
PARENT_EPISODE_SCHEMA = "xunce-mid-dual-g1-coverage-episode/v1"
REPAIR_EPISODE_SCHEMA = "xunce-mid-dual-g1-repair-episode/v1"
REPAIR_MANIFEST_SCHEMA = "xunce-mid-dual-g1-repair-manifest/v1"
AUDIT_SCHEMA = "xunce-mid-dual-g1-independent-offline-audit/v1"
OUTPUT_MANIFEST_SCHEMA = "xunce-mid-dual-g1-independent-audit-manifest/v1"
FORMAL_COVERAGE_THRESHOLD = 0.80
DIAGNOSTIC_COVERAGE_REFERENCE = 0.99
BOOTSTRAP_SEED = 20260726
BOOTSTRAP_RESAMPLES = 2_000
EXPECTED_EPISODES = 24
EXPECTED_REPAIR_EPISODES = 3

_MERGED_KEYS = {
    "episode",
    "episode_id",
    "episode_index",
    "homogeneous_code_execution",
    "lane_id",
    "lineage",
    "mixed_code_incremental_repair",
    "result_origin",
    "result_sha256",
    "row_kind",
    "scenario_id",
    "schema_version",
    "source_line_sha256",
}
_PARENT_LINEAGE_KEYS = {
    "parent_code_sha256",
    "parent_line_bytes_sha256",
    "parent_line_number",
    "parent_line_size_bytes",
    "parent_results_sha256",
}
_REPAIR_LINEAGE_KEYS = {
    "parent_replaced_line_bytes_sha256",
    "parent_replaced_line_number",
    "repair_code_sha256",
    "repair_result_line_sha256",
    "repair_results_sha256",
}


class G1IndependentAuditError(RuntimeError):
    """Raised when the offline evidence fails closed."""


def _canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise G1IndependentAuditError("canonical_json_invalid") from exc


def _pretty_json_bytes(value: object) -> bytes:
    try:
        return (
            json.dumps(
                value,
                ensure_ascii=False,
                allow_nan=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise G1IndependentAuditError("pretty_json_invalid") from exc


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _exact_int(value: object, reason: str) -> int:
    if type(value) is not int:
        raise G1IndependentAuditError(reason)
    return value


def _finite_number(value: object, reason: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise G1IndependentAuditError(reason)
    return float(value)


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, child in pairs:
        if key in value:
            raise G1IndependentAuditError(f"duplicate_json_key:{key}")
        value[key] = child
    return value


def _reject_nonfinite_constant(value: str) -> None:
    raise G1IndependentAuditError(f"nonfinite_json_constant:{value}")


def _strict_json_object(payload: bytes, label: str) -> dict[str, Any]:
    try:
        decoded = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_nonfinite_constant,
        )
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
        G1IndependentAuditError,
    ) as exc:
        if isinstance(exc, G1IndependentAuditError):
            raise
        raise G1IndependentAuditError(f"{label}_json_invalid") from exc
    if type(decoded) is not dict:
        raise G1IndependentAuditError(f"{label}_json_root_invalid")
    return decoded


def _read_jsonl(path: str | Path, label: str) -> dict[str, object]:
    source = Path(path).resolve()
    if not artifact_io.path_is_file(source):
        raise G1IndependentAuditError(f"{label}_missing")
    payload = artifact_io.read_bytes(source)
    lines: list[dict[str, object]] = []
    for line_number, raw_line in enumerate(
        payload.splitlines(keepends=True),
        start=1,
    ):
        if not raw_line.strip():
            continue
        row = _strict_json_object(raw_line, f"{label}_line_{line_number}")
        lines.append(
            {
                "line_number": line_number,
                "raw_line": raw_line,
                "row": row,
                "sha256": _sha256(raw_line),
                "size_bytes": len(raw_line),
            }
        )
    return {
        "path": source,
        "payload": payload,
        "sha256": _sha256(payload),
        "size_bytes": len(payload),
        "lines": lines,
    }


def _read_manifest(path: str | Path) -> dict[str, object]:
    source = Path(path).resolve()
    if not artifact_io.path_is_file(source):
        raise G1IndependentAuditError("repair_manifest_missing")
    payload = artifact_io.read_bytes(source)
    value = _strict_json_object(payload, "repair_manifest")
    if value.get("schema_version") != REPAIR_MANIFEST_SCHEMA:
        raise G1IndependentAuditError("repair_manifest_schema_invalid")
    return {
        "path": source,
        "payload": payload,
        "sha256": _sha256(payload),
        "size_bytes": len(payload),
        "value": value,
    }


def _require_file_hash(
    label: str,
    *,
    actual: object,
    expected: object,
) -> None:
    if not _is_sha256(expected) or actual != expected:
        raise G1IndependentAuditError(f"{label}_sha256_drift")


def _episode_index(
    source: Mapping[str, object],
    *,
    row_kind: str,
    schema_version: str,
    label: str,
) -> dict[int, dict[str, object]]:
    indexed: dict[int, dict[str, object]] = {}
    lines = source.get("lines")
    if not isinstance(lines, list):
        raise G1IndependentAuditError(f"{label}_lines_invalid")
    for line in lines:
        if not isinstance(line, Mapping):
            raise G1IndependentAuditError(f"{label}_line_invalid")
        row = line.get("row")
        if not isinstance(row, dict) or row.get("row_kind") != row_kind:
            continue
        if row.get("schema_version") != schema_version:
            raise G1IndependentAuditError(f"{label}_episode_schema_invalid")
        index = _exact_int(
            row.get("episode_index"),
            f"{label}_episode_index_invalid",
        )
        if index in indexed:
            raise G1IndependentAuditError(
                f"{label}_episode_index_duplicate"
            )
        indexed[index] = {
            "line_number": line["line_number"],
            "raw_line": line["raw_line"],
            "row": row,
            "sha256": line["sha256"],
            "size_bytes": line["size_bytes"],
        }
    return indexed


def _integer_coverage(
    episode: Mapping[str, object],
    *,
    label: str,
) -> tuple[int, int, float, int]:
    denominator = _exact_int(
        episode.get("denominator_cell_count"),
        f"{label}_denominator_invalid",
    )
    final = _exact_int(
        episode.get("final_covered_cell_count"),
        f"{label}_final_count_invalid",
    )
    safety = _exact_int(
        episode.get("safety_violation_count"),
        f"{label}_safety_count_invalid",
    )
    recorded = _finite_number(
        episode.get("coverage"),
        f"{label}_coverage_invalid",
    )
    if (
        denominator <= 0
        or not 0 <= final <= denominator
        or safety < 0
    ):
        raise G1IndependentAuditError(f"{label}_integer_range_invalid")
    recomputed = final / denominator
    if recorded != recomputed:
        raise G1IndependentAuditError("integer_coverage_mismatch")
    return final, denominator, recomputed, safety


def _validate_wrapper_identity(
    wrapper: Mapping[str, object],
    episode: Mapping[str, object],
    index: int,
) -> None:
    if (
        set(wrapper) != _MERGED_KEYS
        or wrapper.get("row_kind") != "mixed_repair_episode"
        or wrapper.get("schema_version") != MERGED_SCHEMA
        or wrapper.get("mixed_code_incremental_repair") is not True
        or wrapper.get("homogeneous_code_execution") is not False
        or wrapper.get("episode_index") != index
        or wrapper.get("episode_id") != episode.get("episode_id")
        or wrapper.get("scenario_id") != episode.get("scenario_id")
        or wrapper.get("lane_id") != episode.get("lane_id")
    ):
        raise G1IndependentAuditError("merged_wrapper_identity_invalid")
    if wrapper.get("result_sha256") != _sha256(_canonical_bytes(episode)):
        raise G1IndependentAuditError("merged_result_sha256_drift")


def _nearest_rank(values: Sequence[float], quantile: float) -> float:
    index = max(1, math.ceil(quantile * len(values))) - 1
    return sorted(values)[index]


def _bootstrap_ci(values: Sequence[float]) -> dict[str, object]:
    generator = random.Random(BOOTSTRAP_SEED)
    sampled_means = sorted(
        statistics.mean(
            generator.choice(values) for _ in range(len(values))
        )
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


def _phase_attempt_hashes(
    source: Mapping[str, object],
) -> dict[str, str]:
    rows = source.get("lines")
    if not isinstance(rows, list):
        raise G1IndependentAuditError("phase_attempts_invalid")
    phase_hashes: dict[str, str] = {}
    for line in rows:
        if not isinstance(line, Mapping):
            raise G1IndependentAuditError("phase_attempts_invalid")
        row = line.get("row")
        if not isinstance(row, Mapping):
            raise G1IndependentAuditError("phase_attempts_invalid")
        phase = row.get("phase_id")
        if phase not in {"p03", "p04"}:
            continue
        digest = row.get("row_sha256")
        if (
            phase in phase_hashes
            or not _is_sha256(digest)
            or row.get("status") != "accepted"
        ):
            raise G1IndependentAuditError("phase_attempts_binding_invalid")
        phase_hashes[str(phase)] = str(digest)
    if set(phase_hashes) != {"p03", "p04"}:
        raise G1IndependentAuditError("phase_attempts_binding_invalid")
    return phase_hashes


def _summarize_test_q24(
    indexed: Mapping[int, Mapping[str, object]],
) -> dict[str, object]:
    if set(indexed) != set(range(EXPECTED_EPISODES)):
        raise G1IndependentAuditError("test_episode_set_invalid")
    values: list[float] = []
    episodes: list[dict[str, object]] = []
    total_safety = 0
    violation_episode_count = 0
    for index in range(EXPECTED_EPISODES):
        source = indexed[index]
        episode = source.get("row")
        if not isinstance(episode, Mapping):
            raise G1IndependentAuditError("test_episode_invalid")
        if (
            episode.get("phase_id") != "p03"
            or episode.get("phase_name") != "test_q24"
            or episode.get("split") != "test_q24"
            or episode.get("episode_id") != f"test_q24-episode-{index:02d}"
        ):
            raise G1IndependentAuditError("test_episode_identity_invalid")
        final, denominator, coverage, safety = _integer_coverage(
            episode,
            label=f"test_episode_{index}",
        )
        values.append(coverage)
        total_safety += safety
        if safety > 0:
            violation_episode_count += 1
        episodes.append(
            {
                "evaluation_split": "test_q24",
                "episode_index": index,
                "episode_id": episode.get("episode_id"),
                "scenario_id": episode.get("scenario_id"),
                "lane_id": episode.get("lane_id"),
                "result_origin": "parent_test_read_only",
                "final_covered_cell_count": final,
                "denominator_cell_count": denominator,
                "coverage": coverage,
                "safety_violation_count": safety,
                "termination_reason": episode.get("termination_reason"),
                "source_line_sha256": source.get("sha256"),
                "result_sha256": _sha256(_canonical_bytes(episode)),
            }
        )
    coverage_80_count = sum(
        value >= FORMAL_COVERAGE_THRESHOLD for value in values
    )
    coverage_99_count = sum(
        value >= DIAGNOSTIC_COVERAGE_REFERENCE for value in values
    )
    return {
        "sample_count": EXPECTED_EPISODES,
        "coverage_80_count": coverage_80_count,
        "coverage_99_count": coverage_99_count,
        "coverage_99_role": "diagnostic_reference_only",
        "mean": sum(values) / len(values),
        "median": statistics.median(values),
        "sample_stddev": statistics.stdev(values),
        "min": min(values),
        "max": max(values),
        "bootstrap_ci": _bootstrap_ci(values),
        "safety": {
            "clean_episode_count": EXPECTED_EPISODES
            - violation_episode_count,
            "violation_episode_count": violation_episode_count,
            "total_violation_count": total_safety,
            "violations_by_origin": {
                "parent_test_read_only": total_safety,
            },
        },
        "below_80_episodes": [
            {
                "episode_index": row["episode_index"],
                "scenario_id": row["scenario_id"],
                "coverage": row["coverage"],
            }
            for row in episodes
            if row["coverage"] < FORMAL_COVERAGE_THRESHOLD
        ],
        "episodes": episodes,
    }


def audit_offline_evidence(
    *,
    merged_results_path: str | Path,
    parent_results_path: str | Path,
    repair_results_path: str | Path,
    repair_manifest_path: str | Path,
    test_results_path: str | Path,
    parent_phase_attempts_path: str | Path,
) -> dict[str, object]:
    """Recompute integrity and statistics without importing runner logic."""

    manifest_source = _read_manifest(repair_manifest_path)
    manifest = manifest_source["value"]
    if not isinstance(manifest, dict):
        raise G1IndependentAuditError("repair_manifest_invalid")
    merged_source = _read_jsonl(merged_results_path, "merged_results")
    parent_source = _read_jsonl(parent_results_path, "parent_results")
    repair_source = _read_jsonl(repair_results_path, "repair_results")
    test_source = _read_jsonl(test_results_path, "test_results")
    phase_attempts_source = _read_jsonl(
        parent_phase_attempts_path,
        "parent_phase_attempts",
    )

    _require_file_hash(
        "merged_results",
        actual=merged_source["sha256"],
        expected=manifest.get("merged_results_sha256"),
    )
    _require_file_hash(
        "parent_results",
        actual=parent_source["sha256"],
        expected=manifest.get("parent_results_sha256"),
    )
    _require_file_hash(
        "repair_results",
        actual=repair_source["sha256"],
        expected=manifest.get("repair_results_sha256"),
    )
    phase_hashes = _phase_attempt_hashes(phase_attempts_source)
    _require_file_hash(
        "test_results",
        actual=test_source["sha256"],
        expected=phase_hashes["p03"],
    )
    _require_file_hash(
        "parent_results",
        actual=parent_source["sha256"],
        expected=phase_hashes["p04"],
    )

    parent_by_index = _episode_index(
        parent_source,
        row_kind="coverage_episode",
        schema_version=PARENT_EPISODE_SCHEMA,
        label="parent",
    )
    repair_by_index = _episode_index(
        repair_source,
        row_kind="repair_coverage_episode",
        schema_version=REPAIR_EPISODE_SCHEMA,
        label="repair",
    )
    test_by_index = _episode_index(
        test_source,
        row_kind="coverage_episode",
        schema_version=PARENT_EPISODE_SCHEMA,
        label="test",
    )
    if set(parent_by_index) != set(range(EXPECTED_EPISODES)):
        raise G1IndependentAuditError("parent_episode_set_invalid")
    repair_indices = sorted(repair_by_index)
    if (
        len(repair_indices) != EXPECTED_REPAIR_EPISODES
        or manifest.get("accepted_episode_indices") != repair_indices
    ):
        raise G1IndependentAuditError("repair_episode_set_invalid")

    merged_lines = merged_source.get("lines")
    if not isinstance(merged_lines, list) or len(merged_lines) != EXPECTED_EPISODES:
        raise G1IndependentAuditError("merged_episode_count_invalid")
    merged_rows = [line.get("row") for line in merged_lines]
    if any(not isinstance(row, dict) for row in merged_rows):
        raise G1IndependentAuditError("merged_row_invalid")
    indices = [row.get("episode_index") for row in merged_rows]
    if indices != list(range(EXPECTED_EPISODES)):
        raise G1IndependentAuditError("merged_episode_order_invalid")

    parent_code_values = {
        source["row"].get("code_sha256")
        for source in parent_by_index.values()
        if isinstance(source.get("row"), Mapping)
    }
    repair_code_values = {
        source["row"].get("repair_code_sha256")
        for source in repair_by_index.values()
        if isinstance(source.get("row"), Mapping)
    }
    if (
        len(parent_code_values) != 1
        or len(repair_code_values) != 1
        or not all(
            _is_sha256(value)
            for value in parent_code_values | repair_code_values
        )
        or parent_code_values == repair_code_values
    ):
        raise G1IndependentAuditError("mixed_code_lineage_invalid")

    coverage_values: list[float] = []
    episode_rows: list[dict[str, object]] = []
    repair_pairs: list[dict[str, object]] = []
    origins = {"parent_read_only": 0, "repair_rerun": 0}
    total_safety = 0
    safety_episode_count = 0
    safety_by_origin = {"parent_read_only": 0, "repair_rerun": 0}

    for index, wrapper_value in enumerate(merged_rows):
        if not isinstance(wrapper_value, dict):
            raise G1IndependentAuditError("merged_row_invalid")
        episode = wrapper_value.get("episode")
        if not isinstance(episode, dict):
            raise G1IndependentAuditError("merged_episode_invalid")
        _validate_wrapper_identity(wrapper_value, episode, index)
        final, denominator, coverage, safety = _integer_coverage(
            episode,
            label=f"merged_episode_{index}",
        )

        expected_origin = (
            "repair_rerun"
            if index in repair_by_index
            else "parent_read_only"
        )
        origin = wrapper_value.get("result_origin")
        lineage = wrapper_value.get("lineage")
        if origin != expected_origin or not isinstance(lineage, dict):
            raise G1IndependentAuditError(
                f"mixed_origin_or_lineage:{index}"
            )
        source = (
            repair_by_index[index]
            if expected_origin == "repair_rerun"
            else parent_by_index[index]
        )
        if (
            episode != source["row"]
            or wrapper_value.get("source_line_sha256") != source["sha256"]
        ):
            raise G1IndependentAuditError(
                f"mixed_origin_or_lineage:{index}"
            )

        parent_source_line = parent_by_index[index]
        if expected_origin == "parent_read_only":
            expected_lineage = {
                "parent_code_sha256": episode.get("code_sha256"),
                "parent_line_bytes_sha256": source["sha256"],
                "parent_line_number": source["line_number"],
                "parent_line_size_bytes": source["size_bytes"],
                "parent_results_sha256": parent_source["sha256"],
            }
            if set(lineage) != _PARENT_LINEAGE_KEYS or lineage != expected_lineage:
                raise G1IndependentAuditError(
                    f"mixed_origin_or_lineage:{index}"
                )
        else:
            expected_lineage = {
                "parent_replaced_line_bytes_sha256": parent_source_line[
                    "sha256"
                ],
                "parent_replaced_line_number": parent_source_line[
                    "line_number"
                ],
                "repair_code_sha256": episode.get("repair_code_sha256"),
                "repair_result_line_sha256": source["sha256"],
                "repair_results_sha256": repair_source["sha256"],
            }
            if set(lineage) != _REPAIR_LINEAGE_KEYS or lineage != expected_lineage:
                raise G1IndependentAuditError(
                    f"mixed_origin_or_lineage:{index}"
                )
            before_episode = parent_source_line["row"]
            if not isinstance(before_episode, Mapping):
                raise G1IndependentAuditError(
                    f"mixed_origin_or_lineage:{index}"
                )
            _, _, before_coverage, _ = _integer_coverage(
                before_episode,
                label=f"parent_replaced_episode_{index}",
            )
            repair_pairs.append(
                {
                    "episode_index": index,
                    "scenario_id": episode.get("scenario_id"),
                    "before_coverage": before_coverage,
                    "after_coverage": coverage,
                }
            )

        origins[expected_origin] += 1
        coverage_values.append(coverage)
        total_safety += safety
        safety_by_origin[expected_origin] += safety
        if safety > 0:
            safety_episode_count += 1
        episode_rows.append(
            {
                "evaluation_split": "unseen24",
                "episode_index": index,
                "episode_id": episode.get("episode_id"),
                "scenario_id": episode.get("scenario_id"),
                "lane_id": episode.get("lane_id"),
                "result_origin": expected_origin,
                "final_covered_cell_count": final,
                "denominator_cell_count": denominator,
                "coverage": coverage,
                "safety_violation_count": safety,
                "termination_reason": episode.get("termination_reason"),
                "source_line_sha256": source["sha256"],
                "result_sha256": wrapper_value.get("result_sha256"),
            }
        )

    if origins != {"parent_read_only": 21, "repair_rerun": 3}:
        raise G1IndependentAuditError("mixed_lineage_cardinality_invalid")

    coverage_80_count = sum(
        value >= FORMAL_COVERAGE_THRESHOLD for value in coverage_values
    )
    coverage_99_count = sum(
        value >= DIAGNOSTIC_COVERAGE_REFERENCE for value in coverage_values
    )
    mean_value = sum(coverage_values) / len(coverage_values)
    manifest_mean = _finite_number(
        manifest.get("mean"),
        "manifest_mean_invalid",
    )
    if (
        manifest.get("coverage_80_count") != coverage_80_count
        or manifest.get("coverage_99_count") != coverage_99_count
        or not math.isclose(
            manifest_mean,
            mean_value,
            rel_tol=0.0,
            abs_tol=1e-15,
        )
        or manifest.get("mixed_code_incremental_repair") is not True
        or manifest.get("homogeneous_code_execution") is not False
    ):
        raise G1IndependentAuditError("repair_manifest_summary_drift")

    test_q24 = _summarize_test_q24(test_by_index)
    return {
        "schema_version": AUDIT_SCHEMA,
        "status": "passed",
        "audit_scope": "offline_completed_artifacts_only/v1",
        "runner_implementation_imported": False,
        "replay_or_scenario_execution_performed": False,
        "mixed_code_incremental_repair": True,
        "homogeneous_code_execution": False,
        "sample_count": EXPECTED_EPISODES,
        "lineage_counts": origins,
        "coverage_80_count": coverage_80_count,
        "coverage_80_all_episodes_passed": (
            coverage_80_count == EXPECTED_EPISODES
        ),
        "coverage_99_count": coverage_99_count,
        "coverage_99_role": "diagnostic_reference_only",
        "mean": mean_value,
        "median": statistics.median(coverage_values),
        "sample_stddev": statistics.stdev(coverage_values),
        "min": min(coverage_values),
        "max": max(coverage_values),
        "bootstrap_ci": _bootstrap_ci(coverage_values),
        "safety": {
            "clean_episode_count": EXPECTED_EPISODES
            - safety_episode_count,
            "violation_episode_count": safety_episode_count,
            "total_violation_count": total_safety,
            "violations_by_origin": safety_by_origin,
        },
        "repair_pairs": repair_pairs,
        "episodes": episode_rows,
        "test_q24": test_q24,
        "overall": {
            "scene_80_count": (
                coverage_80_count + test_q24["coverage_80_count"]
            ),
            "scene_count": EXPECTED_EPISODES * 2,
            "split_mean_80_count": sum(
                value >= FORMAL_COVERAGE_THRESHOLD
                for value in (mean_value, test_q24["mean"])
            ),
            "split_mean_count": 2,
        },
        "input_files": {
            "merged_results": {
                "path": str(merged_source["path"]),
                "sha256": merged_source["sha256"],
                "size_bytes": merged_source["size_bytes"],
            },
            "parent_results": {
                "path": str(parent_source["path"]),
                "sha256": parent_source["sha256"],
                "size_bytes": parent_source["size_bytes"],
            },
            "repair_results": {
                "path": str(repair_source["path"]),
                "sha256": repair_source["sha256"],
                "size_bytes": repair_source["size_bytes"],
            },
            "repair_manifest": {
                "path": str(manifest_source["path"]),
                "sha256": manifest_source["sha256"],
                "size_bytes": manifest_source["size_bytes"],
            },
            "test_results": {
                "path": str(test_source["path"]),
                "sha256": test_source["sha256"],
                "size_bytes": test_source["size_bytes"],
            },
            "parent_phase_attempts": {
                "path": str(phase_attempts_source["path"]),
                "sha256": phase_attempts_source["sha256"],
                "size_bytes": phase_attempts_source["size_bytes"],
            },
        },
    }


def _episodes_csv_bytes(audit: Mapping[str, object]) -> bytes:
    repair_before = {
        row["episode_index"]: row["before_coverage"]
        for row in audit["repair_pairs"]
        if isinstance(row, Mapping)
    }
    output = io.StringIO(newline="")
    fieldnames = [
        "evaluation_split",
        "episode_index",
        "episode_id",
        "scenario_id",
        "lane_id",
        "result_origin",
        "final_covered_cell_count",
        "denominator_cell_count",
        "coverage",
        "parent_before_coverage",
        "safety_violation_count",
        "termination_reason",
        "source_line_sha256",
        "result_sha256",
    ]
    writer = csv.DictWriter(
        output,
        fieldnames=fieldnames,
        lineterminator="\n",
    )
    writer.writeheader()
    rows = [*audit["test_q24"]["episodes"], *audit["episodes"]]
    for row in rows:
        if not isinstance(row, Mapping):
            raise G1IndependentAuditError("audit_episode_invalid")
        writer.writerow(
            {
                **{field: row.get(field) for field in fieldnames},
                "parent_before_coverage": repair_before.get(
                    row["episode_index"],
                    "",
                ),
            }
        )
    return output.getvalue().encode("utf-8")


def _source_data(audit: Mapping[str, object]) -> dict[str, object]:
    return {
        "schema_version": "xunce-mid-dual-g1-independent-figure-data/v1",
        "thresholds": {
            "formal_threshold": FORMAL_COVERAGE_THRESHOLD,
            "diagnostic_reference": DIAGNOSTIC_COVERAGE_REFERENCE,
        },
        "statistics": {
            key: audit[key]
            for key in (
                "sample_count",
                "mean",
                "median",
                "sample_stddev",
                "min",
                "max",
                "coverage_80_count",
                "coverage_99_count",
                "bootstrap_ci",
                "safety",
                "lineage_counts",
            )
        },
        "episodes": audit["episodes"],
        "test_q24_episodes": audit["test_q24"]["episodes"],
        "test_q24_statistics": {
            key: audit["test_q24"][key]
            for key in (
                "sample_count",
                "mean",
                "median",
                "sample_stddev",
                "min",
                "max",
                "coverage_80_count",
                "coverage_99_count",
                "bootstrap_ci",
                "safety",
                "below_80_episodes",
            )
        },
        "overall": audit["overall"],
        "repair_pairs": audit["repair_pairs"],
        "notes": {
            "coverage_definition": (
                "final_covered_cell_count / denominator_cell_count"
            ),
            "ci_method": (
                "95% percentile interval from 2,000 episode-level "
                "bootstrap resamples; seed 20260726"
            ),
            "lineage": "21 parent_read_only + 3 repair_rerun",
            "mixed_code_incremental_repair": True,
            "diagnostic_reference_is_not_gate": True,
        },
    }


def _report_text(audit: Mapping[str, object]) -> str:
    ci = audit["bootstrap_ci"]
    safety = audit["safety"]
    test_q24 = audit["test_q24"]
    test_ci = test_q24["bootstrap_ci"]
    overall = audit["overall"]
    repair_99 = sum(
        row["after_coverage"] >= DIAGNOSTIC_COVERAGE_REFERENCE
        for row in audit["repair_pairs"]
    )
    input_files = audit["input_files"]
    return (
        "# G1 增量修复独立离线完整性审计\n\n"
        "## 结论\n\n"
        f"- Unseen-24 修复后 80% formal threshold："
        f"{audit['coverage_80_count']}/{audit['sample_count']}。\n"
        f"- Test-Q24：{test_q24['coverage_80_count']}/"
        f"{test_q24['sample_count']} 个场景达到 80%；唯一低值为 "
        f"`{test_q24['below_80_episodes'][0]['scenario_id']}`（"
        f"{test_q24['below_80_episodes'][0]['coverage']:.9%}）。\n"
        f"- 两个 split 的集合均值均超过 80%（"
        f"{overall['split_mean_80_count']}/{overall['split_mean_count']}），"
        f"但逐场景仅 {overall['scene_80_count']}/{overall['scene_count']} "
        "达到 80%，并非 48/48 个场景均过线。\n"
        f"- 99% diagnostic reference：{audit['coverage_99_count']}/"
        f"{audit['sample_count']}；仅作诊断参考，不是本审计的考核门槛。\n"
        f"- 三个增量场景中 {repair_99}/3 超过 99% 诊断参考线。\n"
        "- lineage：21 parent + 3 repair；这是 mixed-code incremental "
        "lineage，不是 homogeneous-code 重跑。\n"
        f"- 安全：{safety['total_violation_count']} 次违规，"
        f"{safety['clean_episode_count']}/{audit['sample_count']} episode "
        "无违规。\n\n"
        "## 独立统计\n\n"
        f"- n={audit['sample_count']} episodes。\n"
        f"- Unseen-24 均值：{audit['mean']:.9%}。\n"
        f"- 95% CI：[{ci['lower']:.9%}, {ci['upper']:.9%}]；"
        "2,000 次 episode-level bootstrap，seed=20260726。\n"
        f"- 中位数：{audit['median']:.9%}；样本标准差："
        f"{audit['sample_stddev']:.9%}。\n\n"
        f"- Test-Q24 均值：{test_q24['mean']:.16f}；95% CI："
        f"[{test_ci['lower']:.9%}, {test_ci['upper']:.9%}]。\n\n"
        "## 完整性边界\n\n"
        "- 仅读取已完成 artifact；未执行 replay、环境或任何场景。\n"
        "- 未导入正式 G1 runner、frontier/oracle 或 Stage6 workflow "
        "实现。\n"
        "- merged 嵌套 episode 必须逐字节绑定原始 parent/repair JSONL "
        "行，并同时通过全文件 SHA、lineage、result SHA 与整数覆盖复算。\n"
        f"- merged SHA-256："
        f"`{input_files['merged_results']['sha256']}`。\n"
        f"- parent raw SHA-256："
        f"`{input_files['parent_results']['sha256']}`。\n"
        f"- repair raw SHA-256："
        f"`{input_files['repair_results']['sha256']}`。\n"
        f"- Test-Q24 raw SHA-256："
        f"`{input_files['test_results']['sha256']}`。\n"
    )


def _caption_text(audit: Mapping[str, object]) -> str:
    ci = audit["bootstrap_ci"]
    safety = audit["safety"]
    test_q24 = audit["test_q24"]
    test_ci = test_q24["bootstrap_ci"]
    overall = audit["overall"]
    repair_99 = sum(
        row["after_coverage"] >= DIAGNOSTIC_COVERAGE_REFERENCE
        for row in audit["repair_pairs"]
    )
    return (
        "# Caption draft\n\n"
        "**Independent offline audit of G1 Test-Q24 and repaired Unseen-24 "
        "coverage.** "
        "Each split contains n=24 episodes. "
        f"Panel a ranks the repaired Unseen episodes by coverage recomputed "
        "from integer "
        "covered-cell numerators and denominator-cell counts. The dashed "
        "line denotes the unchanged 80% formal threshold; the dotted 99% "
        "diagnostic reference is shown only for interpretation and is not "
        "an assessment threshold. Low-saturation blue denotes the 21 "
        "read-only parent results and orange denotes the 3 explicit repair "
        "reruns. Panel b presents the original Test-Q24 results, for which "
        f"{test_q24['coverage_80_count']}/24 scenes meet 80% and the split "
        f"mean is {test_q24['mean']:.3%} "
        f"[{test_ci['lower']:.3%}, {test_ci['upper']:.3%}]. Panel c pairs "
        "each replaced parent result with its repair; small deterministic "
        "horizontal offsets separate overlapping traces and do not encode "
        f"data; {repair_99}/3 repairs exceed the diagnostic reference. "
        "Panel d reports split means and 95% percentile intervals from "
        f"2,000 episode-level bootstrap resamples "
        f"(seed={ci['seed']}): {audit['mean']:.3%} "
        f"[{ci['lower']:.3%}, {ci['upper']:.3%}]. "
        f"Safety violations were {safety['total_violation_count']} across "
        f"repaired Unseen-24 and {test_q24['safety']['total_violation_count']} "
        "across original Test-Q24. Both split means exceed 80%, while "
        f"{overall['scene_80_count']}/{overall['scene_count']} individual "
        "scenes meet 80%; this is not a 48/48 scene-level result. The "
        "Unseen evidence is a mixed-code incremental lineage "
        "(21 parent + 3 repair), not a homogeneous-code rerun.\n"
    )


def _hero_repair_label_offset(episode_index: int) -> tuple[float, float]:
    return {
        3: (-12.0, 14.0),
        10: (0.0, 18.0),
        23: (12.0, 14.0),
    }.get(episode_index, (0.0, 16.0))


def _repair_display_layout(
    repair_pairs: Sequence[Mapping[str, object]],
) -> dict[int, dict[str, object]]:
    if len(repair_pairs) != EXPECTED_REPAIR_EPISODES:
        raise G1IndependentAuditError("repair_display_cardinality_invalid")
    by_index = sorted(
        repair_pairs,
        key=lambda pair: int(pair["episode_index"]),
    )
    layout: dict[int, dict[str, object]] = {}
    for pair, x_offset, marker, line_style in zip(
        by_index,
        (-0.035, 0.0, 0.035),
        ("o", "s", "^"),
        ("solid", "dashed", "dotted"),
        strict=True,
    ):
        layout[int(pair["episode_index"])] = {
            "x_offset": x_offset,
            "marker": marker,
            "line_style": line_style,
        }

    previous_label = -math.inf
    labels_by_index: dict[int, float] = {}
    for pair in sorted(
        repair_pairs,
        key=lambda candidate: (
            float(candidate["after_coverage"]),
            int(candidate["episode_index"]),
        ),
    ):
        repair_pct = 100.0 * float(pair["after_coverage"])
        label_pct = max(repair_pct, previous_label + 4.0)
        labels_by_index[int(pair["episode_index"])] = label_pct
        previous_label = label_pct
    if previous_label > 101.0:
        shift = previous_label - 101.0
        labels_by_index = {
            index: value - shift
            for index, value in labels_by_index.items()
        }
    for index, label_y in labels_by_index.items():
        layout[index]["label_y"] = label_y
    return layout


def _render_figure(
    audit: Mapping[str, object],
    *,
    output_base: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = [
        "Arial",
        "DejaVu Sans",
        "Liberation Sans",
    ]
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["font.size"] = 7
    plt.rcParams["axes.linewidth"] = 0.8
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["legend.frameon"] = False
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "white"

    parent_color = "#5B7FA6"
    repair_color = "#D99152"
    threshold_color = "#2F6B5F"
    diagnostic_color = "#767676"

    episodes = list(audit["episodes"])
    test_episodes = list(audit["test_q24"]["episodes"])
    ranked = sorted(
        episodes,
        key=lambda row: (float(row["coverage"]), int(row["episode_index"])),
    )
    ci = audit["bootstrap_ci"]
    safety = audit["safety"]
    test_ci = audit["test_q24"]["bootstrap_ci"]
    test_safety = audit["test_q24"]["safety"]

    figure = plt.figure(figsize=(183 / 25.4, 155 / 25.4))
    grid = figure.add_gridspec(
        2,
        2,
        hspace=0.45,
        wspace=0.34,
    )
    hero = figure.add_subplot(grid[0, 0])
    test_panel = figure.add_subplot(grid[0, 1])
    paired = figure.add_subplot(grid[1, 0])
    summary = figure.add_subplot(grid[1, 1])

    ranks = list(range(1, len(ranked) + 1))
    coverages_pct = [100.0 * float(row["coverage"]) for row in ranked]
    colors = [
        repair_color
        if row["result_origin"] == "repair_rerun"
        else parent_color
        for row in ranked
    ]
    hero.vlines(
        ranks,
        80.0,
        coverages_pct,
        color=colors,
        linewidth=0.75,
        alpha=0.58,
    )
    for rank, row, value, color in zip(
        ranks,
        ranked,
        coverages_pct,
        colors,
        strict=True,
    ):
        marker = "D" if row["result_origin"] == "repair_rerun" else "o"
        hero.scatter(
            rank,
            value,
            s=25 if marker == "D" else 17,
            marker=marker,
            color=color,
            edgecolor="white",
            linewidth=0.55,
            zorder=3,
        )
        if marker == "D":
            label_offset = _hero_repair_label_offset(
                int(row["episode_index"])
            )
            hero.annotate(
                f"E{int(row['episode_index']):02d}",
                (rank, value),
                xytext=label_offset,
                textcoords="offset points",
                ha="center",
                va="bottom",
                color="#8A4F24",
                fontsize=5.9,
                fontweight="bold",
                arrowprops={
                    "arrowstyle": "-",
                    "color": "#B76E36",
                    "linewidth": 0.55,
                },
                bbox={
                    "boxstyle": "square,pad=0.08",
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.88,
                },
                annotation_clip=False,
            )
    hero.axhline(
        80,
        color=threshold_color,
        linestyle="--",
        linewidth=1.15,
    )
    hero.axhline(
        99,
        color=diagnostic_color,
        linestyle=":",
        linewidth=1.0,
    )
    hero.text(
        24.65,
        80,
        "80% formal threshold",
        color=threshold_color,
        fontsize=6.2,
        ha="right",
        va="bottom",
    )
    hero.text(
        1.0,
        100.7,
        "99% diagnostic reference",
        color=diagnostic_color,
        fontsize=6.2,
        ha="left",
        va="bottom",
        bbox={
            "boxstyle": "square,pad=0.08",
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.88,
        },
    )
    hero.set_xlim(0.35, 24.75)
    hero.set_ylim(77.5, 103.2)
    hero.set_xticks((1, 4, 8, 12, 16, 20, 24))
    hero.set_xlabel("Episode rank by recomputed coverage")
    hero.set_ylabel("Covered cells (%)")
    hero.set_title(
        f"Unseen-24 repaired: {audit['coverage_80_count']}/24 scenes "
        "at or above 80%\n(21 parent + 3 repair)",
        loc="left",
        fontsize=7.2,
        fontweight="bold",
    )
    hero.text(
        -0.055,
        1.04,
        "a",
        transform=hero.transAxes,
        fontsize=9,
        fontweight="bold",
    )
    parent_handle = hero.scatter(
        [],
        [],
        s=17,
        marker="o",
        color=parent_color,
        label="Parent read-only (n=21)",
    )
    repair_handle = hero.scatter(
        [],
        [],
        s=25,
        marker="D",
        color=repair_color,
        label="Repair rerun (n=3)",
    )
    hero.legend(
        handles=(parent_handle, repair_handle),
        loc="center right",
        bbox_to_anchor=(0.99, 0.26),
        ncol=1,
        fontsize=5.8,
        handletextpad=0.4,
        labelspacing=0.25,
    )

    ranked_test = sorted(
        test_episodes,
        key=lambda row: (float(row["coverage"]), int(row["episode_index"])),
    )
    test_ranks = list(range(1, len(ranked_test) + 1))
    test_values = [
        100.0 * float(row["coverage"]) for row in ranked_test
    ]
    test_panel.vlines(
        test_ranks,
        0,
        test_values,
        color="#7C78A6",
        linewidth=0.7,
        alpha=0.5,
    )
    test_panel.scatter(
        test_ranks,
        test_values,
        s=15,
        color="#7C78A6",
        edgecolor="white",
        linewidth=0.45,
        zorder=3,
    )
    for rank, row, value in zip(
        test_ranks,
        ranked_test,
        test_values,
        strict=True,
    ):
        if value < 80:
            test_panel.scatter(
                rank,
                value,
                s=28,
                marker="X",
                color="#D99152",
                edgecolor="white",
                linewidth=0.5,
                zorder=4,
            )
            test_panel.annotate(
                "scenario-0094\n1.91%",
                (rank, value),
                xytext=(8, 6),
                textcoords="offset points",
                fontsize=5.8,
                color="#8A4F24",
                ha="left",
                va="bottom",
                fontweight="bold",
            )
    test_panel.axhline(
        80,
        color=threshold_color,
        linestyle="--",
        linewidth=0.9,
    )
    test_panel.axhline(
        99,
        color=diagnostic_color,
        linestyle=":",
        linewidth=0.75,
    )
    test_panel.set_xlim(0.35, 24.75)
    test_panel.set_ylim(0, 103)
    test_panel.set_xticks((1, 8, 16, 24))
    test_panel.set_xlabel("Episode rank")
    test_panel.set_ylabel("Covered cells (%)")
    test_panel.set_title(
        f"Test-Q24 original: {audit['test_q24']['coverage_80_count']}/24 "
        "scenes\nat or above 80%",
        loc="left",
        fontsize=7.2,
        fontweight="bold",
    )
    test_panel.text(
        -0.18,
        1.04,
        "b",
        transform=test_panel.transAxes,
        fontsize=9,
        fontweight="bold",
    )

    repair_pairs = list(audit["repair_pairs"])
    repair_layout = _repair_display_layout(repair_pairs)

    for pair in repair_pairs:
        index = int(pair["episode_index"])
        before = 100.0 * float(pair["before_coverage"])
        after = 100.0 * float(pair["after_coverage"])
        layout = repair_layout[index]
        x_offset = float(layout["x_offset"])
        x_values = (x_offset, 1.0 + x_offset)
        paired.plot(
            x_values,
            (before, after),
            color=repair_color,
            linewidth=1.15,
            alpha=0.82,
            linestyle=str(layout["line_style"]),
        )
        paired.scatter(
            x_values,
            (before, after),
            s=(18, 25),
            color=(parent_color, repair_color),
            marker=str(layout["marker"]),
            edgecolor="white",
            linewidth=0.5,
            zorder=3,
        )
        paired.annotate(
            f"E{index:02d}",
            (1.0 + x_offset, after),
            xytext=(1.10, float(layout["label_y"])),
            textcoords="data",
            fontsize=6.2,
            color="#8A4F24",
            va="center",
            ha="left",
            fontweight="bold",
            arrowprops={
                "arrowstyle": "-",
                "color": "#B76E36",
                "linewidth": 0.55,
            },
        )
    paired.axhline(
        80,
        color=threshold_color,
        linestyle="--",
        linewidth=0.9,
    )
    paired.axhline(
        99,
        color=diagnostic_color,
        linestyle=":",
        linewidth=0.8,
    )
    paired.set_xlim(-0.25, 1.35)
    paired.set_ylim(0, 103)
    paired.set_xticks((0, 1), ("Parent failure", "Repair"))
    paired.set_ylabel("Covered cells (%)")
    paired.set_title(
        "Incremental repair pairs",
        loc="left",
        fontsize=8,
        fontweight="bold",
    )
    paired.text(
        0.5,
        0.03,
        "x-offsets only separate overlapping traces",
        transform=paired.transAxes,
        fontsize=5.4,
        color="#666666",
        ha="center",
        va="bottom",
        bbox={
            "boxstyle": "square,pad=0.12",
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.90,
        },
        zorder=4,
    )
    paired.text(
        -0.12,
        1.04,
        "c",
        transform=paired.transAxes,
        fontsize=9,
        fontweight="bold",
    )

    mean_pct = 100.0 * float(audit["mean"])
    lower_pct = 100.0 * float(ci["lower"])
    upper_pct = 100.0 * float(ci["upper"])
    test_mean_pct = 100.0 * float(audit["test_q24"]["mean"])
    test_lower_pct = 100.0 * float(test_ci["lower"])
    test_upper_pct = 100.0 * float(test_ci["upper"])
    summary.errorbar(
        test_mean_pct,
        1.0,
        xerr=(
            (test_mean_pct - test_lower_pct,),
            (test_upper_pct - test_mean_pct,),
        ),
        fmt="o",
        markersize=4.5,
        color="#7C78A6",
        ecolor="#7C78A6",
        elinewidth=1.2,
        capsize=3,
    )
    summary.errorbar(
        mean_pct,
        0.45,
        xerr=(
            (mean_pct - lower_pct,),
            (upper_pct - mean_pct,),
        ),
        fmt="o",
        markersize=5,
        color="#3E5D7A",
        ecolor="#3E5D7A",
        elinewidth=1.3,
        capsize=3,
    )
    summary.axvline(
        80,
        color=threshold_color,
        linestyle="--",
        linewidth=0.9,
    )
    summary.axvline(
        99,
        color=diagnostic_color,
        linestyle=":",
        linewidth=0.8,
    )
    summary.text(
        test_mean_pct,
        1.13,
        f"Test mean {test_mean_pct:.2f}%\n"
        f"95% CI {test_lower_pct:.2f}–{test_upper_pct:.2f}%",
        fontsize=5.7,
        ha="center",
        va="bottom",
        color="#5B5685",
    )
    summary.text(
        mean_pct,
        0.58,
        f"Unseen mean {mean_pct:.2f}%\n"
        f"95% CI {lower_pct:.2f}–{upper_pct:.2f}%",
        fontsize=5.7,
        ha="center",
        va="bottom",
        color="#3E5D7A",
    )
    summary.text(
        0.02,
        0.05,
        "Split means >=80%: "
        f"{audit['overall']['split_mean_80_count']}/2\n"
        "Scene-level >=80%: "
        f"{audit['overall']['scene_80_count']}/48\n"
        "n=24 per split; 2,000 episode bootstraps\n"
        "Safety violations (Test / Unseen): "
        f"{test_safety['total_violation_count']} / "
        f"{safety['total_violation_count']}",
        transform=summary.transAxes,
        fontsize=5.7,
        ha="left",
        va="bottom",
        linespacing=1.32,
        bbox={
            "boxstyle": "square,pad=0.15",
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.92,
        },
        zorder=4,
    )
    summary.set_xlim(77.5, 101.4)
    summary.set_ylim(0, 1.62)
    summary.set_yticks([])
    summary.set_xlabel("Mean covered cells (%)")
    summary.set_title(
        "Split means and 95% CI",
        loc="left",
        fontsize=8,
        fontweight="bold",
    )
    summary.text(
        -0.12,
        1.04,
        "d",
        transform=summary.transAxes,
        fontsize=9,
        fontweight="bold",
    )
    summary.spines["left"].set_visible(False)

    figure.suptitle(
        "G1 coverage audit: split means pass 80%; scene-level rates differ",
        fontsize=9.2,
        fontweight="bold",
        x=0.08,
        ha="left",
        y=0.995,
    )
    figure.savefig(
        output_base.with_suffix(".svg"),
        bbox_inches="tight",
        facecolor="white",
    )
    figure.savefig(
        output_base.with_suffix(".pdf"),
        bbox_inches="tight",
        facecolor="white",
    )
    figure.savefig(
        output_base.with_suffix(".png"),
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(figure)


def write_deliverables(
    *,
    audit: Mapping[str, object],
    output_dir: str | Path,
    source_paths: Mapping[str, Path],
) -> dict[str, object]:
    """Atomically publish a new audit directory without overwriting."""

    output = Path(output_dir).resolve()
    staging = output.with_name(f".{output.name}.staging")
    if artifact_io.path_exists(output) or artifact_io.path_exists(staging):
        raise G1IndependentAuditError("output_directory_exists")
    artifact_io.make_dirs(staging)

    audit_payload = _pretty_json_bytes(dict(audit))
    artifact_io.write_bytes(staging / "audit.json", audit_payload)
    artifact_io.write_bytes(
        staging / "episodes.csv",
        _episodes_csv_bytes(audit),
    )
    artifact_io.write_bytes(
        staging / "source-data.json",
        _pretty_json_bytes(_source_data(audit)),
    )
    artifact_io.write_text(
        staging / "report.md",
        _report_text(audit),
        encoding="utf-8",
    )
    artifact_io.write_text(
        staging / "caption.md",
        _caption_text(audit),
        encoding="utf-8",
    )
    _render_figure(audit, output_base=staging / "figure")

    artifact_names = (
        "audit.json",
        "caption.md",
        "episodes.csv",
        "figure.pdf",
        "figure.png",
        "figure.svg",
        "report.md",
        "source-data.json",
    )
    artifacts: dict[str, dict[str, object]] = {}
    for name in artifact_names:
        path = staging / name
        if not artifact_io.path_is_file(path):
            raise G1IndependentAuditError(f"deliverable_missing:{name}")
        payload = artifact_io.read_bytes(path)
        artifacts[name] = {
            "sha256": _sha256(payload),
            "size_bytes": len(payload),
        }
    output_manifest = {
        "schema_version": OUTPUT_MANIFEST_SCHEMA,
        "status": "passed",
        "publication_policy": "new-directory-atomic-rename-no-overwrite/v1",
        "audit_sha256": artifacts["audit.json"]["sha256"],
        "artifacts": artifacts,
        "source_paths": {
            key: str(Path(value).resolve())
            for key, value in sorted(source_paths.items())
        },
    }
    artifact_io.write_bytes(
        staging / "manifest.json",
        _pretty_json_bytes(output_manifest),
    )
    os.replace(
        artifact_io.windows_safe_path(staging),
        artifact_io.windows_safe_path(output),
    )
    return output_manifest


def _find_parent_results(
    parent_root: Path,
    *,
    expected_sha256: str,
) -> Path:
    matches: list[Path] = []
    for relative in artifact_io.list_relative_files(parent_root):
        if not relative.endswith("results.jsonl"):
            continue
        candidate = parent_root / Path(relative)
        if _sha256(artifact_io.read_bytes(candidate)) == expected_sha256:
            matches.append(candidate.resolve())
    if len(matches) != 1:
        raise G1IndependentAuditError(
            "parent_results_hash_match_not_unique"
        )
    return matches[0]


def _validate_cli_paths(
    *,
    repair_root: Path,
    parent_root: Path,
    output_dir: Path,
) -> None:
    repair = repair_root.resolve()
    parent = parent_root.resolve()
    output = output_dir.resolve()
    if (
        not artifact_io.path_is_dir(repair)
        or not artifact_io.path_is_dir(parent)
    ):
        raise G1IndependentAuditError("formal_root_missing")
    if output.parent != repair:
        raise G1IndependentAuditError(
            "output_must_be_new_repair_root_child"
        )
    if not output.name.startswith("independent-offline-audit-"):
        raise G1IndependentAuditError("output_directory_name_invalid")


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Independently audit completed G1 merged/parent/repair artifacts "
            "and render publication exports without running any scenario."
        )
    )
    parser.add_argument("--repair-root", type=Path, required=True)
    parser.add_argument("--parent-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    repair_root = args.repair_root.resolve()
    parent_root = args.parent_root.resolve()
    output_dir = args.output_dir.resolve()
    _validate_cli_paths(
        repair_root=repair_root,
        parent_root=parent_root,
        output_dir=output_dir,
    )
    manifest_path = repair_root / "manifest.json"
    manifest_source = _read_manifest(manifest_path)
    manifest = manifest_source["value"]
    if not isinstance(manifest, Mapping):
        raise G1IndependentAuditError("repair_manifest_invalid")
    expected_parent_sha = manifest.get("parent_results_sha256")
    if not _is_sha256(expected_parent_sha):
        raise G1IndependentAuditError("parent_results_expected_sha_invalid")
    parent_results_path = _find_parent_results(
        parent_root,
        expected_sha256=expected_parent_sha,
    )
    test_results_path = (
        parent_root / "phases" / "p03" / "a01" / "results.jsonl"
    )
    phase_attempts_path = parent_root / "phase-attempts.jsonl"
    source_paths = {
        "manifest": manifest_path,
        "merged": repair_root / "merged-results.jsonl",
        "parent": parent_results_path,
        "parent_phase_attempts": phase_attempts_path,
        "repair": repair_root / "raw" / "repair-results.jsonl",
        "test": test_results_path,
    }
    audit = audit_offline_evidence(
        merged_results_path=source_paths["merged"],
        parent_results_path=source_paths["parent"],
        repair_results_path=source_paths["repair"],
        repair_manifest_path=source_paths["manifest"],
        test_results_path=source_paths["test"],
        parent_phase_attempts_path=source_paths["parent_phase_attempts"],
    )
    output_manifest = write_deliverables(
        audit=audit,
        output_dir=output_dir,
        source_paths=source_paths,
    )
    summary = {
        "status": output_manifest["status"],
        "output_dir": str(output_dir),
        "coverage_80": (
            f"{audit['coverage_80_count']}/{audit['sample_count']}"
        ),
        "coverage_99_diagnostic": (
            f"{audit['coverage_99_count']}/{audit['sample_count']}"
        ),
        "mean": audit["mean"],
        "test_coverage_80": (
            f"{audit['test_q24']['coverage_80_count']}/"
            f"{audit['test_q24']['sample_count']}"
        ),
        "test_mean": audit["test_q24"]["mean"],
        "overall_scene_coverage_80": (
            f"{audit['overall']['scene_80_count']}/"
            f"{audit['overall']['scene_count']}"
        ),
        "split_means_coverage_80": (
            f"{audit['overall']['split_mean_80_count']}/"
            f"{audit['overall']['split_mean_count']}"
        ),
        "safety_violations": audit["safety"]["total_violation_count"],
        "test_safety_violations": (
            audit["test_q24"]["safety"]["total_violation_count"]
        ),
    }
    sys.stdout.buffer.write(_pretty_json_bytes(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
