"""Deterministically freeze policy-blind midterm scenario cohorts.

The module deliberately has no policy, checkpoint, reward, coverage-result, or
planner-runtime dependency.  Writing the D: input artifact requires an explicit
``--execute`` invocation; importing it and running its tests are read-only.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

import xunce_artifact_io as artifact_io
from lunar_exploration_ppo.utils.artifact_io import ArtifactStore


SCHEMA_VERSION = "mid-dual-scenario-freeze/v1"
DENOMINATOR_SOURCE = "reachable_observable_free_highres_cells/v1"
DENOMINATOR_ALGORITHM = "exact_reachable_safe_pose_range_los/v1"
DESCRIPTOR_FIELDS = (
    "scenario_id",
    "scenario_hash",
    "slope_p90_deg",
    "hard_obstacle_fraction",
    "start_pose_bin",
    "initial_observed_coverable_fraction",
    "initial_valid_frontier_count",
    "coverable_cell_count",
    "parent_roi",
    "density_profile",
)
NUMERIC_FIELDS = (
    "slope_p90_deg",
    "hard_obstacle_fraction",
    "initial_observed_coverable_fraction",
    "initial_valid_frontier_count",
    "coverable_cell_count",
)
FORBIDDEN_FIELD_FRAGMENTS = (
    "coverage_result",
    "final_coverage",
    "policy",
    "checkpoint",
    "planner_success",
    "runtime",
    "reward",
)
_COHORT_SIZES = {
    "test_q24": 24,
    "test_c24": 24,
    "unseen24": 24,
    "g3_test_q5": 5,
    "g3_unseen5": 5,
    "validation3": 3,
    "replay3": 3,
}


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _require_finite_number(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{field_name} must be a finite number")
    return float(value)


def _factor_value(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def extract_policy_blind_descriptor(candidate: Mapping[str, object]) -> dict[str, object]:
    """Normalize only static/reset-time fields and reject outcome leakage."""

    if not isinstance(candidate, Mapping):
        raise ValueError("scenario descriptor candidate must be a mapping")
    for key in candidate:
        if not isinstance(key, str):
            raise ValueError("scenario descriptor key must be a string")
        if any(fragment in key.lower() for fragment in FORBIDDEN_FIELD_FRAGMENTS):
            raise ValueError(f"forbidden policy or outcome descriptor field: {key}")
    missing = [field for field in DESCRIPTOR_FIELDS if field not in candidate]
    if missing:
        raise ValueError(f"scenario descriptor is missing: {','.join(missing)}")
    scenario_id = candidate["scenario_id"]
    scenario_hash = candidate["scenario_hash"]
    if not isinstance(scenario_id, str) or not scenario_id:
        raise ValueError("scenario_id must be non-empty")
    if (
        not isinstance(scenario_hash, str)
        or len(scenario_hash) != 64
        or any(char not in "0123456789abcdefABCDEF" for char in scenario_hash)
    ):
        raise ValueError("scenario_hash must be a SHA-256 digest")
    pose = candidate["start_pose_bin"]
    if not isinstance(pose, (tuple, list)) or len(pose) != 3 or any(type(value) is not int for value in pose):
        raise ValueError("start_pose_bin must be three integer bins")
    parent_roi = candidate["parent_roi"]
    density_profile = candidate["density_profile"]
    if not isinstance(parent_roi, str) or not parent_roi or not isinstance(density_profile, str) or not density_profile:
        raise ValueError("parent_roi and density_profile must be non-empty")
    frontier_count = candidate["initial_valid_frontier_count"]
    coverable_count = candidate["coverable_cell_count"]
    if type(frontier_count) is not int or type(coverable_count) is not int:
        raise ValueError("static count fields must be exact integers")
    descriptor: dict[str, object] = {
        "scenario_id": scenario_id,
        "scenario_hash": scenario_hash.lower(),
        "slope_p90_deg": _require_finite_number(candidate["slope_p90_deg"], "slope_p90_deg"),
        "hard_obstacle_fraction": _require_finite_number(candidate["hard_obstacle_fraction"], "hard_obstacle_fraction"),
        "start_pose_bin": list(pose),
        "initial_observed_coverable_fraction": _require_finite_number(candidate["initial_observed_coverable_fraction"], "initial_observed_coverable_fraction"),
        "initial_valid_frontier_count": frontier_count,
        "coverable_cell_count": coverable_count,
        "parent_roi": parent_roi,
        "density_profile": density_profile,
    }
    if (
        descriptor["slope_p90_deg"] < 0.0
        or not 0.0 <= float(descriptor["hard_obstacle_fraction"]) <= 1.0
        or not 0.0 <= float(descriptor["initial_observed_coverable_fraction"]) <= 1.0
        or descriptor["initial_valid_frontier_count"] < 0
        or descriptor["coverable_cell_count"] <= 0
    ):
        raise ValueError("static count fields are invalid")
    return descriptor


def assign_stable_rank_tertiles(rows: Sequence[Mapping[str, object]]) -> dict[str, dict[str, int]]:
    """Assign rank tertiles after sorting each numeric field by value then ID."""

    descriptors = [extract_policy_blind_descriptor(row) for row in rows]
    if len({str(row["scenario_id"]) for row in descriptors}) != len(descriptors):
        raise ValueError("scenario IDs must be unique")
    bins = {str(row["scenario_id"]): {} for row in descriptors}
    for field in NUMERIC_FIELDS:
        ordered = sorted(descriptors, key=lambda row: (float(row[field]), str(row["scenario_id"])))
        for index, row in enumerate(ordered):
            bins[str(row["scenario_id"])][field] = min(2, (3 * index) // len(ordered))
    return bins


def _selection_score(
    candidate: Mapping[str, object],
    selected: Sequence[Mapping[str, object]],
    rank_bins: Mapping[str, Mapping[str, int]],
    selection_seed: str,
    *,
    include_distance_bin: bool,
) -> tuple[int, int, int, int, int, str]:
    all_rows = [*selected, candidate]
    factor_counts: dict[tuple[str, str], int] = {}
    stratum_counts: dict[tuple[str, ...], int] = {}
    factor_names = [*NUMERIC_FIELDS, "start_pose_bin"]
    if include_distance_bin:
        factor_names.append("start_to_farthest_candidate_distance_bin")
    for row in all_rows:
        scenario_id = str(row["scenario_id"])
        values = []
        for field in factor_names:
            value: object = rank_bins[scenario_id][field] if field in NUMERIC_FIELDS else row[field]
            encoded = _factor_value(value)
            factor_counts[(field, encoded)] = factor_counts.get((field, encoded), 0) + 1
            values.append(encoded)
        stratum = tuple(values)
        stratum_counts[stratum] = stratum_counts.get(stratum, 0) + 1
    parent_count = sum(row["parent_roi"] == candidate["parent_roi"] for row in all_rows)
    density_count = sum(row["density_profile"] == candidate["density_profile"] for row in all_rows)
    return (
        max(factor_counts.values()),
        sum(count * count for count in factor_counts.values()),
        sum(count - 1 for count in stratum_counts.values() if count > 1),
        parent_count,
        density_count,
        _sha256_text(f"{selection_seed}:{candidate['scenario_id']}"),
    )


def _select(
    candidates: Iterable[Mapping[str, object]],
    *,
    count: int,
    selection_seed: str,
    include_distance_bin: bool = False,
) -> list[str]:
    source_rows = list(candidates)
    normalized = [extract_policy_blind_descriptor(row) for row in source_rows]
    if include_distance_bin:
        for source, row in zip(source_rows, normalized, strict=True):
            raw = source.get("start_to_farthest_candidate_distance_bin")
            if type(raw) is not int or raw < 0:
                raise ValueError("G3 static distance bin is missing")
            row["start_to_farthest_candidate_distance_bin"] = raw
    if len(normalized) < count:
        raise ValueError("not enough policy-blind scenarios to freeze cohort")
    ranks = assign_stable_rank_tertiles(normalized)
    selected: list[dict[str, object]] = []
    remaining = {str(row["scenario_id"]): row for row in normalized}
    while len(selected) < count:
        chosen = min(
            remaining.values(),
            key=lambda row: _selection_score(row, selected, ranks, selection_seed, include_distance_bin=include_distance_bin),
        )
        selected.append(chosen)
        del remaining[str(chosen["scenario_id"])]
    return [str(row["scenario_id"]) for row in selected]


def freeze_selection(rows: Sequence[Mapping[str, object]], *, selection_seed: str) -> dict[str, list[str]]:
    """Choose all frozen cohorts from static descriptors with no result inputs."""

    descriptors = [extract_policy_blind_descriptor(row) for row in rows]
    if len({str(row["scenario_id"]) for row in descriptors}) != len(descriptors):
        raise ValueError("scenario IDs must be unique")
    raw_by_id = {str(row["scenario_id"]): row for row in rows}
    test_q24 = _select(rows, count=_COHORT_SIZES["test_q24"], selection_seed=selection_seed)
    q_ids = set(test_q24)
    remaining = [raw_by_id[str(row["scenario_id"])] for row in descriptors if str(row["scenario_id"]) not in q_ids]
    test_c24 = _select(remaining, count=_COHORT_SIZES["test_c24"], selection_seed=selection_seed)
    qc_ids = q_ids | set(test_c24)
    remaining = [row for row in remaining if str(row["scenario_id"]) not in set(test_c24)]
    unseen24 = _select(remaining, count=_COHORT_SIZES["unseen24"], selection_seed=selection_seed)
    used = qc_ids | set(unseen24)
    tail = [raw_by_id[str(row["scenario_id"])] for row in descriptors if str(row["scenario_id"]) not in used]
    validation3 = _select(tail, count=_COHORT_SIZES["validation3"], selection_seed=selection_seed)
    replay_pool = [row for row in tail if str(row["scenario_id"]) not in set(validation3)]
    replay3 = _select(replay_pool, count=_COHORT_SIZES["replay3"], selection_seed=selection_seed)
    g3_test_q5 = _select([raw_by_id[scenario_id] for scenario_id in test_q24], count=_COHORT_SIZES["g3_test_q5"], selection_seed=selection_seed, include_distance_bin=True)
    g3_unseen5 = _select([raw_by_id[scenario_id] for scenario_id in unseen24], count=_COHORT_SIZES["g3_unseen5"], selection_seed=selection_seed, include_distance_bin=True)
    return {
        "test_q24": test_q24,
        "test_c24": test_c24,
        "unseen24": unseen24,
        "g3_test_q5": g3_test_q5,
        "g3_unseen5": g3_unseen5,
        "validation3": validation3,
        "replay3": replay3,
    }


def prove_denominator_identity(audit: Mapping[str, object], reconstructed_mask: np.ndarray) -> dict[str, object]:
    """Prove semantic aliasing only when exact mask bytes/count/hash agree."""

    mask = np.asarray(reconstructed_mask)
    if mask.dtype != np.dtype(bool) or mask.ndim != 2:
        raise ValueError("denominator reconstruction must be a 2D bool mask")
    geometry = audit.get("geometry")
    if (
        not isinstance(geometry, Mapping)
        or type(geometry.get("width")) is not int
        or type(geometry.get("height")) is not int
        or mask.shape != (geometry["height"], geometry["width"])
    ):
        raise ValueError("denominator geometry drift")
    bytes_hash = hashlib.sha256(np.ascontiguousarray(mask).tobytes()).hexdigest()
    count = int(np.count_nonzero(mask))
    if count <= 0 or audit.get("coverable_cell_count") != count or audit.get("coverable_mask_sha256") != bytes_hash:
        raise ValueError("denominator identity drift")
    if audit.get("algorithm_id") != DENOMINATOR_ALGORITHM or audit.get("exact") is not True:
        raise ValueError("denominator algorithm drift")
    return {
        "scenario_id": audit.get("scenario_id"),
        "scenario_hash": audit.get("scenario_hash"),
        "coverable_mask_sha256": bytes_hash,
        "coverable_cell_count": count,
        "geometry": audit.get("geometry"),
        "coverage_denominator_source": DENOMINATOR_SOURCE,
        "coverage_denominator_algorithm": DENOMINATOR_ALGORITHM,
        "semantic_alias_proven": True,
    }


def build_frozen_manifest(
    rows: Sequence[Mapping[str, object]],
    *,
    config: Mapping[str, object],
    denominator_audits: Sequence[tuple[Mapping[str, object], np.ndarray]] | None = None,
) -> bytes:
    """Return canonical manifest bytes, failing closed before any write on drift."""

    if config.get("schema_version") != SCHEMA_VERSION or not isinstance(config.get("selection_seed"), str) or not config["selection_seed"]:
        raise ValueError("scenario freeze config is invalid")
    descriptors = sorted((extract_policy_blind_descriptor(row) for row in rows), key=lambda row: str(row["scenario_id"]))
    selection = freeze_selection(rows, selection_seed=str(config["selection_seed"]))
    proofs = [] if denominator_audits is None else [prove_denominator_identity(audit, mask) for audit, mask in denominator_audits]
    if denominator_audits is not None and {str(proof["scenario_id"]) for proof in proofs} != {str(row["scenario_id"]) for row in descriptors}:
        raise ValueError("denominator audit set does not bind every frozen candidate")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "selection_seed": config["selection_seed"],
        "descriptor_sha256": hashlib.sha256(ArtifactStore.canonical_json_bytes(descriptors)).hexdigest(),
        "cohort_sizes": dict(_COHORT_SIZES),
        "cohorts": selection,
        "denominator_audits": sorted(proofs, key=lambda proof: str(proof["scenario_id"])),
    }
    return ArtifactStore.canonical_json_bytes(payload)


def _load_json_config(path: str | Path) -> dict[str, Any]:
    return artifact_io.read_json(path)


def _freeze_id(manifest_bytes: bytes) -> str:
    return hashlib.sha256(manifest_bytes).hexdigest()[:16]


def _load_denominator_audits(path: str | Path) -> list[tuple[Mapping[str, object], np.ndarray]]:
    """Read explicit reconstruction evidence without loading a policy or checkpoint."""

    evidence: list[tuple[Mapping[str, object], np.ndarray]] = []
    for row in artifact_io.read_jsonl(path):
        mask_path = row.pop("reconstructed_mask_path", None)
        if not isinstance(mask_path, str) or not mask_path:
            raise ValueError("denominator evidence must name a reconstructed mask")
        try:
            with np.load(io.BytesIO(artifact_io.read_bytes(mask_path)), allow_pickle=False) as archive:
                if tuple(archive.files) != ("coverable_mask",):
                    raise ValueError("reconstructed denominator mask archive schema drifted")
                mask = np.array(archive["coverable_mask"], copy=True)
        except (OSError, ValueError) as exc:
            raise ValueError("reconstructed denominator mask is unreadable") from exc
        evidence.append((row, mask))
    return evidence


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--descriptor-catalog", required=True)
    parser.add_argument("--denominator-audits", required=True)
    parser.add_argument("--execute", action="store_true", help="permit writing the frozen D: input artifact")
    args = parser.parse_args(argv)
    if not args.execute:
        raise SystemExit("refusing to write scenario freeze without --execute")
    config = _load_json_config(args.config)
    rows = artifact_io.read_jsonl(args.descriptor_catalog)
    manifest_bytes = build_frozen_manifest(rows, config=config, denominator_audits=_load_denominator_audits(args.denominator_audits))
    freeze_root = Path("D:/xunce/inputs/mid_dual/scenarios") / _freeze_id(manifest_bytes)
    if artifact_io.path_exists(freeze_root):
        raise FileExistsError(f"freeze root already exists: {freeze_root}")
    artifact_io.make_dirs(freeze_root)
    artifact_io.write_text(freeze_root / "manifest.json", manifest_bytes.decode("utf-8"))
    artifact_io.write_jsonl(freeze_root / "descriptors.jsonl", sorted((extract_policy_blind_descriptor(row) for row in rows), key=lambda row: str(row["scenario_id"])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
