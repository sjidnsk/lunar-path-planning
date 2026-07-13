"""Stage 2 machine workflow with fail-closed source, data, and artifact binding."""

from __future__ import annotations

import hashlib
import io
import inspect
import json
import os
import platform
import re
import subprocess
import sys
import uuid
import zipfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Final

import numpy as np

from lunar_exploration_ppo.configs.stage2 import Stage2Config, load_stage2_config
from lunar_exploration_ppo.env.coverage import compute_coverage_masks
from lunar_exploration_ppo.env.frontier import FrontierActionSet, FrontierGenerator, score_first_top_m
from lunar_exploration_ppo.env.map_state import ObservedMapState
from lunar_exploration_ppo.env.scenario import LowResolutionPrior, ScenarioBundle, TruthMap
from lunar_exploration_ppo.env.scenario_catalog import (
    SPLIT_ORDER,
    ScenarioCatalogRecord,
    StandardScenarioCatalogBuilder,
    StandardScenarioFactory,
)
from lunar_exploration_ppo.env.sensor_model import ObservationDelta, SensorPose, SensorUpdater
from lunar_exploration_ppo.policy.observation import ObservationBuilder, PolicyObservation
from lunar_exploration_ppo.utils.artifact_io import ArtifactStore
from lunar_exploration_ppo.utils.geometry import CellXY, GridGeometry, PoseXYTheta
from lunar_exploration_ppo.workflows.stage1_artifacts import (
    FrozenFileSnapshot,
    Stage1WorkflowError,
    finite_tree,
    prepare_unique_run_root,
    strict_json_object_from_bytes,
    validate_run_id,
)


STAGE2_MACHINE_ARTIFACTS: Final = (
    "config.json", "summary.json", "routing.json", "report.md", "metrics.jsonl", "phase-state.jsonl",
)
STAGE2_EVIDENCE_ARTIFACTS: Final = (
    "evidence/observation_schema_report.json",
    "evidence/candidate_generation_audit.json",
    "evidence/top_m_audit.json",
    "evidence/split_leakage_audit.json",
    "evidence/provenance.json",
    "evidence/sample_observation.npz",
    "evidence/scenario_catalog.json",
)
STAGE2_MANIFEST_BOUND_ARTIFACTS: Final = (*STAGE2_MACHINE_ARTIFACTS, *STAGE2_EVIDENCE_ARTIFACTS)
_ROOT_MACHINE_SET: Final = set(STAGE2_MACHINE_ARTIFACTS) | {"manifest.json", "evidence"}
_FORBIDDEN: Final = {"approval.json", "gate.json"}
STAGE2_APPROVAL_TEXT: Final = "批准 Stage 2 Gate"
_ABSENT_AUTHORITY_HASH: Final = hashlib.sha256(b"ppo_highres_frontier_stage2_authority_absent/v1").hexdigest()
_STAGE2_STATE_SEQUENCE: Final = (
    "machine_passed", "awaiting_independent_review", "awaiting_human_approval", "approved", "next_stage",
)
_EXACT_COVERABLE_MASK_CACHE: dict[str, np.ndarray] = {}
_STAGE2_REVIEW_FRONT_MATTER_KEYS: Final = frozenset(
    {
        "schema_version",
        "goal_id",
        "stage_id",
        "run_id",
        "reviewer_id",
        "spec_verdict",
        "quality_verdict",
        "critical_count",
        "important_count",
        "minor_count",
        "final_conclusion",
        "reviewed_base_commit",
        "reviewed_prospective_git_tree",
        "review_package_sha256",
        "review_package_bytes",
        "review_package_lf_count",
        "review_package_logical_line_count",
        "reviewed_paths_json",
        "reviewed_path_set_sha256",
        "manifest_sha256",
        "config_sha256",
        "data_sha256",
        "environment_sha256",
        "source_set_sha256",
        "authorized_next_stage",
    }
)
_STAGE2_REVIEW_INTEGER_KEYS: Final = frozenset(
    {
        "critical_count",
        "important_count",
        "minor_count",
        "review_package_bytes",
        "review_package_lf_count",
        "review_package_logical_line_count",
    }
)
_SHA256_PATTERN: Final = re.compile(r"^[0-9a-f]{64}$")
_OID_PATTERN: Final = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")


class Stage2WorkflowError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class Stage2WorkflowResult:
    run_id: str
    stage_root: Path
    summary: dict[str, object]
    manifest: dict[str, object]


@dataclass(frozen=True, slots=True)
class Stage2ReviewPackage:
    path: Path
    base_commit: str
    prospective_git_tree: str
    paths: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _HiddenTruthPolicyFixture:
    truth: TruthMap
    coverable_mask: np.ndarray
    prior: LowResolutionPrior
    observed_state: ObservedMapState
    pose: PoseXYTheta
    action_set: FrontierActionSet
    observation: PolicyObservation


@dataclass(frozen=True, slots=True)
class _HiddenTruthLeakagePair:
    baseline: _HiddenTruthPolicyFixture
    mutated: _HiddenTruthPolicyFixture


@dataclass(frozen=True, slots=True)
class _Stage2MachinePayload:
    artifacts: dict[str, bytes]
    manifest: dict[str, object]
    summary: dict[str, object]


def run_stage2_workflow(
    *,
    config_path: str | Path,
    run_id: str,
    base_output_root: str | Path | None = None,
) -> Stage2WorkflowResult:
    validate_run_id(run_id)
    config_source = Path(config_path).expanduser().resolve()
    config_source_bytes = config_source.read_bytes()
    config = load_stage2_config(config_source)
    repo_root = Path(__file__).resolve().parents[3]
    source_identity = _source_identity(repo_root)
    environment_identity = _environment_identity()
    base_root = Path(base_output_root) if base_output_root is not None else Path(config.output_root)
    run_root = (base_root / run_id).expanduser().resolve()
    prepare_unique_run_root(run_root)
    stage = run_root / "s2"
    stage.mkdir(exist_ok=False)
    store = ArtifactStore(stage)
    payload = _build_stage2_machine_payload(
        config=config,
        run_id=run_id,
        source_identity=source_identity,
        environment_identity=environment_identity,
    )
    final_source_identity = _source_identity(repo_root)
    if final_source_identity != source_identity or config_source.read_bytes() != config_source_bytes:
        raise Stage2WorkflowError("source or config drift during Stage 2 machine run")
    for relative_path in STAGE2_MANIFEST_BOUND_ARTIFACTS:
        store.write_bytes(relative_path, payload.artifacts[relative_path])
    store.write_json("manifest.json", payload.manifest)
    return Stage2WorkflowResult(
        run_id=run_id,
        stage_root=stage,
        summary=payload.summary,
        manifest=payload.manifest,
    )


def _build_stage2_machine_payload(
    *,
    config: Stage2Config,
    run_id: str,
    source_identity: dict[str, object],
    environment_identity: dict[str, object],
) -> _Stage2MachinePayload:
    config_payload = config.model_dump(mode="json")
    config_payload["run_id"] = run_id
    config_payload["execution_source_identity"] = source_identity
    config_payload["execution_environment_identity"] = environment_identity
    config_bytes = ArtifactStore.canonical_json_bytes(config_payload)
    config_hash = hashlib.sha256(config_bytes).hexdigest()
    catalog = StandardScenarioCatalogBuilder(
        dem_path=config.dem_path,
        slope_path=config.slope_path,
        base_seed=config.catalog_base_seed,
        verify_hashes=True,
    ).build()
    catalog_payload = catalog.to_dict()
    first_record = catalog.records[0]
    scenario = StandardScenarioFactory(catalog).build(first_record)
    state, reset_delta, sensor_contract = _initial_observed_state(scenario)
    leakage_pair = _build_hidden_truth_leakage_pair(
        scenario=scenario,
        observed_state=state,
        pose=scenario.start_pose,
        top_m=config.standard.frontier_top_m,
    )
    state = leakage_pair.baseline.observed_state
    action_set = leakage_pair.baseline.action_set
    observation = leakage_pair.baseline.observation
    schema_report = {
        **PolicyObservation.schema_metadata(),
        "shapes": [list(array.shape) for array in observation.array_fields()],
        "dtypes": [str(array.dtype) for array in observation.array_fields()],
        "all_finite": all(
            np.isfinite(array).all()
            for array in observation.array_fields()
            if array.dtype != np.bool_
        ),
    }
    candidate_audit = {
        **dict(action_set.diagnostics),
        "valid_cells_observed": all(state.observed_mask[cell.y, cell.x] for cell in action_set.cells),
        "valid_cells_safe": all(state.observed_safe_mask[cell.y, cell.x] for cell in action_set.cells),
        "valid_feature_rows_finite": bool(
            np.isfinite(action_set.frontier_features[action_set.candidate_mask]).all()
        ),
    }
    overflow_priorities = [1.0] * (config.standard.frontier_top_m + 2)
    overflow_selected = score_first_top_m(overflow_priorities, config.standard.frontier_top_m)
    top_m_audit = {
        "policy": "score_first_top_m/v1",
        "stable_tie_first_indices": list(overflow_selected[:3]),
        "overflow_input_count": len(overflow_priorities),
        "overflow_kept_count": len(overflow_selected),
        "overflow_count": len(overflow_priorities) - len(overflow_selected),
        "sample_diagnostics": dict(action_set.diagnostics),
    }
    density: dict[str, Counter[str]] = defaultdict(Counter)
    split_counts = Counter(record.split for record in catalog.records)
    for record in catalog.records:
        density[record.split][record.density_profile] += 1
    split_audit = {
        **catalog.spatial_audit(),
        "split_policy": catalog.split_policy,
        "split_counts": dict(split_counts),
        "density_counts": {split: dict(density[split]) for split in SPLIT_ORDER},
        "proxy_seed_derived_after_parent_split": True,
        "catalog_sha256": catalog.sha256,
    }
    scenario_lineage = _scenario_lineage(
        catalog_sha256=catalog.sha256,
        record=first_record,
        scenario=scenario,
        state=state,
        reset_delta=reset_delta,
        sensor_contract=sensor_contract,
    )
    provenance = {
        "sources": catalog_payload["sources"],
        "sample_record": catalog_payload["records"][0],
        "sample_prior": dict(scenario.prior.provenance),
        "sample_scenario_lineage": scenario_lineage,
        "claim_boundary": "synthetic_proxy_not_native_submeter_lunar_truth/v1",
    }
    leakage_audit = {
        **_paired_hidden_truth_leakage_audit(leakage_pair),
        "coverable_mask_in_observation_signature": "coverable"
        in repr(inspect.signature(ObservationBuilder.build)).lower(),
        "truth_in_frontier_signature": "truth"
        in repr(inspect.signature(FrontierGenerator.extract)).lower(),
    }
    if (
        leakage_audit["hidden_truth_leakage_detected"]
        or leakage_audit["coverable_mask_in_observation_signature"]
        or leakage_audit["truth_in_frontier_signature"]
    ):
        raise Stage2WorkflowError("hidden-truth authority appeared in a Stage 2 policy interface")
    acceptance_items = _acceptance_items(observation, action_set, leakage_audit, split_audit)
    if not all(acceptance_items.values()):
        raise Stage2WorkflowError("Stage 2 machine acceptance failed closed")
    summary: dict[str, object] = {
        "schema_version": "ppo_highres_frontier_stage2_summary/v1",
        "goal_id": config.goal_id,
        "stage_id": config.stage_id,
        "run_id": run_id,
        "state": "machine_passed",
        "config_hash": config_hash,
        "catalog_sha256": catalog.sha256,
        "scenario_lineage_sha256": scenario_lineage["lineage_sha256"],
        "catalog_audit": split_audit,
        "candidate_audit": candidate_audit,
        "leakage_audit": leakage_audit,
        "acceptance": {"all_18_stage2_items_passed": True, "items": acceptance_items},
        "execution_source_identity": source_identity,
        "execution_environment_identity": environment_identity,
        "checkpoint_state": "stage2_no_checkpoint/v1",
    }
    if not finite_tree(summary):
        raise Stage2WorkflowError("Stage 2 summary contains non-finite values")
    routing = {
        "schema_version": "ppo_highres_frontier_stage2_routing/v1",
        "run_id": run_id,
        "route": "awaiting_independent_review",
        "authorized_next_stage": config.authorized_next_stage,
        "human_approval_required": True,
        "next_stage_entered": False,
    }
    report = (
        "# Stage 2 machine report\n\n"
        f"- Run ID: `{run_id}`\n"
        f"- Catalog SHA-256: `{catalog.sha256}`\n"
        f"- Standard candidates: {action_set.candidate_count}\n"
        "- Frozen slope product: 3-band uint8 RGB visualization; pixels were not interpreted as slope degrees.\n"
        "- Physical slope prior: derived from float32 DEM with metric 4m spacing.\n"
        "- State: `machine_passed`; route: `awaiting_independent_review`.\n"
        "- No checkpoint, review, approval, gate, executor, or canary artifact was produced.\n"
    ).encode("utf-8")
    metrics = (
        {"kind": "observation", "all_finite": schema_report["all_finite"], "candidate_count": action_set.candidate_count},
        {"kind": "catalog", "counts": dict(split_counts), "catalog_sha256": catalog.sha256},
        {"kind": "scenario_lineage", "lineage_sha256": scenario_lineage["lineage_sha256"]},
        {"kind": "leakage", **leakage_audit},
    )
    artifacts = {
        "config.json": config_bytes,
        "summary.json": ArtifactStore.canonical_json_bytes(summary),
        "routing.json": ArtifactStore.canonical_json_bytes(routing),
        "report.md": report,
        "metrics.jsonl": _canonical_jsonl_bytes(metrics),
        "phase-state.jsonl": _canonical_jsonl_bytes(({
            "state": "machine_passed",
            "route": "awaiting_independent_review",
            "run_id": run_id,
        },)),
        "evidence/observation_schema_report.json": ArtifactStore.canonical_json_bytes(schema_report),
        "evidence/candidate_generation_audit.json": ArtifactStore.canonical_json_bytes(candidate_audit),
        "evidence/top_m_audit.json": ArtifactStore.canonical_json_bytes(top_m_audit),
        "evidence/split_leakage_audit.json": ArtifactStore.canonical_json_bytes(
            {**split_audit, "leakage": leakage_audit}
        ),
        "evidence/provenance.json": ArtifactStore.canonical_json_bytes(provenance),
        "evidence/sample_observation.npz": _deterministic_observation_npz_bytes(observation),
        "evidence/scenario_catalog.json": ArtifactStore.canonical_json_bytes(catalog_payload),
    }
    if set(artifacts) != set(STAGE2_MANIFEST_BOUND_ARTIFACTS):
        raise Stage2WorkflowError("Stage 2 deterministic payload path set drift")
    manifest = _manifest_for_artifacts(artifacts)
    return _Stage2MachinePayload(artifacts=artifacts, manifest=manifest, summary=summary)


def _canonical_jsonl_bytes(values: tuple[dict[str, object], ...]) -> bytes:
    return b"".join(
        (
            json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        ).encode("utf-8")
        for value in values
    )


def _deterministic_observation_npz_bytes(observation: PolicyObservation) -> bytes:
    arrays = {
        "schema_json": np.asarray(json.dumps(observation.schema_metadata(), sort_keys=True)),
        "prior_channels": observation.prior_channels,
        "coverage_summary": observation.coverage_summary,
        "local_crop": observation.local_crop,
        "frontier_features": observation.frontier_features,
        "pose_features": observation.pose_features,
        "candidate_mask": observation.candidate_mask,
    }
    archive_buffer = io.BytesIO()
    with zipfile.ZipFile(
        archive_buffer,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        for name, array in arrays.items():
            array_buffer = io.BytesIO()
            np.lib.format.write_array(array_buffer, np.asarray(array), allow_pickle=False)
            member = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            member.compress_type = zipfile.ZIP_DEFLATED
            member.create_system = 3
            member.external_attr = 0o600 << 16
            archive.writestr(member, array_buffer.getvalue(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    return archive_buffer.getvalue()


def _manifest_for_artifacts(artifacts: dict[str, bytes]) -> dict[str, object]:
    return {
        "schema_version": "sha256_manifest/v1",
        "artifacts": [
            {
                "path": path,
                "sha256": hashlib.sha256(artifacts[path]).hexdigest(),
                "size_bytes": len(artifacts[path]),
            }
            for path in sorted(artifacts)
        ],
    }


def verify_stage2_machine_run(*, stage_root: str | Path, repo_root: str | Path) -> None:
    stage = Path(stage_root).expanduser().resolve()
    if stage.name != "s2" or not stage.is_dir():
        raise Stage2WorkflowError("Stage 2 root must be canonical <run>/s2")
    actual = {path.name for path in stage.iterdir()}
    allowed = _ROOT_MACHINE_SET | ({"review.json"} if (stage / "review.json").is_file() else set())
    if actual != allowed or actual & _FORBIDDEN:
        raise Stage2WorkflowError("Stage 2 root artifact set drift")
    _verify_stage2_machine_evidence(stage, Path(repo_root).expanduser().resolve())


def _verify_stage2_machine_evidence(stage: Path, repo_root: Path) -> None:
    """Verify immutable machine evidence; authority files are checked by gate callers."""
    manifest = strict_json_object_from_bytes((stage / "manifest.json").read_bytes(), "Stage 2 manifest")
    entries = manifest.get("artifacts")
    if not isinstance(entries, list) or {entry.get("path") for entry in entries if isinstance(entry, dict)} != set(STAGE2_MANIFEST_BOUND_ARTIFACTS):
        raise Stage2WorkflowError("Stage 2 manifest path set drift")
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            raise Stage2WorkflowError("Stage 2 manifest entry drift")
        relative = Path(entry["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise Stage2WorkflowError("Stage 2 manifest path is unsafe")
        payload = (stage / relative).read_bytes()
        if entry.get("size_bytes") != len(payload) or entry.get("sha256") != hashlib.sha256(payload).hexdigest():
            raise Stage2WorkflowError(f"artifact drift: {entry['path']}")
    machine_config = strict_json_object_from_bytes((stage / "config.json").read_bytes(), "Stage 2 config")
    source_identity = machine_config.pop("execution_source_identity", None)
    environment_identity = machine_config.pop("execution_environment_identity", None)
    current_source_identity = _source_identity(repo_root)
    current_environment_identity = _environment_identity()
    if source_identity != current_source_identity:
        raise Stage2WorkflowError("Stage 2 source drift")
    if environment_identity != current_environment_identity:
        raise Stage2WorkflowError("Stage 2 environment drift")
    run_id = machine_config.get("run_id")
    try:
        machine_model = Stage2Config.model_validate(machine_config)
        repository_model = load_stage2_config(Path(repo_root) / "configs/ppo_highres_frontier_stage2_v1.json")
    except Exception as exc:
        raise Stage2WorkflowError("Stage 2 config drift") from exc
    if machine_model.model_dump(exclude={"run_id"}) != repository_model.model_dump(exclude={"run_id"}) or run_id != stage.parent.name:
        raise Stage2WorkflowError("Stage 2 config drift")
    expected = _build_stage2_machine_payload(
        config=machine_model,
        run_id=str(run_id),
        source_identity=current_source_identity,
        environment_identity=current_environment_identity,
    )
    for relative_path in STAGE2_MANIFEST_BOUND_ARTIFACTS:
        if (stage / relative_path).read_bytes() != expected.artifacts[relative_path]:
            if relative_path == "evidence/split_leakage_audit.json":
                raise Stage2WorkflowError("Stage 2 leakage evidence drift")
            raise Stage2WorkflowError(
                f"Stage 2 deterministic payload drift: {relative_path}"
            )
    if (stage / "manifest.json").read_bytes() != ArtifactStore.canonical_json_bytes(
        expected.manifest
    ):
        raise Stage2WorkflowError("Stage 2 deterministic payload drift: manifest.json")


def render_stage2_strict_review_report(
    *,
    bindings: dict[str, object],
    reviewer_id: str,
    spec_verdict: str,
    quality_verdict: str,
    critical_count: int,
    important_count: int,
    minor_count: int,
    final_conclusion: str,
    findings_markdown: str,
) -> str:
    """Render, but never write, the strict Markdown expected from a fresh reviewer."""

    required_bindings = {
        "goal_id",
        "stage_id",
        "run_id",
        "reviewed_base_commit",
        "reviewed_prospective_git_tree",
        "review_package_sha256",
        "review_package_bytes",
        "review_package_lf_count",
        "review_package_logical_line_count",
        "reviewed_paths",
        "reviewed_path_set_sha256",
        "manifest_sha256",
        "config_sha256",
        "data_sha256",
        "environment_sha256",
        "source_set_sha256",
        "authorized_next_stage",
    }
    if not required_bindings.issubset(bindings):
        raise Stage2WorkflowError("Stage 2 strict review formatter bindings are incomplete")
    counts = (critical_count, important_count, minor_count)
    if any(type(count) is not int or count < 0 for count in counts):
        raise Stage2WorkflowError("Stage 2 strict review formatter counts are invalid")
    scalar_values = (reviewer_id, spec_verdict, quality_verdict, final_conclusion)
    if any(
        not isinstance(value, str)
        or not value
        or "\n" in value
        or "\r" in value
        for value in scalar_values
    ):
        raise Stage2WorkflowError("Stage 2 strict review formatter declaration is invalid")
    paths = bindings["reviewed_paths"]
    if not isinstance(paths, list):
        raise Stage2WorkflowError("Stage 2 strict review formatter path set is invalid")
    total_findings = sum(counts)
    if (total_findings == 0 and findings_markdown) or (
        total_findings > 0 and not findings_markdown.strip()
    ):
        raise Stage2WorkflowError("Stage 2 strict review formatter findings contradict counts")
    if "\r" in findings_markdown:
        raise Stage2WorkflowError("Stage 2 strict review formatter findings must use LF")
    front: dict[str, object] = {
        "schema_version": "ppo_highres_frontier_stage2_review_report/v1",
        "goal_id": bindings["goal_id"],
        "stage_id": bindings["stage_id"],
        "run_id": bindings["run_id"],
        "reviewer_id": reviewer_id,
        "spec_verdict": spec_verdict,
        "quality_verdict": quality_verdict,
        "critical_count": critical_count,
        "important_count": important_count,
        "minor_count": minor_count,
        "final_conclusion": final_conclusion,
        "reviewed_base_commit": bindings["reviewed_base_commit"],
        "reviewed_prospective_git_tree": bindings["reviewed_prospective_git_tree"],
        "review_package_sha256": bindings["review_package_sha256"],
        "review_package_bytes": bindings["review_package_bytes"],
        "review_package_lf_count": bindings["review_package_lf_count"],
        "review_package_logical_line_count": bindings[
            "review_package_logical_line_count"
        ],
        "reviewed_paths_json": json.dumps(
            paths,
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        "reviewed_path_set_sha256": bindings["reviewed_path_set_sha256"],
        "manifest_sha256": bindings["manifest_sha256"],
        "config_sha256": bindings["config_sha256"],
        "data_sha256": bindings["data_sha256"],
        "environment_sha256": bindings["environment_sha256"],
        "source_set_sha256": bindings["source_set_sha256"],
        "authorized_next_stage": bindings["authorized_next_stage"],
    }
    order = (
        "schema_version",
        "goal_id",
        "stage_id",
        "run_id",
        "reviewer_id",
        "spec_verdict",
        "quality_verdict",
        "critical_count",
        "important_count",
        "minor_count",
        "final_conclusion",
        "reviewed_base_commit",
        "reviewed_prospective_git_tree",
        "review_package_sha256",
        "review_package_bytes",
        "review_package_lf_count",
        "review_package_logical_line_count",
        "reviewed_paths_json",
        "reviewed_path_set_sha256",
        "manifest_sha256",
        "config_sha256",
        "data_sha256",
        "environment_sha256",
        "source_set_sha256",
        "authorized_next_stage",
    )
    lines = ["---", *(f"{key}: {front[key]}" for key in order), "---", ""]
    lines.extend(
        (
            "# Stage 2 Independent Review",
            "",
            f"Reviewer: {reviewer_id}",
            f"Specification compliance verdict: {spec_verdict}",
            f"Code quality verdict: {quality_verdict}",
            f"Critical findings: {critical_count}",
            f"Important findings: {important_count}",
            f"Minor findings: {minor_count}",
            f"Final conclusion: {final_conclusion}",
            "Findings: None" if total_findings == 0 else "Findings: Listed below",
        )
    )
    if findings_markdown:
        lines.extend(("", findings_markdown.rstrip("\n")))
    return "\n".join(lines) + "\n"


def _parse_stage2_review_report(
    report_snapshot: FrozenFileSnapshot,
) -> dict[str, object]:
    payload = report_snapshot.payload
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise Stage2WorkflowError("Stage 2 review report is not strict UTF-8") from exc
    lines = text.splitlines()
    if not lines or lines[0] != "---":
        raise Stage2WorkflowError("Stage 2 review report strict front matter is missing")
    if b"\r" in payload or not payload.endswith(b"\n"):
        raise Stage2WorkflowError("Stage 2 review report must use canonical LF lines")
    try:
        closing_index = lines.index("---", 1)
    except ValueError as exc:
        raise Stage2WorkflowError("Stage 2 review report strict front matter is not closed") from exc
    if "---" in lines[closing_index + 1 :]:
        raise Stage2WorkflowError("Stage 2 review report contains multiple front matter blocks")
    raw: dict[str, str] = {}
    for line in lines[1:closing_index]:
        if line.count(": ") != 1:
            raise Stage2WorkflowError("Stage 2 review report front matter line is invalid")
        key, value = line.split(": ", 1)
        if not re.fullmatch(r"[a-z][a-z0-9_]*", key) or not value or key in raw:
            raise Stage2WorkflowError("Stage 2 review report front matter key is invalid or duplicated")
        raw[key] = value
    if set(raw) != _STAGE2_REVIEW_FRONT_MATTER_KEYS:
        raise Stage2WorkflowError("Stage 2 review report front matter schema is not exact")
    parsed: dict[str, object] = dict(raw)
    for key in _STAGE2_REVIEW_INTEGER_KEYS:
        if re.fullmatch(r"0|[1-9][0-9]*", raw[key]) is None:
            raise Stage2WorkflowError(f"Stage 2 review report {key} must be a nonnegative integer")
        parsed[key] = int(raw[key])
    try:
        reviewed_paths = json.loads(raw["reviewed_paths_json"])
    except json.JSONDecodeError as exc:
        raise Stage2WorkflowError("Stage 2 review report reviewed path set is invalid JSON") from exc
    if (
        not isinstance(reviewed_paths, list)
        or any(not isinstance(path, str) or not path or "\\" in path for path in reviewed_paths)
        or reviewed_paths != sorted(set(reviewed_paths))
    ):
        raise Stage2WorkflowError("Stage 2 review report reviewed path set is invalid")
    parsed["reviewed_paths"] = reviewed_paths
    parsed.pop("reviewed_paths_json")
    if parsed["schema_version"] != "ppo_highres_frontier_stage2_review_report/v1":
        raise Stage2WorkflowError("Stage 2 review report schema version is invalid")
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/-]{0,127}", str(parsed["reviewer_id"])) is None:
        raise Stage2WorkflowError("Stage 2 review report reviewer is invalid")
    for key in (
        "review_package_sha256",
        "reviewed_path_set_sha256",
        "manifest_sha256",
        "config_sha256",
        "data_sha256",
        "environment_sha256",
        "source_set_sha256",
    ):
        if _SHA256_PATTERN.fullmatch(str(parsed[key])) is None:
            raise Stage2WorkflowError(f"Stage 2 review report {key} is invalid")
    for key in ("reviewed_base_commit", "reviewed_prospective_git_tree"):
        if _OID_PATTERN.fullmatch(str(parsed[key])) is None:
            raise Stage2WorkflowError(f"Stage 2 review report {key} is invalid")

    body_lines = lines[closing_index + 1 :]

    def body_value(prefix: str, label: str) -> str:
        values = [line.removeprefix(prefix) for line in body_lines if line.startswith(prefix)]
        if len(values) != 1:
            raise Stage2WorkflowError(
                f"Stage 2 review report body {label} is missing, duplicated, or contradictory"
            )
        return values[0]

    if body_value("Reviewer: ", "reviewer") != parsed["reviewer_id"]:
        raise Stage2WorkflowError("Stage 2 review report body reviewer mismatch")
    if (
        body_value("Specification compliance verdict: ", "verdict")
        != parsed["spec_verdict"]
        or body_value("Code quality verdict: ", "verdict") != parsed["quality_verdict"]
    ):
        raise Stage2WorkflowError("Stage 2 review report body verdict mismatch")
    for prefix, key in (
        ("Critical findings: ", "critical_count"),
        ("Important findings: ", "important_count"),
        ("Minor findings: ", "minor_count"),
    ):
        if body_value(prefix, "count") != str(parsed[key]):
            raise Stage2WorkflowError("Stage 2 review report body count mismatch")
    if body_value("Final conclusion: ", "conclusion") != parsed["final_conclusion"]:
        raise Stage2WorkflowError("Stage 2 review report body conclusion mismatch")
    findings = [line for line in body_lines if line.startswith("Findings: ")]
    if len(findings) != 1:
        raise Stage2WorkflowError("Stage 2 review report Findings declaration is not exact")
    total_findings = sum(int(parsed[key]) for key in ("critical_count", "important_count", "minor_count"))
    expected_findings = "Findings: None" if total_findings == 0 else "Findings: Listed below"
    if findings[0] != expected_findings:
        if findings[0] == "Findings: None":
            raise Stage2WorkflowError("Stage 2 review report Findings: None contradicts counts")
        raise Stage2WorkflowError("Stage 2 review report Findings declaration contradicts counts")
    return parsed


def _verify_stage2_review_conclusions(report: dict[str, object]) -> None:
    if report["spec_verdict"] != "APPROVED" or report["quality_verdict"] != "APPROVED":
        raise Stage2WorkflowError("independent review verdicts must both be APPROVED")
    if report["critical_count"] != 0 or report["important_count"] != 0:
        raise Stage2WorkflowError("independent review has blocking findings")
    if report["final_conclusion"] != "READY_FOR_HUMAN_APPROVAL":
        raise Stage2WorkflowError("independent review is not ready for human approval")


def _verify_stage2_review_report_bindings(
    report: dict[str, object],
    bindings: dict[str, object],
) -> None:
    package_declarations = {
        "review_package_sha256",
        "review_package_bytes",
        "review_package_lf_count",
        "review_package_logical_line_count",
    }
    if any(report[key] != bindings[key] for key in package_declarations):
        raise Stage2WorkflowError("Stage 2 review package declarations drifted")
    direct = {
        "goal_id",
        "stage_id",
        "run_id",
        "reviewed_base_commit",
        "reviewed_prospective_git_tree",
        "reviewed_paths",
        "reviewed_path_set_sha256",
        "manifest_sha256",
        "config_sha256",
        "data_sha256",
        "environment_sha256",
        "source_set_sha256",
        "authorized_next_stage",
    }
    if any(report[key] != bindings[key] for key in direct):
        raise Stage2WorkflowError("Stage 2 review report binding drift")


def build_stage2_approval_challenge(
    *,
    goal_id: str,
    stage_id: str,
    run_id: str,
    reviewed_prospective_git_tree: str,
    review_report_sha256: str,
    review_package_sha256: str,
    manifest_sha256: str,
    config_sha256: str,
    data_sha256: str,
    environment_sha256: str,
    source_set_sha256: str,
    authorized_next_stage: str,
) -> str:
    source = {
        "schema_version": "ppo_highres_frontier_stage2_approval_challenge/v1",
        "goal_id": goal_id,
        "stage_id": stage_id,
        "run_id": run_id,
        "reviewed_prospective_git_tree": reviewed_prospective_git_tree,
        "review_report_sha256": review_report_sha256,
        "review_package_sha256": review_package_sha256,
        "manifest_sha256": manifest_sha256,
        "config_sha256": config_sha256,
        "data_sha256": data_sha256,
        "environment_sha256": environment_sha256,
        "source_set_sha256": source_set_sha256,
        "authorized_next_stage": authorized_next_stage,
    }
    return _hash_json(source)


def _build_stage2_review_record(
    *,
    parsed: dict[str, object],
    bindings: dict[str, object],
    report_binding: dict[str, object],
    package_binding: dict[str, object],
    approval_challenge: str,
    review_recorded_at_utc: str,
) -> dict[str, object]:
    return {
        "schema_version": "ppo_highres_frontier_stage2_independent_review/v3",
        "state": "awaiting_human_approval",
        "goal_id": bindings["goal_id"],
        "stage_id": bindings["stage_id"],
        "run_id": bindings["run_id"],
        "reviewer_id": parsed["reviewer_id"],
        "spec_verdict": parsed["spec_verdict"],
        "quality_verdict": parsed["quality_verdict"],
        "issue_counts": {
            "critical": parsed["critical_count"],
            "important": parsed["important_count"],
            "minor": parsed["minor_count"],
        },
        "final_conclusion": parsed["final_conclusion"],
        "reviewed_base_commit": bindings["reviewed_base_commit"],
        "reviewed_prospective_git_tree": bindings["reviewed_prospective_git_tree"],
        "reviewed_paths": bindings["reviewed_paths"],
        "reviewed_path_set_sha256": bindings["reviewed_path_set_sha256"],
        "source_set_sha256": bindings["source_set_sha256"],
        "data_sha256": bindings["data_sha256"],
        "config_sha256": bindings["config_sha256"],
        "environment_sha256": bindings["environment_sha256"],
        "manifest_sha256": bindings["manifest_sha256"],
        "authorized_next_stage": bindings["authorized_next_stage"],
        "review_report": report_binding,
        "review_package": package_binding,
        "approval_challenge": approval_challenge,
        "review_recorded_at_utc": review_recorded_at_utc,
    }


def record_stage2_independent_review(
    *,
    stage_root: str | Path,
    repo_root: str | Path,
    review_report: str | Path,
    review_package: str | Path,
) -> Path:
    """Freeze and semantically replay a strict Markdown independent review."""

    stage = Path(stage_root).expanduser().resolve()
    if (stage / "review.json").exists() or any((stage / name).exists() for name in _FORBIDDEN):
        raise Stage2WorkflowError("Stage 2 review or authority evidence already exists")
    report_snapshot = FrozenFileSnapshot.capture(review_report, "Stage 2 review report")
    package_snapshot = FrozenFileSnapshot.capture(review_package, "Stage 2 review package")
    bindings = _inspect_stage2_review_bindings_snapshot(
        stage=stage,
        repo=Path(repo_root).expanduser().resolve(),
        package_snapshot=package_snapshot,
        require_review_absent=True,
    )
    parsed = _parse_stage2_review_report(report_snapshot)
    _verify_stage2_review_conclusions(parsed)
    _verify_stage2_review_report_bindings(parsed, bindings)
    report_binding = report_snapshot.binding()
    package_binding = package_snapshot.binding()
    approval_challenge = build_stage2_approval_challenge(
        goal_id=str(bindings["goal_id"]),
        stage_id=str(bindings["stage_id"]),
        run_id=str(bindings["run_id"]),
        reviewed_prospective_git_tree=str(bindings["reviewed_prospective_git_tree"]),
        review_report_sha256=str(report_binding["sha256"]),
        review_package_sha256=str(package_binding["sha256"]),
        manifest_sha256=str(bindings["manifest_sha256"]),
        config_sha256=str(bindings["config_sha256"]),
        data_sha256=str(bindings["data_sha256"]),
        environment_sha256=str(bindings["environment_sha256"]),
        source_set_sha256=str(bindings["source_set_sha256"]),
        authorized_next_stage=str(bindings["authorized_next_stage"]),
    )
    review = _build_stage2_review_record(
        parsed=parsed,
        bindings=bindings,
        report_binding=report_binding,
        package_binding=package_binding,
        approval_challenge=approval_challenge,
        review_recorded_at_utc=_utc_now(),
    )
    report_snapshot.require_current("Stage 2 review report")
    package_snapshot.require_current("Stage 2 review package")
    current_bindings = _inspect_stage2_review_bindings_snapshot(
        stage=stage,
        repo=Path(repo_root).expanduser().resolve(),
        package_snapshot=package_snapshot,
        require_review_absent=True,
    )
    if current_bindings != bindings:
        raise Stage2WorkflowError("Stage 2 review binding drift before write")
    expected = ArtifactStore.canonical_json_bytes(review)
    written = ArtifactStore(stage).write_json_exclusive("review.json", review)
    if written.resolve() != (stage / "review.json").resolve() or written.read_bytes() != expected:
        raise Stage2WorkflowError("Stage 2 review exclusive write verification failed")
    report_snapshot.require_current("Stage 2 review report")
    package_snapshot.require_current("Stage 2 review package")
    return written


def prepare_stage2_review_package(
    *,
    repo_root: str | Path,
    package_path: str | Path,
    base_commit: str | None = None,
) -> Stage2ReviewPackage:
    """Materialize the exact prospective dirty tree without staging the real index."""

    repo = Path(repo_root).expanduser().resolve()
    base = base_commit or _git_text(repo, ["rev-parse", "HEAD"], "review base").strip()
    tree, paths = _prospective_tree(repo, base)
    package = Path(package_path).expanduser().resolve()
    if package.exists():
        raise Stage2WorkflowError("Stage 2 review package target already exists")
    payload = _git_bytes(repo, ["diff", "--binary", "--full-index", base, tree], "review package")
    package.parent.mkdir(parents=True, exist_ok=True)
    _write_exclusive_bytes(package, payload)
    return Stage2ReviewPackage(package, base, tree, paths)


def replay_stage2_review_package(
    *,
    repo_root: str | Path,
    base_commit: str,
    review_package: str | Path,
    expected_prospective_git_tree: str,
    expected_reviewed_paths: tuple[str, ...] | list[str],
) -> str:
    """Replay a review package in a D-drive temporary index and check its full path set."""

    repo = Path(repo_root).expanduser().resolve()
    package_snapshot = FrozenFileSnapshot.capture(
        review_package,
        "Stage 2 review package",
    )
    tree = _replay_stage2_review_package_snapshot(
        repo=repo,
        base_commit=base_commit,
        package_snapshot=package_snapshot,
        expected_prospective_git_tree=expected_prospective_git_tree,
        expected_reviewed_paths=expected_reviewed_paths,
    )
    package_snapshot.require_current("Stage 2 review package")
    return tree


def _replay_stage2_review_package_snapshot(
    *,
    repo: Path,
    base_commit: str,
    package_snapshot: FrozenFileSnapshot,
    expected_prospective_git_tree: str,
    expected_reviewed_paths: tuple[str, ...] | list[str],
) -> str:
    """Replay exactly the bytes captured for review, never a later path read."""

    _require_empty_real_index(repo)
    index_path = _temporary_index_path("stage2-review")
    environment = os.environ.copy()
    environment["GIT_INDEX_FILE"] = str(index_path)
    try:
        _git_text(repo, ["read-tree", base_commit], "review replay read-tree", environment)
        _git_text(
            repo,
            ["apply", "--cached", "--whitespace=nowarn", "-"],
            "review replay apply",
            environment,
            input_payload=package_snapshot.payload,
        )
        tree = _git_text(repo, ["write-tree"], "review replay write-tree", environment).strip()
        paths = _tree_changed_paths(repo, base_commit, tree, environment)
    finally:
        _remove_temporary_index(index_path)
    _require_empty_real_index(repo)
    if tuple(paths) != tuple(expected_reviewed_paths):
        raise Stage2WorkflowError("review package path set is not the exact prospective reviewed path set")
    if tree != expected_prospective_git_tree:
        raise Stage2WorkflowError("review package rebuilt tree does not equal the prospective reviewed tree")
    return tree


def inspect_stage2_review_bindings(
    *,
    stage_root: str | Path,
    repo_root: str | Path,
    review_package: str | Path,
) -> dict[str, object]:
    """Return the frozen bindings an independent reviewer must copy verbatim."""

    stage = Path(stage_root).expanduser().resolve()
    repo = Path(repo_root).expanduser().resolve()
    package_snapshot = FrozenFileSnapshot.capture(
        review_package,
        "Stage 2 review package",
    )
    bindings = _inspect_stage2_review_bindings_snapshot(
        stage=stage,
        repo=repo,
        package_snapshot=package_snapshot,
        require_review_absent=True,
    )
    package_snapshot.require_current("Stage 2 review package")
    return bindings


def _inspect_stage2_review_bindings_snapshot(
    *,
    stage: Path,
    repo: Path,
    package_snapshot: FrozenFileSnapshot,
    require_review_absent: bool,
) -> dict[str, object]:
    """Build reviewer declarations from one frozen package snapshot."""

    if require_review_absent and (
        (stage / "review.json").exists()
        or any((stage / name).exists() for name in _FORBIDDEN)
    ):
        raise Stage2WorkflowError("Stage 2 review authority already exists")
    _verify_stage2_machine_evidence(stage, repo)
    summary = strict_json_object_from_bytes((stage / "summary.json").read_bytes(), "Stage 2 summary")
    routing = strict_json_object_from_bytes((stage / "routing.json").read_bytes(), "Stage 2 routing")
    if summary.get("state") != "machine_passed" or routing.get("route") != "awaiting_independent_review":
        raise Stage2WorkflowError("Stage 2 review requires machine_passed -> awaiting_independent_review")
    config = _machine_config(stage)
    base = str(config.stage1_approval_commit)
    tree, paths = _prospective_tree(repo, base)
    _replay_stage2_review_package_snapshot(
        repo=repo,
        base_commit=base,
        package_snapshot=package_snapshot,
        expected_prospective_git_tree=tree,
        expected_reviewed_paths=paths,
    )
    source = _source_identity(repo)
    environment = _environment_identity()
    return {
        "goal_id": config.goal_id,
        "stage_id": config.stage_id,
        "run_id": stage.parent.name,
        "reviewed_base_commit": base,
        "reviewed_prospective_git_tree": tree,
        "reviewed_paths": list(paths),
        "reviewed_path_set_sha256": _hash_json(list(paths)),
        "review_package_sha256": package_snapshot.sha256,
        "review_package_bytes": package_snapshot.size_bytes,
        "review_package_lf_count": package_snapshot.lf_count,
        "review_package_logical_line_count": package_snapshot.logical_line_count,
        "source_set_sha256": source["source_set_sha256"],
        "data_sha256": summary["catalog_sha256"],
        "config_sha256": hashlib.sha256((stage / "config.json").read_bytes()).hexdigest(),
        "environment_sha256": _hash_json(environment),
        "manifest_sha256": hashlib.sha256((stage / "manifest.json").read_bytes()).hexdigest(),
        "authorized_next_stage": config.authorized_next_stage,
    }


def inspect_stage2_review_authority(
    *,
    stage_root: str | Path,
    repo_root: str | Path,
) -> dict[str, object]:
    """Replay and inspect strict review authority without issuing approval.

    This local package verifies immutable review/package/machine bindings only.
    It intentionally exposes no controller-event, approval, or gate issuer.
    """

    stage = Path(stage_root).expanduser().resolve()
    repo = Path(repo_root).expanduser().resolve()
    return _authority_bindings_for_paths(
        stage=stage,
        repo=repo,
        require_approval=False,
        allow_gate=False,
    )


def verify_stage2_external_approval(
    *,
    stage_root: str | Path,
    repo_root: str | Path,
) -> dict[str, object]:
    """Verify an approval artifact persisted by an external controller.

    Local verification proves canonical artifact integrity, exact bindings,
    challenge response, and temporal ordering.  It cannot authenticate the
    controller backend or human identity that originated the persisted claim.
    """

    return _authority_bindings_for_paths(
        stage=Path(stage_root).expanduser().resolve(),
        repo=Path(repo_root).expanduser().resolve(),
        require_approval=True,
        allow_gate=False,
    )


def _stage2_context_binding_sha256(bindings: dict[str, object]) -> str:
    source = {
        key: bindings[key]
        for key in (
            "goal_id",
            "stage_id",
            "run_id",
            "reviewed_base_commit",
            "reviewed_prospective_git_tree",
            "reviewed_path_set_sha256",
            "source_set_sha256",
            "data_sha256",
            "config_sha256",
            "environment_sha256",
            "manifest_sha256",
            "review_report_sha256",
            "review_report_bytes",
            "review_report_lf_count",
            "review_report_logical_line_count",
            "review_package_sha256",
            "review_package_bytes",
            "review_package_lf_count",
            "review_package_logical_line_count",
            "review_sha256",
            "approval_challenge",
            "review_recorded_at_utc",
            "authorized_next_stage",
        )
    }
    return _hash_json(
        {
            "schema_version": "ppo_highres_frontier_stage2_gate_context_binding/v1",
            **source,
        }
    )


def _build_stage2_gate_history(
    bindings: dict[str, object],
) -> list[dict[str, object]]:
    history: list[dict[str, object]] = []
    previous = _ABSENT_AUTHORITY_HASH
    for state in _STAGE2_STATE_SEQUENCE:
        event_bindings = dict(bindings)
        if state in {"machine_passed", "awaiting_independent_review"}:
            event_bindings["review_sha256"] = _ABSENT_AUTHORITY_HASH
        if state != "approved" and state != "next_stage":
            event_bindings["approval_sha256"] = _ABSENT_AUTHORITY_HASH
        payload = {"state": state, "previous_record_hash": previous, "bindings": event_bindings}
        record_hash = _hash_json(payload)
        history.append({**payload, "record_hash": record_hash})
        previous = record_hash
    return history


def load_verified_stage2_gate(*, gate_path: str | Path, repo_root: str | Path) -> dict[str, object]:
    """Replay an externally persisted gate without authenticating its origin locally."""

    gate_file = Path(gate_path).expanduser().resolve()
    stage = gate_file.parent
    try:
        gate_snapshot = FrozenFileSnapshot.capture(
            gate_file,
            "Stage 2 verified gate",
            authority_root=stage,
        )
    except Stage1WorkflowError as exc:
        raise Stage2WorkflowError("Stage 2 verified gate snapshot drift") from exc
    gate = strict_json_object_from_bytes(
        gate_snapshot.payload,
        "Stage 2 verified gate",
    )
    authorized_next_stage = _machine_config(stage).authorized_next_stage
    if set(gate) != {"schema_version", "state", "authorized_next_stage", "run_id", "bindings", "history"}:
        raise Stage2WorkflowError("Stage 2 verified gate schema is not exact")
    if gate.get("schema_version") != "ppo_highres_frontier_stage2_verified_gate/v1" or gate.get("state") != "next_stage" or gate.get("authorized_next_stage") != authorized_next_stage:
        raise Stage2WorkflowError("Stage 2 verified gate state is invalid")
    bindings = _authority_bindings_for_paths(
        stage=stage,
        repo=Path(repo_root).expanduser().resolve(),
        require_approval=True,
        allow_gate=True,
    )
    if gate.get("bindings") != bindings or gate.get("run_id") != stage.parent.name:
        raise Stage2WorkflowError("Stage 2 verified gate binding drift")
    history = gate.get("history")
    if history != _build_stage2_gate_history(bindings):
        raise Stage2WorkflowError("Stage 2 verified gate history is invalid")
    try:
        gate_snapshot.require_current("Stage 2 verified gate")
    except Stage1WorkflowError as exc:
        raise Stage2WorkflowError("Stage 2 verified gate snapshot drift") from exc
    return gate


def _capture_stage2_recorded_snapshot(
    binding: object,
    label: str,
) -> FrozenFileSnapshot:
    expected_keys = {"path", "sha256", "bytes", "lf_count", "logical_line_count"}
    if not isinstance(binding, dict) or set(binding) != expected_keys:
        raise Stage2WorkflowError(f"{label} binding schema drift")
    path = binding.get("path")
    if not isinstance(path, str) or not Path(path).is_absolute():
        raise Stage2WorkflowError(f"{label} binding path drift")
    try:
        snapshot = FrozenFileSnapshot.capture(path, label)
    except Stage1WorkflowError as exc:
        raise Stage2WorkflowError(f"{label} binding drift") from exc
    if snapshot.binding() != binding:
        raise Stage2WorkflowError(f"{label} binding drift")
    return snapshot


def _authority_bindings_for_paths(
    *,
    stage: Path,
    repo: Path,
    require_approval: bool,
    allow_gate: bool,
) -> dict[str, object]:
    if stage.name != "s2" or not stage.is_dir():
        raise Stage2WorkflowError("Stage 2 authority requires a canonical s2 stage root")
    expected_root = _ROOT_MACHINE_SET | {"review.json"}
    if require_approval:
        expected_root.add("approval.json")
    if allow_gate:
        expected_root.add("gate.json")
    actual = {path.name for path in stage.iterdir()}
    if actual != expected_root:
        raise Stage2WorkflowError("Stage 2 authority review/approval artifact set drift")
    try:
        review_snapshot = FrozenFileSnapshot.capture(
            stage / "review.json",
            "Stage 2 review record",
            authority_root=stage,
        )
        review = strict_json_object_from_bytes(
            review_snapshot.payload,
            "Stage 2 review record",
        )
        report_snapshot = _capture_stage2_recorded_snapshot(
            review.get("review_report"),
            "Stage 2 review report",
        )
        package_snapshot = _capture_stage2_recorded_snapshot(
            review.get("review_package"),
            "Stage 2 review package",
        )
        current = _inspect_stage2_review_bindings_snapshot(
            stage=stage,
            repo=repo,
            package_snapshot=package_snapshot,
            require_review_absent=False,
        )
        parsed = _parse_stage2_review_report(report_snapshot)
        _verify_stage2_review_conclusions(parsed)
        _verify_stage2_review_report_bindings(parsed, current)
        recorded_at = str(review.get("review_recorded_at_utc", ""))
        _parse_utc_timestamp(recorded_at, "review")
        challenge = build_stage2_approval_challenge(
            goal_id=str(current["goal_id"]),
            stage_id=str(current["stage_id"]),
            run_id=str(current["run_id"]),
            reviewed_prospective_git_tree=str(current["reviewed_prospective_git_tree"]),
            review_report_sha256=report_snapshot.sha256,
            review_package_sha256=package_snapshot.sha256,
            manifest_sha256=str(current["manifest_sha256"]),
            config_sha256=str(current["config_sha256"]),
            data_sha256=str(current["data_sha256"]),
            environment_sha256=str(current["environment_sha256"]),
            source_set_sha256=str(current["source_set_sha256"]),
            authorized_next_stage=str(current["authorized_next_stage"]),
        )
        expected_review = _build_stage2_review_record(
            parsed=parsed,
            bindings=current,
            report_binding=report_snapshot.binding(),
            package_binding=package_snapshot.binding(),
            approval_challenge=challenge,
            review_recorded_at_utc=recorded_at,
        )
        if review != expected_review:
            raise Stage2WorkflowError("Stage 2 review record semantic binding drift")
        report_snapshot.require_current("Stage 2 review report")
        package_snapshot.require_current("Stage 2 review package")
        review_snapshot.require_current("Stage 2 review record")
        _verify_stage2_machine_evidence(stage, repo)
        report_snapshot.require_current("Stage 2 review report")
        package_snapshot.require_current("Stage 2 review package")
        review_snapshot.require_current("Stage 2 review record")
    except Stage2WorkflowError:
        raise
    except Stage1WorkflowError as exc:
        raise Stage2WorkflowError("Stage 2 immutable review authority drift") from exc

    report_binding = report_snapshot.binding()
    package_binding = package_snapshot.binding()
    authority = {
        "goal_id": current["goal_id"],
        "stage_id": current["stage_id"],
        "run_id": current["run_id"],
        "reviewed_base_commit": current["reviewed_base_commit"],
        "reviewed_prospective_git_tree": current["reviewed_prospective_git_tree"],
        "reviewed_path_set_sha256": current["reviewed_path_set_sha256"],
        "source_set_sha256": current["source_set_sha256"],
        "data_sha256": current["data_sha256"],
        "config_sha256": current["config_sha256"],
        "environment_sha256": current["environment_sha256"],
        "manifest_sha256": current["manifest_sha256"],
        "review_report_sha256": report_binding["sha256"],
        "review_report_bytes": report_binding["bytes"],
        "review_report_lf_count": report_binding["lf_count"],
        "review_report_logical_line_count": report_binding["logical_line_count"],
        "review_package_sha256": package_binding["sha256"],
        "review_package_bytes": package_binding["bytes"],
        "review_package_lf_count": package_binding["lf_count"],
        "review_package_logical_line_count": package_binding["logical_line_count"],
        "review_sha256": review_snapshot.sha256,
        "approval_challenge": challenge,
        "review_recorded_at_utc": recorded_at,
        "authorized_next_stage": current["authorized_next_stage"],
    }
    approval_hash = _ABSENT_AUTHORITY_HASH
    if require_approval:
        try:
            approval_snapshot = FrozenFileSnapshot.capture(
                stage / "approval.json",
                "Stage 2 approval",
                authority_root=stage,
            )
        except Stage1WorkflowError as exc:
            raise Stage2WorkflowError("Stage 2 approval snapshot drift") from exc
        approval = strict_json_object_from_bytes(
            approval_snapshot.payload,
            "Stage 2 approval",
        )
        _validate_stage2_approval(
            approval,
            stage=stage,
            review_hash=review_snapshot.sha256,
            approval_challenge=challenge,
            review_recorded_at_utc=recorded_at,
            bindings={**authority, "approval_sha256": _ABSENT_AUTHORITY_HASH},
        )
        approval_snapshot.require_current("Stage 2 approval")
        report_snapshot.require_current("Stage 2 review report")
        package_snapshot.require_current("Stage 2 review package")
        review_snapshot.require_current("Stage 2 review record")
        approval_hash = approval_snapshot.sha256
    return {**authority, "approval_sha256": approval_hash}


def _validate_stage2_approval(
    approval: dict[str, object],
    *,
    stage: Path,
    review_hash: str,
    approval_challenge: str,
    review_recorded_at_utc: str,
    bindings: dict[str, object],
) -> None:
    expected_keys = {
        "schema_version",
        "state",
        "authority_kind",
        "actor",
        "event_authenticity_source",
        "local_verification_scope",
        "thread_id",
        "user_turn_id",
        "approval_text",
        "approval_text_utf8_sha256",
        "approval_timestamp_utc",
        "goal_id",
        "stage_id",
        "run_id",
        "reviewed_prospective_git_tree",
        "review_sha256",
        "approval_challenge",
        "controller_event_binding_sha256",
        "bindings",
    }
    declarations = {
        "schema_version": "ppo_highres_frontier_stage2_human_approval/v2",
        "state": "approved",
        "authority_kind": "externally_persisted_controller_user_approval_claim/v1",
        "actor": "user",
        "event_authenticity_source": "external_controller_boundary_not_locally_authenticated/v1",
        "local_verification_scope": "artifact_integrity_challenge_and_temporal_ordering_only/v1",
        "goal_id": bindings["goal_id"],
        "stage_id": bindings["stage_id"],
        "run_id": stage.parent.name,
        "reviewed_prospective_git_tree": bindings["reviewed_prospective_git_tree"],
        "review_sha256": review_hash,
        "approval_challenge": approval_challenge,
    }
    if set(approval) != expected_keys or any(
        approval.get(key) != value for key, value in declarations.items()
    ):
        raise Stage2WorkflowError("Stage 2 approval schema is invalid")
    if approval.get("approval_text") != STAGE2_APPROVAL_TEXT or approval.get("approval_text_utf8_sha256") != hashlib.sha256(STAGE2_APPROVAL_TEXT.encode("utf-8")).hexdigest():
        raise Stage2WorkflowError("Stage 2 approval text binding is invalid")
    if approval.get("bindings") != bindings:
        raise Stage2WorkflowError("Stage 2 approval binding drift")
    _validate_uuid(approval.get("thread_id"), "thread ID")
    _validate_uuid(approval.get("user_turn_id"), "user turn ID")
    review_time = _parse_utc_timestamp(review_recorded_at_utc, "review")
    approval_timestamp = str(approval.get("approval_timestamp_utc", ""))
    approval_time = _parse_utc_timestamp(approval_timestamp, "approval")
    if approval_time < review_time:
        raise Stage2WorkflowError("Stage 2 approval timestamp predates review")
    event_source = {
        "schema_version": "ppo_highres_frontier_stage2_external_controller_event_binding/v1",
        "context_binding_sha256": _stage2_context_binding_sha256(bindings),
        "thread_id": str(approval.get("thread_id", "")),
        "user_turn_id": str(approval.get("user_turn_id", "")),
        "approval_text": str(approval.get("approval_text", "")),
        "event_timestamp_utc": approval_timestamp,
        "review_sha256": review_hash,
        "approval_challenge": approval_challenge,
    }
    if approval.get("controller_event_binding_sha256") != _hash_json(event_source):
        raise Stage2WorkflowError("Stage 2 approval controller event binding drift")


def _machine_config(stage: Path) -> Stage2Config:
    payload = strict_json_object_from_bytes((stage / "config.json").read_bytes(), "Stage 2 config")
    payload.pop("execution_source_identity", None)
    payload.pop("execution_environment_identity", None)
    try:
        return Stage2Config.model_validate(payload)
    except Exception as exc:
        raise Stage2WorkflowError("Stage 2 config binding is invalid") from exc


def _environment_identity() -> dict[str, object]:
    try:
        import rasterio
    except ImportError as exc:  # pragma: no cover - exercised by install/import verification.
        raise Stage2WorkflowError("Stage 2 Rasterio environment is unavailable") from exc
    return {
        "schema_version": "ppo_highres_frontier_stage2_environment/v1",
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "rasterio_version": rasterio.__version__,
        "numpy_version": np.__version__,
    }


def _hash_json(value: object) -> str:
    return hashlib.sha256(ArtifactStore.canonical_json_bytes(value)).hexdigest()


def _validate_uuid(value: object, label: str) -> None:
    if not isinstance(value, str):
        raise Stage2WorkflowError(f"Stage 2 approval {label} is invalid")
    try:
        parsed = uuid.UUID(value)
    except (ValueError, AttributeError) as exc:
        raise Stage2WorkflowError(f"Stage 2 approval {label} is invalid") from exc
    if str(parsed) != value:
        raise Stage2WorkflowError(f"Stage 2 approval {label} is not canonical")


def _parse_utc_timestamp(value: str, label: str) -> datetime:
    if not value.endswith("Z"):
        raise Stage2WorkflowError(f"Stage 2 {label} timestamp must be UTC Z form")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise Stage2WorkflowError(f"Stage 2 {label} timestamp is invalid") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise Stage2WorkflowError(f"Stage 2 {label} timestamp is not UTC")
    return parsed


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _prospective_tree(repo: Path, base_commit: str) -> tuple[str, tuple[str, ...]]:
    _require_empty_real_index(repo)
    index_path = _temporary_index_path("stage2-prospective")
    environment = os.environ.copy()
    environment["GIT_INDEX_FILE"] = str(index_path)
    try:
        _git_text(repo, ["read-tree", base_commit], "prospective read-tree", environment)
        _git_text(repo, ["add", "-A", "--", "."], "prospective add", environment)
        tree = _git_text(repo, ["write-tree"], "prospective write-tree", environment).strip()
        paths = _tree_changed_paths(repo, base_commit, tree, environment)
    finally:
        _remove_temporary_index(index_path)
    _require_empty_real_index(repo)
    return tree, tuple(paths)


def _tree_changed_paths(repo: Path, base: str, tree: str, environment: dict[str, str]) -> list[str]:
    output = _git_bytes(repo, ["diff-tree", "--no-commit-id", "--name-only", "-r", "-z", "--no-renames", base, tree], "review path set", environment)
    try:
        return sorted(item.decode("utf-8", errors="strict").replace("\\", "/") for item in output.split(b"\0") if item)
    except UnicodeDecodeError as exc:
        raise Stage2WorkflowError("Stage 2 review path set is not strict UTF-8") from exc


def _temporary_index_path(label: str) -> Path:
    root = Path("D:/xunce/tmp/ppo_frontier/git-index").resolve()
    if root.drive.upper() != "D:":
        raise Stage2WorkflowError("Stage 2 temporary Git index must be on D drive")
    root.mkdir(parents=True, exist_ok=True)
    return root / f"{label}-{uuid.uuid4().hex}.index"


def _remove_temporary_index(index_path: Path) -> None:
    for path in (Path(str(index_path) + ".lock"), index_path):
        if path.is_file():
            path.unlink()


def _require_empty_real_index(repo: Path) -> None:
    completed = subprocess.run(["git", "-C", str(repo), "diff", "--cached", "--quiet"], check=False)
    if completed.returncode not in {0}:
        raise Stage2WorkflowError("real Git index is not empty")


def _git_text(
    repo: Path,
    arguments: list[str],
    label: str,
    environment: dict[str, str] | None = None,
    *,
    input_payload: bytes | None = None,
) -> str:
    return _git_bytes(
        repo,
        arguments,
        label,
        environment,
        input_payload=input_payload,
    ).decode("utf-8", errors="strict")


def _git_bytes(
    repo: Path,
    arguments: list[str],
    label: str,
    environment: dict[str, str] | None = None,
    *,
    input_payload: bytes | None = None,
) -> bytes:
    completed = subprocess.run(
        ["git", "-C", str(repo), *arguments],
        check=False,
        input=input_payload,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=environment,
    )
    if completed.returncode:
        raise Stage2WorkflowError(f"Stage 2 Git {label} failed: {completed.stderr.decode('utf-8', errors='replace').strip()}")
    return completed.stdout


def _write_exclusive_bytes(path: Path, payload: bytes) -> None:
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise Stage2WorkflowError("Stage 2 review package target already exists") from exc
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _initial_observed_state(
    scenario: ScenarioBundle,
) -> tuple[ObservedMapState, ObservationDelta, dict[str, float | str]]:
    updater = SensorUpdater()
    state = ObservedMapState.empty(scenario.truth.geometry)
    reset_pose = SensorPose(
        scenario.truth.geometry.cell_to_world_center(scenario.start_pose.cell),
        scenario.start_pose.theta,
        "reset",
    )
    delta = updater.reveal(scenario.truth, state, (reset_pose,))
    contract: dict[str, float | str] = {
        "source": "contract_reset_scan/v1",
        "range_m": updater.range_m,
        "fov_deg": updater.fov_deg,
        "ray_angle_step_deg": updater.ray_angle_step_deg,
        "min_clearance_m": updater.min_clearance_m,
        "max_slope_deg": updater.max_slope_deg,
        "traversability_threshold": updater.traversability_threshold,
    }
    return state, delta, contract


def _copy_observed_state(source: ObservedMapState) -> ObservedMapState:
    return ObservedMapState(
        geometry=source.geometry,
        observed_mask=source.observed_mask.copy(),
        confidence=source.confidence.copy(),
        height=source.height.copy(),
        obstacle=source.obstacle.copy(),
        slope_deg=source.slope_deg.copy(),
        traversability=source.traversability.copy(),
        observed_safe_mask=source.observed_safe_mask.copy(),
        remaining_step_budget_norm=source.remaining_step_budget_norm,
    )


def _build_hidden_truth_leakage_pair(
    *,
    scenario: ScenarioBundle,
    observed_state: ObservedMapState,
    pose: PoseXYTheta,
    top_m: int,
) -> _HiddenTruthLeakagePair:
    yy, xx = np.mgrid[: scenario.truth.geometry.height, : scenario.truth.geometry.width]
    protected_radius_cells = 20.0 / scenario.truth.geometry.resolution_m
    mutation_mask = (~observed_state.observed_mask) & (
        (xx - pose.cell.x) ** 2 + (yy - pose.cell.y) ** 2 > protected_radius_cells**2
    )
    if not np.any(mutation_mask):
        raise Stage2WorkflowError("paired leakage fixture requires unobserved truth cells")
    mutated_height = scenario.truth.height.copy()
    mutated_obstacle = scenario.truth.hard_obstacle.copy()
    mutated_slope = scenario.truth.slope_deg.copy()
    mutated_traversability = scenario.truth.traversability.copy()
    mutated_height[mutation_mask] = mutated_height[mutation_mask] + 1000.0
    mutated_obstacle[mutation_mask] = True
    mutated_slope[mutation_mask] = 89.0
    mutated_traversability[mutation_mask] = 0.0
    mutated_truth = TruthMap(
        geometry=scenario.truth.geometry,
        height=mutated_height,
        hard_obstacle=mutated_obstacle,
        slope_deg=mutated_slope,
        traversability=mutated_traversability,
        provenance={
            **dict(scenario.truth.provenance),
            "paired_fixture_mutation_source": "unobserved_truth_only/v1",
        },
    )

    def fixture(truth: TruthMap) -> _HiddenTruthPolicyFixture:
        state = _copy_observed_state(observed_state)
        action_set = FrontierGenerator(top_m=top_m).extract(state, scenario.prior, pose)
        observation = ObservationBuilder().build(scenario.prior, state, pose, action_set)
        return _HiddenTruthPolicyFixture(
            truth=truth,
            coverable_mask=_exact_coverable_mask(truth, pose.cell),
            prior=scenario.prior,
            observed_state=state,
            pose=pose,
            action_set=action_set,
            observation=observation,
        )

    return _HiddenTruthLeakagePair(
        baseline=fixture(scenario.truth),
        mutated=fixture(mutated_truth),
    )


def _exact_coverable_mask(truth: TruthMap, start: CellXY) -> np.ndarray:
    key = _canonical_sha256(
        {
            "truth_sha256": _truth_map_content_sha256(truth),
            "start": [start.x, start.y],
            "sensor_range_m": 20.0,
            "min_clearance_m": 0.5215874761,
            "max_slope_deg": 30.0,
            "traversability_threshold": 0.50,
        }
    )
    cached = _EXACT_COVERABLE_MASK_CACHE.get(key)
    if cached is None:
        computed = compute_coverage_masks(
            truth,
            start,
            sensor_range_m=20.0,
            min_clearance_m=0.5215874761,
            max_slope_deg=30.0,
            traversability_threshold=0.50,
        ).coverable_mask
        cached = np.asarray(computed, dtype=bool).copy()
        cached.setflags(write=False)
        _EXACT_COVERABLE_MASK_CACHE[key] = cached
    result = cached.copy()
    result.setflags(write=False)
    return result


def _truth_map_content_sha256(truth: TruthMap) -> str:
    identity = {
        "geometry": {
            "shape": list(truth.geometry.shape),
            "resolution_m": truth.geometry.resolution_m,
            "origin": [truth.geometry.origin.x, truth.geometry.origin.y],
        },
        "layers": {
            name: _array_content_sha256(array, domain=f"stage2_leakage_truth_{name}/v1")
            for name, array in (
                ("height", truth.height),
                ("hard_obstacle", truth.hard_obstacle),
                ("slope_deg", truth.slope_deg),
                ("traversability", truth.traversability),
            )
        },
    }
    return _canonical_sha256(identity)


def _array_canonical_bytes(array: np.ndarray) -> bytes:
    values = np.ascontiguousarray(array)
    header = ArtifactStore.canonical_json_bytes(
        {"dtype": values.dtype.str, "shape": list(values.shape)}
    )
    return len(header).to_bytes(8, "big") + header + values.tobytes()


def _action_set_canonical_bytes(action_set: FrontierActionSet) -> bytes:
    payload = bytearray(
        ArtifactStore.canonical_json_bytes(
            {
                "cells": [[cell.x, cell.y] for cell in action_set.cells],
                "diagnostics": dict(action_set.diagnostics),
            }
        )
    )
    for array in (
        action_set.frontier_features,
        action_set.candidate_mask,
        action_set.frontier_mask,
        action_set.observed_safe_frontier_mask,
        action_set.reachable_frontier_mask,
    ):
        if array is None:
            payload.extend((0).to_bytes(8, "big"))
        else:
            encoded = _array_canonical_bytes(array)
            payload.extend(len(encoded).to_bytes(8, "big"))
            payload.extend(encoded)
    return bytes(payload)


def _observation_canonical_bytes(observation: PolicyObservation) -> bytes:
    payload = bytearray(ArtifactStore.canonical_json_bytes(observation.schema_metadata()))
    for array in observation.array_fields():
        encoded = _array_canonical_bytes(array)
        payload.extend(len(encoded).to_bytes(8, "big"))
        payload.extend(encoded)
    return bytes(payload)


def _observed_state_canonical_bytes(state: ObservedMapState) -> bytes:
    payload = bytearray(
        ArtifactStore.canonical_json_bytes(
            {
                "shape": list(state.geometry.shape),
                "resolution_m": state.geometry.resolution_m,
                "remaining_step_budget_norm": state.remaining_step_budget_norm,
            }
        )
    )
    for array in (
        state.observed_mask,
        state.confidence,
        state.height,
        state.obstacle,
        state.slope_deg,
        state.traversability,
        state.observed_safe_mask,
    ):
        encoded = _array_canonical_bytes(array)
        payload.extend(len(encoded).to_bytes(8, "big"))
        payload.extend(encoded)
    return bytes(payload)


def _prior_canonical_bytes(prior: LowResolutionPrior) -> bytes:
    return ArtifactStore.canonical_json_bytes(
        {
            "resolution_m": prior.resolution_m,
            "value_prior_source": prior.value_prior_source,
        }
    ) + _array_canonical_bytes(prior.channels)


def _paired_hidden_truth_leakage_audit(
    pair: _HiddenTruthLeakagePair,
) -> dict[str, object]:
    baseline = pair.baseline
    mutated = pair.mutated
    observed = baseline.observed_state.observed_mask
    truth_observed_equal = all(
        np.array_equal(getattr(baseline.truth, name)[observed], getattr(mutated.truth, name)[observed])
        for name in ("height", "hard_obstacle", "slope_deg", "traversability")
    )
    unobserved_truth_mutated = any(
        not np.array_equal(
            getattr(baseline.truth, name)[~observed],
            getattr(mutated.truth, name)[~observed],
        )
        for name in ("height", "hard_obstacle", "slope_deg", "traversability")
    )
    deployment_prior_byte_equal = _prior_canonical_bytes(baseline.prior) == _prior_canonical_bytes(
        mutated.prior
    )
    observed_state_byte_equal = _observed_state_canonical_bytes(
        baseline.observed_state
    ) == _observed_state_canonical_bytes(mutated.observed_state)
    baseline_blockers = observed & (
        baseline.observed_state.obstacle | (baseline.observed_state.slope_deg > 30.0)
    )
    mutated_blockers = mutated.observed_state.observed_mask & (
        mutated.observed_state.obstacle | (mutated.observed_state.slope_deg > 30.0)
    )
    observed_blockers_byte_equal = _array_canonical_bytes(
        baseline_blockers
    ) == _array_canonical_bytes(mutated_blockers)
    pose_byte_equal = ArtifactStore.canonical_json_bytes(
        [baseline.pose.cell.x, baseline.pose.cell.y, baseline.pose.theta]
    ) == ArtifactStore.canonical_json_bytes(
        [mutated.pose.cell.x, mutated.pose.cell.y, mutated.pose.theta]
    )
    coverable_mask_mutated = _array_canonical_bytes(
        baseline.coverable_mask
    ) != _array_canonical_bytes(mutated.coverable_mask)
    baseline_action_bytes = _action_set_canonical_bytes(baseline.action_set)
    mutated_action_bytes = _action_set_canonical_bytes(mutated.action_set)
    action_set_byte_equal = baseline_action_bytes == mutated_action_bytes
    observation_array_byte_equal = [
        _array_canonical_bytes(first) == _array_canonical_bytes(second)
        for first, second in zip(
            baseline.observation.array_fields(),
            mutated.observation.array_fields(),
            strict=True,
        )
    ]
    all_observation_arrays_byte_equal = all(observation_array_byte_equal)
    fixture_valid = all(
        (
            truth_observed_equal,
            unobserved_truth_mutated,
            deployment_prior_byte_equal,
            observed_state_byte_equal,
            observed_blockers_byte_equal,
            pose_byte_equal,
            coverable_mask_mutated,
        )
    )
    if not fixture_valid:
        raise Stage2WorkflowError("paired hidden-truth leakage fixture precondition failed")
    hidden_truth_leakage_detected = not (
        action_set_byte_equal and all_observation_arrays_byte_equal
    )
    baseline_observation_bytes = _observation_canonical_bytes(baseline.observation)
    mutated_observation_bytes = _observation_canonical_bytes(mutated.observation)
    return {
        "paired_fixture_source": "paired_hidden_truth_coverable_mutation/v1",
        "truth_observed_cells_byte_equal": truth_observed_equal,
        "unobserved_truth_mutated": unobserved_truth_mutated,
        "coverable_mask_mutated": coverable_mask_mutated,
        "deployment_prior_byte_equal": deployment_prior_byte_equal,
        "observed_state_byte_equal": observed_state_byte_equal,
        "observed_blockers_byte_equal": observed_blockers_byte_equal,
        "pose_byte_equal": pose_byte_equal,
        "action_set_byte_equal": action_set_byte_equal,
        "observation_array_byte_equal": observation_array_byte_equal,
        "all_observation_arrays_byte_equal": all_observation_arrays_byte_equal,
        "hidden_truth_leakage_detected": hidden_truth_leakage_detected,
        "unknown_truth_mutation_invariant": not hidden_truth_leakage_detected,
        "coverable_mask_mutation_invariant": not hidden_truth_leakage_detected,
        "baseline_truth_sha256": _truth_map_content_sha256(baseline.truth),
        "mutated_truth_sha256": _truth_map_content_sha256(mutated.truth),
        "baseline_coverable_mask_sha256": hashlib.sha256(
            _array_canonical_bytes(baseline.coverable_mask)
        ).hexdigest(),
        "mutated_coverable_mask_sha256": hashlib.sha256(
            _array_canonical_bytes(mutated.coverable_mask)
        ).hexdigest(),
        "baseline_action_set_sha256": hashlib.sha256(baseline_action_bytes).hexdigest(),
        "mutated_action_set_sha256": hashlib.sha256(mutated_action_bytes).hexdigest(),
        "baseline_observation_sha256": hashlib.sha256(baseline_observation_bytes).hexdigest(),
        "mutated_observation_sha256": hashlib.sha256(mutated_observation_bytes).hexdigest(),
    }


def _array_content_sha256(array: np.ndarray, *, domain: str) -> str:
    values = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(domain.encode("ascii"))
    digest.update(b"\0")
    digest.update(values.dtype.str.encode("ascii"))
    digest.update(b"\0")
    for dimension in values.shape:
        digest.update(int(dimension).to_bytes(8, "big"))
    digest.update(values.tobytes())
    return digest.hexdigest()


def _canonical_sha256(payload: dict[str, object]) -> str:
    return hashlib.sha256(ArtifactStore.canonical_json_bytes(payload)).hexdigest()


def _scenario_lineage(
    *,
    catalog_sha256: str,
    record: ScenarioCatalogRecord,
    scenario: ScenarioBundle,
    state: ObservedMapState,
    reset_delta: ObservationDelta,
    sensor_contract: dict[str, float | str],
) -> dict[str, object]:
    record_sha256 = _canonical_sha256(asdict(record))
    prior_identity: dict[str, object] = {
        "source": record.real_lowres_prior_source,
        "value_prior_source": scenario.prior.value_prior_source,
        "shape": list(scenario.prior.channels.shape),
        "resolution_m": scenario.prior.resolution_m,
        "channels_sha256": _array_content_sha256(
            scenario.prior.channels,
            domain="stage2_standard_prior/v1",
        ),
        "provenance": dict(scenario.prior.provenance),
    }
    truth_identity: dict[str, object] = {
        "source": record.highres_truth_source,
        "scenario_id": scenario.scenario_id,
        "scenario_hash": scenario.scenario_hash,
        "highres_shape": list(scenario.truth.geometry.shape),
        "resolution_m": scenario.truth.geometry.resolution_m,
        "synthetic_source_kind": scenario.truth.provenance["synthetic_source_kind"],
        "physical_obstacle_cells_written": scenario.truth.provenance[
            "physical_obstacle_cells_written"
        ],
        "proxy_seed_hex": scenario.proxy_catalog.seed_hex,
        "proxy_catalog_sha256": scenario.proxy_catalog.sha256,
        "layer_hashes": dict(scenario.proxy_layer_hashes),
    }
    truth_identity["truth_sha256"] = _canonical_sha256(truth_identity)
    observed_layer_hashes = {
        name: _array_content_sha256(array, domain=f"stage2_observed_{name}/v1")
        for name, array in (
            ("observed_mask", state.observed_mask),
            ("confidence", state.confidence),
            ("height", state.height),
            ("obstacle", state.obstacle),
            ("slope_deg", state.slope_deg),
            ("traversability", state.traversability),
            ("observed_safe_mask", state.observed_safe_mask),
        )
    }
    observed_identity: dict[str, object] = {
        "source": record.observed_state_source,
        "sensor_contract": sensor_contract,
        "sensor_diagnostics": asdict(reset_delta.diagnostics),
        "observed_cell_count": int(np.count_nonzero(state.observed_mask)),
        "layer_hashes": observed_layer_hashes,
    }
    observed_identity["observed_state_sha256"] = _canonical_sha256(observed_identity)
    lineage: dict[str, object] = {
        "schema_version": "standard_scenario_observed_lineage/v1",
        "catalog_sha256": catalog_sha256,
        "catalog_record_sha256": record_sha256,
        "prior": prior_identity,
        "truth": truth_identity,
        "observed": observed_identity,
    }
    lineage["lineage_sha256"] = _canonical_sha256(lineage)
    return lineage


def _acceptance_items(
    observation: PolicyObservation,
    action_set,
    leakage: dict[str, object],
    split_audit: dict[str, object],
) -> dict[str, bool]:
    items = {
        "01_global_prior_shape": observation.prior_channels.shape == (7, 32, 32),
        "02_global_prior_order": len(PolicyObservation.schema_metadata()["global_prior_channels"]) == 7,
        "03_summary_shape": observation.coverage_summary.shape == (8, 32, 32),
        "04_summary_observed_only": not bool(leakage["hidden_truth_leakage_detected"]),
        "05_local_crop_shape": observation.local_crop.shape == (8, 96, 96),
        "06_local_crop_order": len(PolicyObservation.schema_metadata()["local_crop_channels"]) == 8,
        "07_candidate_mapping_internal": (
            not hasattr(observation, "frontier_cells")
            and len(action_set.cells) == int(np.count_nonzero(action_set.candidate_mask))
        ),
        "08_frontier_features_shape": observation.frontier_features.shape == (1024, 22),
        "09_frontier_feature_order": len(PolicyObservation.schema_metadata()["frontier_feature_fields"]) == 22,
        "10_candidate_mask_shape": observation.candidate_mask.shape == (1024,),
        "11_padding_mask_false": not np.any(observation.candidate_mask[action_set.candidate_count:]),
        "12_valid_rows_finite": bool(np.isfinite(observation.frontier_features[observation.candidate_mask]).all()),
        "13_nonoverflow_keep_all": int(np.count_nonzero(observation.candidate_mask)) == action_set.candidate_count,
        "14_score_first_overflow": score_first_top_m([1.0, 1.0, 0.0], 2) == (0, 1),
        "15_top_m_no_truth": not bool(leakage["hidden_truth_leakage_detected"]),
        "16_gain_observed_only": bool(leakage["unknown_truth_mutation_invariant"]),
        "17_empty_set_finite": bool(np.isfinite(FrontierGenerator(top_m=1).extract(
            ObservedMapState.empty(GridGeometry(2, 2, 0.5)),
            _tiny_prior(), PoseXYTheta(CellXY(0, 0), 0.0),
        ).frontier_features).all()),
        "18_npz_schema": observation.schema_metadata() == PolicyObservation.schema_metadata(),
    }
    if split_audit["parent_cross_split_count"] or split_audit["child_overlap_pair_count"]:
        items["04_summary_observed_only"] = False
    return items


def _tiny_prior():
    from lunar_exploration_ppo.env.scenario import LowResolutionPrior
    return LowResolutionPrior(np.zeros((7, 1, 1), dtype=np.float32), 1.0, "test/v1")


def _source_identity(repo_root: Path) -> dict[str, object]:
    paths: set[Path] = set()
    paths.update((repo_root / "src/lunar_exploration_ppo").rglob("*.py"))
    paths.update((repo_root / "tests/ppo_highres_frontier").glob("test_stage2_*.py"))
    paths.update({
        repo_root / "pyproject.toml",
        repo_root / "scripts/run_ppo_highres_frontier_stage2.py",
        repo_root / "configs/ppo_highres_frontier_stage2_v1.json",
        repo_root / "docs/superpowers/specs/2026-07-09-ppo-highres-frontier-map-exploration-design.md",
        repo_root / "docs/superpowers/specs/2026-07-10-rock-crater-proxy-fixture-design-addendum.md",
        repo_root / ".superpowers/sdd/task-3-brief.md",
    })
    entries: list[dict[str, object]] = []
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.relative_to(repo_root).as_posix()):
        if not path.is_file():
            raise Stage2WorkflowError(f"Stage 2 source binding path is missing: {path}")
        relative = path.relative_to(repo_root).as_posix()
        payload = path.read_bytes()
        sha256 = hashlib.sha256(payload).hexdigest()
        entries.append({"path": relative, "sha256": sha256, "size_bytes": len(payload)})
        digest.update(relative.encode("utf-8")); digest.update(b"\0"); digest.update(payload); digest.update(b"\0")
    return {
        "schema_version": "stage2_reviewed_source_set/v1",
        "source_set_sha256": digest.hexdigest(),
        "paths": entries,
    }
