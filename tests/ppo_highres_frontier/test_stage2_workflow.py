from __future__ import annotations

import importlib
import json
import hashlib
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "ppo_highres_frontier_stage2_v1.json"


def _module(name: str):
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError:
        pytest.fail(f"Stage 2 module is missing: {name}")


def _rewrite_manifest_entry(workflow: object, stage: Path, relative_path: str) -> None:
    payload = (stage / relative_path).read_bytes()
    manifest_path = stage / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entry = next(item for item in manifest["artifacts"] if item["path"] == relative_path)
    entry["size_bytes"] = len(payload)
    entry["sha256"] = hashlib.sha256(payload).hexdigest()
    manifest_path.write_bytes(workflow.ArtifactStore.canonical_json_bytes(manifest))


def _strict_stage2_review_markdown(
    bindings: dict[str, object],
    *,
    reviewer_id: str = "independent-reviewer",
    spec_verdict: str = "APPROVED",
    quality_verdict: str = "APPROVED",
    critical_count: int = 0,
    important_count: int = 0,
    minor_count: int = 0,
    final_conclusion: str = "READY_FOR_HUMAN_APPROVAL",
    front_overrides: dict[str, object] | None = None,
    body_overrides: dict[str, object] | None = None,
    findings_line: str | None = None,
) -> str:
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
            bindings["reviewed_paths"],
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
    if front_overrides:
        front.update(front_overrides)
    body: dict[str, object] = {
        "reviewer_id": reviewer_id,
        "spec_verdict": spec_verdict,
        "quality_verdict": quality_verdict,
        "critical_count": critical_count,
        "important_count": important_count,
        "minor_count": minor_count,
        "final_conclusion": final_conclusion,
    }
    if body_overrides:
        body.update(body_overrides)
    if findings_line is None:
        total = critical_count + important_count + minor_count
        findings_line = "Findings: None" if total == 0 else "Findings: Listed below"
    ordered_keys = (
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
    lines = ["---", *(f"{key}: {front[key]}" for key in ordered_keys), "---", ""]
    lines.extend(
        (
            "# Stage 2 Independent Review",
            "",
            f"Reviewer: {body['reviewer_id']}",
            f"Specification compliance verdict: {body['spec_verdict']}",
            f"Code quality verdict: {body['quality_verdict']}",
            f"Critical findings: {body['critical_count']}",
            f"Important findings: {body['important_count']}",
            f"Minor findings: {body['minor_count']}",
            f"Final conclusion: {body['final_conclusion']}",
            findings_line,
        )
    )
    return "\n".join(lines) + "\n"


def _stage2_review_inputs(workflow: object, tmp_path: Path, run_id: str):
    result = workflow.run_stage2_workflow(
        config_path=CONFIG,
        run_id=run_id,
        base_output_root=tmp_path,
    )
    machine_config = json.loads(
        (result.stage_root / "config.json").read_text(encoding="utf-8")
    )
    review_base = machine_config["stage1_approval_commit"]
    package = workflow.prepare_stage2_review_package(
        repo_root=ROOT,
        package_path=tmp_path / f"{run_id}-review.patch",
        base_commit=review_base,
    )
    assert package.base_commit == review_base
    bindings = workflow.inspect_stage2_review_bindings(
        stage_root=result.stage_root,
        repo_root=ROOT,
        review_package=package.path,
    )
    package_bytes = package.path.read_bytes()
    bindings = {
        **bindings,
        "goal_id": machine_config["goal_id"],
        "stage_id": machine_config["stage_id"],
        "run_id": run_id,
        "review_package_bytes": len(package_bytes),
        "review_package_lf_count": package_bytes.count(b"\n"),
        "review_package_logical_line_count": len(package_bytes.splitlines()),
    }
    return result, package, bindings


def _record_strict_stage2_review(workflow: object, tmp_path: Path, run_id: str):
    result, package, bindings = _stage2_review_inputs(workflow, tmp_path, run_id)
    report = tmp_path / f"{run_id}-independent-review.md"
    report.write_text(
        _strict_stage2_review_markdown(bindings),
        encoding="utf-8",
        newline="",
    )
    review_path = workflow.record_stage2_independent_review(
        stage_root=result.stage_root,
        repo_root=ROOT,
        review_report=report,
        review_package=package.path,
    )
    return (
        result,
        package,
        bindings,
        report,
        review_path,
        json.loads(review_path.read_text(encoding="utf-8")),
    )


_STAGE2_ABSENT_AUTHORITY_HASH = hashlib.sha256(
    b"ppo_highres_frontier_stage2_authority_absent/v1"
).hexdigest()
_STAGE2_EXTERNAL_AUTHORITY_KIND = (
    "externally_persisted_controller_user_approval_claim/v1"
)
_STAGE2_EXTERNAL_AUTHENTICITY_SOURCE = (
    "external_controller_boundary_not_locally_authenticated/v1"
)
_STAGE2_LOCAL_VERIFICATION_SCOPE = (
    "artifact_integrity_challenge_and_temporal_ordering_only/v1"
)
_STAGE2_CONTEXT_BINDING_KEYS = (
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


def _canonical_sha256(workflow: object, value: object) -> str:
    return hashlib.sha256(workflow.ArtifactStore.canonical_json_bytes(value)).hexdigest()


def _external_stage2_context_binding_sha256(
    workflow: object,
    bindings: dict[str, object],
) -> str:
    return _canonical_sha256(
        workflow,
        {
            "schema_version": "ppo_highres_frontier_stage2_gate_context_binding/v1",
            **{key: bindings[key] for key in _STAGE2_CONTEXT_BINDING_KEYS},
        },
    )


def _external_controller_approval_payload(
    workflow: object,
    stage_root: Path,
    *,
    event_timestamp_utc: str = "2099-07-13T00:00:01Z",
    thread_id: str = "00000000-0000-0000-0000-000000000001",
    user_turn_id: str = "00000000-0000-0000-0000-000000000002",
    review_sha256: str | None = None,
    approval_challenge: str | None = None,
    event_context_bindings: dict[str, object] | None = None,
) -> dict[str, object]:
    """Simulate payload creation at the controller-owned boundary.

    This helper deliberately lives outside production.  It does not prove a real
    backend/user event; tests use it only to exercise local artifact verification.
    """

    bindings = workflow.inspect_stage2_review_authority(
        stage_root=stage_root,
        repo_root=ROOT,
    )
    bound_review_sha256 = (
        str(bindings["review_sha256"])
        if review_sha256 is None
        else review_sha256
    )
    bound_challenge = (
        str(bindings["approval_challenge"])
        if approval_challenge is None
        else approval_challenge
    )
    event_bindings = bindings if event_context_bindings is None else event_context_bindings
    event_source = {
        "schema_version": "ppo_highres_frontier_stage2_external_controller_event_binding/v1",
        "context_binding_sha256": _external_stage2_context_binding_sha256(
            workflow,
            event_bindings,
        ),
        "thread_id": thread_id,
        "user_turn_id": user_turn_id,
        "approval_text": workflow.STAGE2_APPROVAL_TEXT,
        "event_timestamp_utc": event_timestamp_utc,
        "review_sha256": bound_review_sha256,
        "approval_challenge": bound_challenge,
    }
    return {
        "schema_version": "ppo_highres_frontier_stage2_human_approval/v2",
        "state": "approved",
        "authority_kind": _STAGE2_EXTERNAL_AUTHORITY_KIND,
        "actor": "user",
        "event_authenticity_source": _STAGE2_EXTERNAL_AUTHENTICITY_SOURCE,
        "local_verification_scope": _STAGE2_LOCAL_VERIFICATION_SCOPE,
        "thread_id": thread_id,
        "user_turn_id": user_turn_id,
        "approval_text": workflow.STAGE2_APPROVAL_TEXT,
        "approval_text_utf8_sha256": hashlib.sha256(
            workflow.STAGE2_APPROVAL_TEXT.encode("utf-8")
        ).hexdigest(),
        "approval_timestamp_utc": event_timestamp_utc,
        "goal_id": bindings["goal_id"],
        "stage_id": bindings["stage_id"],
        "run_id": bindings["run_id"],
        "reviewed_prospective_git_tree": bindings["reviewed_prospective_git_tree"],
        "review_sha256": bound_review_sha256,
        "approval_challenge": bound_challenge,
        "controller_event_binding_sha256": _canonical_sha256(workflow, event_source),
        "bindings": bindings,
    }


def _write_external_stage2_approval(
    workflow: object,
    stage_root: Path,
    **payload_options: object,
) -> tuple[Path, dict[str, object]]:
    payload = _external_controller_approval_payload(
        workflow,
        stage_root,
        **payload_options,
    )
    path = workflow.ArtifactStore(stage_root).write_json_exclusive("approval.json", payload)
    return path, payload


def _external_stage2_gate_history(
    workflow: object,
    bindings: dict[str, object],
) -> list[dict[str, object]]:
    history: list[dict[str, object]] = []
    previous = _STAGE2_ABSENT_AUTHORITY_HASH
    for state in (
        "machine_passed",
        "awaiting_independent_review",
        "awaiting_human_approval",
        "approved",
        "next_stage",
    ):
        event_bindings = dict(bindings)
        if state in {"machine_passed", "awaiting_independent_review"}:
            event_bindings["review_sha256"] = _STAGE2_ABSENT_AUTHORITY_HASH
        if state not in {"approved", "next_stage"}:
            event_bindings["approval_sha256"] = _STAGE2_ABSENT_AUTHORITY_HASH
        event = {
            "state": state,
            "previous_record_hash": previous,
            "bindings": event_bindings,
        }
        record_hash = _canonical_sha256(workflow, event)
        history.append({**event, "record_hash": record_hash})
        previous = record_hash
    return history


def _write_external_stage2_gate(
    workflow: object,
    stage_root: Path,
) -> tuple[Path, dict[str, object]]:
    bindings = workflow.verify_stage2_external_approval(
        stage_root=stage_root,
        repo_root=ROOT,
    )
    config = json.loads((stage_root / "config.json").read_text(encoding="utf-8"))
    payload = {
        "schema_version": "ppo_highres_frontier_stage2_verified_gate/v1",
        "state": "next_stage",
        "authorized_next_stage": config["authorized_next_stage"],
        "run_id": stage_root.parent.name,
        "bindings": bindings,
        "history": _external_stage2_gate_history(workflow, bindings),
    }
    path = workflow.ArtifactStore(stage_root).write_json_exclusive("gate.json", payload)
    return path, payload


def _simulate_external_controller_authority_writer(
    workflow: object,
    stage_root: Path,
) -> tuple[Path, Path, dict[str, object]]:
    approval_path, _approval = _write_external_stage2_approval(workflow, stage_root)
    gate_path, _gate = _write_external_stage2_gate(workflow, stage_root)
    verified = workflow.load_verified_stage2_gate(gate_path=gate_path, repo_root=ROOT)
    return approval_path, gate_path, verified


def _clone_reviewed_stage(stage_root: Path, case_root: Path) -> Path:
    cloned = case_root / stage_root.parent.name / "s2"
    cloned.parent.mkdir(parents=True, exist_ok=False)
    shutil.copytree(stage_root, cloned)
    return cloned


def test_stage2_config_freezes_exact_scale_catalog_and_data_semantics() -> None:
    config_module = _module("lunar_exploration_ppo.configs.stage2")
    config = config_module.load_stage2_config(CONFIG)
    assert config.stage1_approval_commit == "deb876032054f300c68a4215d597f607df2a356b"
    assert config.catalog_counts == {"train": 700, "validation": 150, "test": 150, "unseen": 64}
    assert config.standard.highres_shape == (256, 256)
    assert config.standard.lowres_shape == (32, 32)
    assert config.standard.lowres_resolution_m == 4.0
    assert config.standard.local_crop_shape == (96, 96)
    assert config.standard.frontier_top_m == 1024
    assert config.kilometer.lowres_resolution_m == 8.0
    assert config.slope_product_semantics == "rgb_visualization_provenance_only/v1"
    assert config.physical_slope_source == "dem_float32_metric_gradient_4m/v1"
    assert config.slope_pixels_used_for_traversability is False
    assert config.authorized_next_stage == "ppo_highres_frontier_stage3_cross_attention_policy/v1"


def test_stage2_workflow_writes_exact_machine_and_manifest_bound_evidence(tmp_path: Path) -> None:
    workflow = _module("lunar_exploration_ppo.workflows.stage2")
    result = workflow.run_stage2_workflow(
        config_path=CONFIG,
        run_id="pytest-stage2-machine",
        base_output_root=tmp_path,
    )
    stage = result.stage_root
    assert {path.name for path in stage.iterdir()} == {
        "config.json", "summary.json", "routing.json", "manifest.json", "report.md",
        "metrics.jsonl", "phase-state.jsonl", "evidence",
    }
    assert not (stage / "review.json").exists()
    assert not (stage / "approval.json").exists()
    assert not (stage / "gate.json").exists()
    summary = json.loads((stage / "summary.json").read_text(encoding="utf-8"))
    routing = json.loads((stage / "routing.json").read_text(encoding="utf-8"))
    assert summary["state"] == "machine_passed"
    assert routing["route"] == "awaiting_independent_review"
    assert routing["authorized_next_stage"] == "ppo_highres_frontier_stage3_cross_attention_policy/v1"
    assert summary["acceptance"]["all_18_stage2_items_passed"] is True
    assert summary["leakage_audit"]["hidden_truth_leakage_detected"] is False
    assert summary["leakage_audit"]["paired_fixture_source"] == "paired_hidden_truth_coverable_mutation/v1"
    assert summary["leakage_audit"]["unobserved_truth_mutated"] is True
    assert summary["leakage_audit"]["coverable_mask_mutated"] is True
    assert summary["leakage_audit"]["deployment_prior_byte_equal"] is True
    assert summary["leakage_audit"]["observed_state_byte_equal"] is True
    assert summary["leakage_audit"]["observed_blockers_byte_equal"] is True
    assert summary["leakage_audit"]["action_set_byte_equal"] is True
    assert summary["leakage_audit"]["all_observation_arrays_byte_equal"] is True
    assert summary["catalog_audit"]["child_overlap_pair_count"] == 0
    provenance = json.loads((stage / "evidence" / "provenance.json").read_text(encoding="utf-8"))
    assert "synthetic_0_5m_highres_truth_proxy" not in provenance
    assert "current_observed_highres_state" not in provenance
    lineage = provenance["sample_scenario_lineage"]
    assert lineage["truth"]["highres_shape"] == [256, 256]
    assert lineage["truth"]["resolution_m"] == 0.5
    assert lineage["truth"]["proxy_seed_hex"] == provenance["sample_record"]["proxy_seed_hex"]
    assert lineage["truth"]["synthetic_source_kind"] == "synthetic_terrain_obstacle_proxy/v1"
    assert lineage["truth"]["physical_obstacle_cells_written"] is False
    assert len(lineage["truth"]["truth_sha256"]) == 64
    assert lineage["observed"]["source"] == "sensor_revealed_highres_state/v1"
    assert 0 < lineage["observed"]["observed_cell_count"] < 256 * 256
    assert len(lineage["observed"]["observed_state_sha256"]) == 64
    assert len(lineage["lineage_sha256"]) == 64
    assert summary["scenario_lineage_sha256"] == lineage["lineage_sha256"]
    evidence_names = {path.name for path in (stage / "evidence").iterdir()}
    assert evidence_names == {
        "observation_schema_report.json", "candidate_generation_audit.json", "top_m_audit.json",
        "split_leakage_audit.json", "provenance.json", "sample_observation.npz",
        "scenario_catalog.json",
    }
    manifest = json.loads((stage / "manifest.json").read_text(encoding="utf-8"))
    paths = {entry["path"] for entry in manifest["artifacts"]}
    assert paths == set(workflow.STAGE2_MANIFEST_BOUND_ARTIFACTS)
    workflow.verify_stage2_machine_run(stage_root=stage, repo_root=ROOT)


def test_paired_hidden_truth_fixture_changes_only_unobserved_truth_and_exact_coverable_mask() -> None:
    workflow = _module("lunar_exploration_ppo.workflows.stage2")
    scenario_module = _module("lunar_exploration_ppo.env.scenario")
    scenario = scenario_module.ScenarioSource().load("smoke-v1")
    observed_state, _, _ = workflow._initial_observed_state(scenario)
    pair = workflow._build_hidden_truth_leakage_pair(
        scenario=scenario,
        observed_state=observed_state,
        pose=scenario.start_pose,
        top_m=512,
    )
    baseline = pair.baseline
    mutated = pair.mutated
    assert baseline.prior is mutated.prior
    assert baseline.pose == mutated.pose
    for name in (
        "observed_mask", "confidence", "height", "obstacle", "slope_deg",
        "traversability", "observed_safe_mask",
    ):
        assert getattr(baseline.observed_state, name).tobytes() == getattr(
            mutated.observed_state,
            name,
        ).tobytes()
    observed = baseline.observed_state.observed_mask
    for name in ("height", "hard_obstacle", "slope_deg", "traversability"):
        assert getattr(baseline.truth, name)[observed].tobytes() == getattr(
            mutated.truth,
            name,
        )[observed].tobytes()
    assert any(
        getattr(baseline.truth, name)[~observed].tobytes()
        != getattr(mutated.truth, name)[~observed].tobytes()
        for name in ("height", "hard_obstacle", "slope_deg", "traversability")
    )
    baseline_blockers = observed & (
        baseline.observed_state.obstacle | (baseline.observed_state.slope_deg > 30.0)
    )
    mutated_blockers = observed & (
        mutated.observed_state.obstacle | (mutated.observed_state.slope_deg > 30.0)
    )
    assert baseline_blockers.tobytes() == mutated_blockers.tobytes()
    assert baseline.coverable_mask.tobytes() != mutated.coverable_mask.tobytes()
    assert workflow._action_set_canonical_bytes(baseline.action_set) == workflow._action_set_canonical_bytes(
        mutated.action_set
    )
    for first, second in zip(
        baseline.observation.array_fields(),
        mutated.observation.array_fields(),
        strict=True,
    ):
        assert first.tobytes() == second.tobytes()
    audit = workflow._paired_hidden_truth_leakage_audit(pair)
    assert audit["hidden_truth_leakage_detected"] is False
    assert audit["unobserved_truth_mutated"] is True
    assert audit["coverable_mask_mutated"] is True


def test_stage2_verifier_recomputes_paired_hidden_truth_leakage_evidence(tmp_path: Path) -> None:
    workflow = _module("lunar_exploration_ppo.workflows.stage2")
    result = workflow.run_stage2_workflow(
        config_path=CONFIG,
        run_id="pytest-stage2-leakage-recompute",
        base_output_root=tmp_path,
    )
    relative = "evidence/split_leakage_audit.json"
    evidence_path = result.stage_root / relative
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    evidence["leakage"]["action_set_byte_equal"] = False
    evidence_path.write_bytes(workflow.ArtifactStore.canonical_json_bytes(evidence))
    _rewrite_manifest_entry(workflow, result.stage_root, relative)
    with pytest.raises(workflow.Stage2WorkflowError, match="leakage evidence drift"):
        workflow.verify_stage2_machine_run(stage_root=result.stage_root, repo_root=ROOT)


@pytest.mark.parametrize("tamper_kind", ("catalog", "summary", "npz"))
def test_stage2_verifier_and_review_bindings_reject_synchronized_payload_tamper(
    tmp_path: Path,
    tamper_kind: str,
) -> None:
    workflow = _module("lunar_exploration_ppo.workflows.stage2")
    result = workflow.run_stage2_workflow(
        config_path=CONFIG,
        run_id=f"pytest-stage2-sync-tamper-{tamper_kind}",
        base_output_root=tmp_path,
    )
    if tamper_kind == "catalog":
        relative = "evidence/scenario_catalog.json"
        path = result.stage_root / relative
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["records"][0]["density_profile"] = "synchronized_tamper"
        path.write_bytes(workflow.ArtifactStore.canonical_json_bytes(payload))
    elif tamper_kind == "summary":
        relative = "summary.json"
        path = result.stage_root / relative
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["candidate_audit"]["candidate_count_before_top_m"] += 1
        path.write_bytes(workflow.ArtifactStore.canonical_json_bytes(payload))
    else:
        relative = "evidence/sample_observation.npz"
        path = result.stage_root / relative
        with np.load(path, allow_pickle=False) as archive:
            arrays = {name: np.asarray(archive[name]).copy() for name in archive.files}
        arrays["frontier_features"][0, 0] += np.float32(0.125)
        np.savez_compressed(path, **arrays)
    _rewrite_manifest_entry(workflow, result.stage_root, relative)
    machine_config = json.loads(
        (result.stage_root / "config.json").read_text(encoding="utf-8")
    )
    review_base = machine_config["stage1_approval_commit"]
    package = workflow.prepare_stage2_review_package(
        repo_root=ROOT,
        package_path=tmp_path / f"{tamper_kind}-review.patch",
        base_commit=review_base,
    )
    assert package.base_commit == review_base

    with pytest.raises(workflow.Stage2WorkflowError, match="deterministic payload drift"):
        workflow.verify_stage2_machine_run(stage_root=result.stage_root, repo_root=ROOT)
    with pytest.raises(workflow.Stage2WorkflowError, match="deterministic payload drift"):
        workflow.inspect_stage2_review_bindings(
            stage_root=result.stage_root,
            repo_root=ROOT,
            review_package=package.path,
        )


def test_stage2_verifier_fails_closed_on_artifact_and_config_drift(tmp_path: Path) -> None:
    workflow = _module("lunar_exploration_ppo.workflows.stage2")
    result = workflow.run_stage2_workflow(
        config_path=CONFIG,
        run_id="pytest-stage2-drift",
        base_output_root=tmp_path,
    )
    with (result.stage_root / "metrics.jsonl").open("ab") as stream:
        stream.write(b'{"drift":true}\n')
    with pytest.raises(workflow.Stage2WorkflowError, match="artifact drift"):
        workflow.verify_stage2_machine_run(stage_root=result.stage_root, repo_root=ROOT)

    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    payload["catalog_counts"]["train"] = 699
    drifted_config = tmp_path / "drifted-config.json"
    drifted_config.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(Exception):
        _module("lunar_exploration_ppo.configs.stage2").load_stage2_config(drifted_config)


def test_stage2_verifier_fails_closed_on_runner_source_drift(tmp_path: Path) -> None:
    workflow = _module("lunar_exploration_ppo.workflows.stage2")
    result = workflow.run_stage2_workflow(
        config_path=CONFIG,
        run_id="pytest-stage2-runner-drift",
        base_output_root=tmp_path,
    )
    machine_config = json.loads((result.stage_root / "config.json").read_text(encoding="utf-8"))
    isolated_repo = tmp_path / "isolated-repo"
    for entry in machine_config["execution_source_identity"]["paths"]:
        relative = Path(entry["path"])
        destination = isolated_repo / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)
    runner_relative = Path("scripts/run_ppo_highres_frontier_stage2.py")
    isolated_runner = isolated_repo / runner_relative
    isolated_runner.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / runner_relative, isolated_runner)
    isolated_runner.write_bytes(isolated_runner.read_bytes() + b"\n# isolated runner drift\n")

    with pytest.raises(workflow.Stage2WorkflowError, match="Stage 2 source drift"):
        workflow.verify_stage2_machine_run(
            stage_root=result.stage_root,
            repo_root=isolated_repo,
        )


def test_stage2_independent_review_rejects_self_made_json_with_valid_bindings(
    tmp_path: Path,
) -> None:
    workflow = _module("lunar_exploration_ppo.workflows.stage2")
    result, package, bindings = _stage2_review_inputs(
        workflow,
        tmp_path,
        "pytest-stage2-json-review-rejected",
    )
    report = tmp_path / "self-made-review.json"
    legacy_binding_keys = {
        "reviewed_base_commit",
        "reviewed_prospective_git_tree",
        "reviewed_paths",
        "reviewed_path_set_sha256",
        "review_package_sha256",
        "source_set_sha256",
        "data_sha256",
        "config_sha256",
        "environment_sha256",
        "manifest_sha256",
        "authorized_next_stage",
    }
    report.write_text(
        json.dumps(
            {
                "reviewer_id": "independent-reviewer",
                "spec_verdict": "approved",
                "quality_verdict": "approved",
                "issue_counts": {"critical": 0, "important": 0, "minor": 0},
                **{key: bindings[key] for key in legacy_binding_keys},
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    with pytest.raises(workflow.Stage2WorkflowError, match="strict front matter"):
        workflow.record_stage2_independent_review(
            stage_root=result.stage_root,
            repo_root=ROOT,
            review_report=report,
            review_package=package.path,
        )


def test_stage2_independent_review_accepts_strict_semantically_bound_markdown(
    tmp_path: Path,
) -> None:
    workflow = _module("lunar_exploration_ppo.workflows.stage2")
    result, package, bindings = _stage2_review_inputs(
        workflow,
        tmp_path,
        "pytest-stage2-strict-review",
    )
    report = tmp_path / "strict-independent-review.md"
    report.write_text(_strict_stage2_review_markdown(bindings), encoding="utf-8", newline="")

    review_path = workflow.record_stage2_independent_review(
        stage_root=result.stage_root,
        repo_root=ROOT,
        review_report=report,
        review_package=package.path,
    )

    review = json.loads(review_path.read_text(encoding="utf-8"))
    assert review["schema_version"] == "ppo_highres_frontier_stage2_independent_review/v3"
    assert review["state"] == "awaiting_human_approval"
    assert review["reviewer_id"] == "independent-reviewer"
    assert review["spec_verdict"] == "APPROVED"
    assert review["quality_verdict"] == "APPROVED"
    assert review["issue_counts"] == {"critical": 0, "important": 0, "minor": 0}
    assert review["final_conclusion"] == "READY_FOR_HUMAN_APPROVAL"
    assert review["review_report"]["sha256"] == hashlib.sha256(report.read_bytes()).hexdigest()
    assert review["review_package"]["sha256"] == hashlib.sha256(package.path.read_bytes()).hexdigest()
    assert review["review_recorded_at_utc"].endswith("Z")
    assert len(review["approval_challenge"]) == 64


def test_stage2_independent_review_rejects_malformed_or_semantically_mismatched_markdown(
    tmp_path: Path,
) -> None:
    workflow = _module("lunar_exploration_ppo.workflows.stage2")
    result, package, bindings = _stage2_review_inputs(
        workflow,
        tmp_path,
        "pytest-stage2-invalid-markdown-review",
    )
    valid = _strict_stage2_review_markdown(bindings)
    cases = (
        ("# no strict front matter\n", "strict front matter"),
        (
            _strict_stage2_review_markdown(
                bindings,
                body_overrides={"spec_verdict": "REJECTED"},
            ),
            "body verdict",
        ),
        (
            _strict_stage2_review_markdown(
                bindings,
                front_overrides={
                    "review_package_bytes": int(bindings["review_package_bytes"]) + 1
                },
            ),
            "package declarations",
        ),
        (
            _strict_stage2_review_markdown(
                bindings,
                body_overrides={"important_count": 1},
            ),
            "body count",
        ),
        (
            _strict_stage2_review_markdown(
                bindings,
                minor_count=1,
                findings_line="Findings: None",
            ),
            "Findings: None",
        ),
    )
    assert valid.startswith("---\n")
    for index, (payload, message) in enumerate(cases):
        report = tmp_path / f"invalid-review-{index}.md"
        report.write_text(payload, encoding="utf-8", newline="")
        with pytest.raises(workflow.Stage2WorkflowError, match=message):
            workflow.record_stage2_independent_review(
                stage_root=result.stage_root,
                repo_root=ROOT,
                review_report=report,
                review_package=package.path,
            )


def test_stage2_strict_review_formatter_is_reusable_without_writing_authority() -> None:
    workflow = _module("lunar_exploration_ppo.workflows.stage2")
    bindings: dict[str, object] = {
        "goal_id": "ppo-highres-frontier-map-exploration",
        "stage_id": "ppo_highres_frontier_stage2_observation_frontier/v1",
        "run_id": "reviewer-fixture",
        "reviewed_base_commit": "1" * 40,
        "reviewed_prospective_git_tree": "2" * 40,
        "review_package_sha256": "3" * 64,
        "review_package_bytes": 123,
        "review_package_lf_count": 4,
        "review_package_logical_line_count": 4,
        "reviewed_paths": ["a.py", "b.py"],
        "reviewed_path_set_sha256": "4" * 64,
        "manifest_sha256": "5" * 64,
        "config_sha256": "6" * 64,
        "data_sha256": "7" * 64,
        "environment_sha256": "8" * 64,
        "source_set_sha256": "9" * 64,
        "authorized_next_stage": "ppo_highres_frontier_stage3_cross_attention_policy/v1",
    }
    rendered = workflow.render_stage2_strict_review_report(
        bindings=bindings,
        reviewer_id="fresh-reviewer",
        spec_verdict="APPROVED",
        quality_verdict="APPROVED",
        critical_count=0,
        important_count=0,
        minor_count=0,
        final_conclusion="READY_FOR_HUMAN_APPROVAL",
        findings_markdown="",
    )
    assert rendered == _strict_stage2_review_markdown(
        bindings,
        reviewer_id="fresh-reviewer",
    )


def test_stage2_authority_path_replays_exact_prospective_tree_and_only_authorizes_stage3(
    tmp_path: Path,
) -> None:
    """Only strict review plus a controller-attested user event reaches the gate."""
    workflow = _module("lunar_exploration_ppo.workflows.stage2")
    result, package, bindings, _report, review_path, review_payload = (
        _record_strict_stage2_review(
            workflow,
            tmp_path,
            "pytest-stage2-authority",
        )
    )
    machine_config = json.loads((result.stage_root / "config.json").read_text(encoding="utf-8"))
    routing = json.loads((result.stage_root / "routing.json").read_text(encoding="utf-8"))
    expected_next_stage = "ppo_highres_frontier_stage3_cross_attention_policy/v1"
    assert not hasattr(workflow, "STAGE2_AUTHORIZED_NEXT_STAGE")
    assert machine_config["authorized_next_stage"] == expected_next_stage
    assert routing["authorized_next_stage"] == expected_next_stage
    assert bindings["authorized_next_stage"] == expected_next_stage
    assert review_payload["state"] == "awaiting_human_approval"
    assert review_payload["authorized_next_stage"] == expected_next_stage
    assert workflow.replay_stage2_review_package(
        repo_root=ROOT,
        base_commit=package.base_commit,
        review_package=package.path,
        expected_prospective_git_tree=package.prospective_git_tree,
        expected_reviewed_paths=package.paths,
    ) == package.prospective_git_tree

    approval, gate, verified = _simulate_external_controller_authority_writer(
        workflow,
        result.stage_root,
    )
    assert hashlib.sha256(workflow.STAGE2_APPROVAL_TEXT.encode("utf-8")).hexdigest() == json.loads(
        approval.read_text(encoding="utf-8")
    )["approval_text_utf8_sha256"]
    approval_payload = json.loads(approval.read_text(encoding="utf-8"))
    assert approval_payload["bindings"]["authorized_next_stage"] == expected_next_stage
    assert approval_payload["authority_kind"] == _STAGE2_EXTERNAL_AUTHORITY_KIND
    assert approval_payload["event_authenticity_source"] == (
        _STAGE2_EXTERNAL_AUTHENTICITY_SOURCE
    )
    assert approval_payload["local_verification_scope"] == (
        _STAGE2_LOCAL_VERIFICATION_SCOPE
    )
    assert approval_payload["approval_challenge"] == review_payload["approval_challenge"]
    assert verified["state"] == "next_stage"
    assert verified["authorized_next_stage"] == expected_next_stage
    assert verified["bindings"]["authorized_next_stage"] == expected_next_stage
    assert [event["state"] for event in verified["history"]] == [
        "machine_passed", "awaiting_independent_review", "awaiting_human_approval", "approved", "next_stage"
    ]
    original_approval = approval.read_bytes()
    tampered_approval = json.loads(original_approval.decode("utf-8"))
    tampered_approval["approval_challenge"] = "0" * 64
    approval.write_bytes(workflow.ArtifactStore.canonical_json_bytes(tampered_approval))
    with pytest.raises(workflow.Stage2WorkflowError, match="approval.*(?:schema|binding|challenge)"):
        workflow.load_verified_stage2_gate(gate_path=gate, repo_root=ROOT)
    approval.write_bytes(original_approval)

    original_review = review_path.read_bytes()
    tampered_review = json.loads(original_review.decode("utf-8"))
    tampered_review["approval_challenge"] = "0" * 64
    review_path.write_bytes(workflow.ArtifactStore.canonical_json_bytes(tampered_review))
    with pytest.raises(workflow.Stage2WorkflowError, match="review.*binding"):
        workflow.load_verified_stage2_gate(gate_path=gate, repo_root=ROOT)
    review_path.write_bytes(original_review)

    verified["authorized_next_stage"] = "ppo_highres_frontier_stage3_unbound_tamper/v1"
    gate.write_bytes(workflow.ArtifactStore.canonical_json_bytes(verified))
    with pytest.raises(workflow.Stage2WorkflowError, match="gate state is invalid"):
        workflow.load_verified_stage2_gate(gate_path=gate, repo_root=ROOT)


def test_untrusted_import_caller_cannot_mint_approval_or_gate_after_strict_review(
    tmp_path: Path,
) -> None:
    """A local importer must have no production authority-issuance route."""
    workflow = _module("lunar_exploration_ppo.workflows.stage2")
    result, _package, _bindings, _report, review_path, review = (
        _record_strict_stage2_review(
            workflow,
            tmp_path,
            "pytest-stage2-untrusted-import",
        )
    )
    issuer_names = {
        "_stage2_trusted_controller_attestor",
        "attest_stage2_controller_user_event",
        "record_stage2_human_approval",
        "record_stage2_verified_gate",
        "verify_stage2_gate_context",
        "Stage2GateContext",
        "Stage2ControllerAttestedUserEvent",
    }
    exposed = sorted(name for name in issuer_names if hasattr(workflow, name))
    assert exposed == []
    production_source = Path(workflow.__file__).read_text(encoding="utf-8")
    assert 'write_json_exclusive("approval.json"' not in production_source
    assert 'write_json_exclusive("gate.json"' not in production_source
    assert "external controller" in workflow.verify_stage2_external_approval.__doc__.lower()
    assert not (result.stage_root / "approval.json").exists()
    assert not (result.stage_root / "gate.json").exists()


def test_stage2_production_package_exposes_verifiers_but_no_authority_capabilities() -> None:
    workflow = _module("lunar_exploration_ppo.workflows.stage2")
    assert callable(workflow.inspect_stage2_review_authority)
    assert callable(workflow.verify_stage2_external_approval)
    assert callable(workflow.load_verified_stage2_gate)
    for forbidden in (
        "Stage2GateContext",
        "Stage2ControllerAttestedUserEvent",
        "_stage2_trusted_controller_attestor",
        "attest_stage2_controller_user_event",
        "record_stage2_human_approval",
        "record_stage2_verified_gate",
    ):
        assert not hasattr(workflow, forbidden)


def test_stage2_raw_uuid_text_and_timestamp_are_not_sufficient_for_approval() -> None:
    workflow = _module("lunar_exploration_ppo.workflows.stage2")
    with pytest.raises(TypeError):
        workflow.verify_stage2_external_approval(
            stage_root=ROOT,
            repo_root=ROOT,
            thread_id="00000000-0000-0000-0000-000000000001",
            user_turn_id="00000000-0000-0000-0000-000000000002",
            approval_text=workflow.STAGE2_APPROVAL_TEXT,
            approval_timestamp_utc="2099-07-13T00:00:01Z",
        )


def test_stage2_external_approval_verifier_rejects_time_challenge_and_review_drift(
    tmp_path: Path,
) -> None:
    workflow = _module("lunar_exploration_ppo.workflows.stage2")
    result, *_rest = _record_strict_stage2_review(
        workflow,
        tmp_path / "template",
        "pytest-stage2-external-rejections",
    )

    before_review = _clone_reviewed_stage(result.stage_root, tmp_path / "before-review")
    before_review.joinpath("review.json").unlink()
    payload = _external_controller_approval_payload(workflow, result.stage_root)
    workflow.ArtifactStore(before_review).write_json_exclusive("approval.json", payload)
    with pytest.raises(workflow.Stage2WorkflowError, match="review/approval artifact set drift"):
        workflow.verify_stage2_external_approval(
            stage_root=before_review,
            repo_root=ROOT,
        )

    predates = _clone_reviewed_stage(result.stage_root, tmp_path / "predates")
    _write_external_stage2_approval(
        workflow,
        predates,
        event_timestamp_utc="2000-01-01T00:00:00Z",
    )
    with pytest.raises(workflow.Stage2WorkflowError, match="predates review"):
        workflow.verify_stage2_external_approval(stage_root=predates, repo_root=ROOT)

    wrong_challenge = _clone_reviewed_stage(result.stage_root, tmp_path / "wrong-challenge")
    _write_external_stage2_approval(
        workflow,
        wrong_challenge,
        approval_challenge="0" * 64,
    )
    with pytest.raises(workflow.Stage2WorkflowError, match="approval schema"):
        workflow.verify_stage2_external_approval(
            stage_root=wrong_challenge,
            repo_root=ROOT,
        )

    wrong_review = _clone_reviewed_stage(result.stage_root, tmp_path / "wrong-review")
    _write_external_stage2_approval(
        workflow,
        wrong_review,
        review_sha256="0" * 64,
    )
    with pytest.raises(workflow.Stage2WorkflowError, match="approval schema"):
        workflow.verify_stage2_external_approval(stage_root=wrong_review, repo_root=ROOT)

    missing_challenge = _clone_reviewed_stage(
        result.stage_root,
        tmp_path / "missing-challenge",
    )
    missing_payload = _external_controller_approval_payload(workflow, missing_challenge)
    missing_payload.pop("approval_challenge")
    workflow.ArtifactStore(missing_challenge).write_json_exclusive(
        "approval.json",
        missing_payload,
    )
    with pytest.raises(workflow.Stage2WorkflowError, match="approval schema"):
        workflow.verify_stage2_external_approval(
            stage_root=missing_challenge,
            repo_root=ROOT,
        )


def test_stage2_approval_rejects_wrong_event_context_and_duplicate_writes(
    tmp_path: Path,
) -> None:
    workflow = _module("lunar_exploration_ppo.workflows.stage2")
    first = _record_strict_stage2_review(
        workflow,
        tmp_path / "first",
        "pytest-stage2-event-first",
    )
    second = _record_strict_stage2_review(
        workflow,
        tmp_path / "second",
        "pytest-stage2-event-second",
    )
    first_bindings = workflow.inspect_stage2_review_authority(
        repo_root=ROOT,
        stage_root=first[0].stage_root,
    )
    wrong_context = _clone_reviewed_stage(
        second[0].stage_root,
        tmp_path / "wrong-context",
    )
    _write_external_stage2_approval(
        workflow,
        wrong_context,
        event_context_bindings=first_bindings,
    )
    with pytest.raises(workflow.Stage2WorkflowError, match="event binding"):
        workflow.verify_stage2_external_approval(
            stage_root=wrong_context,
            repo_root=ROOT,
        )

    approval, _payload = _write_external_stage2_approval(
        workflow,
        second[0].stage_root,
    )
    workflow.verify_stage2_external_approval(
        stage_root=second[0].stage_root,
        repo_root=ROOT,
    )
    with pytest.raises(FileExistsError):
        workflow.ArtifactStore(second[0].stage_root).write_json_exclusive(
            "approval.json",
            json.loads(approval.read_text(encoding="utf-8")),
        )
    gate, gate_payload = _write_external_stage2_gate(workflow, second[0].stage_root)
    workflow.load_verified_stage2_gate(gate_path=gate, repo_root=ROOT)
    with pytest.raises(FileExistsError):
        workflow.ArtifactStore(second[0].stage_root).write_json_exclusive(
            "gate.json",
            gate_payload,
        )


def test_stage2_authority_rejects_review_package_source_and_artifact_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workflow = _module("lunar_exploration_ppo.workflows.stage2")
    result, package, _bindings, report, _review_path, _review = (
        _record_strict_stage2_review(
            workflow,
            tmp_path,
            "pytest-stage2-authority-drift",
        )
    )
    original_report = report.read_bytes()
    original_package = package.path.read_bytes()
    report.write_bytes(original_report + b"drift\n")
    with pytest.raises(workflow.Stage2WorkflowError, match="review report.*drift"):
        workflow.inspect_stage2_review_authority(
            repo_root=ROOT,
            stage_root=result.stage_root,
        )
    report.write_bytes(original_report)
    package.path.write_bytes(original_package + b"drift\n")
    with pytest.raises(workflow.Stage2WorkflowError, match="review package.*drift"):
        workflow.inspect_stage2_review_authority(
            repo_root=ROOT,
            stage_root=result.stage_root,
        )
    package.path.write_bytes(original_package)

    original_source_identity = workflow._source_identity
    monkeypatch.setattr(
        workflow,
        "_source_identity",
        lambda repo: {
            **original_source_identity(repo),
            "source_set_sha256": "0" * 64,
        },
    )
    with pytest.raises(workflow.Stage2WorkflowError, match="source.*drift"):
        workflow.inspect_stage2_review_authority(
            repo_root=ROOT,
            stage_root=result.stage_root,
        )
    monkeypatch.setattr(workflow, "_source_identity", original_source_identity)

    metrics = result.stage_root / "metrics.jsonl"
    metrics.write_bytes(metrics.read_bytes() + b"{}\n")
    with pytest.raises(workflow.Stage2WorkflowError, match="artifact drift"):
        workflow.inspect_stage2_review_authority(
            repo_root=ROOT,
            stage_root=result.stage_root,
        )


def test_stage2_runner_reproducibly_writes_only_machine_artifacts(tmp_path: Path) -> None:
    runner = ROOT / "scripts" / "run_ppo_highres_frontier_stage2.py"
    assert runner.is_file()
    completed = subprocess.run(
        [
            sys.executable,
            str(runner),
            "--config", str(CONFIG),
            "--run-id", "pytest-stage2-runner",
            "--output-root", str(tmp_path),
        ],
        check=False,
        cwd=ROOT,
        env={**__import__("os").environ, "PYTHONPATH": str(ROOT / "src")},
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    stage = tmp_path / "pytest-stage2-runner" / "s2"
    assert {path.name for path in stage.iterdir()} == {
        "config.json", "summary.json", "routing.json", "manifest.json", "report.md",
        "metrics.jsonl", "phase-state.jsonl", "evidence",
    }
