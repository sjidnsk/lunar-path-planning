import json
import sys
from pathlib import Path


def test_missing_default_inputs_route_to_xunce_generation(tmp_path: Path) -> None:
    repo_root = _write_minimal_repo(tmp_path)
    output_root = tmp_path / "audit-output"
    scripts_path = str(Path(__file__).resolve().parents[1] / "scripts")
    if scripts_path not in sys.path:
        sys.path.insert(0, scripts_path)
    from scripts.run_xunce_model_comparison_input_readiness import run_xunce_model_comparison_input_readiness

    audit = run_xunce_model_comparison_input_readiness(
        repo_root=repo_root,
        output_root=output_root,
        roi_expansion_config=repo_root / "configs" / "xunce_high_fidelity_real_map_roi_expansion_v1.json",
        comparison_config=repo_root / "configs" / "xunce_high_fidelity_real_map_comparison_v1.json",
    )

    assert audit["status"] == "blocked"
    assert audit["ready_for_stage_18a"] is False
    assert audit["ready_for_stage_18b"] is False
    assert audit["next_required_change"] == "restore_or_generate_xunce_training_candidate"
    assert "xunce_candidate_checkpoint" in audit["missing_required_inputs"]
    assert "xunce_release_governance_summary" in audit["missing_required_inputs"]
    assert (output_root / "input-readiness-audit.json").is_file()


def test_complete_tiny_inputs_route_to_stage_18b(tmp_path: Path) -> None:
    repo_root = _write_minimal_repo(tmp_path)
    release_root = repo_root / "outputs" / "release"
    domain_root = repo_root / "outputs" / "domain-gap"
    roi_root = repo_root / "outputs" / "roi"
    checkpoint_root = repo_root / "outputs" / "checkpoints"
    output_root = tmp_path / "audit-output"
    _write_json(release_root / "xunce-release-governance-gate-summary.json", {"status": "passed"})
    for filename in (
        "quasi-real-map-domain-gap-summary.json",
        "quasi-real-map-path-feedback-summary.json",
    ):
        _write_json(domain_root / filename, {"status": "passed"})
    (domain_root / "quasi-real-map-slices.jsonl").parent.mkdir(parents=True, exist_ok=True)
    (domain_root / "quasi-real-map-slices.jsonl").write_text('{"scenario_id":"s0"}\n', encoding="utf-8")
    _write_json(roi_root / "xunce-high-fidelity-real-map-roi-expansion-summary.json", {"status": "passed"})
    _write_json(roi_root / "xunce-high-fidelity-path-feedback-audit.json", {"status": "passed"})
    (roi_root / "xunce-high-fidelity-real-map-slices.jsonl").write_text('{"scenario_id":"s0"}\n', encoding="utf-8")
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    (checkpoint_root / "xunce.pt").write_bytes(b"xunce")
    (checkpoint_root / "incumbent.pt").write_bytes(b"incumbent")

    scripts_path = str(Path(__file__).resolve().parents[1] / "scripts")
    if scripts_path not in sys.path:
        sys.path.insert(0, scripts_path)
    from scripts.run_xunce_model_comparison_input_readiness import run_xunce_model_comparison_input_readiness

    audit = run_xunce_model_comparison_input_readiness(
        repo_root=repo_root,
        output_root=output_root,
        roi_expansion_config=repo_root / "configs" / "xunce_high_fidelity_real_map_roi_expansion_v1.json",
        comparison_config=repo_root / "configs" / "xunce_high_fidelity_real_map_comparison_v1.json",
        source_release_root=str(release_root),
        source_domain_gap_root=str(domain_root),
        source_roi_expansion_root=str(roi_root),
        xunce_candidate_checkpoint=str(checkpoint_root / "xunce.pt"),
        incumbent_policy_checkpoint=str(checkpoint_root / "incumbent.pt"),
    )

    assert audit["status"] == "ready"
    assert audit["ready_for_stage_18a"] is True
    assert audit["ready_for_stage_18b"] is True
    assert audit["next_required_change"] == "ready_for_stage_18b"
    assert audit["missing_required_inputs"] == []
    assert audit["publishes_checkpoint"] is False
    assert audit["replaces_default_policy"] is False
    assert audit["connects_real_executor"] is False
    assert audit["starts_online_canary"] is False
    persisted = json.loads((output_root / "input-readiness-audit.json").read_text(encoding="utf-8"))
    assert persisted["schema_version"] == "xunce-model-comparison-input-readiness-audit/v1"


def _write_minimal_repo(root: Path) -> Path:
    repo_root = root / "repo"
    (repo_root / "configs").mkdir(parents=True, exist_ok=True)
    manifest = repo_root / "model-explorer" / "data" / "manifests" / "lunar_south_pole_lro_lola_selection_matrix_v1.json"
    _write_json(manifest, {"schema_version": "model-explorer-quasi-real-evaluation/v1"})
    _write_json(
        repo_root / "configs" / "xunce_high_fidelity_real_map_roi_expansion_v1.json",
        {
            "schema_version": "xunce-high-fidelity-real-map-roi-expansion-config/v1",
            "source_xunce_release_governance_root": "outputs/path_feedback_batch_xunce_release_governance_gate_v1",
            "source_quasi_real_domain_gap_root": "outputs/path_feedback_batch_quasi_real_map_domain_gap_v1",
            "source_matrix_manifest": "model-explorer/data/manifests/lunar_south_pole_lro_lola_selection_matrix_v1.json",
        },
    )
    _write_json(
        repo_root / "configs" / "xunce_high_fidelity_real_map_comparison_v1.json",
        {
            "schema_version": "xunce-high-fidelity-real-map-comparison-config/v1",
            "source_roi_expansion_root": "outputs/path_feedback_batch_xunce_high_fidelity_real_map_roi_expansion_v1",
            "xunce_candidate_checkpoint": "outputs/path_feedback_batch_xunce_sandbox_candidate_preflight_v1/sandbox_package/xunce-controlled-training-candidate.pt",
            "incumbent_policy_checkpoint": "outputs/path_feedback_batch_value_stability_candidate_v1/experimental-hybrid-policy-candidate.pt",
        },
    )
    return repo_root


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
