import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = str(REPO_ROOT / "scripts")
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)


def test_canonical_artifact_is_preferred_over_legacy(tmp_path: Path) -> None:
    import xunce_artifact_io as artifact_io
    from xunce_artifact_paths import STAGE21_1_SUMMARY, artifact_path, read_json_artifact

    root = tmp_path / "s21_1"
    artifact_io.write_json(artifact_path(root, STAGE21_1_SUMMARY, canonical=False), {"status": "legacy"})
    artifact_io.write_json(artifact_path(root, STAGE21_1_SUMMARY), {"status": "canonical"})

    payload, source = read_json_artifact(root, STAGE21_1_SUMMARY)

    assert source == "canonical"
    assert payload["status"] == "canonical"
    assert payload["artifact_source"] == "canonical"


def test_legacy_artifact_is_read_when_canonical_missing(tmp_path: Path) -> None:
    import xunce_artifact_io as artifact_io
    from xunce_artifact_paths import STAGE21_2_REWARDS, artifact_path, read_jsonl_artifact

    root = tmp_path / "s21_2"
    artifact_io.write_jsonl(artifact_path(root, STAGE21_2_REWARDS, canonical=False), [{"transition_id": "t1"}])

    rows, source = read_jsonl_artifact(root, STAGE21_2_REWARDS)

    assert source == "legacy"
    assert rows == [{"transition_id": "t1"}]


def test_dual_write_creates_canonical_and_legacy_files(tmp_path: Path) -> None:
    import xunce_artifact_io as artifact_io
    from xunce_artifact_paths import STAGE21_3_BATCH, STAGE21_3_SUMMARY, artifact_path, write_json_artifact, write_jsonl_artifact

    root = tmp_path / "s21_3"
    write_json_artifact(root, STAGE21_3_SUMMARY, {"status": "passed"})
    write_jsonl_artifact(root, STAGE21_3_BATCH, [{"transition_id": "t1"}])

    assert artifact_io.path_is_file(artifact_path(root, STAGE21_3_SUMMARY))
    assert artifact_io.path_is_file(artifact_path(root, STAGE21_3_SUMMARY, canonical=False))
    assert artifact_io.path_is_file(artifact_path(root, STAGE21_3_BATCH))
    assert artifact_io.path_is_file(artifact_path(root, STAGE21_3_BATCH, canonical=False))
    assert artifact_io.read_json(artifact_path(root, STAGE21_3_SUMMARY))["artifact_source"] == "canonical"
    assert artifact_io.read_json(artifact_path(root, STAGE21_3_SUMMARY, canonical=False))["artifact_source"] == "legacy"


def test_stage21_4_stage21_5_and_stage26_aliases_dual_write(tmp_path: Path) -> None:
    import xunce_artifact_io as artifact_io
    from xunce_artifact_paths import (
        STAGE21_4_CHECKPOINT,
        STAGE21_4_GRADIENT,
        STAGE21_4_LOSS,
        STAGE21_4_MANIFEST,
        STAGE21_4_ROUTING,
        STAGE21_4_SUMMARY,
        STAGE21_5_RESULTS,
        STAGE21_5_ROUTING,
        STAGE21_5_DELTA,
        STAGE21_5_MANIFEST,
        STAGE21_5_SCENARIO_DELTA,
        STAGE21_5_SUMMARY,
        STAGE26_2_BATCH_AUDIT,
        STAGE26_2_CHECKPOINT_BOUNDARY_AUDIT,
        STAGE26_2_LOSS_GRADIENT_AUDIT,
        STAGE26_2_MANIFEST,
        STAGE26_2_ROUTING,
        STAGE26_2_STAGE21_4_CONFIG,
        STAGE26_2_STAGE21_4_SUMMARY,
        STAGE26_2_SUMMARY,
        STAGE26_3_ACTION_AUDIT,
        STAGE26_3_DELTA_AUDIT,
        STAGE26_3_HIGH_FIDELITY_CONFIG,
        STAGE26_3_MANIFEST,
        STAGE26_3_ROUTING,
        STAGE26_3_STAGE21_5_CONFIG,
        STAGE26_3_STAGE21_5_SUMMARY,
        STAGE26_3_SUMMARY,
        artifact_path,
        read_json_artifact,
        read_jsonl_artifact,
        write_json_artifact,
        write_jsonl_artifact,
    )

    root = tmp_path / "aliases"
    write_json_artifact(root / "s21_4", STAGE21_4_SUMMARY, {"status": "passed"})
    write_json_artifact(root / "s21_4", STAGE21_4_CHECKPOINT, {"checkpoint_reload_passed": True})
    write_json_artifact(root / "s21_4", STAGE21_4_GRADIENT, {"grad_norm_finite": True})
    write_json_artifact(root / "s21_4", STAGE21_4_ROUTING, {"next_required_change": "stage21_5"})
    write_json_artifact(root / "s21_4", STAGE21_4_MANIFEST, {"summary_status": "passed"})
    write_jsonl_artifact(root / "s21_4", STAGE21_4_LOSS, [{"epoch_index": 0}])
    write_json_artifact(root / "s21_5", STAGE21_5_SUMMARY, {"status": "passed"})
    write_json_artifact(root / "s21_5", STAGE21_5_RESULTS, {"pre_ppo_xunce": {}})
    write_json_artifact(root / "s21_5", STAGE21_5_DELTA, {"coverage_per_100m_mean_delta": 1.0})
    write_json_artifact(root / "s21_5", STAGE21_5_ROUTING, {"next_required_change": "stage21_6"})
    write_json_artifact(root / "s21_5", STAGE21_5_MANIFEST, {"summary_status": "passed"})
    write_jsonl_artifact(root / "s21_5", STAGE21_5_SCENARIO_DELTA, [{"scenario_id": "s0"}])
    write_json_artifact(root / "s26_2", STAGE26_2_SUMMARY, {"status": "passed"})
    write_json_artifact(root / "s26_2", STAGE26_2_STAGE21_4_CONFIG, {"epochs": 1})
    write_json_artifact(root / "s26_2", STAGE26_2_STAGE21_4_SUMMARY, {"status": "passed"})
    write_json_artifact(root / "s26_2", STAGE26_2_BATCH_AUDIT, {"batch_row_count": 1})
    write_json_artifact(root / "s26_2", STAGE26_2_LOSS_GRADIENT_AUDIT, {"loss_finite": True})
    write_json_artifact(root / "s26_2", STAGE26_2_CHECKPOINT_BOUNDARY_AUDIT, {"checkpoint_boundary_passed": True})
    write_json_artifact(root / "s26_2", STAGE26_2_ROUTING, {"next_required_change": "stage26_3"})
    write_json_artifact(root / "s26_2", STAGE26_2_MANIFEST, {"summary_status": "passed"})
    write_json_artifact(root / "s26_3", STAGE26_3_SUMMARY, {"status": "passed"})
    write_json_artifact(root / "s26_3", STAGE26_3_STAGE21_5_CONFIG, {"rollout_steps": 4})
    write_json_artifact(root / "s26_3", STAGE26_3_HIGH_FIDELITY_CONFIG, {"xunce_only_evaluation": True})
    write_json_artifact(root / "s26_3", STAGE26_3_STAGE21_5_SUMMARY, {"status": "passed"})
    write_json_artifact(root / "s26_3", STAGE26_3_ACTION_AUDIT, {"strong_state_join_available_count": 1})
    write_json_artifact(root / "s26_3", STAGE26_3_DELTA_AUDIT, {"coverage_per_100m_delta": 0.0})
    write_json_artifact(root / "s26_3", STAGE26_3_ROUTING, {"next_required_change": "stage26_4"})
    write_json_artifact(root / "s26_3", STAGE26_3_MANIFEST, {"summary_status": "passed"})

    assert artifact_io.path_is_file(artifact_path(root / "s21_4", STAGE21_4_SUMMARY))
    assert artifact_io.path_is_file(artifact_path(root / "s21_4", STAGE21_4_SUMMARY, canonical=False))
    assert artifact_io.path_is_file(artifact_path(root / "s21_5", STAGE21_5_DELTA))
    assert artifact_io.path_is_file(artifact_path(root / "s21_5", STAGE21_5_DELTA, canonical=False))
    assert read_json_artifact(root / "s21_4", STAGE21_4_CHECKPOINT)[0]["checkpoint_reload_passed"] is True
    assert read_json_artifact(root / "s21_4", STAGE21_4_GRADIENT)[0]["grad_norm_finite"] is True
    assert read_json_artifact(root / "s21_5", STAGE21_5_SUMMARY)[0]["status"] == "passed"
    assert read_json_artifact(root / "s21_5", STAGE21_5_RESULTS)[0]["pre_ppo_xunce"] == {}
    assert read_jsonl_artifact(root / "s21_4", STAGE21_4_LOSS)[0] == [{"epoch_index": 0}]
    assert read_jsonl_artifact(root / "s21_5", STAGE21_5_SCENARIO_DELTA)[0] == [{"scenario_id": "s0"}]
    assert read_json_artifact(root / "s26_2", STAGE26_2_STAGE21_4_CONFIG)[0]["epochs"] == 1
    assert read_json_artifact(root / "s26_2", STAGE26_2_BATCH_AUDIT)[0]["batch_row_count"] == 1
    assert read_json_artifact(root / "s26_3", STAGE26_3_ACTION_AUDIT)[0]["strong_state_join_available_count"] == 1
    assert read_json_artifact(root / "s26_3", STAGE26_3_HIGH_FIDELITY_CONFIG)[0]["xunce_only_evaluation"] is True
