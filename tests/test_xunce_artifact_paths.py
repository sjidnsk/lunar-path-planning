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
