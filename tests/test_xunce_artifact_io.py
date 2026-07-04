import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = str(REPO_ROOT / "scripts")
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)


def test_long_path_json_and_jsonl_roundtrip(tmp_path: Path) -> None:
    import xunce_artifact_io as artifact_io

    long_path = tmp_path
    for index in range(6):
        long_path = long_path / f"segment_{index}_" / ("x" * 32)
    json_path = long_path / ("artifact_" + "y" * 80 + ".json")
    jsonl_path = long_path / ("rows_" + "z" * 80 + ".jsonl")

    artifact_io.write_json(json_path, {"schema_version": "test/v1", "value": 3})
    artifact_io.write_jsonl(jsonl_path, [{"row": 1}, {"row": 2}])

    assert artifact_io.path_length(json_path) > 260
    assert artifact_io.path_is_file(json_path)
    assert artifact_io.read_json(json_path)["value"] == 3
    assert artifact_io.read_jsonl(jsonl_path) == [{"row": 1}, {"row": 2}]
    assert artifact_io.count_jsonl_rows(jsonl_path) == 2
    assert artifact_io.path_exists(json_path)

    copied = long_path / "copied.json"
    artifact_io.copy_file(json_path, copied)
    assert artifact_io.read_bytes(copied) == artifact_io.read_bytes(json_path)


def test_path_length_audit_marks_warn_and_fail(tmp_path: Path) -> None:
    import xunce_artifact_io as artifact_io

    short = tmp_path / "short.json"
    long = tmp_path / ("l" * 120) / ("m" * 120) / "file.json"

    audit = artifact_io.path_length_audit([short, long], warn_at=40, fail_at=80)

    assert audit["path_length_warning_count"] >= 0
    assert audit["path_length_failure_count"] >= 1
    assert audit["max_path_length"] >= artifact_io.path_length(long)
