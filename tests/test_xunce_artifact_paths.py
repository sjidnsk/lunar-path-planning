import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = str(REPO_ROOT / "scripts")
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)


def test_mid_dual_artifact_names_are_canonical_and_have_no_legacy_aliases() -> None:
    from xunce_artifact_paths import (
        MID_DUAL_CONFIG,
        MID_DUAL_MANIFEST,
        MID_DUAL_PHASE_STATE,
        MID_DUAL_REPORT,
        MID_DUAL_RESULTS,
        MID_DUAL_ROUTING,
        MID_DUAL_SUMMARY,
    )

    artifacts = (
        MID_DUAL_CONFIG,
        MID_DUAL_RESULTS,
        MID_DUAL_SUMMARY,
        MID_DUAL_ROUTING,
        MID_DUAL_MANIFEST,
        MID_DUAL_PHASE_STATE,
        MID_DUAL_REPORT,
    )
    assert {artifact.canonical for artifact in artifacts} == {
        "config.json", "results.jsonl", "summary.json", "routing.json", "manifest.json", "phase-state.jsonl", "report.md"
    }
    assert all(artifact.legacy == () for artifact in artifacts)


def test_mid_dual_artifact_io_uses_canonical_paths(tmp_path: Path) -> None:
    import xunce_artifact_io as artifact_io
    from xunce_artifact_paths import MID_DUAL_SUMMARY, artifact_path, read_json_artifact, write_json_artifact

    write_json_artifact(tmp_path, MID_DUAL_SUMMARY, {"status": "passed"})
    payload, source = read_json_artifact(tmp_path, MID_DUAL_SUMMARY)

    assert artifact_io.path_is_file(artifact_path(tmp_path, MID_DUAL_SUMMARY))
    assert source == "canonical"
    assert payload == {"status": "passed", "artifact_source": "canonical"}
