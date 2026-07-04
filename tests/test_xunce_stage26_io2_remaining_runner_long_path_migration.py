import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = str(REPO_ROOT / "scripts")
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)


def test_stage26_io2_passes_with_migrated_runners(tmp_path: Path) -> None:
    from scripts import run_xunce_stage26_io2_remaining_runner_long_path_migration as runner

    summary = runner.run_xunce_stage26_io2_remaining_runner_long_path_migration(
        config_path=_write_config(tmp_path),
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "passed"
    assert summary["next_required_change"] == "rerun_stage26_8n_aggressive_update_sweep_with_aligned_planning_proxy"
    assert summary["alias_contract_passed"] is True
    assert summary["runner_static_io_audit_passed"] is True
    assert summary["short_output_root_contract_passed"] is True
    assert (tmp_path / "out" / "summary.json").is_file()
    assert (tmp_path / "out" / "runner_static_io_audit.json").is_file()


def test_stage26_io2_boundary_route(tmp_path: Path) -> None:
    from scripts import run_xunce_stage26_io2_remaining_runner_long_path_migration as runner

    summary = runner.run_xunce_stage26_io2_remaining_runner_long_path_migration(
        config_path=_write_config(tmp_path, publishes_checkpoint=True),
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "resolve_stage26_io2_boundary_rejections"
    assert "publishes_checkpoint" in summary["boundary_rejections"]


def test_stage26_io2_static_audit_catches_direct_artifact_io(tmp_path: Path) -> None:
    from scripts import run_xunce_stage26_io2_remaining_runner_long_path_migration as runner

    bad_runner = tmp_path / "bad_runner.py"
    bad_runner.write_text("from pathlib import Path\npayload = Path('artifact.json').read_text()\n", encoding="utf-8")
    summary = runner.run_xunce_stage26_io2_remaining_runner_long_path_migration(
        config_path=_write_config(tmp_path, migrated_runner_files=[str(bad_runner)]),
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "repair_stage26_io2_runner_artifact_io_migration"
    assert summary["direct_artifact_io_violation_count"] == 1


def test_stage26_io2_registry_roots_stay_short(tmp_path: Path) -> None:
    from scripts import run_xunce_stage26_io2_remaining_runner_long_path_migration as runner

    summary = runner.run_xunce_stage26_io2_remaining_runner_long_path_migration(
        config_path=_write_config(tmp_path),
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )
    path_audit = json.loads((tmp_path / "out" / "path_audit.json").read_text(encoding="utf-8"))

    assert summary["short_output_root_contract_passed"] is True
    checked = {entry["stage_id"]: entry["path"].replace("\\", "/") for entry in path_audit["entries"]}
    assert checked["xunce-stage26-8o-repair-aggressive-collector-trainable-sample-budget"] == "D:/xunce/out/s26_8o"
    assert checked["xunce-stage26-8p-hybrid-astar-primitive-resolution-sweep"] == "D:/xunce/out/s26_8p"
    assert checked["xunce-stage26-io2-remaining-runner-long-path-migration"] == "D:/xunce/out/s26_io2"


def test_stage26_io2_registry_dry_run_resolves() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_stage.py",
            "--stage",
            "xunce-stage26-io2-remaining-runner-long-path-migration",
            "--dry-run",
        ],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "run_xunce_stage26_io2_remaining_runner_long_path_migration.py" in completed.stdout
    assert "D:\\xunce\\out\\s26_io2" in completed.stdout or "D:/xunce/out/s26_io2" in completed.stdout


def _write_config(tmp_path: Path, **overrides) -> Path:
    payload = {
        "schema_version": "xunce-stage26-io2-remaining-runner-long-path-migration-config/v1",
        "stage_id": "xunce-stage26-io2-remaining-runner-long-path-migration",
        "stage_registry_path": "configs/stage_registry.json",
        "short_root_stage_ids": [
            "xunce-stage26-8m-generalized-resumable-training-pipeline",
            "xunce-stage26-8n-aggressive-sample-update-sweep",
            "xunce-stage26-8o-repair-aggressive-collector-trainable-sample-budget",
            "xunce-stage26-8p-hybrid-astar-primitive-resolution-sweep",
            "xunce-stage26-8q-derived-high-res-planning-proxy-alignment",
            "xunce-stage26-io1-artifact-path-contract-and-long-path-resilience",
            "xunce-stage26-io2-remaining-runner-long-path-migration",
        ],
        "warn_path_length": 180,
        "fail_path_length": 240,
        "release_or_training_authorized": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
    }
    payload.update(overrides)
    path = tmp_path / "stage26_io2_config.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path
