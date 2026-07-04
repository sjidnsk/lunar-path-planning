import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = str(REPO_ROOT / "scripts")
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)


def test_stage26_io1_passes_with_short_roots(tmp_path: Path) -> None:
    from scripts import run_xunce_stage26_io1_artifact_path_contract as runner

    config = _write_config(tmp_path)
    summary = runner.run_xunce_stage26_io1_artifact_path_contract(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "passed"
    assert summary["next_required_change"] == "rerun_stage26_8n_aggressive_update_sweep_with_aligned_planning_proxy"
    assert summary["long_path_io_passed"] is True
    assert summary["alias_contract_passed"] is True
    assert summary["stage21_artifact_compatibility_passed"] is True
    assert summary["short_output_root_contract_passed"] is True
    assert (tmp_path / "out" / "summary.json").is_file()
    assert (tmp_path / "out" / "path_audit.json").is_file()
    assert (tmp_path / "out" / "alias_audit.json").is_file()


def test_stage26_io1_boundary_route(tmp_path: Path) -> None:
    from scripts import run_xunce_stage26_io1_artifact_path_contract as runner

    summary = runner.run_xunce_stage26_io1_artifact_path_contract(
        config_path=_write_config(tmp_path, publishes_checkpoint=True),
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "resolve_stage26_io1_boundary_rejections"
    assert "publishes_checkpoint" in summary["boundary_rejections"]


def test_stage26_io1_short_root_budget_failure_routes_to_repair(tmp_path: Path) -> None:
    from scripts import run_xunce_stage26_io1_artifact_path_contract as runner

    long_root = "D:/xunce/out/" + ("very_long_segment_" * 20)
    summary = runner.run_xunce_stage26_io1_artifact_path_contract(
        config_path=_write_config(tmp_path, short_output_roots=[long_root]),
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "repair_stage26_short_output_root_contract"


def test_stage26_io1_registry_dry_run_resolves() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_stage.py",
            "--stage",
            "xunce-stage26-io1-artifact-path-contract-and-long-path-resilience",
            "--dry-run",
        ],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "run_xunce_stage26_io1_artifact_path_contract.py" in completed.stdout
    assert "D:\\xunce\\out\\s26_io1" in completed.stdout or "D:/xunce/out/s26_io1" in completed.stdout


def _write_config(tmp_path: Path, **overrides) -> Path:
    payload = {
        "schema_version": "xunce-stage26-io1-artifact-path-contract-config/v1",
        "stage_id": "xunce-stage26-io1-artifact-path-contract-and-long-path-resilience",
        "short_output_roots": ["D:/xunce/out/s26_io1", "D:/xunce/out/s26_8m", "D:/xunce/out/s26_8n", "D:/xunce/out/s26_8q"],
        "warn_path_length": 180,
        "fail_path_length": 240,
        "stage26_8q_legacy_root": "",
        "release_or_training_authorized": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
    }
    payload.update(overrides)
    path = tmp_path / "stage26_io1_config.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path
