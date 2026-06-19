import json
import os
import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_model_explorer_path_feedback_cli_runs_tiny_contract_and_sidecar(tmp_path) -> None:
    repo_root = _repo_root()
    contract_path = tmp_path / "tiny.contract.json"
    sidecar_path = tmp_path / "tiny.sidecar.json"
    summary_path = tmp_path / "summary.json"
    report_path = tmp_path / "report.md"
    manifest_path = tmp_path / "manifest.json"
    contract_path.write_text(json.dumps(_tiny_contract(), ensure_ascii=False, indent=2), encoding="utf-8")
    sidecar_path.write_text(json.dumps(_tiny_sidecar(), ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = {
        "schema_version": "path-feedback-manifest/v1",
        "scenario_set": "tiny",
        "diagnostic_profile": "stage18i4_cli_smoke",
        "top_k": 2,
        "planner": {
            "backend": "path_planner_route",
            "python_executable": sys.executable,
            "path_planner_root": str(repo_root / "path-planner"),
        },
        "outputs": {"summary": str(summary_path), "report": str(report_path)},
        "scenarios": [
            {
                "scenario_id": "tiny",
                "scenario_group": "smoke",
                "contract": str(contract_path),
                "sidecar": str(sidecar_path),
                "current_cell": [0, 0],
            }
        ],
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    env = dict(os.environ)
    model_src = str(repo_root / "model-explorer" / "src")
    env["PYTHONPATH"] = model_src if not env.get("PYTHONPATH") else model_src + os.pathsep + env["PYTHONPATH"]

    validate = subprocess.run(
        [sys.executable, "-m", "model_explorer", "path-feedback", "validate", str(manifest_path)],
        cwd=repo_root,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert validate.returncode == 0, validate.stderr or validate.stdout

    run = subprocess.run(
        [sys.executable, "-m", "model_explorer", "path-feedback", "run", str(manifest_path)],
        cwd=repo_root,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert run.returncode == 0, run.stderr or run.stdout
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["schema_version"] == "path-feedback-summary/v1"
    assert summary["scenario_count"] == 1
    assert summary["candidate_count"] == 2
    assert summary["open_grid_fallback_used"] is False


def _tiny_contract() -> dict:
    return {
        "schema_version": "model-explorer-contract/v1",
        "grid": {
            "width": 4,
            "height": 4,
            "resolution": 1.0,
            "frame_id": "test",
            "origin": [0.0, 0.0],
            "layers": ["cost"],
        },
        "constraints": {"violation_count": 0, "passable_ratio": 1.0, "reason_counts": {}},
        "top_goals": [
            {"cell": [1, 0], "utility": 1.0, "reachable": True, "risk": 0.1},
            {"cell": [2, 0], "utility": 0.9, "reachable": True, "risk": 0.2},
        ],
        "top_sequences": [],
        "observation_update": {"coverage_rate_delta": 0.0},
    }


def _tiny_sidecar() -> dict:
    return {
        "schema_version": "path-planner-sidecar/v1",
        "cost": [[1.0, 1.0, 1.0, 1.0] for _ in range(4)],
        "passable_mask": [[True, True, True, True] for _ in range(4)],
        "metadata": {"fixture": "stage18i4-cli-smoke"},
    }
