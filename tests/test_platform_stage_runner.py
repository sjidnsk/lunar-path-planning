import json
import subprocess
import sys
from pathlib import Path


SUPPORTED_STAGE_IDS = {
    "path-feedback-validation",
    "path-feedback-batch-validation",
    "policy-training-readiness-review",
    "policy-gated-sequential-canary-rollout",
    "guarded-ppo-rollout-pilot",
    "iterative-ppo-mini-loop-stability",
    "quasi-real-guarded-ppo-stability-replay",
    "xunce-model-comparison-input-readiness",
    "xunce-design-freeze-audit",
    "xunce-current-head-evidence-refresh",
    "xunce-network-literature-bottleneck-review",
    "xunce-topology-observation-contract",
    "xunce-topology-feature-extraction-audit",
    "xunce-topology-graph-proto",
    "xunce-proto-mechanism-validation",
    "xunce-architecture-contrast-evaluation",
    "xunce-full-network-v1",
    "xunce-full-network-static-contract-validation",
    "xunce-full-network-ablation-experiments",
    "xunce-full-network-stress-evaluation",
    "xunce-guarded-training-candidate-preflight",
    "xunce-controlled-training-candidate",
    "xunce-post-training-offline-evaluation",
    "xunce-high-fidelity-exploration-coverage-comparison",
}


def test_stage_runner_lists_xunce_stages() -> None:
    repo_root = Path(__file__).resolve().parents[1]

    completed = subprocess.run(
        [sys.executable, "scripts/run_stage.py", "--list"],
        cwd=repo_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "xunce-shadow-replay-validation" in completed.stdout
    assert "xunce-high-fidelity-real-map-comparison" in completed.stdout
    for stage_id in SUPPORTED_STAGE_IDS:
        assert stage_id in completed.stdout


def test_stage_runner_dry_run_uses_current_python_without_bash() -> None:
    repo_root = Path(__file__).resolve().parents[1]

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_stage.py",
            "--stage",
            "xunce-shadow-replay-validation",
            "--dry-run",
        ],
        cwd=repo_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert sys.executable in completed.stdout
    assert "scripts/run_xunce_shadow_replay_validation.py" in completed.stdout.replace("\\", "/")
    assert "bash" not in completed.stdout.lower()
    assert "python3" not in completed.stdout.lower()
    assert "/home/kai" not in completed.stdout


def test_stage_runner_supported_stages_dry_run_without_shell_wrappers() -> None:
    repo_root = Path(__file__).resolve().parents[1]

    for stage_id in sorted(SUPPORTED_STAGE_IDS):
        completed = subprocess.run(
            [
                sys.executable,
                "scripts/run_stage.py",
                "--stage",
                stage_id,
                "--dry-run",
            ],
            cwd=repo_root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        assert completed.returncode == 0, completed.stdout + completed.stderr
        normalized = completed.stdout.replace("\\", "/")
        assert sys.executable in completed.stdout
        assert "scripts/run_" in normalized
        assert ".py" in normalized
        assert ".sh" not in normalized
        assert "bash" not in completed.stdout.lower()
        assert "python3" not in completed.stdout.lower()
        assert "/home/kai" not in completed.stdout


def test_stage_runner_rejects_bad_registry_schema(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    registry = tmp_path / "stage_registry.json"
    registry.write_text(
        json.dumps({"schema_version": "bad/v1", "stages": {}}),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_stage.py",
            "--registry",
            str(registry),
            "--list",
        ],
        cwd=repo_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert completed.returncode == 2
    assert "lunar-stage-registry/v1" in completed.stderr


def test_stage_runner_reports_unknown_stage() -> None:
    repo_root = Path(__file__).resolve().parents[1]

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_stage.py",
            "--stage",
            "missing-stage",
            "--dry-run",
        ],
        cwd=repo_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert completed.returncode == 2
    assert "missing-stage" in completed.stderr
    assert "xunce-shadow-replay-validation" in completed.stderr
