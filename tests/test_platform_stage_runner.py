import json
import subprocess
import sys
from pathlib import Path


SUPPORTED_STAGE_IDS = {
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
    "xunce-coverage-discriminability-audit",
    "xunce-candidate-level-coverage-opportunity-materialization",
    "xunce-cost-efficient-coverage-opportunity-refinement",
    "xunce-oracle-separability-benchmark",
    "xunce-true-incumbent-selection-binding",
    "xunce-safe-efficient-opportunity-root-cause-audit",
    "xunce-safe-efficient-candidate-repair",
    "xunce-risk-coverage-cost-quantization-audit",
    "xunce-risk-constrained-frontier-nbv-candidate-generation",
    "xunce-risk-aware-frontier-nbv-candidate-repair",
    "xunce-true-frontier-nbv-candidate-source-replacement",
    "xunce-stage18-5-evidence-attribution-review",
    "xunce-stage18-6-coverage-reward-cost-risk-guard-refinement",
    "xunce-stage18-7-candidate-count-scaling-audit",
    "xunce-stage18-9-trajectory-risk-boundary-reward-audit",
    "xunce-stage18-9-strict-v3-evidence-rollup",
    "xunce-stage18-11-path-cost-weight-calibration",
    "xunce-stage19-evaluator-critic-preflight",
    "xunce-stage20-reward-rerank-oracle-imitation-dataset",
    "xunce-stage20-1-same-candidate-oracle-imitation-evidence",
    "xunce-stage21-0-pure-ppo-readiness-audit",
    "xunce-stage21-1-on-policy-ppo-rollout-collector",
    "xunce-stage21-2-coverage-first-ppo-reward-contract",
    "xunce-stage21-3-ppo-batch-validation",
    "xunce-stage21-4-tiny-ppo-update-smoke",
    "xunce-stage21-5-post-update-offline-trajectory-evaluation",
    "xunce-stage21-6-multi-seed-ppo-pilot",
    "xunce-stage21-7-reward-collector-advantage-horizon-repair",
    "xunce-stage21-8-ppo-update-strength-calibration",
    "xunce-stage21-9-gradient-normalization-loss-scaling-repair",
    "xunce-stage21-10-stage21-9-repaired-multi-seed-ppo-pilot",
    "xunce-stage21-11-coverage-constrained-path-cost-objective-audit",
    "xunce-stage21-12-coverage-constrained-reward-profile-repair",
    "xunce-stage21-13-coverage-constrained-multi-seed-ppo-smoke",
    "xunce-stage21-14-multi-epoch-ppo-update-depth-calibration",
    "xunce-stage21-15-policy-update-signal-strength-calibration",
    "xunce-stage21-16-policy-signal-margin-credit-attribution",
    "xunce-stage21-17-policy-signal-amplification-value-balance",
    "xunce-stage21-18-iterative-ppo-learning-curve-audit",
    "xunce-stage21-19-policy-update-signal-source-repair",
    "xunce-stage22-0-theta-aware-sensor-action-space-contract",
    "xunce-stage22-1-theta-aware-candidate-viewpoint-generation",
    "xunce-stage22-2-theta-aware-coverage-reward-contract",
    "xunce-stage22-3-theta-aware-ppo-collector-smoke",
    "xunce-stage22-4-theta-aware-ppo-update-smoke",
    "xunce-stage23-2b-platform-geometry-sensor-contract-alignment",
    "xunce-stage23-5a-repair-required-inputs-high-res-roi-coverage",
    "xunce-stage23-6-slope-theta-policy-update-signal-strength-repair",
    "xunce-stage24-0-hybrid-astar-pose-path-planner-full-foundation",
    "xunce-stage24-1-hybrid-astar-candidate-path-cost-integration",
    "xunce-stage24-2-hybrid-astar-reward-path-cost-contract",
    "xunce-stage24-3-hybrid-astar-path-cost-ppo-collector-smoke",
    "xunce-stage24-4-hybrid-astar-path-cost-ppo-update-smoke",
    "xunce-stage24-5-hybrid-astar-path-cost-post-update-trajectory-eval-smoke",
    "xunce-stage24-5a-repair-hybrid-path-inference-binding",
    "xunce-stage25-0-continuous-theta-hybrid-action-space-foundation",
    "xunce-stage26-0-synthetic-rock-pit-terrain-augmentation-contract",
    "xunce-stage26-1-synthetic-terrain-collector-smoke",
    "xunce-stage26-2-synthetic-terrain-ppo-update-smoke",
    "xunce-stage26-3-synthetic-terrain-post-update-trajectory-eval-smoke",
    "xunce-stage26-4-synthetic-policy-update-signal-strength-repair",
    "xunce-stage26-4a-parallelize-stage21-1-hybrid-astar-candidate-costs",
    "xunce-stage26-5-synthetic-discrete-margin-crossing-calibration",
    "xunce-stage26-5b-documentation-boundary-consolidation",
    "xunce-stage26-6-synthetic-exploration-credit-assignment",
    "xunce-stage26-7-synthetic-credit-assignment-path-efficiency-repair",
    "xunce-stage26-7b-coverable-cell-semantics-contract",
    "xunce-stage26-7c-main-coverable-coverage-efficiency-rerun",
    "xunce-stage26-7d-repair-synthetic-credit-sampler-continuous-theta-reachability",
    "xunce-stage26-7f-synthetic-credit-ppo-update-stability-sweep",
    "xunce-stage26-7g-repair-behavior-policy-kl-baseline",
    "xunce-stage26-7h-repair-credit-post-update-eval-binding",
    "xunce-stage26-8-synthetic-terrain-multi-seed-coverage-efficiency-pilot",
    "xunce-stage26-8a-expand-seed-or-horizon-budget",
    "xunce-stage26-8b-repair-horizon-collector-terminal-reachability",
    "xunce-stage26-8c-resume-h16-h20-horizon-efficiency",
    "xunce-stage26-8d-resumable-seed-horizon-execution",
    "xunce-stage26-8f-scenario-diversity-and-policy-margin-audit",
    "xunce-stage26-8g-repair-synthetic-scenario-diversity",
    "xunce-stage26-8h-resumable-diverse-scenario-post-update-eval",
    "xunce-stage26-8i-diverse-scenario-policy-signal-strength-repair",
    "xunce-stage26-8m-generalized-resumable-training-pipeline",
    "xunce-stage26-8n-aggressive-sample-update-sweep",
    "xunce-stage26-8o-repair-aggressive-collector-trainable-sample-budget",
    "xunce-stage26-8p-hybrid-astar-primitive-resolution-sweep",
    "xunce-stage26-8q-derived-high-res-planning-proxy-alignment",
    "xunce-stage26-8r-pose-gate-repair",
    "xunce-stage26-8s-terminal-aware-sample-expansion",
    "xunce-stage26-9-synthetic-terrain-long-horizon-efficiency-pilot",
    "xunce-stage26-10-terminal-aware-reward-shaping",
    "xunce-stage26-10a-terminal-aware-reward-weight-sweep",
    "xunce-stage26-10b-completion-capable-shadow-trial",
    "xunce-stage26-io1-artifact-path-contract-and-long-path-resilience",
    "xunce-stage26-io2-remaining-runner-long-path-migration",
    "xunce-stage18i-evidence-closure-audit",
    "xunce-stage18-research-evidence-pipeline",
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


def test_stage_runner_extra_arg_accepts_option_shaped_values() -> None:
    repo_root = Path(__file__).resolve().parents[1]

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_stage.py",
            "--stage",
            "xunce-high-fidelity-exploration-coverage-comparison",
            "--extra-arg",
            "--candidate-refresh-mode",
            "--extra-arg",
            "dynamic_from_coverage_memory",
            "--extra-arg",
            "--include-oracle-baselines",
            "--dry-run",
        ],
        cwd=repo_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "--candidate-refresh-mode" in completed.stdout
    assert "dynamic_from_coverage_memory" in completed.stdout
    assert "--include-oracle-baselines" in completed.stdout


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
