import json
import tempfile
import unittest
from pathlib import Path


class QuasiRealGuardedPpoIterativeMiniLoopEvidenceFreezeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.temp_dir = Path(tempfile.mkdtemp(prefix="qreal-miniloop-freeze-"))
        self.miniloop_root = self.temp_dir / "miniloop"
        self.output_root = self.temp_dir / "freeze"
        self.batch_root = self.temp_dir / "batch"
        self.miniloop_root.mkdir(parents=True)
        self.batch_root.mkdir(parents=True)
        self._write_miniloop_artifacts()

    def test_freeze_passes_and_writes_manifest_readiness_report_and_baseline_counts(self) -> None:
        from scripts.run_quasi_real_guarded_ppo_iterative_miniloop_evidence_freeze import (
            run_quasi_real_guarded_ppo_iterative_miniloop_evidence_freeze,
        )

        result = run_quasi_real_guarded_ppo_iterative_miniloop_evidence_freeze(
            miniloop_root=self.miniloop_root,
            batch_root=self.batch_root,
            output_root=self.output_root,
            config=self._config(),
            repo_root=self.repo_root,
            readiness_runner=self._passing_readiness,
        )

        self.assertEqual(result["schema_version"], "quasi-real-guarded-ppo-iterative-miniloop-evidence-freeze-summary/v1")
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["reason_codes"], [])
        self.assertEqual(result["miniloop_status"], "passed")
        self.assertEqual(result["readiness_status"], "quasi_real_guarded_ppo_iterative_miniloop_stability_evaluated")
        self.assertEqual(result["input_trainable_transition_count"], 684)
        self.assertEqual(result["unique_trainable_context_count"], 684)
        self.assertEqual(result["seed_count"], 3)
        self.assertEqual(result["iteration_count"], 3)
        self.assertEqual(result["passed_iteration_count"], 9)
        self.assertEqual(result["progress_row_count"], 9)
        self.assertEqual(result["iteration_summary_row_count"], 9)
        self.assertEqual(result["controlled_regression_count"], 0)
        self.assertEqual(result["behavior_drift_count"], 0)
        self.assertFalse(result["runs_formal_ppo_rollout"])
        self.assertFalse(result["publishes_checkpoint"])
        self.assertFalse(result["replaces_default_policy"])
        self.assertFalse(result["performance_claimed"])
        self.assertFalse(result["formal_training_ready_claimed"])

        manifest = json.loads(
            (self.output_root / "quasi-real-guarded-ppo-iterative-miniloop-evidence-manifest.json")
            .read_text(encoding="utf-8")
        )
        self.assertEqual(manifest["schema_version"], "quasi-real-guarded-ppo-iterative-miniloop-evidence-manifest/v1")
        self.assertEqual(manifest["required_artifact_missing_count"], 0)
        names = {item["name"] for item in manifest["artifacts"]}
        self.assertIn("miniloop_summary", names)
        self.assertIn("miniloop_progress_jsonl", names)
        self.assertIn("readiness_validate_only", names)
        self.assertIn("iterative_miniloop_test", names)
        self.assertIn("architecture_report_doc", names)
        self.assertTrue(all(item["sha256"] for item in manifest["artifacts"] if item["required"]))

        report = (
            self.output_root
            / "quasi-real-guarded-ppo-iterative-miniloop-evidence-freeze-report.md"
        ).read_text(encoding="utf-8")
        self.assertIn("Quasi-Real Guarded PPO Iterative Mini-Loop Evidence Freeze", report)
        self.assertIn("684", report)
        self.assertIn("quasi_real_guarded_ppo_iterative_miniloop_stability_evaluated", report)

    def test_freeze_fails_when_progress_jsonl_does_not_match_passed_iterations(self) -> None:
        from scripts.run_quasi_real_guarded_ppo_iterative_miniloop_evidence_freeze import (
            run_quasi_real_guarded_ppo_iterative_miniloop_evidence_freeze,
        )

        progress_path = self.miniloop_root / "iterative-miniloop-progress.jsonl"
        rows = progress_path.read_text(encoding="utf-8").splitlines()
        progress_path.write_text("\n".join(rows[:-1]) + "\n", encoding="utf-8")

        result = run_quasi_real_guarded_ppo_iterative_miniloop_evidence_freeze(
            miniloop_root=self.miniloop_root,
            batch_root=self.batch_root,
            output_root=self.output_root,
            config=self._config(),
            repo_root=self.repo_root,
            readiness_runner=self._passing_readiness,
        )

        self.assertEqual(result["status"], "failed")
        self.assertIn("miniloop_progress_row_count_mismatch", result["reason_codes"])

    def test_config_declares_current_miniloop_baseline_and_non_goals(self) -> None:
        config_path = (
            self.repo_root
            / "configs"
            / "quasi_real_guarded_ppo_iterative_miniloop_evidence_freeze_v1.json"
        )
        config = json.loads(config_path.read_text(encoding="utf-8"))

        self.assertEqual(
            config["schema_version"],
            "quasi-real-guarded-ppo-iterative-miniloop-evidence-freeze-config/v1",
        )
        self.assertEqual(config["validation"]["expected_trainable_transition_count"], 684)
        self.assertIn("iterative-miniloop-progress.jsonl", config["required_artifacts"])
        self.assertIn("does_not_start_formal_ppo_rollout", config["non_goals"])
        self.assertIn("docs/算法设计与系统架构报告.md", config["tracked_source_files"])

    def _config(self) -> dict:
        return {
            "schema_version": "quasi-real-guarded-ppo-iterative-miniloop-evidence-freeze-config/v1",
            "readiness": {
                "config": "configs/policy_training_readiness_review_v1.json",
                "expected_status": "quasi_real_guarded_ppo_iterative_miniloop_stability_evaluated",
            },
            "validation": {
                "expected_trainable_transition_count": 684,
                "expected_unique_trainable_context_count": 684,
                "expected_seed_count": 3,
                "expected_iteration_count": 3,
                "expected_passed_iteration_count": 9,
            },
            "output_files": {
                "summary": "quasi-real-guarded-ppo-iterative-miniloop-evidence-freeze-summary.json",
                "manifest": "quasi-real-guarded-ppo-iterative-miniloop-evidence-manifest.json",
                "readiness_validate_only": "quasi-real-guarded-ppo-iterative-miniloop-readiness-validate-only.json",
                "report": "quasi-real-guarded-ppo-iterative-miniloop-evidence-freeze-report.md",
            },
            "tracked_source_files": [
                "README.md",
                "docs/算法设计与系统架构报告.md",
                "tests/test_quasi_real_guarded_ppo_iterative_miniloop_stability.py",
            ],
        }

    def _write_miniloop_artifacts(self) -> None:
        summary = {
            "schema_version": "quasi-real-guarded-ppo-iterative-miniloop-stability-summary/v1",
            "status": "passed",
            "reason_codes": [],
            "seed_count": 3,
            "iteration_count": 3,
            "passed_iteration_count": 9,
            "failed_iteration_count": 0,
            "input_trainable_transition_count": 684,
            "unique_trainable_context_count": 684,
            "ppo_trainable_transition_count": 684,
            "validation_trainable_count": 0,
            "test_trainable_count": 0,
            "source_fallback_trainable_count": 0,
            "teacher_fallback_trainable_count": 0,
            "non_empty_gate_reason_trainable_count": 0,
            "missing_observation_count": 0,
            "missing_log_prob_count": 0,
            "missing_value_count": 0,
            "non_finite_reward_count": 0,
            "non_finite_return_count": 0,
            "non_finite_advantage_count": 0,
            "loss_non_finite_count": 0,
            "non_finite_gradient_count": 0,
            "max_old_log_prob_abs_error": 0.0,
            "max_old_value_abs_error": 0.0,
            "max_abs_approx_kl": 1.0e-5,
            "max_grad_norm_after_clip": 1.0,
            "controlled_regression_count": 0,
            "controlled_safety_regression_count": 0,
            "controlled_contract_regression_count": 0,
            "controlled_path_risk_regression_count": 0,
            "controlled_source_selection_regression_count": 0,
            "behavior_drift_count": 0,
            "min_teacher_agreement_rate": 1.0,
            "readiness_status": "quasi_real_guarded_ppo_iterative_miniloop_stability_evaluated",
            "runs_formal_ppo_rollout": False,
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "performance_claimed": False,
            "formal_training_ready_claimed": False,
            "summary": str(self.miniloop_root / "quasi-real-guarded-ppo-iterative-miniloop-stability-summary.json"),
            "progress_jsonl": str(self.miniloop_root / "iterative-miniloop-progress.jsonl"),
            "iteration_summaries": str(self.miniloop_root / "iterative-miniloop-iteration-summaries.jsonl"),
            "readiness_validate_only": str(self.miniloop_root / "iterative-miniloop-readiness-validate-only.json"),
            "report": str(self.miniloop_root / "iterative-miniloop-stability-report.md"),
        }
        (self.miniloop_root / "quasi-real-guarded-ppo-iterative-miniloop-stability-summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        progress_rows = []
        iteration_rows = []
        for seed in (0, 1, 2):
            for iteration in (0, 1, 2):
                progress_rows.append(
                    {
                        "schema_version": "quasi-real-guarded-ppo-iterative-miniloop-progress/v1",
                        "seed": seed,
                        "iteration": iteration,
                        "status": "passed",
                        "optimizer_train_transition_count": 684,
                        "teacher_agreement_rate": 1.0,
                        "controlled_regression_count": 0,
                        "behavior_drift_count": 0,
                        "approx_kl": 1.0e-5,
                        "max_grad_norm_after_clip": 1.0,
                    }
                )
                iteration_rows.append(
                    {
                        "schema_version": "quasi-real-guarded-ppo-iterative-miniloop-iteration-summary/v1",
                        "seed": seed,
                        "iteration": iteration,
                        "status": "passed",
                        "reason_codes": [],
                    }
                )
        (self.miniloop_root / "iterative-miniloop-progress.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in progress_rows),
            encoding="utf-8",
        )
        (self.miniloop_root / "iterative-miniloop-iteration-summaries.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in iteration_rows),
            encoding="utf-8",
        )
        (self.miniloop_root / "iterative-miniloop-readiness-validate-only.json").write_text(
            json.dumps(self._passing_readiness(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        (self.miniloop_root / "iterative-miniloop-stability-report.md").write_text(
            "# Iterative mini-loop stability\n",
            encoding="utf-8",
        )

    def _passing_readiness(self, **_kwargs) -> dict:
        return {
            "training_readiness_status": "quasi_real_guarded_ppo_iterative_miniloop_stability_evaluated",
            "training_blockers": [],
            "reason_codes": [],
        }
