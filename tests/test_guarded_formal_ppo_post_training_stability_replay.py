import json
import shutil
import tempfile
import unittest
from pathlib import Path


class GuardedFormalPpoPostTrainingStabilityReplayTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.temp_dir = Path(tempfile.mkdtemp(prefix="guarded-formal-post-training-replay-"))
        self.training_root = self.temp_dir / "training-run"
        self.output_root = self.temp_dir / "post-training-replay"
        self.batch_root = self.temp_dir / "batch"
        for path in (self.training_root, self.output_root, self.batch_root):
            path.mkdir(parents=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def test_replays_all_formal_training_seed_candidates(self) -> None:
        from scripts.run_guarded_formal_ppo_post_training_stability_replay import (
            run_guarded_formal_ppo_post_training_stability_replay,
        )

        self._write_training_run_artifacts()
        calls: list[tuple[int, int]] = []

        def replay_runner(**kwargs: object) -> dict:
            seed = int(kwargs["seed"])
            replay_index = int(kwargs["replay_index"])
            calls.append((seed, replay_index))
            return self._replay_summary(seed=seed, replay_index=replay_index)

        result = run_guarded_formal_ppo_post_training_stability_replay(
            training_run_root=self.training_root,
            output_root=self.output_root,
            batch_root=self.batch_root,
            config=self._config(),
            repo_root=self.repo_root,
            replay_runner=replay_runner,
            readiness_runner=self._passing_readiness,
        )

        self.assertEqual(result["schema_version"], "guarded-formal-ppo-post-training-stability-replay-summary/v1")
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["reason_codes"], [])
        self.assertEqual(result["input_formal_training_run_status"], "passed")
        self.assertEqual(result["seed_count"], 5)
        self.assertEqual(result["replay_count_per_seed"], 3)
        self.assertEqual(result["total_replay_count"], 15)
        self.assertEqual(result["passed_replay_count"], 15)
        self.assertEqual(result["optimizer_train_transition_count"], 684)
        self.assertEqual(result["replay_collector_trainable_transition_count"], 684)
        self.assertEqual(result["missing_seed_candidate_checkpoint_count"], 0)
        self.assertEqual(result["replay_behavior_drift_count"], 0)
        self.assertEqual(result["validation_trainable_count"], 0)
        self.assertEqual(result["test_trainable_count"], 0)
        self.assertEqual(result["fallback_trainable_count"], 0)
        self.assertEqual(result["diagnostic_trainable_count"], 0)
        self.assertEqual(result["missing_observation_count"], 0)
        self.assertEqual(result["missing_log_prob_count"], 0)
        self.assertEqual(result["missing_value_count"], 0)
        self.assertEqual(result["non_finite_reward_count"], 0)
        self.assertEqual(result["non_finite_return_count"], 0)
        self.assertEqual(result["non_finite_advantage_count"], 0)
        self.assertEqual(result["teacher_agreement_rate"], 1.0)
        self.assertEqual(result["controlled_regression_count"], 0)
        self.assertEqual(result["post_training_holdout_status"], "passed")
        self.assertEqual(result["post_training_canary_status"], "passed")
        self.assertFalse(result["runs_new_ppo_update"])
        self.assertFalse(result["publishes_checkpoint"])
        self.assertFalse(result["replaces_default_policy"])
        self.assertFalse(result["performance_claimed"])
        self.assertFalse(result["formal_training_ready_claimed"])
        self.assertEqual(
            result["readiness_status"],
            "guarded_formal_ppo_post_training_stability_replay_evaluated",
        )
        self.assertEqual(
            calls,
            [(seed, replay_index) for seed in range(5) for replay_index in range(3)],
        )

        for filename in (
            "formal-ppo-post-training-stability-replay-summary.json",
            "formal-ppo-post-training-stability-replay-seed-summaries.jsonl",
            "formal-ppo-post-training-stability-replay-progress.jsonl",
            "formal-ppo-post-training-stability-replay-drift-report.jsonl",
            "formal-ppo-post-training-stability-replay-gate-audit.json",
            "formal-ppo-post-training-stability-replay-rollback-manifest.json",
            "formal-ppo-post-training-stability-replay-readiness-validate-only.json",
            "formal-ppo-post-training-stability-replay-report.md",
        ):
            self.assertTrue((self.output_root / filename).is_file(), filename)

    def test_fails_when_seed_checkpoint_is_missing(self) -> None:
        from scripts.run_guarded_formal_ppo_post_training_stability_replay import (
            run_guarded_formal_ppo_post_training_stability_replay,
        )

        self._write_training_run_artifacts()
        (
            self.training_root
            / "seed-03"
            / "limited_ppo_update_smoke"
            / "experimental-hybrid-policy-candidate.pt"
        ).unlink()

        result = run_guarded_formal_ppo_post_training_stability_replay(
            training_run_root=self.training_root,
            output_root=self.output_root,
            batch_root=self.batch_root,
            config=self._config(),
            repo_root=self.repo_root,
            replay_runner=lambda **kwargs: self._replay_summary(
                seed=int(kwargs["seed"]),
                replay_index=int(kwargs["replay_index"]),
            ),
            readiness_runner=self._passing_readiness,
        )

        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["missing_seed_candidate_checkpoint_count"], 1)
        self.assertIn("missing_seed_candidate_checkpoint", result["reason_codes"])

    def test_fails_when_replay_behavior_drifts_or_regresses(self) -> None:
        from scripts.run_guarded_formal_ppo_post_training_stability_replay import (
            run_guarded_formal_ppo_post_training_stability_replay,
        )

        self._write_training_run_artifacts()

        def replay_runner(**kwargs: object) -> dict:
            seed = int(kwargs["seed"])
            replay_index = int(kwargs["replay_index"])
            summary = self._replay_summary(seed=seed, replay_index=replay_index)
            if seed == 2 and replay_index == 1:
                summary["replay_collector_trainable_transition_count"] = 683
                summary["controlled_regression_count"] = 1
                summary["post_training_canary_status"] = "failed"
            return summary

        result = run_guarded_formal_ppo_post_training_stability_replay(
            training_run_root=self.training_root,
            output_root=self.output_root,
            batch_root=self.batch_root,
            config=self._config(),
            repo_root=self.repo_root,
            replay_runner=replay_runner,
            readiness_runner=self._passing_readiness,
        )

        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["replay_behavior_drift_count"], 1)
        self.assertIn("replay_behavior_drift_detected", result["reason_codes"])
        self.assertIn("post_training_replay_controlled_regression", result["reason_codes"])
        self.assertIn("post_training_replay_gate_failed", result["reason_codes"])

    def test_config_declares_outputs_docs_and_non_goals(self) -> None:
        config_path = self.repo_root / "configs" / "guarded_formal_ppo_post_training_stability_replay_v1.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))

        self.assertEqual(
            config["schema_version"],
            "guarded-formal-ppo-post-training-stability-replay-config/v1",
        )
        self.assertEqual(config["replay"]["replay_count_per_seed"], 3)
        self.assertIn(
            "formal-ppo-post-training-stability-replay-summary.json",
            config["output_files"].values(),
        )
        self.assertIn("README.md", config["documentation_updates"])
        self.assertIn("docs/算法设计与系统架构报告.md", config["documentation_updates"])
        self.assertIn("does_not_run_new_ppo_update", config["non_goals"])
        self.assertIn("does_not_publish_checkpoint", config["non_goals"])
        self.assertIn("does_not_replace_default_policy", config["non_goals"])

    def _write_training_run_artifacts(self) -> None:
        seed_summaries_path = self.training_root / "formal-ppo-training-run-seed-summaries.jsonl"
        seed_summaries = []
        for seed in range(5):
            seed_root = self.training_root / f"seed-{seed:02d}"
            checkpoint_root = seed_root / "limited_ppo_update_smoke"
            collector_root = seed_root / "collector"
            checkpoint_root.mkdir(parents=True)
            collector_root.mkdir(parents=True)
            (checkpoint_root / "experimental-hybrid-policy-candidate.pt").write_bytes(b"checkpoint")
            (collector_root / "ppo-rollout-collector-summary.json").write_text(
                json.dumps({"status": "passed", "reason_codes": [], "ppo_trainable_transition_count": 684}),
                encoding="utf-8",
            )
            seed_summaries.append(
                {
                    "schema_version": "guarded-formal-ppo-training-run-seed-summary/v1",
                    "status": "passed",
                    "reason_codes": [],
                    "seed": seed,
                    "optimizer_train_transition_count": 684,
                    "post_update_guarded_collector_trainable_transition_count": 684,
                    "limited_ppo_update_smoke_root": str(checkpoint_root),
                    "collector_root": str(collector_root),
                    "teacher_agreement_rate": 1.0,
                    "controlled_regression_count": 0,
                    "controlled_safety_regression_count": 0,
                    "controlled_contract_regression_count": 0,
                    "controlled_path_risk_regression_count": 0,
                    "controlled_source_selection_regression_count": 0,
                    "post_training_holdout_status": "passed",
                    "post_training_canary_status": "passed",
                    "publishes_checkpoint": False,
                    "replaces_default_policy": False,
                    "performance_claimed": False,
                    "formal_training_ready_claimed": False,
                }
            )
        seed_summaries_path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in seed_summaries),
            encoding="utf-8",
        )
        summary = {
            "schema_version": "guarded-formal-ppo-training-run-summary/v1",
            "status": "passed",
            "reason_codes": [],
            "seed_summaries": str(seed_summaries_path),
            "input_authorization_status": "passed",
            "authorization_verdict": "authorized_for_guarded_formal_ppo_training_run",
            "optimizer_train_transition_count": 684,
            "unique_trainable_context_count": 684,
            "seed_count": 5,
            "passed_seed_count": 5,
            "seeds": [0, 1, 2, 3, 4],
            "teacher_agreement_rate": 1.0,
            "controlled_regression_count": 0,
            "post_training_holdout_status": "passed",
            "post_training_canary_status": "passed",
            "runs_guarded_formal_ppo_training_run": True,
            "runs_new_ppo_update": True,
            "experimental_checkpoint": True,
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "performance_claimed": False,
            "formal_training_ready_claimed": False,
            "readiness_status": "guarded_formal_ppo_training_run_evaluated",
            "git_provenance": {"current": {"dirty": False}, "current_matches_sources": True},
        }
        (self.training_root / "formal-ppo-training-run-summary.json").write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )

    def _replay_summary(self, *, seed: int, replay_index: int) -> dict:
        return {
            "schema_version": "guarded-formal-ppo-post-training-stability-replay-row/v1",
            "status": "passed",
            "reason_codes": [],
            "seed": seed,
            "replay_index": replay_index,
            "optimizer_train_transition_count": 684,
            "replay_collector_trainable_transition_count": 684,
            "validation_trainable_count": 0,
            "test_trainable_count": 0,
            "fallback_trainable_count": 0,
            "diagnostic_trainable_count": 0,
            "missing_observation_count": 0,
            "missing_log_prob_count": 0,
            "missing_value_count": 0,
            "non_finite_reward_count": 0,
            "non_finite_return_count": 0,
            "non_finite_advantage_count": 0,
            "teacher_agreement_rate": 1.0,
            "controlled_regression_count": 0,
            "controlled_safety_regression_count": 0,
            "controlled_contract_regression_count": 0,
            "controlled_path_risk_regression_count": 0,
            "controlled_source_selection_regression_count": 0,
            "post_training_holdout_status": "passed",
            "post_training_canary_status": "passed",
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "performance_claimed": False,
            "formal_training_ready_claimed": False,
        }

    def _config(self) -> dict:
        return {
            "schema_version": "guarded-formal-ppo-post-training-stability-replay-config/v1",
            "replay": {"replay_count_per_seed": 3},
            "validation": {
                "expected_seed_count": 5,
                "expected_optimizer_train_transition_count": 684,
                "min_replay_collector_trainable_transition_count": 684,
                "min_teacher_agreement_rate": 0.95,
            },
            "readiness": {
                "config": "configs/policy_training_readiness_review_v1.json",
                "expected_status": "guarded_formal_ppo_post_training_stability_replay_evaluated",
            },
            "output_files": {
                "summary": "formal-ppo-post-training-stability-replay-summary.json",
                "seed_summaries": "formal-ppo-post-training-stability-replay-seed-summaries.jsonl",
                "progress": "formal-ppo-post-training-stability-replay-progress.jsonl",
                "drift_report": "formal-ppo-post-training-stability-replay-drift-report.jsonl",
                "gate_audit": "formal-ppo-post-training-stability-replay-gate-audit.json",
                "rollback_manifest": "formal-ppo-post-training-stability-replay-rollback-manifest.json",
                "readiness_validate_only": "formal-ppo-post-training-stability-replay-readiness-validate-only.json",
                "report": "formal-ppo-post-training-stability-replay-report.md",
            },
        }

    def _passing_readiness(self, **_: object) -> dict:
        return {
            "training_readiness_status": "guarded_formal_ppo_post_training_stability_replay_evaluated",
            "training_blockers": [],
            "reason_codes": [],
            "recommended_next_action": "guarded_formal_ppo_post_training_stability_replay_evaluated",
            "returncode": 0,
        }
