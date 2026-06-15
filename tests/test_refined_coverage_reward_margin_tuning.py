import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path


class RefinedCoverageRewardMarginTuningTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        for path in (
            self.repo_root / "scripts",
            self.repo_root / "model-explorer" / "src",
            self.repo_root / "tests",
        ):
            value = str(path)
            if value not in sys.path:
                sys.path.insert(0, value)
        from test_refined_coverage_driven_ppo_improvement_run import (
            RefinedCoverageDrivenPpoImprovementRunTests,
        )

        self.fixture = RefinedCoverageDrivenPpoImprovementRunTests(methodName="run")
        self.fixture.setUp()
        self.temp_dir = Path(tempfile.mkdtemp(prefix="refined-tuning-"))
        self.refined_root = self.temp_dir / "refined-v2"
        self.output_root = self.temp_dir / "tuning-output"

    def tearDown(self) -> None:
        self.fixture.tearDown()
        shutil.rmtree(self.temp_dir)

    def test_tuning_configs_materialize_explicit_ppo_advantage_and_return_fields(self) -> None:
        from scripts.run_refined_coverage_driven_ppo_improvement_run import (
            run_refined_coverage_driven_ppo_improvement_run,
        )
        from scripts.run_refined_coverage_reward_margin_tuning import (
            run_refined_coverage_reward_margin_tuning,
        )

        self.fixture._write_inputs(safe_better=True)
        run_refined_coverage_driven_ppo_improvement_run(
            stage5a2_root=self.fixture.stage5a2_root,
            coverage_driven_root=self.fixture.coverage_driven_root,
            formal_training_root=self.fixture.formal_root,
            post_training_replay_root=self.fixture.replay_root,
            selected_candidate_root=self.fixture.selected_root,
            coverage_signal_root=self.fixture.signal_root,
            coverage_performance_root=self.fixture.performance_root,
            reward_refinement_root=self.fixture.reward_root,
            output_root=self.refined_root,
            repo_root=self.repo_root,
        )

        summary = run_refined_coverage_reward_margin_tuning(
            refined_root=self.refined_root,
            stage5a2_root=self.fixture.stage5a2_root,
            coverage_driven_root=self.fixture.coverage_driven_root,
            formal_training_root=self.fixture.formal_root,
            post_training_replay_root=self.fixture.replay_root,
            selected_candidate_root=self.fixture.selected_root,
            coverage_signal_root=self.fixture.signal_root,
            coverage_performance_root=self.fixture.performance_root,
            reward_refinement_root=self.fixture.reward_root,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["tuning_config_count"], 4)
        self.assertEqual(summary["safe_better_training_pair_count"], 1)
        self.assertEqual(summary["ppo_advantage_nonzero_count"], 4)
        self.assertIn(summary["best_config_id"], {"advantage_x3", "advantage_x5", "advantage_x8", "reward_margin_x5"})
        self.assertFalse(summary["publishes_checkpoint"])
        self.assertFalse(summary["replaces_default_policy"])
        self.assertFalse(summary["performance_claimed"])

        config_rows = self._read_jsonl(self.output_root / "tuning-config-audit.jsonl")
        self.assertEqual({row["config_id"] for row in config_rows}, {
            "advantage_x3",
            "advantage_x5",
            "advantage_x8",
            "reward_margin_x5",
        })
        self.assertTrue(all(row["ppo_advantage_nonzero_count"] == 1 for row in config_rows))

        for row in config_rows:
            config_id = row["config_id"]
            episodes = self._read_jsonl(
                self.output_root
                / "tuning-configs"
                / config_id
                / "coverage-aware-ppo-batch"
                / "ppo-rollout-episodes.jsonl"
            )
            info = episodes[0]["transitions"][0]["info"]
            self.assertGreater(info["ppo_advantage"], 0.0)
            self.assertGreater(info["ppo_return"], info["old_value"])
            self.assertEqual(info["advantage_scale"], row["advantage_scale"])
            self.assertEqual(info["margin_scale"], row["margin_scale"])
            self.assertGreater(info["teacher_margin_target"], 0.0)
            update_summary = json.loads(
                (
                    self.output_root
                    / "tuning-configs"
                    / config_id
                    / "coverage-driven-ppo-update-summary.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(update_summary["status"], "passed")
            self.assertEqual(update_summary["optimizer_train_transition_count"], 1)

        for filename in (
            "refined-coverage-reward-margin-tuning-summary.json",
            "tuning-config-audit.jsonl",
            "refined-coverage-reward-margin-tuning-report.md",
            "rejection-report.json",
        ):
            self.assertTrue((self.output_root / filename).is_file(), filename)

    def test_tuning_fails_before_update_when_refined_advantage_is_missing(self) -> None:
        from scripts.run_refined_coverage_driven_ppo_improvement_run import (
            run_refined_coverage_driven_ppo_improvement_run,
        )
        from scripts.run_refined_coverage_reward_margin_tuning import (
            run_refined_coverage_reward_margin_tuning,
        )

        self.fixture._write_inputs(safe_better=False, candidate_risk=0.50)
        run_refined_coverage_driven_ppo_improvement_run(
            stage5a2_root=self.fixture.stage5a2_root,
            coverage_driven_root=self.fixture.coverage_driven_root,
            formal_training_root=self.fixture.formal_root,
            post_training_replay_root=self.fixture.replay_root,
            selected_candidate_root=self.fixture.selected_root,
            coverage_signal_root=self.fixture.signal_root,
            coverage_performance_root=self.fixture.performance_root,
            reward_refinement_root=self.fixture.reward_root,
            output_root=self.refined_root,
            repo_root=self.repo_root,
        )

        summary = run_refined_coverage_reward_margin_tuning(
            refined_root=self.refined_root,
            stage5a2_root=self.fixture.stage5a2_root,
            coverage_driven_root=self.fixture.coverage_driven_root,
            formal_training_root=self.fixture.formal_root,
            post_training_replay_root=self.fixture.replay_root,
            selected_candidate_root=self.fixture.selected_root,
            coverage_signal_root=self.fixture.signal_root,
            coverage_performance_root=self.fixture.performance_root,
            reward_refinement_root=self.fixture.reward_root,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("insufficient_tuning_advantage_materialization", summary["reason_codes"])
        self.assertEqual(summary["next_required_change"], "fix_tuning_advantage_materialization")
        self.assertEqual(summary["safe_better_training_pair_count"], 0)
        self.assertEqual(summary["ppo_advantage_nonzero_count"], 0)
        self.assertFalse(summary["runs_new_ppo_update"])
        self.assertFalse(summary["performance_claimed"])

    def _read_jsonl(self, path: Path) -> list[dict]:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


if __name__ == "__main__":
    unittest.main()
