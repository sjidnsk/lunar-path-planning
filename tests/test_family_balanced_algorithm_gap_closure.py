import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path


class FamilyBalancedAlgorithmGapClosureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        scripts = str(self.repo_root / "scripts")
        if scripts not in sys.path:
            sys.path.insert(0, scripts)
        self.temp_dir = Path(tempfile.mkdtemp(prefix="family-balanced-gap-"))
        self.stage16_root = self.temp_dir / "stage16"
        self.stage15_root = self.temp_dir / "stage15"
        self.stage14_root = self.temp_dir / "stage14"
        self.cost_root = self.temp_dir / "cost-efficiency"
        self.low_obs_root = self.temp_dir / "low-observation"
        self.safe_better_root = self.temp_dir / "safe-better"
        self.output_root = self.temp_dir / "output"
        for path in (
            self.stage16_root,
            self.stage15_root,
            self.stage14_root,
            self.cost_root,
            self.low_obs_root,
            self.safe_better_root,
        ):
            path.mkdir(parents=True)
        self._write_docs()
        self._write_stage_summaries()

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def test_runner_merges_low_observation_geometry_and_writes_balanced_inputs(self) -> None:
        from scripts.run_family_balanced_algorithm_gap_closure import (
            run_family_balanced_algorithm_gap_closure,
        )

        self._write_cost_efficiency_inputs(low_observation_count=2)
        self._write_low_observation_inputs(count=36)
        self._write_safe_better_inputs(low_observation_count=36)

        summary = run_family_balanced_algorithm_gap_closure(
            stage16_root=self.stage16_root,
            stage15_root=self.stage15_root,
            stage14_root=self.stage14_root,
            cost_efficiency_root=self.cost_root,
            low_observation_geometry_root=self.low_obs_root,
            safe_better_root=self.safe_better_root,
            output_root=self.output_root,
            repo_root=self.temp_dir,
            minimum_low_observation_count=32,
            minimum_family_pair_count=32,
            minimum_family_balance_ratio=0.15,
        )

        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["reason_codes"], [])
        self.assertEqual(
            summary["gap_closure_verdict"],
            "eligible_for_family_balanced_coverage_driven_ppo_rerun",
        )
        self.assertTrue(summary["family_balanced_algorithm_gap_closure_passed"])
        self.assertTrue(summary["family_balanced_coverage_driven_ppo_rerun_approved"])
        self.assertTrue(summary["low_observation_gap_closed"])
        self.assertTrue(summary["source_backed_counterfactual_audit_passed"])
        self.assertTrue(summary["family_balance_audit_passed"])
        self.assertEqual(summary["next_required_change"], "family_balanced_coverage_driven_ppo_rerun")
        self.assertEqual(summary["low_observation_count"], 36)
        self.assertEqual(summary["minimum_family_pair_count"], 32)
        self.assertGreaterEqual(summary["family_balance_ratio"], 0.15)
        self.assertFalse(summary["checkpoint_publication_approved"])
        self.assertFalse(summary["default_policy_replacement_approved"])
        self.assertFalse(summary["real_executor_connection_approved"])
        self.assertFalse(summary["publishes_checkpoint"])
        self.assertFalse(summary["replaces_default_policy"])
        self.assertFalse(summary["connects_real_executor"])

        pairs = self._read_jsonl(self.output_root / "family-balanced-safe-better-pairs.jsonl")
        counterfactual = self._read_jsonl(
            self.output_root / "family-balanced-counterfactual-rollouts.jsonl"
        )
        self.assertEqual(len(pairs), 132)
        self.assertEqual(len(counterfactual), len(pairs))
        self.assertEqual(
            self._family_counts(pairs),
            {
                "low_observation_count": 36,
                "mixed_risk": 32,
                "rim_or_steep_slope": 32,
                "smooth_high_confidence": 32,
            },
        )
        self.assertTrue(all(row["split"] == "train" for row in pairs))
        self.assertTrue(
            all(
                row["family_balance_source_stage"]
                in {"stage5b7_cost_efficiency_filter", "low_observation_geometry_supplement"}
                for row in pairs
            )
        )
        for filename in (
            "family-balanced-algorithm-gap-closure-summary.json",
            "family-balanced-source-ledger.json",
            "family-balance-audit.json",
            "low-observation-gap-audit.json",
            "family-balanced-coverage-driven-input-manifest.json",
            "family-balanced-release-boundary-audit.json",
            "family-balanced-rejection-report.json",
            "family-balanced-algorithm-gap-closure-report.md",
        ):
            self.assertTrue((self.output_root / filename).is_file(), filename)

    def test_fails_when_stage16_is_not_passed(self) -> None:
        from scripts.run_family_balanced_algorithm_gap_closure import (
            run_family_balanced_algorithm_gap_closure,
        )

        self._write_json(
            self.stage16_root / "default-policy-candidate-authorization-preflight-summary.json",
            {
                "status": "failed",
                "reason_codes": ["fixture_failure"],
                "default_policy_candidate_authorization_preflight_passed": False,
                "checkpoint_publication_approved": False,
                "default_policy_replacement_approved": False,
                "real_executor_connection_approved": False,
            },
        )
        self._write_cost_efficiency_inputs(low_observation_count=2)
        self._write_low_observation_inputs(count=36)
        self._write_safe_better_inputs(low_observation_count=36)

        summary = run_family_balanced_algorithm_gap_closure(
            stage16_root=self.stage16_root,
            stage15_root=self.stage15_root,
            stage14_root=self.stage14_root,
            cost_efficiency_root=self.cost_root,
            low_observation_geometry_root=self.low_obs_root,
            safe_better_root=self.safe_better_root,
            output_root=self.output_root,
            repo_root=self.temp_dir,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("stage16_not_passed", summary["reason_codes"])
        self.assertFalse(summary["family_balanced_coverage_driven_ppo_rerun_approved"])

    def test_fails_when_low_observation_remains_below_threshold(self) -> None:
        from scripts.run_family_balanced_algorithm_gap_closure import (
            run_family_balanced_algorithm_gap_closure,
        )

        self._write_cost_efficiency_inputs(low_observation_count=2)
        self._write_low_observation_inputs(count=4)
        self._write_safe_better_inputs(low_observation_count=4)

        summary = run_family_balanced_algorithm_gap_closure(
            stage16_root=self.stage16_root,
            stage15_root=self.stage15_root,
            stage14_root=self.stage14_root,
            cost_efficiency_root=self.cost_root,
            low_observation_geometry_root=self.low_obs_root,
            safe_better_root=self.safe_better_root,
            output_root=self.output_root,
            repo_root=self.temp_dir,
            minimum_low_observation_count=32,
            minimum_family_pair_count=32,
            minimum_family_balance_ratio=0.15,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("low_observation_gap_not_closed", summary["reason_codes"])
        self.assertIn("family_balance_below_threshold", summary["reason_codes"])

    def test_fails_when_trainable_output_would_include_non_train_split(self) -> None:
        from scripts.run_family_balanced_algorithm_gap_closure import (
            run_family_balanced_algorithm_gap_closure,
        )

        self._write_cost_efficiency_inputs(low_observation_count=2)
        self._write_low_observation_inputs(count=36, split="validation")
        self._write_safe_better_inputs(low_observation_count=36, split="validation")

        summary = run_family_balanced_algorithm_gap_closure(
            stage16_root=self.stage16_root,
            stage15_root=self.stage15_root,
            stage14_root=self.stage14_root,
            cost_efficiency_root=self.cost_root,
            low_observation_geometry_root=self.low_obs_root,
            safe_better_root=self.safe_better_root,
            output_root=self.output_root,
            repo_root=self.temp_dir,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("non_train_split_in_trainable_output", summary["reason_codes"])

    def test_fails_when_release_boundary_is_open(self) -> None:
        from scripts.run_family_balanced_algorithm_gap_closure import (
            run_family_balanced_algorithm_gap_closure,
        )

        self._write_stage_summaries(stage16_overrides={"publishes_checkpoint": True})
        self._write_cost_efficiency_inputs(low_observation_count=2)
        self._write_low_observation_inputs(count=36)
        self._write_safe_better_inputs(low_observation_count=36)

        summary = run_family_balanced_algorithm_gap_closure(
            stage16_root=self.stage16_root,
            stage15_root=self.stage15_root,
            stage14_root=self.stage14_root,
            cost_efficiency_root=self.cost_root,
            low_observation_geometry_root=self.low_obs_root,
            safe_better_root=self.safe_better_root,
            output_root=self.output_root,
            repo_root=self.temp_dir,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("stage16_release_boundary_violation", summary["reason_codes"])
        self.assertIn("release_boundary_violation", summary["reason_codes"])

    def _write_docs(self) -> None:
        self._write_text(
            self.temp_dir / "README.md",
            "Family-Balanced Algorithm Gap Closure v1\n"
            "outputs/path_feedback_batch_family_balanced_algorithm_gap_closure_v1/\n"
            "family_balanced_coverage_driven_ppo_rerun\n"
            "checkpoint_publication_approved=false\n"
            "default_policy_replacement_approved=false\n"
            "real_executor_connection_approved=false\n",
        )
        docs_dir = self.temp_dir / "docs"
        self._write_text(
            docs_dir / "算法设计与系统架构报告.md",
            "Family-Balanced Algorithm Gap Closure v1\n"
            "低观测 family 泛化缺口\n"
            "不发布 checkpoint、不替换 default policy、不连接真实执行器\n",
        )
        self._write_text(
            docs_dir / "superpowers" / "specs" / "2026-06-16-family-balanced-algorithm-gap-closure.md",
            "Family-Balanced Algorithm Gap Closure v1\n"
            "family-balanced-algorithm-gap-closure-summary.json\n"
            "family_balanced_coverage_driven_ppo_rerun\n",
        )

    def _write_stage_summaries(self, stage16_overrides: dict | None = None) -> None:
        stage16 = {
            "status": "passed",
            "reason_codes": [],
            "authorization_verdict": "eligible_for_default_policy_candidate_sandbox_install_preflight",
            "default_policy_candidate_authorization_preflight_passed": True,
            "default_policy_candidate_sandbox_install_preflight_approved": True,
            "checkpoint_publication_approved": False,
            "default_policy_replacement_approved": False,
            "real_executor_connection_approved": False,
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "connects_real_executor": False,
        }
        if stage16_overrides:
            stage16.update(stage16_overrides)
        self._write_json(
            self.stage16_root / "default-policy-candidate-authorization-preflight-summary.json",
            stage16,
        )
        self._write_json(
            self.stage15_root / "checkpoint-publication-sandbox-consumer-replay-canary-summary.json",
            {"status": "passed", "reason_codes": [], "publishes_checkpoint": False},
        )
        self._write_json(
            self.stage14_root / "checkpoint-publication-sandbox-install-dry-run-verification-summary.json",
            {"status": "passed", "reason_codes": [], "publishes_checkpoint": False},
        )

    def _write_cost_efficiency_inputs(self, low_observation_count: int) -> None:
        rows = []
        for family, count in (
            ("low_observation_count", low_observation_count),
            ("mixed_risk", 32),
            ("rim_or_steep_slope", 32),
            ("smooth_high_confidence", 32),
        ):
            rows.extend(
                self._pair(
                    family=family,
                    index=len(rows) + index,
                    source_stage="stage5b7_cost_efficiency_filter",
                )
                for index in range(count)
            )
        self._write_json(
            self.cost_root / "cost-efficiency-aware-coverage-reward-candidate-filter-summary.json",
            {
                "status": "passed",
                "reason_codes": [],
                "trainable_pair_count": len(rows),
                "safe_better_training_family_count": 4,
                "coverage_return_improvement": 54.96083408637,
                "coverage_efficiency_regression": False,
                "fallback_gain_contamination_count": 0,
                "controlled_regression_count": 0,
                "publishes_checkpoint": False,
                "replaces_default_policy": False,
                "connects_real_executor": False,
            },
        )
        self._write_jsonl(self.cost_root / "cost-efficiency-filtered-batch.jsonl", rows)
        self._write_jsonl(self.cost_root / "cost-efficiency-advantage-audit.jsonl", [])
        stage5a2 = self.cost_root / "cost-efficiency-compatible-stage5a2-input"
        stage5a2.mkdir(parents=True)
        self._write_json(
            stage5a2 / "policy-differentiating-counterfactual-coverage-rollouts-summary.json",
            {"status": "passed", "reason_codes": []},
        )
        self._write_jsonl(
            stage5a2 / "counterfactual-coverage-rollouts.jsonl",
            [self._counterfactual(row) for row in rows],
        )

    def _write_low_observation_inputs(self, count: int, split: str = "train") -> None:
        overlay_rows = []
        counterfactual_rows = []
        for index in range(count):
            row = self._pair(
                family="low_observation_count",
                index=1000 + index,
                split=split,
                source_stage="low_observation_geometry_supplement",
            )
            overlay_rows.append(
                {
                    **row,
                    "schema_version": "low-observation-candidate-geometry-overlay-row/v1",
                    "action_index": row["candidate_action_index"],
                    "is_teacher_action": False,
                }
            )
            counterfactual_rows.append(self._counterfactual(row))
        self._write_json(
            self.low_obs_root / "low-observation-candidate-geometry-summary.json",
            {
                "status": "passed",
                "reason_codes": [],
                "low_observation_trainable_safe_better_pair_count": count if split == "train" else 0,
                "missing_counterfactual_source_count": 0,
                "fallback_gain_contamination_count": 0,
                "controlled_regression_count": 0,
                "publishes_checkpoint": False,
                "replaces_default_policy": False,
                "connects_real_executor": False,
            },
        )
        self._write_jsonl(
            self.low_obs_root / "low-observation-candidate-geometry-overlay.jsonl",
            overlay_rows,
        )
        self._write_jsonl(
            self.low_obs_root / "low-observation-candidate-geometry-counterfactual-rollouts.jsonl",
            counterfactual_rows,
        )

    def _write_safe_better_inputs(self, low_observation_count: int, split: str = "train") -> None:
        rows = []
        counterfactual_rows = []
        for family, count in (
            ("low_observation_count", low_observation_count),
            ("mixed_risk", 32),
            ("rim_or_steep_slope", 32),
            ("smooth_high_confidence", 32),
        ):
            row_split = split if family == "low_observation_count" else "train"
            for index in range(count):
                row = self._pair(
                    family=family,
                    index=2000 + len(rows),
                    split=row_split,
                    source_stage="safe_better_pair_expansion",
                )
                rows.append(row)
                counterfactual_rows.append(self._counterfactual(row))
        self._write_json(
            self.safe_better_root / "safe-better-pair-expansion-summary.json",
            {
                "status": "passed",
                "reason_codes": [],
                "safe_better_than_teacher_family_count": 4,
                "family_safe_better_counts": self._family_counts(rows),
                "missing_counterfactual_source_count": 0,
                "fallback_gain_contamination_count": 0,
                "controlled_regression_count": 0,
                "publishes_checkpoint": False,
                "replaces_default_policy": False,
                "connects_real_executor": False,
            },
        )
        self._write_jsonl(self.safe_better_root / "expanded-safe-better-pairs.jsonl", rows)
        self._write_jsonl(
            self.safe_better_root / "expanded-counterfactual-coverage-rollouts.jsonl",
            counterfactual_rows,
        )

    def _pair(
        self,
        *,
        family: str,
        index: int,
        split: str = "train",
        source_stage: str,
    ) -> dict:
        coverage_advantage = 0.02
        return {
            "schema_version": "safe-better-pair-expansion-row/v1",
            "context_id": f"context-{family}-{index}",
            "episode_id": f"episode-{family}-{index}",
            "step_index": index,
            "scenario_id": f"scenario-{family}-{index}",
            "scenario_family": family,
            "split": split,
            "ppo_trainable": split == "train",
            "coverage_source_available": True,
            "safe_better_than_teacher_candidate": True,
            "fallback_like": False,
            "guard_rejected": False,
            "controlled_regression_reason_codes": [],
            "candidate_action_index": 2,
            "teacher_action_index": 1,
            "expected_coverage_rate_delta": 0.06,
            "teacher_expected_coverage_rate_delta": 0.04,
            "expected_new_coverage_area": 10.0,
            "teacher_expected_new_coverage_area": 8.0,
            "information_gain": 0.02,
            "teacher_information_gain": 0.01,
            "valuable_coverage_proxy": 0.5,
            "teacher_valuable_coverage_proxy": 0.4,
            "coverage_advantage": coverage_advantage,
            "valuable_coverage_advantage": 0.1,
            "path_cost": 12.0,
            "teacher_path_cost": 10.0,
            "path_cost_delta": 2.0,
            "risk": 0.2,
            "teacher_risk": 0.2,
            "risk_delta": 0.0,
            "energy_cost": 100.0,
            "teacher_energy_cost": 100.0,
            "energy_delta": 0.0,
            "match_method": source_stage,
            "source_path": str(self.temp_dir / f"{source_stage}.json"),
            "source_confidence": 1.0,
        }

    def _counterfactual(self, row: dict) -> dict:
        return {
            "schema_version": "expanded-counterfactual-coverage-row/v1",
            "context_id": row["context_id"],
            "episode_id": row["episode_id"],
            "step_index": row["step_index"],
            "scenario_id": row["scenario_id"],
            "scenario_family": row["scenario_family"],
            "split": row["split"],
            "action_index": row["candidate_action_index"],
            "teacher_action_index": row["teacher_action_index"],
            "expected_coverage_rate_delta": row["expected_coverage_rate_delta"],
            "teacher_expected_coverage_rate_delta": row["teacher_expected_coverage_rate_delta"],
            "expected_new_coverage_area": row["expected_new_coverage_area"],
            "valuable_coverage_proxy": row["valuable_coverage_proxy"],
            "information_gain": row["information_gain"],
            "coverage_advantage": row["coverage_advantage"],
            "coverage_source_available": True,
            "safe_better_than_teacher_candidate": True,
            "fallback_like": False,
            "guard_rejected": False,
            "controlled_regression_reason_codes": [],
            "source_path": row["source_path"],
        }

    def _family_counts(self, rows: list[dict]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for row in rows:
            family = row["scenario_family"]
            counts[family] = counts.get(family, 0) + 1
        return dict(sorted(counts.items()))

    def _write_json(self, path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _write_jsonl(self, path: Path, rows: list[dict]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )

    def _write_text(self, path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def _read_jsonl(self, path: Path) -> list[dict]:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


if __name__ == "__main__":
    unittest.main()
