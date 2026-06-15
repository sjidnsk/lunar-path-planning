import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path


class SafeBetterPairExpansionAcrossFamiliesTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        scripts_path = str(self.repo_root / "scripts")
        if scripts_path not in sys.path:
            sys.path.insert(0, scripts_path)
        self.temp_dir = Path(tempfile.mkdtemp(prefix="safe-better-expansion-"))
        self.tuning_root = self.temp_dir / "tuning"
        self.stage5a2_root = self.temp_dir / "stage5a2"
        self.quasi_real_root = self.temp_dir / "quasi-real"
        self.formal_root = self.temp_dir / "formal"
        self.replay_root = self.temp_dir / "replay"
        self.signal_root = self.temp_dir / "signal"
        self.reward_root = self.temp_dir / "reward"
        self.output_root = self.temp_dir / "output"
        for path in (
            self.tuning_root,
            self.stage5a2_root,
            self.quasi_real_root,
            self.formal_root,
            self.replay_root,
            self.signal_root,
            self.reward_root,
        ):
            path.mkdir(parents=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def test_passes_when_train_safe_better_pairs_cover_four_families(self) -> None:
        from scripts.run_safe_better_pair_expansion_across_families import (
            run_safe_better_pair_expansion_across_families,
        )

        rows = []
        counterfactual_rows = []
        for family in (
            "low_observation_count",
            "mixed_risk",
            "rim_or_steep_slope",
            "smooth_high_confidence",
        ):
            for index in range(32):
                rows.extend(self._decision_rows(family=family, index=index, split="train"))
                counterfactual_rows.append(self._counterfactual_row(family=family, index=index))
        self._write_inputs(rows, counterfactual_rows)

        summary = run_safe_better_pair_expansion_across_families(
            tuning_root=self.tuning_root,
            stage5a2_root=self.stage5a2_root,
            quasi_real_root=self.quasi_real_root,
            formal_training_root=self.formal_root,
            post_training_replay_root=self.replay_root,
            coverage_signal_root=self.signal_root,
            reward_refinement_root=self.reward_root,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["reason_codes"], [])
        self.assertEqual(summary["next_required_change"], "rerun_refined_coverage_driven_ppo_improvement")
        self.assertEqual(summary["safe_better_than_teacher_candidate_count"], 128)
        self.assertEqual(summary["safe_better_than_teacher_family_count"], 4)
        self.assertEqual(summary["trainable_safe_better_pair_count"], 128)
        self.assertEqual(summary["missing_counterfactual_source_count"], 0)
        self.assertEqual(summary["fallback_gain_contamination_count"], 0)
        self.assertEqual(summary["controlled_regression_count"], 0)
        self.assertFalse(summary["runs_new_ppo_update"])
        self.assertFalse(summary["publishes_checkpoint"])
        self.assertFalse(summary["replaces_default_policy"])
        self.assertFalse(summary["performance_claimed"])

        pairs = self._read_jsonl(self.output_root / "expanded-safe-better-pairs.jsonl")
        self.assertEqual(len(pairs), 128)
        self.assertTrue(all(row["ppo_trainable"] for row in pairs))
        self.assertTrue(all(row["coverage_advantage"] > 0.0 for row in pairs))
        self.assertEqual(
            {
                row["scenario_family"]
                for row in pairs
            },
            {
                "low_observation_count",
                "mixed_risk",
                "rim_or_steep_slope",
                "smooth_high_confidence",
            },
        )
        for filename in (
            "safe-better-pair-expansion-summary.json",
            "expanded-safe-better-pairs.jsonl",
            "expanded-counterfactual-coverage-rollouts.jsonl",
            "family-safe-better-gap-report.json",
            "safe-better-pair-expansion-report.md",
        ):
            self.assertTrue((self.output_root / filename).is_file(), filename)

    def test_fails_with_named_family_gap_when_safe_better_distribution_is_too_narrow(self) -> None:
        from scripts.run_safe_better_pair_expansion_across_families import (
            run_safe_better_pair_expansion_across_families,
        )

        rows = []
        counterfactual_rows = []
        for index in range(128):
            rows.extend(self._decision_rows(family="rim_or_steep_slope", index=index, split="train"))
            counterfactual_rows.append(self._counterfactual_row(family="rim_or_steep_slope", index=index))
        self._write_inputs(rows, counterfactual_rows)

        summary = run_safe_better_pair_expansion_across_families(
            tuning_root=self.tuning_root,
            stage5a2_root=self.stage5a2_root,
            quasi_real_root=self.quasi_real_root,
            formal_training_root=self.formal_root,
            post_training_replay_root=self.replay_root,
            coverage_signal_root=self.signal_root,
            reward_refinement_root=self.reward_root,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("safe_better_family_count_below_threshold", summary["reason_codes"])
        self.assertIn("family_safe_better_gap_low_observation_count", summary["reason_codes"])
        self.assertIn("family_safe_better_gap_mixed_risk", summary["reason_codes"])
        self.assertIn("family_safe_better_gap_smooth_high_confidence", summary["reason_codes"])
        self.assertEqual(summary["next_required_change"], "expand_safe_better_pair_generation_across_families")
        self.assertFalse(summary["performance_claimed"])

    def test_missing_source_and_fallback_rows_are_not_trainable(self) -> None:
        from scripts.run_safe_better_pair_expansion_across_families import (
            run_safe_better_pair_expansion_across_families,
        )

        rows = []
        rows.extend(
            self._decision_rows(
                family="mixed_risk",
                index=0,
                split="train",
                candidate_overrides={"coverage_source_available": False},
            )
        )
        rows.extend(
            self._decision_rows(
                family="rim_or_steep_slope",
                index=1,
                split="train",
                candidate_overrides={"fallback_like": True},
            )
        )
        self._write_inputs(rows, [])

        summary = run_safe_better_pair_expansion_across_families(
            tuning_root=self.tuning_root,
            stage5a2_root=self.stage5a2_root,
            quasi_real_root=self.quasi_real_root,
            formal_training_root=self.formal_root,
            post_training_replay_root=self.replay_root,
            coverage_signal_root=self.signal_root,
            reward_refinement_root=self.reward_root,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertEqual(summary["safe_better_than_teacher_candidate_count"], 0)
        self.assertEqual(summary["trainable_safe_better_pair_count"], 0)
        self.assertEqual(summary["missing_counterfactual_source_count"], 1)
        self.assertEqual(summary["fallback_gain_contamination_count"], 1)
        self.assertIn("safe_better_pair_count_below_threshold", summary["reason_codes"])
        self.assertIn("counterfactual_coverage_source_missing", summary["reason_codes"])
        self.assertIn("fallback_gain_contamination", summary["reason_codes"])

    def test_validation_and_test_rows_are_diagnostic_only(self) -> None:
        from scripts.run_safe_better_pair_expansion_across_families import (
            run_safe_better_pair_expansion_across_families,
        )

        rows = []
        counterfactual_rows = []
        for family in (
            "low_observation_count",
            "mixed_risk",
            "rim_or_steep_slope",
            "smooth_high_confidence",
        ):
            for index in range(40):
                split = "validation" if index % 2 == 0 else "test"
                rows.extend(self._decision_rows(family=family, index=index, split=split))
                counterfactual_rows.append(self._counterfactual_row(family=family, index=index))
        self._write_inputs(rows, counterfactual_rows)

        summary = run_safe_better_pair_expansion_across_families(
            tuning_root=self.tuning_root,
            stage5a2_root=self.stage5a2_root,
            quasi_real_root=self.quasi_real_root,
            formal_training_root=self.formal_root,
            post_training_replay_root=self.replay_root,
            coverage_signal_root=self.signal_root,
            reward_refinement_root=self.reward_root,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertEqual(summary["diagnostic_safe_better_pair_count"], 160)
        self.assertEqual(summary["trainable_safe_better_pair_count"], 0)
        self.assertIn("safe_better_pair_count_below_threshold", summary["reason_codes"])
        pairs = self._read_jsonl(self.output_root / "expanded-safe-better-pairs.jsonl")
        self.assertTrue(pairs)
        self.assertFalse(any(row["ppo_trainable"] for row in pairs))

    def test_supplemental_low_observation_overlay_can_close_family_gap(self) -> None:
        from scripts.run_safe_better_pair_expansion_across_families import (
            run_safe_better_pair_expansion_across_families,
        )

        base_rows = []
        base_counterfactual_rows = []
        for family in ("mixed_risk", "rim_or_steep_slope", "smooth_high_confidence"):
            for index in range(40):
                base_rows.extend(self._decision_rows(family=family, index=index, split="train"))
                base_counterfactual_rows.append(self._counterfactual_row(family=family, index=index))
        self._write_inputs(base_rows, base_counterfactual_rows)

        supplemental_overlay_path = self.temp_dir / "low-observation-supplemental-overlay.jsonl"
        supplemental_counterfactual_path = (
            self.temp_dir / "low-observation-supplemental-counterfactual-rollouts.jsonl"
        )
        supplemental_rows = []
        supplemental_counterfactual_rows = []
        for index in range(8):
            supplemental_rows.extend(
                self._decision_rows(family="low_observation_count", index=index, split="train")
            )
            supplemental_counterfactual_rows.append(
                self._counterfactual_row(family="low_observation_count", index=index)
            )
        self._write_jsonl(supplemental_overlay_path, supplemental_rows)
        self._write_jsonl(supplemental_counterfactual_path, supplemental_counterfactual_rows)

        summary = run_safe_better_pair_expansion_across_families(
            tuning_root=self.tuning_root,
            stage5a2_root=self.stage5a2_root,
            quasi_real_root=self.quasi_real_root,
            formal_training_root=self.formal_root,
            post_training_replay_root=self.replay_root,
            coverage_signal_root=self.signal_root,
            reward_refinement_root=self.reward_root,
            output_root=self.output_root,
            repo_root=self.repo_root,
            supplemental_overlay_paths=[supplemental_overlay_path],
            supplemental_counterfactual_paths=[supplemental_counterfactual_path],
        )

        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["safe_better_than_teacher_candidate_count"], 128)
        self.assertEqual(summary["safe_better_than_teacher_family_count"], 4)
        self.assertEqual(summary["family_safe_better_counts"]["low_observation_count"], 8)
        self.assertEqual(summary["base_candidate_overlay_row_count"], 240)
        self.assertEqual(summary["supplemental_overlay_row_count"], 16)
        self.assertEqual(summary["candidate_overlay_row_count"], 256)
        self.assertEqual(summary["base_counterfactual_coverage_row_count"], 120)
        self.assertEqual(summary["supplemental_counterfactual_row_count"], 8)
        self.assertEqual(summary["counterfactual_coverage_row_count"], 128)
        self.assertEqual(summary["deduplicated_overlay_row_count"], 0)
        self.assertEqual(summary["deduplicated_counterfactual_row_count"], 0)
        self.assertEqual(
            summary["supplemental_input_artifact_paths"]["candidate_coverage_overlays"],
            [str(supplemental_overlay_path)],
        )
        self.assertEqual(
            summary["supplemental_input_artifact_paths"]["counterfactual_coverage_rollouts"],
            [str(supplemental_counterfactual_path)],
        )
        self.assertFalse(summary["performance_claimed"])

    def _write_inputs(self, overlay_rows: list[dict], counterfactual_rows: list[dict]) -> None:
        overlay_path = self.stage5a2_root / "candidate-level-coverage-overlay.jsonl"
        counterfactual_path = self.stage5a2_root / "counterfactual-coverage-rollouts.jsonl"
        self._write_jsonl(overlay_path, overlay_rows)
        self._write_jsonl(counterfactual_path, counterfactual_rows)
        self._write_json(
            self.tuning_root / "refined-coverage-reward-margin-tuning-summary.json",
            {
                "schema_version": "refined-coverage-reward-margin-tuning-summary/v1",
                "status": "failed",
                "reason_codes": ["post_update_policy_teacher_equivalent"],
                "next_required_change": "expand_safe_better_pair_generation_across_families",
                "tuning_config_count": 4,
                "executed_tuning_config_count": 4,
                "safe_better_training_pair_count": 51,
                "ppo_advantage_nonzero_count": 204,
                "policy_argmax_changed_count": 0,
                "coverage_return_improvement": -1.0,
                "cumulative_coverage_rate_delta_improvement": -1.0,
                "fallback_rate": 1.0,
                "performance_claimed": False,
            },
        )
        self._write_json(
            self.stage5a2_root / "policy-differentiating-counterfactual-coverage-rollouts-summary.json",
            {
                "schema_version": "policy-differentiating-counterfactual-coverage-rollouts-summary/v1",
                "status": "passed",
                "reason_codes": [],
                "next_required_change": "rerun_coverage_driven_ppo_with_refined_reward_or_advantage",
                "candidate_coverage_overlay": str(overlay_path),
                "counterfactual_coverage_rollouts": str(counterfactual_path),
                "safe_better_than_teacher_candidate_count": 51,
                "safe_better_than_teacher_family_count": 2,
                "missing_counterfactual_source_count": 0,
                "fallback_gain_contamination_count": 0,
                "controlled_regression_count": 0,
                "runs_new_ppo_update": False,
                "performance_claimed": False,
            },
        )
        self._write_json(
            self.stage5a2_root / "family-action-gap-report.json",
            {"schema_version": "policy-differentiating-counterfactual-family-action-gap-report/v1", "rows": []},
        )
        self._write_json(
            self.quasi_real_root / "quasi-real-map-path-feedback-summary.json",
            {"schema_version": "quasi-real-map-path-feedback-summary/v1", "candidate_count": 0},
        )
        self._write_json(
            self.quasi_real_root / "quasi-real-safe-alternative-opportunity-summary.json",
            {
                "schema_version": "quasi-real-safe-alternative-opportunity-summary/v1",
                "status": "passed",
                "safe_better_opportunity_context_count": 0,
                "performance_claimed": False,
            },
        )
        self._write_json(
            self.formal_root / "formal-ppo-training-run-summary.json",
            {"schema_version": "formal-ppo-training-run-summary/v1", "status": "passed", "performance_claimed": False},
        )
        self._write_json(
            self.replay_root / "formal-ppo-post-training-stability-replay-summary.json",
            {
                "schema_version": "formal-ppo-post-training-stability-replay-summary/v1",
                "status": "passed",
                "performance_claimed": False,
            },
        )
        self._write_json(
            self.signal_root / "exploration-coverage-signal-audit-summary.json",
            {"schema_version": "exploration-coverage-signal-audit-summary/v1", "status": "passed"},
        )
        self._write_json(
            self.reward_root / "coverage-aware-reward-refinement-summary.json",
            {"schema_version": "coverage-aware-reward-refinement-summary/v1", "status": "passed"},
        )

    def _decision_rows(
        self,
        *,
        family: str,
        index: int,
        split: str,
        candidate_overrides: dict | None = None,
    ) -> list[dict]:
        key = f"{family}-{split}-{index}"
        teacher = {
            "schema_version": "candidate-level-coverage-overlay-row/v1",
            "context_id": key,
            "episode_id": f"episode-{key}",
            "step_index": index,
            "scenario_family": family,
            "scenario_id": f"{family}-{split}-{index}",
            "split": split,
            "action_index": 0,
            "teacher_action_index": 0,
            "is_teacher_action": True,
            "action_mask_valid": True,
            "guard_rejected": False,
            "fallback_like": False,
            "coverage_source_available": True,
            "expected_coverage_rate_delta": 0.1,
            "expected_new_coverage_area": 0.1,
            "information_gain": 0.1,
            "valuable_coverage_proxy": 0.05,
            "path_cost": 10.0,
            "risk": 0.1,
            "energy_cost": 1.0,
            "controlled_regression_reason_codes": [],
            "match_method": "executed_action_actual_path_feedback",
            "source_path": "teacher.json",
            "source_confidence": 1.0,
        }
        candidate = {
            **teacher,
            "action_index": 1,
            "is_teacher_action": False,
            "coverage_source_available": True,
            "counterfactual_coverage_observed": True,
            "expected_coverage_rate_delta": 0.25,
            "expected_new_coverage_area": 0.25,
            "information_gain": 0.12,
            "valuable_coverage_proxy": 0.13,
            "value": 0.13,
            "path_cost": 11.0,
            "risk": 0.12,
            "energy_cost": 1.1,
            "match_method": "candidate_counterfactual_path_feedback",
            "source_execution_type": "counterfactual_candidate",
            "source_path": "candidate.json",
            "safe_better_than_teacher_candidate": False,
        }
        if candidate_overrides:
            candidate.update(candidate_overrides)
        return [teacher, candidate]

    def _counterfactual_row(self, *, family: str, index: int) -> dict:
        key = f"{family}-train-{index}"
        return {
            "schema_version": "policy-differentiating-counterfactual-coverage-row/v1",
            "context_id": key,
            "episode_id": f"episode-{key}",
            "step_index": index,
            "scenario_family": family,
            "action_index": 1,
            "coverage_source_available": True,
            "expected_coverage_rate_delta": 0.25,
            "expected_new_coverage_area": 0.25,
            "information_gain": 0.12,
            "valuable_coverage_proxy": 0.13,
            "match_method": "counterfactual_candidate_expanded_cells_sidecar",
        }

    def _write_json(self, path: Path, payload: dict) -> None:
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    def _write_jsonl(self, path: Path, rows: list[dict]) -> None:
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )

    def _read_jsonl(self, path: Path) -> list[dict]:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
