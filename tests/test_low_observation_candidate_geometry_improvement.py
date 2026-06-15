import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path


class LowObservationCandidateGeometryImprovementTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        scripts_path = str(self.repo_root / "scripts")
        if scripts_path not in sys.path:
            sys.path.insert(0, scripts_path)
        self.temp_dir = Path(tempfile.mkdtemp(prefix="low-observation-geometry-"))
        self.config_path = self.temp_dir / "config.json"
        self.path_feedback_summary = self.temp_dir / "quasi-real-map-path-feedback-summary.json"
        self.output_root = self.temp_dir / "geometry-output"
        self.tuning_root = self.temp_dir / "tuning"
        self.stage5a2_root = self.temp_dir / "stage5a2"
        self.quasi_real_root = self.temp_dir / "quasi-real"
        self.formal_root = self.temp_dir / "formal"
        self.replay_root = self.temp_dir / "replay"
        self.signal_root = self.temp_dir / "signal"
        self.reward_root = self.temp_dir / "reward"
        self.safe_better_output_root = self.temp_dir / "safe-better-output"
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
        self._write_config()

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def test_materializes_low_observation_geometry_and_closes_safe_better_family_gate(self) -> None:
        from scripts.run_low_observation_candidate_geometry_improvement import (
            run_low_observation_candidate_geometry_improvement,
        )

        self._write_base_stage5b3_inputs()
        self._write_path_feedback_summary(
            [
                self._scenario(index=index, split="train", teacher_coverage=0.01, candidate_coverage=0.06)
                for index in range(8)
            ]
        )

        summary = run_low_observation_candidate_geometry_improvement(
            config_path=self.config_path,
            path_feedback_summary_paths=[self.path_feedback_summary],
            output_root=self.output_root,
            repo_root=self.repo_root,
            tuning_root=self.tuning_root,
            stage5a2_root=self.stage5a2_root,
            quasi_real_root=self.quasi_real_root,
            formal_training_root=self.formal_root,
            post_training_replay_root=self.replay_root,
            coverage_signal_root=self.signal_root,
            reward_refinement_root=self.reward_root,
            safe_better_output_root=self.safe_better_output_root,
            execute_path_feedback=False,
        )

        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["reason_codes"], [])
        self.assertEqual(summary["low_observation_trainable_safe_better_pair_count"], 8)
        self.assertEqual(summary["low_observation_diagnostic_safe_better_pair_count"], 0)
        self.assertEqual(summary["safe_better_than_teacher_family_count"], 4)
        self.assertEqual(summary["family_safe_better_counts"]["low_observation_count"], 8)
        self.assertEqual(summary["missing_counterfactual_source_count"], 0)
        self.assertEqual(summary["fallback_gain_contamination_count"], 0)
        self.assertEqual(summary["controlled_regression_count"], 0)
        self.assertFalse(summary["runs_new_ppo_update"])
        self.assertFalse(summary["publishes_checkpoint"])
        self.assertFalse(summary["replaces_default_policy"])
        self.assertFalse(summary["performance_claimed"])
        self.assertTrue((self.output_root / "low-observation-candidate-geometry-overlay.jsonl").is_file())
        self.assertTrue(
            (self.output_root / "low-observation-candidate-geometry-counterfactual-rollouts.jsonl").is_file()
        )
        self.assertTrue((self.output_root / "low-observation-candidate-geometry-report.md").is_file())

    def test_frontier_ranking_keeps_best_coverage_candidate_within_top_k(self) -> None:
        from scripts.run_low_observation_candidate_geometry_improvement import (
            run_low_observation_candidate_geometry_improvement,
        )

        self._write_base_stage5b3_inputs()
        self._write_config(top_k=1)
        self._write_path_feedback_summary(
            [
                self._scenario(
                    index=0,
                    split="train",
                    teacher_coverage=0.01,
                    candidate_coverage=0.02,
                    extra_candidates=[
                        self._candidate(action_index=2, cell=[25, 25], coverage=0.08, information_gain=0.9)
                    ],
                )
            ]
        )

        summary = run_low_observation_candidate_geometry_improvement(
            config_path=self.config_path,
            path_feedback_summary_paths=[self.path_feedback_summary],
            output_root=self.output_root,
            repo_root=self.repo_root,
            tuning_root=self.tuning_root,
            stage5a2_root=self.stage5a2_root,
            quasi_real_root=self.quasi_real_root,
            formal_training_root=self.formal_root,
            post_training_replay_root=self.replay_root,
            coverage_signal_root=self.signal_root,
            reward_refinement_root=self.reward_root,
            safe_better_output_root=self.safe_better_output_root,
            execute_path_feedback=False,
        )

        overlay = self._read_jsonl(self.output_root / "low-observation-candidate-geometry-overlay.jsonl")
        non_teacher_actions = [row["action_index"] for row in overlay if not row["is_teacher_action"]]
        self.assertEqual(non_teacher_actions, [2])
        self.assertEqual(summary["geometry_ranked_candidate_count"], 1)

    def test_teacher_uses_post_feedback_selected_cell_when_before_and_after_differ(self) -> None:
        from scripts.run_low_observation_candidate_geometry_improvement import (
            run_low_observation_candidate_geometry_improvement,
        )

        self._write_base_stage5b3_inputs()
        scenario = self._scenario(index=0, split="train", teacher_coverage=0.08, candidate_coverage=0.02)
        scenario["selected_cell_before_path_feedback"] = [20, 20]
        scenario["selected_cell_after_path_feedback"] = [24, 24]
        scenario["path_feedback"]["candidates"].append(
            self._candidate(action_index=2, cell=[26, 26], coverage=0.07)
        )
        self._write_path_feedback_summary([scenario])

        summary = run_low_observation_candidate_geometry_improvement(
            config_path=self.config_path,
            path_feedback_summary_paths=[self.path_feedback_summary],
            output_root=self.output_root,
            repo_root=self.repo_root,
            tuning_root=self.tuning_root,
            stage5a2_root=self.stage5a2_root,
            quasi_real_root=self.quasi_real_root,
            formal_training_root=self.formal_root,
            post_training_replay_root=self.replay_root,
            coverage_signal_root=self.signal_root,
            reward_refinement_root=self.reward_root,
            safe_better_output_root=self.safe_better_output_root,
            execute_path_feedback=False,
        )

        overlay = self._read_jsonl(self.output_root / "low-observation-candidate-geometry-overlay.jsonl")
        teacher = next(row for row in overlay if row["is_teacher_action"])
        self.assertEqual(teacher["action_index"], 1)
        self.assertEqual(summary["low_observation_trainable_safe_better_pair_count"], 2)

    def test_source_missing_fails_without_placeholder_coverage(self) -> None:
        from scripts.run_low_observation_candidate_geometry_improvement import (
            run_low_observation_candidate_geometry_improvement,
        )

        self._write_base_stage5b3_inputs()
        self._write_path_feedback_summary(
            [
                self._scenario(
                    index=0,
                    split="train",
                    teacher_coverage=0.01,
                    candidate_coverage=None,
                )
            ]
        )

        summary = run_low_observation_candidate_geometry_improvement(
            config_path=self.config_path,
            path_feedback_summary_paths=[self.path_feedback_summary],
            output_root=self.output_root,
            repo_root=self.repo_root,
            tuning_root=self.tuning_root,
            stage5a2_root=self.stage5a2_root,
            quasi_real_root=self.quasi_real_root,
            formal_training_root=self.formal_root,
            post_training_replay_root=self.replay_root,
            coverage_signal_root=self.signal_root,
            reward_refinement_root=self.reward_root,
            safe_better_output_root=self.safe_better_output_root,
            execute_path_feedback=False,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertEqual(summary["missing_counterfactual_source_count"], 1)
        self.assertIn("low_observation_counterfactual_source_missing", summary["reason_codes"])
        self.assertFalse(summary["performance_claimed"])

    def test_materialization_enriches_missing_candidate_coverage_from_contract_goals(self) -> None:
        from scripts.run_low_observation_candidate_geometry_improvement import (
            run_low_observation_candidate_geometry_improvement,
        )

        self._write_base_stage5b3_inputs()
        scenario = self._scenario(
            index=0,
            split="train",
            teacher_coverage=None,
            candidate_coverage=None,
        )
        self._write_path_feedback_summary([scenario])
        self._write_contract_goals(
            scenario_id=scenario["scenario_id"],
            goals=[
                {"cell": [20, 20], "expected_coverage_rate_delta": 0.01},
                {"cell": [24, 24], "expected_coverage_rate_delta": 0.08},
            ],
        )

        summary = run_low_observation_candidate_geometry_improvement(
            config_path=self.config_path,
            path_feedback_summary_paths=[self.path_feedback_summary],
            output_root=self.output_root,
            repo_root=self.repo_root,
            tuning_root=self.tuning_root,
            stage5a2_root=self.stage5a2_root,
            quasi_real_root=self.quasi_real_root,
            formal_training_root=self.formal_root,
            post_training_replay_root=self.replay_root,
            coverage_signal_root=self.signal_root,
            reward_refinement_root=self.reward_root,
            safe_better_output_root=self.safe_better_output_root,
            execute_path_feedback=False,
        )

        overlay = self._read_jsonl(self.output_root / "low-observation-candidate-geometry-overlay.jsonl")
        candidate = next(row for row in overlay if not row["is_teacher_action"])
        self.assertTrue(candidate["coverage_source_available"])
        self.assertEqual(candidate["expected_coverage_rate_delta"], 0.08)
        self.assertEqual(summary["missing_counterfactual_source_count"], 0)

    def test_fallback_contamination_fails_even_when_coverage_is_positive(self) -> None:
        from scripts.run_low_observation_candidate_geometry_improvement import (
            run_low_observation_candidate_geometry_improvement,
        )

        self._write_base_stage5b3_inputs()
        self._write_path_feedback_summary(
            [
                self._scenario(
                    index=0,
                    split="train",
                    teacher_coverage=0.01,
                    candidate_coverage=0.08,
                    candidate_overrides={"open_grid_fallback_used": True},
                )
            ]
        )

        summary = run_low_observation_candidate_geometry_improvement(
            config_path=self.config_path,
            path_feedback_summary_paths=[self.path_feedback_summary],
            output_root=self.output_root,
            repo_root=self.repo_root,
            tuning_root=self.tuning_root,
            stage5a2_root=self.stage5a2_root,
            quasi_real_root=self.quasi_real_root,
            formal_training_root=self.formal_root,
            post_training_replay_root=self.replay_root,
            coverage_signal_root=self.signal_root,
            reward_refinement_root=self.reward_root,
            safe_better_output_root=self.safe_better_output_root,
            execute_path_feedback=False,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertEqual(summary["fallback_gain_contamination_count"], 1)
        self.assertIn("fallback_gain_contamination", summary["reason_codes"])

    def test_validation_and_test_low_observation_pairs_are_diagnostic_only(self) -> None:
        from scripts.run_low_observation_candidate_geometry_improvement import (
            run_low_observation_candidate_geometry_improvement,
        )

        self._write_base_stage5b3_inputs()
        self._write_path_feedback_summary(
            [
                self._scenario(index=0, split="validation", teacher_coverage=0.01, candidate_coverage=0.08),
                self._scenario(index=1, split="test", teacher_coverage=0.01, candidate_coverage=0.08),
            ]
        )

        summary = run_low_observation_candidate_geometry_improvement(
            config_path=self.config_path,
            path_feedback_summary_paths=[self.path_feedback_summary],
            output_root=self.output_root,
            repo_root=self.repo_root,
            tuning_root=self.tuning_root,
            stage5a2_root=self.stage5a2_root,
            quasi_real_root=self.quasi_real_root,
            formal_training_root=self.formal_root,
            post_training_replay_root=self.replay_root,
            coverage_signal_root=self.signal_root,
            reward_refinement_root=self.reward_root,
            safe_better_output_root=self.safe_better_output_root,
            execute_path_feedback=False,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertEqual(summary["low_observation_trainable_safe_better_pair_count"], 0)
        self.assertEqual(summary["low_observation_diagnostic_safe_better_pair_count"], 2)
        self.assertIn("low_observation_geometry_no_positive_coverage_advantage", summary["reason_codes"])

    def test_generated_matrix_keeps_required_roi_families_for_model_explorer_validation(self) -> None:
        from scripts.run_low_observation_candidate_geometry_improvement import (
            _low_observation_matrix_payload,
        )

        config = json.loads(self.config_path.read_text(encoding="utf-8"))

        payload = _low_observation_matrix_payload(
            config=config,
            repo_root=self.repo_root,
            target_family="low_observation_count",
        )

        names = {row["name"] for row in payload["rois"]}
        self.assertEqual(
            names,
            {"low_observation_count", "mixed_risk", "rim_or_steep_slope", "smooth_high_confidence"},
        )
        low_observation_rows = [row for row in payload["rois"] if row["name"] == "low_observation_count"]
        smooth_rows = [row for row in payload["rois"] if row["name"] == "smooth_high_confidence"]
        self.assertGreater(len(low_observation_rows), len(smooth_rows))
        self.assertTrue(all(row["candidate_count"] == 16 for row in low_observation_rows))

    def _write_config(self, *, top_k: int = 8) -> None:
        self._write_json(
            self.config_path,
            {
                "schema_version": "quasi-real-low-observation-candidate-geometry-config/v1",
                "target_family": "low_observation_count",
                "candidate_count": 16,
                "top_k": top_k,
                "minimum_trainable_safe_better_pair_count": 8,
                "source_matrix_manifest": "model-explorer/data/manifests/lunar_south_pole_lro_lola_selection_matrix_v1.json",
                "start_cells": [[0, 0], [16, 16], [31, 31]],
                "roi_offsets": [[0, 0], [4, 0], [0, 4]],
            },
        )

    def _write_base_stage5b3_inputs(self) -> None:
        overlay_rows = []
        counterfactual_rows = []
        for family in ("mixed_risk", "rim_or_steep_slope", "smooth_high_confidence"):
            for index in range(40):
                overlay_rows.extend(self._decision_rows(family=family, index=index, split="train"))
                counterfactual_rows.append(self._counterfactual_row(family=family, index=index))
        self._write_stage5b3_inputs(overlay_rows, counterfactual_rows)

    def _write_stage5b3_inputs(self, overlay_rows: list[dict], counterfactual_rows: list[dict]) -> None:
        overlay_path = self.stage5a2_root / "candidate-level-coverage-overlay.jsonl"
        counterfactual_path = self.stage5a2_root / "counterfactual-coverage-rollouts.jsonl"
        self._write_jsonl(overlay_path, overlay_rows)
        self._write_jsonl(counterfactual_path, counterfactual_rows)
        self._write_json(
            self.tuning_root / "refined-coverage-reward-margin-tuning-summary.json",
            {"schema_version": "refined-coverage-reward-margin-tuning-summary/v1", "status": "failed"},
        )
        self._write_json(
            self.stage5a2_root / "policy-differentiating-counterfactual-coverage-rollouts-summary.json",
            {
                "schema_version": "policy-differentiating-counterfactual-coverage-rollouts-summary/v1",
                "status": "passed",
                "candidate_coverage_overlay": str(overlay_path),
                "counterfactual_coverage_rollouts": str(counterfactual_path),
                "safe_better_than_teacher_candidate_count": 120,
                "safe_better_than_teacher_family_count": 3,
                "performance_claimed": False,
            },
        )
        self._write_json(
            self.stage5a2_root / "family-action-gap-report.json",
            {"schema_version": "policy-differentiating-counterfactual-family-action-gap-report/v1", "rows": []},
        )
        self._write_json(self.quasi_real_root / "quasi-real-map-path-feedback-summary.json", {"status": "passed"})
        self._write_json(
            self.quasi_real_root / "quasi-real-safe-alternative-opportunity-summary.json",
            {"status": "passed", "performance_claimed": False},
        )
        self._write_json(
            self.formal_root / "formal-ppo-training-run-summary.json",
            {"schema_version": "formal-ppo-training-run-summary/v1", "status": "passed", "performance_claimed": False},
        )
        self._write_json(
            self.replay_root / "formal-ppo-post-training-stability-replay-summary.json",
            {"schema_version": "formal-ppo-post-training-stability-replay-summary/v1", "status": "passed"},
        )
        self._write_json(
            self.signal_root / "exploration-coverage-signal-audit-summary.json",
            {"schema_version": "exploration-coverage-signal-audit-summary/v1", "status": "passed"},
        )
        self._write_json(
            self.reward_root / "coverage-aware-reward-refinement-summary.json",
            {"schema_version": "coverage-aware-reward-refinement-summary/v1", "status": "passed"},
        )

    def _write_path_feedback_summary(self, scenarios: list[dict]) -> None:
        self._write_json(
            self.path_feedback_summary,
            {
                "schema_version": "path-feedback-summary/v1",
                "scenario_count": len(scenarios),
                "scenario_set": "low_observation_geometry_test",
                "top_k": 8,
                "scenarios": scenarios,
            },
        )

    def _write_contract_goals(self, *, scenario_id: str, goals: list[dict]) -> None:
        contract_dir = self.path_feedback_summary.parent / "path_planner_sidecars"
        contract_dir.mkdir(parents=True, exist_ok=True)
        normalized_goals = []
        for goal in goals:
            normalized = dict(goal)
            normalized.setdefault("expected_new_coverage_area", normalized.get("expected_coverage_rate_delta"))
            normalized.setdefault("information_gain", normalized.get("expected_coverage_rate_delta"))
            normalized.setdefault("valuable_coverage_proxy", normalized.get("expected_coverage_rate_delta"))
            normalized.setdefault("value", normalized.get("expected_coverage_rate_delta"))
            normalized_goals.append(normalized)
        self._write_json(
            contract_dir / f"{scenario_id}.contract.json",
            {"schema_version": "path-feedback-contract/v1", "top_goals": normalized_goals},
        )

    def _scenario(
        self,
        *,
        index: int,
        split: str,
        teacher_coverage: float,
        candidate_coverage: float | None,
        candidate_overrides: dict | None = None,
        extra_candidates: list[dict] | None = None,
    ) -> dict:
        scenario_id = f"lola_qreal_low_observation_count_{split}_{index:03d}"
        teacher = self._candidate(action_index=0, cell=[20, 20], coverage=teacher_coverage)
        candidate = self._candidate(action_index=1, cell=[24, 24], coverage=candidate_coverage)
        if candidate_overrides:
            candidate.update(candidate_overrides)
        candidates = [teacher, candidate]
        candidates.extend(extra_candidates or [])
        return {
            "scenario_id": scenario_id,
            "scenario_group": "low_observation_count",
            "split": split,
            "scenario_seed": 20260615 + index,
            "scenario_variant_id": f"{scenario_id}-variant",
            "selected_cell_after_path_feedback": [20, 20],
            "path_feedback": {"candidate_count": len(candidates), "candidates": candidates},
        }

    def _candidate(
        self,
        *,
        action_index: int,
        cell: list[int],
        coverage: float | None,
        information_gain: float | None = None,
    ) -> dict:
        row = {
            "action_index": action_index,
            "cell": cell,
            "candidate_role": "policy_target",
            "reachable": True,
            "path_cost": 10.0 + action_index,
            "risk": 0.1,
            "utility": 0.5,
            "information_gain": information_gain if information_gain is not None else coverage,
            "expected_new_coverage_area": coverage,
            "valuable_coverage_proxy": coverage,
            "value": coverage,
        }
        if coverage is not None:
            row["expected_coverage_rate_delta"] = coverage
        return row

    def _decision_rows(self, *, family: str, index: int, split: str) -> list[dict]:
        key = f"{family}-{split}-{index}"
        teacher = {
            "schema_version": "candidate-level-coverage-overlay-row/v1",
            "context_id": key,
            "episode_id": f"episode-{key}",
            "step_index": index,
            "scenario_family": family,
            "scenario_id": key,
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
        }
        return [teacher, candidate]

    def _counterfactual_row(self, *, family: str, index: int) -> dict:
        key = f"{family}-train-{index}"
        return {
            "schema_version": "policy-differentiating-counterfactual-coverage-row/v1",
            "context_id": key,
            "episode_id": f"episode-{key}",
            "step_index": index,
            "scenario_family": family,
            "split": "train",
            "action_index": 1,
            "coverage_source_available": True,
            "expected_coverage_rate_delta": 0.25,
            "match_method": "counterfactual_candidate_expanded_cells_sidecar",
            "source_path": "counterfactual.json",
        }

    def _write_json(self, path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    def _write_jsonl(self, path: Path, rows: list[dict]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")

    def _read_jsonl(self, path: Path) -> list[dict]:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
