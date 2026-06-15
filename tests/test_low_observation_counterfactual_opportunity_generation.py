import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path


class LowObservationCounterfactualOpportunityGenerationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        scripts_path = str(self.repo_root / "scripts")
        if scripts_path not in sys.path:
            sys.path.insert(0, scripts_path)
        self.temp_dir = Path(tempfile.mkdtemp(prefix="low-observation-opportunity-"))
        self.output_root = self.temp_dir / "output"
        self.config_path = self.temp_dir / "config.json"
        self.overlay_path = self.temp_dir / "candidate-level-coverage-overlay.jsonl"
        self.counterfactual_path = self.temp_dir / "counterfactual-coverage-rollouts.jsonl"
        self._write_json(
            self.config_path,
            {
                "schema_version": "quasi-real-low-observation-counterfactual-opportunity-config/v1",
                "target_family": "low_observation_count",
                "minimum_trainable_safe_better_pair_count": 8,
                "runs_new_ppo_update": False,
                "publishes_checkpoint": False,
                "replaces_default_policy": False,
            },
        )

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def test_exports_source_backed_train_low_observation_supplemental_pairs(self) -> None:
        from scripts.run_low_observation_counterfactual_opportunity_generation import (
            run_low_observation_counterfactual_opportunity_generation,
        )

        overlay_rows = []
        counterfactual_rows = []
        for index in range(8):
            overlay_rows.extend(
                self._decision_rows(family="low_observation_count", index=index, split="train")
            )
            counterfactual_rows.append(
                self._counterfactual_row(family="low_observation_count", index=index, split="train")
            )
        overlay_rows.extend(self._decision_rows(family="mixed_risk", index=0, split="train"))
        overlay_rows.extend(self._decision_rows(family="low_observation_count", index=99, split="test"))
        self._write_jsonl(self.overlay_path, overlay_rows)
        self._write_jsonl(self.counterfactual_path, counterfactual_rows)

        summary = run_low_observation_counterfactual_opportunity_generation(
            config_path=self.config_path,
            source_overlay_paths=[self.overlay_path],
            source_counterfactual_paths=[self.counterfactual_path],
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["reason_codes"], [])
        self.assertEqual(summary["target_family"], "low_observation_count")
        self.assertEqual(summary["low_observation_candidate_count"], 9)
        self.assertEqual(summary["low_observation_source_available_count"], 9)
        self.assertEqual(summary["low_observation_trainable_safe_better_pair_count"], 8)
        self.assertEqual(summary["low_observation_diagnostic_safe_better_pair_count"], 1)
        self.assertEqual(summary["missing_counterfactual_source_count"], 0)
        self.assertEqual(summary["fallback_gain_contamination_count"], 0)
        self.assertEqual(summary["controlled_regression_count"], 0)
        self.assertFalse(summary["runs_new_ppo_update"])
        self.assertFalse(summary["publishes_checkpoint"])
        self.assertFalse(summary["replaces_default_policy"])
        self.assertFalse(summary["performance_claimed"])

        supplemental_overlay = self._read_jsonl(
            self.output_root / "low-observation-supplemental-overlay.jsonl"
        )
        supplemental_counterfactual = self._read_jsonl(
            self.output_root / "low-observation-supplemental-counterfactual-rollouts.jsonl"
        )
        self.assertEqual(len(supplemental_overlay), 18)
        self.assertEqual(len(supplemental_counterfactual), 9)
        self.assertTrue(all(row["scenario_family"] == "low_observation_count" for row in supplemental_overlay))
        self.assertTrue(all(row["scenario_family"] == "low_observation_count" for row in supplemental_counterfactual))
        self.assertTrue((self.output_root / "low-observation-family-gap-report.json").is_file())
        self.assertTrue((self.output_root / "low-observation-counterfactual-opportunity-report.md").is_file())

    def test_fails_when_low_observation_has_no_positive_coverage_advantage(self) -> None:
        from scripts.run_low_observation_counterfactual_opportunity_generation import (
            run_low_observation_counterfactual_opportunity_generation,
        )

        rows = []
        for index in range(8):
            rows.extend(
                self._decision_rows(
                    family="low_observation_count",
                    index=index,
                    split="train",
                    candidate_overrides={"expected_coverage_rate_delta": 0.05},
                )
            )
        self._write_jsonl(self.overlay_path, rows)
        self._write_jsonl(self.counterfactual_path, [])

        summary = run_low_observation_counterfactual_opportunity_generation(
            config_path=self.config_path,
            source_overlay_paths=[self.overlay_path],
            source_counterfactual_paths=[self.counterfactual_path],
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertEqual(summary["low_observation_trainable_safe_better_pair_count"], 0)
        self.assertIn("low_observation_safe_better_pair_count_below_threshold", summary["reason_codes"])
        self.assertIn("supplemental_overlay_no_positive_coverage_advantage", summary["reason_codes"])
        self.assertFalse(summary["performance_claimed"])

    def test_rejects_source_missing_and_fallback_contamination(self) -> None:
        from scripts.run_low_observation_counterfactual_opportunity_generation import (
            run_low_observation_counterfactual_opportunity_generation,
        )

        rows = []
        rows.extend(
            self._decision_rows(
                family="low_observation_count",
                index=0,
                split="train",
                candidate_overrides={"coverage_source_available": False},
            )
        )
        rows.extend(
            self._decision_rows(
                family="low_observation_count",
                index=1,
                split="train",
                candidate_overrides={"fallback_like": True},
            )
        )
        self._write_jsonl(self.overlay_path, rows)
        self._write_jsonl(self.counterfactual_path, [])

        summary = run_low_observation_counterfactual_opportunity_generation(
            config_path=self.config_path,
            source_overlay_paths=[self.overlay_path],
            source_counterfactual_paths=[self.counterfactual_path],
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertEqual(summary["low_observation_trainable_safe_better_pair_count"], 0)
        self.assertEqual(summary["missing_counterfactual_source_count"], 1)
        self.assertEqual(summary["fallback_gain_contamination_count"], 1)
        self.assertIn("low_observation_counterfactual_source_missing", summary["reason_codes"])
        self.assertIn("fallback_gain_contamination", summary["reason_codes"])

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

    def _counterfactual_row(self, *, family: str, index: int, split: str) -> dict:
        key = f"{family}-{split}-{index}"
        return {
            "schema_version": "policy-differentiating-counterfactual-coverage-row/v1",
            "context_id": key,
            "episode_id": f"episode-{key}",
            "step_index": index,
            "scenario_family": family,
            "split": split,
            "action_index": 1,
            "coverage_source_available": True,
            "expected_coverage_rate_delta": 0.25,
            "expected_new_coverage_area": 0.25,
            "information_gain": 0.12,
            "valuable_coverage_proxy": 0.13,
            "match_method": "counterfactual_candidate_expanded_cells_sidecar",
            "source_path": "counterfactual.json",
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
