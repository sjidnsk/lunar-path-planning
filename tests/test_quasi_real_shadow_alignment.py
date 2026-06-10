from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


class QuasiRealShadowAlignmentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp(prefix="qreal-shadow-align-"))
        self.repo_root = Path(__file__).resolve().parents[1]
        scripts_dir = str(self.repo_root / "scripts")
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)
        self.taxonomy_root = self.temp_dir / "taxonomy"
        self.dataset_root = self.temp_dir / "dataset"
        self.preference_root = self.temp_dir / "preference"
        self.taxonomy_root.mkdir()
        self._write_taxonomy_artifacts()

    def test_dataset_generation_creates_disjoint_train_val_holdout_variants(self) -> None:
        from scripts.run_quasi_real_shadow_alignment_dataset import (
            run_quasi_real_shadow_alignment_dataset,
        )

        summary = run_quasi_real_shadow_alignment_dataset(
            taxonomy_root=self.taxonomy_root,
            output_root=self.dataset_root,
            config=self._dataset_config(),
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "passed")
        self.assertGreater(summary["train_slice_count"], 0)
        self.assertGreater(summary["val_slice_count"], 0)
        self.assertGreater(summary["holdout_slice_count"], 0)
        self.assertEqual(summary["context_id_overlap_count"], 0)
        self.assertEqual(summary["scenario_id_overlap_count"], 0)
        self.assertEqual(summary["slice_id_overlap_count"], 0)
        slices = self._read_jsonl(self.dataset_root / "quasi-real-shadow-alignment-slices.jsonl")
        self.assertEqual({record["split"] for record in slices}, {"train", "val", "holdout"})
        self.assertTrue(all(record["source_failure_scenario_id"] == "lola_qreal_mixed_risk_test_011" for record in slices))

    def test_preference_mining_uses_train_split_only_and_never_adds_hard_positive(self) -> None:
        from scripts.run_quasi_real_shadow_alignment_dataset import (
            run_quasi_real_shadow_alignment_dataset,
        )
        from scripts.run_quasi_real_shadow_alignment_preference_mining import (
            run_quasi_real_shadow_alignment_preference_mining,
        )

        run_quasi_real_shadow_alignment_dataset(
            taxonomy_root=self.taxonomy_root,
            output_root=self.dataset_root,
            config=self._dataset_config(),
            repo_root=self.repo_root,
        )
        summary = run_quasi_real_shadow_alignment_preference_mining(
            taxonomy_root=self.taxonomy_root,
            dataset_root=self.dataset_root,
            output_root=self.preference_root,
            config=self._preference_config(),
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "passed")
        self.assertGreaterEqual(summary["quasi_real_hard_negative_preference_count"], 1)
        self.assertEqual(summary["hard_positive_added_count"], 0)
        self.assertEqual(summary["ppo_transition_added_count"], 0)
        self.assertEqual(summary["holdout_context_leakage_count"], 0)
        samples = self._read_jsonl(
            self.preference_root / "quasi-real-shadow-alignment-preference-samples.jsonl"
        )
        self.assertEqual({sample["split"] for sample in samples}, {"train"})
        self.assertEqual(samples[0]["sample_type"], "raw_policy_regression_preference_pair")
        self.assertEqual(samples[0]["quasi_real_sample_type"], "path_risk_joint_regression")
        self.assertEqual(samples[0]["preferred"]["context_id"], "ctx-source")
        self.assertEqual(samples[0]["alternative"]["context_id"], "ctx-alt")

    def test_readiness_accepts_passed_alignment_summary_and_blocks_holdout_regression(self) -> None:
        from scripts.run_policy_training_readiness_review import (
            _quasi_real_shadow_alignment_readiness,
        )

        passed = _quasi_real_shadow_alignment_readiness(
            {
                "schema_version": "quasi-real-shadow-alignment-summary/v1",
                "status": "passed",
                "reason_codes": [],
                "alignment_verdict": "acceptable_for_quasi_real_shadow_audit",
                "taxonomy_failure_count": 1,
                "quasi_real_hard_negative_preference_count": 1,
                "hard_positive_added_count": 0,
                "ppo_transition_added_count": 0,
                "context_id_overlap_count": 0,
                "scenario_id_overlap_count": 0,
                "slice_id_overlap_count": 0,
                "holdout_policy_changed_gate_rejected_count": 0,
                "holdout_path_cost_regression_count": 0,
                "holdout_risk_regression_count": 0,
                "holdout_source_selection_regression_count": 0,
                "original_roi_regression_count": 0,
                "over_conservative_policy_detected": False,
                "publishes_checkpoint": False,
                "replaces_default_policy": False,
                "performance_claimed": False,
            }
        )
        self.assertTrue(passed["completed"])

        failed = _quasi_real_shadow_alignment_readiness(
            {
                "schema_version": "quasi-real-shadow-alignment-summary/v1",
                "status": "failed",
                "reason_codes": ["quasi_real_shadow_holdout_regression"],
                "alignment_verdict": "holdout_regression",
                "taxonomy_failure_count": 1,
                "quasi_real_hard_negative_preference_count": 1,
                "hard_positive_added_count": 0,
                "ppo_transition_added_count": 0,
                "context_id_overlap_count": 0,
                "scenario_id_overlap_count": 0,
                "slice_id_overlap_count": 0,
                "holdout_policy_changed_gate_rejected_count": 1,
                "holdout_path_cost_regression_count": 1,
                "holdout_risk_regression_count": 1,
                "holdout_source_selection_regression_count": 0,
                "original_roi_regression_count": 0,
                "over_conservative_policy_detected": False,
                "publishes_checkpoint": False,
                "replaces_default_policy": False,
                "performance_claimed": False,
                "next_required_change": "quasi_real_shadow_objective_weight_refinement_required",
            }
        )
        self.assertFalse(failed["completed"])
        self.assertIn("quasi_real_shadow_holdout_regression", failed["training_blockers"])

    def _write_taxonomy_artifacts(self) -> None:
        record = {
            "schema_version": "quasi-real-shadow-failure-taxonomy-record/v1",
            "failure_class": "path_risk_joint_regression",
            "scenario_id": "lola_qreal_mixed_risk_test_011",
            "roi_group": "mixed_risk",
            "roi_name": "mixed_risk",
            "split": "test",
            "map_id": "lola",
            "slice_id": "slice-011",
            "context_id": "ctx-alt",
            "source_action_index": 1,
            "raw_policy_action_index": 2,
            "logit_margin": 0.269,
            "path_cost_delta": 1.9,
            "risk_delta": 0.014,
            "gate_reason_codes": ["path_cost_regression", "risk_regression"],
            "source_candidate": {"context_id": "ctx-source", "path_cost": 10.0, "risk": 0.10},
            "alternative_candidate": {"context_id": "ctx-alt", "path_cost": 11.9, "risk": 0.114},
        }
        (self.taxonomy_root / "quasi-real-shadow-failure-taxonomy.jsonl").write_text(
            json.dumps(record, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        (self.taxonomy_root / "quasi-real-shadow-failure-taxonomy-summary.json").write_text(
            json.dumps(
                {
                    "schema_version": "quasi-real-shadow-failure-taxonomy-summary/v1",
                    "status": "passed",
                    "failure_count": 1,
                    "path_risk_joint_regression_count": 1,
                    "bridge_or_feedback_gap_count": 0,
                    "action_mask_or_contract_gap_count": 0,
                }
            ),
            encoding="utf-8",
        )

    def _dataset_config(self) -> dict[str, object]:
        return {
            "input_files": {
                "taxonomy_summary": "quasi-real-shadow-failure-taxonomy-summary.json",
                "taxonomy": "quasi-real-shadow-failure-taxonomy.jsonl",
            },
            "output_files": {
                "slices": "quasi-real-shadow-alignment-slices.jsonl",
                "path_feedback_summary": "quasi-real-shadow-alignment-path-feedback-summary.json",
                "split_summary": "quasi-real-shadow-alignment-split-summary.json",
            },
            "variants": {"train": 2, "val": 2, "holdout": 2},
        }

    def _preference_config(self) -> dict[str, object]:
        return {
            "input_files": {
                "taxonomy_summary": "quasi-real-shadow-failure-taxonomy-summary.json",
                "taxonomy": "quasi-real-shadow-failure-taxonomy.jsonl",
                "alignment_slices": "quasi-real-shadow-alignment-slices.jsonl",
                "alignment_split_summary": "quasi-real-shadow-alignment-split-summary.json",
            },
            "output_files": {
                "samples": "quasi-real-shadow-alignment-preference-samples.jsonl",
                "summary": "quasi-real-shadow-alignment-preference-summary.json",
                "exclusion_report": "quasi-real-shadow-alignment-exclusion-report.json",
            },
            "validation": {
                "min_quasi_real_hard_negative_preference_count": 1,
                "max_hard_positive_added_count": 0,
                "max_ppo_transition_added_count": 0,
            },
        }

    def _read_jsonl(self, path: Path) -> list[dict[str, object]]:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


if __name__ == "__main__":
    unittest.main()
