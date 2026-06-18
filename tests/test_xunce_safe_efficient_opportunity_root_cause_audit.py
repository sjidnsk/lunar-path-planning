import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path


class XunceSafeEfficientOpportunityRootCauseAuditTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        scripts_path = str(self.repo_root / "scripts")
        if scripts_path not in sys.path:
            sys.path.insert(0, scripts_path)
        self.temp_dir = Path(tempfile.mkdtemp(prefix="xunce-safe-efficient-root-cause-"))
        self.bound_root = self.temp_dir / "bound"
        self.refinement_root = self.temp_dir / "stage18f1"
        self.output_root = self.temp_dir / "audit"
        self.config_path = self.temp_dir / "config.json"
        self._write_bound_root()
        self._write_refinement_root()
        self._write_config()

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def test_audits_cost_risk_efficiency_root_causes(self) -> None:
        from scripts.run_xunce_safe_efficient_opportunity_root_cause_audit import (
            run_xunce_safe_efficient_opportunity_root_cause_audit,
        )

        summary = run_xunce_safe_efficient_opportunity_root_cause_audit(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["coverage_positive_candidate_count"], 9)
        self.assertGreater(summary["coverage_positive_but_cost_regressive_count"], 0)
        self.assertGreater(summary["coverage_positive_but_risk_regressive_count"], 0)
        self.assertGreater(summary["coverage_positive_but_efficiency_regressive_count"], 0)
        self.assertGreater(summary["candidate_generation_cost_bias_score"], 0.0)
        self.assertIn(summary["root_cause_route"], {"candidate_generation_cost_biased", "coverage_cost_tradeoff_too_strict"})
        self.assertNotEqual(summary["next_required_change"], "run_true_incumbent_selection_binding")
        self.assertFalse(summary["publishes_checkpoint"])
        self.assertFalse(summary["starts_online_canary"])
        self.assertTrue((self.output_root / "xunce-safe-efficient-opportunity-root-cause-audit-summary.json").is_file())
        self.assertTrue((self.output_root / "xunce-safe-efficient-opportunity-root-cause-scenarios.jsonl").is_file())
        self.assertTrue((self.output_root / "xunce-safe-efficient-opportunity-root-cause-roi-summary.json").is_file())

    def test_binding_missing_routes_to_binding_stage(self) -> None:
        from scripts.run_xunce_safe_efficient_opportunity_root_cause_audit import (
            run_xunce_safe_efficient_opportunity_root_cause_audit,
        )

        self._write_bound_root(fallback=True)

        summary = run_xunce_safe_efficient_opportunity_root_cause_audit(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("true_incumbent_binding_missing", summary["reason_codes"])
        self.assertEqual(summary["next_required_change"], "run_true_incumbent_selection_binding")

    def test_existing_safe_candidate_routes_to_repair_stage(self) -> None:
        from scripts.run_xunce_safe_efficient_opportunity_root_cause_audit import (
            run_xunce_safe_efficient_opportunity_root_cause_audit,
        )

        self._write_bound_root(with_safe=True)

        summary = run_xunce_safe_efficient_opportunity_root_cause_audit(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertGreater(summary["safe_efficient_opportunity_count"], 0)
        self.assertEqual(summary["root_cause_route"], "ready_for_safe_efficient_candidate_repair")
        self.assertEqual(summary["next_required_change"], "run_safe_efficient_candidate_repair")

    def _write_config(self) -> None:
        payload = {
            "schema_version": "xunce-safe-efficient-opportunity-root-cause-audit-config/v1",
            "source_bound_coverage_root": str(self.bound_root),
            "source_cost_efficient_refinement_root": str(self.refinement_root),
            "required_scenario_count": 3,
            "min_roi_group_count": 3,
            "canary_traffic_fraction": 0.0,
        }
        self.config_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _write_bound_root(self, *, fallback: bool = False, with_safe: bool = False) -> None:
        self.bound_root.mkdir(parents=True, exist_ok=True)
        scenarios = []
        for index in range(3):
            if with_safe:
                candidates = [
                    {"action_index": 0, "cell": [index, 0], "reachable": True, "path_cost": 10.0, "risk": 0.10, "roi_weighted_coverage_delta": 0.02, "path_line_coverage_delta": 0.02, "revisit_penalty": 0.0},
                    {"action_index": 1, "cell": [index, 1], "reachable": True, "path_cost": 8.0, "risk": 0.09, "roi_weighted_coverage_delta": 0.06, "path_line_coverage_delta": 0.06, "revisit_penalty": 0.0},
                    {"action_index": 2, "cell": [index, 2], "reachable": True, "path_cost": 30.0, "risk": 0.30, "roi_weighted_coverage_delta": 0.14, "path_line_coverage_delta": 0.14, "revisit_penalty": 0.0},
                ]
            else:
                candidates = [
                    {"action_index": 0, "cell": [index, 0], "reachable": True, "path_cost": 10.0, "risk": 0.10, "roi_weighted_coverage_delta": 0.02, "path_line_coverage_delta": 0.02, "revisit_penalty": 0.0},
                    {"action_index": 1, "cell": [index, 1], "reachable": True, "path_cost": 30.0, "risk": 0.50, "roi_weighted_coverage_delta": 0.03, "path_line_coverage_delta": 0.03, "revisit_penalty": 0.01},
                    {"action_index": 2, "cell": [index, 2], "reachable": True, "path_cost": 30.0, "risk": 0.30, "roi_weighted_coverage_delta": 0.14, "path_line_coverage_delta": 0.14, "revisit_penalty": 0.02},
                    {"action_index": 3, "cell": [index, 3], "reachable": True, "path_cost": 22.0, "risk": 0.50, "roi_weighted_coverage_delta": 0.08, "path_line_coverage_delta": 0.08, "revisit_penalty": 0.0},
                ]
            scenarios.append(
                {
                    "scenario_id": f"scenario_{index:03d}",
                    "roi_group": f"roi_{index}",
                    "incumbent_selected_action_index": 0,
                    "incumbent_selection_source": "fallback_action_index_0" if fallback else "true_checkpoint_inference",
                    "path_feedback": {"candidates": candidates},
                }
            )
        self._write_json(self.bound_root / "xunce-true-incumbent-selection-binding-summary.json", {"schema_version": "xunce-true-incumbent-selection-binding-summary/v1", "status": "passed", "true_incumbent_selection_bound": not fallback})
        self._write_json(self.bound_root / "xunce-high-fidelity-path-feedback-audit.json", {"schema_version": "xunce-high-fidelity-path-feedback-audit/v1", "scenario_count": 3, "candidate_count": sum(len(s["path_feedback"]["candidates"]) for s in scenarios), "scenarios": scenarios})

    def _write_refinement_root(self) -> None:
        self.refinement_root.mkdir(parents=True, exist_ok=True)
        self._write_json(self.refinement_root / "xunce-cost-efficient-coverage-opportunity-summary.json", {"schema_version": "xunce-cost-efficient-coverage-opportunity-summary/v1", "status": "failed", "safe_efficient_opportunity_count": 0})

    @staticmethod
    def _write_json(path: Path, payload: dict) -> None:
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
