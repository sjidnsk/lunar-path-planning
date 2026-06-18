import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path


class XunceSafeEfficientCandidateRepairTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        scripts_path = str(self.repo_root / "scripts")
        if scripts_path not in sys.path:
            sys.path.insert(0, scripts_path)
        self.temp_dir = Path(tempfile.mkdtemp(prefix="xunce-safe-efficient-repair-"))
        self.bound_root = self.temp_dir / "bound"
        self.audit_root = self.temp_dir / "audit"
        self.output_root = self.temp_dir / "repair"
        self.refinement_output_root = self.temp_dir / "stage18f1"
        self.config_path = self.temp_dir / "config.json"
        self.refinement_config_path = self.temp_dir / "stage18f1-config.json"
        self._write_bound_root()
        self._write_audit_root()
        self._write_config()

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def test_repairs_validated_safe_candidates_and_preserves_bound_root(self) -> None:
        from scripts.run_xunce_safe_efficient_candidate_repair import run_xunce_safe_efficient_candidate_repair

        original_payload = json.loads((self.bound_root / "xunce-high-fidelity-path-feedback-audit.json").read_text(encoding="utf-8"))

        summary = run_xunce_safe_efficient_candidate_repair(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "passed")
        self.assertGreater(summary["safe_efficient_opportunity_count"], 0)
        self.assertGreater(summary["safe_efficient_opportunity_scenario_count"], 0)
        self.assertGreaterEqual(summary["roi_group_with_safe_efficient_opportunity_count"], 3)
        self.assertGreater(summary["cost_efficient_coverage_spread_range"], 0.005)
        self.assertGreater(summary["candidate_pareto_frontier_count"], 1)
        self.assertEqual(summary["fallback_action_index_0_count"], 0)
        self.assertEqual(summary["proposal_unvalidated_positive_count"], 0)
        self.assertEqual(summary["open_grid_fallback_count"], 0)
        self.assertFalse(summary["publishes_checkpoint"])
        self.assertFalse(summary["starts_online_canary"])

        repaired = json.loads((self.output_root / "xunce-high-fidelity-path-feedback-audit.json").read_text(encoding="utf-8"))
        candidates = repaired["scenarios"][0]["path_feedback"]["candidates"]
        safe_candidates = [candidate for candidate in candidates if candidate.get("safe_efficient_opportunity") is True]
        self.assertTrue(safe_candidates)
        self.assertTrue(all(candidate["proposal_validated_by_path_feedback"] for candidate in safe_candidates))
        self.assertTrue(all(candidate["candidate_repair_source"] == "cost_guarded_frontier_interpolation/v1" for candidate in safe_candidates))
        proposal_rows = [candidate for candidate in candidates if candidate.get("proposal_only") is True]
        self.assertTrue(proposal_rows)
        self.assertTrue(all(candidate["proposal_validated_by_path_feedback"] is False for candidate in proposal_rows))
        self.assertTrue((self.output_root / "xunce-safe-efficient-candidate-repair-summary.json").is_file())
        self.assertTrue((self.output_root / "xunce-safe-efficient-candidate-repair-overlay.jsonl").is_file())
        self.assertTrue((self.output_root / "xunce-safe-efficient-candidate-repair-audit.json").is_file())
        self.assertTrue((self.output_root / "xunce-safe-efficient-candidate-repair-report.md").is_file())
        self.assertEqual(json.loads((self.bound_root / "xunce-high-fidelity-path-feedback-audit.json").read_text(encoding="utf-8")), original_payload)

    def test_repaired_root_is_accepted_by_stage18f1_refinement(self) -> None:
        from scripts.run_xunce_cost_efficient_coverage_opportunity_refinement import (
            run_xunce_cost_efficient_coverage_opportunity_refinement,
        )
        from scripts.run_xunce_safe_efficient_candidate_repair import run_xunce_safe_efficient_candidate_repair

        repair_summary = run_xunce_safe_efficient_candidate_repair(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )
        self.assertEqual(repair_summary["status"], "passed")
        self._write_refinement_config()

        refinement_summary = run_xunce_cost_efficient_coverage_opportunity_refinement(
            config_path=self.refinement_config_path,
            output_root=self.refinement_output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(refinement_summary["status"], "passed")
        self.assertGreater(refinement_summary["safe_efficient_opportunity_count"], 0)
        self.assertNotIn("fallback_action_index_0", refinement_summary["incumbent_selection_sources"])

    def test_no_validated_safe_candidates_routes_to_path_feedback_validation(self) -> None:
        from scripts.run_xunce_safe_efficient_candidate_repair import run_xunce_safe_efficient_candidate_repair

        self._write_bound_root(no_safe=True)

        summary = run_xunce_safe_efficient_candidate_repair(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("no_validated_repaired_candidates", summary["reason_codes"])
        self.assertEqual(summary["next_required_change"], "repair_path_feedback_candidate_validation")

    def _write_config(self) -> None:
        payload = {
            "schema_version": "xunce-safe-efficient-candidate-repair-config/v1",
            "source_bound_coverage_root": str(self.bound_root),
            "source_root_cause_audit_root": str(self.audit_root),
            "required_scenario_count": 6,
            "candidate_generation_mode": "cost_guarded_frontier_interpolation",
            "interpolation_fractions": [0.25, 0.5, 0.75],
            "max_repaired_candidates_per_scenario": 6,
            "min_safe_efficient_opportunity_count": 1,
            "min_roi_group_with_safe_efficient_opportunity": 3,
            "path_budget_m": 5000.0,
            "canary_traffic_fraction": 0.0,
        }
        self.config_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _write_refinement_config(self) -> None:
        payload = {
            "schema_version": "xunce-cost-efficient-coverage-opportunity-refinement-config/v1",
            "source_materialized_coverage_root": str(self.output_root),
            "source_oracle_separability_root": str(self.audit_root),
            "required_scenario_count": 6,
            "min_cost_efficient_spread": 0.005,
            "min_safe_efficient_opportunity_count": 1,
            "min_roi_group_with_safe_efficient_opportunity": 3,
            "path_budget_m": 5000.0,
            "epsilon": 1.0e-12,
            "canary_traffic_fraction": 0.0,
        }
        self.refinement_config_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _write_bound_root(self, *, no_safe: bool = False) -> None:
        self.bound_root.mkdir(parents=True, exist_ok=True)
        scenarios = []
        slices = []
        for index in range(6):
            scenario_id = f"scenario_{index:03d}"
            roi_group = f"roi_{index % 3}"
            if no_safe:
                coverage = [0.02, 0.06, 0.14]
                costs = [10.0, 18.0, 30.0]
                risks = [0.10, 0.20, 0.30]
            else:
                coverage = [0.02, 0.08, 0.14]
                costs = [10.0, 8.0, 30.0]
                risks = [0.10, 0.09, 0.30]
            candidates = []
            for action_index, value in enumerate(coverage):
                cell = [index * 10 + action_index, action_index]
                if action_index == 2:
                    cell = [index * 10 + 8, 8]
                candidates.append(
                    {
                        "action_index": action_index,
                        "cell": cell,
                        "reachable": True,
                        "path_cost": costs[action_index],
                        "risk": risks[action_index],
                        "roi_weighted_coverage_delta": value,
                        "path_line_coverage_delta": value,
                        "expected_coverage_rate_delta": value,
                        "revisit_penalty": 0.0,
                    }
                )
            scenarios.append(
                {
                    "scenario_id": scenario_id,
                    "roi_group": roi_group,
                    "incumbent_selected_action_index": 0,
                    "incumbent_selection_source": "true_checkpoint_inference",
                    "path_feedback": {"candidates": candidates},
                    "open_grid_fallback_used": False,
                }
            )
            slices.append({"scenario_id": scenario_id, "roi_name": roi_group, "split": "train", "context_id": f"context-{scenario_id}", "contract": "c.json", "sidecar": "s.json"})
        self._write_json(self.bound_root / "xunce-true-incumbent-selection-binding-summary.json", {"schema_version": "xunce-true-incumbent-selection-binding-summary/v1", "status": "passed", "true_incumbent_selection_bound": True})
        self._write_json(self.bound_root / "xunce-candidate-level-coverage-opportunity-summary.json", {"schema_version": "xunce-candidate-level-coverage-opportunity-summary/v1", "status": "passed", "canary_traffic_fraction": 0.0})
        self._write_json(self.bound_root / "xunce-high-fidelity-real-map-roi-expansion-summary.json", {"schema_version": "xunce-high-fidelity-real-map-roi-expansion-summary/v1", "status": "passed", "slice_count": 6, "roi_group_count": 3, "canary_traffic_fraction": 0.0})
        self._write_jsonl(self.bound_root / "xunce-high-fidelity-real-map-slices.jsonl", slices)
        self._write_json(self.bound_root / "xunce-high-fidelity-path-feedback-audit.json", {"schema_version": "xunce-high-fidelity-path-feedback-audit/v1", "scenario_count": 6, "candidate_count": 18, "reachable_count": 18, "scenarios": scenarios})

    def _write_audit_root(self) -> None:
        self.audit_root.mkdir(parents=True, exist_ok=True)
        self._write_json(self.audit_root / "xunce-safe-efficient-opportunity-root-cause-audit-summary.json", {"schema_version": "xunce-safe-efficient-opportunity-root-cause-audit-summary/v1", "status": "passed", "root_cause_route": "ready_for_safe_efficient_candidate_repair"})
        self._write_json(self.audit_root / "xunce-oracle-separability-summary.json", {"schema_version": "xunce-oracle-separability-summary/v1", "status": "passed", "oracle_separable": False})

    @staticmethod
    def _write_json(path: Path, payload: dict) -> None:
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def _write_jsonl(path: Path, rows: list[dict]) -> None:
        path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
