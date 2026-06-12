from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


class QuasiRealSafeAlternativeOpportunityDiagnosisTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp(prefix="qreal-safe-opportunity-"))
        self.repo_root = Path(__file__).resolve().parents[1]
        scripts_dir = str(self.repo_root / "scripts")
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)
        self.quasi_root = self.temp_dir / "quasi"
        self.guarded_root = self.temp_dir / "guarded"
        self.alignment_root = self.temp_dir / "alignment"
        self.output_root = self.temp_dir / "out"
        for path in (self.quasi_root, self.guarded_root, self.alignment_root):
            path.mkdir(parents=True, exist_ok=True)
        self._write_common_inputs()

    def test_classifies_missing_safe_alternative(self) -> None:
        from scripts.run_quasi_real_safe_alternative_opportunity_diagnosis import (
            run_quasi_real_safe_alternative_opportunity_diagnosis,
        )

        self._write_path_feedback(
            [
                self._scenario(
                    "smooth-001",
                    "smooth_high_confidence",
                    source_action=0,
                    candidates=[
                        self._candidate("smooth-001", 0, 10.0, 0.20, 1.0, source=True),
                        self._candidate("smooth-001", 1, 11.0, 0.25, 0.9),
                    ],
                )
            ]
        )
        self._write_guarded_decisions([self._guarded("smooth-001", "smooth_high_confidence", 0, 0)])

        summary = run_quasi_real_safe_alternative_opportunity_diagnosis(
            quasi_real_root=self.quasi_root,
            guarded_pilot_root=self.guarded_root,
            alignment_root=self.alignment_root,
            output_root=self.output_root,
            config=self._config(min_contexts=1, min_roi_groups=1),
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["opportunity_verdict"], "quasi_real_safe_alternative_opportunity_gap")
        self.assertEqual(summary["opportunity_missing_count"], 1)
        self.assertEqual(summary["safe_alternative_context_count"], 0)
        self.assertEqual(summary["safe_better_opportunity_context_count"], 0)
        self.assertEqual(summary["next_required_change"], "quasi_real_safe_alternative_opportunity_gap")
        rows = self._read_jsonl(self.output_root / "quasi-real-safe-alternative-opportunity-diagnostics.jsonl")
        self.assertEqual(rows[0]["opportunity_class"], "opportunity_missing")

    def test_classifies_safe_alternative_that_is_not_better(self) -> None:
        from scripts.run_quasi_real_safe_alternative_opportunity_diagnosis import (
            run_quasi_real_safe_alternative_opportunity_diagnosis,
        )

        self._write_path_feedback(
            [
                self._scenario(
                    "smooth-001",
                    "smooth_high_confidence",
                    source_action=0,
                    candidates=[
                        self._candidate("smooth-001", 0, 10.0, 0.20, 1.0, source=True),
                        self._candidate("smooth-001", 1, 9.9, 0.20, 1.0),
                    ],
                )
            ]
        )
        self._write_guarded_decisions([self._guarded("smooth-001", "smooth_high_confidence", 0, 0)])

        summary = run_quasi_real_safe_alternative_opportunity_diagnosis(
            quasi_real_root=self.quasi_root,
            guarded_pilot_root=self.guarded_root,
            alignment_root=self.alignment_root,
            output_root=self.output_root,
            config=self._config(min_contexts=1, min_roi_groups=1),
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["safe_alternative_context_count"], 1)
        self.assertEqual(summary["safe_better_opportunity_context_count"], 0)
        self.assertEqual(summary["opportunity_missing_count"], 0)
        self.assertEqual(summary["safe_alternative_exists_but_not_better_count"], 1)
        rows = self._read_jsonl(self.output_root / "quasi-real-safe-alternative-opportunity-diagnostics.jsonl")
        self.assertEqual(rows[0]["opportunity_class"], "safe_alternative_exists_but_not_better")

    def test_classifies_safe_better_opportunity_missed_by_source_aligned_policy(self) -> None:
        from scripts.run_quasi_real_safe_alternative_opportunity_diagnosis import (
            run_quasi_real_safe_alternative_opportunity_diagnosis,
        )

        self._write_path_feedback(
            [
                self._scenario(
                    "mixed-001",
                    "mixed_risk",
                    source_action=0,
                    candidates=[
                        self._candidate("mixed-001", 0, 10.0, 0.20, 1.0, source=True),
                        self._candidate("mixed-001", 1, 9.5, 0.18, 1.02),
                    ],
                )
            ]
        )
        self._write_guarded_decisions([self._guarded("mixed-001", "mixed_risk", 0, 0)])

        summary = run_quasi_real_safe_alternative_opportunity_diagnosis(
            quasi_real_root=self.quasi_root,
            guarded_pilot_root=self.guarded_root,
            alignment_root=self.alignment_root,
            output_root=self.output_root,
            config=self._config(min_contexts=1, min_roi_groups=1),
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["opportunity_verdict"], "acceptable_for_quasi_real_safe_choice_calibration")
        self.assertEqual(summary["safe_better_opportunity_context_count"], 1)
        self.assertEqual(summary["policy_missed_safe_better_opportunity_count"], 1)
        self.assertEqual(summary["policy_selected_safe_better_opportunity_count"], 0)
        self.assertEqual(summary["next_required_change"], "quasi_real_policy_safe_choice_alignment_insufficient")
        rows = self._read_jsonl(self.output_root / "quasi-real-safe-alternative-opportunity-diagnostics.jsonl")
        self.assertEqual(rows[0]["opportunity_class"], "safe_better_opportunity_exists_policy_source_aligned")

    def test_counts_policy_selected_safe_better_opportunity(self) -> None:
        from scripts.run_quasi_real_safe_alternative_opportunity_diagnosis import (
            run_quasi_real_safe_alternative_opportunity_diagnosis,
        )

        self._write_path_feedback(
            [
                self._scenario(
                    "mixed-001",
                    "mixed_risk",
                    source_action=0,
                    candidates=[
                        self._candidate("mixed-001", 0, 10.0, 0.20, 1.0, source=True),
                        self._candidate("mixed-001", 1, 9.5, 0.18, 1.02),
                    ],
                )
            ]
        )
        self._write_guarded_decisions([self._guarded("mixed-001", "mixed_risk", 0, 1)])

        summary = run_quasi_real_safe_alternative_opportunity_diagnosis(
            quasi_real_root=self.quasi_root,
            guarded_pilot_root=self.guarded_root,
            alignment_root=self.alignment_root,
            output_root=self.output_root,
            config=self._config(min_contexts=1, min_roi_groups=1),
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["policy_selected_safe_better_opportunity_count"], 1)
        rows = self._read_jsonl(self.output_root / "quasi-real-safe-alternative-opportunity-diagnostics.jsonl")
        self.assertEqual(rows[0]["opportunity_class"], "safe_better_opportunity_policy_selected")

    def test_excludes_missing_context_id_as_bridge_gap(self) -> None:
        from scripts.run_quasi_real_safe_alternative_opportunity_diagnosis import (
            run_quasi_real_safe_alternative_opportunity_diagnosis,
        )

        self._write_path_feedback(
            [
                self._scenario(
                    "mixed-001",
                    "mixed_risk",
                    source_action=0,
                    candidates=[
                        self._candidate("mixed-001", 0, 10.0, 0.20, 1.0, source=True, context_id=None),
                        self._candidate("mixed-001", 1, 9.5, 0.18, 1.02),
                    ],
                )
            ]
        )
        self._write_guarded_decisions([self._guarded("mixed-001", "mixed_risk", 0, 0)])

        summary = run_quasi_real_safe_alternative_opportunity_diagnosis(
            quasi_real_root=self.quasi_root,
            guarded_pilot_root=self.guarded_root,
            alignment_root=self.alignment_root,
            output_root=self.output_root,
            config=self._config(min_contexts=1, min_roi_groups=1),
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertEqual(summary["opportunity_verdict"], "real_map_bridge_or_feedback_gap")
        self.assertEqual(summary["context_id_missing_count"], 1)
        exclusion = json.loads(
            (self.output_root / "quasi-real-safe-alternative-opportunity-exclusion-report.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(exclusion["opportunity_exclusion_count"], 1)

    def test_readiness_accepts_passed_opportunity_summary(self) -> None:
        from scripts.run_policy_training_readiness_review import (
            _quasi_real_safe_alternative_opportunity_readiness,
        )

        readiness = _quasi_real_safe_alternative_opportunity_readiness(
            {
                "schema_version": "quasi-real-safe-alternative-opportunity-summary/v1",
                "status": "passed",
                "reason_codes": [],
                "opportunity_verdict": "acceptable_for_quasi_real_safe_choice_calibration",
                "quasi_real_context_count": 12,
                "policy_decision_count": 12,
                "roi_group_count": 4,
                "context_id_missing_count": 0,
                "opportunity_exclusion_count": 0,
                "invalid_action_mask_count": 0,
                "fallback_or_open_grid_count": 0,
                "safety_regression_count": 0,
                "contract_violation_count": 0,
                "path_cost_regression_count": 0,
                "risk_regression_count": 0,
                "source_selection_regression_count": 0,
                "safe_better_opportunity_context_count": 1,
                "runs_ppo_update": False,
                "publishes_checkpoint": False,
                "replaces_default_policy": False,
                "performance_claimed": False,
            }
        )

        self.assertTrue(readiness["completed"])
        self.assertEqual(readiness["training_blockers"], [])

    def _write_common_inputs(self) -> None:
        (self.alignment_root / "quasi-real-shadow-alignment-summary.json").write_text(
            json.dumps(
                {
                    "schema_version": "quasi-real-shadow-alignment-summary/v1",
                    "status": "passed",
                    "reason_codes": [],
                    "alignment_verdict": "acceptable_for_quasi_real_shadow_audit",
                    "hard_positive_added_count": 0,
                    "ppo_transition_added_count": 0,
                }
            ),
            encoding="utf-8",
        )
        (self.guarded_root / "quasi-real-guarded-policy-pilot-summary.json").write_text(
            json.dumps(
                {
                    "schema_version": "quasi-real-guarded-policy-pilot-summary/v1",
                    "status": "failed",
                    "reason_codes": ["quasi_real_guarded_policy_over_conservative"],
                    "guarded_pilot_verdict": "policy_over_conservative_on_quasi_real",
                    "quasi_real_context_count": 1,
                    "policy_decision_count": 1,
                    "roi_group_count": 1,
                    "policy_changed_gate_passed_count": 0,
                    "policy_changed_gate_rejected_count": 0,
                    "invalid_action_mask_count": 0,
                    "fallback_or_open_grid_count": 0,
                    "safety_regression_count": 0,
                    "contract_violation_count": 0,
                    "path_cost_regression_count": 0,
                    "risk_regression_count": 0,
                    "source_selection_regression_count": 0,
                }
            ),
            encoding="utf-8",
        )

    def _write_path_feedback(self, scenarios: list[dict[str, object]]) -> None:
        self.quasi_root.mkdir(parents=True, exist_ok=True)
        (self.quasi_root / "quasi-real-map-path-feedback-summary.json").write_text(
            json.dumps(
                {
                    "schema_version": "path-feedback-summary/v1",
                    "status": "completed",
                    "scenario_count": len(scenarios),
                    "scenarios": scenarios,
                }
            ),
            encoding="utf-8",
        )
        slices = [
            {
                "schema_version": "quasi-real-map-slice/v1",
                "scenario_id": scenario["scenario_id"],
                "roi_group": scenario["scenario_group"],
                "roi_name": scenario["scenario_group"],
                "split": "test",
                "map_id": "lola-test",
                "slice_id": scenario["scenario_id"],
                "context_id": f"{scenario['scenario_id']}-slice",
            }
            for scenario in scenarios
        ]
        (self.quasi_root / "quasi-real-map-slices.jsonl").write_text(
            "".join(json.dumps(item) + "\n" for item in slices),
            encoding="utf-8",
        )
        (self.quasi_root / "quasi-real-map-domain-gap-summary.json").write_text(
            json.dumps(
                {
                    "schema_version": "quasi-real-map-domain-gap-summary/v1",
                    "status": "passed",
                    "reason_codes": [],
                    "domain_gap_verdict": "acceptable_for_next_pilot",
                    "slice_count": len(scenarios),
                    "roi_group_count": len({scenario["scenario_group"] for scenario in scenarios}),
                }
            ),
            encoding="utf-8",
        )

    def _write_guarded_decisions(self, decisions: list[dict[str, object]]) -> None:
        (self.guarded_root / "quasi-real-guarded-policy-decisions.jsonl").write_text(
            "".join(json.dumps(item) + "\n" for item in decisions),
            encoding="utf-8",
        )

    def _scenario(
        self,
        scenario_id: str,
        group: str,
        *,
        source_action: int,
        candidates: list[dict[str, object]],
    ) -> dict[str, object]:
        return {
            "scenario_id": scenario_id,
            "scenario_group": group,
            "scenario_seed": 7,
            "scenario_variant_id": f"{scenario_id}-v1",
            "path_feedback": {
                "source_selected_action_index": source_action,
                "candidates": candidates,
            },
        }

    def _candidate(
        self,
        scenario_id: str,
        action_index: int,
        path_cost: float,
        risk: float,
        utility: float,
        *,
        source: bool = False,
        context_id: str | None = "auto",
    ) -> dict[str, object]:
        payload: dict[str, object] = {
            "action_index": action_index,
            "source_action_index": action_index,
            "policy_target_cell": [action_index, action_index + 1],
            "execution_goal_cell": [action_index, action_index + 1],
            "reachable": True,
            "replan_required": False,
            "open_grid_fallback_used": False,
            "path_cost": path_cost,
            "risk": risk,
            "utility": utility,
            "candidate_generation": {
                "source_selection_status": "source_selected" if source else "not_source_candidate"
            },
            "platform_goal_feasibility": {
                "contract_reachable": True,
                "classification": "goal_passable",
            },
        }
        if context_id is not None:
            payload["context_id"] = f"{scenario_id}-{action_index}" if context_id == "auto" else context_id
        return payload

    def _guarded(
        self,
        scenario_id: str,
        group: str,
        source_action: int,
        raw_action: int,
    ) -> dict[str, object]:
        return {
            "schema_version": "quasi-real-guarded-policy-decision/v1",
            "scenario_id": scenario_id,
            "roi_group": group,
            "source_action_index": source_action,
            "raw_policy_action_index": raw_action,
            "decision_class": "source_aligned" if source_action == raw_action else "policy_changed_gate_passed",
            "controlled_choice_source": "source" if source_action == raw_action else "policy",
            "gate_reason_codes": [],
        }

    def _config(self, *, min_contexts: int = 12, min_roi_groups: int = 4) -> dict[str, object]:
        return {
            "schema_version": "quasi-real-safe-alternative-opportunity-diagnosis-config/v1",
            "input_files": {
                "quasi_real_path_feedback_summary": "quasi-real-map-path-feedback-summary.json",
                "quasi_real_slices": "quasi-real-map-slices.jsonl",
                "quasi_real_domain_gap_summary": "quasi-real-map-domain-gap-summary.json",
                "guarded_pilot_summary": "quasi-real-guarded-policy-pilot-summary.json",
                "guarded_decisions": "quasi-real-guarded-policy-decisions.jsonl",
                "alignment_summary": "quasi-real-shadow-alignment-summary.json",
            },
            "output_files": {
                "summary": "quasi-real-safe-alternative-opportunity-summary.json",
                "diagnostics": "quasi-real-safe-alternative-opportunity-diagnostics.jsonl",
                "exclusion_report": "quasi-real-safe-alternative-opportunity-exclusion-report.json",
                "report": "quasi-real-safe-alternative-opportunity-report.md",
            },
            "thresholds": {
                "max_path_cost_regression": 0.0,
                "max_risk_regression": 0.0,
                "better_path_cost_delta": -0.25,
                "better_risk_delta": -0.01,
                "better_utility_delta": 0.005,
            },
            "validation": {
                "min_quasi_real_context_count": min_contexts,
                "min_roi_group_count": min_roi_groups,
                "max_opportunity_exclusion_count": 0,
            },
            "non_goals": ["no_ppo_update", "no_policy_takeover"],
        }

    def _read_jsonl(self, path: Path) -> list[dict[str, object]]:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


if __name__ == "__main__":
    unittest.main()
