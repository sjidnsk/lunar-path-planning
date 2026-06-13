import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path


class GeneratedSequentialLongHorizonTeacherSkillContractAlignmentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        scripts_dir = str(self.repo_root / "scripts")
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)
        self.temp_dir = Path(tempfile.mkdtemp(prefix="generated-seq-long-horizon-"))
        self.diagnosis_root = self.temp_dir / "diagnosis"
        self.accounting_root = self.temp_dir / "accounting"
        self.output_root = self.temp_dir / "alignment"
        self.updated_root = self.diagnosis_root / "updated_generated_sequential_replay"
        self.base_root = self.diagnosis_root / "base_generated_sequential"
        self.updated_root.mkdir(parents=True)
        self.base_root.mkdir(parents=True)
        self.accounting_root.mkdir(parents=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def test_same_as_teacher_counts_as_teacher_equivalent_and_raw_rejections_stay_diagnostic(self) -> None:
        from scripts.run_generated_sequential_long_horizon_teacher_skill_contract_alignment import (
            run_generated_sequential_long_horizon_teacher_skill_contract_alignment,
        )

        steps = [
            self._step("ep-teacher", 0, decision_class="source_aligned", controlled_source="source"),
            self._step("ep-teacher", 1, decision_class="source_aligned", controlled_source="source"),
            self._step(
                "ep-beyond",
                0,
                decision_class="canary_accepted_policy_choice",
                controlled_source="policy",
                policy_path_delta=-2.0,
                policy_risk_delta=-0.2,
            ),
            self._step("ep-beyond", 1, decision_class="source_aligned", controlled_source="source"),
            self._step(
                "ep-diagnostic",
                0,
                decision_class="canary_rejected_policy_choice",
                controlled_source="source_fallback",
                raw_path_delta=4.0,
                raw_risk_delta=0.3,
                canary_reasons=["path_cost_regression", "risk_regression"],
                raw_reasons=["path_cost_regression", "risk_regression"],
            ),
        ]
        self._write_inputs(steps)

        summary = run_generated_sequential_long_horizon_teacher_skill_contract_alignment(
            diagnosis_root=self.diagnosis_root,
            accounting_audit_root=self.accounting_root,
            output_root=self.output_root,
            config=self._config(),
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["reason_codes"], [])
        self.assertEqual(summary["verdict"], "long_horizon_teacher_skill_contract_aligned")
        self.assertEqual(summary["episode_count"], 3)
        self.assertEqual(summary["teacher_equivalent_episode_count"], 3)
        self.assertEqual(summary["beyond_teacher_episode_count"], 1)
        self.assertEqual(summary["teacher_aligned_active_choice_count"], 3)
        self.assertEqual(summary["dominated_raw_choice_count"], 1)
        self.assertEqual(summary["controlled_regression_episode_count"], 0)
        self.assertEqual(summary["controlled_path_cost_regression_count"], 0)
        self.assertEqual(summary["controlled_risk_regression_count"], 0)
        self.assertFalse(summary["publishes_checkpoint"])
        self.assertFalse(summary["formal_training_ready_claimed"])

        comparisons = self._read_jsonl(self.output_root / "teacher-vs-policy-return-comparison.jsonl")
        by_episode = {row["episode_id"]: row for row in comparisons}
        self.assertTrue(by_episode["ep-teacher"]["teacher_equivalent_episode"])
        self.assertTrue(by_episode["ep-beyond"]["beyond_teacher_episode"])
        self.assertTrue(by_episode["ep-diagnostic"]["teacher_equivalent_episode"])

        dominated = self._read_jsonl(self.output_root / "dominated-raw-choice-diagnostics.jsonl")
        self.assertEqual(len(dominated), 1)
        self.assertEqual(dominated[0]["episode_id"], "ep-diagnostic")

    def test_single_step_improvement_does_not_pass_when_multi_step_return_regresses(self) -> None:
        from scripts.run_generated_sequential_long_horizon_teacher_skill_contract_alignment import (
            run_generated_sequential_long_horizon_teacher_skill_contract_alignment,
        )

        steps = [
            self._step(
                "ep-short-sighted",
                0,
                decision_class="canary_accepted_policy_choice",
                controlled_source="policy",
                policy_path_delta=-1.0,
            ),
            self._step(
                "ep-short-sighted",
                1,
                decision_class="canary_accepted_policy_choice",
                controlled_source="policy",
                policy_path_delta=3.0,
                controlled_reasons=["path_cost_regression"],
            ),
        ]
        self._write_inputs(steps)

        summary = run_generated_sequential_long_horizon_teacher_skill_contract_alignment(
            diagnosis_root=self.diagnosis_root,
            accounting_audit_root=self.accounting_root,
            output_root=self.output_root,
            config=self._config(),
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("long_horizon_controlled_regression_detected", summary["reason_codes"])
        self.assertEqual(summary["verdict"], "long_horizon_contract_still_blocked")
        self.assertEqual(summary["teacher_equivalent_episode_count"], 0)
        self.assertEqual(summary["beyond_teacher_episode_count"], 0)
        self.assertEqual(summary["controlled_regression_episode_count"], 1)

    def test_missing_or_stale_input_fails_with_explicit_reason(self) -> None:
        from scripts.run_generated_sequential_long_horizon_teacher_skill_contract_alignment import (
            run_generated_sequential_long_horizon_teacher_skill_contract_alignment,
        )

        summary = run_generated_sequential_long_horizon_teacher_skill_contract_alignment(
            diagnosis_root=self.diagnosis_root,
            accounting_audit_root=self.accounting_root,
            output_root=self.output_root,
            config=self._config(),
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertEqual(summary["verdict"], "missing_or_stale_input_evidence")
        self.assertIn("compatibility_diagnosis_summary_missing", summary["reason_codes"])
        self.assertIn("updated_generated_sequential_steps_missing", summary["reason_codes"])

    def test_readiness_accepts_only_aligned_long_horizon_summary(self) -> None:
        from scripts.run_policy_training_readiness_review import (
            _generated_sequential_long_horizon_teacher_skill_contract_readiness,
        )

        aligned = _generated_sequential_long_horizon_teacher_skill_contract_readiness(
            {
                "schema_version": "generated-sequential-long-horizon-teacher-skill-contract-summary/v1",
                "status": "passed",
                "reason_codes": [],
                "verdict": "long_horizon_teacher_skill_contract_aligned",
                "teacher_equivalent_episode_count": 12,
                "controlled_regression_episode_count": 0,
                "publishes_checkpoint": False,
                "replaces_default_policy": False,
                "performance_claimed": False,
                "formal_training_ready_claimed": False,
                "git_provenance": {"current_matches_sources": True},
            }
        )
        self.assertTrue(aligned["completed"])
        self.assertEqual(aligned["training_blockers"], [])

        blocked = _generated_sequential_long_horizon_teacher_skill_contract_readiness(
            {
                "schema_version": "generated-sequential-long-horizon-teacher-skill-contract-summary/v1",
                "status": "passed",
                "reason_codes": [],
                "verdict": "long_horizon_contract_still_blocked",
                "teacher_equivalent_episode_count": 0,
                "controlled_regression_episode_count": 1,
                "publishes_checkpoint": False,
                "replaces_default_policy": False,
                "performance_claimed": False,
                "formal_training_ready_claimed": False,
                "git_provenance": {"current_matches_sources": True},
            }
        )
        self.assertFalse(blocked["completed"])
        self.assertIn("generated_sequential_long_horizon_contract_still_blocked", blocked["training_blockers"])
        self.assertEqual(blocked["next_required_change"], "generated_sequential_contract_alignment_required")

    def _write_inputs(self, steps: list[dict]) -> None:
        for root in (self.updated_root, self.base_root):
            (root / "policy-gated-sequential-canary-steps.jsonl").write_text(
                "".join(json.dumps(step) + "\n" for step in steps),
                encoding="utf-8",
            )
            self._write_json(
                root / "policy-gated-sequential-canary-rejection-report.json",
                {
                    "schema_version": "policy-gated-sequential-canary-rejection-report/v1",
                    "failed_steps": [
                        step for step in steps if step["decision_class"] == "canary_rejected_policy_choice"
                    ],
                },
            )
            self._write_json(
                root / "policy-gated-sequential-canary-rollout-summary.json",
                {
                    "schema_version": "policy-gated-sequential-canary-rollout-summary/v1",
                    "status": "failed",
                    "reason_codes": ["canary_rejected_policy_choice_count_above_threshold"],
                    "git_provenance": {"current_matches_sources": True},
                },
            )
        self._write_json(
            self.diagnosis_root / "quasi-real-generated-sequential-contract-compatibility-summary.json",
            {
                "schema_version": "quasi-real-generated-sequential-contract-compatibility-summary/v1",
                "status": "passed",
                "reason_codes": [],
                "diagnosis_verdict": "pre_existing_generated_sequential_contract_mismatch",
                "recommended_next_action": "generated_sequential_contract_alignment_required",
                "base_generated_sequential_status": "failed",
                "updated_generated_sequential_status": "failed",
                "quasi_real_teacher_following_status": "passed",
                "quasi_real_collector_status": "passed",
                "publishes_checkpoint": False,
                "replaces_default_policy": False,
                "performance_claimed": False,
                "formal_training_ready_claimed": False,
                "git_provenance": {"current_matches_sources": True},
            },
        )
        self._write_json(
            self.accounting_root / "generated-sequential-gate-metric-accounting-audit-summary.json",
            {
                "schema_version": "generated-sequential-gate-metric-accounting-audit-summary/v1",
                "status": "passed",
                "reason_codes": [],
                "diagnosis_verdict_after_origin_split": "pre_existing_generated_sequential_contract_mismatch",
                "legacy_mismatch_count": 1,
                "controlled_path_cost_regression_count": 0,
                "controlled_risk_regression_count": 0,
                "publishes_checkpoint": False,
                "replaces_default_policy": False,
                "performance_claimed": False,
                "formal_training_ready_claimed": False,
                "git_provenance": {"current_matches_sources": True},
            },
        )
        self._write_json(
            self.diagnosis_root / "post-update-quasi-real-teacher-following-summary.json",
            {"schema_version": "quasi-real-guarded-teacher-following-pilot-summary/v1", "status": "passed"},
        )
        self._write_json(
            self.diagnosis_root / "post-update-quasi-real-collector-summary.json",
            {"schema_version": "ppo-rollout-collector-summary/v1", "status": "passed"},
        )

    def _step(
        self,
        episode_id: str,
        step_index: int,
        *,
        decision_class: str,
        controlled_source: str,
        policy_path_delta: float = 0.0,
        policy_risk_delta: float = 0.0,
        raw_path_delta: float = 0.0,
        raw_risk_delta: float = 0.0,
        canary_reasons: list[str] | None = None,
        raw_reasons: list[str] | None = None,
        controlled_reasons: list[str] | None = None,
    ) -> dict:
        same_as_teacher = controlled_source in {"source", "source_fallback"}
        policy_action = 0 if same_as_teacher else 1
        raw_action = 0 if same_as_teacher and not raw_reasons else 1
        return {
            "schema_version": "policy-gated-sequential-canary-step/v1",
            "episode_id": episode_id,
            "step_index": step_index,
            "scenario_id": f"{episode_id}-{step_index}",
            "scenario_group": "unit",
            "decision_class": decision_class,
            "source_selected_action_index": 0,
            "raw_policy_selected_action_index": raw_action,
            "policy_selected_action_index": policy_action,
            "policy_selected_path_cost_delta": policy_path_delta,
            "policy_selected_risk_delta": policy_risk_delta,
            "policy_selected_utility_delta": 0.0,
            "raw_policy_selected_path_cost_delta": raw_path_delta,
            "raw_policy_selected_risk_delta": raw_risk_delta,
            "raw_policy_selected_utility_delta": 0.0,
            "controlled_choice_source": controlled_source,
            "controlled_regression_reason_codes": controlled_reasons or [],
            "canary_rejection_reason_codes": canary_reasons or [],
            "raw_policy_regression_reason_codes": raw_reasons or [],
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "performance_claimed": False,
        }

    def _config(self) -> dict:
        return {
            "schema_version": "generated-sequential-long-horizon-teacher-skill-contract-alignment-config/v1",
            "input_files": {
                "compatibility_summary": "quasi-real-generated-sequential-contract-compatibility-summary.json",
                "accounting_audit_summary": "generated-sequential-gate-metric-accounting-audit-summary.json",
                "updated_steps": "updated_generated_sequential_replay/policy-gated-sequential-canary-steps.jsonl",
                "updated_rejection_report": "updated_generated_sequential_replay/policy-gated-sequential-canary-rejection-report.json",
                "base_steps": "base_generated_sequential/policy-gated-sequential-canary-steps.jsonl",
                "base_rejection_report": "base_generated_sequential/policy-gated-sequential-canary-rejection-report.json",
                "quasi_real_teacher_following_summary": "post-update-quasi-real-teacher-following-summary.json",
                "quasi_real_collector_summary": "post-update-quasi-real-collector-summary.json",
            },
            "output_files": {
                "summary": "long-horizon-teacher-skill-contract-summary.json",
                "return_comparison": "teacher-vs-policy-return-comparison.jsonl",
                "teacher_equivalent_report": "teacher-equivalent-episode-report.md",
                "beyond_teacher_report": "beyond-teacher-opportunity-report.md",
                "dominated_raw_choice_diagnostics": "dominated-raw-choice-diagnostics.jsonl",
            },
            "evaluation": {
                "horizon_steps": 3,
                "teacher_equivalence_tolerance": 0.01,
                "beyond_teacher_margin": 0.5,
                "max_path_cost_regression": 0.0,
                "max_risk_regression": 0.0,
                "return_weights": {
                    "path_cost": 1.0,
                    "risk": 1.0,
                    "safety_penalty": 10.0,
                    "contract_penalty": 10.0,
                    "source_selection_penalty": 10.0,
                    "progress": 1.0,
                    "terminal": 1.0,
                    "utility": 0.0,
                },
            },
        }

    def _write_json(self, path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _read_jsonl(self, path: Path) -> list[dict]:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
