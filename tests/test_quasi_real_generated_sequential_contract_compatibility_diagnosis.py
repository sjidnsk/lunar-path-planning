import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path


class QuasiRealGeneratedSequentialContractCompatibilityDiagnosisTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        for path in (self.repo_root / "scripts", self.repo_root / "model-explorer" / "src"):
            value = str(path)
            if value not in sys.path:
                sys.path.insert(0, value)
        self.temp_dir = Path(tempfile.mkdtemp(prefix="qreal-gen-seq-compat-"))
        self.update_root = self.temp_dir / "limited-update"
        self.base_candidate_root = self.temp_dir / "base-candidate"
        self.output_root = self.temp_dir / "diagnosis"
        self.source_root = self.temp_dir / "source"
        for path in (self.update_root, self.base_candidate_root, self.source_root):
            path.mkdir(parents=True)
        self._write_base_candidate()
        self._write_update_smoke_root()

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def test_reports_pre_existing_contract_mismatch_when_base_and_updated_fail_same_steps(self) -> None:
        from scripts.run_quasi_real_generated_sequential_contract_compatibility_diagnosis import (
            run_quasi_real_generated_sequential_contract_compatibility_diagnosis,
        )

        summary = run_quasi_real_generated_sequential_contract_compatibility_diagnosis(
            update_smoke_root=self.update_root,
            base_candidate_root=self.base_candidate_root,
            output_root=self.output_root,
            source_root=self.source_root,
            config=self._config(),
            repo_root=self.repo_root,
            replay_runner=self._replay_runner(base_kind="same_failure", updated_kind="same_failure"),
        )

        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["diagnosis_verdict"], "pre_existing_generated_sequential_contract_mismatch")
        self.assertEqual(summary["failed_step_count"], 6)
        self.assertEqual(summary["base_generated_sequential_status"], "failed")
        self.assertEqual(summary["updated_generated_sequential_status"], "failed")
        comparison_rows = self._read_jsonl(self.output_root / "failed-step-comparison.jsonl")
        self.assertEqual(len(comparison_rows), 6)
        self.assertEqual(comparison_rows[0]["episode_id"], "seq-channel_contrast-a")
        self.assertTrue((self.output_root / "compatibility-diagnosis-report.md").is_file())
        self.assertIn("channel_contrast", (self.output_root / "compatibility-diagnosis-report.md").read_text())

    def test_diagnostic_base_candidate_clone_preserves_weights_and_refreshes_provenance(self) -> None:
        from scripts.run_quasi_real_generated_sequential_contract_compatibility_diagnosis import (
            run_quasi_real_generated_sequential_contract_compatibility_diagnosis,
        )
        import torch

        summary = run_quasi_real_generated_sequential_contract_compatibility_diagnosis(
            update_smoke_root=self.update_root,
            base_candidate_root=self.base_candidate_root,
            output_root=self.output_root,
            source_root=self.source_root,
            config=self._config(),
            repo_root=self.repo_root,
            replay_runner=self._replay_runner(base_kind="same_failure", updated_kind="same_failure"),
        )

        original = torch.load(self.base_candidate_root / "experimental-hybrid-policy-candidate.pt", map_location="cpu", weights_only=False)
        cloned = torch.load(self.output_root / "diagnostic-base-candidate" / "experimental-hybrid-policy-candidate.pt", map_location="cpu", weights_only=False)
        self.assertTrue(torch.equal(original["model_state_dict"]["linear.weight"], cloned["model_state_dict"]["linear.weight"]))
        self.assertTrue(cloned["diagnostic_clone"])
        self.assertFalse(cloned["publishes_checkpoint"])
        metadata = json.loads((self.output_root / "diagnostic-base-candidate" / "experimental-hybrid-policy-candidate-metadata.json").read_text())
        self.assertTrue(metadata["diagnostic_clone"])
        self.assertFalse(metadata["replaces_default_policy"])
        self.assertEqual(summary["diagnostic_base_candidate_root"], str(self.output_root / "diagnostic-base-candidate"))

    def test_reports_update_induced_regression_when_base_passes_and_updated_fails(self) -> None:
        from scripts.run_quasi_real_generated_sequential_contract_compatibility_diagnosis import (
            run_quasi_real_generated_sequential_contract_compatibility_diagnosis,
        )

        summary = run_quasi_real_generated_sequential_contract_compatibility_diagnosis(
            update_smoke_root=self.update_root,
            base_candidate_root=self.base_candidate_root,
            output_root=self.output_root,
            source_root=self.source_root,
            config=self._config(),
            repo_root=self.repo_root,
            replay_runner=self._replay_runner(base_kind="passed", updated_kind="same_failure"),
        )

        self.assertEqual(summary["diagnosis_verdict"], "ppo_update_induced_generated_regression")
        self.assertEqual(summary["recommended_next_action"], "update_objective_or_learning_rate_guard_required")

    def test_reports_gate_metric_mismatch_when_rejection_reason_conflicts_with_step_delta(self) -> None:
        from scripts.run_quasi_real_generated_sequential_contract_compatibility_diagnosis import (
            run_quasi_real_generated_sequential_contract_compatibility_diagnosis,
        )

        summary = run_quasi_real_generated_sequential_contract_compatibility_diagnosis(
            update_smoke_root=self.update_root,
            base_candidate_root=self.base_candidate_root,
            output_root=self.output_root,
            source_root=self.source_root,
            config=self._config(),
            repo_root=self.repo_root,
            replay_runner=self._replay_runner(base_kind="passed", updated_kind="controlled_mismatched_failure"),
        )

        self.assertEqual(summary["diagnosis_verdict"], "gate_accounting_or_metric_mismatch")
        self.assertEqual(summary["gate_metric_mismatch_count"], 6)
        self.assertEqual(summary["recommended_next_action"], "generated_sequential_gate_metric_audit_required")

    def test_origin_aware_raw_policy_rejection_is_pre_existing_contract_mismatch(self) -> None:
        from scripts.run_quasi_real_generated_sequential_contract_compatibility_diagnosis import (
            run_quasi_real_generated_sequential_contract_compatibility_diagnosis,
        )

        summary = run_quasi_real_generated_sequential_contract_compatibility_diagnosis(
            update_smoke_root=self.update_root,
            base_candidate_root=self.base_candidate_root,
            output_root=self.output_root,
            source_root=self.source_root,
            config=self._config(),
            repo_root=self.repo_root,
            replay_runner=self._replay_runner(base_kind="raw_probe_failure", updated_kind="raw_probe_failure"),
        )

        self.assertEqual(summary["diagnosis_verdict"], "pre_existing_generated_sequential_contract_mismatch")
        self.assertEqual(summary["gate_metric_mismatch_count"], 0)
        self.assertEqual(summary["recommended_next_action"], "generated_sequential_contract_alignment_required")

    def test_missing_steps_or_rejection_report_fails_with_clear_reason(self) -> None:
        from scripts.run_quasi_real_generated_sequential_contract_compatibility_diagnosis import (
            run_quasi_real_generated_sequential_contract_compatibility_diagnosis,
        )

        shutil.rmtree(self.update_root / "post_update_generated_sequential")

        summary = run_quasi_real_generated_sequential_contract_compatibility_diagnosis(
            update_smoke_root=self.update_root,
            base_candidate_root=self.base_candidate_root,
            output_root=self.output_root,
            source_root=self.source_root,
            config=self._config(),
            repo_root=self.repo_root,
            replay_runner=self._replay_runner(base_kind="passed", updated_kind="same_failure"),
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("generated_sequential_rejection_report_missing", summary["reason_codes"])
        self.assertIn("generated_sequential_steps_missing", summary["reason_codes"])

    def _write_base_candidate(self) -> None:
        import torch

        state = {"linear.weight": torch.tensor([[1.0, 2.0], [3.0, 4.0]])}
        torch.save(
            {
                "schema_version": "controlled-hybrid-policy-candidate-checkpoint/v1",
                "experimental": True,
                "model_state_dict": state,
                "git_provenance": {"current_matches_sources": False},
            },
            self.base_candidate_root / "experimental-hybrid-policy-candidate.pt",
        )
        self._write_json(
            self.base_candidate_root / "experimental-hybrid-policy-candidate-metadata.json",
            {
                "schema_version": "controlled-hybrid-policy-candidate-checkpoint-metadata/v1",
                "experimental": True,
                "checkpoint_path": "experimental-hybrid-policy-candidate.pt",
                "publishes_checkpoint": False,
                "replaces_default_policy": False,
                "performance_claimed": False,
                "git_provenance": {"current_matches_sources": False},
            },
        )
        self._write_json(
            self.base_candidate_root / "raw-policy-generalization-candidate-summary.json",
            {
                "schema_version": "raw-policy-generalization-candidate-summary/v1",
                "status": "passed",
                "reason_codes": [],
                "checkpoint_path": "experimental-hybrid-policy-candidate.pt",
                "checkpoint_metadata_path": "experimental-hybrid-policy-candidate-metadata.json",
                "experimental_checkpoint": True,
                "publishes_checkpoint": False,
                "replaces_default_policy": False,
                "performance_claimed": False,
                "git_provenance": {"current_matches_sources": False},
            },
        )

    def _write_update_smoke_root(self) -> None:
        generated_root = self.update_root / "post_update_generated_sequential"
        generated_root.mkdir(parents=True)
        self._write_generated_sequential(generated_root, kind="same_failure")
        self._write_json(
            self.update_root / "limited-quasi-real-ppo-update-smoke-summary.json",
            {
                "schema_version": "limited-quasi-real-ppo-update-smoke-summary/v1",
                "status": "failed",
                "reason_codes": ["limited_quasi_real_ppo_update_post_update_gate_regression"],
                "base_candidate_root": str(self.base_candidate_root),
                "input_ppo_trainable_transition_count": 36,
                "optimizer_train_transition_count": 36,
                "post_update_quasi_real_teacher_following_status": "passed",
                "post_update_quasi_real_collector_status": "passed",
                "git_provenance": {"current_matches_sources": True},
            },
        )
        self._write_json(
            self.update_root / "post_update_quasi_real_teacher_following" / "quasi-real-guarded-teacher-following-pilot-summary.json",
            {
                "schema_version": "quasi-real-guarded-teacher-following-pilot-summary/v1",
                "status": "passed",
                "reason_codes": [],
                "teacher_agreement_rate": 1.0,
                "unsafe_disagreement_count": 0,
                "git_provenance": {"current_matches_sources": True},
            },
        )
        self._write_json(
            self.update_root / "post_update_quasi_real_collector" / "ppo-rollout-collector-summary.json",
            {
                "schema_version": "ppo-rollout-collector-summary/v1",
                "status": "passed",
                "reason_codes": [],
                "ppo_trainable_transition_count": 36,
                "diagnostic_transition_count": 72,
                "git_provenance": {"current_matches_sources": True},
            },
        )

    def _replay_runner(self, *, base_kind: str, updated_kind: str):
        def run(context: dict) -> dict:
            base_root = Path(context["base_replay_root"])
            updated_root = Path(context["updated_replay_root"])
            self._write_generated_sequential(base_root, kind=base_kind)
            self._write_generated_sequential(updated_root, kind=updated_kind)
            return {
                "base_root": base_root,
                "updated_root": updated_root,
                "base_summary": self._load_json(base_root / "policy-gated-sequential-canary-rollout-summary.json"),
                "updated_summary": self._load_json(updated_root / "policy-gated-sequential-canary-rollout-summary.json"),
            }

        return run

    def _write_generated_sequential(self, root: Path, *, kind: str) -> None:
        root.mkdir(parents=True, exist_ok=True)
        if kind == "passed":
            failed_steps = []
            status = "passed"
            reason_codes = []
        else:
            failed_steps = self._failed_steps(
                mismatched=kind == "mismatched_failure",
                raw_probe=kind == "raw_probe_failure",
                controlled_mismatched=kind == "controlled_mismatched_failure",
            )
            status = "failed"
            reason_codes = [
                "canary_rejected_policy_choice_count_above_threshold",
                "cumulative_path_cost_regression_count_above_threshold",
                "cumulative_risk_regression_count_above_threshold",
            ]
        self._write_json(
            root / "policy-gated-sequential-canary-rollout-summary.json",
            {
                "schema_version": "policy-gated-sequential-canary-rollout-summary/v1",
                "status": status,
                "reason_codes": reason_codes,
                "episode_count": 36,
                "step_count": 108,
                "canary_rejected_policy_choice_count": len(failed_steps),
                "cumulative_path_cost_regression_count": len(failed_steps),
                "cumulative_risk_regression_count": 2 if failed_steps else 0,
                "git_provenance": {"current_matches_sources": True},
            },
        )
        self._write_json(
            root / "policy-gated-sequential-canary-rejection-report.json",
            {
                "schema_version": "policy-gated-sequential-canary-rejection-report/v1",
                "reason_codes": reason_codes,
                "canary_rejection_reason_counts": {
                    "path_cost_regression": len(failed_steps),
                    "risk_regression": 2 if failed_steps else 0,
                },
                "failed_steps": failed_steps,
                "state_continuity_violations": [],
            },
        )
        (root / "policy-gated-sequential-canary-steps.jsonl").write_text(
            "".join(json.dumps(step) + "\n" for step in failed_steps),
            encoding="utf-8",
        )

    def _failed_steps(
        self,
        *,
        mismatched: bool,
        raw_probe: bool = False,
        controlled_mismatched: bool = False,
    ) -> list[dict]:
        families = [
            ("seq-channel_contrast-a", 1, "channel_contrast", ["path_cost_regression"]),
            ("seq-channel_contrast-a", 2, "channel_contrast", ["path_cost_regression"]),
            ("seq-channel_contrast-b", 0, "channel_contrast", ["path_cost_regression", "risk_regression"]),
            ("seq-mixed_stress_detour-a", 1, "mixed_stress_detour", ["path_cost_regression"]),
            ("seq-high_risk_tradeoff-e", 2, "high_risk_tradeoff", ["path_cost_regression"]),
            ("seq-near_blocked_safe_alt-f", 0, "near_blocked_safe_alt", ["path_cost_regression", "risk_regression"]),
        ]
        rows = []
        for index, (episode_id, step_index, family, reasons) in enumerate(families):
            path_delta = 0.0 if (mismatched or raw_probe or controlled_mismatched) else 1.0
            risk_delta = 0.0 if (mismatched or raw_probe or controlled_mismatched) else (1.0 if "risk_regression" in reasons else 0.0)
            rows.append(
                {
                    "schema_version": "policy-gated-sequential-canary-step/v1",
                    "episode_id": episode_id,
                    "step_index": step_index,
                    "scenario_id": f"scenario-{index}",
                    "scenario_group": family,
                    "decision_class": "canary_rejected_policy_choice",
                    "controlled_choice_source": "source_fallback",
                    "canary_rejection_reason_codes": [] if controlled_mismatched else reasons,
                    "controlled_regression_reason_codes": reasons if controlled_mismatched else [],
                    "raw_policy_regression_reason_codes": reasons,
                    "policy_selected_path_cost_delta": path_delta,
                    "policy_selected_risk_delta": risk_delta,
                    "raw_policy_selected_path_cost_delta": 1.0,
                    "raw_policy_selected_risk_delta": 1.0 if "risk_regression" in reasons else 0.0,
                    "raw_policy_logit_margin_vs_source": 0.5 + index,
                    "policy_selected_policy_target_cell": [index, index + 1],
                    "source_selected_policy_target_cell": [index + 2, index + 3],
                }
            )
        return rows

    def _config(self) -> dict:
        return {
            "schema_version": "quasi-real-generated-sequential-contract-compatibility-diagnosis-config/v1",
            "input_files": {
                "update_smoke_summary": "limited-quasi-real-ppo-update-smoke-summary.json",
                "generated_sequential_summary": "post_update_generated_sequential/policy-gated-sequential-canary-rollout-summary.json",
                "generated_sequential_steps": "post_update_generated_sequential/policy-gated-sequential-canary-steps.jsonl",
                "generated_sequential_rejection_report": "post_update_generated_sequential/policy-gated-sequential-canary-rejection-report.json",
                "quasi_real_teacher_following_summary": "post_update_quasi_real_teacher_following/quasi-real-guarded-teacher-following-pilot-summary.json",
                "quasi_real_collector_summary": "post_update_quasi_real_collector/ppo-rollout-collector-summary.json",
                "base_checkpoint": "experimental-hybrid-policy-candidate.pt",
                "base_checkpoint_metadata": "experimental-hybrid-policy-candidate-metadata.json",
                "base_candidate_summary": "raw-policy-generalization-candidate-summary.json",
            },
            "output_files": {
                "summary": "quasi-real-generated-sequential-contract-compatibility-summary.json",
                "failed_step_comparison": "failed-step-comparison.jsonl",
                "baseline_vs_updated_summary": "baseline-vs-updated-sequential-summary.json",
                "report": "compatibility-diagnosis-report.md",
            },
            "replay": {
                "base_replay_root": "base_generated_sequential",
                "updated_replay_root": "updated_generated_sequential_replay",
                "reuse_existing_updated_replay": True,
            },
            "evaluation": {
                "max_path_cost_regression": 0.0,
                "max_risk_regression": 0.0,
            },
            "validation": {
                "expected_failed_step_count": 6
            },
        }

    def _write_json(self, path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _load_json(self, path: Path) -> dict:
        return json.loads(path.read_text(encoding="utf-8"))

    def _read_jsonl(self, path: Path) -> list[dict]:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
