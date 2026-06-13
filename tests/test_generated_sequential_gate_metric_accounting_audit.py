import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path


class GeneratedSequentialGateMetricAccountingAuditTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        scripts_dir = str(self.repo_root / "scripts")
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)
        self.temp_dir = Path(tempfile.mkdtemp(prefix="generated-seq-accounting-"))
        self.diagnosis_root = self.temp_dir / "diagnosis"
        self.output_root = self.temp_dir / "audit"
        self.base_root = self.diagnosis_root / "base_generated_sequential"
        self.updated_root = self.diagnosis_root / "updated_generated_sequential_replay"
        self.base_root.mkdir(parents=True)
        self.updated_root.mkdir(parents=True)
        self._write_diagnosis_root()

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def test_audit_splits_raw_probe_rejections_from_controlled_cumulative_regression(self) -> None:
        from scripts.run_generated_sequential_gate_metric_accounting_audit import (
            run_generated_sequential_gate_metric_accounting_audit,
        )

        summary = run_generated_sequential_gate_metric_accounting_audit(
            diagnosis_root=self.diagnosis_root,
            output_root=self.output_root,
            config=self._config(),
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["reason_codes"], [])
        self.assertEqual(summary["legacy_mismatch_count"], 6)
        self.assertEqual(summary["raw_policy_path_cost_regression_count"], 6)
        self.assertEqual(summary["raw_policy_risk_regression_count"], 2)
        self.assertEqual(summary["controlled_path_cost_regression_count"], 0)
        self.assertEqual(summary["controlled_risk_regression_count"], 0)
        self.assertEqual(
            summary["diagnosis_verdict_after_origin_split"],
            "pre_existing_generated_sequential_contract_mismatch",
        )
        self.assertEqual(
            summary["recommended_next_action"],
            "generated_sequential_contract_alignment_required",
        )

        corrected = self._load_json(self.output_root / "corrected-accounting-shadow-summary.json")
        self.assertEqual(corrected["cumulative_path_cost_regression_count"], 0)
        self.assertEqual(corrected["cumulative_risk_regression_count"], 0)
        self.assertEqual(corrected["canary_rejected_policy_choice_count"], 6)
        self.assertEqual(corrected["status"], "failed")
        self.assertIn("canary_rejected_policy_choice_count_above_threshold", corrected["reason_codes"])

        mismatch_rows = self._read_jsonl(self.output_root / "legacy-mismatch-rows.jsonl")
        self.assertEqual(len(mismatch_rows), 6)
        self.assertTrue(all(row["raw_policy_probe_regression"] for row in mismatch_rows))
        self.assertTrue(all(row["controlled_rollout_regression"] is False for row in mismatch_rows))

        accounting_rows = self._read_jsonl(self.output_root / "origin-aware-failed-step-accounting.jsonl")
        self.assertEqual(len(accounting_rows), 6)
        self.assertEqual(accounting_rows[0]["reason_origin"], "raw_policy_probe")

        report = (self.output_root / "generated-sequential-gate-metric-accounting-report.md").read_text()
        self.assertIn("channel_contrast", report)
        self.assertIn("generated_sequential_contract_alignment_required", report)

    def _write_diagnosis_root(self) -> None:
        failed_steps = self._failed_steps()
        for root in (self.base_root, self.updated_root):
            self._write_json(
                root / "policy-gated-sequential-canary-rollout-summary.json",
                {
                    "schema_version": "policy-gated-sequential-canary-rollout-summary/v1",
                    "status": "failed",
                    "reason_codes": [
                        "canary_rejected_policy_choice_count_above_threshold",
                        "cumulative_path_cost_regression_count_above_threshold",
                        "cumulative_risk_regression_count_above_threshold",
                    ],
                    "episode_count": 36,
                    "step_count": 108,
                    "canary_rejected_policy_choice_count": 6,
                    "cumulative_path_cost_regression_count": 6,
                    "cumulative_risk_regression_count": 2,
                    "git_provenance": {"current_matches_sources": True},
                },
            )
            self._write_json(
                root / "policy-gated-sequential-canary-rejection-report.json",
                {
                    "schema_version": "policy-gated-sequential-canary-rejection-report/v1",
                    "failed_steps": failed_steps,
                    "canary_rejection_reason_counts": {
                        "path_cost_regression": 6,
                        "risk_regression": 2,
                    },
                },
            )
            (root / "policy-gated-sequential-canary-steps.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in failed_steps),
                encoding="utf-8",
            )

        comparison_rows = []
        for step in failed_steps:
            comparison_rows.append(
                {
                    "episode_id": step["episode_id"],
                    "step_index": step["step_index"],
                    "scenario_group": step["scenario_group"],
                    "base_failed": True,
                    "updated_failed": True,
                    "base_reasons": step["canary_rejection_reason_codes"],
                    "updated_reasons": step["canary_rejection_reason_codes"],
                    "base_policy_path_cost_delta": 0.0,
                    "updated_policy_path_cost_delta": 0.0,
                    "base_policy_risk_delta": 0.0,
                    "updated_policy_risk_delta": 0.0,
                    "base_raw_policy_path_cost_delta": step["raw_policy_selected_path_cost_delta"],
                    "updated_raw_policy_path_cost_delta": step["raw_policy_selected_path_cost_delta"],
                    "base_raw_policy_risk_delta": step["raw_policy_selected_risk_delta"],
                    "updated_raw_policy_risk_delta": step["raw_policy_selected_risk_delta"],
                }
            )
        (self.diagnosis_root / "failed-step-comparison.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in comparison_rows),
            encoding="utf-8",
        )
        self._write_json(
            self.diagnosis_root / "quasi-real-generated-sequential-contract-compatibility-summary.json",
            {
                "schema_version": "quasi-real-generated-sequential-contract-compatibility-summary/v1",
                "status": "passed",
                "reason_codes": [],
                "diagnosis_verdict": "gate_accounting_or_metric_mismatch",
                "recommended_next_action": "generated_sequential_gate_metric_audit_required",
                "failed_step_count": 6,
                "base_failed_step_count": 6,
                "updated_failed_step_count": 6,
                "gate_metric_mismatch_count": 6,
                "base_generated_sequential_status": "failed",
                "updated_generated_sequential_status": "failed",
                "base_generated_sequential_root": str(self.base_root),
                "updated_generated_sequential_root": str(self.updated_root),
                "failed_step_comparison": str(self.diagnosis_root / "failed-step-comparison.jsonl"),
                "quasi_real_teacher_following_status": "passed",
                "quasi_real_collector_status": "passed",
                "publishes_checkpoint": False,
                "replaces_default_policy": False,
                "performance_claimed": False,
                "formal_training_ready_claimed": False,
                "git_provenance": {"current_matches_sources": True},
            },
        )

    def _failed_steps(self) -> list[dict]:
        specs = [
            ("seq-channel_contrast-a", 1, "channel_contrast", ["path_cost_regression"]),
            ("seq-channel_contrast-a", 2, "channel_contrast", ["path_cost_regression"]),
            ("seq-channel_contrast-b", 0, "channel_contrast", ["path_cost_regression", "risk_regression"]),
            ("seq-high_risk_tradeoff-e", 2, "high_risk_tradeoff", ["path_cost_regression"]),
            ("seq-mixed_stress_detour-a", 1, "mixed_stress_detour", ["path_cost_regression"]),
            ("seq-near_blocked_safe_alt-f", 0, "near_blocked_safe_alt", ["path_cost_regression", "risk_regression"]),
        ]
        rows = []
        for index, (episode_id, step_index, family, reasons) in enumerate(specs):
            rows.append(
                {
                    "schema_version": "policy-gated-sequential-canary-step/v1",
                    "episode_id": episode_id,
                    "step_index": step_index,
                    "scenario_group": family,
                    "decision_class": "canary_rejected_policy_choice",
                    "controlled_choice_source": "source_fallback",
                    "canary_rejection_reason_codes": reasons,
                    "raw_policy_regression_reason_codes": reasons,
                    "controlled_regression_reason_codes": [],
                    "raw_policy_selected_path_cost_delta": 1.0 + index,
                    "raw_policy_selected_risk_delta": 0.5 if "risk_regression" in reasons else -0.1,
                    "policy_selected_path_cost_delta": 0.0,
                    "policy_selected_risk_delta": 0.0,
                }
            )
        return rows

    def _config(self) -> dict:
        return {
            "schema_version": "generated-sequential-gate-metric-accounting-audit-config/v1",
            "input_files": {
                "diagnosis_summary": "quasi-real-generated-sequential-contract-compatibility-summary.json",
                "failed_step_comparison": "failed-step-comparison.jsonl",
                "base_rejection_report": "base_generated_sequential/policy-gated-sequential-canary-rejection-report.json",
                "updated_rejection_report": "updated_generated_sequential_replay/policy-gated-sequential-canary-rejection-report.json",
            },
            "output_files": {
                "summary": "generated-sequential-gate-metric-accounting-audit-summary.json",
                "legacy_mismatch_rows": "legacy-mismatch-rows.jsonl",
                "origin_aware_failed_step_accounting": "origin-aware-failed-step-accounting.jsonl",
                "corrected_shadow_summary": "corrected-accounting-shadow-summary.json",
                "report": "generated-sequential-gate-metric-accounting-report.md",
            },
            "evaluation": {
                "max_path_cost_regression": 0.0,
                "max_risk_regression": 0.0,
            },
            "validation": {
                "expected_legacy_mismatch_count": 6,
            },
        }

    def _write_json(self, path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _load_json(self, path: Path) -> dict:
        return json.loads(path.read_text(encoding="utf-8"))

    def _read_jsonl(self, path: Path) -> list[dict]:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
