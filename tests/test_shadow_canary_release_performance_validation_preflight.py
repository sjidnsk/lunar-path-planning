import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path


class ShadowCanaryReleasePerformanceValidationPreflightTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        scripts = str(self.repo_root / "scripts")
        if scripts not in sys.path:
            sys.path.insert(0, scripts)
        self.temp_dir = Path(tempfile.mkdtemp(prefix="shadow-canary-preflight-"))
        self.cost_root = self.temp_dir / "cost-efficiency"
        self.formal_root = self.temp_dir / "formal"
        self.replay_root = self.temp_dir / "post-training-replay"
        self.selected_root = self.temp_dir / "selected"
        self.output_root = self.temp_dir / "output"
        for path in (self.cost_root, self.formal_root, self.replay_root, self.selected_root):
            path.mkdir(parents=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def test_preflight_passes_with_long_horizon_shadow_guard_rollback_and_telemetry(self) -> None:
        from scripts.run_shadow_canary_release_performance_validation_preflight import (
            run_shadow_canary_release_performance_validation_preflight,
        )

        self._write_stage6_inputs(row_count_per_family=3)

        summary = run_shadow_canary_release_performance_validation_preflight(
            cost_efficiency_root=self.cost_root,
            formal_training_root=self.formal_root,
            post_training_replay_root=self.replay_root,
            selected_candidate_root=self.selected_root,
            output_root=self.output_root,
            repo_root=self.repo_root,
            horizons=(2, 3),
        )

        self.assertEqual(
            summary["schema_version"],
            "shadow-canary-release-performance-validation-preflight-summary/v1",
        )
        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["reason_codes"], [])
        self.assertEqual(summary["next_required_change"], "formal_performance_claim_release_decision")
        self.assertEqual(summary["horizons"], [2, 3])
        self.assertTrue(summary["long_horizon_shadow_passed"])
        self.assertGreater(summary["coverage_return_improvement"], 0.0)
        self.assertGreater(summary["cumulative_coverage_rate_delta_improvement"], 0.0)
        self.assertGreater(summary["valuable_area_covered_improvement"], 0.0)
        self.assertFalse(summary["coverage_efficiency_regression"])
        self.assertLess(summary["fallback_rate"], 0.5)
        self.assertEqual(summary["safe_better_training_family_count"], 4)
        self.assertEqual(summary["controlled_regression_count"], 0)
        self.assertEqual(summary["fallback_gain_contamination_count"], 0)
        self.assertFalse(summary["shadow_policy_takes_control"])
        self.assertEqual(summary["experimental_control_activation_count"], 0)
        self.assertTrue(summary["kill_switch_audit_passed"])
        self.assertTrue(summary["rollback_audit_passed"])
        self.assertTrue(summary["telemetry_audit_passed"])
        self.assertFalse(summary["publishes_checkpoint"])
        self.assertFalse(summary["replaces_default_policy"])
        self.assertFalse(summary["connects_real_executor"])
        self.assertFalse(summary["performance_claimed"])
        self.assertFalse(summary["uses_old_stage5a2_pairs_as_current_model_evidence"])

        for field in (
            "runtime_manifest",
            "shadow_step_comparison",
            "long_horizon_shadow_validation",
            "coverage_efficiency_audit",
            "guard_fallback_audit",
            "kill_switch_audit",
            "rollback_audit",
            "telemetry_audit",
            "eligibility_ledger",
            "rejection_report",
            "report",
        ):
            self.assertTrue(Path(summary[field]).is_file(), field)

        step_rows = self._read_jsonl(Path(summary["shadow_step_comparison"]))
        self.assertEqual(len(step_rows), 12)
        self.assertTrue(all(row["shadow_policy_takes_control"] is False for row in step_rows))
        self.assertTrue(all(row["experimental_control_activation"] is False for row in step_rows))
        self.assertTrue(
            all(
                {"coverage_rate_delta", "path_cost", "risk", "energy_cost", "fallback_like", "activation"}
                <= set(row["telemetry_fields_recorded"])
                for row in step_rows
            )
        )

        long_horizon = json.loads(Path(summary["long_horizon_shadow_validation"]).read_text(encoding="utf-8"))
        self.assertEqual(long_horizon["horizons"], [2, 3])
        self.assertTrue(long_horizon["long_horizon_shadow_passed"])
        self.assertTrue(
            any(
                row["horizon"] == 3 and row["group_type"] == "all" and row["coverage_return_improvement"] > 0.0
                for row in long_horizon["rollups"]
            )
        )

    def test_input_cost_efficiency_stage_failure_blocks_preflight(self) -> None:
        from scripts.run_shadow_canary_release_performance_validation_preflight import (
            run_shadow_canary_release_performance_validation_preflight,
        )

        self._write_stage6_inputs(row_count_per_family=3)
        summary_path = self.cost_root / "cost-efficiency-aware-coverage-reward-candidate-filter-summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["status"] = "failed"
        summary["reason_codes"] = ["coverage_efficiency_regression"]
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        result = run_shadow_canary_release_performance_validation_preflight(
            cost_efficiency_root=self.cost_root,
            formal_training_root=self.formal_root,
            post_training_replay_root=self.replay_root,
            selected_candidate_root=self.selected_root,
            output_root=self.output_root,
            repo_root=self.repo_root,
            horizons=(2, 3),
        )

        self.assertEqual(result["status"], "failed")
        self.assertIn("input_cost_efficiency_stage_not_passed", result["reason_codes"])
        self.assertEqual(result["next_required_change"], "fix_cost_efficiency_stage_inputs")
        self.assertFalse(result["performance_claimed"])

    def test_stage5a_rerun_diagnostic_failure_does_not_override_cost_efficiency_pass(self) -> None:
        from scripts.run_shadow_canary_release_performance_validation_preflight import (
            run_shadow_canary_release_performance_validation_preflight,
        )

        self._write_stage6_inputs(row_count_per_family=3)
        stage5a_path = self.cost_root / "stage5a-rerun-summary.json"
        stage5a = json.loads(stage5a_path.read_text(encoding="utf-8"))
        stage5a["status"] = "failed"
        stage5a["reason_codes"] = ["fallback_gain_contamination_present"]
        stage5a_path.write_text(json.dumps(stage5a, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        result = run_shadow_canary_release_performance_validation_preflight(
            cost_efficiency_root=self.cost_root,
            formal_training_root=self.formal_root,
            post_training_replay_root=self.replay_root,
            selected_candidate_root=self.selected_root,
            output_root=self.output_root,
            repo_root=self.repo_root,
            horizons=(2, 3),
        )

        self.assertEqual(result["status"], "passed")
        self.assertNotIn("input_cost_efficiency_stage_not_passed", result["reason_codes"])
        self.assertEqual(result["stage5a_rerun_status"], "failed")

    def test_shell_docs_and_spec_contract_are_declared(self) -> None:
        shell_path = self.repo_root / "scripts" / "run_shadow_canary_release_performance_validation_preflight.sh"
        spec_path = (
            self.repo_root
            / "docs"
            / "superpowers"
            / "specs"
            / "2026-06-15-shadow-canary-release-performance-validation-preflight.md"
        )
        self.assertTrue(shell_path.is_file())
        self.assertTrue(spec_path.is_file())
        spec_text = spec_path.read_text(encoding="utf-8")
        self.assertIn("shadow-canary-release-performance-validation-preflight-summary.json", spec_text)
        self.assertIn("README.md", spec_text)
        self.assertIn("docs/算法设计与系统架构报告.md", spec_text)
        self.assertIn("不发布 checkpoint", spec_text)
        self.assertIn("不连接真实执行器", spec_text)

    def _write_stage6_inputs(self, *, row_count_per_family: int) -> None:
        rows = []
        replay_rows = []
        families = (
            "low_observation_count",
            "mixed_risk",
            "rim_or_steep_slope",
            "smooth_high_confidence",
        )
        for family_index, family in enumerate(families):
            for row_index in range(row_count_per_family):
                index = family_index * row_count_per_family + row_index
                fallback = row_index == row_count_per_family - 1
                row = self._filtered_row(index=index, family=family)
                rows.append(row)
                replay_rows.append(self._replay_row(index=index, family=family, source=row, fallback=fallback))

        self._write_jsonl(self.cost_root / "cost-efficiency-filtered-batch.jsonl", rows)
        self._write_jsonl(self.cost_root / "refined-trainable-transitions.jsonl", [self._transition(row) for row in rows])
        self._write_json(
            self.cost_root / "coverage-driven-ppo-replay-audit.json",
            {
                "schema_version": "coverage-driven-ppo-replay-audit/v1",
                "reason_codes": [],
                "raw_policy_action_count": len(replay_rows),
                "accepted_policy_action_count": sum(1 for row in replay_rows if row["policy_action_accepted"]),
                "guard_rejected_action_count": sum(1 for row in replay_rows if row["guard_rejected"]),
                "rows": replay_rows,
            },
        )
        metric_rows = [
            self._metric_row("teacher", rows, replay_rows, role="teacher"),
            self._metric_row("pre_improvement_selected_ppo", rows, replay_rows, role="teacher"),
            self._metric_row("post_improvement_ppo", rows, replay_rows, role="candidate"),
        ]
        self._write_jsonl(self.cost_root / "refined-coverage-driven-ppo-performance-metric-table.jsonl", metric_rows)
        compat_dir = self.cost_root / "cost-efficiency-comparable-performance-input"
        compat_dir.mkdir()
        self._write_jsonl(
            compat_dir / "coverage-performance-metric-table.jsonl",
            [metric_rows[0], metric_rows[1]],
        )
        self._write_json(
            compat_dir / "exploration-coverage-performance-evaluation-summary.json",
            {"schema_version": "exploration-coverage-performance-evaluation-summary/v1", "status": "passed"},
        )
        self._write_json(
            self.cost_root / "stage5a-rerun-summary.json",
            {
                "schema_version": "policy-coverage-opportunity-margin-audit-summary/v1",
                "status": "passed",
                "reason_codes": [],
                "safe_better_than_teacher_family_count": 4,
                "controlled_regression_count": 0,
                "performance_claimed": False,
            },
        )
        cost_summary = {
            "schema_version": "cost-efficiency-aware-coverage-reward-candidate-filter-summary/v1",
            "status": "passed",
            "reason_codes": [],
            "next_required_change": "shadow_canary_release_performance_validation_preflight",
            "filtered_batch": str(self.cost_root / "cost-efficiency-filtered-batch.jsonl"),
            "refined_trainable_transitions": str(self.cost_root / "refined-trainable-transitions.jsonl"),
            "guard_replay_audit": str(self.cost_root / "coverage-driven-ppo-replay-audit.json"),
            "performance_metric_table": str(self.cost_root / "refined-coverage-driven-ppo-performance-metric-table.jsonl"),
            "stage5a_rerun_summary": str(self.cost_root / "stage5a-rerun-summary.json"),
            "compatible_performance_metric_table": str(compat_dir / "coverage-performance-metric-table.jsonl"),
            "compatible_performance_summary": str(compat_dir / "exploration-coverage-performance-evaluation-summary.json"),
            "trainable_pair_count": len(rows),
            "safe_better_training_pair_count": len(rows),
            "safe_better_training_family_count": 4,
            "coverage_return_improvement": 1.0,
            "cumulative_coverage_rate_delta_improvement": 2.0,
            "valuable_area_covered_improvement": 3.0,
            "coverage_efficiency_regression": False,
            "fallback_rate": sum(1 for row in replay_rows if row["fallback_like"]) / len(replay_rows),
            "controlled_regression_count": 0,
            "fallback_gain_contamination_count": 0,
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "performance_claimed": False,
            "connects_real_executor": False,
            "relaxes_guard": False,
            "modifies_network_or_action_space": False,
            "modifies_default_astar": False,
        }
        self._write_json(
            self.cost_root / "cost-efficiency-aware-coverage-reward-candidate-filter-summary.json",
            cost_summary,
        )
        self._write_json(
            self.formal_root / "formal-ppo-training-run-summary.json",
            {"schema_version": "formal-ppo-training-run-summary/v1", "status": "passed", "reason_codes": [], "controlled_regression_count": 0, "performance_claimed": False},
        )
        self._write_json(
            self.replay_root / "formal-ppo-post-training-stability-replay-summary.json",
            {"schema_version": "formal-ppo-post-training-stability-replay-summary/v1", "status": "passed", "reason_codes": [], "controlled_regression_count": 0, "performance_claimed": False},
        )
        self._write_json(
            self.selected_root / "selected-formal-ppo-candidate-promotion-preflight-summary.json",
            {"schema_version": "selected-formal-ppo-candidate-promotion-preflight-summary/v1", "status": "passed", "reason_codes": [], "controlled_regression_count": 0, "publishes_checkpoint": False, "replaces_default_policy": False, "performance_claimed": False},
        )

    def _filtered_row(self, *, index: int, family: str) -> dict:
        teacher_delta = 0.05 + index * 0.001
        candidate_delta = teacher_delta + 0.03
        return {
            "schema_version": "cost-efficiency-aware-filtered-pair-row/v1",
            "context_id": f"ctx-{index}",
            "episode_id": f"episode-{index // 2}",
            "step_index": index,
            "scenario_id": f"scenario-{index}",
            "scenario_family": family,
            "split": "train",
            "candidate_action_index": index % 6,
            "teacher_action_index": (index + 1) % 6,
            "expected_coverage_rate_delta": candidate_delta,
            "teacher_expected_coverage_rate_delta": teacher_delta,
            "expected_new_coverage_area": 20.0 + index,
            "teacher_expected_new_coverage_area": 10.0 + index,
            "valuable_coverage_proxy": 2.0 + index * 0.1,
            "teacher_valuable_coverage_proxy": 1.0 + index * 0.1,
            "information_gain": 0.2,
            "teacher_information_gain": 0.1,
            "path_cost": 10.0,
            "teacher_path_cost": 12.0,
            "risk": 1.0,
            "teacher_risk": 1.5,
            "energy_cost": 20.0,
            "teacher_energy_cost": 25.0,
            "coverage_source_available": True,
            "ppo_trainable": True,
            "safe_better_than_teacher_candidate": True,
            "fallback_like": False,
            "guard_rejected": False,
            "controlled_regression_reason_codes": [],
            "cost_efficiency_ppo_advantage": 0.02,
        }

    def _replay_row(self, *, index: int, family: str, source: dict, fallback: bool) -> dict:
        accepted = not fallback
        return {
            "schema_version": "coverage-driven-ppo-replay-row/v1",
            "actor": "post_improvement_ppo",
            "replay_index": index,
            "context_id": source["context_id"],
            "episode_id": source["episode_id"],
            "step_index": source["step_index"],
            "scenario_id": source["scenario_id"],
            "scenario_family": family,
            "raw_policy_action_index": source["candidate_action_index"],
            "controlled_action_index": source["candidate_action_index"],
            "teacher_action_index": source["teacher_action_index"],
            "coverage_rate_delta": 0.0 if fallback else source["expected_coverage_rate_delta"],
            "final_coverage_rate": None if fallback else source["expected_coverage_rate_delta"],
            "new_area_covered": 0.0 if fallback else source["expected_new_coverage_area"],
            "valuable_area_covered": 0.0 if fallback else source["valuable_coverage_proxy"],
            "information_gain": 0.0 if fallback else source["information_gain"],
            "path_cost": 0.0 if fallback else source["path_cost"],
            "risk": 0.0 if fallback else source["risk"],
            "energy_cost": 0.0 if fallback else source["energy_cost"],
            "policy_action_accepted": accepted,
            "fallback_like": fallback,
            "guard_rejected": fallback,
            "controlled_regression_reason_codes": [],
        }

    def _transition(self, row: dict) -> dict:
        return {
            "schema_version": "refined-coverage-trainable-transition/v1",
            "context_id": row["context_id"],
            "episode_id": row["episode_id"],
            "step_index": row["step_index"],
            "scenario_id": row["scenario_id"],
            "scenario_family": row["scenario_family"],
            "action_index": row["candidate_action_index"],
            "teacher_action_index": row["teacher_action_index"],
            "controlled_regression_reason_codes": [],
            "ppo_trainable": True,
        }

    def _metric_row(self, actor: str, filtered_rows: list[dict], replay_rows: list[dict], *, role: str) -> dict:
        if role == "teacher":
            total_gain = sum(row["teacher_expected_coverage_rate_delta"] for row in filtered_rows)
            path = sum(row["teacher_path_cost"] for row in filtered_rows)
            risk = sum(row["teacher_risk"] for row in filtered_rows)
            energy = sum(row["teacher_energy_cost"] for row in filtered_rows)
            valuable = sum(row["teacher_valuable_coverage_proxy"] for row in filtered_rows)
            coverage_return = sum(
                row["teacher_expected_coverage_rate_delta"] * (0.99 ** row["step_index"])
                for row in filtered_rows
            )
            activation = 0.0 if actor == "teacher" else 1.0
            fallback = 0.0
            teacher_agreement = 1.0
        else:
            total_gain = sum(row["coverage_rate_delta"] for row in replay_rows)
            path = sum(row["path_cost"] for row in replay_rows)
            risk = sum(row["risk"] for row in replay_rows)
            energy = sum(row["energy_cost"] for row in replay_rows)
            valuable = sum(row["valuable_area_covered"] for row in replay_rows)
            coverage_return = sum(row["coverage_rate_delta"] * (0.99 ** row["step_index"]) for row in replay_rows)
            activation = sum(1 for row in replay_rows if row["policy_action_accepted"]) / len(replay_rows)
            fallback = sum(1 for row in replay_rows if row["fallback_like"]) / len(replay_rows)
            teacher_agreement = 0.0
        return {
            "schema_version": "coverage-driven-ppo-performance-metric-row/v1",
            "actor": actor,
            "row_count": len(filtered_rows),
            "episode_count": len({row["episode_id"] for row in filtered_rows}),
            "actual_coverage_row_count": len(filtered_rows),
            "coverage_return": coverage_return,
            "cumulative_coverage_rate_delta": total_gain,
            "final_coverage_rate": total_gain / len(filtered_rows),
            "new_area_covered": total_gain * 100.0,
            "valuable_area_covered": valuable,
            "information_gain": 1.0,
            "path_cost": path,
            "risk": risk,
            "energy_cost": energy,
            "coverage_gain_per_path_cost": total_gain / path,
            "coverage_gain_per_risk": total_gain / risk,
            "coverage_gain_per_energy": total_gain / energy,
            "accepted_policy_activation_rate": activation,
            "fallback_rate": fallback,
            "teacher_agreement_rate": teacher_agreement,
            "controlled_regression_count": 0,
            "fallback_coverage_gain": 0.0,
            "scenario_family_gain_count": 4,
            "scenario_families_with_gain": [
                "low_observation_count",
                "mixed_risk",
                "rim_or_steep_slope",
                "smooth_high_confidence",
            ],
            "comparator_basis": ["same_cost_efficiency_filtered_decision_set"],
        }

    def _write_json(self, path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def _write_jsonl(self, path: Path, rows: list[dict]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")

    def _read_jsonl(self, path: Path) -> list[dict]:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
