import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path


class FamilyBalancedShadowCanaryPreflightTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        scripts = str(self.repo_root / "scripts")
        if scripts not in sys.path:
            sys.path.insert(0, scripts)
        self.temp_dir = Path(tempfile.mkdtemp(prefix="family-balanced-shadow-canary-"))
        self.rerun_root = self.temp_dir / "rerun"
        self.gap_root = self.temp_dir / "gap"
        self.formal_root = self.temp_dir / "formal"
        self.replay_root = self.temp_dir / "post-training-replay"
        self.selected_root = self.temp_dir / "selected"
        self.output_root = self.temp_dir / "output"
        for path in (self.rerun_root, self.gap_root, self.formal_root, self.replay_root, self.selected_root):
            path.mkdir(parents=True)
        self._write_docs()

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def test_preflight_passes_with_family_balanced_long_horizon_shadow(self) -> None:
        from scripts.run_family_balanced_shadow_canary_preflight import (
            run_family_balanced_shadow_canary_preflight,
        )

        self._write_stage_inputs(row_count_per_family=4)

        summary = run_family_balanced_shadow_canary_preflight(
            family_balanced_rerun_root=self.rerun_root,
            family_balanced_gap_root=self.gap_root,
            formal_training_root=self.formal_root,
            post_training_replay_root=self.replay_root,
            selected_candidate_root=self.selected_root,
            output_root=self.output_root,
            repo_root=self.temp_dir,
            horizons=(2, 4),
        )

        self.assertEqual(summary["schema_version"], "family-balanced-shadow-canary-preflight-summary/v1")
        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["reason_codes"], [])
        self.assertEqual(
            summary["preflight_verdict"],
            "eligible_for_family_balanced_formal_performance_claim_release_decision",
        )
        self.assertTrue(summary["family_balanced_shadow_canary_preflight_passed"])
        self.assertTrue(summary["family_balanced_formal_performance_claim_release_decision_approved"])
        self.assertTrue(summary["long_horizon_shadow_passed"])
        self.assertEqual(summary["horizons"], [2, 4])
        self.assertGreater(summary["coverage_return_improvement"], 0.0)
        self.assertGreater(summary["cumulative_coverage_rate_delta_improvement"], 0.0)
        self.assertGreater(summary["valuable_area_covered_improvement"], 0.0)
        self.assertFalse(summary["coverage_efficiency_regression"])
        self.assertEqual(summary["safe_better_training_family_count"], 4)
        self.assertTrue(summary["low_observation_shadow_passed"])
        self.assertTrue(summary["family_generalization_audit_passed"])
        self.assertLess(summary["fallback_rate"], 0.5)
        self.assertEqual(summary["fallback_gain_contamination_count"], 0)
        self.assertEqual(summary["controlled_regression_count"], 0)
        self.assertTrue(summary["kill_switch_audit_passed"])
        self.assertTrue(summary["rollback_audit_passed"])
        self.assertTrue(summary["telemetry_audit_passed"])
        self.assertFalse(summary["shadow_policy_takes_control"])
        self.assertEqual(summary["experimental_control_activation_count"], 0)
        self.assertEqual(summary["next_required_change"], "family_balanced_formal_performance_claim_release_decision")
        self.assertFalse(summary["checkpoint_publication_approved"])
        self.assertFalse(summary["default_policy_replacement_approved"])
        self.assertFalse(summary["real_executor_connection_approved"])
        self.assertFalse(summary["publishes_checkpoint"])
        self.assertFalse(summary["replaces_default_policy"])
        self.assertFalse(summary["connects_real_executor"])
        self.assertFalse(summary["performance_claimed"])

        for field in (
            "runtime_manifest",
            "shadow_step_comparison",
            "long_horizon_shadow_validation",
            "family_generalization_audit",
            "coverage_efficiency_audit",
            "guard_fallback_audit",
            "kill_switch_audit",
            "rollback_audit",
            "telemetry_audit",
            "lineage_audit",
            "release_boundary_audit",
            "rejection_report",
            "report",
        ):
            self.assertTrue(Path(summary[field]).is_file(), field)

        step_rows = self._read_jsonl(Path(summary["shadow_step_comparison"]))
        self.assertEqual(len(step_rows), 16)
        self.assertTrue(all(row["shadow_policy_takes_control"] is False for row in step_rows))
        self.assertTrue(all(row["experimental_control_activation"] is False for row in step_rows))
        self.assertTrue(
            all(
                {"coverage_rate_delta", "path_cost", "risk", "energy_cost", "fallback_like", "activation"}
                <= set(row["telemetry_fields_recorded"])
                for row in step_rows
            )
        )

        family_audit = json.loads(Path(summary["family_generalization_audit"]).read_text(encoding="utf-8"))
        self.assertTrue(family_audit["family_generalization_audit_passed"])
        self.assertTrue(family_audit["low_observation_shadow_passed"])
        self.assertEqual(set(family_audit["family_rollups"]), set(self._families()))
        self.assertGreater(family_audit["family_rollups"]["low_observation_count"]["candidate_coverage_return"], 0.0)

    def test_rerun_failure_blocks_preflight(self) -> None:
        from scripts.run_family_balanced_shadow_canary_preflight import (
            run_family_balanced_shadow_canary_preflight,
        )

        self._write_stage_inputs(row_count_per_family=4)
        summary_path = self.rerun_root / "family-balanced-coverage-driven-ppo-rerun-summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["status"] = "failed"
        summary["reason_codes"] = ["coverage_efficiency_regression"]
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        result = run_family_balanced_shadow_canary_preflight(
            family_balanced_rerun_root=self.rerun_root,
            family_balanced_gap_root=self.gap_root,
            formal_training_root=self.formal_root,
            post_training_replay_root=self.replay_root,
            selected_candidate_root=self.selected_root,
            output_root=self.output_root,
            repo_root=self.temp_dir,
            horizons=(2, 4),
        )

        self.assertEqual(result["status"], "failed")
        self.assertIn("family_balanced_rerun_not_passed", result["reason_codes"])
        self.assertEqual(result["next_required_change"], "fix_family_balanced_shadow_canary_inputs")

    def test_low_observation_regression_blocks_preflight(self) -> None:
        from scripts.run_family_balanced_shadow_canary_preflight import (
            run_family_balanced_shadow_canary_preflight,
        )

        self._write_stage_inputs(row_count_per_family=4, low_observation_candidate_delta=-0.08)

        result = run_family_balanced_shadow_canary_preflight(
            family_balanced_rerun_root=self.rerun_root,
            family_balanced_gap_root=self.gap_root,
            formal_training_root=self.formal_root,
            post_training_replay_root=self.replay_root,
            selected_candidate_root=self.selected_root,
            output_root=self.output_root,
            repo_root=self.temp_dir,
            horizons=(2, 4),
        )

        self.assertEqual(result["status"], "failed")
        self.assertIn("low_observation_shadow_regression", result["reason_codes"])
        self.assertIn("family_generalization_regression", result["reason_codes"])

    def test_shell_docs_and_spec_contract_are_declared(self) -> None:
        shell_path = self.repo_root / "scripts" / "run_family_balanced_shadow_canary_preflight.sh"
        spec_path = (
            self.repo_root
            / "docs"
            / "superpowers"
            / "specs"
            / "2026-06-16-family-balanced-shadow-canary-preflight.md"
        )
        self.assertTrue(shell_path.is_file())
        self.assertTrue(spec_path.is_file())
        spec_text = spec_path.read_text(encoding="utf-8")
        self.assertIn("family-balanced-shadow-canary-preflight-summary.json", spec_text)
        self.assertIn("README.md", spec_text)
        self.assertIn("docs/算法设计与系统架构报告.md", spec_text)
        self.assertIn("不发布 checkpoint", spec_text)
        self.assertIn("不连接真实执行器", spec_text)

    def _write_docs(self) -> None:
        self._write_text(
            self.temp_dir / "README.md",
            "Family-Balanced Shadow/Canary Preflight v1\n"
            "outputs/path_feedback_batch_family_balanced_shadow_canary_preflight_v1/\n"
            "family_balanced_formal_performance_claim_release_decision\n"
            "checkpoint_publication_approved=false\n"
            "default_policy_replacement_approved=false\n"
            "real_executor_connection_approved=false\n",
        )
        self._write_text(
            self.temp_dir / "docs" / "算法设计与系统架构报告.md",
            "Family-Balanced Shadow/Canary Preflight v1\n"
            "离线 shadow/canary 预检\n"
            "不发布 checkpoint、不替换 default policy、不连接真实执行器\n",
        )
        self._write_text(
            self.temp_dir
            / "docs"
            / "superpowers"
            / "specs"
            / "2026-06-16-family-balanced-shadow-canary-preflight.md",
            "Family-Balanced Shadow/Canary Preflight v1\n"
            "family-balanced-shadow-canary-preflight-summary.json\n"
            "README.md\n"
            "docs/算法设计与系统架构报告.md\n"
            "不发布 checkpoint\n"
            "不连接真实执行器\n",
        )

    def _write_stage_inputs(self, *, row_count_per_family: int, low_observation_candidate_delta: float = 0.04) -> None:
        filtered_rows = []
        replay_rows = []
        refined_rows = []
        for family_index, family in enumerate(self._families()):
            for row_index in range(row_count_per_family):
                index = family_index * row_count_per_family + row_index
                row = self._filtered_row(
                    index=index,
                    family=family,
                    candidate_delta=low_observation_candidate_delta if family == "low_observation_count" else 0.04,
                )
                fallback = row_index == row_count_per_family - 1
                filtered_rows.append(row)
                replay_rows.append(self._replay_row(index=index, source=row, fallback=fallback))
                refined_rows.append(self._transition(row))

        self._write_jsonl(self.rerun_root / "family-balanced-safe-better-pairs.jsonl", filtered_rows)
        self._write_jsonl(self.rerun_root / "refined-coverage-ppo-batch" / "refined-trainable-transitions.jsonl", refined_rows)
        self._write_json(
            self.rerun_root / "coverage-driven-ppo-replay-audit.json",
            {
                "schema_version": "coverage-driven-ppo-replay-audit/v1",
                "status": "passed",
                "reason_codes": [],
                "raw_policy_action_count": len(replay_rows),
                "accepted_policy_action_count": sum(1 for row in replay_rows if row["policy_action_accepted"]),
                "guard_rejected_action_count": sum(1 for row in replay_rows if row["guard_rejected"]),
                "rows": replay_rows,
            },
        )
        metric_rows = [
            self._metric_row("teacher", filtered_rows, replay_rows, role="teacher"),
            self._metric_row("post_improvement_ppo", filtered_rows, replay_rows, role="candidate"),
        ]
        self._write_jsonl(self.rerun_root / "refined-coverage-driven-ppo-performance-metric-table.jsonl", metric_rows)
        compat_dir = self.rerun_root / "family-balanced-compatible-performance-evaluation"
        compat_dir.mkdir(parents=True)
        self._write_jsonl(compat_dir / "coverage-performance-metric-table.jsonl", [metric_rows[0]])
        self._write_json(
            compat_dir / "exploration-coverage-performance-evaluation-summary.json",
            {
                "schema_version": "family-balanced-compatible-performance-evaluation-summary/v1",
                "status": "passed",
                "reason_codes": [],
                "metric_table": str(compat_dir / "coverage-performance-metric-table.jsonl"),
                "performance_claimed": False,
            },
        )
        self._write_json(
            self.rerun_root / "stage5a-rerun-summary.json",
            {
                "schema_version": "policy-coverage-opportunity-margin-audit-summary/v1",
                "status": "passed",
                "reason_codes": [],
                "safe_better_than_teacher_family_count": 4,
                "policy_argmax_changed_count": len(filtered_rows),
                "controlled_regression_count": 0,
                "performance_claimed": False,
            },
        )
        self._write_json(
            self.rerun_root / "coverage-driven-experimental-policy-candidate-metadata.json",
            {
                "schema_version": "experimental-policy-candidate-metadata/v1",
                "experimental": True,
                "publishes_checkpoint": False,
                "replaces_default_policy": False,
                "connects_real_executor": False,
                "performance_claimed": False,
            },
        )
        self._write_json(
            self.rerun_root / "family-balanced-coverage-driven-ppo-rerun-summary.json",
            {
                "schema_version": "family-balanced-coverage-driven-ppo-rerun-summary/v1",
                "status": "passed",
                "reason_codes": [],
                "rerun_verdict": "eligible_for_family_balanced_shadow_canary_preflight",
                "family_balanced_coverage_driven_ppo_rerun_passed": True,
                "family_balanced_shadow_canary_preflight_approved": True,
                "next_required_change": "family_balanced_shadow_canary_preflight",
                "compatible_performance_metric_table": str(compat_dir / "coverage-performance-metric-table.jsonl"),
                "compatible_performance_root": str(compat_dir),
                "family_balanced_safe_better_pairs": str(self.rerun_root / "family-balanced-safe-better-pairs.jsonl"),
                "refined_summary": str(self.rerun_root / "refined-coverage-driven-ppo-improvement-run-summary.json"),
                "guard_replay_audit": str(self.rerun_root / "coverage-driven-ppo-replay-audit.json"),
                "performance_metric_table": str(self.rerun_root / "refined-coverage-driven-ppo-performance-metric-table.jsonl"),
                "stage5a_rerun_summary": str(self.rerun_root / "stage5a-rerun-summary.json"),
                "refined_trainable_transitions": str(
                    self.rerun_root / "refined-coverage-ppo-batch" / "refined-trainable-transitions.jsonl"
                ),
                "checkpoint_metadata_path": str(self.rerun_root / "coverage-driven-experimental-policy-candidate-metadata.json"),
                "coverage_return_improvement": 1.0,
                "cumulative_coverage_rate_delta_improvement": 1.0,
                "valuable_area_covered_improvement": 1.0,
                "coverage_efficiency_regression": False,
                "fallback_rate": sum(1 for row in replay_rows if row["fallback_like"]) / len(replay_rows),
                "controlled_regression_count": 0,
                "fallback_gain_contamination_count": 0,
                "safe_better_training_family_count": 4,
                "low_observation_trainable_transition_count": row_count_per_family,
                "checkpoint_publication_approved": False,
                "default_policy_replacement_approved": False,
                "real_executor_connection_approved": False,
                "publishes_checkpoint": False,
                "replaces_default_policy": False,
                "connects_real_executor": False,
                "performance_claimed": False,
                "relaxes_guard": False,
                "modifies_network_or_action_space": False,
                "modifies_default_astar": False,
            },
        )
        self._write_json(
            self.rerun_root / "refined-coverage-driven-ppo-improvement-run-summary.json",
            {
                "schema_version": "refined-coverage-driven-ppo-improvement-run-summary/v2",
                "status": "passed",
                "reason_codes": [],
                "guard_replay_audit": str(self.rerun_root / "coverage-driven-ppo-replay-audit.json"),
                "performance_metric_table": str(self.rerun_root / "refined-coverage-driven-ppo-performance-metric-table.jsonl"),
                "coverage_return_improvement": 1.0,
                "coverage_efficiency_regression": False,
                "publishes_checkpoint": False,
                "replaces_default_policy": False,
                "connects_real_executor": False,
                "performance_claimed": False,
            },
        )
        self._write_json(
            self.gap_root / "family-balanced-algorithm-gap-closure-summary.json",
            {
                "schema_version": "family-balanced-algorithm-gap-closure-summary/v1",
                "status": "passed",
                "reason_codes": [],
                "family_balanced_algorithm_gap_closure_passed": True,
                "family_safe_better_counts": {family: row_count_per_family for family in self._families()},
                "low_observation_count": row_count_per_family,
                "next_required_change": "family_balanced_coverage_driven_ppo_rerun",
                "publishes_checkpoint": False,
                "replaces_default_policy": False,
                "connects_real_executor": False,
            },
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

    def _filtered_row(self, *, index: int, family: str, candidate_delta: float) -> dict:
        teacher_delta = 0.05 + index * 0.001
        candidate_gain = teacher_delta + candidate_delta
        return {
            "schema_version": "family-balanced-safe-better-pair-row/v1",
            "context_id": f"ctx-{index}",
            "episode_id": f"episode-{index // 2}",
            "step_index": index,
            "scenario_id": f"scenario-{index}",
            "scenario_family": family,
            "split": "train",
            "candidate_action_index": index % 6,
            "teacher_action_index": (index + 1) % 6,
            "expected_coverage_rate_delta": candidate_gain,
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
            "family_balanced_ppo_advantage": 0.02,
        }

    def _replay_row(self, *, index: int, source: dict, fallback: bool) -> dict:
        accepted = not fallback
        return {
            "schema_version": "coverage-driven-ppo-replay-row/v1",
            "actor": "post_improvement_ppo",
            "replay_index": index,
            "context_id": source["context_id"],
            "episode_id": source["episode_id"],
            "step_index": source["step_index"],
            "scenario_id": source["scenario_id"],
            "scenario_family": source["scenario_family"],
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
            coverage_return = sum(row["teacher_expected_coverage_rate_delta"] * (0.99 ** row["step_index"]) for row in filtered_rows)
            activation = 0.0
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
            "coverage_gain_per_path_cost": total_gain / path if path else None,
            "coverage_gain_per_risk": total_gain / risk if risk else None,
            "coverage_gain_per_energy": total_gain / energy if energy else None,
            "accepted_policy_activation_rate": activation,
            "fallback_rate": fallback,
            "teacher_agreement_rate": teacher_agreement,
            "controlled_regression_count": 0,
            "fallback_coverage_gain": 0.0,
            "scenario_family_gain_count": 4,
            "scenario_families_with_gain": list(self._families()),
            "comparator_basis": ["same_family_balanced_decision_set"],
        }

    def _families(self) -> tuple[str, ...]:
        return ("low_observation_count", "mixed_risk", "rim_or_steep_slope", "smooth_high_confidence")

    def _write_json(self, path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def _write_jsonl(self, path: Path, rows: list[dict]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")

    def _write_text(self, path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def _read_jsonl(self, path: Path) -> list[dict]:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


if __name__ == "__main__":
    unittest.main()
