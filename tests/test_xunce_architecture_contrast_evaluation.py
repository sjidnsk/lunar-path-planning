import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path


class XunceArchitectureContrastEvaluationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(tempfile.mkdtemp(prefix="xunce-contrast-"))
        scripts_path = str(Path(__file__).resolve().parents[1] / "scripts")
        if scripts_path not in sys.path:
            sys.path.insert(0, scripts_path)
        self.stage6_root = self.repo_root / "outputs" / "stage6"
        self.stage4_root = self.repo_root / "outputs" / "stage4"
        self.output_root = self.repo_root / "outputs" / "stage7"
        self.config_path = self.repo_root / "configs" / "xunce_architecture_contrast_evaluation_v1.json"
        self._write_stage6()
        self._write_stage4()
        self._write_config()

    def tearDown(self) -> None:
        shutil.rmtree(self.repo_root)

    def test_default_fixture_passes_and_writes_artifacts(self) -> None:
        from scripts.run_xunce_architecture_contrast_evaluation import run_xunce_architecture_contrast_evaluation

        summary = run_xunce_architecture_contrast_evaluation(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["reason_codes"], [])
        self.assertEqual(summary["source_mechanism_validation_status"], "passed")
        self.assertEqual(summary["next_required_change"], "full_xunce_network_v1_design")
        self.assertEqual(summary["architecture_count"], 4)
        self.assertIn("topology_aware_coverage_graph_proto_v1", summary["evaluated_architectures"])
        self.assertTrue(summary["parameter_efficiency_gate_passed"])
        self.assertTrue(summary["latency_gate_passed"])
        self.assertTrue(summary["ranking_quality_gate_passed"])
        self.assertTrue(summary["architecture_contrast_passed"])
        self.assertIn("sha", summary["git_provenance"]["current"]["parent"])
        self.assertFalse(summary["publishes_checkpoint"])
        self.assertFalse(summary["replaces_default_policy"])
        self.assertFalse(summary["connects_real_executor"])
        self.assertFalse(summary["starts_online_canary"])
        self.assertFalse(summary["runs_new_ppo_update"])
        self.assertFalse(summary["modifies_action_space"])
        self.assertFalse(summary["modifies_default_astar"])

        for filename in (
            "xunce-architecture-contrast-evaluation-summary.json",
            "xunce-architecture-contrast-evaluation-manifest.json",
            "xunce-architecture-contrast-results.jsonl",
            "xunce-architecture-contrast-parameter-latency-audit.json",
            "xunce-architecture-contrast-ranking-audit.json",
            "xunce-architecture-contrast-boundary-audit.json",
            "xunce-architecture-contrast-rejection-report.json",
            "xunce-architecture-contrast-evaluation-report.md",
        ):
            self.assertTrue((self.output_root / filename).is_file(), filename)

    def test_missing_stage6_routes_to_mechanism_fix(self) -> None:
        from scripts.run_xunce_architecture_contrast_evaluation import run_xunce_architecture_contrast_evaluation

        shutil.rmtree(self.stage6_root)
        summary = run_xunce_architecture_contrast_evaluation(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("missing_proto_mechanism_validation_summary", summary["reason_codes"])
        self.assertEqual(summary["next_required_change"], "fix_xunce_proto_mechanism_validation")

    def test_parameter_gate_failure_blocks_complete_network(self) -> None:
        from scripts.run_xunce_architecture_contrast_evaluation import run_xunce_architecture_contrast_evaluation

        self._write_config(max_parameter_ratio_vs_candidate_attention=0.01)
        summary = run_xunce_architecture_contrast_evaluation(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("xunce_parameter_efficiency_gate_failed", summary["reason_codes"])
        self.assertFalse(summary["parameter_efficiency_gate_passed"])
        self.assertEqual(summary["next_required_change"], "fix_architecture_contrast_evaluation")

    def test_latency_gate_failure_blocks_complete_network(self) -> None:
        from scripts.run_xunce_architecture_contrast_evaluation import run_xunce_architecture_contrast_evaluation

        self._write_config(max_forward_latency_ms=0.000001)
        summary = run_xunce_architecture_contrast_evaluation(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("xunce_latency_gate_failed", summary["reason_codes"])
        self.assertFalse(summary["latency_gate_passed"])

    def test_ranking_gate_failure_blocks_complete_network(self) -> None:
        from scripts.run_xunce_architecture_contrast_evaluation import run_xunce_architecture_contrast_evaluation

        self._write_config(
            max_xunce_rank=1,
            architecture_seeds={
                "mlp_v1": 1,
                "mlp_missing_v1": 1,
                "candidate_attention_v1": 1,
                "topology_aware_coverage_graph_proto_v1": 4,
            },
        )
        summary = run_xunce_architecture_contrast_evaluation(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("xunce_ranking_quality_gate_failed", summary["reason_codes"])
        self.assertFalse(summary["ranking_quality_gate_passed"])

    def test_stage6_boundary_violation_fails(self) -> None:
        from scripts.run_xunce_architecture_contrast_evaluation import run_xunce_architecture_contrast_evaluation

        self._write_stage6(summary_updates={"publishes_checkpoint": True})
        summary = run_xunce_architecture_contrast_evaluation(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("architecture_contrast_boundary_violation", summary["reason_codes"])
        self.assertFalse(summary["boundary_audit_passed"])

    def _write_config(
        self,
        *,
        max_parameter_ratio_vs_candidate_attention: float = 2.0,
        max_forward_latency_ms: float = 1000.0,
        max_xunce_rank: int = 3,
        architecture_seeds: dict[str, int] | None = None,
    ) -> None:
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": "xunce-architecture-contrast-evaluation-config/v1",
            "source_proto_mechanism_validation_root": str(self.stage6_root),
            "source_topology_feature_extraction_root": str(self.stage4_root),
            "architectures": [
                "mlp_v1",
                "mlp_missing_v1",
                "candidate_attention_v1",
                "topology_aware_coverage_graph_proto_v1",
            ],
            "architecture_seeds": architecture_seeds
            or {
                "mlp_v1": 1,
                "mlp_missing_v1": 1,
                "candidate_attention_v1": 1,
                "topology_aware_coverage_graph_proto_v1": 1,
            },
            "hidden_dim": 16,
            "attention_heads": 1,
            "message_passing_layers": 1,
            "dropout": 0.0,
            "latency_repeats": 1,
            "max_parameter_ratio_vs_candidate_attention": max_parameter_ratio_vs_candidate_attention,
            "max_latency_ratio_vs_candidate_attention": 1000.0,
            "max_forward_latency_ms": max_forward_latency_ms,
            "max_xunce_rank": max_xunce_rank,
            "action_mask_override": [True, True, True, True, True],
        }
        self.config_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _write_stage6(self, *, summary_updates: dict | None = None) -> None:
        self.stage6_root.mkdir(parents=True, exist_ok=True)
        summary = {
            "schema_version": "xunce-proto-mechanism-validation-summary/v1",
            "status": "passed",
            "reason_codes": [],
            "next_required_change": "architecture_contrast_evaluation",
            "edge_rank_changed": True,
            "memory_rank_changed": True,
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "connects_real_executor": False,
            "starts_online_canary": False,
            "runs_new_ppo_update": False,
            "modifies_network": False,
            "modifies_action_space": False,
            "modifies_default_astar": False,
        }
        if summary_updates:
            summary.update(summary_updates)
        self._write_json(self.stage6_root / "xunce-proto-mechanism-validation-summary.json", summary)

    def _write_stage4(self) -> None:
        self.stage4_root.mkdir(parents=True, exist_ok=True)
        self._write_jsonl(
            self.stage4_root / "xunce-topology-feature-extraction-candidates.jsonl",
            [
                self._candidate(0, [3, 1], 0, 0, 6, 1, 1, 0.01, 0.0, 2, 0.25),
                self._candidate(1, [2, 2], 2, 0, 4, 1, 1, 0.01, 0.0, 3, 0.25),
                self._candidate(2, [1, 3], 3, 2, 8, 3, 3, 0.03, 0.0, 3, 0.0),
                self._candidate(3, [3, 0], 0, 0, 4, 2, 1, 0.02, 0.0, 2, 0.25),
                self._candidate(4, [0, 2], 1, 0, 3, 3, 3, 0.03, 0.0, 4, 0.25),
            ],
        )
        self._write_jsonl(
            self.stage4_root / "xunce-topology-feature-extraction-edges.jsonl",
            [
                self._edge(0, 1, 2, 0.25, False, True, False),
                self._edge(0, 2, 4, 0.11764705882352941, False, False, False),
                self._edge(0, 3, 1, 0.75, True, True, False),
                self._edge(0, 4, 4, 0.07142857142857142, False, True, False),
                self._edge(1, 2, 2, 0.5, False, False, False),
                self._edge(1, 3, 3, 0.18181818181818182, False, True, False),
                self._edge(1, 4, 2, 0.4, False, True, False),
                self._edge(2, 3, 5, 0.0625, False, False, False),
                self._edge(2, 4, 2, 0.5, False, False, False),
                self._edge(3, 4, 5, 0.08333333333333333, False, True, False),
            ],
        )
        self._write_json(
            self.stage4_root / "xunce-topology-feature-extraction-memory.json",
            {
                "schema_version": "xunce-topology-feature-extraction-memory/v1",
                "scenario_id": "xunce_contrast_fixture",
                "features": {
                    "coverage_rate": 0.1590909090909091,
                    "fallback_rate": 0.0,
                    "recent_new_coverage_trend": -1.0,
                    "recent_path_cost_trend": 0.025,
                    "remaining_budget_fraction": 0.95,
                    "revisit_rate": 1.0,
                    "roi_group_completion_ratio": 0.1590909090909091,
                },
            },
        )

    def _candidate(
        self,
        index: int,
        waypoint: list[int],
        cluster: int,
        roi_group: int,
        new_coverage: int,
        distance: int,
        revisit: int,
        budget_fraction_cost: float,
        fallback_risk: float,
        overlap_count: int,
        bottleneck: float,
    ) -> dict:
        return {
            "schema_version": "xunce-topology-feature-extraction-candidate-row/v1",
            "scenario_id": "xunce_contrast_fixture",
            "candidate_index": index,
            "selected_waypoint": waypoint,
            "baseline_rank": index,
            "event_target_cell_count": new_coverage + overlap_count,
            "path": [waypoint],
            "features": {
                "bfs_distance_from_current": distance,
                "budget_fraction_cost": budget_fraction_cost,
                "coverage_overlap_count": overlap_count,
                "fallback_risk": fallback_risk,
                "frontier_cluster_id": cluster,
                "new_coverage_cell_count": new_coverage,
                "path_bottleneck_score": bottleneck,
                "revisit_path_cell_count": revisit,
                "roi_group_id": roi_group,
            },
            "missing_indicators": {
                "bfs_distance_from_current_missing": False,
                "budget_fraction_cost_missing": False,
                "coverage_overlap_count_missing": False,
                "fallback_risk_missing": False,
                "frontier_cluster_id_missing": False,
                "new_coverage_cell_count_missing": False,
                "path_bottleneck_score_missing": False,
                "revisit_path_cell_count_missing": False,
                "roi_group_id_missing": False,
            },
        }

    def _edge(
        self,
        left: int,
        right: int,
        distance: int,
        overlap: float,
        same_cluster: bool,
        same_roi: bool,
        shared_bottleneck: bool,
    ) -> dict:
        return {
            "schema_version": "xunce-topology-feature-extraction-edge-row/v1",
            "scenario_id": "xunce_contrast_fixture",
            "candidate_left_index": left,
            "candidate_right_index": right,
            "candidate_pair_key": f"{left:04d}-{right:04d}",
            "features": {
                "bfs_distance_between_candidates": distance,
                "coverage_overlap_ratio": overlap,
                "mutual_redundancy_score": overlap / 2.0,
                "same_frontier_cluster": same_cluster,
                "same_roi_group": same_roi,
                "shared_bottleneck": shared_bottleneck,
            },
            "missing_indicators": {
                "bfs_distance_between_candidates_missing": False,
                "coverage_overlap_ratio_missing": False,
                "mutual_redundancy_score_missing": False,
                "same_frontier_cluster_missing": False,
                "same_roi_group_missing": False,
                "shared_bottleneck_missing": False,
            },
        }

    def _write_json(self, path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _write_jsonl(self, path: Path, rows: list[dict]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
