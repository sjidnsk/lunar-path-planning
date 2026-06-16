import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import torch


class XunceFullNetworkV1Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(tempfile.mkdtemp(prefix="xunce-full-network-"))
        scripts_path = str(Path(__file__).resolve().parents[1] / "scripts")
        if scripts_path not in sys.path:
            sys.path.insert(0, scripts_path)
        self.stage7_root = self.repo_root / "outputs" / "stage7"
        self.stage4_root = self.repo_root / "outputs" / "stage4"
        self.output_root = self.repo_root / "outputs" / "stage8"
        self.config_path = self.repo_root / "configs" / "xunce_full_network_v1.json"
        self._write_stage7()
        self._write_stage4()
        self._write_config()

    def tearDown(self) -> None:
        shutil.rmtree(self.repo_root)

    def test_network_forward_shapes_mask_and_metadata(self) -> None:
        from scripts.xunce_full_network_common import XunceFullNetworkV1

        network = XunceFullNetworkV1(
            candidate_feature_count=4,
            edge_feature_count=3,
            memory_feature_count=2,
            context_feature_count=2,
            missing_indicator_count=1,
            hidden_dim=8,
            message_passing_layers=1,
            dropout=0.0,
        )
        with torch.no_grad():
            output = network(
                candidate_features=torch.randn(1, 3, 4),
                edge_features=torch.randn(3, 3),
                edge_index=torch.tensor([[0, 1], [0, 2], [1, 2]], dtype=torch.long),
                memory_features=torch.randn(1, 2),
                context_features=torch.randn(1, 3, 2),
                action_mask=torch.tensor([[True, False, True]]),
                candidate_missing_indicators=torch.zeros(1, 3, 1),
            )

        self.assertEqual(tuple(output.logits.shape), (1, 3))
        self.assertEqual(tuple(output.masked_logits.shape), (1, 3))
        self.assertEqual(tuple(output.action_probs.shape), (1, 3))
        self.assertEqual(tuple(output.value.shape), (1,))
        self.assertLessEqual(float(output.masked_logits[0, 1]), -1.0e8)
        self.assertEqual(float(output.action_probs[0, 1]), 0.0)
        self.assertAlmostEqual(float(output.action_probs[0, 0] + output.action_probs[0, 2]), 1.0, places=6)
        metadata = network.metadata()
        self.assertEqual(metadata["architecture"], "xunce_full_network_v1")
        self.assertTrue(metadata["candidate_graph_encoder_used"])
        self.assertTrue(metadata["topology_bias_used"])
        self.assertTrue(metadata["coverage_memory_token_used"])
        self.assertTrue(metadata["roi_budget_fusion_used"])

    def test_default_fixture_passes_and_writes_artifacts(self) -> None:
        from scripts.run_xunce_full_network_v1 import run_xunce_full_network_v1

        summary = run_xunce_full_network_v1(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["reason_codes"], [])
        self.assertEqual(summary["source_architecture_contrast_status"], "passed")
        self.assertEqual(summary["architecture"], "xunce_full_network_v1")
        self.assertEqual(summary["next_required_change"], "full_network_static_contract_validation")
        self.assertTrue(summary["candidate_graph_encoder_used"])
        self.assertTrue(summary["topology_bias_used"])
        self.assertTrue(summary["coverage_memory_token_used"])
        self.assertTrue(summary["roi_budget_fusion_used"])
        self.assertTrue(summary["masked_logits_valid"])
        self.assertTrue(summary["value_head_valid"])
        self.assertTrue(summary["metadata_audit_passed"])
        self.assertTrue(summary["parameter_latency_audit_passed"])
        self.assertTrue(summary["full_network_v1_passed"])
        self.assertFalse(summary["publishes_checkpoint"])
        self.assertFalse(summary["replaces_default_policy"])
        self.assertFalse(summary["connects_real_executor"])
        self.assertFalse(summary["starts_online_canary"])
        self.assertFalse(summary["runs_new_ppo_update"])
        self.assertFalse(summary["modifies_network"])
        self.assertFalse(summary["modifies_action_space"])
        self.assertFalse(summary["modifies_default_astar"])

        for filename in (
            "xunce-full-network-v1-summary.json",
            "xunce-full-network-v1-manifest.json",
            "xunce-full-network-v1-forward-audit.json",
            "xunce-full-network-v1-metadata-audit.json",
            "xunce-full-network-v1-parameter-latency-audit.json",
            "xunce-full-network-v1-logits.jsonl",
            "xunce-full-network-v1-boundary-audit.json",
            "xunce-full-network-v1-rejection-report.json",
            "xunce-full-network-v1-report.md",
        ):
            self.assertTrue((self.output_root / filename).is_file(), filename)

    def test_missing_stage7_routes_to_contrast_fix(self) -> None:
        from scripts.run_xunce_full_network_v1 import run_xunce_full_network_v1

        shutil.rmtree(self.stage7_root)
        summary = run_xunce_full_network_v1(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("missing_architecture_contrast_evaluation_summary", summary["reason_codes"])
        self.assertEqual(summary["next_required_change"], "fix_architecture_contrast_evaluation")

    def test_parameter_gate_failure_blocks_static_contract(self) -> None:
        from scripts.run_xunce_full_network_v1 import run_xunce_full_network_v1

        self._write_config(max_parameter_count=1)
        summary = run_xunce_full_network_v1(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("full_network_parameter_latency_gate_failed", summary["reason_codes"])
        self.assertFalse(summary["parameter_latency_audit_passed"])
        self.assertEqual(summary["next_required_change"], "fix_xunce_full_network_v1")

    def test_stage7_boundary_violation_fails(self) -> None:
        from scripts.run_xunce_full_network_v1 import run_xunce_full_network_v1

        self._write_stage7(summary_updates={"publishes_checkpoint": True})
        summary = run_xunce_full_network_v1(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("full_network_boundary_violation", summary["reason_codes"])
        self.assertFalse(summary["boundary_audit_passed"])

    def _write_config(self, *, max_parameter_count: int = 50000) -> None:
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": "xunce-full-network-v1-config/v1",
            "source_architecture_contrast_root": str(self.stage7_root),
            "source_topology_feature_extraction_root": str(self.stage4_root),
            "architecture": "xunce_full_network_v1",
            "hidden_dim": 16,
            "message_passing_layers": 2,
            "dropout": 0.0,
            "seed": 5,
            "latency_repeats": 1,
            "max_parameter_count": max_parameter_count,
            "max_forward_latency_ms": 1000.0,
            "action_mask_override": [True, False, True],
        }
        self.config_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _write_stage7(self, *, summary_updates: dict | None = None) -> None:
        self.stage7_root.mkdir(parents=True, exist_ok=True)
        summary = {
            "schema_version": "xunce-architecture-contrast-evaluation-summary/v1",
            "status": "passed",
            "reason_codes": [],
            "next_required_change": "full_xunce_network_v1_design",
            "architecture_contrast_passed": True,
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
        self._write_json(self.stage7_root / "xunce-architecture-contrast-evaluation-summary.json", summary)

    def _write_stage4(self) -> None:
        self.stage4_root.mkdir(parents=True, exist_ok=True)
        self._write_jsonl(
            self.stage4_root / "xunce-topology-feature-extraction-candidates.jsonl",
            [
                self._candidate(0, [3, 1], 0, 0, 6, 1, 1, 0.01, 0.0, 2, 0.25),
                self._candidate(1, [2, 2], 2, 0, 4, 1, 1, 0.01, 0.0, 3, 0.25),
                self._candidate(2, [1, 3], 3, 2, 8, 3, 3, 0.03, 0.0, 3, 0.0),
            ],
        )
        self._write_jsonl(
            self.stage4_root / "xunce-topology-feature-extraction-edges.jsonl",
            [
                self._edge(0, 1, 2, 0.25, False, True, False),
                self._edge(0, 2, 4, 0.11764705882352941, False, False, False),
                self._edge(1, 2, 2, 0.5, False, False, False),
            ],
        )
        self._write_json(
            self.stage4_root / "xunce-topology-feature-extraction-memory.json",
            {
                "schema_version": "xunce-topology-feature-extraction-memory/v1",
                "scenario_id": "xunce_full_network_fixture",
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
            "scenario_id": "xunce_full_network_fixture",
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
            "scenario_id": "xunce_full_network_fixture",
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
