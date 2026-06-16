import json
import math
import shutil
import sys
import tempfile
import unittest
from pathlib import Path


class XunceTopologyGraphProtoTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(tempfile.mkdtemp(prefix="xunce-graph-proto-"))
        scripts_path = str(Path(__file__).resolve().parents[1] / "scripts")
        if scripts_path not in sys.path:
            sys.path.insert(0, scripts_path)
        self.stage4_root = self.repo_root / "outputs" / "stage4"
        self.output_root = self.repo_root / "outputs" / "stage5"
        self.config_path = self.repo_root / "configs" / "xunce_topology_graph_proto_v1.json"
        self._write_stage4()
        self._write_config()

    def tearDown(self) -> None:
        shutil.rmtree(self.repo_root)

    def test_default_fixture_passes_and_writes_artifacts(self) -> None:
        from scripts.run_xunce_topology_graph_proto import run_xunce_topology_graph_proto

        summary = run_xunce_topology_graph_proto(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["reason_codes"], [])
        self.assertEqual(summary["next_required_change"], "xunce_proto_mechanism_validation")
        self.assertEqual(summary["architecture"], "topology_aware_coverage_graph_proto_v1")
        self.assertEqual(summary["candidate_count"], 3)
        self.assertEqual(summary["edge_count"], 3)
        self.assertTrue(summary["candidate_graph_used"])
        self.assertTrue(summary["memory_token_used"])
        self.assertTrue(summary["forward_pass_audit_passed"])
        self.assertTrue(summary["shape_audit_passed"])
        self.assertTrue(summary["parameter_audit_passed"])
        self.assertTrue(summary["boundary_audit_passed"])
        self.assertFalse(summary["publishes_checkpoint"])
        self.assertFalse(summary["replaces_default_policy"])
        self.assertFalse(summary["connects_real_executor"])
        self.assertFalse(summary["starts_online_canary"])
        self.assertFalse(summary["runs_new_ppo_update"])
        self.assertFalse(summary["modifies_action_space"])
        self.assertFalse(summary["modifies_default_astar"])

        for filename in (
            "xunce-topology-graph-proto-summary.json",
            "xunce-topology-graph-proto-manifest.json",
            "xunce-topology-graph-proto-forward-pass-audit.json",
            "xunce-topology-graph-proto-shape-audit.json",
            "xunce-topology-graph-proto-parameter-audit.json",
            "xunce-topology-graph-proto-logits.jsonl",
            "xunce-topology-graph-proto-boundary-audit.json",
            "xunce-topology-graph-proto-rejection-report.json",
            "xunce-topology-graph-proto-report.md",
        ):
            self.assertTrue((self.output_root / filename).is_file(), filename)

    def test_forward_shapes_and_finite_outputs(self) -> None:
        from scripts.run_xunce_topology_graph_proto import run_xunce_topology_graph_proto

        summary = run_xunce_topology_graph_proto(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )
        forward = json.loads((self.output_root / "xunce-topology-graph-proto-forward-pass-audit.json").read_text(encoding="utf-8"))

        self.assertEqual(forward["logits_shape"], [1, 3])
        self.assertEqual(forward["masked_logits_shape"], [1, 3])
        self.assertEqual(forward["action_probs_shape"], [1, 3])
        self.assertEqual(forward["value_shape"], [1])
        self.assertEqual(summary["non_finite_output_count"], 0)
        self.assertAlmostEqual(forward["valid_action_probability_sum"], 1.0, places=6)
        self.assertTrue(math.isfinite(float(forward["value"][0])))

    def test_invalid_action_mask_override_masks_logits(self) -> None:
        from scripts.run_xunce_topology_graph_proto import run_xunce_topology_graph_proto

        self._write_config(action_mask_override=[True, False, True])
        summary = run_xunce_topology_graph_proto(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )
        logits = self._read_jsonl(self.output_root / "xunce-topology-graph-proto-logits.jsonl")

        self.assertEqual(summary["status"], "passed")
        self.assertLessEqual(logits[1]["masked_logit"], -1.0e8)
        self.assertEqual(logits[1]["action_probability"], 0.0)
        self.assertAlmostEqual(logits[0]["action_probability"] + logits[2]["action_probability"], 1.0, places=6)

    def test_missing_stage4_routes_to_feature_extraction_fix(self) -> None:
        from scripts.run_xunce_topology_graph_proto import run_xunce_topology_graph_proto

        shutil.rmtree(self.stage4_root)
        summary = run_xunce_topology_graph_proto(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("missing_topology_feature_extraction_summary", summary["reason_codes"])
        self.assertEqual(summary["next_required_change"], "fix_xunce_topology_feature_extraction_audit")

    def test_stage4_boundary_violation_fails(self) -> None:
        from scripts.run_xunce_topology_graph_proto import run_xunce_topology_graph_proto

        self._write_stage4(summary_updates={"publishes_checkpoint": True})
        summary = run_xunce_topology_graph_proto(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("topology_graph_proto_boundary_violation", summary["reason_codes"])
        self.assertFalse(summary["boundary_audit_passed"])
        self.assertEqual(summary["next_required_change"], "fix_xunce_topology_graph_proto")

    def _write_config(self, *, action_mask_override: list[bool] | None = None) -> None:
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": "xunce-topology-graph-proto-config/v1",
            "source_topology_feature_extraction_root": str(self.stage4_root),
            "architecture": "topology_aware_coverage_graph_proto_v1",
            "hidden_dim": 8,
            "message_passing_layers": 1,
            "dropout": 0.0,
            "seed": 7,
            "max_forward_passes": 1,
            "max_parameter_count": 10000,
            "max_forward_latency_ms": 1000.0,
        }
        if action_mask_override is not None:
            payload["action_mask_override"] = action_mask_override
        self.config_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _write_stage4(self, *, summary_updates: dict | None = None) -> None:
        self.stage4_root.mkdir(parents=True, exist_ok=True)
        summary = {
            "schema_version": "xunce-topology-feature-extraction-audit-summary/v1",
            "status": "passed",
            "reason_codes": [],
            "next_required_change": "topology_aware_coverage_graph_proto",
            "candidate_count": 3,
            "edge_count": 3,
            "candidate_missing_field_count": 0,
            "edge_missing_field_count": 0,
            "memory_missing_field_count": 0,
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "connects_real_executor": False,
            "starts_online_canary": False,
            "runs_new_ppo_update": False,
            "modifies_action_space": False,
            "modifies_default_astar": False,
        }
        if summary_updates:
            summary.update(summary_updates)
        self._write_json(self.stage4_root / "xunce-topology-feature-extraction-audit-summary.json", summary)
        self._write_jsonl(
            self.stage4_root / "xunce-topology-feature-extraction-candidates.jsonl",
            [
                self._candidate(0, [2, 1], 0, 0, 8, 2, 1, 0.2, 0.0),
                self._candidate(1, [2, 2], 0, 0, 6, 3, 1, 0.3, 0.0),
                self._candidate(2, [4, 1], 1, 1, 5, 4, 0, 0.4, 0.0),
            ],
        )
        self._write_jsonl(
            self.stage4_root / "xunce-topology-feature-extraction-edges.jsonl",
            [
                self._edge(0, 1, same_cluster=True, same_roi=True, distance=1.0, overlap=0.5),
                self._edge(0, 2, same_cluster=False, same_roi=False, distance=3.0, overlap=0.1),
                self._edge(1, 2, same_cluster=False, same_roi=False, distance=2.0, overlap=0.2),
            ],
        )
        self._write_json(
            self.stage4_root / "xunce-topology-feature-extraction-memory.json",
            {
                "schema_version": "xunce-topology-feature-extraction-memory/v1",
                "scenario_id": "xunce_proto_fixture",
                "features": {
                    "coverage_rate": 0.25,
                    "remaining_budget_fraction": 0.8,
                    "recent_path_cost_trend": 0.03,
                    "recent_new_coverage_trend": -1.0,
                    "revisit_rate": 0.5,
                    "fallback_rate": 0.0,
                    "roi_group_completion_ratio": 0.2,
                },
                "missing_indicators": {
                    "coverage_rate_missing": False,
                    "remaining_budget_fraction_missing": False,
                    "recent_path_cost_trend_missing": False,
                    "recent_new_coverage_trend_missing": False,
                    "revisit_rate_missing": False,
                    "fallback_rate_missing": False,
                    "roi_group_completion_ratio_missing": False,
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
        bfs_distance: int,
        revisit: int,
        budget_fraction: float,
        fallback_risk: float,
    ) -> dict:
        features = {
            "frontier_cluster_id": cluster,
            "roi_group_id": roi_group,
            "new_coverage_cell_count": new_coverage,
            "coverage_overlap_count": 2,
            "bfs_distance_from_current": bfs_distance,
            "path_bottleneck_score": 0.25,
            "revisit_path_cell_count": revisit,
            "budget_fraction_cost": budget_fraction,
            "fallback_risk": fallback_risk,
        }
        return {
            "schema_version": "xunce-topology-feature-extraction-candidate-row/v1",
            "scenario_id": "xunce_proto_fixture",
            "candidate_index": index,
            "baseline_rank": index,
            "selected_waypoint": waypoint,
            "features": features,
            "missing_indicators": {f"{name}_missing": False for name in features},
        }

    def _edge(
        self,
        left: int,
        right: int,
        *,
        same_cluster: bool,
        same_roi: bool,
        distance: float,
        overlap: float,
    ) -> dict:
        features = {
            "same_frontier_cluster": same_cluster,
            "same_roi_group": same_roi,
            "bfs_distance_between_candidates": distance,
            "coverage_overlap_ratio": overlap,
            "shared_bottleneck": False,
            "mutual_redundancy_score": overlap * 0.5,
        }
        return {
            "schema_version": "xunce-topology-feature-extraction-edge-row/v1",
            "scenario_id": "xunce_proto_fixture",
            "candidate_left_index": left,
            "candidate_right_index": right,
            "candidate_pair_key": f"{left:04d}-{right:04d}",
            "features": features,
            "missing_indicators": {f"{name}_missing": False for name in features},
        }

    def _read_jsonl(self, path: Path) -> list[dict]:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def _write_json(self, path: Path, payload: dict) -> None:
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _write_jsonl(self, path: Path, rows: list[dict]) -> None:
        path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
