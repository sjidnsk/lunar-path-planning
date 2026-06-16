import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path


class XunceProtoMechanismValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(tempfile.mkdtemp(prefix="xunce-mechanism-"))
        scripts_path = str(Path(__file__).resolve().parents[1] / "scripts")
        if scripts_path not in sys.path:
            sys.path.insert(0, scripts_path)
        self.stage5_root = self.repo_root / "outputs" / "stage5"
        self.stage4_root = self.repo_root / "outputs" / "stage4"
        self.output_root = self.repo_root / "outputs" / "stage6"
        self.config_path = self.repo_root / "configs" / "xunce_proto_mechanism_validation_v1.json"
        self._write_stage5()
        self._write_stage4()
        self._write_config()

    def tearDown(self) -> None:
        shutil.rmtree(self.repo_root)

    def test_default_fixture_passes_and_writes_artifacts(self) -> None:
        from scripts.run_xunce_proto_mechanism_validation import run_xunce_proto_mechanism_validation

        summary = run_xunce_proto_mechanism_validation(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["reason_codes"], [])
        self.assertEqual(summary["next_required_change"], "architecture_contrast_evaluation")
        self.assertTrue(summary["signal_audit_passed"])
        self.assertTrue(summary["mask_audit_passed"])
        self.assertTrue(summary["fallback_audit_passed"])
        self.assertTrue(summary["boundary_audit_passed"])
        self.assertTrue(summary["edge_rank_changed"])
        self.assertTrue(summary["memory_rank_changed"])
        self.assertGreaterEqual(summary["max_edge_logit_delta"], 0.001)
        self.assertGreaterEqual(summary["max_memory_logit_delta"], 0.001)
        self.assertFalse(summary["publishes_checkpoint"])
        self.assertFalse(summary["replaces_default_policy"])
        self.assertFalse(summary["connects_real_executor"])
        self.assertFalse(summary["starts_online_canary"])
        self.assertFalse(summary["runs_new_ppo_update"])

        for filename in (
            "xunce-proto-mechanism-validation-summary.json",
            "xunce-proto-mechanism-validation-manifest.json",
            "xunce-proto-mechanism-validation-signal-audit.json",
            "xunce-proto-mechanism-validation-mask-audit.json",
            "xunce-proto-mechanism-validation-fallback-audit.json",
            "xunce-proto-mechanism-validation-logit-comparison.jsonl",
            "xunce-proto-mechanism-validation-boundary-audit.json",
            "xunce-proto-mechanism-validation-rejection-report.json",
            "xunce-proto-mechanism-validation-report.md",
        ):
            self.assertTrue((self.output_root / filename).is_file(), filename)

    def test_mask_override_keeps_invalid_candidate_masked(self) -> None:
        from scripts.run_xunce_proto_mechanism_validation import run_xunce_proto_mechanism_validation

        self._write_config(action_mask_override=[True, False, True, True, True])
        summary = run_xunce_proto_mechanism_validation(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )
        rows = self._read_jsonl(self.output_root / "xunce-proto-mechanism-validation-logit-comparison.jsonl")

        self.assertEqual(summary["status"], "passed")
        self.assertTrue(summary["mask_audit_passed"])
        self.assertLessEqual(rows[1]["full_masked_logit"], -1.0e8)
        self.assertEqual(rows[1]["full_action_probability"], 0.0)

    def test_missing_stage5_routes_to_proto_fix(self) -> None:
        from scripts.run_xunce_proto_mechanism_validation import run_xunce_proto_mechanism_validation

        shutil.rmtree(self.stage5_root)
        summary = run_xunce_proto_mechanism_validation(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("missing_topology_graph_proto_summary", summary["reason_codes"])
        self.assertEqual(summary["next_required_change"], "fix_xunce_topology_graph_proto")

    def test_insufficient_signal_delta_fails_mechanism_validation(self) -> None:
        from scripts.run_xunce_proto_mechanism_validation import run_xunce_proto_mechanism_validation

        self._write_config(min_logit_delta=100.0)
        summary = run_xunce_proto_mechanism_validation(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("edge_signal_delta_too_small", summary["reason_codes"])
        self.assertIn("memory_signal_delta_too_small", summary["reason_codes"])
        self.assertEqual(summary["next_required_change"], "fix_xunce_proto_mechanism_validation")

    def test_stage5_boundary_violation_fails(self) -> None:
        from scripts.run_xunce_proto_mechanism_validation import run_xunce_proto_mechanism_validation

        self._write_stage5(summary_updates={"publishes_checkpoint": True})
        summary = run_xunce_proto_mechanism_validation(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("proto_mechanism_boundary_violation", summary["reason_codes"])
        self.assertFalse(summary["boundary_audit_passed"])

    def _write_config(
        self,
        *,
        min_logit_delta: float = 0.001,
        action_mask_override: list[bool] | None = None,
    ) -> None:
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": "xunce-proto-mechanism-validation-config/v1",
            "source_topology_graph_proto_root": str(self.stage5_root),
            "source_topology_feature_extraction_root": str(self.stage4_root),
            "architecture": "topology_aware_coverage_graph_proto_v1",
            "hidden_dim": 16,
            "message_passing_layers": 1,
            "dropout": 0.0,
            "mechanism_seed": 4,
            "min_logit_delta": min_logit_delta,
            "require_edge_rank_change": True,
            "require_memory_rank_change": True,
            "action_mask_override": action_mask_override or [True, True, True, True, True],
        }
        self.config_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _write_stage5(self, *, summary_updates: dict | None = None) -> None:
        self.stage5_root.mkdir(parents=True, exist_ok=True)
        summary = {
            "schema_version": "xunce-topology-graph-proto-summary/v1",
            "status": "passed",
            "reason_codes": [],
            "next_required_change": "xunce_proto_mechanism_validation",
            "architecture": "topology_aware_coverage_graph_proto_v1",
            "candidate_graph_used": True,
            "memory_token_used": True,
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
        self._write_json(self.stage5_root / "xunce-topology-graph-proto-summary.json", summary)

    def _write_stage4(self) -> None:
        self.stage4_root.mkdir(parents=True, exist_ok=True)
        self._write_jsonl(
            self.stage4_root / "xunce-topology-feature-extraction-candidates.jsonl",
            [
                self._candidate(0, [3, 1], 0, 0, 6, 1, 1, 0.01, 0.0),
                self._candidate(1, [2, 2], 2, 0, 4, 1, 1, 0.01, 0.0),
                self._candidate(2, [1, 3], 3, 2, 8, 3, 3, 0.03, 0.0),
                self._candidate(3, [3, 0], 0, 0, 4, 2, 1, 0.02, 0.0),
                self._candidate(4, [0, 2], 1, 0, 3, 3, 3, 0.03, 0.0),
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
        budget: float,
        fallback: float,
    ) -> dict:
        features = {
            "bfs_distance_from_current": distance,
            "budget_fraction_cost": budget,
            "coverage_overlap_count": 2,
            "fallback_risk": fallback,
            "frontier_cluster_id": cluster,
            "new_coverage_cell_count": new_coverage,
            "path_bottleneck_score": 0.25,
            "revisit_path_cell_count": revisit,
            "roi_group_id": roi_group,
        }
        return {
            "candidate_index": index,
            "selected_waypoint": waypoint,
            "features": features,
            "missing_indicators": {f"{name}_missing": False for name in features},
        }

    def _edge(
        self,
        left: int,
        right: int,
        distance: float,
        overlap: float,
        same_cluster: bool,
        same_roi: bool,
        shared_bottleneck: bool,
    ) -> dict:
        features = {
            "bfs_distance_between_candidates": distance,
            "coverage_overlap_ratio": overlap,
            "mutual_redundancy_score": overlap * 0.5,
            "same_frontier_cluster": same_cluster,
            "same_roi_group": same_roi,
            "shared_bottleneck": shared_bottleneck,
        }
        return {
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
