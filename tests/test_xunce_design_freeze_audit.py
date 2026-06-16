import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path


class XunceDesignFreezeAuditTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(tempfile.mkdtemp(prefix="xunce-design-freeze-"))
        self.scripts_path = str(Path(__file__).resolve().parents[1] / "scripts")
        if self.scripts_path not in sys.path:
            sys.path.insert(0, self.scripts_path)
        self.output_root = self.repo_root / "outputs" / "xunce-freeze"
        self.controlled_root = self.repo_root / "outputs" / "controlled"
        self.network_root = self.repo_root / "outputs" / "network"
        self.real_map_root = self.repo_root / "outputs" / "real-map"
        self.config_path = self.repo_root / "configs" / "xunce_design_freeze_v1.json"
        self._write_docs()
        self._write_config()
        self._write_sources()

    def tearDown(self) -> None:
        shutil.rmtree(self.repo_root)

    def test_design_freeze_passes_and_writes_artifacts(self) -> None:
        from scripts.run_xunce_design_freeze_audit import run_xunce_design_freeze_audit

        summary = run_xunce_design_freeze_audit(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["reason_codes"], [])
        self.assertTrue(summary["design_freeze_passed"])
        self.assertTrue(summary["document_audit_passed"])
        self.assertTrue(summary["evidence_audit_passed"])
        self.assertTrue(summary["boundary_audit_passed"])
        self.assertEqual(summary["next_required_change"], "current_head_evidence_refresh")
        self.assertEqual(summary["document_count"], 4)
        self.assertEqual(summary["complete_stage_chain_count"], 18)
        self.assertEqual(summary["source_controlled_installation_status"], "passed")
        self.assertEqual(
            summary["source_controlled_installation_next_required_change"],
            "eligible_for_controlled_default_policy_candidate_installation_review",
        )
        self.assertFalse(summary["publishes_checkpoint"])
        self.assertFalse(summary["replaces_default_policy"])
        self.assertFalse(summary["connects_real_executor"])
        self.assertFalse(summary["starts_online_canary"])
        self.assertFalse(summary["runs_new_ppo_update"])
        self.assertFalse(summary["modifies_network"])
        self.assertFalse(summary["modifies_action_space"])
        self.assertFalse(summary["modifies_default_astar"])

        for filename in (
            "xunce-design-freeze-summary.json",
            "xunce-design-freeze-manifest.json",
            "xunce-design-document-audit.json",
            "xunce-design-evidence-audit.json",
            "xunce-design-boundary-audit.json",
            "xunce-design-freeze-rejection-report.json",
            "xunce-design-freeze-report.md",
        ):
            self.assertTrue((self.output_root / filename).is_file(), filename)

    def test_missing_complete_stage_chain_fails(self) -> None:
        from scripts.run_xunce_design_freeze_audit import run_xunce_design_freeze_audit

        spec = self.repo_root / "docs" / "superpowers" / "specs" / "2026-06-16-topology-aware-coverage-policy-network-design.md"
        spec.write_text("巡策 Topology-Aware Coverage Policy Network Design\n", encoding="utf-8")

        summary = run_xunce_design_freeze_audit(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("xunce_stage_chain_incomplete", summary["reason_codes"])
        self.assertEqual(summary["next_required_change"], "fix_xunce_design_freeze")

    def test_source_boundary_violation_fails(self) -> None:
        from scripts.run_xunce_design_freeze_audit import run_xunce_design_freeze_audit

        self._write_sources(controlled_overrides={"replaces_default_policy": True})

        summary = run_xunce_design_freeze_audit(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("source_boundary_violation", summary["reason_codes"])
        self.assertFalse(summary["boundary_audit_passed"])
        self.assertEqual(summary["next_required_change"], "fix_global_99_evidence_chain")

    def _write_config(self) -> None:
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        self.config_path.write_text(
            json.dumps(
                {
                    "schema_version": "xunce-design-freeze-config/v1",
                    "documents": [
                        "README.md",
                        "docs/算法设计与系统架构报告.md",
                        "docs/superpowers/specs/2026-06-16-global-99-exploration-coverage-line.md",
                        "docs/superpowers/specs/2026-06-16-topology-aware-coverage-policy-network-design.md",
                    ],
                    "source_controlled_installation_root": str(self.controlled_root),
                    "source_network_readiness_root": str(self.network_root),
                    "source_real_map_multi_roi_root": str(self.real_map_root),
                    "require_complete_stage_chain": True,
                    "require_plan_first_rule": True,
                    "require_closed_boundaries": True,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    def _write_docs(self) -> None:
        for rel in (
            "README.md",
            "docs/算法设计与系统架构报告.md",
            "docs/superpowers/specs/2026-06-16-global-99-exploration-coverage-line.md",
            "docs/superpowers/specs/2026-06-16-topology-aware-coverage-policy-network-design.md",
        ):
            path = self.repo_root / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(self._doc_body(), encoding="utf-8")

    def _doc_body(self) -> str:
        stages = "\n".join(
            [
                "0 冻结巡策设计",
                "1 Current-HEAD Evidence Refresh",
                "2 文献与项目瓶颈审计",
                "3 Topology Observation Contract",
                "4 Topology Feature Extraction Audit",
                "5 小型巡策原型",
                "6 原型机制验证",
                "7 架构对比评测",
                "8 完整巡策网络 v1",
                "9 完整网络静态合同验证",
                "10 完整网络消融实验",
                "11 完整网络压力评测",
                "12 Guarded Training Candidate Preflight",
                "13 受控训练候选",
                "14 训练后离线评测",
                "15 Shadow / Replay 验证",
                "16 Sandbox Candidate Preflight",
                "17 发布治理门禁",
            ]
        )
        return (
            "巡策 / Topology-Aware Coverage Policy Network Design\n"
            "每一个阶段都必须先设计详细计划，再执行实现。\n"
            f"{stages}\n"
            "不连接真实 executor；不启动 online canary；不直接替换 default policy；"
            "不宣称真实世界性能；不修改 action space/default A*；不发布 checkpoint。\n"
        )

    def _write_sources(self, controlled_overrides: dict | None = None) -> None:
        boundary = {
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "connects_real_executor": False,
            "starts_online_canary": False,
            "runs_new_ppo_update": False,
            "modifies_network": False,
            "modifies_action_space": False,
            "modifies_default_astar": False,
        }
        controlled = {
            "schema_version": "global-99-controlled-default-policy-candidate-installation-preflight-summary/v1",
            "status": "passed",
            "reason_codes": [],
            "next_required_change": "eligible_for_controlled_default_policy_candidate_installation_review",
            "controlled_installation_verdict": "eligible_for_controlled_default_policy_candidate_installation_review",
            **boundary,
        }
        if controlled_overrides:
            controlled.update(controlled_overrides)
        network = {
            "schema_version": "network-architecture-upgrade-readiness-summary/v1",
            "status": "passed",
            "reason_codes": [],
            "next_required_change": "global_99_release_governance_preflight",
            "network_upgrade_recommended": False,
            **boundary,
        }
        real_map = {
            "schema_version": "global-99-real-map-multi-roi-generalization-summary/v1",
            "status": "passed",
            "reason_codes": [],
            "next_required_change": "default_policy_candidate_authorization_preflight",
            "failed_required_scenario_count": 0,
            **boundary,
        }
        self.controlled_root.mkdir(parents=True, exist_ok=True)
        self.network_root.mkdir(parents=True, exist_ok=True)
        self.real_map_root.mkdir(parents=True, exist_ok=True)
        (self.controlled_root / "global-99-controlled-default-policy-candidate-installation-preflight-summary.json").write_text(
            json.dumps(controlled, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (self.network_root / "network-architecture-upgrade-readiness-summary.json").write_text(
            json.dumps(network, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (self.real_map_root / "global-99-real-map-multi-roi-generalization-summary.json").write_text(
            json.dumps(real_map, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
