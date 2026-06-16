import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path


class XunceCurrentHeadEvidenceRefreshTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        scripts_path = str(self.repo_root / "scripts")
        if scripts_path not in sys.path:
            sys.path.insert(0, scripts_path)
        self.temp_dir = Path(tempfile.mkdtemp(prefix="xunce-current-head-"))
        self.stage0_root = self.temp_dir / "stage0"
        self.controlled_root = self.temp_dir / "controlled"
        self.real_map_shadow_root = self.temp_dir / "real-map-shadow"
        self.real_map_multi_roi_root = self.temp_dir / "real-map-multi-roi"
        self.network_root = self.temp_dir / "network"
        self.output_root = self.temp_dir / "output"
        self.config_path = self.temp_dir / "config.json"
        self.current_git = {
            "parent": {"sha": "abc123", "dirty": False},
            "submodules": {},
            "dirty": False,
        }
        self._write_config()
        self._write_sources()

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def test_current_head_evidence_refresh_passes_with_legacy_provenance_counted(self) -> None:
        from scripts.run_xunce_current_head_evidence_refresh import run_xunce_current_head_evidence_refresh

        summary = run_xunce_current_head_evidence_refresh(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
            current_git=self.current_git,
        )

        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["reason_codes"], [])
        self.assertTrue(summary["current_head_evidence_refresh_passed"])
        self.assertTrue(summary["source_status_audit_passed"])
        self.assertTrue(summary["git_provenance_audit_passed"])
        self.assertTrue(summary["boundary_audit_passed"])
        self.assertEqual(summary["next_required_change"], "network_literature_bottleneck_review")
        self.assertEqual(summary["source_count"], 5)
        self.assertEqual(summary["source_current_git_match_count"], 2)
        self.assertEqual(summary["legacy_missing_git_provenance_count"], 3)
        self.assertFalse(summary["current_git_dirty"])
        self.assertFalse(summary["publishes_checkpoint"])
        self.assertFalse(summary["replaces_default_policy"])
        self.assertFalse(summary["connects_real_executor"])
        self.assertFalse(summary["starts_online_canary"])
        self.assertFalse(summary["runs_new_ppo_update"])
        self.assertFalse(summary["modifies_network"])
        self.assertFalse(summary["modifies_action_space"])
        self.assertFalse(summary["modifies_default_astar"])

        for filename in (
            "xunce-current-head-evidence-refresh-summary.json",
            "xunce-current-head-evidence-refresh-manifest.json",
            "xunce-current-head-source-status-audit.json",
            "xunce-current-head-git-provenance-audit.json",
            "xunce-current-head-boundary-audit.json",
            "xunce-current-head-evidence-refresh-rejection-report.json",
            "xunce-current-head-evidence-refresh-report.md",
        ):
            self.assertTrue((self.output_root / filename).is_file(), filename)

    def test_source_git_provenance_mismatch_fails(self) -> None:
        from scripts.run_xunce_current_head_evidence_refresh import run_xunce_current_head_evidence_refresh

        self._write_sources(real_map_shadow_git={"parent": {"sha": "old", "dirty": False}, "submodules": {}, "dirty": False})
        summary = run_xunce_current_head_evidence_refresh(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
            current_git=self.current_git,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("source_git_provenance_mismatch", summary["reason_codes"])
        self.assertEqual(summary["next_required_change"], "fix_current_head_evidence_refresh")

    def test_failed_stage0_routes_to_design_freeze_fix(self) -> None:
        from scripts.run_xunce_current_head_evidence_refresh import run_xunce_current_head_evidence_refresh

        self._write_sources(stage0_updates={"status": "failed", "next_required_change": "fix_xunce_design_freeze"})
        summary = run_xunce_current_head_evidence_refresh(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
            current_git=self.current_git,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("stage0_not_passed", summary["reason_codes"])
        self.assertEqual(summary["next_required_change"], "fix_xunce_design_freeze")

    def test_source_boundary_violation_fails(self) -> None:
        from scripts.run_xunce_current_head_evidence_refresh import run_xunce_current_head_evidence_refresh

        self._write_sources(controlled_updates={"connects_real_executor": True})
        summary = run_xunce_current_head_evidence_refresh(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
            current_git=self.current_git,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("source_boundary_violation", summary["reason_codes"])
        self.assertFalse(summary["boundary_audit_passed"])
        self.assertEqual(summary["next_required_change"], "fix_current_head_evidence_refresh")

    def _write_config(self) -> None:
        self.config_path.write_text(
            json.dumps(
                {
                    "schema_version": "xunce-current-head-evidence-refresh-config/v1",
                    "source_xunce_design_freeze_root": str(self.stage0_root),
                    "source_controlled_installation_root": str(self.controlled_root),
                    "source_real_map_shadow_replay_root": str(self.real_map_shadow_root),
                    "source_real_map_multi_roi_root": str(self.real_map_multi_roi_root),
                    "source_network_readiness_root": str(self.network_root),
                    "require_current_worktree_clean": True,
                    "require_stage0_passed": True,
                    "allow_legacy_missing_git_provenance": True,
                    "require_closed_boundaries": True,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    def _write_sources(
        self,
        *,
        stage0_updates: dict | None = None,
        controlled_updates: dict | None = None,
        real_map_shadow_git: dict | None = None,
    ) -> None:
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
        stage0 = {
            "schema_version": "xunce-design-freeze-summary/v1",
            "status": "passed",
            "reason_codes": [],
            "next_required_change": "current_head_evidence_refresh",
            "git_provenance": {"current": self.current_git},
            **boundary,
        }
        if stage0_updates:
            stage0.update(stage0_updates)
        controlled = {
            "schema_version": "global-99-controlled-default-policy-candidate-installation-preflight-summary/v1",
            "status": "passed",
            "reason_codes": [],
            "next_required_change": "eligible_for_controlled_default_policy_candidate_installation_review",
            **boundary,
        }
        if controlled_updates:
            controlled.update(controlled_updates)
        real_map_shadow = {
            "schema_version": "global-99-real-map-shadow-replay-summary/v1",
            "status": "passed",
            "reason_codes": [],
            "next_required_change": "global_99_real_map_release_governance_preflight",
            "git_provenance": {"current": real_map_shadow_git or self.current_git},
            **boundary,
        }
        real_map_multi_roi = {
            "schema_version": "global-99-real-map-multi-roi-generalization-summary/v1",
            "status": "passed",
            "reason_codes": [],
            "next_required_change": "default_policy_candidate_authorization_preflight",
            **boundary,
        }
        network = {
            "schema_version": "network-architecture-upgrade-readiness-summary/v1",
            "status": "passed",
            "reason_codes": [],
            "next_required_change": "global_99_release_governance_preflight",
            "network_upgrade_recommended": False,
            **boundary,
        }
        for root in (
            self.stage0_root,
            self.controlled_root,
            self.real_map_shadow_root,
            self.real_map_multi_roi_root,
            self.network_root,
        ):
            root.mkdir(parents=True, exist_ok=True)
        (self.stage0_root / "xunce-design-freeze-summary.json").write_text(
            json.dumps(stage0, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (self.controlled_root / "global-99-controlled-default-policy-candidate-installation-preflight-summary.json").write_text(
            json.dumps(controlled, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (self.real_map_shadow_root / "global-99-real-map-shadow-replay-summary.json").write_text(
            json.dumps(real_map_shadow, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (self.real_map_multi_roi_root / "global-99-real-map-multi-roi-generalization-summary.json").write_text(
            json.dumps(real_map_multi_roi, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (self.network_root / "network-architecture-upgrade-readiness-summary.json").write_text(
            json.dumps(network, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
