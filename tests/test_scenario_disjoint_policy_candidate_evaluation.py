import json
import os
import shutil
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path


class ScenarioDisjointPolicyCandidateEvaluationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.temp_dir = Path(tempfile.mkdtemp(prefix="scenario-disjoint-"))
        self.python = Path("/home/kai/anaconda3/envs/lunar-explorer/bin/python")

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def _env(self) -> dict[str, str]:
        env = os.environ.copy()
        env["PYTHON"] = str(self.python)
        return env

    def test_strict_config_requires_context_id_and_scenario_disjoint(self) -> None:
        config = json.loads(
            (self.repo_root / "configs" / "scenario_disjoint_policy_candidate_evaluation_v1.json").read_text(
                encoding="utf-8"
            )
        )

        validation = config["validation"]
        self.assertTrue(validation["require_context_id"])
        self.assertTrue(validation["require_scenario_disjoint"])
        self.assertEqual(validation["max_scenario_overlap_count"], 0)
        self.assertEqual(validation["max_identity_overlap_count"], 0)
        self.assertEqual(validation["max_legacy_identity_fallback_count"], 0)
        self.assertTrue(validation["require_candidate_git_current_match"])

    def test_batch_matrix_accepts_holdout_scenario_set(self) -> None:
        fake_single_run = self.temp_dir / "fake-single-run.sh"
        fake_single_run.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
        fake_single_run.chmod(fake_single_run.stat().st_mode | stat.S_IXUSR)

        completed = subprocess.run(
            [
                str(self.python),
                str(self.repo_root / "scripts" / "run_batch_path_feedback_validation.py"),
                "--matrix",
                str(self.repo_root / "configs" / "path_feedback_batch_scenario_disjoint_policy_candidate_evaluation_v1.json"),
                "--single-run-script",
                str(fake_single_run),
                "--validate-only",
            ],
            cwd=self.repo_root,
            env=self._env(),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        payload = json.loads(completed.stdout.splitlines()[0])
        self.assertEqual(payload["run_count"], 2)

    def test_holdout_generator_emits_seeded_variant_ids_disjoint_from_existing_sets(self) -> None:
        script = self.repo_root / "dev-platform-constraints" / "scripts" / "generate_npz_validation_maps.py"

        holdout = self._generator_dry_run(script, "holdout")
        all_existing = self._generator_dry_run(script, "all")

        holdout_ids = {item["scenario_id"] for item in holdout["scenarios"]}
        existing_ids = {item["scenario_id"] for item in all_existing["scenarios"]}
        holdout_seeds = {item["seed"] for item in holdout["scenarios"]}
        existing_seeds = {item["seed"] for item in all_existing["scenarios"]}

        self.assertTrue(holdout_ids)
        self.assertFalse(holdout_ids & existing_ids)
        self.assertFalse(holdout_seeds & existing_seeds)
        self.assertTrue(all(str(item["scenario_id"]).startswith("npz_holdout_") for item in holdout["scenarios"]))
        self.assertTrue(
            all(
                item["scenario_variant_id"] == f"{item['scenario_id']}-seed-{item['seed']}"
                for item in holdout["scenarios"]
            )
        )

    def _generator_dry_run(self, script: Path, scenario_set: str) -> dict:
        completed = subprocess.run(
            [
                str(self.python),
                str(script),
                "--scenario-set",
                scenario_set,
                "--dry-run",
            ],
            cwd=script.parents[1],
            env=self._env(),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        return json.loads(completed.stdout)


if __name__ == "__main__":
    unittest.main()
